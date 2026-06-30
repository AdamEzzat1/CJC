# State-Space-Model Primitives — Design Note

A short design / scope document. Authoritative architectural decisions live
in [ADR-0020](../../CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0020%20State-Space-Model%20Primitives.md).

## Scope

Add the smallest useful set of deterministic temporal-memory primitives to
CJC-Lang, demonstrate them in an RL chess agent, do not break the existing
chess demo, do not weaken determinism.

## Non-goals

- Mamba / S4 selective-scan kernels
- Diagonal-plus-low-rank `A` parameterization
- Continuous-time discretization (zero-order hold etc.)
- Trainable SSM (no GradGraph integration in this phase — see "Deferred")

## What temporal memory buys an RL agent

Without it, every position is processed in isolation. Two distinct
problems result on chess RL workloads:

1. **Repetition blindness.** The agent has no signal that it just played
   a position it visited two plies ago, so it cannot learn to avoid
   threefold-repetition draws or break out of cycle traps.
2. **Phase drift.** Opening, middlegame, and endgame differ in what
   matters (development vs. king safety vs. promotion). A position-only
   feature vector loses the move number's context.

A small (input=6, hidden=12, output=8) SSM cell, updated once per ply,
gives the MLP head a temporal summary it can learn to use.

## Recurrence

```text
h_t = tanh(A · h_{t-1} + B · x_t)
y_t = C · h_t + b_o
```

Shapes: `A: [hidden, hidden]`, `B: [hidden, input]`, `C: [output, hidden]`,
`b_o: [output]`. `tanh` keeps the hidden state bounded in `[-1, 1]`,
preventing saturation explosions across long sequences.

For the demo, `x_t` is the 6-D summary
`[ply_norm, side, material_diff, repetition_count, in_check, last_move_dist]`.
The `y_t` readout is concatenated with the 16-D board feature vector and
fed to the MLP.

## Determinism story

- **SplitMix64** for weight initialization. Per-tensor seed offsets
  (`+0xA1`, `+0xB2`, `+0xC3`) prevent shape changes from disturbing
  unrelated parameters.
- **No FMA** in the matvec — purely scalar `acc += w[i,j] * x[j]`. Gives
  bit-identical results across platforms.
- **`BTreeMap` arena.** Iteration order is sorted. No `HashMap`.
- **Monotonic, never-reused handles.** Stale handles after `state_space_clear`
  return clean error messages instead of aliasing fresh cells.

## Phase 2 (ADR-0021) — Performance Primitives + Differentiable Composition

Added 9 builtins on top of Phase 1:

| Builtin | Role |
|---|---|
| `tensor_concat_1d(a, b) → c` | Native 1-D concat replacing user-space O(N²) `array_push` chain |
| `state_space_step_with_readout(handle, x) → [y, h]` | Fused step + readout; bit-identical to split path |
| `state_space_step_batched(handle, xs[B,input]) → ys[B,output]` | Independent-row batched step (zeroes h between rows) |
| `state_space_get_A` / `_get_B` / `_get_C` / `_get_b_o` | Extract cell weight tensors as `Tensor` |

**Differentiable SSM via composition.** With the four extractors, user code
can build a fully-differentiable SSM forward pass *without* a new GradOp
variant: wrap the extracted matrices in `grad_graph_input(...)`, compose
the recurrence using existing `matmul + add + tanh` ops, and gradients
flow through `A`, `B`, `h`, `x` automatically. Trade-off vs a fused
`GradOp::SsmStep`: graph-node count is O(layers × sequence_length).
Adequate for short-unroll RL; the trigger to ship the fused variant is
when this composition saturates.

**Demo speedup is modest (~10% AST wall-clock).** The demo is dominated
by *other* user-space loops (`legal_moves`, `encode_board`,
`position_signature`) that aren't SSM-specific. The Phase-2 primitives
shave the SSM-related dispatch; further gains require chess-specific
fused kernels (deferred).

**Cold-start measurement gotcha.** During investigation a single cold
run timed `tensor_concat_1d` at 6.5s vs. the warm 0.28s — a phantom 23×
regression. Always run ≥ 3 warm iterations before reporting wall-clock
numbers.

## Test coverage

62 tests in `tests/state_space_tests/` (up from 33 in Phase 1):

| Layer | Tests | Purpose |
|---|---|---|
| Dispatch unit (Phase 1) | 13 | Direct calls to `dispatch_state_space`, no parser |
| Dispatch parity (Phase 1) | 6 | AST↔MIR byte-equal across executors |
| Properties (Phase 1) | 6 (×128 cases) | Same-seed determinism, scan==step, snapshot round-trip, etc. |
| Demo smoke (Phase 1) | 4 | End-to-end `.cjcl` execution under both backends |
| Demo replay (Phase 1) | 4 | Bit-identical double-run with fixed seed |
| Perf primitives (Phase 2) | 18 | Unit tests for fused/native primitives + extractors |
| Perf parity (Phase 2) | 4 | AST↔MIR byte-equal for new primitives |
| Perf proptest (Phase 2) | 5 (×128 cases = 640) | Concat associativity/preservation, fused≡split, etc. |
| Perf fuzz (Phase 2, bolero) | 2 | Structural state recovery + concat numerical |

Plus 8 tests in `cjc-runtime/src/state_space.rs` (in-crate unit coverage).

## Deferred

### 1. SSM autodiff integration

Plumbing `state_space_step` into `cjc-ad::GradGraph` requires:

- A new `GradOp::SsmStep { handle, x_idx, h_in_idx, h_out_idx }` variant
  (or finer-grained ops if we want to compose recurrences).
- Backward-through-time arithmetic. `dL/dh_{t-1}` flows back through the
  recurrence; `dL/dA`, `dL/dB`, `dL/dC` accumulate across all timesteps.
- Either fixed-shape allocation in the GradGraph arena or per-step ops
  (the latter scales node count with sequence length).

Until done, the chess demo treats SSM readouts as **frozen features**.
This is not a hack — it's a legitimate architecture (fixed feature
extractor + trainable head). The trigger for unfreezing is empirical:
benchmarks showing the frozen approach saturates on chess RL quality
gates that the v2.x line currently misses.

### 2. Selective state spaces

S4/Mamba use input-dependent `A(x_t)` to make the system "selective" —
the recurrence parameters change per step. We ship time-invariant `A` for
simplicity. Adding selectivity is a subsequent design phase: it requires
either a learned input-dependent transform or per-step `A` matrices, both
of which interact with the deferred autodiff design.

### 3. Parallel scan

The current `state_space_scan` is purely sequential (one matvec per step).
For long sequences this becomes the bottleneck; the standard trick is the
Blelloch parallel scan with associative-friendly `A` parameterizations.
Out of scope until SSM autodiff lands and benchmarks justify it.

## File / line references

- `crates/cjc-runtime/src/state_space.rs` — module + dispatch (≈460 LOC)
- `crates/cjc-runtime/src/builtins.rs` — fall-through wiring (`other =>`
  arm in `dispatch_builtin`)
- `crates/cjc-runtime/src/lib.rs` — `pub mod state_space;`
- `demos/state_space_chess/demo.cjcl` — ~500 LOC self-contained RL demo
- `tests/state_space_tests/` — 33 tests across 5 files

## What could go wrong

- **Thread-local arena leaks across tests.** The arena is per-thread, and
  cargo's test pool reuses threads. We mitigate with explicit
  `state_space_clear()` at the top of every test snippet, plus a monotonic
  handle counter so a stale handle from a prior test errors instead of
  aliasing.
- **Long sequences saturate `tanh`.** Mitigation: keep input magnitudes
  bounded (the demo divides material by 1000, ply by max_plies, etc.).
  The `[-1, 1]` Glorot-lite weight scale also helps.
- **Numerical drift from FMA.** Avoided by hand-rolled scalar matvec.
