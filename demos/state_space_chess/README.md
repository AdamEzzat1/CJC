# `state_space_chess` — chess RL with state-space-model temporal memory

A reinforcement-learning chess demo that augments the standard
`board_features → policy/value` architecture with a deterministic
state-space cell. The cell carries a hidden state across moves so the
agent can condition on temporal context — repetition, threat development,
phase drift — instead of treating every position as fully isolated.

This is **not** a replacement for the existing
[`tests/chess_rl_v2/`](../../tests/chess_rl_v2/) demo. The two coexist.
This one tracks the new state-space primitives shipped under
[ADR-0020](../../CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0020%20State-Space-Model%20Primitives.md).

## Run it

```bash
cargo build --release --bin cjcl
target/release/cjcl run demos/state_space_chess/demo.cjcl       # AST executor
target/release/cjcl run --mir-opt demos/state_space_chess/demo.cjcl  # MIR executor
```

The demo runs three episodes:

1. Episode 1 — verbose self-play, printing per-move policy probability,
   value estimate, and SSM hidden-state L2² norm.
2. Episode 2 — silent self-play, used to disturb the SSM.
3. Episode 3 — replay from a mid-game snapshot, proving
   `state_space_restore` round-trips.

Each run with the same SEED produces bit-identical output (verified by
`tests/state_space_tests/test_demo_replay.rs`).

## Architecture

```text
              ┌──────────────────────────┐
   board ─────│  encode_board → 16-D     │──┐
   state      └──────────────────────────┘  │
                                            ▼
   move        ssm_input = [ply, side, ...] │
   history ──► state_space_step(handle, x)  │
                       │                    │
                       ▼                    │
              state_space_readout → 8-D ────┤
                                            ▼
                              concat → 24-D │
                                            ▼
                       trainable MLP head (GradGraph-built, eager forward)
                                            ▼
                          policy logits + value scalar
                                            ▼
                          softmax + categorical_sample
```

The SSM cell is built once per process via
`state_space_init(input=6, hidden=12, output=8, SEED)` and reset to zero
hidden state at the start of every episode. Its weights are drawn from
SplitMix64 with the user-supplied seed and never change during the demo —
the cell is a *frozen* temporal-feature extractor, not a trainable layer.
The trainable head sits downstream.

## Differentiable SSM (Phase 2 update)

The demo runs the SSM as a frozen feature extractor, but as of
[ADR-0021](../../CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0021%20State-Space%20Performance%20Primitives%20and%20Differentiable%20Composition.md)
**user code can build a differentiable SSM today, without a new GradOp variant.**
The extractors `state_space_get_A`, `_get_B`, `_get_C`, `_get_b_o` return
the cell's weight tensors. Wrap them with `grad_graph_input(...)` (or
`grad_graph_param(...)` for trainable variants) and compose the recurrence
from existing primitives:

```cjclang
let A_idx = grad_graph_input(state_space_get_A(handle));
let B_idx = grad_graph_input(state_space_get_B(handle));
let h_idx = grad_graph_input(state_space_state(handle));
let x_idx = grad_graph_input(x_t);
let pre   = grad_graph_add(grad_graph_matmul(A_idx, h_idx),
                           grad_graph_matmul(B_idx, x_idx));
let h_new = grad_graph_tanh(pre);
// ...continue building loss; gradients flow through A, B, h, x.
```

Trade-off vs. a future fused `GradOp::SsmStep`: graph-node count is
O(layers × sequence_length). Adequate for short-unroll RL but eventually
saturates — that's the trigger to ship the fused variant.

Full rationale and the deferral roadmap (parallel scan blocked by tanh
non-associativity, selective SSMs layered on top of `SsmStep`) live in
[ADR-0020 § Consequences](../../CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0020%20State-Space-Model%20Primitives.md#what-we-deferred--and-why)
and [ADR-0021](../../CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0021%20State-Space%20Performance%20Primitives%20and%20Differentiable%20Composition.md).

## Determinism guarantees

- **SplitMix64 seeding** — `state_space_init(_, _, _, seed)` produces
  bit-identical weights for the same seed.
- **No FMA / no parallel reductions** in the matvec hot path.
- **`BTreeMap<usize, Cell>`** arena — iteration order is sorted, not
  randomized.
- **Snapshot / restore** — `state_space_snapshot(h)` returns a deep copy
  of the hidden tensor; `state_space_restore(h, snap)` reinstalls it
  exactly. Verified by proptest.

## Tests

`tests/state_space_tests/` (62 tests across two phases):

| File | Coverage |
|---|---|
| `test_dispatch_unit.rs` | 13 unit tests against the dispatch directly |
| `test_dispatch_parity.rs` | 6 AST↔MIR byte-equal parity tests |
| `test_properties.rs` | 6 proptest properties × 128 cases |
| `test_demo_smoke.rs` | 4 end-to-end demo execution checks |
| `test_demo_replay.rs` | 4 deterministic replay double-runs |
| `test_perf_primitives.rs` *(Phase 2)* | 18 unit tests for fused/native primitives |
| `test_perf_parity.rs` *(Phase 2)* | 4 AST↔MIR parity tests for new primitives |
| `test_perf_proptest.rs` *(Phase 2)* | 5 proptest properties × 128 cases (640 generated) |
| `test_perf_fuzz.rs` *(Phase 2)* | 2 bolero fuzz targets (structural + numerical) |

Run with:

```bash
cargo test --release --test state_space_tests
```

## Limitations and known scope

- 4×4 board with a tiny piece set — exists to keep the demo file readable
  (~500 LOC). The SSM primitives themselves work for any tensor shapes.
- King capture rule (no checkmate logic). Stalemate ends the game in a
  draw.
- No castling, en passant, or promotion.
- The chess demo's full output diverges between AST and MIR executors
  because `Tensor.randn` (used to init the trainable head) threads the
  RNG differently in the two backends. This is pre-existing behavior;
  the SSM primitives themselves *are* AST↔MIR byte-identical (verified by
  `test_dispatch_parity.rs`).
