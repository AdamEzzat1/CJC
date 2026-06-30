---
title: "ADR-0020: State-Space-Model Primitives"
tags: [adr, ml, rl, state-space, dispatch]
status: Accepted
date: 2026-04-29
---

# ADR-0020: State-Space-Model Primitives

## Status

Accepted

## Context

The Chess RL v2.x line ([[Chess RL v2]]) treats every board position as
fully isolated: the trainable MLP head sees a 774-D feature vector and has
no memory of prior plies. It cannot learn to react to repetition,
threat-development, or phase drift across a game. The Phase 3c
[[ADR-0016 Language-Level GradGraph Primitives|`grad_graph_*` surface]]
gave us autodiff in CJC-Lang source but did nothing for *temporal* memory.

We want a deterministic, inspectable, testable temporal-memory primitive
that:

- Plugs into the existing dispatch pipeline without breaking parity (G-8).
- Preserves the [[ADR-0004 SplitMix64 RNG|SplitMix64]] / [[ADR-0002 Kahan Accumulator|Kahan]]
  determinism story.
- Can be used from `.cjcl` source directly — same ergonomic level as
  `grad_graph_*` and the chess-engine builtins.
- Stays small. We are *not* shipping a Mamba clone in this ADR; we are
  shipping the smallest useful kernel that demonstrates state-space-style
  temporal reasoning in an RL agent.

## Decision

### 1. New module `cjc-runtime/src/state_space.rs`

A thread-local arena of `StateSpaceCell` structs, addressed by `usize`
handles that cross the language boundary as `Value::Int(i64)` — same
representation as [[ADR-0016 Language-Level GradGraph Primitives|grad_graph]]
node indices. This preserves `Value` enum layout (HARD RULE #1).

Each cell implements a discrete-time recurrence:

```text
h_t = tanh(A · h_{t-1} + B · x_t)        // hidden update
y_t = C · h_t + b_o                       // readout
```

with shapes `A: [hidden, hidden]`, `B: [hidden, input]`, `C: [output, hidden]`.

### 2. Dispatch placement: inside `cjc-runtime::dispatch_builtin`

Unlike `dispatch_grad_graph` (which had to be a satellite to avoid a
`cjc-runtime → cjc-ad` cycle), state-space cells use only `Tensor` types
that `cjc-runtime` already owns. The dispatch fall-through inside
`dispatch_builtin` calls `crate::state_space::dispatch_state_space`
directly — **no executor changes are needed**. Both `cjc-eval` and
`cjc-mir-exec` already call `dispatch_builtin` and inherit the new
builtins for free.

### 3. The thirteen builtins

| Builtin | Signature | Behavior |
|---|---|---|
| `state_space_init` | `(input, hidden, output, seed) → Int` | Construct cell, return handle |
| `state_space_step` | `(handle, x[input]) → Tensor[output]` | One forward step; mutates `h` |
| `state_space_scan` | `(handle, xs[T,input]) → Tensor[T,output]` | Apply step over T inputs |
| `state_space_reset` | `(handle) → Void` | Zero `h` |
| `state_space_state` | `(handle) → Tensor[hidden]` | Read `h` |
| `state_space_set_state` | `(handle, h) → Void` | Direct write |
| `state_space_readout` | `(handle) → Tensor[output]` | `C·h + b_o` without stepping |
| `state_space_snapshot` | `(handle) → Tensor[hidden]` | Deep-copy `h` |
| `state_space_restore` | `(handle, snap) → Void` | Restore from snapshot |
| `state_space_hidden_dim` | `(handle) → Int` | Introspection |
| `state_space_input_dim` | `(handle) → Int` | Introspection |
| `state_space_output_dim` | `(handle) → Int` | Introspection |
| `state_space_clear` | `() → Void` | Drop the entire arena |
| `state_space_len` | `() → Int` | Cells alive |

### 4. Determinism

- Cell weights are drawn from an inlined SplitMix64 stream, seeded by the
  user via the `seed` argument. Per-tensor seed offsets (`+0xA1`, `+0xB2`,
  `+0xC3`) prevent shape changes from disturbing unrelated parameters.
- All matmuls are hand-rolled scalar `i*cols+j` loops — no FMA, no parallel
  reduction. Bit-identical across runs and platforms.
- Arena is `BTreeMap<usize, StateSpaceCell>` for deterministic iteration.
- Handle counter is monotonic and never reused; a stale handle from a
  cleared arena surfaces as "does not refer to a live cell" rather than
  silently aliasing.

### 5. Demo

`demos/state_space_chess/demo.cjcl` is a self-contained 4×4 micro-chess RL
demo that uses an `(input=6, hidden=12, output=8)` cell. The 6-D input is
a temporal summary of the position (`[ply_norm, side, material_diff,
repetition_count, in_check_dummy, last_move_dist]`); the 8-D readout is
concatenated with a 16-D board feature vector and fed into a trainable
MLP head. The demo runs three episodes per invocation:

1. Verbose — prints policy, value, hidden-state norm per move
2. Silent self-play
3. Replay from snapshot — proves SSM `snapshot/restore` round-trips

## Consequences

### What we got

- Chess RL agents can now condition on temporal context (repetition, phase,
  prior policy confidence) in pure CJC-Lang source.
- The 33 new tests in `tests/state_space_tests/` give:
  - 13 dispatch unit tests
  - 6 AST↔MIR parity tests (byte-identical output across executors)
  - 6 proptest properties × 128 cases each = 768 generated cases
  - 4 demo smoke tests
  - 4 demo replay determinism tests
- Phase 3c-style satellite dispatch precedent is now generalized: dispatch
  modules can live *inside* `cjc-runtime` when there's no dependency cycle.

### What we deferred — and why

**SSM ops do NOT participate in `GradGraph`.** Plumbing `state_space_step`
into a `GradOp` variant requires:

1. A new `GradOp::SsmStep { handle, x_idx, h_in_idx, h_out_idx }` variant
2. Custom `backward()` arithmetic for the recurrence: `dL/dh_{t-1}` flows
   *backwards through time* (BPTT), and `dL/dA`, `dL/dB` accumulate across
   all timesteps in a sequence.
3. Either fixed-shape allocation in the GradGraph arena or a `Vec<Op>` per
   timestep — the latter explodes node counts on long sequences.

This is a bigger design choice than fits in the current scope. Until it's
done, the chess demo treats SSM readouts as **frozen features**: the
trainable MLP head sees them as inputs, learns to use them, but does not
backprop through the recurrence. This is the same architecture as a
fixed-feature embedding layer with a trainable downstream classifier — a
fully respectable design choice, just not the only one.

The deferred work is tracked as a future ADR. Trigger conditions:
benchmarks showing the frozen-feature approach saturates on chess RL
quality gates.

**Not a Mamba clone.** Real structured-state-space-models (Mamba, S4) use
diagonal-plus-low-rank `A` matrices, parallel-scan algorithms, and
continuous-time discretization. We use a dense `[hidden, hidden]` matrix
and a sequential scan. For `hidden ≤ 64` this is fine; for larger cells
the dense matmul is the bottleneck, not the algorithm.

## Related

- [[ADR-0016 Language-Level GradGraph Primitives]] — established the
  thread-local-arena + `Value::Int(handle)` pattern this ADR reuses.
- [[ADR-0004 SplitMix64 RNG]] — the same RNG primitive seeds SSM weights.
- [[Chess RL v2]] — the demo target; SSM agent lives in
  `demos/state_space_chess/`, separate from the chess_rl_v2 line.
