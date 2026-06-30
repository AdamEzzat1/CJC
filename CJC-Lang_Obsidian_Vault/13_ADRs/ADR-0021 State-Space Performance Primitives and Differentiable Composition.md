---
title: "ADR-0021: State-Space Performance Primitives and Differentiable Composition"
tags: [adr, ml, rl, state-space, performance, autodiff]
status: Accepted
date: 2026-04-29
---

# ADR-0021: State-Space Performance Primitives and Differentiable Composition

## Status

Accepted

## Context

[[ADR-0020 State-Space-Model Primitives|ADR-0020]] shipped the first
SSM surface for CJC-Lang and explicitly deferred three items:

1. **Full SSM autodiff (BPTT through `state_space_step`)** — requires a
   new `GradOp::SsmStep` variant and reverse arithmetic.
2. **Selective state spaces (input-dependent `A(x_t)`)** — Mamba/S4-style.
3. **Parallel scan** — Blelloch parallel-scan over the recurrence.

A Stack Role Group revisit asked: *which of these can be tackled now,
and what other performance wins are tractable?*

The investigation produced four findings:

1. **Demo bottleneck is interpreter overhead, not SSM math.** At
   `hidden=12`, the inner matvec is 144 floating-point ops — noise.
   The dominant cost in the chess RL demo is dispatch overhead in
   user-space hot loops (`concat_1d`'s O(N²) `array_push` chain,
   `score_moves`'s per-move scan, `position_signature`'s 16-square walk).
2. **Full SSM autodiff is a multi-site change.** Adding a new `GradOp`
   variant requires updates to ≥ 6 dispatch sites in `cjc-ad/src/lib.rs`
   (`parents`, `forward_node`, `backward`, `reforward`, `backward_with_seed`,
   `backward_collect`). Each is non-trivial, and naive BPTT scales node
   count linearly with sequence length. **Out of scope for one phase.**
3. **Differentiable SSM is achievable today *without* a new variant.**
   Expose the cell's weight tensors (`A`, `B`, `C`, `b_o`); user code
   wraps them with `grad_graph_input(...)` and composes the recurrence
   from existing `matmul + add + tanh` GradOps. Gradients flow through
   `h`, `A`, `B`, `C`, `x` for as many timesteps as the user unrolls.
   This is the recommended pattern until a fused variant ships.
4. **Parallel scan is mathematically blocked.** The tanh recurrence is
   not associative; parallel scan requires associativity (Blelloch). A
   *linear* SSM (drop tanh) would unlock it — but that's a different
   model. **Genuine deferral, not effort-bound.**

## Decision

### 1. Three native fused performance primitives

| Builtin | Signature | Replaces |
|---|---|---|
| `tensor_concat_1d` | `(a, b) → Tensor` | User-space O(N²) `array_push` loop |
| `state_space_step_with_readout` | `(handle, x) → [y, h]` | `step` + `state` + `readout` triple |
| `state_space_step_batched` | `(handle, xs[B,input]) → ys[B,output]` | Per-row Python-style loop |

`tensor_concat_1d` is general (not SSM-specific) — a 1-D tensor concat
that any pipeline can use. The SSM-specific fused step+readout saves one
arena borrow, one `Tensor::clone`, and one dispatch boundary per call.
The batched step zeroes `h` between rows so independent positions can be
evaluated through the same cell without state leakage.

All three preserve bit-identical numerics relative to the unfused path —
proven by parity tests under both `cjc-eval` and `cjc-mir-exec`.

### 2. Four weight-extraction primitives

| Builtin | Returns |
|---|---|
| `state_space_get_A` | `Tensor[hidden, hidden]` |
| `state_space_get_B` | `Tensor[hidden, input]` |
| `state_space_get_C` | `Tensor[output, hidden]` |
| `state_space_get_b_o` | `Tensor[output]` |

These return *clones* of the cell's internal weight tensors. User code
wraps them via `grad_graph_input(...)` (or `grad_graph_param(...)` for
trainable variants) and reconstructs the recurrence from existing GradOps:

```cjclang
let A_idx = grad_graph_input(state_space_get_A(handle));
let B_idx = grad_graph_input(state_space_get_B(handle));
let h_idx = grad_graph_input(state_space_get_state(handle));
let x_idx = grad_graph_input(x);
let pre   = grad_graph_add(grad_graph_matmul(A_idx, h_idx),
                           grad_graph_matmul(B_idx, x_idx));
let h_new = grad_graph_tanh(pre);
// ...continue building loss, run grad_graph_backward(loss_idx)
```

Gradients flow through `A`, `B`, `h`, `x` automatically — full BPTT for
the unrolled length, all using existing primitives. The trade-off vs.
a future `GradOp::SsmStep` variant: graph node count is O(layers × length)
instead of O(length).

### 3. Demo retrofit

`demos/state_space_chess/demo.cjcl` now uses `tensor_concat_1d` and
`state_space_step_with_readout` on its hot path. Speedup is modest
(≈10% AST wall-clock, bit-identical output) — the demo is dominated by
*other* user-space loops (`legal_moves`, `encode_board`,
`position_signature`) that aren't SSM-specific. Future ADRs may add
chess-specific fused kernels (`encode_micro_chess_state`, etc.).

## Consequences

### What we got

- **9 new builtins** (3 fused, 4 extractors, 2 already in Phase-1 scope).
  Builtin count: 489 → 498.
- **29 additional tests** in `tests/state_space_tests/`:
  - `test_perf_primitives.rs` — 18 unit tests
  - `test_perf_parity.rs` — 4 AST↔MIR parity tests
  - `test_perf_proptest.rs` — 5 proptest properties × 128 cases (640 generated)
  - `test_perf_fuzz.rs` — 2 bolero fuzz targets (structural + numerical)
- **Differentiable SSM is now possible from CJC-Lang source** without
  any new GradOp variant. ADR-0021 is the canonical reference for this
  pattern.

### What stays deferred — and why precisely

1. **Fused `GradOp::SsmStep` variant.** Adds ≥ 6 dispatch-site
   modifications in `cjc-ad/src/lib.rs`. Trigger to ship:
   - Empirical benchmarks showing the GradGraph-composition pattern
     (Decision #2) saturates due to per-step graph node overhead, OR
   - User feedback that long-unroll PINN-style training is slow.

   Until then, composition is the recommended pattern.

2. **Parallel scan.** Tanh-recurrence is not associative. Mathematical
   blocker, not effort-bound. A *linear* SSM (no tanh) would unlock
   Blelloch scan — but that's a different model.

3. **Selective SSMs (Mamba/S4 input-dependent `A(x_t)`).** Layered on
   top of #1: requires either a learned input-dependent transform or
   per-step `A` matrices. Out of scope until #1 lands.

### What could go wrong

- **Cold-start measurement noise.** During investigation, a single cold
  run timed `tensor_concat_1d` at 6.5s (vs. warm 0.28s — a 23× phantom
  regression). Lesson: always run ≥ 3 warm iterations before reporting
  speedup. Documented in the perf-investigation note.
- **Extracted-weight aliasing.** `get_A/B/C/b_o` returns `Tensor::clone()`
  (Rc-counted buffer). User code that mutates the returned tensor sees
  copy-on-write (because `Tensor` is COW), so the cell's internal
  weights are unaffected. Verified by unit tests. The doc comment on
  each extractor calls this out.

## Related

- [[ADR-0020 State-Space-Model Primitives]] — the Phase-1 surface this
  ADR extends.
- [[ADR-0016 Language-Level GradGraph Primitives]] — the GradGraph
  surface user code composes against for differentiable SSM.
- [[ADR-0004 SplitMix64 RNG]] — same RNG seeds the cell's weights.
