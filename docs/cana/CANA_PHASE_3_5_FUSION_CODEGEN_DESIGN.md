# CANA Phase 3.5 — Fusion Codegen Design

**Status:** session foundation; multi-session scope.
**Predecessor:** Phase 3 — fusion candidate *identification* (in [`cjc-cana/src/fusion.rs`](../../crates/cjc-cana/src/fusion.rs)).
**Goal:** turn each identified `FusionCandidate` into a real native primitive that runs faster than the original chain, byte-identical in both executors.

---

## 1. The mismatch that scopes this session

Phase 3 produces `FusionPlan { candidates: Vec<FusionCandidate> }` where every entry is a chain of names from `NATIVE_PRIMITIVES`:

```rust
pub const NATIVE_PRIMITIVES: &[&str] = &[
    "matmul", "transpose", "dot", "sum", "mean", "tensor_concat_1d",
    "mlp_forward", "mlp_layer", "encode_state_fast", "score_moves_batch",
    "adam_step",
];
```

This set treats every name as a free-function call (`MirExprKind::Call` with a `Var`/`VarLocal` callee). But in the actual codebase the primitives split into two dispatch contexts:

| Context | Primitives | Where it lives | Signature shape |
|---|---|---|---|
| **A — tensor builtins** | `matmul`, `transpose`, `dot`, `sum`, `mean`, `tensor_concat_1d`, `encode_state_fast`, `score_moves_batch`, `adam_step` | [`cjc-runtime/src/builtins.rs`](../../crates/cjc-runtime/src/builtins.rs) `dispatch_builtin` | `(Tensor, Tensor, ...) → Value` |
| **B — GradGraph methods** | `mlp_layer` (and would-be `mlp_forward` if exposed) | [`cjc-ad/src/lib.rs`](../../crates/cjc-ad/src/lib.rs) `GradGraph::mlp_layer`; dispatched in [`cjc-eval/src/lib.rs:3708`](../../crates/cjc-eval/src/lib.rs:3708) and [`cjc-mir-exec/src/lib.rs:3765`](../../crates/cjc-mir-exec/src/lib.rs:3765) | `(node_idx, node_idx, ..., Activation) → node_idx` |

A literal `fused_mlp_matmul_dot(input, w1, b1, "tanh", w2, v) → f64` therefore can't exist as a single dispatch arm. `mlp_layer` produces a node index, `matmul` consumes a `Tensor`, `dot` returns `f64`. The types only line up if you collapse the chain **within one dispatch context**.

## 2. Scope split

### Phase 3.5a — tensor-level fusion (this session)

- Add `fused_matmul_dot(a: Tensor, w: Tensor, v: Tensor) → Float` to `cjc-runtime/src/builtins.rs`.
- Math: `dot(matmul(a, w), v)` with the same `binned_sum_f64` accumulator that `dot` already uses, eliminating the intermediate `Tensor` allocation that `matmul` would otherwise produce.
- Bit-identical to running the chain unfused.
- Wire through the three-place pattern (runtime / cjc-eval / cjc-mir-exec).
- Parity tests + bolero fuzz + proptest.
- Add to `cjc-cana::fusion::NATIVE_PRIMITIVES` so the identifier knows it exists (Phase 3 already lists the unfused names; we add the *fused* name so a future codegen step can detect already-fused chains).
- **Out of scope:** MIR rewriter that automatically replaces matched chains with fused calls. That comes in 3.5c.

### Phase 3.5b — GradGraph-level fusion (next session)

- Add `GradOp::MlpMatmulDot { input, weight1, bias1, activation, weight2, target }` to `cjc-ad/src/lib.rs`.
- Forward: `sum(activation(input @ w1ᵀ + b1) @ w2 * target)` returning a scalar tensor.
- Backward: a single hand-derived gradient computation for `d/d{input, w1, b1, w2}`.
- New `GradGraph::fused_mlp_matmul_dot` constructor method.
- Wire via the existing dispatch pattern in cjc-eval/cjc-mir-exec (method-call routing).
- Parity tests against `mlp_layer → matmul → sum(* target)` chain.
- This is where the literal `fused_mlp_matmul_dot` name lands. Reuses the same scaffolding patterns proven in 3.5a.

### Phase 3.5c — MIR rewriter (later session)

- Walk every `MirBody`, for each `FusionCandidate` of length ≥ 2:
  - Verify the chain shape matches a known fused primitive.
  - Ask the `LegalityGate` for approval (NoGC, determinism, intermediate-value liveness).
  - Replace the chain of `MirStmt::Let` with a single `MirStmt::Let { init: Call("fused_...", ...) }`.
- Drop the intermediate bindings (DCE catches them if needed).
- Parity gate after rewriting (this is the load-bearing check).

## 3. Why `fused_matmul_dot` first

It's the smallest possible fusion that:
- Spans two distinct native primitives (proves the pattern).
- Saves a real `Tensor` allocation (the `matmul` output, which has shape `[M, K]` for `a: [M, N]`, `w: [N, K]`).
- Has a verifiable bit-identical reference: the unfused `dot(matmul(a, w), v)` chain that we can test against.
- Uses the same `binned_sum_f64` accumulator as the existing `dot` — determinism story is already proven.

It also matches a real production pattern: per-step inference in chess RL, PINN final-layer residual computation, and BLR posterior projection all end in `dot(matmul(layer_out, head_w), target)`.

## 4. Determinism contract

- The fused kernel must produce **byte-identical** results to the unfused chain on the same inputs.
- This is checked by parity tests (`tests/cana_phase_3_5/`) that compute both forms and compare via `f64::to_bits()`.
- The accumulation order in the fused kernel matches the unfused chain: matmul row-major, then `binned_sum_f64` over `result * v`.

## 5. File map (this session)

```
NEW:
  docs/cana/CANA_PHASE_3_5_FUSION_CODEGEN_DESIGN.md  (this file)
  tests/cana_phase_3_5/                              (parity + fuzz tests)

MODIFIED:
  crates/cjc-runtime/src/builtins.rs   (+fused_matmul_dot dispatch arm)
  crates/cjc-cana/src/fusion.rs        (+ NATIVE_PRIMITIVES entry for the fused name)
```

The cjc-eval and cjc-mir-exec executors both call `dispatch_builtin` for top-level free-function names, so no changes are needed there — adding to `cjc-runtime` exposes the function to both executors automatically.

## 6. Open question for Phase 3.5b

Should `GradOp::MlpMatmulDot` carry the `target` (final-projection vector) as a node index or a constant tensor? Node index keeps the autodiff story uniform; constant tensor reduces graph size when the target doesn't change across epochs (e.g., supervised training labels). Decision deferred to the 3.5b session.

---

**Bottom line:** this session ships `fused_matmul_dot` — small, real, parity-proven — and lays out the scaffolding pattern so 3.5b can drop a GradGraph variant in without re-thinking the architecture.
