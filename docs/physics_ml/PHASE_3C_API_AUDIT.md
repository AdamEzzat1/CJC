# Phase 3c — `cjc-ad` API Audit

**Date:** 2026-04-26
**Owner:** Lead Language Architect (Phase 3c)
**Source:** `crates/cjc-ad/src/lib.rs` (3,963 LOC), `crates/cjc-ad/src/pinn.rs` (4,855 LOC)

This audit classifies every public method on `GradGraph` as **PRIMITIVE** (single op
node construction or trivial state read/write — eligible for language exposure) or
**DRIVER** (composite/algorithmic — must stay in Rust or be re-implemented in user
`.cjcl` code on top of primitives). Phase 3c exposes only PRIMITIVES.

---

## 1. Storage shape

```rust
pub struct GradGraph {
    ops: Vec<GradOp>,
    tensors: Vec<Tensor>,
    param_grads: Vec<Option<Tensor>>,
}
// derives Clone, Debug. Send + Sync (no Rc / no RefCell).
```

Flat-arena design (post-v2.4). Node index = `usize` offset into all three vectors.
**Cloneable cheaply** for graphs up to ~1000 nodes. **No interior mutability** —
all mutators take `&mut self`. This is what makes Option B (ambient context held
behind a thread-local `RefCell`) clean: the borrow boundary is at the ambient
container, not inside `GradGraph`.

## 2. Public methods on `GradGraph`

### Constructors / metadata — PRIMITIVE
- `new() -> Self` — empty graph
- `len() -> usize`, `is_empty() -> bool`

### Node construction — PRIMITIVE
- `input(t: Tensor) -> usize` — non-trainable input
- `parameter(t: Tensor) -> usize` — trainable parameter (grad slot allocated)
- `push_node(op, tensor, grad) -> usize` — raw arena append (do not expose)

### Pointwise ops (forward immediate, store result tensor) — PRIMITIVE
- `add(a,b)`, `sub(a,b)`, `mul(a,b)`, `div(a,b)`
- `neg(a)`, `scalar_mul(a, s: f64)`
- `exp(a)`, `ln(a)`, `log2(a)`, `sqrt(a)`, `pow(a, n: f64)`, `abs(a)`
- `sin(a)`, `cos(a)`
- `sigmoid(a)`, `relu(a)`, `tanh_act(a)`, `gelu(a)`, `silu(a)`, `elu(a)`, `selu(a)`

### Reductions / shape — PRIMITIVE
- `sum(a) -> usize` — scalar
- `mean(a) -> usize` — scalar
- `transpose_op(a)`, `reshape(a, shape: &[usize])`
- `cat(inputs: &[usize], axis: usize)`, `gather(a, indices: &[usize], axis: usize)`

### Composite ops still *graph-level* (single fused op variant) — PRIMITIVE
- `matmul(a, b)` — 2-D matmul
- `mlp_layer(input, weight, bias, activation: Activation) -> usize`
  fused `act(x @ Wᵀ + b)`. **Activation enum:** Tanh, Sigmoid, Relu, None, Gelu,
  Silu, Elu, Selu, **SinAct** (PINN-friendly).
- `softmax(a)`, `cross_entropy(logits, targets)`, `layer_norm(a)`, `batch_norm(a)`
- `clamp(a, min, max)`, `where_cond(cond, t, f)`

### State accessors — PRIMITIVE
- `value(idx) -> f64` (scalar shorthand)
- `tensor(idx) -> Tensor`
- `grad(idx) -> Option<Tensor>`
- `set_tensor(&mut self, idx, t)` — parameter update / perturbation
- `zero_grad(&mut self)`

### Backward pass — PRIMITIVE (single call into reverse-mode tape)
- `backward(&mut self, loss_idx)`
- `backward_with_seed(&mut self, loss_idx, seed: &Tensor)`
- `clip_grad_norm(&mut self, max_norm) -> f64` — single global-norm pass

### DRIVER (do **not** expose; reproduce in `.cjcl` if needed)
- `clip_grad(&mut self, max_norm)` — element-wise clip *loop* over params
- `backward_collect(loss, params: &[usize]) -> Vec<Option<Tensor>>` — convenience
  wrapper around `zero_grad + backward + grad`-loop
- `reforward(start, end)` — loop-based partial re-evaluation
- `jacobian(out, param)` — for-loop over output dims, repeated `backward_with_seed`
- `hessian_diag`, `hessian` — finite-diff over parameter dims
- `double_backward` — finite-diff over parameter dims
- `vmap_forward(input, batch: &[Tensor]) -> Vec<usize>` — explicit batch loop

These contain control flow that user code will write directly.

## 3. `GradOp` variants (34)

`Input`, `Parameter`, `Add`, `Sub`, `Mul`, `Div`, `Neg`, `ScalarMul`, `MatMul`,
`Sum`, `Mean`, `Exp`, `Ln`, `Log2`, `Sqrt`, `Pow`, `Abs`, `Sin`, `Cos`, `Sigmoid`,
`Relu`, `TanhAct`, `Gelu`, `Silu`, `Elu`, `Selu`, `Softmax`, `CrossEntropy`,
`LayerNorm`, `BatchNorm`, `Clamp`, `Where`, `Reshape`, `TransposeOp`, `CatOp`,
`GatherOp`, `MlpLayer`.

## 4. Higher-order AD

**No native graph-node-level higher-order AD exists.** All current second-order
APIs (`hessian_diag`, `double_backward`, `jacobian`) return *concrete tensors*
(final gradients), not graph nodes. Implementing `grad_graph_grad_of(out, in)
-> NodeIdx` would require a new `GradOp::GradOf` variant + reverse-on-reverse
tape replay across all 34 op variants. Estimated cost > 500 LOC, plus a fresh
risk surface for determinism bugs.

**Decision:** ship **finite-difference fallback** for Phase 3c. Bit-equality
relaxes to `RMSE < 1e-6` as authorized by the brief. See `PHASE_3C_DESIGN.md`.

## 5. PINN heat-1D trainer outline (`pinn_heat_1d_nn_train`)

```text
init params (seeded SplitMix64 → Vec<Vec<f64>>)
for epoch in 0..E {
  graph = GradGraph::new()
  p_idx = params.iter().map(|p| graph.parameter(t(p)))
  // IC loss
  ic_terms = ic_pts.map(|x,u*| (mlp(x,0)-u*)²)
  ic_loss  = mean(ic_terms)
  // BC loss
  bc_terms = bc_pts.map(|x,t| mlp(x,t)²)         // u=0 at boundary
  bc_loss  = mean(bc_terms)
  // Physics residual: u_t - α·u_xx via finite differences
  phys_terms = collocation.map(|x,t|
      ut  = (mlp(x, t+ε) - mlp(x, t-ε)) / (2ε)
      uxx = (mlp(x+ε,t) - 2*mlp(x,t) + mlp(x-ε,t)) / ε²
      (ut - α*uxx)²
  )
  phys_loss = mean(phys_terms)
  loss = phys_loss + λ_bc·(ic_loss + bc_loss)
  graph.zero_grad(); graph.backward(loss)
  for (p, g) in zip(params, p_idx.map(graph.grad)):
      adam_step(p, g, lr, …)
}
```

Five primitives carry the algorithm: `parameter`, `mlp_layer`, `sub/mul/mean`,
`backward`, `grad`. Everything else is glue, and glue is what we move to `.cjcl`.

## 6. Phase 3c builtin surface (final, ~24 names)

Construction:
- `grad_graph_new()` *(resets ambient graph)*
- `grad_graph_param(t)`, `grad_graph_input(t)`, `grad_graph_const(t)`

Op nodes:
- `grad_graph_add/sub/mul/div(a,b)`
- `grad_graph_neg/exp/ln/sqrt(a)`
- `grad_graph_pow(a, n_f64)`, `grad_graph_scalar_mul(a, s)`
- `grad_graph_sin/cos/tanh(a)`
- `grad_graph_sum/mean(a)`
- `grad_graph_matmul(a, b)`
- `grad_graph_mlp_layer(x, w, b, activation_str)`

Forward/backward/state:
- `grad_graph_forward(idx) -> Tensor`
- `grad_graph_set_tensor(idx, t)`
- `grad_graph_zero_grad()`
- `grad_graph_backward(loss_idx)`
- `grad_graph_param_grad(idx) -> Tensor`
- `grad_graph_clip_grad_norm(max_norm)`

FD helper for residuals:
- `grad_graph_fd_first(...)` and `grad_graph_fd_second(...)` are *not* builtins —
  they're written in `.cjcl` and call `set_tensor` + `forward` directly.

---

**End of audit.** All data needed by `PHASE_3C_DESIGN.md` is here.
