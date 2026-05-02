# ADR-0023 — Native Higher-Order Autodiff (Phase 3d)

**Date:** 2026-05-01
**Status:** Accepted, partial implementation (Phase 3e scope deferred)
**Supersedes:** none
**Related:** [[ADR-0016 Language-Level GradGraph Primitives]],
[[ADR-0022 Typed Index Newtypes for ML Metadata]]

## Context

Phase 3c (ADR-0016) shipped 24 language-level `grad_graph_*` builtins,
exposing the existing reverse-mode autodiff tape to user `.cjcl`
source. ADR-0016 explicitly deferred *higher-order* AD:

> PINN residuals need `u_t`, `u_xx`, etc. Native higher-order AD
> requires a graph-of-graphs rewrite of the existing tape, which is a
> major change affecting every `GradOp` backward closure. The brief
> explicitly authorized **central finite differences** (ε=1e-3,
> 3-point stencil) as the Phase 3c fallback.

The flagship pure-CJC-Lang heat-1D PINN demo therefore uses 5 forward
passes per collocation point to compute `u_t` and `u_xx` via FD —
which is correct (truncation error O(ε²) ≈ 1e-6) but quintuples the
forward-pass cost relative to native AD.

ADR-0016 closed with: *"Phase 3d will close that gap."*

## Decision

Ship a **native graph-of-graphs `grad_of`** as `GradGraph::grad_of(f,
x)` and the corresponding `grad_graph_grad_of(f, x)` builtin.

Where `backward()` accumulates `dF/dParam` into `param_grads` (concrete
tensors), `grad_of` builds a **sub-graph** that represents `dF/dx` as a
function of the existing parameters and inputs. The returned node:

- Forward-evaluates like any other node (`grad_graph_forward(dx)`).
- Is itself differentiable — `grad_of(grad_of(f, x), x)` gives native
  second-order derivatives, with no FD truncation.
- Composes with all existing `grad_graph_*` ops at construction sites.

### Polynomial subset (Phase 3d, this ADR)

Phase 3d ships native `grad_of` for the **polynomial-arithmetic op
subset only**:

| GradOp variant | Derivative rule |
|---|---|
| `Input` / `Parameter` | leaf — gradient stops at this node |
| `Add(a, b)` | `dF/da += upstream`, `dF/db += upstream` |
| `Sub(a, b)` | `dF/da += upstream`, `dF/db += -upstream` (via `Neg`) |
| `Mul(a, b)` | `dF/da += b * upstream`, `dF/db += a * upstream` |
| `ScalarMul(a, s)` | `dF/da += scalar_mul(upstream, s)` |
| `Neg(a)` | `dF/da += -upstream` (via `Neg`) |

These suffice to express any **polynomial in the parameters**.
Higher-order derivatives of such polynomials evaluate exactly without
finite-difference truncation, which is what makes the
`d³(x³)/dx³ = 6` test possible bit-exactly.

### Deferred to Phase 3e

The following GradOp variants Err with a clean message ("not yet
supported in Phase 3d") rather than panic:

- **Reductions:** `Sum`, `Mean` — require building a graph node that
  broadcasts a scalar upstream gradient to a vector shape, which is
  not expressible with the current pointwise op set. Either a new
  `GradOp::ScalarBroadcast` is added, or `Sum/Mean` is decomposed.
- **Matmul:** requires building a `Matmul(b.transpose(), upstream)`
  graph; doable but needs `transpose` exposed at the graph level.
- **Transcendentals:** `Exp`, `Ln`, `Sqrt`, `Sin`, `Cos`, `Pow`,
  `Log2`, `Abs` — each has a derivative rule (e.g.,
  `d(exp(x))/dx = exp(x)`); 5-15 LOC each, but requires careful
  testing.
- **Activations:** `Sigmoid`, `Relu`, `TanhAct`, `Gelu`, `Silu`,
  `Elu`, `Selu` — 5-10 LOC each.
- **Fused ops:** `MlpLayer`, `Softmax`, `LayerNorm`, `BatchNorm`,
  `CrossEntropy`, `Reshape`, `CatOp`, `GatherOp`, `Where`, `Clamp`,
  `TransposeOp`, `Div` — each requires either a per-op gradient
  formula or a "decompose-then-grad_of" fallback.

Phase 3e will expand the supported set incrementally — likely in two
PRs: (a) reductions + transcendentals + activations (most ops, all
straightforward), (b) fused ops (more involved per-op work).

### Why split here?

The polynomial subset has a special property: **every per-op
derivative formula is expressible using ops already in the polynomial
subset itself**. `Add`'s gradient uses `Add` for accumulation; `Mul`'s
gradient uses `Mul`; `Neg`'s gradient uses `Neg`; `ScalarMul`'s
gradient uses `ScalarMul`. The subset is *closed under
differentiation*. That's what makes `grad_of(grad_of(f, x), x)`
trivial — the gradient sub-graph uses only ops we already know how to
differentiate.

Other ops break this closure. `d(exp(x))/dx = exp(x) * upstream`
introduces `Exp` into the gradient graph, so to differentiate the
gradient we'd need to support `Exp` too. That's not a problem in
principle, but it means **Phase 3e must add ops in topological order
of their derivative dependencies**.

## Consequences

### Positive

- **Native HOAD works today** for polynomial expressions, validated
  with bit-exact tests (`d³(x³)/dx³ = 6` byte-equal across eval and
  MIR).
- **Same dispatch surface as ADR-0016 promised.** Users add one
  builtin (`grad_graph_grad_of`) — no new dispatch table.
- **Cross-checked against `backward()`.** A new test
  `grad_of_agrees_with_backward_on_polynomial` verifies that for
  polynomial expressions, the new graph-of-graphs path produces
  byte-identical gradients to the existing tensor-accumulating path.
- **Composes with Phase 2a typed boundary.** The dispatch arm receives
  `NodeIdx` from `arg_idx_checked`; the wiring is identical to other
  arms.

### Negative

- **PINN residuals still need FD today.** The heat 1D PINN demo uses
  `Sum` and `MlpLayer(tanh)` — neither in Phase 3d's subset. The demo
  can't switch to native HOAD until Phase 3e adds those ops.
- **Op-by-op expansion is a long tail.** Roughly 25+ ops still to
  cover. Each is small but the bookkeeping (per-op gradient formulas,
  per-op tests, per-op proptests) compounds.
- **Subset is non-obvious.** A user calling `grad_of(my_loss, w)`
  where `my_loss` happens to use `tanh` will get a runtime Err. The
  Err message names the unsupported op, but a user expecting
  full-coverage HOAD can be surprised.

### Neutral

- The decision to *not* use the FD fallback as a builtin is
  intentional. Wrapping FD as `grad_graph_grad_of_fd(f, x, eps)`
  would be sugar for what user code already does manually
  (`set_tensor` + `reforward` + scalar arithmetic). The bar Phase 3d
  needed to clear was *native* AD, not "FD with a friendlier
  signature."

## Implementation notes

### Shape of the algorithm

`grad_of` mirrors `backward()`:

1. Walk reachability from `f` back to ancestors.
2. Initialize upstream at `f` to a ones-tensor (an `Input` node).
3. Reverse-topological pass over original graph nodes (0..=f).
4. For each visited node, look up upstream graph-node-index, and
   build new graph nodes representing the contribution to each
   input's gradient.
5. Stop propagation when `i == x` — `upstream_grad[x]` holds the
   final answer.
6. Return `upstream_grad[x]` as the node index.

The key difference from `backward()`: instead of operating on
`Tensor` values (e.g., `grad.mul_elem_unchecked(&b_val)`), every
intermediate is a `usize` graph-node index, and operations are
`self.mul(b, upstream)` / `self.scalar_mul(upstream, s)` / etc. —
graph-construction calls that append to the tape.

### Accumulation across multiple paths

When a node has multiple downstream uses (e.g., `x` used by both
`Mul(x, x)` and `ScalarMul(x, 5)`), gradients arrive from each path.
The implementation accumulates via `Add(existing, contribution)` —
exactly mirroring how `accumulate_grad` works in `backward()`, but
producing a graph node instead of a tensor sum.

### One-tensor initialization

The upstream gradient at `f` is initialized to a tensor of ones with
`f`'s shape, exposed as an `Input` node. For scalar-shaped `f` (the
typical case after a `Sum` reduction would normally produce — though
`Sum` itself isn't yet supported), this is just `[1.0]`.

## Test matrix

Phase 3d ships **18 new tests** in
`tests/physics_ml/grad_graph_phase3d_higher_order.rs`:

- **7 wiring tests** — eval ↔ MIR byte-equal printed output for each
  supported op + the second-order test.
- **7 analytic-correctness tests** — Rust-direct `GradGraph` calls
  comparing `grad_of`'s forward-evaluated output to closed-form
  derivatives. Includes the cubic third-derivative test
  (`d³(x³)/dx³ = 6`).
- **1 cross-check test** — `grad_of_agrees_with_backward_on_polynomial`
  validates that for a polynomial expression, the new graph-of-graphs
  path produces gradients byte-identical to the existing
  tensor-accumulating `backward()`.
- **2 error-path tests** — unsupported op (`tanh`) Errs cleanly;
  `x` unreachable from `f` Errs.
- **1 dispatch-boundary test** — exercises the language-level builtin
  via `dispatch_grad_graph(...)` with `Value::Int` indices, ensuring
  Phase 2a's typed boundary plumbs through correctly.

All 18 tests pass on first run.

## Acceptance criteria

- [x] `crates/cjc-ad/src/lib.rs` defines `pub fn grad_of(&mut self,
  f: usize, x: usize) -> Result<usize, String>` with the polynomial
  subset above.
- [x] `crates/cjc-ad/src/dispatch.rs` adds the `grad_graph_grad_of`
  arm (uses Phase 2a's typed `arg_idx_checked → NodeIdx` helper).
- [x] `cargo test -p cjc-ad --release` passes (no regression).
- [x] `cargo test --test physics_ml --release grad_graph_phase3d`
  passes (18/18).
- [x] Cross-check test demonstrates byte-equal gradients between
  `grad_of` and `backward()` on a polynomial expression.
- [x] This ADR documents the polynomial subset, the deferred set, and
  the closure-under-differentiation rationale.

## Phase 3e expansion roadmap

A future PR should expand `grad_of` coverage in this order, since
each tier's ops use the previous tier's ops in their gradient
formulas:

**Tier 1 — Reductions and broadcasts ✅ SHIPPED**

Status: implemented in a follow-up PR. Adds:

- `GradOp::BroadcastScalar { input, target_shape }` — new variant
  expressing "scalar value replicated to target shape." Required
  because `Sum`/`Mean` reduce a vector to a scalar; their gradient
  must broadcast a scalar gradient back to the input's shape, and
  the broadcast itself must be a graph node so higher-order
  derivatives compose.
- `grad_of` arms for `Sum`, `Mean`, and `BroadcastScalar`. `Sum` and
  `BroadcastScalar` are **mutually closed under differentiation**
  (Sum's gradient uses BroadcastScalar; BroadcastScalar's gradient
  uses Sum), preserving the polynomial subset's closure property.
  Adding both at once was necessary — adding only one would break
  closure.
- 15 new tests covering wiring (eval ↔ MIR byte-equal), analytic
  correctness vs closed-form derivatives, second-order
  (`d²(Σ x_i²)/dx_i² = 2`), cross-check vs `backward()`, and
  BroadcastScalar primitive forward/backward.

Coverage extension after Tier 1:

- Inputs: `Input`, `Parameter` (leaves)
- Pointwise: `Add`, `Sub`, `Mul`, `ScalarMul`, `Neg`
- Reductions: `Sum`, `Mean` ← Tier 1
- Broadcast: `BroadcastScalar` ← Tier 1 (helper)

**Tier 2 — Smooth transcendentals and activations ✅ PARTIALLY SHIPPED**

Status: 9 of the 14 originally-listed ops shipped in a follow-up PR.

Shipped (smooth, derivative formula expressible via Tier 0 + Tier 1
ops):

- `Exp` — `dF/da = upstream * exp(a)`. Reuses current node's tensor
  rather than building a fresh exp.
- `Ln` — `dF/da = upstream / a`.
- `Sqrt` — `dF/da = upstream / (2*sqrt(a))`. Reuses current node.
- `Sin`, `Cos` — derivatives are each other (with sign).
- `Pow(a, n)` — `dF/da = n * a^(n-1) * upstream`. Builds a fresh Pow
  with reduced exponent.
- `Log2` — `dF/da = upstream / (a * ln(2))`.
- `Sigmoid` — `dF/da = σ(a) * (1 - σ(a)) * upstream`. Uses
  BroadcastScalar (Tier 1) to construct a same-shape ones tensor.
- `TanhAct` — `dF/da = (1 - tanh²(a)) * upstream`. Same shape trick.

22 new tests including `d²(exp(x))/dx² = exp(x)` (self-similar
derivative) and `d²(x³)/dx² = 6x` via Pow.

Deferred to Tier 3 (need Where-like op or piecewise-tensor
primitives the current GradGraph can't express cleanly):

- `Abs` — sub-gradient `sign(x)` requires conditional.
- `Relu` — `1[x>0]` requires conditional indicator.
- `Elu` / `Selu` — piecewise `x>0 ? f(x) : g(x)`.
- `Gelu` — composite formula referencing `Φ(x)` (CDF of normal); no
  graph-level erf/erfinv yet.
- `Silu` — `σ + a*σ*(1-σ)` is composable but requires Mul of three
  factors and would benefit from a fused primitive.

**Tier 3 — Compound ops** (each requires per-op care):

- `Matmul` — needs `transpose` at the graph level.
- `Div` — needs `Pow(b, -1)` or new `Inv` GradOp.
- `MlpLayer`, `LayerNorm`, `BatchNorm`, `CrossEntropy`, `Softmax`.

**Tier 4 — Shape/index ops** (defer until concrete need):

- `Reshape`, `CatOp`, `GatherOp`, `TransposeOp`, `Where`, `Clamp`.

The recommended cadence is one tier per PR. Each tier has a clear
test pattern (analytic derivative via grad_of equals
analytic-derivative-by-hand for representative inputs).
