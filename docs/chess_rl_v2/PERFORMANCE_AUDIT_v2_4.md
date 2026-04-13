---
title: CJC-Lang v2.4 — Performance Architecture Audit
date: 2026-04-10
status: Phase 3 complete — P1/P2/P3/P5 implemented, P4 deferred (Tensor lacks add_assign_unchecked)
scope: GradGraph backward pass, COW arrays, gradient strategies, MIR optimizer
---

# Phase 1: Performance Architecture Audit

## Executive Summary

Four independent audits identify **three dominant bottlenecks** in the
chess RL v2.3 training loop (82.2 s/episode baseline):

| Bottleneck | % of episode | Root cause | Fix complexity |
|---|---|---|---|
| **GradGraph backward pass** | ~35-45% | Tensor cloning from `Rc<RefCell>`, no dead-node skip, per-node allocation | Medium |
| **Interpreter loop overhead** | ~25-35% | COW array copies in outer while-loop (partially optimized) | Small |
| **Forward-pass dispatch** | ~15-20% | BTreeMap scope lookup, per-expression match dispatch | Medium-Large |

The remaining ~10% is inherent compute (matmul, tensor arithmetic).

---

## 1A — GradGraph Backward Pass Audit

### Node storage

```rust
// cjc-ad/src/lib.rs:326-328
pub struct GradGraph {
    pub nodes: Vec<Rc<RefCell<GradNode>>>,
}

// cjc-ad/src/lib.rs:319-323
pub struct GradNode {
    pub op: GradOp,           // enum, 39+ variants
    pub tensor: Tensor,       // cached forward value
    pub grad: Option<Tensor>, // accumulates during backward
}
```

Every node is heap-allocated behind `Rc<RefCell<>>`. Each access requires
a runtime borrow check.

### Clone count per backward pass

**Per-node baseline** (lines 1049-1052): every visited node clones both
its `op` and `tensor`:

```rust
let (op, node_tensor) = {
    let node = self.nodes[i].borrow();
    (node.op.clone(), node.tensor.clone())  // 2 clones always
};
```

**Per-operation additional clones:**

| GradOp | Extra clones | Line | Notes |
|---|---|---|---|
| Add | 0 | 1064 | Grad passed by ref |
| Sub | 0+1 alloc | 1068 | `grad.neg()` allocates |
| Mul | 2 | 1073 | `a_val.clone()` + `b_val.clone()` |
| Div | 2+2 alloc | 1083 | Clones + derived tensors |
| MatMul | 2+2 alloc | 1101 | `a_val`, `b_val` + transpose + matmul results |
| Sum | 1 alloc | 1116 | Constructs expanded gradient |
| Relu | 1+1 alloc | 1219 | `a_val.clone()` + mask |
| TanhAct | 0+1 alloc | 1228 | Uses `node_tensor` + `one_minus_sq` |
| Sigmoid | 0+1 alloc | 1208 | Uses `node_tensor` + derived |
| Exp | 0 | 1140 | Uses `node_tensor` |
| Softmax | 0+2 alloc | 1261 | `.to_vec()` on grad & softmax |

**`accumulate_grad()`** (line 2457): allocates a NEW tensor on every call
via `add_unchecked()` or `.clone()`:

```rust
fn accumulate_grad(grads: &mut [Option<Tensor>], idx: usize, grad: &Tensor) {
    if let Some(existing) = &grads[idx] {
        grads[idx] = Some(existing.add_unchecked(grad));  // new alloc
    } else {
        grads[idx] = Some(grad.clone());  // clone
    }
}
```

**Total for N-node graph:** ~2N baseline clones + ~1.5N operation clones
+ accumulation clones = **~4N tensor operations** per backward pass.

For a chess RL episode graph with ~5,000 nodes: **~20,000 tensor clone/alloc
operations per backward pass.**

### Dead node traversal

**No reachability analysis exists.** Line 1042:
```rust
for i in (0..=loss_idx).rev() {
```

Every node from 0 to loss_idx is visited. Nodes with `grads[i] == None`
are skipped (line 1043-1045), but the op+tensor clone at lines 1049-1052
happens before the skip check would apply to children.

For multi-head networks (policy + value), nodes contributing only to the
unused head are still visited but contribute no gradient. Estimated dead
node ratio: **20-30%** for the chess RL factored policy/value architecture.

### `.grad()` accessor clones

```rust
pub fn grad(&self, idx: usize) -> Option<Tensor> {
    self.nodes[idx].borrow().grad.clone()  // clones entire gradient
}
```

Collecting gradients for all 10 trainable tensors clones 10 tensors.

### Arena replacement feasibility

Replace `Vec<Rc<RefCell<GradNode>>>` with flat `Vec<GradNode>`:
- Eliminate 2 heap indirections per node (Rc + RefCell)
- Eliminate runtime borrow checking (~120 `.borrow()` calls in backward)
- Store gradients in separate `Vec<Option<Tensor>>` (already done for
  intermediate grads; just extend to parameters)
- Cache locality improvement: nodes are contiguous in memory

**Estimated backward speedup from arena alone: 15-25%.**

---

## 1B — COW Array Overhead Audit

### Storage model

```rust
// cjc-runtime/src/value.rs:157
Array(Rc<Vec<Value>>),
```

COW via `Rc<Vec<Value>>`. Clone is O(1) (refcount increment).
Mutation uses `Rc::make_mut()` — clones only when refcount > 1.

### `array_push` — already optimized (line 3027)

```rust
Rc::make_mut(&mut arr_rc).push(args[1].clone());
```

Uses `Rc::make_mut()`. For `arr = array_push(arr, val)` where the old
binding is overwritten, refcount is 1 → zero-copy push (amortized O(1)).

**However**: if any other reference to the array exists (e.g., it was
passed to a function, or stored in a closure), refcount > 1 and the
entire array is deep-cloned.

### Other array builtins — NOT optimized

| Builtin | Clone pattern | Line | Cost |
|---|---|---|---|
| `array_pop` | `(**a).clone()` | 3038 | O(N) always |
| `array_reverse` | `(**a).clone()` | 3053 | O(N) always |
| `array_flatten` | Element-wise clone | 3060 | O(N) always |
| `array_map` | `a.as_ref().clone()` | eval:2235 | O(N) always |
| `array_filter` | `a.as_ref().clone()` | eval:2254 | O(N) always |
| `array_reduce` | `a.as_ref().clone()` | eval:2273 | O(N) always |

### Chess RL loop cost

The rollout outer loop does 5 `array_push` calls per iteration for 80
iterations. If `array_push` COW works correctly (refcount=1), total cost
is O(80 × 5) = O(400) amortized pushes.

**But**: the arrays are passed to `compute_gae()` and used in the A2C
update, which may create additional references. If any reference escapes,
the next push becomes O(N) instead of O(1).

**Worst case** (all shared): 5 × Σ(1..80) = **16,200 element clones.**
**Best case** (all single-owner): 5 × 80 = **400 pushes** (amortized O(1) each).

### Quick fix: apply `Rc::make_mut()` to `array_pop` and `array_reverse`

These are 1-line changes each. Not high-impact for chess RL but prevents
regression for other workloads.

---

## 1C — Gradient Strategy Unification Audit

### Three strategies compared

| | Reverse-Mode (GradGraph) | Parameter-Shift (VQE) | Finite-Diff (PINN Hessian) |
|---|---|---|---|
| **Gradient cost** | 1 backward per loss | 2p forward evals | 2p forward + backward |
| **Memory** | O(N) node storage | O(1) per param | O(N) graph + overhead |
| **Graph rebuild** | Once per forward | None | Every epoch |
| **Tensor cloning** | ~4N per backward | None (scalar) | ~4N per backward |
| **Second-order** | O(p²) via FD | N/A | O(p) Hessian-diag |

### Key insight: PINN rebuilds graph every epoch

```rust
for epoch in 0..config.epochs {
    let mut graph = GradGraph::new();  // Fresh graph each epoch!
    // ... rebuild all parameters, inputs, operations
}
```

This is O(epochs × graph_size) allocations. A single graph with in-place
parameter updates would reduce this to O(graph_size) total.

### Cross-strategy opportunities

1. **Arena allocation** benefits all three strategies (GradGraph, PINN,
   and any future reverse-mode quantum gradient).
2. **Graph reuse in PINN** — build once, update parameter values per epoch.
3. **Hessian-vector products** via forward-over-reverse could replace the
   O(p²) finite-difference Hessian with O(1) backward passes.

### MLP forward+backward as native builtin

PINN calls `mlp_forward()` 500+ times per epoch. Each call creates
graph nodes. A fused `mlp_layer_fused()` builtin could:
- Reduce graph size by 2-3× (one node per layer instead of matmul+add+activation)
- Fuse backward kernel (avoid intermediate tensor allocation)
- Expected speedup: 1.5-2× on PINN epoch time.

---

## 1D — MIR Optimizer Audit

### Current passes (6)

1. Constant Folding — folds literal-only expressions
2. Strength Reduction — algebraic identities (`a*0→0`, `a+0→a`)
3. Dead Code Elimination — removes unused let-bindings
4. Common Subexpression Elimination — reuses identical pure bindings
5. Loop-Invariant Code Motion — hoists invariant bindings out of while
6. Second Constant Folding — catches opportunities exposed by other passes

### MIR instruction count: 32 variants

Literals (8), containers (5), special (3), variables (2), operators (3),
memory access (3), control flow (4), closures (2), linalg (5).

### Loop analysis infrastructure exists but is underutilized

`crates/cjc-mir/src/loop_analysis.rs` provides:
- Loop tree construction from CFG
- Trip count detection (`is_countable`, `trip_count_hint`)
- Schedule hints (Sequential, Tiled, Vectorized, Partitioned)

**None of these are used by the tree-form optimizer.** Only CFG-aware
passes leverage loop metadata.

### Reduction analysis detects accumulation patterns

`crates/cjc-mir/src/reduction.rs` recognizes sum/product/min/max folds
but NOT array accumulation (`arr = array_push(arr, ...)`).

### Per-instruction dispatch overhead

Each `eval_expr()` call in the MIR executor:
- 32-way pattern match (~50 cycles)
- BTreeMap scope lookup for Var (~100-200 cycles)
- Value cloning for results (~1-100 cycles)
- **Total: ~200-500 cycles per instruction**

For a tight 80-iteration loop with ~20 instructions/iteration, that's
~320,000-800,000 cycles of pure dispatch overhead per episode.

### Key insight: no instruction fusion exists

Multiple statements per iteration = multiple dispatch cycles. A fused
"array accumulate" or "increment-and-test" instruction could eliminate
repeated dispatch for hot patterns.

---

## Baseline Measurements

| Metric | Value | Source |
|---|---|---|
| Chess RL per-episode (v2.3) | 82.2 s | Phase D v2.3 summary |
| Rollout forward pass | ~2.6 s | v2.3 native kernels |
| `a2c_update_adam` backward | ~25-45 s | Profile zones |
| Interpreter loop overhead | ~30-50 s | Difference estimate |
| PINN harmonic per-epoch | ~15-30 ms | Example timing note |
| Total tests (workspace) | ~5,353 | Pre-v2.4 baseline |
