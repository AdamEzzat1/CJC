---
title: CJC-Lang v2.4 — Ranked Design Proposals
date: 2026-04-10
status: Phase 3 complete — P1/P2/P3/P5 implemented, measurement pending
predecessor: docs/chess_rl_v2/PERFORMANCE_AUDIT_v2_4.md
---

# Phase 2: Ranked Design Proposals

Proposals ranked by **expected-speedup / implementation-complexity**.

---

## P1: Arena-Based GradGraph (RECOMMENDED — implement first)

**One-line:** Replace `Vec<Rc<RefCell<GradNode>>>` with flat `Vec<GradNode>`
and index-based references.

**Expected speedup:** 20-30% on backward pass (~5-10 s/episode saved)

**Implementation complexity:** Medium (1-2 files, ~200 lines changed)

**Risk to determinism:** None — same traversal order, same arithmetic

**New builtins:** None

**Files affected:**
- `crates/cjc-ad/src/lib.rs` — GradGraph struct, all methods

**Design:**
```rust
pub struct GradGraph {
    ops: Vec<GradOp>,           // flat array of operations
    tensors: Vec<Tensor>,       // forward values (one per node)
    grads: Vec<Option<Tensor>>, // parameter gradients (sparse)
}
```

Benefits:
1. Eliminate ~120 `.borrow()` / `.borrow_mut()` calls per backward pass
2. Eliminate per-node heap allocation (Rc overhead)
3. Cache-contiguous node access (better prefetching)
4. The backward loop no longer needs to clone `op` and `tensor` out of
   RefCell — it can borrow directly from `&self.ops[i]` and `&self.tensors[i]`

**Migration path:** Keep the public API identical. `GradGraph::parameter()`,
`GradGraph::input()`, etc. still return `usize` node indices. The internal
storage changes are invisible to callers.

**Test plan:**
- All existing `cjc-ad` unit tests must pass unchanged
- All existing chess RL parity tests must produce identical weight hashes
- New proptest: random graph construction + backward produces same gradients
  as old implementation

---

## P2: Dead Node Elimination in Backward (RECOMMENDED — implement second)

**One-line:** Build a reachability set from loss node to parameters before
backward traversal; skip unreachable nodes.

**Expected speedup:** 15-25% on backward pass for multi-head networks

**Implementation complexity:** Small (~30 lines added to backward)

**Risk to determinism:** None — same gradients, fewer wasted operations

**New builtins:** None

**Files affected:**
- `crates/cjc-ad/src/lib.rs` — `backward()` method only

**Design:**
```rust
pub fn backward(&self, loss_idx: usize) {
    // NEW: Build reachability set
    let mut reachable = vec![false; self.ops.len()];
    reachable[loss_idx] = true;
    for i in (0..=loss_idx).rev() {
        if !reachable[i] { continue; }
        // Mark children as reachable
        match &self.ops[i] {
            GradOp::Add(a, b) | GradOp::Sub(a, b) | ... => {
                reachable[*a] = true;
                reachable[*b] = true;
            }
            ...
        }
    }

    // Backward pass — skip unreachable nodes
    for i in (0..=loss_idx).rev() {
        if !reachable[i] { continue; }
        // ... existing backward logic
    }
}
```

**Test plan:**
- Existing tests pass unchanged
- New test: multi-head graph where only one head backpropagates — verify
  unused head parameters have zero gradient

---

## P3: Eliminate Tensor Cloning in Backward (RECOMMENDED — implement with P1)

**One-line:** With arena storage (P1), borrow `op` and `tensor` by
reference instead of cloning them out of RefCell.

**Expected speedup:** 15-20% on backward pass (eliminates 2N clones)

**Implementation complexity:** Small (comes free with P1)

**Risk to determinism:** None

**Files affected:**
- `crates/cjc-ad/src/lib.rs` — backward loop body

**Design (post-P1):**
```rust
for i in (0..=loss_idx).rev() {
    let grad = match grads[i].take() {
        Some(g) => g,
        None => continue,
    };

    // NO CLONE — borrow directly from flat arrays
    let op = &self.ops[i];
    let node_tensor = &self.tensors[i];

    match op {
        GradOp::MatMul(a, b) => {
            let a_val = &self.tensors[*a];  // NO CLONE
            let b_val = &self.tensors[*b];  // NO CLONE
            // ... compute gradients using references
        }
        ...
    }
}
```

**Caveat:** Some operations (MatMul backward) need temporary tensors
for transpose + matmul results. These allocations are inherent and cannot
be eliminated without a tensor pool. But the 2 input clones per MatMul
node ARE eliminated.

---

## P4: In-Place Gradient Accumulation (RECOMMENDED — implement with P1)

**One-line:** Replace `accumulate_grad()` with in-place `add_assign` when
the gradient tensor already exists.

**Expected speedup:** 10-15% on backward (eliminates N/2 tensor allocations)

**Implementation complexity:** Small (~20 lines)

**Risk to determinism:** None — same arithmetic, different allocation pattern

**Design:**
```rust
fn accumulate_grad(grads: &mut [Option<Tensor>], idx: usize, grad: &Tensor) {
    match &mut grads[idx] {
        Some(existing) => {
            existing.add_assign_unchecked(grad);  // NEW: in-place add
        }
        slot @ None => {
            *slot = Some(grad.clone());
        }
    }
}
```

Requires adding `add_assign_unchecked()` to `Tensor`:
```rust
pub fn add_assign_unchecked(&mut self, other: &Tensor) {
    let data = self.data_mut();
    let other_data = other.to_vec();
    for (a, b) in data.iter_mut().zip(other_data.iter()) {
        *a += b;
    }
}
```

---

## P5: COW Mutation for array_pop and array_reverse (QUICK WIN)

**One-line:** Apply `Rc::make_mut()` pattern (already used by `array_push`)
to `array_pop` and `array_reverse`.

**Expected speedup:** Minimal for chess RL; prevents O(N) regression in
other workloads.

**Implementation complexity:** Tiny (2 one-line changes)

**Risk to determinism:** None

**Files affected:**
- `crates/cjc-runtime/src/builtins.rs` — lines 3036 and 3051

**Design:**
```rust
// array_pop: before
let arr = match &args[0] { Value::Array(a) => (**a).clone(), ... };
// array_pop: after
let mut arr_rc = match &args[0] { Value::Array(a) => Rc::clone(a), ... };
let last = Rc::make_mut(&mut arr_rc).pop().ok_or("...")?;
Ok(Some(Value::Tuple(Rc::new(vec![last, Value::Array(arr_rc)]))))
```

---

## P6: Native Backward Pass Builtin (DEFERRED — high impact but high risk)

**One-line:** `grad_backward_native(weights, graph_data, loss)` — execute
entire backward traversal in Rust, bypassing interpreter dispatch.

**Expected speedup:** 2-5× on backward pass (eliminates all interpreter
overhead for the backward traversal itself)

**Implementation complexity:** Large (~500+ lines)

**Risk to determinism:** Medium — must produce bit-identical gradients

**Why deferred:** P1-P4 together should achieve 40-60% backward speedup
with much lower risk. If that's insufficient, P6 becomes the next tier.

---

## P7: PINN Graph Reuse (SEPARATE WORKLOAD)

**One-line:** Build PINN graph once, update parameter values per epoch
instead of rebuilding.

**Expected speedup:** 1.5-2× on PINN epoch time

**Implementation complexity:** Medium

**Risk to determinism:** Low

**Files affected:**
- `crates/cjc-ad/src/pinn.rs` — training loop
- `crates/cjc-ad/src/lib.rs` — add `update_parameter_value()` method

**Design:** Add to GradGraph:
```rust
pub fn update_parameter_value(&mut self, idx: usize, new_tensor: &Tensor) {
    self.tensors[idx] = new_tensor.clone();
}
```

Then PINN training loop builds graph once and calls `update_parameter_value()`
+ `reforward()` + `backward()` per epoch.

---

## P8: Fused MLP Layer GradOp (FUTURE)

**One-line:** Add `GradOp::MlpLayer { input, weight, bias, activation }`
that fuses matmul+add+activation in both forward and backward.

**Expected speedup:** 1.5-2× on PINN; marginal for chess RL

**Implementation complexity:** Medium-Large

**Deferred because:** P7 (graph reuse) gives a larger win for less effort.

---

## Implementation Order

```
Phase 3a: P1 (arena) + P3 (no-clone) + P4 (in-place accum)  ← biggest win
Phase 3b: P2 (dead node elimination)                          ← quick add-on
Phase 3c: P5 (COW array_pop/reverse)                          ← tiny quick win
Phase 3d: P7 (PINN graph reuse)                               ← if time permits
```

**Combined expected speedup (P1+P2+P3+P4):** 40-60% reduction in backward
pass time. For a 25-45 s backward pass, this saves 10-25 s/episode.

**New per-episode target:** 82.2 s - 15 s (avg) = **~65-70 s/episode.**
Gate T1 (≤40 s) will likely still miss — that requires P6 (native backward)
or MIR-level optimizations.
