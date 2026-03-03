# Phase C Changelog -- RL Infrastructure

**Baseline**: 2,050+ tests, 0 failures
**Final**: 2,050+ tests + 75 new tests, 0 failures, 0 regressions

Phase C addressed all gaps identified in the RL readiness analysis across 6
sub-sprints (C1--C6). The goal is to expose reverse-mode AD, optimizers,
bitwise ops, tensor indexing, Map/Set types, and collection utilities to CJC
programs -- everything needed for a chess reinforcement learning demo.

**Architecture constraint**: `cjc-ad` depends on `cjc-runtime`, so
`cjc-runtime` CANNOT depend on `cjc-ad`. GradGraph builtins use type-erased
`Value::GradGraph(Rc<RefCell<dyn Any>>)` and are dispatched directly in
`cjc-eval` and `cjc-mir-exec`, NOT in `builtins.rs`.

---

## C1: GradGraph Language API

**Files**: `crates/cjc-ad/src/lib.rs`, `crates/cjc-runtime/src/value.rs`,
`crates/cjc-eval/src/lib.rs`, `crates/cjc-mir-exec/src/lib.rs`

**New Value variant**: `GradGraph(Rc<RefCell<dyn Any>>)` in `value.rs`

**5 new forward methods added to cjc-ad**: `div()`, `neg()`, `scalar_mul()`,
`exp()`, `ln()` (GradOp variants existed from Phase B but had no public
forward API).

**Constructor**: `GradGraph.new()` -- dispatched in executor `dispatch_call`

| CJC Method | Args | Returns | Notes |
|------------|------|---------|-------|
| `g.parameter(tensor)` | Tensor | Int (node idx) | Creates parameter node |
| `g.input(tensor)` | Tensor | Int (node idx) | Creates input node |
| `g.add(a, b)` | Int, Int | Int | Element-wise add |
| `g.sub(a, b)` | Int, Int | Int | Element-wise subtract |
| `g.mul(a, b)` | Int, Int | Int | Element-wise multiply |
| `g.div(a, b)` | Int, Int | Int | Element-wise divide |
| `g.neg(a)` | Int | Int | Negate |
| `g.matmul(a, b)` | Int, Int | Int | Matrix multiply |
| `g.sum(a)` | Int | Int | Sum all elements |
| `g.mean(a)` | Int | Int | Mean all elements |
| `g.scalar_mul(a, s)` | Int, Float | Int | Scale by float |
| `g.sigmoid(a)` | Int | Int | Sigmoid activation |
| `g.relu(a)` | Int | Int | ReLU activation |
| `g.tanh(a)` | Int | Int | Tanh activation |
| `g.sin(a)` | Int | Int | Sine |
| `g.cos(a)` | Int | Int | Cosine |
| `g.sqrt(a)` | Int | Int | Square root |
| `g.pow(a, n)` | Int, Float | Int | Power |
| `g.exp(a)` | Int | Int | Exponential |
| `g.ln(a)` | Int | Int | Natural log |
| `g.backward(loss_idx)` | Int | Void | Reverse-mode backprop |
| `g.value(idx)` | Int | Float | Scalar value at node |
| `g.tensor(idx)` | Int | Tensor | Full tensor at node |
| `g.grad(idx)` | Int | Tensor/Void | Gradient (Void if None) |
| `g.set_tensor(idx, t)` | Int, Tensor | Void | Update parameter tensor |
| `g.zero_grad()` | -- | Void | Reset all gradients |

**Tests**: 5 unit tests (cjc-ad) + 17 integration tests

---

## C2: Optimizer & Loss Builtins

**Files**: `crates/cjc-runtime/src/value.rs`, `crates/cjc-runtime/src/builtins.rs`,
`crates/cjc-eval/src/lib.rs`, `crates/cjc-mir-exec/src/lib.rs`

**New Value variant**: `OptimizerState(Rc<RefCell<dyn Any>>)` in `value.rs`

Constructors live in `builtins.rs` (no circular dep -- `AdamState`/`SgdState`
are in `cjc-runtime/src/ml.rs`). The `step()` method is dispatched in both
executors using `downcast_mut` to detect Adam vs SGD.

| Builtin | Args | Returns | Effect |
|---------|------|---------|--------|
| `Adam.new(n_params, lr)` | Int, Float | OptimizerState | ALLOC |
| `Sgd.new(n_params, lr, momentum)` | Int, Float, Float | OptimizerState | ALLOC |
| `opt.step(params, grads)` | Tensor, Tensor | Tensor (updated) | ALLOC+MUTATES |

**Note**: Loss functions (`mse_loss`, `cross_entropy_loss`,
`binary_cross_entropy`, `huber_loss`, `hinge_loss`) were already fully wired
from Phase B -- no additional work needed.

**Tests**: 8 integration tests (includes full train loop with GradGraph)

---

## C3: Bitwise Operations

**Files**: `crates/cjc-runtime/src/builtins.rs`

All operate on `Int(i64)` values. Shift operations use `u64` casts with
range checks (0--63).

| Builtin | Implementation | Effect |
|---------|---------------|--------|
| `bit_and(a, b)` | `a & b` | PURE |
| `bit_or(a, b)` | `a \| b` | PURE |
| `bit_xor(a, b)` | `a ^ b` | PURE |
| `bit_not(a)` | `!a` | PURE |
| `bit_shl(a, n)` | `(a as u64) << n` | PURE |
| `bit_shr(a, n)` | `(a as u64) >> n` | PURE |
| `popcount(a)` | `(a as u64).count_ones()` | PURE |

**Tests**: 11 integration tests (includes chess bitboard validation)

---

## C4: Sorting & Tensor Indexing

**Files**: `crates/cjc-runtime/src/tensor.rs`, `crates/cjc-runtime/src/builtins.rs`

Added 4 new methods to `Tensor` impl: `argsort()`, `gather()`, `scatter()`,
`index_select()`. `argsort` uses `f64::total_cmp` for deterministic ordering.
`gather`/`scatter`/`index_select` support both 1D and 2D tensors.

| Builtin | Args | Returns | Effect |
|---------|------|---------|--------|
| `argsort(tensor)` | Tensor | Tensor (indices as f64) | ALLOC |
| `gather(tensor, dim, indices)` | Tensor, Int, Tensor | Tensor | ALLOC |
| `scatter(tensor, dim, indices, src)` | Tensor, Int, Tensor, Tensor | Tensor | ALLOC |
| `index_select(tensor, dim, indices)` | Tensor, Int, Tensor | Tensor | ALLOC |

**Note**: `one_hot(indices, depth)` was already wired from a previous phase --
verified working.

**Tests**: 10 integration tests

---

## C5: Map Completion & Set Type

**Files**: `crates/cjc-runtime/src/builtins.rs`, `crates/cjc-eval/src/lib.rs`,
`crates/cjc-mir-exec/src/lib.rs`

Map operations were registered in `is_known_builtin` and `effect_registry`
but had **zero dispatch implementation**. C5 added full method dispatch for
all Map operations.

**Set type**: Implemented as `Value::Map` with all values = `Value::Void` --
no new Value variant needed. `Set.new()` returns an empty Map; Set methods
(`add`, `contains`, `remove`, `len`, `to_array`) dispatch on the same
`Value::Map` match arm.

| Map Method | Returns | Effect |
|------------|---------|--------|
| `Map.new()` | Map | ALLOC |
| `m.insert(k, v)` | Void | MUTATES |
| `m.get(k)` | Value/Void | PURE |
| `m.remove(k)` | Void | MUTATES |
| `m.len()` | Int | PURE |
| `m.contains_key(k)` | Bool | PURE |
| `m.keys()` | Array | ALLOC |
| `m.values()` | Array | ALLOC |

| Set Method | Returns | Effect |
|------------|---------|--------|
| `Set.new()` | Map (empty) | ALLOC |
| `s.add(v)` | Void | MUTATES |
| `s.contains(v)` | Bool | PURE |
| `s.remove(v)` | Void | MUTATES |
| `s.len()` | Int | PURE |
| `s.to_array()` | Array | ALLOC |

**Tests**: 12 integration tests

---

## C6: I/O & Collection Utilities

**Files**: `crates/cjc-runtime/src/builtins.rs`, `crates/cjc-eval/src/lib.rs`,
`crates/cjc-mir-exec/src/lib.rs`

`read_line` is dispatched as a stateful builtin in both executors'
`dispatch_call` (needs interpreter state for stdin). All array utilities are
stateless and dispatched in `builtins.rs`.

| Builtin | Args | Returns | Effect |
|---------|------|---------|--------|
| `read_line()` | -- | String | IO+NONDET |
| `array_push(arr, val)` | Array, Value | Array (new) | ALLOC |
| `array_pop(arr)` | Array | Tuple(last, rest) | ALLOC |
| `array_contains(arr, val)` | Array, Value | Bool | PURE |
| `array_reverse(arr)` | Array | Array | ALLOC |
| `array_flatten(arr)` | Array (nested) | Array | ALLOC |
| `array_len(arr)` | Array | Int | PURE |
| `array_slice(arr, start, end)` | Array, Int, Int | Array | ALLOC |

**Tests**: 12 integration tests (read_line skipped -- requires stdin)

---

## Test Summary

| Sub-Sprint | Unit Tests | Integration Tests | Total |
|------------|-----------|-------------------|-------|
| C1 | 5 | 17 | 22 |
| C2 | 0 | 8 | 8 |
| C3 | 0 | 11 | 11 |
| C4 | 0 | 10 | 10 |
| C5 | 0 | 12 | 12 |
| C6 | 0 | 12 | 12 |
| **Total** | **5** | **70** | **75** |

Integration test directory: `tests/rl_phase/` (6 modules)

---

## Files Modified

| File | Changes |
|------|---------|
| `crates/cjc-ad/src/lib.rs` | C1: 5 new forward methods (div, neg, scalar_mul, exp, ln) + unit tests |
| `crates/cjc-runtime/src/value.rs` | C1+C2: GradGraph + OptimizerState Value variants |
| `crates/cjc-runtime/src/builtins.rs` | C2--C6: Adam/Sgd constructors, bitwise ops, tensor indexing, Map/Set constructors, array utilities |
| `crates/cjc-runtime/src/tensor.rs` | C4: argsort, gather, scatter, index_select methods |
| `crates/cjc-eval/src/lib.rs` | C1--C6: is_known_builtin entries, GradGraph/Map/Set/OptimizerState dispatch_call + dispatch_method, read_line |
| `crates/cjc-mir-exec/src/lib.rs` | C1--C6: mirrors cjc-eval exactly |
| `crates/cjc-types/src/effect_registry.rs` | C1--C6: ~50 new effect entries |
| `Cargo.toml` | `[[test]]` entry for test_rl_phase |
| `tests/test_rl_phase.rs` | Module declaration |
| `tests/rl_phase/mod.rs` | 6 sub-module declarations |
| `tests/rl_phase/test_c1_gradgraph.rs` | 17 integration tests |
| `tests/rl_phase/test_c2_optimizer.rs` | 8 integration tests |
| `tests/rl_phase/test_c3_bitwise.rs` | 11 integration tests |
| `tests/rl_phase/test_c4_tensor_index.rs` | 10 integration tests |
| `tests/rl_phase/test_c5_map_set.rs` | 12 integration tests |
| `tests/rl_phase/test_c6_collections.rs` | 12 integration tests |

---

## Invariants Maintained

1. **Determinism**: Every sub-sprint includes a `_determinism` test verifying
   bit-identical output across runs. All implementations use `BTreeMap`,
   `f64::total_cmp`, and deterministic iteration where applicable.
2. **No regressions**: Full workspace test suite passes before and after.
3. **Zero external dependencies**: All algorithms implemented from scratch.
4. **Dual-executor parity**: All builtins registered and dispatched identically
   in both `cjc-eval` and `cjc-mir-exec`.
5. **Circular dependency safety**: GradGraph dispatch lives in executors (which
   depend on both `cjc-ad` and `cjc-runtime`), never in `builtins.rs`.
