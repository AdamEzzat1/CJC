# CJC Phase C: RL Infrastructure -- Stack Role Implementation Prompt

## Instructions for Use

This is a **master prompt** to be given to an AI coding assistant to implement
all infrastructure needed for a chess reinforcement learning demo in CJC. The
work is organized into **6 sub-sprints** (C1 through C6). Each sub-sprint should
be run as a separate conversation. Copy the relevant sub-sprint section plus the
"Context" sections into each conversation.

**CRITICAL**: All Phase A (S1-S6), Phase B (B1-B8), and hardening (H1-H18) work
is **already fully implemented**. Do NOT re-implement anything listed in the
"What Already Exists" section below. Phase C fills the **infrastructure gaps**
needed for the chess RL demo.

---

## Context (Include in EVERY sub-sprint conversation)

### What CJC Is

CJC is a deterministic scientific computing language with:
- Two parallel executors: `cjc-eval` (AST tree-walk) and `cjc-mir-exec` (MIR interpreter)
- Every builtin must be registered in **4-6 places** (the "wiring pattern")
- All floating-point reductions must use Kahan or Binned summation
- `BTreeMap`/`BTreeSet` everywhere -- no `HashMap` with random iteration order
- Same input must produce bit-identical output on every run
- 18 crates in the workspace, zero external dependencies

### Workspace Layout

```
crates/
  cjc-runtime/src/
    builtins.rs        -- shared stateless builtin dispatch (BOTH executors call this)
    tensor.rs          -- Tensor type (~1500 lines, has matmul/attention/conv/sigmoid/tanh/etc.)
    linalg.rs          -- LU, QR, Cholesky, inverse, det, solve, eigh, svd, schur, matrix_exp
    stats.rs           -- variance, sd, median, quantile, cor, spearman, kendall, weighted, etc.
    distributions.rs   -- Normal/t/chi2/F/beta/gamma/exp/weibull CDF/PDF/PPF
    hypothesis.rs      -- t-test, chi2, ANOVA, lm, wls, tukey_hsd, mann_whitney, etc.
    ml.rs              -- loss functions, SGD, Adam, batch_norm, dropout, lr_schedules
    fft.rs             -- Cooley-Tukey FFT, IFFT, RFFT, PSD, Bluestein, 2D-FFT
    det_map.rs         -- DetMap (deterministic hash map with insertion-order iteration)
    value.rs           -- Value enum (Int, Float, Bool, String, Tensor, Array, Struct, etc.)
    complex.rs         -- ComplexF64
    sparse.rs          -- SparseCsr, SparseCoo
    buffer.rs          -- COW Buffer<T>
    lib.rs             -- pub mod declarations + re-exports
  cjc-types/src/
    effect_registry.rs -- EffectSet classification for ALL builtins
    lib.rs             -- Type enum, TypeEnv, TypeChecker
  cjc-eval/src/
    lib.rs             -- AST interpreter with is_known_builtin() list + dispatch_method()
  cjc-mir-exec/src/
    lib.rs             -- MIR executor with is_known_builtin() list + dispatch_method()
  cjc-ad/src/
    lib.rs             -- Forward (Dual) + Reverse (GradGraph) AD (~780 lines)
  cjc-data/src/
    lib.rs             -- DataFrame, TidyView, Column, joins, pivots
  cjc-repro/src/
    lib.rs             -- Rng (SplitMix64), KahanAccumulatorF64, pairwise_sum
tests/
  audit_phase_b/       -- B1-B8 integration tests (106 tests)
  rl_phase/            -- C1-C6 integration tests (this phase)
  hardening_tests/     -- H1-H18 tests
```

### Dependency Constraint (CRITICAL for C1)

```
cjc-ad  →  cjc-runtime   (cjc-ad uses Tensor from cjc-runtime)
cjc-eval →  cjc-ad + cjc-runtime + cjc-data
cjc-mir-exec → cjc-ad + cjc-runtime + cjc-data
```

**cjc-runtime CANNOT depend on cjc-ad** (circular dependency). This means:
- GradGraph builtins **CANNOT** go in `builtins.rs`
- GradGraph must use a **type-erased** Value variant: `GradGraph(Rc<RefCell<dyn Any>>)`
- GradGraph construction and method dispatch happens in **cjc-eval** and **cjc-mir-exec** directly
- This follows the same pattern as `Value::TidyView(Rc<dyn Any>)` which is dispatched via `cjc_data::tidy_dispatch`

OptimizerState (Adam/SGD) lives in `cjc-runtime/src/ml.rs`, so its constructors
CAN go in `builtins.rs`.

### What Already Exists (DO NOT re-implement)

**Phase A (S1-S6)**: All core stats, distributions, hypothesis tests, linalg,
ML utilities, FFT. See `docs/STACK_ROLE_PROMPT.md` for full list.

**Phase B (B1-B8)**: 56 additional builtins including weighted stats, rank
correlations, linalg extensions, ML training (cat, stack, topk, batch_norm,
dropout, lr_schedules, l1/l2_penalty), analyst QoL (case_when, ntile,
percent_rank, cume_dist, wls), advanced FFT & distributions (window functions,
Bluestein FFT, 2D FFT, beta/gamma/exp/weibull), non-parametric tests
(tukey_hsd, mann_whitney, kruskal_wallis, wilcoxon, bonferroni, fdr_bh,
logistic_regression), and autodiff ops (Sin, Cos, Sqrt, Pow, Sigmoid, Relu, Tanh).

**Loss functions already wired**: `mse_loss`, `cross_entropy_loss`,
`binary_cross_entropy`, `huber_loss`, `hinge_loss` -- these are in `builtins.rs`
and callable from CJC. Do NOT add these again.

**Sort already wired**: `sort(array)` exists in `builtins.rs` for arrays.

**GradGraph Rust API exists** (in `cjc-ad/src/lib.rs`) with these public methods:
`new`, `input`, `parameter`, `add`, `sub`, `mul`, `matmul`, `sum`, `mean`,
`sin`, `cos`, `sqrt`, `pow`, `sigmoid`, `relu`, `tanh_act`, `value`, `tensor`,
`set_tensor`, `grad`, `zero_grad`, `backward`. GradOp variants also exist for
`Div`, `Neg`, `ScalarMul`, `Exp`, `Ln` but these lack public forward methods.

### The Wiring Pattern (CRITICAL -- follow for every new builtin)

**For builtins whose implementation is in cjc-runtime** (C2 constructors, C3-C6):

**1. Implementation module** (e.g., `crates/cjc-runtime/src/builtins.rs`):
   - Write the dispatch arm or implementation function
   - Add `#[cfg(test)] mod tests` with unit tests
   - Use `KahanAccumulatorF64` from `cjc_repro` for any summation

**2. `crates/cjc-runtime/src/builtins.rs`** -- add dispatch arm:
```rust
"bit_and" => {
    if args.len() != 2 { return Err("bit_and requires 2 arguments".into()); }
    let a = match &args[0] { Value::Int(i) => *i, _ => return Err("bit_and: expected Int".into()) };
    let b = match &args[1] { Value::Int(i) => *i, _ => return Err("bit_and: expected Int".into()) };
    Ok(Some(Value::Int(a & b)))
}
```

**3. `crates/cjc-mir-exec/src/lib.rs`** -- add to `is_known_builtin()`:
```rust
| "bit_and"
```

**4. `crates/cjc-eval/src/lib.rs`** -- add to `is_known_builtin()`:
```rust
| "bit_and"
```

**5. `crates/cjc-types/src/effect_registry.rs`** -- add effect classification:
```rust
m.insert("bit_and", pure);
```

### Phase C Wiring Differences (CRITICAL for C1 and C2)

**For GradGraph builtins** (C1): These bypass `builtins.rs` entirely.

**Constructor** -- dispatched in each executor's `dispatch_call()` method,
between the shared builtins fallthrough and the CSV builtins:

```rust
// In cjc-mir-exec/src/lib.rs dispatch_call(), AFTER Ok(None) from dispatch_builtin:
"GradGraph.new" => {
    use std::any::Any;
    let g = cjc_ad::GradGraph::new();
    let erased: Rc<RefCell<dyn Any>> = Rc::new(RefCell::new(g));
    return Ok(Value::GradGraph(erased));
}
```

**Method dispatch** -- in each executor's `dispatch_method()`, pattern-match on
the type-erased Value variant and downcast:

```rust
(Value::GradGraph(inner), "parameter") => {
    if args.len() != 1 { return Err("GradGraph.parameter requires 1 argument".into()); }
    let t = match &args[0] {
        Value::Tensor(t) => t.clone(),
        _ => return Err("GradGraph.parameter: expected Tensor".into()),
    };
    let mut borrow = inner.borrow_mut();
    let graph = borrow.downcast_mut::<cjc_ad::GradGraph>().unwrap();
    Ok(Value::Int(graph.parameter(t) as i64))
}
```

**For OptimizerState builtins** (C2): Constructors go in `builtins.rs` (since
AdamState/SgdState live in cjc-runtime/ml.rs). The `opt.step()` method is
dispatched in each executor's `dispatch_method()` with downcast.

### Effect Classification Guide

| Effect | Flag | When to use |
|--------|------|-------------|
| PURE | `pure` | No side effects, no allocation |
| ALLOC | `alloc` | Allocates new Value (arrays, strings, tensors) |
| IO | `io` | File/network/clock/stdin access |
| NONDET | `EffectSet::new(EffectSet::NONDET)` | Result depends on external state |
| MUTATES | `mutates` | Modifies an argument in place |
| MUTATES+ALLOC | `mutates_alloc` | Modifies state AND returns new value |
| GC | `gc` | Triggers garbage collection |

### Return Value Conventions

**Scalar**: `Ok(Some(Value::Float(...)))` or `Ok(Some(Value::Int(...)))`

**Array**: `Ok(Some(Value::Array(Rc::new(vec_of_values))))`

**Tensor**: `Ok(Some(Value::Tensor(tensor)))`

**Struct** (for result objects):
```rust
let mut fields = std::collections::HashMap::new();
fields.insert("statistic".into(), Value::Float(r.statistic));
fields.insert("p_value".into(), Value::Float(r.p_value));
Ok(Some(Value::Struct { name: "ResultName".into(), fields }))
```

**Tuple** (for paired results):
```rust
Ok(Some(Value::Tuple(Rc::new(vec![Value::Float(a), Value::Float(b)]))))
```

**Void** (for mutations): `Ok(Value::Void)`

### Testing Pattern

Phase C tests go in `tests/rl_phase/` (new directory). Each sub-sprint creates
a test file `tests/rl_phase/test_c{N}_{name}.rs`:

```rust
//! Phase C test C{N}: {Description}

fn run_mir(src: &str) -> Vec<String> {
    let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

#[test]
fn c{N}_feature_name() {
    let out = run_mir(r#"
let x = bit_and(0xFF, 0x0F);
print(x);
"#);
    assert_eq!(out, vec!["15"]);
}
```

**IMPORTANT CJC syntax reminders**:
- CJC requires **semicolons** at end of statements
- Tensor creation: `Tensor.from_vec(data, shape)` NOT `tensor([[...]])`
- CJC does NOT support `.0` tuple field access -- use pattern destructuring
- CJC does NOT support `==` for Array comparison
- `Value::Struct` uses `HashMap<String, Value>` (NOT BTreeMap)

Create `tests/rl_phase/mod.rs`:
```rust
pub mod test_c1_gradgraph;
pub mod test_c2_optimizer;
pub mod test_c3_bitwise;
pub mod test_c4_tensor_index;
pub mod test_c5_map_set;
pub mod test_c6_collections;
```

Create `tests/test_rl_phase.rs`:
```rust
mod rl_phase;
```

### Determinism Rules

1. **No `HashMap` with iteration** -- use `BTreeMap` or `Vec` with deterministic ordering
2. **No `f64` as hash key** -- use integer indices or `to_bits()` for exact comparison
3. **No `par_iter()` in new code** -- sequential only
4. **Kahan summation for all reductions** -- `KahanAccumulatorF64::new()` / `.add()` / `.finalize()`
5. **Deterministic sorting** -- `sort_by(|a, b| a.total_cmp(b))` for NaN handling
6. **No `SystemTime`** -- only `datetime_now()` is allowed to be NONDET
7. **Fixed iteration order** -- for-loops over ranges, not iterators over hash structures

---

## Sub-Sprint C1: GradGraph Language API

### Goal

Expose the existing `cjc_ad::GradGraph` reverse-mode automatic differentiation
engine to CJC programs. This is the **core blocker** for the chess RL demo --
without it, a CJC user cannot build a computation graph, run backpropagation,
or access gradients from CJC source code.

After C1, a CJC program can write:
```
let g = GradGraph.new();
let W = g.parameter(Tensor.from_vec([0.5, 0.3], [2, 1]));
let x = g.input(Tensor.from_vec([1.0, 2.0], [1, 2]));
let pred = g.matmul(x, W);
let target = g.input(Tensor.from_vec([1.1], [1, 1]));
let diff = g.sub(pred, target);
let sq = g.mul(diff, diff);
let loss = g.sum(sq);
g.backward(loss);
let grad_W = g.grad(W);
print(grad_W);
```

### Step 1: Add Missing GradGraph Forward Methods

Add 5 public forward methods to `crates/cjc-ad/src/lib.rs`. The GradOp variants
and backward cases already exist -- only the public `impl GradGraph` methods are
missing.

```rust
/// Element-wise division: a / b.
/// GradOp::Div(a, b) already has backward implementation.
pub fn div(&mut self, a: usize, b: usize) -> usize {
    let a_tensor = self.nodes[a].borrow().tensor.clone();
    let b_tensor = self.nodes[b].borrow().tensor.clone();
    let result = a_tensor.div_elem(&b_tensor);
    let node = GradNode { op: GradOp::Div(a, b), tensor: result, grad: None };
    self.nodes.push(Rc::new(RefCell::new(node)));
    self.nodes.len() - 1
}

/// Element-wise negation: -a.
/// GradOp::Neg(a) already has backward implementation.
pub fn neg(&mut self, a: usize) -> usize {
    let a_tensor = self.nodes[a].borrow().tensor.clone();
    let result = a_tensor.neg();
    let node = GradNode { op: GradOp::Neg(a), tensor: result, grad: None };
    self.nodes.push(Rc::new(RefCell::new(node)));
    self.nodes.len() - 1
}

/// Scalar multiply: a * s (where s is an f64 constant).
/// GradOp::ScalarMul(a, s) already has backward implementation.
pub fn scalar_mul(&mut self, a: usize, s: f64) -> usize {
    let a_tensor = self.nodes[a].borrow().tensor.clone();
    let result = a_tensor.scalar_mul(s);
    let node = GradNode { op: GradOp::ScalarMul(a, s), tensor: result, grad: None };
    self.nodes.push(Rc::new(RefCell::new(node)));
    self.nodes.len() - 1
}

/// Element-wise exponential: exp(a).
/// GradOp::Exp(a) already has backward implementation.
pub fn exp(&mut self, a: usize) -> usize {
    let a_tensor = self.nodes[a].borrow().tensor.clone();
    let result = Tensor::from_vec(
        a_tensor.to_vec().iter().map(|x| x.exp()).collect(),
        a_tensor.shape().to_vec(),
    );
    let node = GradNode { op: GradOp::Exp(a), tensor: result, grad: None };
    self.nodes.push(Rc::new(RefCell::new(node)));
    self.nodes.len() - 1
}

/// Element-wise natural logarithm: ln(a).
/// GradOp::Ln(a) already has backward implementation.
pub fn ln(&mut self, a: usize) -> usize {
    let a_tensor = self.nodes[a].borrow().tensor.clone();
    let result = Tensor::from_vec(
        a_tensor.to_vec().iter().map(|x| x.ln()).collect(),
        a_tensor.shape().to_vec(),
    );
    let node = GradNode { op: GradOp::Ln(a), tensor: result, grad: None };
    self.nodes.push(Rc::new(RefCell::new(node)));
    self.nodes.len() - 1
}
```

Add unit tests for these 5 methods following the existing pattern in cjc-ad:
- `test_reverse_div`: d/dx(x/2) at x=6 should be 0.5
- `test_reverse_neg`: d/dx(-x) at x=3 should be -1
- `test_reverse_scalar_mul`: d/dx(3*x) at x=2 should be 3
- `test_reverse_exp`: d/dx(exp(x)) at x=1 should be exp(1)
- `test_reverse_ln`: d/dx(ln(x)) at x=2 should be 0.5

### Step 2: Add Value::GradGraph Variant

In `crates/cjc-runtime/src/value.rs`:

```rust
use std::any::Any;

// Add to the Value enum:
/// Type-erased reverse-mode AD graph. Concrete type: cjc_ad::GradGraph.
/// Uses Rc<RefCell<dyn Any>> because cjc-runtime cannot depend on cjc-ad.
/// Construction and method dispatch happen in cjc-eval and cjc-mir-exec.
GradGraph(Rc<RefCell<dyn Any>>),
```

Add to `type_name()`:
```rust
Value::GradGraph(_) => "GradGraph",
```

Add to `Display`:
```rust
Value::GradGraph(_) => write!(f, "<GradGraph>"),
```

### Step 3: Wire GradGraph into Both Executors

In **both** `crates/cjc-eval/src/lib.rs` AND `crates/cjc-mir-exec/src/lib.rs`:

**a) Add to `is_known_builtin()`**:
```rust
| "GradGraph.new"
```

**b) Add constructor to `dispatch_call()`** (after shared builtins fallthrough,
before CSV builtins):
```rust
"GradGraph.new" => {
    use std::any::Any;
    if !args.is_empty() {
        return Err("GradGraph.new takes 0 arguments".into());
    }
    let g = cjc_ad::GradGraph::new();
    let erased: Rc<RefCell<dyn Any>> = Rc::new(RefCell::new(g));
    return Ok(Value::GradGraph(erased));
}
```

**c) Add method dispatch in `dispatch_method()`** -- add this block for all
GradGraph methods. Every method follows the same downcast pattern:

```rust
// -- GradGraph method dispatch --
(Value::GradGraph(inner), method) => {
    use std::any::Any;
    match method {
        "parameter" => {
            if args.len() != 1 { return Err("parameter requires 1 arg: Tensor".into()); }
            let t = match &args[0] { Value::Tensor(t) => t.clone(), _ => return Err("expected Tensor".into()) };
            let mut borrow = inner.borrow_mut();
            let graph = borrow.downcast_mut::<cjc_ad::GradGraph>().unwrap();
            Ok(Value::Int(graph.parameter(t) as i64))
        }
        "input" => {
            if args.len() != 1 { return Err("input requires 1 arg: Tensor".into()); }
            let t = match &args[0] { Value::Tensor(t) => t.clone(), _ => return Err("expected Tensor".into()) };
            let mut borrow = inner.borrow_mut();
            let graph = borrow.downcast_mut::<cjc_ad::GradGraph>().unwrap();
            Ok(Value::Int(graph.input(t) as i64))
        }
        // Binary ops: add, sub, mul, div, matmul
        "add" | "sub" | "mul" | "div" | "matmul" => {
            if args.len() != 2 { return Err(format!("{method} requires 2 args: node_a, node_b")); }
            let a = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected Int (node index)".into()) };
            let b = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("expected Int (node index)".into()) };
            let mut borrow = inner.borrow_mut();
            let graph = borrow.downcast_mut::<cjc_ad::GradGraph>().unwrap();
            let idx = match method {
                "add" => graph.add(a, b),
                "sub" => graph.sub(a, b),
                "mul" => graph.mul(a, b),
                "div" => graph.div(a, b),
                "matmul" => graph.matmul(a, b),
                _ => unreachable!(),
            };
            Ok(Value::Int(idx as i64))
        }
        // Unary ops: neg, sum, mean, sigmoid, relu, tanh, sin, cos, sqrt, exp, ln
        "neg" | "sum" | "mean" | "sigmoid" | "relu" | "tanh" | "sin" | "cos" | "sqrt" | "exp" | "ln" => {
            if args.len() != 1 { return Err(format!("{method} requires 1 arg: node_index")); }
            let a = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected Int (node index)".into()) };
            let mut borrow = inner.borrow_mut();
            let graph = borrow.downcast_mut::<cjc_ad::GradGraph>().unwrap();
            let idx = match method {
                "neg" => graph.neg(a),
                "sum" => graph.sum(a),
                "mean" => graph.mean(a),
                "sigmoid" => graph.sigmoid(a),
                "relu" => graph.relu(a),
                "tanh" => graph.tanh_act(a),
                "sin" => graph.sin(a),
                "cos" => graph.cos(a),
                "sqrt" => graph.sqrt(a),
                "exp" => graph.exp(a),
                "ln" => graph.ln(a),
                _ => unreachable!(),
            };
            Ok(Value::Int(idx as i64))
        }
        // pow(a, n) and scalar_mul(a, s) -- unary + float arg
        "pow" => {
            if args.len() != 2 { return Err("pow requires 2 args: node_index, exponent".into()); }
            let a = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected Int".into()) };
            let n = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("expected number".into()) };
            let mut borrow = inner.borrow_mut();
            let graph = borrow.downcast_mut::<cjc_ad::GradGraph>().unwrap();
            Ok(Value::Int(graph.pow(a, n) as i64))
        }
        "scalar_mul" => {
            if args.len() != 2 { return Err("scalar_mul requires 2 args: node_index, scalar".into()); }
            let a = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected Int".into()) };
            let s = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("expected number".into()) };
            let mut borrow = inner.borrow_mut();
            let graph = borrow.downcast_mut::<cjc_ad::GradGraph>().unwrap();
            Ok(Value::Int(graph.scalar_mul(a, s) as i64))
        }
        // Backward pass
        "backward" => {
            if args.len() != 1 { return Err("backward requires 1 arg: loss_node_index".into()); }
            let loss_idx = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected Int".into()) };
            let borrow = inner.borrow();
            let graph = borrow.downcast_ref::<cjc_ad::GradGraph>().unwrap();
            graph.backward(loss_idx);
            Ok(Value::Void)
        }
        // Value/tensor/gradient access
        "value" => {
            if args.len() != 1 { return Err("value requires 1 arg: node_index".into()); }
            let idx = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected Int".into()) };
            let borrow = inner.borrow();
            let graph = borrow.downcast_ref::<cjc_ad::GradGraph>().unwrap();
            Ok(Value::Float(graph.value(idx)))
        }
        "tensor" => {
            if args.len() != 1 { return Err("tensor requires 1 arg: node_index".into()); }
            let idx = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected Int".into()) };
            let borrow = inner.borrow();
            let graph = borrow.downcast_ref::<cjc_ad::GradGraph>().unwrap();
            Ok(Value::Tensor(graph.tensor(idx)))
        }
        "grad" => {
            if args.len() != 1 { return Err("grad requires 1 arg: node_index".into()); }
            let idx = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected Int".into()) };
            let borrow = inner.borrow();
            let graph = borrow.downcast_ref::<cjc_ad::GradGraph>().unwrap();
            match graph.grad(idx) {
                Some(t) => Ok(Value::Tensor(t)),
                None => Ok(Value::Void),
            }
        }
        "set_tensor" => {
            if args.len() != 2 { return Err("set_tensor requires 2 args: node_index, tensor".into()); }
            let idx = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected Int".into()) };
            let t = match &args[1] { Value::Tensor(t) => t.clone(), _ => return Err("expected Tensor".into()) };
            let borrow = inner.borrow();
            let graph = borrow.downcast_ref::<cjc_ad::GradGraph>().unwrap();
            graph.set_tensor(idx, t);
            Ok(Value::Void)
        }
        "zero_grad" => {
            if !args.is_empty() { return Err("zero_grad takes 0 arguments".into()); }
            let borrow = inner.borrow();
            let graph = borrow.downcast_ref::<cjc_ad::GradGraph>().unwrap();
            graph.zero_grad();
            Ok(Value::Void)
        }
        _ => Err(format!("no method `{method}` on GradGraph")),
    }
}
```

### Step 4: Effect Registry

Add to `crates/cjc-types/src/effect_registry.rs`:

```rust
// Phase C1: GradGraph Language API
m.insert("GradGraph.new", alloc);
m.insert("GradGraph.parameter", mutates_alloc);
m.insert("GradGraph.input", mutates_alloc);
m.insert("GradGraph.add", mutates_alloc);
m.insert("GradGraph.sub", mutates_alloc);
m.insert("GradGraph.mul", mutates_alloc);
m.insert("GradGraph.div", mutates_alloc);
m.insert("GradGraph.neg", mutates_alloc);
m.insert("GradGraph.matmul", mutates_alloc);
m.insert("GradGraph.sum", mutates_alloc);
m.insert("GradGraph.mean", mutates_alloc);
m.insert("GradGraph.sigmoid", mutates_alloc);
m.insert("GradGraph.relu", mutates_alloc);
m.insert("GradGraph.tanh", mutates_alloc);
m.insert("GradGraph.sin", mutates_alloc);
m.insert("GradGraph.cos", mutates_alloc);
m.insert("GradGraph.sqrt", mutates_alloc);
m.insert("GradGraph.pow", mutates_alloc);
m.insert("GradGraph.exp", mutates_alloc);
m.insert("GradGraph.ln", mutates_alloc);
m.insert("GradGraph.scalar_mul", mutates_alloc);
m.insert("GradGraph.backward", mutates);
m.insert("GradGraph.value", pure);
m.insert("GradGraph.tensor", alloc);
m.insert("GradGraph.grad", alloc);
m.insert("GradGraph.set_tensor", mutates);
m.insert("GradGraph.zero_grad", mutates);
```

### Unit Tests (add to cjc-ad/src/lib.rs)

- `test_reverse_div`: d/dx(x/2) at x=6 → gradient = 0.5
- `test_reverse_neg`: d/dx(-x) at x=3 → gradient = -1.0
- `test_reverse_scalar_mul`: d/dx(3x) at x=2 → gradient = 3.0
- `test_reverse_exp`: d/dx(exp(x)) at x=1 → gradient ≈ e
- `test_reverse_ln`: d/dx(ln(x)) at x=2 → gradient = 0.5

### Integration Tests

Create `tests/rl_phase/test_c1_gradgraph.rs` -- 15+ tests:

- `c1_construct_graph`: GradGraph.new() creates without error
- `c1_forward_add`: 2 + 3 = 5 via graph
- `c1_forward_matmul`: [1,2] @ [[1],[1]] = [[3]] via graph
- `c1_backward_x_squared`: d/dx(x^2) at x=3 → grad = 6
- `c1_backward_linear`: d/dW(x @ W) for known x and W
- `c1_backward_sigmoid`: sigmoid derivative matches formula
- `c1_backward_relu`: relu gradient is 1 for positive, 0 for negative
- `c1_backward_chain_rule`: d/dx(sin(x^2)) via chain rule
- `c1_gradient_descent_step`: W -= lr * grad reduces loss
- `c1_parameter_update_cycle`: set_tensor + forward + backward cycle
- `c1_multi_param_net`: two-layer network with matmul + relu + sum
- `c1_zero_grad_reset`: verify gradients reset to zero
- `c1_exp_ln_roundtrip`: exp(ln(x)) ≈ x, gradients cancel
- `c1_scalar_mul_grad`: d/dx(3x) = 3
- `c1_determinism`: double-run produces bit-identical gradients

### Validation

```
cargo test -p cjc-ad                            # unit tests (new + existing)
cargo test --test test_rl_phase -- c1            # integration tests
cargo test --workspace                           # 0 regressions
```

---

## Sub-Sprint C2: Optimizer & Loss Builtins

### Goal

Expose Adam and SGD optimizers as callable CJC objects so users can train neural
networks entirely from CJC source code. Combined with C1's GradGraph API, this
enables a complete training loop: forward → backward → optimizer step → repeat.

After C2, a CJC program can write:
```
let opt = Adam.new(2, 0.001);
let params = Tensor.from_vec([1.0, 2.0], [2]);
let grads = Tensor.from_vec([0.1, 0.2], [2]);
let new_params = opt.step(params, grads);
print(new_params);
```

### Step 1: Add Value::OptimizerState Variant

In `crates/cjc-runtime/src/value.rs`:

```rust
/// Type-erased optimizer state. Concrete types: AdamState or SgdState (from ml.rs).
/// Uses Rc<RefCell<dyn Any>> for interior mutability (step updates internal state).
OptimizerState(Rc<RefCell<dyn Any>>),
```

Add to `type_name()`: `"OptimizerState"`
Add to `Display`: `"<OptimizerState>"`

### Step 2: Add Constructors to builtins.rs

In `crates/cjc-runtime/src/builtins.rs`:

```rust
"Adam.new" => {
    if args.len() != 2 { return Err("Adam.new requires 2 args: n_params, lr".into()); }
    let n = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected Int".into()) };
    let lr = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("expected number".into()) };
    Ok(Some(Value::OptimizerState(Rc::new(RefCell::new(
        crate::ml::AdamState::new(n, lr)
    )))))
}
"Sgd.new" => {
    if args.len() != 3 { return Err("Sgd.new requires 3 args: n_params, lr, momentum".into()); }
    let n = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected Int".into()) };
    let lr = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("expected number".into()) };
    let momentum = match &args[2] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("expected number".into()) };
    Ok(Some(Value::OptimizerState(Rc::new(RefCell::new(
        crate::ml::SgdState::new(n, lr, momentum)
    )))))
}
```

### Step 3: Add Method Dispatch

In **both** executors' `dispatch_method()`:

```rust
(Value::OptimizerState(inner), "step") => {
    if args.len() != 2 { return Err("step requires 2 args: params_tensor, grads_tensor".into()); }
    let params_t = match &args[0] { Value::Tensor(t) => t.clone(), _ => return Err("expected Tensor".into()) };
    let grads_t = match &args[1] { Value::Tensor(t) => t.clone(), _ => return Err("expected Tensor".into()) };
    let mut params_data = params_t.to_vec();
    let grads_data = grads_t.to_vec();
    let mut borrow = inner.borrow_mut();
    // Try Adam first, then SGD
    if let Some(adam) = borrow.downcast_mut::<cjc_runtime::ml::AdamState>() {
        cjc_runtime::ml::adam_step(&mut params_data, &grads_data, adam);
    } else if let Some(sgd) = borrow.downcast_mut::<cjc_runtime::ml::SgdState>() {
        cjc_runtime::ml::sgd_step(&mut params_data, &grads_data, sgd);
    } else {
        return Err("unknown optimizer type".into());
    }
    Ok(Value::Tensor(Tensor::from_vec(params_data, params_t.shape().to_vec())))
}
```

### Step 4: Wire into is_known_builtin and effect_registry

```rust
// is_known_builtin (both executors):
| "Adam.new"
| "Sgd.new"

// effect_registry:
m.insert("Adam.new", alloc);
m.insert("Sgd.new", alloc);
m.insert("OptimizerState.step", mutates_alloc);
```

### Integration Tests

Create `tests/rl_phase/test_c2_optimizer.rs` -- 10+ tests:

- `c2_adam_new`: creates Adam optimizer without error
- `c2_sgd_new`: creates SGD optimizer without error
- `c2_adam_basic_step`: one step reduces parameter in gradient direction
- `c2_adam_bias_correction`: verify bias correction activates early steps
- `c2_sgd_momentum`: verify momentum accumulation across steps
- `c2_sgd_zero_momentum`: SGD with momentum=0 is vanilla gradient descent
- `c2_optimizer_determinism`: double-run produces identical params
- `c2_multi_step_convergence`: 100 steps on simple quadratic converges
- `c2_param_size_mismatch`: wrong tensor size → error
- `c2_full_train_loop`: GradGraph + Adam end-to-end training loop

### Validation

```
cargo test -p cjc-runtime                        # unit tests
cargo test --test test_rl_phase -- c2             # integration tests
cargo test --workspace                            # 0 regressions
```

---

## Sub-Sprint C3: Bitwise Operations

### Goal

Add 7 bitwise operations on integers. These enable efficient bitboard
representations for chess (each board position encoded as a 64-bit integer).

### Functions to Implement

All go directly in `crates/cjc-runtime/src/builtins.rs` (trivial one-liners):

```rust
/// Bitwise AND of two i64 values.
"bit_and" => {
    if args.len() != 2 { return Err("bit_and requires 2 arguments".into()); }
    let a = match &args[0] { Value::Int(i) => *i, _ => return Err("bit_and: expected Int".into()) };
    let b = match &args[1] { Value::Int(i) => *i, _ => return Err("bit_and: expected Int".into()) };
    Ok(Some(Value::Int(a & b)))
}

/// Bitwise OR of two i64 values.
"bit_or" => {
    if args.len() != 2 { return Err("bit_or requires 2 arguments".into()); }
    let a = match &args[0] { Value::Int(i) => *i, _ => return Err("bit_or: expected Int".into()) };
    let b = match &args[1] { Value::Int(i) => *i, _ => return Err("bit_or: expected Int".into()) };
    Ok(Some(Value::Int(a | b)))
}

/// Bitwise XOR of two i64 values.
"bit_xor" => {
    if args.len() != 2 { return Err("bit_xor requires 2 arguments".into()); }
    let a = match &args[0] { Value::Int(i) => *i, _ => return Err("bit_xor: expected Int".into()) };
    let b = match &args[1] { Value::Int(i) => *i, _ => return Err("bit_xor: expected Int".into()) };
    Ok(Some(Value::Int(a ^ b)))
}

/// Bitwise NOT (complement) of an i64 value.
"bit_not" => {
    if args.len() != 1 { return Err("bit_not requires 1 argument".into()); }
    let a = match &args[0] { Value::Int(i) => *i, _ => return Err("bit_not: expected Int".into()) };
    Ok(Some(Value::Int(!a)))
}

/// Left shift: interpret a as u64, shift left by n bits. n must be 0-63.
"bit_shl" => {
    if args.len() != 2 { return Err("bit_shl requires 2 arguments".into()); }
    let a = match &args[0] { Value::Int(i) => *i, _ => return Err("bit_shl: expected Int".into()) };
    let n = match &args[1] { Value::Int(i) => *i, _ => return Err("bit_shl: expected Int".into()) };
    if n < 0 || n > 63 { return Err("bit_shl: shift amount must be 0-63".into()); }
    Ok(Some(Value::Int(((a as u64) << (n as u32)) as i64)))
}

/// Right shift (logical): interpret a as u64, shift right by n bits. n must be 0-63.
"bit_shr" => {
    if args.len() != 2 { return Err("bit_shr requires 2 arguments".into()); }
    let a = match &args[0] { Value::Int(i) => *i, _ => return Err("bit_shr: expected Int".into()) };
    let n = match &args[1] { Value::Int(i) => *i, _ => return Err("bit_shr: expected Int".into()) };
    if n < 0 || n > 63 { return Err("bit_shr: shift amount must be 0-63".into()); }
    Ok(Some(Value::Int(((a as u64) >> (n as u32)) as i64)))
}

/// Population count: number of 1-bits in the integer (treating as u64).
"popcount" => {
    if args.len() != 1 { return Err("popcount requires 1 argument".into()); }
    let a = match &args[0] { Value::Int(i) => *i, _ => return Err("popcount: expected Int".into()) };
    Ok(Some(Value::Int((a as u64).count_ones() as i64)))
}
```

### Builtin Names for Wiring

| Function | Builtin name | Args | Returns | Effect |
|----------|-------------|------|---------|--------|
| `bit_and(a, b)` | `"bit_and"` | 2 (Int, Int) | Int | PURE |
| `bit_or(a, b)` | `"bit_or"` | 2 (Int, Int) | Int | PURE |
| `bit_xor(a, b)` | `"bit_xor"` | 2 (Int, Int) | Int | PURE |
| `bit_not(a)` | `"bit_not"` | 1 (Int) | Int | PURE |
| `bit_shl(a, n)` | `"bit_shl"` | 2 (Int, Int) | Int | PURE |
| `bit_shr(a, n)` | `"bit_shr"` | 2 (Int, Int) | Int | PURE |
| `popcount(a)` | `"popcount"` | 1 (Int) | Int | PURE |

### Unit Tests

- `test_bit_and_mask`: `bit_and(0xFF, 0x0F)` → 15
- `test_bit_or_combine`: `bit_or(0xF0, 0x0F)` → 255
- `test_bit_xor_toggle`: `bit_xor(0xFF, 0xFF)` → 0
- `test_bit_not_complement`: `bit_not(0)` → -1 (all bits set in i64)
- `test_bit_shl_basic`: `bit_shl(1, 3)` → 8
- `test_bit_shr_basic`: `bit_shr(8, 3)` → 1
- `test_bit_shl_overflow_error`: `bit_shl(1, 64)` → error
- `test_popcount_allones`: `popcount(-1)` → 64 (all bits set)
- `test_popcount_zero`: `popcount(0)` → 0
- `test_popcount_powers`: `popcount(bit_shl(1, 5))` → 1

### Integration Tests

Create `tests/rl_phase/test_c3_bitwise.rs` -- 10+ tests:

- `c3_and_mask`: `print(bit_and(255, 15));` → "15"
- `c3_or_combine`: `print(bit_or(240, 15));` → "255"
- `c3_xor_toggle`: `print(bit_xor(255, 255));` → "0"
- `c3_not_zero`: `print(bit_not(0));` → "-1"
- `c3_shl_one`: `print(bit_shl(1, 3));` → "8"
- `c3_shr_eight`: `print(bit_shr(8, 3));` → "1"
- `c3_popcount_full`: `print(popcount(-1));` → "64"
- `c3_popcount_byte`: `print(popcount(255));` → "8"
- `c3_chess_bitboard`: test setting/checking specific board squares
- `c3_determinism`: double-run produces identical results

### Validation

```
cargo test -p cjc-runtime                        # unit tests
cargo test --test test_rl_phase -- c3             # integration tests
cargo test --workspace                            # 0 regressions
```

---

## Sub-Sprint C4: Sorting & Tensor Indexing

### Goal

Add `argsort`, `gather`, `scatter`, and `index_select` for efficient move
ordering and feature vector construction. `argsort` returns indices that
would sort a tensor, enabling move ordering by evaluation score. `gather` and
`index_select` enable efficient embedding lookups (board position → feature).

### Functions to Implement

Add to `crates/cjc-runtime/src/tensor.rs`:

```rust
impl Tensor {
    /// Return indices that would sort the 1D tensor in ascending order.
    /// Uses stable sort with total_cmp for deterministic NaN handling.
    /// Returns a new Tensor of f64 indices.
    pub fn argsort(&self) -> Result<Tensor, String> {
        if self.shape().len() != 1 {
            return Err("argsort: only 1D tensors supported".into());
        }
        let data = self.to_vec();
        let mut indices: Vec<usize> = (0..data.len()).collect();
        indices.sort_by(|&a, &b| data[a].total_cmp(&data[b]));
        let result: Vec<f64> = indices.iter().map(|&i| i as f64).collect();
        Ok(Tensor::from_vec(result, vec![data.len()]))
    }

    /// Gather elements along a dimension using index tensor.
    /// For 1D: output[i] = self[index[i]]
    /// For 2D dim=0: output[i][j] = self[index[i][j]][j]
    /// For 2D dim=1: output[i][j] = self[i][index[i][j]]
    pub fn gather(&self, dim: usize, index: &Tensor) -> Result<Tensor, String>

    /// Scatter source values into a new tensor at positions given by index.
    /// Inverse of gather: output[index[i]] = src[i] for 1D.
    pub fn scatter(&self, dim: usize, index: &Tensor, src: &Tensor) -> Result<Tensor, String>

    /// Select slices along a dimension using a 1D index tensor.
    /// For 2D dim=0: selects rows at given indices.
    /// For 2D dim=1: selects columns at given indices.
    pub fn index_select(&self, dim: usize, index: &Tensor) -> Result<Tensor, String>
}
```

### Builtin Names for Wiring

| Function | Builtin name | Args | Returns | Effect |
|----------|-------------|------|---------|--------|
| `argsort(tensor)` | `"argsort"` | 1 (Tensor) | Tensor | ALLOC |
| `gather(tensor, dim, indices)` | `"gather"` | 3 (Tensor, Int, Tensor) | Tensor | ALLOC |
| `scatter(tensor, dim, indices, src)` | `"scatter"` | 4 (Tensor, Int, Tensor, Tensor) | Tensor | ALLOC |
| `index_select(tensor, dim, indices)` | `"index_select"` | 3 (Tensor, Int, Tensor) | Tensor | ALLOC |

### Dispatch in builtins.rs

```rust
"argsort" => {
    if args.len() != 1 { return Err("argsort requires 1 argument".into()); }
    let t = match &args[0] { Value::Tensor(t) => t, _ => return Err("argsort: expected Tensor".into()) };
    Ok(Some(Value::Tensor(t.argsort()?)))
}
"gather" => {
    if args.len() != 3 { return Err("gather requires 3 arguments: tensor, dim, indices".into()); }
    let t = match &args[0] { Value::Tensor(t) => t, _ => return Err("expected Tensor".into()) };
    let dim = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("expected Int".into()) };
    let idx = match &args[2] { Value::Tensor(t) => t, _ => return Err("expected Tensor".into()) };
    Ok(Some(Value::Tensor(t.gather(dim, idx)?)))
}
"scatter" => {
    if args.len() != 4 { return Err("scatter requires 4 arguments: tensor, dim, indices, src".into()); }
    let t = match &args[0] { Value::Tensor(t) => t, _ => return Err("expected Tensor".into()) };
    let dim = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("expected Int".into()) };
    let idx = match &args[2] { Value::Tensor(t) => t, _ => return Err("expected Tensor".into()) };
    let src = match &args[3] { Value::Tensor(t) => t, _ => return Err("expected Tensor".into()) };
    Ok(Some(Value::Tensor(t.scatter(dim, idx, src)?)))
}
"index_select" => {
    if args.len() != 3 { return Err("index_select requires 3 arguments: tensor, dim, indices".into()); }
    let t = match &args[0] { Value::Tensor(t) => t, _ => return Err("expected Tensor".into()) };
    let dim = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("expected Int".into()) };
    let idx = match &args[2] { Value::Tensor(t) => t, _ => return Err("expected Tensor".into()) };
    Ok(Some(Value::Tensor(t.index_select(dim, idx)?)))
}
```

### Unit Tests (add to tensor.rs)

- `test_argsort_ascending`: already sorted → [0,1,2,3,4]
- `test_argsort_descending`: reversed → [4,3,2,1,0]
- `test_argsort_with_ties`: stable sort preserves order of equal elements
- `test_argsort_determinism`: double-run → bit-identical
- `test_gather_1d`: gather([10,20,30], indices=[2,0,1]) → [30,10,20]
- `test_gather_2d_rows`: gather along dim=0
- `test_scatter_1d`: scatter into zeros at given indices
- `test_index_select_rows`: select rows 0 and 2 from 3x3 matrix
- `test_index_select_cols`: select columns 1 and 2

### Integration Tests

Create `tests/rl_phase/test_c4_tensor_index.rs` -- 10+ tests:

- `c4_argsort_basic`: sort indices match expected order
- `c4_argsort_descending`: verify reverse sort via index reversal
- `c4_gather_1d`: gather elements by index
- `c4_gather_2d`: gather from 2D tensor
- `c4_scatter_1d`: scatter values into zeros
- `c4_index_select_rows`: row selection from matrix
- `c4_index_select_cols`: column selection from matrix
- `c4_argsort_then_gather_sorts`: `gather(t, 0, argsort(t))` equals `sort(t)`
- `c4_index_select_identity`: selecting all indices returns original
- `c4_determinism`: double-run produces identical results

### Validation

```
cargo test -p cjc-runtime tensor                 # unit tests
cargo test --test test_rl_phase -- c4             # integration tests
cargo test --workspace                            # 0 regressions
```

---

## Sub-Sprint C5: Map Completion & Set Type

### Goal

Complete the Map builtin dispatch (operations are registered but lack
implementation) and add a lightweight Set type backed by DetMap. These enable
transposition tables and visited-state tracking for game search.

### Step 1: Wire Map Constructor and Methods

Map operations are registered in `is_known_builtin()` and `effect_registry.rs`
but have **no dispatch implementation**. Add dispatch:

**Constructor in `builtins.rs`**:

```rust
"Map.new" => {
    if !args.is_empty() { return Err("Map.new takes 0 arguments".into()); }
    Ok(Some(Value::Map(Rc::new(RefCell::new(
        crate::det_map::DetMap::new()
    )))))
}
```

**Method dispatch in both executors' `dispatch_method()`**:

```rust
(Value::Map(m), method) => {
    match method {
        "insert" => {
            if args.len() != 2 { return Err("insert requires 2 args: key, value".into()); }
            m.borrow_mut().insert(args[0].clone(), args[1].clone());
            Ok(Value::Void)
        }
        "get" => {
            if args.len() != 1 { return Err("get requires 1 arg: key".into()); }
            match m.borrow().get(&args[0]) {
                Some(v) => Ok(v.clone()),
                None => Ok(Value::Void),
            }
        }
        "remove" => {
            if args.len() != 1 { return Err("remove requires 1 arg: key".into()); }
            m.borrow_mut().remove(&args[0]);
            Ok(Value::Void)
        }
        "len" => {
            Ok(Value::Int(m.borrow().len() as i64))
        }
        "contains_key" => {
            if args.len() != 1 { return Err("contains_key requires 1 arg: key".into()); }
            Ok(Value::Bool(m.borrow().get(&args[0]).is_some()))
        }
        "keys" => {
            let keys: Vec<Value> = m.borrow().iter().map(|(k, _)| k.clone()).collect();
            Ok(Value::Array(Rc::new(keys)))
        }
        "values" => {
            let vals: Vec<Value> = m.borrow().iter().map(|(_, v)| v.clone()).collect();
            Ok(Value::Array(Rc::new(vals)))
        }
        _ => Err(format!("no method `{method}` on Map")),
    }
}
```

NOTE: Check if `DetMap` has `get`, `insert`, `remove`, `iter` methods. If any are
missing, add them to `crates/cjc-runtime/src/det_map.rs` before wiring dispatch.

### Step 2: Add Set Type

A Set is simply a `Map` where all values are `Value::Void`. No new Value variant
needed -- Set operations dispatch on `Value::Map`.

**Constructor in `builtins.rs`**:

```rust
"Set.new" => {
    if !args.is_empty() { return Err("Set.new takes 0 arguments".into()); }
    Ok(Some(Value::Map(Rc::new(RefCell::new(
        crate::det_map::DetMap::new()
    )))))
}
```

**Method dispatch** -- add to the `(Value::Map(m), method)` match block:

```rust
"add" => {
    if args.len() != 1 { return Err("add requires 1 arg: value".into()); }
    m.borrow_mut().insert(args[0].clone(), Value::Void);
    Ok(Value::Void)
}
"contains" => {
    if args.len() != 1 { return Err("contains requires 1 arg: value".into()); }
    Ok(Value::Bool(m.borrow().get(&args[0]).is_some()))
}
"to_array" => {
    let keys: Vec<Value> = m.borrow().iter().map(|(k, _)| k.clone()).collect();
    Ok(Value::Array(Rc::new(keys)))
}
```

### Builtin Names for Wiring

| Function | Builtin name | Args | Returns | Effect |
|----------|-------------|------|---------|--------|
| `Map.new()` | `"Map.new"` | 0 | Map | ALLOC |
| `m.insert(k, v)` | method | 2 | Void | MUTATES |
| `m.get(k)` | method | 1 | Value/Void | PURE |
| `m.remove(k)` | method | 1 | Void | MUTATES |
| `m.len()` | method | 0 | Int | PURE |
| `m.contains_key(k)` | method | 1 | Bool | PURE |
| `m.keys()` | method | 0 | Array | ALLOC |
| `m.values()` | method | 0 | Array | ALLOC |
| `Set.new()` | `"Set.new"` | 0 | Map | ALLOC |
| `s.add(v)` | method | 1 | Void | MUTATES |
| `s.contains(v)` | method | 1 | Bool | PURE |
| `s.to_array()` | method | 0 | Array | ALLOC |

Add to `effect_registry.rs`:
```rust
// Phase C5: Map Completion & Set Type
// Map.new, Map.insert, Map.get, Map.remove, Map.len, Map.contains_key already registered
// Add any missing:
m.insert("Map.keys", alloc);
m.insert("Map.values", alloc);
m.insert("Set.new", alloc);
```

### Integration Tests

Create `tests/rl_phase/test_c5_map_set.rs` -- 12+ tests:

- `c5_map_new`: creates empty map
- `c5_map_insert_get`: insert and retrieve a value
- `c5_map_overwrite`: inserting same key overwrites
- `c5_map_remove`: remove deletes entry
- `c5_map_contains_key`: true for present, false for absent
- `c5_map_len`: tracks count accurately
- `c5_map_keys_values`: returns arrays of keys and values
- `c5_set_new`: creates empty set
- `c5_set_add_contains`: add element, check contains
- `c5_set_dedup`: adding same element twice, len still 1
- `c5_set_to_array`: converts to array
- `c5_determinism`: double-run produces identical results

### Validation

```
cargo test -p cjc-runtime                        # unit tests
cargo test --test test_rl_phase -- c5             # integration tests
cargo test --workspace                            # 0 regressions
```

---

## Sub-Sprint C6: I/O & Collection Utilities

### Goal

Add `read_line` for interactive play and array utility builtins for game state
management (push, pop, contains, reverse, flatten, len, slice).

### Functions to Implement

**I/O** -- dispatched in each executor's `dispatch_call()` (needs interpreter
state for stdout capture, similar to `print`):

```rust
"read_line" => {
    use std::io::BufRead;
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line)
        .map_err(|e| format!("read_line: {e}"))?;
    // Remove trailing newline
    if line.ends_with('\n') { line.pop(); }
    if line.ends_with('\r') { line.pop(); }
    return Ok(Value::String(Rc::new(line)));
}
```

**Array utilities** -- all in `builtins.rs`:

```rust
/// Append a value to the end of an array, returning a new array.
"array_push" => {
    if args.len() != 2 { return Err("array_push requires 2 args: array, value".into()); }
    let arr = match &args[0] { Value::Array(a) => a.as_ref().clone(), _ => return Err("expected Array".into()) };
    let mut new_arr = arr;
    new_arr.push(args[1].clone());
    Ok(Some(Value::Array(Rc::new(new_arr))))
}

/// Remove and return the last element as (last, remaining_array).
/// Error if array is empty.
"array_pop" => {
    if args.len() != 1 { return Err("array_pop requires 1 argument".into()); }
    let arr = match &args[0] { Value::Array(a) => a.as_ref().clone(), _ => return Err("expected Array".into()) };
    if arr.is_empty() { return Err("array_pop: empty array".into()); }
    let mut new_arr = arr;
    let last = new_arr.pop().unwrap();
    Ok(Some(Value::Tuple(Rc::new(vec![last, Value::Array(Rc::new(new_arr))]))))
}

/// Check if an array contains a value (uses Display-based equality).
"array_contains" => {
    if args.len() != 2 { return Err("array_contains requires 2 args: array, value".into()); }
    let arr = match &args[0] { Value::Array(a) => a, _ => return Err("expected Array".into()) };
    let needle = format!("{}", args[1]);
    let found = arr.iter().any(|v| format!("{v}") == needle);
    Ok(Some(Value::Bool(found)))
}

/// Reverse an array, returning a new array.
"array_reverse" => {
    if args.len() != 1 { return Err("array_reverse requires 1 argument".into()); }
    let arr = match &args[0] { Value::Array(a) => a.as_ref().clone(), _ => return Err("expected Array".into()) };
    let mut reversed = arr;
    reversed.reverse();
    Ok(Some(Value::Array(Rc::new(reversed))))
}

/// Flatten one level of nesting: [[1,2],[3]] → [1,2,3].
"array_flatten" => {
    if args.len() != 1 { return Err("array_flatten requires 1 argument".into()); }
    let arr = match &args[0] { Value::Array(a) => a, _ => return Err("expected Array".into()) };
    let mut flat = Vec::new();
    for item in arr.iter() {
        match item {
            Value::Array(inner) => flat.extend(inner.iter().cloned()),
            other => flat.push(other.clone()),
        }
    }
    Ok(Some(Value::Array(Rc::new(flat))))
}

/// Length of an array.
"array_len" => {
    if args.len() != 1 { return Err("array_len requires 1 argument".into()); }
    let arr = match &args[0] { Value::Array(a) => a, _ => return Err("expected Array".into()) };
    Ok(Some(Value::Int(arr.len() as i64)))
}

/// Slice an array from start (inclusive) to end (exclusive).
"array_slice" => {
    if args.len() != 3 { return Err("array_slice requires 3 args: array, start, end".into()); }
    let arr = match &args[0] { Value::Array(a) => a, _ => return Err("expected Array".into()) };
    let start = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("expected Int".into()) };
    let end = match &args[2] { Value::Int(i) => *i as usize, _ => return Err("expected Int".into()) };
    if start > end || end > arr.len() {
        return Err(format!("array_slice: invalid range [{start}, {end}) for array of len {}", arr.len()));
    }
    Ok(Some(Value::Array(Rc::new(arr[start..end].to_vec()))))
}
```

### Builtin Names for Wiring

| Function | Builtin name | Args | Returns | Effect |
|----------|-------------|------|---------|--------|
| `read_line()` | `"read_line"` | 0 | String | IO+NONDET |
| `array_push(arr, val)` | `"array_push"` | 2 | Array | ALLOC |
| `array_pop(arr)` | `"array_pop"` | 1 | Tuple(Value, Array) | ALLOC |
| `array_contains(arr, val)` | `"array_contains"` | 2 | Bool | PURE |
| `array_reverse(arr)` | `"array_reverse"` | 1 | Array | ALLOC |
| `array_flatten(arr)` | `"array_flatten"` | 1 | Array | ALLOC |
| `array_len(arr)` | `"array_len"` | 1 | Int | PURE |
| `array_slice(arr, start, end)` | `"array_slice"` | 3 | Array | ALLOC |

Effect registry:
```rust
// Phase C6: I/O & Collection Utilities
let io_nondet = EffectSet::new(EffectSet::IO | EffectSet::NONDET);
m.insert("read_line", io_nondet);
m.insert("array_push", alloc);
m.insert("array_pop", alloc);
m.insert("array_contains", pure);
m.insert("array_reverse", alloc);
m.insert("array_flatten", alloc);
m.insert("array_len", pure);
m.insert("array_slice", alloc);
```

### Integration Tests

Create `tests/rl_phase/test_c6_collections.rs` -- 10+ tests:

- `c6_array_push`: `array_push([1,2], 3)` → [1,2,3]
- `c6_array_pop`: `array_pop([1,2,3])` → (3, [1,2])
- `c6_array_pop_empty_error`: `array_pop([])` → error
- `c6_array_contains_found`: `array_contains([1,2,3], 2)` → true
- `c6_array_contains_missing`: `array_contains([1,2,3], 4)` → false
- `c6_array_reverse`: `array_reverse([1,2,3])` → [3,2,1]
- `c6_array_flatten`: `array_flatten([[1,2],[3]])` → [1,2,3]
- `c6_array_len`: `array_len([1,2,3])` → 3
- `c6_array_slice`: `array_slice([1,2,3,4], 1, 3)` → [2,3]
- `c6_determinism`: double-run produces identical results

Note: `read_line` cannot be easily tested through MIR executor (needs stdin).
Test it manually or with a mock. The integration test file should include a
comment noting this limitation.

### Validation

```
cargo test -p cjc-runtime                        # unit tests
cargo test --test test_rl_phase -- c6             # integration tests
cargo test --workspace                            # 0 regressions
```

---

## Test Plan Summary

### Directory Structure

```
tests/
  rl_phase/
    mod.rs                        -- pub mod declarations for C1-C6
    test_c1_gradgraph.rs          -- 15+ tests
    test_c2_optimizer.rs          -- 10+ tests
    test_c3_bitwise.rs            -- 10+ tests
    test_c4_tensor_index.rs       -- 10+ tests
    test_c5_map_set.rs            -- 12+ tests
    test_c6_collections.rs        -- 10+ tests
  test_rl_phase.rs                -- mod rl_phase;
```

### Test Categories per File

Each test file includes:
1. **Correctness tests**: output matches known values
2. **Edge case tests**: empty input, zero values, boundary conditions
3. **Determinism tests**: run twice → assert bit-identical output
4. **Error tests**: invalid arguments produce clear error messages

### Expected Test Count

| Sub-Sprint | Min Integration Tests | Min Unit Tests |
|------------|----------------------|----------------|
| C1 | 15 | 5 |
| C2 | 10 | 0 |
| C3 | 10 | 10 |
| C4 | 10 | 9 |
| C5 | 12 | 0 |
| C6 | 10 | 0 |
| **Total** | **67+** | **24+** |

---

## Regression Testing Instructions

After EACH sub-sprint, run:

```bash
# Unit tests for modified crates:
cargo test -p cjc-runtime        # builtins, tensor, ml, det_map
cargo test -p cjc-ad             # GradGraph additions (C1 only)

# Phase C integration tests:
cargo test --test test_rl_phase

# All existing tests must remain green:
cargo test --test test_hardening          # H1-H18
cargo test --test test_audit_phase_b      # B1-B8

# Full workspace regression:
cargo test --workspace

# Expected: 0 failures, all previous tests still passing
```

If any existing test breaks, **fix the regression before proceeding**. Phase C
additions are purely additive -- no behavioral change to existing functions.

---

## Documentation Requirements

After all 6 sub-sprints, create:

1. **Create `docs/phase_c_changelog.md`**: List every new builtin by sub-sprint
   with brief descriptions, test counts, and files modified.

2. **Include invariants maintained**: determinism, zero regressions, zero
   external dependencies, dual-executor parity.

---

## Final Validation (after all sub-sprints)

```bash
# Full workspace must pass with 0 failures:
cargo test --workspace

# Run ALL test suites:
cargo test --test test_hardening         # H1-H18 all green
cargo test --test test_audit_phase_b     # B1-B8 all green
cargo test --test test_rl_phase          # C1-C6 all green

# Expected: 0 failures across the board
```

### Post-Phase C Checklist

- [ ] Every new builtin appears in `is_known_builtin()` in BOTH executors
- [ ] Every new builtin has an effect classification in `effect_registry.rs`
- [ ] Every new function has MIR-executor integration tests
- [ ] GradGraph methods dispatched identically in cjc-eval and cjc-mir-exec
- [ ] OptimizerState methods dispatched identically in cjc-eval and cjc-mir-exec
- [ ] Map methods dispatched identically in cjc-eval and cjc-mir-exec
- [ ] No `HashMap` with iteration in any new code
- [ ] All new sorting uses `f64::total_cmp`
- [ ] GradGraph operations produce bit-identical results on double-run
- [ ] Optimizer steps produce bit-identical results on double-run
- [ ] Bitwise operations are pure and have no allocation
- [ ] `cargo test --workspace` passes with 0 failures
- [ ] Documentation updated in `docs/phase_c_changelog.md`
