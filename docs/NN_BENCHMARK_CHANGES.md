# Neural Network Benchmark: Required CJC Changes

## Context

A benchmark suite was built to evaluate base CJC (no external libraries) against
Python + NumPy on training a fully-connected Multi-Layer Perceptron (MLP) with
manual backpropagation. This document catalogs every gap the benchmark exposed
and the changes made to close them.

**Neural network type**: Fully-connected MLP (Multi-Layer Perceptron)
- Stacked linear layers: `D -> H -> H -> ... -> H -> O`
- ReLU activation on hidden layers
- MSE (Mean Squared Error) loss
- Vanilla SGD optimizer
- Manual backpropagation (no automatic differentiation)

---

## 1. No `.transpose()` Exposed to CJC Code — CRITICAL

### Problem

Backpropagation requires matrix transposition in every layer:
- `dW = X^T @ dZ` — weight gradient needs input transposed
- `dX = dZ @ W^T` — upstream gradient needs weight matrix transposed

The Rust runtime already had `Tensor::transpose()` (zero-copy stride swap), but the
tree-walk interpreter's `dispatch_method` never wired it up. CJC code could not call
`t.transpose()`.

Without this, the only workaround would be to manually index every element in nested
loops to build a transposed copy — catastrophically slow even by interpreter standards.

### Fix

**File**: `crates/cjc-eval/src/lib.rs` — `dispatch_method`

```rust
(Value::Tensor(t), "transpose") => {
    if t.ndim() != 2 {
        return Err(EvalError::Runtime(
            "transpose requires a 2-D tensor".to_string(),
        ));
    }
    Ok(Value::Tensor(t.transpose()))
}
```

### CJC Usage

```cjc
let a = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
let at = a.transpose();   // shape [3, 2], zero-copy
let dW = matmul(X.transpose(), dZ);  // weight gradient
let dX = matmul(dZ, W.transpose());  // upstream gradient
```

---

## 2. No Tensor-Scalar Arithmetic — CRITICAL

### Problem

Every neural network optimizer multiplies gradients by a scalar learning rate:
```
W = W - lr * dW     // SGD update
```

CJC only supported `Tensor * Tensor` (element-wise). `Tensor * Float` produced a
runtime error: `"cannot apply * to Tensor and Float"`.

This blocked:
- SGD updates: `weights - dW * 0.01`
- He initialization: `Tensor.randn([D, H]) * sqrt(2.0 / D)`
- MSE gradient: `diff * (2.0 / n)`
- Noise scaling: `Tensor.randn([B, O]) * 0.01`

### Fix

**File**: `crates/cjc-eval/src/lib.rs` — `eval_binop`

Added 6 new binary operation variants:

```rust
// Tensor x Float (scalar broadcast)
(Value::Tensor(t), Value::Float(s)) => match op {
    BinOp::Mul => Ok(Value::Tensor(t.scalar_mul(*s))),
    BinOp::Div => Ok(Value::Tensor(t.scalar_mul(1.0 / *s))),
    BinOp::Add => Ok(Value::Tensor(t.map(|x| x + *s))),
    BinOp::Sub => Ok(Value::Tensor(t.map(|x| x - *s))),
    ...
},
// Float x Tensor (scalar broadcast)
(Value::Float(s), Value::Tensor(t)) => match op {
    BinOp::Mul => Ok(Value::Tensor(t.scalar_mul(*s))),
    BinOp::Add => Ok(Value::Tensor(t.map(|x| *s + x))),
    BinOp::Sub => Ok(Value::Tensor(t.map(|x| *s - x))),
    BinOp::Div => Ok(Value::Tensor(t.map(|x| *s / x))),
    ...
},
// Tensor x Int (scalar broadcast, promote to float)
(Value::Tensor(t), Value::Int(s)) => match op {
    BinOp::Mul => Ok(Value::Tensor(t.scalar_mul(*s as f64))),
    BinOp::Div => Ok(Value::Tensor(t.scalar_mul(1.0 / *s as f64))),
    ...
},
// Int x Tensor (scalar broadcast, promote to float)
(Value::Int(s), Value::Tensor(t)) => match op {
    BinOp::Mul => Ok(Value::Tensor(t.scalar_mul(*s as f64))),
    ...
},
```

### CJC Usage

```cjc
let W = Tensor.randn([512, 1024]) * 0.05;   // Tensor * Float
let update = 0.01 * dW;                      // Float * Tensor
let scaled = W / 2.0;                        // Tensor / Float
let shifted = W + 1.0;                       // Tensor + Float
let W_new = W - dW * lr;                     // combined
```

---

## 3. No `sum_axis()` — Axis-Wise Reduction — CRITICAL

### Problem

The bias gradient requires summing over the batch dimension:
```
db = sum(dZ, axis=0)   // [B, H] -> [1, H]
```

CJC only had `.sum()` which collapses all elements to a single `f64` scalar. There was
no way to sum along a specific axis while preserving the other dimensions.

### Fix

**File**: `crates/cjc-runtime/src/lib.rs` — new method on `Tensor`

```rust
pub fn sum_axis(&self, axis: usize) -> Result<Tensor, RuntimeError> {
    // For 2-D tensors:
    // axis=0: sum columns -> [1, N]
    // axis=1: sum rows -> [M, 1]
    // Uses Kahan summation for numerical stability
}
```

**File**: `crates/cjc-eval/src/lib.rs` — `dispatch_method`

```rust
(Value::Tensor(t), "sum_axis") => {
    let axis = self.value_to_usize(&args[0])?;
    Ok(Value::Tensor(t.sum_axis(axis)?))
}
```

### CJC Usage

```cjc
let dZ = Tensor.randn([32, 256]);    // [batch, hidden]
let db = dZ.sum_axis(0);             // [1, 256] — bias gradient
let row_sums = dZ.sum_axis(1);       // [32, 1] — row sums
```

---

## 4. No `.scalar_mul()`, `.neg()`, `.mul()` Methods Exposed — IMPORTANT

### Problem

The Rust `Tensor` struct had `scalar_mul()`, `neg()`, and `mul_elem()` methods, but
the interpreter's `dispatch_method` didn't expose them. These are needed for:
- `scalar_mul(s)`: Scaling tensors by a constant
- `neg()`: Negating gradients
- `mul(other)`: Element-wise multiply for ReLU mask: `dA * relu_mask`

### Fix

**File**: `crates/cjc-eval/src/lib.rs` — `dispatch_method`

```rust
(Value::Tensor(t), "scalar_mul") => {
    match &args[0] {
        Value::Float(s) => Ok(Value::Tensor(t.scalar_mul(*s))),
        Value::Int(s) => Ok(Value::Tensor(t.scalar_mul(*s as f64))),
        ...
    }
}
(Value::Tensor(t), "neg") => {
    Ok(Value::Tensor(t.neg()))
}
(Value::Tensor(t), "mul") => {
    let other = self.value_to_tensor(&args[0])?;
    Ok(Value::Tensor(t.mul_elem(&other)?))
}
```

### CJC Usage

```cjc
let scaled = W.scalar_mul(0.99);     // weight decay
let negated = grad.neg();             // negate gradient
let masked = dA.mul(relu_mask);       // apply ReLU derivative
```

---

## 5. No Utility Builtins: `push`, `sort`, `sqrt`, `floor`, `int`, `float` — IMPORTANT

### Problem

Six essential utility functions were either missing entirely or listed as builtins
but never implemented in `dispatch_call`.

| Function | Needed For | Prior State |
|----------|-----------|-------------|
| `push(arr, val)` | Building arrays of step times, loss samples | Listed in `is_known_builtin` but **not implemented** in `dispatch_call` |
| `sort(arr)` | Percentile computation (p50/p95/p99) | Not present |
| `sqrt(x)` | He init scale `sqrt(2/fan_in)`, gradient norm | Not present |
| `floor(x)` | Index math, hash computation | Not present |
| `int(x)` | Float-to-int conversion for indexing | Not present |
| `float(x)` | Int-to-float for division | Not present |

### Fix

**File**: `crates/cjc-eval/src/lib.rs` — `dispatch_call`

```rust
"push" => {
    // push(array, value) -> new array with value appended
    // Returns a NEW array (CJC uses value semantics)
    match arr {
        Value::Array(mut a) => {
            a.push(val);
            Ok(Value::Array(a))
        }
        ...
    }
}
"sort" => {
    // sort(array) -> sorted copy (ascending by numeric value)
    let mut sorted = arr.clone();
    sorted.sort_by(|a, b| { /* compare as f64 */ });
    Ok(Value::Array(sorted))
}
"sqrt" => {
    match &args[0] {
        Value::Float(f) => Ok(Value::Float(f.sqrt())),
        Value::Int(i) => Ok(Value::Float((*i as f64).sqrt())),
        ...
    }
}
"floor" => {
    match &args[0] {
        Value::Float(f) => Ok(Value::Float(f.floor())),
        ...
    }
}
"int" => {
    // Truncate float to integer
    match &args[0] {
        Value::Float(f) => Ok(Value::Int(*f as i64)),
        ...
    }
}
"float" => {
    // Promote integer to float
    match &args[0] {
        Value::Int(i) => Ok(Value::Float(*i as f64)),
        ...
    }
}
```

### CJC Usage

```cjc
let times = [];
times = push(times, 0.0023);            // build array dynamically
times = push(times, 0.0019);

let sorted = sort(times);               // ascending sort
let p50 = sorted[len(sorted) * 50 / 100];  // median

let scale = sqrt(2.0 / float(fan_in));  // He initialization
let idx = int(floor(x));                // float -> index
```

---

## 6. No `isnan()`, `isinf()`, `abs()` Builtins — IMPORTANT

### Problem

Neural network training requires detecting numerical divergence (NaN/Inf in loss
or gradients). Without builtins, the benchmark had to use hacky workarounds:

```cjc
// Hacky workaround — fragile
fn is_nan(x: f64) -> bool { x != x }
fn is_inf(x: f64) -> bool { x > 1e308 }
fn abs_f(x: f64) -> f64 { if x < 0.0 { 0.0 - x } else { x } }
```

These fail on edge cases (e.g., `-Inf` not caught by `x > 1e308`), add
function-call overhead, and are not discoverable as language features.

### Fix

**File**: `crates/cjc-eval/src/lib.rs` — `dispatch_call`

```rust
"isnan" => {
    match &args[0] {
        Value::Float(f) => Ok(Value::Bool(f.is_nan())),
        Value::Int(_) => Ok(Value::Bool(false)), // integers are never NaN
        ...
    }
}
"isinf" => {
    match &args[0] {
        Value::Float(f) => Ok(Value::Bool(f.is_infinite())),
        Value::Int(_) => Ok(Value::Bool(false)), // integers are never Inf
        ...
    }
}
"abs" => {
    match &args[0] {
        Value::Float(f) => Ok(Value::Float(f.abs())),
        Value::Int(i) => Ok(Value::Int(i.abs())),
        ...
    }
}
```

### CJC Usage

```cjc
if isnan(loss) { print("Training diverged — NaN detected"); }
if isinf(loss) { print("Training diverged — Inf detected"); }
let magnitude = abs(-3.14);   // 3.14
```

---

## Summary of All Changes

### Files Modified

| File | Changes |
|------|---------|
| `crates/cjc-runtime/src/lib.rs` | Added `Tensor::sum_axis()` method |
| `crates/cjc-eval/src/lib.rs` | Added 6 tensor methods, 6 binary op variants, 9 builtins |

### Changes by Category

| Category | Item | Severity | Status |
|----------|------|----------|--------|
| **Tensor Methods** | `.transpose()` | Critical | Done |
| **Tensor Methods** | `.scalar_mul(s)` | Important | Done |
| **Tensor Methods** | `.neg()` | Important | Done |
| **Tensor Methods** | `.mul(other)` (element-wise) | Important | Done |
| **Tensor Methods** | `.sum_axis(axis)` | Critical | Done |
| **Tensor Methods** | `.set(indices, val)` | Important | Done |
| **Binary Ops** | `Tensor * Float` | Critical | Done |
| **Binary Ops** | `Float * Tensor` | Critical | Done |
| **Binary Ops** | `Tensor / Float` | Critical | Done |
| **Binary Ops** | `Tensor + Float` | Critical | Done |
| **Binary Ops** | `Tensor - Float` | Critical | Done |
| **Binary Ops** | `Float - Tensor` | Important | Done |
| **Binary Ops** | `Float / Tensor` | Important | Done |
| **Binary Ops** | `Tensor * Int` | Important | Done |
| **Binary Ops** | `Int * Tensor` | Important | Done |
| **Builtins** | `push(arr, val)` | Important | Done |
| **Builtins** | `sort(arr)` | Important | Done |
| **Builtins** | `sqrt(x)` | Important | Done |
| **Builtins** | `floor(x)` | Important | Done |
| **Builtins** | `int(x)` | Important | Done |
| **Builtins** | `float(x)` | Important | Done |
| **Builtins** | `isnan(x)` | Important | Done |
| **Builtins** | `isinf(x)` | Important | Done |
| **Builtins** | `abs(x)` | Important | Done |

### Test Results

- All 500+ existing tests pass with zero regressions
- All 3 demos (matmul, gradient descent, neural network pipeline) work unchanged
- New operations validated via dedicated test files
- Benchmark suite runs end-to-end (CJC + NumPy, mini case, determinism validated)

---

## What's Still Missing for Production ML

| Gap | Impact | Effort |
|-----|--------|--------|
| `f32` tensors | Memory bandwidth, GPU compat | Medium — needs generic `Tensor<T>` |
| Optimized matmul (SIMD, cache blocking) | 10-100x throughput on large matrices | High — needs LLVM or hand-tuned kernels |
| In-place operations (`W -= lr * dW`) | Avoids allocation per update | Medium — COW helps but still allocates |
| Higher-dim tensor ops (batched matmul, einsum) | CNNs, attention, transformers | High |
| Automatic differentiation integration | Usability for training | Medium — AD engine exists, needs wiring |
| Convolution, pooling ops | CNNs | High |
| Attention primitives (softmax, layer norm) | Transformers | Medium |
