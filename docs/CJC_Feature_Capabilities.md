> **Pre-v0.1.4 document.** Uses legacy naming: CJC (now CJC-Lang), `cjc` (now `cjcl`), `.cjc` (now `.cjcl`). Kept unmodified for historical accuracy. See [REBRAND_NOTICE.md](REBRAND_NOTICE.md) for the full mapping.

# CJC Language — Feature Capabilities Reference

> **Audit revision:** Post-Hardening (Phase 5) — 2025 tests passing, 0 failures
> **Auditor roles:** Role 3 (Runtime), Role 4 (ML/Numerical), Role 6 (Data), Role 7 (Regex)
> **Status:** READ-ONLY audit — no source modifications

---

## Table of Contents

1. [Compiler Pipeline](#1-compiler-pipeline)
2. [Runtime Value Model](#2-runtime-value-model)
3. [Memory Management (Three-Layer Model)](#3-memory-management-three-layer-model)
4. [Tensor System](#4-tensor-system)
5. [Numerical Stability — Stable Reductions](#5-numerical-stability--stable-reductions)
6. [Matrix Operations](#6-matrix-operations)
7. [Linear Algebra (Linalg)](#7-linear-algebra-linalg)
8. [Automatic Differentiation](#8-automatic-differentiation)
9. [Data System — DataFrame & CSV](#9-data-system--dataframe--csv)
10. [Regex System](#10-regex-system)
11. [Closures and Higher-Order Functions](#11-closures-and-higher-order-functions)
12. [MIR Control-Flow Graph](#12-mir-control-flow-graph)
13. [Determinism & Reproducibility](#13-determinism--reproducibility)
14. [GC System](#14-gc-system)
15. [Dispatch System](#15-dispatch-system)
16. [CLI Interface](#16-cli-interface)
17. [Feature Status Matrix](#17-feature-status-matrix)

---

## 1. Compiler Pipeline

CJC uses a multi-stage lowering pipeline:

```
Source text
    │
    ▼  cjc-lexer
Tokens (Vec<Token>)
    │
    ▼  cjc-parser
AST (Program / Decl / Expr / Pattern)
    │
    ▼  cjc-hir
HIR (desugaring: pipes, try-?, closures → captures)
    │
    ▼  cjc-mir
MIR tree-form (lambda-lifted, match-compiled)
    │
    ├──► cjc-mir::cfg — CFG (BasicBlock / Terminator) [NEW]
    │
    ▼  cjc-mir-exec
Interpreter (MirExecutor — tree-walking over MIR)
    │
    ▼
Value (runtime result)
```

### Crate Dependency Graph

```
cjc-ast        ← leaf (no deps except std)
cjc-diag       ← leaf
cjc-lexer      ← cjc-diag
cjc-parser     ← cjc-ast, cjc-diag, cjc-lexer
cjc-hir        ← cjc-ast
cjc-repro      ← (leaf: Rng, Kahan, pairwise)
cjc-runtime    ← cjc-repro
cjc-types      ← cjc-ast, cjc-diag
cjc-mir        ← cjc-ast, cjc-hir, cjc-repro
cjc-data       ← cjc-runtime
cjc-ad         ← cjc-runtime
cjc-regex      ← (leaf: regex engine)
cjc-eval       ← cjc-parser, cjc-runtime, cjc-data, ...
cjc-mir-exec   ← cjc-mir, cjc-runtime, cjc-data, cjc-repro, cjc-ast
cjc-dispatch   ← cjc-runtime
cjc-cli        ← all of the above
```

### Public Execution API

```rust
// Parse source text → AST
let (prog, diags) = parse_source(src);

// Execute (no type checking gate)
let value = run_program(&prog, seed)?;

// Execute with executor reference (for output inspection)
let (value, executor) = run_program_with_executor(&prog, seed)?;

// Execute with optimization pass first
let value = run_program_optimized(&prog, seed)?;

// Execute with compile-time type checking gate
let value = run_program_type_checked(&prog, seed)?;  // returns Err(TypeErrors) if any

// Standalone type-check only (no execution)
let result = type_check_program(&prog)?;             // Ok(()) or Err(TypeErrors)

// Lower to MIR only (no execution)
let mir_program = lower_to_mir(&prog);
```

---

## 2. Runtime Value Model

The `Value` enum in `cjc_runtime` represents all CJC runtime values:

```rust
pub enum Value {
    Void,
    Bool(bool),
    Int(i64),
    Float(f64),
    U8(u8),
    String(Rc<String>),
    ByteSlice(Rc<Vec<u8>>),
    Tensor(Tensor),
    Array(Vec<Value>),
    Tuple(Vec<Value>),
    Struct {
        name: String,
        fields: HashMap<String, Value>,
    },
    Class(GcRef),       // GC-managed class instance
    Enum {
        enum_name: String,
        variant: String,
        fields: Vec<Value>,
    },
    Fn(FnValue),        // named function reference
    Closure {           // lambda-lifted closure with captured env
        fn_name: String,
        env: Vec<Value>,
        arity: usize,
    },
    Regex {
        pattern: String,
        flags: String,
    },
    DataFrame(DataFrame),
    Column(Column),
    Bytes(Rc<Vec<u8>>),
}
```

### Value Characteristics

| Value Type | Stack/Heap | Copy/Clone | GC-managed |
|------------|-----------|------------|-----------|
| `Void`, `Bool`, `Int`, `Float`, `U8` | Stack | Copy | No |
| `String(Rc<...>)` | RC heap | Clone (shared) | No |
| `ByteSlice(Rc<...>)` | RC heap | Clone (shared) | No |
| `Tensor` | COW Buffer | Clone (shared until write) | No |
| `Array(Vec<Value>)` | Heap | Clone (deep) | No |
| `Tuple(Vec<Value>)` | Heap | Clone (deep) | No |
| `Struct { fields }` | Heap | Clone (deep) | No |
| `Class(GcRef)` | GC heap | Clone (reference) | **Yes** |
| `Enum { fields }` | Heap | Clone (deep) | No |
| `Fn(FnValue)` | Stack | Clone | No |
| `Closure { env }` | Heap | Clone (deep) | No |

---

## 3. Memory Management (Three-Layer Model)

CJC implements a three-layer memory model:

### Layer 1 — NoGC (Static / Stack)
**Types:** `i32`, `i64`, `u8`, `f32`, `f64`, `bf16`, `bool`, `void`, `ByteSlice`, `StrView`

- Zero allocation — values live on the stack
- No garbage collection pressure
- Checked at compile time via `nogc { ... }` blocks
- `is_nogc_safe()` method on `Type` enforces this at type-checking time
- `nogc_verify` module in `cjc-mir` validates MIR functions marked `is_nogc = true`

```cjc
nogc fn dot_product(a: ByteSlice, b: ByteSlice) -> f64 {
    // Only NoGC-safe operations allowed here
    ...
}
```

### Layer 2 — COW Buffer (Deterministic RC)
**Types:** `Bytes`, `Buffer<T>`, `Tensor<T>`, `Array<T>`, `Struct`, `Enum`, `String`, `Map<K,V>`, `SparseTensor<T>`

- Backed by `Rc<RefCell<Vec<T>>>` (reference counted, not GC)
- **Copy-on-write semantics:** `Buffer::set()` deep-copies if `refcount > 1`
- Deterministic lifetime — freed when last `Rc` drops
- `Buffer::make_unique()` forces exclusive ownership
- `Buffer::refcount()` exposes current reference count for testing

```rust
let b1 = Buffer::from_vec(vec![1.0, 2.0, 3.0]);
let b2 = b1.clone();  // shallow — same Rc
b2.set(0, 99.0);      // COW: deep copy triggered here
// b1 still has [1.0, 2.0, 3.0]; b2 has [99.0, 2.0, 3.0]
```

### Layer 3 — GC Mark-Sweep
**Types:** `class` instances (`GcRef`)

- `GcHeap` manages a contiguous array of `GcObject`s
- Simple mark-sweep collector: mark phase traverses `GcRef` fields, sweep reclaims unmarked
- `GcRef`: opaque index into `GcHeap::objects`
- Triggered manually or at allocation threshold
- Supports self-referential class graphs (linked lists, trees)

```cjc
class ListNode {
    value: i64,
    next: ListNode,  // self-referential — requires GC
}
```

---

## 4. Tensor System

The `Tensor` type in `cjc_runtime` is the core numerical container.

### Construction

```rust
Tensor::zeros(&[rows, cols])        // all-zeros
Tensor::ones(&[rows, cols])         // all-ones
Tensor::randn(&[rows, cols], rng)   // Normal(0,1) via Box-Muller, seeded
Tensor::from_vec(data, &[2, 3])     // from Vec<f64>
```

### Indexing and Slicing

```rust
tensor.get(&[i, j])                 // element access → f64
tensor.set(&[i, j], value)          // mutation (COW)
tensor.slice(&[(r0..r1), (c0..c1)]) // zero-copy view
tensor.to_contiguous()              // materialize non-contiguous
```

### Element-Wise Operations

```rust
tensor.add(&other)?      // element-wise +
tensor.sub(&other)?      // element-wise -
tensor.mul(&other)?      // element-wise *
tensor.div(&other)?      // element-wise /
tensor.scale(scalar)     // scalar multiply (in-place)
tensor.neg()             // negate all elements
tensor.abs_tensor()      // absolute value
tensor.sqrt_tensor()     // element-wise sqrt
tensor.exp_tensor()      // element-wise exp
tensor.log_tensor()      // element-wise log
tensor.sigmoid()         // σ(x) = 1/(1+e^{-x})
tensor.relu()            // max(0, x)
tensor.tanh_tensor()     // tanh
tensor.clamp(min, max)   // clamp to range
```

### Reduction Operations

```rust
tensor.sum()             // Kahan sum (stable)
tensor.mean()            // Kahan sum / numel
tensor.max()             // element maximum
tensor.min()             // element minimum
tensor.norm()            // L2 norm (Kahan sum of squares then sqrt)
tensor.softmax()         // numerically stable softmax (exp-max trick)
```

### Shape Operations

```rust
tensor.shape()                // &[usize]
tensor.ndim()                 // number of dimensions
tensor.len()                  // total element count
tensor.reshape(&[new_shape])? // contiguous reshape
tensor.transpose()            // 2D transpose (stride swap, zero-copy)
tensor.flatten()              // 1D view
tensor.broadcast_to(&[shape])? // broadcast (stride-0 trick)
```

### Broadcasting
`broadcast_to` uses stride-0 broadcasting — no data copy, just stride manipulation:

```rust
let v = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3])?;
let m = v.broadcast_to(&[4, 3])?;  // 4 rows, zero-copy
```

---

## 5. Numerical Stability — Stable Reductions

CJC provides three reduction strategies for deterministic, numerically stable floating-point accumulation:

### 5.1 Kahan Summation (`cjc-repro`)

**Use case:** Sequential, ordered reduction with good precision and low overhead.

```rust
// Functional form
let sum = kahan_sum_f64(&values);

// Accumulator form (allocation-free, post-hardening)
let mut acc = KahanAccumulatorF64::new();
acc.add(x);
acc.add(y);
let result = acc.finalize();
```

**Properties:**
- O(1) memory (2 registers: sum + compensation)
- O(n) time
- Error bounded by O(ε) instead of O(nε) for naive summation
- **Same-order determinism:** bit-identical for same input sequence

`KahanAccumulatorF64` and `KahanAccumulatorF32` are exported from `cjc-repro` root via:
```rust
pub use kahan::{KahanAccumulatorF32, KahanAccumulatorF64};
```

### 5.2 Pairwise Summation (`cjc-repro`)

```rust
let sum = pairwise_sum_f64(&values);
```

Recursive halving: if `len <= 32` uses Kahan, else splits and sums halves recursively.
- Error bounded by O(log n · ε) — better than Kahan for large arrays

### 5.3 Binned (Superaccumulator) — `BinnedAccumulatorF64/F32`

**Use case:** Order-invariant reduction — parallel or distributed summation where input order is not fixed.

```rust
let mut acc = BinnedAccumulatorF64::new();
acc.add(x);
acc.add(y);
// or
acc.add_slice(&values);
// parallel merge:
let mut acc2 = BinnedAccumulatorF64::new();
acc2.add_slice(&chunk);
acc.merge(&acc2);  // commutativity + associativity guaranteed
let result = acc.finalize();
```

**Properties:**
- **Stack-allocated:** `[f64; 2048]` bins + `[f64; 2048]` compensation + `[u32; 2048]` counts — no heap allocation ever
- **Order-invariant:** `a.merge(b)` == `b.merge(a)` (commutativity); `(a + b) + c` == `a + (b + c)` (associativity)
- **Bit-identical** regardless of add order, chunk size, thread count
- Exponent binning: each value classified by IEEE-754 biased exponent (11-bit for f64 → 2048 bins)
- Merges use Knuth 2Sum error-free transformation
- Finalize: Kahan-fold bins in ascending exponent order

**IEEE-754 special values:**
| Input | Result |
|-------|--------|
| NaN | sets `has_nan` flag → result is canonical NaN |
| +Inf | tracked separately → +Inf |
| -Inf | tracked separately → -Inf |
| +Inf + -Inf | → NaN |
| ±0.0 | binned into exponent 0 |
| subnormals | binned into exponent 0, not flushed |

### 5.4 Hybrid Dispatch (`cjc-dispatch`)

The `dispatch` module selects the appropriate summation strategy automatically:
- Small arrays (< threshold) → Kahan
- Large arrays → Binned
- User-configurable threshold

---

## 6. Matrix Operations

### 6.1 Matmul (Post-Hardening — Allocation-Free)

After Production Hardening Phase 5, **all matmul inner loops are allocation-free**.

**Before hardening:**
```rust
let products: Vec<f64> = (0..k).map(|p| a[i*k+p] * b[p*n+j]).collect();
result[i*n+j] = kahan_sum_f64(&products);  // allocates Vec per dot product
```

**After hardening:**
```rust
let mut acc = KahanAccumulatorF64::new();
for p in 0..k {
    acc.add(a[i * k + p] * b[p * n + j]);
}
result[i * n + j] = acc.finalize();  // zero allocations
```

This was applied to **5 locations**:

| Function | Description |
|----------|-------------|
| `Tensor::matmul` | Standard 2D matrix multiply |
| `Tensor::bmm` | Batched matrix multiply |
| `Tensor::linear_layer` | Fully connected layer (Ax + b) |
| `kernel::matmul_raw` | Raw buffer matmul (no Tensor overhead) |
| `kernel::linear_raw` | Raw buffer linear layer |

**Results:**
- Zero heap allocations per dot product
- Bit-identical output (same Kahan accumulation order)
- Verified by 10 determinism tests + 10 allocation-free tests

### 6.2 Matmul API

```rust
// High-level (Tensor method)
let c = a.matmul(&b)?;         // [M, K] x [K, N] -> [M, N]
let batched = a.bmm(&b)?;      // [B, M, K] x [B, K, N] -> [B, M, N]
let fc = a.linear_layer(&w, bias_opt)?;  // [N, in] x [in, out] + bias -> [N, out]

// Low-level (raw, no Tensor overhead)
kernel::matmul_raw(&a, &b, &mut c, m, k, n);
kernel::linear_raw(&input, &weights, &mut output, batch, in_feat, out_feat, bias_opt);
```

### 6.3 Matmul Error Handling

```rust
// Dimension mismatch returns Err
let err = Tensor::from_vec(vec![1.0, 2.0], &[1, 2])?
    .matmul(&Tensor::from_vec(vec![1.0, 2.0], &[1, 2])?);
assert!(err.is_err());  // k mismatch: 2 vs 1
```

---

## 7. Linear Algebra (Linalg)

Dedicated MIR opcodes for matrix decompositions ensure they are recognized and potentially hardware-dispatched:

### MIR Linalg Opcodes

```
LinalgLU { operand }         → (L, U) tuple — LU decomposition with pivoting
LinalgQR { operand }         → (Q, R) tuple — QR decomposition (Gram-Schmidt)
LinalgCholesky { operand }   → L — Cholesky decomposition (A = L·Lᵀ)
LinalgInv { operand }        → A⁻¹ — matrix inverse
```

### CJC Syntax

```cjc
fn solve(a: Tensor<f64>, b: Tensor<f64>) -> Tensor<f64> {
    let (l, u) = linalg_lu(a);
    // solve L·U·x = b ...
}

fn pca(data: Tensor<f64>) -> (Tensor<f64>, Tensor<f64>) {
    let (q, r) = linalg_qr(data);
    (q, r)
}

fn precision(cov: Tensor<f64>) -> Tensor<f64> {
    linalg_inv(cov)
}
```

### Runtime Dispatch

The MIR executor dispatches linalg opcodes to `Tensor` methods:
- `tensor.lu_decompose()` → `(L, U, pivots)`
- `tensor.qr_decompose()` → `(Q, R)`
- `tensor.cholesky()` → `L`
- `tensor.inverse()` → `A⁻¹`

---

## 8. Automatic Differentiation

The `cjc-ad` crate provides forward-mode automatic differentiation.

### Dual Numbers

```rust
#[derive(Debug, Clone, Copy)]
pub struct DualF64 {
    pub value: f64,      // primal (function value)
    pub tangent: f64,    // derivative
}
```

**Operations supported:** `+`, `-`, `*`, `/`, `neg`, `abs`, `sqrt`, `sin`, `cos`, `tan`, `exp`, `ln`, `powi`, `powf`, `min`, `max`, `sin_cos`.

### Gradient Computation

```rust
// Forward mode: compute f(x) and f'(x) simultaneously
let x = DualF64::new(2.0, 1.0);  // value=2, seed tangent=1
let result = x * x + DualF64::constant(3.0);
// result.value = 7.0, result.tangent = 4.0 (derivative of x² + 3 at x=2)
```

### Tensor AD

```rust
let grad = compute_gradient(&tensor, &loss_fn, &rng);
let grad_at = compute_gradient_at(&tensor, &loss_fn, idx, &rng);
let approx = numerical_gradient(&tensor, &loss_fn, epsilon, &rng);
```

---

## 9. Data System — DataFrame & CSV

### DataFrame

```rust
pub struct DataFrame {
    pub columns: HashMap<String, Column>,
    pub row_count: usize,
}
```

**Operations:**

```rust
// Construction
let df = DataFrame::from_csv("data.csv")?;

// Column access
let col = df.column("price")?;

// Filtering
let filtered = df.filter(|row| row["price"] > 100.0)?;

// Aggregation
let mean_price = df.aggregate("price", AggFunc::Mean)?;

// Joining
let joined = df1.join(&df2, "id", JoinType::Inner)?;
```

### Column Types
```rust
pub enum Column {
    Int(Vec<i64>),
    Float(Vec<f64>),
    Str(Vec<String>),
    Bool(Vec<bool>),
    Nullable(Vec<Option<f64>>),
}
```

### CSV Processing

**Batch reader:**
```rust
let cfg = CsvConfig::default();
let (df, diags) = CsvReader::read_file("data.csv", &cfg)?;
```

**Streaming processor (memory-efficient):**
```rust
let mut processor = StreamingCsvProcessor::new(cfg);
processor.process_file("large.csv", |row| {
    // handle each row incrementally
})?;
```

### DSL Sugar

```cjc
// Column reference in data expressions
let revenue = col("price") * col("quantity");

// Pipe-based data manipulation
df |> filter(col("age") > 18) |> group_by(col("region"))
```

---

## 10. Regex System

### Literal Syntax

```cjc
let pattern = /\d{4}-\d{2}-\d{2}/;            // date pattern
let email   = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/i;
```

**Supported flags:**
| Flag | Meaning |
|------|---------|
| `i`  | Case-insensitive |
| `g`  | Global (find all matches) |
| `m`  | Multiline (`^`/`$` match line boundaries) |
| `s`  | Dotall (`.` matches `\n`) |
| `x`  | Extended (whitespace ignored in pattern) |

### Operators

```cjc
let is_match = text ~= /pattern/i;    // true if matches
let no_match = text !~ /pattern/;     // true if does NOT match
```

### Disambiguation
The lexer is context-sensitive: `/` is lexed as a regex literal start only when the previous token was **not** a value-producing token (identifier, `)`, `]`, `}`, literal). This prevents ambiguity with the division operator:

```cjc
let x = a / b;         // division (prev token is identifier)
let r = /pattern/;     // regex literal (prev token is `=`)
let bad = (a) / b;     // division (prev token is `)`)
```

### Runtime Implementation

The `cjc-regex` crate provides the regex compilation and execution engine:
- Pattern compilation → `Regex` value
- Match/no-match binary operators
- Capture group support
- Raw string literals avoid double-escaping: `r"\d+"` == `/\\d+/`

---

## 11. Closures and Higher-Order Functions

### Lambda Syntax

```cjc
|x: f64| x * 2.0
|a: i64, b: i64| a + b
|items: [i64; N]| items[0]
```

### Closures (Capture by Value)

```cjc
let threshold = 0.5;
let filter_fn = |x: f64| x > threshold;  // captures threshold

let values = [0.1, 0.6, 0.3, 0.8];
let filtered = values |> filter(filter_fn);
```

### Lambda Lifting (Implementation)

During HIR→MIR lowering, closures are **lambda-lifted** to top-level functions:
1. Free variables in the lambda body are identified as captures
2. A fresh `__closure_N` function is created with captures as leading parameters
3. At the closure creation site, `MakeClosure { fn_name, captures }` is emitted
4. At call site, captured values are prepended to the argument list

```rust
// HIR input:
let thresh = 0.5;
let f = |x: f64| x > thresh;

// MIR output:
// function __closure_0(thresh: f64, x: f64) -> bool { x > thresh }
// let thresh = 0.5;
// let f = MakeClosure { fn_name: "__closure_0", captures: [thresh] };
```

### Pipe Operator

```cjc
data
  |> filter(is_valid)
  |> map(normalize)
  |> reduce(sum)
```

Pipe `a |> f(b, c)` desugars to `f(a, b, c)` — first argument is the left-hand side.

---

## 12. MIR Control-Flow Graph

The `cjc_mir::cfg` module (added in Production Hardening Phase 4) provides an explicit **graph representation** of MIR function bodies.

### When to Use CFG vs Tree-Form

| Analysis | Tree-form `MirBody` | CFG `MirCfg` |
|----------|---------------------|--------------|
| Tree-walking interpretation | ✅ | — |
| Live variable analysis | ❌ | ✅ |
| Loop-Invariant Code Motion | ❌ | ✅ |
| SSA construction | ❌ | ✅ |
| Dominator tree | ❌ | ✅ |
| Back-edge detection | ❌ | ✅ |

### CFG Construction

```rust
use cjc_mir::cfg::CfgBuilder;
let cfg = CfgBuilder::build(&mir_function.body);
```

- Deterministic: same `MirBody` always produces same `MirCfg` (same block IDs, same ordering)
- Single depth-first traversal — O(n) in MIR node count

### CFG API

```rust
// Access blocks
let entry = cfg.entry_block();          // &BasicBlock
let block = cfg.block(id);             // &BasicBlock

// Graph traversal
let succs = cfg.successors(id);        // Vec<BlockId>
let preds = cfg.predecessors();        // Vec<Vec<BlockId>> (all blocks)

// Loop detection
let is_header = cfg.is_loop_header(id);  // true if any predecessor has higher ID
```

### Block Layout Rules

| Construct | Blocks Created |
|-----------|----------------|
| Straight-line (let, expr) | stays in current block |
| `if/else` | branch-block + then-block + else-block + merge-block (≥3 new) |
| `while` | header-block + body-block + exit-block |
| `return` | terminates current; new dead block for dead code after |
| `nogc { }` | transparent (inlined into current block) |

---

## 13. Determinism & Reproducibility

Determinism is a **first-class language guarantee** in CJC, not an afterthought.

### Random Number Generation (`cjc-repro`)

```rust
pub struct Rng {
    state: u64,
}
```

**SplitMix64 algorithm:**
- `next_u64()`: `state += 0x9e3779b97f4a7c15; z = state; ...`
- `next_f64()`: `(next_u64() >> 11) as f64 / (1 << 53) as f64` → [0, 1)
- `next_f32()`: similar, 24-bit mantissa
- `next_normal_f64()`: Box-Muller transform (two uniforms → normal)
- `fork()`: deterministic child RNG from current state

**Guarantee:** Same seed → identical sequence on all platforms (no OS, CPU, or thread dependency).

### `run_program_with_executor(prog, seed)`

Seed is threaded through the entire execution:
- Initializes `MirExecutor::rng = Rng::seeded(seed)`
- `rand_tensor` builtin uses this RNG
- All stochastic operations consume from this single RNG

### Matmul Determinism (Post-Hardening)

After replacing `Vec<f64>` allocations with `KahanAccumulatorF64`:
- Same inputs → bit-identical outputs (same Kahan accumulation order)
- No allocation-order nondeterminism

### CFG Determinism

`CfgBuilder::build()` is deterministic:
- Block IDs assigned in single DFS traversal order
- Same `MirBody` → same `MirCfg` every time

### `BinnedAccumulatorF64` — Order-Invariant Determinism

Even when addition order varies (e.g., parallel execution):
- Commutativity: `acc.merge(a); acc.merge(b)` == `acc.merge(b); acc.merge(a)`
- Associativity: `(a + b) + c` == `a + (b + c)` in bit-exact terms

### ReproConfig

```rust
pub struct ReproConfig {
    pub enabled: bool,
    pub seed: u64,
}
```

Enables/disables seeded execution for reproducibility auditing.

---

## 14. GC System

### `GcHeap`

```rust
pub struct GcHeap {
    objects: Vec<GcObject>,
    capacity: usize,
    roots: Vec<GcRef>,
}
```

**Operations:**
```rust
let heap = GcHeap::new(1024);              // initial capacity
let obj_ref = heap.allocate(fields)?;      // returns GcRef
heap.set_field(ref, "name", value)?;       // mutate field
let val = heap.get_field(ref, "name")?;    // read field
heap.collect(&roots);                      // mark-sweep GC
```

### Mark-Sweep Algorithm
1. **Mark phase:** BFS/DFS from roots, mark all reachable `GcObject`s
2. **Sweep phase:** Iterate all objects; reclaim unmarked ones
3. **Compact (optional):** Not currently implemented (indices remain stable)

### Class Instance Lifecycle
```cjc
class Node { value: i64, next: Node, }

let n1 = Node { value: 1, next: null };
let n2 = Node { value: 2, next: n1 };
// n1 and n2 are GcRefs; managed by heap
// when no longer reachable → collected at next GC cycle
```

---

## 15. Dispatch System

The `cjc-dispatch` crate provides strategy dispatch for reduction operations.

### Summation Strategy Selection
```rust
pub fn dispatch_sum(values: &[f64], config: &DispatchConfig) -> f64 {
    if values.len() < config.kahan_threshold {
        kahan_sum_f64(values)
    } else {
        binned_sum_f64(values)
    }
}
```

### Matmul Strategy
```rust
pub fn dispatch_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, RuntimeError> {
    // Selects between:
    // - kernel::matmul_raw (small matrices)
    // - blocked matmul (cache-optimized for large)
    // Currently: always kernel::matmul_raw
}
```

---

## 16. CLI Interface

The `cjc-cli` crate provides the `cjc` binary.

### Commands

```bash
cjc run file.cjc              # parse, type-check, execute
cjc run --seed 42 file.cjc    # deterministic execution with seed
cjc check file.cjc            # parse + type check only (no execution)
cjc parse file.cjc            # parse only, print AST
cjc mir file.cjc              # lower to MIR, print
cjc cfg file.cjc              # build CFG, print block structure
```

### Output Modes
- Default: print final value
- `--verbose`: print diagnostics + timing
- `--json`: JSON output for tooling integration

---

## 17. Feature Status Matrix

| Feature | Crate | Status | Notes |
|---------|-------|--------|-------|
| Lexer | `cjc-lexer` | ✅ Complete | All literal types, nested comments |
| Parser | `cjc-parser` | ✅ Complete (limitation: `impl Trait for Type`) | All AST nodes |
| AST | `cjc-ast` | ✅ Complete | Full pretty-printer included |
| HIR lowering | `cjc-hir` | ✅ Complete | Desugars pipes, try, closures |
| MIR tree-form | `cjc-mir` | ✅ Complete | Lambda-lifting, match compilation |
| MIR CFG | `cjc-mir::cfg` | ✅ **NEW (Phase 4)** | BasicBlock/Terminator, DFS builder |
| MIR NoGC verify | `cjc-mir::nogc_verify` | ✅ Complete | Validates nogc annotations |
| MIR monomorph | `cjc-mir::monomorph` | ✅ Partial | Generic instantiation |
| MIR optimize | `cjc-mir::optimize` | ✅ Partial | Basic constant folding |
| Type checker | `cjc-types` | ✅ Complete | HM unification + span plumbing |
| Span-aware unify | `cjc-types::unify_spanned` | ✅ **NEW (Phase 1)** | E0100 with source location |
| Match exhaustiveness | `cjc-types` | ✅ **FIXED (Phase 2)** | E0130 compile-time error |
| Trait resolution | `cjc-types::check_impl` | ✅ **NEW (Phase 3)** | E0200/E0201/E0202 |
| MIR executor | `cjc-mir-exec` | ✅ Complete | Tree-walking interpreter |
| Type-checked exec | `cjc-mir-exec` | ✅ **NEW (Phase 2)** | `run_program_type_checked` |
| Runtime Buffer | `cjc-runtime` | ✅ Complete | COW, RC, `make_unique` |
| Tensor (basic) | `cjc-runtime` | ✅ Complete | All elem-wise + reductions |
| Tensor (matmul) | `cjc-runtime` | ✅ **HARDENED (Phase 5)** | Allocation-free, bit-identical |
| Tensor (linalg) | `cjc-runtime` | ✅ Complete | LU, QR, Cholesky, inv |
| GC heap | `cjc-runtime` | ✅ Complete | Mark-sweep, class support |
| Kahan accumulator | `cjc-repro` | ✅ Complete | `KahanAccumulatorF64/F32` |
| Binned accumulator | `cjc-runtime::accumulator` | ✅ Complete | Order-invariant superaccumulator |
| RNG | `cjc-repro` | ✅ Complete | SplitMix64, seeded, fork |
| Auto-diff | `cjc-ad` | ✅ Complete | Forward-mode DualF64 |
| DataFrame | `cjc-data` | ✅ Complete | Filter, agg, join, CSV |
| Streaming CSV | `cjc-data` | ✅ Complete | Memory-efficient reader |
| Regex | `cjc-regex` | ✅ Complete | Literals, flags, ~= / !~ ops |
| Closures | via MIR | ✅ Complete | Lambda-lifted at MIR level |
| Pipe operator | via HIR | ✅ Complete | `|>` desugared in HIR |
| Dispatch | `cjc-dispatch` | ✅ Partial | Kahan/Binned strategy |
| CLI | `cjc-cli` | ✅ Partial | run, check, parse, mir |
| Numeric generics | via `cjc-mir::monomorph` | ⚠️ Partial | Not fully instantiated |
| `impl Trait for Type` syntax | `cjc-parser` | ❌ Not supported | Use `impl Type : Trait` |
| SSA construction | `cjc-mir` | ❌ Planned | Requires CFG (now available) |
| Register allocation | `cjc-mir` | ❌ Planned | Requires live ranges |
| SIMD intrinsics | — | ❌ Planned | No intrinsic layer yet |
| Parallel execution | — | ❌ Planned | BinnedAccumulator ready |

---

*Generated by the CJC Full Repo Audit (Read-Only, Post-Hardening Phase 5)*
*Total passing tests at time of audit: **2025***
