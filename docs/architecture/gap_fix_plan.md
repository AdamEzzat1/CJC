# CJC Gap-Fix Plan: Data Science & Mathematics Completeness

## Stacked Role Group

| Role | Responsibility |
|------|---------------|
| **Type System Architect** | Typed tensors, new Value/Type variants, byte-first layout decisions |
| **Numerical Computing Lead** | SVD, interpolation, optimization, sparse arithmetic, distribution sampling |
| **ML Pipeline Engineer** | Clustering, PCA, AD graph expansion, categorical encoding |
| **Tensor Ops Engineer** | Sorting, boolean ops, axis reductions, concat/stack, einsum |
| **Snap Serialization Architect** | Chunked format, typed tensor encoding, streaming, schema evolution |
| **Determinism Auditor** | Every new feature must preserve bit-identical cross-run output |

---

## Phase 1: Typed Tensors (Foundation — Everything Else Builds on This)

### Current State
- `Tensor` wraps `Buffer<f64>` — **f64 only**
- `Buffer<T>` is generic (`Rc<RefCell<Vec<T>>>`) but `Tensor` hardcodes `f64`
- All ops (matmul, softmax, SIMD, etc.) operate on `f64` data

### Plan

#### 1A. Introduce `DType` enum and `TypedStorage`

```rust
// cjc-runtime/src/tensor.rs (or new tensor_typed.rs)

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F64,
    F32,
    I64,
    I32,
    U8,
    Bool,    // packed bitset storage (1 bit per element)
    Bf16,
    F16,
    Complex, // pair of f64 (16 bytes per element)
}

impl DType {
    pub fn byte_width(&self) -> usize {
        match self {
            DType::F64 | DType::I64 | DType::Complex => 8, // Complex = 16 but stored as 2x f64
            DType::F32 | DType::I32 => 4,
            DType::Bf16 | DType::F16 => 2,
            DType::U8 => 1,
            DType::Bool => 0, // special: packed bits, not byte-per-element
        }
    }
}
```

#### 1B. `TypedStorage` — Byte-First Backing Store

```rust
/// Byte-first tensor storage. Raw bytes + DType tag.
/// Views are computed on demand from the byte buffer.
pub struct TypedStorage {
    /// The raw byte buffer (8-byte aligned minimum, 16-byte for SIMD)
    bytes: Rc<RefCell<Vec<u8>>>,
    /// Element type determines how bytes are interpreted
    dtype: DType,
    /// Number of logical elements (not bytes)
    len: usize,
}
```

**Byte-first rationale:** Tensors are the #1 data-heavy type in CJC. Storing raw bytes means:
- `snap_encode(tensor)` = tag + dtype + shape + memcpy(bytes) — zero conversion
- Memory-mapped I/O: load a tensor file and use it directly without parsing
- SIMD operates on aligned byte buffers directly
- f32/i32/bf16 tensors use 50-75% less memory than f64

**View-second:** When you need typed access, create a view:
```rust
impl TypedStorage {
    pub fn as_f64(&self) -> &[f64] {
        assert_eq!(self.dtype, DType::F64);
        unsafe { std::slice::from_raw_parts(self.bytes.borrow().as_ptr() as *const f64, self.len) }
    }
    pub fn as_i64(&self) -> &[i64] { ... }
    pub fn as_bool_bits(&self) -> &BitVec { ... }  // packed bit access
}
```

#### 1C. Update `Tensor` struct

```rust
pub struct Tensor {
    storage: TypedStorage,   // WAS: Buffer<f64>
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}
```

All existing f64 ops continue to work — they call `storage.as_f64()` internally.
New typed ops are added incrementally.

#### 1D. Update `Value` and `Type` enums

- `Value::Tensor(Tensor)` — no change to variant, but Tensor now carries DType
- `Type::Tensor` — add `dtype: Option<DType>` field (None = inferred)

#### 1E. Byte-First Assessment

| Property | Benefit |
|----------|---------|
| Snap serialization | **Major** — tensor bytes are the encoded form |
| Memory efficiency | **Major** — f32 tensors use half the memory |
| SIMD compatibility | **Major** — aligned byte buffer is SIMD-ready |
| COW semantics | **Same** — Rc<RefCell<Vec<u8>>> works identically |
| Determinism | **Preserved** — byte layout is platform-canonical (LE) |

**Decision: YES — byte-first for TypedStorage.**

### Files Changed
- `crates/cjc-runtime/src/tensor.rs` — Tensor struct, DType, TypedStorage
- `crates/cjc-runtime/src/buffer.rs` — may need `Buffer<u8>` specialization
- `crates/cjc-runtime/src/tensor_simd.rs` — dispatch on DType
- `crates/cjc-types/src/lib.rs` — `Type::Tensor { dtype }`
- `crates/cjc-runtime/src/builtins.rs` — tensor constructor builtins
- `crates/cjc-snap/src/encode.rs` + `decode.rs` — typed tensor encoding
- `crates/cjc-eval/src/lib.rs` + `crates/cjc-mir-exec/src/lib.rs` — wiring

### Tests Required
- Roundtrip: create i64/f32/bool tensor → verify values
- Parity: eval vs MIR for typed tensor ops
- Determinism: typed tensor snap encode → identical bytes across runs
- SIMD: element-wise ops on f32 tensors use SIMD path
- Bool tensor: packed bit storage, logical ops

---

## Phase 2: Core Tensor Ops (Sort, Boolean, Reductions, Concat)

### Current State
Tensor already has: `argmax`, `argmin`, `topk`, `cat`, `stack`, `argsort`, `gather`, `scatter`, `index_select`, `one_hot`, `clamp`

### Missing Ops to Add

#### 2A. Boolean / Masking Ops
```
tensor.where(condition, other)  → element-wise conditional select
tensor.any()                     → bool (any element > 0)
tensor.all()                     → bool (all elements > 0)
tensor.nonzero()                 → indices tensor where elements are non-zero
tensor.masked_fill(mask, value)  → fill where mask is true
```

**Byte-first relevance:** Bool tensors (from Phase 1) make mask operations memory-efficient. A 1M-element mask is 125KB instead of 8MB.

#### 2B. Axis Reductions with keepdim
```
tensor.mean_axis(axis, keepdim)
tensor.max_axis(axis, keepdim)   → (values, indices)
tensor.min_axis(axis, keepdim)   → (values, indices)
tensor.var_axis(axis, keepdim)
tensor.std_axis(axis, keepdim)
tensor.prod_axis(axis, keepdim)
```

All reductions MUST use BinnedAccumulator for f64/f32 paths.

#### 2C. Sort Operations
```
tensor.sort(axis, descending)    → sorted tensor
tensor.argsort(axis, descending) → already exists (1D only, extend to N-D)
```

Sorting must be **stable** (deterministic tie-breaking).

#### 2D. Einsum
```
einsum("ij,jk->ik", [a, b])     → matmul
einsum("ii->i", [a])             → diagonal
einsum("ij->ji", [a])            → transpose
einsum("ijk,ikl->ijl", [a, b])   → batched matmul
```

Parse Einstein notation string → generate contraction plan → execute.
Deterministic: fixed contraction order, BinnedAccumulator for reductions.

#### 2E. Reshape / View Enhancements
```
tensor.unsqueeze(dim)            → add dimension of size 1
tensor.squeeze(dim)              → remove dimension of size 1
tensor.expand(shape)             → broadcast without copying
tensor.contiguous()              → already exists (to_contiguous)
tensor.flatten(start, end)       → flatten range of dims
tensor.chunk(n, dim)             → split into n chunks
tensor.split(sizes, dim)         → split by sizes
```

### Files Changed
- `crates/cjc-runtime/src/tensor.rs` — new methods
- `crates/cjc-runtime/src/builtins.rs` — wire new tensor builtins
- `crates/cjc-eval/src/lib.rs` + `crates/cjc-mir-exec/src/lib.rs` — wiring

### Byte-First Assessment
These are operations ON tensors, not new data types. They benefit from byte-first TypedStorage because:
- `where()` on bool mask + f64 tensor: mask reads from packed bits, result writes to f64 buffer
- `sort` on i64 tensor: no f64 conversion overhead
- `einsum`: contraction plan can optimize memory layout based on dtype byte width

---

## Phase 3: SVD and Dependent Algorithms

### Current State
- `eigh` (symmetric eigenvalue via QR iteration) exists — ~100 LOC
- `qr_decompose` exists — the building block for SVD
- No SVD, no PCA, no pseudoinverse

### Plan

#### 3A. SVD via Golub-Kahan Bidiagonalization
```
tensor.svd()          → (U, S, Vt)  where A = U @ diag(S) @ Vt
tensor.svd_truncated(k) → top-k singular values/vectors
```

Implementation: Householder bidiagonalization → implicit QR iteration on bidiagonal matrix.
Uses existing `qr_decompose` internally.

All intermediate reductions use BinnedAccumulator.

#### 3B. PCA (uses SVD)
```
pca(data, n_components)       → (transformed, components, explained_variance)
pca_fit(data, n_components)   → PcaModel
pca_transform(model, data)    → projected data
```

#### 3C. Pseudoinverse (Moore-Penrose, uses SVD)
```
tensor.pinv()   → A+ = V @ diag(1/s_i) @ Ut   (with tolerance for near-zero singular values)
```

### Files Changed
- `crates/cjc-runtime/src/linalg.rs` — `svd()`, `pinv()`
- `crates/cjc-runtime/src/ml.rs` — `pca()`
- `crates/cjc-runtime/src/builtins.rs` — wire as builtins

### Byte-First Assessment
SVD is an algorithm, not a data type. **No byte-first change needed.** The input/output tensors already benefit from byte-first TypedStorage (Phase 1). SVD on f32 tensors would be 2x faster due to cache efficiency.

---

## Phase 4: Distribution Sampling

### Current State
- `cjc-repro::Rng` has: `next_f64()`, `next_normal_f64()`, `fork()`
- All distributions have PDF/CDF/PPF but **no sampling**
- `Tensor::randn()` exists (standard normal only)

### Plan

Add to `crates/cjc-runtime/src/distributions.rs`:

```
normal_sample(mu, sigma, n, rng)      → Vec<f64>
uniform_sample(a, b, n, rng)          → Vec<f64>
exponential_sample(lambda, n, rng)    → Vec<f64>
poisson_sample(lambda, n, rng)        → Vec<i64>
binomial_sample(n_trials, p, n, rng)  → Vec<i64>
gamma_sample(k, theta, n, rng)        → Vec<f64>
beta_sample(a, b, n, rng)             → Vec<f64>
chi2_sample(df, n, rng)               → Vec<f64>
t_sample(df, n, rng)                  → Vec<f64>
bernoulli_sample(p, n, rng)           → Vec<bool>
categorical_sample(probs, n, rng)     → Vec<i64>   // already exists (scalar), extend to batch
multinomial_sample(probs, n, rng)     → Vec<i64>
dirichlet_sample(alpha, rng)          → Vec<f64>
multivariate_normal_sample(mu, cov, n, rng) → Tensor  // uses Cholesky
```

All sampling uses `cjc_repro::Rng` (SplitMix64) — deterministic with seed.

Algorithms:
- Normal: Box-Muller (already in Rng::next_normal_f64)
- Gamma: Marsaglia-Tsang
- Beta: via Gamma
- Poisson: Knuth (small lambda), transformed rejection (large lambda)
- Binomial: inverse CDF (small n), BTPE (large n)

### Byte-First Assessment
Sampling produces Vec<f64> or Vec<i64>. With typed tensors from Phase 1, sampling can return `Tensor` with appropriate DType directly:
- `poisson_sample` → `Tensor { dtype: I64, bytes: [...] }` — **byte-first native**
- `bernoulli_sample` → `Tensor { dtype: Bool, bytes: packed_bits }` — **major win**

**Decision: YES — sampling should return typed Tensors, not Vec.**

### Files Changed
- `crates/cjc-runtime/src/distributions.rs` — all sampling functions
- `crates/cjc-runtime/src/builtins.rs` — wire as builtins
- Both executors — wiring

---

## Phase 5: Optimization & Root Finding

### Current State
- Nothing. Zero optimization infrastructure.

### Plan

Add `crates/cjc-runtime/src/optimize.rs`:

#### 5A. Scalar Root Finding
```
bisect(f, a, b, tol)              → x where f(x) ≈ 0
brentq(f, a, b, tol)              → Brent's method (faster convergence)
newton(f, df, x0, tol, max_iter)  → Newton-Raphson
secant(f, x0, x1, tol, max_iter)  → secant method
```

#### 5B. Unconstrained Optimization
```
minimize_gd(f, grad, x0, lr, max_iter)         → gradient descent
minimize_bfgs(f, grad, x0, tol, max_iter)       → quasi-Newton BFGS
minimize_lbfgs(f, grad, x0, m, tol, max_iter)   → limited-memory BFGS
minimize_nelder_mead(f, x0, tol, max_iter)       → derivative-free
```

#### 5C. Constrained Optimization (stubs)
```
minimize_projected_gd(f, grad, proj, x0, ...)   → projected gradient descent
```

**Determinism:** All methods use fixed iteration order. BFGS uses BTreeMap for any keyed storage. Line search uses deterministic bisection.

### Byte-First Assessment
**No byte-first relevance.** These are algorithmic functions operating on scalars or small vectors. The function closures (f, grad) are CJC `Value::Fn` / `Value::Closure`.

### Files Changed
- NEW: `crates/cjc-runtime/src/optimize.rs`
- `crates/cjc-runtime/src/lib.rs` — add module
- `crates/cjc-runtime/src/builtins.rs` — wire builtins
- Both executors

---

## Phase 6: Interpolation & Curve Fitting

### Current State
- Nothing.

### Plan

Add `crates/cjc-runtime/src/interpolate.rs`:

```
interp1d(x_data, y_data, x_query, method)  → y values
    methods: "linear", "nearest", "cubic"

polyfit(x, y, degree)          → coefficients [a0, a1, ..., an]
polyval(coeffs, x)             → evaluated polynomial

spline_cubic(x, y)             → SplineCoeffs (natural cubic spline)
spline_eval(coeffs, x_query)   → interpolated values
```

**Determinism:** Cubic spline solves a tridiagonal system — deterministic by construction. Polynomial fitting via QR (already exists in linalg).

### Byte-First Assessment
**Minimal relevance.** `SplineCoeffs` could be byte-first (array of f64 coefficient segments), but it's a small struct. Not worth the complexity.

### Files Changed
- NEW: `crates/cjc-runtime/src/interpolate.rs`
- `crates/cjc-runtime/src/builtins.rs` — wire
- Both executors

---

## Phase 7: Sparse Matrix Arithmetic & Solvers

### Current State
- `SparseCsr`: `matvec`, `to_dense`, `from_coo`, `nnz`, `get`
- `SparseCoo`: `new`, `to_csr`, `sum`
- `lanczos_eigsh`, `arnoldi_eigs` in sparse_eigen.rs

### Plan

#### 7A. Sparse Arithmetic
```
sparse_add(a: SparseCsr, b: SparseCsr)       → SparseCsr
sparse_sub(a: SparseCsr, b: SparseCsr)       → SparseCsr
sparse_mul(a: SparseCsr, b: SparseCsr)       → SparseCsr (element-wise)
sparse_matmul(a: SparseCsr, b: SparseCsr)    → SparseCsr (SpGEMM)
sparse_scalar_mul(a: SparseCsr, s: f64)      → SparseCsr
sparse_transpose(a: SparseCsr)               → SparseCsr
```

#### 7B. Iterative Solvers
```
cg_solve(A: SparseCsr, b: Vec<f64>, tol, max_iter)      → x (conjugate gradient, SPD matrices)
gmres_solve(A: SparseCsr, b: Vec<f64>, tol, max_iter)   → x (general matrices)
bicgstab_solve(A: SparseCsr, b: Vec<f64>, tol, max_iter) → x (non-symmetric)
```

All solvers use BinnedAccumulator for dot products and norms.

#### 7C. Sparse Decompositions (later phase)
- Sparse Cholesky (incomplete — for preconditioners)
- Sparse LU (incomplete — for preconditioners)

### Byte-First Assessment
`SparseCsr` has three arrays: `row_ptr: Vec<usize>`, `col_idx: Vec<usize>`, `values: Vec<f64>`.

**Moderate benefit from byte-first:**
```
SparseCsr byte layout:
[nrows: u64][nnz: u64][row_ptr bytes...][col_idx bytes...][values bytes...]
```
This enables:
- Snap serialization = memcpy
- Memory-mapped sparse matrices
- With typed storage, values could be f32 for 2x compression

**Decision: DEFER — keep SparseCsr as-is for now. Revisit when typed tensors (Phase 1) prove the pattern.**

### Files Changed
- `crates/cjc-runtime/src/sparse.rs` — arithmetic ops
- NEW: `crates/cjc-runtime/src/sparse_solvers.rs` — CG, GMRES
- `crates/cjc-runtime/src/builtins.rs` — wire

---

## Phase 8: AD Graph Expansion

### Current State
GradGraph TapeOps: Add, Sub, Mul, MatMul, Sum, Mean, Sin, Cos, Sqrt, Pow, Sigmoid, Relu, Tanh, Div, Neg, ScalarMul, Exp, Ln

Also has: `backward`, `jacobian`, `hessian_diag`, `clip_grad`, `backward_with_seed`

### Missing TapeOps
```
Abs         → backward: grad * sign(x)
Log2        → backward: grad / (x * ln(2))
Softmax     → backward: grad * (softmax * (δ_ij - softmax))
CrossEntropy → backward: combined softmax+CE gradient (numerically stable)
LayerNorm   → backward: standard layer norm backward
BatchNorm   → backward: batch norm backward
Clamp       → backward: grad * (min <= x <= max)
Where       → backward: grad to selected branch
Reshape     → backward: grad.reshape(original_shape)
Transpose   → backward: grad.transpose()
Cat         → backward: split grad along cat axis
Gather      → backward: scatter_add
```

### Byte-First Assessment
**Not applicable.** AD tape is a runtime execution structure, not a serializable data type. The tape nodes reference tensors by index — they're inherently view-based.

### Files Changed
- `crates/cjc-ad/src/lib.rs` — new TapeOp variants + backward implementations

---

## Phase 9: Clustering & Unsupervised ML

### Plan

Add to `crates/cjc-runtime/src/ml.rs` (or new `ml_unsupervised.rs`):

#### 9A. K-Means
```
kmeans(data: Tensor, k: usize, max_iter: usize, seed: u64)
    → (centroids: Tensor, labels: Tensor<i64>, inertia: f64)
```
**Determinism:** Fixed seed for initialization (k-means++), deterministic assignment (BinnedAccumulator for distances), fixed iteration order.

#### 9B. DBSCAN
```
dbscan(data: Tensor, eps: f64, min_samples: usize)
    → labels: Tensor<i64>   // -1 for noise
```
**Determinism:** Use BTreeMap for neighbor lists, fixed point ordering.

#### 9C. Hierarchical / Agglomerative
```
agglomerative(data: Tensor, n_clusters: usize, linkage: &str)
    → (labels: Tensor<i64>, dendrogram: Tensor)
    linkage: "single", "complete", "average", "ward"
```

### Byte-First Assessment
Labels tensors benefit from typed tensors (Phase 1): `Tensor { dtype: I64 }` or even `Tensor { dtype: I32 }` for label indices. **This is a consumer of Phase 1, not a new byte-first type.**

---

## Phase 10: Categorical Encoding & DataFrame Enhancement

### Current State
- `Column::Str(Vec<String>)` — heap-allocated string per element
- No factor levels, no encoding

### Plan

#### 10A. Categorical Column Type
```rust
pub enum Column {
    Int(Vec<i64>),
    Float(Vec<f64>),
    Str(Vec<String>),
    Bool(Vec<bool>),
    Categorical {           // NEW
        levels: Vec<String>,     // unique level names, sorted
        codes: Vec<u32>,         // index into levels per row
    },
    DateTime(Vec<i64>),     // NEW — epoch millis
}
```

**Byte-first assessment:** Categorical is a natural byte-first type:
```
Byte layout: [n_levels: u32][level_offsets...][level_bytes...][n_rows: u64][codes as u32...]
```
- `codes` array is a flat u32 buffer — 4 bytes per element vs ~40 bytes for String
- Snap serialization = memcpy the codes + small string table
- Grouping/filtering operates on u32 codes, not string comparison

**Decision: YES — byte-first for Categorical column.**

#### 10B. Encoding Functions
```
one_hot_encode(col: Categorical)     → DataFrame (one column per level)
label_encode(col: Str)               → (Categorical, mapping)
ordinal_encode(col: Str, order: Vec) → Categorical
```

### Files Changed
- `crates/cjc-data/src/lib.rs` — Column::Categorical, DateTime
- `crates/cjc-runtime/src/builtins.rs` — encoding builtins

---

## Phase 11: Time Series

### Current State
- `adf_test`, `kpss_test`, `pp_test` — stationarity
- `window_sum/mean/min/max` — rolling windows
- `lag`, `lead` — shift operations

### Plan

Add `crates/cjc-runtime/src/timeseries.rs`:

```
acf(data, max_lag)            → autocorrelation function
pacf(data, max_lag)           → partial autocorrelation (via Durbin-Levinson)
ewma(data, alpha)             → exponential weighted moving average
ema(data, span)               → exponential moving average
seasonal_decompose(data, period, model)  → (trend, seasonal, residual)
    model: "additive" | "multiplicative"
diff(data, periods)           → differenced series
```

### Byte-First Assessment
**Not applicable.** These are algorithms on f64 slices. Output is Vec<f64> or Tensor.

---

## Snap Upgrade Plan (Phase 12)

### Current State
- `snap_encode(value) → Vec<u8>`: recursive tag+data encoding
- `snap_decode(bytes) → Value`: recursive parsing
- `snap(value) → SnapBlob`: encode + SHA-256 content hash
- `snap_save/snap_load`: file I/O
- 18 tag bytes (0x00-0x11) for all Value variants
- NaN canonicalized to 0x7FF8_0000_0000_0000
- Struct fields sorted alphabetically for deterministic encoding
- SHA-256 hand-rolled, zero external deps

### Upgrade Plan

#### 12A. Version Header
```
Current:  [tag][data...]
Proposed: [MAGIC: "CJS\x01"][version: u8][flags: u8][tag][data...]
```
- Magic bytes enable format detection
- Version byte for backward-compatible evolution
- Flags: 0x01 = compressed, 0x02 = chunked, 0x04 = includes schema

#### 12B. Typed Tensor Encoding (byte-first native)
```
Current TAG_TENSOR (0x08):
  [0x08][ndim: u64][shape: ndim×u64][data: nelems×f64]

New TAG_TYPED_TENSOR (0x12):
  [0x12][dtype: u8][ndim: u64][shape: ndim×u64][raw_bytes: nelems×byte_width]
```
dtype values: 0=f64, 1=f32, 2=i64, 3=i32, 4=u8, 5=bool(packed), 6=bf16, 7=f16, 8=complex

**This is THE key byte-first win for snap:** The encoded tensor bytes ARE the in-memory bytes. Decode = cast pointer. No conversion.

#### 12C. Chunked Encoding for Large Tensors
```
TAG_CHUNKED_TENSOR (0x13):
  [0x13][dtype: u8][ndim: u64][shape: ndim×u64]
  [n_chunks: u64]
  [chunk_0_size: u64][chunk_0_hash: 32 bytes][chunk_0_bytes...]
  [chunk_1_size: u64][chunk_1_hash: 32 bytes][chunk_1_bytes...]
  ...
```
Benefits:
- Stream large tensors without loading entire thing into memory
- Content-addressable chunks: dedup across snapshots
- Resumable save/load
- Chunk size default: 4MB (configurable)

#### 12D. Sparse Matrix Encoding
```
TAG_SPARSE_CSR (0x14):
  [0x14][dtype: u8][nrows: u64][ncols: u64][nnz: u64]
  [row_ptr: (nrows+1)×u64][col_idx: nnz×u64][values: nnz×byte_width]
```

#### 12E. Categorical Column Encoding
```
TAG_CATEGORICAL (0x15):
  [0x15][n_levels: u32][level_data: encoded strings][n_rows: u64][codes: n_rows×u32]
```

#### 12F. Schema / Metadata
```
TAG_SCHEMA (0x16):
  [0x16][n_fields: u32]
  [field_name: str][field_type: u8][field_metadata: optional]...
```
Enables:
- Reading tensor shape/dtype without loading data
- DataFrame schema inspection
- Lazy loading: read metadata first, load columns on demand

#### 12G. Optional Compression
For non-tensor data or highly compressible tensors:
- LZ4 frame compression (fast, deterministic)
- Applied per-chunk, indicated by flag byte
- Compression is OPTIONAL — uncompressed is always valid
- Would need to implement LZ4 from scratch (zero-dep constraint)

#### 12H. New Tag Registry

| Tag | Value | Type | Byte-First? |
|-----|-------|------|-------------|
| 0x00 | TAG_VOID | Void | N/A |
| 0x01 | TAG_INT | i64 | Yes (8 bytes inline) |
| 0x02 | TAG_FLOAT | f64 | Yes (8 bytes inline) |
| 0x03 | TAG_BOOL | bool | Yes (1 byte inline) |
| 0x04 | TAG_STRING | String | Yes (len + UTF-8 bytes) |
| 0x05 | TAG_ARRAY | Array | Partial (homogeneous = yes) |
| 0x06 | TAG_TUPLE | Tuple | Partial |
| 0x07 | TAG_STRUCT | Struct | Yes (sorted fields, offsets) |
| 0x08 | TAG_TENSOR | Tensor(f64) | LEGACY — keep for compat |
| 0x09 | TAG_ENUM | Enum | Yes (variant_id + payload) |
| 0x0A | TAG_BYTES | Bytes | Yes (native) |
| 0x0B | TAG_BYTESLICE | ByteSlice | Yes (native) |
| 0x0C | TAG_STRVIEW | StrView | Yes (native) |
| 0x0D | TAG_U8 | u8 | Yes (1 byte inline) |
| 0x0E | TAG_BF16 | Bf16 | Yes (2 bytes inline) |
| 0x0F | TAG_F16 | F16 | Yes (2 bytes inline) |
| 0x10 | TAG_COMPLEX | Complex | Yes (16 bytes inline) |
| 0x11 | TAG_MAP | Map | Partial (DetMap order preserved) |
| 0x12 | TAG_TYPED_TENSOR | TypedTensor | **YES — primary byte-first target** |
| 0x13 | TAG_CHUNKED_TENSOR | ChunkedTensor | YES |
| 0x14 | TAG_SPARSE_CSR | SparseCsr | YES |
| 0x15 | TAG_CATEGORICAL | Categorical | YES |
| 0x16 | TAG_SCHEMA | Schema | Metadata |
| 0x17 | TAG_DATAFRAME | DataFrame | YES (columnar) |

### Files Changed
- `crates/cjc-snap/src/encode.rs` — new tags, version header, typed tensor, chunked
- `crates/cjc-snap/src/decode.rs` — corresponding decoders
- `crates/cjc-snap/src/lib.rs` — SnapBlob v2, backward compat
- `crates/cjc-snap/src/persist.rs` — streaming save/load for chunked
- NEW: `crates/cjc-snap/src/schema.rs` — schema reading
- NEW: `crates/cjc-snap/src/chunk.rs` — chunked encoding

---

## Implementation Order (Dependency Graph)

```
Phase 1: Typed Tensors ─────────┐
                                 │
Phase 2: Tensor Ops ─────────────┤
                                 │
Phase 4: Distribution Sampling ──┤ (returns typed Tensors)
                                 │
Phase 3: SVD ────────────────────┤
    └─ Phase 9C: PCA             │
                                 │
Phase 5: Optimization ───────────┤ (independent)
Phase 6: Interpolation ──────────┤ (independent)
Phase 7: Sparse Arithmetic ──────┤ (independent)
Phase 8: AD Graph Expansion ─────┤ (independent)
Phase 10: Categorical ───────────┤
Phase 11: Time Series ───────────┤
                                 │
Phase 12: Snap Upgrade ──────────┘ (depends on all new types being defined)
    Phase 9: Clustering ─────────  (independent, uses typed Tensors)
```

**Recommended execution order:**
1. Phase 1 (typed tensors) — **MUST be first**, everything depends on it
2. Phase 2 (tensor ops) — high value, many algorithms need these
3. Phase 4 (sampling) + Phase 5 (optimization) + Phase 6 (interpolation) — **parallel**
4. Phase 3 (SVD → PCA)
5. Phase 7 (sparse) + Phase 8 (AD) + Phase 9 (clustering) — **parallel**
6. Phase 10 (categorical) + Phase 11 (time series) — **parallel**
7. Phase 12 (snap upgrade) — **last**, incorporates all new types

---

## Byte-First Summary

| New Type/Structure | Byte-First? | Rationale |
|-------------------|-------------|-----------|
| **TypedStorage (Tensor)** | **YES** | #1 priority. Raw byte buffer IS the in-memory form. Zero-copy snap, mmap, SIMD. |
| **Bool Tensor** | **YES** | Packed bits in byte buffer. 64x more memory efficient than f64. |
| **Categorical Column** | **YES** | u32 codes as byte array. 10x smaller than String column. |
| **SparseCsr** | **DEFER** | Benefit exists but lower priority. Revisit after typed tensors prove pattern. |
| **Chunked Tensor (Snap)** | **YES** | Content-addressable byte chunks for large tensors. |
| **DataFrame (Snap)** | **YES** | Columnar byte layout enables lazy column loading. |
| SVD/PCA output | No | Algorithm, not data type. Uses existing typed tensors. |
| Optimization result | No | Small scalar/vector output. |
| Interpolation coeffs | No | Small struct. |
| AD tape | No | Runtime execution structure, not serializable data. |
| Clustering labels | No | Uses typed Tensor<i64> from Phase 1. |
| Time series output | No | Uses existing f64 slices/tensors. |

---

## Estimated Scope

| Phase | New LOC (est.) | Files Changed | Risk |
|-------|---------------|---------------|------|
| 1. Typed Tensors | 800-1200 | 8-10 | **HIGH** — foundational change |
| 2. Tensor Ops | 600-800 | 3-4 | Medium |
| 3. SVD + PCA | 400-500 | 3 | Medium |
| 4. Distribution Sampling | 500-700 | 3 | Low |
| 5. Optimization | 400-600 | 3 (new file) | Low |
| 6. Interpolation | 300-400 | 3 (new file) | Low |
| 7. Sparse Arithmetic | 500-700 | 3 | Medium |
| 8. AD Expansion | 400-500 | 1 | Medium |
| 9. Clustering | 400-500 | 2 | Low |
| 10. Categorical | 300-400 | 2 | Low |
| 11. Time Series | 300-400 | 2 (new file) | Low |
| 12. Snap Upgrade | 600-800 | 5-6 | Medium |
| **Total** | **~5500-7500** | | |

---

## Non-Negotiable Constraints

1. **Determinism** — Every new function must produce bit-identical output given same inputs + seed
2. **Dual executor parity** — Every builtin works in both cjc-eval and cjc-mir-exec
3. **BinnedAccumulator** — All f64/f32 reductions. No naive `.sum()`
4. **BTreeMap/DetMap** — No HashMap/HashSet anywhere
5. **SplitMix64** — All randomness through cjc_repro::Rng
6. **Zero external deps** — Everything hand-rolled (CJC constraint)
7. **Backward compatibility** — Old TAG_TENSOR (0x08) snap files must still decode
