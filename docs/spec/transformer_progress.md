> **Pre-v0.1.4 document.** Uses legacy naming: CJC (now CJC-Lang), `cjc` (now `cjcl`), `.cjc` (now `.cjcl`). Kept unmodified for historical accuracy. See [../REBRAND_NOTICE.md](../REBRAND_NOTICE.md) for the full mapping.

# CJC Transformer Kernel — Silicon Transformer Milestone

**Generated:** 2025-02-20 (updated 2025-02-20)
**Golden Baseline:** 908 → **964 workspace tests** (680 integration + 284 crate unit), all passing
**Branch:** main

---

## 1. Deep Audit — Test Reconciliation

The discrepancy between "624" and "908" is Rust's split test model:
- `cargo test` (default) runs only the **root crate** — integration tests in `tests/` plus root lib unit tests.
- `cargo test --workspace` runs **all crates** — each subcrate's `#[cfg(test)]` inline tests *plus* root integration tests.

### 1.1 Integration Tests (root crate: `tests/`)

| Test File | Count | Status |
|-----------|-------|--------|
| test_ad | 12 | ✅ |
| test_bytes_strings | 34 | ✅ |
| test_closures | 26 | ✅ |
| test_data | 10 | ✅ |
| test_determinism | 19 | ✅ |
| test_diag | 3 | ✅ |
| test_dispatch | 8 | ✅ |
| test_eval | 28 | ✅ |
| test_for_loops | 34 | ✅ |
| test_hir | 18 | ✅ |
| test_lexer | 14 | ✅ |
| test_match_patterns | 51 | ✅ |
| test_milestone_2_4 | 62 | ✅ |
| test_milestone_2_5 | 70 | ✅ |
| test_milestone_2_6 | 62 | ✅ |
| test_mir | 6 | ✅ |
| test_mir_exec | 8 | ✅ |
| test_parser | 33 | ✅ |
| test_reference_kernels | 10 | ✅ |
| test_regex | 77 | ✅ |
| test_regression_gate | 9 | ✅ |
| test_repro | 6 | ✅ |
| test_runtime | 17 | ✅ |
| test_transformer | 56 | ✅ |
| test_types | 7 | ✅ |
| **Integration Subtotal** | **680** | **ALL PASS** |

### 1.2 Crate Unit Tests (`#[cfg(test)]` inline)

| Crate | Count | Status |
|-------|-------|--------|
| cjc-ad | 12 | ✅ |
| cjc-data | 10 | ✅ |
| cjc-diag | 3 | ✅ |
| cjc-dispatch | 8 | ✅ |
| cjc-eval | 28 | ✅ |
| cjc-hir | 18 | ✅ |
| cjc-lexer | 38 | ✅ |
| cjc-mir | 30 | ✅ |
| cjc-mir-exec | 8 | ✅ |
| cjc-parser | 33 | ✅ |
| cjc-regex | 27 | ✅ |
| cjc-repro | 6 | ✅ |
| cjc-runtime | 31 | ✅ |
| cjc-types | 32 | ✅ |
| **Unit Subtotal** | **284** | **ALL PASS** |

### 1.3 Golden Baseline

| Metric | Value |
|--------|-------|
| `cargo test -- --list \| grep ": test" \| wc -l` | **680** |
| `cargo test --workspace -- --list \| grep ": test" \| wc -l` | **964** |
| Failures | **0** |
| Ignored | **0** |

---

## 2. Existing Infrastructure Inventory

### 2.1 Tensor Core (`cjc-runtime`)

```
struct Tensor {
    buffer: Buffer<f64>,     // Rc<RefCell<Vec<f64>>> with COW
    shape:  Vec<usize>,      // Dimension sizes
    strides: Vec<usize>,     // Row-major strides
    offset: usize,           // Offset into buffer (for views)
}
```

**Already implemented:**
- Construction: `zeros`, `ones`, `randn`, `from_vec`
- Element-wise: `add`, `sub`, `mul_elem`, `div_elem`
- MatMul: 2D, O(M×K×N), **Kahan summation** on dot products
- Reductions: `sum`, `mean`, `sum_axis` — all using Kahan
- Views (zero-copy): `slice`, `transpose`, `reshape`, `broadcast_to`
- Utilities: `to_vec`, `to_contiguous`, `is_contiguous`, `map`, `neg`, `scalar_mul`

### 2.2 Memory Model (3 Layers)

| Layer | Type | Mechanism | NoGC Safe? |
|-------|------|-----------|------------|
| 1 | i32/i64/u8/f32/f64/bf16/bool | Stack | ✅ Yes |
| 2 | Buffer/Tensor/String/Bytes/Array | Rc + COW | ❌ No (allocation) |
| 3 | Class instances | GcHeap (mark-sweep) | ❌ No |

**ByteSlice/StrView** are special Layer 1.5 — zero-copy views over Layer 2 data, NoGC safe.

### 2.3 NoGC Verifier (`cjc-mir/nogc_verify.rs`)

Static analysis via fixpoint iteration:
- Computes `may_gc` for every function in the MIR program
- Rejects: direct GC builtins, transitive GC calls, indirect calls, unknown externals
- `is_nogc_safe()` types: i32, i64, u8, f32, f64, bf16, bool, void, ByteSlice, StrView
- `is_safe_builtin()`: matmul, Tensor.zeros/ones/randn/from_vec/slice/transpose/broadcast_to, etc.

### 2.4 Numerical Stability (`cjc-repro`)

- `kahan_sum_f64(values: &[f64]) -> f64` — compensated summation
- `kahan_sum_f32(values: &[f32]) -> f32`
- `pairwise_sum_f64(values: &[f64]) -> f64` — recursive divide-and-conquer
- Deterministic RNG: `Rng` with fixed seed, `next_u64`, `next_f64`, `next_normal_f64`, `fork()`
- All math is deterministic: murmurhash3 with fixed seed `0x5f3759df`

### 2.5 Type System (`cjc-types`)

```
Type::Tensor { elem: Box<Type>, shape: Option<Vec<ShapeDim>> }
Type::Bf16
ShapeDim::Known(usize) | ShapeDim::Symbolic(String)
```

### 2.6 Bf16 (`cjc-runtime`)

```
struct Bf16(pub u16);
from_f32: truncate lower 16 bits
to_f32: left-shift 16 bits
Arithmetic: widen to f32, compute, narrow back
```

---

## 3. Silicon Transformer — Implementation Plan

### 3.1 Transformer Kernel Architecture

```
Input tokens [B, T]
    ↓
Embedding lookup → [B, T, D]
    ↓
┌─────────────────────────────────────┐
│ Transformer Block (×N layers)       │
│                                     │
│  LayerNorm → Q,K,V projections      │
│      ↓                              │
│  Multi-Head Attention               │
│    • Q×Kᵀ / √d_k  (MatMul)        │
│    • Softmax (two-pass stable)      │
│    • Attn×V  (MatMul)              │
│      ↓                              │
│  Residual Add                       │
│      ↓                              │
│  LayerNorm                          │
│      ↓                              │
│  FFN: Linear→ReLU→Linear           │
│      ↓                              │
│  Residual Add                       │
└─────────────────────────────────────┘
    ↓
Final LayerNorm → Logits [B, T, V]
```

### 3.2 Kernel Specifications

#### MatMul Kernel (EXISTING — enhance for batched)
- Current: 2D only, Kahan summation, O(M×K×N)
- **Enhancement:** Add batched matmul `bmm([B,M,K], [B,K,N]) -> [B,M,N]`
- Precision: f64 accumulation, no platform dependency
- NoGC strategy: operate on existing Tensor buffers (Layer 2)

#### Softmax Kernel (NEW)
- Algorithm: Two-pass stable softmax
  1. Pass 1: Find max across last dimension
  2. Pass 2: exp(x - max), accumulate sum
  3. Normalize: divide by sum
- Precision: All intermediates in f64
- Special cases: handle -inf, NaN propagation

#### LayerNorm Kernel (NEW)
- Algorithm: Two-pass normalization
  1. Pass 1: Compute mean = Σx/n (Kahan)
  2. Pass 2: Compute variance = Σ(x-μ)²/n (Kahan)
  3. Normalize: (x - μ) / √(σ² + ε)
  4. Scale + shift: γ * normalized + β
- Epsilon: 1e-5 (standard)
- Precision: f64 throughout

### 3.3 TensorView over ByteSlice (NEW)

Zero-copy weight mapping:
```
ByteSlice (raw model weights on disk)
    ↓ as_tensor::<f32>([1, 12, 64])
TensorView { data: &[u8], shape, strides, dtype }
```

- Maps raw bytes to typed tensor without copying
- Supports f32 and f64 element types
- NoGC safe (view over existing ByteSlice)
- Read-only (inference only, no gradient mutation)

### 3.4 Tensor Literal Syntax `[| ... |]`

```cjc
let weights = [| 1.0, 0.0; 0.0, 1.0 |];  // 2×2 matrix
let bias = [| 0.1, 0.2, 0.3 |];           // 1D vector
let cube = [| [| 1, 2; 3, 4 |], [| 5, 6; 7, 8 |] |]; // 2×2×2
```

Lexer tokens: `LBracketPipe` (`[|`), `PipeRBracket` (`|]`)
Parser: parse as nested literal with `;` as row separator

---

## 4. Implementation Checklist

### Phase 1: Tensor Primitives — **COMPLETE**

| Task | Status |
|------|--------|
| `[| ... |]` lexer tokens (`LBracketPipe`, `PipeRBracket`) | ✅ |
| `[| ... |]` parser → `ExprKind::TensorLit` | ✅ |
| `TensorLit` through HIR/MIR pipeline | ✅ |
| TensorLit eval (cjc-eval + cjc-mir-exec) | ✅ |
| Shape inference (1D: single row, 2D: semicolon rows) | ✅ |

### Phase 2: Kernel Functions — **COMPLETE**

| Task | Status |
|------|--------|
| `softmax()` — two-pass stable (max-subtraction + Kahan) | ✅ |
| `layer_norm(gamma, beta, eps)` — Kahan mean/variance | ✅ |
| `bmm()` — batched matrix multiplication | ✅ |
| `relu()` — max(0, x) element-wise | ✅ |
| `gelu()` — approximate GELU activation | ✅ |
| `linear(weight, bias)` — matmul + bias | ✅ |
| `scaled_dot_product_attention(Q, K, V)` — softmax(QK^T/√d)V | ✅ |
| `transpose_last_two()` — swap last two dims | ✅ |
| `attention()` builtin function (eval + mir-exec) | ✅ |
| Register all as `is_safe_builtin` in NoGC verifier | ✅ |

### Phase 3: Transformer Forward Pass — **PARTIAL**

| Task | Status |
|------|--------|
| LayerNorm → Linear → ReLU → Linear block test | ✅ |
| Attention forward pass (Q/K/V projection → attention) | ✅ |
| AST/MIR parity on transformer blocks | ✅ |
| Full `transformer_forward.cjc` reference implementation | ❌ (future) |
| KV-cache with pre-allocated scratch buffer | ❌ (future) |
| Multi-head attention (head splitting) | ❌ (future) |

### Phase 4: Regression Gate — **COMPLETE**

| Task | Status |
|------|--------|
| All 908 baseline workspace tests pass | ✅ |
| New tensor syntax tests pass (12 tests) | ✅ |
| New kernel tests pass (44 tests) | ✅ |
| Determinism double-run on softmax/layernorm/attention | ✅ |
| Full pipeline determinism double-run | ✅ |
| **Total: 964 workspace tests, 0 failures** | ✅ |

---

## 5. Determinism Contract (Extended)

| Guarantee | Status |
|-----------|--------|
| Fixed hash seed (`0x5f3759df`) | ✅ |
| murmurhash3 algorithm | ✅ |
| Kahan summation in all reductions | ✅ |
| Kahan summation in MatMul dot products | ✅ |
| f64 accumulation (no platform-dependent intermediate precision) | ✅ |
| Deterministic RNG (Box-Muller for normal) | ✅ |
| Softmax: max-subtraction overflow prevention | ✅ |
| LayerNorm: Kahan for mean and variance | ✅ |
| Transformer block: bit-exact across double-runs | ✅ |
