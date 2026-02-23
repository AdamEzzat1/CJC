# Phase 3 — Zero-Copy Inference Engine: Progress Report

## 1. Deep Audit — Golden Baseline

| Metric | Count |
|---|---|
| **Workspace tests (pre-Phase 3)** | 964 |
| **New Phase 3 tests** | 45 |
| **Workspace tests (post-Phase 3)** | **1009** |
| Integration tests | 725 |
| Crate unit tests | 284 |
| Failures | **0** |
| Regressions | **0** |

### Per-File Test Breakdown (integration tests, top files)

| File | Tests |
|---|---|
| `tests/milestone_2_5/` | 70 |
| `tests/milestone_2_4/` | 62 |
| `tests/milestone_2_6/` | 62 |
| `tests/test_transformer.rs` | 56 |
| `tests/test_match_patterns.rs` | 51 |
| **`tests/test_phase3_inference.rs`** | **45** |
| `tests/test_regex.rs` | 77 |
| Other integration tests | 302 |

---

## 2. Phase 3 Deliverables

### 2.1 Zero-Copy Weight Mapping — `Tensor::from_bytes` ✅

**File:** `crates/cjc-runtime/src/lib.rs`

```
pub fn from_bytes(bytes: &[u8], shape: &[usize], dtype: &str) -> Result<Tensor, RuntimeError>
```

- Supports `f64` (8 bytes LE per element) and `f32` (4 bytes LE, promoted to f64)
- Shape validation: `bytes.len()` must equal `numel * sizeof(dtype)`
- Exactly **one allocation** for the data vector — no intermediate buffers
- Registered in both eval and mir-exec as `Tensor.from_bytes(bytes, shape, [dtype])`
- `ByteSlice.as_tensor(shape, [dtype])` instance method also wired

**Parity:** `from_bytes` produces bit-identical results to `Tensor::from_vec` ✅

### 2.2 KV-Cache Scratchpad — `Scratchpad` Type ✅

**File:** `crates/cjc-runtime/src/lib.rs`

```
pub struct Scratchpad {
    buffer: Buffer<f64>,     // pre-allocated [max_seq_len * dim]
    max_seq_len: usize,
    dim: usize,
    current_len: usize,      // cursor — grows with each append
}
```

**Methods:**

| Method | Signature | Allocation |
|---|---|---|
| `new` | `(max_seq_len, dim) -> Scratchpad` | One upfront |
| `append` | `(&mut self, &[f64]) -> Result` | **Zero** |
| `append_tensor` | `(&mut self, &Tensor) -> Result` | **Zero** |
| `as_tensor` | `(&self) -> Tensor` | Zero (Rc clone) |
| `clear` | `(&mut self)` | **Zero** |
| `len` / `capacity` / `dim` / `is_empty` | Accessors | Zero |

**Key property:** After initial `Scratchpad::new()`, every subsequent operation allocates **zero bytes of heap memory**. The `as_tensor()` view shares the underlying buffer via `Rc` clone (pointer copy only).

**Value variant:** `Value::Scratchpad(Rc<RefCell<Scratchpad>>)` added to the runtime Value enum.

### 2.3 Multi-Head Attention — `split_heads` / `merge_heads` ✅

**File:** `crates/cjc-runtime/src/lib.rs`

```
pub fn split_heads(&self, num_heads: usize) -> Result<Tensor, RuntimeError>
pub fn merge_heads(&self) -> Result<Tensor, RuntimeError>
pub fn view_reshape(&self, new_shape: &[usize]) -> Result<Tensor, RuntimeError>
```

- `split_heads`: `[B, S, H*D]` → `[B, H, S, D]` via stride manipulation (zero-copy when contiguous)
- `merge_heads`: `[B, H, S, D]` → `[B, S, H*D]` (materializes if non-contiguous after transpose)
- `view_reshape`: Delegates to `reshape()` — zero-copy for contiguous tensors

**Roundtrip property:** `t.split_heads(h).merge_heads() == t` (bit-exact) ✅

### 2.4 Dispatch Wiring ✅

All new operations registered in:

| Layer | File | Methods Added |
|---|---|---|
| **eval** | `crates/cjc-eval/src/lib.rs` | `split_heads`, `merge_heads`, `view_reshape`, `from_bytes`, `Scratchpad.*`, `ByteSlice.as_tensor` |
| **mir-exec** | `crates/cjc-mir-exec/src/lib.rs` | Mirror of eval |
| **NoGC verifier** | `crates/cjc-mir/src/nogc_verify.rs` | 17 new safe builtins |
| **is_known_builtin** | Both eval + mir-exec | `Tensor.from_bytes`, `Scratchpad.new`, `attention` |

### 2.5 NoGC Safe Builtins Added

```
Tensor.from_bytes, Tensor.split_heads, Tensor.merge_heads,
Tensor.view_reshape, ByteSlice.as_tensor,
Scratchpad.new, Scratchpad.append, Scratchpad.append_tensor,
Scratchpad.as_tensor, Scratchpad.len, Scratchpad.capacity,
Scratchpad.dim, Scratchpad.clear, Scratchpad.is_empty
```

All are pure math or pre-allocated buffer operations — **zero GC pressure**.

---

## 3. Master Pipeline — `examples/transformer_forward.cjc` ✅

Full transformer inference script demonstrating:

1. **Simulated input embeddings** `[1, 4, 8]`
2. **Weight initialization** via `Tensor.from_vec` (Q/K/V/O projections + FFN)
3. **KV-Cache setup** via `Scratchpad.new(32, 8)`
4. **Pre-norm LayerNorm** → Q/K/V projections → cache append
5. **Multi-head attention**: `split_heads(2)` → `attention()` → `merge_heads()`
6. **Output projection** + residual connection
7. **FFN block**: `linear(W1, b1)` → `gelu()` → `linear(W2, b2)` → residual
8. **Determinism gate**: Two identical forward passes → `assert_eq(val, val2)`

---

## 4. Test Coverage — `tests/test_phase3_inference.rs` (45 tests)

| Section | Tests | Coverage |
|---|---|---|
| 1. from_bytes (Rust) | 9 | f64, f32 promotion, shape mismatch, invalid dtype, 1D/3D, parity |
| 2. Scratchpad (Rust) | 8 | new, append single/multiple/tensor, overflow, dim mismatch, clear, buffer sharing |
| 3. split_heads (Rust) | 6 | basic, wrong dim, indivisible, roundtrip, 4-head, merge error |
| 4. view_reshape (Rust) | 2 | basic, mismatch |
| 5. Scratchpad (CJC eval) | 3 | basic, append+read, clear+reuse |
| 6. split/merge (CJC eval) | 2 | roundtrip, view_reshape |
| 7. Multi-head attention (CJC eval) | 2 | full pipeline, from_bytes |
| 8. KV-cache + attention (CJC eval) | 1 | Scratchpad → attention integration |
| 9. Parity (AST ↔ MIR) | 6 | scratchpad, clear, split_heads, view_reshape, multihead, kv-cache |
| 10. Transformer block | 2 | integration, parity |
| 11. Determinism double-run | 4 | scratchpad, split/merge, multihead attention, full pipeline |
| **Total** | **45** | |

---

## 5. Bug Fix: `to_vec()` Buffer Truncation

**Issue:** `Tensor::to_vec()` returned the entire backing buffer when contiguous, even when the tensor's logical size was smaller (e.g., `Scratchpad` pre-allocates `max_seq_len * dim` but only `current_len * dim` elements are valid).

**Fix:** Added truncation in `to_vec()`:
```rust
if self.is_contiguous() {
    let full = self.buffer.as_slice();
    let numel = self.len();
    if full.len() == numel { return full; }
    return full[..numel].to_vec(); // Truncate to logical size
}
```

This fix is correct for all existing tensors (where `buffer.len() == numel`) and essential for Scratchpad-backed tensors.

---

## 6. Determinism Contract

| Operation | Bit-Exact Double-Run | Status |
|---|---|---|
| `Tensor::from_bytes` | ✅ deterministic (LE byte decode) | PASS |
| `Scratchpad::append` + `as_tensor` | ✅ deterministic (memcpy into fixed buffer) | PASS |
| `split_heads` → `merge_heads` roundtrip | ✅ bit-identical to original | PASS |
| Multi-head attention pipeline | ✅ Kahan summation in softmax/matmul | PASS |
| Full transformer block + KV-cache | ✅ all operations deterministic | PASS |

---

## 7. Architecture Summary

```
                    ┌─────────────────────────────┐
                    │   transformer_forward.cjc    │
                    │   (Master Pipeline Script)   │
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
    ┌─────────────┐  ┌─────────────┐  ┌──────────────┐
    │  ByteSlice  │  │ Scratchpad  │  │   Tensor     │
    │ .as_tensor()│  │ (KV-Cache)  │  │ .split_heads │
    │ from_bytes  │  │ .append()   │  │ .merge_heads │
    └──────┬──────┘  │ .as_tensor()│  │ .view_reshape│
           │         └──────┬──────┘  └──────┬───────┘
           │                │                │
           ▼                ▼                ▼
    ┌────────────────────────────────────────────────┐
    │              Tensor Runtime (Layer 2)           │
    │   Buffer<f64> with Rc+COW, stride-based views  │
    │   softmax / layer_norm / linear / attention     │
    └────────────────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
         cjc-eval         cjc-mir-exec     NoGC verifier
        (AST v1)          (MIR v2)        (safe builtins)
```

---

## 8. Files Modified

| File | Changes |
|---|---|
| `crates/cjc-runtime/src/lib.rs` | `Tensor::from_bytes`, `split_heads`, `merge_heads`, `view_reshape`, `Scratchpad` struct, `Value::Scratchpad`, `to_vec()` fix |
| `crates/cjc-eval/src/lib.rs` | `Tensor.from_bytes`, `Scratchpad.new` constructors; method dispatch for `split_heads`, `merge_heads`, `view_reshape`, `ByteSlice.as_tensor`, all Scratchpad methods |
| `crates/cjc-mir-exec/src/lib.rs` | Mirror of eval dispatch |
| `crates/cjc-mir/src/nogc_verify.rs` | 17 new safe builtins |
| `examples/transformer_forward.cjc` | Full transformer inference pipeline |
| `tests/test_phase3_inference.rs` | 45 tests (new file) |

---

## 9. Status

| Phase | Status |
|---|---|
| Phase 1: Zero-Copy Weight Mapping | ✅ **COMPLETE** |
| Phase 2: KV-Cache Scratchpad | ✅ **COMPLETE** |
| Phase 3: Multi-Head Splitting | ✅ **COMPLETE** |
| Phase 4: Master Pipeline | ✅ **COMPLETE** |
| Phase 5: Full Regression (1009/1009) | ✅ **COMPLETE** |
| Phase 6: Documentation | ✅ **COMPLETE** |

**Final Golden Baseline: 1009 workspace tests, 0 failures, 0 regressions.**

---

## 10. Looking Ahead: RNN Benchmark

The infrastructure built here directly supports RNN architectures:

| Phase 3 Feature | RNN Application |
|---|---|
| `Scratchpad` (KV-Cache) | Hidden state persistence (`h_t` buffer) |
| `Tensor::from_bytes` | Zero-copy weight loading for RNN cells |
| `split_heads` / `merge_heads` | Multi-layer RNN state management |
| `view_reshape` | Gate splitting (LSTM: `[4*hidden]` → `[4, hidden]`) |
| Deterministic Kahan math | Reproducible RNN unrolling |
