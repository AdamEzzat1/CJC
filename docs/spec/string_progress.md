# CJC String & Bytes-First Progress Audit

**Generated:** 2025-02-20 (updated 2025-02-20)
**Baseline:** 547 tests → **624 tests**, all passing (0 failures, 0 ignored)
**Branch:** main

---

## 1. Test Suite Baseline

| Test File | Count | Status |
|-----------|-------|--------|
| cjc (doc-tests) | 0 | ok |
| test_ad | 12 | ok |
| test_bytes_strings | 34 | ok |
| test_closures | 26 | ok |
| test_data | 10 | ok |
| test_diag | 19 | ok |
| test_dispatch | 3 | ok |
| test_eval | 8 | ok |
| test_for_loops | 28 | ok |
| test_hir | 34 | ok |
| test_lexer | 18 | ok |
| test_match_patterns | 51 | ok |
| test_mir | 14 | ok |
| test_mir_exec | 62 | ok |
| test_parser | 70 | ok |
| milestone_2_4 (nogc+opt+parity+shape) | 62 | ok |
| milestone_2_5 | 6 | ok |
| test_repro | 8 | ok |
| test_runtime | 33 | ok |
| test_types | 10 | ok |
| test_for_loops (MIR parity) | 9 | ok |
| test_regex | 77 | ok |
| (remaining files) | 6+7 | ok |
| **TOTAL** | **624** | **ALL PASS** |

---

## 2. Bytes-First Layer 1 Checklist

### 2.1 Literal Syntax (Lexer + Parser)

| Feature | Lexer Token | Parser AST | Status |
|---------|------------|-----------|--------|
| `b"..."` byte string | `ByteStringLit` | `ExprKind::ByteStringLit(Vec<u8>)` | ✅ Complete |
| `b'c'` byte char | `ByteCharLit` | `ExprKind::ByteCharLit(u8)` | ✅ Complete |
| `r"..."` raw string | `RawStringLit` | `ExprKind::RawStringLit(String)` | ✅ Complete |
| `br"..."` raw byte string | `RawByteStringLit` | `ExprKind::RawByteStringLit(Vec<u8>)` | ✅ Complete |
| `r#"..."#` delimited raw | `RawStringLit` | `ExprKind::RawStringLit(String)` | ✅ Complete |
| `br#"..."#` delimited raw byte | `RawByteStringLit` | `ExprKind::RawByteStringLit(Vec<u8>)` | ✅ Complete |
| Escape sequences (`\n \t \r \\ \" \0 \xNN`) | — | — | ✅ Complete |
| `/pattern/flags` regex | `RegexLit` | `ExprKind::RegexLit { pattern, flags }` | ✅ Complete |

### 2.2 Type System

| Type | Enum Variant | `is_nogc_safe()` | `is_value_type()` | Status |
|------|-------------|-------------------|---------------------|--------|
| `Bytes` | `Type::Bytes` | No | Yes | ✅ Defined |
| `ByteSlice` | `Type::ByteSlice` | **Yes** | Yes | ✅ Defined |
| `StrView` | `Type::StrView` | **Yes** | Yes | ✅ Defined |
| `U8` | `Type::U8` | Yes | Yes | ✅ Defined |
| `Regex` | `Type::Regex` | No | Yes | ✅ Defined |

### 2.3 Runtime Value Representation

| Value Variant | Internal Repr | Status |
|---------------|---------------|--------|
| `Value::Bytes(Rc<RefCell<Vec<u8>>>)` | Owning, COW | ✅ Implemented |
| `Value::ByteSlice(Rc<Vec<u8>>)` | Shared immutable | ✅ Implemented |
| `Value::StrView(Rc<Vec<u8>>)` | Shared immutable (validated) | ✅ Implemented |
| `Value::U8(u8)` | Scalar | ✅ Implemented |
| `Value::String(Rc<String>)` | Existing | ✅ Implemented |
| `Value::Regex { pattern, flags }` | Compiled NFA | ✅ Implemented |

### 2.4 Hashing & Equality

| Feature | Status | Notes |
|---------|--------|-------|
| `murmurhash3` fixed seed `0x5f3759df` | ✅ | Platform-independent |
| ByteSlice content equality | ✅ | byte-by-byte |
| StrView content equality | ✅ | byte-by-byte |
| Bytes content equality | ✅ | via inner vec |

---

## 3. Pipeline Coverage

### 3.1 AST Evaluator (v1) — **FULLY COMPLETE**

| Method | Type | Returns | Tested |
|--------|------|---------|--------|
| `len()` | ByteSlice | i64 | ✅ |
| `is_empty()` | ByteSlice | bool | ✅ |
| `get(i)` | ByteSlice | u8 | ✅ |
| `slice(start, end)` | ByteSlice | ByteSlice | ✅ |
| `find_byte(b)` | ByteSlice | i64 (-1 if not found) | ✅ |
| `split_byte(delim)` | ByteSlice | Array<ByteSlice> | ✅ |
| `trim_ascii()` | ByteSlice | ByteSlice | ✅ |
| `strip_prefix(p)` | ByteSlice | Result<ByteSlice, Error> | ✅ |
| `strip_suffix(s)` | ByteSlice | Result<ByteSlice, Error> | ✅ |
| `starts_with(p)` | ByteSlice | bool | ✅ |
| `ends_with(s)` | ByteSlice | bool | ✅ |
| `count_byte(b)` | ByteSlice | i64 | ✅ |
| `as_str_utf8()` | ByteSlice | Result<StrView, Utf8Error> | ✅ |
| `eq(other)` | ByteSlice | bool | ✅ |
| `len_bytes()` | StrView | i64 | ✅ |
| `as_bytes()` | StrView | ByteSlice | ✅ |
| `to_string()` | StrView | String | ✅ |
| `eq(other)` | StrView | bool | ✅ |
| `len()` | String | i64 | ✅ |
| `as_bytes()` | String | ByteSlice | ✅ |

### 3.2 HIR Layer — **COMPLETE**

All byte literal kinds flow through AST→HIR lowering with correct variants:
- `ExprKind::ByteStringLit` → `HirExprKind::ByteStringLit`
- `ExprKind::ByteCharLit` → `HirExprKind::ByteCharLit`
- `ExprKind::RawByteStringLit` → `HirExprKind::RawByteStringLit`
- `ExprKind::RegexLit` → `HirExprKind::RegexLit`

### 3.3 MIR Layer — **FULLY COMPLETE**

Literal lowering works:
- `HirExprKind::ByteStringLit` → `MirExprKind::ByteStringLit` ✅
- `HirExprKind::ByteCharLit` → `MirExprKind::ByteCharLit` ✅
- `HirExprKind::RawByteStringLit` → `MirExprKind::RawByteStringLit` ✅
- `HirExprKind::RegexLit` → `MirExprKind::RegexLit` ✅

Monomorph handles byte literals in all three analysis passes. ✅
Optimizer (CF+DCE) handles byte literals as pure constants. ✅

MIR-exec `dispatch_method` covers:
- Tensor methods (sum, mean, shape, len, to_vec, matmul, add, sub, reshape, get) ✅
- Array.len() ✅
- String.len() ✅
- String.as_bytes() ✅
- Struct qualified method lookup ✅
- ByteSlice: len, is_empty, get, slice, find_byte, split_byte, trim_ascii, strip_prefix, strip_suffix, starts_with, ends_with, count_byte, as_str_utf8, eq ✅
- StrView: len_bytes, as_bytes, to_string, eq ✅

### 3.4 NoGC Verifier

- `Type::ByteSlice` → `is_nogc_safe() == true` ✅
- `Type::StrView` → `is_nogc_safe() == true` ✅
- `Type::Bytes` → `is_nogc_safe() == false` ✅ (correctly unsafe)
- NoGC verifier does not need ByteSlice-specific logic (handled via type system) ✅

---

## 4. Layer 2 Features — Status

### 4.1 Regex Literals — **FULLY COMPLETE**

| Feature | Status |
|---------|--------|
| `/pattern/flags` lexer token | ✅ Complete (`RegexLit` token, context-sensitive `/`) |
| `~=` match operator | ✅ Complete (`TildeEq` token, `BinOp::Match`) |
| `!~` negative match operator | ✅ Complete (`BangTilde` token, `BinOp::NotMatch`) |
| `RegexLit` AST/HIR/MIR variant | ✅ Complete (all pipeline layers) |
| Regex engine (NFA compiler) | ✅ Complete (`cjc-regex` crate, Thompson NFA) |
| `is_match`, `find`, `find_all`, `split` | ✅ Complete (runtime dispatch in eval + MIR-exec) |
| `Type::Regex` + `Value::Regex` | ✅ Complete |
| 77 integration tests (`test_regex.rs`) | ✅ All passing |

### 4.2 Tensor Syntax Extensions

| Feature | Status |
|---------|--------|
| `[| ... |]` tensor init syntax | ❌ Not implemented |
| `TensorView<T, Rank>` zero-copy wrapper | ❌ Not implemented |
| Transformer kernels (MatMul, Softmax, LayerNorm) | ❌ Not implemented |

### 4.3 MIR-exec ByteSlice Parity — **FULLY COMPLETE**

| Feature | Status |
|---------|--------|
| ByteSlice methods in MIR-exec dispatch | ✅ Complete (14 methods) |
| StrView methods in MIR-exec dispatch | ✅ Complete (4 methods) |
| String.as_bytes() in MIR-exec dispatch | ✅ Complete |
| Regex `~=` / `!~` in MIR-exec | ✅ Complete |

---

## 5. Implementation Priority

1. ~~**MIR-exec ByteSlice/StrView method parity**~~ ✅ Done
2. ~~**Regex literal lexing + parsing**~~ ✅ Done — `/pattern/flags` token + AST + `~=`/`!~` operators
3. ~~**Regex engine**~~ ✅ Done — NFA-based ByteSlice matcher (`cjc-regex` crate)
4. ~~**Regex runtime wiring**~~ ✅ Done — dispatch_method for Regex type in eval + MIR-exec
5. **Tensor syntax** — `[| ... |]` init, TensorView wrapper *(next)*
6. **Transformer kernels** — MatMul/Softmax/LayerNorm deterministic math
7. ~~**SYNTAX.md**~~ ✅ Done — Updated with full grammar spec

---

## 6. Determinism Contract

| Guarantee | Status |
|-----------|--------|
| Fixed hash seed (`0x5f3759df`) | ✅ Spec'd & implemented |
| murmurhash3 algorithm | ✅ Spec'd & implemented |
| Lexicographic byte ordering | ✅ Spec'd |
| No randomized seeds | ✅ Enforced |
| Double-run harness | ❌ Not yet built |

---

## 7. Documentation Status

| Document | Status |
|----------|--------|
| `docs/spec/bytes_and_strings.md` | ✅ Locked (v2.7-draft) |
| `docs/SYNTAX.md` | ✅ Updated with regex, byte literals, operators, EBNF grammar |
| `docs/spec/determinism_hashing.md` | ❌ Does not exist yet |
| `docs/spec/string_progress.md` | ✅ This document |
