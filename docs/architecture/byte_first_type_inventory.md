# Byte-First Type Inventory

## Overview

CJC maintains 30+ runtime `Value` variants and 30+ static `Type` variants. This document catalogs every type, its byte-level representation, determinism properties, and VM strategy classification.

## Value Enum Variants

### Primitive Types

| Variant | Storage | Size | Deterministic | Notes |
|---------|---------|------|---------------|-------|
| `Int(i64)` | 8-byte LE | 8B | Yes | Two's complement, platform-independent |
| `Float(f64)` | IEEE 754 double | 8B | Yes | Bit-identical via `.to_bits()` comparison |
| `Bool(bool)` | 1 byte | 1B | Yes | `true`/`false` |
| `U8(u8)` | 1 byte | 1B | Yes | Unsigned byte |
| `Bf16(Bf16)` | Custom struct, `u16` storage | 2B | Yes | Brain float, truncate-and-round from f32 |
| `F16(F16)` | Custom struct, `u16` storage | 2B | Yes | IEEE 754 half-precision emulation |
| `Complex { re: f64, im: f64 }` | Two f64 | 16B | Yes | Fixed-point arithmetic methods |
| `Void` | Unit | 0B | Yes | No data |

### Container Types

| Variant | Storage | Deterministic | Notes |
|---------|---------|---------------|-------|
| `String(Rc<String>)` | COW via Rc | Yes | UTF-8, immutable sharing |
| `Array(Rc<Vec<Value>>)` | COW via Rc | Yes | Ordered, immutable sharing |
| `Tuple(Rc<Vec<Value>>)` | COW via Rc | Yes | Fixed-length ordered |
| `Bytes(Rc<RefCell<Vec<u8>>>)` | Mutable buffer | Yes | Interior mutability |
| `ByteSlice(Rc<Vec<u8>>)` | Immutable view | Yes | Read-only byte view |
| `StrView(Rc<Vec<u8>>)` | Immutable view | Yes | Read-only string view |
| `Struct { name, fields: BTreeMap }` | Sorted map | Yes | **BTreeMap guarantees alphabetical field order** |
| `Enum { enum_name, variant, fields }` | Tagged union | Yes | Variant name + payload |
| `Map(Rc<RefCell<DetMap>>)` | Deterministic hash map | Yes | **DetMap: MurmurHash3 + insertion-order iteration** |

### Tensor Types

| Variant | Storage | Deterministic | Notes |
|---------|---------|---------------|-------|
| `Tensor(Tensor)` | `Buffer<f64>` + shape/strides | Yes | Row-major, COW via `Rc<RefCell<Vec<f64>>>` |
| `SparseTensor(SparseTensor)` | COO format | Yes | Sorted indices |

### Function Types

| Variant | Storage | Deterministic | Notes |
|---------|---------|---------------|-------|
| `Fn(FnValue)` | Name reference | Yes | Points to registered function |
| `Closure { params, body, captures }` | Captured environment | Yes | Captures are cloned values |

### Runtime/Library Types

| Variant | Storage | Deterministic | Notes |
|---------|---------|---------------|-------|
| `Regex { pattern, flags }` | String pair | Yes | NFA-based engine |
| `ClassRef(GcRef)` | ObjectSlab index | Yes | RC-based, deterministic allocation order |
| `Scratchpad(Scratchpad)` | Pre-allocated buffer | Yes | Fixed-size workspace |
| `PagedKvCache(PagedKvCache)` | Paged buffer | Yes | Attention cache |
| `AlignedBytes(AlignedBytes)` | Aligned allocation | Yes | SIMD-friendly |
| `GradGraph(GradGraph)` | AD tape | Yes | Topological ordering |
| `OptimizerState(OptimizerState)` | Optimizer params | Yes | Adam/SGD state |
| `TidyView(TidyView)` | DataFrame view | Yes | Column-major |
| `GroupedTidyView(GroupedTidyView)` | Grouped view | Yes | Sorted groups |
| `VizorPlot(VizorPlot)` | Plot specification | Yes | Grammar-of-graphics |

## Type Enum Variants

The static type system mirrors the runtime types with additional inference machinery:

- **Concrete types:** I32, I64, U8, F32, F64, Bool, Str, Void, Bytes, ByteSlice, StrView, Tensor, Buffer, Regex, Bf16, F16, Complex, Range, Slice, SparseTensor, TidyView, GroupedTidyView
- **Composite types:** Array(Box<Type>), Tuple(Vec<Type>), Struct{..}, Class{..}, Record{..}, Enum{..}, Map(Box<Type>, Box<Type>), Fn{params, ret}
- **Inference types:** Var(u32), Unresolved(String), Error

## Determinism Guarantees

### Data Structure Choices

| Structure | Usage | Why |
|-----------|-------|-----|
| `BTreeMap` | Struct fields, scopes, type env | Alphabetical iteration order |
| `DetMap` | CJC `Map` type | MurmurHash3 (fixed seed) + insertion-order iteration |
| `Vec` | Arrays, tuples, tensor data | Ordered by construction |
| `Rc<RefCell<..>>` | COW buffers | Deterministic reference counting |
| `ObjectSlab` | GC replacement | Deterministic index allocation |

### Zero HashMap Usage

Verified: `grep -r "use std::collections::HashMap" crates/` returns zero matches across all 20 crates. All map-like structures use `BTreeMap`, `BTreeSet`, or `DetMap`.

## Byte-Level Serialization (cjc-snap)

Every Value variant has a canonical binary encoding:
- Tag byte identifies the variant
- NaN canonicalized to `0x7FF8_0000_0000_0000`
- Struct fields serialized in sorted key order
- Content-addressable via SHA-256 hash
- Encoding is platform-independent (little-endian throughout)
