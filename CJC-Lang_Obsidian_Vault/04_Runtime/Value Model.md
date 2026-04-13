---
title: Value Model
tags: [runtime, types]
status: Implemented
---

# Value Model

**Source**: `crates/cjc-runtime/src/value.rs`.

## Summary

The `Value` enum is the unified runtime representation of every CJC-Lang value. Both [[cjc-eval]] and [[cjc-mir-exec]] use it.

## Variants (surveyed)

- **Numeric**: `Int(i64)`, `Float(f64)`, `F16`, `Bf16`, `Complex(ComplexF64)`
- **Boolean**: `Bool(bool)`
- **Text**: `String(...)`
- **Container**: `Array(...)`, `Tuple(...)`, `Struct(...)`, `Enum(...)`
- **Tensor**: `Tensor(Tensor)` — see [[Tensor Runtime]]
- **Specialized**: `Span(...)`, `DateTime(...)`, `Complex(...)`, `Quantized(...)`
- **Functions**: closure values with captured environment
- **Nil / Unit**

(The code survey identified at least 23 distinct variants across the runtime.)

## Immediate vs reference-counted

From `docs/architecture/byte_first_type_inventory.md`, runtime values split into:

- **Immediate types** — stack-friendly scalars, no heap.
- **Reference-counted (COW) types** — tensors, strings, large arrays; share backing storage via [[COW Buffers]].
- **Interior-mutable types** — wrapped in `Rc<RefCell<..>>` where mutation is needed.
- **Computed types** — closures, lazy views.

The [[NoGC Verifier]] draws a line: if a function only manipulates immediate types and known-stack/arena-allocated buffers, it is provably `@nogc`.

## Canonical byte form

For serialization and hashing, NaN is canonicalized to `0x7FF8_0000_0000_0000` and floats are emitted in a fixed bit order. This keeps [[Binary Serialization]] bit-identical across runs.

## Related

- [[Tensor Runtime]]
- [[Memory Model]]
- [[COW Buffers]]
- [[NoGC Verifier]]
- [[Binary Serialization]]
