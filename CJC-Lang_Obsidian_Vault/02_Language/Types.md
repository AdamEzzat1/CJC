---
title: Types
tags: [language, types]
status: Implemented
---

# Types

CJC-Lang has a statically-checked type system with Hindley-Milner style unification, grounded in the [[cjc-types]] crate.

## Base types

| Type | Description |
|---|---|
| `i32`, `i64` | Signed integers |
| `f32`, `f64` | IEEE 754 floats |
| `bool` | Booleans |
| `str` | UTF-8 strings |
| `Tensor` | N-dimensional array with dtype |
| `Array` | Dynamic array |
| `Tuple` | Fixed-size heterogeneous |
| `Struct`, `Class`, `Record` | Named aggregates |
| `Enum` | Tagged unions |
| `Function` | First-class function type |
| `Any` | Dynamic / polymorphic escape hatch |

The code survey of `crates/cjc-types/src/lib.rs` also reports `Complex`, `F16`, `Bf16` as runtime variants (see [[Value Model]]).

## Type inference

The type checker uses unification with a `TypeSubst` backed by `BTreeMap` for deterministic substitution order — every corner of the compiler prefers `BTreeMap` over `HashMap` to preserve [[Deterministic Ordering]].

## Effect registry

`cjc-types` also tracks an `EffectSet` per function with flags for `IO`, `GC`, `allocation`, and `nondeterminism`. These effects feed into the [[NoGC Verifier]] and are part of how the compiler proves that `@nogc` functions really do not allocate.

## Type annotations are required on function parameters

```cjcl
fn f(x: i64) { ... }   // required
fn f(x) { ... }        // rejected
```

This is intentional — it keeps parser + type inference simple and makes overload dispatch in [[Dispatch Layer]] predictable.

## Tensor types

Tensor types carry shape metadata: `Tensor { shape: ... }`. Not every shape is resolved statically; dynamic shapes are allowed. The [[Tensor Runtime]] classifies dtype at runtime: `f64`, `f32`, `i64`, `bool`, and (in some modules) `f16`, `bf16`, `complex64`.

## The `Any` type

`Any` is the explicit dynamic escape hatch. Use it for polymorphic helpers where the type system can't statically prove soundness. It exists because closing off all dynamism would make the standard library clumsy.

## Traits

CJC-Lang has trait declarations (`trait` keyword) with partial implementation. **Needs verification** of current completeness — the `CJC_CODEBASE_AUDIT.md` lists trait resolution as part of hardening work.

## Related

- [[Syntax]]
- [[cjc-types]]
- [[Type Checker]]
- [[Dispatch Layer]]
- [[Value Model]]
- [[Tensor Runtime]]
- [[NoGC Verifier]]
