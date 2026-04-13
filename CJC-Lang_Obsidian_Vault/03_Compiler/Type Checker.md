---
title: Type Checker
tags: [compiler, types]
status: Implemented
---

# Type Checker

**Crate**: `cjc-types` — `crates/cjc-types/src/lib.rs` (~5,926 LOC).

## Summary

Static type checking with Hindley-Milner style unification. Annotates the [[AST]] with types before lowering to [[HIR]]. Tracks an effect set per function to support verification later in the pipeline.

## Public types

- `Type` — the type enum: `I32`, `I64`, `F32`, `F64`, `Bool`, `Str`, `Tensor { shape }`, `Array`, `Tuple`, `Struct`, `Class`, `Record`, `Enum`, `Function`, `Var`, `Error`.
- `TypeEnv` — symbol tables (`BTreeMap` for [[Deterministic Ordering]]).
- `TypeSubst` — substitution map from type variables to types.
- `EffectSet` — flags: `IO`, `GC`, `allocation`, `nondeterminism`.

## Algorithm

1. Walk the AST recording a `TypeEnv` as scopes are entered.
2. For every expression, generate a type using either a concrete rule or a fresh `Var`.
3. Unify when constraints meet (calls, assignments, operator arguments, return types).
4. Resolve substitutions at the end of each declaration.
5. Report unresolved variables as errors in the [[Diagnostics]] bag.

Type variables and substitution are backed by `BTreeMap` — even in type inference, insertion order must be deterministic.

## Effect registry

The effect set attached to every function is used by:
- [[NoGC Verifier]] — checks that `@nogc` functions do not transitively reach anything with `allocation` or `GC` set.
- Future determinism verification — `nondeterminism` is reserved.
- [[MIR Optimizer]] — some passes refuse to hoist IO-effectful code out of loops.

## Error codes

Type errors fall in the `E0100–E0249` range per `docs/spec/error_codes.md`. See [[Error Codes]].

## Related

- [[AST]]
- [[Types]]
- [[HIR]]
- [[NoGC Verifier]]
- [[Diagnostics]]
- [[Error Codes]]
