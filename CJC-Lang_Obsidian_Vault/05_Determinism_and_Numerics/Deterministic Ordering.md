---
title: Deterministic Ordering
tags: [determinism, data-structures]
status: Implemented
---

# Deterministic Ordering

## The rule

> **No `HashMap` or `HashSet` in CJC-Lang compiler or hot-path runtime code.** Use `BTreeMap` and `BTreeSet`.

This is in CLAUDE.md, the architecture docs, and every crate's module docstrings (e.g., `cjc-module/src/lib.rs` says "All internal maps use BTreeMap / BTreeSet for deterministic iteration order").

## Why

`HashMap` uses a randomized hash function (DoS protection in Rust's default) which means iteration order is nondeterministic across runs, platforms, and even program invocations. Any output that depends on iteration order — error messages, MIR register numbering, serialized format, pretty-printer output — would drift across runs.

`BTreeMap` iterates in key order (which is stable). That alone is enough to make a lot of the compiler pipeline deterministic.

## Where it matters

- **[[Type Checker]]** — `TypeEnv`, `TypeSubst` use BTreeMap.
- **[[HIR]]** — capture lists sorted deterministically.
- **[[MIR]]** — block enumeration, phi operand order, use-def lists.
- **[[NoGC Verifier]]** — call-graph traversal order.
- **[[Module System]]** — `ModuleGraph`, cycle detection, topological merge.
- **[[cjc-eval]]** and **[[cjc-mir-exec]]** — variable bindings, closure captures.
- **[[Binary Serialization]]** — field order.

## Exceptions

There is a `det_map.rs` in `cjc-runtime` that provides a deterministic user-facing map — documented in `byte_first_type_inventory.md` as using MurmurHash3 + insertion-order tracking. This is for *user* maps where a BTreeMap's sort order would be the wrong semantic.

## Related

- [[Determinism Contract]]
- [[Value Model]]
- [[Module System]]
