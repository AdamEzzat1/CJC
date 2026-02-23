# ADR-0009: Vec COW for Value::Array and Value::Tuple

**Status:** Proposed
**Date:** 2025-01-01
**Deciders:** Systems Architect, Technical Lead
**Supersedes:** none

## Context

`Value::Array(Vec<Value>)` and `Value::Tuple(Vec<Value>)` in `cjc-runtime/src/lib.rs` (line 3361, 3366) use plain `Vec<Value>`, which means every pass-by-value in the interpreter performs a **full heap clone** of the array contents. This includes:

- Passing arrays as function arguments
- Returning arrays from functions
- Pattern matching that captures array bindings
- Assignment: `let b = a` where `a` is an Array

For array-heavy CJC programs (data processing, ML feature vectors, for-loop accumulators), this creates O(n) allocations per array operation where O(1) is achievable.

**Audit data:** `grep -rn "Value::Array(" crates/ tests/` produces ~98 match sites across 6 crates:
- `cjc-runtime/src/lib.rs`: ~12 sites (construction, Display, type_name, value_hash, value_equal)
- `cjc-eval/src/lib.rs`: ~35 sites (eval_expr array/tuple construction + destructuring)
- `cjc-mir-exec/src/lib.rs`: ~28 sites (exec_stmt/eval_expr + array dispatch methods)
- `cjc-types/src/lib.rs`: ~8 sites (type inference inspection)
- `cjc-hir/src/lib.rs`: ~6 sites (HIR array literal lowering)
- `cjc-data/src/lib.rs`: ~9 sites (DataFrame row construction)

## Decision

Change `Value::Array` and `Value::Tuple` to use `Rc<Vec<Value>>`:

```rust
// Before
Array(Vec<Value>),
Tuple(Vec<Value>),

// After
Array(Rc<Vec<Value>>),
Tuple(Rc<Vec<Value>>),
```

**COW semantics on mutation:**
- Read-only access: `v.iter()`, `v.len()`, `v[i]` — no change, `Rc<Vec<Value>>` derefs to `Vec<Value>`
- Mutation (`v.push(x)`, `v[i] = y`): use `Rc::make_mut(&mut v).push(x)` — triggers deep copy only when `Rc::strong_count > 1`

**Construction sites:** All `Value::Array(vec![...])` become `Value::Array(Rc::new(vec![...]))`.

**The `Tuple` variant** uses the same treatment.

## Rationale

- **O(1) array passing**: `clone()` on `Value::Array(Rc<Vec<Value>>)` increments a reference count — no heap allocation.
- **COW correctness**: When a shared array is mutated, `Rc::make_mut` ensures the mutation is private to the mutating site. The parity gate (milestone_2_4/parity) validates identical semantics before and after.
- **Minimal change surface**: Only construction and mutation sites change. All read-only match arms (`if let Value::Array(ref v) = ...`) compile without modification due to `Deref`.

## Consequences

**Positive:**
- Array-heavy programs (e.g., for-loop accumulators building large arrays) reduce heap allocation significantly.
- Functions that pass arrays without mutating them pay O(1) instead of O(n).

**Known limitations:**
- `Rc::make_mut` requires `Vec<Value>: Clone`, which means individual elements are cloned on COW trigger. For deeply nested structures, this is still O(n) for the triggered copy (same as before, but less frequent).
- `Rc` is not `Send` — CJC is single-threaded by design, so this is acceptable.
- 98 match sites must be audited. The migration is mechanical but large.

## Implementation Notes

- Crates affected: `cjc-runtime`, `cjc-eval`, `cjc-mir-exec`, `cjc-types`, `cjc-hir`, `cjc-data`
- Primary file: `crates/cjc-runtime/src/value.rs` (after module split per ADR-0008)
- Migration script: `grep -rn "Value::Array(vec" crates/ | xargs sed -i 's/Value::Array(vec/Value::Array(Rc::new(vec/g'` (dry-run only — verify each site)
- Regression gate: `cargo test --workspace` must pass with 0 failures; `cargo test milestone_2_4 -- parity` must pass
- New audit test: `tests/audit_tests/test_audit_cow_array.rs` validates COW semantics and parity
