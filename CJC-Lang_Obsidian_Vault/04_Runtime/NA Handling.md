---
title: NA Handling
tags: [runtime, values, data]
status: Implemented (v0.1.7, 2026-04-19)
---

# NA Handling

CJC-Lang models missing data as a first-class value, distinct from `nil`. This allows tidy data operations to preserve typed columns while still representing absent observations, as in R or pandas.

## The `NA` keyword

`NA` is a lexer keyword (`TokenKind::Na`) introduced in v0.1.7. It evaluates to a runtime `Value::Na`.

```cjcl
let x: Any = NA;
if is_na(x) {
    print("missing");
}
```

`NA` is **not** the same as `nil`:

- `nil` means "no value, unit" — typically function return or uninitialized binding.
- `NA` means "a value is expected here but is missing" — a statistical/data modelling concept.

## Value model

- Lexer: `TokenKind::Na`
- Parser / AST: produces an `ExprKind::Literal(Literal::Na)` (or equivalent NA literal form)
- Runtime: `Value::Na` — carried through both [[cjc-eval]] and [[cjc-mir-exec]] via the canonical [[Wiring Pattern]]

See [[Value Model]] for the full value enum.

## Builtins

| Builtin | Signature | Meaning |
|---|---|---|
| `is_na(x)` | `Any -> Bool` | `true` iff `x` is `Value::Na` |
| `is_not_null(x)` | `Any -> Bool` | inverse of `is_na` (also rejects `nil`) |
| `fill_na(x, default)` / `fillna(x, default)` | `Any, Any -> Any` | returns `default` if `x` is NA, else `x` |
| `drop_na(arr)` | `Array -> Array` | filter array, dropping NA entries; preserves order |
| `coalesce(a, b, ...)` | `Any... -> Any` | first non-NA argument |

Note: `na_count` is **not** a builtin. Compose it when you need it:

```cjcl
fn na_count(arr: Array) -> i64 {
    let mut total = 0;
    let mut i = 0;
    while i < len(arr) {
        if is_na(arr[i]) { total += 1; }
        i += 1;
    }
    total
}
```

## DataFrame integration

`cjc-data` exposes per-column NA helpers (v0.1.7):

- `df_fill_na(df, col_name, fill_val)` — replace NA in one column
- `df_drop_na(df)` — drop rows where any column is NA

See [[DataFrame DSL]] for the full data-frame surface.

## Determinism

`drop_na` and `coalesce` iterate in array order; `df_drop_na` iterates in row-insertion order. No hash-based reordering is introduced. See [[Determinism Contract]].

## Test coverage

`tests/test_na_handling.rs` — **18 tests, all passing**. Covers `NA` lexing, `is_na` across value kinds, `fill_na` type coercion, `drop_na` preserves order, and `coalesce` short-circuit semantics.

## Related

- [[Value Model]]
- [[Builtins Catalog]]
- [[DataFrame DSL]]
- [[Wiring Pattern]]
- [[Version History]]
