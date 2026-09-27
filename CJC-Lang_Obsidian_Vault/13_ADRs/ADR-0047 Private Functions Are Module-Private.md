# ADR-0047 Module-Scoped Name Resolution; Private Functions Are Module-Private

- **Status:** Accepted (2026-09-27). Supersedes this ADR's first version (2026-09-26, commit `94111b5`). That version kept the aliases and rejected cross-module uses of private names with a static check. As a result, a private `fn abs` in one module made the builtin `abs` unusable in the modules that imported it, and it did not fix the two executor divergences below.
- **Crates:**
  - `cjc-module`: new `resolve` module (`resolve_modules`, `explain_undefined`); `merge_programs` now resolves first and creates no aliases; `merge_resolved`; symbol-import fallback.
  - `cjc-ast`: new `visit_mut` (mutating visitor).
  - `cjc-eval` and `cjc-mir-exec`: multi-file entry points run the resolved ASTs.
- **Related:** [[Module System]], [[ADR-0003 Backward-compatible run_program]]

## Context

In multi-file programs, a function declared without `pub` was callable from
any module that imported its module. The two executors also resolved
multi-file names in different ways, and each was wrong somewhere:

| Program | AST-eval | MIR-exec | Correct |
|---|---|---|---|
| Modules `alpha` and `beta` each have a private `helper`; `alpha.fa()*100 + beta.fb()` | 202 | 101 | 102 |
| `alpha` imports `beta` and calls `beta`'s pub `fb` | 11 | error: undefined `fb` | 11 |
| `alpha` has a private `fn abs`; `main` calls the builtin `abs` | alpha's `abs` | alpha's `abs` | builtin |

The two causes:

- **`merge_programs`** (MIR-exec) renamed each module's functions to
  `module::name`, but left call sites untouched. It then aliased *every*
  function of the entry module's imports under its bare name, private ones
  included. So private functions were callable, the first same-named alias
  won, and nested imports had no alias at all.
- **The AST evaluator** ran every module in one interpreter with a single
  namespace, so the last function registered under a name won.

`check_visibility` checked only `import m.Symbol`. Each half of the code
said the other half enforced privacy.

## Decision

**One name-resolution pass, shared by both executors.**
`cjc_module::resolve_modules` clones each module's AST and rewrites every
free identifier that names a function to that function's qualified name.
It also renames top-level `fn` declarations to match. For a bare name `n`
in module `M`:

1. If a binding in scope shadows `n`, leave it: parameters, `let`, `for`,
   lambda parameters, match-pattern bindings, and module-level `let`/`const`.
2. Otherwise, if `M` defines a top-level `fn n`, it becomes `M::n`. The
   entry module's names stay bare, so `main` is still `main`.
3. Otherwise, if one of `M`'s direct imports provides `n`, it becomes that
   module's `X::n`. `import X` provides `X`'s `pub` functions;
   `import X.f [as g]` provides `f` as `g` if `f` is `pub`. The first
   import in source order wins.
4. Otherwise, leave `n` alone. The executor resolves it as in a
   single-file program: a builtin, an enum variant, or "undefined".

Consequences of these rules:

- **Private functions are out of scope outside their module.** This is
  rule 3: only `pub` functions are provided by imports.
- **A `pub fn` can call its module's private helpers.** Rule 2 applies
  inside the module.
- **Builtins are never shadowed by another module's function.** Rule 4.

**Both executors run the resolved ASTs.** `merge_programs` no longer
creates aliases, and the AST evaluator's shared namespace can no longer
collide, because names are qualified.

**Diagnostics.** A private function of another module is simply not in
scope. That is exact, but by itself unhelpful. So the resolver records
unresolved names that are private functions elsewhere, and
`explain_undefined` appends a note to the executors' "undefined
function/variable" errors. Both executors call it, so the text is
identical:

```
undefined function `double`
note: `double` is a private function of module `utils`; mark it `pub` to use it from other modules
```

This is a note, not a static error: a name may be a private function
elsewhere and also a builtin (rule 4), and `cjc-module` has no authoritative
list of builtins.

**Also fixed.** `import m.f` with a lowercase `f` was classified as the
module path `m/f`. When no such file existed, the import was dropped
silently, even though `classify_import`'s comment described a fallback.
`build_module_graph` now tries `(m, symbol f)` in that case.

## Evidence

- All three table rows now give the correct value in both executors
  (`tests/final_phase_hardening_before_vm/test_module_visibility.rs`,
  `same_named_private_helpers_do_not_collide`,
  `nested_import_resolves_in_both_executors`,
  `private_fn_does_not_shadow_builtin_in_importer`).
- The tests run every program through both executors' multi-file entry
  points. Before this, the AST-eval multi-file path had no tests.
- Scoping, precedence, recursion and lambdas inside modules, `import m.f as
  g`, and determinism are covered (the resolver's output is compared across
  repeated runs).

## Consequences

- **Breaking:** calling a non-`pub` function of another module now fails.
  One test relied on it (`test_module_system::module_exec_two_files`) and
  now marks its function `pub`. Programs that depended on either
  executor's old collision behaviour change results. The old results were
  wrong, and the two executors disagreed.
- `VisibilityViolation` and `check_visibility` are unchanged from before
  this ADR.

## Not decided here

- **Qualified calls** (`utils::double(x)`) do not parse. The docs showed
  them; the docs are corrected. Supporting them needs parser work.
- **Module-level `let` globals and struct/enum names** are still shared
  across modules the old way. Only functions are resolved.
- **Transitive access.** A module sees only its direct imports. Importing
  `a` does not make `a`'s imports visible.
