# ADR-0047 Private Functions Are Module-Private

- **Status:** Accepted (2026-09-26)
- **Crates:** `cjc-module` (`check_visibility`, new `enforce_visibility`), `cjc-eval` and `cjc-mir-exec` (multi-file entry points call it)
- **Related:** [[Module System]], [[ADR-0003 Backward-compatible run_program]]

## Context

A function declared without `pub` was callable from any module that
imported its module. Nothing enforced it, and the two halves of the
visibility code each said the other one did:

- `merge_programs` aliased *every* function of an imported module under its
  unprefixed name, citing `check_visibility` for enforcement.
- `check_visibility` said module-level imports were "enforced during alias
  creation" and checked nothing for them. It only checked `import m.Symbol`.

The aliasing of private functions is load-bearing. Merging renames each
function to `module::name` but does not rewrite call sites inside bodies, so
a `pub fn` that calls a private helper finds the helper only through its
unprefixed alias.

The executors also disagree on how multi-file names resolve. `cjc-mir-exec`
uses the prefixed merge plus aliases; `cjc-eval` runs every module in one
interpreter with a single namespace. A fix in only one resolver would make
the executors accept different programs.

The gap was found when `tests/final_phase_hardening_before_vm/`, a suite
that had never been compiled, was wired in. Its test asserting that private
functions are not aliased failed.

## Decision

1. **Rule:** a module may not reference, by bare name, a function that
   another module declares without `pub`. This covers calls `f(x)` and
   function values `g(f)`. `import m` exposes `m`'s `pub` functions only.
2. **Enforcement is a static check on the ASTs** (`check_visibility`, rule 2),
   not a change to name resolution. A name is reported only if all of these
   hold:
   - the referencing module neither defines it nor binds it anywhere (as a
     `let`, `const`, `for` variable, parameter, or pattern binding);
   - another module declares it as a private top-level `fn`;
   - no module declares a `pub fn` with that name.

   The binding test is scope-insensitive on purpose, so shadowing can
   never cause a false positive.
3. **Both executors' multi-file entry points call `enforce_visibility`**
   before running, so they reject the same programs with the same message.
   The CLI (`cjcl run --multi-file`) already called `check_visibility`, so
   it gets the rule automatically.
4. **Aliasing and resolution are unchanged.** Private functions are still
   aliased, so intra-module helper calls keep working. No other module can
   reach the alias, because the check rejects any reference to it.

## Consequences

- **Breaking:** multi-file programs that call a non-`pub` function of an
  imported module now fail with, for example:
  ``visibility error: function `double` is private to module `mathlib` and
  cannot be used from module `main` (mark it `pub` to export it)``. One
  existing test relied on this (`test_module_system::module_exec_two_files`)
  and now marks its function `pub`. The documented examples already use
  `pub fn`.
- **Builtin shadowing is still loud rather than right.** Suppose a module
  has a private `fn mean` and the importer calls `mean(x)` intending the
  builtin. Previously both executors silently called the private function;
  now the program is rejected. Resolving it to the builtin needs
  module-scoped name resolution in both executors.
- Parity: the multi-file AST-eval entry point had no tests. The new tests
  run valid programs through both executors and compare the results.

## Not decided here

- **Call-site rewriting and module-scoped resolution.** These would let the
  merge alias only `pub` functions, fix the builtin-shadowing case, and
  unify the two executors' resolvers.
- **Private structs and records used by bare name.** Only `import m.Struct`
  is checked today.
