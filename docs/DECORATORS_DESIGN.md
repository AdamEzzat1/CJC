# Decorators as a Language Feature — Current State + Path Forward

**Date:** 2026-06-09
**Status:** Survey + design — most infrastructure already shipped; gap is user-defined decorators
**Scope:** §4.5 of `HANDOFF_NEXT_SESSION_v3.md`

This doc surveys what's already in the tree, identifies the actual
remaining gap, and lays out the implementation path for the missing
piece.

---

## 1. What's already shipped (much more than the handoff suggested)

The handoff said: "Required work: Parser → AST → HIR → MIR → runtime
wrapper execution." A code survey shows **every layer except
user-defined decorator semantics is already implemented**:

| Layer | Status | Where |
|---|---|---|
| Lexer | ✅ shipped | `TokenKind::At` (`@`) recognized; `crates/cjc-lexer/src/lib.rs` |
| Parser | ✅ shipped | `parse_fn_decl_with_vis` accepts `decorators: Vec<cjc_ast::Decorator>`; parses `@name` and `@name(arg1, arg2)` form. `crates/cjc-parser/src/lib.rs:610` |
| AST | ✅ shipped | `cjc_ast::Decorator` struct with name + optional args |
| HIR | ✅ shipped | `pub decorators: Vec<String>` field on HIR function items (`crates/cjc-hir/src/lib.rs:88`) |
| MIR | ✅ shipped | `pub decorators: Vec<String>` field on `MirFunction` (`crates/cjc-mir/src/lib.rs:149`) |
| Eval runtime | ✅ partial | Hardcoded support for `@memoize` (cache by arg-hash) and `@trace` (log entry/exit). `crates/cjc-eval/src/lib.rs:3949-3950` |
| MIR-exec runtime | ✅ partial | Same as eval. `crates/cjc-mir-exec/src/lib.rs:4014-4015` |
| Tests | ✅ shipped | `tests/test_decorators.rs` — 35 tests covering lex / parse / eval / mir-exec / parity for `@memoize` and `@trace` |

The actual user-visible feature works today:

```cjcl
@memoize
fn fib(n: i64) -> i64 {
    if n < 2 { return n; }
    return fib(n - 1) + fib(n - 2);
}
print(fib(30));  // O(n) with memoization, not O(2^n)

@trace
fn step(x: i64) -> i64 {
    return x * 2;
}
print(step(5));  // prints "[trace] step(5) enter" + "[trace] step(5) => 10"
```

Both `@memoize` and `@trace` are runtime-active in both executors with
parity-tested behavior.

---

## 2. The actual gap — user-defined decorators

The current implementation is **hardcoded-builtin only**. If a user
writes:

```cjcl
fn my_logger(f: Fn, args: Tuple) -> Any {
    print("calling user-decorated function");
    let result = f(args);
    print("done");
    return result;
}

@my_logger
fn my_fn(x: i64) -> i64 {
    return x + 1;
}
print(my_fn(5));
```

…nothing happens with the `@my_logger` decoration. The executor scans
the decorator list and only matches against the literal strings
`"memoize"` and `"trace"`; everything else is silently ignored.

That's the §4.5 gap. **General user-defined decorators need:**

1. The executor to recognize that `my_logger` is a user-defined
   function (not a builtin).
2. A way to wrap the decorated function (`my_fn`) as a callable Value
   that can be passed as an argument.
3. A way to invoke `my_logger(wrapped_fn, args)` at the right point
   in the dispatch sequence.

`Value::Closure` exists (`crates/cjc-runtime/src/value.rs:268`), so
representation is solved. What's missing is the wiring.

---

## 3. Two implementation strategies

### 3.1 At-call rewrite (strategy A — recommended)

When `f(args)` is invoked and `f` has a decorator `@d`:

1. Look up `d` as a user-defined function. If not found AND not a
   hardcoded builtin (memoize/trace), emit a parse-error-quality
   diagnostic (`unknown decorator @d on function f`).
2. If found: construct a `Value::Closure` wrapping `f`'s body
   (capturing the function's parameters + body MIR).
3. Invoke `d(wrapped_f_closure, args_tuple)`. The decorator's body
   gets to do whatever — log, time (caveat: not deterministic!),
   gate, cache.
4. Return whatever `d` returns.

**Pros:**
- Minimal MIR change — the wrapping happens dynamically at dispatch
  time. No HIR/MIR rewrite needed.
- Decorators can be defined anywhere, even after the decorated
  function (single-pass parser is fine because the lookup is at
  runtime).
- Composition (`@a @b fn foo`) is straightforward: `b` wraps first,
  `a` wraps the result.

**Cons:**
- Per-call overhead: the closure construction + dispatch indirection
  fire every call. Acceptable for `@log` / `@gate` / `@trace`, costly
  for hot loops.
- The "wrap as closure" step needs to encode the function's full
  signature (parameter names, types, body, captures). This is
  ~50-100 LOC of MIR-to-closure conversion.

### 3.2 At-definition rewrite (strategy B)

When the function is defined (HIR or MIR lowering time):

1. Lookup `@d` in scope.
2. Generate a wrapper function: `fn d_wrapped_f(args) { return d(orig_f, args); }`.
3. Replace `f`'s body with a call to `d_wrapped_f`.

**Pros:**
- Decorator dispatch happens at compile time → zero per-call overhead
  past the static wrapper.
- Easier to reason about — the wrapping is visible in the lowered
  IR, not a runtime opaque step.

**Cons:**
- Requires HIR/MIR rewriting pass (~300-500 LOC).
- Forward references break — `@d` must be defined before `f` for the
  lookup to succeed. CJC-Lang's existing function definitions allow
  forward references (recursion, mutual recursion), so this is a
  visible step backward.
- Composition gets more complex (multiple wrapper functions chained).

### 3.3 Recommendation

**Strategy A** for the MVP, **Strategy B** as a Phase 5+ optimization.

The per-call overhead of strategy A is acceptable for the
decorator-as-aspect use cases people actually want (logging,
gating, instrumentation). When and if a workload appears where
decorator overhead is on a hot path, strategy B can be added later
as a transparent optimization — the user-visible semantics don't
change.

---

## 4. Implementation plan (strategy A)

### 4.1 New runtime helpers

```rust
// crates/cjc-runtime/src/decorator.rs (new file, ~150 LOC)

/// Construct a Value::Closure that wraps a regular MirFunction so it
/// can be passed as an argument to a decorator.
pub fn wrap_function_as_closure(
    fn_id: MirFnId,
    fn_name: &str,
    params: &[MirParam],
) -> Value {
    Value::Closure {
        fn_id,
        captured: BTreeMap::new(), // no captures — it's a static fn
        name: fn_name.to_string(),
    }
}
```

### 4.2 Executor dispatch change

In both `cjc-eval` and `cjc-mir-exec`, replace the hardcoded
`if has_memoize { ... } if has_trace { ... }` block with:

```rust
// New: handle ANY decorator, hardcoded or user-defined.
for decorator_name in &func.decorators {
    match decorator_name.as_str() {
        "memoize" => { /* existing memoize logic */ }
        "trace"   => { /* existing trace logic */ }
        // User-defined decorator: look up + invoke.
        other => {
            let decorator_fn = self.find_user_fn(other).ok_or_else(|| {
                MirExecError::Runtime(format!(
                    "unknown decorator @{other} on function {current_name}"
                ))
            })?;
            let wrapped = wrap_function_as_closure(func.id, &func.name, &func.params);
            let args_tuple = Value::Tuple(Rc::new(current_args.clone()));
            // Replace direct execution with a recursive call into the decorator.
            return self.call_user_fn(decorator_fn, vec![wrapped, args_tuple]);
        }
    }
}
// If no decorator matched, fall through to regular execution.
```

### 4.3 Two-fold parity gate

The change MUST preserve:

1. **AST/MIR parity** for every fixture program. Decorator dispatch
   happens identically in both executors.
2. **Behavioral parity with the hardcoded builtins**. Specifically:
   `@memoize` on `fib` must still cache; `@trace` must still produce
   the existing trace output format. The 35 existing tests in
   `tests/test_decorators.rs` are the regression gate.

### 4.4 New tests

```rust
// tests/test_decorators_user_defined.rs (new, ~150 LOC, 8-12 tests)

#[test] fn user_defined_decorator_invoked() { /* basic call */ }
#[test] fn user_defined_decorator_can_modify_args() { ... }
#[test] fn user_defined_decorator_can_replace_result() { ... }
#[test] fn user_defined_decorator_composes_with_builtin() { ... }
#[test] fn multiple_user_defined_decorators_compose_bottom_up() { ... }
#[test] fn unknown_decorator_errors_clearly() { ... }
#[test] fn user_defined_decorator_ast_mir_parity() { /* parity */ }
#[test] fn user_defined_decorator_with_recursive_function() { ... }
```

---

## 5. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Decorator overhead on hot paths | Medium | Document the per-call cost; recommend strategy B optimization if it bites |
| Decorator infinite recursion (decorator calls decorated function) | Medium | Detect by checking the call stack at decorator dispatch — refuse to re-decorate if the decorated function is already on the stack |
| Decorator captures changing types | Low | The wrapped closure has the decorated function's type signature; type-checking decorators stays as-is (decorators take `Fn` + `Tuple`) |
| Determinism break from clock-using decorators | High | Refuse to ship a `@timed` decorator that reads system clock. Document the constraint. Existing memoize/trace are deterministic by construction. |
| Composition order ambiguity (`@a @b fn foo` — which wraps first?) | Low | Document: bottom-up. `@b` wraps first, then `@a` wraps the result. Matches Python convention. |

---

## 6. Estimated effort

| Slice | LOC | Calendar |
|---|---|---|
| `wrap_function_as_closure` helper + executor dispatch change | ~250 | 2-3 days |
| New tests (user-defined decorator suite) | ~200 | 1 day |
| Doc updates (CLAUDE.md, language reference) | ~50 | 0.5 day |
| **Total** | **~500 LOC** | **3-4 days** |

Smaller than the handoff's "largest single item" framing because
most of the infrastructure already exists. The handoff predates the
realization that lexer/parser/AST/HIR/MIR/test-suite are all shipped.

---

## 7. What this design does NOT cover

- **Decorator arguments** beyond simple literal values
  (`@cache_ttl(3600)`). Current parser accepts them; semantics need
  a separate spec.
- **Class-style decorator factories** (`@authorized("admin")` where
  `authorized` itself returns a decorator). Functional but needs
  the strategy-A change to land first.
- **Decorators on non-function items** (struct fields, type aliases).
  Out of scope for §4.5; would need its own design pass.
- **Builtin decorator catalog expansion** (`@log`, `@timed`,
  `@deprecated`, etc.). Easy to add via the existing
  hardcoded-builtin pattern but each one has its own determinism
  and semantic considerations — not bundled with this design.

---

## 8. Pick-up checklist for a future session

When implementing strategy A:

- [ ] Create `crates/cjc-runtime/src/decorator.rs` with `wrap_function_as_closure`
- [ ] Update `crates/cjc-eval/src/lib.rs` decorator dispatch block (around line 3949)
- [ ] Update `crates/cjc-mir-exec/src/lib.rs` decorator dispatch block (around line 4014)
- [ ] Create `tests/test_decorators_user_defined.rs` with 8-12 tests
- [ ] Re-run `tests/test_decorators.rs` (35 existing tests) — none must regress
- [ ] Re-run AST/MIR parity gate
- [ ] Update CLAUDE.md §5 (decorators) to describe user-defined form
- [ ] Update the file-level doc comment in `cjc-mir/src/lib.rs` near `pub decorators: Vec<String>` to note the runtime now supports arbitrary names

---

*Generated as part of the §4.4 / §4.5 sweep. Real implementation
deferred to a focused future session (3-4 days, smaller than the
handoff's "largest single item" budget because most infrastructure is
already shipped).*
