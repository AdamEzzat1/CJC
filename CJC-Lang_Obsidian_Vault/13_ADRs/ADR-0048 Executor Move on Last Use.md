---
title: "ADR-0048: Executor Move on Last Use"
tags: [adr, executor, mir, eval, runtime, perf, cow]
status: Accepted
date: 2026-09-25
---

# ADR-0048: Executor Move on Last Use

- **Status:** Accepted (2026-09-25)
- **Crates:** `cjc-runtime` (owned-args dispatch), `cjc-eval`, `cjc-mir-exec`
- **Related:** [[ADR-0024 Tier-0 Slot Resolution]] (MIR binding storage),
  [[ADR-0044 Quantum Value Semantics]] (COW circuit builders, a future
  consumer of this mechanism)
- **Numbering note:** drafted as ADR-0047. That number was taken on
  `master` by [[ADR-0047 Private Functions Are Module-Private]] before this
  landed, so this record is ADR-0048.
- **Tests:** `tests/test_move_on_last_use.rs` (parity, aliasing, closures,
  error path, determinism, O(n) scaling) plus
  `cjc-runtime` unit tests `builtins::owned_dispatch_tests`.

## Context

CJC-Lang values are immutable from the program's point of view. Arrays are
`Value::Array(Rc<Vec<Value>>)`, and the "mutating" collection builtins
return a new value:

```cjcl
a = array_push(a, v);
```

`array_push`, `array_pop` and `array_reverse` implement this with
copy-on-write: `Rc::make_mut` mutates in place when the `Rc` is unique and
clones the `Vec` otherwise. Their comments claimed that for
`arr = array_push(arr, val)` "refcount is 1 → zero-copy push". **That was
false.** A temporary probe measured `Rc::strong_count` of the argument
inside the builtin dispatcher:

| Situation | strong_count |
|---|---|
| `a = q_x(a, 0)`, `a` unaliased | 2 (binding + argument vector) |
| same, `a` aliased | 3 |
| same, inside a function | 5 |

Both executors behave the same way. There are two independent causes, and
both have to be fixed:

1. **The executor copies the binding into the argument vector.** Evaluating
   the argument `a` clones the `Value` (an `Rc` increment) and leaves the
   binding alive until the assignment overwrites it *after* the call.
2. **The builtin only borrows its arguments.**
   `dispatch_builtin(name, args: &[Value])` cannot move out of `args`, so
   `array_push` must `Rc::clone` the array. The argument vector's own
   reference then pins the count at ≥ 2 even when nothing else holds
   the value.

So `make_mut` always copied, and every push was O(n). A timed MIR loop
(`cjcl run --mir-opt`) showed quadratic growth: 10k pushes took 1.9 s, 20k
took 10.3 s and 40k took 46.9 s. The quantum circuit builders under
ADR-0044's value semantics pay O(gates) per gate for the same reason.

## Decision

Add **move on last use** at the executor level, paired with an
**owned-argument entry point** in the runtime.

### 1. The rewrite (both executors)

For an assignment

```
x = f(e₁, …, x, …, eₙ)
```

the executor takes the fast path iff **all** of the following hold:

| # | Condition | Why |
|---|---|---|
| C1 | The target is a bare variable `x` (eval `Ident`; MIR `Var(name)` or `VarLocal{slot}`) | Field and index targets read and write a containing value; out of scope. |
| C2 | The value is a direct call whose callee is a bare name `f` (eval `Ident`; MIR `Var`) | Method calls, closures held in locals and computed callees can run user code. |
| C3 | `f ∈ OWNED_ARG_BUILTINS` (`array_push`, `array_pop`, `array_reverse`) | These are stateless, never call back into the program, and follow the error contract in §3. |
| C4 | `f` is not shadowed: no user function named `f`, no variant constructor `f` (eval), and no binding named `f` | The call must resolve to the shared builtin exactly as ordinary dispatch would. The binding check is stricter than ordinary dispatch, deliberately. |
| C5 | Exactly one argument is the bare variable `x` (MIR: the same `Var(name)` or the same `VarLocal{slot}` as the target) | `f(x, x)` would need `x` twice. The ordinary path handles it with ordinary semantics. |
| C6 | `x` currently has a binding in the storage `exec_assign` writes | See §4. If the binding can't be found, fall back. |

When the pattern does not match, the fast path returns *before evaluating
anything* and the ordinary path runs unchanged.

When it matches:

1. Evaluate every **other** argument left to right, exactly as `eval_call`
   would.
2. **Move** `x` out of its binding with `mem::replace(binding, Value::Void)`
   into its argument slot. The binding's reference is gone, so the
   argument vector holds the only one (unless the value is aliased
   elsewhere).
3. (MIR only) Run the same CANA trace accounting that `dispatch_call`
   runs, now factored into `trace_builtin_call`, so traces are the same
   on both paths.
4. Call `cjc_runtime::builtins::dispatch_builtin_owned(f, &mut args)`.
5. On `Ok(Some(v))`: write `v` into the binding.
   On `Err(msg)`: **restore** the argument into the binding, then return
   the same `Runtime(msg)` error the ordinary path would.

#### Why the move happens after the sibling arguments are evaluated

Reading a bare variable has no side effects. Taking `x` at the end is
therefore indistinguishable from reading it in its argument position, and
it has one important benefit: sibling expressions that also mention `x`
(`a = array_push(a, len(a))`) run against the intact binding. So C5 only
needs to restrict *bare* occurrences, and nested reads need no scan.
"Exactly once as a bare argument" is the precise rule.

### 2. The runtime half: `dispatch_builtin_owned`

```rust
pub const OWNED_ARG_BUILTINS: &[&str] = &["array_push", "array_pop", "array_reverse"];
pub fn is_owned_arg_builtin(name: &str) -> bool;
pub fn dispatch_builtin_owned(name: &str, args: &mut [Value])
    -> Result<Option<Value>, String>;
```

The owned arms move the array out of `args` (leaving `Void`) instead of
`Rc::clone`-ing it. Any other name is forwarded to `dispatch_builtin` and
never modifies `args`. The borrowed arms in `dispatch_builtin` stay for
every other caller. They share the validation and the `make_mut` kernel
(`check_array_*`, `array_*_rc`), so results and error messages are the same
by construction, and a unit test checks this for every owned name.

The signature change is local on purpose. Changing
`dispatch_builtin(&[Value])` across roughly 370 arms would be a broad
refactor with no benefit for builtins that don't do COW.

### 3. Error path: the placeholder is never observable

CJC-Lang has no in-language `try/catch`, so a runtime error ends the
program. The executor state can still be seen afterwards through the
embedding APIs: the REPL reuses one `Interpreter` and exposes
`list_bindings()`, and a `MirExecutor` can `exec` a second program. The
binding must therefore never be left as `Void` after a failure:

- **Sibling argument fails** (step 1): the move hasn't happened yet, so
  the binding is untouched.
- **The builtin fails** (step 4): `dispatch_builtin_owned` guarantees that
  **`args` is unmodified whenever it returns `Err`**. Every owned arm
  validates arity, types and preconditions (for example, `array_pop` on an
  empty array) *before* it takes anything. The executor moves `args[pos]`
  back into the binding and then propagates the error.
- **Between take and write-back** no user code runs (C3), so nothing can
  read the placeholder in the meantime. This is why the allowlist is
  explicit and not "any builtin": `array_map` or `array_sort_by`
  invoke user closures, and a closure that reads a top-level binding would
  see `Void`.

Tests: `error_path_restores_binding_{eval,mir}`,
`error_path_empty_pop_restores_binding_both` and
`error_in_sibling_arg_leaves_binding_untouched` fail the call, then run a
follow-up program on the same executor that reads the binding.

### 4. MIR slot resolution (ADR-0024)

After Stage 5a, MIR bindings live in exactly one place:

- `VarLocal { slot }`: the frame slot `frame[frame_base + slot]`.
  `exec_assign` writes the frame **only** and errors if no frame is active.
- `Var(name)`: the scope chain (top-level `__main`, lifted closure bodies,
  and the unresolved fallback).

`binding_mut(target)` returns exactly the storage `exec_assign` writes: the
frame slot (or `None` when no frame is active, which falls back to the
ordinary path) or the innermost scope binding. C5 compares slots, not
names, for `VarLocal`, so a shadowing inner `let a` (a distinct slot, since
slot counters are monotonic per function) is never confused with the outer
`a`. MIR builtin callees are always emitted as `Var(name)`. A `VarLocal`
callee is a local binding that holds a callable and never reaches the
builtin tables, so C2 excludes it.

`cjc-eval` has a single scope chain; `lookup_mut` finds the innermost
binding, which is the same one `assign` writes.

### 5. Closures and captures

Both executors capture **by value at closure creation**. The captured
`Value`s are cloned into the closure's `env` (eval: `Value::Closure` built
by lexical capture; MIR: `MakeClosure`). Consequences:

- A binding that was captured is never *shared storage* with the closure.
  Moving `x` out of the binding only drops the binding's own reference. The
  closure's snapshot keeps its reference, so the refcount stays ≥ 2 and
  `make_mut` copies. The closure keeps seeing its snapshot.
  "`x` is not captured by a closure" is thus guaranteed by the capture
  model, not by a static check, and it holds by construction
  (`parity_closure_capture_is_a_snapshot`).
- Inside a lifted closure body, captured names are ordinary parameters of
  the lifted function (MIR `Var`, eval scope bindings) that are rebound
  from `env` on every call. Moving one mutates only that call's
  parameter (`parity_move_inside_closure_body`).

**Pre-existing bug fixed along the way (`cjc-hir`).** The closure parity
tests exposed a MIR-only failure that had nothing to do with the move. HIR
capture analysis treated *every* free name as a capture unless it was in a
hard-coded 20-name `is_builtin` list. A closure calling `array_len` or
`array_push` therefore "captured" the builtin, `MakeClosure` evaluated an
unbound `Var`, and MIR failed with `undefined variable` while eval
succeeded. The fix adds a single condition: a free name is captured only if
`is_defined` in an enclosing HIR scope, which is cjc-eval's rule ("captured
iff it resolves to a live local binding"). Every binding form (`let`,
`const`, params, `for` variables, pattern bindings, lambda params)
registers with `define_var`, so no real capture is dropped, and the
`is_builtin` skip is kept unchanged.

If CJC-Lang ever adds by-reference capture (shared cells), C1–C6 would
need a "not captured by reference" condition. The capture analysis in
`cjc-hir` (and `collect_var_refs` in eval) is where to compute it.

### 6. Aliasing semantics are unchanged

The move removes *one* reference: the binding being overwritten. Every
other holder (another binding, an array element, a closure env, the
caller's binding of a parameter) keeps its reference, and COW copies as
before. Programs cannot tell the fast path from the ordinary path except
by timing. `determinism_matches_non_move_reference` checks this by running
the same computation written so the move never applies and comparing the
outputs byte for byte.

## Consequences

**Wins**

- `x = array_push(x, v)` is amortized O(1) per push when `x` is unaliased,
  at top level and inside functions, in both executors. See the measurements
  below.
- The same pattern is available to any future COW builtin: add it to
  `OWNED_ARG_BUILTINS` and give it an owned arm that follows the
  §3 contract. ADR-0044's quantum builders need an owned
  `dispatch_quantum` arm (the executor half is shared).

**Costs and limits**

- One pattern check per assignment. It returns early on the first
  mismatched condition. Non-call assignments fail at C2.
- Only direct `x = f(…x…)` assignments benefit. `let y = f(x)` (with `x`
  still live), `x = g(f(x))` and user-function calls do not. Moving into a
  user function would let user code (the callee) run while the caller's
  binding is a placeholder. That is sound for true locals but not for
  top-level bindings that functions can read, and it would need the
  liveness analysis this ADR avoids.
- `array_pop` returns `(last, rest)`, so the common `let t = array_pop(a); a = t[1];`
  shape is not a direct self-assignment and still copies.

## Measurements

Release build on an idle machine. CLI figures are the best of 3 runs of the
top-level push loop (`cjcl run --mir-opt`, ~128 ms process startup included).

| Push loop | 10k | 20k | 40k | 160k |
|---|---|---|---|---|
| before, `cjcl run --mir-opt` | 1.9 s | 10.3 s | 46.9 s | – |
| after, `cjcl run --mir-opt` | 125 ms | 135 ms | 156 ms | 285 ms |

Scaling gate (`scaling_*` in `tests/test_move_on_last_use.rs`, 40k/20k,
minimum of 7 interleaved runs, three consecutive runs of the suite):

| Mode | thread-CPU ratio | wall at 40k |
|---|---|---|
| eval, top level | 1.76–2.77 | 38–46 ms |
| eval, in function | 1.93–2.24 | 35–40 ms |
| MIR-opt, top level | 1.76–2.57 | 28–35 ms |
| MIR-opt, in function | 1.71–2.47 | 21–28 ms |

Temporary instrumentation (a counter of `make_mut` calls that saw
refcount > 1) recorded **zero** copies in every mode, size and scope.

### Measuring complexity on a loaded machine

The first gate compared wall-clock time and reported MIR at 5–9× while
eval stayed at ~2×. It wasn't copying (the counter read zero), and the
cost didn't grow with array length: timing 10k-push chunks from inside
the program showed chunks alternating between ~15 ms and ~300 ms,
independent of array length, in *both* executors. The machine was at
100% CPU with 15+ `rustc` processes from parallel builds, so the thread
was being descheduled. MIR's 20k run (~17 ms) fits in one scheduling
slice while its 40k run (~35 ms) doesn't, so the 40k run absorbed a
stall almost every time and the minimum of N runs couldn't avoid it.
With the machine idle, the same MIR runs are linear.

The gate therefore uses **thread CPU time**: Windows
`QueryThreadCycleTime` (cycle-accurate; `GetThreadTimes` only advances
at the 15.6 ms tick) and `clock_gettime(CLOCK_THREAD_CPUTIME_ID)` on
Linux and macOS. It measures the two sizes interleaved and serializes
the four scaling tests. Wall time is printed alongside.
