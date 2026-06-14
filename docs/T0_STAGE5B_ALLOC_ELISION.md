# Tier-0 Stage 5b (partial) — `call_function` allocation elision

**Date:** 2026-06-13 · branch `claude/happy-jennings-d673b9`
**Scope:** the allocation-elision sub-win of Stage 5b (ADR-0024). Does
NOT include the larger Stage-5b cleanup (delete `Var(String)` + the
`scopes` chain) — that remains open.
**Gate:** `docs/cana/DETERMINISM_CONTRACT.md`. Allocation-only change;
invariants 1–10 untouched.

## 0. How this started (and a correction)

A session investigating speed/memory wins first proposed "S1: slot-resolve
the fallback bodies," sourced from a comment at `cjc-mir-exec/src/lib.rs`
listing `__main` / closures / match-arm bodies as name-keyed fallbacks.
**That comment was stale (Stage-2 era).** Verified reality (ADR-0024 +
current code): closures and match-arm bodies are ALREADY slot-resolved
(Stage 4); `__main` is excluded *by design* (top-level lets are
name-visible globals — slot-resolving them would break name lookup by
functions; pinned by `t0b_stage2_main_function_not_slot_resolved`). The
named-function hot loops in the bench corpus are already on the fast
path. So S1 dissolved on contact — THE RULE, paid again.

The genuinely live work in the slot-resolution area is **Stage 5b**,
which ADR-0024 flagged with a *known, repeatable* chess_rl_v2 regression
(680s → ~950s on Stage 5a).

## 1. The Stage-5a "regression" reassessed: mostly noise

The documented regression (680 → avg of 892 & 1005) is ~1.4×, **inside
the handoff's own stated ~2× Windows run-to-run noise envelope**
(`T0_INTERPRETER_PERF_HANDOFF.md` finding #3), and the two post-Stage-5a
runs themselves spanned 892–1005 (13%). The controlled min-of-N
microbench, by contrast, showed a clean 2× *improvement*. A new
closure-call microbench (below), aggregated min-of-5, shows **no
closure-call wall-clock regression** on this branch.

Conclusion: the "regression" is largely measurement noise. The ADR's
leading hypothesis (the `VarLocal`-callee path lost `dispatch_call`
early-exits) is also refuted by static analysis: for a *local* closure
the `frame_get(slot).cloned()` path is strictly *less* work than the old
`contains_key` + `is_known_builtin` + scope-walk path. The Stage-5b
cleanup can proceed without fear of a real regression; this doc lands the
contained alloc win found along the way.

## 2. The closure-call microbench (new — fills the gap)

`bench/interp_micro/main.rs` gained two workloads — `clos1` (1 capture)
and `clos8` (8 captures) — each a 50k-iteration loop calling a
slot-resolved local closure (`let f = |x| ...; f(i)`). **Their absence
is precisely why the Stage-5a regression was invisible to the
microbench** (its `call` workload exercises a *builtin*, which never
enters `call_function`).

It also gained a **deterministic global allocation counter**
(`CountingAlloc`). Wall-clock on this stack is dominated by ~2× noise;
allocation *count* is reproducible run-to-run and is the correct
instrument for allocation-level changes.

Discovered cost structure: **~7 heap allocations per closure call**, and
clos1 vs clos8 allocate near-identically (the `env` Vec is one allocation
whether it holds 1 or 8 inline `i64`s) — so env *size* is not the cost,
contrary to the ADR's env-clone hypothesis.

## 3. Root cause + fix

`call_function(name: &str, args: &[Value])` eagerly did
`current_name = name.to_string()` + `current_args = args.to_vec()` on
**every** call — owned state that only the tail-call trampoline needs
(it reassigns them on a `TailCall`). The closure-callee path paid it
twice: it already built an owned `full_args`, which `call_function` then
re-copied.

Fix: `Cow` the trampoline state — `Cow::Borrowed` on the first
(common, non-tail-call) iteration, `Cow::Owned` only when a tail call
actually loops. Contained to one function; no signature or caller
changes.

```rust
let mut current_name: Cow<'_, str>   = Cow::Borrowed(name);
let mut current_args: Cow<'_, [Value]> = Cow::Borrowed(args);
// ... TailCall arm: current_name = Cow::Owned(tco_name); current_args = Cow::Owned(tco_args);
```

## 4. Measured result (deterministic alloc count, before → after)

| workload | before | after | delta |
|---|---|---|---|
| `clos1` (50k closure calls) | 450,056 | 350,055 | **−100,001 (−22.2%)** = −2/call |
| `clos8` (50k closure calls) | 450,105 | 350,104 | **−100,001 (−22.2%)** = −2/call |
| `call` / `mixed` / `arith` / `lookup` | — | — | −1 each (`main`'s name; empty args don't allocate) |

The delta matches the code prediction to the single allocation: 50,000
calls × 2 saved (`to_string` + `to_vec`) + 1 for `main`. Wall-clock
remained within the noise floor (as expected for a 2-of-7 alloc
reduction) — which is exactly why the alloc counter, not the stopwatch,
is the load-bearing measurement here.

## 5. Determinism / parity gate (all green)

- `test_builtin_parity` 10/10 (the load-bearing AST-eval ≡ MIR-exec gate)
- `test_closures` 26/26 · `test_match_patterns` 51/51
- `fixtures` (full parity runner) pass · `test_parity_stress` 11/11 (50-seed FP)

Allocation-only change: no value, iteration order, FP, or RNG touched.

## 6. Follow-ups (still open)

- **Stage 5b proper:** delete `MirExprKind::Var(String)` + the
  `scopes: Vec<BTreeMap<String, Value>>` chain once `__main` is the only
  remaining name-path user. (The `__main`-globals constraint must be
  handled first — likely a dedicated global table.)
- **Deeper closure-call win:** the remaining ~5 allocs/call
  (`frame_get(slot).cloned()` of `fn_name`+`env`, `push_scope`,
  `ArenaStore::new`, frame resize). Interning `fn_name` as `Rc<str>` and
  pooling `ArenaStore` are the next levers — each measurable with the new
  alloc counter.
- **Parity gap (separate bug):** `let f = closure; f(x)` works in
  MIR-exec but errors `undefined function f` in AST-eval. The parity
  suite does not currently cover this form. Invariant-7 concern; worth a
  dedicated fix + fixture.
