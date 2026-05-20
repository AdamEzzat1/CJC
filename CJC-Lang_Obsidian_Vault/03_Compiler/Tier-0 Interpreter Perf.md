---
title: Tier-0 Interpreter Perf
tags: [compiler, runtime, executor, perf]
status: In progress (T0-a + T0-c + T0-b S1+S2+S3+S4+S5a shipped; S5b pending)
---

# Tier-0 Interpreter Perf

Multi-stage perf work for [[cjc-mir-exec]]. Goal: 2-5× speedup pre-JIT,
preserving determinism and parity with [[cjc-eval]].

The textbook framing of Tier-0 (computed-goto dispatch, typed register
slots, opcode superinstructions) **does not apply** to this codebase
because `cjc-mir-exec` is a tree-walker, not a register machine. See
[[ADR-0024 Tier-0 Slot Resolution]] for the architectural finding.

## Status board (2026-05-20)

| Item | State | Commit | Notes |
|---|---|---|---|
| **T0-a** Microbench harness | ✅ shipped | `0b4d007` | `bench/interp_micro/` |
| **T0-c** Inline cache for `dispatch_call` | ✅ shipped | `0b4d007` | Cache hits below bench noise floor |
| **T0-b Stage 1** Data foundation | ✅ shipped | `d005d40` | `VarLocal` variant + `MirFunction.local_count`; purely additive |
| **T0-b Stage 2** HIR→MIR slot resolution | ✅ shipped | `bd99522` | Producer + executor pattern coverage |
| **T0-b Stage 3** Executor frame fast-path | ✅ shipped | `9e65aa5` | `Vec<Value>` per call frame; `frame[slot]` reads; perf payoff (double-bookkeeping keeps writes at parity until Stage 5) |
| **T0-b Stage 4** Closures + match patterns | ✅ shipped | `5edadd6` | Lifts `local_count = 0` cap from closures and match arm bodies. **15% wall-clock speedup on chess_rl_v2** (802s → 680s) — match-heavy workloads now hit the frame fast-path |
| **T0-b Stage 5a** Drop double-bookkeeping | ✅ shipped | `2f8db84` | Slot-resolved Let/Assign/match-bind/param-bind skip `define`/`assign`. **Microbench lookup ratio 0.70 → 0.50 (~2× win)** but **chess_rl_v2 regressed 680s → ~950s (-40%)** — workload-specific, flagged for Stage 5b investigation |
| **T0-b Stage 5b** Investigate + finish cleanup | ⏸ next | — | Profile chess_rl_v2 to find the regression; then audit + delete `Var(String)` variant and `scopes` field |
| **T0-d** `eval_binary` fast-paths | optional | — | Per-shape arithmetic; ~1 hr |
| **T0-e** `is_known_builtin` static set | optional | — | ~30 min |

Regression baseline (post-Stage 5a):
- `cargo test --workspace --lib` — 2,524 / 2,524 pass
- `cargo test --test test_builtin_parity` — 10 / 10 pass
- `cargo test --test test_match_patterns` — 26 / 26 pass
- `cargo test --test test_closures` — 26 / 26 pass
- `cargo test --test test_chess_rl_v2 --release` — 97 / 97 pass (correct), but **wall-clock regressed**: Stage 4 680s → Stage 5a 892s/1005s (two runs, ~30-50% slower). Workload-specific regression flagged for Stage 5b investigation.

## The three building blocks (Stages 1+2)

```rust
// 1. New MirExprKind variant (Stage 1)
pub enum MirExprKind {
    ...
    Var(String),                              // unresolved fallback
    VarLocal { name: String, slot: u32 },     // resolved fast path
    ...
}

// 2. New MirFunction field (Stage 1)
pub struct MirFunction {
    ...
    pub local_count: u32,    // 0 = no resolution; use name lookup
}

// 3. HirToMir slot tracker (Stage 2)
pub struct HirToMir {
    ...
    scope_stack: Vec<BTreeMap<String, u32>>,
    slot_counter: u32,                  // monotonic per function
    slot_resolution_active: bool,       // gated for closures/match arms
}
```

Currently `VarLocal` and `Var` route to the same name-lookup code paths
in the executor — Stage 2 is purely a structural refactor, no
behaviour change. Stage 3 lights up the actual fast path.

## Stage 3 as it shipped

`MirExecutor` gained two fields:

```rust
pub struct MirExecutor {
    ...
    scopes: Vec<BTreeMap<String, Value>>,    // existing fallback path
    frame: Vec<Value>,                        // NEW: flat slot array
    frame_stack: Vec<usize>,                  // saved frame.len() per call
}
```

Hook points wired:
- **`call_function` entry**: if `func.local_count > 0`, push
  `frame_stack(frame.len())` + resize `frame` to `base + local_count`
  + bind params to `frame[base..base+n_params]`.
- **`call_function` exit (Return, Err, TailCall)**: pop `frame_stack`,
  truncate `frame` back to base. The TailCall path is critical —
  without it, tight tail-call loops would grow the frame unbounded.
- **`MirStmt::Let`**: if `slot` is `Some`, write `frame[base+slot]`.
  Also call `self.define(name, val)` for the safety net (closure
  captures + match arm bodies reference by name).
- **`eval_expr::VarLocal { slot, .. }`**: try `frame_get(slot)` first,
  fall back to scope-chain lookup if no frame is active.
- **`exec_assign::VarLocal { slot, name }`**: write `frame[base+slot]`
  AND `self.assign(name, val)`. Same safety-net rationale.

The `slot` field had to land on `MirStmt::Let` (not derivable by the
executor at runtime) because **branch-unbalanced shadowing** —
`if c { let x } else { let y }` — assigns DIFFERENT slots to
DIFFERENT names depending on which branch executes. Re-deriving by
counting Lets at runtime would give the wrong slot.

The helpers `frame_base()`, `frame_get(slot)`, `frame_set(slot, val)`
are marked `#[inline(always)]` — they're on the per-reference hot
path (~50,000 calls per inner bench iteration).

## Stage 4 as it shipped (commit `5edadd6`)

Stage 4 lifts the `local_count = 0` cap that Stage 2 placed on
**lambda-lifted closure bodies** and **match arm bodies**. After
Stage 4, the only paths still on the name-only fallback are
top-level statements in `__main` (the synthetic wrapper).

### Closures: drop the disable-and-restore hack

In `HirToMir::lower_expr::Closure`, the Stage 2 pattern was:

```rust
let saved = self.save_tracker();
self.slot_resolution_active = false;
self.scope_stack.clear();
let lifted_body = ...lower(body)...;
self.restore_tracker(saved);
// lifted MirFunction { ..., local_count: 0 }
```

Stage 4 replaces this with the standard function-lowering pattern:

```rust
let saved = self.save_tracker();
self.enter_function(&lifted_params);  // captures + original params -> slots 0..N
let lifted_body = ...lower(body)...;
let local_count = self.exit_function();
self.restore_tracker(saved);
// lifted MirFunction { ..., local_count }
```

The lifted closure is now treated exactly like any other function.
Its params (captures first, then original) get slots `0..N`; lets in
the body get slots after. The executor's Stage 3 `call_function`
entry/exit handles the closure frame automatically — lambda-lifted
closures dispatch through the same code path as user-defined
functions.

### Match patterns: slot field on `MirPattern::Binding`

`MirPattern::Binding(String)` became `MirPattern::Binding { name:
String, slot: Option<u32> }`. The `lower_pattern` method (now
`&mut self`) walks the pattern tree, calls `define_local` for every
`Binding` it finds (including nested ones in `Tuple`/`Struct`/
`Variant` patterns), and records the assigned slot in the returned
MIR pattern.

`HirExprKind::Match` lowering now opens a lexical scope per arm:

```rust
arms.iter().map(|arm| {
    self.push_scope();
    let pattern = self.lower_pattern(&arm.pattern);  // assigns slots
    let body = self.lower_expr(&arm.body);           // refs slot-resolve
    self.pop_scope();
    MirMatchArm { pattern, body }
})
```

The arm body's references to pattern-bound names AND outer-scope
locals both resolve to slots. Sibling arms consume distinct slot
ranges (slot counter is monotonic per function — same trade-off as
`if`/`else` branches).

The executor's `match_pattern` return type widened from
`Vec<(String, Value)>` to `Vec<(String, Option<u32>, Value)>`. When
a binding carries `Some(slot)`, the arm handler writes
`frame[base + slot] = val` in addition to `self.define(name, val)`
(same double-bookkeeping pattern as Stage 3's Let, retired in Stage 5).

### Measured win (chess_rl_v2)

The chess RL v2 test suite runs through `match` heavily — every
piece type, every move category, every board state evaluation uses
match expressions. This is exactly the workload Stage 4 targeted:

| Stage | chess_rl_v2 wall-clock (release) |
|---|---|
| Pre-Tier-0 (Stage 0) | 802 s |
| Stage 3 | 802 s (frame fast-path, but match arm bodies still on name fallback) |
| **Stage 4** | **680 s** (-15%) |

This is the first **clearly measurable Tier-0 perf win on a real
program** — Stage 3's lookup-workload microbench gains were below
Windows noise, but Stage 4 moves the needle on the real chess RL
workload because the match expressions inside the per-move scoring
loop now hit the frame fast-path.

## Stage 5a as it shipped (commit `2f8db84`)

Stage 5a drops the `self.define()` / `self.assign()` double-bookkeeping
from slot-resolved paths and establishes single-source-of-truth
discipline:

> `slot.is_some()` → frame is the only path
> `slot.is_none()` → name binding is the only path

The two never overlap. For slot-resolved fns (every regular fn +
every lifted closure + every match arm body after Stages 2-4), the
name path is unused; for `__main` (`local_count = 0`) the name path
is the only path.

### Sites updated

- `MirStmt::Let` handler — `match slot { Some -> frame_set, None -> define }`.
- Match arm binding loop — same conditional.
- `exec_assign::VarLocal` — `frame_set(slot, val)` only.
- `call_function` param binding — `frame[i] = val` when `pushed_frame`, else `define`.
- `eval_call::VarLocal` callee — split from `Var(name)`; uses `frame_get(*slot).cloned()` to inspect the Value and dispatch as `Fn`/`Closure` directly (the old shared path went through `dispatch_call`, which relied on scope-chain `lookup` that's now empty for slot-resolved bindings).

### Sharper microbench win

The lookup workload finally moved into the "≥30% win" range the
handoff said would clear Windows noise:

| Workload | Stage 4 mir/eval | Stage 5a mir/eval |
|---|---|---|
| arith | ~0.70 | ~0.40-0.80 |
| **lookup** | **~0.70** | **~0.50 (≈2× faster than eval)** |
| call | ~0.70 | ~0.50 |
| mixed | ~0.50 | ~0.50 |

### Chess RL regression (workload-specific, flagged for 5b)

The chess_rl_v2 wall-clock REGRESSED:

| Stage | chess_rl_v2 |
|---|---|
| Stage 3 | 802 s |
| Stage 4 | 680 s |
| **Stage 5a run 1** | 892 s |
| **Stage 5a run 2** | 1005 s |

Both Stage 5a runs are consistently slower than Stage 4 (30-50%).
Since the microbench shows the opposite signal, this is
workload-specific — not a global regression.

**Hypothesis**: the new `eval_call::VarLocal` callee path
(`frame_get(*slot).cloned()` to inspect the Value) is exercised on
every indirect call through a local-bound `Value::Closure` (Adam
optimizer state, training callbacks). The clone cost matches the
old `lookup(name).cloned()` path, but the OLD `dispatch_call` had
early-exit fast paths for `functions.contains_key(name)` and
`is_known_builtin(name)` that the new VarLocal path skips. For
chess RL's hot paths, those early exits may have been firing more
than expected.

**Stage 5b should**:
1. Profile chess_rl_v2 (use the existing `profile_zone_*` builtins
   from the v2.3 work) to confirm the regression source.
2. Either restore early-exit fast paths at the front of the new
   VarLocal callee dispatch, OR find a way to inspect the Value
   via `&Value` instead of clone-and-match.
3. Decide whether to ship Stage 5b's deeper cleanup (delete
   `Var(String)` + `scopes` field) once the chess RL regression
   is resolved.

The Stage 5a commit ships the regression alongside the win because:
1. Functional correctness is solid (97/97 chess_rl_v2 still passes).
2. The microbench gains are real and reproducible.
3. Reverting would lose the lookup-workload win and put us back at
   Stage 4's double-bookkeeping cost.

### Measured win (Windows microbench)

5 runs of the `lookup` workload (mir_warm vs eval, ms) at Stage 3:

| Run | mir_warm | eval | ratio |
|---|---|---|---|
| 1 | 112.24 | 128.27 | 0.88 |
| 2 | 90.00 | 127.98 | 0.70 |
| 3 | 84.68 | 131.76 | 0.64 |
| 4 | 106.17 | 113.76 | 0.93 |
| 5 | 53.23 | 77.34 | 0.69 |

Stage 4 ratios on the same workload stayed in the same range
(0.63-0.82) — `lookup` doesn't use match or closures, so Stage 4
doesn't directly help it. The 15% chess_rl_v2 speedup at Stage 4 is
the more informative measurement: it confirms that real, match-heavy
programs *do* see a clear Tier-0 win.

mir-exec is 7-36% faster than eval (median ~30%) on the lookup
workload. The handoff warned that Windows ~2× run-to-run variance
hides anything below 30% — and that's what we see. The frame
fast-path saves on reads, but the **double-bookkeeping** (still
calling `self.define()` alongside `frame_set()` to keep closure
captures and match arm body references working) keeps writes at
parity. **Stage 5 is where the sharper microbench win should land**
— once every variable reference is slot-indexed, `define()` can
disappear and the BTreeMap scope chain can be deleted entirely.

## Why AST-eval is faster than MIR-exec on the bench

Both `cjc-eval` and `cjc-mir-exec` are tree-walkers. `cjc-eval` walks
the AST directly; `cjc-mir-exec` walks MIR. MIR is more uniform (fewer
variants, simpler dispatch), but the MIR-exec call path adds:

1. HIR→MIR lowering (currently per `run_program_with_executor`; the
   bench amortises this by lowering once outside the timing loop)
2. Generated MIR is slightly more verbose than AST (e.g. wrapping
   block expressions, explicit if-as-expression nodes)

Stage 3's frame fast-path is the single biggest lever to flip this
back: `frame[slot]` vs `BTreeMap::get(name)` is dozens of nanoseconds vs
hundreds.

## Determinism budget (preserved)

- `BTreeMap` (not `HashMap`) for the slot tracker scope_stack. Even
  though the tracker is internal to lowering and never leaks into
  output, the rule is uniform across the codebase.
- Slot counter is monotonic per function: lowering twice with the same
  input produces byte-identical MIR (same slot indices for the same
  names).
- Save/restore around closure bodies preserves outer slot state across
  nested lowerings.

## Why Windows bench noise matters

Identical microbench workloads vary by ~2× run-to-run on Windows
(scheduling, JIT warmup of supporting code, etc.). Only optimisations
producing **≥30% wins** have a cleanly measurable signal. Stage 3's
expected 3-5× on the `lookup` workload will clear the noise floor;
smaller wins like T0-c and T0-d will not show up cleanly without
re-engineering the harness (multi-run aggregation, OS scheduler pinning).

## Reading order

1. [[ADR-0024 Tier-0 Slot Resolution]] — the design decision
2. `docs/T0_INTERPRETER_PERF_HANDOFF.md` — the operational resume point
   for Stage 3
3. [[cjc-mir-exec]] — the tree-walker being accelerated
4. [[MIR]] — the data model `VarLocal` extends

## Related

- [[ADR-0001 Tree-form MIR]] — why we keep tree form
- [[ADR-0010 Scope Stack SmallVec]] — alternative scope optimisation
  (proposed, not the same lever)
- [[Parity Gates]] — the load-bearing correctness gate
