---
title: Tier-0 Interpreter Perf
tags: [compiler, runtime, executor, perf]
status: In progress (T0-a + T0-c + T0-b S1 + T0-b S2 shipped; S3-S5 pending)
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
| **T0-b Stage 2** HIR→MIR slot resolution | ✅ shipped | `bd99522` | Producer + executor pattern coverage; this concept note's home |
| **T0-b Stage 3** Executor frame fast-path | ⏸ next | — | `Vec<Value>` per call frame; `frame[slot]` reads; perf payoff |
| **T0-b Stage 4** Closures + match patterns | ⏸ later | — | Lift `local_count = 0` cap from closures and arm bodies |
| **T0-b Stage 5** Remove name fallback | ⏸ later | — | Delete `Var(String)` once everything is slot-indexed |
| **T0-d** `eval_binary` fast-paths | optional | — | Per-shape arithmetic; ~1 hr |
| **T0-e** `is_known_builtin` static set | optional | — | ~30 min |

Regression baseline (post-Stage 2):
- `cargo test --workspace --lib` — 2,515 / 2,515 pass
- `cargo test --test test_builtin_parity` — 10 / 10 pass
- `cargo test --test test_chess_rl_v2 --release` — 97 / 97 pass

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

## Stage 3 plan (next)

Add a flat `Vec<Value>` call frame and route `VarLocal` reads through
it:

```rust
pub struct MirExecutor {
    ...
    scopes: Vec<BTreeMap<String, Value>>,    // existing fallback path
    frame: Vec<Value>,                        // NEW: flat slot array
    frame_stack: Vec<usize>,                  // saved frame.len() per call
}
```

**Hook points:**
- On function entry (`call_function`): grow `frame` by `local_count`;
  push the prior length to `frame_stack`; bind params to
  `frame[base..base+n_params]`
- On function return: pop `frame_stack`; truncate `frame`
- On `MirStmt::Let`: write to `frame[slot]` (requires either adding
  `slot: Option<u32>` to `MirStmt::Let` or maintaining a parallel
  scope→slot map in the executor)
- On `MirExprKind::VarLocal { slot, .. }`:
  `frame[frame_stack.last().unwrap() + slot]`
- On `MirExprKind::Var(name)`: unchanged (closures, top-level, captures)

Expected: 3-5× speedup on the `lookup` workload in the microbench.
This is the **actual** perf payoff.

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
