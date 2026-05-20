---
title: Tier-0 Interpreter Perf
tags: [compiler, runtime, executor, perf]
status: In progress (T0-a + T0-c + T0-b S1+S2+S3 shipped; S4-S5 pending)
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
| **T0-b Stage 4** Closures + match patterns | ⏸ next | — | Lift `local_count = 0` cap from closures and arm bodies |
| **T0-b Stage 5** Remove name fallback | ⏸ later | — | Delete `Var(String)` + scopes chain; double-bookkeeping vanishes here |
| **T0-d** `eval_binary` fast-paths | optional | — | Per-shape arithmetic; ~1 hr |
| **T0-e** `is_known_builtin` static set | optional | — | ~30 min |

Regression baseline (post-Stage 3):
- `cargo test --workspace --lib` — 2,523 / 2,523 pass (was 2,515; +8 new Stage 3 tests)
- `cargo test --test test_builtin_parity` — 10 / 10 pass
- `cargo test --test test_chess_rl_v2 --release` — 97 / 97 pass (~13 min, MIR-heavy end-to-end ML training)

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

### Measured win (Windows microbench)

5 runs of the `lookup` workload (mir_warm vs eval, ms):

| Run | mir_warm | eval | ratio |
|---|---|---|---|
| 1 | 112.24 | 128.27 | 0.88 |
| 2 | 90.00 | 127.98 | 0.70 |
| 3 | 84.68 | 131.76 | 0.64 |
| 4 | 106.17 | 113.76 | 0.93 |
| 5 | 53.23 | 77.34 | 0.69 |

mir-exec is 7-36% faster than eval (median ~30%). The handoff
warned that Windows ~2× run-to-run variance hides anything below
30% — and that's what we see. The frame fast-path saves on reads,
but the **double-bookkeeping** (still calling `self.define()`
alongside `frame_set()` to keep closure captures and match arm body
references working) keeps writes at parity. **Stage 5 is where the
sharper win lands** — once every variable reference is slot-indexed,
`define()` can disappear and the BTreeMap scope chain can be
deleted entirely.

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
