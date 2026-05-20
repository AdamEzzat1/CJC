---
title: "ADR-0024: Tier-0 Slot Resolution"
tags: [adr, mir, executor, perf]
status: Accepted (Stages 1+2+3+4 shipped; Stage 5 pending)
date: 2026-05-20
---

# ADR-0024: Tier-0 Slot Resolution

## Status

**Accepted.** Stages 1 (data foundation), 2 (slot resolution + executor
pattern coverage), 3 (executor frame fast-path), and 4 (closures +
match patterns slot-resolved) shipped on `master` as of 2026-05-20.
Stage 5 (retire `Var(String)`) is the sequenced follow-up tracked in
[[Tier-0 Interpreter Perf]].

**First measurable win:** Stage 4 produced a 15% wall-clock speedup on
chess_rl_v2 (802s → 680s) — the load-bearing real workload. Stage 3's
gains were below Windows microbench noise; Stage 4 surfaces above it
because match-heavy code is finally hitting the frame fast-path.

## Context

[[cjc-mir-exec]] is, despite its README framing, a **tree-walking
interpreter**, not a register machine. There is no opcode dispatch loop,
no register file, no bytecode. `eval_expr` recursively walks `MirExpr`
nodes via Rust's match dispatch. (See [[ADR-0001 Tree-form MIR]] — the
tree form is intentional and load-bearing.)

The pre-Tier-0 bench shows two unexpected facts:

1. **`cjc-eval` is often *faster* than `cjc-mir-exec`**. Both are
   tree-walkers; MIR-exec adds HIR→MIR lowering overhead. The "MIR is
   faster" assumption is inverted.
2. **`lookup` is the heaviest single workload** in the microbench at
   `bench/interp_micro/`. Variable resolution walks
   `Vec<BTreeMap<String, Value>>` by string key on every read.

The textbook Tier-0 optimisations for an interpreter — computed-goto
dispatch, typed register slots, opcode superinstructions — **do not
apply** here because there is no bytecode loop. The optimisations that
*do* apply are:

- **Variable lookup acceleration** (this ADR — the biggest lever)
- Dispatch-call inline caching (shipped as T0-c, see `0b4d007`)
- Per-shape arithmetic fast-paths (T0-d, optional)

## Decision

Replace string-keyed `BTreeMap` scope-chain lookup with **statically
resolved slot indices** for function-local variables.

### Design — three discriminator types, two emission gates

#### 1. Two `MirExprKind` variants for variable references

```rust
pub enum MirExprKind {
    ...
    /// Unresolved fallback path. Used for top-level functions,
    /// captured variables, pattern bindings, and anything outside
    /// the slot tracker's view.
    Var(String),
    /// Slot-resolved local. Emitted by HirToMir when the reference
    /// statically resolves to a function-local binding (param or let).
    VarLocal { name: String, slot: u32 },
    ...
}
```

The `name` field on `VarLocal` is retained for debugging and
printing; runtime dispatch uses `slot` only (eventually — Stage 3).

#### 2. `MirFunction.local_count: u32`

Records the total slot count needed by the function's frame. The Stage 3
executor will use this to size `Vec<Value>` in one allocation. `0` means
"no slot resolution performed; fall back to name-based lookup."

#### 3. `HirToMir` slot tracker (Stage 2)

Per-function state added to `HirToMir`:

```rust
scope_stack: Vec<BTreeMap<String, u32>>,
slot_counter: u32,                  // monotonic; never decrements
slot_resolution_active: bool,       // gated for closures & match arms
```

Rules:

- **`lower_fn`** wraps each function body in `enter_function(params)` /
  `exit_function()` and sets `local_count = slot_counter` on exit.
  Param defaults are lowered with the outer tracker (saved/restored).
- **`lower_block`** opens/closes a lexical scope via `push_scope` /
  `pop_scope`. The slot counter is **monotonic per function** —
  shadowing across sibling blocks consumes distinct slots; the small
  space cost buys a much simpler implementation than reclaiming slots.
- **`lower_stmt::Let`** lowers the initialiser *before* binding the new
  name, so `let x = x + 1` resolves the RHS `x` to the outer slot
  (parent binding) and the LHS `x` gets a fresh slot.
- **`lower_expr::Var(name)`** searches the scope stack innermost-first.
  Hit → `VarLocal { name, slot }`. Miss → `Var(name)` (executor
  fallback).
- **Closures** save/restore the outer tracker around the lifted body
  lowering and emit `local_count: 0` for the lifted function. Capture
  *expressions* (evaluated in the outer scope at `MakeClosure` time)
  *do* slot-resolve.
- **Match arm bodies** temporarily disable slot resolution. Pattern-
  bound names are not tracked at lowering time; emitting a `VarLocal`
  with an outer slot would happen to work in Stage 2 (executor still
  does name lookup) but bake in a latent bug for Stage 3's
  `frame[slot]` reads.

### Executor frame fast-path (Stage 3, commit `9e65aa5`)

`MirExecutor` gains two fields:

```rust
frame: Vec<Value>,           // flat slot array, sliced per call
frame_stack: Vec<usize>,     // saved frame.len() per active call
```

On `call_function` entry, if `func.local_count > 0`:
1. `frame_stack.push(self.frame.len())` — save the base for this call
2. `frame.resize(base + local_count, Value::Void)` — reserve N slots
3. Bind params: write `frame[base + i] = arg_i` for each param

On `call_function` exit (return, error, or tail-call trampoline):
1. `frame_stack.pop()` — discard this call's base
2. `frame.truncate(base)` — release this call's slots

`MirStmt::Let` gained a `slot: Option<u32>` field, populated by
`HirToMir::lower_stmt`. When `Some(s)`, the executor writes
`frame[base + s] = init_value`. `MirExprKind::VarLocal { slot, .. }`
reads via `frame[base + slot]` (a single indexed `Vec` access),
falling back to scope-chain lookup only if no frame is active
(Stage 2 structural compat).

The slot identity has to live on `MirStmt::Let` because
**branch-unbalanced shadowing**: `if c { let x } else { let y }`
assigns slots `N` and `N+1` to DIFFERENT names depending on which
branch runs. The executor cannot re-derive the right slot at runtime
by counting Lets — only one branch executes, so the counter would
be off for the other branch's references.

#### Double-bookkeeping (the Stage 5 cleanup target)

Stage 3 still calls `self.define(name, val)` alongside every
`frame_set(slot, val)`, and `self.assign(name, val)` alongside every
slot-targeted assign. Reason: closure captures (`MakeClosure`
captures address outer locals by name), match arm body references
(Stage 2 disabled slot resolution there, so references emit
`Var(name)` and rely on the scope chain), and pattern bindings all
still need the name binding. Until Stage 4 lifts the closure/match
restriction and Stage 5 retires `Var(String)`, the double cost
stays.

This is why the measured Stage 3 win is below the Windows noise
floor — `frame_get` is faster than `BTreeMap::get`, but the
`define`+`frame_set` pair on writes nullifies most of the read-side
savings. Stage 5 is where the sharper signal lands.

### Closures + match patterns (Stage 4, commit `5edadd6`)

Stage 4 lifts the `local_count = 0` cap that Stages 2-3 placed on
lambda-lifted closure bodies and match arm bodies.

**Closures**: in `HirToMir::lower_expr::Closure`, the Stage 2
disable-and-restore tracker hack is replaced with the standard
function-lowering pattern:

```rust
let saved = self.save_tracker();
self.enter_function(&lifted_params);     // params -> slots 0..N
let lifted_body = ...lower(body)...;     // refs slot-resolve
let local_count = self.exit_function();
self.restore_tracker(saved);
```

Lifted closures now go through the same `call_function` entry/exit
in the executor as regular functions — no new executor code needed.

**Match patterns**: `MirPattern::Binding(String)` became
`MirPattern::Binding { name: String, slot: Option<u32> }`. The slot
is populated by `lower_pattern` (now `&mut self`) which walks the
pattern tree and `define_local`s each Binding it finds, including
nested ones in `Tuple`/`Struct`/`Variant`.

`HirExprKind::Match` lowering opens a lexical scope per arm:
push_scope → lower_pattern (assigns slots) → lower body (refs
resolve) → pop_scope. Sibling arms consume distinct slot ranges
(monotonic counter — same trade-off as if/else branches).

The executor's `match_pattern` return type widened from
`Vec<(String, Value)>` to `Vec<(String, Option<u32>, Value)>`. The
arm handler writes `frame[base + slot]` when slot is `Some`, AND
still calls `self.define(name, val)` (same double-bookkeeping
pattern as Stage 3's Let — Stage 5 cleanup target).

**Measured impact**:

| Stage | chess_rl_v2 wall-clock (release) |
|---|---|
| Stage 3 | 802 s |
| **Stage 4** | **680 s** (-15%) |

First clearly measurable Tier-0 perf win on a real program. The
microbench `lookup` workload is unchanged (it doesn't use match or
closures), confirming Stage 4's gains are specific to the workloads
that actually exercise the new fast paths.

### Executor pattern coverage (Stage 2)

The standalone `VarLocal` read arm in `eval_expr` was added in Stage 1.
But the executor had **six other pattern-match sites** that recognised
only `Var(name)`:

1. Tail-call detection in `exec_body` (callee position)
2. Tail-call detection in `MirStmt::Return` (callee position)
3. `eval_call` main dispatch
4. `eval_call` Field-object static-method shortcut
5. `exec_assign` target
6. `exec_assign` Field/Index `.object` inner

If Stage 2 emitted `VarLocal` for, say, an assignment target (`x = 5`
where `x` is a local) and the executor's pattern only matched `Var`,
the catch-all returned `"invalid assignment target"`. Stage 2 updates
each site to match `Var(name) | VarLocal { name, .. }` via or-patterns
so both variants route identically. No semantic change.

## Consequences

### Positive

- **Foundation for Stage 3 fast path**: the actual `frame[slot]` read
  is a single indexed array access. Expected ≥3× speedup on the
  `lookup` workload (clear of Windows noise).
- **Determinism unchanged**: `BTreeMap`-backed tracker preserves
  insertion-order iteration; slots are assigned in declaration order
  (params first, then `let` bindings depth-first through the body).
- **Backward compatible**: every `MirFunction` constructor not touched
  in Stage 1/2 gets `local_count = 0` (and the executor falls back to
  name lookup), so downstream MIR-builder code still works.

### Negative

- **Public struct breaking change**: adding `local_count: u32` to
  `MirFunction` and a new variant to `MirExprKind` means every
  exhaustive match and every `MirFunction { ... }` constructor must be
  updated. Stage 1's "additive variant + new field" was *intended* to
  be invisible behaviour-wise but still required updating 9 source
  files + (in Stage 2) 8 integration-test files for the
  `MirFunction.local_count` field. Future struct evolution should
  consider `#[non_exhaustive]`.
- **Slot counter waste**: a function with
  `if c { let x } else { let y }` consumes two slots even though only
  one branch executes. Trade-off accepted (frame is `Vec<Value>` so
  the waste is small).
- **Closures + match arm bodies sit at `local_count = 0`**: Stage 2
  leaves them on the name fallback. Stage 4 will lift them.

### Risk register

- Stage 3 must be careful with **match-arm bodies**. If a future
  refactor enables slot resolution inside arm bodies without also
  tracking pattern bindings, references to a shadowed outer local
  will read `frame[outer_slot]` and silently get the wrong value.
  Stage 2 disables resolution inside arm bodies as a safety net; this
  ADR documents that the safety net is **intentional**.
- The pre-existing `examples/physics_ml/pinn_heat_1d_pure.cjcl`
  parity tests fail in any worktree because the demo file was never
  committed to git. This is **not** a Tier-0 regression and is called
  out here so future Tier-0 work doesn't mistake it for one.

## Alternatives considered

### A. Skip slot emission for callee positions

To preserve the existing `if let MirExprKind::Var(callee_name)` pattern
in tail-call detection and dispatch, we could refuse to emit `VarLocal`
when a `Var(name)` is in callee position. This would let us skip
updating the six executor pattern sites.

**Rejected** because:
- The asymmetric emission rule is hard to keep correct
  (every recursion through `lower_expr` has to know its "callee-ness")
- Stage 3's fast path needs the slot in callee position anyway (a
  local-bound function value read should still hit `frame[slot]`)
- The or-pattern fix at the six sites is a 6-line change with zero
  behaviour delta

### B. Bytecode VM

Move `cjc-mir-exec` to a real bytecode VM with opcode dispatch loop.
This unlocks all the textbook interpreter optimisations.

**Rejected for Tier-0** because:
- It's a multi-week rewrite with high regression risk against the
  AST-eval parity gate
- ADR-0001 says tree-form is the canonical MIR; CFG and SSA overlays
  are *derived*
- Tier-0's goal is pre-JIT wins. The natural next move after the
  Tier-0 ceiling is a real backend (Cranelift / copy-and-patch), not
  a bytecode VM

### C. Bind slots to `MirStmt::Let` (annotate the binding itself)

Instead of looking the slot up from the lowering's parallel scope map,
attach `slot: Option<u32>` directly to `MirStmt::Let`. The executor's
Let path reads the slot directly.

**Deferred to Stage 3** — the handoff doc notes that the cleanest
implementation does add a slot to `MirStmt::Let` at Stage 3 time, but
Stage 2 left `MirStmt::Let` untouched to keep the diff focused. Either
the executor maintains its own scope-stack-to-slot map during exec, or
Stage 3 amends `MirStmt::Let` with the slot. Both paths are open.

## Tests

10 new focused unit tests in `crates/cjc-mir/src/lib.rs` cover:

| Test | What it pins down |
|---|---|
| `t0b_stage2_params_get_sequential_slots` | Params slot in declaration order; `local_count` matches |
| `t0b_stage2_let_binding_gets_next_slot_after_params` | `let` continues the counter past the param range |
| `t0b_stage2_let_rhs_resolves_to_outer_for_shadowing` | `let x = x + 1` — RHS sees outer, LHS gets new slot |
| `t0b_stage2_main_function_not_slot_resolved` | `__main` stays at `local_count: 0` |
| `t0b_stage2_unresolved_name_stays_as_var` | Top-level / globals emit `Var(name)` |
| `t0b_stage2_closure_body_stays_on_name_fallback` | Lifted closure has `local_count: 0` and emits `Var` |
| `t0b_stage2_capture_expr_in_outer_is_slot_resolved` | Capture expressions in `MakeClosure` *do* slot-resolve |
| `t0b_stage2_nested_blocks_use_distinct_slots` | Sibling shadowing consumes two slots (counter monotonic) |
| `t0b_stage2_match_arm_bodies_disabled` | Arm bodies emit `Var`, not `VarLocal` (Stage 4 defer) |
| `t0b_stage2_function_calls_dont_disturb_outer_slots` | Sequential `lower_fn` calls don't contaminate each other |

Plus the load-bearing parity gate (`cargo test --test
test_builtin_parity` — 10/10) and the chess RL v2 end-to-end suite
(`cargo test --test test_chess_rl_v2 --release` — 97/97) confirm Stage 2
introduces zero behaviour change.

## Source

- Commit: `bd99522` (Stage 2)
- Commit: `d005d40` (Stage 1)
- Handoff: `docs/T0_INTERPRETER_PERF_HANDOFF.md`
- Concept note: [[Tier-0 Interpreter Perf]]
- Crates touched: `cjc-mir`, `cjc-mir-exec`

## Related

- [[MIR]] — `MirExprKind` and `MirFunction` data model
- [[cjc-mir-exec]] — the tree-walker getting accelerated
- [[ADR-0001 Tree-form MIR]] — why we keep tree form (not bytecode)
- [[ADR-0010 Scope Stack SmallVec]] — alternative scope optimisation
  approach (proposed, not implemented; this ADR is a different lever)
