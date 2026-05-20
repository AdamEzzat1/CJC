# Tier-0 Interpreter Perf — Handoff

**Status as of 2026-05-19** · master at `d005d40` · scope: pre-JIT perf wins for the tree-walking executor, before any backend work.

This handoff carries enough context for a fresh chat session to resume Tier-0
work without needing the prior session's context window. Read top to bottom.

---

## TL;DR — what's done, what's next

| Item | State | Commit | Notes |
|---|---|---|---|
| **T0-a** Microbench harness | ✅ shipped | `0b4d007` | `cargo run -p interp-micro --release` |
| **T0-c** Inline cache for `dispatch_call` | ✅ shipped | `0b4d007` | Correctness-preserving; cache hits below bench noise floor |
| **T0-b Stage 1** Data foundation | ✅ shipped | `d005d40` | `VarLocal` variant + `MirFunction.local_count`; purely additive |
| **T0-b Stage 2** HIR→MIR lowering | ⏸ **NEXT** | — | Walk fn bodies, assign slots, emit `VarLocal` |
| **T0-b Stage 3** Executor `frame[slot]` fast-path | ⏸ after Stage 2 | — | The actual perf payoff lives here |
| **T0-b Stage 4** Closures + captures | ⏸ later | — | Stays on name fallback until 4 |
| **T0-b Stage 5** Remove name fallback | ⏸ later | — | Once everything is slot-indexed |
| **T0-d** `eval_binary` fast-paths | optional | — | ~1 hr, marginal but cheap |
| **T0-e** `is_known_builtin` static set | optional | — | ~30 min |

Regression baseline:
- `cargo test --workspace --lib` → **2,505 / 2,505 pass**
- `cargo test --test test_builtin_parity` → **10 / 10 pass**

---

## Critical architectural findings from the prior session

**Read these before touching code. The textbook plan does not match the
codebase.**

### Finding 1 — `cjc-mir-exec` is a tree-walker, not a register machine

Despite the README's "register-machine executor" framing, `cjc-mir-exec`
recursively walks `MirExpr` nodes via `eval_expr()` at
[`crates/cjc-mir-exec/src/lib.rs:517`](../crates/cjc-mir-exec/src/lib.rs).
There is no opcode dispatch loop, no register file, no bytecode. The
"register-machine" framing is aspirational.

**Implication:** the textbook Tier-0 items (computed-goto dispatch,
typed register slots, opcode superinstructions) **do not apply**. They
would apply to a future real bytecode VM. The optimizations that *do*
apply for this codebase are:
- Variable lookup acceleration (T0-b — biggest leverage)
- Dispatch-call inline caching (T0-c — shipped)
- Per-shape arithmetic fast-paths (T0-d)

### Finding 2 — AST eval is often *faster* than MIR-exec

The bench shows `cjc-eval` beating `cjc-mir-exec` on most workloads.
Both are tree-walkers; MIR-exec adds HIR→MIR lowering overhead per call
of `run_program_with_executor`. The bench harness already lowers once
outside the timing loop (see `bench/interp_micro/main.rs`) to factor
this out.

**Implication:** Tier-0 wins should first bring MIR-exec at least to
*parity* with eval, then beyond. The "MIR is faster" assumption is
inverted here.

### Finding 3 — Windows bench noise is severe (~2× run-to-run variance)

Identical workloads vary by ~2× across consecutive runs on Windows. The
bench harness works, but only optimizations producing ≥30% wins will
have a cleanly measurable signal. Stage 3's expected 3–5× on the
`lookup` workload should clear the noise floor. Smaller wins (T0-c,
T0-d) will not show up cleanly without re-engineering the harness.

---

## How to verify the baseline

In a fresh session:

```bash
cd C:\Users\adame\CJC
git log --oneline -5             # should show d005d40 at the top
cargo test --workspace --lib     # 2,505 / 2,505
cargo test --test test_builtin_parity   # 10 / 10
cargo run -p interp-micro --release     # bench, ~1 min
```

Bench output format (JSONL to stdout + scorecard to stderr):

```
arith     mir_cold:   25.67 ms   mir_warm:   27.49 ms   eval:   29.90 ms
lookup    mir_cold:   83.38 ms   mir_warm:  117.03 ms   eval:  100.69 ms
call      mir_cold:   51.96 ms   mir_warm:   65.28 ms   eval:   51.90 ms
mixed     mir_cold:   55.14 ms   mir_warm:   41.97 ms   eval:   35.54 ms
```

`lookup` is consistently the heaviest workload — that's the one Stage 3
targets.

---

## Stage 1 (shipped) — the foundation you'll build on

### What was added

**[`crates/cjc-mir/src/lib.rs`](../crates/cjc-mir/src/lib.rs):**

```rust
pub enum MirExprKind {
    // ... existing variants ...
    Var(String),                              // unresolved (fallback)
    VarLocal { name: String, slot: u32 },    // resolved (fast path)
    // ... rest ...
}

pub struct MirFunction {
    // ... existing fields ...
    pub local_count: u32,    // 0 = no slot resolution; use name fallback
}
```

**Match-arm coverage:** `VarLocal` is handled by every exhaustive match
across `cjc-mir/src/{escape,monomorph,nogc_verify,optimize}.rs` and
`cjc-mir-exec/src/lib.rs:546`. Each arm mirrors the existing `Var`
arm's semantics (treats name field identically, ignores `slot`).

**Constructor coverage:** every `MirFunction { ... }` constructor across
9 files now has `local_count: 0`.

### What still does NOT happen yet

- Nothing emits `VarLocal` — the lowering at
  [`crates/cjc-mir/src/lib.rs:772`](../crates/cjc-mir/src/lib.rs) still
  does `HirExprKind::Var(name) => MirExprKind::Var(name.clone())`.
- `local_count` is always 0 in every `MirFunction`.
- Executor's `VarLocal` arm falls back to name lookup (treats it
  exactly like `Var`).

This means Stage 1 is **purely additive, zero behavior change**. All
2,505 tests pass because nothing actually invokes the new code path.

---

## STAGE 2 — HIR → MIR slot resolution (start here)

This is the next session's primary task. Roughly 200–400 lines of
changes, mostly in `crates/cjc-mir/src/lib.rs`. No new public APIs.

### Goal

In `HirToMir`, walk each function body and assign slot indices to
parameters + `let` bindings. Emit `MirExprKind::VarLocal { name, slot }`
for variable references that resolve to a function-local slot.
Emit `MirExprKind::Var(name)` for everything else (closures, top-level,
unresolved).

### Files to modify

| File | What changes |
|---|---|
| `crates/cjc-mir/src/lib.rs` | Add slot tracking state to `HirToMir`; populate during fn lowering |
| (nothing else in Stage 2) | Stage 3 will touch the executor; tests come for free |

### Algorithm sketch

For each `HirFn` being lowered:

```
1. enter_function():
   - clear scope stack
   - push initial scope
   - slot_counter = 0
   - for each param: assign slot to slot_counter, increment, insert into top scope
   - record params' slot count

2. walk body:
   - on Block { stmts, expr }:
     - push_scope()
     - walk stmts (each may add Let bindings)
     - walk expr
     - pop_scope()
   - on Let { name, init, ... }:
     - lower init (may emit Var or VarLocal)
     - assign slot_counter to name in top scope
     - emit MirStmt::Let { name, init, ... } (existing variant -- slot is encoded
       implicitly via the scope-stack lookup below)
     - increment slot_counter
   - on Var(name):
     - search scope stack from top down
     - if found: emit MirExprKind::VarLocal { name: name.clone(), slot }
     - else: emit MirExprKind::Var(name.clone())  (existing path; closures,
       top-level, captures)

3. exit_function():
   - set MirFunction.local_count = slot_counter
   - reset state for next function
```

### Closure handling in Stage 2

**Skip slot resolution entirely for lambda-lifted closures.** Closures
in CJC-Lang are lambda-lifted in HIR→MIR at
[`crates/cjc-mir/src/lib.rs:826-886`](../crates/cjc-mir/src/lib.rs) —
they become regular `MirFunction`s with captures prepended as extra
params. The simplest Stage 2 path is:

- Detect closure-lifted functions by name pattern (`__closure_N`) OR by
  presence of captures-as-params
- For those, leave `local_count = 0` and emit `Var(name)` for every
  variable reference (name fallback)
- This means closures stay slow until Stage 4

### Edge cases to handle

1. **Shadowing in nested blocks.** A `let x` in an inner block shadows
   the outer `x`. The inner scope's slot is used while in that scope;
   the outer slot resumes when the inner scope pops.
2. **Slot counter is monotonic per function** — slots from popped
   scopes are NOT reclaimed. This wastes some frame slots but keeps the
   implementation simple. A function with `if { let x = 1 } else { let y = 2 }`
   uses 2 slots, not 1.
3. **`MirStmt::Let` does NOT carry slot info in Stage 2.** The lowering
   only needs to track slots for the *emission of `Var` references*.
   Let is left as-is. Stage 3 will handle the Let path differently
   (the executor looks up the name in the current frame's slot map at
   Let time -- one-time cost per binding).
4. **`Assign` expressions** target a Var (or a Field/Index of a Var). The
   target's Var should also become VarLocal when the variable is a
   local. Look at how `Assign` is lowered; you may need to recursively
   slot-resolve the target's Var.
5. **Match patterns that bind names** — patterns like `Some(x)` bind
   `x` as a local. The pattern compiler emits implicit Let-like bindings.
   For Stage 2, handle this if it's straightforward; otherwise defer to
   Stage 4 (keep these on the name fallback path).

### Parity testing

After Stage 2, all 2,505 workspace tests must still pass. The
`test_builtin_parity` (10 tests) is the load-bearing one — eval and
mir-exec must produce byte-identical output. Since Stage 2 only changes
WHICH variant is emitted (Var → VarLocal) but not its semantics,
parity is preserved by construction *as long as* the executor's
`VarLocal` arm continues to behave identically to `Var`.

The executor change is Stage 3; Stage 2 should pass tests on its own
because the executor still does name-based lookup for both variants.

---

## STAGE 3 — Executor frame fast-path

After Stage 2 ships and tests pass, light up the actual perf payoff in
the executor.

### Files to modify

| File | What changes |
|---|---|
| `crates/cjc-mir-exec/src/lib.rs` | Add `frame: Vec<Value>` per call frame; route `VarLocal` reads to it; route `Let` writes to it; size the frame from `MirFunction.local_count` |

### Where to hook

The executor's `MirExecutor` struct is at
[`crates/cjc-mir-exec/src/lib.rs:209`](../crates/cjc-mir-exec/src/lib.rs).
Currently has `scopes: Vec<BTreeMap<String, Value>>` for the name-based
chain. Add alongside (don't replace yet):

```rust
pub struct MirExecutor {
    // ... existing fields ...
    scopes: Vec<BTreeMap<String, Value>>,    // existing fallback path
    frame: Vec<Value>,                        // NEW: flat slot array
    frame_stack: Vec<usize>,                  // NEW: saved frame.len() per call
}
```

### Hook points

**On function entry (`call_function`):**

```rust
if mir_fn.local_count > 0 {
    self.frame_stack.push(self.frame.len());
    self.frame.resize(self.frame.len() + mir_fn.local_count as usize, Value::Void);
    // bind params to frame[base..base+n_params]
    let base = self.frame_stack.last().copied().unwrap();
    for (i, val) in args.iter().enumerate() {
        self.frame[base + i] = val.clone();
    }
}
```

**On function return:**

```rust
if let Some(base) = self.frame_stack.pop() {
    self.frame.truncate(base);
}
```

**On `Let` statement:** lookup the binding's slot from the lowering's
scope map (you'll need to expose it from MIR, OR re-derive it in the
executor by maintaining a parallel scope-stack during exec).

Cleanest approach: **add `slot: Option<u32>` to `MirStmt::Let`** during
Stage 3 so the executor doesn't need to look it up. This is the one
place where Stage 1's "additive variant only" approach has to give a
bit — Let needs the slot too.

**On `MirExprKind::VarLocal { slot, .. }`:**

```rust
MirExprKind::VarLocal { slot, .. } => {
    let base = self.frame_stack.last().copied().unwrap_or(0);
    Ok(self.frame[base + *slot as usize].clone())
}
```

**On `MirExprKind::Var(name)`:** unchanged (the existing scope-chain
fallback for closures, top-level, captures).

### Mutables / assignment

Assign through `VarLocal` must write to `frame[base + slot]`. Walk the
existing `exec_assign` code path to find where Var-targeted assignment
lives and add a parallel VarLocal-targeted path.

### Determinism check

The `frame_stack` is push/pop-balanced and deterministic given the same
program. No HashMap iteration order issues. Tests should pass — if they
don't, suspect:
- Slot collision in nested scopes (Stage 2 bug)
- Missing frame setup on function entry
- Asymmetric Var/VarLocal handling (one path mutates, the other doesn't)

### Measuring the win

After Stage 3:

```bash
cargo run -p interp-micro --release
```

Look at the `lookup` workload — that's where the win lives. Expected:
3–5× improvement on `mir_warm` vs the pre-Stage-3 baseline. The
`arith`, `call`, `mixed` workloads should also improve modestly.

---

## Stage 4 & 5 — for later sessions, not next

Stage 4: extend slot resolution to closures. Captures live in the
closure's `env: Vec<Value>` and are currently accessed by name via the
fallback. Adding slot-indexed captures requires touching the closure
capture analysis at HIR lowering time + the env-prepending logic at
[`crates/cjc-mir-exec/src/lib.rs:1258`](../crates/cjc-mir-exec/src/lib.rs).

Stage 5: once everything is slot-indexed, the `Var(String)` variant and
the `scopes: Vec<BTreeMap<String, Value>>` field can both be deleted.
Pure cleanup — only do this when you're certain nothing emits `Var`
anymore.

---

## Project rules to remember

These come from `CLAUDE.md` and the project memory; they apply to every
change:

1. **Determinism is sacred.** Same seed = bit-identical output. The
   parity tests enforce this between eval and mir-exec. Never weaken.
2. **`BTreeMap` everywhere, never `HashMap`.** Random iteration order
   breaks determinism. (Exception: caches used only via `get()`, never
   iterated.)
3. **Both executors must agree.** Every change to mir-exec semantics
   must be mirrored or shown to be perf-only. Stage 2 (slot emission)
   doesn't change semantics; Stage 3 (frame access) also doesn't change
   semantics. Both should preserve `test_builtin_parity`.
4. **No FMA / no FP reassociation.** Doesn't apply to Tier-0 (no FP
   touched), but worth remembering for later perf work.
5. **Commit each stage independently** with the regression gate
   (`cargo test --workspace --lib`) passing.
6. **Don't push to origin** without an explicit ask.

---

## Quick-reference file map

| Path | Purpose |
|---|---|
| `crates/cjc-mir/src/lib.rs` | `MirExprKind`, `MirFunction`, `HirToMir` lowering — Stage 2 home |
| `crates/cjc-mir-exec/src/lib.rs` | `MirExecutor`, `eval_expr`, `dispatch_call` — Stage 3 home |
| `crates/cjc-eval/src/lib.rs` | AST tree-walker — parity reference; do NOT modify in T0-b |
| `bench/interp_micro/main.rs` | Microbench harness |
| `tests/test_builtin_parity.rs` | The load-bearing parity gate |

## Quick-reference command map

```bash
# fast feedback during dev
cargo check -p cjc-mir -p cjc-mir-exec

# regression gate (run before every commit)
cargo test --workspace --lib

# parity gate (load-bearing)
cargo test --test test_builtin_parity

# perf measurement
cargo run -p interp-micro --release

# build the cjcl binary for smoke tests
cargo build --bin cjcl
target/debug/cjcl.exe run <some.cjcl>
```

---

## What "done" looks like for T0-b

- `MirFunction.local_count > 0` for every non-closure function
- `MirExprKind::VarLocal` emitted for every non-closure variable reference
- `MirExecutor` reads `frame[slot]` for `VarLocal` and writes `frame[slot]`
  for `Let` (with known slot)
- `lookup` workload in the bench: 3-5× faster (clear of Windows noise)
- All parity tests still pass
- (Stage 5) `Var(String)` deleted from `MirExprKind`; `scopes` field
  deleted from `MirExecutor`

At that point Tier-0 is meaningfully complete and the natural next
move is the JIT / Cranelift backend / copy-and-patch experimental
work.
