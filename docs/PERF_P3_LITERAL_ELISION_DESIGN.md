# P3 — Non-escaping literal elision: verified status + dedicated-session design

**Date:** 2026-06-13 · branch `claude/happy-jennings-d673b9`
**Source:** the "do P3" pass of the speed/memory arc. Roadmap item:
`docs/cana/PERFORMANCE_ROADMAP.md` §P3. This doc REPLACES the roadmap's
one-paragraph sketch with a code-verified status and an executable spec.
**Gate for any implementation:** `docs/cana/DETERMINISM_CONTRACT.md`.

## 0. TL;DR (THE RULE, applied)

Verifying P3 against the code changed its scope. The easy half is
**already done**; the hard half is **genuinely a dedicated-session
refactor**, not a wire-up. This session did NOT force it, because every
viable implementation either changes the `Value` representation
(pervasive, determinism-sensitive) or adds a MIR pass (ripples into the
whole CANA corpus). Both violate "don't rush pervasive changes." This doc
makes the future session turnkey.

## 1. What was verified in the code

| Claim | Verified reality | Evidence |
|---|---|---|
| Non-escaping literals are detected | YES — escape analysis already classifies `let cell = [i,i+1]` (non-escaping) as `AllocHint::Arena` | `cjc-mir/src/escape.rs` (Container init + no escape path → Arena) |
| **Dead** non-escaping literals are elided | YES — DCE removes them when the plan includes `dce` | `cjc-mir/src/optimize.rs:763`; `is_pure_expr` returns true for `ArrayLit`/`TupleLit` with pure elements (`:868`). This is the Phase-D selector win on `mem_grad_*`. |
| The `Arena` hint elides an allocation today | **NO — it is a diagnostic no-op** | `cjc-mir-exec/src/lib.rs:864`: `if let Some(AllocHint::Arena) = alloc_hint { self.arena_alloc_count += 1 }`. The value is still `Value::Array(Rc::new(vec))` from `eval_expr`. |
| The existing arena can back a `Value` array cheaply | **NO** | `cjc-runtime/src/frame_arena.rs::ArenaStore::alloc` stores `Rc::new(RefCell::new(Box::new(value)))` — it provides bulk-free *discipline*, not allocation *elision* for `Value::Array`'s `Rc<Vec>`. |

**Conclusion:** the only remaining P3 value is the **live-but-ephemeral**
literal — read once (so DCE can't drop it) then dead before function exit
(so escape analysis says `Arena`). Example DCE cannot help:
```cjcl
let t = (lo, hi);          // non-escaping, but...
let mid = (t.0 + t.1) / 2; // ...t IS read, so not dead
// t never used again
```
Today `(lo, hi)` allocates a `Value::Tuple(Rc<Vec>)` that dies one
statement later.

## 2. Why it is not a contained change (the blast radius)

Three implementation routes, each dedicated-session:

### Route A — arena-backed `Value` array variant
Add `Value::ArrayArena { frame: u32, idx: u32 }` (or similar) holding an
index into a real bump arena of `Vec<Value>`, used when `alloc_hint ==
Arena`. Frees with the frame (escape analysis guarantees it does not
outlive the frame).
- **Blast radius:** every `match` on `Value::Array` across
  `cjc-eval`/`cjc-mir-exec`/`cjc-runtime`/`cjc-dispatch` must also handle
  the arena variant (read path), OR every read must go through an
  accessor that normalizes. Plus the unsafe lifetime management (the
  arena value must not escape — a miscompiled escape proof is UB).
- **Determinism:** safe IF the escape proof is conservative; the risk is
  the proof, not the arithmetic.

### Route B — inline small-array storage (`SmallVec`-style)
Store ≤N elements inline in `Value` (no heap) for small arrays/tuples.
- **Blast radius:** changes `Value`'s SIZE (the thing Phase I worked to
  keep at 72 B) and its `Array`/`Tuple` access pattern everywhere.
  Pervasive; size-regression risk.

### Route C — SROA MIR pass (scalar replacement of aggregates)
A new optimizer pass: when a non-escaping tuple/array is consumed only by
constant-index/field reads, replace it with its scalar elements and drop
the literal. No `Value` change, no unsafe — the cleanest *semantically*.
- **Blast radius:** **a new MIR pass ripples into CANA.** The pass plans,
  the trained energy/thermal/memory heads, and the committed corpus
  (`bench_results/cana_ablation/profiles.cpdb`, 4,740 rows) are all keyed
  on the pass set / FeatureHash. Adding a pass = regen corpus → retrain
  all heads → re-shadow → re-measure (handoff §4). That is a multi-hour
  arc, not a contained edit.

## 3. Recommended path (for the dedicated session)

**Route C (SROA), staged behind the CANA regen, is the cleanest.** It is
determinism-safe by construction (it is a semantic rewrite, no
representation/unsafe), and it generalizes (helps any "build a tuple to
return two values, immediately destructure" pattern — extremely common).
Plan:
1. Implement `sroa` as an `optimize.rs` pass: for each non-escaping
   `let t = TupleLit/ArrayLit` whose every use is a constant
   `Index`/`Field` into `t`, substitute the element exprs and delete the
   binding. Conservative: any non-constant index, any whole-`t` use, or
   any escape → leave as-is.
2. Heavy `proptest` (random nestings) + `bolero` (structural) proving
   AST-eval ≡ MIR-exec byte-identical before/after the pass (invariant 7).
3. Add `sroa` to the pass registry → **CANA regen + retrain + re-shadow**
   (the ripple — budget for it).
4. Measure: a NEW `interp_micro` workload with the live-ephemeral pattern
   (the current `mem_grad`/`clos` workloads do not exercise it — the
   literals there are dead, already DCE'd), via the deterministic alloc
   counter; plus `cana_diagnostics` wall-clock.

If Route C's CANA ripple is undesirable, **Route A** is the
executor-local alternative (no CANA ripple) but carries the unsafe
escape-lifetime risk and the multi-crate `Value::Array` match surface.

## 4. What this session DID ship toward P3

- **Verified** the status above (so the dedicated session starts from
  fact, not the roadmap's estimate).
- The **deterministic alloc counter** in `bench/interp_micro` (added in
  the Stage-5b arc) is the measurement instrument P3 needs — a
  live-ephemeral workload dropped into it will quantify the win before and
  after, exactly as the contract requires.
- The **arena pool** (Stage 5b) already proves the "reuse ephemeral
  backing instead of fresh-allocate" pattern is determinism-safe in this
  executor — a useful precedent for Route A's arena.

## 5. Verdict

P3's dead-literal half is already delivered (DCE). Its live-ephemeral
half is a real win but a dedicated-session refactor (Route C SROA
recommended, with its CANA regen budgeted; Route A as the no-CANA-ripple
alternative). This session scoped it precisely and honestly rather than
forcing a `Value`-representation change late — the same call Phase I made,
now backed by fresh code verification.
