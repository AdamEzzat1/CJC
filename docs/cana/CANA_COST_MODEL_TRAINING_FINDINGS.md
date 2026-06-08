# CANA Cost Model Training — Findings (v1 → v2 comparison)

**Date:** 2026-06-07 (TIER B #4 from handoff)
**Status:** v2 shipped — all four improvements from v1 findings implemented.
**Output:** `LinearCostModel::trained()` with v2 coefficients + 60-program corpus.

---

## 1. What v2 changed vs v1

The v1 findings doc identified four improvements that "would move the
needle." v2 implements all four:

| Improvement | Implementation |
|---|---|
| **1. Bigger workloads** (10-100× n) | Inner-loop iteration counts scaled. Most programs now run for 100μs-10ms instead of 1-100μs. |
| **2. 60-program corpus** (was 18) | 42 new programs densifying the (expr_count, loop_depth, branch_count) corners. |
| **3. MIR-instruction-count delta** for size-shrinking passes | New `count_mir_nodes` walker. Used as label for CF + DCE — deterministic, zero variance. |
| **4. Multi-seed averaging** (N_ITERS 5 → 21) | `sqrt(21/5) ≈ 2.05×` reduction in wall-clock variance. |

A fifth thing happened that wasn't planned: discovering that **CSE and SR
don't shrink MIR structure** in this codebase. CSE replaces variable uses
(the dead lets it creates get cleaned up by a later DCE pass — but
DEFAULT_PASS_SEQUENCE runs DCE *before* CSE, so the cleanup never
happens). SR rewrites operations in place (`x * 8` → `x << 3`) without
changing node shape. So the MIR-count proxy reports zero benefit for
these passes — correctly, structurally, but useless as a training
signal. They were moved to wall-clock measurement.

**Final signal assignment in v2:**

```
mir_count (deterministic):  constant_fold, dce
wall_clock (noisy):         strength_reduce, cse, licm
```

---

## 2. Headline results

### Per-pass RMSE: v1 → v2

| Pass            | v1 RMSE | v2 RMSE | Reduction | v2 Signal      |
|-----------------|---------|---------|-----------|----------------|
| constant_fold   | 0.182   | **0.055** | **3.3×**  | mir_count      |
| dce             | 0.166   | **0.097** | **1.7×**  | mir_count      |
| strength_reduce | 0.130   | 0.087   | 1.5×      | wall_clock     |
| licm            | 0.169   | 0.135   | 1.3×      | wall_clock     |
| cse             | 0.137   | 0.140   | ~1.0×     | wall_clock     |

### Per-pass confidence: v1 → v2

| Pass            | v1 conf | v2 conf | Gain  |
|-----------------|---------|---------|-------|
| constant_fold   | 0.10    | **0.65** | 6.5×  |
| dce             | 0.16    | **0.47** | 2.9×  |
| strength_reduce | 0.32    | 0.51    | 1.6×  |
| licm            | 0.14    | 0.29    | 2.1×  |
| cse             | 0.29    | 0.27    | ~flat |

### What the numbers mean

CF and DCE got the **deterministic signal treatment** — their RMSE
dropped 1.7-3.3×, their confidence rose 3-6×. These trained
coefficients are now in **the same quality band as the hand-tuned
defaults**. They're reproducible bit-for-bit on re-runs.

SR, CSE, LICM stayed on wall-clock — their RMSE improved 1.0-1.5×
purely from bigger workloads + 4× more samples per program. The
remaining noise is microsecond-scale scheduler variance that no amount
of corpus tuning will eliminate. **These coefficients vary 10-30%
between re-runs.**

CSE was the only pass that didn't budge much. Its measured benefit is
genuinely small in this codebase (mean 0.07 across the corpus) — the
optimizer's CSE pass mostly does its work for cases where the savings
are too small to measure cleanly at microsecond scale, regardless of
N_ITERS.

---

## 3. Determinism contract by signal

A reader trying to reproduce these numbers from the binary needs to know
what's reproducible:

| Pass | Reproducible across runs? |
|---|---|
| `constant_fold` | **YES** — mir_count is a structural property of the IR, identical across runs/OS/CPU |
| `dce` | **YES** — same reason |
| `strength_reduce` | NO — wall-clock dependent; expect ±10-30% drift per run |
| `cse` | NO — same |
| `licm` | NO — same |

The trained CF and DCE coefficients you see in `linear_cost_model.rs`
will regenerate identically every time. The trained SR/CSE/LICM
coefficients will drift; the file gets stamped with the most-recent
run's values when someone runs the training binary.

This is a **load-bearing distinction** — it tells maintainers which
fields are safe to copy verbatim into design docs and which are
representative samples that callers should treat as noisy.

---

## 4. The wall-clock plateau

We've reached the limit of what microsecond-scale wall-clock can give
us. Future-proof options if cleaner SR/CSE/LICM coefficients are
needed:

**Option A — instruction-count proxy (custom per pass)**
- For SR: instrument MIR walker to count `Mul` vs `Shl` ops. Direct measure.
- For CSE: count `let X = ...` bindings whose `init` matches a previously-
  seen pure expression. Direct measure.
- For LICM: count *hoisted nodes* via a marker pass that tags movements.
  Requires optimizer instrumentation.

Estimated effort: 2-3 days per pass; estimated RMSE improvement: ~4-10× on each.

**Option B — millisecond-scale workloads + perf counter instead of wall-clock**
- Use `windows::Performance::QueryPerformanceCounter` for sub-microsecond
  resolution; scale workloads to 10-100ms each.
- Total benchmark wall-clock would be 30-60 min (vs 6-7 min today).
- Estimated RMSE improvement: ~3× on wall-clock passes.

**Option C — accept the plateau, document it, move on**
- Ship v2 coefficients with the confidence-floor safety property.
- Document that further refinement is gated on someone caring enough
  to invest in A or B.

This commit takes Option C. The v2 trained model is good enough for the
deterministic-signal passes (CF, DCE) and honestly noisy on the others
(documented confidence < 0.51, automatic fallback when confidence is low).

---

## 5. How to regenerate

```bash
cargo run --release -p cana-train-cost-model
```

Wall-clock: ~6-7 minutes on a modern Windows release-mode build.
- Phase 1 (collect training data): ~5 minutes (60 programs × 5 passes × measurements)
- Phase 2 (per-pass OLS fit): <1 second
- Output: Rust source ready to paste into `linear_cost_model.rs`

Confidence values are derived from training RMSE: lower RMSE → higher
confidence, capped at [0.1, 0.95].

---

## 6. What v2 measured benefits look like in the wild

### Per-program leaderboard (which pass "won" each program in v2)

The leaderboard is now more sensible than v1:

| Program category | Expected winner | v2 actual winners |
|---|---|---|
| arith_* (CF programs) | constant_fold | constant_fold, sometimes SR (noise) |
| loop_* (LICM programs) | licm | mostly licm or SR (wall-clock noise mixing them) |
| cse_* (CSE programs) | cse | mostly licm (CSE's wall-clock signal is weak) |
| sr_* (SR programs) | strength_reduce | SR for half, licm for the rest (noise) |
| dce_* (DCE programs) | dce | **dce wins decisively for all four** (mir_count clean) |

The clean wins for DCE confirm the deterministic signal works. The
ambiguous winners for SR and CSE confirm the wall-clock noise is real.

---

## 7. API surface — unchanged in v2

The v1 API still works identically; v2 only changed the coefficients:

```rust
let m_default = LinearCostModel::new();      // hand-tuned coefficients
let m_trained = LinearCostModel::trained();  // v2 fit coefficients

let r_default = default_ranker();   // uses hand-tuned
let r_trained = trained_ranker();   // uses trained
```

The seven tests in `linear_cost_model::tests` all still pass. New
coefficients are still in `[0.0, 0.5]` range, still deterministic for
a given coefficient set, still differ from hand-tuned on at least one
pass.

---

## 8. Where to read the code

| File | Purpose |
|---|---|
| `bench/cana_train_cost_model/programs.rs` | 60-program training corpus |
| `bench/cana_train_cost_model/main.rs` | Measurement (mir_count or wall_clock per pass) + OLS fit + code generation |
| `crates/cjc-cana/src/linear_cost_model.rs` | `trained_pass_coefficients` fn |
| `crates/cjc-cana/src/pass_ranker.rs` | `trained_ranker()` helper |

---

**Bottom line:** the four improvements landed. CF and DCE now have
clean, reproducible, audit-worthy trained coefficients. SR/CSE/LICM
have noisier-but-better trained coefficients than v1. The infrastructure
for cleaner signal on wall-clock passes (Option A above) is well-scoped
when someone has reason to invest in it.
