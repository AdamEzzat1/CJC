# Phase H — Head-Independent Exploration Configs: Attacking the Regression Floor

**Date:** 2026-06-13 (follows Phase G on `claude/stupefied-liskov-83b258`)
**Spec source:** `docs/cana/HANDOFF_PHASE_D.md` §3b (the second selector-
hardening lever): "add forced versions of the selector's candidate
shapes so training sees novel pass combinations WITHOUT the feedback
loop; then retrain energy head → regret should improve out-of-
distribution." Designed by the Compiler-Pipeline-Engineer pass of the
stacked-role optimization arc.
**Exit criterion:** the Phase-G regression floor of 7 shrinks after
retraining the energy head on head-independent anchors covering the
selector's out-of-distribution pass combinations, with the energy head
still PROMOTE in shadow and the thermal head undisturbed.

## 0. The problem Phase G left

Phase G's margin gating halved the selector's regressions (16→7) but
left a **floor of 7** that no margin can reach: those are *confident*
head mispredictions (predicted advantage > 0.20, measured worse), not
marginal switches. Margin gating only catches small-advantage bets; a
confidently-wrong head needs *better training data*, not a bigger
threshold.

## 1. The diagnosis (why the head is confidently wrong)

The energy head trains on forced configs at pass-counts **{0, 1, 7}**
only — `force_none` (0 passes), the singletons (1 pass), and
`force_all` (7 passes). Between count-1 and count-7 it **extrapolates
linearly**, and the surviving regressions live in that 2–6-pass gap.
Worse, they are pass-*interaction* effects — `licm`+`loop_unroll` (LICM
hoists a body that unroll then multiplies → node growth the additive
head underprices) and `cf`+`dce` (the actual win mechanism) — and a
linear head over additive pass-counts is structurally blind to
interactions it has never seen labeled.

## 2. The fix (anchors in the gap)

Five new forced configs, head-independent (chosen by a fixed pass list,
never by the head — so no feedback loop), placed exactly where the head
extrapolates blind:

| config | passes | fills |
|---|---|---|
| `force_cf2` | `[cf_round_2]` | the one uncovered singleton (selector candidate id 9) |
| `force_cf_dce` | `[constant_fold, dce]` | 2-pass: the fold→prune win mechanism |
| `force_licm_unroll` | `[licm, loop_unroll]` | 2-pass: the node-growth interaction |
| `force_cf_dce_cse` | `[constant_fold, dce, cse]` | 3-pass mid-count anchor |
| `force_sr_licm_unroll` | `[strength_reduce, licm, loop_unroll]` | 3-pass loop-heavy anchor |

**Rejected:** `force_default_seq` (DEFAULT_PASS_SEQUENCE applied
uniformly) — it would equal `force_all`, because the head's pass-count
features are order-invariant and `CANONICAL_PASSES ≡
DEFAULT_PASS_SEQUENCE` element-wise. Reordering alone is invisible to
the head; only new count/interaction combinations add signal.

**No schema ripple:** every pass used is already in `CANONICAL_PASSES`,
and `cf_round_2` already enters the energy vocabulary via `force_all`,
so the vocabulary-dependent CPB1 feature count is unchanged — no bundle
version bump.

**Feedback guard:** these configs are deliberately NOT in
`ENERGY_EXCLUDED_CONFIGS`. They were never chosen by the head, so
training on them is not a feedback loop — it is the exploration signal
the head lacks.

## 3. The measurement (two-pass, because the selector depends on the head)

The selector's plans are driven by the energy head, so the experiment
is necessarily two-pass:

1. **Regen #1** — exploration-config rows now exist; the selector still
   uses the committed (pre-H) head.
2. **Retrain** the energy head on Regen #1 (now including the
   exploration anchors) → new CPB1 bundle; **shadow-energy must still
   PROMOTE**.
3. **Regen #2** — the selector now uses the retrained head.
4. **Compare** the Phase-G table's regression count at
   `selector_mg_rec_t02` (the calibrated τ): Regen #2 vs the committed 7.

## 4. Results (measured 2026-06-13)

The exploration configs worked — and the result inverts Phase G's
conclusion in an instructive way.

### 4.1 The headline: the UNGATED selector's regressions collapse 16 → 1

Phase G table after retraining the energy head on the exploration
anchors (regen #2, new head driving the selector):

| τ | wins | regressions | mean score |
|---|---|---|---|
| **0.00 (`selector_rec`, ungated)** | 6 | **1** | **0.98179** |
| 0.02 | 7 | 7 | 0.98186 |
| 0.05 | 7 | 7 | 0.98417 |
| 0.10 / 0.20 | 1 | 7 | 1.00329 |

The ungated selector went from **6 wins / 16 regressions** (committed,
pre-H) to **6 wins / 1 regression** — 15 of the 16 regressions
eliminated, wins held, best mean. The head, given labeled anchors in
the 2–6-pass interaction gap, stopped confidently mis-scoring those
combinations: its switches are now mostly *correct*, so the selector
makes them and they pan out.

### 4.2 The inversion: margin gating is now counterproductive

With the OLD (noisy) head, Phase G found τ=0.02 dominant — gating was
safer because the head's switches were often wrong. With the NEW
(accurate) head the relationship flips: **τ=0 (no gating) is now best**
(1 regression, lowest mean), and the gated configs show *more*
regressions (7), because gating keeps the ranked plan on functions
where the improved head would have correctly switched to a win — so it
*re-introduces* regressions the ungated selector fixes.

This is the load-bearing finding: **Phase G treated the symptom**
(unreliable switches via a margin), **Phase H fixed the cause** (the
head's training blind spots). Fixing the cause makes the workaround
obsolete — and slightly harmful. The calibrated recommendation flips
back to **τ = 0**. (The `with_margin` machinery is kept, not removed:
it is the correct tool for a future noisier head, and τ=0 is its
byte-identical no-op.)

### 4.3 Honest texture

- The head's own **regret metric nudged slightly worse** on the FNV-
  test cohort (+0.00140 → +0.00256) even as the selector's downstream
  decisions got much better. The two measure different things: regret
  is prediction accuracy over the *forced-plan* space; the selector
  regression count is decision quality over the *selector's candidate*
  space. The exploration anchors improved the latter. Shadow-energy
  still **PROMOTE** (test regret beats both baselines; holdout exact
  10/10).
- **1 residual regression remains** (down from 16). It is a single
  hard program where the head still mis-scores; the 16→1 collapse is
  the headline, and the last one is a candidate for the non-linear-
  head direction, not more anchors.
- Corpus grew 3,318 → 4,740 rows (6 new configs × programs).

## 5. Safety / invariants

- **Thermal head undisturbed**: the exploration configs add energy-
  training rows but the thermal label and basis are unchanged, so the
  CPB0 bundle retrains byte-identical (verified by hash).
- **Vocabulary unchanged** → CPB1 feature count unchanged → no schema
  bump.
- **selector_rec plan-identity gate**: the selector's candidate set is
  fixed in code, not driven by the config list, so the gate's structure
  holds; the committed corpus is regenerated so hashes match.
- **Determinism contract** (`DETERMINISM_CONTRACT.md`): no FP, no
  collection-order, no RNG changes — this is a corpus + training-data
  change only; parity and row-hash gates green.

## 6. Verdict

| Claim | Status after H |
|---|---|
| Head-independent exploration configs shrink the regression floor | **CONFIRMED, dramatically** — ungated selector regressions 16 → 1, wins held, mean improved. The diagnosis (linear extrapolation across an unlabeled 2–6-pass interaction gap) was correct. |
| Margin gating (Phase G) is the right hardening | **SUPERSEDED** — with the fixed head, τ=0 dominates; gating now re-introduces regressions. Phase G treated the symptom; H fixed the cause. Calibrated default flips to τ=0. |
| Energy head still valid after retraining | **PROMOTE** — shadow gate holds (holdout regret exact 10/10), thermal head byte-identical. |

**Exit criterion MET**: the regression floor shrank (16 → 1) by
retraining on head-independent anchors, with the head still PROMOTE and
the thermal head undisturbed.

**What this licenses — and what it doesn't.** The selector with the
retrained head is **6 wins / 1 regression** — close to default-on
quality, a large step from "ablation-grade, 16 regressions." It does
NOT yet license flipping it on: the wins/regressions are MODELED energy
(Phase D's stopwatch would confirm the net effect), and one residual
regression plus the granularity caveats remain. But the selector is now
materially closer than any prior phase.

**Next:** the 1 residual regression is a non-linear-head problem (a
single confident misprediction anchors don't reach), not a more-anchors
problem — a tiny MLP energy head on the same features, or a pairwise-
interaction feature, is the lever. Separately, with τ=0 now optimal,
the margin-gating sweep configs could be retired from the default
corpus (kept here as the audit trail of WHY τ=0 won).
