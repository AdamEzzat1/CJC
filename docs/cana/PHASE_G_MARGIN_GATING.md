# Phase G — Margin Gating: Trading the Selector's Regressions for Stability

**Date:** 2026-06-13 (follows Phase F1 on `claude/stupefied-liskov-83b258`)
**Spec source:** `docs/cana/HANDOFF_PHASE_D.md` §3 (selector hardening,
queued after Phase D's verdict): "keep the ranked plan unless predicted
gain exceeds a corpus-calibrated threshold — kills most of the 16
regressions at the cost of some wins; tune on test, verify on frozen
holdout."
**Exit criterion:** a calibrated margin τ that measurably reduces the
selector's regressions while preserving its wins, with the tradeoff
curve measured (not asserted) and the calibrated default verified on
the frozen-holdout cohort.

## 0. The problem Phase C left

The energy selector (Phase C) was the first config with mean measured
energy below baseline (0.98230), but it makes **bold bets**: 6 large
wins (allocation-churn DCE, down to 0.496) against **16 modest
regressions** (worst +14%). Those regressions are the out-of-
distribution effect — the head scores novel pass combinations it never
saw in training, and a few small predicted gains turn into small real
losses. The 16 regressions are exactly what disqualifies the selector
from default-on.

Phase D then proved the 6 wins are **real on the stopwatch** (2.7–3.5×,
frozen holdout included). So the wins are worth keeping; the
regressions are worth killing. That is a margin-gating problem.

## 1. The mechanism

The selector scores 10 candidate plans per function by predicted
`ln(score)` (lower = cheaper) and takes the argmin; candidate 0 is the
ranked plan. The **predicted advantage** of a switch is
`ranked_predicted − argmin_predicted` — a log-ratio, so τ is in
ln-score units (τ = 0.05 ≈ "only switch if the predicted energy ratio
is ≥ ~5% better"; e^0.05 ≈ 1.051).

The 6 wins have a *large* predicted advantage (the alloc-churn programs
predict ≈ ln(0.5) − ln(1.0) ≈ −0.70 of margin); the 16 regressions are
*small* advantages near zero. A margin τ between those two scales keeps
the wins and sheds the regressions.

`PassPlanSelector::with_margin(τ)` (`crates/cjc-cana/src/plan_selector.rs`):
a non-ranked argmin is taken ONLY when its advantage ≥ τ; otherwise the
ranked plan is kept (`gated_to_ranked`). **`τ = 0.0` reproduces the
Phase C selector byte-for-byte** (proven by
`margin_zero_is_byte_identical_to_ungated`), so `selector_rec` is
unchanged and the existing diagnostics/corpus gates still hold.

## 2. Calibration (measured, not asserted)

The ablation harness runs the selector at a τ sweep as separate
recorded configs — `selector_mg_rec_t{02,05,10,20}` (τ =
0.02/0.05/0.10/0.20) — exactly the pattern the thermal-cap `_c80`/`_c60`
variants used. All are added to `ENERGY_EXCLUDED_CONFIGS` (the feedback-
loop guard: the energy head must never train on plans it chose, gated
or not). The report prints wins / regressions / mean / switches per τ.

### 2.1 The tradeoff curve (measured, 158 programs)

| config (τ) | wins | regressions | mean score | switches |
|---|---|---|---|---|
| `selector_rec` (0.00) | 6 | **16** | 0.98230 | 158 |
| **`selector_mg_rec_t02` (0.02)** | **7** | **7** | **0.98186** | 141 |
| `selector_mg_rec_t05` (0.05) | 2 | 7 | 1.00010 | 136 |
| `selector_mg_rec_t10` (0.10) | 1 | 7 | 1.00329 | 136 |
| `selector_mg_rec_t20` (0.20) | 1 | 7 | 1.00329 | 136 |

Two findings the curve forced, neither hypothesized:

1. **The optimal τ is SMALL (0.02), not ~0.70.** The handoff reasoned
   the wins' margin was ≈ ln(0.5) ≈ 0.70, but that is the *whole-
   program* measured ratio; the selector gates on *per-function*
   predicted advantages, which are compressed by the program→function
   granularity mismatch the selector docs flagged. Above τ=0.05 real
   wins get gated away and the mean climbs ABOVE baseline (1.0003) —
   over-gating is worse than not gating.

2. **There is a regression FLOOR of 7.** Every τ ≥ 0.02 leaves exactly
   7 regressions. So τ=0.02 kills 9 of the 16, but the remaining 7
   survive arbitrarily large margins — they are switches the head
   predicts with HIGH confidence (advantage > 0.20) that still regress.
   Those are genuine head *mispredictions*, not marginal bets; margin
   gating structurally cannot reach them. Fixing them needs a better-
   trained head (the head-independent exploration configs, §4), not a
   bigger τ.

### 2.2 The calibrated default: τ = 0.02

τ = 0.02 **strictly dominates** the ungated selector on all three axes:
regressions more than halved (16 → 7, −56%), wins UP (6 → 7), and the
best mean (0.98186 vs 0.98230). The extra win is real, not noise: in a
multi-function program a single bad switch can tip the whole-program
score into regression while a genuine win elsewhere survives — gating
the bad switch flips the program back to a net win. The frozen-holdout
program `holdout_alloc_pulse` remains a win under τ=0.02 (so the gate
preserves the generalizing mechanism Phase D proved on silicon, not
just the trained shapes).

## 3. Safety / invariants

- **`selector_rec` (τ=0) byte-identical**: the 6 wins reproduce at
  0.49613–0.49855, parity 100%, row-hash stable, and the
  `gate_selector_rec_plans_unchanged_on_committed_corpus` gate passes.
- **Energy bundle byte-identical**: the gated configs are excluded from
  energy training, so retraining reproduces the committed CPB1 bundle —
  the feedback-loop guard, verified by hash.
- **Margin can only make the selector more conservative**: negative /
  non-finite τ clamps to 0; the gate never inverts. The
  never-worse-than-ranked invariant is preserved (the gate's only
  effect is to keep the ranked plan more often).

## 4. Status / what this licenses

**Exit criterion MET**: a calibrated margin (τ=0.02) measurably reduces
the selector's regressions (16 → 7) while preserving — slightly
improving — its wins (6 → 7) and mean (0.98230 → 0.98186), tradeoff
curve measured, frozen-holdout win preserved, safety guards
(selector_rec byte-identical, energy bundle byte-identical) intact.

**What it does NOT license: default-on.** τ=0.02 leaves a hard floor of
7 regressions that gating cannot touch — confident head mispredictions,
not marginal switches. The selector is *closer* to default-on
(regressions roughly halved, mean below baseline more robustly) but not
*at* it. The honest one-line summary: *a τ=0.02 margin halves the
selector's regressions for free — it strictly dominates the ungated
selector — but 7 residual regressions from head mispredictions keep it
ablation-grade until the head itself improves.*

**The next lever (head-independent exploration configs, handoff §3b):**
the 7-regression floor is an out-of-distribution problem — the head
mis-scores pass combinations it never trained on. Adding forced
versions of the selector's candidate shapes as SEPARATE training
configs (head-independent, so no feedback loop) would let the energy
head see those combinations and learn them, shrinking the floor. That
is a corpus + retraining arc of its own.

**Modeled vs silicon:** these are MODELED energy wins/regressions. τ=0.02
is calibrated against the modeled metric; the Phase D harness
(`bench/cana_diagnostics`) is the stopwatch that would confirm the
gated selector's net effect — a natural follow-up, not done here (Phase
D already proved the wins are real; what's unverified is that the
gated-away regressions were real wall-clock regressions, which the
modeled +14% strongly implies).

## 5. Determinism

Pure function of `(head, margin, MIR, features, ranked plan, gate)`:
fixed candidate order, `total_cmp` argmin, threshold comparison, no
RNG, no clocks. Margin-gated configs take a distinct report version
(`ENERGY_SELECTOR_VERSION_MG`) so a gated and an ungated plan never
collide in row identity.
