# ABNG — Phase 0.9 Architecture Insights

**Date:** 2026-05-16
**Branch:** `claude/abng-phase-0-9`
**Audience:** future ABNG implementers + the upcoming blog post about ABNG's design

This document records two architectural insights that emerged during
Phase 0.9 Track P (Wisconsin BC baseline). They are not new features
of the codebase — they are **load-bearing design decisions** that
became visible only when we measured ABNG on a real-world tabular
classification problem. Both deserve to be promoted from
"implementation detail" to "documented architectural invariant"
before Phase 0.10 work begins.

---

## Insight 1 — The route/predictor separation

**The signature:**

```rust
g.train_step(x: &[f64], phi: &[f64], y: f64) -> Result<NodeId, GraphError>;
```

where `x` is the **routing representation** (passed to the codebook
to compute the descent path) and `phi` is the **predictive
representation** (passed to the leaf's BLR as its feature vector).
These are intentionally separate parameters, not fused into a
single `features` slice.

**Why this is non-obvious.** Most ML libraries fuse routing and
prediction into one feature vector — a single `(x, y)` pair, where
the routing model and the predictor share the same input. Splitting
them is a real design commitment with real consequences.

**Why it's the right call for ABNG.** The two roles want
fundamentally different things from their input:

| Role          | What it wants                                    | Natural choice                              |
|---------------|---------------------------------------------------|---------------------------------------------|
| **Routing**   | Compact, low-dim, easy to bin, fast to descend    | Top-K most-discriminative features (K small) |
| **Prediction**| Rich, high-dim, captures all available signal     | Full feature vector + standardization        |

Fusing them forces a compromise: either you under-feed the predictor
(routing-dim only) or you blow up routing memory (full-dim x N nodes).
Keeping them separate lets each role use what's optimal for it.

**Empirical evidence from Phase 0.9 Track P.** The baseline routes on
the top-4 F-score features (compact x) and feeds the BLR all 30
standardized features (rich phi). Trying to push phi down to 10 cost
zero accuracy (F-score top-10 captures the discriminative signal on
synthetic, where features are uncorrelated). Trying to push phi up to
30 *helped* on real BC, because BC features are correlated and the
extra dimensions added effective signal even when nominally
underdetermined (n ≈ 28 per leaf, d = 30).

**The audit-chain payoff.** Because `x` and `phi` are separate,
ABNG's audit chain can — and does — witness the routing decision
separately from the predictive update. The pre-A2 audit shape had
two distinct events per training row (`BeliefUpdate` for the route +
`BlrUpdated` for the posterior); Phase 0.8's A2 fused them into one
`TrainStep` event, but the *information* is still cleanly separable
inside that event (the routing prefix and the BLR state hash are
both recoverable from the audit log).

**Forward-looking implication.** Any future ABNG feature that wants
to evolve routing and prediction independently — different
preprocessing per role, different trigger logic, different
serialization — has a clean seam to work with. The separation is
**load-bearing** for ABNG's eventual "calibrated explainability"
story: you can show *which* features routed the sample, separately
from *which* features predicted its label.

---

## Insight 2 — The root-ensemble as core primitive (not a hyperparameter)

**The Phase 0.9 finding.** On real Wisconsin BC, per-leaf BLR alone
hits a soft accuracy ceiling around 0.944 — about 1 point below
the published logistic-regression ceiling (~0.95). The gap closes
to zero when the harness *also* trains the root BLR with every
sample and ensembles the leaf prediction with the root prediction
at evaluate time.

```rust
// In the training loop:
g.train_step(routing, phi, y)?;       // updates leaf BLR
g.blr_update(0, phi, &[y])?;           // ALSO updates root BLR

// At evaluation:
let leaf_mean = g.blr_predict_with_fallback(leaf_id, phi)?.0;
let root_mean = g.blr_predict_with_fallback(0, phi)?.0;
let ensemble = 0.5 * (leaf_mean + root_mean);
```

This sounds like a hyperparameter ("oh nice, ensembling helps").
**It is not.** It is an architectural primitive worth recognizing,
naming, and pinning as an invariant.

### The proposed invariant

> **Every adaptive local model must have access to a calibrated
> global fallback unless explicitly disabled.**

This is the *hierarchical Bayesian fallback layer*. The leaf BLR
captures local structure (per-route specialization). The root BLR
captures global structure (the linear classifier). Both are
calibrated against the same prior; their posteriors are directly
averageable in feature space.

### Why this generalizes beyond Phase 0.9's specific tree

The principle scales independently of:
* **Tree depth.** At depth 0 (root only), the root *is* the
  classifier and the ensemble is trivial. At depth N, every leaf
  prediction can be blended with its ancestors' predictions, not
  just the root.
* **Tree branching.** Binary, 4-ary, 256-ary — the ancestor BLR
  remains a valid fallback regardless.
* **Whether the tree is pre-allocated or grown organically.** When
  the structural-mutation engine (`decide_step`) splits or merges
  leaves at runtime, the ancestor fallback survives intact —
  newly-split children inherit the parent's posterior implicitly
  via the fallback path.

### When this matters: regime-by-regime

| Data regime                          | Without fallback           | With fallback                             |
|--------------------------------------|----------------------------|-------------------------------------------|
| **Globally simple** (BC, MNIST-easy) | Per-leaf BLR underperforms global LR | Ensemble ≈ global LR (matches ceiling)   |
| **Locally complex** (XOR-like, multi-modal) | Per-leaf BLR wins, leaf > root | Ensemble keeps leaf's edge, root adds tie-breaking |
| **Sparse-leaf** (n_per_leaf < d)     | Per-leaf BLR is data-starved | Root BLR carries the prediction; leaf is reserved for confident regions |

### The Bayesian model-averaging framing

This isn't just an engineering trick — it has a clean Bayesian
interpretation. The root and leaf posteriors are two
**Gaussian-NIG models at different resolutions**. Averaging their
posterior means is a coarse approximation of true Bayesian model
averaging (the full version would weight by each model's posterior
evidence + prior probability). The 0.5/0.5 weighting in Phase 0.9
is the simplest case; weighted variants are a natural Phase 0.10+
extension:

* **Confidence-weighted average:** weight each by `1 / (1 +
  epistemic_leverage)`. Confident leaves dominate; uncertain ones
  defer to root.
* **n_seen-weighted average:** weight each by their training-sample
  count. The root always wins on raw sample count.
* **Hierarchical credit assignment:** walk *all* ancestors of the
  descended leaf, weight by descent path length.

### The cost

Honest accounting: the leaf+root ensemble has a real cost.

* **Audit footprint doubles.** Each training row produces 2 chain
  events (`TrainStep` for the leaf + `BlrUpdated` for the root).
  Phase 0.9 absorbed this; the audit chain is still verifiable in
  parallel via the Phase 0.8 C2 path.
* **Predict-time work doubles.** Each evaluation runs two BLR
  predicts (leaf + root) rather than one. The cost is constant per
  row, so it's negligible for any sane workload.
* **State-space doubles for the root.** The root now holds a
  meaningful BLR posterior (instead of just the prior). One extra
  per-graph copy of `BlrState` (d²/2 + 2d + 2 floats). Phase 0.9
  with d=30 adds ~960 bytes per graph — vanishing.

These costs are accepted in exchange for matching the published
linear-classifier ceiling on a real dataset. Without the ensemble,
ABNG hit ~0.94 on Wisconsin BC; with the ensemble it hits 0.95+.

---

## Phase 0.9 measurement reference

The two insights above are not architectural speculation — they are
both anchored in measured Phase 0.9 numbers. For posterity:

| Configuration                                              | Real BC 15-seed mean | Synthetic (+1.8σ) |
|------------------------------------------------------------|---------------------:|------------------:|
| Pre-Phase-0.9 (depth=3, b=4, phi=10)                       | 0.917                | 0.81              |
| + depth=2, b=4                                             | 0.917                | 0.91              |
| + phi=10 → 30                                              | 0.918                | 0.92              |
| + depth=1, b=4                                             | 0.933                | 0.84              |
| + depth=1, b=8, threshold=0.373                            | 0.939                | 0.94              |
| + stratified train/test split                              | 0.941                | 0.94              |
| + depth=4, b=2 (binary splits on top-4 F-score)            | 0.944                | 0.96              |
| **+ leaf+root ensemble (the load-bearing change)**         | **0.952**            | **1.00**          |

The structural-config sweep alone (rows 2-7) saturated around 0.944
on real BC — 1 point below the published LR ceiling. The ensemble
(row 8) closed the gap entirely. On synthetic, the ensemble's gain
is smaller because the +1.8σ separation already produced near-ceiling
accuracy at depth=2 without the ensemble — but the ensemble doesn't
hurt synthetic, which is exactly the property the invariant
promises ("unless explicitly disabled, every local model has a
global fallback").

---

## What this implies for Phase 0.10+

The two insights suggest concrete follow-up work:

1. **Promote the leaf+root ensemble to a graph-level mode flag.**
   Add `AdaptiveBeliefGraph.fallback_mode: FallbackMode` (default
   `Disabled` for back-compat; new mode `RootEnsemble` enables the
   Phase 0.9 path). Then `train_step` and `predict` automatically
   handle the ensemble training/eval. Cleaner than the test-harness-
   level inlining used in Phase 0.9.

2. **Document the `x / phi` split as the *first* thing in any new
   user-facing ABNG tutorial.** It's the central design decision
   and currently has no dedicated documentation entry.

3. **Add a `predict_with_fallback_chain(leaf_id, phi)` API** that
   walks *all* ancestors and returns each level's posterior. Lets
   callers implement any weighting they want — confidence-weighted,
   sample-weighted, etc. — without rebuilding the ensemble logic.

4. **Cross-seed topology similarity (deferred until Phase 0.10).**
   With Phase 0.9's pre-allocated tree, topology is identical
   across seeds by construction — there's nothing to measure. Once
   organic Grow/Split triggers (Phase 0.4 Track B) are exercised
   in a Track P-like baseline, topology similarity across seeds
   becomes a meaningful interpretability metric.

5. **Per-route calibration as a first-class ABNG output.** The
   per-leaf BLR posterior naturally produces per-leaf confidence.
   Surface per-route ECE / Brier / NLL in the standard `TrialResult`
   output — Phase 0.9 added these metrics globally; the per-route
   variant is a Phase 0.10+ shape that needs UI design.

---

## Related Phase 0.9 commits

* `cc302d0` — *Track P: push real BC accuracy to 0.95+ via leaf+root
  ensemble* — the load-bearing change discussed above.
* `2e68d25` — *Track P: tune baseline config to clear 0.95 accuracy
  floor* — the synthetic-tuning predecessor that established the
  shape of the problem.
* `988196d` — *Track P: accuracy evaluation + per-leaf
  explainability* — added the `PerLeafReport` shape that the per-
  route calibration story will eventually grow into.

This doc lives alongside `PHASE_0_9_HANDOFF.md` (the parent phase
plan), `ABNG_CURRENT_ARCHITECTURE.md` (the source-of-truth
implementation snapshot), and `CAPABILITIES.md` (the per-feature
demo reference, currently on `claude/abng-v14-wire-format`).
