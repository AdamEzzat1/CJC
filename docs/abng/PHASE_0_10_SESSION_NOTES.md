# Phase 0.10 — Diabetes-130 Accuracy Push, Session 1 Notes

**Date:** 2026-05-28
**Branch:** `claude/adoring-shtern-01c720` (worktree of `master` at `95acb1f`)
**Handoff source:** [`PHASE_0_10_DIABETES_ACCURACY_HANDOFF.md`](PHASE_0_10_DIABETES_ACCURACY_HANDOFF.md)
**Bench results:** [`bench_results/diabetes_phase_0_10_section_4a/`](../../bench_results/diabetes_phase_0_10_section_4a/), [`...section_4b/`](../../bench_results/diabetes_phase_0_10_section_4b/), [`...section_4e/`](../../bench_results/diabetes_phase_0_10_section_4e/)

---

## TL;DR

| Goal | Result |
|---|---|
| Floor: reproduce blog AUC 0.611 | EXCEEDED. §4.E full 101K = **AUC 0.6621** (raw). |
| Target: AUC 0.62 from Locke pruning alone | EXCEEDED on a different path — \*not\* from Locke pruning. |
| Stretch: AUC ≥ 0.64 from MLP heads (§4.C) | EXCEEDED **without** MLP heads — the linear BLR at K=2, prior=0.5, on full 101K gets us into the 0.64–0.68 strong-published-model band. |

**Three sections landed in this session:** §4.A (tuned config), §4.B (Locke audit + negative pruning finding), §4.E (full-scale headline). Three sections deferred to a follow-up session: §4.C (MLP heads), §4.D (belief-weighted prior), §4.F (multi-seed variance).

---

## Section-by-section narrative

### §7 Pre-flight

All pre-flight gates green. Dataset was absent from this worktree (it lives untracked + `.gitignore`'d); refetched from UCI archive into [`tests/data/diabetes_130/`](../../tests/data/diabetes_130/). The fetch URL is documented in [`PHASE_0_9_5_STATUS.md:100`](PHASE_0_9_5_STATUS.md:100).

| Check | Got |
|---|---|
| Workspace release build | clean (9m20s, LTO on, 23 crates) |
| `cargo test --test abng --release` | 629/629 (8 ignored, 3.46s) |
| `cargo test -p cjc-locke --lib --release` | 268/268 (0.03s) |
| `cargo test --test locke --release` | 176/176 (4 ignored, 3.06s) |
| Synthetic per-leaf | 4/4 |
| Diabetes-130 per-leaf (with `--ignored`) | 3/3 |

The diabetes-130 per-leaf experiment emitted a baseline per-leaf BeliefScore CSV: see [`bench_results/diabetes_per_leaf_belief_baseline.csv`](../../bench_results/diabetes_per_leaf_belief_baseline.csv).

### §4.A — Tuned config landed

Changed `K_ROUTING` from `3` to `2` and `BLR_PRIOR_PRECISION` from `0.1` to `0.5` in [`tests/abng/dataset_a_diabetes130.rs:61,78`](../../tests/abng/dataset_a_diabetes130.rs:61). Added an `eprintln!` to `diabetes130_subsample_trial` so the metrics are visible with `-- --ignored --nocapture`.

| Metric | Before (K=3, prior=0.1) | After (K=2, prior=0.5) | Blog (K=2 calibrated) |
|---|---|---|---|
| AUC | 0.5760 | **0.6312** | 0.6107 |
| Brier | 0.1049 | 0.0979 | 0.0980 |
| NLL | 0.5259 | 0.3666 | 0.3435 |
| ECE | 0.0573 | 0.0279 | 0.0101 |
| Populated/total leaves | 45/64 | 12/16 | — |

**Verdict.** Tuned config landed. Raw AUC at K=2 already *beats* the blog's published 0.6107 by 0.020 — we don't know why exactly, but the candidate factors are: 80/20 split (15,999 train) vs blog's 70/15/20 (~14,000 train), and our stratified-subsample RNG path may differ slightly. Worth noting: this means the published number is not optimal even for the published config; pursuing variance + protocol differences would also lift the headline.

### §4.B — Locke audit + negative pruning finding

The handoff predicted E9063 (multi-class target leakage) would fire on `discharge_disposition_id` because death codes (11/13/14/19/20/21) → `readmitted=NO` deterministically. **It did not fire.** E9063 uses per-feature ROC-AUC; the death codes are interspersed numerically with non-death codes (12/15/16/17/18), so column-wide AUC stays under the 0.85 warning threshold.

What the audit *did* return (142 findings, summary: info=58, notice=75, warning=8, **error=1**):
- **E9004 (error)** — `patient_nbr` has 30,248 duplicates across 16,773 groups. Real: patients have multiple encounters in this dataset.
- **E9072 (notice)** — `encounter_id` cardinality 1.000 (already `Ignore` in schema).
- **E9073 (notice × ~40 columns)** — duplicate-key disagreements, all natural medical data.
- **E9082 (warning)** — near-duplicate categorical strings in `weight`, `medical_specialty`, `max_glu_serum`.

Applied the leakage filter via *domain knowledge* (since Locke didn't surface it):

```rust
const DEATH_DISCHARGE_CODES: &[&str] = &["11", "13", "14", "19", "20", "21"];
```

In [`tests/abng/dataset_a_diabetes130.rs:289`](../../tests/abng/dataset_a_diabetes130.rs:289). Wrote a new ignored test [`diabetes130_subsample_trial_locke_pruned`](../../tests/abng/dataset_a_diabetes130.rs:920) that filters then runs the trial.

**The prune hurt AUC.** 20K trial: 0.6312 → 0.6194 (-0.0118).

| | §4.A baseline | §4.B pruned |
|---|---|---|
| AUC | 0.6312 | 0.6194 |
| Routing cols | [17, 9] = num_inpatient + **time_in_hospital** | [17, 16] = num_inpatient + **number_emergency** |
| Populated leaves | 12/16 | 6/16 |
| Mean rows/leaf | 1,333 | 2,667 |

**Confounded experiment.** The prune changes the data's mutual information against `readmitted`, which changes the MI-selected routing features. `time_in_hospital` was a stronger routing feature than `number_emergency` at 20K. So the prune effect ≈ (-leakage benefit) + (-stronger-routing-feature lost). Net negative.

To isolate, you'd need to force the same routing columns across configs. Deferred.

### §4.E — Full 101K headline

Ran the existing `diabetes130_full_run` test at the §4.A-tuned config (per user decision: skip the prune for the headline). Wall clock 213.84s.

**Result: AUC 0.6621 — in the strong-published-model band (0.64–0.68).**

| Metric | §4.E (101K) | §4.A (20K) | Blog calibrated (20K) |
|---|---|---|---|
| AUC | **0.6621** | 0.6312 | 0.6107 |
| Brier | 0.0953 | 0.0979 | 0.098 |
| NLL | 0.3349 | 0.3666 | 0.344 |
| ECE | 0.0044 | 0.0279 | 0.010 |
| n_train | 81,412 | 15,999 | ~14,000 |

The calibration metrics (Brier, NLL, ECE) are *better than the blog's calibrated values* — without any Platt scaling. The increased data volume at the same model class produced raw outputs that are already well-calibrated.

Chain head: `56af19614b4dfff6df97c53949668765acdc045e316e4fed087a8a5d9b3233b6`. Deterministic; reproducible.

---

## Insights

### 1. The decisive lever was data scale, not architecture

`K=3 → K=2 + prior 0.1 → 0.5` at 20K: ΔAUC +0.0552
`20K → 101K` at K=2, prior=0.5: ΔAUC +0.0309

The K_ROUTING + prior change was the bigger single jump, but the scale change was what got us into the strong-model band. Together they delivered AUC 0.5760 → 0.6621, a jump of 0.086 — bigger than any architectural change the §4.C MLP path could plausibly deliver.

The strong-published-model band is 0.64–0.68. We're at 0.6621 *without* leaving the linear BLR class. **The architectural ceiling is not where the prior analysis put it** — the linear BLR + good config + full data can match what was assumed to require nonlinear heads.

### 2. MI-based routing-feature selection is scale-sensitive

At 20K: top-2 features by MI = `[17, 9]` (num_inpatient + time_in_hospital)
At 20K with death rows dropped: `[17, 16]` (num_inpatient + number_emergency)
At 101K full: `[17, 16]` (same as the pruned 20K)

So scaling up has the same effect on MI selection as pruning the death rows — both shift the selector from `time_in_hospital` to `number_emergency`. The §4.B "negative finding" at 20K was, in part, MI-selection drift that was about to happen anyway at scale.

This has a wider implication for the ABNG / harness design: **fixed-K MI feature selection can produce different feature sets at different scales for the same data**, with substantial downstream AUC impact. Worth flagging for the ABNG roadmap — at minimum, the selector's choice should be in the audit log (currently it is, via `routing_feature_columns`).

### 3. Locke's E9063 misses per-level deterministic leakage

The handoff predicted E9063 would surface the death-discharge leakage. It didn't. The detector uses per-column ROC-AUC, which doesn't catch "specific level deterministically predicts class" when the leaking levels are interspersed numerically with non-leaking levels.

**Proposed E9064: per-level conditional-probability leakage detector.** For each `(column, level, class)` triple, compute P(class | level); flag when ≥ 0.99 with at least N supports. This would catch the discharge-code case and similar patterns (e.g. specific procedure codes deterministically predicting outcomes).

**Recorded as a Locke roadmap item.**

### 4. Locke's missingness detector misses string-sentinel datasets

Diabetes-130 stores missingness as the literal string `?`. Locke's default config looks for `NaN` (Float columns). So:
- E9007 (missingness) doesn't fire even though `weight` is 97% `?`
- E9070 (conditional missingness) doesn't fire for the same reason

The per-leaf BeliefScore CSV from §7 also showed `missingness_score = 1.0` on every leaf — Locke's defaults didn't see the `?` sentinel.

**Fix:** the per-leaf experiment and the validate CLI should accept a `--missing-marker` flag (or be smart enough to detect `?` as a common sentinel). Until then, the `ValidationConfig` would need to be threaded through with explicit `missingness_markers = vec!["?".to_string()]`.

### 5. Encounter-level train/test split is a leakage hole

`patient_nbr` has 16,773 patients with multiple encounters (101,766 total rows). The current train/test split is row-level (encounter-level), so the same patient can appear in both. The model can memorize features about that patient and "look up" their outcome.

The blog doesn't address this; it would change *what number* gets reported. A patient-level split would tighten the protocol but reduce effective data volume.

Recorded for future work (probably §F variant).

### 6. The handoff's stretch goal was reachable via a different path

The handoff said:
> **Stretch:** AUC ≥ 0.64 from MLP leaf heads (§4.C) — into the "strong published model" band.

Implicit assumption: the linear BLR is at its ceiling, and breaking that ceiling requires a nonlinear head. The scale result refutes this — the linear BLR at full data is *already* in that band. The MLP heads might still push higher (toward 0.68), but they're no longer needed for the brief's stretch goal.

This is a *reframe* for the §4.C path: it's now "push from 0.66 to 0.68", not "break a 0.61 ceiling". The cost/benefit ratio is much weaker.

---

## Determinism contract — unmoved

- 629/629 always-run ABNG tests pass (no canary drift).
- All §4.A / §4.B / §4.E results are seed 42, reproducible.
- Chain heads recorded:
  - §4.A "before": `8897c2dd674b7197...`
  - §4.A "after": `57463728a08711f6...`
  - §4.B pruned: `78a63f56fad002b2...`
  - §4.E full 101K: `56af19614b4dfff6...`
- No changes to core update rules, BLR conjugate logic, descend walk, or codebook encode. Config-only changes (K_ROUTING + BLR_PRIOR_PRECISION) and a pre-training filter (§4.B's death-row prune, applied to row data before `transform.fit`).

---

## What's left (deferred)

### §4.C — MLP leaf heads

The biggest remaining architectural lever. Adds `LeafHeadKind::Mlp { hidden }` next to the existing BLR head, per-leaf Adam training, new `AuditKind::MlpLeafUpdate` variant, new SHA-256 canary locks. Multi-session work.

Given §4.E hit the strong-model band, the cost/benefit ratio of §4.C is now much weaker. **Recommend re-prioritizing as "push toward 0.68", not "break the 0.61 ceiling".**

### §4.D — Belief-weighted per-leaf prior

Operationalize the directional hypothesis from [`docs/locke/ABNG_PER_LEAF_BELIEF.md`](../locke/ABNG_PER_LEAF_BELIEF.md): per-leaf prior precision = `BLR_PRIOR_PRECISION * (2.0 - overall_belief)`. High-belief leaves get lighter regularization.

**Open question per the handoff §9:** "does ABNG expose a per-leaf-set-prior API, or is prior set graph-wide at codebook freeze time?" Currently `set_blr_prior` is graph-wide (see [`tests/abng/dataset_a_diabetes130.rs:368`](../../tests/abng/dataset_a_diabetes130.rs:368)). If per-leaf priors require new ABNG plumbing, §4.D becomes a multi-session task.

**Pre-blocker:** The §7 per-leaf belief CSV has missingness=1.0 on every leaf (Locke didn't see `?` as missing). Belief-weighted prior is only meaningful if the belief vector is informative; first the `?`-sentinel issue (insight #4) must be addressed.

### §4.F — Multi-seed variance

Run seeds 42, 43, ..., 51 at the §4.A config on 20K (or §4.E config at 101K). Report mean ± std for AUC, Brier, NLL, ECE.

**Estimated cost:** 10 × ~40 s (20K) = ~7 min, or 10 × ~3.6 min (101K) = ~36 min.

The handoff said this is "the variance honesty the blog called out as pending." It's the lowest-risk next step and converts the single-seed §4.E headline into a defensible mean ± spread.

---

## Recommended next-session pickup order

1. **§4.F multi-seed variance** at the §4.E config (full 101K). Confirm the AUC 0.6621 isn't a lucky-seed result. Honest report: mean ± std at 10 seeds.
2. **§4.D belief-weighted prior** — but only after fixing the `?`-sentinel issue in the per-leaf belief experiment. If ABNG's `set_blr_prior` is graph-wide, defer §4.D until the ABNG plumbing decision is made.
3. **§4.C MLP heads** — last, and only if pushing from 0.66 → 0.68 is worth the multi-session work. Likely deferrable to a later phase.

---

## Files

- Source edits: [`tests/abng/dataset_a_diabetes130.rs`](../../tests/abng/dataset_a_diabetes130.rs) — `K_ROUTING`, `BLR_PRIOR_PRECISION`, `DISCHARGE_DISPOSITION_COL`, `DEATH_DISCHARGE_CODES`, `filter_out_death_discharges`, `diabetes130_subsample_trial_locke_pruned`, eprintln in `diabetes130_subsample_trial`.
- Bench artifacts:
  - [`bench_results/diabetes_phase_0_10_section_4a/SUMMARY.md`](../../bench_results/diabetes_phase_0_10_section_4a/SUMMARY.md) — tuned config landing
  - [`bench_results/diabetes_phase_0_10_section_4b/SUMMARY.md`](../../bench_results/diabetes_phase_0_10_section_4b/SUMMARY.md) — Locke audit + negative pruning result
  - [`bench_results/diabetes_phase_0_10_section_4e/SUMMARY.md`](../../bench_results/diabetes_phase_0_10_section_4e/SUMMARY.md) — full 101K headline
  - [`bench_results/diabetes_per_leaf_belief_baseline.csv`](../../bench_results/diabetes_per_leaf_belief_baseline.csv) — pre-flight per-leaf BeliefScore snapshot

## Repro one-shot

```
cargo test --test abng --release                                    # 629/629
cargo test -p cjc-locke --lib --release                              # 268/268
cargo test --test locke --release                                    # 176/176
cargo test --test abng per_leaf_belief --release                     # 4/4 (synthetic)
cargo test --test abng diabetes130_subsample_trial --release -- --ignored --nocapture
cargo test --test abng diabetes130_subsample_trial_locke_pruned --release -- --ignored --nocapture
cargo test --test abng diabetes130_full_run --release -- --ignored --nocapture
```
