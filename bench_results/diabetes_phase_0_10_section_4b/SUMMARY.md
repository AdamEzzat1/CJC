# Phase 0.10 §4.B — Locke pre-training feature audit

Runs `cjcl locke validate` on the full diabetes-130 CSV plus a Locke-driven
row filter (drop death/hospice discharge codes 11/13/14/19/20/21).

## Locke audit (the audit step)

```
./target/release/cjcl.exe locke validate \
    tests/data/diabetes_130/diabetic_data.csv \
    --target readmitted \
    --primary-key patient_nbr \
    --save-json bench_results/diabetes_phase_0_10_section_4b/diabetes_locke_baseline.json \
    --html  bench_results/diabetes_phase_0_10_section_4b/diabetes_locke_baseline.html
```

| Statistic | Value |
|---|---|
| Total findings | 142 |
| Errors | 1 |
| Warnings | 8 |
| Notices | 75 |
| Info | 58 |
| Worst severity | **error** |

**Findings worth acting on:**

| Code | Severity | Subject | Action taken |
|---|---|---|---|
| E9004 | error | `patient_nbr` has 30,248 duplicates across 16,773 groups (multiple encounters per patient) | Not acted on — `patient_nbr` is `Ignore` in the schema. Real concern is encounter-level train/test split letting same patient appear in train + test; out of §4.B scope but recorded for §F multi-seed variance. |
| E9072 | notice | `encounter_id` cardinality 1.000 (ID-like) | Already `Ignore` in schema. |
| E9073 | notice (×~40 columns) | duplicate-key groups disagree on column values | Natural medical data — same patient, different encounters, different values. Not acted on. |
| E9082 | warning | `weight`, `medical_specialty`, `max_glu_serum`: near-duplicate category strings | Not acted on; these are real distinctions (e.g. `[0-25)` vs `[50-75)`). |

**Findings NOT raised but worth a note:**

| Predicted | Actual | Explanation |
|---|---|---|
| E9063 multi-class leakage on `discharge_disposition_id` (death codes) | **DID NOT FIRE** | E9063 uses per-feature ROC-AUC. Death codes (11/13/14/19/20/21) are interspersed with non-death codes in the numeric range, so the column-wide AUC stays below the 0.85 threshold. A per-LEVEL conditional-probability detector (proposed E9064) would catch this. **Recorded as a Locke roadmap item.** |
| E9070 conditional missingness on `weight`/`payer_code`/`medical_specialty` | **DID NOT FIRE** | The detector looks for NaN-implication patterns. Diabetes-130 stores missing as the literal `?` string, not NaN — Locke's default missing-marker config doesn't recognise `?`. (Same caveat as the §7 per-leaf belief observation.) |

## Row-filter trial (the action step)

Apply the filter via domain knowledge (since E9063 didn't fire), then re-run the 20K trial:

```
cargo test --test abng diabetes130_subsample_trial_locke_pruned --release -- --ignored --nocapture
```

| Metric | §4.A baseline (K=2, prior=0.5, no prune) | §4.B (K=2, prior=0.5, **with prune**) | Δ from §4.A |
|---|---|---|---|
| Rows | 101,766 | 99,343 (dropped 2,423 = 2.38%) | — |
| Train rows | 15,999 | 15,999 | — |
| AUC | **0.6312** | **0.6194** | **-0.0118** |
| Brier | 0.0979 | 0.0993 | +0.0014 |
| NLL | 0.3666 | 0.3680 | +0.0014 |
| ECE | 0.0279 | 0.0229 | -0.0050 |
| Balanced accuracy | 0.5084 | 0.5094 | +0.0010 |
| Routing cols (MI-selected) | [17, 9] (num_inpatient, **time_in_hospital**) | [17, 16] (num_inpatient, **number_emergency**) | **CHANGED** |
| Populated leaves | 12/16 | 6/16 | -6 |
| Mean rows/leaf | 1,333 | 2,667 | +1,334 |

## Headline

**The Locke-driven leakage prune did NOT improve AUC on the linear-BLR-head ABNG.** Removing death/hospice rows pushed AUC down 0.6312 → 0.6194 because:

1. The MI-based routing-feature selector picked a different K=2 set after the prune (`time_in_hospital` → `number_emergency`). `time_in_hospital` was the stronger routing feature.
2. The pruned dataset's MI is shifted in a way that surfaces less predictive features.
3. The net effect: routing-feature loss > leakage removal gain.

This is a confounded experiment — the prune affects both leakage (removed) and routing (changed). A controlled comparison would force the same routing columns; deferred.

## Decision for §4.E

Use **§4.A's config (K=2, prior=0.5, no prune)** for the full 101,766-row headline run. The §4.B pruned variant is preserved in the codebase as `diabetes130_subsample_trial_locke_pruned` for the record and for a potential future §4.C MLP-head experiment that may benefit from cleaner labels.

## Files

- [`diabetes_locke_baseline.txt`](diabetes_locke_baseline.txt) — full Locke text report (1,454 lines, 142 findings)
- [`diabetes_locke_baseline.json`](diabetes_locke_baseline.json) — structured findings (88 KB)
- [`diabetes_locke_baseline.html`](diabetes_locke_baseline.html) — HTML report with inline-SVG Pearson heatmap (68 KB)
- [`pruned_K2_prior0.5_trial.txt`](pruned_K2_prior0.5_trial.txt) — full trial output post-filter
- [`../diabetes_phase_0_10_section_4a/SUMMARY.md`](../diabetes_phase_0_10_section_4a/SUMMARY.md) — §4.A baseline

## Code

- Helper `filter_out_death_discharges` at [`tests/abng/dataset_a_diabetes130.rs:289`](../../tests/abng/dataset_a_diabetes130.rs:289)
- Test `diabetes130_subsample_trial_locke_pruned` at [`tests/abng/dataset_a_diabetes130.rs:920`](../../tests/abng/dataset_a_diabetes130.rs:920)
- Constants `DISCHARGE_DISPOSITION_COL` and `DEATH_DISCHARGE_CODES` at [`tests/abng/dataset_a_diabetes130.rs:49`](../../tests/abng/dataset_a_diabetes130.rs:49)
