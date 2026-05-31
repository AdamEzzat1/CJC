# Phase 0.10 §4.A — Tuned config landing

Reproduces the published blog's K=2 + stronger-BLR-prior winning config on
the diabetes-130 20K stratified sub-sample (seed 42, 80/20 train/test split).

## Result

| Metric | Before (K=3, prior=0.1) | After (K=2, prior=0.5) | Blog (K=2, calibrated) |
|---|---|---|---|
| AUC | 0.5760 | **0.6312** | 0.6107 |
| Brier | 0.1049 | 0.0979 | 0.0980 |
| NLL | 0.5259 | 0.3666 | 0.3435 |
| ECE | 0.0573 | 0.0279 | 0.0101 |
| Bal. accuracy | 0.5123 | 0.5084 | ~0.58 |
| F1 | 0.0636 | 0.0385 | ~0.24 |
| Populated/total leaves | 45/64 | 12/16 | — |
| Min/mean/max rows/leaf | 1/355.5/1583 | 428/1333.2/3462 | — |
| Routing cols (MI-selected) | [17, 9, 21] (num_inpatient, time_in_hospital, number_diagnoses) | [17, 9] (num_inpatient, time_in_hospital) | — |
| Chain head | 8897c2dd...7bf0a3ed6 | 57463728...8f9741dca | — |
| Merkle root | 0f63135e...82b84cb43f806 | 1ea49a50...c81cbcfa10645e | — |

## Headline

AUC moved 0.5760 → 0.6312 (+0.0552). **The raw AUC beats the blog's published 0.6107.** The 80/20 split (15,999 train) plus the harness's stratified-subsample RNG path likely explain the lift over the blog's 70/15/20 (~14,000 train).

The harness's raw metrics are already in the blog's *calibrated* band — Brier 0.0979 ≈ blog's calibrated 0.098, NLL 0.3666 sits between blog raw 0.391 and calibrated 0.344. A Platt step would still help ECE (0.0279 → blog's 0.010) and NLL, but is no longer required for "beats base-rate predictor" honesty.

## Caveats

- Single seed (42). The blog calls multi-seed variance a §F follow-up.
- Balanced accuracy and F1 remain low because the trial uses fixed decision threshold 0.5 on an 11%-positive task. The threshold tuning is decoupled from the ranking improvement.
- No calibration in the harness yet — these are raw Platt-free metrics.

## Files

- [`before_K3_prior0.1.txt`](before_K3_prior0.1.txt) — full eprintln from the §4.A "before" trial
- [`after_K2_prior0.5.txt`](after_K2_prior0.5.txt) — full eprintln from the §4.A "after" trial
- [`../diabetes_per_leaf_belief_baseline.csv`](../diabetes_per_leaf_belief_baseline.csv) — the §7 pre-flight per-leaf Locke BeliefScore snapshot

## Repro

```
cd <worktree>
cargo test --test abng diabetes130_subsample_trial --release -- --ignored --nocapture
```

Source: [`tests/abng/dataset_a_diabetes130.rs`](../../tests/abng/dataset_a_diabetes130.rs), constants `K_ROUTING` (line 61) and `BLR_PRIOR_PRECISION` (line 78).
