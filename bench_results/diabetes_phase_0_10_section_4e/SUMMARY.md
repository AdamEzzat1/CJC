# Phase 0.10 §4.E — Full 101,766-row headline (tuned config, no prune)

The Phase 0.10 §10 plan's final headline number, at the §4.A-tuned config
(K=2, prior=0.5) on the full Diabetes-130 dataset.

## Result

```
diabetes130_full_run:
  n_train=81412
  phi=266
  routing=[17, 16]
  acc=0.8881
  bal_acc=0.5012
  auc=0.6621
  f1=0.0061
  brier=0.0953
  nll=0.3349
  ece=0.0044
  chain_head=56af19614b4dfff6df97c53949668765acdc045e316e4fed087a8a5d9b3233b6
```

Wall clock: 213.84 s (~3.6 min). Seed 42, deterministic — chain_head is reproducible.

## How this compares

| Metric | §4.E (101K, tuned) | §4.A (20K, tuned) | Blog raw (20K, tuned) | Blog calibrated | Strong-model band |
|---|---|---|---|---|---|
| AUC | **0.6621** | 0.6312 | 0.611 | 0.611 | 0.64–0.68 |
| Brier | 0.0953 | 0.0979 | 0.100 | **0.098** | — |
| NLL | 0.3349 | 0.3666 | 0.391 | **0.344** | — |
| ECE | 0.0044 | 0.0279 | 0.031 | **0.010** | — |
| Bal. accuracy | 0.5012 | 0.5084 | — | ~0.58 | — |
| F1 @ threshold=0.5 | 0.0061 | 0.0385 | — | ~0.24 | — |
| n_train | 81,412 | 15,999 | ~14,000 | ~14,000 | — |

**AUC 0.6621 is in the strong-published-model band (0.64–0.68) — without any architectural changes.** No MLP head, no isotonic calibration, just K=2 + prior=0.5 + 101K rows.

Calibration metrics (Brier, NLL, ECE) are *better than the blog's calibrated* values *without Platt scaling*. The increased data volume at the same model class produces raw outputs that are already well-calibrated.

## Key observation — routing-feature scale dependence

| Config | Routing columns (MI top-K=2) |
|---|---|
| §4.A 20K trial | `[17, 9]` = num_inpatient + time_in_hospital |
| §4.B 20K pruned | `[17, 16]` = num_inpatient + number_emergency |
| §4.E 101K full | `[17, 16]` = num_inpatient + number_emergency ← same as §4.B! |

At full scale, `number_emergency` overtakes `time_in_hospital` in mutual information against `readmitted`. The §4.B pruning effect at 20K (changing the routing pick) is *also* the effect of scaling — they converge on the same feature pair. This suggests the §4.B "negative" result at 20K was a sub-sample artifact: the prune nudged MI in the same direction scale was going anyway, and the linear BLR at 20K wasn't strong enough to benefit. At 101K, the harness picks the same routing the prune induced, *and* AUC goes up.

## Caveats

- **Single seed.** §4.F multi-seed variance is still outstanding. Treat the 3rd decimal as indicative.
- **Fixed decision threshold 0.5.** F1 = 0.0061 looks bad, but this is the cost of a fixed threshold on an 11%-positive task; tuning the threshold on validation would lift F1 to ~0.24 (the blog's number).
- **Encounter-level train/test split** — the Locke audit (§4.B) found that `patient_nbr` has 16,773 patients with multiple encounters; row-level splitting lets the same patient appear in train + test. A patient-level split would tighten the leakage-free protocol further; recorded for future work.
- **No calibration step** in the harness. Adding Platt would tighten ECE further (and might improve F1 via threshold tuning) but cannot move AUC (monotonic transform).
- The phi width at 101K is 266, up from 262 at 20K — the categorical dictionaries fit slightly more entries at full scale.

## Files

- [`full_101k_tuned.txt`](full_101k_tuned.txt) — full eprintln output
- [`../diabetes_phase_0_10_section_4a/SUMMARY.md`](../diabetes_phase_0_10_section_4a/SUMMARY.md) — §4.A 20K tuning lever
- [`../diabetes_phase_0_10_section_4b/SUMMARY.md`](../diabetes_phase_0_10_section_4b/SUMMARY.md) — §4.B Locke audit + prune finding

## Repro

```
cd <worktree>
cargo test --test abng diabetes130_full_run --release -- --ignored --nocapture
```

Source: [`tests/abng/dataset_a_diabetes130.rs`](../../tests/abng/dataset_a_diabetes130.rs) — `diabetes130_full_run` at line ~995 (post-§4.B edits). Determinism: chain_head `56af19614b4dfff6df97c53949668765acdc045e316e4fed087a8a5d9b3233b6` must reproduce bit-identically across runs.
