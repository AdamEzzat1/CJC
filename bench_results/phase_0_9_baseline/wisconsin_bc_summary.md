# Phase 0.9 Wisconsin Breast Cancer — Baseline Summary

Computational Jacobian Core (CJC-Lang) — Adaptive Belief
Neighborhood Graph (ABNG). Phase 0.9 Track P close-out.

## Headline

* **Real Wisconsin BC 15-seed mean accuracy: 0.9519** (min 0.9217, max 0.9913)
* Synthetic seed=1 accuracy: 0.9739 (Bayes-optimal LDA ≈ 0.998)
* 5-run determinism gate: ✓ byte-equal chain heads at fixed seed (both datasets)

## Architecture (Phase 0.9 baseline config)

| Knob | Value | Rationale |
|---|---|---|
| `N_ROUTING_FEATURES` | 4 | Top-K F-score features used for routing |
| `N_BINS_PER_FEATURE` | 2 | Binary splits per routing feature |
| `N_PHI_FEATURES` | 30 | All standardized features in the BLR phi |
| Tree | 2^4 = 16 leaves | Pre-allocated full-depth |
| Ensemble | leaf + root | Bayesian fallback layer (the Phase 0.9 architectural insight) |
| Threshold | 0.30 | Calibration-tuned for the ensemble average |
| Train/test split | 80/20 stratified | Preserves 357/212 class ratio per seed |

## Calibration (test-set, leaf+root ensemble)

| Metric | Real BC seed=1 | Real BC 15-seed mean | Synthetic seed=1 |
|---|---:|---:|---:|
| Accuracy | 0.9391 | 0.9519 | 0.9739 |
| Brier    | 0.0752 | 0.0753 | 0.0163 |
| NLL      | 0.2302 | 0.3603 | 0.0859 |
| ECE (10 bins) | 0.1264 | 0.1268 | 0.0760 |

## Route utilization (real BC seed=1)

* **16** total leaves (pre-allocated `branching^depth` tree)
* **12** populated leaves (received ≥ 1 train sample)
* **4** dead leaves (routes the codebook can produce but the data never visits)
* Per-populated-leaf train counts: min=1, max=227, mean=37.83, std=66.99

## Audit chain (one trial, deterministic)

* Total audit events: 1019 (1,019)
* Chain head: `557bbbde07750796`
* Merkle root: `4603a2d2b5853829`

## Files in this bundle

| File | Bytes (approx) | Description |
|---|---:|---|
| `wisconsin_bc_summary.md` | this file | human-readable headline |
| `wisconsin_bc_real_15runs.csv` | ~3 KB | 15 real-BC trial rows |
| `wisconsin_bc_synthetic_5runs.csv` | ~1 KB | 5 synthetic trial rows |
| `wisconsin_bc_per_leaf_seed1.csv` | ~2 KB | 12 populated leaves (seed=1, real BC) |
| `wisconsin_bc_chain_heads.txt` | ~3 KB | 20 chain heads + Merkle roots |
| `wisconsin_bc_accuracy.svg` | ~5 KB | accuracy box plot (deterministic) |
| `wisconsin_bc_route_utilization.svg` | ~5 KB | per-leaf sample bars (deterministic) |
| `wisconsin_bc_per_leaf_calibration.svg` | ~5 KB | calibration scatter (deterministic) |
| `wisconsin_bc_runtime.svg` | ~4 KB | wall-clock per seed (NOT byte-stable) |

## Reproduce

```bash
cargo test --test abng --release -- --ignored \
  baseline_wisconsin_bc_produce_artifacts
```

Generated 2026-05-16 by `claude/abng-phase-0-9`.
