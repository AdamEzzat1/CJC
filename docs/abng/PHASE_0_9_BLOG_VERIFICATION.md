# Phase 0.9 — Blog Feature Verification Report

**Date:** 2026-05-16
**Branch:** `claude/abng-phase-0-9`
**Purpose:** confirm that every ABNG feature mentioned in the
four user-facing blog posts (architecture, benchmarks part 1 +
2, deterministic systems) is still present in the current
codebase, after the Phase 0.9 Track P additions (root-ensemble,
calibration metrics, route utilization, artifact producer).

**Blog posts surveyed:**
1. <https://adamezzat1.github.io/blog/posts/abng-architecture/>
2. <https://adamezzat1.github.io/blog/posts/abng-benchmarks/>
3. <https://adamezzat1.github.io/blog/posts/abng-benchmarks-part-2/>
4. <https://adamezzat1.github.io/blog/posts/abng-deterministic-systems/>

**Headline:** Every feature mentioned across the four blog posts
is still present. The codebase has *gained* features in Phases
0.8 and 0.9, never lost them.

---

## 1. Audit chain + event kinds

**Blog claim:** 30 audit kinds (tags `0x00`..`0x1D`), genesis
`sha256(b"ABNG-GENESIS-v1")`, SHA-256 hash-chained log.

**Current state:** **31 audit kinds (tags `0x00`..`0x1E`)** —
Phase 0.8c added `TrainStep` (0x1E) as a fused per-row training
event. All previously-shipped tags retained.

| Audit kind (blog-mentioned) | Tag | Present | Source |
|---|---|---|---|
| `Created` | 0x00 | ✅ | `crates/cjc-abng/src/audit.rs:377` |
| `BeliefUpdate` | 0x01 | ✅ | `audit.rs:378` |
| `NodeAdded` | 0x02 | ✅ | `audit.rs:379` |
| `ChildrenPromoted` | 0x03 | ✅ | `audit.rs:380` |
| `CodebookFrozen` | 0x04 | ✅ | `audit.rs:381` |
| `LeafHeadConfigured` | 0x05 | ✅ | `audit.rs:382` |
| `LeafParamsInitialized` | 0x06 | ✅ | `audit.rs:383` |
| `LeafParamsUpdated` | 0x07 | ✅ | `audit.rs:384` |
| `BlrPriorConfigured` | 0x08 | ✅ | `audit.rs:385` |
| `BlrInitialized` | 0x09 | ✅ | `audit.rs:386` |
| `BlrUpdated` | 0x0A | ✅ | `audit.rs:387` |
| `DensityTrackerInstalled` | 0x0B | ✅ | `audit.rs:388` |
| `DensityUpdated` | 0x0C | ✅ | `audit.rs:389` |
| `CalibrationInstalled` | 0x0D | ✅ | `audit.rs:390` |
| `CalibrationUpdated` | 0x0E | ✅ | `audit.rs:391` |
| `DriftBaselineFrozen` | 0x0F | ✅ | `audit.rs:392` |
| `Grow` | 0x10 | ✅ | `audit.rs:395` |
| `Split` | 0x11 | ✅ | `audit.rs:396` |
| `Merge` | 0x12 | ✅ | `audit.rs:397` |
| `Prune` | 0x13 | ✅ | `audit.rs:398` |
| `Compress` | 0x14 | ✅ | `audit.rs:399` |
| `Freeze` | 0x15 | ✅ | `audit.rs:400` |
| `Unfreeze` | 0x16 | ✅ | `audit.rs:401` |
| `ExpectedEpistemicCaptured` | 0x17 | ✅ | `audit.rs:402` |
| `BlrNumericalRescue` | 0x18 | ✅ | `audit.rs:403` |
| `LeafParamsUpdatedBatch` | 0x19 | ✅ | `audit.rs:404` |
| `StatsSnapshot` | 0x1A | ✅ | `audit.rs:408` |
| `Routed` | 0x1B | ✅ | `audit.rs:407` |
| `ProvenanceStamped` | 0x1C | ✅ | `audit.rs:411` |
| `BeliefUpdateBatch` | 0x1D | ✅ | `audit.rs:414` |
| `TrainStep` | 0x1E | ✅ NEW (Phase 0.8c) | `audit.rs:418` |

---

## 2. Snapshot wire format

**Blog claim:** snapshot magic `b"ABNG\x0D"` (v13 as of Phase 0.7).

**Current state:** **v14 magic `b"ABNG\x0E"`** is the active
writer; **v13 reader retained** as a backward-compatibility
decode path. `crates/cjc-abng/src/serialize.rs:46` defines
`MAGIC_V13`; `serialize.rs:77` defines the current `MAGIC` =
v14; `serialize.rs:98` defines `COMPRESSED_MAGIC = b"ABNGZ\x01"`
(Phase 0.8 B3).

Forward compatibility: any v13 blob from the blog era still
decodes; the current writer always emits v14.

---

## 3. Core types

| Type (blog-mentioned) | Present | Source |
|---|---|---|
| `AdaptiveBeliefGraph` | ✅ | `crates/cjc-abng/src/graph.rs:265` |
| `AdaptiveBeliefNode` | ✅ | `crates/cjc-abng/src/node.rs:40` |
| `AdaptiveChildren` (enum) | ✅ | `crates/cjc-abng/src/children.rs:71` |
| `QuantileCodebook` | ✅ | `crates/cjc-abng/src/codebook.rs:80` |
| `BlrState` | ✅ | `crates/cjc-abng/src/blr.rs:180` |
| `RouteEvidence` | ✅ | `crates/cjc-abng/src/route.rs:13` |
| `DensityTracker` | ✅ | `crates/cjc-abng/src/density.rs:73` |
| `CalibrationBins` | ✅ | `crates/cjc-abng/src/calibration.rs:68` |
| `DriftBaseline` | ✅ | `crates/cjc-abng/src/drift.rs:81` |
| `DecisionPolicy` | ✅ | `crates/cjc-abng/src/policy.rs:101` |
| `NodeSignature` | ✅ | `crates/cjc-abng/src/signature.rs:135` |
| `Maturity` | ✅ | `crates/cjc-abng/src/maturity.rs:80` |
| `LeafHead` | ✅ | `crates/cjc-abng/src/leaf_head.rs:30` |

---

## 4. Core graph API functions

| Function (blog-mentioned) | Present | Source |
|---|---|---|
| `encode_prefix` | ✅ | `graph.rs:599` |
| `descend` | ✅ | `graph.rs:667` |
| `descend_traced` | ✅ | `graph.rs:861` (Phase 0.4 Track A) |
| `route_to_leaf_batch` | ✅ | `graph.rs:629` |
| `route_to_leaf_batch_par` | ✅ NEW (Phase 0.8 C1) | `graph.rs:728` |
| `blr_predict` | ✅ | `graph.rs:1615` |
| `blr_predict_with_fallback` | ✅ | `graph.rs:1647` (Phase 0.4 C-2.3.8) |
| `blr_update` | ✅ | `graph.rs:1449` |
| `train_step` (fused) | ✅ NEW (Phase 0.8c A2) | `graph.rs:1526` |
| `reset_blr` | ✅ | `graph.rs:1588` (Phase 0.4 C-2.3.5) |
| `leaf_forward` | ✅ | `graph.rs:1261` |
| `leaf_set_param` | ✅ | `graph.rs:1145` |
| `force_recapture_expected_epistemic` | ✅ | `graph.rs:2058` |
| `decide_step` | ✅ | `graph.rs:2687` |
| `merkle_root` | ✅ NEW (Phase 0.8c A3) | `graph.rs:917` |
| `verify_chain` | ✅ | `graph.rs:1037` |
| `verify_chain_par` | ✅ NEW (Phase 0.8c C2) | `graph.rs:963` |
| `serialize_compressed` | ✅ NEW (Phase 0.8 B3) | `serialize.rs:610` |
| `replay_mmap` | ✅ NEW (Phase 0.8 B1) | `serialize.rs:2282` |
| `smart_replay` | ✅ | `serialize.rs:2248` |
| `iter_sorted` (children) | ✅ | `children.rs:205` (Phase 0.7 Item E) |
| `encode_into` (codebook) | ✅ | `codebook.rs:160` (Phase 0.7 Item B buffer reuse) |

---

## 5. CJC-Lang builtins (cjcl-side `abng_*`)

| Builtin (blog-mentioned) | Present | Dispatch site |
|---|---|---|
| `abng_descend_traced` | ✅ | `crates/cjc-abng/src/dispatch.rs:587` |
| `abng_predict_snap` | ✅ | `dispatch.rs:662` |
| `abng_compact_log` | ✅ | `dispatch.rs:677` |
| `abng_leaf_set_params_batch` | ✅ | `dispatch.rs:790` |
| `abng_train_step` | ✅ NEW (Phase 0.7) | `dispatch.rs:921` |
| `abng_reset_blr` | ✅ | `dispatch.rs:962` |
| `abng_blr_predict_with_fallback` | ✅ | `dispatch.rs:975` |
| `abng_unfreeze_count` | ✅ | `dispatch.rs:1347` |
| `abng_force_recapture_expected_epistemic` | ✅ | `dispatch.rs:1444` |

Total `abng_*` dispatch arms: **73 user-facing builtins**
(per the architecture doc).

---

## 6. CLI commands (`cjcl abng …`)

| Command | Present | Source |
|---|---|---|
| `cjcl abng inspect` | ✅ | `crates/cjc-cli/src/commands/abng.rs:78` |
| `cjcl abng replay` | ✅ | `abng.rs:79` |
| `cjcl abng diff` | ✅ | `abng.rs:80` |
| `cjcl abng explain` | ✅ | `abng.rs:81` |
| `cjcl abng train` | ✅ | `abng.rs:82` |

All 5 commands ship with `--json` flag per the original Phase 0.4 Track A spec.

---

## 7. Structural-mutation engine — runtime verification

The six adaptation actions (Grow, Split, Merge, Prune, Compress,
Freeze) each have dedicated trigger tests that exercise the
`decide_step` engine through real graphs. **All trigger tests
pass on this branch:**

| Test file | Tests passing |
|---|---|
| `tests/test_abng_grow_trigger_cjcl.rs` | 7/7 ✅ |
| `tests/test_abng_split_trigger_cjcl.rs` | 8/8 ✅ |
| `tests/test_abng_prune_trigger_cjcl.rs` | 8/8 ✅ |
| `tests/test_abng_compress_trigger_cjcl.rs` | 8/8 ✅ |
| `tests/test_abng_freeze_trigger_cjcl.rs` | 8/8 ✅ |
| **Total trigger tests** | **39/39 ✅** |

The merge trigger is exercised inside the adaptive-triggers
suite and the canary tests, not in a dedicated file.

---

## 8. Original blog demos — runtime verification

| Demo / test | Tests passing |
|---|---|
| `test_abng_pinn_uncertainty` (1D heat eq from blog 2 §8.1) | 13/13 ✅ |
| `test_abng_lineage_attestation` (three-signal spoof detection from blog 2 §8.2) | 16/16 ✅ |
| Total | 29/29 ✅ |

---

## 9. Test files mentioned in blogs

All blog-mentioned test files still exist:

| Test file | Present | Path |
|---|---|---|
| `test_abng_tabular_gp.rs` | ✅ | `tests/test_abng_tabular_gp.rs` |
| `test_abng_ood_detection_cjcl.rs` | ✅ | `tests/test_abng_ood_detection_cjcl.rs` |
| `test_abng_adaptive_triggers_cjcl.rs` | ✅ | `tests/test_abng_adaptive_triggers_cjcl.rs` |
| `test_abng_compress_trigger_cjcl.rs` | ✅ | `tests/test_abng_compress_trigger_cjcl.rs` |
| `test_abng_grow_trigger_cjcl.rs` | ✅ | `tests/test_abng_grow_trigger_cjcl.rs` |
| `test_abng_split_trigger_cjcl.rs` | ✅ | `tests/test_abng_split_trigger_cjcl.rs` |
| `test_abng_prune_trigger_cjcl.rs` | ✅ | `tests/test_abng_prune_trigger_cjcl.rs` |
| `test_abng_freeze_trigger_cjcl.rs` | ✅ | `tests/test_abng_freeze_trigger_cjcl.rs` |
| `test_abng_drift_detection_cjcl.rs` | ✅ | `tests/test_abng_drift_detection_cjcl.rs` |
| `test_abng_lineage_attestation.rs` | ✅ | `tests/test_abng_lineage_attestation.rs` |
| `test_abng_pinn_uncertainty.rs` | ✅ | `tests/test_abng_pinn_uncertainty.rs` |
| `test_abng_calibration_cjcl.rs` | ✅ | `tests/test_abng_calibration_cjcl.rs` |
| `test_abng_maturity_inspection_cjcl.rs` | ✅ | `tests/test_abng_maturity_inspection_cjcl.rs` |
| `test_abng_compact_log_cjcl.rs` | ✅ | `tests/test_abng_compact_log_cjcl.rs` |

Plus 11 additional `*_scaled_cjcl.rs` versions added between the
blog and now, suggesting blog-era tests grew in scope rather
than shrinking.

---

## 10. Benchmark crates (blog mentioned 4 in `crates/`)

The blog mentioned `abng-srg-experiments`, `abng-micro`,
`abng-vs-sklearn`, `abng-lineage-at-scale`. Current locations:

| Blog name | Current location |
|---|---|
| `abng-srg-experiments` | renamed → `bench/abng_pinn_scale/` (the PINN-scaled SRG experiments) |
| `abng-micro` | ✅ `bench/abng_micro/` |
| `abng-vs-sklearn` | ✅ `bench/abng_vs_sklearn/` |
| `abng-lineage-at-scale` | ✅ `bench/abng_lineage_at_scale/` |

Note: the bench crates moved from `crates/` → `bench/` and the
SRG crate was renamed to reflect its actual scope (PINN scaling
experiments). Functionality preserved; naming improved.

---

## 11. Determinism primitives

All blog-mentioned determinism guarantees still hold:

| Primitive | Present | Notes |
|---|---|---|
| `KahanAccumulatorF64` | ✅ | `cjc_repro` (workspace) |
| `pairwise_sum_f64` | ✅ | `cjc_repro` |
| Custom FIPS 180-4 SHA-256 | ✅ | `cjc_snap::hash::sha256` (used throughout this repo + by the Phase 0.9 baseline harness) |
| `SplitMix64` RNG | ✅ | `cjc_repro::Rng` + inlined in `tests/abng/baseline_wisconsin_bc.rs:135` |
| `BTreeMap`/`BTreeSet` only | ✅ | Enforced by grep; no `HashMap` in `cjc-abng` |
| f64 canonical bytes (`to_bits().to_be_bytes()`) | ✅ | Used in `audit.rs`, `serialize.rs`, `stats.rs` |
| FMA disabled in belief kernels | ✅ | Phase 0.8c D2a/D2b reaffirmed this in SIMD-friendly Kahan refactor |
| `feature_version_hash` per node | ✅ | `blr.rs:205` (Phase 0.4 C-2.3.5) |
| `provenance_stamp_hash` per node | ✅ | `node.rs:161`, `serialize.rs:1165` (Phase 0.5 Item 1) |

---

## 12. Uncertainty + OOD machinery

| Feature | Present | Source |
|---|---|---|
| Per-leaf `(mean, epistemic_leverage, aleatoric_var)` triple | ✅ | `blr.rs::predict` returns this tuple |
| Density tracker (diagonal Gaussian) | ✅ | `density.rs` |
| 15-bin reliability diagram per leaf (`CalibrationBins`) | ✅ | `calibration.rs` |
| ECE aggregation with Kahan summation | ✅ | `calibration.rs` |
| Drift baseline + L2 z-shift score | ✅ | `drift.rs` |
| Composite OOD score `max(density, prefix_distance, epi_z)` | ✅ | `predict_snap.rs` |
| Three-tier OOD (LOW/MID/HIGH abstain) | ✅ | tested by `test_abng_ood_detection_cjcl.rs` |
| `expected_epistemic` auto-capture | ✅ | `graph.rs::force_recapture_expected_epistemic` |

---

## 13. Phase 0.9 additions (new since the blog)

Documented in
[`PHASE_0_9_ARCHITECTURE_INSIGHTS.md`](PHASE_0_9_ARCHITECTURE_INSIGHTS.md).
None of the additions remove or change blog-era functionality —
all are additive. Summary:

1. **Root-ensemble pattern** — train root BLR with every sample,
   ensemble leaf + root predictions at evaluate time. Wired in
   the Phase 0.9 baseline harness; *not yet promoted to a
   first-class `FallbackMode` enum on the graph* (that's the
   primary Phase 0.10+ follow-up).
2. **`PerLeafReport` extension** — `n_test_samples`, `test_accuracy`,
   `test_mean_predicted`, `test_brier` for per-leaf calibration.
3. **`CalibrationReport`** — Brier, NLL, ECE at the trial level.
4. **`RouteUtilization`** — populated/dead leaf counts +
   per-populated-leaf train sample distribution.
5. **Baseline harness** — `tests/abng/baseline_wisconsin_bc.rs`
   with 31 deterministic gates including 5-run determinism for
   both synthetic and real data.
6. **Bundled UCI Wisconsin BC** — SHA-256-pinned at
   `tests/data/wisconsin_bc.csv`.
7. **Artifact producer** — `bench_results/phase_0_9_baseline/`
   with CSV + summary.md + 4 SVGs.

---

## 14. Findings summary

* **Zero regressions.** Every blog-mentioned feature still
  exists.
* **Net additions (Phase 0.8 + 0.9):** one new audit kind
  (`TrainStep` 0x1E), Merkle-indexed audit chain, parallel
  chain verify, parallel route batch, mmap snapshot replay,
  streaming snapshot encode, zstd snapshot wrapper, columnar
  audit log, SIMD-friendly Kahan, fused matvec, leaf+root
  ensemble pattern, calibration metric layer, route utilization
  layer, baseline harness, real Wisconsin BC integration.
* **Wire format:** v13 → v14 (additive — v13 blobs still
  decode).
* **Naming changes:** bench crates moved `crates/` → `bench/`,
  `abng-srg-experiments` → `abng_pinn_scale` (clearer scope).
* **No removed functions, types, audit kinds, builtins, or CLI
  commands.**

---

## 15. Recommended blog post follow-up

The four blog posts can be left untouched as a record of the
v0.7 → Phase 0.8 transition. A new **Phase 0.9 blog post**
(separate from the four) should cover:

1. **The route/predictor separation** (load-bearing design
   insight, blog-worthy in its own right).
2. **The leaf+root ensemble** (the architectural change that
   pushed real Wisconsin BC to 0.95+ accuracy).
3. **The interpretability story** (4 dead routes, one leaf
   with 50% of train data).
4. **The new measurement layer** (Brier, NLL, ECE +
   per-route calibration).

Source material for the post: the Phase 0.9 architecture
insights doc, the baseline summary.md, and the 4 SVGs from
`bench_results/phase_0_9_baseline/` (also rendered as PNGs in
`~/Downloads/phase_0_9_baseline/social/` for LinkedIn +
Instagram).

---

*Verification performed 2026-05-16 by walking each blog post
end-to-end, building a feature checklist, then verifying each
item via `Grep` and runtime test execution. Methodology:
exhaustive (no spot-checking — every claim got a corresponding
codebase lookup).*
