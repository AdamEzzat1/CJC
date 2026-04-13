> **Pre-v0.1.4 document.** Uses legacy naming: CJC (now CJC-Lang), `cjc` (now `cjcl`), `.cjc` (now `.cjcl`). Kept unmodified for historical accuracy. See [../REBRAND_NOTICE.md](../REBRAND_NOTICE.md) for the full mapping.

# Bastion Final Audit

**Version:** 2.0 (Phase 2 complete)
**Date:** 2026-03-08

---

## What Landed

### P-Layer (Primitives — Rust builtins)

| # | Primitive | File | LOC | Status |
|---|-----------|------|-----|--------|
| 1 | sum_kahan | `cjc-repro/src/lib.rs:55` | pre-existing | Unchanged |
| 2 | mean (Kahan) | `cjc-runtime/src/stats.rs:18` | pre-existing | Unchanged |
| 3 | variance (two-pass Kahan) | `cjc-runtime/src/stats.rs:41` | pre-existing | Unchanged |
| 4 | min / max | `cjc-mir-exec/src/lib.rs` | pre-existing | Unchanged |
| 5 | stable_sort | `cjc-runtime/src/builtins.rs:356` | pre-existing | Unchanged |
| 6 | argsort_stable | `cjc-runtime/src/tensor.rs:1719` | pre-existing | Unchanged |
| 7 | **nth_element** | `cjc-runtime/src/stats.rs` | **~45 NEW** | Introselect via `select_nth_unstable_by` |
| 8 | rank | `cjc-runtime/src/stats.rs:430` | pre-existing | Unchanged |
| 9 | map | `cjc-runtime/src/tensor.rs:495` | pre-existing | Unchanged |
| 10 | zip_map (broadcast2) | `cjc-runtime/src/builtins.rs:2360` | pre-existing | Unchanged |
| 11 | **filter_mask** | `cjc-runtime/src/stats.rs` | **~10 NEW** | Boolean mask selection |
| 12 | gather | `cjc-runtime/src/tensor.rs:1731` | pre-existing | Unchanged |
| 13 | rolling_reduce (concrete) | `cjc-runtime/src/window.rs:33-122` | pre-existing | Unchanged |
| 14 | prefix_scan (concrete) | `cjc-runtime/src/stats.rs:359-395` | pre-existing | Unchanged |
| 15 | **sample_indices** | `cjc-runtime/src/stats.rs` | **~30 NEW** | Fisher-Yates + SplitMix64 |

### Special Functions

| # | Function | File | LOC | Status |
|---|----------|------|-----|--------|
| S1 | **erf** | `cjc-runtime/src/distributions.rs` | **~5 NEW** | Via erfc |
| S2 | **erfc** | `cjc-runtime/src/distributions.rs` | **~25 NEW** | A&S 7.1.26 |
| S3 | normal_cdf | `cjc-runtime/src/distributions.rs:177` | pre-existing | Unchanged |
| S4 | normal_pdf | `cjc-runtime/src/distributions.rs:195` | pre-existing | Unchanged |

### R-Layer (Runtime kernels — derived Rust functions)

| Function | File | LOC | Built On |
|----------|------|-----|----------|
| **median_fast** | `cjc-runtime/src/stats.rs` | **~15 NEW** | nth_element |
| **quantile_fast** | `cjc-runtime/src/stats.rs` | **~25 NEW** | nth_element |
| nth_element_copy | `cjc-runtime/src/stats.rs` | **~8 NEW** | nth_element |

### Builtin Wiring

| File | Changes |
|------|---------|
| `cjc-runtime/src/builtins.rs` | +7 dispatch entries (nth_element, median_fast, quantile_fast, filter_mask, erf, erfc) |
| `cjc-mir-exec/src/lib.rs` | +7 names in is_known_builtin, +1 stateful handler (sample_indices) |
| `cjc-types/src/effect_registry.rs` | +7 effect classifications |

---

## New Code Summary

| Category | New LOC (Rust) | Files Modified |
|----------|---------------|----------------|
| Primitives (stats.rs) | ~133 | 1 |
| Special functions (distributions.rs) | ~30 | 1 |
| Builtin wiring (builtins.rs) | ~55 | 1 |
| MIR executor wiring (lib.rs) | ~40 | 1 |
| Effect registry (effect_registry.rs) | ~8 | 1 |
| Integration tests (test_bastion_primitives.rs) | ~135 | 1 (new) |
| Unit tests (stats.rs + distributions.rs) | ~250 | 2 |
| **Total new code** | **~651** | **6 files** |

---

## Why Each Runtime Addition Was Necessary

### nth_element (~45 LOC)
**Justification:** Enables O(n) median, quantile, percentile, IQR, MAD, trimmed_mean,
winsorize, and all robust estimators. Without it, these require O(n log n) full sort.
**Leverage:** Unlocks ~15 higher-level functions.

### filter_mask (~10 LOC)
**Justification:** Boolean mask selection is fundamental for NaN-aware statistics,
outlier removal, and conditional computation. Could be emulated via gather + where,
but native is cleaner and faster.
**Leverage:** Unlocks ~8 higher-level functions.

### sample_indices (~30 LOC)
**Justification:** Required for bootstrap, jackknife, permutation tests, cross-validation,
and all resampling methods. Trivial wrapper over existing SplitMix64 RNG.
**Leverage:** Unlocks ~12 higher-level functions.

### erf/erfc (~30 LOC)
**Justification:** Completes the special function set. Required for higher-precision
normal CDF, probit, and Gaussian kernel density. Foundation for future minimax upgrade.
**Leverage:** Unlocks ~6 higher-level functions.

### median_fast / quantile_fast (~40 LOC)
**Justification:** R-layer convenience functions that compose nth_element into the
two most common use cases. Provides O(n) alternatives to O(n log n) sort-based versions.

---

## Phase 2 Additions

### `mean` as Standalone Builtin

`mean(array)` was added as a free function (Kahan-compensated via `cjc_repro::kahan_sum_f64`).
Previously only available as `Tensor.mean()`. Required by every Bastion library function.

### Stationarity Tests (R-layer, Rust)

| Function | File | LOC | Algorithm |
|----------|------|-----|-----------|
| `adf_test` | `stationarity.rs` | ~60 | OLS on diff series + DF critical values |
| `kpss_test` | `stationarity.rs` | ~40 | Level regression + Newey-West LRV |
| `pp_test` | `stationarity.rs` | ~50 | ADF + HAC correction |
| Helper: `long_run_variance` | `stationarity.rs` | ~30 | Bartlett kernel, Schwert bandwidth |
| Helper: `df_pvalue` | `stationarity.rs` | ~20 | MacKinnon table interpolation |
| Helper: `kpss_pvalue` | `stationarity.rs` | ~15 | Kwiatkowski table interpolation |

**Total new Rust:** ~280 LOC (stationarity.rs) + ~30 LOC (builtin wiring)

### Bastion Pure CJC Library (B-layer)

| Module | File | Functions | LOC |
|--------|------|-----------|-----|
| Descriptive | `lib/bastion/descriptive.cjc` | cummean, diff, pct_change, ecdf, quantile_bins | ~100 |
| Rolling | `lib/bastion/rolling.cjc` | rolling_var/std/zscore/cov/corr/beta, ewma | ~130 |
| Robust | `lib/bastion/robust.cjc` | trimmed_std, winsorized_mean, mad_std, biweight_midvariance, huber_location | ~115 |
| Resampling | `lib/bastion/resampling.cjc` | bootstrap_mean/ci/se, jackknife_mean/ci, permutation_test_mean/corr | ~170 |
| TSA | `lib/bastion/tsa.cjc` | acovf, acf, pacf, ccf, acf_with_ci, durbin_watson, spectral_entropy/flatness, dominant_frequency, band_power, rolling_autocorr, fractional_diff | ~280 |
| Inference | `lib/bastion/infer.cjc` | cohens_d, hedges_g, mean_diff_ci, odds_ratio, rank_biserial, cliffs_delta, jarque_bera | ~130 |
| Distributions | `lib/bastion/dist.cjc` | unif_pdf/cdf/ppf, exp_ppf, norm_sf/logpdf/logsf, norm_cdf/pdf/ppf_param, lognorm_pdf/cdf | ~100 |
| Transform | `lib/bastion/transform.cjc` | minmax_scale, robust_scale, demean, standard_scale, log_transform, rank_transform, boxcox | ~100 |
| **Total** | | **~36 functions** | **~1,125** |

---

## Test Coverage

| Test Suite | Count | Location |
|-----------|-------|----------|
| stats.rs unit tests | 84 (incl. ~22 new) | `cjc-runtime/src/stats.rs` |
| distributions.rs unit tests | 29 (incl. ~8 new) | `cjc-runtime/src/distributions.rs` |
| stationarity.rs unit tests | 13 (all new) | `cjc-runtime/src/stationarity.rs` |
| Bastion primitive integration tests | 14 (all new) | `tests/test_bastion_primitives.rs` |
| Bastion library integration tests | 21 (all new) | `tests/test_bastion_library.rs` |
| **Total Bastion-specific tests** | **~78 new** | |
| **Full workspace** | **2,916 passing, 0 failed** | All crates |

### Test Categories

1. **Correctness:** Known-value tests against reference implementations
2. **Edge cases:** Empty arrays, single elements, NaN, Inf, zero
3. **Determinism:** Same input + same seed = identical output (bitwise)
4. **Parity:** median_fast vs median (sort-based) produce identical results
5. **Error handling:** Out-of-bounds, length mismatches, invalid parameters
6. **Identity:** erf(x) + erfc(x) = 1
7. **Statistical validity:** ADF rejects random walks, KPSS accepts stationary series
8. **Library composition:** CJC library functions compose correctly from builtins

---

## What Was NOT Done (Deferred)

| Item | Reason | When to Revisit |
|------|--------|-----------------|
| Fused min_max | Separate min/max work fine; negligible perf gain | When profiling shows bottleneck |
| Generalized rolling_reduce | 4 concrete window ops suffice | When CJC adds closures-as-values |
| Generalized prefix_scan | 4 concrete cum ops suffice | Same as above |
| Welford online mean/var | Two-pass Kahan is numerically equivalent | If CJC adds streaming data support |
| Higher-precision erf (~1e-12) | A&S 1.5e-7 is sufficient for all current tests | When tail probability precision matters |
| Parallel bootstrap | CJC has no threading | After Stage 3 |
| STL decomposition | Needs LOESS + complex iteration | v2.0 |

---

## Dependency Impact

**New external dependencies:** 0

All new code uses only:
- Rust standard library (`Vec`, `f64` methods)
- `cjc-repro` (existing: `Rng`, `kahan_sum_f64`)
- IEEE 754 arithmetic

**Binary size impact:** Estimated <4KB incremental (stationarity tables + polynomial coefficients).

---

## Architecture Compliance

### Layer Separation

| Rule | Status |
|------|--------|
| P-layer functions are pure Rust, no CJC AST dependency | PASS |
| R-layer functions compose only P-layer primitives | PASS |
| No B-layer code in this phase (pure CJC not yet written) | PASS |
| No external crate dependencies added | PASS |
| All new functions accessible from CJC code via builtins | PASS |

### Primitive Admission Rule

Every new primitive enables 5+ higher-level functions:
- nth_element: ~15 functions
- filter_mask: ~8 functions
- sample_indices: ~12 functions
- erf/erfc: ~6 functions

All pass the admission threshold.

---

## Documents Produced

| Document | Path | Content |
|----------|------|---------|
| Structural Audit | `docs/bastion/AUDIT.md` | bunker-stats analysis |
| Feature Classification | `docs/bastion/CLASSIFICATION.md` | 118 features classified |
| Migration Map | `docs/bastion/MIGRATION_MAP.md` | Phased build plan |
| Primitive ABI Audit | `docs/bastion/PRIMITIVE_ABI_AUDIT.md` | Phase 0 gap analysis |
| Primitive ABI Spec | `docs/bastion/BASTION_PRIMITIVE_ABI.md` | 15-primitive model |
| Determinism Contract | `docs/bastion/bastion_determinism_contract.md` | Invariants & mechanisms |
| Accuracy Report | `docs/bastion/bastion_accuracy_report.md` | Numerical precision data |
| Final Audit | `docs/bastion/bastion_final_audit.md` | This document |
