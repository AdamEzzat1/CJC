# ADR-0035 Locke v0.6.3 — Distribution Shape, Multi-class Leakage, CategoricalAdaptive

- **Status:** Accepted (v0.6.3, 2026-05-28)
- **Crate:** `cjc-locke` (extended from ADRs 0028–0034)
- **Companion docs:** [[Locke Data Skepticism]] §v0.6.3, [[Locke Roadmap]] §v0.6.3, [[Locke CLI]]

## Context

Batches 1 and 2 (ADR-0033 and ADR-0034) shipped the cheap and most of the medium items from the v0.6 roadmap. Three medium items remained:

1. **Distribution-shape diagnostics** — skewness / excess kurtosis / top-K modes on numeric columns.
2. **Multi-class target-leakage AUC** — v0.5 supports only binary targets via E9060/E9061; many real-world targets are multi-class.
3. **`Column::CategoricalAdaptive` support** — every v0.6 categorical detector previously skipped this variant.

This ADR ships all three.

## Decisions

### 1. Distribution-shape diagnostics — E9024 (new module `shape.rs`)

Computes population skewness `g1 = m3 / m2^(3/2)` and excess kurtosis `g2 = m4 / m2^2 - 3` via Kahan-summed central moments. Returns `None` when:

- fewer than 2 valid (non-NaN) values,
- variance is zero or non-finite,
- m3 or m4 overflows to non-finite (e.g. on f64 vectors with `±1e150` extremes),
- the resulting skew or excess_kurt is non-finite (defensive guard against ratio overflow).

Fires E9024 (Notice) on numeric columns where either `|skew| > skew_threshold` (default 2.0) or `|excess_kurt| > kurt_threshold` (default 7.0), provided `n_valid >= min_n_valid` (default 20).

Evidence list always includes the top-K modes (default 3) so the per-column triage view shows what's dominating the distribution — sentinels, defaults, and other concentration patterns become visible without further inspection.

Auto-wired into `validate_dataframe`. Belief mapping: **none** — distribution shape is informational; constrains nothing about the column's validity by itself. A skewed column may be exactly correct for its domain.

Public API:
```rust
pub fn detect_distribution_shape(df, cfg) -> Vec<ValidationFinding>;
pub fn skew_and_kurtosis(values: &[f64]) -> Option<(f64, f64)>;
pub fn top_k_modes(col: &Column, k: usize) -> Vec<(String, u64)>;
pub struct ShapeConfig { skew_threshold, kurt_threshold, min_n_valid, top_k_modes };
```

### 2. Multi-class target-leakage AUC — E9063 (`leakage.rs`)

Extends the existing binary-target leakage path with a one-vs-rest variant for Int targets with 3–`multiclass_max_classes` (default 20) distinct values.

| Code | Severity | When |
|---|---|---|
| **E9063** | Warning | max OVR `|AUC| ≥ auc_warn_threshold` (default 0.85) |
| **E9063** | Error | max OVR `|AUC| ≥ auc_error_threshold` (default 0.95) |

For each class `c`, computes binary AUC of feature vs `(target == c)` via the existing `binary_target_auc`. Takes the **max** over classes — a feature that perfectly identifies one class but is noise on the others is still a leakage signal. Per-class top-3 AUCs included as evidence (sample label `top_per_class_aucs`) so the user can see which specific class is leaking.

Helper for callers wanting macro-averaged or per-class results directly:
```rust
pub fn multiclass_max_one_vs_rest_auc(
    feature: &[f64], target: &[u32], n_classes: u32, min_class_count: u64,
) -> Option<f64>;
```

Auto-wired into the CLI's `validate --target COL` path: when `--target` is supplied, both `detect_target_leakage` (binary) and `detect_target_leakage_multiclass` (multi-class) run. They return empty for inapplicable target shapes — calling both is safe and inexpensive.

Belief mapping: same `constraint_score` axis as E9090–E9093 since leakage is a structural violation, not a schema-shape one.

### 3. `Column::CategoricalAdaptive` support across the v0.6 categorical battery

`category_counts` in `categorical.rs` previously returned `None` for `Column::CategoricalAdaptive`, silently skipping the adaptive-width categorical storage that ships in `cjc-data`'s `byte_dict` module. Fixed:

```rust
Column::CategoricalAdaptive(cc) => {
    let dict = cc.dictionary();
    for code in cc.codes().iter() {
        let Some(bytes) = dict.get(code) else { continue };
        let label = String::from_utf8_lossy(bytes).to_string();
        *counts.entry(label).or_insert(0u64) += 1;
    }
    Some(counts)
}
```

All eight v0.6 categorical detectors (E9016 / E9017 / E9080 / E9081 / E9082 / E9083 / E9084 / E9085 / E9086) now work uniformly on `Column::Str`, `Column::Categorical`, and `Column::CategoricalAdaptive`.

Also extended `detect_high_cardinality_categorical` (E9015) to read `Column::CategoricalAdaptive`'s dictionary size directly.

`String::from_utf8_lossy` is used because `ByteDictionary` stores bytes (not strings) by design; the lossy step only triggers on corrupted streams, which would already have failed at the ingestion layer.

## Wiring summary

| Detector | Auto-wired into |
|---|---|
| `shape::detect_distribution_shape` | `validate_dataframe()` (always-on) |
| `leakage::detect_target_leakage_multiclass` | `cjcl locke validate --target COL` CLI path (alongside binary) |
| `categorical::detect_all_categorical_quality` (now adaptive-aware) | `validate_dataframe()` (already wired in batch 1) |
| `validation::detect_high_cardinality_categorical` (now adaptive-aware) | `validate_dataframe()` (existing wiring) |

## Belief axis mapping

| Code | Axis weakened |
|---|---|
| E9024 | (informational — no belief penalty) |
| E9063 | (drift / leakage axis via existing target-leakage scoring path) |
| Adaptive-aware E9016–E9086 | same as their non-adaptive counterparts |

## Test infrastructure

| Layer | Location | New count |
|---|---|---|
| Unit (in-module) | `shape.rs` (9 in-module tests for moments + detector + top-K) | 10 |
| Integration | `tests/locke/shape_tests.rs` (7), `multiclass_leakage_tests.rs` (6), `categorical_tests.rs` (+2 adaptive) | 15 |
| Property (proptest) | `locke_proptest.rs` (3 new: shape skew/kurt finiteness, shape determinism, multiclass AUC in `[0,1]`) | 3 |
| Bolero structural fuzz | `locke_fuzz.rs` (2 new: shape on arbitrary floats, multiclass leakage on arbitrary ints) | 2 |

Plus one hardening discovery during fuzz: `skew_and_kurtosis` needed a defensive `is_finite()` check on the m3/m4 computations because f64 vectors with `±1e150` values produced finite m2 but overflowing m3. The fix is documented inline in `skew_and_kurtosis`.

Final totals after batch 3: **cjc-locke 247 lib** (was 237 batch 2, +10) + **tests/locke 161** (was 141 batch 2, +20) + **cjc-cli 154** (no regressions). Workspace builds clean.

## Out of scope (deferred to v0.7+)

Five **heavy** items each requiring multiple batches remain:

- **Text drift** — vocabulary KS, token-entropy drift, language-distribution shift. Needs a tokenizer.
- **Ontology / taxonomy consistency** — hyphen/underscore variants, common-prefix taxonomy inference.
- **Per-value category lineage** — `raw → normalized → grouped → encoded → embedding` chain.
- **Governance workflows** — suppression files, owner annotations, required-finding policies.
- **Per-axis BeliefScore composition rules** — formalise the v0.2 plan as code with property tests.

## Consequences

1. **The v0.6 medium-complexity tier is now closed.** Combined with batches 1 and 2, Locke v0.6 ships all the detectors named in the original roadmap except the heavy items above.
2. **Multi-class targets are now fully supported.** Real-world medical and recommendation pipelines (UCI Diabetes-130's `readmitted` target has classes `<30`, `>30`, `NO`; product-category targets have many classes) get proper leakage analysis.
3. **`CategoricalAdaptive` works seamlessly** — `cjc-data`'s adaptive-width storage no longer disables Locke's categorical checks. Important for the ABNG Phase 0.9.5 integration, where the diabetes-130 harness uses categorical storage extensively.
4. **Distribution shape diagnostics make the per-column triage view actionable.** Top-K modes inline in evidence catch sentinel-value patterns (`-1`, `9999`) that the dedicated E9007 sometimes misses below threshold.

## Open questions

- Should the multi-class detector report **macro-averaged** AUC alongside max? The max is more sensitive to single-class leaks but less informative for "the feature has weak global signal." A `--multiclass-mode max|macro|both` CLI knob is a v0.6.4 question.
- Should `detect_distribution_shape` also fire on **categorical** columns with extremely concentrated modes (e.g. 99% top-1 share)? Currently it only fires on numeric columns; the categorical case is partially covered by E9011 (encoding risk) and E9015 (suspicious cardinality).
- The Latin-1 → ASCII base mapping in E9086 (from ADR-0034) is still Western-Europe-only. Extending to Cyrillic / Greek diacritics is a v0.7 question.

Related: [[ADR-0028 Locke Data Skepticism Layer]], [[ADR-0032 Locke v0.5 Capabilities]], [[ADR-0033 Locke v0.6 Categorical and Drift Capabilities]], [[ADR-0034 Locke v0.6 Batch 2 — PII, Drift, Label-Encoding, Confidence Summary, Seasonality]], [[Locke Roadmap]], [[Locke Data Skepticism]], [[Locke CLI]].
