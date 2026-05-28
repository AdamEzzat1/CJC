# ADR-0032 Locke v0.5 Capabilities

- **Status:** Accepted (v0.5, 2026-05-28)
- **Crate:** `cjc-locke` (extended from ADRs 0028–0031)

## Context

Locke v0.4 closed five of six Tier-1 priorities. The user-facing assessment named two **must-have v0.5 features for customer-churn ML projects** plus five smaller data-hygiene additions:

1. **Time-aware validation** (`--time-col`)
2. **Target leakage detection** (per-feature AUC)
3. **Conditional missingness** ("missing(A) implies missing(B)")
4. **Imbalanced-class warning**
5. **ID-like cardinality hint**
6. **Duplicate-key conditioning**
7. **Cross-column correlation matrix in HTML**

All seven shipped in v0.5.

## Decisions

### 1. Time-aware validation (`crates/cjc-locke/src/temporal.rs`)

- New `TimeColumnConfig` with optional `max_timestamp`, optional `gap_threshold`, `unit_is_millis` flag.
- New `detect_temporal_issues(df, time_col, &cfg)` — per-column scan emitting:
  - **E9050** (Warning) — column not sorted (non-decreasing)
  - **E9052** (Error) — row has timestamp > caller-supplied cutoff (future-leakage)
  - **E9053** (Notice) — successive timestamps with gap > threshold
  - **E9054** (Error) — column not found or not a temporal type
- New `detect_train_test_temporal_overlap(train, test, time_col)` — emits **E9051** (Error) when any test row's timestamp ≤ the maximum train timestamp.
- Accepts `Column::DateTime` (epoch millis), `Column::Int` (epoch seconds or millis), `Column::Float` (fractional seconds via i64 cast).
- Determinism: all scans single-pass O(n) with i64 arithmetic. 8 unit tests.

### 2. Target leakage detection (`crates/cjc-locke/src/leakage.rs`)

- New `LeakageConfig` (AUC thresholds 0.85 warn / 0.95 error, `min_class_count = 10`).
- New `binary_target_auc(feature, target, min_class_count) -> Option<f64>` — exact rank-sum ROC AUC over a sorted (feature, target) pair list. O(n log n) via `f64::total_cmp`. Handles ties via average ranks.
- New `detect_target_leakage(df, target_col, &cfg)` — per-feature AUC, emits:
  - **E9060** (Error) — |AUC| ≥ 0.95 (almost certainly leakage)
  - **E9061** (Warning) — 0.85 ≤ |AUC| < 0.95 (worth investigating)
  - **E9062** (Notice) — target is not binary (check skipped, multi-class deferred to v0.6)
- Uses `max(AUC, 1-AUC)` so negatively-perfect features are caught too.
- New `detect_id_like_columns(df, &cfg)` — emits **E9072** (Notice) when distinct/n_rows ≥ 0.95.
- 10 unit tests including perfect-leak detection, random-noise non-detection, AUC formula correctness.

### 3. Conditional missingness (extension to `validation.rs`)

- New `ConditionalMissingnessConfig` (`min_missing_in_a = 5`, `implication_threshold = 0.95`).
- New `detect_conditional_missingness(df, &cfg)` — pairwise scan over Float columns:
  - For each ordered pair `(A, B)`, computes `P(missing(B) | missing(A))`.
  - Emits **E9070** (Notice) when the probability crosses the threshold.
- Catches the classic churn-pipeline bug: a join failure produces jointly-null columns that look independently missing.
- 2 unit tests (positive + negative case).

### 4. Imbalanced-class warning (extension to `validation.rs`)

- New `detect_imbalanced_target(df, target_col, min_minority_rate)` — for Bool or binary-Int target columns:
  - Severity **Warning** if minority < 1%, **Notice** otherwise (below the threshold).
  - Emits **E9071** with evidence containing minority rate, n_pos, n_neg.
  - **E9075** (Error) if the target column isn't found.
- 2 unit tests (Warning at 0.5%, no finding at 50/50).

### 5. ID-like cardinality hint

- Shipped alongside leakage in `leakage.rs` since they share `LeakageConfig`. See above.

### 6. Duplicate-key conditioning (extension to `validation.rs`)

- New `detect_duplicate_key_conditioning(df, key_column)` — for each non-key column, count duplicate-key groups whose values DISAGREE on that column.
- Emits **E9073** (Notice) per disagreeing column with the count of affected groups.
- Helps debug bad joins ("user_id 7 appears twice with different last_login values").
- 2 unit tests (positive + negative).

### 7. Cross-column correlation matrix (extension to `html_emit.rs`)

- New `emit_locke_report_html_with_df(report, df)` — adds an inline-SVG correlation heatmap to the report.
- Cells colored by `|r|` using `hsl()` (white at 0, deep red at +1, deep blue at -1).
- Skipped automatically when < 3 numeric columns; capped at 30 columns (variance-ranked) when more.
- Hover tooltip shows exact `r` per cell; cells with `|r| > 0.5` display the value in-place.
- No JS, no external assets, byte-identical across runs.
- The v0.4 `emit_locke_report_html(report)` entry point still works (no DataFrame, no correlation matrix).
- 3 unit tests (rendered when ≥ 3 cols, skipped otherwise, deterministic).

## CLI surface

New flags on `cjcl locke validate`:

| Flag | Purpose |
|---|---|
| `--time-col COL` | declare the time column; enables temporal checks |
| `--max-timestamp N` | future-leakage cutoff (i64, unit follows time column) |
| `--gap-threshold N` | minimum gap between successive timestamps to fire E9053 |
| `--target COL` | enables target-leakage + imbalanced-class checks |
| `--primary-key COL` | enables duplicate-key conditioning |

`--target` previously existed only on `cjcl locke causal`; it now also applies to `validate`.

## Test counts

| Bucket | v0.4 | v0.5 | Δ |
|---|---|---|---|
| `cargo test -p cjc-locke --lib` | 156 | **182** | +26 |
| `cargo test --test locke` | 74 | 74 | 0 |
| `cargo test -p cjc-cli locke` | 8 | 8 | 0 |
| **Total Locke** | 238 | **264** | **+26** |

Adjacent: cjc-data 258/258, cjc-eval 28/28, cjc-mir-exec 17/17 — zero regression.

## Error code allocation

| Range | v0.5 additions |
|---|---|
| E9050–E9054 | temporal |
| E9060–E9062 | target leakage |
| E9070 | conditional missingness |
| E9071, E9075 | imbalanced class / missing target |
| E9072 | ID-like cardinality |
| E9073 | duplicate-key conditioning |

Remaining E9000–E9099 surface: ~30 codes free, plenty for v0.6 (Wasserstein, mutual information, conditional drift, etc.).

## Consequences

**Positive**

- Customer-churn ML projects now get first-class support for the three classic bugs (target leakage, temporal leakage, joined-null pipeline failures).
- `cjcl locke validate data.csv --time-col ts --max-timestamp 2026Q3_cutoff --target churned --primary-key customer_id --html report.html` runs a complete data-hygiene pass + emits a stakeholder-ready report in one command.
- The correlation heatmap closes the most-asked-for usability gap from the v0.4 "Locke doesn't look like pandas-profiling" feedback.

**Negative**

- Target-leakage check is binary-target only in v0.5. Multi-class one-vs-rest AUC is the v0.6 task.
- Conditional missingness uses NaN-based missingness only (Float columns); non-Float missingness via NullMask is a v0.6 extension.
- Temporal checks accept three column types but don't yet understand human-readable timestamp formats (ISO 8601 in a Str column). Parser is a v0.6 task.
- The correlation matrix renders Pearson only — non-linear relationships (which AUC catches) don't show. Spearman / Cramér's V deferred.

## Related

- [[ADR-0028 Locke Data Skepticism Layer]] (v0.1 base)
- [[ADR-0029 Locke v0.2 Capabilities]]
- [[ADR-0030 Locke v0.3 Capabilities]]
- [[ADR-0031 Locke v0.4 Capabilities]]
- [[Locke Overview]], [[Locke Roadmap]]
