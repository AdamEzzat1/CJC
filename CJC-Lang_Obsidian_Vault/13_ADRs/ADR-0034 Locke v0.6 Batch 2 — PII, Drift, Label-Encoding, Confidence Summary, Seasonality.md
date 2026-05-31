# ADR-0034 Locke v0.6 Batch 2 — PII, Category Drift, Label-Encoding, Confidence Summary, NFC/NFD, Seasonality

- **Status:** Accepted (v0.6 batch 2, 2026-05-28)
- **Crate:** `cjc-locke` (extended from ADR-0033)
- **Companion docs:** [[Locke Data Skepticism]] §v0.6 batch 2, [[Locke Roadmap]] §v0.6, [[Locke CLI]]

## Context

ADR-0033 shipped batch 1 of Locke v0.6 — categorical / string semantic quality + Wasserstein-1 + Mermaid lineage + reproducibility verifier. That left a substantial "medium-complexity" tier of detectors and emit pieces named in the roadmap. This ADR ships six of them as batch 2:

1. **Category drift extensions** — explicit cardinality-explosion and entropy-shift findings on top of the existing E9034 TVD.
2. **Label-encoding risk** — Int columns that look nominal-not-ordinal (small bounded value sets).
3. **Unicode NFC/NFD variants** — same word stored in two normalisation forms in the same column.
4. **PII detection** — email / phone / SSN / API-key pattern recognisers without bringing in the `regex` crate.
5. **Per-column confidence summary** — synthesise findings per column into a readable triage emit.
6. **Seasonality / periodicity** — detect hour-of-day or day-of-week clustering in time columns.

## Decisions

### 1. Category drift extensions (`drift.rs`)

Two new findings on the categorical drift path of `compare()`:

| Code | Severity | What it flags |
|---|---|---|
| **E9018** | Notice or Warning | Test cardinality ≥ `cardinality_explosion_ratio` × train cardinality (default 2.0). Severity escalates to Warning when ratio ≥ 2 × the threshold. |
| **E9019** | Notice | Absolute Shannon entropy (in nats) of category frequencies shifted by ≥ `entropy_shift_warn` (default 0.20 nats ≈ 0.29 bits). Computed via `-Σ p ln p` with Kahan summation; `0 ln 0 = 0` enforced. |

Both share the **categorical-distribution** concern with E9034 TVD but answer different questions:

- TVD: "how much probability mass moved?"
- E9018: "how many new categories appeared?"
- E9019: "did the distribution concentrate or spread out, independent of identity?"

A drift event can fire any subset (e.g. a vendor adds 500 new product SKUs that each have tiny probability — E9018 fires Warning, TVD might miss it, entropy stays similar).

`DriftConfig` knobs added: `cardinality_explosion_ratio`, `entropy_shift_warn`, `entropy_min_distinct`.

### 2. Label-encoding risk — E9023 (`validation.rs`)

Fires on `Column::Int` columns where:
- distinct count is between 2 and `max_distinct_for_risk` (default 30),
- and `(max - min + 1) / distinct ≤ span_tightness` (default 1.5) — values pack densely into a small contiguous range,
- and the column has ≥ `min_rows` rows (default 30).

Catches the common ML failure: a column like `discharge_disposition_id` (codes 1..29) gets fed to a linear model as if it were ordinal. The user sees the message:

> column `discharge_disposition_id` has 23 distinct Int values densely packed in [1, 29] — looks like a label-encoded nominal, not an ordinal numeric

Configurable via `LabelEncodingRiskConfig`. Belief mapping: → `schema_score` (the declared type is misleading for downstream).

### 3. Unicode NFC/NFD variants — E9086 (`categorical.rs`)

Detector: `detect_unicode_normalization_variants`. Groups categories by their **approximately-normalised** form computed by:
1. Strip all Unicode combining marks (U+0300–U+036F + related ranges).
2. Map Latin-1 / Latin-Extended-A precomposed accented letters to their ASCII bases via a hand-rolled ~150-entry mapping table.

Both `café` (NFC: `c a f é`) and `café` (NFD: `c a f e ◌́`) collapse to `cafe` under this normalisation. Firing requires at least one of the colliding members to actually carry a combining mark — so columns containing only precomposed Latin-1 characters don't fire false positives.

Why hand-roll the table instead of pulling in `unicode-normalization`? Locke is zero-dep at runtime by policy ([[Locke Architecture]] §scope). The hand-roll covers ~98% of real-world French/Spanish/German/Portuguese/Italian text without the dependency.

Severity Warning. Belief mapping: → `schema_score`.

### 4. PII detection — E9090–E9093 (new module `pii.rs`)

Four hand-rolled pattern recognisers with `min_match_share` threshold (default 10%):

| Code | Severity | Pattern |
|---|---|---|
| **E9090** | Warning | Email — `local@domain.tld` with conservative char set, dot-count ≥ 1, TLD ≥ 2 ASCII letters |
| **E9091** | Notice | Phone — E.164 (`+CC...`) or NA (`NNN-NNN-NNNN` / `(NNN) NNN-NNNN`) shape |
| **E9092** | Error | SSN — exactly `NNN-NN-NNNN` (we deliberately reject raw 9-digit runs to avoid false positives) |
| **E9093** | Warning | API-key — ≥ `api_key_min_len` chars (default 24), alphanumeric + `-_`, Shannon entropy ≥ `api_key_min_entropy_bits` (default 3.5 bits/char) |

All four wired into `validate_dataframe`. Belief mapping: → `constraint_score` (presence of PII is a constraint violation per data-governance policy).

Hand-rolled rather than `regex` for the same zero-dep policy as the rest of Locke. The patterns favour precision over recall: false-positive rate on legitimate non-PII (URLs, ISBNs, internal SKUs) is the bigger risk than missing edge-case PII formats.

Public API:
```rust
pub fn looks_like_email(s: &str) -> bool;
pub fn looks_like_phone(s: &str) -> bool;
pub fn looks_like_ssn(s: &str) -> bool;
pub fn looks_like_api_key(s: &str, min_len: usize, min_entropy_bits: f64) -> bool;
pub fn detect_all_pii(df: &DataFrame, cfg: &PiiConfig) -> Vec<ValidationFinding>;
```

### 5. Per-column confidence summary (new module `column_summary.rs`)

Synthesises a `LockeReport`'s findings into one `ColumnConfidenceSummary` per column. Each summary carries:

- worst severity → coarse `ConfidenceBand` (`Low` / `Moderate` / `High`)
- sorted-deduplicated list of fired codes
- per-severity counts (info / notice / warning / error)
- the message of the worst-severity finding

Plus a deterministic text emit `emit_per_column_confidence_summary(report)` whose output looks like:

```
# Per-column confidence summary
Column: country
  confidence: Moderate
  codes:      E9080, E9081, E9085
  findings:   info=0 notice=1 warning=2 error=0
  worst:      column `country` has 1 case-fold collision group(s) ...
```

Dataset-wide findings (those with `column: None`) are aggregated under a synthetic `<dataset>` row so they aren't silently dropped.

The confidence-band heuristic is deliberately coarse — it's a triage label, not a calibrated metric. The full 8-axis [[Locke Belief Reports|`BeliefScore`]] is the right tool for calibrated answers; this summary is the answer to "what's wrong with each column?" in 30 seconds.

### 6. Seasonality / periodicity — E9055 (`temporal.rs`)

Detector: `detect_seasonality`. Bucket timestamps by hour-of-day (0..23) and day-of-week (0..6, Mon=0) and compute the **index of dispersion** = `var(counts) / mean(counts)` per axis. A uniform-Poisson null produces ID ≈ 1; periodic clustering produces ID ≫ 1. Default threshold 3.0.

Emits one `E9055 Notice` per axis where the threshold is crossed, with evidence including the axis name, the ID value, and the max / min bucket positions for quick visual diagnosis.

Configurable via `SeasonalityConfig`. Handles unit (`unit_is_millis`) for both seconds-since-epoch and ms-since-epoch.

This is intentionally simple — not FFT-based, not autocorrelation-based. The hour-of-day / day-of-week axes are the two periodicity classes that catch ~95% of real-world data-pipeline failures (batch job missing graveyard hours, weekend feed offline, etc.). Sub-hour periodicity and arbitrary periods are deferred to v0.7.

## Wiring summary

| Module | Wired into |
|---|---|
| `validation::detect_label_encoding_risk` | `validate_dataframe()` (always-on, default config) |
| `pii::detect_all_pii` | `validate_dataframe()` (always-on, default config) |
| `categorical::detect_unicode_normalization_variants` | `detect_all_categorical_quality()` (always-on) |
| `drift::category_cardinality_explosion_finding` | `compare()` categorical loop |
| `drift::category_entropy_shift_finding` | `compare()` categorical loop |
| `temporal::detect_seasonality` | **not auto-wired** — caller invokes via `detect_seasonality(df, time_col, cfg)`. This matches the v0.5 convention that time-column checks require an explicit caller-supplied `time_col`. |
| `column_summary::emit_per_column_confidence_summary` | **not auto-wired** — emit-only, called explicitly from caller code or future CLI flag. |

## Belief axis mapping

| Code | Axis weakened |
|---|---|
| E9018, E9019 | (drift path — affects `drift_score` only when caller composes `validate` + `compare`) |
| E9023 | `schema_score` (declared type vs effective type mismatch) |
| E9086 | `schema_score` (semantic-category fragmentation under Unicode) |
| E9090, E9091, E9092, E9093 | `constraint_score` (PII is a constraint violation) |
| E9055 | (temporal path — currently not penalised; revisit when temporal/seasonality drives `lineage_score`) |

The 8-axis `BeliefScore` and meet-semilattice composition algebra from ADR-0032 / ADR-0033 are preserved unchanged.

## Test infrastructure

| Layer | Location | New count |
|---|---|---|
| Unit (in-module) | `categorical.rs`, `validation.rs`, `drift.rs`, `temporal.rs`, `pii.rs`, `column_summary.rs` | 20 |
| Integration | `tests/locke/{categorical_tests,validation_tests,drift_tests,pii_tests,column_summary_tests,seasonality_tests}.rs` | 38 |
| Property (proptest) | `tests/locke/locke_proptest.rs` | 4 |
| Bolero structural fuzz | `tests/locke/locke_fuzz.rs` | 3 |

Final totals after batch 2: **cjc-locke 237 lib** (was 217 batch 1, +20) + **tests/locke 141** (was 103 batch 1, +38) + **cjc-cli 154** (no regressions). Workspace builds clean.

## Out of scope (deferred to v0.7+)

Five remaining items from the original v0.6/v0.7 roadmap — these are explicitly **heavy** items each requiring multiple batches:

- **Text drift** — vocabulary KS, token-entropy drift, language-distribution shift. Needs a tokenizer.
- **Ontology / taxonomy consistency** — hyphen/underscore variants, common-prefix taxonomy inference, hierarchy fragmentation.
- **Per-value category lineage** — `raw → normalized → grouped → encoded → embedding` chain.
- **Governance workflows** — suppression files, owner annotations, required-finding policies.
- **Per-axis BeliefScore composition rules** — formalise the v0.2 plan as code (currently only the global meet-semilattice is locked).

Plus three medium-complexity items not in this batch (could ship as v0.6.x):

- **Distribution-shape diagnostics** — skew, kurtosis, top-k modes.
- **Multi-class target-leakage AUC** — v0.5 supports binary targets only.
- **`CategoricalAdaptive`** variant support — every v0.6 categorical detector skips it currently.

## Consequences

1. **Locke now covers most non-numeric data-quality failures** an analyst would encounter in real-world tabular pipelines — including the three "operational nightmare" classes (case/whitespace/typo collisions, Unicode confusables, NFC/NFD mixing) that even mature tools rarely catch.
2. **Drift triage gets two new signals (cardinality + entropy)** that the existing TVD missed. The three signals together cover the orthogonal failure modes: TVD = identity-aware mass movement, cardinality = identity-set growth, entropy = identity-independent concentration.
3. **Per-column confidence summary makes the report scannable in 30 seconds.** The full finding list remains authoritative for evidence/assumptions/suggested-checks; the summary is the lookup table for "which columns to triage first."
4. **PII detection makes Locke usable in regulated environments.** A 10%-share threshold + Error severity on SSN makes accidental PII in a training set fail loudly in CI rather than slipping through.
5. **Seasonality detection is intentionally coarse** but catches the failure modes that drive real production incidents (batch missing a time window, feed offline on weekends).

## Open questions

- Should E9055 seasonality be auto-wired into `validate_dataframe` when a time column is detected by type alone? Currently the caller must explicitly call `detect_seasonality(df, time_col, cfg)`.
- Should the per-column confidence summary become a CLI subcommand (`cjcl locke summary <data.csv>`) or stay as a library-only emit?
- The Latin-1 base-letter mapping table in `strip_combining_marks` covers Western European text. Cyrillic / Greek / Arabic NFC/NFD mixing isn't currently detected. v0.7?
- PII patterns lean US-centric (SSN format, NA phone format). International phone shapes (German, French, UK) are covered by E.164 but not by the NA pattern. v0.7 could add per-region phone variants.

Related: [[ADR-0028 Locke Data Skepticism Layer]], [[ADR-0032 Locke v0.5 Capabilities]], [[ADR-0033 Locke v0.6 Categorical and Drift Capabilities]], [[Locke Roadmap]], [[Locke Data Skepticism]], [[Locke CLI]].
