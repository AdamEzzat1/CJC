# ADR-0038 Locke v0.7+ A2 — Per-Value Category Lineage

- **Status:** Accepted (2026-05-29)
- **Crate:** `cjc-locke` (extended from ADRs 0028–0037)
- **Companion docs:** [[Locke Belief Reports]], [[Locke Roadmap]] §v0.7+ heavy, [[ADR-0037 Locke v0.6.4 — Auto String-Sentinel Detection and Per-Level Leakage]]

## Context

[[ADR-0037 Locke v0.6.4 — Auto String-Sentinel Detection and Per-Level Leakage]] shipped E9008 — Locke now auto-detects `?` / `NA` / `NULL` / etc. in `Str` columns and folds them into the null mask. The Phase 0.10 §4.D Part 1 finding was that the diabetes-130 `weight` column (96.9% `?`) was previously reported as `missingness_score = 1.0000`; after v0.6.4 the missingness is correctly surfaced.

The follow-up gap, identified by the v0.7+ heavy handoff §A2: *the user can see that E9008 fired, but cannot **trace** what each `?` became after the auto-mask, nor what `"Premium"` would become after case-fold consolidation, nor what `"USA. "` would become after whitespace-strip*. The existing [[ADR-0029 Locke v0.2 Capabilities|`TracedDataFrame`]] tracks lineage at the **DataFrame** level — "filter then group_by" as Idea nodes parented by Impressions — which is coarse-grained. What's missing is **per-distinct-value** lineage: for each `(column, value)` pair, the canonicalisation chain Locke would apply if the user adopted its suggested normalisations.

This ADR ships that.

## Decisions

### 1. New module `crates/cjc-locke/src/per_value_lineage.rs` (~600 LOC)

Public API:

```rust
pub struct PerValueLineageConfig {
    pub stages: PerValueStageSet,
    pub max_distinct_per_column: usize,        // default 10_000
    pub additional_sentinels: Vec<String>,
    pub trim_terminal_chars: &'static str,     // default ".,!?;:"
    pub rare_threshold: u64,                   // default 5
}

pub struct PerValueStageSet {
    pub sentinel: bool,
    pub case_fold: bool,
    pub whitespace_punct: bool,
    pub unicode_normalize: bool,
    pub rare_candidate: bool,
}

pub enum ValueTransform {
    SentinelMask { sentinel: String },                                  // E9008
    CaseFold,                                                           // E9080
    WhitespacePunctStrip { terminal_chars: String },                    // E9081
    UnicodeNormalize,                                                   // E9086
    RareCandidate { count: u64, threshold: u64 },                       // E9016
    TooManyDistinctValuesSkipped { n_distinct: usize, limit: usize },   // synthetic
}

pub struct LineageStage {
    pub transform: ValueTransform,
    pub canonical: Option<String>,
    pub siblings: BTreeSet<String>,
}

pub struct PerValueLineage {
    pub column: String,
    pub original_value: String,
    pub stages: Vec<LineageStage>,
}

pub type PerValueLineageMap = BTreeMap<(String, String), PerValueLineage>;

pub fn build_per_value_lineage(df, cfg) -> PerValueLineageMap;
pub fn trace_value(df, cfg, column, value) -> Option<PerValueLineage>;
pub fn emit_value_trace_text(lineage) -> String;
```

All types + functions re-exported from the crate root for ergonomic access.

### 2. Single source of truth for canonicalisation

Per-value lineage *reuses* the existing canonicalisation helpers from `crates/cjc-locke/src/categorical.rs` rather than reimplementing them:

- `normalize_whitespace_punct(s, terminal_chars)` — promoted to `pub(crate)`.
- `strip_combining_marks(s)` — promoted to `pub(crate)`.
- `category_counts(col)` — promoted to `pub(crate)` (handles `Column::Str` / `Column::Categorical` / `Column::CategoricalAdaptive` uniformly).

The `BUILTIN_STRING_SENTINELS` constant from `crates/cjc-locke/src/validation.rs` is already `pub` (v0.6.4).

**Why:** if Locke ever changes how it canonicalises (e.g., updates the Latin-1→ASCII table), per-value lineage automatically tracks the change. No drift risk.

### 3. Stage emission rule

A stage is emitted into a value's lineage iff one of the following holds:

1. The transform produces a **terminal** result (currently only `SentinelMask`, which maps to null and short-circuits the chain).
2. The transform's canonical form **differs** from the original value (e.g., `"Caucasian"` → `"caucasian"` under `CaseFold`).
3. The transform's canonical form has **siblings** (other distinct originals that map to the same canonical, e.g., `"Premium"` / `"premium"` / `"PREMIUM"` all reduce to `"premium"`).

This keeps clean values' lineage empty (the typical case in well-curated data) while ensuring all genuine canonicalisations show up.

`RareCandidate` is a *tag*, not a canonicalisation; it fires only when the value's count is below the threshold AND at least one other value in the column is above the threshold (so "every value rare → no signal" is correctly suppressed).

### 4. Determinism contract

- `PerValueLineageMap` is a `BTreeMap` keyed by `(column, original_value)` — sorted iteration on both axes.
- `LineageStage::siblings` is `BTreeSet<String>` — sorted output.
- `PerValueLineage::stages` is a `Vec` in fixed code order (sentinel → case_fold → whitespace_punct → unicode_normalize → rare_candidate) — never permuted by collection-iteration order.
- `emit_value_trace_text` produces byte-identical strings across runs for byte-identical inputs.

### 5. CLI surface

New subcommand `cjcl locke trace-value <data.csv> <column> <value>`:

- Reads the CSV via the existing `read_csv` helper.
- Runs `trace_value(df, PerValueLineageConfig::default(), column, value)`.
- Emits the canonical text via `emit_value_trace_text`.
- Exit `2` (caller-mapped) when the column or value isn't present.

Help text updated to surface the v0.7+ A2 reference.

### 6. Boundedness

Columns with more than `cfg.max_distinct_per_column` (default `10_000`) distinct values are skipped, emitting one synthetic entry under the key `(column, "__skipped__")` with a `TooManyDistinctValuesSkipped` transform. The caller sees the column was *considered* but bypassed, distinguishing it from "this column isn't categorical."

## Use-case map

| Need | Stage | Example |
|---|---|---|
| "Why did `weight` show 96.9% missing? What were the values?" | `SentinelMask` | `?` → null |
| "Are `Premium` and `premium` the same category?" | `CaseFold` | siblings = `["premium", "PREMIUM"]` |
| "Is `USA.` the same as `USA`?" | `WhitespacePunctStrip` | canonical = `usa` |
| "Are `café` (NFC) and `café` (NFD) the same?" | `UnicodeNormalize` | canonical = `cafe` |
| "Is `Boston` rare enough to warrant `__other__` grouping?" | `RareCandidate` | count=1 vs threshold=5 |

## Consequences

1. **Closes the v0.6.4 silent-failure story**. E9008 surfaces *that* a sentinel exists; A2 surfaces *what happened to it*. Together they explain the diabetes-130 result end-to-end.
2. **Single source of canonicalisation truth**. By promoting categorical's helpers to `pub(crate)` and reusing them, per-value lineage is *guaranteed* to apply the same transforms the detectors fire on. No drift between detection logic and lineage logic.
3. **The CLI gets a debugging primitive**. `cjcl locke trace-value` is now the canonical way to investigate "why did Locke flag this value?" — analogous to `git blame` for data quality.
4. **No detector signature changes**. Findings are unchanged; this is purely additive infrastructure. All existing snapshot tests pass without modification.
5. **Boundedness is explicit**. The synthetic `__skipped__` entry keeps the output map size predictable regardless of input alphabet width.

## Test infrastructure

| Layer | Location | New count |
|---|---|---|
| Unit (in-module) | `per_value_lineage.rs::tests` | 18 |
| Integration | `tests/locke/per_value_lineage_tests.rs` | 12 |
| Property (proptest) | `tests/locke/locke_proptest.rs` | 2 (determinism + emit well-formedness) |
| Bolero structural fuzz | `tests/locke/locke_fuzz.rs` | 1 (arbitrary strings, no panic, deterministic) |
| CLI parser | `crates/cjc-cli/src/commands/locke.rs::tests` | 3 (subcommand + missing-positional + `?` value) |

**Net delta after v0.7+ A2:**

- cjc-locke `--lib`: **291 → 309** (+18 unit tests in `per_value_lineage::tests`)
- tests/locke: **196 → 215** (+12 integration + 2 proptest + 1 bolero; net +15 visible test markers, +1 each for the per-fuzz/proptest properties counted within the existing test files)
- cjc-cli `--lib`: **154 → 157** (+3 CLI parser tests)
- tests/abng: **629** unchanged (no ABNG surface touched)
- Workspace builds clean (2 pre-existing v0.7 part 1 warnings).

## Out of scope

- **Per-value lineage for numeric columns.** Locke does not canonicalise numbers; if/when [[ADR-0035 Locke v0.6.3 — Distribution Shape, Multi-class Leakage, CategoricalAdaptive|E9024 binning]] becomes an active transform, this could be extended.
- **Stage attachment on `ValidationFinding`s.** The lineage is computed independently — findings continue to carry their existing `evidence` payload. A future revision could link findings to lineage stages via shared IDs.
- **`__other__` grouping execution.** Per-value lineage *flags* rare candidates; it does not rewrite the DataFrame. The remediation suggestion (group into `__other__`) remains a user decision.
- **E9082 (near-duplicate edit-distance) lineage.** Clusters aren't a strict canonical form — they're a relation. Deferred until "cluster summary" semantics are decided.
- **E9083 / E9084 (mixed-script, mojibake).** Property flags, not canonicalisations. Out of scope.

## Open questions

- Should `trace_value` emit *every* applicable transform (current behaviour: emit only when canonical-differs OR siblings exist), or strict-only (siblings required)? The current design is "informative pipeline view"; a `--strict` flag could be added if users prefer the focused view.
- Should `PerValueLineageMap` carry a fingerprint hash (analogous to `LockeReport::run_id`) for snapshot comparison via `cjcl locke gate`? Likely yes, v0.7.x.
- Is the lineage useful in `BeliefReport`? E.g., axis "schema" could carry a "fragmentation depth" derived from average siblings count. Not in this batch.

Related: [[ADR-0028 Locke Data Skepticism Layer]], [[ADR-0032 Locke v0.5 Capabilities]], [[ADR-0034 Locke v0.6 Batch 2 — PII, Drift, Label-Encoding, Confidence Summary, Seasonality]], [[ADR-0036 Locke v0.7 — Per-axis BeliefScore Composition Algebra]], [[ADR-0037 Locke v0.6.4 — Auto String-Sentinel Detection and Per-Level Leakage]], [[Locke Belief Reports]], [[Locke Roadmap]].
