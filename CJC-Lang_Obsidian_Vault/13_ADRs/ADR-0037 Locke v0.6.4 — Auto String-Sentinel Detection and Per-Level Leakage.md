# ADR-0037 Locke v0.6.4 — Auto String-Sentinel Detection (E9008) and Per-Level Deterministic Leakage (E9064)

- **Status:** Accepted (v0.6.4, 2026-05-29)
- **Crate:** `cjc-locke` (extends ADR-0028 / 0033 / 0035)
- **Companion docs:** [[Locke Data Skepticism]], [[Locke Roadmap]] §v0.6.4
- **Motivating dataset:** UCI Diabetes 130-US Hospitals (1999–2008), via the [Phase 0.10 blog post](https://adamezzat1.github.io/blog/posts/abng-diabetes-revisited-with-locke/)
- **Audit log:** [`docs/locke/SILENT_FAILURES_AUDIT.md`](../../docs/locke/SILENT_FAILURES_AUDIT.md)

## Context

Two concrete misses surfaced in the Phase 0.10 §4.D Part 1 and §4.B blog work, both members of a broader pattern — Locke's default behaviour reporting "no concern" while a human reader can immediately see one.

### Miss 1 — `?` sentinels invisible to `detect_missingness`

`detect_missingness` (v0.2) treats only `f64::NAN` (Float columns) as missing without explicit configuration. For `Str` / `Int` / `Bool` / `Categorical` / `DateTime`, the caller must supply a `NullMaskMap`. Diabetes-130's `weight` column is **96.9% the literal string `?`** — Locke's default reports `missingness_score = 1.0000` ("perfect").

The §4.D Part 1 workaround in the harness (`build_question_mark_null_masks` at `tests/abng/per_leaf_belief_diabetes130.rs:455`) is a 15-line helper that every consumer would have to re-implement. It belongs in Locke.

### Miss 2 — Per-level deterministic leakage not detectable

`detect_target_leakage_multiclass` (E9063, v0.6.3) computes per-feature ROC AUC against a multi-class target. AUC is a column-wide rank statistic — it misses leakage that hides at the *level* of a column.

On diabetes-130, `discharge_disposition_id` codes 11 / 13 / 14 / 19 / 20 / 21 (death / hospice) deterministically predict `readmitted = NO` (a dead patient cannot be readmitted). But death codes are interspersed *numerically* with non-death codes (12 / 15 / 16 / 17 / 18), so the column-wide rank statistic stays below E9063's 0.85 warning threshold. The §4.B blog work documented this miss as a false-negative motivating a new detector.

### Miss 3 — `detect_conditional_missingness` was also Float-only

The pairwise NaN-implication check (E9070) had the same blind spot as `detect_missingness` — it inspected only `Column::Float` and ignored caller-supplied `NullMaskMap` entirely. Even a fully-configured caller couldn't see `?` co-missingness on `Str` columns.

## Decisions

### 1. New finding code: `E9008` — auto-detected string sentinels (Info)

New public function `detect_string_sentinels(df, cfg) -> (NullMaskMap, Vec<ValidationFinding>)` in `crates/cjc-locke/src/validation.rs`. Pre-scans every `Str` column for matches against:

```rust
pub const BUILTIN_STRING_SENTINELS: &[&str] = &[
    "?",       // UCI convention
    "NA",      // R convention
    "N/A",     // forms / spreadsheets
    "NULL",    // SQL convention (upper)
    "null",    // SQL convention (lower)
    "nan",     // pandas / numpy stringified
    "NaN",     // ditto, capitalised
    "None",    // Python stringified
    "-",       // dash placeholder
    "",        // empty string
];
```

Plus any string in `cfg.additional_sentinels`. Matching is **exact** (no trimming, no case folding) — false-positive cost is one info finding the user can read and ignore; false-negative cost is the §4.D Part 1 reality (96.9%-missing column reported as perfectly clean).

For every affected column, emits one `E9008` info finding naming the sentinel(s), the count, and the per-row rate. Returns a `NullMaskMap` that the caller unions with their own mask before further validators run.

### 2. New finding code: `E9064` — per-level deterministic-outcome leakage (Error)

New public function `detect_per_level_target_leakage(df, target_col, cfg) -> Vec<ValidationFinding>` in `crates/cjc-locke/src/leakage.rs`. For every `(column, level, target_class)` triple:

```
1. support       = count of rows where column == level
2. concentration = count of those rows where target == class
3. P(class|level) = concentration / support
4. P(class)       = base rate of class in the dataset
5. Emit E9064 when:
   - P(class|level) ≥ conditional_threshold (default 0.99)
   - support ≥ min_support (default 10)
   - P(class) < conditional_threshold (the level adds information)
```

`PerLevelLeakageConfig` defaults: `conditional_threshold = 0.99`, `min_support = 10`, `max_levels = 1000`, `max_classes = 20`. `Float` columns are skipped (continuous; binning is the caller's job).

Supports binary (`Bool`, 2-class `Int`) and multi-class (`Int` up to `max_classes` distinct) targets. The detector iterates levels in `BTreeMap` order; findings are byte-canonical across runs.

### 3. Wiring — auto-mask flows through `validate_dataframe` + `validate()`

`validate_dataframe` (the pipeline used by all callers) runs `detect_string_sentinels` first, merges the resulting mask with the caller-supplied one via the new public `merge_null_mask_maps` helper, and passes the merged map to both `detect_missingness` and `detect_conditional_missingness`. `validate()` (api.rs) also runs auto-detection so the per-column `ColumnBeliefReport.missingness_rate` reflects auto-detected sentinels, not just user-supplied masks.

Opt-out: `ValidationConfig.auto_detect_sentinels = false` restores v0.6.3 behaviour exactly.

E9064 is wired into the `cjcl locke validate --target` CLI path (`crates/cjc-cli/src/commands/locke.rs`) alongside the existing E9063 multi-class leakage check.

### 4. `detect_conditional_missingness` accepts `&NullMaskMap`

Signature changed: `detect_conditional_missingness(df, cfg, null_masks: &NullMaskMap)`. The per-column missing-row set is now the union of `Float`-NaN positions (for `Column::Float`) *and* the mask-driven positions for **all** column types. Direct callers (CLI command at `commands/locke.rs:306`, two unit tests at `validation.rs:2219, 2241`) updated.

### 5. Fix for pre-existing `detect_label_encoding_risk` overflow

Hardened `(hi - lo + 1) as f64` at `validation.rs:1983` to use `checked_sub` + `checked_add` and skip the column when the i64 range overflows. The existing bolero fuzz target `fuzz_label_encoding_risk_arbitrary_ints_never_panics` had been latently broken — bolero's deterministic input sequence shifted when v0.6.4's new fuzz targets were added to the test binary, finally exercising `i64::MIN`/`i64::MAX`-extreme inputs.

## Belief-axis mapping for v0.6.4 codes

| Code | Severity | BeliefScore axis | Notes |
|---|---|---|---|
| E9008 | Info | (none directly) | Sentinels feed into the `missingness` axis indirectly by raising E9001 missingness counts on affected columns. |
| E9064 | Error | `leakage` (when computed against a target) | Joins E9060/E9061/E9063 as a `leakage` penalty source. |

Default `BeliefPenalty` mapping is unchanged — both new codes route to existing axes through their severities.

## Test infrastructure

Discipline matches ADR-0028 / 0033 / 0035 ("wiring + unit + proptest + bolero fuzz on every new detector"):

| Layer | Location | Count delta |
|---|---|---|
| **Unit** (sentinel-detection) | bottom of `crates/cjc-locke/src/validation.rs` | +7 tests |
| **Unit** (E9064) | bottom of `crates/cjc-locke/src/leakage.rs` | +8 tests |
| **Integration** (wiring + cross-detector) | `tests/locke/sentinel_e9064_tests.rs` (new file) | +12 tests |
| **Proptest** (determinism, monotonicity, opt-out) | `tests/locke/locke_proptest.rs` | +4 properties × 256 cases |
| **Bolero fuzz** (no-panic + canonical evidence) | `tests/locke/locke_fuzz.rs` | +2 fuzz targets |

Net delta: cjc-locke --lib **284** (was 268, +16) + tests/locke **194** (was 176, +18) + cjc-cli **154** (no regressions). ABNG suite **629/629** unchanged.

## Consequences

- Diabetes-130's `weight` / `medical_specialty` / `payer_code` now show real missingness in Locke's default config. The §4.D Part 1 helper in the harness is now redundant — the per-leaf belief experiment can drop `build_question_mark_null_masks` (kept for now to avoid touching test files in this release).
- E9064 detects discharge-code death/hospice leakage on diabetes-130 without any custom rule. Validated against the §4.B blog post's prediction.
- One pre-existing latent bug (`i64::MAX - i64::MIN` overflow) closed as a side-effect.
- Backward compatibility: `auto_detect_sentinels = false` reproduces v0.6.3 behaviour exactly. Detector signature changes are narrow — only `detect_conditional_missingness` gained a parameter.

## Still deferred to v0.7+ heavy

The five heavy items each requiring multiple batches stay deferred (text drift, ontology consistency, per-value lineage, governance workflows, BeliefScore-axis composition migration from ADR-0036 v0.7 part 2).

## References

- Phase 0.10 blog: https://adamezzat1.github.io/blog/posts/abng-diabetes-revisited-with-locke/ §4.B, §4.D, §6
- Audit doc: [`docs/locke/SILENT_FAILURES_AUDIT.md`](../../docs/locke/SILENT_FAILURES_AUDIT.md)
- New code: `crates/cjc-locke/src/validation.rs` (sentinels), `crates/cjc-locke/src/leakage.rs` (E9064)
- Wiring: `crates/cjc-cli/src/commands/locke.rs` (E9064 CLI)
- Tests: `tests/locke/sentinel_e9064_tests.rs` + extensions to `locke_proptest.rs` + `locke_fuzz.rs`
