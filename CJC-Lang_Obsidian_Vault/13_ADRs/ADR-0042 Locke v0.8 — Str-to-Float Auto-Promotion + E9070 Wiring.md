# ADR-0042 Locke v0.8 — Str-to-Float Auto-Promotion + E9070 Wiring

- **Status:** Accepted (2026-06-02)
- **Crate:** `cjc-locke` (extends ADRs 0028–0041)
- **Companion docs:** [[ADR-0028 Locke Data Skepticism Layer]], [[ADR-0037 Locke v0.6.4 — Auto String-Sentinel Detection]], [[ADR-0041 Locke v0.8 — Custom Detector Extension Layer]]
- **Demo proof:** `demos/lendingclub/` — same audit on the full LC dataset now emits **E9009 × 22** (auto-promoted columns) + **E9070 × 1879** (conditional missingness, was silently zero pre-v0.8).

## Context

The LendingClub demo's after-action review (cross_validate.md §3 commentary, 2026-06-01) identified the gap that this ADR closes:

> **Str-with-sentinels columns silently bypass three detectors** (E9070 conditional missingness, E9039 numeric drift, E9050+ temporal). The CSV reader infers a column as `Str` if the first row has empty/sentinel values. Once typed `Str`, the column carries the sentinel string instead of NaN, and the Float-only detectors skip it. `annual_inc_joint` is the canonical example — logically numeric, but the reader sees `""` in row 1 and types it `Str`.

When I dug into the v0.7 code path to fix this, the gap turned out to be **two-fold**, not one-fold:

1. **CSV-reader type-inference gap** — `cjc-data::CsvReader` types a column from its first data row only. A numeric column with empty / `?` / `NA` first row gets typed `Str`, blinding downstream Float-only detectors.
2. **Missing call-site** — `cjc-locke::validation::detect_conditional_missingness` (E9070) was defined, unit-tested, and exported in v0.5, but **never invoked** from `validate_dataframe`. The pipeline that runs every other detector skipped over it. Even if the column had been typed Float, E9070 would still have fired zero times.

The second gap was invisible from the LC demo's external behavior because the first gap masked it. Once I added the wiring fix and re-ran with `auto_promote_str_to_float = false`, E9070 *still* fired via the sentinel-mask path on the joint columns. So both fixes were necessary; either alone partly closes the gap.

A **third gap** surfaced during the LC verification but is out of scope: `cjc-data::CsvReader` does not support RFC-4180 quoted strings. The LC dataset's `desc` (free-text loan description) column contains commas; without quote handling, subsequent columns get shifted left, polluting `annual_inc_joint` with values like `'Individual'` (from the `application_type` column). The parseable-fraction guard correctly refuses to promote in that case. ADR-0042 documents this as a known limitation; the upstream fix in `cjc-data` will land separately.

## Decisions

### 1. New module `crates/cjc-locke/src/auto_promote.rs` (~450 LOC)

```rust
pub const PROMOTION_FINDING_CODE: &str = "E9009";

pub struct ColumnScan { /* n_total, n_sentinel, n_parseable, n_other, parseable_fraction, will_promote */ }

pub fn auto_promote_str_columns(
    df: &DataFrame,
    cfg: &ValidationConfig,
) -> (Option<DataFrame>, Vec<ValidationFinding>);
```

Scans every `Str` column. Each value is classified as:

- **Sentinel** — matches `BUILTIN_STRING_SENTINELS` (`""`, `"?"`, `"NA"`, `"N/A"`, `"NULL"`, `"null"`, `"nan"`, `"NaN"`, `"None"`, `"-"`) or `cfg.additional_sentinels`.
- **Parseable** — `str::parse::<f64>()` succeeds. Accepts integer and float literals, including scientific notation (`1.5e-3`).
- **Other** — text, malformed numbers, anything else.

A column is **promoted** when:

- `non_sentinel_rows >= cfg.min_non_sentinel_rows_for_promotion` (default 10), AND
- `n_parseable / non_sentinel >= cfg.min_parseable_fraction_for_promotion` (default 0.80).

Promoted columns are rebuilt as `Column::Float` where sentinels and unparseable values become NaN. Each promoted column emits one E9009 `Info` finding documenting the counts.

Return shape:

- `(Some(new_df), findings)` if any column was promoted.
- `(None, vec![])` if no column qualified — caller uses the original DataFrame, zero allocation cost.

### 2. New `ValidationConfig` fields

```rust
pub auto_promote_str_to_float: bool,        // default true
pub min_parseable_fraction_for_promotion: f64,  // default 0.80
pub min_non_sentinel_rows_for_promotion: usize, // default 10
```

**Default ON.** The pre-v0.8 behavior silently disabled three detector families on real-world data. A flag that's off by default doesn't fix the gap for users who don't know it exists. Users with strict byte-identity requirements can opt out.

### 3. `api::validate` integration (~10 LOC modified)

```rust
let (maybe_promoted_df, promo_findings) =
    crate::auto_promote::auto_promote_str_columns(df, &opts.config);
let working_df: &DataFrame = maybe_promoted_df.as_ref().unwrap_or(df);

let mut findings = validate_dataframe(working_df, ...);
findings.extend(promo_findings);
```

`working_df` is then used everywhere downstream: column type inference, column reports, custom detectors, per-value lineage. The original `df` is no longer referenced after the promotion step. The report's `input.column_types` map reflects post-promotion types, so consumers reading the JSON see the correct view.

### 4. `validate_dataframe` wires `detect_conditional_missingness`

```rust
out.extend(detect_missingness(df, cfg, &effective_masks));
out.extend(detect_conditional_missingness(
    df,
    &ConditionalMissingnessConfig::default(),
    &effective_masks,
));  // <-- new in v0.8
out.extend(detect_duplicates_full_row(df, cfg));
```

This is the "second gap" fix. Default config: `implication_threshold = 0.95`, `min_missing_in_a = 5`. The detector honors `effective_masks` (which includes auto-detected sentinels), so it works for both Float-NaN-typed columns AND Str-with-sentinel-mask columns.

### 5. Determinism

Auto-promotion is a deterministic function of `(df, cfg)`:

- Column iteration order matches `df.columns` (a `Vec`, insertion-ordered).
- Per-column scan is single-pass, no hash-based iteration.
- The parser is `f64::from_str` from std.
- Findings are emitted in column order; their `sort_key` orders them canonically against built-in findings during the final report sort.

Proptest-locked: same `df` + same `cfg` → byte-identical output across runs.

### 6. What's explicitly NOT promoted

- **Str → Int** — `Int` columns can't represent NaN. Use Float; downstream consumers can `as i64` if they need integer semantics.
- **Str → Bool** — string `"true"`/`"false"` columns are rare in production. Excluded for scope.
- **Categorical / CategoricalAdaptive / DateTime** — already typed; not Str.
- **Numeric-with-suffix** (`"15.0%"`, `"$45000"`) — would require domain-specific format hints, rabbit hole.
- **Quoted-comma-pollution columns** — when CSV parsing shifts unrelated text into a numeric column (the LC `annual_inc_joint` case), the parseable fraction drops below threshold and the column correctly stays `Str`. The user sees no E9009 for that column and can investigate.

## Tests shipped

| File                                              | Tests | Kind                                  |
|---------------------------------------------------|-------|---------------------------------------|
| `crates/cjc-locke/src/auto_promote.rs` (in-module) | 12    | Unit (classify, scan, promote, edge cases) |
| `tests/locke/auto_promote_tests.rs`               | 14    | Integration (8) + proptest (4) + bolero (2) |
| Existing 423 cjc-locke lib tests                  | 423   | All still pass (zero regressions) |
| Existing 277 locke integration tests              | 277   | All still pass (zero regressions) |

**Total: 26 new tests + 700 existing tests, all passing.** Notable test cases:

- `e9070_fires_on_joint_columns_after_v08_fixes` — the headline integration test, proves both fixes together unblock E9070 on LC-style synthetic data.
- `e9070_fires_via_sentinel_mask_path_when_promotion_disabled` — proves the wiring fix alone (without promotion) also fires E9070 via the existing mask path. Demonstrates the opt-out behavior preserves the detector.
- `proptest_promotion_is_deterministic` — random Str column → same output across runs, including bit-identical NaN positions.
- `fuzz_arbitrary_str_columns_never_panic` — arbitrary byte sequences (lossily UTF-8 decoded) → no panics in the scan or rebuild.

## LC demo proof point

Re-running `lendingclub_demo` on the full 2.26M-row LC CSV with v0.8:

| Code  | Pre-v0.8 (2026-06-01) | Post-v0.8 (2026-06-02) | What changed |
|-------|-----------------------|------------------------|--------------|
| E9008 | 55                    | 33                     | Some sentinel-masked Str cols are now Float (E9009 supersedes E9008 for those) |
| E9009 | **n/a** (not implemented) | **22**             | Auto-promoted columns |
| E9070 | **0** (never fired)   | **1879**               | Now fires on ~54 unique columns (pairwise across the promoted set) |
| All others | unchanged within ±5 | unchanged within ±5  | |

Total findings: 577 → 2421. Total wall: 8 min → 9 min (minor cost of the additional scan and pairwise check).

The promoted column set on full LC (22 columns):

- `sec_app_*` (secondary applicant): 12 cols (fico_range_high/low, mort_acc, num_rev_accts, etc.)
- `hardship_*` (hardship plan): 5 cols (amount, dpd, length, payoff_balance_amount, last_payment_amount)
- `settlement_*` (debt settlement): 3 cols (amount, percentage, term)
- `revol_bal_joint` (joint applicant)
- `orig_projected_additional_accrued_interest`
- `mths_since_last_record`

Columns that do NOT promote despite being numeric in spirit:

- `annual_inc_joint`, `dti_joint`, `verification_status_joint` etc. — quoted-comma pollution from `desc` reduces their parseable fraction below the threshold. Documented limitation; upstream fix needed in `cjc-data::CsvReader`.

## Trade-offs explicitly accepted

- **Default ON breaks byte-identity with v0.7 reports.** Two reports over the same input will differ between v0.7 and v0.8 in: (a) ≥10 new E9009 findings, (b) potentially thousands of new E9070 findings, (c) per-column type strings change for promoted columns. Users locked to v0.7 byte-identity must set `auto_promote_str_to_float = false` AND skip the new E9070 wiring (which we cannot easily disable — they'd have to filter post-hoc).
- **One-time DataFrame clone per validate.** Promotion builds a new `DataFrame` when any column qualifies. For wide frames this is O(rows × cols), bounded by the input size. Negligible on the LC demo (1.3M rows × 152 cols = ~1.3 GB peak, well within budget).
- **`Box::leak`-style heap costs do NOT apply here** — unlike the Python custom-detector bridge, the auto-promote path uses no leaked `&'static str` because all finding codes (`E9009`) are compile-time constants.
- **Threshold defaults (0.80 / 10 rows) are heuristic.** Could be wrong for some datasets. Both are tunable; the LC demo uses defaults.

## Known limitations / out of scope

1. **CSV quoted-string handling** — `cjc-data::CsvReader` does not yet support RFC-4180 quoted strings. The LC `desc` column contains commas; without quoting, subsequent columns get column-shift-corrupted. Fix lives in `cjc-data`, not `cjc-locke`; tracked as ADR-0043 (future).
2. **`cjc-locke::drift::compare()` does NOT auto-promote.** The function still operates on `&DataFrame`. Users running drift detection on CSV-loaded data should call `auto_promote_str_columns` themselves before `compare()`. The wrapping change in `compare` is straightforward but out of scope for this commit.
3. **`StreamingValidator::ingest_chunk` does NOT auto-promote per chunk.** Same reason; streaming has its own state management and the integration deserves its own ADR.
4. **The Python bridge does not surface a separate `auto_promote_str_columns(...)` entry point.** Users get the default behavior through `validate(...)`; calling `compare_drift` from Python won't promote automatically.

## Vault updates

- New ADR-0042 (this file).
- [[Locke Roadmap]] v0.8 row marked as expanded with the auto-promotion + E9070 wiring item.
- [[LendingClub Demo Notes]] updated with the v0.8 post-fix measurements.

## File inventory

```
crates/cjc-locke/src/auto_promote.rs               +450 (new module, 12 tests)
crates/cjc-locke/src/api.rs                        ~10  (working_df wiring)
crates/cjc-locke/src/lib.rs                        +1   (mod declaration)
crates/cjc-locke/src/validation.rs                 +30  (3 new ValidationConfig fields + E9070 wire)
tests/locke/auto_promote_tests.rs                  +320 (14 tests)
tests/locke/mod.rs                                 +1   (mod declaration)
demos/lendingclub/src/lib.rs                       +6   (config field additions)
demos/lendingclub/expected_findings.json           ~30  (E9009/E9070 entries + notes)
CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0042 ...md     +new
```

Total: ~850 LOC (incl. 26 new tests + docs).
