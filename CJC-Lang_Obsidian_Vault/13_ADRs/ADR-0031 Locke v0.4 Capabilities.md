# ADR-0031 Locke v0.4 Capabilities

- **Status:** Accepted (v0.4, 2026-05-27)
- **Crate:** `cjc-locke` (extended from ADR-0028 / ADR-0029 / ADR-0030)

## Context

Locke v0.3 (ADR-0030) closed every v0.2 "thin or oversold" gap. The honest follow-up assessment named six v0.4 Tier-1 priorities. v0.4 closes five of them (gate command, outlier detection, sentinel detection, HTML output, no-reconstruction streaming) and ships a *structural-skeleton* Parquet reader as honest progress on the sixth.

## Decisions

### 1. `cjcl locke gate` + canonical JSON Schema

- New module `crates/cjc-locke/src/json_emit.rs` with `emit_locke_report_json(&LockeReport) -> String` and `parse_locke_report_json(&str) -> Result<LockeReport, String>`.
- **No `serde_json` dependency** — hand-written serializer (~150 LOC) and a tightly-scoped parser tuned to the canonical emission shape.
- Schema versioned via `LockeReport::SCHEMA_VERSION` (currently 1).
- Floats: NaN / Infinity / -Infinity serialize as JSON strings (`"NaN"`, `"Infinity"`, `"-Infinity"`), normal floats via Rust's `{:?}` round-trippable format.
- BTreeMap iteration + sorted finding `Vec`s give byte-identical output across repeated runs.
- New module `crates/cjc-locke/src/gate.rs` with `diff_reports(reference, current) -> ReportDiff` and `emit_diff_text`. Findings are matched by content-addressed `id`; a finding whose severity changed gets a new id, so it cleanly appears as "disappeared (old severity)" + "appeared (new severity)" in the diff.
- CLI: `cjcl locke validate ... --save-json PATH` writes the canonical JSON; `cjcl locke gate <reference.json> <current>` diffs and exits non-zero (with `--fail-on SEV`) when any *appeared* finding meets the threshold.

### 2. Outlier detection (E9040 / E9041)

- New `OutlierConfig` (configurable IQR multipliers + modified-Z thresholds + `min_n_for_outliers`).
- New `detect_outliers(df, &OutlierConfig)` validator: per-numeric-column, applies both 1.5×IQR and |modified-Z| ≥ 3.5 tests in parallel. A value is **extreme** if either gives `extreme`; **mild** otherwise.
- Modified-Z uses the median and MAD (median absolute deviation) with the Iglewicz & Hoaglin 0.6745 scaling.
- Codes: **E9040** (Notice) for mild outliers, **E9041** (Warning) for extreme outliers. Both include `iqr`, `mad`, and Q1/Q3 in their evidence record so consumers can compute their own thresholds.
- New `stats::quantile_f64`, `stats::median_absolute_deviation`, `stats::outlier_baselines` — deterministic, NaN-filtered, `f64::total_cmp`-sorted.
- The validator does **not** fire on samples with `n_valid < min_n_for_outliers` (default 20) — outlier detection is statistically unreliable below ~20 samples.

### 3. Sentinel-value detection (E9007)

- New `SentinelConfig` with default numeric sentinels (`-1`, `-99`, `-999`, `-9999`, `999`, `9999`) and default string sentinels (empty, `"NA"`, `"N/A"`, `"null"`, `"Unknown"`, `"missing"`, `"-"`, `"?"`, etc.).
- New `detect_sentinel_values(df, &SentinelConfig)` validator: per Float / Int / Str column, flags candidate sentinel values that occur `>= min_count` times AND `>= min_rate` fraction of the column.
- Code: **E9007** at Info severity — the wording is explicitly hedged ("may be a sentinel missing value") because Locke can't know whether `-1` is sensor-error or real signed delta without domain context.
- Suggests `NullMask::from_indices(...)` as the explicit remediation.

### 4. HTML report output

- New module `crates/cjc-locke/src/html_emit.rs` with `emit_locke_report_html(&LockeReport) -> String`.
- **Single self-contained file**: inline CSS, no JS, no external assets, no network requests.
- Severity color-coded summary cards (info / notice / warning / error) + findings table + assumptions panel + footer with `run_id`.
- All user-supplied strings are HTML-escaped (`<`, `>`, `&`, `"`).
- Determinism: BTreeMap iteration + sorted findings → byte-identical HTML across runs.
- CLI: `cjcl locke validate <data> --html PATH` (also accepts `--save-html`).
- ~5KB for a typical report.

### 5. No-reconstruction streaming

- Extended `FloatState` with Welford state (`welford_mean`, `welford_m2`, `n_valid`) for incremental mean+variance, plus a `BTreeMap<u64, u64>` ECDF keyed by `f64::to_bits()` for sorted-insertion access.
- New public type `StreamingColumnSummary` and new methods on `StreamingValidator`:
  - `streaming_summaries() -> BTreeMap<String, StreamingColumnSummary>` — per-column mean/variance/min/max/distinct directly from running state.
  - `streaming_ks_d(name, reference_sorted) -> Option<f64>` — exact two-sample KS D-statistic via merge-walk of (this side's ECDF map) and (reference's sorted slice). **Tested to match single-shot KS D bit-for-bit.**
- Verified: `streaming_summary_is_fidelity_correct_at_sample_cap_zero` — Welford state gives accurate mean/std/min/max on 10 000 rows with `sample_cap=0`, closing v0.3's "streaming is half-real" caveat.

### 6. Parquet recognition (structural skeleton)

- New module `crates/cjc-locke/src/parquet_reader.rs` with `inspect_parquet_file(path) -> Result<u64, ParquetOpenError>`.
- Validates: file size ≥ 12 bytes, starts with `PAR1`, ends with `PAR1`, footer length sane.
- Returns `Err(ParquetOpenError::UnsupportedV04 { ... })` for structurally-valid files with a clear diagnostic message pointing to v0.5.
- **Scope honesty**: a full Parquet reader (Thrift compact protocol + Snappy decompression + PLAIN/Dictionary/RLE encodings + page navigation + nested types) is 1,500-2,500 LOC. Building it in one session would compromise either scope or quality. v0.4 ships the recognition layer with clearer diagnostics; the full decoder is the v0.5 priority.
- CLI improvement: `cjcl locke validate file.parquet` now reports *what* in the file Locke could parse ("structural framing OK") versus "not a Parquet file at all."

## Test counts

| Bucket                          | v0.3 | v0.4 | Δ   |
|---------------------------------|------|------|-----|
| `cargo test -p cjc-locke --lib` | 122  | 156  | +34 |
| `cargo test --test locke`       | 74   | 74   | 0   |
| `cargo test -p cjc-cli locke`   | 6    | 8    | +2  |
| **Total**                       | **202** | **238** | **+36** |

Adjacent regression: `cjc-data` 258/258, `cjc-eval` 28/28, `cjc-mir-exec` 17/17 — zero regression.

New unit tests by module (+34):
- `json_emit` (+6): round-trip, determinism, empty report, NaN/Infinity, special-char escaping, schema-version mismatch.
- `gate` (+5): clean diff, appeared findings, gate failure threshold, determinism, canonical text.
- `html_emit` (+5): determinism, dataset label, severity classes, character escaping, empty report.
- `validation` (+8): 4 outlier tests + 4 sentinel tests.
- `stats` (+0, but new fns: quantile_f64, median_absolute_deviation, outlier_baselines — exercised via validation tests).
- `streaming` (+5): Welford mean matches arithmetic, fidelity at sample_cap=0, streaming KS matches single-shot, streaming KS zero for identical, chunk-invariance.
- `parquet_reader` (+5): too small, not Parquet, missing trailing magic, well-formed shell unsupported, invalid footer length.

New CLI tests (+2): `parse_validate_with_save_json`, `parse_gate_subcommand`.

## Consequences

**Positive**

- `cjcl locke validate ... --save-json` + `cjcl locke gate ref.json current` makes the full "snapshot reference → diff against current" CI workflow possible.
- Outlier and sentinel detection close two of the most-requested missing-validator gaps. Locke v0.3 told you about missingness and duplicates; v0.4 tells you about *value-level* anomalies too.
- HTML output makes Locke reports digestible by non-technical stakeholders. The ~5KB self-contained file works on any browser, online or off, no dependencies.
- No-reconstruction streaming closes the v0.3 "sample_cap fidelity loss" caveat. KS / mean / std / min / max are now lossless past any cap.
- The Parquet diagnostic is honest progress: users who try Parquet now get "we recognise this file structurally; v0.5 will decode it" instead of "we can't read this." That's a better UX even without the full decoder.

**Negative**

- The Parquet decoder is **not** complete in v0.4. v0.5 will land Thrift compact protocol + Snappy + PLAIN encoding for the common-case Parquet files.
- Sentinel detection is heuristic — there will be false positives in domains where `-1` is real data. Severity is Info to limit noise.
- `html_emit` does not embed SVG histograms (deferred to v0.5 when we wire `cjc-vizor` properly).

## Related

- [[ADR-0028 Locke Data Skepticism Layer]] (v0.1 base)
- [[ADR-0029 Locke v0.2 Capabilities]]
- [[ADR-0030 Locke v0.3 Capabilities]]
- [[ADR-0017 Adaptive TidyView Selection]]
- [[TidyView Architecture]]
- [[Locke Overview]], [[Locke Roadmap]]
