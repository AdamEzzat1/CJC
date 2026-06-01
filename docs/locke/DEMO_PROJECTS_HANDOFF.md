# Locke demo projects — laptop-budget handoff

**Date**: 2026-06-01
**Author**: ChatGPT (Claude Opus 4.7) under direct prompting from Adam
**Owner of next session**: whoever picks this up; can be a fresh agent
**Target**: a single laptop, ~1 hour of compute, ~5 GB SSD headroom, ~4 GB RAM peak

This handoff is self-contained. Read top to bottom, then start working. No prior conversation context is required.

---

## 0. TL;DR

The next session executes a paired Locke demo:

1. **LendingClub default prediction** — RAM-fits, ~30s validate runs, *the leakage story*
2. **NYC Taxi 2019-Q1 through 2020-Q2** (six pandemic-bracketing quarters) — streaming-required, ~20-45 min sweep, *the streaming + drift story*

Together they cover all four claimed Locke axes (numerical rigor, determinism, thermal control, memory efficiency). Separately they each cover two.

The deliverable is a `demos/` directory with two reproducible projects, each producing a Locke report + a cross-validation file confirming the result matches a published baseline.

---

## 1. State of the worktree at handoff

**Branch**: `claude/stupefied-shtern-6e58f6`
**HEAD**: `3d74d00` (`locke: distinct+top-freq single pass, binary_search class lookup, algebra docs (C6/C7)`)

**Test counts to expect on a clean build**:
- `cargo test --release -p cjc-locke --lib` → **408 passed, 0 failed**
- `cargo test --release -p cjc-lang --test locke` → **277 passed, 0 failed, 7 ignored** (the 7 ignored are scale benches gated behind `--ignored`)
- `cargo test --release -p cjc-locke --doc` → **2 passed, 2 ignored**

**Working tree should be clean**:
```
$ git status --short
# (empty)
```

If `cargo build` fails or test counts differ, do not start the demo — diagnose first.

**Recent performance work that landed on this branch** (don't undo any of this):
- C2-C5: api.rs / validation.rs facade cleanup (committed earlier on the parent branch)
- C6: `distinct_count_and_top_freq` single-pass helper (this commit)
- C7: `binary_search` on sorted `labels` Vec in `extract_multiclass_target` (this commit)
- Stable bench harness (median of 5 + stddev%) in `tests/locke/scale_benchmark.rs`
- Algebra section added to `crates/cjc-locke/src/lib.rs` front-page docs
- `docs/locke/PERF_C6_REVERT_NOTES.md` exists as the mechanical revert recipe if a cool-system bench later shows C6 is a net loss

---

## 2. Constraints (read before scoping anything)

**Hardware budget** (laptop only):
- RAM peak: ≤4 GB working set across both demos
- SSD: ≤5 GB total downloaded + processed data
- CPU: assume thermal throttling kicks in around 20-30 min of sustained load; demo should accommodate
- Power: assume plugged in for the Taxi run; battery for LendingClub is fine
- OS: Windows 11 (worktree path is `C:\Users\adame\CJC\.claude\worktrees\...`); paths must use backslash or forward slash consistently. Treat the demos as `cjc-lang`-workspace tests, not as standalone crates.

**Zero-external-dep invariant**:
- Anything we add must use the workspace's existing crates (`cjc-data`, `cjc-repro`, `cjc-runtime`, `cjc-locke`).
- If a demo needs CSV parsing, use `cjc-data::csv`.
- If it needs Parquet, check `crates/cjc-data/src/parquet_reader.rs` first; if it's not there, write a thin streaming reader using `std::io::BufReader` + the existing Parquet dependency that `cjc-locke::parquet_reader` already pulls in.
- **DO NOT add `polars`, `arrow`, `parquet` as direct deps to a demo crate** — they're already transitively available via the workspace chain. Use what's there.

**Time budget**:
- Total demo compute budget: ~60 min on a laptop, of which ~45 min is the NYC Taxi streaming sweep and the rest is LendingClub + report rendering.
- If a single iteration of either demo exceeds these budgets by >50%, stop and re-scope — don't grind through it.

---

## 3. Project A — LendingClub Default Prediction Leakage Audit

### 3.1 Dataset acquisition

**Source**: LendingClub historical loan-level data, 2007-2018 (LC discontinued investor program in 2020; full dumps preserved on Kaggle).

**Preferred snapshot**: `wordsforthewise/lending-club` on Kaggle — has the complete 2007-2018 corpus as a single ~600 MB CSV (`accepted_2007_to_2018Q4.csv.gz` or equivalent).

**Alternative**: any of the per-quarter CSV files on the same Kaggle dataset; concatenation is straightforward.

**Download flow** (assume the next session is offline-friendly):
1. Browse to https://www.kaggle.com/datasets/wordsforthewise/lending-club
2. Download `accepted_2007_to_2018q4.csv.gz` (~600 MB compressed)
3. Place at `demos/lendingclub/data/accepted_2007_to_2018q4.csv.gz`
4. Do not commit the data file (already covered by `.gitignore` patterns, but verify with `git status` before committing anything).

**Schema reference**: https://resources.lendingclub.com/LCDataDictionary.xlsx (LendingClub's own data dictionary). Approximately 150 columns. Key ones for the demo:
- `loan_status` — target (charge-off / fully paid / etc.)
- `issue_d` — the prediction horizon anchor; anything dated after this is leakage
- `int_rate`, `loan_amnt`, `term`, `installment` — pre-origination, safe features
- `total_pymnt`, `total_rec_prncp`, `total_rec_int`, `last_pymnt_amnt`, `recoveries` — **all post-origination, all leak the target**
- `out_prncp`, `out_prncp_inv` — outstanding principal, partial leak
- `id`, `member_id` — ID-like columns

### 3.2 Locke configuration

The demo's `validate()` call should use:

```rust
use cjc_locke::api::{validate, ValidateOptions};
use cjc_locke::validation::{ImpossibleValueRule, ValidationConfig};

let opts = ValidateOptions {
    dataset_label: "lendingclub-2007-2018".into(),
    config: ValidationConfig {
        auto_detect_sentinels: true,
        // LC-specific sentinels seen in real data
        additional_sentinels: vec!["n/a".into(), "NONE".into()],
        near_constant_threshold: 0.99,
        high_cardinality_ratio: 0.5,
        duplicate_sample_limit: 5,
        collect_per_value_lineage: false,
    },
    impossible_rules: vec![
        ImpossibleValueRule::numeric_range("loan_amnt", 0.0, 1_000_000.0),
        ImpossibleValueRule::numeric_range("int_rate", 0.0, 50.0),
        ImpossibleValueRule::numeric_range("dti", 0.0, 100.0),
        ImpossibleValueRule::numeric_range("annual_inc", 0.0, 100_000_000.0),
    ],
    expected_schema: None,        // optional — can wire up later
    primary_key: Some("id".into()),
    null_masks: Default::default(),
};
let report = validate(&df, &opts);
```

Then run the leakage detector separately:

```rust
use cjc_locke::leakage::{detect_target_leakage, LeakageConfig};

let leakage_findings = detect_target_leakage(
    &df,
    "loan_status",
    &LeakageConfig::default(),
);
```

### 3.3 E-codes expected to fire

This is the verification checklist. If these don't fire, something is wrong:

| Code | Severity | Where | What |
|---|---|---|---|
| **E9008** | Info | `term` column | Leading-space sentinel (`" 36 months"`) |
| **E9008** | Info | `emp_length` column | `"n/a"` sentinel |
| **E9001** | Warning | ~60 columns | Co-borrower columns are NA on ~85% of rows |
| **E9060** | Error | `total_pymnt`, `total_rec_int`, `last_pymnt_amnt`, `recoveries`, `collection_recovery_fee` | AUC ≥ 0.95 against `loan_status` (`charged_off` vs `fully_paid`) |
| **E9061** | Warning | `out_prncp`, `total_pymnt_inv` | AUC in [0.85, 0.95] |
| **E9070** | Warning | `annual_inc_joint`, `dti_joint`, `verification_status_joint` etc. | Conditional missingness: present iff `application_type == "Joint App"` |
| **E9071** | Info | `loan_status` itself if used as multi-class | Minority class < 5% (some statuses are extremely rare) |
| **E9072** | Warning | `id`, `member_id` | distinct/n_rows ≥ 0.95 → ID-like |

### 3.4 Published baselines for cross-validation

After Locke removes the leakage features, the model should produce AUCs in the *honest* range:

| Baseline | Reported AUC | Notes |
|---|---|---|
| **FICO score alone** | ~0.70 | canonical reference for credit risk |
| **Tsai & Wu (2008)** | ~0.69 | classical credit-scoring on similar dataset |
| **Bao et al. (2019)** | ~0.74 | gradient boosting on LC subset |
| **LendingClub's own 10-K vintage tables** | charge-off rates ~14-18% per vintage 2010-2014 | aggregate-level cross-check |

**Pre-Locke baseline (deliberately broken)**: include `total_pymnt` as a feature → AUC > 0.99 → useless model. This is the "before" half of the demo's headline visualization.

### 3.5 Deliverable for Project A

A directory `demos/lendingclub/` containing:
- `README.md` — what this demo does, how to reproduce
- `src/main.rs` — the binary that loads the CSV, runs validate + leakage detection, emits a `LockeReport`
- `expected_findings.json` — committed file enumerating every E-code that must fire (acts as a regression gate)
- `cross_validate.md` — the table comparing honest-AUC range vs published baselines, with the Locke version's measured AUC
- `data/.gitkeep` — empty marker; the actual CSV is gitignored

### 3.6 Demo runtime budget

- CSV load + parse: 30-60s
- `validate()` call: 5-30s (depends on whether streaming or single-shot)
- `detect_target_leakage`: 10-30s
- Report serialization: <5s
- **Total: under 2 minutes per iteration.** Iterate freely.

---

## 4. Project B — NYC Taxi 2019-2020 H1 Pandemic Drift

### 4.1 Dataset acquisition

**Source**: NYC Taxi & Limousine Commission Trip Record Data
- Index: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Per-month Parquet files at predictable URLs:
  - `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-01.parquet`
  - `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-02.parquet`
  - ... through ...
  - `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-06.parquet`

**Exact files to download** (18 months, ~3.6 GB compressed total):

| Year | Months | Approx size each |
|---|---|---|
| 2019 | 01 02 03 04 05 06 07 08 09 10 11 12 | ~700 MB raw / ~500 MB Parquet each |
| 2020 | 01 02 03 04 05 06 | ~500 MB Parquet pre-pandemic, ~50 MB during lockdown |

**Tighter alternative if SSD is constrained**: just download Q4-2019 + Q1-Q2-2020 (9 months, ~3 GB) — still captures the pandemic drift event with one quarter of pre-pandemic context.

**Place at**: `demos/nyc_taxi/data/yellow_tripdata_YYYY-MM.parquet`

**Total bytes on disk after download**: ~3.6 GB for full 18-month set, ~2 GB for the tighter 9-month set. Either fits comfortably within the 5 GB SSD budget.

**Schema reference**: https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf

Key columns:
- `tpep_pickup_datetime`, `tpep_dropoff_datetime` — temporal anchors
- `passenger_count`, `trip_distance`, `fare_amount`, `total_amount`, `tip_amount` — numeric distribution targets
- `PULocationID`, `DOLocationID` — pickup/dropoff zone IDs (categorical)
- `payment_type` — categorical, sparse codes
- `congestion_surcharge` — added 2019-01, mostly zero pre-2020

### 4.2 Locke configuration

This is the streaming demo. The single-shot path is impossible at 80M rows on a 4 GB RAM budget.

```rust
use cjc_locke::streaming::{StreamingConfig, StreamingValidator};

let cfg = StreamingConfig {
    sample_cap: 1_000_000,           // 1M-row sample retained for KS/PSI checks
    distinct_cap: 100_000,           // bounded distinct tracking
    duplicate_hash_cap: 5_000_000,   // dup detection cap
};
let mut sv = StreamingValidator::new("nyc-taxi-2019-2020", cfg);

// For each month in sequence:
for month in months {
    let df_chunk = load_month_parquet(month)?;   // your own loader
    sv.ingest_chunk(&df_chunk)?;
}

let report = sv.into_report(None)?;
```

For the drift detection (the headline finding), run `compare()` on pairs of months:

```rust
use cjc_locke::drift::{compare, DriftConfig};

let dec_2019 = load_month_parquet("2019-12")?;
let mar_2020 = load_month_parquet("2020-03")?;
let drift = compare(&dec_2019, &mar_2020, &DriftConfig::default());
```

This is where E9039 should fire on `passenger_count`, `trip_distance`, `fare_amount`, `total_amount` — the pandemic shock is unmistakable in the distribution.

### 4.3 E-codes expected to fire

| Code | Severity | Where | What |
|---|---|---|---|
| **E9039** | Warning/Error | `passenger_count`, `trip_distance`, `fare_amount`, `total_amount`, `tip_amount` | Numeric drift Dec 2019 → Mar 2020 (KS-D > 0.2 for most; > 0.5 for trip count if you aggregate first) |
| **E9001** | Warning | `tpep_dropoff_datetime` on select 2019 files | Known TLC raw-data issue |
| **E9010** | Warning | `congestion_surcharge` on early-2019 files | Constant 0 before 2019-01 introduction was fully rolled out |
| **E9011** | Notice | `passenger_count` on 2020-04, 2020-05 | Near-constant at 1 during lockdown |
| **E9050-E9054** | Error | rows where `tpep_dropoff_datetime < tpep_pickup_datetime` | Temporal leakage — real, ~0.001% of records, varies by month |
| **E9072** | not expected | n/a | Zone IDs aren't unique per row, so no ID-like flag |

### 4.4 Published baselines for cross-validation

**The headline cross-check**: Locke's reported total-trip count per month must match TLC's published quarterly totals exactly (Kahan-summed should be bit-identical to TLC's accounting).

| Quarter | TLC published yellow-cab trip count (approx) | Locke must match |
|---|---|---|
| 2019-Q1 | ~85M | ✓ |
| 2019-Q4 | ~80M | ✓ |
| 2020-Q1 | ~57M (collapses March) | ✓ |
| 2020-Q2 | ~6.5M (lockdown low) | ✓ |

Source for the published totals: TLC's quarterly factbook, https://www.nyc.gov/site/tlc/about/factbook.page

**Drift detection target**: Locke's E9039 must fire on the Feb-2020 → Mar-2020 comparison, and the magnitude must match the known event. Trip count fell ~70% in March 2020 vs February. Fare-amount distribution shifted because the surviving trips were disproportionately longer-distance (people leaving the city).

**Academic cross-references**:
- Zheng et al. (2021) "Pandemic shock to NYC taxi demand"
- Erhardt et al. (2021) on congestion pricing and ride-share

### 4.5 Deliverable for Project B

A directory `demos/nyc_taxi/` containing:
- `README.md` — what this demo does, how to reproduce
- `src/main.rs` — binary that streams the 6-quarter Parquet set through `StreamingValidator`
- `download.sh` (or `download.ps1` for Windows) — script that fetches the 18 Parquet files
- `expected_drift.json` — committed: the expected E9039 firings month-pair-by-month-pair with thresholds
- `tlc_published_totals.csv` — committed: TLC's quarterly factbook trip counts that Locke must match
- `cross_validate.md` — comparison table: TLC published vs Locke-streamed totals, with the bit-identity claim
- `determinism_check.sh` — runs the demo twice, compares JSON output hashes byte-for-byte
- `data/.gitkeep` — empty marker; Parquet files are gitignored

### 4.6 Demo runtime budget

On a laptop (16 GB RAM, modern CPU, plug-in, fans free):
- Download (one-time, 3.6 GB): 5-15 min depending on bandwidth
- Streaming ingest of 18 months: 20-40 min sustained CPU
- Drift compare on pair: 2-5 min
- Report serialization: <30 s
- **Total: 30-60 min per iteration after download.** Plan one or two clean iterations per session, not ten.

### 4.7 Thermal observability sub-experiment

Because Project B is the only demo with long-enough runtime to observe thermal effects, the deliverable should include a small thermal log:

```rust
// pseudocode — emit a CSV with one row per chunk ingested:
// chunk_index, wall_clock_ms, ns_per_row, cpu_temp_c_if_available
```

`cpu_temp_c` can be left blank if reading thermal sensors is impractical on Windows; the wall-clock per-chunk trend alone shows throttling (ns/row increases as the laptop heats up).

This thermal log + the determinism check together cover all four axes:
- numerical rigor → cross-validation totals
- determinism → two-run hash comparison
- thermal → per-chunk wall-clock trend
- memory → bounded sample_cap proves out (4 GB RAM peak never crossed)

---

## 5. Suggested directory structure

```
demos/
  README.md                              # top-level entry point
  lendingclub/
    README.md
    Cargo.toml                           # depends on cjc-locke, cjc-data, cjc-repro
    src/
      main.rs                            # the demo binary
    expected_findings.json
    cross_validate.md
    data/
      .gitkeep                           # data file gitignored
  nyc_taxi/
    README.md
    Cargo.toml
    src/
      main.rs
    download.ps1                         # Windows download script
    expected_drift.json
    tlc_published_totals.csv
    cross_validate.md
    determinism_check.ps1
    data/
      .gitkeep
```

Adjust `.gitignore` to cover `demos/*/data/*` so nobody accidentally commits a 600 MB CSV.

---

## 6. Cross-cutting design notes

### 6.1 Determinism contract

Both demos must produce **byte-identical** JSON reports across runs on the same machine, and **byte-identical** across machines (Mac/Linux/Windows) given the same input bytes.

Mechanism: Locke already guarantees this via Kahan summation, BTreeMap-everywhere iteration, `total_cmp` ordering, and seeded RNG. The demos just need to *expose* it — run twice, hash the output, compare. The `determinism_check.ps1` script in Project B is the canonical way to demonstrate this.

### 6.2 Thermal control

Locke's perf optimizations (C2-C7) reduce wall-clock by ~30-40% at large N (1M+ rows). On a laptop, this maps directly to fewer minutes at maximum fan, fewer watt-hours drawn, lower CPU package temperature averaged over the run. The Project B thermal log is the only place this is *measured*; everywhere else it's implied.

### 6.3 Memory efficiency

Project A uses single-shot (fits in RAM, ~3-4 GB peak). Project B uses streaming (~500 MB working set against an 80M-row underlying dataset). The Project B demo should print working-set RSS periodically as proof.

### 6.4 Numerical rigor

For both demos, the cross-validation column in the deliverable spreadsheet is the proof. Locke matches a published number exactly; an alternative tool using float32 or non-Kahan reduction would not. The point is *visible* matching.

---

## 7. Open decisions for the next session

These were *not* settled in the planning phase. Pick one approach, document the choice in the demo's README.

1. **Parquet reader for Project B**: roll our own with `cjc-locke::parquet_reader` as a starting point, OR pull in a thin wrapper. Recommendation: extend `parquet_reader.rs` if it doesn't already handle multi-row-group sequential reads efficiently. Check first.
2. **CSV parser for Project A**: `cjc-data::csv` should handle this. Confirm before writing any custom code.
3. **AUC computation outside Locke**: Project A's cross-validation requires training a credit-risk model. Recommend `cjc-runtime` + a simple logistic regression for the post-Locke "honest model" run, OR delegate to Python via the `cjc-locke-py` wrapper for visualization-only purposes. Make a clean call and document it.
4. **Drift-event visualization**: Project B's headline is "Locke detected the pandemic on March 13, 2020." That's powerful as a visualization (timeline of E9039 firings). Vizor (`cjc-vizor`) can produce SVG charts deterministically — use it.
5. **Whether to commit data**: NO. Both demos must download data on first run. The cross-validation files (TLC published totals, expected_findings.json) are committed; the raw data is not.

---

## 8. Failure modes to watch for

These are likely points where the demo can break:

- **Kaggle download requires login**: the `wordsforthewise/lending-club` Kaggle dataset requires a Kaggle account + API token. The download script should detect missing creds and print a clear error pointing to `https://www.kaggle.com/docs/api`. Do NOT hardcode credentials.
- **TLC URL changes**: the `d37ci6vzurychx.cloudfront.net` CDN has been stable for years but is not guaranteed. The fallback is the official index page at `nyc.gov/site/tlc/about/tlc-trip-record-data.page` — script the download to follow whatever the index links to.
- **Parquet decoder mismatch**: if `cjc-data` doesn't decode TLC's specific Parquet flavor (dictionary-encoded INT64s, etc.), the streaming pipeline will silently produce wrong values. The first sanity check after loading the first month is to verify row count matches TLC's published total for that month exactly. If not, the loader is broken, not Locke.
- **`StreamingValidator::ingest_chunk` returns Err on schema mismatch**: the TLC schema changed slightly across months (column `airport_fee` added 2021, `congestion_surcharge` added 2019-01). For the 2019-2020 H1 window we should be schema-stable, but the demo must handle the case gracefully — log the problematic month, skip it, continue. Don't crash mid-sweep.
- **Thermal throttling mid-run can look like a Locke regression**: it's the laptop, not the code. The thermal log will confirm this. Don't revert Locke perf changes based on a laptop bench.

---

## 9. Definition of "demo complete"

The demo is *complete* when, for each project:

1. A fresh clone + `cargo build` succeeds
2. The data download script fetches the inputs (with manual credential setup steps documented)
3. `cargo run --release -p locke-demo-<project>` produces a `LockeReport` JSON
4. The cross-validation file shows Locke's number matching the published baseline (or, for the leakage demo, the E-code firings matching `expected_findings.json` exactly)
5. The `determinism_check` script reports two byte-identical hashes
6. The Project B thermal log shows the per-chunk wall-clock trend
7. All of the above runs to completion within the 60-min compute budget

The repository must contain enough material — the cross-validation tables, the expected findings file, the determinism checker — that an independent reader who has never run the demo can verify the claims by reading the committed artifacts alone.

---

## 10. What this handoff does not cover

- Visualization beyond Vizor SVG charts (no matplotlib, no Plotly, no JS dashboards)
- Performance optimizations beyond what's in `cjc-locke` HEAD
- Cross-platform shell scripts (Windows-first; Linux/Mac left as a follow-up)
- The MIMIC-IV, SEC EDGAR, ATLAS Higgs, and Common Crawl projects that were considered earlier — they were dropped as out-of-scope for laptop budget; revisit only if a desktop-class machine becomes available

---

## 11. Pointers to existing code that will help

- `tests/locke/scale_benchmark.rs` — the benchmark harness; reuse the `synthesize()` helper as a sanity check on the loader
- `crates/cjc-locke/src/streaming.rs` — `StreamingValidator` API; the `streaming_summaries()` method is what Project B uses for the per-month rolling stats
- `crates/cjc-locke/src/drift.rs` — `compare(&train, &test)` and `DriftConfig`
- `crates/cjc-locke/src/leakage.rs` — `detect_target_leakage`, `detect_id_like_columns`, `LeakageConfig`
- `crates/cjc-locke/src/validation.rs` — every E-code, with comments documenting what triggers each
- `docs/locke/PERF_C6_REVERT_NOTES.md` — if a cool-system bench shows C6 regressing on the demo, this is the revert recipe
- `crates/cjc-vizor/src/` — deterministic SVG plotting; use this for the drift-timeline chart in Project B

Good luck. Both demos are bounded, the published baselines are real, and the laptop budget is generous enough that you can iterate freely without running afoul of thermals or memory. Don't add external dependencies. Don't break determinism. The rest is just code.
