# LendingClub default-prediction Locke demo

A self-contained Locke audit of the LendingClub 2007-2018 accepted-loans
dataset. Demonstrates the *target leakage* story: features that look
like strong predictors but actually leak the outcome because they are
recorded *after* the loan completes.

This is Project A from the handoff at
[../../docs/locke/DEMO_PROJECTS_HANDOFF.md](../../docs/locke/DEMO_PROJECTS_HANDOFF.md).

## What this demo does

1. Loads the LendingClub CSV (gzip-decompressed in memory)
2. Derives a binary `target_default` column from the multi-class
   `loan_status`, dropping mid-life rows whose outcome is not yet known
3. Runs `cjc_locke::api::validate()` — single-shot path (RAM-fits)
4. Runs `cjc_locke::leakage::detect_target_leakage()` against
   `target_default`
5. Runs `cjc_locke::leakage::detect_id_like_columns()`
6. Emits a single merged `LockeReport` as deterministic JSON

The interesting output is the set of `E9060` (Error: |AUC| ≥ 0.95) and
`E9061` (Warning: |AUC| ≥ 0.85) findings — the columns Locke flags are
exactly the post-origination payment streams that would silently inflate
a credit-risk model's AUC from a realistic 0.70 to a nonsensical > 0.99.

## Prerequisites

Hardware: 4 GB RAM headroom (peak ~3-4 GB on the full dataset). `--max-rows`
subsamples for smaller machines.

The CSV is not committed — see [Data acquisition](#data-acquisition) below.

## Data acquisition

The canonical source is the Kaggle dataset
[`wordsforthewise/lending-club`](https://www.kaggle.com/datasets/wordsforthewise/lending-club).
The specific file is `accepted_2007_to_2018Q4.csv.gz` (~600 MB compressed,
~2.5 GB decompressed).

Pick one of:

### Option A — Kaggle CLI (recommended)

1. Get a Kaggle API token: <https://www.kaggle.com/docs/api>
   (Account → Settings → API → Create New Token → downloads `kaggle.json`)
2. Place `kaggle.json` at `~/.kaggle/kaggle.json` (Linux/Mac) or
   `%USERPROFILE%/.kaggle/kaggle.json` (Windows).
3. Run:
   ```powershell
   kaggle datasets download `
       -d wordsforthewise/lending-club `
       -f accepted_2007_to_2018Q4.csv.gz `
       -p demos/lendingclub/data/
   ```

### Option B — manual download

1. Visit <https://www.kaggle.com/datasets/wordsforthewise/lending-club>
2. Log in, download `accepted_2007_to_2018Q4.csv.gz`
3. Place at `demos/lendingclub/data/accepted_2007_to_2018Q4.csv.gz`

## Running

### Baseline run (built-in detectors only)

```powershell
cargo run --release -p lendingclub-demo -- `
    --input  demos/lendingclub/data/accepted_2007_to_2018Q4.csv.gz `
    --output demos/lendingclub/out/report.json
```

### With the `PostOriginationByName` custom detector (ADR-0041)

```powershell
cargo run --release -p lendingclub-demo -- `
    --input  demos/lendingclub/data/accepted_2007_to_2018Q4.csv.gz `
    --output demos/lendingclub/out/report_with_custom.json `
    --use-custom-detectors
```

Registers the demo's `PostOriginationByNameDetector` (in
[`src/lib.rs`](src/lib.rs)) via `ValidateOptions::custom_detectors`. The
detector emits E9500 findings for every column whose name matches a
known post-origination pattern (`total_*`, `last_pymnt_*`, `recoveries`,
etc.). The report's E9500 + E9061 columns together cover the broader
leakage set the analyst's domain triage would have hand-curated.

For a faster smoke run on a smaller machine:

```powershell
cargo run --release -p lendingclub-demo -- `
    --input  demos/lendingclub/data/accepted_2007_to_2018Q4.csv.gz `
    --output demos/lendingclub/out/report_500k.json `
    --max-rows 500000
```

Expected runtime on a modern laptop:

| `--max-rows` | Wall clock | Peak RSS |
| ------------ | ---------- | -------- |
| (full ~2.5M) | 30-90 s    | ~3-4 GB  |
| 500 K        | ~15 s      | ~1.5 GB  |
| 100 K        | ~5 s       | ~700 MB  |

## Verifying the result

```powershell
cargo test --release -p lendingclub-demo --test expected_findings
```

This compares `out/report.json` against
[expected_findings.json](expected_findings.json). The test passes when
every E-code listed in the fixture fires at least `min_count` times *and*
fires against every column listed in `fingerprint_columns`. See the
fixture file for the full list. Cross-references to credit-risk literature
live in [cross_validate.md](cross_validate.md).

## Reproducibility

Locke produces byte-identical JSON across runs over the same input bytes.
To verify:

```powershell
cargo run --release -p lendingclub-demo -- --input <csv.gz> --output run1.json
cargo run --release -p lendingclub-demo -- --input <csv.gz> --output run2.json
Get-FileHash run1.json -Algorithm SHA256
Get-FileHash run2.json -Algorithm SHA256
# hashes must match
```

## Files

| Path                                 | What                                                              |
| ------------------------------------ | ----------------------------------------------------------------- |
| `src/lib.rs`                         | CSV loader + binarization + audit composition + honest-model helpers |
| `src/main.rs`                        | CLI shim — runs Locke audit, emits report.json                    |
| `src/bin/honest_model.rs`            | Second binary — fits 3 logistic regressions, reports test AUCs    |
| `expected_findings.json`             | Required E-codes the regression gate enforces                     |
| `cross_validate.md`                  | Mapping from Locke's findings to credit-risk literature + measured AUC table |
| `tests/expected_findings.rs`         | Regression-gate integration test                                  |
| `data/`                              | Downloaded CSV (gitignored)                                       |
| `out/`                               | Emitted reports (gitignored)                                      |

## Honest-model evaluation

A second binary, `honest_model`, fits three logistic regressions and
reports test-set |AUC| for each, validating the cross_validate.md §3
claim. Run via:

```powershell
cargo run --release -p lendingclub-demo --bin honest_model -- `
    --input demos/lendingclub/data/accepted_2007_to_2018Q4.csv.gz `
    --sample-rows 200000 --seed 42
```

For the 4-way comparison (including the ADR-0041 custom detector
result), pass `--from-report` pointing at the report produced with
`--use-custom-detectors`:

```powershell
.\target\release\honest_model.exe `
    --input demos/lendingclub/data/accepted_2007_to_2018Q4.csv.gz `
    --sample-rows 200000 --seed 42 `
    --from-report demos/lendingclub/out/report_with_custom.json
```

Expected output (2026-06-01 measurement, see
[cross_validate.md §3](cross_validate.md#3-honest-model-auc-measurement)):

| Variant                 | \|AUC\| | Interpretation                                                |
| ----------------------- | ------- | ------------------------------------------------------------- |
| Pre-Locke (naive)       | 0.9993  | catastrophic overfitting via leakage                          |
| Locke-filtered          | 0.9995  | removing only Locke's E9061 flags is not sufficient           |
| **Locke + custom det.** | **0.7388** | **ADR-0041 custom detector closes the gap inside Locke**  |
| Domain-honest           | 0.7394  | matches Bao et al. (2019) AUC ≈ 0.74 — hand-curated baseline  |

Implementation uses `cjc-runtime::hypothesis::logistic_regression`
(IRLS), `cjc-runtime::ml::auc_roc`, and
`cjc-runtime::ml::train_test_split`. Wall: ~3-4 min on 200K subsample.
