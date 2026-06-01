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

```powershell
cargo run --release -p lendingclub-demo -- `
    --input  demos/lendingclub/data/accepted_2007_to_2018Q4.csv.gz `
    --output demos/lendingclub/out/report.json
```

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
| `src/lib.rs`                         | CSV loader + binarization + audit composition                     |
| `src/main.rs`                        | CLI shim                                                          |
| `expected_findings.json`             | Required E-codes the regression gate enforces                     |
| `cross_validate.md`                  | Mapping from Locke's findings to credit-risk literature           |
| `tests/expected_findings.rs`         | Regression-gate integration test                                  |
| `data/`                              | Downloaded CSV (gitignored)                                       |
| `out/`                               | Emitted reports (gitignored)                                      |

## Honest-model evaluation (optional follow-up)

This demo does not itself train a credit-risk model. To complete the
"before vs after Locke" comparison in
[cross_validate.md](cross_validate.md), you need a separate scoring
script that:

1. Splits the post-binarization frame 70/30 (train/test)
2. Fits one logistic regression *with* the E9060-flagged columns, AUC on
   test → should be > 0.99 (useless)
3. Fits a second logistic regression *without* the E9060-flagged columns,
   AUC on test → should fall into the 0.70 - 0.75 band cited by Tsai &
   Wu (2008) and Bao et al. (2019)

`cjc-runtime` exposes the necessary primitives (logistic regression via
GradGraph, ROC curve via Kahan-summed binning). Wiring is left as an
exercise to keep the demo's primary deliverable scoped to the Locke
audit itself.
