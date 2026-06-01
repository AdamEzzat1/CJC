//! Honest-model AUC harness — cross_validate.md §3 follow-up.
//!
//! Trains three logistic regression models on a subsample of the
//! LendingClub binarized frame and reports test-set AUC for each:
//!
//! 1. **Pre-Locke (naive)** — every numeric column except the target and
//!    ID-likes. Expected to inflate AUC because post-origination signals
//!    leak in.
//! 2. **Locke-filtered** — naive set minus the 3 columns Locke flagged
//!    E9061 (`last_fico_range_high/low`, `total_rec_prncp`). Tests
//!    whether trusting only Locke's flags is enough to recover an
//!    honest model — predicted to still be inflated because other
//!    post-origination columns (`total_pymnt`, `out_prncp`, etc.) are
//!    sub-threshold for Locke's |AUC| heuristic.
//! 3. **Domain-honest** — naive set minus the full handoff §3.3
//!    post-origination column list. Expected to land in the
//!    FICO/Tsai-Wu/Bao literature band (0.65 – 0.75).
//!
//! The three-way comparison is the demo's punchline: it shows where
//! Locke's flags catch the leakage and where they don't.
//!
//! Usage:
//! ```text
//! honest_model --input demos/lendingclub/data/accepted_2007_to_2018Q4.csv.gz \
//!              [--sample-rows 200000] [--seed 42]
//! ```

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use cjc_runtime::hypothesis::{logistic_regression, LogisticResult};
use cjc_runtime::ml::{auc_roc, train_test_split};

use lendingclub_demo::{
    apply_column_transform, binarize_loan_status, drop_useless_columns, extract_target_bool,
    fit_column_stats, flatten_subset_row_major, load_csv_gz, score_logistic,
    select_numeric_columns, ALWAYS_EXCLUDED_COLUMNS, DOMAIN_POST_ORIGINATION_COLUMNS,
    LOCKE_E9061_COLUMNS,
};

struct Args {
    input: PathBuf,
    sample_rows: usize,
    seed: u64,
    test_fraction: f64,
}

fn parse_args() -> Result<Args, String> {
    let mut input: Option<PathBuf> = None;
    let mut sample_rows: usize = 200_000;
    let mut seed: u64 = 42;
    let mut test_fraction: f64 = 0.3;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--input" | "-i" => {
                input = Some(PathBuf::from(it.next().ok_or("--input needs a value")?));
            }
            "--sample-rows" => {
                sample_rows = it
                    .next()
                    .ok_or("--sample-rows needs a value")?
                    .parse()
                    .map_err(|e| format!("--sample-rows: {}", e))?;
            }
            "--seed" => {
                seed = it
                    .next()
                    .ok_or("--seed needs a value")?
                    .parse()
                    .map_err(|e| format!("--seed: {}", e))?;
            }
            "--test-fraction" => {
                test_fraction = it
                    .next()
                    .ok_or("--test-fraction needs a value")?
                    .parse()
                    .map_err(|e| format!("--test-fraction: {}", e))?;
            }
            "-h" | "--help" => {
                eprintln!("honest_model --input <path.csv.gz> [--sample-rows N] [--seed S] [--test-fraction F]");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {}", other)),
        }
    }
    Ok(Args {
        input: input.ok_or("--input required")?,
        sample_rows,
        seed,
        test_fraction,
    })
}

/// One full train-and-score iteration for a named feature set.
fn run_variant(
    label: &str,
    excluded: Vec<&str>,
    df: &cjc_data::DataFrame,
    sample_indices: &[usize],
    seed: u64,
    test_fraction: f64,
) -> Result<VariantResult, String> {
    let t0 = Instant::now();

    let (raw_names, raw_columns) = select_numeric_columns(df, &excluded);
    let (names, columns) = drop_useless_columns(raw_names, raw_columns);
    let p = names.len();
    if p == 0 {
        return Err(format!("variant {}: no usable feature columns", label));
    }

    // Split the SAMPLE indices into train/test. Determinism: seeded RNG.
    let (train_local, test_local) = train_test_split(sample_indices.len(), test_fraction, seed);
    let train_indices: Vec<usize> = train_local.iter().map(|&i| sample_indices[i]).collect();
    let test_indices: Vec<usize> = test_local.iter().map(|&i| sample_indices[i]).collect();

    // Per-column fit on train; apply to both train and test. This is the
    // critical determinism + no-leakage guarantee.
    let mut train_columns_z: Vec<Vec<f64>> = Vec::with_capacity(p);
    let mut test_columns_z: Vec<Vec<f64>> = Vec::with_capacity(p);
    for col in &columns {
        let train_slice: Vec<f64> = train_indices.iter().map(|&i| col[i]).collect();
        let (mean, std) = fit_column_stats(&train_slice);
        let train_z = apply_column_transform(&train_slice, mean, std);
        let test_slice: Vec<f64> = test_indices.iter().map(|&i| col[i]).collect();
        let test_z = apply_column_transform(&test_slice, mean, std);
        train_columns_z.push(train_z);
        test_columns_z.push(test_z);
    }

    // Flatten row-major. For train we want indices = 0..n_train against
    // the already-subsetted train_columns_z, so build a 0..n_train index list.
    let n_train = train_indices.len();
    let n_test = test_indices.len();
    let train_row_idx: Vec<usize> = (0..n_train).collect();
    let test_row_idx: Vec<usize> = (0..n_test).collect();
    let x_train_flat = flatten_subset_row_major(&train_columns_z, &train_row_idx);
    let x_test_flat = flatten_subset_row_major(&test_columns_z, &test_row_idx);

    // Targets.
    let target_full = extract_target_bool(df)?;
    let y_train: Vec<f64> = train_indices
        .iter()
        .map(|&i| if target_full[i] { 1.0 } else { 0.0 })
        .collect();
    let y_test_bool: Vec<bool> = test_indices.iter().map(|&i| target_full[i]).collect();

    // Train. Returns an Err if IRLS fails.
    let logreg_t = Instant::now();
    let res: LogisticResult = logistic_regression(&x_train_flat, &y_train, n_train, p)
        .map_err(|e| format!("variant {} IRLS failed: {}", label, e))?;
    let logreg_secs = logreg_t.elapsed().as_secs_f64();

    // Score test set.
    let eta = score_logistic(&res.coefficients, &x_test_flat, n_test, p);
    let auc = auc_roc(&eta, &y_test_bool).map_err(|e| format!("variant {} auc_roc: {}", label, e))?;
    let abs_auc = auc.max(1.0 - auc);

    Ok(VariantResult {
        label: label.into(),
        n_features: p,
        feature_names: names,
        n_train,
        n_test,
        iterations: res.iterations,
        wall_secs: t0.elapsed().as_secs_f64(),
        logreg_secs,
        auc,
        abs_auc,
    })
}

#[allow(dead_code)]
struct VariantResult {
    label: String,
    n_features: usize,
    feature_names: Vec<String>,
    n_train: usize,
    n_test: usize,
    iterations: usize,
    wall_secs: f64,
    logreg_secs: f64,
    auc: f64,
    abs_auc: f64,
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("argument error: {}", e);
            return ExitCode::from(2);
        }
    };

    eprintln!("[load] {} (max {} rows)", args.input.display(), args.sample_rows.saturating_mul(2));
    let t_load = Instant::now();
    // Load a generous superset of sample_rows so binarization filtering
    // (~58% terminal-outcome) still leaves us with ~sample_rows.
    let load_cap = (args.sample_rows as f64 / 0.5) as usize;
    let (raw_df, n_bytes) = match load_csv_gz(&args.input, Some(load_cap)) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("load error: {}", e);
            return ExitCode::FAILURE;
        }
    };
    eprintln!(
        "[load] {:.1} MB decompressed, {} rows × {} cols, {:.1}s",
        n_bytes as f64 / 1_048_576.0,
        raw_df.nrows(),
        raw_df.ncols(),
        t_load.elapsed().as_secs_f64()
    );

    let df = match binarize_loan_status(raw_df) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("binarize error: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let n_kept = df.nrows();
    let actual_sample = args.sample_rows.min(n_kept);
    eprintln!("[derive] kept {} terminal-outcome rows", n_kept);
    eprintln!("[sample] using first {} rows as the modeling pool", actual_sample);

    // Deterministic modeling pool: indices 0..actual_sample. Because the
    // load step already capped row count (and the CSV's row order is what
    // the file gives us), a fixed seed + fixed input produces a fixed pool.
    let sample_indices: Vec<usize> = (0..actual_sample).collect();

    // Run all three variants. Failures don't abort — each variant is
    // independent and even a partial result is useful.
    let mut results: Vec<VariantResult> = Vec::new();
    for (label, excludes) in [
        ("pre-Locke (naive)", vec![]),
        ("Locke-filtered", LOCKE_E9061_COLUMNS.to_vec()),
        ("domain-honest", DOMAIN_POST_ORIGINATION_COLUMNS.to_vec()),
    ] {
        let mut full_excludes: Vec<&str> = ALWAYS_EXCLUDED_COLUMNS.to_vec();
        full_excludes.extend(excludes);
        eprintln!("[variant] {} — fitting...", label);
        match run_variant(
            label,
            full_excludes,
            &df,
            &sample_indices,
            args.seed,
            args.test_fraction,
        ) {
            Ok(r) => {
                eprintln!(
                    "[variant] {} → AUC = {:.4} (|AUC| = {:.4}), p = {}, iter = {}, train wall = {:.1}s",
                    r.label, r.auc, r.abs_auc, r.n_features, r.iterations, r.logreg_secs
                );
                results.push(r);
            }
            Err(e) => {
                eprintln!("[variant] {} failed: {}", label, e);
            }
        }
    }

    eprintln!();
    eprintln!("=== Summary ===");
    eprintln!(
        "{:<22} {:>10} {:>10} {:>10} {:>10}",
        "variant", "|AUC|", "n_train", "n_test", "p"
    );
    for r in &results {
        eprintln!(
            "{:<22} {:>10.4} {:>10} {:>10} {:>10}",
            r.label, r.abs_auc, r.n_train, r.n_test, r.n_features
        );
    }

    // Sanity: pre-Locke should clearly beat domain-honest. If not, something
    // is upstream-broken (data quality, sampling, etc.) and we surface it.
    if results.len() == 3 {
        let pre = results[0].abs_auc;
        let domain = results[2].abs_auc;
        if pre <= domain {
            eprintln!();
            eprintln!(
                "WARNING: pre-Locke AUC ({:.4}) is not greater than domain-honest AUC ({:.4}).",
                pre, domain
            );
            eprintln!("This is unexpected. Investigate before drawing conclusions.");
        }
    }

    ExitCode::SUCCESS
}
