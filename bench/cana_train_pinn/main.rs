//! PINN v2 §2.1 — data-sanity pass over the Phase-A6 ablation corpus.
//!
//! Read-only audit of `bench_results/cana_ablation/profiles.cpdb`
//! answering, BEFORE any training code is written (handoff §2.1):
//!
//! 1. How much variance do the labels really have — per config and per
//!    program family?
//! 2. Where does the `score` signal (baseline-relative modeled energy)
//!    actually live? Rows whose plan equals the baseline plan score
//!    exactly 1.0 by construction, so the trainable signal is confined
//!    to plan-divergent rows.
//! 3. Are the recorded pressure labels per-row or per-program? The
//!    recorded predictor is built from ONE instrumented run of the
//!    unoptimized program, so every `*_rec` row of a program should
//!    carry identical pressure labels — verified bitwise, not assumed.
//! 4. Does a LINEAR model already saturate the feature→label signal?
//!    If OLS held-out R² is already high, a 2-layer MLP adds nothing
//!    and v2 should stay linear (re-fit coefficients, skip the MLP).
//!
//! Determinism: Kahan accumulation for every FP reduction, BTreeMap /
//! BTreeSet everywhere, `f64::total_cmp` for ordering, FNV-1a (via
//! `cjc_cana::hash::hash_bytes`) for the program-level train/test
//! split. No RNG, no wall-clock in any reported number.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

use cjc_cana::hash::hash_bytes;
use cjc_cana_compress::profile_db::{read_all, CompilationProfile};
use cjc_repro::KahanAccumulatorF64;

const DB_PATH: &str = "bench_results/cana_ablation/profiles.cpdb";

/// Score == 1.0 tolerance, matching the ablation harness's "≠baseline"
/// column (`bench/cana_ablation/main.rs`).
const SCORE_DIVERGENCE_EPS: f64 = 1e-9;

/// Programs whose FNV-1a(name) % SPLIT_MOD == 0 form the held-out test
/// set (~20% of programs; all 11 rows of a program land on one side so
/// no program leaks across the split).
const SPLIT_MOD: u64 = 5;

// =============================================================================
// Deterministic statistics helpers (Kahan two-pass)
// =============================================================================

#[derive(Debug, Clone, Copy)]
struct Stats {
    n: usize,
    min: f64,
    max: f64,
    mean: f64,
    std: f64,
}

fn stats(values: &[f64]) -> Stats {
    if values.is_empty() {
        return Stats {
            n: 0,
            min: f64::NAN,
            max: f64::NAN,
            mean: f64::NAN,
            std: f64::NAN,
        };
    }
    let mut sum = KahanAccumulatorF64::new();
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in values {
        sum.add(v);
        min = min.min(v);
        max = max.max(v);
    }
    let mean = sum.finalize() / values.len() as f64;
    let mut sq = KahanAccumulatorF64::new();
    for &v in values {
        let d = v - mean;
        sq.add(d * d);
    }
    let std = (sq.finalize() / values.len() as f64).sqrt();
    Stats {
        n: values.len(),
        min,
        max,
        mean,
        std,
    }
}

/// Pearson correlation, two-pass with Kahan accumulation. Returns NAN
/// when either side has zero variance.
fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len());
    if xs.len() < 2 {
        return f64::NAN;
    }
    let sx = stats(xs);
    let sy = stats(ys);
    if sx.std == 0.0 || sy.std == 0.0 {
        return f64::NAN;
    }
    let mut cov = KahanAccumulatorF64::new();
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        cov.add((x - sx.mean) * (y - sy.mean));
    }
    cov.finalize() / (xs.len() as f64 * sx.std * sy.std)
}

// =============================================================================
// Deterministic OLS (ridge λ=1e-8) via normal equations + Gaussian elimination
// =============================================================================

/// Solve `A x = b` by Gaussian elimination with partial pivoting.
/// Deterministic: pivot choice uses `f64::total_cmp` on |a|.
fn gauss_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
            .unwrap();
        if a[pivot][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in (col + 1)..n {
            let f = a[row][col] / a[col][col];
            // Plain loop (not iterator zip) keeps the subtraction order
            // explicit and fixed.
            for k in col..n {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for col in (0..n).rev() {
        let mut acc = KahanAccumulatorF64::new();
        acc.add(b[col]);
        for k in (col + 1)..n {
            acc.add(-a[col][k] * x[k]);
        }
        x[col] = acc.finalize() / a[col][col];
    }
    Some(x)
}

/// A fitted standardized ridge model: train-set column statistics +
/// coefficients (intercept LAST in `beta`).
struct FittedLinear {
    col_mean: Vec<f64>,
    col_std: Vec<f64>,
    beta: Vec<f64>,
}

impl FittedLinear {
    fn predict(&self, x: &[f64]) -> f64 {
        let d = self.col_mean.len();
        let mut acc = KahanAccumulatorF64::new();
        for c in 0..d {
            let z = (x[c] - self.col_mean[c]) / self.col_std[c];
            acc.add(z * self.beta[c]);
        }
        acc.add(self.beta[d]); // intercept
        acc.finalize()
    }
}

/// Fit ridge-OLS (λ=1e-8) on the rows selected by `train_idx`.
///
/// Columns are z-scored with TRAIN-set statistics (zero-variance
/// columns collapse to 0), then an unscaled intercept column is
/// appended. The ridge keeps the normal equations solvable under the
/// collinearity the pass-count / config-one-hot features inevitably
/// have.
fn fit_ridge(rows: &[Vec<f64>], targets: &[f64], train_idx: &[usize]) -> Option<FittedLinear> {
    let d = rows.first()?.len();
    if train_idx.len() <= d + 1 {
        return None;
    }
    let mut col_mean = vec![0.0f64; d];
    let mut col_std = vec![0.0f64; d];
    for c in 0..d {
        let vals: Vec<f64> = train_idx.iter().map(|&i| rows[i][c]).collect();
        let s = stats(&vals);
        col_mean[c] = s.mean;
        col_std[c] = if s.std > 0.0 { s.std } else { 1.0 };
    }
    let design = |i: usize| -> Vec<f64> {
        let mut x: Vec<f64> = (0..d)
            .map(|c| (rows[i][c] - col_mean[c]) / col_std[c])
            .collect();
        x.push(1.0); // intercept
        x
    };

    let dd = d + 1;
    let mut ata = vec![vec![KahanAccumulatorF64::new(); dd]; dd];
    let mut atb = vec![KahanAccumulatorF64::new(); dd];
    for &i in train_idx {
        let x = design(i);
        for r in 0..dd {
            for c in r..dd {
                ata[r][c].add(x[r] * x[c]);
            }
            atb[r].add(x[r] * targets[i]);
        }
    }
    let mut a = vec![vec![0.0f64; dd]; dd];
    let mut b = vec![0.0f64; dd];
    for r in 0..dd {
        for c in r..dd {
            let v = ata[r][c].finalize();
            a[r][c] = v;
            a[c][r] = v;
        }
        a[r][r] += 1e-8; // ridge
        b[r] = atb[r].finalize();
    }
    let beta = gauss_solve(a, b)?;
    Some(FittedLinear {
        col_mean,
        col_std,
        beta,
    })
}

/// R² of `model` over the subset `idx` (subset's own mean for SS_tot).
fn r2_over(model: &FittedLinear, rows: &[Vec<f64>], targets: &[f64], idx: &[usize]) -> f64 {
    if idx.is_empty() {
        return f64::NAN;
    }
    let ys: Vec<f64> = idx.iter().map(|&i| targets[i]).collect();
    let s = stats(&ys);
    let mut ss_res = KahanAccumulatorF64::new();
    let mut ss_tot = KahanAccumulatorF64::new();
    for &i in idx {
        let e = targets[i] - model.predict(&rows[i]);
        ss_res.add(e * e);
        let dm = targets[i] - s.mean;
        ss_tot.add(dm * dm);
    }
    let tot = ss_tot.finalize();
    if tot == 0.0 {
        return f64::NAN;
    }
    1.0 - ss_res.finalize() / tot
}

struct OlsReport {
    n_train: usize,
    n_test: usize,
    r2_train: f64,
    r2_test: f64,
}

/// Fit on the non-test rows, report in-sample and held-out R².
fn ols(rows: &[Vec<f64>], targets: &[f64], is_test: &[bool]) -> Option<OlsReport> {
    assert_eq!(rows.len(), targets.len());
    assert_eq!(rows.len(), is_test.len());
    let train_idx: Vec<usize> = (0..rows.len()).filter(|&i| !is_test[i]).collect();
    let test_idx: Vec<usize> = (0..rows.len()).filter(|&i| is_test[i]).collect();
    let model = fit_ridge(rows, targets, &train_idx)?;
    Some(OlsReport {
        n_train: train_idx.len(),
        n_test: test_idx.len(),
        r2_train: r2_over(&model, rows, targets, &train_idx),
        r2_test: r2_over(&model, rows, targets, &test_idx),
    })
}

// =============================================================================
// Corpus views
// =============================================================================

fn family(program_name: &str) -> &'static str {
    if program_name.starts_with("grad_") {
        "grad"
    } else if program_name.starts_with("train_") {
        "train"
    } else if program_name.starts_with("tensor_") {
        "tensor"
    } else {
        "static"
    }
}

fn is_test_program(program_name: &str) -> bool {
    hash_bytes(program_name.as_bytes()) % SPLIT_MOD == 0
}

/// Frozen holdout (Phase A item 7): `holdout_`-prefixed programs are
/// excluded from BOTH the train set and the FNV-split test set — they
/// were never part of any training or tuning decision and are
/// evaluated only as a separate cohort at promotion gates. The FNV
/// test split erodes as heads get tuned against it; this set does not.
fn is_holdout_program(program_name: &str) -> bool {
    program_name.starts_with("holdout_")
}

fn log1p_u64(v: u64) -> f64 {
    (v as f64).ln_1p()
}

/// Static FP-op density: the schema-v2 analog of the runtime
/// `thermal_intensity` (FP ops over total ops).
fn fp_density(row: &CompilationProfile) -> f64 {
    if row.estimated_flops == 0 {
        return 0.0;
    }
    row.estimated_float_ops as f64 / row.estimated_flops as f64
}

/// Total occurrences of each pass name across a row's per-function plan.
fn pass_counts(row: &CompilationProfile, vocab: &[String]) -> Vec<f64> {
    let mut counts: BTreeMap<&str, f64> = BTreeMap::new();
    for (_, passes) in &row.pass_sequence {
        for p in passes {
            *counts.entry(p.as_str()).or_insert(0.0) += 1.0;
        }
    }
    vocab
        .iter()
        .map(|p| counts.get(p.as_str()).copied().unwrap_or(0.0))
        .collect()
}

/// Feature vector for the score-prediction task. `with_structural`
/// additionally exposes the post-plan node count (computable at compile
/// time by statically applying the plan — cheap, deterministic).
fn score_features(
    row: &CompilationProfile,
    vocab: &[String],
    configs: &[String],
    with_structural: bool,
) -> Vec<f64> {
    let mut x = vec![
        log1p_u64(row.estimated_flops),
        log1p_u64(row.estimated_bytes_read),
        log1p_u64(row.estimated_bytes_written),
        log1p_u64(row.estimated_alloc_bytes),
        log1p_u64(row.estimated_working_set),
        log1p_u64(row.estimated_float_ops),
        fp_density(row),
        log1p_u64(row.mir_nodes_before),
        row.recommended_count as f64,
        row.dropped_count as f64,
    ];
    x.extend(pass_counts(row, vocab));
    for c in configs.iter().filter(|c| c.as_str() != "baseline") {
        x.push(if &row.config_id == c { 1.0 } else { 0.0 });
    }
    if with_structural {
        x.push(log1p_u64(row.mir_nodes_after));
        let ratio = if row.mir_nodes_before > 0 {
            row.mir_nodes_after as f64 / row.mir_nodes_before as f64
        } else {
            1.0
        };
        x.push(ratio);
    }
    x
}

// =============================================================================
// Report sections
// =============================================================================

fn section(title: &str) {
    println!("\n=== {title} ===");
}

fn print_stats_line(label: &str, s: &Stats) {
    println!(
        "  {label:<28} n={:<5} min={:>10.4} mean={:>10.4} max={:>10.4} std={:>10.4}",
        s.n, s.min, s.mean, s.max, s.std
    );
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "sanity".into());
    match mode.as_str() {
        "sanity" => run_sanity(),
        "train" => run_train(),
        "shadow" => run_shadow(),
        "sanity-energy" => run_sanity_energy(),
        "train-energy" => run_train_energy(),
        "shadow-energy" => run_shadow_energy(),
        other => {
            eprintln!(
                "unknown mode '{other}' — available: sanity, train, shadow, sanity-energy, train-energy, shadow-energy"
            );
            std::process::exit(2);
        }
    }
}

// =============================================================================
// Shared corpus views for train/shadow
// =============================================================================

/// Default output path for the trained bundle.
const BUNDLE_PATH: &str = "bench_results/cana_train_pinn/pinn_thermal_v2.cpb";

/// Lift a corpus row's workload sums into a `PhysicalCostQuery` so the
/// trainer uses the EXACT feature basis the predictor will
/// (`cjc_cana::pinn_thermal_v2::features_from_query`) — any drift
/// between the two would silently invalidate the weights.
fn query_from_row(r: &CompilationProfile) -> cjc_cana::physical_cost::PhysicalCostQuery<'_> {
    cjc_cana::physical_cost::PhysicalCostQuery {
        function_name: &r.program_name,
        strategy_id: "row",
        flops_estimate: r.estimated_flops,
        bytes_read_estimate: r.estimated_bytes_read,
        bytes_written_estimate: r.estimated_bytes_written,
        allocation_bytes_estimate: r.estimated_alloc_bytes,
        working_set_bytes_estimate: r.estimated_working_set,
        thread_count: 1,
        batch_size: 1,
        compression_overhead_bytes: 0,
        float_ops_estimate: r.estimated_float_ops,
    }
}

/// Program-level dataset: one row per program (taken from its first
/// `*_rec` row — pressure labels are bitwise-identical across recorded
/// configs, verified by the sanity pass), v2 feature basis, recorded
/// thermal label, deterministic split.
struct ProgramDataset {
    names: Vec<String>,
    feats: Vec<Vec<f64>>,
    labels: Vec<f64>,
    /// v1 closed-form predictions, for the shadow comparison.
    v1_preds: Vec<f64>,
    train_idx: Vec<usize>,
    test_idx: Vec<usize>,
    /// Frozen holdout (`holdout_` prefix) — never trained or tuned on;
    /// reported as its own cohort at promotion gates only.
    holdout_idx: Vec<usize>,
}

fn load_program_dataset() -> ProgramDataset {
    let rows = read_all(&PathBuf::from(DB_PATH)).unwrap_or_else(|e| {
        panic!("cannot read corpus at {DB_PATH} — run `cargo run --release -p cana-ablation`: {e}")
    });
    let mut by_prog: BTreeMap<&str, &CompilationProfile> = BTreeMap::new();
    for r in &rows {
        if r.config_id.contains("_rec") {
            by_prog.entry(r.program_name.as_str()).or_insert(r);
        }
    }
    let mut names = Vec::new();
    let mut feats = Vec::new();
    let mut labels = Vec::new();
    let mut v1_preds = Vec::new();
    let mut train_idx = Vec::new();
    let mut test_idx = Vec::new();
    let mut holdout_idx = Vec::new();
    for (i, (name, r)) in by_prog.iter().enumerate() {
        names.push(name.to_string());
        let q = query_from_row(r);
        feats.push(cjc_cana::pinn_thermal_v2::features_from_query(&q).to_vec());
        labels.push(r.nss_predicted_thermal_max);
        v1_preds.push(r.pinn_predicted_thermal_max);
        if is_holdout_program(name) {
            holdout_idx.push(i);
        } else if is_test_program(name) {
            test_idx.push(i);
        } else {
            train_idx.push(i);
        }
    }
    ProgramDataset {
        names,
        feats,
        labels,
        v1_preds,
        train_idx,
        test_idx,
        holdout_idx,
    }
}

fn mae_over(preds: &[f64], labels: &[f64], idx: &[usize]) -> f64 {
    if idx.is_empty() {
        return f64::NAN;
    }
    let mut acc = KahanAccumulatorF64::new();
    for &i in idx {
        acc.add((preds[i] - labels[i]).abs());
    }
    acc.finalize() / idx.len() as f64
}

// =============================================================================
// Mode: train — fit the v2 thermal head, persist as CPB0 (§2.3 + §2.4)
// =============================================================================

fn run_train() {
    use cjc_cana::pinn_thermal_v2::{
        PinnThermalV2, PINN_V2_FEATURE_COUNT, PINN_V2_MODEL_ID, PINN_V2_MODEL_VERSION,
    };
    use cjc_cana_compress::pinn_bundle::{write_bundle, PinnBundle};

    let ds = load_program_dataset();
    println!("=== PINN v2 thermal-head training (deterministic ridge OLS) ===");
    println!(
        "  programs: {} ({} train / {} test by FNV%{SPLIT_MOD} / {} holdout-frozen, untouched)",
        ds.names.len(),
        ds.train_idx.len(),
        ds.test_idx.len(),
        ds.holdout_idx.len()
    );

    let model = fit_ridge(&ds.feats, &ds.labels, &ds.train_idx)
        .expect("fit failed — corpus too small for the 7-feature basis?");

    // -- Fit quality -----------------------------------------------------------
    let r2_train = r2_over(&model, &ds.feats, &ds.labels, &ds.train_idx);
    let r2_test = r2_over(&model, &ds.feats, &ds.labels, &ds.test_idx);
    let preds: Vec<f64> = ds.feats.iter().map(|x| model.predict(x)).collect();
    let mae_train = mae_over(&preds, &ds.labels, &ds.train_idx);
    let mae_test = mae_over(&preds, &ds.labels, &ds.test_idx);
    println!("  R²(train) = {r2_train:.4}   R²(test) = {r2_test:.4}");
    println!("  MAE(train) = {mae_train:.4}  MAE(test) = {mae_test:.4}");

    // -- Physics post-fit checks (the §4.3 residuals adapted to a
    //    closed-form fit: verified properties, not penalty terms) -----------
    let density_coeff = model.beta[6];
    println!("  physics: standardized FP-density coefficient = {density_coeff:+.4} (must be > 0: more FP work rate → more heat)");
    let zero_q = cjc_cana::physical_cost::PhysicalCostQuery {
        function_name: "zero",
        strategy_id: "row",
        flops_estimate: 0,
        bytes_read_estimate: 0,
        bytes_written_estimate: 0,
        allocation_bytes_estimate: 0,
        working_set_bytes_estimate: 0,
        thread_count: 1,
        batch_size: 1,
        compression_overhead_bytes: 0,
        float_ops_estimate: 0,
    };
    let zero_feats = cjc_cana::pinn_thermal_v2::features_from_query(&zero_q);
    let zero_pred_raw = model.predict(&zero_feats);
    println!("  physics: zero-workload raw prediction = {zero_pred_raw:+.4} (clamped to ≥ 0 at predict time)");
    let out_of_range = preds.iter().filter(|p| **p < 0.0 || **p > 1.0).count();
    println!(
        "  physics: pre-clamp out-of-[0,1] predictions = {}/{} (clamp handles these)",
        out_of_range,
        preds.len()
    );
    assert!(
        density_coeff > 0.0,
        "physics violation: thermal must increase with FP density"
    );

    // -- Persist ----------------------------------------------------------------
    let mut feature_means = [0.0f64; PINN_V2_FEATURE_COUNT];
    let mut feature_stds = [1.0f64; PINN_V2_FEATURE_COUNT];
    let mut coefficients = [0.0f64; PINN_V2_FEATURE_COUNT];
    feature_means.copy_from_slice(&model.col_mean);
    feature_stds.copy_from_slice(&model.col_std);
    coefficients.copy_from_slice(&model.beta[..PINN_V2_FEATURE_COUNT]);
    let head = PinnThermalV2 {
        feature_means,
        feature_stds,
        coefficients,
        intercept: model.beta[PINN_V2_FEATURE_COUNT],
    };
    assert!(head.is_valid(), "trained head must validate");
    let bundle = PinnBundle {
        model_id: PINN_V2_MODEL_ID.to_string(),
        model_version: PINN_V2_MODEL_VERSION,
        head,
    };
    let path = PathBuf::from(BUNDLE_PATH);
    fs::create_dir_all(path.parent().unwrap()).expect("create bundle dir");
    write_bundle(&path, &bundle).expect("write bundle");
    println!(
        "  bundle written: {} ({} / v{})",
        path.display(),
        bundle.model_id,
        bundle.model_version
    );

    // Double-write determinism canary.
    let first = fs::read(&path).unwrap();
    write_bundle(&path, &bundle).expect("re-write bundle");
    assert_eq!(
        first,
        fs::read(&path).unwrap(),
        "bundle must be byte-stable"
    );
    println!("  bundle double-write: byte-identical");
    println!(
        "\nNext: `cargo run --release -p cana-train-pinn -- shadow` for the §2.5 promotion gate."
    );
}

// =============================================================================
// Mode: shadow — v1 closed form vs v2 trained head against measured labels (§2.5)
// =============================================================================

fn run_shadow() {
    use cjc_cana_compress::pinn_bundle::read_bundle;

    let ds = load_program_dataset();
    let bundle = read_bundle(&PathBuf::from(BUNDLE_PATH)).unwrap_or_else(|e| {
        panic!("cannot read bundle at {BUNDLE_PATH} — run `-- train` first: {e}")
    });
    println!(
        "=== PINN v2 shadow mode — v1 closed form vs trained head ({} v{}) ===",
        bundle.model_id, bundle.model_version
    );
    println!("  ground truth: recorded per-program thermal labels (Option-B instrumented runs)");

    // Predict through the REAL head API (clamping included), not the
    // raw fit — shadow mode must measure what the compiler would see.
    let rows = read_all(&PathBuf::from(DB_PATH)).expect("corpus");
    let mut by_prog: BTreeMap<&str, &CompilationProfile> = BTreeMap::new();
    for r in &rows {
        if r.config_id.contains("_rec") {
            by_prog.entry(r.program_name.as_str()).or_insert(r);
        }
    }
    let v2_preds: Vec<f64> = ds
        .names
        .iter()
        .map(|n| {
            let r = by_prog[n.as_str()];
            bundle.head.predict_thermal(&query_from_row(r))
        })
        .collect();

    let all_idx: Vec<usize> = (0..ds.names.len()).collect();
    let report = |label: &str, idx: &[usize]| {
        let v1_mae = mae_over(&ds.v1_preds, &ds.labels, idx);
        let v2_mae = mae_over(&v2_preds, &ds.labels, idx);
        let pick = |src: &[f64]| -> (Vec<f64>, Vec<f64>) {
            (
                idx.iter().map(|&i| src[i]).collect(),
                idx.iter().map(|&i| ds.labels[i]).collect(),
            )
        };
        let (v1_p, lab) = pick(&ds.v1_preds);
        let (v2_p, _) = pick(&v2_preds);
        println!(
            "  {label:<10} n={:<4} | v1: MAE={v1_mae:.4} corr={:+.4} | v2: MAE={v2_mae:.4} corr={:+.4}",
            idx.len(),
            pearson(&v1_p, &lab),
            pearson(&v2_p, &lab),
        );
        (v1_mae, v2_mae)
    };
    report("train", &ds.train_idx);
    let (v1_test, v2_test) = report("held-out", &ds.test_idx);
    // Frozen holdout (Phase A item 7): never used in any training or
    // tuning decision — THE promotion-gate cohort that doesn't erode.
    if !ds.holdout_idx.is_empty() {
        report("holdout", &ds.holdout_idx);
    }
    let (v1_all, v2_all) = report("overall", &all_idx);

    let promote = v2_test < v1_test && v2_all < v1_all;
    println!(
        "\n§2.5 promotion gate (v2 must beat v1 on held-out AND overall MAE): {}",
        if promote {
            "PROMOTE — attach the trained head via PinnPhysicalCostModel::with_thermal_head"
        } else {
            "DO NOT PROMOTE — v1 closed form stays active"
        }
    );
}

// =============================================================================
// Mode: sanity-energy — Phase B §B1 data-sanity pass over the energy signal
// =============================================================================
//
// Settles the research doc's hypothesis (ridge + loop features reaches
// R²(test) 0.65–0.75 on energy) BEFORE any training code ships. The
// only empirical datum so far is the −32 held-out R² from the v2
// corpus. Two design facts shape the grid:
//
// 1. The deployed consumer is Phase C's PLAN SELECTOR, which scores
//    candidate plans — those carry pass counts, workload, loop shape,
//    and a statically-computable post-plan node count, but NO config
//    identity. Config one-hots (the diagnosed collinearity culprit)
//    are therefore excluded from every deployable feature set; one
//    grid row keeps them solely to replicate the −32 failure.
// 2. The selector-relevant quality metric is REGRET (measured score of
//    the predicted-argmin plan minus the measured minimum), not R².
//    Both are reported; regret against honest baselines (always-pick-
//    baseline-plan, structural-ratio-argmin) decides Phase C's go/no-go.

/// Loop-feature aggregates from the schema-v3 per-function records:
/// total countable loops + max nesting depth across functions
/// (mirrors `build_physical_query`'s amplification view).
fn loop_features(row: &CompilationProfile) -> (f64, f64) {
    let mut countable = 0u64;
    let mut depth = 0u32;
    for (_f, fp) in &row.per_function {
        countable = countable.saturating_add(fp.countable_loop_count as u64);
        depth = depth.max(fp.max_loop_depth);
    }
    (countable as f64, depth as f64)
}

/// Deployable feature surface for energy prediction (NO config
/// one-hots — see mode docs). `with_loops` / `with_structural` toggle
/// the grid axes.
fn energy_features(
    row: &CompilationProfile,
    vocab: &[String],
    with_loops: bool,
    with_structural: bool,
) -> Vec<f64> {
    let mut x = vec![
        log1p_u64(row.estimated_flops),
        log1p_u64(row.estimated_bytes_read),
        log1p_u64(row.estimated_bytes_written),
        log1p_u64(row.estimated_alloc_bytes),
        log1p_u64(row.estimated_working_set),
        log1p_u64(row.estimated_float_ops),
        fp_density(row),
        log1p_u64(row.mir_nodes_before),
        row.recommended_count as f64,
        row.dropped_count as f64,
    ];
    x.extend(pass_counts(row, vocab));
    if with_loops {
        let (countable, depth) = loop_features(row);
        x.push((1.0 + countable).ln());
        x.push(depth);
    }
    if with_structural {
        x.push(log1p_u64(row.mir_nodes_after));
        let ratio = if row.mir_nodes_before > 0 {
            row.mir_nodes_after as f64 / row.mir_nodes_before as f64
        } else {
            1.0
        };
        x.push(ratio);
    }
    x
}

/// Selector regret on one cohort of programs: fit predictions are
/// supplied per row; for each program pick the argmin-predicted row
/// among its configs and charge the measured-score gap to the true
/// minimum. Deterministic: ties break by config id (BTreeMap order).
fn mean_regret(
    rows_by_prog: &BTreeMap<&str, Vec<(&CompilationProfile, f64)>>,
) -> (f64, usize, usize) {
    let mut acc = KahanAccumulatorF64::new();
    let mut n = 0usize;
    let mut hits = 0usize;
    for per_prog in rows_by_prog.values() {
        let measured_min = per_prog
            .iter()
            .map(|(r, _)| r.score)
            .fold(f64::INFINITY, f64::min);
        // argmin by predicted value; first-in-BTreeMap-order wins ties.
        let mut best_pred = f64::INFINITY;
        let mut chosen_score = f64::NAN;
        for (r, pred) in per_prog {
            if *pred < best_pred {
                best_pred = *pred;
                chosen_score = r.score;
            }
        }
        let regret = chosen_score - measured_min;
        acc.add(regret);
        n += 1;
        if regret.abs() < 1e-9 {
            hits += 1;
        }
    }
    (acc.finalize() / n.max(1) as f64, hits, n)
}

// =============================================================================
// Mode: train-energy — fit the energy head, persist as CPB1 (Phase B §B2)
// =============================================================================

/// Default output path for the trained energy bundle.
const ENERGY_BUNDLE_PATH: &str = "bench_results/cana_train_pinn/pinn_energy_v1.cpb";

/// Lift one corpus row into the head's [`EnergyQuery`] — the single
/// basis definition lives in `cjc-cana::pinn_energy_v1`; this only
/// adapts row fields onto it.
fn energy_query_from_row(
    head: &cjc_cana::pinn_energy_v1::PinnEnergyV1,
    r: &CompilationProfile,
) -> cjc_cana::pinn_energy_v1::EnergyQuery {
    let (countable, depth) = loop_features(r);
    cjc_cana::pinn_energy_v1::EnergyQuery {
        flops_estimate: r.estimated_flops,
        bytes_read_estimate: r.estimated_bytes_read,
        bytes_written_estimate: r.estimated_bytes_written,
        allocation_bytes_estimate: r.estimated_alloc_bytes,
        working_set_bytes_estimate: r.estimated_working_set,
        float_ops_estimate: r.estimated_float_ops,
        mir_nodes_before: r.mir_nodes_before,
        recommended_count: r.recommended_count,
        dropped_count: r.dropped_count,
        pass_counts: head.pass_counts(
            r.pass_sequence
                .iter()
                .flat_map(|(_, ps)| ps.iter().map(|s| s.as_str())),
        ),
        countable_loop_count: countable as u64,
        max_loop_depth: depth as u32,
        mir_nodes_after: r.mir_nodes_after,
    }
}

fn run_train_energy() {
    use cjc_cana::pinn_energy_v1::{
        PinnEnergyV1, ENERGY_TAIL_FEATURES, ENERGY_WORKLOAD_FEATURES, PINN_ENERGY_V1_MODEL_ID,
        PINN_ENERGY_V1_MODEL_VERSION,
    };
    use cjc_cana_compress::energy_bundle::{write_energy_bundle, EnergyBundle};

    let rows = read_all(&PathBuf::from(DB_PATH)).expect("corpus");
    let vocab: Vec<String> = {
        let set: BTreeSet<String> = rows
            .iter()
            .flat_map(|r| r.pass_sequence.iter())
            .flat_map(|(_, ps)| ps.iter().cloned())
            .collect();
        set.into_iter().collect()
    };
    let n_features = ENERGY_WORKLOAD_FEATURES + vocab.len() + ENERGY_TAIL_FEATURES;
    // Template head: vocabulary fixed, parameters identity — used only
    // to build the design matrix through THE shared basis definition.
    let template = PinnEnergyV1 {
        pass_names: vocab.clone(),
        feature_means: vec![0.0; n_features],
        feature_stds: vec![1.0; n_features],
        coefficients: vec![0.0; n_features],
        intercept: 0.0,
    };

    // Evidence-chosen recipe (sanity-energy §3): fit on ALL working
    // train rows (ties included), ln(score) target.
    let fit_rows: Vec<&CompilationProfile> = rows
        .iter()
        .filter(|r| !is_holdout_program(&r.program_name) && !is_test_program(&r.program_name))
        .collect();
    let test_rows: Vec<&CompilationProfile> = rows
        .iter()
        .filter(|r| !is_holdout_program(&r.program_name) && is_test_program(&r.program_name))
        .collect();

    println!("=== PINN energy-head training (deterministic ridge OLS, ln target) ===");
    println!(
        "  rows: {} fit / {} FNV-test / holdout frozen; pass vocabulary ({}): {:?}",
        fit_rows.len(),
        test_rows.len(),
        vocab.len(),
        vocab
    );

    let feats: Vec<Vec<f64>> = fit_rows
        .iter()
        .map(|r| template.features_from_query(&energy_query_from_row(&template, r)))
        .collect();
    let targets: Vec<f64> = fit_rows.iter().map(|r| r.score.max(1e-12).ln()).collect();
    let train_idx: Vec<usize> = (0..fit_rows.len()).collect();
    let model = fit_ridge(&feats, &targets, &train_idx).expect("fit failed");

    let r2_fit = r2_over(&model, &feats, &targets, &train_idx);
    let test_feats: Vec<Vec<f64>> = test_rows
        .iter()
        .map(|r| template.features_from_query(&energy_query_from_row(&template, r)))
        .collect();
    let test_targets: Vec<f64> = test_rows.iter().map(|r| r.score.max(1e-12).ln()).collect();
    let test_idx: Vec<usize> = (0..test_rows.len()).collect();
    let fit_for_eval = FittedLinear {
        col_mean: model.col_mean.clone(),
        col_std: model.col_std.clone(),
        beta: model.beta.clone(),
    };
    let r2_test = r2_over(&fit_for_eval, &test_feats, &test_targets, &test_idx);
    println!("  R²(fit rows, ln) = {r2_fit:.4}   R²(FNV-test rows, ln) = {r2_test:.4}");
    println!("  (all-rows R² is tie-dominated by design — the shadow gate judges on regret;");
    println!("   diverged-subset R² is reported there.)");

    let head = PinnEnergyV1 {
        pass_names: vocab,
        feature_means: model.col_mean.clone(),
        feature_stds: model.col_std.clone(),
        coefficients: model.beta[..n_features].to_vec(),
        intercept: model.beta[n_features],
    };
    assert!(head.is_valid(), "trained head must validate");

    // Sanity: a baseline-identical plan should predict near ln(1)=0.
    if let Some(baseline_row) = rows.iter().find(|r| r.config_id == "baseline") {
        let pred = head.predict_ln_score(&energy_query_from_row(&head, baseline_row));
        println!("  sanity: baseline-plan prediction = {pred:+.4} (expect ≈ 0 = tie)");
    }

    let bundle = EnergyBundle {
        model_id: PINN_ENERGY_V1_MODEL_ID.to_string(),
        model_version: PINN_ENERGY_V1_MODEL_VERSION,
        head,
    };
    let path = PathBuf::from(ENERGY_BUNDLE_PATH);
    fs::create_dir_all(path.parent().unwrap()).expect("create bundle dir");
    write_energy_bundle(&path, &bundle).expect("write bundle");
    println!(
        "  bundle written: {} ({} / v{})",
        path.display(),
        bundle.model_id,
        bundle.model_version
    );
    let first = fs::read(&path).unwrap();
    write_energy_bundle(&path, &bundle).expect("re-write bundle");
    assert_eq!(first, fs::read(&path).unwrap(), "bundle must be byte-stable");
    println!("  bundle double-write: byte-identical");
    println!("\nNext: `cargo run --release -p cana-train-pinn -- shadow-energy`.");
}

// =============================================================================
// Mode: shadow-energy — trained head vs baselines against measured labels
// =============================================================================

fn run_shadow_energy() {
    use cjc_cana_compress::energy_bundle::read_energy_bundle;

    let rows = read_all(&PathBuf::from(DB_PATH)).expect("corpus");
    let bundle = read_energy_bundle(&PathBuf::from(ENERGY_BUNDLE_PATH)).unwrap_or_else(|e| {
        panic!("cannot read bundle at {ENERGY_BUNDLE_PATH} — run `-- train-energy` first: {e}")
    });
    println!(
        "=== Energy shadow mode — trained head ({} v{}) vs plan-choice baselines ===",
        bundle.model_id, bundle.model_version
    );
    println!("  ground truth: measured baseline-relative modeled energy (row score)");

    // Predictions through the REAL persisted head (CPB1 round-trip is
    // part of what this gate proves).
    let predict = |r: &CompilationProfile| bundle.head.predict_ln_score(&energy_query_from_row(&bundle.head, r));

    let test_rows: Vec<&CompilationProfile> = rows
        .iter()
        .filter(|r| !is_holdout_program(&r.program_name) && is_test_program(&r.program_name))
        .collect();
    let holdout_rows: Vec<&CompilationProfile> = rows
        .iter()
        .filter(|r| is_holdout_program(&r.program_name))
        .collect();

    // R² in ln space on the diverged FNV-test subset (prediction-
    // quality diagnostic; the gate itself judges regret).
    let div_test: Vec<&CompilationProfile> = test_rows
        .iter()
        .copied()
        .filter(|r| (r.score - 1.0).abs() > SCORE_DIVERGENCE_EPS)
        .collect();
    let r2_diag = {
        let ys: Vec<f64> = div_test.iter().map(|r| r.score.max(1e-12).ln()).collect();
        let preds: Vec<f64> = div_test.iter().map(|r| predict(r)).collect();
        let s = stats(&ys);
        if s.std == 0.0 || ys.is_empty() {
            f64::NAN
        } else {
            let mut ss_res = KahanAccumulatorF64::new();
            let mut ss_tot = KahanAccumulatorF64::new();
            for (y, p) in ys.iter().zip(preds.iter()) {
                let e = y - p;
                ss_res.add(e * e);
                let d = y - s.mean;
                ss_tot.add(d * d);
            }
            1.0 - ss_res.finalize() / ss_tot.finalize()
        }
    };
    println!(
        "  diagnostic: R²(ln, diverged FNV-test rows, n={}) = {r2_diag:.4}",
        div_test.len()
    );

    // Regret per cohort: model vs always-baseline vs structural-argmin.
    let cohort = |label: &str, rows_in: &[&CompilationProfile]| -> (f64, f64, f64) {
        let mut by_model: BTreeMap<&str, Vec<(&CompilationProfile, f64)>> = BTreeMap::new();
        let mut by_struct: BTreeMap<&str, Vec<(&CompilationProfile, f64)>> = BTreeMap::new();
        for r in rows_in {
            by_model
                .entry(r.program_name.as_str())
                .or_default()
                .push((r, predict(r)));
            by_struct
                .entry(r.program_name.as_str())
                .or_default()
                .push((r, r.mir_nodes_after as f64));
        }
        let (model_regret, hits, n) = mean_regret(&by_model);
        let (struct_regret, struct_hits, _) = mean_regret(&by_struct);
        let mut base_acc = KahanAccumulatorF64::new();
        for per_prog in by_model.values() {
            let min = per_prog
                .iter()
                .map(|(r, _)| r.score)
                .fold(f64::INFINITY, f64::min);
            base_acc.add(1.0 - min);
        }
        let base_regret = base_acc.finalize() / by_model.len().max(1) as f64;
        println!(
            "  {label:<10} model regret {model_regret:+.5} (exact {hits}/{n}) | always-baseline {base_regret:+.5} | structural {struct_regret:+.5} (exact {struct_hits}/{n})"
        );
        (model_regret, base_regret, struct_regret)
    };
    let (t_model, t_base, t_struct) = cohort("test", &test_rows);
    let (h_model, _h_base, h_struct) = cohort("holdout", &holdout_rows);

    // Promotion gate: the deployment metric decides. R² is a floor
    // sanity (must be positive on the diverged test subset), not the
    // criterion — sanity-energy measured that the regret-best fit is
    // NOT the R²-best fit.
    let promote =
        t_model < t_base && t_model < t_struct && h_model <= h_struct && r2_diag > 0.0;
    println!(
        "\nPhase-B promotion gate (test regret beats BOTH baselines, holdout regret ≤ structural, diverged R² > 0): {}",
        if promote {
            "PROMOTE — the trained energy head is the selector criterion for Phase C"
        } else {
            "DO NOT PROMOTE — hand-tuned criteria stay; record the numbers and stop"
        }
    );
}

fn run_sanity_energy() {
    let path = PathBuf::from(DB_PATH);
    let rows = read_all(&path).unwrap_or_else(|e| {
        panic!(
            "cannot read corpus at {} — run `cargo run --release -p cana-ablation` first: {e}",
            path.display()
        )
    });
    let vocab: Vec<String> = {
        let set: BTreeSet<String> = rows
            .iter()
            .flat_map(|r| r.pass_sequence.iter())
            .flat_map(|(_, ps)| ps.iter().cloned())
            .collect();
        set.into_iter().collect()
    };
    let configs: Vec<String> = {
        let set: BTreeSet<String> = rows.iter().map(|r| r.config_id.clone()).collect();
        set.into_iter().collect()
    };

    // Cohorts: holdout rows are excluded from BOTH fit and the FNV test
    // split, reported separately (frozen promotion cohort).
    let working: Vec<&CompilationProfile> = rows
        .iter()
        .filter(|r| !is_holdout_program(&r.program_name))
        .collect();
    let holdout: Vec<&CompilationProfile> = rows
        .iter()
        .filter(|r| is_holdout_program(&r.program_name))
        .collect();
    let diverged: Vec<&CompilationProfile> = working
        .iter()
        .copied()
        .filter(|r| (r.score - 1.0).abs() > SCORE_DIVERGENCE_EPS)
        .collect();

    section("1. Energy label distribution (schema-v3 corpus)");
    let all_scores: Vec<f64> = working.iter().map(|r| r.score).collect();
    let div_scores: Vec<f64> = diverged.iter().map(|r| r.score).collect();
    print_stats_line("score (working rows)", &stats(&all_scores));
    print_stats_line("score (diverged only)", &stats(&div_scores));
    println!(
        "  diverged rows: {}/{} working ({} holdout rows kept frozen)",
        diverged.len(),
        working.len(),
        holdout.len()
    );

    // ---- 2. The OLS grid -----------------------------------------------------
    section("2. Ridge grid (program-level split; holdout excluded everywhere)");
    println!("  target ln(score) is evaluated in ln space; argmin/ranking is transform-invariant.");
    let grid_ols = |label: &str,
                    subset: &[&CompilationProfile],
                    feats_of: &dyn Fn(&CompilationProfile) -> Vec<f64>,
                    log_target: bool| {
        let feats: Vec<Vec<f64>> = subset.iter().map(|r| feats_of(r)).collect();
        let targets: Vec<f64> = subset
            .iter()
            .map(|r| if log_target { r.score.max(1e-12).ln() } else { r.score })
            .collect();
        let mask: Vec<bool> = subset
            .iter()
            .map(|r| is_test_program(&r.program_name))
            .collect();
        match ols(&feats, &targets, &mask) {
            Some(rep) => println!(
                "  {label:<58} R²(train)={:>8.4}  R²(test)={:>8.4}  (n={}+{})",
                rep.r2_train, rep.r2_test, rep.n_train, rep.n_test
            ),
            None => println!("  {label:<58} — insufficient rows"),
        }
    };

    // Replication row: the −32 failure recipe (config one-hots, raw
    // score, diverged rows) on the v3 corpus.
    let v2_recipe = |r: &CompilationProfile| score_features(r, &vocab, &configs, false);
    grid_ols(
        "REPLICATION: one-hots, raw score, diverged rows",
        &diverged,
        &v2_recipe,
        false,
    );
    // Deployable surfaces.
    for (rows_label, subset) in [("all rows", &working), ("diverged", &diverged)] {
        for log_target in [false, true] {
            for (with_loops, with_structural, fl) in [
                (false, false, "base"),
                (true, false, "+loops"),
                (false, true, "+structural"),
                (true, true, "+loops+structural"),
            ] {
                let f = |r: &CompilationProfile| {
                    energy_features(r, &vocab, with_loops, with_structural)
                };
                let t = if log_target { "ln(score)" } else { "score" };
                grid_ols(
                    &format!("{rows_label:<9} {t:<9} {fl}"),
                    subset,
                    &f,
                    log_target,
                );
            }
        }
    }

    // ---- 3. Selector regret (the Phase-C go/no-go number) --------------------
    // Two fit recipes compete: ALL train rows (ties included — teaches
    // the model where score == 1.0) vs DIVERGED-only (the R²-0.82 fit).
    // The shipped recipe is whichever wins held-out + holdout regret.
    section("3. Selector regret (ln target, +loops+structural; model vs baselines)");
    let feats_of = |r: &CompilationProfile| energy_features(r, &vocab, true, true);
    let fit_on = |label: &str, fit_rows: &[&CompilationProfile]| -> Option<FittedLinear> {
        let fit_feats: Vec<Vec<f64>> = fit_rows.iter().map(|r| feats_of(r)).collect();
        let fit_targets: Vec<f64> = fit_rows.iter().map(|r| r.score.max(1e-12).ln()).collect();
        let train_idx: Vec<usize> = (0..fit_rows.len()).collect();
        let m = fit_ridge(&fit_feats, &fit_targets, &train_idx);
        if m.is_none() {
            println!("  {label}: fit failed");
        }
        m
    };
    let all_train: Vec<&CompilationProfile> = working
        .iter()
        .copied()
        .filter(|r| !is_test_program(&r.program_name))
        .collect();
    let div_train: Vec<&CompilationProfile> = diverged
        .iter()
        .copied()
        .filter(|r| !is_test_program(&r.program_name))
        .collect();
    let test_rows: Vec<&CompilationProfile> = working
        .iter()
        .copied()
        .filter(|r| is_test_program(&r.program_name))
        .collect();

    let cohort_regret = |label: &str, cohort: &[&CompilationProfile], model: &FittedLinear| {
        let mut by_prog: BTreeMap<&str, Vec<(&CompilationProfile, f64)>> = BTreeMap::new();
        for r in cohort {
            by_prog
                .entry(r.program_name.as_str())
                .or_default()
                .push((r, model.predict(&feats_of(r))));
        }
        let (regret, hits, n) = mean_regret(&by_prog);
        println!("    {label:<10} mean regret {regret:+.5} (exact-best {hits}/{n})");
    };
    for (fit_label, fit_rows) in [("fit on ALL train rows", &all_train), ("fit on DIVERGED train rows", &div_train)]
    {
        if let Some(model) = fit_on(fit_label, fit_rows) {
            println!("  {fit_label}:");
            cohort_regret("test", &test_rows, &model);
            cohort_regret("holdout", &holdout, &model);
        }
    }

    // Baselines (model-free), per cohort.
    let baselines = |label: &str, cohort: &[&CompilationProfile]| {
        let mut by_prog: BTreeMap<&str, Vec<(&CompilationProfile, f64)>> = BTreeMap::new();
        for r in cohort {
            by_prog
                .entry(r.program_name.as_str())
                .or_default()
                .push((r, r.mir_nodes_after as f64));
        }
        let (struct_regret, struct_hits, n) = mean_regret(&by_prog);
        let mut base_acc = KahanAccumulatorF64::new();
        for per_prog in by_prog.values() {
            let min = per_prog
                .iter()
                .map(|(r, _)| r.score)
                .fold(f64::INFINITY, f64::min);
            base_acc.add(1.0 - min);
        }
        let base_regret = base_acc.finalize() / by_prog.len().max(1) as f64;
        println!(
            "  baselines {label:<10} always-baseline: {base_regret:+.5} | structural-argmin: {struct_regret:+.5} (exact {struct_hits}/{n})"
        );
    };
    baselines("test", &test_rows);
    baselines("holdout", &holdout);

    println!("\nSanity-energy pass complete — corpus read-only, nothing written.");
}

fn run_sanity() {
    let path = PathBuf::from(DB_PATH);
    let rows = read_all(&path).unwrap_or_else(|e| {
        panic!(
            "cannot read corpus at {} — run `cargo run --release -p cana-ablation` first: {e}",
            path.display()
        )
    });

    // ---- 1. Inventory -------------------------------------------------------
    section("1. Corpus inventory");
    let configs: Vec<String> = {
        let set: BTreeSet<String> = rows.iter().map(|r| r.config_id.clone()).collect();
        set.into_iter().collect()
    };
    let programs: BTreeSet<&str> = rows.iter().map(|r| r.program_name.as_str()).collect();
    let mut per_config_count: BTreeMap<&str, usize> = BTreeMap::new();
    let mut per_family_count: BTreeMap<&str, usize> = BTreeMap::new();
    for r in &rows {
        *per_config_count.entry(r.config_id.as_str()).or_insert(0) += 1;
        *per_family_count.entry(family(&r.program_name)).or_insert(0) += 1;
    }
    println!("  rows: {}", rows.len());
    println!("  programs: {}", programs.len());
    println!("  configs ({}): {:?}", configs.len(), configs);
    println!("  rows per config: {:?}", per_config_count);
    println!("  rows per family: {:?}", per_family_count);
    let parity_ok = rows.iter().filter(|r| r.parity_match == Some(true)).count();
    let legal_ok = rows.iter().filter(|r| r.legality_approved).count();
    let models: BTreeSet<(String, u32)> = rows
        .iter()
        .map(|r| (r.cost_model_id.clone(), r.cost_model_version))
        .collect();
    println!("  parity Some(true): {}/{}", parity_ok, rows.len());
    println!("  legality approved: {}/{}", legal_ok, rows.len());
    println!("  cost models seen: {:?}", models);
    let holdout_programs = programs.iter().filter(|p| is_holdout_program(p)).count();
    let test_programs = programs
        .iter()
        .filter(|p| !is_holdout_program(p) && is_test_program(p))
        .count();
    println!(
        "  train/test split (FNV%{SPLIT_MOD}==0 → test): {} train / {} test / {} holdout-frozen programs",
        programs.len() - test_programs - holdout_programs,
        test_programs,
        holdout_programs
    );

    // ---- 2. Score label: distribution per config ---------------------------
    section("2. Score (baseline-relative energy) per config");
    let mut diverged_rows = 0usize;
    for config in &configs {
        let scores: Vec<f64> = rows
            .iter()
            .filter(|r| &r.config_id == config)
            .map(|r| r.score)
            .collect();
        let s = stats(&scores);
        let diverged: Vec<f64> = scores
            .iter()
            .copied()
            .filter(|v| (v - 1.0).abs() > SCORE_DIVERGENCE_EPS)
            .collect();
        diverged_rows += diverged.len();
        let (dmin, dmax) = if diverged.is_empty() {
            (f64::NAN, f64::NAN)
        } else {
            let ds = stats(&diverged);
            (ds.min, ds.max)
        };
        println!(
            "  {config:<18} n={:<4} mean={:>8.5} std={:>8.5} | ≠1.0: {:>3} rows in [{dmin:.5}, {dmax:.5}]",
            s.n,
            s.mean,
            s.std,
            diverged.len(),
        );
    }
    println!(
        "  total rows with score ≠ 1.0: {diverged_rows}/{}",
        rows.len()
    );
    let distinct_scores: BTreeSet<u64> = rows.iter().map(|r| r.score.to_bits()).collect();
    println!(
        "  distinct score values (bitwise): {}",
        distinct_scores.len()
    );

    // ---- 3. Where does score divergence live? ------------------------------
    section("3. Score divergence by family × config");
    for config in &configs {
        let mut per_fam: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
        for r in rows.iter().filter(|r| &r.config_id == config) {
            let e = per_fam.entry(family(&r.program_name)).or_insert((0, 0));
            e.1 += 1;
            if (r.score - 1.0).abs() > SCORE_DIVERGENCE_EPS {
                e.0 += 1;
            }
        }
        let cells: Vec<String> = per_fam
            .iter()
            .map(|(f, (d, n))| format!("{f}:{d}/{n}"))
            .collect();
        println!("  {config:<18} {}", cells.join("  "));
    }

    // ---- 4. Pressure labels: per-row or per-program? ------------------------
    section("4. Pressure-label structure (bitwise identity across configs of a program)");
    let rec_configs: Vec<&String> = configs.iter().filter(|c| c.contains("_rec")).collect();
    let syn_configs: Vec<&String> = configs.iter().filter(|c| !c.contains("_rec")).collect();
    for (cohort_name, cohort) in [
        ("recorded (*_rec)", &rec_configs),
        ("synthetic", &syn_configs),
    ] {
        let mut identical = [0usize; 3]; // cpu, mem, thermal
        let mut total = 0usize;
        for prog in &programs {
            let cohort_rows: Vec<&CompilationProfile> = rows
                .iter()
                .filter(|r| &r.program_name == prog && cohort.iter().any(|c| *c == &r.config_id))
                .collect();
            if cohort_rows.is_empty() {
                continue;
            }
            total += 1;
            let distinct = |f: fn(&CompilationProfile) -> f64| -> usize {
                cohort_rows
                    .iter()
                    .map(|r| f(r).to_bits())
                    .collect::<BTreeSet<u64>>()
                    .len()
            };
            if distinct(|r| r.nss_predicted_cpu_max) == 1 {
                identical[0] += 1;
            }
            if distinct(|r| r.nss_predicted_memory_max) == 1 {
                identical[1] += 1;
            }
            if distinct(|r| r.nss_predicted_thermal_max) == 1 {
                identical[2] += 1;
            }
        }
        println!(
            "  {cohort_name:<18} programs with identical labels across configs: cpu {}/{total}, mem {}/{total}, thermal {}/{total}",
            identical[0], identical[1], identical[2]
        );
    }

    // Program-level recorded labels (one value per program, from the first
    // *_rec row — identity across rec configs is verified above).
    let mut rec_by_prog: BTreeMap<&str, &CompilationProfile> = BTreeMap::new();
    for r in &rows {
        if r.config_id.contains("_rec") {
            rec_by_prog.entry(r.program_name.as_str()).or_insert(r);
        }
    }
    let rec_cpu: Vec<f64> = rec_by_prog
        .values()
        .map(|r| r.nss_predicted_cpu_max)
        .collect();
    let rec_mem: Vec<f64> = rec_by_prog
        .values()
        .map(|r| r.nss_predicted_memory_max)
        .collect();
    let rec_thermal: Vec<f64> = rec_by_prog
        .values()
        .map(|r| r.nss_predicted_thermal_max)
        .collect();
    println!(
        "  recorded labels at PROGRAM granularity ({} programs):",
        rec_by_prog.len()
    );
    print_stats_line("cpu_max", &stats(&rec_cpu));
    print_stats_line("memory_max", &stats(&rec_mem));
    print_stats_line("thermal_max", &stats(&rec_thermal));
    let mut bands: BTreeMap<u32, usize> = BTreeMap::new();
    for &t in &rec_thermal {
        *bands.entry((t * 10.0).floor() as u32).or_insert(0) += 1;
    }
    let band_view: Vec<String> = bands
        .iter()
        .map(|(b, n)| format!("[0.{b}0,0.{}0): {n}", b + 1))
        .collect();
    println!("  thermal bands: {}", band_view.join("  "));

    // Schema v3 (Phase A item 2): per-FUNCTION recorded labels — the
    // effective-label-count multiplier the per-program MAX was hiding.
    let mut fn_cpu: Vec<f64> = Vec::new();
    let mut fn_memory: Vec<f64> = Vec::new();
    let mut fn_thermal: Vec<f64> = Vec::new();
    for r in rec_by_prog.values() {
        for (_fname, fp) in &r.per_function {
            fn_cpu.push(fp.nss_cpu);
            fn_memory.push(fp.nss_memory);
            fn_thermal.push(fp.nss_thermal);
        }
    }
    println!(
        "  recorded labels at FUNCTION granularity ({} labels):",
        fn_thermal.len()
    );
    print_stats_line("fn cpu", &stats(&fn_cpu));
    print_stats_line("fn memory", &stats(&fn_memory));
    print_stats_line("fn thermal", &stats(&fn_thermal));

    // ---- 5. PINN v1 closed-form vs recorded labels (the bar v2 must beat) --
    section("5. PINN v1 predictions vs recorded labels (program granularity)");
    let v1_thermal: Vec<f64> = rec_by_prog
        .values()
        .map(|r| r.pinn_predicted_thermal_max)
        .collect();
    let v1_energy: Vec<f64> = rec_by_prog
        .values()
        .map(|r| r.pinn_predicted_energy_max)
        .collect();
    let v1_bandwidth: Vec<f64> = rec_by_prog
        .values()
        .map(|r| r.pinn_predicted_bandwidth_max)
        .collect();
    print_stats_line("v1 thermal_max", &stats(&v1_thermal));
    print_stats_line("v1 energy_max", &stats(&v1_energy));
    print_stats_line("v1 bandwidth_max", &stats(&v1_bandwidth));
    println!(
        "  corr(v1 thermal, recorded thermal) = {:+.4}",
        pearson(&v1_thermal, &rec_thermal)
    );
    println!(
        "  corr(v1 energy,  recorded thermal) = {:+.4}",
        pearson(&v1_energy, &rec_thermal)
    );
    println!(
        "  corr(v1 energy,  recorded cpu)     = {:+.4}",
        pearson(&v1_energy, &rec_cpu)
    );
    println!(
        "  corr(v1 energy,  recorded memory)  = {:+.4}",
        pearson(&v1_energy, &rec_mem)
    );

    // ---- 6. Workload feature spread -----------------------------------------
    section("6. Workload features (program granularity)");
    let feat = |f: fn(&CompilationProfile) -> u64| -> Vec<f64> {
        rec_by_prog.values().map(|r| f(r) as f64).collect()
    };
    print_stats_line("estimated_flops", &stats(&feat(|r| r.estimated_flops)));
    print_stats_line(
        "estimated_bytes_read",
        &stats(&feat(|r| r.estimated_bytes_read)),
    );
    print_stats_line(
        "estimated_bytes_written",
        &stats(&feat(|r| r.estimated_bytes_written)),
    );
    print_stats_line(
        "estimated_alloc_bytes",
        &stats(&feat(|r| r.estimated_alloc_bytes)),
    );
    print_stats_line(
        "estimated_working_set",
        &stats(&feat(|r| r.estimated_working_set)),
    );
    print_stats_line(
        "estimated_float_ops",
        &stats(&feat(|r| r.estimated_float_ops)),
    );
    print_stats_line("mir_nodes_before", &stats(&feat(|r| r.mir_nodes_before)));
    let densities: Vec<f64> = rec_by_prog.values().map(|r| fp_density(r)).collect();
    print_stats_line("fp_density (float/flops)", &stats(&densities));
    println!(
        "  corr(fp_density, recorded thermal) = {:+.4}   ← the §2.2 information-gap check",
        pearson(&densities, &rec_thermal)
    );

    // ---- 7. Pass-plan vocabulary --------------------------------------------
    section("7. Pass-plan vocabulary and plan sizes");
    let vocab: Vec<String> = {
        let set: BTreeSet<String> = rows
            .iter()
            .flat_map(|r| r.pass_sequence.iter())
            .flat_map(|(_, ps)| ps.iter().cloned())
            .collect();
        set.into_iter().collect()
    };
    println!("  vocabulary ({}): {:?}", vocab.len(), vocab);
    for config in &configs {
        let mut acc = KahanAccumulatorF64::new();
        let mut n = 0usize;
        for r in rows.iter().filter(|r| &r.config_id == config) {
            let total: usize = r.pass_sequence.iter().map(|(_, p)| p.len()).sum();
            acc.add(total as f64);
            n += 1;
        }
        println!(
            "  {config:<18} mean passes/row = {:.3}",
            acc.finalize() / n.max(1) as f64
        );
    }

    // ---- 8. Feature ↔ score correlations ------------------------------------
    section("8. Per-feature Pearson correlation with score");
    let diverged: Vec<&CompilationProfile> = rows
        .iter()
        .filter(|r| (r.score - 1.0).abs() > SCORE_DIVERGENCE_EPS)
        .collect();
    println!(
        "  (over the {} plan-divergent rows; all-row correlations are diluted by exact-1.0 scores)",
        diverged.len()
    );
    let div_scores: Vec<f64> = diverged.iter().map(|r| r.score).collect();
    let corr_with = |label: &str, vals: Vec<f64>| {
        println!(
            "  corr({label:<26}, score) = {:+.4}",
            pearson(&vals, &div_scores)
        );
    };
    corr_with(
        "log1p(flops)",
        diverged
            .iter()
            .map(|r| log1p_u64(r.estimated_flops))
            .collect(),
    );
    corr_with(
        "log1p(working_set)",
        diverged
            .iter()
            .map(|r| log1p_u64(r.estimated_working_set))
            .collect(),
    );
    corr_with(
        "log1p(mir_nodes_before)",
        diverged
            .iter()
            .map(|r| log1p_u64(r.mir_nodes_before))
            .collect(),
    );
    corr_with(
        "nodes_after/nodes_before",
        diverged
            .iter()
            .map(|r| {
                if r.mir_nodes_before > 0 {
                    r.mir_nodes_after as f64 / r.mir_nodes_before as f64
                } else {
                    1.0
                }
            })
            .collect(),
    );
    corr_with(
        "recorded thermal_max",
        diverged
            .iter()
            .map(|r| r.nss_predicted_thermal_max)
            .collect(),
    );
    corr_with(
        "recommended_count",
        diverged
            .iter()
            .map(|r| r.recommended_count as f64)
            .collect(),
    );
    for p in &vocab {
        corr_with(
            &format!("plan #{p}"),
            diverged
                .iter()
                .map(|r| pass_counts(r, std::slice::from_ref(p))[0])
                .collect(),
        );
    }

    // ---- 9. Linear saturation check -----------------------------------------
    section("9. OLS saturation check (ridge 1e-8, by-program split)");
    let run_score_ols = |label: &str, subset: &[&CompilationProfile], structural: bool| {
        let feats: Vec<Vec<f64>> = subset
            .iter()
            .map(|r| score_features(r, &vocab, &configs, structural))
            .collect();
        let targets: Vec<f64> = subset.iter().map(|r| r.score).collect();
        let mask: Vec<bool> = subset
            .iter()
            .map(|r| is_test_program(&r.program_name))
            .collect();
        match ols(&feats, &targets, &mask) {
            Some(rep) => println!(
                "  {label:<44} R²(train)={:>7.4}  R²(test)={:>7.4}  (n={}+{})",
                rep.r2_train, rep.r2_test, rep.n_train, rep.n_test
            ),
            None => println!("  {label:<44} — insufficient rows for fit"),
        }
    };
    let all_refs: Vec<&CompilationProfile> = rows.iter().collect();
    let rec_refs: Vec<&CompilationProfile> = rows
        .iter()
        .filter(|r| r.config_id.contains("_rec"))
        .collect();
    run_score_ols("score ~ workload+plan+config (all rows)", &all_refs, false);
    run_score_ols(
        "score ~ ... + structural outcome (all rows)",
        &all_refs,
        true,
    );
    run_score_ols("score ~ workload+plan+config (rec rows)", &rec_refs, false);
    run_score_ols(
        "score ~ workload+plan+config (diverged rows)",
        &diverged,
        false,
    );
    run_score_ols("score ~ ... + structural (diverged rows)", &diverged, true);

    // Pressure-prediction task at program granularity: can the workload
    // estimates linearly explain the recorded pressures?
    let prog_rows: Vec<&CompilationProfile> = rec_by_prog.values().copied().collect();
    let prog_feats: Vec<Vec<f64>> = prog_rows
        .iter()
        .map(|r| {
            vec![
                log1p_u64(r.estimated_flops),
                log1p_u64(r.estimated_bytes_read),
                log1p_u64(r.estimated_bytes_written),
                log1p_u64(r.estimated_alloc_bytes),
                log1p_u64(r.estimated_working_set),
                log1p_u64(r.estimated_float_ops),
                fp_density(r),
                log1p_u64(r.mir_nodes_before),
            ]
        })
        .collect();
    let prog_mask: Vec<bool> = prog_rows
        .iter()
        .map(|r| is_test_program(&r.program_name))
        .collect();
    let run_pressure_ols =
        |label: &str, targets: Vec<f64>| match ols(&prog_feats, &targets, &prog_mask) {
            Some(rep) => println!(
                "  {label:<44} R²(train)={:>7.4}  R²(test)={:>7.4}  (n={}+{})",
                rep.r2_train, rep.r2_test, rep.n_train, rep.n_test
            ),
            None => println!("  {label:<44} — insufficient rows for fit"),
        };
    run_pressure_ols(
        "rec thermal ~ workload (program-level)",
        prog_rows
            .iter()
            .map(|r| r.nss_predicted_thermal_max)
            .collect(),
    );
    run_pressure_ols(
        "rec cpu ~ workload (program-level)",
        prog_rows.iter().map(|r| r.nss_predicted_cpu_max).collect(),
    );
    run_pressure_ols(
        "rec memory ~ workload (program-level)",
        prog_rows
            .iter()
            .map(|r| r.nss_predicted_memory_max)
            .collect(),
    );

    println!("\nSanity pass complete — corpus read-only, nothing written.");
}

// =============================================================================
// Tests — protect the deterministic math the report depends on
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_solves_known_system() {
        // 2x + y = 5 ; x + 3y = 10  →  x = 1, y = 3
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 10.0];
        let x = gauss_solve(a, b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn pearson_on_perfect_line_is_one() {
        let xs: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 3.0 * x - 7.0).collect();
        assert!((pearson(&xs, &ys) - 1.0).abs() < 1e-12);
        let neg: Vec<f64> = xs.iter().map(|x| -2.0 * x + 1.0).collect();
        assert!((pearson(&xs, &neg) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn pearson_zero_variance_is_nan() {
        let xs = vec![1.0, 1.0, 1.0];
        let ys = vec![1.0, 2.0, 3.0];
        assert!(pearson(&xs, &ys).is_nan());
    }

    #[test]
    fn ols_recovers_linear_target() {
        // y = 2*x0 - x1 + 4, deterministic synthetic grid.
        let mut rows = Vec::new();
        let mut targets = Vec::new();
        let mut mask = Vec::new();
        for i in 0..40 {
            let x0 = (i % 7) as f64;
            let x1 = (i % 5) as f64;
            rows.push(vec![x0, x1]);
            targets.push(2.0 * x0 - x1 + 4.0);
            mask.push(i % 4 == 0);
        }
        let rep = ols(&rows, &targets, &mask).unwrap();
        assert!(rep.r2_train > 0.999999, "train R² {}", rep.r2_train);
        assert!(rep.r2_test > 0.999999, "test R² {}", rep.r2_test);
    }

    #[test]
    fn stats_handles_constant_and_empty() {
        let s = stats(&[2.5, 2.5, 2.5]);
        assert_eq!(s.mean, 2.5);
        assert_eq!(s.std, 0.0);
        assert_eq!(stats(&[]).n, 0);
    }
}
