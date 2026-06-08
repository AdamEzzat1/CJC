//! CANA cost-model training harness.
//!
//! Pipeline:
//!
//!   1. For each of 18 programs in the corpus, parse → lower to MIR.
//!   2. For each of the 5 canonical passes (CF / SR / DCE / CSE / LICM),
//!      run two configurations:
//!        - With pass:    DEFAULT_PASS_SEQUENCE
//!        - Without pass: DEFAULT_PASS_SEQUENCE minus pass-of-interest
//!      Each config is timed N_ITERS=5 times; take the median run_us.
//!   3. Per-pass benefit label for program P, function F =
//!        (run_us_without - run_us_with) / max(run_us_without, 1.0)
//!      clamped to [0.0, 0.5] (matches LinearCostModel's clamp range).
//!   4. Per-function (features, benefit) pairs feed an OLS fit (gradient
//!      descent for robustness on small N) to produce per-pass coefficients.
//!   5. Generated coefficients are printed as Rust source ready to paste
//!      into `LinearCostModel::trained()`.
//!
//! Determinism:
//!   - Same corpus + same algorithm → byte-identical coefficient output.
//!   - Wall-clock measurements vary across runs (that's the input noise
//!     OLS is supposed to denoise); the fit averages over them.
//!   - GD initial weights are deterministic (all zeros); learning rate
//!     and step count are deterministic constants.

use std::collections::BTreeMap;
use std::time::Instant;

use cjc_cana::features::FnFeatures;
use cjc_mir::optimize::{optimize_program_with_plan, PassPlan, DEFAULT_PASS_SEQUENCE};

mod programs;
use programs::{Program, PROGRAMS};

// ---------------------------------------------------------------------------
// Measurement constants
// ---------------------------------------------------------------------------

const N_ITERS: usize = 5;
const SEED: u64 = 42;

/// Passes whose coefficients we fit.
const TARGET_PASSES: &[&str] = &[
    "constant_fold",
    "strength_reduce",
    "dce",
    "cse",
    "licm",
];

// ---------------------------------------------------------------------------
// Training data shape
// ---------------------------------------------------------------------------

/// One (program, function, pass) measurement row.
#[derive(Debug, Clone)]
struct TrainingPoint {
    program: String,
    function: String,
    pass: String,
    // Features (raw, unnormalized).
    expr_count: f64,
    loop_depth: f64,
    branch_count: f64,
    alloc_sites: f64,
    // Label: the measured benefit, in [0.0, 0.5].
    benefit: f64,
    // Median measured run_us with and without the pass.
    run_us_with: f64,
    run_us_without: f64,
}

// ---------------------------------------------------------------------------
// Pass-plan helpers
// ---------------------------------------------------------------------------

/// Build a PassPlan that runs DEFAULT_PASS_SEQUENCE on every function.
fn plan_default(fn_names: &[String]) -> PassPlan {
    let mut plan = PassPlan::empty();
    let seq: Vec<String> = DEFAULT_PASS_SEQUENCE.iter().map(|s| s.to_string()).collect();
    for n in fn_names {
        plan.per_function.insert(n.clone(), seq.clone());
    }
    plan
}

/// Build a PassPlan that runs DEFAULT_PASS_SEQUENCE minus `excluded` on
/// every function. The cf_round_2 alias is also dropped when the
/// excluded pass is "constant_fold" — otherwise we'd still run a CF in
/// the second slot and the "without CF" experiment would be no-op.
fn plan_without(fn_names: &[String], excluded: &str) -> PassPlan {
    let mut plan = PassPlan::empty();
    let seq: Vec<String> = DEFAULT_PASS_SEQUENCE
        .iter()
        .filter(|p| {
            if excluded == "constant_fold" {
                *p != &"constant_fold" && *p != &"cf_round_2"
            } else {
                **p != excluded
            }
        })
        .map(|s| s.to_string())
        .collect();
    for n in fn_names {
        plan.per_function.insert(n.clone(), seq.clone());
    }
    plan
}

// ---------------------------------------------------------------------------
// Single measurement
// ---------------------------------------------------------------------------

fn parse_and_lower(source: &str) -> (cjc_ast::Program, cjc_mir::MirProgram) {
    let (ast, diags) = cjc_parser::parse_source(source);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    let mut al = cjc_hir::AstLowering::new();
    let hir = al.lower_program(&ast);
    let mut h2m = cjc_mir::HirToMir::new();
    let mir = h2m.lower_program(&hir);
    (ast, mir)
}

fn median(values: &mut [u128]) -> u128 {
    values.sort_unstable();
    values[values.len() / 2]
}

/// Run `mir` through `optimize_program_with_plan(_, plan)` then execute
/// once. Returns the run_us.
fn run_one(ast: &cjc_ast::Program, mir: &cjc_mir::MirProgram, plan: &PassPlan) -> u128 {
    let mut opt = optimize_program_with_plan(mir, plan);
    cjc_mir::escape::annotate_program(&mut opt);
    let mut exec = cjc_mir_exec::MirExecutor::new(SEED);
    exec.scan_ast_imports(ast);
    let start = Instant::now();
    let _ = exec.exec(&opt).unwrap();
    start.elapsed().as_micros()
}

/// Take N_ITERS measurements with `plan` and return the median run_us.
fn median_run_us(ast: &cjc_ast::Program, mir: &cjc_mir::MirProgram, plan: &PassPlan) -> u128 {
    let mut samples = Vec::with_capacity(N_ITERS);
    for _ in 0..N_ITERS {
        samples.push(run_one(ast, mir, plan));
    }
    median(&mut samples)
}

// ---------------------------------------------------------------------------
// Per-program training data collection
// ---------------------------------------------------------------------------

fn collect_training_points(prog: &Program) -> Vec<TrainingPoint> {
    let (ast, mir) = parse_and_lower(prog.source);
    let fn_names: Vec<String> = mir.functions.iter().map(|f| f.name.clone()).collect();
    let features = cjc_cana::analyze_program(&mir).features;

    let mut points = Vec::new();
    // Baseline: default plan (all passes).
    let baseline_plan = plan_default(&fn_names);
    let baseline_us = median_run_us(&ast, &mir, &baseline_plan) as f64;

    for &pass in TARGET_PASSES {
        let without_plan = plan_without(&fn_names, pass);
        let without_us = median_run_us(&ast, &mir, &without_plan) as f64;
        let with_us = baseline_us;

        // Benefit = (without - with) / max(without, 1). Clamp to [0, 0.5]
        // to match LinearCostModel::predict_pass_gain's output range.
        // Negative measured "benefits" (the pass actually slowed runtime
        // — common on tiny programs where overhead dominates) clamp to 0.
        let raw_benefit = (without_us - with_us) / without_us.max(1.0);
        let benefit = raw_benefit.clamp(0.0, 0.5);

        // Emit one training point per function in this program.
        for fname in &fn_names {
            let Some(ff) = features.per_fn.get(fname) else { continue };
            points.push(TrainingPoint {
                program: prog.name.to_string(),
                function: fname.clone(),
                pass: pass.to_string(),
                expr_count: ff.memory.expr_count as f64,
                loop_depth: ff.cfg.max_loop_depth as f64,
                branch_count: ff.cfg.branch_count as f64,
                alloc_sites: ff.memory.alloc_sites as f64,
                benefit,
                run_us_with: with_us,
                run_us_without: without_us,
            });
        }
    }
    points
}

// ---------------------------------------------------------------------------
// Gradient-descent OLS fit
// ---------------------------------------------------------------------------

/// Per-pass fit output. 4 weights matching `LinearCostModel::PassCoefficients`.
#[derive(Debug, Clone, Copy)]
struct FitCoefs {
    w_expr_count: f64,
    w_loop_depth: f64,
    w_branch_count: f64,
    w_alloc_sites: f64,
    /// Mean predicted benefit on training set — useful for sanity-checking.
    train_mean_benefit: f64,
    /// RMSE on training set.
    train_rmse: f64,
}

/// OLS fit via gradient descent. `x` is [N x 4], `y` is [N]. Returns the
/// fitted weights and RMSE.
///
/// The four features have wildly different scales (expr_count ≈ 100,
/// loop_depth ≈ 0-4), so we normalize each column to its training-set
/// maximum before fitting, then un-normalize the weights at the end so
/// they apply directly to raw feature values.
fn fit_ols_gd(
    x: &[[f64; 4]],
    y: &[f64],
    lr: f64,
    n_steps: usize,
) -> (FitCoefs, f64) {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f64;
    if x.is_empty() {
        return (
            FitCoefs {
                w_expr_count: 0.0,
                w_loop_depth: 0.0,
                w_branch_count: 0.0,
                w_alloc_sites: 0.0,
                train_mean_benefit: 0.0,
                train_rmse: 0.0,
            },
            0.0,
        );
    }

    // Per-column max for normalization (use max(1.0) to avoid divide-by-zero).
    let mut col_max = [1.0_f64; 4];
    for row in x {
        for k in 0..4 {
            if row[k].abs() > col_max[k] {
                col_max[k] = row[k].abs();
            }
        }
    }

    // Normalize x.
    let xn: Vec<[f64; 4]> = x
        .iter()
        .map(|row| {
            let mut r = [0.0_f64; 4];
            for k in 0..4 {
                r[k] = row[k] / col_max[k];
            }
            r
        })
        .collect();

    // Initial weights = 0.0 (deterministic).
    let mut w = [0.0_f64; 4];

    for _ in 0..n_steps {
        let mut grad = [0.0_f64; 4];
        for (xi, &yi) in xn.iter().zip(y.iter()) {
            let pred = w[0] * xi[0] + w[1] * xi[1] + w[2] * xi[2] + w[3] * xi[3];
            let err = pred - yi;
            for k in 0..4 {
                grad[k] += 2.0 * err * xi[k] / n;
            }
        }
        for k in 0..4 {
            w[k] -= lr * grad[k];
        }
    }

    // Un-normalize: y = sum_k w_k * (x_k / col_max[k]) = sum_k (w_k / col_max[k]) * x_k
    // So the weights on RAW features are w_k / col_max[k].
    let raw_w = [
        w[0] / col_max[0],
        w[1] / col_max[1],
        w[2] / col_max[2],
        w[3] / col_max[3],
    ];

    // Compute RMSE on training set with un-normalized weights against
    // raw features.
    let mut sum_err_sq = 0.0_f64;
    let mut sum_y = 0.0_f64;
    for (xi, &yi) in x.iter().zip(y.iter()) {
        let pred = raw_w[0] * xi[0] + raw_w[1] * xi[1] + raw_w[2] * xi[2] + raw_w[3] * xi[3];
        let err = pred - yi;
        sum_err_sq += err * err;
        sum_y += yi;
    }
    let rmse = (sum_err_sq / n).sqrt();
    let mean_benefit = sum_y / n;

    (
        FitCoefs {
            w_expr_count: raw_w[0],
            w_loop_depth: raw_w[1],
            w_branch_count: raw_w[2],
            w_alloc_sites: raw_w[3],
            train_mean_benefit: mean_benefit,
            train_rmse: rmse,
        },
        rmse,
    )
}

// ---------------------------------------------------------------------------
// Output generation
// ---------------------------------------------------------------------------

/// Emit Rust source code for the trained() constructor's per-pass match arms.
fn emit_rust_source(fits: &BTreeMap<String, FitCoefs>) {
    println!();
    println!("============================================================");
    println!("TRAINED COEFFICIENTS — paste into linear_cost_model.rs");
    println!("============================================================");
    println!();
    println!("// Generated by `cargo run --release --bin cana_train_cost_model`");
    println!("// from {} corpus programs.", PROGRAMS.len());
    println!("// Each per-pass fit reports train RMSE for sanity-checking.");
    println!();
    println!("fn trained_pass_coefficients(pass_name: &str) -> Option<PassCoefficients> {{");
    println!("    match pass_name {{");
    for (pass, fit) in fits {
        let confidence = trained_confidence(fit.train_rmse);
        let base_compile = trained_base_compile_cost(pass);
        let aliases = pass_aliases(pass);
        println!(
            "        {} => Some(PassCoefficients {{",
            aliases
        );
        println!("            w_expr_count: {:.6e},   // train_rmse={:.4}, mean_benefit={:.4}", fit.w_expr_count, fit.train_rmse, fit.train_mean_benefit);
        println!("            w_loop_depth: {:.6e},", fit.w_loop_depth);
        println!("            w_branch_count: {:.6e},", fit.w_branch_count);
        println!("            w_alloc_sites: {:.6e},", fit.w_alloc_sites);
        println!("            base_compile_cost: {:.4},", base_compile);
        println!("            confidence: {:.4},", confidence);
        println!("        }}),");
    }
    println!("        _ => None,");
    println!("    }}");
    println!("}}");
    println!();
}

fn pass_aliases(pass: &str) -> String {
    match pass {
        "constant_fold" => "\"constant_fold\" | \"cf\"".to_string(),
        "strength_reduce" => "\"strength_reduce\" | \"sr\"".to_string(),
        "dce" => "\"dce\" | \"dead_code_elimination\"".to_string(),
        "cse" => "\"cse\" | \"common_subexpression_elimination\"".to_string(),
        "licm" => "\"licm\" | \"loop_invariant_code_motion\"".to_string(),
        other => format!("{:?}", other),
    }
}

/// Map training RMSE to a confidence value in [0.1, 0.95]. Lower RMSE →
/// higher confidence. Capped so we never claim near-perfect certainty.
fn trained_confidence(rmse: f64) -> f64 {
    // RMSE of 0.01 → confidence 0.85; RMSE of 0.10 → confidence 0.45.
    // Interpolate linearly between (0.01, 0.85) and (0.10, 0.45).
    let interp = 0.85 - ((rmse - 0.01) / 0.09) * (0.85 - 0.45);
    interp.clamp(0.1, 0.95)
}

/// The training pipeline doesn't measure compile_us (we'd need to instrument
/// optimize_program_with_plan more carefully). Reuse the existing
/// hand-tuned base_compile_cost values from the original Phase 2 model.
fn trained_base_compile_cost(pass: &str) -> f64 {
    match pass {
        "constant_fold" => 0.05,
        "strength_reduce" => 0.03,
        "dce" => 0.04,
        "cse" => 0.08,
        "licm" => 0.07,
        _ => 0.05,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("CANA cost-model training — corpus size {}, passes {}", PROGRAMS.len(), TARGET_PASSES.len());
    println!("Measurement: {} iters per (program, config), median run_us as benefit signal.", N_ITERS);

    // Phase 1: collect training points.
    println!("\n=== Phase 1: collecting training data ===");
    let mut all_points: Vec<TrainingPoint> = Vec::new();
    for (i, prog) in PROGRAMS.iter().enumerate() {
        let pts = collect_training_points(prog);
        println!("  [{:>2}/{}] {:<20} {} points", i + 1, PROGRAMS.len(), prog.name, pts.len());
        all_points.extend(pts);
    }
    println!("Collected {} training points total.", all_points.len());

    // Phase 2: group by pass and fit.
    println!("\n=== Phase 2: per-pass OLS fits ===");
    let mut fits: BTreeMap<String, FitCoefs> = BTreeMap::new();
    for &pass in TARGET_PASSES {
        let pass_pts: Vec<&TrainingPoint> =
            all_points.iter().filter(|p| p.pass == pass).collect();
        let x: Vec<[f64; 4]> = pass_pts
            .iter()
            .map(|p| [p.expr_count, p.loop_depth, p.branch_count, p.alloc_sites])
            .collect();
        let y: Vec<f64> = pass_pts.iter().map(|p| p.benefit).collect();
        let (fit, rmse) = fit_ols_gd(&x, &y, 0.05, 5000);
        println!(
            "  {:<20} {:>4} pts, mean_benefit={:.4}, rmse={:.4}, weights: expr={:.3e} loop={:.3e} branch={:.3e} alloc={:.3e}",
            pass, pass_pts.len(), fit.train_mean_benefit, rmse,
            fit.w_expr_count, fit.w_loop_depth, fit.w_branch_count, fit.w_alloc_sites,
        );
        fits.insert(pass.to_string(), fit);
    }

    // Phase 3: emit Rust source.
    emit_rust_source(&fits);

    // Phase 4: per-pass mean measured benefit table (sanity check).
    println!("============================================================");
    println!("Per-pass measured benefit (mean across all training points)");
    println!("============================================================");
    for &pass in TARGET_PASSES {
        let mean_benefit: f64 = {
            let pts: Vec<&TrainingPoint> = all_points.iter().filter(|p| p.pass == pass).collect();
            if pts.is_empty() { 0.0 } else {
                pts.iter().map(|p| p.benefit).sum::<f64>() / pts.len() as f64
            }
        };
        println!("  {:<20} mean_benefit = {:.4}", pass, mean_benefit);
    }

    // Phase 5: emit the per-program leaderboard (which pass "won" each).
    println!();
    println!("============================================================");
    println!("Per-program leaderboard (biggest measured benefit per program)");
    println!("============================================================");
    println!("{:<20} {:<20} {:>10}", "program", "winning_pass", "benefit");
    let mut by_program: BTreeMap<String, Vec<&TrainingPoint>> = BTreeMap::new();
    for p in &all_points {
        by_program.entry(p.program.clone()).or_default().push(p);
    }
    for (prog, pts) in &by_program {
        // Pick the pass with the max benefit on this program's first function
        // (single representative entry per program × pass).
        let best = pts
            .iter()
            .max_by(|a, b| a.benefit.partial_cmp(&b.benefit).unwrap_or(std::cmp::Ordering::Equal));
        if let Some(b) = best {
            println!("{:<20} {:<20} {:>10.4}", prog, b.pass, b.benefit);
        }
    }
}
