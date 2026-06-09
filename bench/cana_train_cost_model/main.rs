//! CANA cost-model training harness — **v2 with the four improvements.**
//!
//! Changes vs v1:
//!
//!   1. **N_ITERS 5 → 21.** sqrt(21/5) ≈ 2× reduction in wall-clock noise
//!      from median-of-N. Bench wall-clock grows ~4×, still well under
//!      10 minutes total.
//!
//!   2. **MIR-instruction-count proxy for size-shrinking passes.**
//!      CF, DCE, CSE, SR all reduce code size; the label is now the
//!      structural-size delta from each pass, measured deterministically.
//!      Zero variance. LICM keeps wall-clock (it rearranges code, doesn't
//!      shrink it).
//!
//!   3. **Bigger workloads.** Inner-loop iteration counts in
//!      `programs.rs` scaled 10-100×. Per-program runtime is now in the
//!      100μs-10ms range, putting per-pass benefits well above the
//!      scheduler/cache noise floor.
//!
//!   4. **60-program corpus.** 3.3× more training data with denser
//!      feature-space coverage; each pass sees 10-15 affinity-aligned
//!      programs instead of 2-4.
//!
//! The combination should drop RMSE significantly per-pass. The findings
//! doc records before/after numbers after each re-run.

use std::collections::BTreeMap;
use std::time::Instant;

use cjc_cana::features::FnFeatures;
use cjc_mir::optimize::{
    apply_pass_with_diagnostics, optimize_program_with_plan, PassPlan, DEFAULT_PASS_SEQUENCE,
};
use cjc_mir::{MirBody, MirExpr, MirExprKind, MirProgram, MirStmt};

mod programs;
use programs::{Program, PROGRAMS};

mod external_corpus;
use external_corpus::EXTERNAL_PROGRAMS;

// ---------------------------------------------------------------------------
// Measurement constants
// ---------------------------------------------------------------------------

const N_ITERS: usize = 21;
const SEED: u64 = 42;

const TARGET_PASSES: &[&str] = &[
    "constant_fold",
    "strength_reduce",
    "dce",
    "cse",
    "licm",
    // `loop_unroll` shipped with hand-tuned default coefficients and
    // placeholder trained coefficients. Add unrollable programs to the
    // corpus (see `bench/cana_train_cost_model/programs.rs` — short
    // fixed-trip-count `while` patterns) before regenerating, otherwise
    // the OLS fit has too few points to be meaningful and the resulting
    // confidence will be near zero.
    "loop_unroll",
];

/// **Option A** (cost-model training findings doc §4): every pass gets a
/// deterministic signal sourced from its own native diagnostic count.
///
/// For each (program, pass) we run a fresh copy of the function through
/// `apply_pass_with_diagnostics(pass, &mut fn)` and use:
///
///   benefit = changes_applied / max(nodes_before, 1)
///
/// where `changes_applied` is the pass's NATIVE count of rewrites:
///   * CF: node-count delta (CF only collapses literal subexpressions —
///     delta is exact)
///   * SR: count of `try_strength_reduce` successes
///   * DCE: node-count delta
///   * CSE: count of variable-replacement applications
///   * LICM: count of statements hoisted out of `while` loops
///
/// This eliminates wall-clock measurement entirely. Every signal is
/// deterministic, reproducible bit-for-bit across runs and platforms,
/// and tracks each pass's own understanding of its work.
const ALL_PASSES_DETERMINISTIC: bool = true;

// ---------------------------------------------------------------------------
// Training data shape
// ---------------------------------------------------------------------------

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
    // Label: measured benefit in [0.0, 0.5].
    benefit: f64,
    /// Which signal produced the benefit: "mir_count" (deterministic) or
    /// "wall_clock" (noisy). Useful for downstream analysis.
    #[allow(dead_code)]
    signal: &'static str,
}

// ---------------------------------------------------------------------------
// Pass-plan helpers
// ---------------------------------------------------------------------------

fn plan_default(fn_names: &[String]) -> PassPlan {
    let mut plan = PassPlan::empty();
    let seq: Vec<String> = DEFAULT_PASS_SEQUENCE.iter().map(|s| s.to_string()).collect();
    for n in fn_names {
        plan.per_function.insert(n.clone(), seq.clone());
    }
    plan
}

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
// MIR node counting (the new deterministic signal)
// ---------------------------------------------------------------------------

/// Count all MirStmt + MirExpr nodes in a program. Used as a
/// deterministic proxy for "how much code is here" — size-shrinking
/// passes reduce this count; the delta is the benefit signal.
fn count_mir_nodes(program: &MirProgram) -> usize {
    let mut total = 0;
    for func in &program.functions {
        total += count_body_nodes(&func.body);
    }
    total
}

fn count_body_nodes(body: &MirBody) -> usize {
    let mut count = 0;
    for stmt in &body.stmts {
        count += count_stmt_nodes(stmt);
    }
    if let Some(e) = &body.result {
        count += count_expr_nodes(e);
    }
    count
}

fn count_stmt_nodes(stmt: &MirStmt) -> usize {
    1 + match stmt {
        MirStmt::Let { init, .. } => count_expr_nodes(init),
        MirStmt::Expr(e) => count_expr_nodes(e),
        MirStmt::If { cond, then_body, else_body } => {
            count_expr_nodes(cond)
                + count_body_nodes(then_body)
                + else_body.as_ref().map(count_body_nodes).unwrap_or(0)
        }
        MirStmt::While { cond, body } => count_expr_nodes(cond) + count_body_nodes(body),
        MirStmt::Return(Some(e)) => count_expr_nodes(e),
        MirStmt::Return(None) | MirStmt::Break | MirStmt::Continue => 0,
        MirStmt::NoGcBlock(b) => count_body_nodes(b),
    }
}

fn count_expr_nodes(expr: &MirExpr) -> usize {
    1 + match &expr.kind {
        MirExprKind::Binary { left, right, .. } => {
            count_expr_nodes(left) + count_expr_nodes(right)
        }
        MirExprKind::Unary { operand, .. } => count_expr_nodes(operand),
        MirExprKind::Call { callee, args } => {
            count_expr_nodes(callee) + args.iter().map(count_expr_nodes).sum::<usize>()
        }
        MirExprKind::Assign { target, value } => {
            count_expr_nodes(target) + count_expr_nodes(value)
        }
        MirExprKind::Field { object, .. } => count_expr_nodes(object),
        MirExprKind::Index { object, index } => {
            count_expr_nodes(object) + count_expr_nodes(index)
        }
        MirExprKind::ArrayLit(es) | MirExprKind::TupleLit(es) => {
            es.iter().map(count_expr_nodes).sum::<usize>()
        }
        MirExprKind::StructLit { fields, .. } => {
            fields.iter().map(|(_, e)| count_expr_nodes(e)).sum::<usize>()
        }
        MirExprKind::MakeClosure { captures, .. } => {
            captures.iter().map(count_expr_nodes).sum::<usize>()
        }
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Single measurement helpers
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

fn run_one(ast: &cjc_ast::Program, mir: &cjc_mir::MirProgram, plan: &PassPlan) -> u128 {
    let mut opt = optimize_program_with_plan(mir, plan);
    cjc_mir::escape::annotate_program(&mut opt);
    let mut exec = cjc_mir_exec::MirExecutor::new(SEED);
    exec.scan_ast_imports(ast);
    let start = Instant::now();
    let _ = exec.exec(&opt).unwrap();
    start.elapsed().as_micros()
}

fn median_run_us(ast: &cjc_ast::Program, mir: &cjc_mir::MirProgram, plan: &PassPlan) -> u128 {
    let mut samples = Vec::with_capacity(N_ITERS);
    for _ in 0..N_ITERS {
        samples.push(run_one(ast, mir, plan));
    }
    median(&mut samples)
}

// ---------------------------------------------------------------------------
// Per-program label collection — switches signal based on pass
// ---------------------------------------------------------------------------

/// **Option A signal — per-function, per-pass diagnostic counts.**
///
/// We run the target pass alone on a fresh copy of each function in the
/// program (default-sequence-optimized first, so we see what the pass
/// does on already-CF'd/DCE'd code — closest to the real pipeline
/// position). The pass's own `changes_applied` from
/// `PassDiagnostics` is the numerator; the function's pre-pass node
/// count is the denominator. The result is the function-level benefit.
fn measure_pass_native_benefit_per_function(
    mir: &cjc_mir::MirProgram,
    pass: &str,
) -> Vec<(String, f64)> {
    let mut results = Vec::new();
    // Pre-optimize through the default sequence MINUS the target pass.
    // This is what the function looks like just before our target pass
    // would run, so the diagnostic count reflects realistic opportunity.
    let fn_names: Vec<String> = mir.functions.iter().map(|f| f.name.clone()).collect();
    let pre_plan = plan_without(&fn_names, pass);
    let pre_optimized = optimize_program_with_plan(mir, &pre_plan);

    for func in &pre_optimized.functions {
        let mut clone = func.clone();
        let Some(d) = apply_pass_with_diagnostics(pass, &mut clone) else {
            // Unknown pass — shouldn't happen for TARGET_PASSES, skip.
            continue;
        };
        let raw = if d.nodes_before == 0 {
            0.0
        } else {
            (d.changes_applied as f64) / (d.nodes_before as f64)
        };
        let benefit = raw.clamp(0.0, 0.5);
        results.push((func.name.clone(), benefit));
    }
    results
}

fn collect_training_points(prog: &Program) -> Vec<TrainingPoint> {
    let (_ast, mir) = parse_and_lower(prog.source);
    let features = cjc_cana::analyze_program(&mir).features;

    let mut points = Vec::new();
    for &pass in TARGET_PASSES {
        let per_fn_benefits = measure_pass_native_benefit_per_function(&mir, pass);
        for (fname, benefit) in per_fn_benefits {
            let Some(ff) = features.per_fn.get(&fname) else { continue };
            points.push(TrainingPoint {
                program: prog.name.to_string(),
                function: fname,
                pass: pass.to_string(),
                expr_count: ff.memory.expr_count as f64,
                loop_depth: ff.cfg.max_loop_depth as f64,
                branch_count: ff.cfg.branch_count as f64,
                alloc_sites: ff.memory.alloc_sites as f64,
                benefit,
                signal: "pass_native",
            });
        }
    }
    points
}

// ---------------------------------------------------------------------------
// Gradient-descent OLS fit (unchanged from v1)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct FitCoefs {
    w_expr_count: f64,
    w_loop_depth: f64,
    w_branch_count: f64,
    w_alloc_sites: f64,
    train_mean_benefit: f64,
    train_rmse: f64,
}

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

    let mut col_max = [1.0_f64; 4];
    for row in x {
        for k in 0..4 {
            if row[k].abs() > col_max[k] {
                col_max[k] = row[k].abs();
            }
        }
    }

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

    let raw_w = [
        w[0] / col_max[0],
        w[1] / col_max[1],
        w[2] / col_max[2],
        w[3] / col_max[3],
    ];

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

/// Print a per-feature distribution report over the training corpus.
///
/// For each of the 4 features (expr_count, loop_depth, branch_count,
/// alloc_sites) we report min/q25/q50/q75/max and the unique-value count.
/// Features with `unique <= 2` or `max == 0` get a WARNING — they
/// effectively don't contribute signal to the OLS fit, so the
/// corresponding coefficient is meaningless and the model is implicitly
/// extrapolating on any real program that varies along that axis.
///
/// Implements §3A.4 (steps 1-2) from the cost-model handoff. Step 3-4
/// (corpus augmentation + drift comparison) are deferred follow-up.
fn print_feature_audit(points: &[TrainingPoint]) {
    println!();
    println!("============================================================");
    println!("§3A.4 — Feature distribution audit");
    println!("============================================================");
    if points.is_empty() {
        println!("(no points — skipping)");
        return;
    }

    let n = points.len();
    let features: [(&str, fn(&TrainingPoint) -> f64); 4] = [
        ("expr_count", |p| p.expr_count),
        ("loop_depth", |p| p.loop_depth),
        ("branch_count", |p| p.branch_count),
        ("alloc_sites", |p| p.alloc_sites),
    ];

    println!(
        "{:<14} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8}",
        "feature", "min", "q25", "q50", "q75", "max", "unique",
    );
    println!("{}", "-".repeat(66));

    let mut warnings: Vec<String> = Vec::new();
    for (name, get) in features {
        let mut vals: Vec<f64> = points.iter().map(get).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let min = vals[0];
        let max = *vals.last().unwrap();
        let q25 = vals[n / 4];
        let q50 = vals[n / 2];
        let q75 = vals[(3 * n) / 4];
        let unique: usize = {
            let mut seen: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
            for v in &vals {
                seen.insert(v.to_bits());
            }
            seen.len()
        };

        println!(
            "{:<14} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>8}",
            name, min, q25, q50, q75, max, unique,
        );

        if unique <= 2 {
            warnings.push(format!(
                "  WARNING: feature `{}` has only {} unique value(s) in corpus.\n           Trained coefficient is effectively meaningless — the OLS fit\n           cannot distinguish this column from a constant. Real programs\n           that vary along `{}` will get extrapolated predictions.",
                name, unique, name,
            ));
        }
        if max == 0.0 {
            warnings.push(format!(
                "  WARNING: feature `{}` is zero across the entire corpus.\n           Coefficient `w_{}` is mathematically free; OLS sets it to 0.\n           Adds zero signal to predictions on programs where `{} > 0`.",
                name, name, name,
            ));
        }
    }

    if warnings.is_empty() {
        println!();
        println!("All four features have variance > 2 and non-zero range.");
        println!("Trained model is not blind on any feature dimension.");
    } else {
        println!();
        for w in &warnings {
            println!("{}", w);
        }
        println!();
        println!(
            "Total feature blind spots: {}. The trained model is implicitly",
            warnings.len(),
        );
        println!("extrapolating on these dimensions. §3A.4 follow-up: extend the");
        println!("corpus with programs that exercise these features non-trivially");
        println!("(see docs/cana/CANA_COST_MODEL_TRAINING_FINDINGS.md §12).");
    }
}

/// Evaluate a fitted model on held-out data. Returns RMSE in raw (unnormalized)
/// feature space, matching what `FitCoefs` stores.
fn evaluate_fit(fit: &FitCoefs, test_pts: &[&TrainingPoint]) -> f64 {
    if test_pts.is_empty() {
        return 0.0;
    }
    let mut sum_err_sq = 0.0_f64;
    for p in test_pts {
        let pred = fit.w_expr_count * p.expr_count
            + fit.w_loop_depth * p.loop_depth
            + fit.w_branch_count * p.branch_count
            + fit.w_alloc_sites * p.alloc_sites;
        let err = pred - p.benefit;
        sum_err_sq += err * err;
    }
    (sum_err_sq / test_pts.len() as f64).sqrt()
}

// ---------------------------------------------------------------------------
// Output generation (unchanged)
// ---------------------------------------------------------------------------

fn emit_rust_source(fits: &BTreeMap<String, FitCoefs>) {
    println!();
    println!("============================================================");
    println!("TRAINED COEFFICIENTS — paste into linear_cost_model.rs");
    println!("============================================================");
    println!();
    println!("// Generated by `cargo run --release --bin cana_train_cost_model` v2");
    println!(
        "// ({} programs, N_ITERS={}, pass-native diagnostic count signal for every pass).",
        PROGRAMS.len(),
        N_ITERS,
    );
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
        "loop_unroll" => "\"loop_unroll\" | \"unroll\"".to_string(),
        other => format!("{:?}", other),
    }
}

/// Map training RMSE to a confidence value in [0.1, 0.95]. With the v2
/// improvements (MIR-count signal for 4 of 5 passes), most RMSEs should
/// drop into the 0.01-0.05 range — confidences ascend correspondingly.
fn trained_confidence(rmse: f64) -> f64 {
    let interp = 0.85 - ((rmse - 0.01) / 0.09) * (0.85 - 0.45);
    interp.clamp(0.1, 0.95)
}

fn trained_base_compile_cost(pass: &str) -> f64 {
    match pass {
        "constant_fold" => 0.05,
        "strength_reduce" => 0.03,
        "dce" => 0.04,
        "cse" => 0.08,
        "licm" => 0.07,
        "loop_unroll" => 0.06,
        _ => 0.05,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!(
        "CANA cost-model training v2 — {} programs, {} passes, N_ITERS={}.",
        PROGRAMS.len(),
        TARGET_PASSES.len(),
        N_ITERS,
    );
    println!("Signal: MIR-count delta for CF/SR/DCE/CSE; wall-clock for LICM.");

    println!("\n=== Phase 1: collecting training data ===");
    let collect_start = Instant::now();
    let mut all_points: Vec<TrainingPoint> = Vec::new();
    let mut train_points: Vec<TrainingPoint> = Vec::new();
    let mut test_points: Vec<TrainingPoint> = Vec::new();
    for (i, prog) in PROGRAMS.iter().enumerate() {
        let pts = collect_training_points(prog);
        let is_test = i % 4 == 0;
        println!(
            "  [{:>2}/{}] {:<26} {} points{}",
            i + 1,
            PROGRAMS.len(),
            prog.name,
            pts.len(),
            if is_test { " [test]" } else { "" },
        );
        all_points.extend(pts.iter().cloned());
        if is_test {
            test_points.extend(pts);
        } else {
            train_points.extend(pts);
        }
    }
    let collect_elapsed = collect_start.elapsed();
    println!(
        "Collected {} training points in {:.1}s.",
        all_points.len(),
        collect_elapsed.as_secs_f64(),
    );
    println!(
        "  → train: {} points, test: {} points",
        train_points.len(),
        test_points.len(),
    );

    print_feature_audit(&all_points);

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
        let signal = "pass_native";
        println!(
            "  {:<20} {:>4} pts, signal={:<10} mean_benefit={:.4}, rmse={:.4}",
            pass, pass_pts.len(), signal, fit.train_mean_benefit, rmse,
        );
        fits.insert(pass.to_string(), fit);
    }

    emit_rust_source(&fits);

    println!("============================================================");
    println!("Per-pass mean measured benefit");
    println!("============================================================");
    for &pass in TARGET_PASSES {
        let pts: Vec<&TrainingPoint> = all_points.iter().filter(|p| p.pass == pass).collect();
        let mean_benefit: f64 = if pts.is_empty() { 0.0 }
            else { pts.iter().map(|p| p.benefit).sum::<f64>() / pts.len() as f64 };
        let signal = "pass_native";
        println!("  {:<20} signal={:<10} mean_benefit = {:.4}", pass, signal, mean_benefit);
    }

    println!();
    println!("============================================================");
    println!("Per-program leaderboard (best-measured pass per program)");
    println!("============================================================");
    println!("{:<26} {:<20} {:>10}", "program", "winning_pass", "benefit");
    let mut by_program: BTreeMap<String, Vec<&TrainingPoint>> = BTreeMap::new();
    for p in &all_points {
        by_program.entry(p.program.clone()).or_default().push(p);
    }
    for (prog, pts) in &by_program {
        let best = pts
            .iter()
            .max_by(|a, b| a.benefit.partial_cmp(&b.benefit).unwrap_or(std::cmp::Ordering::Equal));
        if let Some(b) = best {
            println!("{:<26} {:<20} {:>10.4}", prog, b.pass, b.benefit);
        }
    }

    // =========================================================================
    // Phase 3: Held-out validation (§3A.1 from CANA handoff)
    // =========================================================================
    //
    // The Phase 2 fit above uses the entire 73-program corpus and emits its
    // coefficients into linear_cost_model.rs. That fit's RMSE is *training*
    // RMSE — what the model gets right on the corpus it saw. It tells us
    // nothing about whether the model generalizes to held-out programs.
    //
    // Phase 3 splits the corpus into train (75%) and test (25%) by program
    // index — every 4th program goes to test. We re-fit using train-only data,
    // then evaluate the resulting coefficients on the test set. The gap
    // (test_rmse - train_rmse) and ratio (test_rmse / train_rmse) tell us
    // whether the "14× headline" from the full-corpus fit is real
    // generalization or an artifact of training on the evaluation set.
    //
    // Interpretation thresholds (from handoff §3A.1):
    //   * ratio ≤ 1.5  → model generalizes well; trust the full-corpus fit.
    //   * 1.5 < ratio ≤ 2.0  → mild overfitting; usable but flag in docs.
    //   * ratio > 2.0  → overfit; the headline doesn't hold, downgrade claims.
    println!();
    println!("============================================================");
    println!("§3A.1 — Held-out validation (train/test split)");
    println!("============================================================");
    let test_program_count = PROGRAMS
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 4 == 0)
        .count();
    let train_program_count = PROGRAMS.len() - test_program_count;
    println!("Split rule: every 4th program (indices 0, 4, 8, ...) → test.");
    println!(
        "Train: {} programs ({} points)  |  Test: {} programs ({} points)",
        train_program_count,
        train_points.len(),
        test_program_count,
        test_points.len(),
    );
    println!();
    println!(
        "{:<20} {:>7} {:>7} {:>11} {:>11} {:>9} {:>6}",
        "pass", "N_train", "N_test", "train_rmse", "test_rmse", "gap", "ratio",
    );
    println!("{}", "-".repeat(76));

    let mut held_out_fits: BTreeMap<String, (FitCoefs, f64)> = BTreeMap::new();
    for &pass in TARGET_PASSES {
        let train_pts: Vec<&TrainingPoint> =
            train_points.iter().filter(|p| p.pass == pass).collect();
        let test_pts: Vec<&TrainingPoint> =
            test_points.iter().filter(|p| p.pass == pass).collect();

        let train_x: Vec<[f64; 4]> = train_pts
            .iter()
            .map(|p| [p.expr_count, p.loop_depth, p.branch_count, p.alloc_sites])
            .collect();
        let train_y: Vec<f64> = train_pts.iter().map(|p| p.benefit).collect();
        let (train_fit, train_rmse) = fit_ols_gd(&train_x, &train_y, 0.05, 5000);

        let test_rmse = evaluate_fit(&train_fit, &test_pts);
        let gap = test_rmse - train_rmse;
        let ratio = if train_rmse > 1e-9 {
            test_rmse / train_rmse
        } else {
            f64::NAN
        };

        println!(
            "{:<20} {:>7} {:>7} {:>11.4} {:>11.4} {:>+9.4} {:>6.2}",
            pass,
            train_pts.len(),
            test_pts.len(),
            train_rmse,
            test_rmse,
            gap,
            ratio,
        );
        held_out_fits.insert(pass.to_string(), (train_fit, test_rmse));
    }
    println!();
    println!("Verdict (per pass, by handoff §3A.1 threshold):");
    for &pass in TARGET_PASSES {
        let Some((fit, test_rmse)) = held_out_fits.get(pass) else { continue };
        let train_rmse = fit.train_rmse;
        let ratio = if train_rmse > 1e-9 {
            test_rmse / train_rmse
        } else {
            f64::NAN
        };
        let verdict = if ratio.is_nan() {
            "(no signal)"
        } else if ratio <= 1.5 {
            "generalizes well"
        } else if ratio <= 2.0 {
            "mild overfitting"
        } else {
            "overfit — downgrade claims"
        };
        println!("  {:<20} ratio={:>5.2}  →  {}", pass, ratio, verdict);
    }

    // =========================================================================
    // Phase 4: Cross-corpus validation (§3A.3 from CANA handoff)
    // =========================================================================
    //
    // §3A.1 measured generalization within our 73-program corpus by holding
    // out 25%. §3A.3 asks a harder question: does the trained model
    // generalize to programs hand-written by a *different* author for a
    // *different* purpose? We evaluate the full-corpus Phase 2 fit against
    // the 8 programs in bench/cana_pass_ordering (snapshotted in
    // external_corpus.rs). If the external RMSE is close to held-out RMSE
    // from Phase 3, the model is robust across authors. If external RMSE
    // is much worse, the model has overfit to the corpus author's style
    // (here: the integer-arithmetic emphasis from `programs.rs`).
    println!();
    println!("============================================================");
    println!("§3A.3 — Cross-corpus validation");
    println!("============================================================");
    println!(
        "External corpus: bench/cana_pass_ordering, {} programs (snapshotted",
        EXTERNAL_PROGRAMS.len(),
    );
    println!("in external_corpus.rs). Evaluating the full-corpus Phase 2 fit.");
    println!();

    let mut ext_points: Vec<TrainingPoint> = Vec::new();
    for (i, prog) in EXTERNAL_PROGRAMS.iter().enumerate() {
        let pts = collect_training_points(prog);
        println!(
            "  [{:>1}/{}] {:<14} {} points",
            i + 1,
            EXTERNAL_PROGRAMS.len(),
            prog.name,
            pts.len(),
        );
        ext_points.extend(pts);
    }
    println!("Collected {} external points.", ext_points.len());
    println!();
    println!(
        "{:<20} {:>6} {:>11} {:>10} {:>11} {:>10} {:>8}",
        "pass", "N_ext", "train_rmse", "ext_rmse", "held_out", "ext/trn", "ext/held",
    );
    println!("{}", "-".repeat(80));

    for &pass in TARGET_PASSES {
        let ext_pts: Vec<&TrainingPoint> = ext_points.iter().filter(|p| p.pass == pass).collect();
        let Some(full_fit) = fits.get(pass) else { continue };

        let ext_rmse = evaluate_fit(full_fit, &ext_pts);
        let train_rmse = full_fit.train_rmse;
        let held_out_rmse = held_out_fits.get(pass).map(|(_, r)| *r).unwrap_or(0.0);

        let ratio_train = if train_rmse > 1e-9 {
            ext_rmse / train_rmse
        } else {
            f64::NAN
        };
        let ratio_held = if held_out_rmse > 1e-9 {
            ext_rmse / held_out_rmse
        } else {
            f64::NAN
        };

        println!(
            "{:<20} {:>6} {:>11.4} {:>10.4} {:>11.4} {:>10.2} {:>8.2}",
            pass,
            ext_pts.len(),
            train_rmse,
            ext_rmse,
            held_out_rmse,
            ratio_train,
            ratio_held,
        );
    }
    println!();
    println!("Verdict (cross-corpus generalization):");
    for &pass in TARGET_PASSES {
        let ext_pts: Vec<&TrainingPoint> = ext_points.iter().filter(|p| p.pass == pass).collect();
        let Some(full_fit) = fits.get(pass) else { continue };
        let ext_rmse = evaluate_fit(full_fit, &ext_pts);
        let held_out_rmse = held_out_fits.get(pass).map(|(_, r)| *r).unwrap_or(0.0);
        let ratio = if held_out_rmse > 1e-9 {
            ext_rmse / held_out_rmse
        } else {
            f64::NAN
        };
        let verdict = if ratio.is_nan() {
            "(no signal)"
        } else if ratio <= 1.5 {
            "external matches held-out — robust"
        } else if ratio <= 2.5 {
            "external worse than held-out — corpus-style dependency"
        } else {
            "external much worse — model overfit to corpus author"
        };
        println!("  {:<20} ext/held={:>5.2}  →  {}", pass, ratio, verdict);
    }
}
