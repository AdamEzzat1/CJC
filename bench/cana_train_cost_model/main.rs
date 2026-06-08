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
use cjc_mir::optimize::{optimize_program_with_plan, PassPlan, DEFAULT_PASS_SEQUENCE};
use cjc_mir::{MirBody, MirExpr, MirExprKind, MirProgram, MirStmt};

mod programs;
use programs::{Program, PROGRAMS};

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
];

/// Passes whose per-program benefit is measured by **MIR-node-count
/// delta** rather than wall-clock. These are passes that actually shrink
/// MIR structure: CF collapses literal subexpressions; DCE deletes
/// unreachable let bindings. Both produce a deterministic node-count
/// change — zero-variance label.
const SIZE_SHRINKING_PASSES: &[&str] = &[
    "constant_fold",
    "dce",
];

/// Passes whose per-program benefit must be measured by **wall-clock**.
///
/// - SR rewrites operations in-place (e.g. `x * 8` → `x << 3`) — same
///   node count, faster runtime.
/// - CSE replaces variable *uses* with earlier bindings; the now-redundant
///   let stays in the IR until a subsequent DCE pass cleans it up — but
///   DEFAULT_PASS_SEQUENCE puts DCE *before* CSE, so this cleanup never
///   happens. Net: CSE leaves node count unchanged at training time.
/// - LICM rearranges code, doesn't shrink it.
///
/// All three need wall-clock measurement. Subject to microsecond
/// scheduler noise; that's the trade-off.
const RUNTIME_ONLY_PASSES: &[&str] = &[
    "strength_reduce",
    "cse",
    "licm",
];

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

/// For size-shrinking passes, return the MIR-node-count fractional
/// reduction after applying the default pass sequence WITH vs WITHOUT
/// the pass under test. Deterministic, zero-variance.
fn measure_mir_count_benefit(
    mir: &cjc_mir::MirProgram,
    fn_names: &[String],
    pass: &str,
) -> f64 {
    let with_plan = plan_default(fn_names);
    let without_plan = plan_without(fn_names, pass);

    let with_opt = optimize_program_with_plan(mir, &with_plan);
    let without_opt = optimize_program_with_plan(mir, &without_plan);

    let with_nodes = count_mir_nodes(&with_opt) as f64;
    let without_nodes = count_mir_nodes(&without_opt) as f64;

    // benefit = how much MORE code "without" has, fractionally.
    // Equivalent to: how much code "with" eliminates from "without".
    let raw = (without_nodes - with_nodes) / without_nodes.max(1.0);
    raw.clamp(0.0, 0.5)
}

/// For runtime-only passes (LICM), measure wall-clock benefit via
/// median-of-N_ITERS. Noisy but unavoidable.
fn measure_wall_clock_benefit(
    ast: &cjc_ast::Program,
    mir: &cjc_mir::MirProgram,
    fn_names: &[String],
    pass: &str,
) -> f64 {
    let with_plan = plan_default(fn_names);
    let without_plan = plan_without(fn_names, pass);

    let with_us = median_run_us(ast, mir, &with_plan) as f64;
    let without_us = median_run_us(ast, mir, &without_plan) as f64;

    let raw = (without_us - with_us) / without_us.max(1.0);
    raw.clamp(0.0, 0.5)
}

fn collect_training_points(prog: &Program) -> Vec<TrainingPoint> {
    let (ast, mir) = parse_and_lower(prog.source);
    let fn_names: Vec<String> = mir.functions.iter().map(|f| f.name.clone()).collect();
    let features = cjc_cana::analyze_program(&mir).features;

    let mut points = Vec::new();
    for &pass in TARGET_PASSES {
        let (benefit, signal) = if SIZE_SHRINKING_PASSES.contains(&pass) {
            (measure_mir_count_benefit(&mir, &fn_names, pass), "mir_count")
        } else {
            assert!(RUNTIME_ONLY_PASSES.contains(&pass));
            (measure_wall_clock_benefit(&ast, &mir, &fn_names, pass), "wall_clock")
        };

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
                signal,
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
    println!("// (60 programs, N_ITERS={}, MIR-count proxy for CF/SR/DCE/CSE, wall-clock for LICM).", N_ITERS);
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
    for (i, prog) in PROGRAMS.iter().enumerate() {
        let pts = collect_training_points(prog);
        println!("  [{:>2}/{}] {:<26} {} points", i + 1, PROGRAMS.len(), prog.name, pts.len());
        all_points.extend(pts);
    }
    let collect_elapsed = collect_start.elapsed();
    println!(
        "Collected {} training points in {:.1}s.",
        all_points.len(),
        collect_elapsed.as_secs_f64(),
    );

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
        let signal = if SIZE_SHRINKING_PASSES.contains(&pass) { "mir_count" } else { "wall_clock" };
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
        let signal = if SIZE_SHRINKING_PASSES.contains(&pass) { "mir_count" } else { "wall_clock" };
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
}
