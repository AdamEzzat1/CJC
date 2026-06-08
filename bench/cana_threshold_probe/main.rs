//! §17 — Skip-threshold sensitivity probe.
//!
//! The `PassRanker::skip_threshold` defaults to
//! [`cjc_cana::DEFAULT_SKIP_THRESHOLD`] = 0.005. The §3A.2 PINN AB test
//! finding was that both default and trained rankers reject every
//! candidate pass on PINN at this threshold. The §16 cana_ab_corpus
//! finding revealed the trained ranker DOES produce decisions at the
//! same threshold on the 8 pass_ordering programs — so the threshold
//! isn't universally over-aggressive, just specifically blocking on
//! PINN-like workloads.
//!
//! This probe sweeps the threshold from 1e-2 down to 1e-6 and reports
//! how many Run-recommendations the trained ranker produces at each
//! value, on:
//!
//!   - PINN heat 1D (`examples/08_pinn_heat_equation.cjcl`, n_epochs
//!     substituted to 20 to match cana_ab_pinn).
//!   - Each of the 8 pass_ordering programs.
//!
//! Tells us at what threshold PINN starts emitting non-empty PassPlans,
//! and how aggressively the threshold needs to drop for that to happen.
//! That's the data needed to decide whether threshold tuning is a
//! workable activation strategy, or whether PINN's structural features
//! (small expr_count per function, narrow loop_depth and branch_count)
//! mean even a near-zero threshold won't produce meaningful decisions.

use cjc_cana::legality::PerPassLegalityGate;
use cjc_cana::pass_ranker::PassRanker;
use cjc_cana::{analyze_program, LinearCostModel};

// =============================================================================
// PINN workload (mirrored from cana_ab_pinn)
// =============================================================================

const PINN_FULL_SOURCE: &str = include_str!("../../examples/08_pinn_heat_equation.cjcl");

fn pinn_source_short() -> String {
    PINN_FULL_SOURCE.replace("let n_epochs = 80;", "let n_epochs = 20;")
}

// =============================================================================
// 8 pass_ordering programs (snapshot)
// =============================================================================

struct Program {
    name: &'static str,
    source: &'static str,
}

const PROG_ARITH: &str = r#"
fn compute(n: i64) -> i64 {
    let a: i64 = 10 * 5 + 2;
    let b: i64 = (a + 100) * 2;
    let c: i64 = b - 50 + n;
    return c + a + b;
}
print(compute(7));
"#;

const PROG_LOOP: &str = r#"
fn sum_to(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        total = total + i;
        i = i + 1;
    }
    return total;
}
print(sum_to(1000));
"#;

const PROG_NESTED: &str = r#"
fn nested(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let mut j: i64 = 0;
        while j < n {
            total = total + i * j;
            j = j + 1;
        }
        i = i + 1;
    }
    return total;
}
print(nested(30));
"#;

const PROG_MANY_FN: &str = r#"
fn add1(x: i64) -> i64 { return x + 1; }
fn add2(x: i64) -> i64 { return x + 2; }
fn add3(x: i64) -> i64 { return x + 3; }
fn mul2(x: i64) -> i64 { return x * 2; }
fn mul3(x: i64) -> i64 { return x * 3; }
fn driver() -> i64 {
    let mut r: i64 = 0;
    r = add1(r);
    r = add2(r);
    r = add3(r);
    r = mul2(r);
    r = mul3(r);
    return r;
}
print(driver());
"#;

const PROG_MIXED: &str = r#"
fn classify(n: i64) -> i64 {
    let mut sum: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let inc: i64 = if i * 2 > n { i } else { 0 };
        sum = sum + inc;
        i = i + 1;
    }
    return sum;
}
print(classify(40));
"#;

const PROG_FLOAT: &str = r#"
fn polynomial(x: f64) -> f64 {
    let a: f64 = 3.14;
    let b: f64 = 2.71;
    let c: f64 = 1.41;
    return a * x * x + b * x + c;
}
print(polynomial(1.5));
"#;

const PROG_RECURSIVE: &str = r#"
fn factorial(n: i64) -> i64 {
    let result: i64 = if n <= 1 { 1 } else { n * factorial(n - 1) };
    return result;
}
print(factorial(10));
"#;

const PROG_LARGE: &str = r#"
fn count_evens(n: i64) -> i64 {
    let mut c: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        if i * 2 / 2 == i {
            c = c + 1;
        }
        i = i + 1;
    }
    return c;
}
fn count_squares(n: i64) -> i64 {
    let mut c: i64 = 0;
    let mut i: i64 = 0;
    while i * i < n {
        c = c + 1;
        i = i + 1;
    }
    return c;
}
fn sum_to(n: i64) -> i64 {
    let mut s: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        s = s + i;
        i = i + 1;
    }
    return s;
}
fn combined(n: i64) -> i64 {
    let a: i64 = count_evens(n);
    let b: i64 = count_squares(n);
    let c: i64 = sum_to(n);
    return a + b + c;
}
print(combined(50));
"#;

const PASS_ORDERING_PROGRAMS: &[Program] = &[
    Program { name: "arith",     source: PROG_ARITH },
    Program { name: "loop",      source: PROG_LOOP },
    Program { name: "nested",    source: PROG_NESTED },
    Program { name: "many_fn",   source: PROG_MANY_FN },
    Program { name: "mixed",     source: PROG_MIXED },
    Program { name: "float",     source: PROG_FLOAT },
    Program { name: "recursive", source: PROG_RECURSIVE },
    Program { name: "large",     source: PROG_LARGE },
];

// =============================================================================
// Probe
// =============================================================================

const THRESHOLDS: &[f64] = &[
    0.01,
    0.005,   // DEFAULT — same as DEFAULT_SKIP_THRESHOLD
    0.002,
    0.001,
    0.0005,
    0.0001,
    0.00001,
    0.0,     // accept everything the cost model returns > 0
];

fn parse_and_lower(source: &str) -> cjc_mir::MirProgram {
    let (ast, diags) = cjc_parser::parse_source(source);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    let mut al = cjc_hir::AstLowering::new();
    let hir = al.lower_program(&ast);
    let mut h2m = cjc_mir::HirToMir::new();
    h2m.lower_program(&hir)
}

fn sweep_thresholds(program_name: &str, source: &str) -> Vec<(f64, usize, usize)> {
    let mir = parse_and_lower(source);
    let features = analyze_program(&mir).features;
    let mut results = Vec::with_capacity(THRESHOLDS.len());

    for &thresh in THRESHOLDS {
        // Rebuild the trained ranker with a custom threshold each iteration.
        let ranker = PassRanker::new(
            LinearCostModel::trained(),
            PerPassLegalityGate::new(),
        )
        .with_skip_threshold(thresh);

        let report = ranker.rank(&mir, &features);
        let rec = report.total_recommended();
        let drop = report.total_dropped();
        results.push((thresh, rec, drop));
    }
    let _ = program_name;
    results
}

// =============================================================================
// Main
// =============================================================================

/// Print per-function feature counts. Useful for diagnosing why a
/// program produces 0 Run-recommendations: if the cost-model predictions
/// are above threshold but the ranker still drops them, the cause is
/// downstream (e.g. DefaultLegalityGate rejecting functions with
/// `strict_count > 0` from float reductions). The feature dump shows
/// at a glance what the trained model is being asked about.
#[allow(dead_code)]
fn dump_features(name: &str, source: &str) {
    let mir = parse_and_lower(source);
    let features = analyze_program(&mir).features;
    println!();
    println!("--- {} feature dump (per-function) ---", name);
    println!(
        "  {:<22} {:>10} {:>10} {:>12} {:>11} {:>13}",
        "function", "expr_count", "loop_depth", "branch_count", "alloc_sites", "strict_reds",
    );
    for (fname, ff) in &features.per_fn {
        println!(
            "  {:<22} {:>10} {:>10} {:>12} {:>11} {:>13}",
            fname,
            ff.memory.expr_count,
            ff.cfg.max_loop_depth,
            ff.cfg.branch_count,
            ff.memory.alloc_sites,
            ff.reductions.strict_count(),
        );
    }
}

fn main() {
    println!("============================================================");
    println!("§17 — Skip-threshold sensitivity probe");
    println!("============================================================");
    println!();
    println!("Sweeping PassRanker.skip_threshold from 1e-2 down to 0.0 on");
    println!("(a) PINN heat 1D (n_epochs=20), (b) 8 pass_ordering programs.");
    println!("Trained ranker only — its coefficients are the v5 production set.");
    println!();
    // Feature dump on PINN — reveals that PINN's functions have
    // non-trivial strict reductions (from float arithmetic), which
    // DefaultLegalityGate rejects regardless of cost-model verdict.
    dump_features("PINN", &pinn_source_short());

    // PINN
    println!("============================================================");
    println!("PINN heat 1D");
    println!("============================================================");
    let pinn_src = pinn_source_short();
    let pinn_results = sweep_thresholds("pinn", &pinn_src);
    println!("{:<12} {:>10} {:>10}", "threshold", "Run-rec", "Skip-drop");
    println!("{}", "-".repeat(36));
    for (t, r, d) in &pinn_results {
        let marker = if (*t - 0.005).abs() < 1e-9 { " ← DEFAULT" } else { "" };
        println!("{:<12.6} {:>10} {:>10}{}", t, r, d, marker);
    }
    let pinn_first_rec_thresh = pinn_results
        .iter()
        .find(|(_, r, _)| *r > 0)
        .map(|(t, _, _)| *t);
    println!();
    if let Some(t) = pinn_first_rec_thresh {
        println!(
            "First non-zero Run-recommendation on PINN: threshold = {:.6} \
             ({:.1}× lower than DEFAULT 0.005)",
            t,
            0.005 / t,
        );
    } else {
        println!("PINN never produces any Run-recommendation at any threshold ≥ 0.0.");
        println!("That would mean every predicted_benefit is exactly zero — the model");
        println!("is clamping to zero (negative raw → clamp(0, 0.5) → 0).");
    }

    // pass_ordering programs
    println!();
    println!("============================================================");
    println!("8 pass_ordering programs");
    println!("============================================================");
    for prog in PASS_ORDERING_PROGRAMS {
        let results = sweep_thresholds(prog.name, prog.source);
        let at_default = results.iter().find(|(t, _, _)| (*t - 0.005).abs() < 1e-9);
        let at_zero = results.iter().find(|(t, _, _)| *t == 0.0);
        let default_rec = at_default.map(|(_, r, _)| *r).unwrap_or(0);
        let zero_rec = at_zero.map(|(_, r, _)| *r).unwrap_or(0);
        let first_rec_thresh = results
            .iter()
            .find(|(_, r, _)| *r > 0)
            .map(|(t, _, _)| *t);
        let saturation = if let Some(ft) = first_rec_thresh {
            if (ft - 0.01).abs() < 1e-9 {
                "active at 1e-2".to_string()
            } else if (ft - 0.005).abs() < 1e-9 {
                "active at default".to_string()
            } else if ft >= 0.001 {
                format!("active only at ≤ {:.4}", ft)
            } else if ft > 0.0 {
                format!("needs threshold ≤ {:.6}", ft)
            } else {
                "needs threshold = 0".to_string()
            }
        } else {
            "never recommends".to_string()
        };
        println!(
            "{:<12} default(0.005)→{} Run | t=0→{} Run | {}",
            prog.name, default_rec, zero_rec, saturation,
        );
    }

    // Interpretation
    println!();
    println!("============================================================");
    println!("Interpretation");
    println!("============================================================");
    println!();
    if pinn_first_rec_thresh.is_none() {
        println!("PINN's predicted_benefit is identically zero across all candidate");
        println!("passes — the linear cost model's raw value is negative or exactly");
        println!("zero, and clamp(0, 0.5) pushes everything to 0. Lowering the");
        println!("threshold has NO effect because the prediction itself is zero.");
        println!();
        println!("Implication: PINN's features (large expr_count, loop_depth >= 1,");
        println!("branch_count >= 1, alloc_sites = 0) interact with the trained");
        println!("coefficients to produce a NEGATIVE raw prediction. The negative");
        println!("loop_depth/branch_count coefficients on most passes overwhelm the");
        println!("small positive expr_count contribution.");
        println!();
        println!("This is a coefficient-sign problem, not a threshold problem.");
        println!("Fixing it requires either:");
        println!("  - Refitting with PINN-like programs in the corpus, or");
        println!("  - Adjusting clamping (e.g. allow negative predictions through");
        println!("    if the model is confident enough), or");
        println!("  - Adding hidden layers / non-linearity to the cost model.");
        println!();
        println!("Threshold tuning will NOT activate PINN — the prediction is the");
        println!("bottleneck, not the threshold.");
    } else if pinn_first_rec_thresh.unwrap() < 0.0001 {
        println!("PINN starts producing recommendations only below threshold");
        println!("{:.6} — that's {:.0}× lower than DEFAULT (0.005).",
                 pinn_first_rec_thresh.unwrap(),
                 0.005 / pinn_first_rec_thresh.unwrap());
        println!();
        println!("That magnitude of threshold drop is too aggressive for production");
        println!("— it would let through every pass with even microscopically positive");
        println!("predicted benefit, including ones that compile-cost would dominate.");
        println!("PINN's predictions sit so close to zero that the threshold");
        println!("approach isn't viable.");
    } else {
        println!("PINN starts producing recommendations at threshold {:.6} —",
                 pinn_first_rec_thresh.unwrap());
        println!("{:.1}× lower than DEFAULT (0.005).",
                 0.005 / pinn_first_rec_thresh.unwrap());
        println!();
        println!("This is in the range where threshold tuning would be a viable");
        println!("activation strategy. Worth experimenting with");
        println!("--mir-opt + custom_skip_threshold on PINN to see actual compile +");
        println!("runtime impact.");
    }
}
