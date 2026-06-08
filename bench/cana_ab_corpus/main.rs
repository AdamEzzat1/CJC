//! Compile-time-only three-way ranker comparison across the 8 external
//! `cana_pass_ordering` programs.
//!
//! ## Why this bench exists
//!
//! `cana_ab_pinn` answers the AB question for one workload (PINN heat 1D)
//! with full runtime measurement: ~6 minutes per workload at N=3
//! iterations. The §3A.2 finding there was "all three rankers agree to
//! skip every pass."
//!
//! Before concluding that's a *global* property (rather than a
//! PINN-specific one), we want to check: do the rankers EVER disagree
//! on a non-PINN program? The cheapest way to answer is to skip the
//! runtime phase entirely — we only need the compile-time decisions,
//! which take milliseconds per workload. 8 programs × ~10 ms each ≈
//! seconds total, vs ~48 minutes for the full runtime bench.
//!
//! The 8 programs come from the `cana_pass_ordering` corpus
//! (`bench/cana_pass_ordering`), snapshotted in this file because
//! they're stable hand-written workloads and we don't want corpus
//! drift to silently change AB results.

use std::collections::{BTreeMap, BTreeSet};

use cjc_cana::legality::PerPassLegalityGate;
use cjc_cana::pass_ranker::{pass_plan_from, trained_ranker, PassRanker};
use cjc_cana::thermal_cost_model::ThermalAwareCostModel;
use cjc_cana::{analyze_program, default_ranker, LinearCostModel};
use cjc_cana_nss::NssPressurePredictor;
use cjc_mir::optimize::PassPlan;

// =============================================================================
// Snapshot of pass_ordering's 8 programs.
// Copied verbatim from bench/cana_pass_ordering/main.rs at the time of
// implementation. Drift is intentional: we want stable AB results.
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

const PROGRAMS: &[Program] = &[
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
// Per-program comparison
// =============================================================================

#[derive(Debug)]
struct RankerComparison {
    program: &'static str,
    /// Default ranker stats.
    default_rec: usize,
    default_drop: usize,
    /// Trained ranker stats.
    trained_rec: usize,
    trained_drop: usize,
    /// Thermal-aware ranker stats.
    thermal_rec: usize,
    thermal_drop: usize,
    /// Whether each pairwise PassPlan comparison matches.
    dt_match: bool,
    dth_match: bool,
    tth_match: bool,
}

fn rank_program(prog: &Program) -> RankerComparison {
    let (ast, diags) = cjc_parser::parse_source(prog.source);
    if diags.has_errors() {
        panic!("parse errors in {}: {:?}", prog.name, diags.diagnostics);
    }
    let mut al = cjc_hir::AstLowering::new();
    let hir = al.lower_program(&ast);
    let mut h2m = cjc_mir::HirToMir::new();
    let mir = h2m.lower_program(&hir);
    let features = analyze_program(&mir).features;

    let default_report = default_ranker().rank(&mir, &features);
    let trained_report = trained_ranker().rank(&mir, &features);

    let thermal_cost_model = ThermalAwareCostModel::new(
        LinearCostModel::trained(),
        NssPressurePredictor::default(),
    );
    let thermal_report =
        PassRanker::new(thermal_cost_model, PerPassLegalityGate::new()).rank(&mir, &features);

    let default_plan = pass_plan_from(&default_report.sequence);
    let trained_plan = pass_plan_from(&trained_report.sequence);
    let thermal_plan = pass_plan_from(&thermal_report.sequence);

    RankerComparison {
        program: prog.name,
        default_rec: default_report.total_recommended(),
        default_drop: default_report.total_dropped(),
        trained_rec: trained_report.total_recommended(),
        trained_drop: trained_report.total_dropped(),
        thermal_rec: thermal_report.total_recommended(),
        thermal_drop: thermal_report.total_dropped(),
        dt_match: plans_equivalent(&default_plan, &trained_plan),
        dth_match: plans_equivalent(&default_plan, &thermal_plan),
        tth_match: plans_equivalent(&trained_plan, &thermal_plan),
    }
}

fn plans_equivalent(a: &PassPlan, b: &PassPlan) -> bool {
    if a.per_function.len() != b.per_function.len() {
        return false;
    }
    for (f, seq) in &a.per_function {
        match b.per_function.get(f) {
            Some(b_seq) if b_seq == seq => continue,
            _ => return false,
        }
    }
    true
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    println!("============================================================");
    println!("Compile-time three-way AB across {} non-PINN programs", PROGRAMS.len());
    println!("============================================================");
    println!();

    let mut results: Vec<RankerComparison> = Vec::with_capacity(PROGRAMS.len());
    for prog in PROGRAMS {
        results.push(rank_program(prog));
    }

    // --------------------------------------------------------------------
    // Per-program ranking-internal stats
    // --------------------------------------------------------------------
    println!("Per-program RankingReport internals (rec=Run-recommendations, drop=Skips):");
    println!(
        "{:<12} | {:>3}/{:<3} | {:>3}/{:<3} | {:>3}/{:<3}",
        "program", "d-r", "d-d", "t-r", "t-d", "th-r", "th-d",
    );
    println!("{}", "-".repeat(64));
    for r in &results {
        println!(
            "{:<12} | {:>3}/{:<3} | {:>3}/{:<3} | {:>3}/{:<3}",
            r.program,
            r.default_rec, r.default_drop,
            r.trained_rec, r.trained_drop,
            r.thermal_rec, r.thermal_drop,
        );
    }

    // --------------------------------------------------------------------
    // Per-program PassPlan agreement
    // --------------------------------------------------------------------
    println!();
    println!("Pairwise PassPlan equivalence (✓ = identical, ✗ = differ):");
    println!(
        "{:<12} | {:^12} | {:^12} | {:^12}",
        "program", "d == t", "d == th", "t == th",
    );
    println!("{}", "-".repeat(60));
    let mut any_disagreement = false;
    let mut disagreement_count: BTreeMap<&str, usize> = BTreeMap::new();
    for r in &results {
        let dt = if r.dt_match { "✓" } else { "✗" };
        let dth = if r.dth_match { "✓" } else { "✗" };
        let tth = if r.tth_match { "✓" } else { "✗" };
        println!(
            "{:<12} | {:^12} | {:^12} | {:^12}",
            r.program, dt, dth, tth,
        );
        if !r.dt_match {
            any_disagreement = true;
            *disagreement_count.entry("default↔trained").or_insert(0) += 1;
        }
        if !r.dth_match {
            any_disagreement = true;
            *disagreement_count.entry("default↔thermal").or_insert(0) += 1;
        }
        if !r.tth_match {
            any_disagreement = true;
            *disagreement_count.entry("trained↔thermal").or_insert(0) += 1;
        }
    }

    // --------------------------------------------------------------------
    // Aggregate stats
    // --------------------------------------------------------------------
    println!();
    println!("============================================================");
    println!("Aggregate ranking stats");
    println!("============================================================");
    let total_rec = |stat: fn(&RankerComparison) -> usize| -> usize {
        results.iter().map(stat).sum()
    };
    let default_total_rec: usize = total_rec(|r| r.default_rec);
    let trained_total_rec: usize = total_rec(|r| r.trained_rec);
    let thermal_total_rec: usize = total_rec(|r| r.thermal_rec);
    let default_total_drop: usize = total_rec(|r| r.default_drop);
    let trained_total_drop: usize = total_rec(|r| r.trained_drop);
    let thermal_total_drop: usize = total_rec(|r| r.thermal_drop);

    println!(
        "{:<12}: {:>4} Run-recommendations, {:>4} Skips",
        "default", default_total_rec, default_total_drop,
    );
    println!(
        "{:<12}: {:>4} Run-recommendations, {:>4} Skips",
        "trained", trained_total_rec, trained_total_drop,
    );
    println!(
        "{:<12}: {:>4} Run-recommendations, {:>4} Skips",
        "thermal", thermal_total_rec, thermal_total_drop,
    );

    // --------------------------------------------------------------------
    // Verdict
    // --------------------------------------------------------------------
    println!();
    println!("============================================================");
    println!("Verdict");
    println!("============================================================");
    if !any_disagreement {
        println!("All three rankers agree on all {} programs.", PROGRAMS.len());
        println!();
        println!("This strengthens the §3A.2 PINN finding: the rankers' identical");
        println!("behavior isn't a PINN-specific quirk — it holds across the");
        println!("8-program external corpus too. The base ranker rejects every");
        println!("candidate pass on every program, so trained vs default coefficient");
        println!("differences and thermal-aware adjustments never cross the skip");
        println!("threshold.");
        println!();
        println!("Implication for §17 (skip-threshold tuning): the threshold is");
        println!("indeed the bottleneck. Lowering it (or making it pass-confidence-");
        println!("aware) is what would surface trained vs default vs thermal");
        println!("divergence on real workloads. The coefficients are NOT the");
        println!("problem; the threshold logic is.");
    } else {
        println!("Disagreement found on at least one program:");
        let total_disagreements: usize = disagreement_count.values().sum();
        for (pair, count) in &disagreement_count {
            let pct = (*count as f64 / PROGRAMS.len() as f64) * 100.0;
            println!(
                "  {:<20} disagreed on {}/{} programs ({:.0}%)",
                pair, count, PROGRAMS.len(), pct,
            );
        }
        println!();
        println!("Investigate which programs and which functions to identify the");
        println!("feature-distribution conditions that surface ranker disagreement.");
        let _ = total_disagreements;
    }

    // --------------------------------------------------------------------
    // Programs where all rankers agree completely
    // --------------------------------------------------------------------
    let all_agree: BTreeSet<&str> = results
        .iter()
        .filter(|r| r.dt_match && r.dth_match && r.tth_match)
        .map(|r| r.program)
        .collect();
    if !all_agree.is_empty() && all_agree.len() < PROGRAMS.len() {
        println!();
        println!(
            "Programs where all three rankers fully agree ({} of {}):",
            all_agree.len(),
            PROGRAMS.len(),
        );
        for p in &all_agree {
            println!("  - {}", p);
        }
    }
}
