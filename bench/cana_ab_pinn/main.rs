//! §3A.2 + §4B.3 extension — three-way AB test: `default_ranker` vs
//! `trained_ranker` vs **thermal-aware ranker** on a real workload.
//!
//! Originally implemented §3A.2 (default vs trained). After §4B.3 landed,
//! extended to a third config: the thermal-aware ranker constructed from
//! `ThermalAwareCostModel<LinearCostModel::trained, NssPressurePredictor>`
//! — the exact composition that `cjcl run --thermal-aware` uses through
//! `cjc_mir_exec::run_program_optimized_thermal_aware`.
//!
//! The §4B.2 design doc and §3A.2 bench-level checks predict that
//! `NssPressurePredictor` in Option C mode returns empty thermal maps,
//! so the thermal-wrapped cost model is a behavioural no-op vs the
//! base trained model. This bench confirms that prediction at the
//! end-to-end compile + run level on a substantial real program.
//!
//! Workload: `examples/08_pinn_heat_equation.cjcl` — a physics-informed
//! neural network with 4 functions, ~180 LOC of CJC-Lang, dominated by
//! nested while-loops + float arithmetic + branchy if/else. The original
//! example runs 80 epochs (4 minutes wall-clock through MIR-exec on the
//! bench machine); for AB testing we substitute n_epochs=20 to keep
//! total wall-clock tractable while preserving the program structure
//! that determines compile-time decisions.
//!
//! Per-config N=3 iterations to get a stable median timing.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use cjc_cana::legality::DefaultLegalityGate;
use cjc_cana::pass_ranker::{trained_ranker, PassRanker};
use cjc_cana::thermal_cost_model::ThermalAwareCostModel;
use cjc_cana::{analyze_program, default_ranker, pass_plan_from, LinearCostModel};
use cjc_cana_nss::NssPressurePredictor;
use cjc_mir::optimize::{optimize_program_with_plan, PassPlan};

/// The PINN program is loaded from the example tree. We do a
/// surgical string substitution to reduce the epoch count from
/// 80 → 20 so total AB-test wall-clock stays in single-digit minutes.
const PINN_FULL_SOURCE: &str = include_str!("../../examples/08_pinn_heat_equation.cjcl");

const SEED: u64 = 42;
const N_ITERS: usize = 3;
const SUBSTITUTE_FROM: &str = "let n_epochs = 80;";
const SUBSTITUTE_TO: &str = "let n_epochs = 20;";

fn pinn_source_short() -> String {
    let modified = PINN_FULL_SOURCE.replace(SUBSTITUTE_FROM, SUBSTITUTE_TO);
    assert_ne!(
        modified, PINN_FULL_SOURCE,
        "expected '{}' in PINN source — substitution failed. Did the example change?",
        SUBSTITUTE_FROM,
    );
    modified
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ranker {
    Default,
    Trained,
    Thermal,
}

impl Ranker {
    fn label(self) -> &'static str {
        match self {
            Ranker::Default => "default",
            Ranker::Trained => "trained",
            Ranker::Thermal => "thermal",
        }
    }
}

/// All three rankers, in canonical comparison order.
const ALL_RANKERS: &[Ranker] = &[Ranker::Default, Ranker::Trained, Ranker::Thermal];

struct AbResult {
    ranker: Ranker,
    compile_us_samples: Vec<u128>,
    run_us_samples: Vec<u128>,
    pass_plan: PassPlan,
    output: String,
    /// Internal ranker stats from the first iteration: count of Run
    /// recommendations across all functions, and count of dropped
    /// recommendations. Together they reveal whether an empty PassPlan
    /// means "ranker recommends nothing" or "ranker recommends but
    /// `pass_plan_from` filtered everything".
    total_recommended: usize,
    total_dropped: usize,
    /// Per-function Run-recommendation count from the first iteration.
    per_fn_run_count: BTreeMap<String, usize>,
}

fn parse_and_lower(source: &str) -> (cjc_ast::Program, cjc_mir::MirProgram) {
    let (ast, diags) = cjc_parser::parse_source(source);
    assert!(
        !diags.has_errors(),
        "parse errors in PINN source: {:#?}",
        diags.diagnostics,
    );
    let mut al = cjc_hir::AstLowering::new();
    let hir = al.lower_program(&ast);
    let mut h2m = cjc_mir::HirToMir::new();
    let mir = h2m.lower_program(&hir);
    (ast, mir)
}

struct OneShot {
    compile_us: u128,
    run_us: u128,
    plan: PassPlan,
    output: String,
    total_recommended: usize,
    total_dropped: usize,
    per_fn_run_count: BTreeMap<String, usize>,
}

fn measure_one(source: &str, ranker: Ranker) -> OneShot {
    let compile_start = Instant::now();
    let (ast, mir) = parse_and_lower(source);
    let features = analyze_program(&mir).features;
    let report = match ranker {
        Ranker::Default => default_ranker().rank(&mir, &features),
        Ranker::Trained => trained_ranker().rank(&mir, &features),
        Ranker::Thermal => {
            // The exact composition that `cjcl run --thermal-aware`
            // builds inside `cjc_mir_exec::cana_thermal_aware_plan_for`.
            // Keep this construction in sync with that helper — if it
            // ever changes (e.g. NssPressurePredictor takes a non-default
            // seed), update here too.
            let cost_model = ThermalAwareCostModel::new(
                LinearCostModel::trained(),
                NssPressurePredictor::default(),
            );
            PassRanker::new(cost_model, DefaultLegalityGate::new())
                .rank(&mir, &features)
        }
    };
    let plan = pass_plan_from(&report.sequence);
    let mut optimized = optimize_program_with_plan(&mir, &plan);
    cjc_mir::escape::annotate_program(&mut optimized);
    let compile_us = compile_start.elapsed().as_micros();

    let total_recommended = report.total_recommended();
    let total_dropped = report.total_dropped();
    let mut per_fn_run_count: BTreeMap<String, usize> = BTreeMap::new();
    for (fn_name, ranking) in &report.per_fn {
        per_fn_run_count.insert(fn_name.clone(), ranking.recommended.len());
    }

    let run_start = Instant::now();
    let mut exec = cjc_mir_exec::MirExecutor::new(SEED);
    exec.scan_ast_imports(&ast);
    let _val = exec
        .exec(&optimized)
        .unwrap_or_else(|e| panic!("exec failed for {}: {:?}", ranker.label(), e));
    let run_us = run_start.elapsed().as_micros();
    let output = exec.output.join("\n");

    OneShot {
        compile_us,
        run_us,
        plan,
        output,
        total_recommended,
        total_dropped,
        per_fn_run_count,
    }
}

fn measure(source: &str, ranker: Ranker) -> AbResult {
    let mut compile_samples = Vec::with_capacity(N_ITERS);
    let mut run_samples = Vec::with_capacity(N_ITERS);
    let mut canonical: Option<OneShot> = None;

    for i in 0..N_ITERS {
        let one = measure_one(source, ranker);
        println!(
            "  [{:>1}/{}] {:<8} compile_us={:>8}  run_us={:>10}  rec={:>2} drop={:>2}",
            i + 1,
            N_ITERS,
            ranker.label(),
            one.compile_us,
            one.run_us,
            one.total_recommended,
            one.total_dropped,
        );
        compile_samples.push(one.compile_us);
        run_samples.push(one.run_us);
        if canonical.is_none() {
            canonical = Some(one);
        }
    }

    let canon = canonical.unwrap();
    AbResult {
        ranker,
        compile_us_samples: compile_samples,
        run_us_samples: run_samples,
        pass_plan: canon.plan,
        output: canon.output,
        total_recommended: canon.total_recommended,
        total_dropped: canon.total_dropped,
        per_fn_run_count: canon.per_fn_run_count,
    }
}

fn median(samples: &[u128]) -> u128 {
    let mut s = samples.to_vec();
    s.sort_unstable();
    s[s.len() / 2]
}

fn diff_plans(a: &PassPlan, b: &PassPlan, label_a: &str, label_b: &str) -> Vec<String> {
    let mut diffs = Vec::new();
    let all_fns: BTreeSet<&String> = a
        .per_function
        .keys()
        .chain(b.per_function.keys())
        .collect();
    for f in all_fns {
        let av = a.per_function.get(f);
        let bv = b.per_function.get(f);
        match (av, bv) {
            (Some(av), Some(bv)) if av == bv => {}
            (Some(av), Some(bv)) => diffs.push(format!(
                "  fn {}\n     {}: {:?}\n     {}: {:?}",
                f, label_a, av, label_b, bv,
            )),
            (Some(av), None) => diffs.push(format!(
                "  fn {} (only in {})\n     {}: {:?}",
                f, label_a, label_a, av,
            )),
            (None, Some(bv)) => diffs.push(format!(
                "  fn {} (only in {})\n     {}: {:?}",
                f, label_b, label_b, bv,
            )),
            (None, None) => unreachable!(),
        }
    }
    diffs
}

fn summarize_plan(plan: &PassPlan) -> BTreeMap<&'static str, usize> {
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    let known: &[&'static str] = &[
        "constant_fold",
        "cf_round_2",
        "strength_reduce",
        "dce",
        "cse",
        "licm",
    ];
    for k in known {
        counts.insert(k, 0);
    }
    counts.insert("other", 0);
    for (_, seq) in &plan.per_function {
        for pass in seq {
            let key = known
                .iter()
                .copied()
                .find(|&k| k == pass.as_str())
                .unwrap_or("other");
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

fn main() {
    println!("============================================================");
    println!("§3A.2 + §4B.3 — Three-way AB test on PINN heat 1D");
    println!("                default / trained / thermal-aware");
    println!("============================================================");
    let src = pinn_source_short();
    println!(
        "Workload: examples/08_pinn_heat_equation.cjcl (n_epochs 80 → 20 for AB tractability)",
    );
    println!(
        "Lines: {}, chars: {}, N_ITERS: {}",
        src.lines().count(),
        src.len(),
        N_ITERS,
    );

    // Pre-extract function names for context.
    let (_ast, preview_mir) = parse_and_lower(&src);
    println!(
        "MIR functions: {} ({})",
        preview_mir.functions.len(),
        preview_mir
            .functions
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>()
            .join(", "),
    );
    println!();

    // ---------------------------------------------------------------------
    // Measure all three configs.
    // ---------------------------------------------------------------------
    let mut results: BTreeMap<&'static str, AbResult> = BTreeMap::new();
    for &ranker in ALL_RANKERS {
        println!("Running {}-ranker timings:", ranker.label());
        let res = measure(&src, ranker);
        results.insert(ranker.label(), res);
        println!();
    }

    let get = |label: &str| -> &AbResult {
        results
            .get(label)
            .unwrap_or_else(|| panic!("missing result for ranker {}", label))
    };
    let default_res = get("default");
    let trained_res = get("trained");
    let thermal_res = get("thermal");

    // ---------------------------------------------------------------------
    // Timing summary (3 rows, deltas vs default baseline)
    // ---------------------------------------------------------------------
    let medians: BTreeMap<&'static str, (u128, u128)> = results
        .iter()
        .map(|(k, r)| (*k, (median(&r.compile_us_samples), median(&r.run_us_samples))))
        .collect();
    let (base_compile, base_run) = medians["default"];

    println!("============================================================");
    println!("Timing summary (median across {} iterations per config)", N_ITERS);
    println!("============================================================");
    println!(
        "{:<12} {:>14} {:>14} {:>12} {:>12}",
        "ranker", "compile_us", "run_us", "Δcompile%", "Δrun%",
    );
    println!("{}", "-".repeat(68));
    for &ranker in ALL_RANKERS {
        let label = ranker.label();
        let (c, r) = medians[label];
        let dc_pct = ((c as f64 - base_compile as f64) / base_compile as f64) * 100.0;
        let dr_pct = ((r as f64 - base_run as f64) / base_run as f64) * 100.0;
        if label == "default" {
            println!("{:<12} {:>14} {:>14} {:>12} {:>12}", label, c, r, "(base)", "(base)");
        } else {
            println!(
                "{:<12} {:>14} {:>14} {:>+11.2}% {:>+11.2}%",
                label, c, r, dc_pct, dr_pct,
            );
        }
    }

    // ---------------------------------------------------------------------
    // Internal ranking summary across all 3 configs
    // ---------------------------------------------------------------------
    println!();
    println!("============================================================");
    println!("Internal RankingReport summary");
    println!("============================================================");
    println!(
        "{:<12} {:>20} {:>14}",
        "ranker", "total_recommended (Run)", "total_dropped",
    );
    println!("{}", "-".repeat(50));
    for &ranker in ALL_RANKERS {
        let r = get(ranker.label());
        println!(
            "{:<12} {:>20} {:>14}",
            ranker.label(),
            r.total_recommended,
            r.total_dropped,
        );
    }

    println!();
    println!("Per-function Run-recommendation count:");
    let all_fns: BTreeSet<&String> = ALL_RANKERS
        .iter()
        .flat_map(|r| get(r.label()).per_fn_run_count.keys())
        .collect();
    println!(
        "  {:<20} {:>10} {:>10} {:>10}",
        "function", "default", "trained", "thermal",
    );
    println!("  {}", "-".repeat(53));
    for f in all_fns {
        let d = default_res.per_fn_run_count.get(f).copied().unwrap_or(0);
        let t = trained_res.per_fn_run_count.get(f).copied().unwrap_or(0);
        let th = thermal_res.per_fn_run_count.get(f).copied().unwrap_or(0);
        println!("  {:<20} {:>10} {:>10} {:>10}", f, d, t, th);
    }

    // ---------------------------------------------------------------------
    // Behavioural checks — three-way byte-identity
    // ---------------------------------------------------------------------
    println!();
    println!("============================================================");
    println!("Behavioural checks — three-way byte-identity");
    println!("============================================================");
    let dt_match = default_res.output == trained_res.output;
    let dth_match = default_res.output == thermal_res.output;
    let tth_match = trained_res.output == thermal_res.output;
    let all_match = dt_match && dth_match && tth_match;

    println!("  default == trained: {}", dt_match);
    println!("  default == thermal: {}", dth_match);
    println!("  trained == thermal: {}", tth_match);
    println!();
    if all_match {
        println!("Three-way output byte-identical: ✓");
        println!("  All three rankers produced byte-identical stdout. No correctness");
        println!("  regression in the §4B.3 thermal-aware path. The §4B.2 design-doc");
        println!("  claim 'NssPressurePredictor empty thermal map → ThermalAwareCostModel");
        println!("  is a no-op vs base cost model' is confirmed end-to-end on a real");
        println!("  workload, not just via the cjc-cana-nss unit test.");
    } else {
        println!("Three-way output byte-identical: FAIL");
        println!("  CRITICAL — at least one ranker changes output bytes vs another.");
        println!("  Investigate before shipping.");
    }

    // ---------------------------------------------------------------------
    // PassPlan diff — pairwise across all three
    // ---------------------------------------------------------------------
    println!();
    println!("============================================================");
    println!("PassPlan diffs (pairwise)");
    println!("============================================================");
    for (label_a, label_b) in [("default", "trained"), ("default", "thermal"), ("trained", "thermal")] {
        let a = &get(label_a).pass_plan;
        let b = &get(label_b).pass_plan;
        let diffs = diff_plans(a, b, label_a, label_b);
        if diffs.is_empty() {
            println!("  {} vs {}: NONE", label_a, label_b);
        } else {
            println!("  {} vs {}: {} function(s) differ.", label_a, label_b, diffs.len());
            for d in &diffs {
                println!("{}", d);
            }
        }
    }

    println!();
    println!("============================================================");
    println!("Per-pass invocation counts (across all functions)");
    println!("============================================================");
    let counts: BTreeMap<&'static str, BTreeMap<&'static str, usize>> = results
        .iter()
        .map(|(k, r)| (*k, summarize_plan(&r.pass_plan)))
        .collect();
    println!(
        "{:<20} {:>10} {:>10} {:>10}",
        "pass", "default", "trained", "thermal",
    );
    println!("{}", "-".repeat(54));
    let pass_keys = counts["default"].keys().copied().collect::<Vec<_>>();
    for key in &pass_keys {
        let d = counts["default"].get(key).copied().unwrap_or(0);
        let t = counts["trained"].get(key).copied().unwrap_or(0);
        let th = counts["thermal"].get(key).copied().unwrap_or(0);
        println!("{:<20} {:>10} {:>10} {:>10}", key, d, t, th);
    }

    // ---------------------------------------------------------------------
    // Verdict
    // ---------------------------------------------------------------------
    println!();
    println!("============================================================");
    println!("Verdict");
    println!("============================================================");
    let dt_plans_match = default_res.pass_plan == trained_res.pass_plan;
    let dth_plans_match = default_res.pass_plan == thermal_res.pass_plan;
    if !all_match {
        println!("CRITICAL: output divergence between rankers — investigate before shipping.");
    } else if dt_plans_match && dth_plans_match {
        println!("All three rankers produced identical PassPlans on this workload.");
        println!();
        println!("Implications:");
        println!("  §3A.2 (default vs trained):   trained_ranker is inactive here (see §3A.4 audit).");
        println!("  §4B.3 (thermal vs trained):   thermal-aware wrapper is currently a no-op,");
        println!("                                consistent with §4B.2 Option C empty-map design.");
        println!();
        println!("Both wrappers are wired and validated. They will become differentially active");
        println!("once the base ranker starts producing non-trivial plans on real workloads");
        println!("(corpus expansion / threshold tuning) or once NssPressurePredictor migrates");
        println!("from Option C (empty maps) to Option A (synthetic trace) or Option B (real");
        println!("instrumentation). The bench is ready to detect either transition immediately.");
    } else {
        println!("PassPlans differ across configs but output stays byte-identical:");
        println!("  default vs trained plan: {}", if dt_plans_match { "match" } else { "differ" });
        println!("  default vs thermal plan: {}", if dth_plans_match { "match" } else { "differ" });
        println!();
        println!("Investigate: which functions differ, and what the per-pass changes mean.");
        println!("The 'PassPlan diffs (pairwise)' section above lists specific differences.");
    }
}
