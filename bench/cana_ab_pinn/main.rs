//! §3A.2 — AB test: `trained_ranker` vs `default_ranker` on a real workload.
//!
//! Implements the experiment from `docs/cana/HANDOFF_NEXT_SESSION.md` §3A.2:
//! pick a real CJC-Lang program, compile it twice (once with each ranker),
//! diff the PassPlans, measure compile + runtime, verify byte-identical
//! output.
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

use cjc_cana::pass_ranker::trained_ranker;
use cjc_cana::{analyze_program, default_ranker, pass_plan_from};
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

#[derive(Clone, Copy, Debug)]
enum Ranker {
    Default,
    Trained,
}

impl Ranker {
    fn label(self) -> &'static str {
        match self {
            Ranker::Default => "default",
            Ranker::Trained => "trained",
        }
    }
}

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

fn diff_plans(a: &PassPlan, b: &PassPlan) -> Vec<String> {
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
                "  fn {}\n     default: {:?}\n     trained: {:?}",
                f, av, bv,
            )),
            (Some(av), None) => diffs.push(format!(
                "  fn {} (only in default)\n     default: {:?}",
                f, av,
            )),
            (None, Some(bv)) => diffs.push(format!(
                "  fn {} (only in trained)\n     trained: {:?}",
                f, bv,
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
    println!("§3A.2 — AB test: trained vs default ranker on PINN heat 1D");
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

    println!("Running default-ranker timings:");
    let default_res = measure(&src, Ranker::Default);
    println!();
    println!("Running trained-ranker timings:");
    let trained_res = measure(&src, Ranker::Trained);

    // ---------------------------------------------------------------------
    // Timing summary
    // ---------------------------------------------------------------------
    let default_compile = median(&default_res.compile_us_samples);
    let default_run = median(&default_res.run_us_samples);
    let trained_compile = median(&trained_res.compile_us_samples);
    let trained_run = median(&trained_res.run_us_samples);

    println!();
    println!("============================================================");
    println!("Timing summary (median across {} iterations per config)", N_ITERS);
    println!("============================================================");
    println!(
        "{:<12} {:>14} {:>14}",
        "ranker", "compile_us", "run_us",
    );
    println!("{}", "-".repeat(44));
    println!(
        "{:<12} {:>14} {:>14}",
        "default", default_compile, default_run,
    );
    println!(
        "{:<12} {:>14} {:>14}",
        "trained", trained_compile, trained_run,
    );

    let compile_delta_us = trained_compile as i128 - default_compile as i128;
    let run_delta_us = trained_run as i128 - default_run as i128;
    let compile_pct = (compile_delta_us as f64 / default_compile as f64) * 100.0;
    let run_pct = (run_delta_us as f64 / default_run as f64) * 100.0;
    println!();
    println!(
        "delta (trained − default):  compile = {:+.1}% ({:+}us)   run = {:+.2}% ({:+}us)",
        compile_pct, compile_delta_us, run_pct, run_delta_us,
    );

    // ---------------------------------------------------------------------
    // Internal ranking summary (what the ranker itself decided, before
    // pass_plan_from filtered Skip recommendations away).
    // ---------------------------------------------------------------------
    println!();
    println!("============================================================");
    println!("Internal RankingReport summary");
    println!("============================================================");
    println!(
        "  default: total_recommended (Run) = {:>3}, total_dropped = {:>3}",
        default_res.total_recommended, default_res.total_dropped,
    );
    println!(
        "  trained: total_recommended (Run) = {:>3}, total_dropped = {:>3}",
        trained_res.total_recommended, trained_res.total_dropped,
    );
    println!();
    println!("Per-function Run-recommendation count:");
    let all_fns: BTreeSet<&String> = default_res
        .per_fn_run_count
        .keys()
        .chain(trained_res.per_fn_run_count.keys())
        .collect();
    println!("  {:<20} {:>10} {:>10} {:>9}", "function", "default", "trained", "delta");
    println!("  {}", "-".repeat(53));
    for f in all_fns {
        let d = default_res.per_fn_run_count.get(f).copied().unwrap_or(0);
        let t = trained_res.per_fn_run_count.get(f).copied().unwrap_or(0);
        let delta = t as i64 - d as i64;
        println!("  {:<20} {:>10} {:>10} {:>+9}", f, d, t, delta);
    }

    // ---------------------------------------------------------------------
    // Behavioural checks
    // ---------------------------------------------------------------------
    println!();
    println!("============================================================");
    println!("Behavioural checks");
    println!("============================================================");
    let output_match = default_res.output == trained_res.output;
    println!("Output byte-identical: {}", output_match);
    if !output_match {
        let common_prefix_lines = default_res
            .output
            .lines()
            .zip(trained_res.output.lines())
            .take_while(|(a, b)| a == b)
            .count();
        println!(
            "  Diverged after {} matching lines (correctness regression — investigate!).",
            common_prefix_lines,
        );
    }

    // ---------------------------------------------------------------------
    // PassPlan diff
    // ---------------------------------------------------------------------
    println!();
    let plan_diffs = diff_plans(&default_res.pass_plan, &trained_res.pass_plan);
    if plan_diffs.is_empty() {
        println!("PassPlan diff: NONE — both rankers produced identical plans for every function.");
        println!("  Interpretation: on this workload, the trained coefficients lead to the");
        println!("  same per-function pass sequence as the default coefficients. The compile +");
        println!("  runtime deltas above (if any) are pure noise.");
    } else {
        println!("PassPlan diff: {} function(s) differ.", plan_diffs.len());
        for d in &plan_diffs {
            println!("{}", d);
        }
    }

    println!();
    println!("============================================================");
    println!("Per-pass invocation counts (across all functions)");
    println!("============================================================");
    let default_counts = summarize_plan(&default_res.pass_plan);
    let trained_counts = summarize_plan(&trained_res.pass_plan);
    println!("{:<20} {:>10} {:>10} {:>+9}", "pass", "default", "trained", "delta");
    println!("{}", "-".repeat(54));
    for key in default_counts.keys() {
        let d = default_counts.get(key).copied().unwrap_or(0);
        let t = trained_counts.get(key).copied().unwrap_or(0);
        let delta = t as i64 - d as i64;
        println!("{:<20} {:>10} {:>10} {:>+9}", key, d, t, delta);
    }

    println!();
    println!("============================================================");
    println!("Verdict");
    println!("============================================================");
    if plan_diffs.is_empty() {
        println!("Trained ranker produces no behavioural change vs default on this workload.");
        println!("Recommendation: §3A.1+§3A.3 validation results stand; trained_ranker() is");
        println!("safe but inactive here. Try a workload with more diverse feature ranges to");
        println!("provoke a divergent decision (e.g. one with branch_count > 5).");
    } else if !output_match {
        println!("CRITICAL: trained ranker changes output bytes. Investigate before shipping.");
    } else if compile_pct < -5.0 && run_pct < -5.0 {
        println!("Trained ranker is faster on BOTH dimensions. Strong positive signal.");
    } else if run_pct < -5.0 {
        println!("Trained ranker produces measurably faster code (compile-time tradeoff acceptable).");
    } else if compile_pct < -10.0 {
        println!("Trained ranker is materially faster to compile (runtime indistinguishable).");
    } else if run_pct > 5.0 {
        println!("Trained ranker is SLOWER at runtime. Held-out validation passed but this workload");
        println!("hit a generalization weak spot. See §3A.1 for SR overfitting context.");
    } else {
        println!("Both rankers within noise band (±5%). Trained ranker produces different plans");
        println!("but no measurable runtime impact on this workload.");
    }
}
