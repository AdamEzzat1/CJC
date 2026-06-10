//! Phase A6 — 5-way ablation harness emitting `CompilationProfile` rows.
//!
//! For every `(program × configuration)` pair this harness:
//!
//! 1. Compiles the program (parse → HIR → MIR → features).
//! 2. Ranks passes under the configuration's cost-model stack.
//! 3. Applies the resulting `PassPlan` and counts MIR nodes
//!    before/after (the deterministic score: size ratio, lower =
//!    better).
//! 4. Runs BOTH executors (AST tree-walk on the source program,
//!    MIR-exec on the optimized program) and compares captured print
//!    output → `parity_match`.
//! 5. Records NSS + PINN predictions for the row.
//! 6. Appends one [`CompilationProfile`] row to
//!    `bench_results/cana_ablation/profiles.cpdb`.
//!
//! ## The five ablations (Phase-A handoff §5.2)
//!
//! | id | stack |
//! |---|---|
//! | `baseline` | `LinearCostModel::trained()` only |
//! | `nss` | + `ThermalAwareCostModel` over `NssPressurePredictor` |
//! | `quantum` | + `EnergyAwarePassRanker` re-ranking (null pressures) |
//! | `nss_quantum` | + both advisory layers |
//! | `full_pinn` | `PinnPhysicalCostModel` + NSS + energy re-ranking |
//!
//! Every ablation runs the same corpus with the same seed. Wall-clock
//! is recorded as diagnostic metadata only — the score and every
//! decision input are deterministic counters (invariant #7).
//!
//! ## Program corpus
//!
//! Snapshotted from `bench/cana_ab_corpus` (which snapshotted from
//! `bench/cana_pass_ordering`). Drift is intentional: stable
//! hand-written workloads, so ablation rows stay comparable across
//! sessions.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use cjc_cana::cost_model::CostModel;
use cjc_cana::features::CanaFeatures;
use cjc_cana::legality::{LegalityVerdict, PerPassLegalityGate};
use cjc_cana::pass_ranker::{pass_plan_from, PassRanker, RankingReport};
use cjc_cana::physical_cost::{build_physical_query, predict_physical, PhysicalCoefficients};
use cjc_cana::pinn_cost_model::PinnPhysicalCostModel;
use cjc_cana::pressure::{NullPressurePredictor, PressurePredictor};
use cjc_cana::thermal_cost_model::ThermalAwareCostModel;
use cjc_cana::{analyze_program, LinearCostModel};
use cjc_cana_compress::profile_db::{
    append_row, read_all, CompilationProfile, PROFILE_SCHEMA_VERSION,
};
use cjc_cana_compress::EnergyAwarePassRanker;
use cjc_cana_nss::{NssPressurePredictor, RecordedPressurePredictor};
use cjc_mir::optimize::optimize_program_with_plan;
use cjc_mir::MirProgram;
use cjc_mir_exec::run_program_instrumented;

const SEED: u64 = 42;

// =============================================================================
// Program corpus (snapshot — see module docs)
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

/// FP-hot workload — ADDED for the Option-B re-run (not part of the
/// original cana_ab_corpus snapshot). The Track-2 ablation concluded
/// the corpus had no program whose thermal signal could differentiate
/// the stacks; this one runs dense float arithmetic inside a nested
/// loop, so the recorded FP-op density (→ Thermal) is high while the
/// integer programs' stays near zero.
const PROG_FP_HOT: &str = r#"
fn horner(x: f64, n: i64) -> f64 {
    let mut acc: f64 = 0.0;
    let mut i: i64 = 0;
    while i < n {
        let mut j: i64 = 0;
        let mut p: f64 = 1.0;
        while j < 16 {
            p = p * x + 0.5;
            acc = acc + p * 0.001;
            j = j + 1;
        }
        i = i + 1;
    }
    return acc;
}
print(horner(1.01, 200));
"#;

const PROGRAMS: &[Program] = &[
    Program {
        name: "arith",
        source: PROG_ARITH,
    },
    Program {
        name: "loop",
        source: PROG_LOOP,
    },
    Program {
        name: "nested",
        source: PROG_NESTED,
    },
    Program {
        name: "many_fn",
        source: PROG_MANY_FN,
    },
    Program {
        name: "mixed",
        source: PROG_MIXED,
    },
    Program {
        name: "float",
        source: PROG_FLOAT,
    },
    Program {
        name: "recursive",
        source: PROG_RECURSIVE,
    },
    Program {
        name: "large",
        source: PROG_LARGE,
    },
    Program {
        name: "fp_hot",
        source: PROG_FP_HOT,
    },
];

// =============================================================================
// Ablation configurations
// =============================================================================

/// Synthetic-predictor configurations (Option A) — the original
/// Track-2 set, kept for cross-session comparability.
const CONFIG_IDS: &[&str] = &["baseline", "nss", "quantum", "nss_quantum", "full_pinn"];

/// Recorded-trace configurations (Option B). Same stacks as the three
/// pressure-consuming synthetic configs, but the predictor is a
/// [`RecordedPressurePredictor`] built from a real instrumented run of
/// the program. `baseline` and `quantum` don't consume pressure, so
/// they have no recorded variant.
const CONFIG_IDS_RECORDED: &[&str] = &["nss_rec", "nss_quantum_rec", "full_pinn_rec"];

/// Rank `mir` under one ablation configuration. Returns the report plus
/// the cost-model identity that drove it. `recorded` backs the `*_rec`
/// configs.
fn rank_under(
    config: &str,
    mir: &MirProgram,
    features: &CanaFeatures,
    recorded: &RecordedPressurePredictor,
) -> (RankingReport, String, u32) {
    match config {
        "baseline" => {
            let model = LinearCostModel::trained();
            let (id, ver) = (model.name().to_string(), model.version());
            let report = PassRanker::new(model, PerPassLegalityGate::new()).rank(mir, features);
            (report, id, ver)
        }
        "nss" => {
            let model = ThermalAwareCostModel::new(
                LinearCostModel::trained(),
                NssPressurePredictor::default(),
            );
            let (id, ver) = (model.name().to_string(), model.version());
            let report = PassRanker::new(model, PerPassLegalityGate::new()).rank(mir, features);
            (report, id, ver)
        }
        "nss_rec" => {
            let model = ThermalAwareCostModel::new(LinearCostModel::trained(), recorded.clone());
            let (id, ver) = (model.name().to_string(), model.version());
            let report = PassRanker::new(model, PerPassLegalityGate::new()).rank(mir, features);
            (report, id, ver)
        }
        "quantum" => {
            let model = LinearCostModel::trained();
            let (id, ver) = (model.name().to_string(), model.version());
            let adapter = EnergyAwarePassRanker::new(
                PassRanker::new(model, PerPassLegalityGate::new()),
                Box::new(NullPressurePredictor),
            );
            (adapter.rank(mir, features), id, ver)
        }
        "nss_quantum" => {
            let model = LinearCostModel::trained();
            let (id, ver) = (model.name().to_string(), model.version());
            let adapter = EnergyAwarePassRanker::new(
                PassRanker::new(model, PerPassLegalityGate::new()),
                Box::new(NssPressurePredictor::default()),
            );
            (adapter.rank(mir, features), id, ver)
        }
        "nss_quantum_rec" => {
            let model = LinearCostModel::trained();
            let (id, ver) = (model.name().to_string(), model.version());
            let adapter = EnergyAwarePassRanker::new(
                PassRanker::new(model, PerPassLegalityGate::new()),
                Box::new(recorded.clone()),
            );
            (adapter.rank(mir, features), id, ver)
        }
        "full_pinn" => {
            let model = PinnPhysicalCostModel::new(
                LinearCostModel::trained(),
                NssPressurePredictor::default(),
            );
            let (id, ver) = (model.name().to_string(), model.version());
            let adapter = EnergyAwarePassRanker::new(
                PassRanker::new(model, PerPassLegalityGate::new()),
                Box::new(NssPressurePredictor::default()),
            );
            (adapter.rank(mir, features), id, ver)
        }
        "full_pinn_rec" => {
            let model = PinnPhysicalCostModel::new(LinearCostModel::trained(), recorded.clone());
            let (id, ver) = (model.name().to_string(), model.version());
            let adapter = EnergyAwarePassRanker::new(
                PassRanker::new(model, PerPassLegalityGate::new()),
                Box::new(recorded.clone()),
            );
            (adapter.rank(mir, features), id, ver)
        }
        other => panic!("unknown ablation config {other}"),
    }
}

// =============================================================================
// Per-(program × config) experiment
// =============================================================================

fn total_expr_count(features: &CanaFeatures) -> u64 {
    features
        .per_fn
        .values()
        .map(|f| f.memory.expr_count as u64)
        .fold(0u64, |a, b| a.saturating_add(b))
}

fn run_experiment(
    prog: &Program,
    config: &str,
    recorded: &RecordedPressurePredictor,
) -> CompilationProfile {
    let wall_start = Instant::now();

    // -- Compile + rank ----------------------------------------------------
    let (ast, diags) = cjc_parser::parse_source(prog.source);
    assert!(
        !diags.has_errors(),
        "parse errors in {}: {:?}",
        prog.name,
        diags.diagnostics
    );
    let mut al = cjc_hir::AstLowering::new();
    let hir = al.lower_program(&ast);
    let mut h2m = cjc_mir::HirToMir::new();
    let mir = h2m.lower_program(&hir);
    let features = analyze_program(&mir).features;

    let (report, cost_model_id, cost_model_version) = rank_under(config, &mir, &features, recorded);
    let plan = pass_plan_from(&report.sequence);
    let mut optimized = optimize_program_with_plan(&mir, &plan);
    cjc_mir::escape::annotate_program(&mut optimized);
    let compile_wall_micros = wall_start.elapsed().as_micros() as u64;

    // -- Deterministic size metric ------------------------------------------
    let mir_nodes_before = total_expr_count(&features);
    let optimized_features = cjc_cana::features::extract(&optimized);
    let mir_nodes_after = total_expr_count(&optimized_features);
    let score = mir_nodes_after as f64 / (mir_nodes_before.max(1)) as f64;

    // -- Workload estimates + PINN predictions ------------------------------
    // Neutral pass ("dce" has identity physical amplification) so the
    // row captures the program's intrinsic workload, not a per-pass
    // variant.
    let coeffs = PhysicalCoefficients::default();
    let mut est_flops = 0u64;
    let mut est_read = 0u64;
    let mut est_written = 0u64;
    let mut est_alloc = 0u64;
    let mut est_ws = 0u64;
    let mut pinn_energy_max = 0.0f64;
    let mut pinn_thermal_max = 0.0f64;
    let mut pinn_bandwidth_max = 0.0f64;
    for (fn_name, ff) in &features.per_fn {
        let q = build_physical_query(fn_name, "dce", ff);
        est_flops = est_flops.saturating_add(q.flops_estimate);
        est_read = est_read.saturating_add(q.bytes_read_estimate);
        est_written = est_written.saturating_add(q.bytes_written_estimate);
        est_alloc = est_alloc.saturating_add(q.allocation_bytes_estimate);
        est_ws = est_ws.saturating_add(q.working_set_bytes_estimate);
        if let Some(est) = predict_physical(&q, &coeffs) {
            pinn_energy_max = pinn_energy_max.max(est.energy_estimate);
            pinn_thermal_max = pinn_thermal_max.max(est.thermal_pressure);
            pinn_bandwidth_max = pinn_bandwidth_max.max(est.bandwidth_pressure);
        }
    }

    // -- NSS predictions -----------------------------------------------------
    // Recorded configs record the recorded predictor's view; synthetic
    // configs record Option A's — the row reflects what the ranker saw.
    let max_of = |m: BTreeMap<String, f64>| m.values().copied().fold(0.0f64, f64::max);
    let (nss_cpu_max, nss_memory_max, nss_thermal_max) = if config.ends_with("_rec") {
        (
            max_of(recorded.predict_cpu_saturation(&mir, &features)),
            max_of(recorded.predict_memory_peak(&mir, &features)),
            max_of(recorded.predict_thermal(&mir, &features)),
        )
    } else {
        let nss = NssPressurePredictor::default();
        (
            max_of(nss.predict_cpu_saturation(&mir, &features)),
            max_of(nss.predict_memory_peak(&mir, &features)),
            max_of(nss.predict_thermal(&mir, &features)),
        )
    };

    // -- Parity: AST-eval vs MIR-exec on the OPTIMIZED program ---------------
    let mut interp = cjc_eval::Interpreter::new(SEED);
    let eval_result = interp.exec(&ast);
    let mut exec = cjc_mir_exec::MirExecutor::new(SEED);
    exec.scan_ast_imports(&ast);
    let exec_result = exec.exec(&optimized);
    let parity_match = match (&eval_result, &exec_result) {
        (Ok(_), Ok(_)) => Some(interp.output == exec.output),
        _ => Some(false),
    };

    // -- Legality + counts ----------------------------------------------------
    let (legality_approved, legality_violation_count) = match &report.verdict {
        LegalityVerdict::Approved => (true, 0u32),
        LegalityVerdict::Rejected(v) => (false, v.len() as u32),
    };

    let pass_sequence: Vec<(String, Vec<String>)> = plan
        .per_function
        .iter()
        .map(|(f, seq)| (f.clone(), seq.clone()))
        .collect();

    CompilationProfile {
        schema_version: PROFILE_SCHEMA_VERSION,
        program_name: prog.name.to_string(),
        program_hash: features.program_hash.0,
        feature_hash: features.feature_hash.0,
        sidecar_bundle_hash: 0, // no sidecar in the ablation harness (yet)
        config_id: config.to_string(),
        cost_model_id,
        cost_model_version,
        pass_sequence,
        estimated_flops: est_flops,
        estimated_bytes_read: est_read,
        estimated_bytes_written: est_written,
        estimated_alloc_bytes: est_alloc,
        estimated_working_set: est_ws,
        nss_predicted_cpu_max: nss_cpu_max,
        nss_predicted_memory_max: nss_memory_max,
        nss_predicted_thermal_max: nss_thermal_max,
        pinn_predicted_energy_max: pinn_energy_max,
        pinn_predicted_thermal_max: pinn_thermal_max,
        pinn_predicted_bandwidth_max: pinn_bandwidth_max,
        mir_nodes_before,
        mir_nodes_after,
        recommended_count: report.total_recommended() as u32,
        dropped_count: report.total_dropped() as u32,
        legality_approved,
        legality_violation_count,
        parity_match,
        compile_wall_micros,
        score,
    }
}

// =============================================================================
// Main — run all experiments, emit rows, print comparison + §5.2 gate
// =============================================================================

fn main() {
    let out_dir = PathBuf::from("bench_results/cana_ablation");
    fs::create_dir_all(&out_dir).expect("create bench_results/cana_ablation");
    let db_path = out_dir.join("profiles.cpdb");
    // Fresh file per invocation: the harness is deterministic, so
    // re-running appends identical rows; truncating keeps the corpus
    // duplicate-free for training. (Cross-session accumulation can
    // concatenate archives.)
    let _ = fs::remove_file(&db_path);

    println!("=================================================================");
    println!(
        "Phase A6 — ablation over {} programs (seed {SEED}): 5 synthetic + 3 recorded configs",
        PROGRAMS.len()
    );
    println!("=================================================================\n");

    // Per program: one instrumented run (Option B) feeds the recorded
    // predictor used by the *_rec configs.
    let mut recorded_preds: BTreeMap<&str, RecordedPressurePredictor> = BTreeMap::new();
    for prog in PROGRAMS {
        let (ast, diags) = cjc_parser::parse_source(prog.source);
        assert!(!diags.has_errors(), "parse errors in {}", prog.name);
        let (_val, exec, events) = run_program_instrumented(&ast, SEED).expect("instrumented run");
        let recorded = RecordedPressurePredictor::from_recorded_events(
            events,
            exec.trace_node_assignments().clone(),
        );
        recorded_preds.insert(prog.name, recorded);
    }

    // rows[program][config] = profile
    let mut rows: BTreeMap<&str, BTreeMap<&str, CompilationProfile>> = BTreeMap::new();
    for prog in PROGRAMS {
        let recorded = &recorded_preds[prog.name];
        for config in CONFIG_IDS.iter().chain(CONFIG_IDS_RECORDED.iter()) {
            let row = run_experiment(prog, config, recorded);
            append_row(&db_path, &row).expect("append profile row");
            rows.entry(prog.name).or_default().insert(config, row);
        }
    }

    // -- Synthetic cohort (Option A predictors) -----------------------------
    println!("Synthetic (Option A) cohort — scores (lower = better):");
    println!(
        "{:<10} | {:>9} | {:>9} | {:>9} | {:>11} | {:>9}",
        "program", "baseline", "nss", "quantum", "nss_quantum", "full_pinn"
    );
    println!("{}", "-".repeat(74));
    for (prog, per_config) in &rows {
        let score = |c: &str| per_config.get(c).map(|r| r.score).unwrap_or(f64::NAN);
        println!(
            "{:<10} | {:>9.4} | {:>9.4} | {:>9.4} | {:>11.4} | {:>9.4}",
            prog,
            score("baseline"),
            score("nss"),
            score("quantum"),
            score("nss_quantum"),
            score("full_pinn"),
        );
    }

    // -- Recorded cohort (Option B real traces) ------------------------------
    println!("\nRecorded (Option B) cohort — scores + recorded max thermal:");
    println!(
        "{:<10} | {:>9} | {:>15} | {:>13} | {:>11} | parity",
        "program", "nss_rec", "nss_quantum_rec", "full_pinn_rec", "rec_thermal"
    );
    println!("{}", "-".repeat(78));
    for (prog, per_config) in &rows {
        let score = |c: &str| per_config.get(c).map(|r| r.score).unwrap_or(f64::NAN);
        let rec_thermal = per_config
            .get("full_pinn_rec")
            .map(|r| r.nss_predicted_thermal_max)
            .unwrap_or(f64::NAN);
        let parity_all = per_config.values().all(|r| r.parity_match == Some(true));
        println!(
            "{:<10} | {:>9.4} | {:>15.4} | {:>13.4} | {:>11.4} | {}",
            prog,
            score("nss_rec"),
            score("nss_quantum_rec"),
            score("full_pinn_rec"),
            rec_thermal,
            if parity_all { "ok" } else { "MISMATCH" },
        );
    }

    // -- Differentiation check: did real traces change ANY plan? ------------
    let mut diverged: Vec<String> = Vec::new();
    for (prog, per_config) in &rows {
        for (syn, rec) in [
            ("nss", "nss_rec"),
            ("nss_quantum", "nss_quantum_rec"),
            ("full_pinn", "full_pinn_rec"),
        ] {
            if per_config[syn].pass_sequence != per_config[rec].pass_sequence {
                diverged.push(format!("{prog}:{rec}"));
                // Show exactly which passes the real pressure withheld
                // (or added) per function — union of both plans, so a
                // function dropped ENTIRELY (PINN hard limit zeroing
                // every benefit) still prints.
                let syn_map: BTreeMap<&String, &Vec<String>> = per_config[syn]
                    .pass_sequence
                    .iter()
                    .map(|(f, p)| (f, p))
                    .collect();
                let rec_map: BTreeMap<&String, &Vec<String>> = per_config[rec]
                    .pass_sequence
                    .iter()
                    .map(|(f, p)| (f, p))
                    .collect();
                let empty: Vec<String> = Vec::new();
                let mut all_fns: Vec<&String> =
                    syn_map.keys().chain(rec_map.keys()).copied().collect();
                all_fns.sort();
                all_fns.dedup();
                for func in all_fns {
                    let syn_passes = syn_map.get(func).copied().unwrap_or(&empty);
                    let rec_passes = rec_map.get(func).copied().unwrap_or(&empty);
                    if syn_passes != rec_passes {
                        println!(
                            "  [plan diff] {prog}/{func} under {rec}: {:?} -> {:?}",
                            syn_passes, rec_passes
                        );
                    }
                }
            }
        }
    }

    // -- §5.2 promotion gate over the recorded cohort -------------------------
    // full_pinn_rec vs every other config (both cohorts). Lower = better.
    let margin = 0.1;
    let mut pinn_wins = 0usize;
    let mut ties = 0usize;
    for per_config in rows.values() {
        let pinn = per_config["full_pinn_rec"].score;
        let best_other = per_config
            .iter()
            .filter(|(c, _)| **c != "full_pinn_rec")
            .map(|(_, r)| r.score)
            .fold(f64::INFINITY, f64::min);
        if pinn <= best_other - margin {
            pinn_wins += 1;
        } else if (pinn - best_other).abs() < 1e-12 {
            ties += 1;
        }
    }
    let total = rows.len();
    let parity_all = rows
        .values()
        .flat_map(|m| m.values())
        .all(|r| r.parity_match == Some(true));

    println!("\n----------------------------------------------------------------");
    println!(
        "Plan divergence (synthetic vs recorded): {}",
        if diverged.is_empty() {
            "NONE — recorded pressures did not change any plan".to_string()
        } else {
            format!(
                "{} config-program pairs: {}",
                diverged.len(),
                diverged.join(", ")
            )
        }
    );
    println!(
        "§5.2 gate (full_pinn_rec): wins (≥{margin} margin): {pinn_wins}/{total}   ties: {ties}/{total}"
    );
    println!(
        "Parity (all rows): {}",
        if parity_all { "100%" } else { "FAILED" }
    );

    // Row-hash stability canary: re-run one recorded experiment.
    let again = run_experiment(
        &PROGRAMS[0],
        "full_pinn_rec",
        &recorded_preds[PROGRAMS[0].name],
    );
    let stable = rows[PROGRAMS[0].name]["full_pinn_rec"].row_hash() == again.row_hash();
    println!(
        "Row-hash stability (double-run, wall-clock excluded): {}",
        if stable { "byte-identical" } else { "DRIFT" }
    );

    let back = read_all(&db_path).expect("read back profile db");
    println!("\nProfile DB: {} rows at {}", back.len(), db_path.display());
    let verdict = if pinn_wins * 10 >= total * 6 {
        "PINN v2 promotion gate: WOULD PASS (≥60% wins)"
    } else {
        "PINN v2 promotion gate: NOT MET — see §5.4 (also requires 1000+ rows)"
    };
    println!("{verdict}");
    assert!(parity_all, "parity must hold across every ablation row");
    assert!(stable, "row hash must be double-run stable");
}
