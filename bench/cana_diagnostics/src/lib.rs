//! Phase D — silicon diagnostics: wall-clock + peak-RSS A/B harness.
//!
//! Everything CANA has measured so far is MODELED energy (executed
//! statements + weighted FP ops + heap pages, from instrumented runs).
//! This crate asks the only question that justifies the stack to a
//! user: do those modeled wins appear on a stopwatch? A negative
//! answer is as load-bearing as a positive one — it recalibrates the
//! energy formula before Phases E/F build on it.
//!
//! ## Contract (research doc §3, the auditor's design)
//!
//! **Determinism gate FIRST.** Before any clock is read, every subject
//! must pass, hard-error otherwise:
//!
//! 1. **Corpus identity** — the snapshot source's recomputed
//!    `program_hash` equals the committed corpus row's. The subject
//!    sources here are snapshots of `bench/cana_ablation` generators
//!    (the established snapshot pattern); this gate makes the snapshot
//!    self-verifying, so drift can never silently time the wrong
//!    program.
//! 2. **Plan identity** — the recomputed arm plans byte-equal the
//!    corpus rows' `pass_sequence`. We time exactly the plans Phase C
//!    measured, not a lookalike.
//! 3. **Output determinism** — AST-eval, MIR-exec(arm A) and
//!    MIR-exec(arm B) outputs are byte-identical. THE gate.
//! 4. **Modeled-energy reproduction** — the energy ratio recomputed in
//!    THIS build matches the corpus `score` to 1e-9. Without it, a
//!    null stopwatch result is ambiguous between "modeled wins don't
//!    materialize" and "we timed a build where the win doesn't exist".
//!
//! Gates 1/2/4 are skipped only for the non-corpus example subject;
//! gate 3 never.
//!
//! ## Protocol
//!
//! Iterations are calibrated ONCE on arm A (identical count for both
//! arms — per-run times stay directly comparable), then one warm-up
//! phase per arm followed by interleaved measured phases A/B/A/B…,
//! median-of-5 per arm. Each phase is a **fresh child process**:
//! `PeakWorkingSetSize` is process-monotonic, so per-arm peak RSS
//! requires process isolation, and fresh processes give every phase
//! identical allocator/startup state. The PARENT plans both arms once
//! (gate 2 pins the plans to the corpus) and hands each child its plan
//! as a file; the child compiles + applies it untimed, then times only
//! the sustained `iters × MirExecutor` loop (~5 s target), re-checking
//! the output FNV every iteration. Children never plan: selector
//! candidate-probing was measured at a 1.63 GB peak on the real
//! workload subject, which would have made child RSS a measurement of
//! the planner instead of the program.
//!
//! ## Hard wall (determinism invariant)
//!
//! Wall-clock and RSS land ONLY in `bench_results/cana_diagnostics/`
//! artifacts. Nothing measured here feeds decisions, hashes, or
//! profile-row stable fields — the same rule as `compile_wall_micros`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use cjc_cana::features::CanaFeatures;
use cjc_cana::hash::CanaHasher;
use cjc_cana::legality::PerPassLegalityGate;
use cjc_cana::pass_ranker::{pass_plan_from, PassRanker};
use cjc_cana::pinn_cost_model::PinnPhysicalCostModel;
use cjc_cana::pinn_energy_v1::PinnEnergyV1;
use cjc_cana::pinn_thermal_v2::PinnThermalV2;
use cjc_cana::plan_selector::PassPlanSelector;
use cjc_cana::{analyze_program, LinearCostModel};
use cjc_cana_compress::profile_db::{read_all, CompilationProfile};
use cjc_cana_compress::EnergyAwarePassRanker;
use cjc_cana_nss::RecordedPressurePredictor;
use cjc_mir::optimize::{optimize_program_with_plan, PassPlan};
use cjc_mir::MirProgram;
use cjc_mir_exec::{run_program_instrumented, trace, MirExecutor};
use cjc_repro::KahanAccumulatorF64;

/// Same seed as every CANA harness — the corpus rows being reproduced
/// were emitted under it.
pub const SEED: u64 = 42;

/// Snapshot of `bench/cana_ablation/main.rs::FP_ENERGY_WEIGHT`. The
/// modeled-energy formula DEFINES the corpus score labels; gate 4
/// cross-checks this snapshot against the committed corpus on every
/// run, so silent drift fails loudly instead of skewing the report.
pub const FP_ENERGY_WEIGHT: f64 = 3.0;

/// Workspace root, resolved at compile time from this crate's
/// manifest. Children inherit it regardless of their spawn cwd.
pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

// =============================================================================
// Subjects — snapshot sources, self-verified against the corpus (gate 1)
// =============================================================================

/// One A/B subject: a program plus the two configs whose plans race.
#[derive(Debug, Clone)]
pub struct Subject {
    /// Corpus program name (or a fresh name for non-corpus subjects).
    pub name: String,
    /// Family label: `selector` | `thermal` | `tensor` | `nonsynthetic`.
    pub family: &'static str,
    pub source: String,
    /// Arm A config id (always `baseline` today — gate 4 relies on it).
    pub arm_a: &'static str,
    /// Arm B config id.
    pub arm_b: &'static str,
    /// Whether gates 1/2/4 apply (false only for non-corpus subjects).
    pub corpus_verified: bool,
}

/// Snapshot of `cana_ablation::PROG_FP_HOT` (gate-1-verified).
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

/// Snapshot of `cana_ablation::holdout_workloads()`'s
/// `holdout_alloc_pulse` — the frozen-holdout selector win.
const PROG_HOLDOUT_ALLOC_PULSE: &str = r#"
fn pulse(n: i64) -> i64 {
    let mut keep: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let burst: Any = [i, i + 1, i + 2, i + 3];
        let tag: Any = (i, i * i);
        keep = keep + i % 7;
        i = i + 1;
    }
    return keep;
}
print(pulse(5000));
"#;

/// The non-synthetic subject: a real PINN training loop shipped as a
/// user-facing example (sustained scalar FP, deterministic, never part
/// of any training corpus). The handoff's broad-claim guard: the six
/// selector wins are all one synthetic allocation-churn shape, so at
/// least one real workload must sit next to them on the stopwatch.
const PROG_EXAMPLE_08: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../examples/08_pinn_heat_equation.cjcl"
));

/// Snapshot of `cana_ablation::memory_gradient_workloads()` for one
/// `k` — the `mem_grad_a{k}` allocation-churn family (5 of the 6
/// selector wins).
fn mem_grad_source(k: u32) -> String {
    let iters = 4i64.pow(k) * 64; // 256 .. 65,536
    format!(
        r#"
fn churn(n: i64) -> i64 {{
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {{
        let cell: Any = [i, i + 1];
        let pair: Any = (i, i * 2);
        total = total + i;
        i = i + 1;
    }}
    return total;
}}
print(churn({iters}));
"#
    )
}

/// Snapshot of `cana_ablation::thermal_gradient_workloads()` restricted
/// to `fp_k = 9` — the `grad_f90_*` sustained-FP family (Track 3's
/// original thermal subjects).
fn grad_f90_source(outer: i64, depth: u32) -> String {
    let fp_k = 9u32;
    let int_k = 10 - fp_k;
    let mut body = String::new();
    for f in 0..fp_k {
        body.push_str(&format!("            facc = facc + 0.5{f:01};\n"));
    }
    for i in 0..int_k {
        body.push_str(&format!("            iacc = iacc + i * {};\n", i + 3));
    }
    if depth == 1 {
        format!(
            r#"
fn work(n: i64) -> i64 {{
    let mut facc: f64 = 0.0;
    let mut iacc: i64 = 0;
    let mut i: i64 = 0;
    while i < n {{
{body}            i = i + 1;
    }}
    print(facc);
    return iacc;
}}
print(work({outer}));
"#
        )
    } else {
        format!(
            r#"
fn work(n: i64) -> i64 {{
    let mut facc: f64 = 0.0;
    let mut iacc: i64 = 0;
    let mut o: i64 = 0;
    while o < n {{
        let mut i: i64 = 0;
        while i < 8 {{
{body}            i = i + 1;
        }}
        o = o + 1;
    }}
    print(facc);
    return iacc;
}}
print(work({outer}));
"#
        )
    }
}

/// Snapshot of `cana_ablation::TENSOR_BUILDER`.
const TENSOR_BUILDER: &str = r#"
fn build(n: i64, scale: f64) -> Tensor {
    let mut buf: Any = [];
    let mut i: i64 = 0;
    while i < n * n {
        buf = array_push(buf, scale * float(i % 7) + 0.25);
        i = i + 1;
    }
    return Tensor.from_vec(buf, [n, n]);
}
"#;

/// Snapshot of `cana_ablation::tensor_workloads()` — the Phase A1
/// tensor family (validates the energy formula's FP weighting on
/// tensor dispatch).
fn tensor_sources() -> Vec<(String, String)> {
    let mut out = Vec::new();
    out.push((
        "tensor_mm_n16_i50".to_string(),
        format!(
            r#"{TENSOR_BUILDER}
fn mm_hot(a: Tensor, b: Tensor, iters: i64) -> f64 {{
    let mut s: f64 = 0.0;
    let mut i: i64 = 0;
    while i < iters {{
        let c: Tensor = matmul(a, b);
        s = c.sum();
        i = i + 1;
    }}
    return s;
}}
let a: Tensor = build(16, 0.5);
let b: Tensor = build(16, 0.25);
print(mm_hot(a, b, 50));
"#
        ),
    ));
    out.push((
        "tensor_ew_n32_i200".to_string(),
        format!(
            r#"{TENSOR_BUILDER}
fn ew_hot(a: Tensor, b: Tensor, iters: i64) -> Tensor {{
    let mut c: Tensor = a + b;
    let mut i: i64 = 1;
    while i < iters {{
        c = a * b;
        c = c + a;
        i = i + 1;
    }}
    return c;
}}
let a: Tensor = build(32, 0.5);
let b: Tensor = build(32, 0.25);
let r: Tensor = ew_hot(a, b, 200);
print(r.sum());
"#
        ),
    ));
    out.push((
        "tensor_red_n64_i100".to_string(),
        format!(
            r#"{TENSOR_BUILDER}
fn red_hot(a: Tensor, iters: i64) -> f64 {{
    let mut s: f64 = 0.0;
    let mut i: i64 = 0;
    while i < iters {{
        s = a.sum();
        i = i + 1;
    }}
    return s;
}}
let a: Tensor = build(64, 0.5);
print(red_hot(a, 100));
"#
        ),
    ));
    out.push((
        "tensor_mix_n16_i50".to_string(),
        format!(
            r#"{TENSOR_BUILDER}
fn mix_hot(a: Tensor, b: Tensor, iters: i64) -> f64 {{
    let mut acc: f64 = 0.0;
    let mut i: i64 = 0;
    while i < iters {{
        let c: Tensor = matmul(a, b);
        acc = acc + 0.001;
        i = i + 1;
    }}
    return acc;
}}
let a: Tensor = build(16, 0.5);
let b: Tensor = build(16, 0.25);
print(mix_hot(a, b, 50));
"#
        ),
    ));
    for k in 0u32..=4 {
        let mut body = String::new();
        for _ in 0..k {
            body.push_str("        u = u * 0.999;\n");
        }
        for j in 0..(4 - k) {
            body.push_str(&format!("        facc = facc + 0.5{j:01};\n"));
        }
        out.push((
            format!("tensor_tg_k{k}"),
            format!(
                r#"{TENSOR_BUILDER}
fn work(t: Tensor, n: i64) -> f64 {{
    let mut facc: f64 = 0.0;
    let mut u: Tensor = t * 1.0;
    let mut i: i64 = 0;
    while i < n {{
{body}        i = i + 1;
    }}
    print(facc);
    return u.sum();
}}
let t: Tensor = build(16, 0.5);
print(work(t, 128));
"#
            ),
        ));
    }
    out
}

/// The full subject list, deterministic order. Family sizes: 6
/// (selector wins) + 7 (thermal: fp_hot + grad_f90 family) + 9
/// (tensor) + 1 (non-synthetic) = 23.
pub fn subjects() -> Vec<Subject> {
    let mut out = Vec::new();

    // -- Family 1: the six named selector wins (handoff §2 arm 1).
    //    All one mechanism — dead per-iteration allocations that DCE
    //    halves — so read any win as ONE discovery, not six.
    for k in 1u32..=5 {
        out.push(Subject {
            name: format!("mem_grad_a{k}"),
            family: "selector",
            source: mem_grad_source(k),
            arm_a: "baseline",
            arm_b: "selector_rec",
            corpus_verified: true,
        });
    }
    out.push(Subject {
        name: "holdout_alloc_pulse".to_string(),
        family: "selector",
        source: PROG_HOLDOUT_ALLOC_PULSE.to_string(),
        arm_a: "baseline",
        arm_b: "selector_rec",
        corpus_verified: true,
    });

    // -- Family 2: sustained scalar FP, thermal-aware stack vs plain
    //    (handoff §2 arm 2 — the original Track-3 subjects).
    out.push(Subject {
        name: "fp_hot".to_string(),
        family: "thermal",
        source: PROG_FP_HOT.to_string(),
        arm_a: "baseline",
        arm_b: "full_pinn_v2_rec",
        corpus_verified: true,
    });
    for &outer in &[64i64, 256, 1024] {
        for &depth in &[1u32, 2] {
            out.push(Subject {
                name: format!("grad_f90_d{depth}_n{outer}"),
                family: "thermal",
                source: grad_f90_source(outer, depth),
                arm_a: "baseline",
                arm_b: "full_pinn_v2_rec",
                corpus_verified: true,
            });
        }
    }

    // -- Family 3: the A1 tensor family (handoff §2 arm 3).
    for (name, source) in tensor_sources() {
        out.push(Subject {
            name,
            family: "tensor",
            source,
            arm_a: "baseline",
            arm_b: "selector_rec",
            corpus_verified: true,
        });
    }

    // -- Family 4: one real workload next to the synthetics.
    out.push(Subject {
        name: "example_08_pinn_heat".to_string(),
        family: "nonsynthetic",
        source: PROG_EXAMPLE_08.to_string(),
        arm_a: "baseline",
        arm_b: "selector_rec",
        corpus_verified: false,
    });

    out
}

// =============================================================================
// Compilation + arm planning (mirrors cana_ablation's pipeline exactly)
// =============================================================================

/// Parsed + lowered subject, ready for planning and execution.
pub struct CompiledSubject {
    pub ast: cjc_ast::Program,
    pub mir: MirProgram,
    pub features: CanaFeatures,
}

pub fn compile_subject(source: &str) -> Result<CompiledSubject, String> {
    let (ast, diags) = cjc_parser::parse_source(source);
    if diags.has_errors() {
        return Err(format!("parse errors: {:?}", diags.diagnostics));
    }
    let mut al = cjc_hir::AstLowering::new();
    let hir = al.lower_program(&ast);
    let mut h2m = cjc_mir::HirToMir::new();
    let mir = h2m.lower_program(&hir);
    let features = analyze_program(&mir).features;
    Ok(CompiledSubject { ast, mir, features })
}

/// The two trained heads the `*_rec` arms need. Offline-trained,
/// loaded read-only (training never runs during diagnostics either).
pub struct TrainedHeads {
    pub thermal: PinnThermalV2,
    pub energy: PinnEnergyV1,
}

pub fn load_heads() -> Result<TrainedHeads, String> {
    let root = workspace_root();
    let thermal = cjc_cana_compress::pinn_bundle::read_bundle(
        &root.join("bench_results/cana_train_pinn/pinn_thermal_v2.cpb"),
    )
    .map_err(|e| format!("CPB0 thermal bundle missing/corrupt: {e:?}"))?
    .head;
    let energy = cjc_cana_compress::energy_bundle::read_energy_bundle(
        &root.join("bench_results/cana_train_pinn/pinn_energy_v1.cpb"),
    )
    .map_err(|e| format!("CPB1 energy bundle missing/corrupt: {e:?}"))?
    .head;
    Ok(TrainedHeads { thermal, energy })
}

/// One instrumented run of the SOURCE program feeds the recorded
/// predictor — same Option-B pattern as the ablation harness.
pub fn record_pressures(ast: &cjc_ast::Program) -> Result<RecordedPressurePredictor, String> {
    let (_val, exec, events) =
        run_program_instrumented(ast, SEED).map_err(|e| format!("instrumented run failed: {e:?}"))?;
    Ok(RecordedPressurePredictor::from_recorded_events(
        events,
        exec.trace_node_assignments().clone(),
    ))
}

/// The three configs Phase D races. Plans are recomputed through the
/// SAME code paths the ablation harness used (gate 2 then proves they
/// match the committed corpus rows byte-for-byte).
pub fn plan_for_config(
    config: &str,
    mir: &MirProgram,
    features: &CanaFeatures,
    recorded: &RecordedPressurePredictor,
    heads: &TrainedHeads,
) -> Result<PassPlan, String> {
    match config {
        "baseline" => {
            let report = PassRanker::new(LinearCostModel::trained(), PerPassLegalityGate::new())
                .rank(mir, features);
            Ok(pass_plan_from(&report.sequence))
        }
        "full_pinn_v2_rec" => {
            let model = PinnPhysicalCostModel::new(LinearCostModel::trained(), recorded.clone())
                .with_thermal_head(heads.thermal.clone());
            let adapter = EnergyAwarePassRanker::new(
                PassRanker::new(model, PerPassLegalityGate::new()),
                Box::new(recorded.clone()),
            );
            Ok(pass_plan_from(&adapter.rank(mir, features).sequence))
        }
        "selector_rec" => {
            let ranked = plan_for_config("full_pinn_v2_rec", mir, features, recorded, heads)?;
            let selector = PassPlanSelector::new(heads.energy.clone())
                .ok_or("committed energy head failed validation")?;
            Ok(selector
                .select(mir, features, &ranked, &PerPassLegalityGate::new())
                .plan)
        }
        other => Err(format!("unsupported diagnostics config {other}")),
    }
}

/// Optimize under a plan + escape-annotate + run BOTH safety verifiers
/// — byte-for-byte the ablation harness's post-plan path.
pub fn optimize_with(mir: &MirProgram, plan: &PassPlan, label: &str) -> Result<MirProgram, String> {
    let mut optimized = optimize_program_with_plan(mir, plan);
    cjc_mir::escape::annotate_program(&mut optimized);
    if let Err(errors) = cjc_mir::nogc_verify::verify_nogc(&optimized) {
        return Err(format!("NoGC verifier rejected {label}: {errors:?}"));
    }
    let legality = cjc_mir::verify::verify_mir_legality(&optimized);
    if !legality.is_ok() {
        return Err(format!(
            "MIR legality verifier rejected {label}: {:?}",
            legality.errors()
        ));
    }
    Ok(optimized)
}

/// One uninstrumented MIR-exec run; returns captured print output.
pub fn run_mir(ast: &cjc_ast::Program, optimized: &MirProgram) -> Result<Vec<String>, String> {
    let mut exec = MirExecutor::new(SEED);
    exec.scan_ast_imports(ast);
    exec.exec(optimized).map_err(|e| format!("MIR-exec failed: {e:?}"))?;
    Ok(std::mem::take(&mut exec.output))
}

/// Stable FNV-1a digest of an output transcript (length-prefixed so
/// line boundaries can't alias).
pub fn output_fnv(lines: &[String]) -> u64 {
    let mut h = CanaHasher::new();
    h.write_u64(lines.len() as u64);
    for line in lines {
        h.write_u64(line.len() as u64);
        h.write(line.as_bytes());
    }
    h.finish()
}

/// Deterministic modeled energy of one optimized run — the EXACT
/// formula and accumulation order of `cana_ablation::run_experiment`
/// (gate 4 holds this to the committed corpus at 1e-9).
pub fn modeled_energy(ast: &cjc_ast::Program, optimized: &MirProgram) -> Result<f64, String> {
    trace::with_trace(|c| {
        c.reset();
        c.enable();
    });
    let mut exec = MirExecutor::new(SEED);
    exec.scan_ast_imports(ast);
    let result = exec.exec(optimized);
    let events = trace::with_trace(|c| {
        c.disable();
        let e = c.take();
        c.reset();
        e
    });
    result.map_err(|e| format!("instrumented MIR-exec failed: {e:?}"))?;

    let mut instr_total: u64 = 0;
    let mut heap_max: u64 = 0;
    let mut fp_acc = KahanAccumulatorF64::new();
    for ev in &events {
        instr_total = instr_total.saturating_add(ev.instruction_count as u64);
        heap_max = heap_max.max(ev.heap_bytes_in_use);
        let fp_in_window = ev.thermal_intensity * ev.instruction_count as f64;
        fp_acc.add(fp_in_window);
    }
    let fp_total = fp_acc.finalize();
    let fp_term = FP_ENERGY_WEIGHT * fp_total;
    let heap_term = heap_max as f64 / 4096.0;
    let energy_partial = instr_total as f64 + fp_term;
    Ok(energy_partial + heap_term)
}

// =============================================================================
// Corpus index + the four gates
// =============================================================================

/// `(program, config) → row` view of the committed ablation corpus.
pub struct CorpusIndex {
    rows: BTreeMap<(String, String), CompilationProfile>,
}

impl CorpusIndex {
    pub fn load() -> Result<Self, String> {
        let path = workspace_root().join("bench_results/cana_ablation/profiles.cpdb");
        let all = read_all(&path).map_err(|e| {
            format!(
                "cannot read committed corpus {}: {e:?} — run `cargo run --release -p cana-ablation`",
                path.display()
            )
        })?;
        let mut rows = BTreeMap::new();
        for row in all {
            rows.insert((row.program_name.clone(), row.config_id.clone()), row);
        }
        Ok(Self { rows })
    }

    pub fn get(&self, program: &str, config: &str) -> Option<&CompilationProfile> {
        self.rows.get(&(program.to_string(), config.to_string()))
    }
}

/// Gate 1 — the snapshot source IS the corpus program.
pub fn gate1_program_hash(
    subject: &Subject,
    features: &CanaFeatures,
    corpus: &CorpusIndex,
) -> Result<(), String> {
    let row = corpus
        .get(&subject.name, "baseline")
        .ok_or_else(|| format!("gate 1: {} has no baseline corpus row", subject.name))?;
    if features.program_hash.0 != row.program_hash {
        return Err(format!(
            "gate 1 FAILED: {} snapshot drifted from corpus (hash {:#x} != committed {:#x})",
            subject.name, features.program_hash.0, row.program_hash
        ));
    }
    Ok(())
}

/// A plan as the corpus stores it (BTreeMap iteration = sorted, the
/// same construction `run_experiment` used).
pub fn plan_as_sorted_vec(plan: &PassPlan) -> Vec<(String, Vec<String>)> {
    plan.per_function
        .iter()
        .map(|(f, p)| (f.clone(), p.clone()))
        .collect()
}

/// Gate 2 — the recomputed plan IS the plan Phase C measured.
pub fn gate2_plan_identity(
    subject: &Subject,
    config: &str,
    plan: &PassPlan,
    corpus: &CorpusIndex,
) -> Result<(), String> {
    let row = corpus
        .get(&subject.name, config)
        .ok_or_else(|| format!("gate 2: {}/{config} has no corpus row", subject.name))?;
    let recomputed = plan_as_sorted_vec(plan);
    if recomputed != row.pass_sequence {
        return Err(format!(
            "gate 2 FAILED: {}/{config} recomputed plan differs from corpus row\n  recomputed: {:?}\n  committed:  {:?}",
            subject.name, recomputed, row.pass_sequence
        ));
    }
    Ok(())
}

/// Gate 3 — THE determinism gate: AST-eval, arm A and arm B outputs
/// byte-identical. Returns the agreed transcript + its FNV (children
/// re-verify against it every sustained-load iteration).
pub fn gate3_output_determinism(
    ast: &cjc_ast::Program,
    opt_a: &MirProgram,
    opt_b: &MirProgram,
) -> Result<(Vec<String>, u64), String> {
    let mut interp = cjc_eval::Interpreter::new(SEED);
    interp
        .exec(ast)
        .map_err(|e| format!("AST-eval failed: {e:?}"))?;
    let out_eval = interp.output.clone();
    let out_a = run_mir(ast, opt_a)?;
    let out_b = run_mir(ast, opt_b)?;
    if out_a != out_eval {
        return Err(format!(
            "gate 3 FAILED: arm A output diverges from AST-eval\n  eval: {out_eval:?}\n  armA: {out_a:?}"
        ));
    }
    if out_b != out_a {
        return Err(format!(
            "gate 3 FAILED: arm B output diverges from arm A\n  armA: {out_a:?}\n  armB: {out_b:?}"
        ));
    }
    let fnv = output_fnv(&out_a);
    Ok((out_a, fnv))
}

/// Modeled energies of both arms + the baseline-normalized ratio
/// (`max(e_a, 1.0)` mirrors the ablation's normalizer exactly).
pub struct EnergyEvidence {
    pub energy_a: f64,
    pub energy_b: f64,
    /// `energy_b / max(energy_a, 1.0)` — comparable to corpus `score`.
    pub ratio_b: f64,
}

pub fn measure_modeled_energy(
    ast: &cjc_ast::Program,
    opt_a: &MirProgram,
    opt_b: &MirProgram,
) -> Result<EnergyEvidence, String> {
    let energy_a = modeled_energy(ast, opt_a)?;
    let energy_b = modeled_energy(ast, opt_b)?;
    let ratio_b = energy_b / energy_a.max(1.0);
    Ok(EnergyEvidence {
        energy_a,
        energy_b,
        ratio_b,
    })
}

/// Gate 4 — the modeled win exists in THIS build. Tolerance 1e-9: the
/// pipeline is deterministic, so anything beyond rounding noise means
/// we are not timing what the corpus claimed.
pub fn gate4_energy_reproduction(
    subject: &Subject,
    evidence: &EnergyEvidence,
    corpus: &CorpusIndex,
) -> Result<(), String> {
    if subject.arm_a != "baseline" {
        return Err(format!(
            "gate 4: arm A of {} is {}, but corpus scores are baseline-normalized",
            subject.name, subject.arm_a
        ));
    }
    let row_a = corpus
        .get(&subject.name, "baseline")
        .ok_or_else(|| format!("gate 4: {} has no baseline row", subject.name))?;
    let row_b = corpus
        .get(&subject.name, subject.arm_b)
        .ok_or_else(|| format!("gate 4: {}/{} has no corpus row", subject.name, subject.arm_b))?;
    let ratio_a = evidence.energy_a / evidence.energy_a.max(1.0);
    if (ratio_a - row_a.score).abs() > 1e-9 {
        return Err(format!(
            "gate 4 FAILED: {} baseline energy ratio {ratio_a:.12} != corpus score {:.12}",
            subject.name, row_a.score
        ));
    }
    if (evidence.ratio_b - row_b.score).abs() > 1e-9 {
        return Err(format!(
            "gate 4 FAILED: {}/{} energy ratio {:.12} != corpus score {:.12}",
            subject.name, subject.arm_b, evidence.ratio_b, row_b.score
        ));
    }
    Ok(())
}

// =============================================================================
// Child workload — the in-process core both the child mode and tests run
// =============================================================================

/// Which arm of a subject a child measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arm {
    A,
    B,
}

impl Arm {
    pub fn letter(self) -> &'static str {
        match self {
            Arm::A => "a",
            Arm::B => "b",
        }
    }

    pub fn parse(s: &str) -> Option<Arm> {
        match s {
            "a" => Some(Arm::A),
            "b" => Some(Arm::B),
            _ => None,
        }
    }
}

/// What one sustained-load phase measured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChildMeasurement {
    pub iters: u64,
    /// Wall-clock of the `iters`-run loop ONLY (compile+plan excluded).
    pub wall_micros: u64,
    /// Process peak RSS right after compile + plan application, before
    /// the first run — subtracting from `peak_rss_final_kb` isolates
    /// the execution contribution from compile-side memory.
    pub peak_rss_plan_kb: u64,
    /// Process peak RSS after the loop (the headline RSS number).
    pub peak_rss_final_kb: u64,
    pub output_fnv: u64,
}

/// Compile + apply a PARENT-SUPPLIED plan (untimed), then run the
/// sustained-load loop. The output FNV is re-checked every iteration:
/// drift DURING load is a determinism violation and hard-errors.
///
/// The child deliberately does NOT plan its own arm. Planning in the
/// child would (a) re-run the instrumented recording per phase and
/// (b) put the SELECTOR'S candidate-probing memory into the child's
/// peak RSS — measured at 1.63 GB vs a 206 MB baseline arm on the
/// real-workload subject, which would make the RSS comparison a
/// measurement of the planner, not the program. The parent computes
/// the plans once, gate 2 pins them to the corpus, and the child only
/// applies them (both safety verifiers still run here).
pub fn run_child_workload(
    subject: &Subject,
    arm: Arm,
    iters: u64,
    plan: &PassPlan,
) -> Result<ChildMeasurement, String> {
    if iters == 0 {
        return Err("iters must be >= 1".to_string());
    }
    let compiled = compile_subject(&subject.source)?;
    let config = match arm {
        Arm::A => subject.arm_a,
        Arm::B => subject.arm_b,
    };
    let optimized = optimize_with(
        &compiled.mir,
        plan,
        &format!("{}/{config}", subject.name),
    )?;

    let peak_rss_plan_kb = cjc_runtime::builtins::peak_rss_kb();

    let mut expected: Option<u64> = None;
    let start = Instant::now();
    for i in 0..iters {
        let out = run_mir(&compiled.ast, &optimized)?;
        let fnv = output_fnv(&out);
        match expected {
            None => expected = Some(fnv),
            Some(e) if e != fnv => {
                return Err(format!(
                    "{}/{config}: output drift during sustained load at iteration {i} \
                     ({e:#018x} -> {fnv:#018x})",
                    subject.name
                ))
            }
            _ => {}
        }
    }
    let wall_micros = start.elapsed().as_micros() as u64;
    let peak_rss_final_kb = cjc_runtime::builtins::peak_rss_kb();

    Ok(ChildMeasurement {
        iters,
        wall_micros,
        peak_rss_plan_kb,
        peak_rss_final_kb,
        output_fnv: expected.expect("iters >= 1 guarantees one run"),
    })
}

// =============================================================================
// Plan file protocol (parent -> child)
// =============================================================================

pub const PLAN_FILE_HEADER: &str = "CANA_PLAN_V1";

/// Serialize a plan for the child. Only PRESENT entries are written —
/// absence means "full default sequence" in `optimize_program_with_plan`
/// and must survive the round trip as absence (the absence-means-default
/// trap). Present-but-empty entries serialize as a bare function name.
pub fn serialize_plan(plan: &PassPlan) -> String {
    let mut out = String::from(PLAN_FILE_HEADER);
    out.push('\n');
    for (fn_name, passes) in &plan.per_function {
        out.push_str(fn_name);
        out.push('\t');
        out.push_str(&passes.join(","));
        out.push('\n');
    }
    out
}

/// Parse a plan file. `None` on any malformation — never panics
/// (bolero-fuzzed alongside the child line parser).
pub fn parse_plan(text: &str) -> Option<PassPlan> {
    let mut lines = text.lines();
    if lines.next()? != PLAN_FILE_HEADER {
        return None;
    }
    let mut plan = PassPlan::empty();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let (fn_name, passes) = line.split_once('\t')?;
        if fn_name.is_empty() {
            return None;
        }
        let pass_list: Vec<String> = if passes.is_empty() {
            Vec::new()
        } else {
            passes.split(',').map(|s| s.to_string()).collect()
        };
        plan.per_function.insert(fn_name.to_string(), pass_list);
    }
    Some(plan)
}

// =============================================================================
// Child line protocol (parent <-> child over stdout)
// =============================================================================

pub const CHILD_LINE_PREFIX: &str = "CANA_DIAG_CHILD_V1";

pub fn format_child_line(subject: &str, arm: Arm, m: &ChildMeasurement) -> String {
    format!(
        "{CHILD_LINE_PREFIX} subject={subject} arm={} iters={} wall_micros={} \
         rss_plan_kb={} rss_final_kb={} output_fnv={:016x}",
        arm.letter(),
        m.iters,
        m.wall_micros,
        m.peak_rss_plan_kb,
        m.peak_rss_final_kb,
        m.output_fnv
    )
}

/// Parsed child report. Field-for-field what `format_child_line` emits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedChild {
    pub subject: String,
    pub arm: Arm,
    pub measurement: ChildMeasurement,
}

/// Parse one child line. Returns `None` on ANY malformation — never
/// panics (bolero-fuzzed). Unknown keys are ignored for forward
/// compatibility; missing keys fail the parse.
pub fn parse_child_line(line: &str) -> Option<ParsedChild> {
    let mut tokens = line.split_whitespace();
    if tokens.next()? != CHILD_LINE_PREFIX {
        return None;
    }
    let mut subject: Option<String> = None;
    let mut arm: Option<Arm> = None;
    let mut iters: Option<u64> = None;
    let mut wall_micros: Option<u64> = None;
    let mut rss_plan_kb: Option<u64> = None;
    let mut rss_final_kb: Option<u64> = None;
    let mut output_fnv: Option<u64> = None;
    for tok in tokens {
        let (key, value) = tok.split_once('=')?;
        match key {
            "subject" => subject = Some(value.to_string()),
            "arm" => arm = Some(Arm::parse(value)?),
            "iters" => iters = Some(value.parse().ok()?),
            "wall_micros" => wall_micros = Some(value.parse().ok()?),
            "rss_plan_kb" => rss_plan_kb = Some(value.parse().ok()?),
            "rss_final_kb" => rss_final_kb = Some(value.parse().ok()?),
            "output_fnv" => output_fnv = Some(u64::from_str_radix(value, 16).ok()?),
            _ => {}
        }
    }
    Some(ParsedChild {
        subject: subject?,
        arm: arm?,
        measurement: ChildMeasurement {
            iters: iters?,
            wall_micros: wall_micros?,
            peak_rss_plan_kb: rss_plan_kb?,
            peak_rss_final_kb: rss_final_kb?,
            output_fnv: output_fnv?,
        },
    })
}

// =============================================================================
// Stats — medians, bands, ratios (diagnostics math, deterministic)
// =============================================================================

/// `(min, median, max)` of a sample — the confidence band the exit
/// criterion asks for (median-of-N spread).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Band {
    pub min: f64,
    pub med: f64,
    pub max: f64,
}

/// Median under `f64::total_cmp` (deterministic for all inputs,
/// NaN included). Even-length samples average the middle pair.
pub fn median(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    Some(if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    })
}

pub fn band(xs: &[f64]) -> Option<Band> {
    let med = median(xs)?;
    let min = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Some(Band { min, med, max })
}

/// B/A ratio band: the median ratio plus the most conservative bounds
/// the two bands allow (`lo` = best case for B, `hi` = worst case).
pub fn ratio_band(a: &Band, b: &Band) -> (f64, f64, f64) {
    let lo = b.min / a.max.max(f64::MIN_POSITIVE);
    let med = b.med / a.med.max(f64::MIN_POSITIVE);
    let hi = b.max / a.min.max(f64::MIN_POSITIVE);
    (lo, med, hi)
}

/// Phase iteration count: enough single runs to sustain
/// `target_phase_micros`, clamped to `[1, max_iters]`. Calibrated ONCE
/// on arm A so both arms run identical work.
pub fn calibrate_iters(single_run_micros: u64, target_phase_micros: u64, max_iters: u64) -> u64 {
    let single = single_run_micros.max(1);
    (target_phase_micros / single).clamp(1, max_iters.max(1))
}

/// A/B verdict from the conservative ratio band: a WIN only when the
/// ENTIRE band sits below 1.0, a REGRESSION only when it sits above.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Win,
    Regression,
    Inconclusive,
}

pub fn verdict(ratio_lo: f64, ratio_hi: f64) -> Verdict {
    if ratio_hi < 1.0 {
        Verdict::Win
    } else if ratio_lo > 1.0 {
        Verdict::Regression
    } else {
        Verdict::Inconclusive
    }
}

impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Win => "WIN",
            Verdict::Regression => "REGRESSION",
            Verdict::Inconclusive => "inconclusive",
        }
    }
}

// =============================================================================
// Parent orchestration
// =============================================================================

/// Protocol knobs. `standard()` is the auditor's spec; `quick()` is a
/// smoke profile for development and the e2e check in CI docs.
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// The diagnostics binary itself (children are self-respawns).
    pub child_exe: PathBuf,
    /// Empty = all subjects; otherwise exact-name membership (the
    /// config-name trap rule: never substring matching).
    pub subject_filter: Vec<String>,
    /// Warm-up phases per arm (excluded from stats).
    pub warmup_phases: u32,
    /// Measured phases per arm (median-of-N).
    pub measured_phases: u32,
    /// Sustained-load target per phase, microseconds.
    pub phase_target_micros: u64,
    /// Upper bound on per-phase iterations.
    pub max_iters: u64,
    pub out_dir: PathBuf,
}

impl RunOptions {
    pub fn standard(child_exe: PathBuf) -> Self {
        Self {
            child_exe,
            subject_filter: Vec::new(),
            warmup_phases: 1,
            measured_phases: 5,
            phase_target_micros: 5_000_000,
            max_iters: 200_000,
            out_dir: workspace_root().join("bench_results/cana_diagnostics"),
        }
    }

    pub fn quick(child_exe: PathBuf) -> Self {
        Self {
            measured_phases: 3,
            phase_target_micros: 500_000,
            ..Self::standard(child_exe)
        }
    }
}

/// One phase's raw record (CSV row).
#[derive(Debug, Clone)]
pub struct PhaseRecord {
    pub subject: String,
    pub family: &'static str,
    pub arm: Arm,
    pub config: String,
    pub phase_idx: u32,
    pub warmup: bool,
    pub measurement: ChildMeasurement,
}

/// Aggregated per-subject outcome (REPORT.md row).
#[derive(Debug, Clone)]
pub struct SubjectOutcome {
    pub name: String,
    pub family: &'static str,
    pub arm_a: &'static str,
    pub arm_b: &'static str,
    pub corpus_verified: bool,
    /// Identical plans ⇒ the subject is a noise-floor CONTROL.
    pub plans_differ: bool,
    pub modeled_ratio_b: f64,
    pub corpus_score_b: Option<f64>,
    pub iters: u64,
    /// Per-run wall micros bands over the measured phases.
    pub wall_a: Band,
    pub wall_b: Band,
    /// Peak-RSS (final) bands, kilobytes.
    pub rss_a: Band,
    pub rss_b: Band,
    pub wall_ratio: (f64, f64, f64),
    pub rss_ratio: (f64, f64, f64),
    pub wall_verdict: Verdict,
    pub rss_verdict: Verdict,
}

/// Spawn one child phase and parse + validate its report line.
fn spawn_child_phase(
    opts: &RunOptions,
    subject: &Subject,
    arm: Arm,
    iters: u64,
    plan_path: &Path,
    expected_fnv: u64,
) -> Result<ChildMeasurement, String> {
    let output = Command::new(&opts.child_exe)
        .arg("--child")
        .arg(&subject.name)
        .arg(arm.letter())
        .arg(iters.to_string())
        .arg(plan_path)
        .output()
        .map_err(|e| format!("failed to spawn child for {}: {e}", subject.name))?;
    if !output.status.success() {
        return Err(format!(
            "child {}/{} exited with {:?}: {}",
            subject.name,
            arm.letter(),
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed = stdout
        .lines()
        .find_map(parse_child_line)
        .ok_or_else(|| {
            format!(
                "child {}/{} produced no parseable report line; stdout: {stdout}",
                subject.name,
                arm.letter()
            )
        })?;
    if parsed.subject != subject.name || parsed.arm != arm {
        return Err(format!(
            "child identity mismatch: asked {}/{}, got {}/{}",
            subject.name,
            arm.letter(),
            parsed.subject,
            parsed.arm.letter()
        ));
    }
    if parsed.measurement.iters != iters {
        return Err(format!(
            "child {}/{} ran {} iters, asked {iters}",
            subject.name,
            arm.letter(),
            parsed.measurement.iters
        ));
    }
    if parsed.measurement.output_fnv != expected_fnv {
        return Err(format!(
            "DETERMINISM VIOLATION: child {}/{} output fnv {:#018x} != gate-3 transcript {:#018x}",
            subject.name,
            arm.letter(),
            parsed.measurement.output_fnv,
            expected_fnv
        ));
    }
    Ok(parsed.measurement)
}

/// Run the full protocol for every selected subject. Returns outcomes
/// and writes `phases.csv` + `REPORT.md` under `opts.out_dir`.
pub fn run_diagnostics(opts: &RunOptions) -> Result<Vec<SubjectOutcome>, String> {
    let heads = load_heads()?;
    let corpus = CorpusIndex::load()?;
    let all = subjects();
    let selected: Vec<&Subject> = if opts.subject_filter.is_empty() {
        all.iter().collect()
    } else {
        let missing: Vec<&String> = opts
            .subject_filter
            .iter()
            .filter(|f| !all.iter().any(|s| s.name == **f))
            .collect();
        if !missing.is_empty() {
            return Err(format!("unknown subject(s): {missing:?}"));
        }
        all.iter()
            .filter(|s| opts.subject_filter.iter().any(|f| *f == s.name))
            .collect()
    };

    fs::create_dir_all(&opts.out_dir)
        .map_err(|e| format!("cannot create {}: {e}", opts.out_dir.display()))?;

    let mut outcomes: Vec<SubjectOutcome> = Vec::new();
    let mut phase_records: Vec<PhaseRecord> = Vec::new();

    for subject in selected {
        println!("== {} ({}) ==", subject.name, subject.family);

        // -- Gates (all hard errors, before any clock) --------------------
        let compiled = compile_subject(&subject.source)?;
        let recorded = record_pressures(&compiled.ast)?;
        let plan_a = plan_for_config(
            subject.arm_a,
            &compiled.mir,
            &compiled.features,
            &recorded,
            &heads,
        )?;
        let plan_b = plan_for_config(
            subject.arm_b,
            &compiled.mir,
            &compiled.features,
            &recorded,
            &heads,
        )?;
        if subject.corpus_verified {
            gate1_program_hash(subject, &compiled.features, &corpus)?;
            gate2_plan_identity(subject, subject.arm_a, &plan_a, &corpus)?;
            gate2_plan_identity(subject, subject.arm_b, &plan_b, &corpus)?;
        }
        let opt_a = optimize_with(
            &compiled.mir,
            &plan_a,
            &format!("{}/{}", subject.name, subject.arm_a),
        )?;
        let opt_b = optimize_with(
            &compiled.mir,
            &plan_b,
            &format!("{}/{}", subject.name, subject.arm_b),
        )?;
        let (_transcript, expected_fnv) =
            gate3_output_determinism(&compiled.ast, &opt_a, &opt_b)?;
        let evidence = measure_modeled_energy(&compiled.ast, &opt_a, &opt_b)?;
        let corpus_score_b = if subject.corpus_verified {
            gate4_energy_reproduction(subject, &evidence, &corpus)?;
            corpus.get(&subject.name, subject.arm_b).map(|r| r.score)
        } else {
            None
        };
        let plans_differ = plan_as_sorted_vec(&plan_a) != plan_as_sorted_vec(&plan_b);
        println!(
            "   gates OK | plans differ: {} | modeled energy ratio B/A: {:.5}",
            if plans_differ { "yes" } else { "NO (control)" },
            evidence.ratio_b
        );

        // -- Persist the gate-2-verified plans for the children -----------
        let plans_dir = opts.out_dir.join("plans");
        fs::create_dir_all(&plans_dir)
            .map_err(|e| format!("cannot create {}: {e}", plans_dir.display()))?;
        let plan_path_a = plans_dir.join(format!("{}_a.plan", subject.name));
        let plan_path_b = plans_dir.join(format!("{}_b.plan", subject.name));
        fs::write(&plan_path_a, serialize_plan(&plan_a))
            .map_err(|e| format!("cannot write {}: {e}", plan_path_a.display()))?;
        fs::write(&plan_path_b, serialize_plan(&plan_b))
            .map_err(|e| format!("cannot write {}: {e}", plan_path_b.display()))?;

        // -- Calibration (arm A, one untimed-warm run then one timed) -----
        let _ = run_mir(&compiled.ast, &opt_a)?;
        let t0 = Instant::now();
        let _ = run_mir(&compiled.ast, &opt_a)?;
        let single_micros = t0.elapsed().as_micros() as u64;
        let iters = calibrate_iters(single_micros, opts.phase_target_micros, opts.max_iters);
        println!("   calibration: single run ~{single_micros} µs -> {iters} iters/phase");

        // -- Phases: warm-ups, then interleaved A/B/A/B... ----------------
        let mut wall_a: Vec<f64> = Vec::new();
        let mut wall_b: Vec<f64> = Vec::new();
        let mut rss_a: Vec<f64> = Vec::new();
        let mut rss_b: Vec<f64> = Vec::new();
        let total_phases = opts.warmup_phases + opts.measured_phases;
        for phase_idx in 0..total_phases {
            let warmup = phase_idx < opts.warmup_phases;
            for arm in [Arm::A, Arm::B] {
                let plan_path = match arm {
                    Arm::A => &plan_path_a,
                    Arm::B => &plan_path_b,
                };
                let m = spawn_child_phase(opts, subject, arm, iters, plan_path, expected_fnv)?;
                let per_run = m.wall_micros as f64 / m.iters as f64;
                if !warmup {
                    match arm {
                        Arm::A => {
                            wall_a.push(per_run);
                            rss_a.push(m.peak_rss_final_kb as f64);
                        }
                        Arm::B => {
                            wall_b.push(per_run);
                            rss_b.push(m.peak_rss_final_kb as f64);
                        }
                    }
                }
                println!(
                    "   phase {phase_idx}{} arm {}: {:.1} µs/run, peak RSS {} KB",
                    if warmup { " (warmup)" } else { "" },
                    arm.letter(),
                    per_run,
                    m.peak_rss_final_kb
                );
                phase_records.push(PhaseRecord {
                    subject: subject.name.clone(),
                    family: subject.family,
                    arm,
                    config: match arm {
                        Arm::A => subject.arm_a.to_string(),
                        Arm::B => subject.arm_b.to_string(),
                    },
                    phase_idx,
                    warmup,
                    measurement: m,
                });
            }
        }

        let wall_a_band = band(&wall_a).ok_or("no measured phases for arm A")?;
        let wall_b_band = band(&wall_b).ok_or("no measured phases for arm B")?;
        let rss_a_band = band(&rss_a).ok_or("no RSS samples for arm A")?;
        let rss_b_band = band(&rss_b).ok_or("no RSS samples for arm B")?;
        let wall_ratio = ratio_band(&wall_a_band, &wall_b_band);
        let rss_ratio = ratio_band(&rss_a_band, &rss_b_band);

        outcomes.push(SubjectOutcome {
            name: subject.name.clone(),
            family: subject.family,
            arm_a: subject.arm_a,
            arm_b: subject.arm_b,
            corpus_verified: subject.corpus_verified,
            plans_differ,
            modeled_ratio_b: evidence.ratio_b,
            corpus_score_b,
            iters,
            wall_a: wall_a_band,
            wall_b: wall_b_band,
            rss_a: rss_a_band,
            rss_b: rss_b_band,
            wall_ratio,
            rss_ratio,
            wall_verdict: verdict(wall_ratio.0, wall_ratio.2),
            rss_verdict: verdict(rss_ratio.0, rss_ratio.2),
        });
    }

    let csv = render_csv(&phase_records);
    fs::write(opts.out_dir.join("phases.csv"), csv)
        .map_err(|e| format!("cannot write phases.csv: {e}"))?;
    let report = render_report(&outcomes, opts);
    fs::write(opts.out_dir.join("REPORT.md"), report)
        .map_err(|e| format!("cannot write REPORT.md: {e}"))?;

    Ok(outcomes)
}

// =============================================================================
// Artifacts
// =============================================================================

pub fn render_csv(records: &[PhaseRecord]) -> String {
    let mut out = String::from(
        "subject,family,arm,config,phase_idx,warmup,iters,wall_micros,per_run_micros,rss_plan_kb,rss_final_kb\n",
    );
    for r in records {
        let m = &r.measurement;
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{:.3},{},{}\n",
            r.subject,
            r.family,
            r.arm.letter(),
            r.config,
            r.phase_idx,
            r.warmup,
            m.iters,
            m.wall_micros,
            m.wall_micros as f64 / m.iters as f64,
            m.peak_rss_plan_kb,
            m.peak_rss_final_kb
        ));
    }
    out
}

fn fmt_band(b: &Band) -> String {
    format!("{:.1} [{:.1}, {:.1}]", b.med, b.min, b.max)
}

fn fmt_ratio(r: (f64, f64, f64)) -> String {
    format!("{:.4} [{:.4}, {:.4}]", r.1, r.0, r.2)
}

pub fn render_report(outcomes: &[SubjectOutcome], opts: &RunOptions) -> String {
    let mut md = String::new();
    md.push_str("# Phase D — silicon diagnostics report\n\n");
    md.push_str(
        "Wall-clock + peak-RSS A/B of CANA plan choices. All subjects passed\n\
         the determinism gates (output byte-equality across AST-eval and both\n\
         arms; corpus program-hash, plan and modeled-energy identity where\n\
         applicable) BEFORE any timing was read. Ratios are arm B / arm A,\n\
         lower = B better. Bands are `median [min, max]` over the measured\n\
         phases; a verdict is only WIN/REGRESSION when the entire conservative\n\
         ratio band clears 1.0.\n\n",
    );
    md.push_str(&format!(
        "Protocol: {} warm-up + {} measured phases per arm, interleaved A/B; \
         ~{:.1} s sustained-load target per phase; fresh child process per \
         phase; per-phase iteration count calibrated once on arm A.\n\n",
        opts.warmup_phases,
        opts.measured_phases,
        opts.phase_target_micros as f64 / 1e6
    ));
    md.push_str(
        "Measured on one Windows machine; wall-clock and peak RSS only (CPU\n\
         frequency/temperature are out of MVP scope per the research doc §3\n\
         signal-reality audit). Within-machine deltas only — never compare\n\
         absolute numbers across machines.\n\n",
    );

    for family in ["selector", "thermal", "tensor", "nonsynthetic"] {
        let rows: Vec<&SubjectOutcome> = outcomes.iter().filter(|o| o.family == family).collect();
        if rows.is_empty() {
            continue;
        }
        md.push_str(&format!("## Family: {family}\n\n"));
        md.push_str(
            "| subject | arms (A vs B) | plans differ | modeled B/A | iters | wall A µs/run | wall B µs/run | wall ratio B/A | wall verdict | RSS A KB | RSS B KB | RSS ratio B/A | RSS verdict |\n",
        );
        md.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
        for o in rows {
            md.push_str(&format!(
                "| {} | {} vs {} | {} | {:.5} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                o.name,
                o.arm_a,
                o.arm_b,
                if o.plans_differ { "yes" } else { "NO (control)" },
                o.modeled_ratio_b,
                o.iters,
                fmt_band(&o.wall_a),
                fmt_band(&o.wall_b),
                fmt_ratio(o.wall_ratio),
                o.wall_verdict.label(),
                fmt_band(&o.rss_a),
                fmt_band(&o.rss_b),
                fmt_ratio(o.rss_ratio),
                o.rss_verdict.label()
            ));
        }
        md.push('\n');
    }

    // -- Headline: modeled vs measured on the subjects with real deltas --
    md.push_str("## Modeled vs measured (the Phase D question)\n\n");
    md.push_str("| subject | modeled energy B/A | wall-clock B/A (median) | agree? |\n");
    md.push_str("|---|---|---|---|\n");
    for o in outcomes.iter().filter(|o| o.plans_differ) {
        let modeled_says_win = o.modeled_ratio_b < 1.0 - 1e-9;
        let agree = match o.wall_verdict {
            Verdict::Win => modeled_says_win,
            Verdict::Regression => o.modeled_ratio_b > 1.0 + 1e-9,
            Verdict::Inconclusive => false,
        };
        md.push_str(&format!(
            "| {} | {:.5} | {:.4} | {} |\n",
            o.name,
            o.modeled_ratio_b,
            o.wall_ratio.1,
            if agree {
                "yes"
            } else if o.wall_verdict == Verdict::Inconclusive {
                "inconclusive"
            } else {
                "NO"
            }
        ));
    }
    md.push('\n');

    let wins = outcomes
        .iter()
        .filter(|o| o.wall_verdict == Verdict::Win)
        .count();
    let regressions = outcomes
        .iter()
        .filter(|o| o.wall_verdict == Verdict::Regression)
        .count();
    let controls = outcomes.iter().filter(|o| !o.plans_differ).count();
    md.push_str(&format!(
        "Summary: {} subjects ({} controls with identical plans); wall-clock \
         verdicts: {wins} WIN, {regressions} REGRESSION, {} inconclusive.\n",
        outcomes.len(),
        controls,
        outcomes.len() - wins - regressions
    ));
    md.push_str(
        "\nHard wall: nothing in this report feeds back into compile decisions,\n\
         hashes, or profile rows.\n",
    );
    md
}
