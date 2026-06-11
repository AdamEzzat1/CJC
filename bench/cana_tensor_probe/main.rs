//! Phase A item 1 — tensor-blindness probe (handoff §3.1).
//!
//! ## The question
//!
//! `TypeMix` counts SCALAR float binops; the runtime FP-op counter
//! (`cjc-mir-exec` trace site 4) increments only on scalar
//! `(Float, Float)` / `(Int, Float)` / `(Float, Int)` binop dispatch.
//! Tensor binary ops (element-wise add/mul/...) and tensor builtins
//! (`matmul`, `.sum()`, ...) execute thousands of hardware FP
//! operations per MIR statement yet increment NEITHER instrument.
//!
//! Hypothesis (handoff §6, "high prior"): the thermal head whiffs on
//! tensor workloads. The subtlety this probe must resolve: blindness
//! is suspected on BOTH sides — the recorded label AND the static
//! feature. If both are blind, head-vs-label MAE looks perfect on
//! tensor programs while both mismeasure physical reality. So the
//! probe reports three comparisons per program:
//!
//! 1. recorded thermal label vs v2-head prediction (shadow-style),
//! 2. trace-counted FP ops vs an ANALYTIC lower bound on true FP work
//!    (hand-derived from loop trip counts × tensor shapes — the
//!    tie-breaker between "accurate" and "agreeing blind"),
//! 3. static `float_ops_estimate` / `tensor_heavy_ops` vs the same
//!    bound (does the feature basis carry signal the head COULD use?).
//!
//! ## Determinism
//!
//! Seed 42, no wall-clock in any reported number, Kahan reductions,
//! BTreeMap iteration. Read-only: writes nothing to disk.

use std::collections::BTreeMap;
use std::path::PathBuf;

use cjc_cana::physical_cost::{build_physical_query, predict_physical, PhysicalCoefficients};
use cjc_cana::pinn_thermal_v2::PinnThermalV2;
use cjc_cana::pressure::PressurePredictor;
use cjc_cana::{analyze_program, features::CanaFeatures};
use cjc_cana_compress::pinn_bundle::read_bundle;
use cjc_cana_nss::RecordedPressurePredictor;
use cjc_mir::MirProgram;
use cjc_mir_exec::run_program_instrumented;
use cjc_repro::KahanAccumulatorF64;

const SEED: u64 = 42;
const BUNDLE_PATH: &str = "bench_results/cana_train_pinn/pinn_thermal_v2.cpb";

// =============================================================================
// Probe programs
// =============================================================================
//
// Analytic FP lower bounds count ONLY float arithmetic that provably
// executes (mul/add per matmul MAC, per element-wise op, per Kahan
// accumulation ≥1 add per element, per scalar float binop). Loop
// counters / comparisons on ints are excluded. Bounds are LOWER
// bounds: Kahan compensation and internal temporaries only add work.

struct ProbeProgram {
    name: &'static str,
    source: &'static str,
    /// Analytic lower bound on scalar float binops (the population the
    /// runtime counter CAN see).
    true_scalar_fp: u64,
    /// Analytic lower bound on tensor-internal FP ops (the population
    /// both instruments currently CANNOT see).
    true_tensor_fp: u64,
}

/// Control 1 — pure integer loop. Every instrument should read ~0.
const PROBE_INT: &str = r#"
fn work(n: i64) -> i64 {
    let mut acc: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        acc = acc + i * 3;
        i = i + 1;
    }
    return acc;
}
print(work(2000));
"#;

/// Control 2 — dense SCALAR float arithmetic (the fp_hot pattern the
/// corpus already measures well). 200 × 16 inner iterations × 4 float
/// binops = 12,800 scalar FP ops.
const PROBE_SCALAR_FP: &str = r#"
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

/// Tensor workload 1 — matmul hot loop. build(16): 256 elements × 2
/// float binops = 512 scalar FP per tensor (1,024 for two). mm_hot:
/// 50 matmuls of [16×16]·[16×16] ≥ 2·16³ = 8,192 FP each → 409,600
/// tensor FP. The `.get` extraction adds no float arithmetic.
const PROBE_TENSOR_MATMUL: &str = r#"
fn build(n: i64, scale: f64) -> Tensor {
    let mut buf: Any = [];
    let mut i: i64 = 0;
    while i < n * n {
        buf = array_push(buf, scale * float(i % 7) + 0.25);
        i = i + 1;
    }
    return Tensor.from_vec(buf, [n, n]);
}
fn mm_hot(a: Tensor, b: Tensor, iters: i64) -> f64 {
    let mut last: f64 = 0.0;
    let mut i: i64 = 0;
    while i < iters {
        let c: Tensor = matmul(a, b);
        last = c.get([0, 0]);
        i = i + 1;
    }
    return last;
}
let a: Tensor = build(16, 0.5);
let b: Tensor = build(16, 0.25);
print(mm_hot(a, b, 50));
"#;

/// Tensor workload 2 — element-wise binops (the EXACT runtime sites
/// that skip the FP counter: `(Tensor, Tensor)` binary dispatch).
/// build(32): 1,024 elements × 2 = 2,048 scalar FP per tensor (4,096
/// for two). ew_hot: 200 iterations × 2 element-wise ops × 1,024
/// elements = 409,600 tensor FP.
const PROBE_TENSOR_ELEMWISE: &str = r#"
fn build(n: i64, scale: f64) -> Tensor {
    let mut buf: Any = [];
    let mut i: i64 = 0;
    while i < n * n {
        buf = array_push(buf, scale * float(i % 7) + 0.25);
        i = i + 1;
    }
    return Tensor.from_vec(buf, [n, n]);
}
fn ew_hot(a: Tensor, b: Tensor, iters: i64) -> Tensor {
    let mut c: Tensor = a + b;
    let mut i: i64 = 1;
    while i < iters {
        c = a * b;
        c = c + a;
        i = i + 1;
    }
    return c;
}
let a: Tensor = build(32, 0.5);
let b: Tensor = build(32, 0.25);
let r: Tensor = ew_hot(a, b, 200);
print(1);
"#;

/// Tensor workload 3 — reduction hot loop. build(64): 4,096 elements
/// × 2 = 8,192 scalar FP. red_hot: 100 × `.sum()` over 4,096 elements
/// ≥ 409,600 tensor FP (Kahan adds more).
const PROBE_TENSOR_REDUCE: &str = r#"
fn build(n: i64, scale: f64) -> Tensor {
    let mut buf: Any = [];
    let mut i: i64 = 0;
    while i < n * n {
        buf = array_push(buf, scale * float(i % 7) + 0.25);
        i = i + 1;
    }
    return Tensor.from_vec(buf, [n, n]);
}
fn red_hot(a: Tensor, iters: i64) -> f64 {
    let mut s: f64 = 0.0;
    let mut i: i64 = 0;
    while i < iters {
        s = a.sum();
        i = i + 1;
    }
    return s;
}
let a: Tensor = build(64, 0.5);
print(red_hot(a, 100));
"#;

/// Tensor workload 4 — realistic mix: matmul dominates, a scalar
/// accumulator gives the counter a faint heartbeat. Scalar: builder
/// 1,024 + 50 × 1 accumulate ≈ 1,074. Tensor: 50 × 8,192 = 409,600.
const PROBE_TENSOR_MIX: &str = r#"
fn build(n: i64, scale: f64) -> Tensor {
    let mut buf: Any = [];
    let mut i: i64 = 0;
    while i < n * n {
        buf = array_push(buf, scale * float(i % 7) + 0.25);
        i = i + 1;
    }
    return Tensor.from_vec(buf, [n, n]);
}
fn mix_hot(a: Tensor, b: Tensor, iters: i64) -> f64 {
    let mut acc: f64 = 0.0;
    let mut i: i64 = 0;
    while i < iters {
        let c: Tensor = matmul(a, b);
        acc = acc + 0.001;
        i = i + 1;
    }
    return acc;
}
let a: Tensor = build(16, 0.5);
let b: Tensor = build(16, 0.25);
print(mix_hot(a, b, 50));
"#;

const PROBES: &[ProbeProgram] = &[
    ProbeProgram {
        name: "probe_int_control",
        source: PROBE_INT,
        true_scalar_fp: 0,
        true_tensor_fp: 0,
    },
    ProbeProgram {
        name: "probe_scalar_fp_hot",
        source: PROBE_SCALAR_FP,
        true_scalar_fp: 12_800,
        true_tensor_fp: 0,
    },
    ProbeProgram {
        name: "probe_tensor_matmul",
        source: PROBE_TENSOR_MATMUL,
        true_scalar_fp: 1_024,
        true_tensor_fp: 409_600,
    },
    ProbeProgram {
        name: "probe_tensor_elemwise",
        source: PROBE_TENSOR_ELEMWISE,
        true_scalar_fp: 4_096,
        true_tensor_fp: 409_600,
    },
    ProbeProgram {
        name: "probe_tensor_reduce",
        source: PROBE_TENSOR_REDUCE,
        true_scalar_fp: 8_192,
        true_tensor_fp: 409_600,
    },
    ProbeProgram {
        name: "probe_tensor_mix",
        source: PROBE_TENSOR_MIX,
        true_scalar_fp: 1_074,
        true_tensor_fp: 409_600,
    },
];

// =============================================================================
// Per-program measurement
// =============================================================================

struct Measurement {
    /// Recorded thermal label — what v2 trains/shadows against.
    recorded_thermal_max: f64,
    /// v2 trained head, max over per-function queries (predict path).
    v2_pred_max: f64,
    /// v1 closed form, same granularity.
    v1_pred_max: f64,
    /// Σ instruction_count over trace events.
    instr_total: u64,
    /// Σ thermal_intensity·instruction_count — reconstructs the
    /// runtime FP counter (same identity the ablation energy formula
    /// uses).
    trace_fp_total: f64,
    /// Static feature sums across functions.
    est_float_ops: u64,
    est_flops: u64,
    float_binops_static: u64,
    tensor_heavy_ops_static: u64,
}

fn lower_to_mir(src: &str, name: &str) -> (cjc_ast::Program, MirProgram, CanaFeatures) {
    let (ast, diags) = cjc_parser::parse_source(src);
    assert!(
        !diags.has_errors(),
        "parse errors in {name}: {:?}",
        diags.diagnostics
    );
    let mut al = cjc_hir::AstLowering::new();
    let hir = al.lower_program(&ast);
    let mut h2m = cjc_mir::HirToMir::new();
    let mir = h2m.lower_program(&hir);
    let features = analyze_program(&mir).features;
    (ast, mir, features)
}

fn measure(probe: &ProbeProgram, v2_head: &PinnThermalV2) -> Measurement {
    let (ast, mir, features) = lower_to_mir(probe.source, probe.name);

    // -- Label path: one instrumented run, recorded predictor, max thermal --
    let (_val, exec, events) =
        run_program_instrumented(&ast, SEED).unwrap_or_else(|e| panic!("{}: {e:?}", probe.name));
    let mut instr_total = 0u64;
    let mut fp_acc = KahanAccumulatorF64::new();
    for ev in &events {
        instr_total = instr_total.saturating_add(ev.instruction_count as u64);
        fp_acc.add(ev.thermal_intensity * ev.instruction_count as f64);
    }
    let recorded = RecordedPressurePredictor::from_recorded_events(
        events,
        exec.trace_node_assignments().clone(),
    );
    let max_of = |m: BTreeMap<String, f64>| m.values().copied().fold(0.0f64, f64::max);
    let recorded_thermal_max = max_of(recorded.predict_thermal(&mir, &features));

    // -- Prediction path: per-function queries, neutral pass (mirrors the
    //    ablation harness's row construction) --------------------------------
    let coeffs = PhysicalCoefficients::default();
    let mut v2_pred_max = 0.0f64;
    let mut v1_pred_max = 0.0f64;
    let mut est_float_ops = 0u64;
    let mut est_flops = 0u64;
    let mut float_binops_static = 0u64;
    let mut tensor_heavy_ops_static = 0u64;
    for (fn_name, ff) in &features.per_fn {
        let q = build_physical_query(fn_name, "dce", ff);
        est_float_ops = est_float_ops.saturating_add(q.float_ops_estimate);
        est_flops = est_flops.saturating_add(q.flops_estimate);
        v2_pred_max = v2_pred_max.max(v2_head.predict_thermal(&q));
        if let Some(est) = predict_physical(&q, &coeffs) {
            v1_pred_max = v1_pred_max.max(est.thermal_pressure);
        }
        float_binops_static =
            float_binops_static.saturating_add(ff.type_mix.float_binop_count as u64);
        tensor_heavy_ops_static =
            tensor_heavy_ops_static.saturating_add(ff.memory.tensor_heavy_ops as u64);
    }

    Measurement {
        recorded_thermal_max,
        v2_pred_max,
        v1_pred_max,
        instr_total,
        trace_fp_total: fp_acc.finalize(),
        est_float_ops,
        est_flops,
        float_binops_static,
        tensor_heavy_ops_static,
    }
}

// =============================================================================
// Main — table + verdict
// =============================================================================

fn main() {
    let v2_head = read_bundle(&PathBuf::from(BUNDLE_PATH))
        .expect("CPB0 bundle missing — run `cargo run --release -p cana-train-pinn -- train`")
        .head;

    println!("=================================================================");
    println!("Phase A1 — tensor-blindness probe (seed {SEED}, read-only)");
    println!("=================================================================\n");

    println!(
        "{:<22} | {:>9} | {:>8} | {:>8} | {:>10} | {:>12} | {:>12} | {:>12}",
        "program", "rec_label", "v2_pred", "v1_pred", "trace_fp", "true_scalar", "true_tensor", "instr_total"
    );
    println!("{}", "-".repeat(118));

    let mut results: Vec<(&ProbeProgram, Measurement)> = Vec::new();
    for probe in PROBES {
        let m = measure(probe, &v2_head);
        println!(
            "{:<22} | {:>9.4} | {:>8.4} | {:>8.4} | {:>10.0} | {:>12} | {:>12} | {:>12}",
            probe.name,
            m.recorded_thermal_max,
            m.v2_pred_max,
            m.v1_pred_max,
            m.trace_fp_total,
            probe.true_scalar_fp,
            probe.true_tensor_fp,
            m.instr_total,
        );
        results.push((probe, m));
    }

    println!("\nStatic feature view (what the head could possibly see):");
    println!(
        "{:<22} | {:>14} | {:>12} | {:>14} | {:>12}",
        "program", "est_float_ops", "est_flops", "float_binops", "tensor_ops"
    );
    println!("{}", "-".repeat(86));
    for (probe, m) in &results {
        println!(
            "{:<22} | {:>14} | {:>12} | {:>14} | {:>12}",
            probe.name, m.est_float_ops, m.est_flops, m.float_binops_static, m.tensor_heavy_ops_static,
        );
    }

    // -- Verdict ---------------------------------------------------------------
    //
    // Probe validity (controls):
    //  * int control: trace FP must be exactly 0 (no phantom FP).
    //  * scalar control: label must read HOT (≥0.9) and the trace must
    //    recover ≥50% of the analytic scalar bound. Full recovery is
    //    impossible by construction: `thermal_intensity` is capped at
    //    1.0 per window (`thermal_raw.min(1.0)`), so statements with
    //    >1 FP binop clip — a separate probe finding, quantified below.
    //
    // Blindness (per tensor program): the trace total must match the
    // analytic SCALAR subset almost exactly (the counter saw the
    // scalar ops and nothing else) while the tensor bound goes
    // unseen, and the label must read ≤0.05 where ≥400k FP ops ran.
    println!("\nVerdict inputs:");
    let mut label_blind_programs = 0usize;
    let mut scalar_exact_matches = 0usize;
    let mut tensor_programs = 0usize;
    for (probe, m) in &results {
        if probe.true_tensor_fp == 0 {
            continue;
        }
        tensor_programs += 1;
        let true_total = (probe.true_scalar_fp + probe.true_tensor_fp) as f64;
        let coverage = m.trace_fp_total / true_total;
        let scalar_delta = (m.trace_fp_total - probe.true_scalar_fp as f64).abs()
            / (probe.true_scalar_fp.max(1)) as f64;
        let agree = (m.v2_pred_max - m.recorded_thermal_max).abs();
        println!(
            "  {:<22} trace covers {:>5.2}% of analytic FP | trace-vs-scalar-only Δ = {:>5.2}% | label = {:.4} | head-vs-label |Δ| = {:.4}",
            probe.name,
            coverage * 100.0,
            scalar_delta * 100.0,
            m.recorded_thermal_max,
            agree
        );
        if scalar_delta < 0.05 {
            scalar_exact_matches += 1;
        }
        if coverage < 0.10 {
            label_blind_programs += 1;
        }
    }

    let int_control = results
        .iter()
        .find(|(p, _)| p.name == "probe_int_control")
        .expect("control present");
    let scalar_control = results
        .iter()
        .find(|(p, _)| p.name == "probe_scalar_fp_hot")
        .expect("control present");
    let scalar_coverage =
        scalar_control.1.trace_fp_total / scalar_control.0.true_scalar_fp as f64;
    println!(
        "  {:<22} int-control trace FP = {} (must be 0); scalar-control coverage = {:.2}% (cap-lossy floor 50%), label = {:.4}",
        "controls",
        int_control.1.trace_fp_total,
        scalar_coverage * 100.0,
        scalar_control.1.recorded_thermal_max
    );
    println!(
        "  note: scalar-control shortfall below 100% measures the per-window
       1.0 intensity cap (multi-FP-op statements clip), not counter loss."
    );

    let controls_valid = int_control.1.trace_fp_total == 0.0
        && scalar_coverage >= 0.50
        && scalar_control.1.recorded_thermal_max >= 0.90;

    // The decision-relevant instrument is the LABEL: a program running
    // 400k+ FP ops must read hot. (The trace-coverage column above is
    // informational only — the event stream carries a per-window
    // density capped at 1.0, so the reconstruction is bounded by
    // instruction count even when the counter itself saw every FP op.)
    let labels_blind = results
        .iter()
        .filter(|(p, _)| p.true_tensor_fp > 0)
        .filter(|(_, m)| m.recorded_thermal_max < 0.05)
        .count();
    let labels_hot = results
        .iter()
        .filter(|(p, _)| p.true_tensor_fp > 0)
        .filter(|(_, m)| m.recorded_thermal_max >= 0.50)
        .count();
    let _ = (label_blind_programs, scalar_exact_matches); // table-only inputs

    println!("\n=================================================================");
    if !controls_valid {
        println!("PROBE INVALID — controls not recovered; fix the probe before");
        println!("drawing any blindness conclusion.");
    } else if labels_blind == tensor_programs && tensor_programs > 0 {
        println!("CONFIRMED: tensor blindness — the recorded thermal label reads");
        println!("≈ 0 on all {tensor_programs} tensor programs despite 409,600+ analytic FP ops");
        println!("(where the dense-scalar control reads 1.0). The 2026-06-11 A1");
        println!("probe measured exactly this (labels 0.0000, est_float_ops ≈");
        println!("scalar-only) before the dual-side accounting fix landed.");
        println!("→ Fix BOTH sides: runtime counter (label) and static feature,");
        println!("  then regen + retrain + re-shadow.");
    } else if labels_hot == tensor_programs && tensor_programs > 0 {
        println!("NOT BLIND (post-fix state): all {tensor_programs} tensor programs read hot");
        println!("(label ≥ 0.5) and the static basis prices their tensor FP work.");
        println!("This is the expected output after the Phase A1 accounting fix;");
        println!("a return to 'CONFIRMED' above means the fix regressed.");
    } else {
        println!(
            "MIXED: {labels_blind}/{tensor_programs} tensor programs blind, {labels_hot}/{tensor_programs} hot — \
             the accounting covers some op forms but not others; check the per-program table."
        );
    }
    println!("=================================================================");
}
