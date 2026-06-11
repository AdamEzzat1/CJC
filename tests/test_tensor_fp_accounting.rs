//! Phase A1 — tensor FP accounting WIRING tests.
//!
//! The A1 probe (`bench/cana_tensor_probe`) showed the recorded
//! thermal label and the static feature basis were both blind to
//! tensor FP work (label 0.0 on programs running 409,600+ FP ops).
//! These tests lock the dual-side fix end-to-end:
//!
//! 1. The runtime counter fires through the REAL instrumented-runner
//!    path (binop arms, free-call dispatch, method dispatch).
//! 2. The recorded label reads tensor programs as thermally hot.
//! 3. The static pipeline (TypeMix tensor propagation → MemoryProxy
//!    method classification → `build_physical_query`) prices tensor FP.
//! 4. Instrumentation does not perturb output (the Option-B identity).
//! 5. Eval ↔ MIR-exec parity holds on tensor programs — the
//!    precondition for the ablation harness's tensor family.

use std::collections::BTreeMap;

use cjc_cana::analyze_program;
use cjc_cana::physical_cost::build_physical_query;
use cjc_cana::pressure::PressurePredictor;
use cjc_cana_nss::RecordedPressurePredictor;
use cjc_mir_exec::run_program_instrumented;

const SEED: u64 = 42;

/// Element-wise tensor binop hot loop — exercises the `(Tensor, Tensor)`
/// and `(Tensor, Float)` binary arms. 200 iterations × 2 ops × 1,024
/// elements ≥ 409,600 tensor FP; builder contributes 4,096 scalar FP.
const TENSOR_EW: &str = r#"
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

/// Matmul + method-call reduction — exercises the free-call dispatch
/// (`matmul`) and the method dispatch (`.sum()`).
const TENSOR_MM_SUM: &str = r#"
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
    let mut s: f64 = 0.0;
    let mut i: i64 = 0;
    while i < iters {
        let c: Tensor = matmul(a, b);
        s = c.sum();
        i = i + 1;
    }
    return s;
}
let a: Tensor = build(16, 0.5);
let b: Tensor = build(16, 0.25);
print(mm_hot(a, b, 50));
"#;

/// Scalar-only control: the accounting must not invent FP work.
const INT_CONTROL: &str = r#"
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

fn instrumented_label(src: &str) -> f64 {
    let (ast, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse: {:?}", diags.diagnostics);
    let mut al = cjc_hir::AstLowering::new();
    let hir = al.lower_program(&ast);
    let mut h2m = cjc_mir::HirToMir::new();
    let mir = h2m.lower_program(&hir);
    let features = analyze_program(&mir).features;
    let (_val, exec, events) = run_program_instrumented(&ast, SEED).expect("instrumented run");
    let recorded = RecordedPressurePredictor::from_recorded_events(
        events,
        exec.trace_node_assignments().clone(),
    );
    let max_of = |m: BTreeMap<String, f64>| m.values().copied().fold(0.0f64, f64::max);
    max_of(recorded.predict_thermal(&mir, &features))
}

fn trace_fp_total(src: &str) -> f64 {
    let (ast, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse: {:?}", diags.diagnostics);
    let (_val, _exec, events) = run_program_instrumented(&ast, SEED).expect("instrumented run");
    events
        .iter()
        .map(|ev| ev.thermal_intensity * ev.instruction_count as f64)
        .sum()
}

fn static_float_ops(src: &str) -> u64 {
    let (ast, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse: {:?}", diags.diagnostics);
    let mut al = cjc_hir::AstLowering::new();
    let hir = al.lower_program(&ast);
    let mut h2m = cjc_mir::HirToMir::new();
    let mir = h2m.lower_program(&hir);
    let features = analyze_program(&mir).features;
    features
        .per_fn
        .iter()
        .map(|(name, ff)| build_physical_query(name, "dce", ff).float_ops_estimate)
        .fold(0u64, |a, b| a.saturating_add(b))
}

// ---------------------------------------------------------------------------
// 1+2 — runtime counter fires; recorded label reads hot
// ---------------------------------------------------------------------------

#[test]
fn tensor_programs_read_thermally_hot() {
    // Pre-fix these labels were 0.0000 (A1 probe, 2026-06-11).
    assert!(
        instrumented_label(TENSOR_EW) >= 0.5,
        "element-wise tensor program must read hot"
    );
    assert!(
        instrumented_label(TENSOR_MM_SUM) >= 0.5,
        "matmul+sum tensor program must read hot"
    );
}

#[test]
fn int_control_stays_cold_with_zero_fp() {
    assert_eq!(trace_fp_total(INT_CONTROL), 0.0);
    assert!(instrumented_label(INT_CONTROL) < 0.05);
}

#[test]
fn trace_counter_exceeds_scalar_only_bound() {
    // The element-wise program's builder contributes exactly 4,096
    // scalar FP; pre-fix the trace total matched it exactly. Post-fix
    // the tensor windows add (cap-bounded) density on top.
    let total = trace_fp_total(TENSOR_EW);
    assert!(
        total > 4096.0 * 1.05,
        "trace FP total {total} must exceed the scalar-only bound"
    );
}

// ---------------------------------------------------------------------------
// 3 — static pipeline prices tensor FP
// ---------------------------------------------------------------------------

#[test]
fn static_basis_prices_tensor_fp_work() {
    // Pre-fix: 16 (scalar builder only). Post-fix: tensor binops and
    // call sites enter at TENSOR_FP_PER_OP each, loop-amplified.
    assert!(
        static_float_ops(TENSOR_EW) > 1000,
        "element-wise binops must be priced"
    );
    assert!(
        static_float_ops(TENSOR_MM_SUM) > 1000,
        "matmul free call + .sum() method must be priced"
    );
    assert_eq!(static_float_ops(INT_CONTROL), 0);
}

// ---------------------------------------------------------------------------
// 4 — instrumentation does not perturb output
// ---------------------------------------------------------------------------

#[test]
fn instrumented_output_identical_on_tensor_programs() {
    for src in [TENSOR_EW, TENSOR_MM_SUM] {
        let (ast, diags) = cjc_parser::parse_source(src);
        assert!(!diags.has_errors());
        let (_v1, exec_plain) =
            cjc_mir_exec::run_program_with_executor(&ast, SEED).expect("plain run");
        let (_v2, exec_inst, _events) =
            run_program_instrumented(&ast, SEED).expect("instrumented run");
        assert_eq!(
            exec_plain.output, exec_inst.output,
            "tracing must not change program output"
        );
    }
}

// ---------------------------------------------------------------------------
// 5 — eval ↔ MIR-exec parity (precondition for the ablation tensor family)
// ---------------------------------------------------------------------------

#[test]
fn eval_mir_parity_on_tensor_programs() {
    for (name, src) in [
        ("tensor_ew", TENSOR_EW),
        ("tensor_mm_sum", TENSOR_MM_SUM),
        ("int_control", INT_CONTROL),
    ] {
        let (ast, diags) = cjc_parser::parse_source(src);
        assert!(!diags.has_errors(), "{name}: {:?}", diags.diagnostics);
        let mut interp = cjc_eval::Interpreter::new(SEED);
        let eval_result = interp.exec(&ast);
        let (_val, exec) =
            cjc_mir_exec::run_program_with_executor(&ast, SEED).expect("mir-exec run");
        assert!(eval_result.is_ok(), "{name}: eval failed: {eval_result:?}");
        assert_eq!(
            interp.output, exec.output,
            "{name}: eval and MIR-exec outputs must be byte-identical"
        );
    }
}
