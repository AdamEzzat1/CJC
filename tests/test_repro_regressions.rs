//! Milestone 2.7 — Repro Regression Tests
//!
//! Verifies that the hybrid deterministic summation infrastructure
//! integrates correctly with the CJC language runtime, eval, and MIR-exec.

use cjc_runtime::accumulator::{binned_sum_f64, binned_sum_f32};
use cjc_runtime::dispatch::{
    ExecMode, ReproMode, ReductionContext, SumStrategy,
    select_strategy, dispatch_sum_f64, dispatch_dot_f64,
};
use cjc_runtime::Tensor;
use cjc_repro::kahan_sum_f64;
use cjc_eval::Interpreter;
use cjc_parser::parse_source;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn eval_output(src: &str) -> Vec<String> {
    let (program, diag) = parse_source(src);
    assert!(!diag.has_errors(), "Parse errors:\n{}", diag.render_all(src, "<test>"));
    let mut interp = Interpreter::new(42);
    match interp.exec(&program) {
        Ok(_) => {}
        Err(e) => panic!("eval error: {:?}", e),
    }
    interp.output.clone()
}

fn mir_output(src: &str) -> Vec<String> {
    let (program, diag) = parse_source(src);
    assert!(!diag.has_errors(), "Parse errors:\n{}", diag.render_all(src, "<test>"));
    let result = cjc_mir_exec::run_program_with_executor(&program, 42);
    match result {
        Ok((_, exec)) => exec.output.clone(),
        Err(e) => panic!("MIR error: {:?}", e),
    }
}

fn assert_parity(src: &str) {
    let eval = eval_output(src);
    let mir = mir_output(src);
    assert_eq!(eval, mir, "Parity failure:\neval={eval:?}\nmir={mir:?}\nsrc={src}");
}

// ---------------------------------------------------------------------------
// 1. Accumulator Unit Tests
// ---------------------------------------------------------------------------

#[test]
fn test_binned_f64_exact_integers() {
    let values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
    assert_eq!(binned_sum_f64(&values), 5050.0);
}

#[test]
fn test_binned_f32_exact_integers() {
    let values: Vec<f32> = (1..=100).map(|i| i as f32).collect();
    assert_eq!(binned_sum_f32(&values), 5050.0);
}

#[test]
fn test_binned_negative_values() {
    let values = vec![-1.0, -2.0, -3.0, -4.0, -5.0];
    assert_eq!(binned_sum_f64(&values), -15.0);
}

#[test]
fn test_binned_mixed_sign() {
    let values = vec![10.0, -3.0, 5.0, -7.0, 1.0];
    assert_eq!(binned_sum_f64(&values), 6.0);
}

#[test]
fn test_binned_empty() {
    assert_eq!(binned_sum_f64(&[]), 0.0);
}

#[test]
fn test_binned_single() {
    assert_eq!(binned_sum_f64(&[42.0]), 42.0);
}

// ---------------------------------------------------------------------------
// 2. Dispatch Strategy Tests
// ---------------------------------------------------------------------------

#[test]
fn test_strategy_serial_on_is_kahan() {
    let ctx = ReductionContext {
        exec_mode: ExecMode::Serial,
        repro_mode: ReproMode::On,
        in_nogc: false,
        is_linalg: false,
    };
    assert_eq!(select_strategy(&ctx), SumStrategy::Kahan);
}

#[test]
fn test_strategy_serial_off_is_kahan() {
    let ctx = ReductionContext {
        exec_mode: ExecMode::Serial,
        repro_mode: ReproMode::Off,
        in_nogc: false,
        is_linalg: false,
    };
    assert_eq!(select_strategy(&ctx), SumStrategy::Kahan);
}

#[test]
fn test_strategy_parallel_is_binned() {
    let ctx = ReductionContext {
        exec_mode: ExecMode::Parallel,
        repro_mode: ReproMode::On,
        in_nogc: false,
        is_linalg: false,
    };
    assert_eq!(select_strategy(&ctx), SumStrategy::Binned);
}

#[test]
fn test_strategy_strict_is_binned() {
    let ctx = ReductionContext {
        exec_mode: ExecMode::Serial,
        repro_mode: ReproMode::Strict,
        in_nogc: false,
        is_linalg: false,
    };
    assert_eq!(select_strategy(&ctx), SumStrategy::Binned);
}

#[test]
fn test_strategy_nogc_is_binned() {
    let ctx = ReductionContext {
        exec_mode: ExecMode::Serial,
        repro_mode: ReproMode::On,
        in_nogc: true,
        is_linalg: false,
    };
    assert_eq!(select_strategy(&ctx), SumStrategy::Binned);
}

#[test]
fn test_strategy_linalg_is_binned() {
    let ctx = ReductionContext {
        exec_mode: ExecMode::Serial,
        repro_mode: ReproMode::On,
        in_nogc: false,
        is_linalg: true,
    };
    assert_eq!(select_strategy(&ctx), SumStrategy::Binned);
}

// ---------------------------------------------------------------------------
// 3. Dispatch Function Tests
// ---------------------------------------------------------------------------

#[test]
fn test_dispatch_sum_kahan_path() {
    let ctx = ReductionContext::default_serial();
    let values = vec![1.0, 2.0, 3.0];
    let kahan = kahan_sum_f64(&values);
    let dispatched = dispatch_sum_f64(&values, &ctx);
    assert_eq!(dispatched.to_bits(), kahan.to_bits());
}

#[test]
fn test_dispatch_sum_binned_path() {
    let ctx = ReductionContext::strict_parallel();
    let values = vec![1.0, 2.0, 3.0];
    let binned = binned_sum_f64(&values);
    let dispatched = dispatch_sum_f64(&values, &ctx);
    assert_eq!(dispatched.to_bits(), binned.to_bits());
}

#[test]
fn test_dispatch_dot_agreement() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let kahan_ctx = ReductionContext::default_serial();
    let binned_ctx = ReductionContext::strict_parallel();

    let k = dispatch_dot_f64(&a, &b, &kahan_ctx);
    let bn = dispatch_dot_f64(&a, &b, &binned_ctx);

    assert_eq!(k, 70.0);
    assert_eq!(bn, 70.0);
}

// ---------------------------------------------------------------------------
// 4. Tensor Method Tests
// ---------------------------------------------------------------------------

#[test]
fn test_tensor_sum_kahan_default() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
    assert_eq!(t.sum(), 15.0);
}

#[test]
fn test_tensor_binned_sum() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
    assert_eq!(t.binned_sum(), 15.0);
}

#[test]
fn test_tensor_dispatched_sum() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
    let ctx = ReductionContext::linalg();
    assert_eq!(t.dispatched_sum(&ctx), 15.0);
}

#[test]
fn test_tensor_sum_and_binned_agree_simple() {
    let t = Tensor::from_vec((1..=100).map(|i| i as f64).collect(), &[100]).unwrap();
    assert_eq!(t.sum(), t.binned_sum());
}

// ---------------------------------------------------------------------------
// 5. CJC Language Integration (Eval)
// ---------------------------------------------------------------------------

#[test]
fn test_cjc_eval_binned_sum_basic() {
    let out = eval_output("let t = Tensor.ones([100]); print(t.binned_sum());");
    assert_eq!(out[0], "100");
}

#[test]
fn test_cjc_eval_binned_sum_matches_sum() {
    // For a small tensor, Kahan and Binned should agree closely.
    let out = eval_output(
        "let t = Tensor.ones([50]); \
         let s1 = t.sum(); \
         let s2 = t.binned_sum(); \
         print(s1 == s2);"
    );
    assert_eq!(out[0], "true");
}

// ---------------------------------------------------------------------------
// 6. CJC Language Integration (MIR-exec)
// ---------------------------------------------------------------------------

#[test]
fn test_cjc_mir_binned_sum_basic() {
    let out = mir_output("let t = Tensor.ones([100]); print(t.binned_sum());");
    assert_eq!(out[0], "100");
}

#[test]
fn test_cjc_mir_binned_sum_matches_sum() {
    let out = mir_output(
        "let t = Tensor.ones([50]); \
         let s1 = t.sum(); \
         let s2 = t.binned_sum(); \
         print(s1 == s2);"
    );
    assert_eq!(out[0], "true");
}

// ---------------------------------------------------------------------------
// 7. Parity Tests (Eval vs MIR-exec)
// ---------------------------------------------------------------------------

#[test]
fn test_parity_binned_sum() {
    assert_parity("let t = Tensor.ones([200]); print(t.binned_sum());");
}

#[test]
fn test_parity_binned_sum_randn() {
    assert_parity("let t = Tensor.randn([100]); print(t.binned_sum());");
}

// ---------------------------------------------------------------------------
// 8. Kernel Dispatched Variants
// ---------------------------------------------------------------------------

#[test]
fn test_kernel_matmul_dispatched() {
    let ctx = ReductionContext::linalg();
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let mut c = vec![0.0; 4];
    cjc_runtime::kernel::matmul_dispatched(&a, &b, &mut c, 2, 2, 2, &ctx);
    assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_kernel_matmul_dispatched_vs_raw() {
    let ctx = ReductionContext::linalg();
    let m = 8;
    let k = 16;
    let n = 8;

    let mut rng = cjc_repro::Rng::seeded(42);
    let a: Vec<f64> = (0..m * k).map(|_| rng.next_normal_f64()).collect();
    let b: Vec<f64> = (0..k * n).map(|_| rng.next_normal_f64()).collect();

    let mut c_raw = vec![0.0; m * n];
    let mut c_disp = vec![0.0; m * n];
    cjc_runtime::kernel::matmul_raw(&a, &b, &mut c_raw, m, k, n);
    cjc_runtime::kernel::matmul_dispatched(&a, &b, &mut c_disp, m, k, n, &ctx);

    for i in 0..c_raw.len() {
        let diff = (c_raw[i] - c_disp[i]).abs();
        assert!(diff < 1e-8,
            "Matmul raw vs dispatched differ at {i}: {} vs {} (diff={diff})",
            c_raw[i], c_disp[i]);
    }
}

#[test]
fn test_kernel_linear_dispatched() {
    let ctx = ReductionContext::linalg();
    let x = vec![1.0, 2.0, 3.0];
    let w = vec![1.0, 0.0, 1.0,  0.0, 1.0, 0.0];
    let bias = vec![0.5, -0.5];
    let mut out = vec![0.0; 2];
    cjc_runtime::kernel::linear_dispatched(&x, &w, &bias, &mut out, 1, 3, 2, &ctx);
    assert_eq!(out, vec![4.5, 1.5]);
}

#[test]
fn test_kernel_conv1d_dispatched() {
    let ctx = ReductionContext::linalg();
    let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let filters = vec![1.0, 0.0, -1.0];
    let bias = vec![0.0];
    let mut out = vec![0.0; 3];
    cjc_runtime::kernel::conv1d_dispatched(&signal, &filters, &bias, &mut out, 5, 1, 3, &ctx);
    assert_eq!(out, vec![-2.0, -2.0, -2.0]);
}

// ---------------------------------------------------------------------------
// 9. NoGC Verification
// ---------------------------------------------------------------------------

#[test]
fn test_binned_sum_in_nogc_function() {
    let src = "\
nogc fn compute() -> Float {\n\
    let t = Tensor.ones([50]);\n\
    return t.binned_sum();\n\
}\n\
print(compute());\n";
    let eval_out = eval_output(src);
    assert_eq!(eval_out[0], "50");

    let mir_out = mir_output(src);
    assert_eq!(mir_out[0], "50");
}

// ---------------------------------------------------------------------------
// 10. Determinism Across Summation Methods
// ---------------------------------------------------------------------------

#[test]
fn test_all_methods_agree_on_integers() {
    let values: Vec<f64> = (1..=1000).map(|i| i as f64).collect();
    let expected = 500500.0;

    assert_eq!(kahan_sum_f64(&values), expected);
    assert_eq!(binned_sum_f64(&values), expected);
    assert_eq!(dispatch_sum_f64(&values, &ReductionContext::default_serial()), expected);
    assert_eq!(dispatch_sum_f64(&values, &ReductionContext::strict_parallel()), expected);
    assert_eq!(dispatch_sum_f64(&values, &ReductionContext::nogc()), expected);
    assert_eq!(dispatch_sum_f64(&values, &ReductionContext::linalg()), expected);
}
