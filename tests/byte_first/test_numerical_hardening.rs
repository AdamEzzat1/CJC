//! Numerical hardening tests for the byte-first audit.

fn eval_float(src: &str, seed: u64) -> f64 {
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>());
    match cjc_eval::Interpreter::new(seed).exec(&prog) {
        Ok(cjc_runtime::Value::Float(f)) => f,
        Ok(other) => panic!("expected Float, got {:?}", other),
        Err(e) => panic!("eval error: {:?}", e),
    }
}

fn mir_float(src: &str, seed: u64) -> f64 {
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>());
    match cjc_mir_exec::run_program_with_executor(&prog, seed) {
        Ok((cjc_runtime::Value::Float(f), _)) => f,
        Ok((other, _)) => panic!("expected Float, got {:?}", other),
        Err(e) => panic!("mir error: {e}"),
    }
}

#[test]
fn dot_product_binned() {
    let src = r#"fn main() -> f64 {
        let a = Tensor.from_vec([1e16, 1.0, -1e16], [3]);
        let b = Tensor.from_vec([1.0, 1.0, 1.0], [3]);
        dot(a, b)
    }"#;
    assert_eq!(eval_float(src, 42), 1.0, "dot should use binned sum");
}

#[test]
fn dot_product_parity() {
    let src = r#"fn main() -> f64 {
        let a = Tensor.from_vec([1.5, 2.5, 3.5], [3]);
        let b = Tensor.from_vec([4.0, 5.0, 6.0], [3]);
        dot(a, b)
    }"#;
    let fe = eval_float(src, 42);
    let fm = mir_float(src, 42);
    assert_eq!(fe.to_bits(), fm.to_bits(), "dot parity: {fe} vs {fm}");
}

#[test]
fn norm_l2_basic() {
    let src = r#"fn main() -> f64 { norm(Tensor.from_vec([3.0, 4.0], [2]), 2) }"#;
    assert_eq!(eval_float(src, 42), 5.0);
}

#[test]
fn float_nan_propagation() {
    let src = r#"fn main() -> f64 { 0.0 / 0.0 }"#;
    assert!(eval_float(src, 42).is_nan());
}

#[test]
fn float_infinity() {
    let src = r#"fn main() -> f64 { 1.0 / 0.0 }"#;
    let f = eval_float(src, 42);
    assert!(f.is_infinite() && f > 0.0);
}

#[test]
fn tensor_sum_deterministic() {
    let src = r#"fn main() -> f64 { Tensor.randn([100]).sum() }"#;
    assert_eq!(eval_float(src, 42).to_bits(), eval_float(src, 42).to_bits());
}

#[test]
fn matmul_deterministic() {
    let src = r#"fn main() -> f64 {
        let a = Tensor.randn([8, 8]);
        let b = Tensor.randn([8, 8]);
        a.matmul(b).sum()
    }"#;
    assert_eq!(eval_float(src, 42).to_bits(), eval_float(src, 42).to_bits());
}

#[test]
fn binned_accumulator_signed_zero() {
    use cjc_runtime::accumulator::BinnedAccumulatorF64;
    let mut acc = BinnedAccumulatorF64::new();
    acc.add(0.0);
    acc.add(-0.0);
    assert_eq!(acc.finalize(), 0.0);
}

#[test]
fn binned_accumulator_extreme_magnitudes() {
    use cjc_runtime::accumulator::BinnedAccumulatorF64;
    let mut acc = BinnedAccumulatorF64::new();
    acc.add(1e308);
    acc.add(1e-308);
    acc.add(-1e308);
    assert_eq!(acc.finalize(), 1e-308);
}

#[test]
fn binned_accumulator_mixed_inf_nan() {
    use cjc_runtime::accumulator::BinnedAccumulatorF64;
    let mut acc = BinnedAccumulatorF64::new();
    acc.add(f64::INFINITY);
    acc.add(f64::NAN);
    assert!(acc.finalize().is_nan());
}

#[test]
fn snap_nan_canonical() {
    let b1 = cjc_snap::snap_encode(&cjc_runtime::Value::Float(f64::NAN));
    let b2 = cjc_snap::snap_encode(&cjc_runtime::Value::Float(f64::from_bits(0x7FF8_0000_0000_0001)));
    assert_eq!(b1, b2, "NaN canonicalization");
}
