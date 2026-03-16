//! Parallel determinism tests for the byte-first audit.

fn eval_float(src: &str, seed: u64) -> f64 {
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>());
    match cjc_eval::Interpreter::new(seed).exec(&prog) {
        Ok(cjc_runtime::Value::Float(f)) => f,
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn matmul_repeated_runs_identical() {
    let src = r#"fn main() -> f64 {
        let a = Tensor.randn([16, 16]);
        let b = Tensor.randn([16, 16]);
        a.matmul(b).sum()
    }"#;
    let bits: Vec<u64> = (0..5).map(|_| eval_float(src, 42).to_bits()).collect();
    for i in 1..bits.len() {
        assert_eq!(bits[0], bits[i], "run {i} differs");
    }
}

#[test]
fn element_wise_ops_deterministic() {
    let src = r#"fn main() -> f64 {
        let a = Tensor.randn([1000]);
        let b = Tensor.randn([1000]);
        (a + b - a * b).sum()
    }"#;
    let f1 = eval_float(src, 42);
    let f2 = eval_float(src, 42);
    assert_eq!(f1.to_bits(), f2.to_bits());
}

#[test]
fn simd_binop_large_tensor() {
    use cjc_runtime::tensor::Tensor;
    use cjc_repro::Rng;
    let mut rng1 = Rng::seeded(42);
    let a = Tensor::randn(&[200, 200], &mut rng1);
    let mut rng2 = Rng::seeded(99);
    let b = Tensor::randn(&[200, 200], &mut rng2);
    let c1 = a.add(&b).unwrap();
    let c2 = a.add(&b).unwrap();
    let v1 = c1.to_vec();
    let v2 = c2.to_vec();
    for i in 0..v1.len() {
        assert_eq!(v1[i].to_bits(), v2[i].to_bits(), "elem {i} differs");
    }
}

#[test]
fn eval_mir_parity_tensor_ops() {
    let src = r#"fn main() -> f64 {
        let a = Tensor.randn([5, 5]);
        let b = Tensor.randn([5, 5]);
        a.matmul(b).sum()
    }"#;
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>());
    let fe = match cjc_eval::Interpreter::new(42).exec(&prog) {
        Ok(cjc_runtime::Value::Float(f)) => f, other => panic!("eval: {:?}", other),
    };
    let fm = match cjc_mir_exec::run_program_with_executor(&prog, 42) {
        Ok((cjc_runtime::Value::Float(f), _)) => f, _ => panic!("mir: expected Float"),
    };
    assert_eq!(fe.to_bits(), fm.to_bits());
}

#[test]
fn tiled_matmul_deterministic() {
    use cjc_runtime::tensor_tiled::TiledMatmul;
    let a: Vec<f64> = (0..100).map(|i| (i as f64) * 0.1).collect();
    let b: Vec<f64> = (0..100).map(|i| ((100 - i) as f64) * 0.1).collect();
    let c1 = TiledMatmul::with_tile_size(4).matmul(&a, 10, 10, &b, 10);
    let c2 = TiledMatmul::with_tile_size(4).matmul(&a, 10, 10, &b, 10);
    for (i, (x, y)) in c1.iter().zip(c2.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "elem {i} differs");
    }
}

#[test]
fn no_hashmap_in_value_types() {
    use std::collections::BTreeMap;
    use cjc_runtime::Value;
    let mut fields = BTreeMap::new();
    fields.insert("x".to_string(), Value::Int(1));
    fields.insert("y".to_string(), Value::Int(2));
    let s = Value::Struct { name: "P".to_string(), fields };
    if let Value::Struct { fields, .. } = &s {
        let keys: Vec<_> = fields.keys().cloned().collect();
        assert_eq!(keys, vec!["x".to_string(), "y".to_string()]);
    }
}
