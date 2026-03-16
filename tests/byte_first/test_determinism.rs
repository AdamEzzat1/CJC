//! Determinism tests for the byte-first audit.

#[test]
fn rng_splitmix64_deterministic() {
    use cjc_repro::Rng;
    let mut r1 = Rng::seeded(42);
    let mut r2 = Rng::seeded(42);
    for _ in 0..1000 { assert_eq!(r1.next_u64(), r2.next_u64()); }
}

#[test]
fn rng_normal_deterministic() {
    use cjc_repro::Rng;
    let mut r1 = Rng::seeded(99);
    let mut r2 = Rng::seeded(99);
    for _ in 0..100 {
        assert_eq!(r1.next_normal_f64().to_bits(), r2.next_normal_f64().to_bits());
    }
}

#[test]
fn rng_fork_deterministic() {
    use cjc_repro::Rng;
    let mut r1 = Rng::seeded(42);
    let mut r2 = Rng::seeded(42);
    let mut f1 = r1.fork();
    let mut f2 = r2.fork();
    for _ in 0..100 { assert_eq!(f1.next_u64(), f2.next_u64()); }
    assert_eq!(r1.next_u64(), r2.next_u64());
}

#[test]
fn detmap_iteration_order_stable() {
    use cjc_runtime::det_map::DetMap;
    use cjc_runtime::Value;
    use std::rc::Rc;
    let mut m1 = DetMap::new();
    let mut m2 = DetMap::new();
    for i in 0..50 {
        let k = Value::String(Rc::new(format!("key_{}", i)));
        let v = Value::Int(i);
        m1.insert(k.clone(), v.clone());
        m2.insert(k, v);
    }
    let k1: Vec<_> = m1.keys();
    let k2: Vec<_> = m2.keys();
    assert_eq!(k1.len(), k2.len());
    for (a, b) in k1.iter().zip(k2.iter()) {
        assert!(cjc_runtime::det_map::values_equal_static(a, b));
    }
}

#[test]
fn program_determinism_complex() {
    let src = r#"fn main() -> f64 {
        let t = Tensor.randn([10, 10]);
        let r = t.matmul(t.transpose());
        r.sum()
    }"#;
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}",
        diags.diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>());
    let f1 = match cjc_eval::Interpreter::new(42).exec(&prog) {
        Ok(cjc_runtime::Value::Float(f)) => f, v => panic!("expected Float, got {:?}", v),
    };
    let f2 = match cjc_eval::Interpreter::new(42).exec(&prog) {
        Ok(cjc_runtime::Value::Float(f)) => f, v => panic!("expected Float, got {:?}", v),
    };
    assert_eq!(f1.to_bits(), f2.to_bits());
}

#[test]
fn different_seeds_differ() {
    let src = r#"fn main() -> f64 { Tensor.randn([10]).sum() }"#;
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let f1 = match cjc_eval::Interpreter::new(42).exec(&prog) {
        Ok(cjc_runtime::Value::Float(f)) => f, v => panic!("expected Float, got {:?}", v),
    };
    let f2 = match cjc_eval::Interpreter::new(99).exec(&prog) {
        Ok(cjc_runtime::Value::Float(f)) => f, v => panic!("expected Float, got {:?}", v),
    };
    assert_ne!(f1.to_bits(), f2.to_bits());
}

#[test]
fn btreemap_ordering_deterministic() {
    use std::collections::BTreeMap;
    let mut m1 = BTreeMap::new();
    let mut m2 = BTreeMap::new();
    m1.insert("z", 1); m1.insert("a", 2); m1.insert("m", 3);
    m2.insert("m", 3); m2.insert("z", 1); m2.insert("a", 2);
    let k1: Vec<_> = m1.keys().collect();
    let k2: Vec<_> = m2.keys().collect();
    assert_eq!(k1, k2);
    assert_eq!(k1, vec![&"a", &"m", &"z"]);
}

#[test]
fn kahan_sum_deterministic() {
    let data: Vec<f64> = (1..=1000).map(|i| i as f64 * 0.001).collect();
    let r1 = cjc_repro::kahan_sum_f64(&data);
    let r2 = cjc_repro::kahan_sum_f64(&data);
    assert_eq!(r1.to_bits(), r2.to_bits());
}

#[test]
fn binned_sum_order_invariant() {
    use cjc_runtime::accumulator::binned_sum_f64;
    let fwd: Vec<f64> = (1..=100).map(|i| i as f64).collect();
    let rev: Vec<f64> = fwd.iter().rev().copied().collect();
    assert_eq!(binned_sum_f64(&fwd).to_bits(), binned_sum_f64(&rev).to_bits());
}
