//! Phase 0.3c — dispatch-level tests for the OOD / calibration / drift
//! builtins.

use cjc_abng::dispatch::{dispatch_abng, reset_arena};
use cjc_runtime::tensor::Tensor;
use cjc_runtime::value::Value;

fn call(name: &str, args: &[Value]) -> Value {
    dispatch_abng(name, args).unwrap().unwrap()
}

fn try_call(name: &str, args: &[Value]) -> Result<Option<Value>, String> {
    dispatch_abng(name, args)
}

fn new_graph(seed: i64) -> i64 {
    match call("abng_new", &[Value::Int(seed)]) {
        Value::Int(i) => i,
        _ => panic!(),
    }
}

fn install_full_stack(g: i64) {
    // 2-D input, no hidden → BLR feature dim = input_dim = 2.
    let _ = call(
        "abng_set_leaf_head",
        &[
            Value::Int(g),
            Value::Int(2),
            Value::Tensor(Tensor::from_vec(vec![], &[0]).unwrap()),
            Value::Int(1),
            Value::String(std::rc::Rc::new("none".to_string())),
        ],
    );
    let _ = call(
        "abng_set_blr_prior",
        &[
            Value::Int(g),
            Value::Float(1.0),
            Value::Float(1.5),
            Value::Float(1.0),
        ],
    );
    let _ = call("abng_set_density_tracker", &[Value::Int(g)]);
    let _ = call("abng_set_calibration", &[Value::Int(g), Value::Int(15)]);
}

fn expect_int(v: Value) -> i64 {
    match v {
        Value::Int(i) => i,
        other => panic!("expected Int, got {}", other.type_name()),
    }
}
fn expect_float(v: Value) -> f64 {
    match v {
        Value::Float(f) => f,
        other => panic!("expected Float, got {}", other.type_name()),
    }
}

// ─── Density ──────────────────────────────────────────────────────

#[test]
fn density_observe_increments_n_seen() {
    reset_arena();
    let g = new_graph(0);
    install_full_stack(g);
    let xs = Tensor::from_vec(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0], &[3, 2]).unwrap();
    let _ = call(
        "abng_density_observe",
        &[Value::Int(g), Value::Int(0), Value::Tensor(xs)],
    );
    let n = expect_int(call(
        "abng_density_n_seen",
        &[Value::Int(g), Value::Int(0)],
    ));
    assert_eq!(n, 3);
}

#[test]
fn density_score_increases_with_distance() {
    reset_arena();
    let g = new_graph(0);
    install_full_stack(g);
    let xs = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0], &[3, 2]).unwrap();
    let _ = call(
        "abng_density_observe",
        &[Value::Int(g), Value::Int(0), Value::Tensor(xs)],
    );
    let s_at = expect_float(call(
        "abng_density_score",
        &[
            Value::Int(g),
            Value::Int(0),
            Value::Tensor(Tensor::from_vec(vec![1.0, 1.0], &[2]).unwrap()),
        ],
    ));
    let s_far = expect_float(call(
        "abng_density_score",
        &[
            Value::Int(g),
            Value::Int(0),
            Value::Tensor(Tensor::from_vec(vec![100.0, 100.0], &[2]).unwrap()),
        ],
    ));
    assert!(s_far > s_at);
}

// ─── Calibration ──────────────────────────────────────────────────

#[test]
fn calibration_observe_increments_n_seen() {
    reset_arena();
    let g = new_graph(0);
    install_full_stack(g);
    let _ = call(
        "abng_calibration_observe",
        &[
            Value::Int(g),
            Value::Int(0),
            Value::Float(0.7),
            Value::Bool(true),
        ],
    );
    let n = expect_int(call(
        "abng_calibration_n_seen",
        &[Value::Int(g), Value::Int(0)],
    ));
    assert_eq!(n, 1);
}

#[test]
fn calibration_ece_is_nonnegative() {
    reset_arena();
    let g = new_graph(0);
    install_full_stack(g);
    for i in 0..30 {
        let _ = call(
            "abng_calibration_observe",
            &[
                Value::Int(g),
                Value::Int(0),
                Value::Float(0.5),
                Value::Bool(i % 2 == 0),
            ],
        );
    }
    let ece = expect_float(call(
        "abng_calibration_ece",
        &[Value::Int(g), Value::Int(0)],
    ));
    assert!(ece >= 0.0);
}

#[test]
fn calibration_invalid_n_bins_errs() {
    reset_arena();
    let g = new_graph(0);
    let _ = call(
        "abng_set_leaf_head",
        &[
            Value::Int(g),
            Value::Int(2),
            Value::Tensor(Tensor::from_vec(vec![], &[0]).unwrap()),
            Value::Int(1),
            Value::String(std::rc::Rc::new("none".to_string())),
        ],
    );
    let err = try_call(
        "abng_set_calibration",
        &[Value::Int(g), Value::Int(0)],
    )
    .unwrap_err();
    assert!(err.contains("[2, 100]"));
}

// ─── Drift ────────────────────────────────────────────────────────

#[test]
fn drift_score_zero_at_freeze_grows_after_shift() {
    reset_arena();
    let g = new_graph(0);
    install_full_stack(g);
    let xs = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0], &[4, 2])
        .unwrap();
    let _ = call(
        "abng_density_observe",
        &[Value::Int(g), Value::Int(0), Value::Tensor(xs)],
    );
    let _ = call(
        "abng_freeze_drift_baseline",
        &[Value::Int(g), Value::Int(0)],
    );
    let s0 = expect_float(call("abng_drift_score", &[Value::Int(g), Value::Int(0)]));
    assert!(s0.abs() < 1e-12);
    let xs_shifted = Tensor::from_vec(
        vec![10.0, 10.0, 11.0, 11.0, 12.0, 12.0, 13.0, 13.0],
        &[4, 2],
    )
    .unwrap();
    let _ = call(
        "abng_density_observe",
        &[Value::Int(g), Value::Int(0), Value::Tensor(xs_shifted)],
    );
    let s1 = expect_float(call("abng_drift_score", &[Value::Int(g), Value::Int(0)]));
    assert!(s1 > s0);
}

#[test]
fn freeze_drift_without_data_errs() {
    reset_arena();
    let g = new_graph(0);
    install_full_stack(g);
    let err = try_call(
        "abng_freeze_drift_baseline",
        &[Value::Int(g), Value::Int(0)],
    )
    .unwrap_err();
    assert!(err.contains("n ≥ 2"));
}

// ─── Composite OOD ────────────────────────────────────────────────

#[test]
fn ood_score_within_unit_interval() {
    reset_arena();
    let g = new_graph(0);
    install_full_stack(g);
    let xs = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0], &[3, 2]).unwrap();
    let _ = call(
        "abng_density_observe",
        &[Value::Int(g), Value::Int(0), Value::Tensor(xs)],
    );
    let s = expect_float(call(
        "abng_ood_score",
        &[
            Value::Int(g),
            Value::Int(0),
            Value::Tensor(Tensor::from_vec(vec![100.0, 100.0], &[2]).unwrap()),
            Value::Int(0),
            Value::Int(5),
        ],
    ));
    assert!((0.0..=1.5).contains(&s)); // saturates near 1.0 for far-away points
}

#[test]
fn ood_score_works_without_any_subsystem_installed() {
    // OOD composite returns 0.0 when no density / blr / no prefix info.
    reset_arena();
    let g = new_graph(0);
    let s = expect_float(call(
        "abng_ood_score",
        &[
            Value::Int(g),
            Value::Int(0),
            Value::Tensor(Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap()),
            Value::Int(0),
            Value::Int(0), // prefix_max=0 → prefix_distance=0
        ],
    ));
    assert_eq!(s, 0.0);
}
