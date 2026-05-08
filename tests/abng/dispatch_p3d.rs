//! Phase 0.3d-1 — dispatch-level tests for the maturity + signature
//! builtins.

use std::cell::RefCell;
use std::rc::Rc;

use cjc_abng::dispatch::{dispatch_abng, reset_arena};
use cjc_runtime::tensor::Tensor;
use cjc_runtime::value::Value;

fn call(name: &str, args: &[Value]) -> Value {
    dispatch_abng(name, args).unwrap().unwrap()
}

fn call_err(name: &str, args: &[Value]) -> String {
    dispatch_abng(name, args).unwrap_err()
}

fn new_graph(seed: i64) -> i64 {
    match call("abng_new", &[Value::Int(seed)]) {
        Value::Int(i) => i,
        _ => panic!("abng_new did not return Int"),
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
            Value::String(Rc::new("none".to_string())),
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

fn expect_tensor(v: Value) -> Tensor {
    match v {
        Value::Tensor(t) => t,
        other => panic!("expected Tensor, got {}", other.type_name()),
    }
}

fn expect_bytes(v: Value) -> Rc<RefCell<Vec<u8>>> {
    match v {
        Value::Bytes(b) => b,
        other => panic!("expected Bytes, got {}", other.type_name()),
    }
}

// ─── abng_node_maturity ───────────────────────────────────────────

#[test]
fn maturity_fresh_root_is_all_zeros() {
    reset_arena();
    let g = new_graph(0);
    let v = call(
        "abng_node_maturity",
        &[Value::Int(g), Value::Int(0)],
    );
    let t = expect_tensor(v);
    assert_eq!(t.shape(), &[4]);
    let data = t.to_vec();
    assert_eq!(data, vec![0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn maturity_climbs_with_samples_via_dispatch() {
    reset_arena();
    let g = new_graph(0);
    for _ in 0..64 {
        let _ = call(
            "abng_observe",
            &[Value::Int(g), Value::Int(0), Value::Float(1.0)],
        );
    }
    let v = call(
        "abng_node_maturity",
        &[Value::Int(g), Value::Int(0)],
    );
    let data = expect_tensor(v).to_vec();
    assert_eq!(data[0], 64.0);
    assert_eq!(data[1], 0.0); // calibration_stable stub
    assert_eq!(data[2], 0.0); // uncertainty_stable stub
    assert_eq!(data[3], 1.0); // trust_level
}

#[test]
fn maturity_bad_node_id_errs() {
    reset_arena();
    let g = new_graph(0);
    let err = call_err(
        "abng_node_maturity",
        &[Value::Int(g), Value::Int(99)],
    );
    assert!(err.contains("out of range"), "err = {err}");
}

#[test]
fn maturity_unknown_graph_errs() {
    reset_arena();
    let err = call_err(
        "abng_node_maturity",
        &[Value::Int(9_999_999), Value::Int(0)],
    );
    assert!(err.contains("no graph"), "err = {err}");
}

#[test]
fn maturity_arity_check() {
    reset_arena();
    let g = new_graph(0);
    let err = call_err("abng_node_maturity", &[Value::Int(g)]);
    assert!(err.contains("expected 2 arguments"), "err = {err}");
}

#[test]
fn maturity_negative_node_id_errs() {
    reset_arena();
    let g = new_graph(0);
    let err = call_err(
        "abng_node_maturity",
        &[Value::Int(g), Value::Int(-1)],
    );
    assert!(err.contains("non-negative"), "err = {err}");
}

// ─── abng_node_signature ──────────────────────────────────────────

#[test]
fn signature_fresh_root_is_32_bytes() {
    reset_arena();
    let g = new_graph(0);
    let v = call(
        "abng_node_signature",
        &[Value::Int(g), Value::Int(0)],
    );
    let bytes = expect_bytes(v);
    let data = bytes.borrow();
    // Phase 0.4 Track B-2.2.1 — signatures populate via Welford
    // observations driven by `decide_step`. A fresh root has zero
    // observations on every profile; all 32 bytes are the all-zeros
    // sentinel.
    assert_eq!(data.len(), 32);
    assert_eq!(&data[..], &[0u8; 32]);
}

#[test]
fn signature_changes_with_subsystem_install() {
    // Phase 0.4 Track B-2.2.1 — to populate signatures, drive a
    // `decide_step` call after subsystem install. The Welfords fold
    // one observation per profile per call.
    reset_arena();
    let g_bare = new_graph(0);
    let bare = expect_bytes(call(
        "abng_node_signature",
        &[Value::Int(g_bare), Value::Int(0)],
    ))
    .borrow()
    .clone();

    let g_full = new_graph(0);
    install_full_stack(g_full);
    // Train BLR so its posterior has a non-zero mean — the
    // `epistemic_leverage_at_posterior_mean` helper returns None
    // when leverage is zero (which it is at the prior with mean=0,
    // since `predict([0; d])` evaluates leverage at zero), so the
    // uncertainty Welford never observes. A single update lifts the
    // posterior off the prior.
    let features = Tensor::from_vec(vec![1.0, 0.5], &[1, 2]).unwrap();
    let y = Tensor::from_vec(vec![1.0], &[1]).unwrap();
    let _ = call(
        "abng_blr_update",
        &[
            Value::Int(g_full),
            Value::Int(0),
            Value::Tensor(features),
            Value::Tensor(y),
        ],
    );
    // Install policy + decide_step so the Welfords observe.
    let _ = call(
        "abng_set_decision_policy",
        &[Value::Int(g_full), Value::Tensor(ok_thresholds_tensor())],
    );
    let _ = call("abng_decide_step", &[Value::Int(g_full)]);
    let full = expect_bytes(call(
        "abng_node_signature",
        &[Value::Int(g_full), Value::Int(0)],
    ))
    .borrow()
    .clone();

    assert_ne!(bare, full);
    // After decide_step folded one observation per profile, all four
    // 8-byte profile fields are now non-zero.
    assert_ne!(&full[0..8], &[0u8; 8]);
    assert_ne!(&full[8..16], &[0u8; 8]);
    assert_ne!(&full[16..24], &[0u8; 8]);
    assert_ne!(&full[24..32], &[0u8; 8]);
}

#[test]
fn signature_bad_node_id_errs() {
    reset_arena();
    let g = new_graph(0);
    let err = call_err(
        "abng_node_signature",
        &[Value::Int(g), Value::Int(99)],
    );
    assert!(err.contains("out of range"), "err = {err}");
}

#[test]
fn signature_unknown_graph_errs() {
    reset_arena();
    let err = call_err(
        "abng_node_signature",
        &[Value::Int(9_999_999), Value::Int(0)],
    );
    assert!(err.contains("no graph"), "err = {err}");
}

#[test]
fn signature_arity_check() {
    reset_arena();
    let g = new_graph(0);
    let err = call_err("abng_node_signature", &[Value::Int(g)]);
    assert!(err.contains("expected 2 arguments"), "err = {err}");
}

#[test]
fn signature_dispatch_double_run_byte_identical() {
    let mk = || {
        reset_arena();
        let g = new_graph(42);
        install_full_stack(g);
        let _ = call(
            "abng_observe",
            &[Value::Int(g), Value::Int(0), Value::Float(1.5)],
        );
        let v = call(
            "abng_node_signature",
            &[Value::Int(g), Value::Int(0)],
        );
        expect_bytes(v).borrow().clone()
    };
    assert_eq!(mk(), mk());
}

// ─── abng_set_expected_epistemic / abng_expected_epistemic ───────

fn expect_float(v: Value) -> f64 {
    match v {
        Value::Float(f) => f,
        other => panic!("expected Float, got {}", other.type_name()),
    }
}

#[test]
fn expected_epistemic_uncaptured_returns_sentinel() {
    reset_arena();
    let g = new_graph(0);
    install_full_stack(g);
    let v = call(
        "abng_expected_epistemic",
        &[Value::Int(g), Value::Int(0)],
    );
    assert_eq!(expect_float(v), -1.0);
}

#[test]
fn set_then_get_expected_epistemic() {
    reset_arena();
    let g = new_graph(0);
    install_full_stack(g);
    let _ = call(
        "abng_set_expected_epistemic",
        &[Value::Int(g), Value::Int(0), Value::Float(0.5)],
    );
    let v = call(
        "abng_expected_epistemic",
        &[Value::Int(g), Value::Int(0)],
    );
    assert!((expect_float(v) - 0.5).abs() < 1e-12);
}

#[test]
fn set_expected_epistemic_one_shot_dispatch_errs() {
    reset_arena();
    let g = new_graph(0);
    install_full_stack(g);
    let _ = call(
        "abng_set_expected_epistemic",
        &[Value::Int(g), Value::Int(0), Value::Float(0.5)],
    );
    let err = call_err(
        "abng_set_expected_epistemic",
        &[Value::Int(g), Value::Int(0), Value::Float(0.7)],
    );
    assert!(err.contains("already captured"), "err = {err}");
}

#[test]
fn set_expected_epistemic_no_blr_errs() {
    reset_arena();
    let g = new_graph(0); // bare graph — no BLR installed
    let err = call_err(
        "abng_set_expected_epistemic",
        &[Value::Int(g), Value::Int(0), Value::Float(0.5)],
    );
    assert!(err.contains("BLR posterior"), "err = {err}");
}

#[test]
fn set_expected_epistemic_invalid_value_errs() {
    reset_arena();
    let g = new_graph(0);
    install_full_stack(g);
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let err = call_err(
            "abng_set_expected_epistemic",
            &[Value::Int(g), Value::Int(0), Value::Float(bad)],
        );
        assert!(
            err.contains("strictly positive"),
            "expected positive-error for {bad}, got {err}"
        );
    }
}

#[test]
fn set_expected_epistemic_arity_check() {
    reset_arena();
    let g = new_graph(0);
    let err = call_err(
        "abng_set_expected_epistemic",
        &[Value::Int(g), Value::Int(0)],
    );
    assert!(err.contains("expected 3 arguments"), "err = {err}");
}

#[test]
fn expected_epistemic_arity_check() {
    reset_arena();
    let g = new_graph(0);
    let err = call_err("abng_expected_epistemic", &[Value::Int(g)]);
    assert!(err.contains("expected 2 arguments"), "err = {err}");
}

// ─── Phase 0.3d-3 — DecisionPolicy + force-* + inspection ─────

fn ok_thresholds_tensor() -> Tensor {
    // Phase 0.4 Track B-2.2.7 — drift_unfreeze added at index 11.
    // Phase 0.4-extended (v11) — ece_stability_max_delta + sigma_stability_ratio at 12, 13.
    Tensor::from_vec(
        vec![
            0.5, 64.0, 128.0, 0.05, 0.02, 4.0, 0.1, 32.0, 10.0, 8.0, 20.0, f64::MAX,
            0.005, 1.05,
        ],
        &[14],
    )
    .unwrap()
}

fn expect_bool(v: Value) -> bool {
    match v {
        Value::Bool(b) => b,
        other => panic!("expected Bool, got {}", other.type_name()),
    }
}

fn expect_int(v: Value) -> i64 {
    match v {
        Value::Int(i) => i,
        other => panic!("expected Int, got {}", other.type_name()),
    }
}

fn expect_string(v: Value) -> std::rc::Rc<String> {
    match v {
        Value::String(s) => s,
        other => panic!("expected String, got {}", other.type_name()),
    }
}

#[test]
fn set_decision_policy_then_hash() {
    reset_arena();
    let g = new_graph(0);
    // Hash is empty before install.
    let h0 = expect_string(call("abng_decision_policy_hash", &[Value::Int(g)]));
    assert!(h0.is_empty());
    let _ = call(
        "abng_set_decision_policy",
        &[Value::Int(g), Value::Tensor(ok_thresholds_tensor())],
    );
    let h1 = expect_string(call("abng_decision_policy_hash", &[Value::Int(g)]));
    assert_eq!(h1.len(), 64); // 32-byte hash → 64 hex chars
}

#[test]
fn set_decision_policy_one_shot_dispatch_errs() {
    reset_arena();
    let g = new_graph(0);
    let _ = call(
        "abng_set_decision_policy",
        &[Value::Int(g), Value::Tensor(ok_thresholds_tensor())],
    );
    let err = call_err(
        "abng_set_decision_policy",
        &[Value::Int(g), Value::Tensor(ok_thresholds_tensor())],
    );
    assert!(err.contains("already frozen"), "err = {err}");
}

#[test]
fn force_grow_then_action_count() {
    reset_arena();
    let g = new_graph(0);
    let c = expect_int(call(
        "abng_force_grow",
        &[Value::Int(g), Value::Int(0), Value::Int(7)],
    ));
    assert_eq!(c, 1);
    let n = expect_int(call(
        "abng_action_count",
        &[Value::Int(g), Value::Int(0)], // index 0 = Grow
    ));
    assert_eq!(n, 1);
}

#[test]
fn force_grow_invalid_key_errs() {
    reset_arena();
    let g = new_graph(0);
    let err = call_err(
        "abng_force_grow",
        &[Value::Int(g), Value::Int(0), Value::Int(300)],
    );
    assert!(err.contains("key_byte must be in"), "err = {err}");
}

#[test]
fn force_split_returns_two_node_ids() {
    reset_arena();
    let g = new_graph(0);
    let v = call("abng_force_split", &[Value::Int(g), Value::Int(0)]);
    let t = match v {
        Value::Tensor(t) => t,
        _ => panic!("expected Tensor"),
    };
    assert_eq!(t.shape(), &[2]);
    let data = t.to_vec();
    assert_eq!(data[0] as i64, 1);
    assert_eq!(data[1] as i64, 2);
}

#[test]
fn force_merge_then_inspect_via_action_count() {
    reset_arena();
    let g = new_graph(0);
    // Prepare two children to merge.
    let a = expect_int(call(
        "abng_add_node",
        &[Value::Int(g), Value::Int(0), Value::Int(1)],
    ));
    let b = expect_int(call(
        "abng_add_node",
        &[Value::Int(g), Value::Int(0), Value::Int(2)],
    ));
    let _ = call(
        "abng_force_merge",
        &[Value::Int(g), Value::Int(a), Value::Int(b)],
    );
    let n = expect_int(call(
        "abng_action_count",
        &[Value::Int(g), Value::Int(2)], // Merge
    ));
    assert_eq!(n, 1);
}

#[test]
fn force_merge_self_dispatch_errs() {
    reset_arena();
    let g = new_graph(0);
    let err = call_err(
        "abng_force_merge",
        &[Value::Int(g), Value::Int(0), Value::Int(0)],
    );
    assert!(err.contains("absorb itself"), "err = {err}");
}

#[test]
fn force_prune_via_dispatch() {
    reset_arena();
    let g = new_graph(0);
    let c = expect_int(call(
        "abng_add_node",
        &[Value::Int(g), Value::Int(0), Value::Int(7)],
    ));
    let _ = call(
        "abng_force_prune",
        &[Value::Int(g), Value::Int(c)],
    );
    let n = expect_int(call(
        "abng_action_count",
        &[Value::Int(g), Value::Int(3)], // Prune
    ));
    assert_eq!(n, 1);
}

#[test]
fn force_compress_then_freeze_via_dispatch() {
    reset_arena();
    let g = new_graph(0);
    let _ = call(
        "abng_force_freeze",
        &[Value::Int(g), Value::Int(0)],
    );
    let frozen = expect_bool(call(
        "abng_is_frozen",
        &[Value::Int(g), Value::Int(0)],
    ));
    assert!(frozen);
    // Compressing a frozen node errors.
    let err = call_err(
        "abng_force_compress",
        &[Value::Int(g), Value::Int(0)],
    );
    assert!(err.contains("frozen"), "err = {err}");
}

#[test]
fn is_frozen_default_false_via_dispatch() {
    reset_arena();
    let g = new_graph(0);
    let frozen = expect_bool(call(
        "abng_is_frozen",
        &[Value::Int(g), Value::Int(0)],
    ));
    assert!(!frozen);
}

#[test]
fn action_count_unknown_kind_errs() {
    reset_arena();
    let g = new_graph(0);
    let err = call_err(
        "abng_action_count",
        &[Value::Int(g), Value::Int(99)],
    );
    assert!(err.contains("out of range"), "err = {err}");
}

#[test]
fn action_count_arity_check() {
    reset_arena();
    let g = new_graph(0);
    let err = call_err("abng_action_count", &[Value::Int(g)]);
    assert!(err.contains("expected 2 arguments"), "err = {err}");
}

#[test]
fn force_split_blocked_on_non_leaf_via_dispatch() {
    reset_arena();
    let g = new_graph(0);
    let _ = call(
        "abng_add_node",
        &[Value::Int(g), Value::Int(0), Value::Int(7)],
    );
    let err = call_err(
        "abng_force_split",
        &[Value::Int(g), Value::Int(0)],
    );
    assert!(err.contains("non-frozen leaf") || err.contains("ForceSplit"), "err = {err}");
}

#[test]
fn double_run_p3d3_dispatch_byte_identical() {
    let mk = || {
        reset_arena();
        let g = new_graph(42);
        let _ = call(
            "abng_set_decision_policy",
            &[Value::Int(g), Value::Tensor(ok_thresholds_tensor())],
        );
        let _ = call(
            "abng_force_grow",
            &[Value::Int(g), Value::Int(0), Value::Int(11)],
        );
        let _ = call("abng_force_freeze", &[Value::Int(g), Value::Int(0)]);
        let h = expect_string(call("abng_chain_head", &[Value::Int(g)]));
        let n_grow = expect_int(call(
            "abng_action_count",
            &[Value::Int(g), Value::Int(0)],
        ));
        let n_freeze = expect_int(call(
            "abng_action_count",
            &[Value::Int(g), Value::Int(5)],
        ));
        ((*h).clone(), n_grow, n_freeze)
    };
    assert_eq!(mk(), mk());
}

// ─── Phase 0.3d-4 — decide_step + unfreeze ───────────────────────

#[test]
fn decide_step_returns_six_counts_tensor() {
    reset_arena();
    let g = new_graph(0);
    let _ = call(
        "abng_set_decision_policy",
        &[Value::Int(g), Value::Tensor(ok_thresholds_tensor())],
    );
    let v = call("abng_decide_step", &[Value::Int(g)]);
    let t = match v {
        Value::Tensor(t) => t,
        _ => panic!("expected Tensor"),
    };
    assert_eq!(t.shape(), &[6]);
    let data = t.to_vec();
    // Idle graph, first call → no actions yet.
    for c in data {
        assert_eq!(c, 0.0);
    }
}

#[test]
fn decide_step_arity_check() {
    reset_arena();
    let g = new_graph(0);
    let err = call_err("abng_decide_step", &[Value::Int(g), Value::Int(0)]);
    assert!(err.contains("expected 1 arguments"), "err = {err}");
}

#[test]
fn decide_step_no_policy_returns_zeros() {
    reset_arena();
    let g = new_graph(0);
    // No policy installed → no-op.
    let v = call("abng_decide_step", &[Value::Int(g)]);
    let data = match v {
        Value::Tensor(t) => t.to_vec(),
        _ => panic!(),
    };
    for c in data {
        assert_eq!(c, 0.0);
    }
}

#[test]
fn unfreeze_then_is_frozen_false() {
    reset_arena();
    let g = new_graph(0);
    let _ = call("abng_force_freeze", &[Value::Int(g), Value::Int(0)]);
    assert!(expect_bool(call(
        "abng_is_frozen",
        &[Value::Int(g), Value::Int(0)]
    )));
    let _ = call("abng_unfreeze", &[Value::Int(g), Value::Int(0)]);
    assert!(!expect_bool(call(
        "abng_is_frozen",
        &[Value::Int(g), Value::Int(0)]
    )));
}

#[test]
fn unfreeze_arity_check() {
    reset_arena();
    let g = new_graph(0);
    let err = call_err("abng_unfreeze", &[Value::Int(g)]);
    assert!(err.contains("expected 2 arguments"), "err = {err}");
}

#[test]
fn unfreeze_bad_node_id_errs() {
    reset_arena();
    let g = new_graph(0);
    let err = call_err("abng_unfreeze", &[Value::Int(g), Value::Int(99)]);
    assert!(err.contains("out of range"), "err = {err}");
}

#[test]
fn decide_step_grow_via_dispatch() {
    reset_arena();
    let g = new_graph(0);
    let _ = call(
        "abng_set_decision_policy",
        &[Value::Int(g), Value::Tensor(ok_thresholds_tensor())],
    );
    for _ in 0..70 {
        let _ = call(
            "abng_observe",
            &[Value::Int(g), Value::Int(0), Value::Float(1.0)],
        );
    }
    let v = call("abng_decide_step", &[Value::Int(g)]);
    let counts = match v {
        Value::Tensor(t) => t.to_vec(),
        _ => panic!(),
    };
    // Grow fires (index 0).
    assert_eq!(counts[0], 1.0);
}
