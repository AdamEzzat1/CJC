//! Direct-dispatch unit tests — bypass the parser and call
//! `cjc_runtime::state_space::dispatch_state_space` directly.
//!
//! These exercise every primitive against handwritten reference values to
//! catch numerical or shape regressions independent of the language pipeline.

use crate::harness::*;
use cjc_runtime::tensor::Tensor;

#[test]
fn init_produces_zero_hidden_state() {
    clear();
    let h = ssm_init(4, 6, 3, 42);
    let st = ssm_state(h);
    assert_eq!(st.shape(), &[6]);
    assert!(st.to_vec().iter().all(|&v| v == 0.0));
}

#[test]
fn init_with_same_seed_produces_same_weights() {
    // Two cells with same params → step on same input → identical output.
    clear();
    let a = ssm_init(3, 5, 2, 123);
    let b = ssm_init(3, 5, 2, 123);
    let x = Tensor::from_vec(vec![1.0, 0.5, -0.25], &[3]).unwrap();
    let y_a = ssm_step(a, x.clone());
    let y_b = ssm_step(b, x);
    assert_eq!(y_a.to_vec(), y_b.to_vec());
}

#[test]
fn init_with_different_seed_produces_different_weights() {
    clear();
    let a = ssm_init(3, 5, 2, 1);
    let b = ssm_init(3, 5, 2, 2);
    let x = Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3]).unwrap();
    let y_a = ssm_step(a, x.clone());
    let y_b = ssm_step(b, x);
    assert_ne!(
        y_a.to_vec(),
        y_b.to_vec(),
        "different seeds must yield different SSM trajectories"
    );
}

#[test]
fn step_advances_hidden_state() {
    clear();
    let h = ssm_init(2, 4, 2, 7);
    let s0 = ssm_state(h).to_vec();
    let x = Tensor::from_vec(vec![1.0, -0.5], &[2]).unwrap();
    let _ = ssm_step(h, x);
    let s1 = ssm_state(h).to_vec();
    assert_ne!(s0, s1, "step must change hidden state");
}

#[test]
fn reset_zeros_hidden_state() {
    clear();
    let h = ssm_init(2, 3, 1, 11);
    let x = Tensor::from_vec(vec![1.0, 1.0], &[2]).unwrap();
    let _ = ssm_step(h, x.clone());
    let _ = ssm_step(h, x.clone());
    ssm_reset(h);
    let st = ssm_state(h);
    assert!(st.to_vec().iter().all(|&v| v == 0.0));
}

#[test]
fn snapshot_restore_round_trip_preserves_future() {
    clear();
    let h = ssm_init(2, 4, 2, 99);
    let x = Tensor::from_vec(vec![0.7, -0.3], &[2]).unwrap();
    let y = Tensor::from_vec(vec![-0.2, 0.5], &[2]).unwrap();
    // Walk forward
    let _ = ssm_step(h, x.clone());
    let _ = ssm_step(h, y.clone());
    let snap = ssm_snapshot(h);
    let future_a = ssm_step(h, x.clone());
    // Disturb and restore
    let _ = ssm_step(h, y.clone());
    let _ = ssm_step(h, x.clone());
    ssm_restore(h, snap);
    let future_b = ssm_step(h, x);
    assert_eq!(future_a.to_vec(), future_b.to_vec());
}

#[test]
fn scan_equals_repeated_step() {
    clear();
    let ha = ssm_init(2, 4, 2, 31);
    let hb = ssm_init(2, 4, 2, 31);
    let xs_data = vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0, -0.3, 0.2];
    let xs = Tensor::from_vec(xs_data.clone(), &[4, 2]).unwrap();
    let ys = ssm_scan(ha, xs);
    let mut acc = Vec::new();
    for step in 0..4 {
        let row = xs_data[step * 2..(step + 1) * 2].to_vec();
        let x = Tensor::from_vec(row, &[2]).unwrap();
        let y = ssm_step(hb, x);
        acc.extend(y.to_vec());
    }
    assert_eq!(ys.to_vec(), acc);
    assert_eq!(ys.shape(), &[4, 2]);
}

#[test]
fn shape_mismatch_in_step_errors() {
    clear();
    let h = ssm_init(3, 4, 2, 1);
    let bad = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let err = cjc_runtime::state_space::dispatch_state_space(
        "state_space_step",
        &[
            cjc_runtime::value::Value::Int(h),
            cjc_runtime::value::Value::Tensor(bad),
        ],
    )
    .unwrap_err();
    assert!(err.contains("shape"), "unexpected error: {err}");
}

#[test]
fn shape_mismatch_in_scan_errors() {
    clear();
    let h = ssm_init(3, 4, 2, 1);
    // Wrong inner dim
    let bad = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let err = cjc_runtime::state_space::dispatch_state_space(
        "state_space_scan",
        &[
            cjc_runtime::value::Value::Int(h),
            cjc_runtime::value::Value::Tensor(bad),
        ],
    )
    .unwrap_err();
    assert!(err.contains("shape"), "unexpected error: {err}");
}

#[test]
fn hidden_state_length_matches_dim() {
    clear();
    for hd in [1, 2, 8, 32] {
        let h = ssm_init(2, hd, 2, 1);
        let st = ssm_state(h);
        assert_eq!(st.shape(), &[hd as usize]);
    }
}

#[test]
fn dead_handle_after_clear_errors() {
    clear();
    let h = ssm_init(2, 2, 1, 0);
    clear();
    let err = cjc_runtime::state_space::dispatch_state_space(
        "state_space_state",
        &[cjc_runtime::value::Value::Int(h)],
    )
    .unwrap_err();
    assert!(
        err.contains("does not refer to a live cell"),
        "unexpected error: {err}"
    );
}

#[test]
fn readout_without_step_returns_bias_only() {
    // After init, hidden state is zero, so readout = C·0 + b_o = b_o = zeros.
    clear();
    let h = ssm_init(2, 4, 3, 5);
    let y = ssm_readout(h);
    assert_eq!(y.shape(), &[3]);
    assert!(y.to_vec().iter().all(|&v| v == 0.0));
}

#[test]
fn set_state_then_readout_reflects_new_state() {
    clear();
    let h = ssm_init(2, 3, 2, 7);
    let custom = Tensor::from_vec(vec![1.0, -1.0, 0.5], &[3]).unwrap();
    ssm_set_state(h, custom.clone());
    assert_eq!(ssm_state(h).to_vec(), custom.to_vec());
}
