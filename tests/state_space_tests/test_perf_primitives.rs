//! Unit tests for the Phase-2 performance primitives:
//!   - `tensor_concat_1d`
//!   - `state_space_step_with_readout`
//!   - `state_space_step_batched`
//!   - `state_space_get_A` / `_get_B` / `_get_C` / `_get_b_o`

use crate::harness::*;
use cjc_runtime::state_space::{dispatch_state_space, tensor_concat_1d};
use cjc_runtime::tensor::Tensor;
use cjc_runtime::value::Value;

// ─── tensor_concat_1d ───────────────────────────────────────────────────────

#[test]
fn concat_basic_two_tensors() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let b = Tensor::from_vec(vec![4.0, 5.0], &[2]).unwrap();
    let c = tensor_concat_1d(&a, &b).unwrap();
    assert_eq!(c.shape(), &[5]);
    assert_eq!(c.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn concat_empty_left_returns_right() {
    let a = Tensor::from_vec(vec![], &[0]).unwrap();
    let b = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let c = tensor_concat_1d(&a, &b).unwrap();
    assert_eq!(c.to_vec(), vec![1.0, 2.0]);
}

#[test]
fn concat_2d_input_errors() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let err = tensor_concat_1d(&a, &b).unwrap_err();
    assert!(err.contains("must be 1-D"), "unexpected error: {err}");
}

#[test]
fn concat_via_dispatch() {
    let a = Tensor::from_vec(vec![10.0, 20.0], &[2]).unwrap();
    let b = Tensor::from_vec(vec![30.0], &[1]).unwrap();
    let v = dispatch_state_space(
        "tensor_concat_1d",
        &[Value::Tensor(a), Value::Tensor(b)],
    )
    .unwrap()
    .unwrap();
    let t = match v {
        Value::Tensor(t) => t,
        _ => panic!("expected Tensor"),
    };
    assert_eq!(t.to_vec(), vec![10.0, 20.0, 30.0]);
}

// ─── state_space_step_with_readout ─────────────────────────────────────────

#[test]
fn fused_step_matches_separate_step_then_readout() {
    clear();
    // Two cells with same params → fused step on one, separate step+readout
    // on the other → outputs and final hidden states must agree byte-for-byte.
    let h_fused = ssm_init(2, 4, 3, 71);
    let h_split = ssm_init(2, 4, 3, 71);
    let x = Tensor::from_vec(vec![0.5, -0.5], &[2]).unwrap();

    let v = dispatch_state_space(
        "state_space_step_with_readout",
        &[Value::Int(h_fused), Value::Tensor(x.clone())],
    )
    .unwrap()
    .unwrap();
    let arr = match v {
        Value::Array(a) => a,
        _ => panic!("expected Array"),
    };
    let y_fused = match &arr[0] {
        Value::Tensor(t) => t.clone(),
        _ => panic!("first element must be Tensor"),
    };
    let h_after_fused = match &arr[1] {
        Value::Tensor(t) => t.clone(),
        _ => panic!("second element must be Tensor"),
    };

    // Separate path
    let _ = ssm_step(h_split, x);
    let h_after_split = ssm_state(h_split);
    let y_split = ssm_readout(h_split);

    assert_eq!(y_fused.to_vec(), y_split.to_vec());
    assert_eq!(h_after_fused.to_vec(), h_after_split.to_vec());
}

#[test]
fn fused_step_returns_two_element_array() {
    clear();
    let h = ssm_init(2, 3, 4, 1);
    let x = Tensor::from_vec(vec![1.0, 0.0], &[2]).unwrap();
    let v = dispatch_state_space(
        "state_space_step_with_readout",
        &[Value::Int(h), Value::Tensor(x)],
    )
    .unwrap()
    .unwrap();
    let arr = match v {
        Value::Array(a) => a,
        _ => panic!("expected Array"),
    };
    assert_eq!(arr.len(), 2);
    let y_shape = match &arr[0] {
        Value::Tensor(t) => t.shape().to_vec(),
        _ => panic!(),
    };
    let h_shape = match &arr[1] {
        Value::Tensor(t) => t.shape().to_vec(),
        _ => panic!(),
    };
    assert_eq!(y_shape, vec![4]); // output_dim
    assert_eq!(h_shape, vec![3]); // hidden_dim
}

#[test]
fn fused_step_shape_mismatch_errors() {
    clear();
    let h = ssm_init(3, 4, 2, 1);
    let bad = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let err = dispatch_state_space(
        "state_space_step_with_readout",
        &[Value::Int(h), Value::Tensor(bad)],
    )
    .unwrap_err();
    assert!(err.contains("expected x of shape"), "unexpected: {err}");
}

// ─── state_space_step_batched ──────────────────────────────────────────────

#[test]
fn batched_step_independent_rows() {
    clear();
    // The batched step zeroes hidden state before each row, so the two outputs
    // for IDENTICAL rows must agree byte-for-byte.
    let h = ssm_init(2, 3, 2, 13);
    let xs = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], &[2, 2]).unwrap();
    let ys = unwrap_tensor(
        dispatch_state_space(
            "state_space_step_batched",
            &[Value::Int(h), Value::Tensor(xs)],
        )
        .unwrap(),
    );
    assert_eq!(ys.shape(), &[2, 2]);
    let row0 = ys.to_vec()[0..2].to_vec();
    let row1 = ys.to_vec()[2..4].to_vec();
    assert_eq!(row0, row1, "identical inputs must produce identical batched outputs");
}

#[test]
fn batched_step_distinct_rows_yield_distinct_outputs() {
    clear();
    let h = ssm_init(2, 3, 2, 17);
    let xs = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
    let ys = unwrap_tensor(
        dispatch_state_space(
            "state_space_step_batched",
            &[Value::Int(h), Value::Tensor(xs)],
        )
        .unwrap(),
    );
    let row0 = ys.to_vec()[0..2].to_vec();
    let row1 = ys.to_vec()[2..4].to_vec();
    assert_ne!(row0, row1);
}

#[test]
fn batched_step_shape_mismatch_errors() {
    clear();
    let h = ssm_init(3, 4, 2, 1);
    let bad = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap(); // wrong inner dim
    let err = dispatch_state_space(
        "state_space_step_batched",
        &[Value::Int(h), Value::Tensor(bad)],
    )
    .unwrap_err();
    assert!(err.contains("shape"), "unexpected: {err}");
}

// ─── Weight extractors ─────────────────────────────────────────────────────

#[test]
fn get_a_returns_hidden_by_hidden_matrix() {
    clear();
    let h = ssm_init(3, 5, 2, 19);
    let v = dispatch_state_space("state_space_get_A", &[Value::Int(h)])
        .unwrap()
        .unwrap();
    let t = match v {
        Value::Tensor(t) => t,
        _ => panic!(),
    };
    assert_eq!(t.shape(), &[5, 5]);
}

#[test]
fn get_b_returns_hidden_by_input_matrix() {
    clear();
    let h = ssm_init(3, 5, 2, 19);
    let v = dispatch_state_space("state_space_get_B", &[Value::Int(h)])
        .unwrap()
        .unwrap();
    let t = match v {
        Value::Tensor(t) => t,
        _ => panic!(),
    };
    assert_eq!(t.shape(), &[5, 3]);
}

#[test]
fn get_c_returns_output_by_hidden_matrix() {
    clear();
    let h = ssm_init(3, 5, 2, 19);
    let v = dispatch_state_space("state_space_get_C", &[Value::Int(h)])
        .unwrap()
        .unwrap();
    let t = match v {
        Value::Tensor(t) => t,
        _ => panic!(),
    };
    assert_eq!(t.shape(), &[2, 5]);
}

#[test]
fn get_b_o_returns_output_dim_vector() {
    clear();
    let h = ssm_init(3, 5, 2, 19);
    let v = dispatch_state_space("state_space_get_b_o", &[Value::Int(h)])
        .unwrap()
        .unwrap();
    let t = match v {
        Value::Tensor(t) => t,
        _ => panic!(),
    };
    assert_eq!(t.shape(), &[2]);
    assert!(t.to_vec().iter().all(|&v| v == 0.0)); // bias init is zero
}

#[test]
fn extracted_weights_match_independent_init() {
    // Two cells with same seed must have byte-identical extracted weights.
    clear();
    let ha = ssm_init(2, 3, 2, 99);
    let hb = ssm_init(2, 3, 2, 99);
    for name in ["state_space_get_A", "state_space_get_B", "state_space_get_C", "state_space_get_b_o"] {
        let ta = unwrap_tensor(
            dispatch_state_space(name, &[Value::Int(ha)]).unwrap(),
        );
        let tb = unwrap_tensor(
            dispatch_state_space(name, &[Value::Int(hb)]).unwrap(),
        );
        assert_eq!(ta.to_vec(), tb.to_vec(), "{name} mismatch");
    }
}

#[test]
fn extracted_weights_dead_handle_errors() {
    clear();
    let h = ssm_init(2, 2, 2, 1);
    clear();
    let err = dispatch_state_space("state_space_get_A", &[Value::Int(h)]).unwrap_err();
    // `arg_handle` is checked first and surfaces "does not refer to a live cell";
    // the inner op's "no longer alive" message is unreachable. Both forms
    // are acceptable — the contract is that a dead handle errors cleanly.
    assert!(
        err.contains("does not refer to a live cell") || err.contains("no longer alive"),
        "unexpected: {err}"
    );
}

// ─── Reconstructing the recurrence from extracted weights (autodiff path) ──

#[test]
fn reconstruct_step_from_extracted_weights() {
    // The "differentiable SSM via GradGraph composition" pattern: extract
    // A, B from the cell, manually compute h_new = tanh(A·h + B·x), and
    // assert it matches the cell's own step. This is the contract the
    // get_A/B/C/b_o extractors are supposed to satisfy — they should be
    // the same matrices the cell uses internally.
    clear();
    let h = ssm_init(2, 4, 3, 31);
    let a_t = unwrap_tensor(dispatch_state_space("state_space_get_A", &[Value::Int(h)]).unwrap());
    let b_t = unwrap_tensor(dispatch_state_space("state_space_get_B", &[Value::Int(h)]).unwrap());
    let h0 = ssm_state(h);

    let x = Tensor::from_vec(vec![0.4, -0.6], &[2]).unwrap();

    // Reference: ask the cell to step.
    let _ = ssm_step(h, x.clone());
    let h1_cell = ssm_state(h).to_vec();

    // Reconstruction: compute by hand using the extracted matrices.
    let a = a_t.to_vec();
    let b = b_t.to_vec();
    let h0_d = h0.to_vec();
    let x_d = x.to_vec();
    let mut h1_recon = vec![0.0; 4];
    for i in 0..4 {
        let mut acc = 0.0;
        for j in 0..4 {
            acc += a[i * 4 + j] * h0_d[j];
        }
        for j in 0..2 {
            acc += b[i * 2 + j] * x_d[j];
        }
        h1_recon[i] = acc.tanh();
    }
    assert_eq!(h1_cell, h1_recon, "reconstructed step must match cell step");
}
