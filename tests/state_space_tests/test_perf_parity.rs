//! AST↔MIR parity tests for the Phase-2 performance primitives.
//!
//! Each snippet runs through `cjc-eval` and `cjc-mir-exec`; printed output
//! must be byte-identical.

use crate::harness::assert_parity;

#[test]
fn parity_tensor_concat_1d_basic() {
    assert_parity(
        "tensor_concat_1d basic",
        r#"
        let r = state_space_clear();
        let a = Tensor.from_vec([1.0, 2.0, 3.0], [3]);
        let b = Tensor.from_vec([4.0, 5.0], [2]);
        let c = tensor_concat_1d(a, b);
        print(c.get([0]));
        print(c.get([1]));
        print(c.get([2]));
        print(c.get([3]));
        print(c.get([4]));
        "#,
    );
}

#[test]
fn parity_step_with_readout_matches_split_path() {
    assert_parity(
        "fused step+readout matches separate calls",
        r#"
        let r = state_space_clear();
        // Two independent cells with the same seed → fused step on A,
        // separate step+readout on B → outputs and final hidden states
        // must agree byte-for-byte.
        let ha = state_space_init(2, 3, 2, 71);
        let hb = state_space_init(2, 3, 2, 71);
        let x = Tensor.from_vec([0.5, 0.0 - 0.5], [2]);
        let pair = state_space_step_with_readout(ha, x);
        let y_fused = pair[0];
        let h_fused = pair[1];

        let _ystep = state_space_step(hb, x);
        let h_split = state_space_state(hb);
        let y_split = state_space_readout(hb);

        // y agreement
        print(y_fused.get([0]) - y_split.get([0]));
        print(y_fused.get([1]) - y_split.get([1]));
        // h agreement
        print(h_fused.get([0]) - h_split.get([0]));
        print(h_fused.get([1]) - h_split.get([1]));
        print(h_fused.get([2]) - h_split.get([2]));
        "#,
    );
}

#[test]
fn parity_step_batched_independent_rows() {
    assert_parity(
        "batched step zeroes h between rows",
        r#"
        let r = state_space_clear();
        let h = state_space_init(2, 3, 2, 13);
        // Two identical rows — batched output must be identical too.
        let xs = Tensor.from_vec([1.0, 0.0, 1.0, 0.0], [2, 2]);
        let ys = state_space_step_batched(h, xs);
        print(ys.get([0, 0]) - ys.get([1, 0]));
        print(ys.get([0, 1]) - ys.get([1, 1]));
        "#,
    );
}

#[test]
fn parity_get_weights_matches_recurrence() {
    assert_parity(
        "extracted A/B/b_o reconstruct step",
        r#"
        let r = state_space_clear();
        let h = state_space_init(1, 2, 1, 7);
        let a = state_space_get_A(h);   // [2,2]
        let b = state_space_get_B(h);   // [2,1]
        // The hidden state is initially zero. After one step with x=[v]:
        //   h_new[i] = tanh(B[i,0] * v)
        // Reconstruction via extracted B must equal cell's step.
        let v = 0.4;
        let x = Tensor.from_vec([v], [1]);
        let _ystep = state_space_step(h, x);
        let h_after = state_space_state(h);
        // Compute the analytic prediction:
        //   pre0 = B[0,0] * v;   h_new[0] = tanh(pre0)
        //   pre1 = B[1,0] * v;   h_new[1] = tanh(pre1)
        let pre0 = b.get([0, 0]) * v;
        let pre1 = b.get([1, 0]) * v;
        let pred0 = tanh(pre0);
        let pred1 = tanh(pre1);
        print(h_after.get([0]) - pred0);
        print(h_after.get([1]) - pred1);
        "#,
    );
}
