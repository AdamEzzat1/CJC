//! AST↔MIR parity tests for `state_space_*` builtins.
//!
//! Each test runs a small `.cjcl` snippet through `cjc-eval` (AST tree-walk)
//! and `cjc-mir-exec` (register-machine MIR), asserting byte-identical
//! printed output. Because the SSM dispatch lives in `cjc-runtime` and is
//! reached via the same `dispatch_builtin` entry that both backends call,
//! parity is structural — but we test it explicitly to catch regressions
//! in the routing.

use crate::harness::assert_parity;

/// Every parity test starts by clearing the thread-local arena. cargo's test
/// pool reuses threads, so a stale cell from a prior test could otherwise
/// alias the new handle and silently corrupt output.

#[test]
fn parity_init_returns_handle() {
    assert_parity(
        "init returns handle",
        r#"
        let r = state_space_clear();
        let h = state_space_init(2, 4, 2, 7);
        print(h);
        print(state_space_len());
        "#,
    );
}

#[test]
fn parity_step_then_state_is_deterministic() {
    assert_parity(
        "step deterministic",
        r#"
        let r = state_space_clear();
        let h = state_space_init(2, 3, 2, 11);
        let x = Tensor.from_vec([1.0, 0.5], [2]);
        let y = state_space_step(h, x);
        let s = state_space_state(h);
        print(y.get([0]));
        print(y.get([1]));
        print(s.get([0]));
        print(s.get([1]));
        print(s.get([2]));
        "#,
    );
}

#[test]
fn parity_scan_matches_repeated_step() {
    // Use one cell for scan, another for stepwise; with the same seed +
    // same input, output sequences must agree byte-for-byte across both
    // backends.
    assert_parity(
        "scan vs step",
        r#"
        let r = state_space_clear();
        let ha = state_space_init(2, 3, 2, 33);
        let hb = state_space_init(2, 3, 2, 33);
        let xs = Tensor.from_vec([1.0, 0.0, 0.5, 0.5, 0.0, 1.0], [3, 2]);
        let ys = state_space_scan(ha, xs);
        let x0 = Tensor.from_vec([1.0, 0.0], [2]);
        let x1 = Tensor.from_vec([0.5, 0.5], [2]);
        let x2 = Tensor.from_vec([0.0, 1.0], [2]);
        let y0 = state_space_step(hb, x0);
        let y1 = state_space_step(hb, x1);
        let y2 = state_space_step(hb, x2);
        print(ys.get([0, 0]) - y0.get([0]));
        print(ys.get([1, 1]) - y1.get([1]));
        print(ys.get([2, 0]) - y2.get([0]));
        "#,
    );
}

#[test]
fn parity_reset_zeros_state() {
    assert_parity(
        "reset zeros",
        r#"
        let r = state_space_clear();
        let h = state_space_init(2, 4, 2, 5);
        let x = Tensor.from_vec([1.0, -0.5], [2]);
        let y = state_space_step(h, x);
        let r2 = state_space_reset(h);
        let s = state_space_state(h);
        print(s.get([0]));
        print(s.get([1]));
        print(s.get([2]));
        print(s.get([3]));
        "#,
    );
}

#[test]
fn parity_snapshot_restore_round_trip() {
    assert_parity(
        "snapshot/restore",
        r#"
        let r = state_space_clear();
        let h = state_space_init(2, 4, 2, 9);
        let x = Tensor.from_vec([1.0, 0.0], [2]);
        let y = Tensor.from_vec([0.0, 1.0], [2]);
        let y0 = state_space_step(h, x);
        let snap = state_space_snapshot(h);
        let y1 = state_space_step(h, y);
        let r2 = state_space_restore(h, snap);
        let y2 = state_space_step(h, y);
        // y1 and y2 must agree element-wise
        print(y1.get([0]) - y2.get([0]));
        print(y1.get([1]) - y2.get([1]));
        "#,
    );
}

#[test]
fn parity_dim_introspection() {
    assert_parity(
        "dim introspection",
        r#"
        let r = state_space_clear();
        let h = state_space_init(5, 12, 3, 1);
        print(state_space_input_dim(h));
        print(state_space_hidden_dim(h));
        print(state_space_output_dim(h));
        "#,
    );
}
