//! Wiring tests — `fused_matmul_norm` byte-identical across both executors.
//!
//! The matmul→norm chain is the first fused pair that ALSO works as an
//! explicit user-written sequence in CJC-Lang. These tests verify
//! `fused_matmul_norm(A, W, ord)` matches `norm(matmul(A, W), ord)` in
//! both executors at the language level (not just the kernel level).

#[path = "../fused_test_helpers/mod.rs"]
mod helpers;
use helpers::{run_eval, run_mir, run_parity};

#[test]
fn fused_matmul_norm_l2_default_byte_identical() {
    let src = r#"
fn main() {
    let a = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]);
    let w = Tensor.from_vec([3.0, 0.0, 0.0, 4.0], [2, 2]);
    let n = fused_matmul_norm(a, w);
    print(n);
}
"#;
    let out = run_parity(src, 0);
    assert_eq!(out, vec!["5"]);
}

#[test]
fn fused_matmul_norm_l1_explicit_byte_identical() {
    let src = r#"
fn main() {
    let a = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]);
    let w = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]);
    let n = fused_matmul_norm(a, w, 1);
    print(n);
}
"#;
    let out = run_parity(src, 0);
    assert_eq!(out, vec!["10"]);
}

#[test]
fn fused_matmul_norm_equals_unfused_chain_in_both_executors() {
    // The headline language-level guarantee: fused_matmul_norm(A, W) returns
    // the same string as norm(matmul(A, W)) when both run through the
    // executors. This is the rewriter's correctness contract.
    let src_fused = r#"
fn main() {
    let a = Tensor.from_vec([1.5, -0.5, 2.0, 0.5, 1.0, -1.0], [3, 2]);
    let w = Tensor.from_vec([0.5, 1.5, 1.0, -0.5, -1.0, 2.0], [2, 3]);
    let n = fused_matmul_norm(a, w);
    print(n);
}
"#;
    let src_unfused = r#"
fn main() {
    let a = Tensor.from_vec([1.5, -0.5, 2.0, 0.5, 1.0, -1.0], [3, 2]);
    let w = Tensor.from_vec([0.5, 1.5, 1.0, -0.5, -1.0, 2.0], [2, 3]);
    let h = matmul(a, w);
    let n = norm(h);
    print(n);
}
"#;
    let fused_eval = run_eval(src_fused, 0);
    let fused_mir = run_mir(src_fused, 0);
    let unfused_eval = run_eval(src_unfused, 0);
    let unfused_mir = run_mir(src_unfused, 0);

    // Eval vs MIR parity for each form.
    assert_eq!(fused_eval, fused_mir, "fused eval ≠ fused mir");
    assert_eq!(unfused_eval, unfused_mir, "unfused eval ≠ unfused mir");
    // Fused == unfused at the user level.
    assert_eq!(fused_eval, unfused_eval, "fused ≠ unfused (eval)");
}

#[test]
fn fused_matmul_norm_1x1_byte_identical() {
    let src = r#"
fn main() {
    let a = Tensor.from_vec([3.0], [1, 1]);
    let w = Tensor.from_vec([4.0], [1, 1]);
    let n = fused_matmul_norm(a, w);
    print(n);
}
"#;
    let out = run_parity(src, 7);
    assert_eq!(out, vec!["12"]);
}
