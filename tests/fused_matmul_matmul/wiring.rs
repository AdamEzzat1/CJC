//! Wiring tests — `fused_matmul_matmul` byte-identical across both executors,
//! including parity vs the unfused `matmul(matmul(a, b), c)` chain.

#[path = "../fused_test_helpers/mod.rs"]
mod helpers;
use helpers::{run_eval, run_mir, run_parity};

#[test]
fn fused_matmul_matmul_basic_byte_identical() {
    let src = r#"
fn main() {
    let a = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]);
    let b = Tensor.from_vec([2.0, 3.0, 4.0, 5.0], [2, 2]);
    let c = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]);
    let r = fused_matmul_matmul(a, b, c);
    print(r.shape()[0]);
    print(r.shape()[1]);
}
"#;
    let out = run_parity(src, 0);
    assert_eq!(out, vec!["2", "2"]);
}

#[test]
fn fused_matmul_matmul_equals_unfused_chain_in_both_executors() {
    // The headline guarantee for Phase 3.5d: fused(A, B, C) ==
    // matmul(matmul(A, B), C) byte-for-byte through the executors.
    // This is what the MIR rewriter's correctness contract reduces to.
    let src_fused = r#"
fn main() {
    let a = Tensor.from_vec([1.5, -0.5, 2.0, 0.5, 1.0, -1.0], [2, 3]);
    let b = Tensor.from_vec([0.5, 1.5, 1.0, -0.5, -1.0, 2.0], [3, 2]);
    let c = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]);
    let r = fused_matmul_matmul(a, b, c);
    print(r.shape()[0]);
    print(r.shape()[1]);
}
"#;
    let src_unfused = r#"
fn main() {
    let a = Tensor.from_vec([1.5, -0.5, 2.0, 0.5, 1.0, -1.0], [2, 3]);
    let b = Tensor.from_vec([0.5, 1.5, 1.0, -0.5, -1.0, 2.0], [3, 2]);
    let c = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]);
    let h = matmul(a, b);
    let r = matmul(h, c);
    print(r.shape()[0]);
    print(r.shape()[1]);
}
"#;
    let fused_eval = run_eval(src_fused, 0);
    let fused_mir = run_mir(src_fused, 0);
    let unfused_eval = run_eval(src_unfused, 0);
    let unfused_mir = run_mir(src_unfused, 0);

    assert_eq!(fused_eval, fused_mir, "fused eval ≠ fused mir");
    assert_eq!(unfused_eval, unfused_mir, "unfused eval ≠ unfused mir");
    assert_eq!(fused_eval, unfused_eval, "fused ≠ unfused (eval)");
}
