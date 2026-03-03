//! Test Suite 10: Eval vs MIR-exec parity verification

use super::helpers::*;

fn assert_parity(src: &str) {
    let mir_out = run_mir(src);
    let eval_out = run_eval(src);
    assert_eq!(
        mir_out, eval_out,
        "Parity mismatch:\n  MIR: {mir_out:?}\n  Eval: {eval_out:?}\n  Source: {src}"
    );
}

#[test]
fn parity_sin() {
    assert_parity("print(sin(1.0));");
}

#[test]
fn parity_cos() {
    assert_parity("print(cos(1.0));");
}

#[test]
fn parity_tan() {
    assert_parity("print(tan(1.0));");
}

#[test]
fn parity_pow() {
    assert_parity("print(pow(2.0, 10.0));");
}

#[test]
fn parity_log2() {
    assert_parity("print(log2(16.0));");
}

#[test]
fn parity_log10() {
    assert_parity("print(log10(100.0));");
}

#[test]
fn parity_ceil() {
    assert_parity("print(ceil(2.3));");
}

#[test]
fn parity_round() {
    assert_parity("print(round(2.7));");
}

#[test]
fn parity_min_max() {
    assert_parity("print(min(3.0, 5.0));");
    assert_parity("print(max(3.0, 5.0));");
}

#[test]
fn parity_sign() {
    assert_parity("print(sign(0.0 - 42.0));");
}

#[test]
fn parity_hypot() {
    assert_parity("print(hypot(3.0, 4.0));");
}

#[test]
fn parity_constants() {
    assert_parity("print(PI());");
    assert_parity("print(E());");
    assert_parity("print(TAU());");
}

#[test]
fn parity_tensor_eye() {
    assert_parity(r#"
let I = Tensor.eye(3);
print(I.sum());
"#);
}

#[test]
fn parity_tensor_linspace() {
    assert_parity(r#"
let t = Tensor.linspace(0.0, 1.0, 5);
print(t.sum());
"#);
}

#[test]
fn parity_dot() {
    assert_parity(r#"
let a = Tensor.from_vec([1.0, 2.0, 3.0], [3]);
let b = Tensor.from_vec([4.0, 5.0, 6.0], [3]);
print(dot(a, b));
"#);
}

#[test]
fn parity_tensor_max_min() {
    assert_parity(r#"
let t = Tensor.from_vec([3.0, 1.0, 4.0, 1.0, 5.0], [5]);
print(t.max());
print(t.min());
"#);
}

#[test]
fn parity_tensor_var_std() {
    assert_parity(r#"
let t = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0], [5]);
print(t.var());
print(t.std());
"#);
}
