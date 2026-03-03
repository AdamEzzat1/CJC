//! Test Suite 08: Tensor reductions (max, min, var, std, abs, mean_axis, max_axis, min_axis)

use super::helpers::*;

#[test]
fn tensor_max_basic() {
    let out = run_mir(r#"
let t = Tensor.from_vec([1.0, 5.0, 3.0, 2.0], [4]);
print(t.max());
"#);
    assert_close(parse_float(&out[0]), 5.0, 1e-15);
}

#[test]
fn tensor_min_basic() {
    let out = run_mir(r#"
let t = Tensor.from_vec([1.0, 5.0, 3.0, 2.0], [4]);
print(t.min());
"#);
    assert_close(parse_float(&out[0]), 1.0, 1e-15);
}

#[test]
fn tensor_max_negative() {
    let out = run_mir(r#"
let t = Tensor.from_vec([0.0 - 5.0, 0.0 - 1.0, 0.0 - 3.0], [3]);
print(t.max());
"#);
    assert_close(parse_float(&out[0]), -1.0, 1e-15);
}

#[test]
fn tensor_var_basic() {
    // var([1, 2, 3, 4, 5]) = 2.0 (population variance)
    let out = run_mir(r#"
let t = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0], [5]);
print(t.var());
"#);
    assert_close(parse_float(&out[0]), 2.0, 1e-14);
}

#[test]
fn tensor_std_basic() {
    // std([1, 2, 3, 4, 5]) = sqrt(2)
    let out = run_mir(r#"
let t = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0], [5]);
print(t.std());
"#);
    assert_close(parse_float(&out[0]), 2.0_f64.sqrt(), 1e-14);
}

#[test]
fn tensor_var_constant() {
    // var of constant tensor = 0
    let out = run_mir(r#"
let t = Tensor.full([10], 5.0);
print(t.var());
"#);
    assert_close(parse_float(&out[0]), 0.0, 1e-15);
}

#[test]
fn tensor_abs_basic() {
    let out = run_mir(r#"
let t = Tensor.from_vec([0.0 - 1.0, 2.0, 0.0 - 3.0, 4.0], [4]);
let a = t.abs();
print(a.sum());
"#);
    assert_close(parse_float(&out[0]), 10.0, 1e-15);
}

#[test]
fn tensor_mean_axis_rows() {
    // [[1, 2], [3, 4]] mean_axis(0) = [2, 3]
    let out = run_mir(r#"
let t = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]);
let m = t.mean_axis(0);
print(m.sum());
"#);
    assert_close(parse_float(&out[0]), 5.0, 1e-15);
}

#[test]
fn tensor_mean_axis_cols() {
    // [[1, 2], [3, 4]] mean_axis(1) = [1.5, 3.5]
    let out = run_mir(r#"
let t = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]);
let m = t.mean_axis(1);
print(m.sum());
"#);
    assert_close(parse_float(&out[0]), 5.0, 1e-15);
}

#[test]
fn tensor_max_axis_rows() {
    // [[1, 4], [3, 2]] max_axis(0) = [3, 4]
    let out = run_mir(r#"
let t = Tensor.from_vec([1.0, 4.0, 3.0, 2.0], [2, 2]);
let m = t.max_axis(0);
print(m.sum());
"#);
    assert_close(parse_float(&out[0]), 7.0, 1e-15);
}

#[test]
fn tensor_min_axis_cols() {
    // [[1, 4], [3, 2]] min_axis(1) = [1, 2]
    let out = run_mir(r#"
let t = Tensor.from_vec([1.0, 4.0, 3.0, 2.0], [2, 2]);
let m = t.min_axis(1);
print(m.sum());
"#);
    assert_close(parse_float(&out[0]), 3.0, 1e-15);
}
