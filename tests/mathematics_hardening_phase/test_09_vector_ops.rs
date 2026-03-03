//! Test Suite 09: Vector operations (dot, outer, cross, norm)

use super::helpers::*;

#[test]
fn dot_basic() {
    let out = run_mir(r#"
let a = Tensor.from_vec([1.0, 2.0, 3.0], [3]);
let b = Tensor.from_vec([4.0, 5.0, 6.0], [3]);
print(dot(a, b));
"#);
    assert_close(parse_float(&out[0]), 32.0, 1e-15);
}

#[test]
fn dot_orthogonal() {
    let out = run_mir(r#"
let a = Tensor.from_vec([1.0, 0.0, 0.0], [3]);
let b = Tensor.from_vec([0.0, 1.0, 0.0], [3]);
print(dot(a, b));
"#);
    assert_close(parse_float(&out[0]), 0.0, 1e-15);
}

#[test]
fn dot_self_is_norm_squared() {
    let out = run_mir(r#"
let a = Tensor.from_vec([3.0, 4.0], [2]);
print(dot(a, a));
"#);
    assert_close(parse_float(&out[0]), 25.0, 1e-15);
}

#[test]
fn outer_basic() {
    let out = run_mir(r#"
let a = Tensor.from_vec([1.0, 2.0], [2]);
let b = Tensor.from_vec([3.0, 4.0, 5.0], [3]);
let M = outer(a, b);
print(M.sum());
"#);
    // outer product sum = (1+2)*(3+4+5) = 3*12 = 36
    assert_close(parse_float(&out[0]), 36.0, 1e-15);
}

#[test]
fn outer_shape() {
    let out = run_mir(r#"
let a = Tensor.from_vec([1.0, 2.0], [2]);
let b = Tensor.from_vec([3.0, 4.0, 5.0], [3]);
let M = outer(a, b);
print(len(M));
"#);
    assert_eq!(out[0].trim(), "6");
}

#[test]
fn cross_basic() {
    // i × j = k
    let out = run_mir(r#"
let i_hat = Tensor.from_vec([1.0, 0.0, 0.0], [3]);
let j_hat = Tensor.from_vec([0.0, 1.0, 0.0], [3]);
let k_hat = cross(i_hat, j_hat);
print(k_hat.sum());
"#);
    // k_hat should be [0, 0, 1], sum = 1
    assert_close(parse_float(&out[0]), 1.0, 1e-15);
}

#[test]
fn cross_anticommutative() {
    // a × b = -(b × a)
    let out = run_mir(r#"
let a = Tensor.from_vec([1.0, 2.0, 3.0], [3]);
let b = Tensor.from_vec([4.0, 5.0, 6.0], [3]);
let ab = cross(a, b);
let ba = cross(b, a);
let diff = ab.sum() + ba.sum();
print(diff);
"#);
    assert_close(parse_float(&out[0]), 0.0, 1e-14);
}

#[test]
fn cross_parallel_is_zero() {
    // a × a = 0
    let out = run_mir(r#"
let a = Tensor.from_vec([1.0, 2.0, 3.0], [3]);
let c = cross(a, a);
print(c.sum());
"#);
    assert_close(parse_float(&out[0]), 0.0, 1e-15);
}

#[test]
fn norm_l2_basic() {
    let out = run_mir(r#"
let a = Tensor.from_vec([3.0, 4.0], [2]);
print(norm(a));
"#);
    assert_close(parse_float(&out[0]), 5.0, 1e-15);
}

#[test]
fn norm_l1() {
    let out = run_mir(r#"
let a = Tensor.from_vec([0.0 - 1.0, 2.0, 0.0 - 3.0], [3]);
print(norm(a, 1));
"#);
    assert_close(parse_float(&out[0]), 6.0, 1e-15);
}

#[test]
fn norm_l2_unit() {
    let out = run_mir(r#"
let a = Tensor.from_vec([1.0, 0.0, 0.0], [3]);
print(norm(a));
"#);
    assert_close(parse_float(&out[0]), 1.0, 1e-15);
}
