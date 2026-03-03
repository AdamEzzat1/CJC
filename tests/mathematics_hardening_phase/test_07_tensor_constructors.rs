//! Test Suite 07: Tensor constructors (linspace, arange, eye, full, diag, uniform)

use super::helpers::*;

#[test]
fn linspace_basic() {
    let out = run_mir(r#"
let t = Tensor.linspace(0.0, 1.0, 5);
print(len(t));
print(t.to_vec());
"#);
    assert_eq!(out[0].trim(), "5");
}

#[test]
fn linspace_endpoints() {
    let out = run_mir(r#"
let t = Tensor.linspace(0.0, 10.0, 3);
let v = t.to_vec();
print(v[0]);
print(v[1]);
print(v[2]);
"#);
    assert_close(parse_float(&out[0]), 0.0, 1e-15);
    assert_close(parse_float(&out[1]), 5.0, 1e-14);
    assert_close(parse_float(&out[2]), 10.0, 1e-14);
}

#[test]
fn linspace_single_point() {
    let out = run_mir(r#"
let t = Tensor.linspace(5.0, 5.0, 1);
print(len(t));
"#);
    assert_eq!(out[0].trim(), "1");
}

#[test]
fn linspace_empty() {
    let out = run_mir(r#"
let t = Tensor.linspace(0.0, 1.0, 0);
print(len(t));
"#);
    assert_eq!(out[0].trim(), "0");
}

#[test]
fn arange_basic() {
    let out = run_mir(r#"
let t = Tensor.arange(0.0, 5.0, 1.0);
print(len(t));
"#);
    assert_eq!(out[0].trim(), "5");
}

#[test]
fn arange_step() {
    let out = run_mir(r#"
let t = Tensor.arange(0.0, 10.0, 2.0);
print(len(t));
"#);
    assert_eq!(out[0].trim(), "5");
}

#[test]
fn arange_default_step() {
    let out = run_mir(r#"
let t = Tensor.arange(0.0, 3.0);
print(len(t));
"#);
    assert_eq!(out[0].trim(), "3");
}

#[test]
fn eye_3x3() {
    let out = run_mir(r#"
let I = Tensor.eye(3);
print(I.shape());
"#);
    assert!(out[0].contains("3") && out[0].contains("3"));
}

#[test]
fn eye_diagonal_ones() {
    let out = run_mir(r#"
let I = Tensor.eye(3);
print(I.sum());
"#);
    assert_close(parse_float(&out[0]), 3.0, 1e-15);
}

#[test]
fn full_2d() {
    let out = run_mir(r#"
let t = Tensor.full([2, 3], 7.0);
print(t.sum());
"#);
    assert_close(parse_float(&out[0]), 42.0, 1e-15);
}

#[test]
fn full_1d() {
    let out = run_mir(r#"
let t = Tensor.full([5], 3.0);
print(len(t));
print(t.sum());
"#);
    assert_eq!(out[0].trim(), "5");
    assert_close(parse_float(&out[1]), 15.0, 1e-15);
}

#[test]
fn diag_1d_to_2d() {
    let out = run_mir(r#"
let v = Tensor.from_vec([1.0, 2.0, 3.0], [3]);
let D = Tensor.diag(v);
print(D.sum());
"#);
    assert_close(parse_float(&out[0]), 6.0, 1e-15);
}

#[test]
fn diag_2d_to_1d() {
    let out = run_mir(r#"
let M = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]);
let d = Tensor.diag(M);
print(d.sum());
"#);
    assert_close(parse_float(&out[0]), 5.0, 1e-15);
}

#[test]
fn diag_roundtrip() {
    // diag(diag(v)) should have the same diagonal as v
    let out = run_mir(r#"
let v = Tensor.from_vec([10.0, 20.0, 30.0], [3]);
let M = Tensor.diag(v);
let d = Tensor.diag(M);
print(d.sum());
"#);
    assert_close(parse_float(&out[0]), 60.0, 1e-15);
}

#[test]
fn uniform_shape() {
    let out = run_mir(r#"
let t = Tensor.uniform([2, 3]);
print(len(t));
"#);
    assert_eq!(out[0].trim(), "6");
}

#[test]
fn uniform_range() {
    // All values should be in [0, 1)
    let out = run_mir(r#"
let t = Tensor.uniform([100]);
let mx = t.max();
let mn = t.min();
print(mn);
print(mx);
"#);
    let mn = parse_float(&out[0]);
    let mx = parse_float(&out[1]);
    assert!(mn >= 0.0, "min should be >= 0, got {mn}");
    assert!(mx < 1.0, "max should be < 1, got {mx}");
}
