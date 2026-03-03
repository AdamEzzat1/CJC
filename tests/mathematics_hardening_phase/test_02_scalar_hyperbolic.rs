//! Test Suite 02: Hyperbolic functions (sinh, cosh, tanh_scalar)

use super::helpers::*;

#[test]
fn sinh_zero() {
    let out = run_mir("print(sinh(0.0));");
    assert_close(parse_float(&out[0]), 0.0, 1e-15);
}

#[test]
fn sinh_one() {
    let out = run_mir("print(sinh(1.0));");
    assert_close(parse_float(&out[0]), 1.0_f64.sinh(), 1e-14);
}

#[test]
fn sinh_integer() {
    let out = run_mir("print(sinh(0));");
    assert_close(parse_float(&out[0]), 0.0, 1e-15);
}

#[test]
fn cosh_zero() {
    let out = run_mir("print(cosh(0.0));");
    assert_close(parse_float(&out[0]), 1.0, 1e-15);
}

#[test]
fn cosh_one() {
    let out = run_mir("print(cosh(1.0));");
    assert_close(parse_float(&out[0]), 1.0_f64.cosh(), 1e-14);
}

#[test]
fn tanh_scalar_zero() {
    let out = run_mir("print(tanh_scalar(0.0));");
    assert_close(parse_float(&out[0]), 0.0, 1e-15);
}

#[test]
fn tanh_scalar_large() {
    let out = run_mir("print(tanh_scalar(100.0));");
    assert_close(parse_float(&out[0]), 1.0, 1e-10);
}

#[test]
fn cosh_sinh_identity() {
    // cosh²(x) - sinh²(x) = 1
    let out = run_mir(r#"
let x = 2.5;
let c = cosh(x);
let s = sinh(x);
print(c * c - s * s);
"#);
    assert_close(parse_float(&out[0]), 1.0, 1e-12);
}
