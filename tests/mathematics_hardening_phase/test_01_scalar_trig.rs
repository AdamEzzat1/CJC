//! Test Suite 01: Trigonometric functions (sin, cos, tan, asin, acos, atan, atan2)

use super::helpers::*;

#[test]
fn sin_zero() {
    let out = run_mir("print(sin(0.0));");
    assert_close(parse_float(&out[0]), 0.0, 1e-15);
}

#[test]
fn sin_pi_over_2() {
    let out = run_mir("print(sin(PI() / 2.0));");
    assert_close(parse_float(&out[0]), 1.0, 1e-15);
}

#[test]
fn sin_integer_input() {
    let out = run_mir("print(sin(0));");
    assert_close(parse_float(&out[0]), 0.0, 1e-15);
}

#[test]
fn cos_zero() {
    let out = run_mir("print(cos(0.0));");
    assert_close(parse_float(&out[0]), 1.0, 1e-15);
}

#[test]
fn cos_pi() {
    let out = run_mir("print(cos(PI()));");
    assert_close(parse_float(&out[0]), -1.0, 1e-15);
}

#[test]
fn tan_zero() {
    let out = run_mir("print(tan(0.0));");
    assert_close(parse_float(&out[0]), 0.0, 1e-15);
}

#[test]
fn tan_pi_over_4() {
    let out = run_mir("print(tan(PI() / 4.0));");
    assert_close(parse_float(&out[0]), 1.0, 1e-14);
}

#[test]
fn asin_one() {
    let out = run_mir("print(asin(1.0));");
    assert_close(parse_float(&out[0]), std::f64::consts::FRAC_PI_2, 1e-15);
}

#[test]
fn asin_nan_out_of_domain() {
    let out = run_mir("print(asin(2.0));");
    assert_nan(parse_float(&out[0]));
}

#[test]
fn acos_one() {
    let out = run_mir("print(acos(1.0));");
    assert_close(parse_float(&out[0]), 0.0, 1e-15);
}

#[test]
fn atan_one() {
    let out = run_mir("print(atan(1.0));");
    assert_close(parse_float(&out[0]), std::f64::consts::FRAC_PI_4, 1e-15);
}

#[test]
fn atan2_basic() {
    let out = run_mir("print(atan2(1.0, 1.0));");
    assert_close(parse_float(&out[0]), std::f64::consts::FRAC_PI_4, 1e-15);
}

#[test]
fn atan2_negative_y() {
    let out = run_mir("print(atan2(0.0 - 1.0, 0.0));");
    assert_close(parse_float(&out[0]), -std::f64::consts::FRAC_PI_2, 1e-15);
}

#[test]
fn sin_cos_pythagorean_identity() {
    // sin²(x) + cos²(x) = 1 for any x
    let out = run_mir(r#"
let x = 1.23456;
let s = sin(x);
let c = cos(x);
print(s * s + c * c);
"#);
    assert_close(parse_float(&out[0]), 1.0, 1e-14);
}
