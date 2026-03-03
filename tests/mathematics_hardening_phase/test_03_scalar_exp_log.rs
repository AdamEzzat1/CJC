//! Test Suite 03: Exponentiation & logarithms (pow, log2, log10, log1p, expm1)

use super::helpers::*;

#[test]
fn pow_basic() {
    let out = run_mir("print(pow(2.0, 3.0));");
    assert_close(parse_float(&out[0]), 8.0, 1e-15);
}

#[test]
fn pow_fractional() {
    let out = run_mir("print(pow(4.0, 0.5));");
    assert_close(parse_float(&out[0]), 2.0, 1e-15);
}

#[test]
fn pow_integer_args() {
    let out = run_mir("print(pow(3, 4));");
    assert_close(parse_float(&out[0]), 81.0, 1e-15);
}

#[test]
fn pow_zero_exponent() {
    let out = run_mir("print(pow(5.0, 0.0));");
    assert_close(parse_float(&out[0]), 1.0, 1e-15);
}

#[test]
fn log2_powers() {
    let out = run_mir("print(log2(8.0));");
    assert_close(parse_float(&out[0]), 3.0, 1e-15);
}

#[test]
fn log2_one() {
    let out = run_mir("print(log2(1.0));");
    assert_close(parse_float(&out[0]), 0.0, 1e-15);
}

#[test]
fn log10_hundred() {
    let out = run_mir("print(log10(100.0));");
    assert_close(parse_float(&out[0]), 2.0, 1e-15);
}

#[test]
fn log10_integer() {
    let out = run_mir("print(log10(1000));");
    assert_close(parse_float(&out[0]), 3.0, 1e-14);
}

#[test]
fn log1p_near_zero() {
    // log1p(x) is more precise than log(1+x) for small x
    let out = run_mir("print(log1p(1e-15));");
    assert_close(parse_float(&out[0]), 1e-15, 1e-28);
}

#[test]
fn log1p_zero() {
    let out = run_mir("print(log1p(0.0));");
    assert_close(parse_float(&out[0]), 0.0, 1e-15);
}

#[test]
fn expm1_near_zero() {
    // expm1(x) is more precise than exp(x)-1 for small x
    let out = run_mir("print(expm1(1e-15));");
    assert_close(parse_float(&out[0]), 1e-15, 1e-28);
}

#[test]
fn expm1_zero() {
    let out = run_mir("print(expm1(0.0));");
    assert_close(parse_float(&out[0]), 0.0, 1e-15);
}

#[test]
fn exp_log_inverse() {
    // exp(log(x)) = x
    let out = run_mir("print(exp(log(42.0)));");
    assert_close(parse_float(&out[0]), 42.0, 1e-12);
}

#[test]
fn pow_log_consistency() {
    // log(pow(b, e)) = e * log(b)
    let out = run_mir(r#"
let b = 3.0;
let e = 2.5;
let lhs = log(pow(b, e));
let rhs = e * log(b);
print(lhs - rhs);
"#);
    assert_close(parse_float(&out[0]), 0.0, 1e-12);
}
