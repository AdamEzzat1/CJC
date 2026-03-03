use super::helpers::*;

// ── Power operator (**) via pow() builtin ──

#[test]
fn pow_int_basic() {
    let out = run_mir(r#"
        print(pow(2, 10));
    "#);
    assert_close(parse_float(&out[0]), 1024.0, 1e-10);
}

#[test]
fn pow_float_basic() {
    let out = run_mir(r#"
        print(pow(2.0, 0.5));
    "#);
    assert_close(parse_float(&out[0]), std::f64::consts::SQRT_2, 1e-10);
}

#[test]
fn pow_float_negative_exponent() {
    let out = run_mir(r#"
        print(pow(2.0, -1.0));
    "#);
    assert_close(parse_float(&out[0]), 0.5, 1e-10);
}

#[test]
fn pow_zero_exponent() {
    let out = run_mir(r#"
        print(pow(5.0, 0.0));
    "#);
    assert_close(parse_float(&out[0]), 1.0, 1e-10);
}

#[test]
fn pow_one_exponent() {
    let out = run_mir(r#"
        print(pow(7.0, 1.0));
    "#);
    assert_close(parse_float(&out[0]), 7.0, 1e-10);
}

#[test]
fn pow_cubed() {
    let out = run_mir(r#"
        print(pow(3.0, 3.0));
    "#);
    assert_close(parse_float(&out[0]), 27.0, 1e-10);
}

// ── Power operator: ** syntax ──

#[test]
fn pow_operator_syntax_int() {
    let out = run_eval(r#"
        let x: i64 = 2 ** 10;
        print(x);
    "#);
    assert_eq!(out[0], "1024");
}

#[test]
fn pow_operator_syntax_float() {
    let out = run_eval(r#"
        let x: f64 = 3.0 ** 2.0;
        print(x);
    "#);
    assert_close(parse_float(&out[0]), 9.0, 1e-10);
}

// ── Power operator: right-associative ──

#[test]
fn pow_right_associative() {
    // 2 ** 3 ** 2 should be 2 ** (3 ** 2) = 2 ** 9 = 512
    // not (2 ** 3) ** 2 = 8 ** 2 = 64
    let out = run_eval(r#"
        let x: i64 = 2 ** 3 ** 2;
        print(x);
    "#);
    assert_eq!(out[0], "512");
}

// ── Power with precedence ──

#[test]
fn pow_higher_than_mul() {
    // 2 * 3 ** 2 should be 2 * 9 = 18, not 6 ** 2 = 36
    let out = run_eval(r#"
        let x: i64 = 2 * 3 ** 2;
        print(x);
    "#);
    assert_eq!(out[0], "18");
}
