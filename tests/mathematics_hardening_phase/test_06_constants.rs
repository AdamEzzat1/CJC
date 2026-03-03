//! Test Suite 06: Mathematical constants (PI, E, TAU, INF, NAN_VAL)

use super::helpers::*;

#[test]
fn pi_value() {
    let out = run_mir("print(PI());");
    assert_close(parse_float(&out[0]), std::f64::consts::PI, 1e-15);
}

#[test]
fn e_value() {
    let out = run_mir("print(E());");
    assert_close(parse_float(&out[0]), std::f64::consts::E, 1e-15);
}

#[test]
fn tau_value() {
    let out = run_mir("print(TAU());");
    assert_close(parse_float(&out[0]), std::f64::consts::TAU, 1e-15);
}

#[test]
fn tau_is_2pi() {
    let out = run_mir("print(TAU() - 2.0 * PI());");
    assert_close(parse_float(&out[0]), 0.0, 1e-14);
}

#[test]
fn inf_value() {
    let out = run_mir("print(INF());");
    assert!(parse_float(&out[0]).is_infinite());
    assert!(parse_float(&out[0]) > 0.0);
}

#[test]
fn nan_val_is_nan() {
    let out = run_mir("print(NAN_VAL());");
    assert_nan(parse_float(&out[0]));
}

#[test]
fn pi_in_expression() {
    // sin(PI) should be approximately 0
    let out = run_mir("print(sin(PI()));");
    assert_close(parse_float(&out[0]), 0.0, 1e-14);
}

#[test]
fn e_exp_log() {
    // log(E()) = 1
    let out = run_mir("print(log(E()));");
    assert_close(parse_float(&out[0]), 1.0, 1e-15);
}
