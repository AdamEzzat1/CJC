//! Test Suite 04: Rounding functions (ceil, round, floor already tested)

use super::helpers::*;

#[test]
fn ceil_positive_frac() {
    let out = run_mir("print(ceil(2.3));");
    assert_close(parse_float(&out[0]), 3.0, 1e-15);
}

#[test]
fn ceil_negative_frac() {
    let out = run_mir("print(ceil(0.0 - 2.3));");
    assert_close(parse_float(&out[0]), -2.0, 1e-15);
}

#[test]
fn ceil_integer() {
    let out = run_mir("print(ceil(5));");
    let val = out[0].trim().parse::<i64>().unwrap();
    assert_eq!(val, 5);
}

#[test]
fn round_half_up() {
    let out = run_mir("print(round(2.5));");
    assert_close(parse_float(&out[0]), 3.0, 1e-15);
}

#[test]
fn round_down() {
    let out = run_mir("print(round(2.3));");
    assert_close(parse_float(&out[0]), 2.0, 1e-15);
}

#[test]
fn round_negative() {
    let out = run_mir("print(round(0.0 - 2.7));");
    assert_close(parse_float(&out[0]), -3.0, 1e-15);
}

#[test]
fn round_integer() {
    let out = run_mir("print(round(7));");
    let val = out[0].trim().parse::<i64>().unwrap();
    assert_eq!(val, 7);
}

#[test]
fn floor_ceil_bracket() {
    // floor(x) <= x <= ceil(x)
    let out = run_mir(r#"
let x = 3.7;
let f = floor(x);
let c = ceil(x);
print(f);
print(c);
"#);
    assert_close(parse_float(&out[0]), 3.0, 1e-15);
    assert_close(parse_float(&out[1]), 4.0, 1e-15);
}
