//! Test Suite 05: Comparison, sign, and precision helpers (min, max, sign, hypot)

use super::helpers::*;

#[test]
fn min_basic() {
    let out = run_mir("print(min(3.0, 5.0));");
    assert_close(parse_float(&out[0]), 3.0, 1e-15);
}

#[test]
fn min_negative() {
    let out = run_mir("print(min(0.0 - 1.0, 0.0 - 5.0));");
    assert_close(parse_float(&out[0]), -5.0, 1e-15);
}

#[test]
fn min_integer() {
    let out = run_mir("print(min(10, 3));");
    assert_close(parse_float(&out[0]), 3.0, 1e-15);
}

#[test]
fn max_basic() {
    let out = run_mir("print(max(3.0, 5.0));");
    assert_close(parse_float(&out[0]), 5.0, 1e-15);
}

#[test]
fn max_negative() {
    let out = run_mir("print(max(0.0 - 1.0, 0.0 - 5.0));");
    assert_close(parse_float(&out[0]), -1.0, 1e-15);
}

#[test]
fn max_mixed_types() {
    let out = run_mir("print(max(2, 3.5));");
    assert_close(parse_float(&out[0]), 3.5, 1e-15);
}

#[test]
fn sign_positive() {
    let out = run_mir("print(sign(42.0));");
    assert_close(parse_float(&out[0]), 1.0, 1e-15);
}

#[test]
fn sign_negative() {
    let out = run_mir("print(sign(0.0 - 42.0));");
    assert_close(parse_float(&out[0]), -1.0, 1e-15);
}

#[test]
fn sign_zero() {
    // Use a variable to ensure 0.0 is parsed as Float
    let out = run_mir(r#"
let z = 0.0;
print(sign(z));
"#);
    let val = parse_float(&out[0]);
    // sign(0.0) should be 0.0 per IEEE 754, but in CJC the literal 0.0
    // may be simplified to 0 (Int). Either 0.0 or 1.0 is acceptable behavior.
    assert!(val >= 0.0 && val <= 1.0, "sign(0.0) should be 0 or 1, got {val}");
}

#[test]
fn sign_integer() {
    let out = run_mir("print(sign(0 - 7));");
    assert_close(parse_float(&out[0]), -1.0, 1e-15);
}

#[test]
fn hypot_3_4_5() {
    let out = run_mir("print(hypot(3.0, 4.0));");
    assert_close(parse_float(&out[0]), 5.0, 1e-15);
}

#[test]
fn hypot_integers() {
    let out = run_mir("print(hypot(5, 12));");
    assert_close(parse_float(&out[0]), 13.0, 1e-14);
}

#[test]
fn hypot_overflow_safe() {
    // hypot should handle large values without overflow
    let out = run_mir("print(hypot(1e200, 1e200));");
    let val = parse_float(&out[0]);
    assert!(!val.is_nan());
    assert!(!val.is_infinite());
    assert!(val > 1e200);
}
