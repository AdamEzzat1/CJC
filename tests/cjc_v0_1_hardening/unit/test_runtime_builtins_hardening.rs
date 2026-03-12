//! Runtime builtins hardening — verify shared dispatch works for key functions.

#[path = "../helpers.rs"]
mod helpers;
use helpers::*;

// ============================================================
// Math builtins
// ============================================================

#[test]
fn builtin_abs() {
    let out = run_mir("fn main() -> f64 { abs(-5.0) }");
    assert_eq!(out, vec![] as Vec<String>); // no print output
    // Verify via print
    let out = run_mir("fn main() { print(abs(-5.0)); }");
    assert_eq!(out, vec!["5"]);
}

#[test]
fn builtin_sqrt() {
    let out = run_mir("fn main() { print(sqrt(4.0)); }");
    assert_eq!(out, vec!["2"]);
}

#[test]
fn builtin_sin_cos() {
    let out = run_mir("fn main() { print(sin(0.0)); print(cos(0.0)); }");
    assert_eq!(out, vec!["0", "1"]);
}

#[test]
fn builtin_log_exp() {
    let out = run_mir("fn main() { print(log(1.0)); print(exp(0.0)); }");
    assert_eq!(out, vec!["0", "1"]);
}

#[test]
fn builtin_floor_ceil_round() {
    let out = run_mir("fn main() { print(floor(3.7)); print(ceil(3.2)); print(round(3.5)); }");
    assert_eq!(out, vec!["3", "4", "4"]);
}

#[test]
fn builtin_pow() {
    let out = run_mir("fn main() { print(pow(2.0, 10.0)); }");
    assert_eq!(out, vec!["1024"]);
}

#[test]
fn builtin_min_max() {
    let out = run_mir("fn main() { print(min(3.0, 7.0)); print(max(3.0, 7.0)); }");
    assert_eq!(out, vec!["3", "7"]);
}

#[test]
fn builtin_sign() {
    // sign returns f64 signum — test via individual calls
    let out = run_mir("fn main() { print(sign(-5.0)); }");
    let val: f64 = out[0].parse().unwrap();
    assert!(val < 0.0, "sign(-5.0) should be negative, got {val}");
}

#[test]
fn builtin_hypot() {
    let out = run_mir("fn main() { print(hypot(3.0, 4.0)); }");
    assert_eq!(out, vec!["5"]);
}

// ============================================================
// Trig builtins
// ============================================================

#[test]
fn builtin_tan() {
    let out = run_mir("fn main() { print(tan(0.0)); }");
    assert_eq!(out, vec!["0"]);
}

#[test]
fn builtin_asin_acos_atan() {
    let out = run_mir("fn main() { print(asin(0.0)); print(acos(1.0)); print(atan(0.0)); }");
    assert_eq!(out, vec!["0", "0", "0"]);
}

#[test]
fn builtin_sinh_cosh() {
    let out = run_mir("fn main() { print(sinh(0.0)); print(cosh(0.0)); }");
    assert_eq!(out, vec!["0", "1"]);
}

// ============================================================
// Array builtins
// ============================================================

#[test]
fn builtin_array_push() {
    let out = run_mir(r#"
fn main() {
    let arr: Any = [1, 2, 3];
    let arr2 = array_push(arr, 4);
    print(array_len(arr2));
}
"#);
    assert_eq!(out, vec!["4"]);
}

#[test]
fn builtin_array_contains() {
    let out = run_mir(r#"
fn main() {
    let arr: Any = [1, 2, 3];
    print(array_contains(arr, 2));
    print(array_contains(arr, 5));
}
"#);
    assert_eq!(out, vec!["true", "false"]);
}

#[test]
fn builtin_array_reverse() {
    let out = run_mir(r#"
fn main() {
    let arr: Any = [1, 2, 3];
    let rev = array_reverse(arr);
    print(rev);
}
"#);
    assert_eq!(out, vec!["[3, 2, 1]"]);
}

// ============================================================
// Bitwise builtins
// ============================================================

#[test]
fn builtin_bitwise_and() {
    let out = run_mir("fn main() { print(bit_and(0xFF, 0x0F)); }");
    assert_eq!(out, vec!["15"]);
}

#[test]
fn builtin_bitwise_or() {
    let out = run_mir("fn main() { print(bit_or(0xF0, 0x0F)); }");
    assert_eq!(out, vec!["255"]);
}

#[test]
fn builtin_bitwise_xor() {
    let out = run_mir("fn main() { print(bit_xor(0xFF, 0xFF)); }");
    assert_eq!(out, vec!["0"]);
}

#[test]
fn builtin_popcount() {
    let out = run_mir("fn main() { print(popcount(0xFF)); }");
    assert_eq!(out, vec!["8"]);
}

// ============================================================
// String conversion
// ============================================================

#[test]
fn builtin_to_string() {
    let out = run_mir(r#"fn main() { print(to_string(42)); }"#);
    assert_eq!(out, vec!["42"]);
}

// ============================================================
// Statistics builtins
// ============================================================

#[test]
fn builtin_variance() {
    let out = run_mir(r#"
fn main() {
    let a: Any = [1.0, 2.0, 3.0, 4.0, 5.0];
    print(variance(a));
}
"#);
    let val: f64 = out[0].parse().unwrap();
    assert!((val - 2.0).abs() < 0.6 || (val - 2.5).abs() < 0.6, "variance should be near 2 or 2.5, got {val}");
}

#[test]
fn builtin_sd() {
    let out = run_mir(r#"
fn main() {
    let a: Any = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    print(sd(a));
}
"#);
    let val: f64 = out[0].parse().unwrap();
    assert!(val > 1.0 && val < 3.0, "sd should be reasonable, got {val}");
}

#[test]
fn builtin_median() {
    let out = run_mir(r#"
fn main() {
    let a: Any = [1.0, 3.0, 2.0];
    print(median(a));
}
"#);
    assert_eq!(out, vec!["2"]);
}

// ============================================================
// Constants
// ============================================================

#[test]
fn builtin_constants() {
    let out = run_mir(r#"
fn main() {
    print(PI());
    print(E());
}
"#);
    let pi: f64 = out[0].parse().unwrap();
    let e: f64 = out[1].parse().unwrap();
    assert!((pi - std::f64::consts::PI).abs() < 1e-10, "PI should be pi");
    assert!((e - std::f64::consts::E).abs() < 1e-10, "E should be e");
}
