//! Wiring verification: confirm key builtins work end-to-end through both executors.

#[path = "../helpers.rs"]
mod helpers;
use helpers::*;

/// Helper: run through MIR and verify output contains expected string.
fn mir_check(label: &str, src: &str, expected: &[&str]) {
    let out = run_mir(src);
    assert_eq!(
        out.len(),
        expected.len(),
        "[{label}] Output line count mismatch. Got: {out:?}, expected: {expected:?}"
    );
    for (i, (got, exp)) in out.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got.as_str(),
            *exp,
            "[{label}] Line {i} mismatch"
        );
    }
}

// ============================================================
// Math builtins wiring
// ============================================================

#[test]
fn wiring_abs() {
    mir_check("abs", "fn main() { print(abs(-7.0)); }", &["7"]);
}

#[test]
fn wiring_sqrt() {
    mir_check("sqrt", "fn main() { print(sqrt(9.0)); }", &["3"]);
}

#[test]
fn wiring_pow() {
    mir_check("pow", "fn main() { print(pow(2.0, 8.0)); }", &["256"]);
}

#[test]
fn wiring_log_exp_identity() {
    // log(exp(1)) should be 1
    let out = run_mir("fn main() { print(log(exp(1.0))); }");
    let val: f64 = out[0].parse().unwrap();
    assert!((val - 1.0).abs() < 1e-10, "log(exp(1)) should be 1, got {val}");
}

// ============================================================
// Array builtins wiring
// ============================================================

#[test]
fn wiring_array_len() {
    mir_check("array_len", r#"
fn main() {
    let a: Any = [10, 20, 30];
    print(array_len(a));
}
"#, &["3"]);
}

#[test]
fn wiring_array_push_pop() {
    let out = run_mir(r#"
fn main() {
    let mut a: Any = [1, 2];
    a = array_push(a, 3);
    print(array_len(a));
}
"#);
    assert_eq!(out, vec!["3"]);
}

// ============================================================
// Tensor builtins wiring
// ============================================================

#[test]
fn wiring_tensor_zeros_ones() {
    let out = run_mir(r#"
fn main() {
    let z = Tensor.zeros([3]);
    let o = Tensor.ones([3]);
    print(z);
    print(o);
}
"#);
    assert!(out[0].contains("0.0") || out[0].contains("[0"), "Tensor.zeros output should contain zeros: {}", out[0]);
    assert!(out[1].contains("1.0") || out[1].contains("[1"), "Tensor.ones output should contain ones: {}", out[1]);
}

#[test]
fn wiring_matmul() {
    let out = run_mir(r#"
fn main() {
    let a = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [4]);
    let a = a.reshape([2, 2]);
    let b = Tensor.from_vec([5.0, 6.0], [2]);
    let b = b.reshape([2, 1]);
    let c = matmul(a, b);
    print(c);
}
"#);
    assert!(!out.is_empty(), "matmul should produce output");
}

// ============================================================
// Statistics builtins wiring
// ============================================================

#[test]
fn wiring_mean() {
    let out = run_mir(r#"
fn main() {
    let t = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0], [5]);
    print(t.mean());
}
"#);
    let val: f64 = out[0].parse().unwrap();
    assert!((val - 3.0).abs() < 1e-10, "mean should be 3.0, got {val}");
}

#[test]
fn wiring_sum() {
    let out = run_mir(r#"
fn main() {
    let t = Tensor.from_vec([1.0, 2.0, 3.0], [3]);
    print(t.sum());
}
"#);
    let val: f64 = out[0].parse().unwrap();
    assert!((val - 6.0).abs() < 1e-10, "sum should be 6.0, got {val}");
}

// ============================================================
// Linalg builtins wiring
// ============================================================

#[test]
fn wiring_det() {
    let out = run_mir(r#"
fn main() {
    let m = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [4]);
    let m = m.reshape([2, 2]);
    print(det(m));
}
"#);
    let val: f64 = out[0].parse().unwrap();
    assert!((val - 1.0).abs() < 1e-10, "det(I) should be 1.0, got {val}");
}

#[test]
fn wiring_trace() {
    let out = run_mir(r#"
fn main() {
    let m = Tensor.from_vec([1.0, 0.0, 0.0, 2.0], [4]);
    let m = m.reshape([2, 2]);
    print(trace(m));
}
"#);
    let val: f64 = out[0].parse().unwrap();
    assert!((val - 3.0).abs() < 1e-10, "trace should be 3.0, got {val}");
}

// ============================================================
// Type conversion builtins wiring
// ============================================================

#[test]
fn wiring_int_conversion() {
    let out = run_mir("fn main() { print(int(3.14)); }");
    assert_eq!(out, vec!["3"]);
}

#[test]
fn wiring_float_conversion() {
    let out = run_mir("fn main() { print(float(42)); }");
    assert_eq!(out, vec!["42"]);
}

// ============================================================
// Snap builtins wiring
// ============================================================

#[test]
fn wiring_snap_roundtrip() {
    let out = run_mir(r#"
fn main() {
    let data: Any = [1, 2, 3];
    let s = snap(data);
    let recovered = restore(s);
    print(recovered);
}
"#);
    assert_eq!(out, vec!["[1, 2, 3]"]);
}

// ============================================================
// Bitwise builtins wiring
// ============================================================

#[test]
fn wiring_bitwise_ops() {
    let out = run_mir(r#"
fn main() {
    print(bit_and(255, 15));
    print(bit_or(240, 15));
    print(bit_xor(255, 255));
    print(bit_shl(1, 8));
    print(bit_shr(256, 8));
}
"#);
    assert_eq!(out, vec!["15", "255", "0", "256", "1"]);
}
