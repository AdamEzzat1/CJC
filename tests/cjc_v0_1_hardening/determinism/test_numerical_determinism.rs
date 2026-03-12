//! Numerical determinism: floating-point operations must be bit-identical.

#[path = "../helpers.rs"]
mod helpers;
use helpers::*;

/// Helper: run twice and verify bit-identical float output.
fn assert_float_deterministic(label: &str, src: &str) {
    let out1 = run_mir(src);
    let out2 = run_mir(src);
    assert_eq!(out1.len(), out2.len(), "[{label}] Output length mismatch");
    for (i, (a, b)) in out1.iter().zip(out2.iter()).enumerate() {
        if let (Ok(fa), Ok(fb)) = (a.parse::<f64>(), b.parse::<f64>()) {
            assert_eq!(
                fa.to_bits(),
                fb.to_bits(),
                "[{label}] Line {i}: bit mismatch: {fa} (bits: {}) vs {fb} (bits: {})",
                fa.to_bits(),
                fb.to_bits()
            );
        } else {
            assert_eq!(a, b, "[{label}] Line {i}: string mismatch");
        }
    }
}

// ============================================================
// Basic float operations
// ============================================================

#[test]
fn num_det_addition() {
    assert_float_deterministic("float add", r#"
fn main() {
    print(0.1 + 0.2);
    print(0.1 + 0.2 + 0.3);
    print(1e-15 + 1.0);
}
"#);
}

#[test]
fn num_det_multiplication() {
    assert_float_deterministic("float mul", r#"
fn main() {
    print(0.1 * 0.2);
    print(1e100 * 1e-100);
    print(3.14159 * 2.0);
}
"#);
}

#[test]
fn num_det_division() {
    assert_float_deterministic("float div", r#"
fn main() {
    print(1.0 / 3.0);
    print(22.0 / 7.0);
    print(1.0 / 7.0);
}
"#);
}

// ============================================================
// Transcendental functions
// ============================================================

#[test]
fn num_det_trig() {
    assert_float_deterministic("trig", r#"
fn main() {
    print(sin(0.5));
    print(cos(0.5));
    print(tan(0.5));
    print(asin(0.5));
    print(acos(0.5));
    print(atan(0.5));
}
"#);
}

#[test]
fn num_det_exp_log() {
    assert_float_deterministic("exp/log", r#"
fn main() {
    print(exp(1.0));
    print(log(2.0));
    print(log2(8.0));
    print(log10(100.0));
    print(expm1(0.001));
    print(log1p(0.001));
}
"#);
}

#[test]
fn num_det_hyperbolic() {
    assert_float_deterministic("hyperbolic", r#"
fn main() {
    print(sinh(1.0));
    print(cosh(1.0));
    print(tanh_scalar(1.0));
}
"#);
}

#[test]
fn num_det_sqrt_pow() {
    assert_float_deterministic("sqrt/pow", r#"
fn main() {
    print(sqrt(2.0));
    print(pow(2.0, 0.5));
    print(pow(10.0, 3.0));
    print(hypot(3.0, 4.0));
}
"#);
}

// ============================================================
// Tensor reductions
// ============================================================

#[test]
fn num_det_tensor_sum() {
    assert_float_deterministic("tensor sum", r#"
fn main() {
    let t = Tensor.from_vec([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [10]);
    print(t.sum());
}
"#);
}

#[test]
fn num_det_tensor_mean() {
    assert_float_deterministic("tensor mean", r#"
fn main() {
    let t = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0], [5]);
    print(t.mean());
}
"#);
}

#[test]
fn num_det_dot_product() {
    assert_float_deterministic("dot product", r#"
fn main() {
    let a = Tensor.from_vec([1.0, 2.0, 3.0], [3]);
    let b = Tensor.from_vec([4.0, 5.0, 6.0], [3]);
    print(dot(a, b));
}
"#);
}

#[test]
fn num_det_matmul() {
    assert_float_deterministic("matmul", r#"
fn main() {
    let a = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6]);
    let a = a.reshape([2, 3]);
    let b = Tensor.from_vec([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [6]);
    let b = b.reshape([3, 2]);
    let c = matmul(a, b);
    print(c);
}
"#);
}

// ============================================================
// Statistics determinism
// ============================================================

#[test]
fn num_det_variance() {
    assert_float_deterministic("variance", r#"
fn main() {
    let a: Any = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    print(variance(a));
    print(sd(a));
}
"#);
}

#[test]
fn num_det_median() {
    assert_float_deterministic("median", r#"
fn main() {
    let a: Any = [5.0, 3.0, 1.0, 4.0, 2.0];
    print(median(a));
}
"#);
}

// ============================================================
// Edge cases
// ============================================================

#[test]
fn num_det_special_values() {
    assert_float_deterministic("special values", r#"
fn main() {
    print(1.0 / 0.0);
    print(-1.0 / 0.0);
    print(0.0 / 0.0);
}
"#);
}

#[test]
fn num_det_very_small_numbers() {
    assert_float_deterministic("small numbers", r#"
fn main() {
    print(1e-300 + 1e-300);
    print(1e-300 * 1e-300);
}
"#);
}

#[test]
fn num_det_very_large_numbers() {
    assert_float_deterministic("large numbers", r#"
fn main() {
    print(1e300 + 1e300);
    print(1e300 * 2.0);
}
"#);
}
