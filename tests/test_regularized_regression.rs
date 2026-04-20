//! Integration tests for Ridge, Lasso, and ElasticNet regression builtins.
//!
//! Covers: correctness (known solution), R² range, parity (eval == mir-exec),
//! convergence flags, determinism (repeat run → same output), and error handling.

use cjc_runtime::{builtins::dispatch_builtin, Tensor, Value};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn make_x(data: Vec<f64>, n: usize, p: usize) -> Value {
    Value::Tensor(Tensor::from_vec(data, &[n, p]).unwrap())
}

fn make_y(data: Vec<f64>) -> Value {
    let n = data.len();
    Value::Tensor(Tensor::from_vec(data, &[n]).unwrap())
}

fn get_f64(val: &Value, field: &str) -> f64 {
    match val {
        Value::Struct { fields, .. } => match fields.get(field).unwrap() {
            Value::Float(v) => *v,
            Value::Int(v) => *v as f64,
            other => panic!("field {field} is not numeric: {other:?}"),
        },
        _ => panic!("expected Struct, got {val:?}"),
    }
}

fn get_bool(val: &Value, field: &str) -> bool {
    match val {
        Value::Struct { fields, .. } => match fields.get(field).unwrap() {
            Value::Bool(b) => *b,
            other => panic!("field {field} is not Bool: {other:?}"),
        },
        _ => panic!("expected Struct"),
    }
}

fn get_i64(val: &Value, field: &str) -> i64 {
    match val {
        Value::Struct { fields, .. } => match fields.get(field).unwrap() {
            Value::Int(v) => *v,
            Value::Float(v) => *v as i64,
            other => panic!("field {field} is not Int: {other:?}"),
        },
        _ => panic!("expected Struct"),
    }
}

fn get_coef_vec(val: &Value) -> Vec<f64> {
    match val {
        Value::Struct { fields, .. } => match fields.get("coefficients").unwrap() {
            Value::Tensor(t) => t.to_vec(),
            other => panic!("coefficients is not Tensor: {other:?}"),
        },
        _ => panic!("expected Struct"),
    }
}

/// Linear data: y = 2x + 1 + tiny_noise (nearly perfect fit).
fn linear_x_y() -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|xi| 2.0 * xi + 1.0).collect();
    (x, y)
}

/// 2-feature data: y = 3*x1 + -1*x2 + 5 (exact linear relationship).
fn two_feature_x_y() -> (Vec<f64>, Vec<f64>, usize, usize) {
    // 8 samples, 2 features
    let data = vec![
        1.0, 2.0,
        2.0, 1.0,
        3.0, 4.0,
        4.0, 3.0,
        5.0, 2.0,
        6.0, 5.0,
        7.0, 1.0,
        8.0, 3.0,
    ];
    let y: Vec<f64> = (0..8)
        .map(|i| 3.0 * data[i * 2] + -1.0 * data[i * 2 + 1] + 5.0)
        .collect();
    (data, y, 8, 2)
}

// ── Ridge regression tests ────────────────────────────────────────────────────

#[test]
fn ridge_returns_struct_with_required_fields() {
    let (x, y) = linear_x_y();
    let n = y.len();
    let result = dispatch_builtin(
        "ridge_regression",
        &[make_x(x, n, 1), make_y(y), Value::Float(0.1)],
    )
    .unwrap()
    .unwrap();
    match &result {
        Value::Struct { name, fields } => {
            assert_eq!(name, "RidgeResult");
            assert!(fields.contains_key("coefficients"), "missing coefficients");
            assert!(fields.contains_key("intercept"), "missing intercept");
            assert!(fields.contains_key("r_squared"), "missing r_squared");
            assert!(fields.contains_key("alpha"), "missing alpha");
        }
        _ => panic!("expected Struct, got {result:?}"),
    }
}

#[test]
fn ridge_r_squared_in_range() {
    let (x, y) = linear_x_y();
    let n = y.len();
    let result = dispatch_builtin(
        "ridge_regression",
        &[make_x(x, n, 1), make_y(y), Value::Float(0.01)],
    )
    .unwrap()
    .unwrap();
    let r2 = get_f64(&result, "r_squared");
    assert!(r2 >= 0.0 && r2 <= 1.0, "R² out of [0,1]: {r2}");
}

#[test]
fn ridge_near_ols_at_zero_alpha() {
    // With very small alpha, ridge ≈ OLS → R² should be very high for linear data.
    let (x, y) = linear_x_y();
    let n = y.len();
    let result = dispatch_builtin(
        "ridge_regression",
        &[make_x(x, n, 1), make_y(y), Value::Float(1e-8)],
    )
    .unwrap()
    .unwrap();
    let r2 = get_f64(&result, "r_squared");
    assert!(r2 > 0.999, "expected near-perfect R² for near-zero alpha, got {r2}");
}

#[test]
fn ridge_high_alpha_shrinks_toward_zero() {
    // Very high alpha forces coefficients toward zero; R² should drop.
    let (x, y) = linear_x_y();
    let n = y.len();

    let low_alpha = dispatch_builtin(
        "ridge_regression",
        &[make_x(x.clone(), n, 1), make_y(y.clone()), Value::Float(1e-6)],
    )
    .unwrap()
    .unwrap();
    let high_alpha = dispatch_builtin(
        "ridge_regression",
        &[make_x(x, n, 1), make_y(y), Value::Float(1000.0)],
    )
    .unwrap()
    .unwrap();

    let r2_low = get_f64(&low_alpha, "r_squared");
    let r2_high = get_f64(&high_alpha, "r_squared");
    assert!(r2_low > r2_high, "higher alpha should reduce R²: low={r2_low}, high={r2_high}");
}

#[test]
fn ridge_two_features_reasonable_r_squared() {
    let (x_data, y, n, p) = two_feature_x_y();
    let result = dispatch_builtin(
        "ridge_regression",
        &[make_x(x_data, n, p), make_y(y), Value::Float(0.01)],
    )
    .unwrap()
    .unwrap();
    let r2 = get_f64(&result, "r_squared");
    // Exact linear relationship → should be very high even with small regularization
    assert!(r2 > 0.99, "expected high R² for exact linear data, got {r2}");
}

#[test]
fn ridge_coefficient_count_matches_n_features() {
    let (x_data, y, n, p) = two_feature_x_y();
    let result = dispatch_builtin(
        "ridge_regression",
        &[make_x(x_data, n, p), make_y(y), Value::Float(0.1)],
    )
    .unwrap()
    .unwrap();
    let coef = get_coef_vec(&result);
    assert_eq!(coef.len(), p, "expected {p} coefficients, got {}", coef.len());
}

#[test]
fn ridge_alpha_stored_in_result() {
    let (x, y) = linear_x_y();
    let n = y.len();
    let result = dispatch_builtin(
        "ridge_regression",
        &[make_x(x, n, 1), make_y(y), Value::Float(0.42)],
    )
    .unwrap()
    .unwrap();
    let alpha = get_f64(&result, "alpha");
    assert!((alpha - 0.42).abs() < 1e-12, "alpha not stored correctly: {alpha}");
}

#[test]
fn ridge_deterministic_repeated_call() {
    let (x, y) = linear_x_y();
    let n = y.len();
    let args = || vec![make_x(x.clone(), n, 1), make_y(y.clone()), Value::Float(0.1)];
    let r1 = dispatch_builtin("ridge_regression", &args()).unwrap().unwrap();
    let r2 = dispatch_builtin("ridge_regression", &args()).unwrap().unwrap();
    assert_eq!(
        get_f64(&r1, "r_squared"),
        get_f64(&r2, "r_squared"),
        "ridge_regression must be deterministic"
    );
    let c1 = get_coef_vec(&r1);
    let c2 = get_coef_vec(&r2);
    assert_eq!(c1, c2, "coefficients must be bit-identical across calls");
}

#[test]
fn ridge_wrong_arg_count_errors() {
    let (x, y) = linear_x_y();
    let n = y.len();
    let result = dispatch_builtin("ridge_regression", &[make_x(x, n, 1), make_y(y)]);
    assert!(result.is_err(), "expected error with 2 args");
}

#[test]
fn ridge_mismatched_rows_errors() {
    // X has 10 rows, y has 5 elements
    let x = make_x(vec![1.0; 10], 10, 1);
    let y = make_y(vec![1.0; 5]);
    let result = dispatch_builtin("ridge_regression", &[x, y, Value::Float(0.1)]);
    assert!(result.is_err(), "expected error for mismatched dimensions");
}

// ── Lasso regression tests ────────────────────────────────────────────────────

#[test]
fn lasso_returns_struct_with_required_fields() {
    let (x, y) = linear_x_y();
    let n = y.len();
    let result = dispatch_builtin(
        "lasso_regression",
        &[make_x(x, n, 1), make_y(y), Value::Float(0.01)],
    )
    .unwrap()
    .unwrap();
    match &result {
        Value::Struct { name, fields } => {
            assert_eq!(name, "LassoResult");
            assert!(fields.contains_key("coefficients"), "missing coefficients");
            assert!(fields.contains_key("intercept"), "missing intercept");
            assert!(fields.contains_key("r_squared"), "missing r_squared");
            assert!(fields.contains_key("alpha"), "missing alpha");
            assert!(fields.contains_key("n_iter"), "missing n_iter");
            assert!(fields.contains_key("converged"), "missing converged");
        }
        _ => panic!("expected Struct, got {result:?}"),
    }
}

#[test]
fn lasso_r_squared_in_range() {
    let (x, y) = linear_x_y();
    let n = y.len();
    let result = dispatch_builtin(
        "lasso_regression",
        &[make_x(x, n, 1), make_y(y), Value::Float(0.01)],
    )
    .unwrap()
    .unwrap();
    let r2 = get_f64(&result, "r_squared");
    assert!(r2 >= 0.0 && r2 <= 1.0, "R² out of [0,1]: {r2}");
}

#[test]
fn lasso_converges_on_linear_data() {
    let (x, y) = linear_x_y();
    let n = y.len();
    let result = dispatch_builtin(
        "lasso_regression",
        &[
            make_x(x, n, 1),
            make_y(y),
            Value::Float(0.001),
            Value::Int(2000),
            Value::Float(1e-6),
        ],
    )
    .unwrap()
    .unwrap();
    assert!(
        get_bool(&result, "converged"),
        "lasso should converge on simple linear data"
    );
}

#[test]
fn lasso_n_iter_positive() {
    let (x, y) = linear_x_y();
    let n = y.len();
    let result = dispatch_builtin(
        "lasso_regression",
        &[make_x(x, n, 1), make_y(y), Value::Float(0.01)],
    )
    .unwrap()
    .unwrap();
    let n_iter = get_i64(&result, "n_iter");
    assert!(n_iter > 0, "n_iter should be positive, got {n_iter}");
}

#[test]
fn lasso_near_zero_alpha_high_r_squared() {
    let (x, y) = linear_x_y();
    let n = y.len();
    let result = dispatch_builtin(
        "lasso_regression",
        &[make_x(x, n, 1), make_y(y), Value::Float(1e-6)],
    )
    .unwrap()
    .unwrap();
    let r2 = get_f64(&result, "r_squared");
    assert!(r2 > 0.99, "expected near-perfect R² for near-zero alpha, got {r2}");
}

#[test]
fn lasso_two_features() {
    let (x_data, y, n, p) = two_feature_x_y();
    let result = dispatch_builtin(
        "lasso_regression",
        &[make_x(x_data, n, p), make_y(y), Value::Float(0.01)],
    )
    .unwrap()
    .unwrap();
    let coef = get_coef_vec(&result);
    assert_eq!(coef.len(), p);
    let r2 = get_f64(&result, "r_squared");
    assert!(r2 > 0.9, "expected high R² for exact linear data, got {r2}");
}

#[test]
fn lasso_default_args_3_param() {
    // 3-arg form: should use default max_iter=1000, tol=1e-4
    let (x, y) = linear_x_y();
    let n = y.len();
    let result = dispatch_builtin(
        "lasso_regression",
        &[make_x(x, n, 1), make_y(y), Value::Float(0.01)],
    );
    assert!(result.is_ok(), "3-arg form should succeed");
}

#[test]
fn lasso_deterministic_repeated_call() {
    let (x, y) = linear_x_y();
    let n = y.len();
    let args = || {
        vec![
            make_x(x.clone(), n, 1),
            make_y(y.clone()),
            Value::Float(0.01),
        ]
    };
    let r1 = dispatch_builtin("lasso_regression", &args()).unwrap().unwrap();
    let r2 = dispatch_builtin("lasso_regression", &args()).unwrap().unwrap();
    assert_eq!(get_f64(&r1, "r_squared"), get_f64(&r2, "r_squared"));
    assert_eq!(get_i64(&r1, "n_iter"), get_i64(&r2, "n_iter"));
    assert_eq!(get_bool(&r1, "converged"), get_bool(&r2, "converged"));
}

#[test]
fn lasso_wrong_arg_count_errors() {
    let (x, y) = linear_x_y();
    let n = y.len();
    let result = dispatch_builtin("lasso_regression", &[make_x(x, n, 1), make_y(y)]);
    assert!(result.is_err(), "expected error with 2 args");
}

// ── ElasticNet tests ──────────────────────────────────────────────────────────

#[test]
fn elastic_net_returns_struct_with_required_fields() {
    let (x, y) = linear_x_y();
    let n = y.len();
    let result = dispatch_builtin(
        "elastic_net",
        &[
            make_x(x, n, 1),
            make_y(y),
            Value::Float(0.01),
            Value::Float(0.5),
        ],
    )
    .unwrap()
    .unwrap();
    match &result {
        Value::Struct { name, fields } => {
            assert_eq!(name, "ElasticNetResult");
            assert!(fields.contains_key("coefficients"), "missing coefficients");
            assert!(fields.contains_key("intercept"), "missing intercept");
            assert!(fields.contains_key("r_squared"), "missing r_squared");
            assert!(fields.contains_key("alpha"), "missing alpha");
            assert!(fields.contains_key("l1_ratio"), "missing l1_ratio");
            assert!(fields.contains_key("n_iter"), "missing n_iter");
            assert!(fields.contains_key("converged"), "missing converged");
        }
        _ => panic!("expected Struct, got {result:?}"),
    }
}

#[test]
fn elastic_net_r_squared_in_range() {
    let (x, y) = linear_x_y();
    let n = y.len();
    let result = dispatch_builtin(
        "elastic_net",
        &[
            make_x(x, n, 1),
            make_y(y),
            Value::Float(0.01),
            Value::Float(0.5),
        ],
    )
    .unwrap()
    .unwrap();
    let r2 = get_f64(&result, "r_squared");
    assert!(r2 >= 0.0 && r2 <= 1.0, "R² out of [0,1]: {r2}");
}

#[test]
fn elastic_net_l1_ratio_stored() {
    let (x, y) = linear_x_y();
    let n = y.len();
    let result = dispatch_builtin(
        "elastic_net",
        &[
            make_x(x, n, 1),
            make_y(y),
            Value::Float(0.01),
            Value::Float(0.75),
        ],
    )
    .unwrap()
    .unwrap();
    let l1r = get_f64(&result, "l1_ratio");
    assert!((l1r - 0.75).abs() < 1e-12, "l1_ratio not stored correctly: {l1r}");
}

#[test]
fn elastic_net_l1_ratio_1_matches_lasso_behavior() {
    // l1_ratio=1.0 → pure Lasso behavior; both should give similar R²
    let (x, y) = linear_x_y();
    let n = y.len();
    let alpha = 0.001;

    let en = dispatch_builtin(
        "elastic_net",
        &[
            make_x(x.clone(), n, 1),
            make_y(y.clone()),
            Value::Float(alpha),
            Value::Float(1.0),
            Value::Int(5000),
            Value::Float(1e-6),
        ],
    )
    .unwrap()
    .unwrap();
    let lasso = dispatch_builtin(
        "lasso_regression",
        &[
            make_x(x, n, 1),
            make_y(y),
            Value::Float(alpha),
            Value::Int(5000),
            Value::Float(1e-6),
        ],
    )
    .unwrap()
    .unwrap();

    let en_r2 = get_f64(&en, "r_squared");
    let lasso_r2 = get_f64(&lasso, "r_squared");
    // Should be very close (same algorithm)
    assert!(
        (en_r2 - lasso_r2).abs() < 0.01,
        "elastic_net(l1_ratio=1) should ≈ lasso: en={en_r2}, lasso={lasso_r2}"
    );
}

#[test]
fn elastic_net_l1_ratio_0_closer_to_ridge() {
    // l1_ratio=0.0 → pure Ridge behavior (no L1 penalty)
    let (x, y) = linear_x_y();
    let n = y.len();
    let alpha = 0.1;

    let en = dispatch_builtin(
        "elastic_net",
        &[
            make_x(x.clone(), n, 1),
            make_y(y.clone()),
            Value::Float(alpha),
            Value::Float(0.0), // pure ridge
        ],
    )
    .unwrap()
    .unwrap();
    let ridge = dispatch_builtin(
        "ridge_regression",
        &[make_x(x, n, 1), make_y(y), Value::Float(alpha)],
    )
    .unwrap()
    .unwrap();

    let en_r2 = get_f64(&en, "r_squared");
    let ridge_r2 = get_f64(&ridge, "r_squared");
    // Should be close (same effective penalty structure)
    assert!(
        (en_r2 - ridge_r2).abs() < 0.05,
        "elastic_net(l1_ratio=0) should ≈ ridge: en={en_r2}, ridge={ridge_r2}"
    );
}

#[test]
fn elastic_net_converges_on_linear_data() {
    let (x, y) = linear_x_y();
    let n = y.len();
    let result = dispatch_builtin(
        "elastic_net",
        &[
            make_x(x, n, 1),
            make_y(y),
            Value::Float(0.001),
            Value::Float(0.5),
            Value::Int(2000),
            Value::Float(1e-6),
        ],
    )
    .unwrap()
    .unwrap();
    assert!(
        get_bool(&result, "converged"),
        "elastic_net should converge on simple linear data"
    );
}

#[test]
fn elastic_net_two_features() {
    let (x_data, y, n, p) = two_feature_x_y();
    let result = dispatch_builtin(
        "elastic_net",
        &[
            make_x(x_data, n, p),
            make_y(y),
            Value::Float(0.01),
            Value::Float(0.5),
        ],
    )
    .unwrap()
    .unwrap();
    let coef = get_coef_vec(&result);
    assert_eq!(coef.len(), p);
    let r2 = get_f64(&result, "r_squared");
    assert!(r2 > 0.9, "expected high R² for exact linear data, got {r2}");
}

#[test]
fn elastic_net_deterministic_repeated_call() {
    let (x, y) = linear_x_y();
    let n = y.len();
    let args = || {
        vec![
            make_x(x.clone(), n, 1),
            make_y(y.clone()),
            Value::Float(0.01),
            Value::Float(0.5),
        ]
    };
    let r1 = dispatch_builtin("elastic_net", &args()).unwrap().unwrap();
    let r2 = dispatch_builtin("elastic_net", &args()).unwrap().unwrap();
    assert_eq!(get_f64(&r1, "r_squared"), get_f64(&r2, "r_squared"));
    assert_eq!(get_f64(&r1, "l1_ratio"), get_f64(&r2, "l1_ratio"));
    assert_eq!(get_i64(&r1, "n_iter"), get_i64(&r2, "n_iter"));
}

#[test]
fn elastic_net_wrong_arg_count_errors() {
    let (x, y) = linear_x_y();
    let n = y.len();
    // needs at least 4 args (X, y, alpha, l1_ratio)
    let result = dispatch_builtin(
        "elastic_net",
        &[make_x(x, n, 1), make_y(y), Value::Float(0.01)],
    );
    assert!(result.is_err(), "expected error with only 3 args");
}

// ── Cross-method parity tests ─────────────────────────────────────────────────

#[test]
fn all_three_give_positive_r_squared_on_linear_data() {
    let (x, y) = linear_x_y();
    let n = y.len();

    let ridge = dispatch_builtin(
        "ridge_regression",
        &[make_x(x.clone(), n, 1), make_y(y.clone()), Value::Float(0.001)],
    )
    .unwrap()
    .unwrap();
    let lasso = dispatch_builtin(
        "lasso_regression",
        &[make_x(x.clone(), n, 1), make_y(y.clone()), Value::Float(0.001)],
    )
    .unwrap()
    .unwrap();
    let en = dispatch_builtin(
        "elastic_net",
        &[
            make_x(x, n, 1),
            make_y(y),
            Value::Float(0.001),
            Value::Float(0.5),
        ],
    )
    .unwrap()
    .unwrap();

    for (name, result) in [("ridge", &ridge), ("lasso", &lasso), ("elastic_net", &en)] {
        let r2 = get_f64(result, "r_squared");
        assert!(r2 > 0.0, "{name} R² should be positive for linear data, got {r2}");
    }
}

#[test]
fn eval_mir_parity_ridge() {
    // Run via CJC-Lang source through both executors and compare R²
    // Use Tensor.from_vec([data], [shape]) to create tensors in CJC-Lang source.
    let src = r#"
let x = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [8, 1]);
let y = Tensor.from_vec([2.1, 4.0, 5.9, 7.8, 10.2, 12.0, 14.1, 16.0], [8]);
let r = ridge_regression(x, y, 0.01);
print(r.r_squared);
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);

    let mut interp = cjc_eval::Interpreter::new(42);
    let _ = interp.exec(&program).unwrap();
    let eval_out = interp.output.clone();

    let (_, exec) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    let mir_out = exec.output.clone();

    assert_eq!(eval_out, mir_out, "eval vs mir-exec parity failed for ridge_regression");
}

#[test]
fn eval_mir_parity_lasso() {
    let src = r#"
let x = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [8, 1]);
let y = Tensor.from_vec([2.1, 4.0, 5.9, 7.8, 10.2, 12.0, 14.1, 16.0], [8]);
let r = lasso_regression(x, y, 0.01);
print(r.r_squared);
print(r.converged);
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);

    let mut interp = cjc_eval::Interpreter::new(42);
    let _ = interp.exec(&program).unwrap();
    let eval_out = interp.output.clone();

    let (_, exec) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    let mir_out = exec.output.clone();

    assert_eq!(eval_out, mir_out, "eval vs mir-exec parity failed for lasso_regression");
}

#[test]
fn eval_mir_parity_elastic_net() {
    let src = r#"
let x = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [8, 1]);
let y = Tensor.from_vec([2.1, 4.0, 5.9, 7.8, 10.2, 12.0, 14.1, 16.0], [8]);
let r = elastic_net(x, y, 0.01, 0.5);
print(r.r_squared);
print(r.converged);
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);

    let mut interp = cjc_eval::Interpreter::new(42);
    let _ = interp.exec(&program).unwrap();
    let eval_out = interp.output.clone();

    let (_, exec) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    let mir_out = exec.output.clone();

    assert_eq!(eval_out, mir_out, "eval vs mir-exec parity failed for elastic_net");
}
