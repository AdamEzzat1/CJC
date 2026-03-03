//! v0.1 Contract Tests: lm() Linear Model
//!
//! Locks down: intercept auto-add, coefficient counts, R² range,
//! residuals length, argument validation, rank deficiency, eval/MIR parity.

// ── Helpers ──────────────────────────────────────────────────────

fn parse(src: &str) -> cjc_ast::Program {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    program
}

fn eval_output(src: &str) -> Vec<String> {
    let program = parse(src);
    let mut interp = cjc_eval::Interpreter::new(42);
    let _ = interp.exec(&program).unwrap();
    interp.output.clone()
}

fn mir_output(src: &str) -> Vec<String> {
    let program = parse(src);
    let (_, exec) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    exec.output.clone()
}

// ── Tests ────────────────────────────────────────────────────────

#[test]
fn lm_intercept_auto_add() {
    // p=1 predictor → 2 coefficients (intercept + slope)
    let src = r#"
let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let y = [2.1, 4.0, 5.8, 8.1, 9.9, 12.2, 13.8, 16.1, 18.0, 20.1];
let result = lm(x, y, 10, 1);
let coeffs = result.coefficients;
print(len(coeffs));
"#;
    let out = eval_output(src);
    assert_eq!(out, vec!["2"], "p=1 → should have 2 coefficients (intercept + slope)");
}

#[test]
fn lm_coefficient_count_p2() {
    // p=2 predictors → 3 coefficients (intercept + 2 slopes)
    let src = r#"
let x = [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0,
         6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 9.0, 10.0, 10.0, 11.0];
let y = [5.0, 8.0, 10.0, 13.0, 15.0, 18.0, 20.0, 23.0, 25.0, 28.0];
let result = lm(x, y, 10, 2);
let coeffs = result.coefficients;
print(len(coeffs));
"#;
    let out = eval_output(src);
    assert_eq!(out, vec!["3"], "p=2 → should have 3 coefficients");
}

#[test]
fn lm_r_squared_range() {
    // R² must be in [0, 1]
    let src = r#"
let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let y = [2.1, 4.0, 5.8, 8.1, 9.9, 12.2, 13.8, 16.1, 18.0, 20.1];
let result = lm(x, y, 10, 1);
print(result.r_squared);
"#;
    let out = eval_output(src);
    let r2: f64 = out[0].parse().expect("R² should be a float");
    assert!(r2 >= 0.0 && r2 <= 1.0, "R² = {} out of [0,1]", r2);
    // For near-perfect linear data, R² should be very high
    assert!(r2 > 0.99, "R² = {} should be > 0.99 for near-linear data", r2);
}

#[test]
fn lm_residuals_length() {
    // residuals count must equal n
    let src = r#"
let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let y = [2.1, 4.0, 5.8, 8.1, 9.9, 12.2, 13.8, 16.1, 18.0, 20.1];
let result = lm(x, y, 10, 1);
print(len(result.residuals));
"#;
    let out = eval_output(src);
    assert_eq!(out, vec!["10"], "residuals length should equal n=10");
}

#[test]
fn lm_requires_four_args() {
    // lm() with wrong arg count should error
    let src = "let r = lm([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 3);";
    let program = parse(src);
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_err(), "lm() with 3 args should fail");
}

#[test]
fn lm_rank_deficient() {
    // n <= p+1 should error (need n > p+1 for regression with intercept)
    let src = r#"
let x = [1.0, 2.0];
let y = [1.0, 2.0];
let result = lm(x, y, 2, 1);
"#;
    let program = parse(src);
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_err(), "n=2, p=1 means n <= p+1 → should error");
}

#[test]
fn lm_parity_eval_mir() {
    let src = r#"
let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let y = [2.1, 4.0, 5.8, 8.1, 9.9, 12.2, 13.8, 16.1, 18.0, 20.1];
let result = lm(x, y, 10, 1);
print(result.r_squared);
print(len(result.coefficients));
"#;
    let eval_out = eval_output(src);
    let mir_out = mir_output(src);
    assert_eq!(eval_out, mir_out, "lm parity: eval vs MIR must match");
}
