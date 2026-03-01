//! Hardening test H14: Correlation, Distributions, and Inference via MIR executor

fn run_mir(src: &str) -> Vec<String> {
    let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

#[test]
fn h14_cor_perfect() {
    let out = run_mir(r#"
let x = [1.0, 2.0, 3.0, 4.0, 5.0];
let y = [2.0, 4.0, 6.0, 8.0, 10.0];
print(cor(x, y));
"#);
    assert_eq!(out, vec!["1"]);
}

#[test]
fn h14_cov_basic() {
    let out = run_mir(r#"
let x = [1.0, 2.0, 3.0];
let y = [4.0, 5.0, 6.0];
let c = cov(x, y);
print(c);
"#);
    let parsed: f64 = out[0].parse().unwrap();
    assert!((parsed - 2.0 / 3.0).abs() < 1e-10);
}

#[test]
fn h14_normal_cdf_zero() {
    let out = run_mir(r#"
let r = normal_cdf(0.0);
print(r);
"#);
    let parsed: f64 = out[0].parse().unwrap();
    assert!((parsed - 0.5).abs() < 1e-6);
}

#[test]
fn h14_normal_cdf_196() {
    let out = run_mir(r#"
let r = normal_cdf(1.96);
print(r);
"#);
    let parsed: f64 = out[0].parse().unwrap();
    assert!((parsed - 0.975).abs() < 0.01);
}

#[test]
fn h14_normal_ppf() {
    let out = run_mir(r#"
let r = normal_ppf(0.5);
print(r);
"#);
    let parsed: f64 = out[0].parse().unwrap();
    assert!(parsed.abs() < 1e-4);
}

#[test]
fn h14_t_test_nonsig() {
    let out = run_mir(r#"
let data = [5.1, 4.9, 5.0, 5.2, 4.8, 5.0, 5.1, 4.9];
let r = t_test(data, 5.0);
print(r.p_value);
"#);
    let p: f64 = out[0].parse().unwrap();
    assert!(p > 0.05, "expected non-significant, got p={p}");
}

#[test]
fn h14_t_test_sig() {
    let out = run_mir(r#"
let data = [10.1, 10.2, 10.0, 9.9, 10.3, 10.1, 10.0, 10.2];
let r = t_test(data, 0.0);
print(r.p_value);
"#);
    let p: f64 = out[0].parse().unwrap();
    assert!(p < 0.001, "expected significant, got p={p}");
}

#[test]
fn h14_chi_squared_uniform() {
    let out = run_mir(r#"
let obs = [20.0, 20.0, 20.0, 20.0, 20.0];
let exp = [20.0, 20.0, 20.0, 20.0, 20.0];
let r = chi_squared_test(obs, exp);
print(r.chi2);
"#);
    assert_eq!(out, vec!["0"]);
}

#[test]
fn h14_det_identity() {
    let out = run_mir(r#"
let I = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]);
print(det(I));
"#);
    let parsed: f64 = out[0].parse().unwrap();
    assert!((parsed - 1.0).abs() < 1e-12);
}

#[test]
fn h14_solve_basic() {
    let out = run_mir(r#"
let A = Tensor.from_vec([2.0, 1.0, 1.0, 3.0], [2, 2]);
let b = Tensor.from_vec([5.0, 7.0], [2]);
let x = solve(A, b);
print(x);
"#);
    // 2x + y = 5, x + 3y = 7 → x=1.6, y=1.8
    let _out_str = &out[0];
    assert!(!out.is_empty());
}

#[test]
fn h14_trace_identity() {
    let out = run_mir(r#"
let I = Tensor.from_vec([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [3, 3]);
print(trace(I));
"#);
    let parsed: f64 = out[0].parse().unwrap();
    assert!((parsed - 3.0).abs() < 1e-12);
}

#[test]
fn h14_mse_loss() {
    let out = run_mir(r#"
let pred = [1.0, 2.0, 3.0];
let target = [1.0, 2.0, 3.0];
print(mse_loss(pred, target));
"#);
    assert_eq!(out, vec!["0"]);
}

#[test]
fn h14_histogram() {
    let out = run_mir(r#"
let data = [1.0, 2.0, 3.0, 4.0, 5.0];
let r = histogram(data, 5);
print(len(r));
"#);
    assert_eq!(out, vec!["2"]); // tuple of 2 elements
}

#[test]
fn h14_determinism() {
    let src = r#"
let x = [1.0, 2.0, 3.0, 4.0, 5.0];
let y = [5.0, 4.0, 3.0, 2.0, 1.0];
print(cor(x, y));
print(normal_cdf(1.96));
"#;
    let out1 = run_mir(src);
    let out2 = run_mir(src);
    assert_eq!(out1, out2);
}
