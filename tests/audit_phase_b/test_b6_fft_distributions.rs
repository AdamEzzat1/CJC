// ── Phase B6: Advanced FFT & Distributions ──────────────────────────
// Integration tests for windows, fft_arbitrary, fft_2d, and new distributions.

fn run_mir(src: &str) -> Vec<String> {
    let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

fn parse_float(s: &str) -> f64 {
    s.parse::<f64>().unwrap()
}

// ── Window functions ──

#[test]
fn b6_hann_length() {
    let out = run_mir(r#"
let w = hann(8);
print(len(w));
"#);
    assert_eq!(out, vec!["8"]);
}

#[test]
fn b6_hann_endpoints_zero() {
    let out = run_mir(r#"
let w = hann(8);
print(w);
"#);
    // hann[0] and hann[7] should be 0 (or very close)
    let inner = out[0].trim_matches(|c| c == '[' || c == ']');
    let vals: Vec<f64> = inner.split(", ").map(|s| s.parse::<f64>().unwrap()).collect();
    assert!(vals[0].abs() < 1e-10, "hann[0] = {}", vals[0]);
    assert!(vals[7].abs() < 1e-10, "hann[7] = {}", vals[7]);
}

#[test]
fn b6_hamming_endpoints() {
    let out = run_mir(r#"
let w = hamming(8);
print(w);
"#);
    let inner = out[0].trim_matches(|c| c == '[' || c == ']');
    let vals: Vec<f64> = inner.split(", ").map(|s| s.parse::<f64>().unwrap()).collect();
    assert!((vals[0] - 0.08).abs() < 1e-10, "hamming[0] = {}", vals[0]);
    assert!((vals[7] - 0.08).abs() < 1e-10, "hamming[7] = {}", vals[7]);
}

#[test]
fn b6_blackman_length() {
    let out = run_mir(r#"
let w = blackman(16);
print(len(w));
"#);
    assert_eq!(out, vec!["16"]);
}

// ── Distribution functions ──

#[test]
fn b6_beta_pdf_symmetric() {
    let out = run_mir(r#"
let r = beta_pdf(0.5, 2.0, 2.0);
print(r);
"#);
    let v = parse_float(&out[0]);
    assert!((v - 1.5).abs() < 1e-6, "beta_pdf(0.5, 2, 2) = {v}");
}

#[test]
fn b6_beta_cdf_uniform() {
    let out = run_mir(r#"
let r = beta_cdf(0.5, 1.0, 1.0);
print(r);
"#);
    let v = parse_float(&out[0]);
    assert!((v - 0.5).abs() < 1e-6, "beta_cdf(0.5, 1, 1) = {v}");
}

#[test]
fn b6_gamma_pdf_positive() {
    let out = run_mir(r#"
let r = gamma_pdf(1.0, 2.0, 1.0);
print(r);
"#);
    let v = parse_float(&out[0]);
    // gamma_pdf(1, k=2, theta=1) = 1 * exp(-1) = e^-1 ≈ 0.3679
    assert!((v - (-1.0_f64).exp()).abs() < 1e-6, "gamma_pdf = {v}");
}

#[test]
fn b6_gamma_cdf_basic() {
    let out = run_mir(r#"
let r = gamma_cdf(2.0, 1.0, 1.0);
print(r);
"#);
    // gamma_cdf(2, k=1, theta=1) = 1 - exp(-2) ≈ 0.8647
    let v = parse_float(&out[0]);
    assert!((v - (1.0 - (-2.0_f64).exp())).abs() < 1e-4, "gamma_cdf = {v}");
}

#[test]
fn b6_exp_cdf_memoryless() {
    let out = run_mir(r#"
let r = exp_cdf(0.5, 2.0);
print(r);
"#);
    // exp_cdf(0.5, lambda=2) = 1 - exp(-1) ≈ 0.6321
    let v = parse_float(&out[0]);
    let expected = 1.0 - (-1.0_f64).exp();
    assert!((v - expected).abs() < 1e-6, "exp_cdf = {v}");
}

#[test]
fn b6_exp_pdf_at_zero() {
    let out = run_mir(r#"
let r = exp_pdf(0.0, 3.0);
print(r);
"#);
    // exp_pdf(0, lambda=3) = 3
    let v = parse_float(&out[0]);
    assert!((v - 3.0).abs() < 1e-10, "exp_pdf(0, 3) = {v}");
}

#[test]
fn b6_weibull_cdf_basic() {
    let out = run_mir(r#"
let r = weibull_cdf(1.0, 1.0, 1.0);
print(r);
"#);
    // weibull_cdf(1, k=1, lambda=1) = 1 - exp(-1)
    let v = parse_float(&out[0]);
    let expected = 1.0 - (-1.0_f64).exp();
    assert!((v - expected).abs() < 1e-10, "weibull_cdf = {v}");
}

#[test]
fn b6_weibull_pdf_positive() {
    let out = run_mir(r#"
let r = weibull_pdf(1.0, 2.0, 1.0);
print(r);
"#);
    // weibull_pdf(1, k=2, lambda=1) = 2 * 1 * exp(-1) = 2/e
    let v = parse_float(&out[0]);
    let expected = 2.0 * (-1.0_f64).exp();
    assert!((v - expected).abs() < 1e-6, "weibull_pdf = {v}");
}

// ── Determinism ──

#[test]
fn b6_determinism() {
    let src = r#"
print(hann(5));
print(beta_pdf(0.3, 2.0, 5.0));
print(exp_cdf(1.0, 1.5));
"#;
    let out1 = run_mir(src);
    let out2 = run_mir(src);
    assert_eq!(out1, out2);
}

#[test]
fn b6_beta_cdf_endpoints() {
    let out = run_mir(r#"
print(beta_cdf(0.0, 2.0, 3.0));
print(beta_cdf(1.0, 2.0, 3.0));
"#);
    let v0 = parse_float(&out[0]);
    let v1 = parse_float(&out[1]);
    assert!(v0.abs() < 1e-10, "beta_cdf(0) = {v0}");
    assert!((v1 - 1.0).abs() < 1e-10, "beta_cdf(1) = {v1}");
}
