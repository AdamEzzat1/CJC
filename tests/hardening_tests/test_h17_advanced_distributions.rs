//! Hardening test H17: Advanced Distributions, FFT, and Discrete Distributions via MIR executor

fn run_mir(src: &str) -> Vec<String> {
    let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

#[test]
fn h17_t_ppf_median() {
    let out = run_mir(r#"
let r = t_ppf(0.5, 10.0);
print(r);
"#);
    let parsed: f64 = out[0].parse().unwrap();
    assert!(parsed.abs() < 1e-4, "t_ppf(0.5, 10) should be ~0, got {parsed}");
}

#[test]
fn h17_chi2_ppf_median() {
    let out = run_mir(r#"
let r = chi2_ppf(0.5, 1.0);
print(r);
"#);
    let parsed: f64 = out[0].parse().unwrap();
    // chi2 median with df=1 is ~0.455
    assert!(parsed > 0.3 && parsed < 0.6, "chi2_ppf(0.5, 1) should be ~0.455, got {parsed}");
}

#[test]
fn h17_f_ppf_basic() {
    let out = run_mir(r#"
let r = f_ppf(0.95, 5.0, 10.0);
print(r);
"#);
    let parsed: f64 = out[0].parse().unwrap();
    // F critical value at 0.95 with df1=5, df2=10 should be ~3.33
    assert!(parsed > 2.5 && parsed < 4.5, "f_ppf(0.95, 5, 10) should be ~3.33, got {parsed}");
}

#[test]
fn h17_binomial_pmf() {
    let out = run_mir(r#"
let r = binomial_pmf(5, 10, 0.5);
print(r);
"#);
    let parsed: f64 = out[0].parse().unwrap();
    // C(10,5) * 0.5^10 ≈ 0.2461
    assert!((parsed - 0.2461).abs() < 0.01, "binomial_pmf(5,10,0.5) should be ~0.246, got {parsed}");
}

#[test]
fn h17_binomial_cdf() {
    let out = run_mir(r#"
let r = binomial_cdf(5, 10, 0.5);
print(r);
"#);
    let parsed: f64 = out[0].parse().unwrap();
    // P(X <= 5) for Bin(10, 0.5) ≈ 0.623
    assert!((parsed - 0.623).abs() < 0.01, "binomial_cdf(5,10,0.5) should be ~0.623, got {parsed}");
}

#[test]
fn h17_poisson_pmf() {
    let out = run_mir(r#"
let r = poisson_pmf(0, 1.0);
print(r);
"#);
    let parsed: f64 = out[0].parse().unwrap();
    // P(X=0) for Poisson(1) = e^(-1) ≈ 0.368
    assert!((parsed - 0.3679).abs() < 0.01, "poisson_pmf(0,1) should be ~0.368, got {parsed}");
}

#[test]
fn h17_poisson_cdf() {
    let out = run_mir(r#"
let r = poisson_cdf(2, 1.0);
print(r);
"#);
    let parsed: f64 = out[0].parse().unwrap();
    // P(X <= 2) for Poisson(1) = e^(-1)(1 + 1 + 1/2) ≈ 0.920
    assert!((parsed - 0.920).abs() < 0.01, "poisson_cdf(2,1) should be ~0.920, got {parsed}");
}

#[test]
fn h17_rfft_impulse() {
    let out = run_mir(r#"
let data = [1.0, 0.0, 0.0, 0.0];
let spectrum = rfft(data);
print(len(spectrum));
"#);
    assert_eq!(out, vec!["4"]);
}

#[test]
fn h17_psd_impulse() {
    let out = run_mir(r#"
let data = [1.0, 0.0, 0.0, 0.0];
let power = psd(data);
let first = power[0];
print(first);
"#);
    let parsed: f64 = out[0].parse().unwrap();
    assert!((parsed - 1.0).abs() < 1e-10, "PSD of impulse should be 1.0, got {parsed}");
}

#[test]
fn h17_normal_pdf() {
    let out = run_mir(r#"
let r = normal_pdf(0.0);
print(r);
"#);
    let parsed: f64 = out[0].parse().unwrap();
    // 1/sqrt(2*pi) ≈ 0.3989
    assert!((parsed - 0.3989).abs() < 0.01, "normal_pdf(0) should be ~0.399, got {parsed}");
}

#[test]
fn h17_t_cdf_symmetry() {
    let out = run_mir(r#"
let r1 = t_cdf(1.0, 10.0);
let r2 = t_cdf(-1.0, 10.0);
let sum = r1 + r2;
print(sum);
"#);
    let parsed: f64 = out[0].parse().unwrap();
    assert!((parsed - 1.0).abs() < 1e-6, "t_cdf symmetry: t_cdf(1,10) + t_cdf(-1,10) should be 1.0, got {parsed}");
}

#[test]
fn h17_determinism() {
    let src = r#"
print(binomial_pmf(3, 10, 0.5));
print(poisson_pmf(2, 3.0));
print(t_ppf(0.975, 20.0));
let data = [1.0, 0.0, 0.0, 0.0];
let p = psd(data);
print(p[0]);
"#;
    let out1 = run_mir(src);
    let out2 = run_mir(src);
    assert_eq!(out1, out2);
}
