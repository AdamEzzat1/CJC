//! Hardening test H13: Descriptive Statistics via MIR executor

fn run_mir(src: &str) -> Vec<String> {
    let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

#[test]
fn h13_variance() {
    // variance uses sample (N-1) denominator (R/pandas default)
    let out = run_mir(r#"
let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
let v = variance(data);
print(v);
"#);
    // mean=5, sum_sq_dev=32, sample_var=32/7≈4.571
    let parsed: f64 = out[0].parse().unwrap();
    assert!((parsed - 32.0 / 7.0).abs() < 1e-10, "expected sample variance 32/7, got {parsed}");
}

#[test]
fn h13_sd() {
    let out = run_mir(r#"
let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
let s = sd(data);
print(s);
"#);
    let parsed: f64 = out[0].parse().unwrap();
    let expected = (32.0_f64 / 7.0).sqrt();
    assert!((parsed - expected).abs() < 1e-10, "expected sample sd sqrt(32/7), got {parsed}");
}

#[test]
fn h13_median_odd() {
    let out = run_mir(r#"
let data = [5.0, 1.0, 3.0];
print(median(data));
"#);
    assert_eq!(out, vec!["3"]);
}

#[test]
fn h13_median_even() {
    let out = run_mir(r#"
let data = [4.0, 1.0, 2.0, 3.0];
print(median(data));
"#);
    assert_eq!(out, vec!["2.5"]);
}

#[test]
fn h13_quantile() {
    let out = run_mir(r#"
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
print(quantile(data, 0.5));
"#);
    assert_eq!(out, vec!["5.5"]);
}

#[test]
fn h13_iqr() {
    let out = run_mir(r#"
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let i = iqr(data);
print(i);
"#);
    // IQR for 1..10 = Q3-Q1
    let parsed: f64 = out[0].parse().unwrap();
    assert!(parsed > 0.0);
}

#[test]
fn h13_z_score() {
    let out = run_mir(r#"
let data = [10.0, 20.0, 30.0];
let z = z_score(data);
print(len(z));
"#);
    assert_eq!(out, vec!["3"]);
}

#[test]
fn h13_standardize() {
    let out = run_mir(r#"
let data = [0.0, 5.0, 10.0];
let s = standardize(data);
print(s);
"#);
    assert_eq!(out, vec!["[0, 0.5, 1]"]);
}

#[test]
fn h13_cumsum() {
    let out = run_mir(r#"
let data = [1.0, 2.0, 3.0, 4.0];
print(cumsum(data));
"#);
    assert_eq!(out, vec!["[1, 3, 6, 10]"]);
}

#[test]
fn h13_n_distinct() {
    let out = run_mir(r#"
let data = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0];
print(n_distinct(data));
"#);
    assert_eq!(out, vec!["3"]);
}

#[test]
fn h13_determinism() {
    let src = r#"
let data = [1.1, 2.2, 3.3, 4.4, 5.5];
print(variance(data));
print(sd(data));
print(median(data));
"#;
    let out1 = run_mir(src);
    let out2 = run_mir(src);
    assert_eq!(out1, out2, "determinism: two runs must produce identical output");
}
