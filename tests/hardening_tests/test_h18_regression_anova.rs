//! Hardening test H18: Regression (lm), ANOVA, F-test, and QoL builtins via MIR executor

fn run_mir(src: &str) -> Vec<String> {
    let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

#[test]
fn h18_sample_variance() {
    let out = run_mir(r#"
let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
print(sample_variance(data));
"#);
    let parsed: f64 = out[0].parse().unwrap();
    assert!((parsed - 4.571).abs() < 0.01, "sample variance should be ~4.571, got {parsed}");
}

#[test]
fn h18_sample_sd() {
    let out = run_mir(r#"
let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
print(sample_sd(data));
"#);
    let parsed: f64 = out[0].parse().unwrap();
    assert!((parsed - 2.138).abs() < 0.01, "sample sd should be ~2.138, got {parsed}");
}

#[test]
fn h18_sample_cov() {
    let out = run_mir(r#"
let x = [1.0, 2.0, 3.0];
let y = [4.0, 5.0, 6.0];
print(sample_cov(x, y));
"#);
    let parsed: f64 = out[0].parse().unwrap();
    // sample_cov with n-1 denominator = 1.0
    assert!((parsed - 1.0).abs() < 1e-10, "sample_cov should be 1.0, got {parsed}");
}

#[test]
fn h18_row_number() {
    let out = run_mir(r#"
let data = [30.0, 10.0, 20.0];
let rn = row_number(data);
print(rn[0]);
print(rn[1]);
print(rn[2]);
"#);
    // row_number sorts by value, returns positions
    // 10.0 → rank 1, 20.0 → rank 2, 30.0 → rank 3
    let r0: f64 = out[0].parse().unwrap();
    let r1: f64 = out[1].parse().unwrap();
    let r2: f64 = out[2].parse().unwrap();
    assert!((r0 - 3.0).abs() < 1e-10, "30.0 should be rank 3");
    assert!((r1 - 1.0).abs() < 1e-10, "10.0 should be rank 1");
    assert!((r2 - 2.0).abs() < 1e-10, "20.0 should be rank 2");
}

#[test]
fn h18_t_test_paired() {
    let out = run_mir(r#"
let before = [200.0, 190.0, 210.0, 205.0, 195.0];
let after = [190.0, 185.0, 195.0, 200.0, 190.0];
let r = t_test_paired(before, after);
print(r.p_value);
"#);
    let p: f64 = out[0].parse().unwrap();
    assert!(p > 0.0 && p < 1.0, "p-value should be between 0 and 1, got {p}");
}

#[test]
fn h18_f_test() {
    let out = run_mir(r#"
let x = [1.0, 2.0, 3.0, 4.0, 5.0];
let y = [10.0, 20.0, 30.0, 40.0, 50.0];
let r = f_test(x, y);
print(r.f_statistic);
print(r.p_value);
"#);
    let f_stat: f64 = out[0].parse().unwrap();
    let p_val: f64 = out[1].parse().unwrap();
    assert!(f_stat > 0.0, "F statistic should be positive, got {f_stat}");
    assert!(p_val > 0.0 && p_val <= 1.0, "p-value should be in (0,1], got {p_val}");
}

#[test]
fn h18_anova_similar_groups() {
    // Groups with same mean but some within-group variance
    let out = run_mir(r#"
let g1 = [4.9, 5.0, 5.1, 5.0];
let g2 = [4.8, 5.0, 5.2, 5.0];
let g3 = [4.7, 5.0, 5.3, 5.0];
let r = anova_oneway(g1, g2, g3);
print(r.p_value);
"#);
    let p: f64 = out[0].parse().unwrap();
    assert!(p > 0.05, "similar groups should have p > 0.05, got {p}");
}

#[test]
fn h18_anova_different_groups() {
    let out = run_mir(r#"
let g1 = [1.0, 2.0, 3.0];
let g2 = [10.0, 11.0, 12.0];
let g3 = [100.0, 101.0, 102.0];
let r = anova_oneway(g1, g2, g3);
print(r.p_value);
"#);
    let p: f64 = out[0].parse().unwrap();
    assert!(p < 0.001, "very different groups should have p < 0.001, got {p}");
}

#[test]
fn h18_lm_simple() {
    // Simple linear regression: y = 2x + 1, lm auto-adds intercept
    let out = run_mir(r#"
let X = [1.0, 2.0, 3.0, 4.0, 5.0];
let y = [3.0, 5.0, 7.0, 9.0, 11.0];
let r = lm(X, y, 5, 1);
print(r.r_squared);
"#);
    let r2: f64 = out[0].parse().unwrap();
    assert!((r2 - 1.0).abs() < 1e-10, "perfect linear fit should have R²=1, got {r2}");
}

#[test]
fn h18_lm_coefficients() {
    // y = 2x + 1 → coefficients should be [intercept=1, slope=2]
    // lm(X, y, n, p) auto-adds intercept, so pass just the predictor
    let out = run_mir(r#"
let X = [1.0, 2.0, 3.0, 4.0, 5.0];
let y = [3.0, 5.0, 7.0, 9.0, 11.0];
let r = lm(X, y, 5, 1);
let c = r.coefficients;
print(c[0]);
print(c[1]);
"#);
    let c0: f64 = out[0].parse().unwrap();
    let c1: f64 = out[1].parse().unwrap();
    assert!((c0 - 1.0).abs() < 1e-6, "intercept should be ~1, got {c0}");
    assert!((c1 - 2.0).abs() < 1e-6, "slope should be ~2, got {c1}");
}

#[test]
fn h18_cumsum_cummax() {
    let out = run_mir(r#"
let data = [1.0, 3.0, 2.0, 5.0];
let cs = cumsum(data);
print(cs[3]);
let cm = cummax(data);
print(cm[2]);
"#);
    let cs3: f64 = out[0].parse().unwrap();
    let cm2: f64 = out[1].parse().unwrap();
    assert!((cs3 - 11.0).abs() < 1e-10, "cumsum[3] should be 11");
    assert!((cm2 - 3.0).abs() < 1e-10, "cummax[2] should be 3");
}

#[test]
fn h18_determinism() {
    let src = r#"
let x = [1.0, 2.0, 3.0, 4.0, 5.0];
let y = [10.0, 20.0, 30.0, 40.0, 50.0];
let r = f_test(x, y);
print(r.f_statistic);
print(sample_variance(x));
let rn = row_number(x);
print(rn[0]);
"#;
    let out1 = run_mir(src);
    let out2 = run_mir(src);
    assert_eq!(out1, out2);
}
