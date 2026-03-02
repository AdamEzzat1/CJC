// ── Phase B7: Non-parametric Tests & Multiple Comparisons ───────────
// Integration tests for mann_whitney, kruskal_wallis, wilcoxon_signed_rank,
// bonferroni, fdr_bh, tukey_hsd, logistic_regression.

fn run_mir(src: &str) -> Vec<String> {
    let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

fn parse_float(s: &str) -> f64 {
    s.parse::<f64>().unwrap()
}

// ── Mann-Whitney ──

#[test]
fn b7_mann_whitney_identical() {
    let out = run_mir(r#"
let r = mann_whitney([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]);
print(r.p_value);
"#);
    let p = parse_float(&out[0]);
    assert!(p > 0.05, "p = {p}");
}

#[test]
fn b7_mann_whitney_separated() {
    let out = run_mir(r#"
let r = mann_whitney([1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 11.0, 12.0, 13.0, 14.0]);
print(r.p_value);
"#);
    let p = parse_float(&out[0]);
    assert!(p < 0.05, "p = {p}");
}

// ── Kruskal-Wallis ──

#[test]
fn b7_kruskal_wallis_different() {
    let out = run_mir(r#"
let r = kruskal_wallis([1.0, 2.0, 3.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0]);
print(r.p_value);
"#);
    let p = parse_float(&out[0]);
    assert!(p < 0.05, "p = {p}");
}

#[test]
fn b7_kruskal_wallis_identical() {
    let out = run_mir(r#"
let r = kruskal_wallis([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]);
print(r.p_value);
"#);
    let p = parse_float(&out[0]);
    assert!(p > 0.05, "p = {p}");
}

// ── Wilcoxon signed-rank ──

#[test]
fn b7_wilcoxon_clear_shift() {
    let out = run_mir(r#"
let r = wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]);
print(r.p_value);
"#);
    let p = parse_float(&out[0]);
    assert!(p < 0.05, "p = {p}");
}

#[test]
fn b7_wilcoxon_no_difference() {
    let out = run_mir(r#"
let r = wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1.1, 1.9, 3.1, 3.9, 5.1, 5.9, 7.1, 7.9]);
print(r.p_value);
"#);
    let p = parse_float(&out[0]);
    assert!(p > 0.05, "p = {p}");
}

// ── Bonferroni ──

#[test]
fn b7_bonferroni_basic() {
    let out = run_mir(r#"
let adj = bonferroni([0.01, 0.04, 0.5]);
print(adj);
"#);
    assert_eq!(out, vec!["[0.03, 0.12, 1]"]);
}

// ── FDR BH ──

#[test]
fn b7_fdr_bh_basic() {
    let out = run_mir(r#"
let adj = fdr_bh([0.01, 0.04, 0.03, 0.5]);
print(adj);
"#);
    // Verify first adjusted p-value is still significant
    let inner = out[0].trim_matches(|c| c == '[' || c == ']');
    let vals: Vec<f64> = inner.split(", ").map(|s| s.parse::<f64>().unwrap()).collect();
    assert!(vals[0] < 0.05, "adj[0] = {}", vals[0]);
}

// ── Tukey HSD ──

#[test]
fn b7_tukey_hsd_one_different() {
    let out = run_mir(r#"
let results = tukey_hsd([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0], [20.0, 21.0, 22.0, 23.0, 24.0]);
print(len(results));
"#);
    // 3 groups -> 3 pairwise comparisons
    assert_eq!(out, vec!["3"]);
}

// ── Logistic regression ──

#[test]
fn b7_logistic_positive_coefficient() {
    let out = run_mir(r#"
let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let y = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
let r = logistic_regression(x, y, 10, 1);
let coeffs = r.coefficients;
print(coeffs);
"#);
    // Parse the array - coefficient for x should be positive
    let inner = out[0].trim_matches(|c| c == '[' || c == ']');
    let vals: Vec<f64> = inner.split(", ").map(|s| s.parse::<f64>().unwrap()).collect();
    assert!(vals[1] > 0.0, "beta_1 = {} should be positive", vals[1]);
}

#[test]
fn b7_logistic_aic() {
    let out = run_mir(r#"
let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let y = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
let r = logistic_regression(x, y, 10, 1);
print(r.aic);
"#);
    let aic = parse_float(&out[0]);
    assert!(aic > 0.0, "AIC = {aic}");
}

// ── Determinism ──

#[test]
fn b7_determinism() {
    let src = r#"
let r = mann_whitney([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
print(r.u_statistic);
print(r.p_value);
"#;
    let out1 = run_mir(src);
    let out2 = run_mir(src);
    assert_eq!(out1, out2);
}

#[test]
fn b7_fdr_preserves_order() {
    let out = run_mir(r#"
let adj = fdr_bh([0.5, 0.01, 0.3]);
print(adj);
"#);
    // Original p[1]=0.01 should have smallest adjusted value
    let inner = out[0].trim_matches(|c| c == '[' || c == ']');
    let vals: Vec<f64> = inner.split(", ").map(|s| s.parse::<f64>().unwrap()).collect();
    assert!(vals[1] < vals[0], "adj[1]={} should be < adj[0]={}", vals[1], vals[0]);
}
