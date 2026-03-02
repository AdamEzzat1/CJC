// ── Phase B5: Analyst QoL Extensions ─────────────────────────────────
// Integration tests for case_when, ntile, percent_rank, cume_dist, wls.

fn run_mir(src: &str) -> Vec<String> {
    let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

// ── case_when tests ──

#[test]
fn test_case_when_first_true() {
    let out = run_mir(r#"
let r = case_when([true, false], ["a", "b"], "default");
print(r);
"#);
    assert_eq!(out, vec!["a"]);
}

#[test]
fn test_case_when_second_true() {
    let out = run_mir(r#"
let r = case_when([false, true], ["a", "b"], "default");
print(r);
"#);
    assert_eq!(out, vec!["b"]);
}

#[test]
fn test_case_when_none_true() {
    let out = run_mir(r#"
let r = case_when([false, false], [1, 2], 99);
print(r);
"#);
    assert_eq!(out, vec!["99"]);
}

#[test]
fn test_case_when_all_true() {
    let out = run_mir(r#"
let r = case_when([true, true, true], [10, 20, 30], 0);
print(r);
"#);
    assert_eq!(out, vec!["10"]);
}

// ── ntile tests ──

#[test]
fn test_ntile_quartiles() {
    let out = run_mir(r#"
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let groups = ntile(data, 4);
print(groups);
"#);
    assert_eq!(out, vec!["[1, 1, 2, 2, 3, 3, 4, 4]"]);
}

#[test]
fn test_ntile_all_same_group() {
    let out = run_mir(r#"
let groups = ntile([10.0, 20.0, 30.0], 1);
print(groups);
"#);
    assert_eq!(out, vec!["[1, 1, 1]"]);
}

// ── percent_rank tests ──

#[test]
fn test_percent_rank_sorted() {
    let out = run_mir(r#"
let pr = percent_rank([1.0, 2.0, 3.0, 4.0, 5.0]);
print(pr);
"#);
    assert_eq!(out, vec!["[0, 0.25, 0.5, 0.75, 1]"]);
}

#[test]
fn test_percent_rank_single() {
    let out = run_mir(r#"
let pr = percent_rank([42.0]);
print(pr);
"#);
    assert_eq!(out, vec!["[0]"]);
}

// ── cume_dist tests ──

#[test]
fn test_cume_dist_basic() {
    let out = run_mir(r#"
let cd = cume_dist([1.0, 2.0, 3.0, 4.0, 5.0]);
print(cd);
"#);
    assert_eq!(out, vec!["[0.2, 0.4, 0.6, 0.8, 1]"]);
}

#[test]
fn test_cume_dist_ties() {
    let out = run_mir(r#"
let cd = cume_dist([1.0, 2.0, 2.0, 4.0]);
print(cd);
"#);
    assert_eq!(out, vec!["[0.25, 0.75, 0.75, 1]"]);
}

// ── wls tests ──

#[test]
fn test_wls_uniform_weights() {
    let out = run_mir(r#"
let x = [1.0, 2.0, 3.0, 4.0, 5.0];
let y = [3.0, 5.0, 7.0, 9.0, 11.0];
let w = [1.0, 1.0, 1.0, 1.0, 1.0];
let r = wls(x, y, w, 5, 1);
let coeffs = r.coefficients;
print(coeffs);
"#);
    // y = 2*x + 1 -> coefficients should be [1, 2]
    let coeffs_str = &out[0];
    let inner = coeffs_str.trim_matches(|c| c == '[' || c == ']');
    let vals: Vec<f64> = inner.split(", ").map(|s| s.parse::<f64>().unwrap()).collect();
    assert!((vals[0] - 1.0).abs() < 1e-6, "intercept = {}", vals[0]);
    assert!((vals[1] - 2.0).abs() < 1e-6, "slope = {}", vals[1]);
}

#[test]
fn test_wls_r_squared() {
    let out = run_mir(r#"
let x = [1.0, 2.0, 3.0, 4.0, 5.0];
let y = [3.0, 5.0, 7.0, 9.0, 11.0];
let w = [1.0, 1.0, 1.0, 1.0, 1.0];
let r = wls(x, y, w, 5, 1);
print(r.r_squared);
"#);
    let r2: f64 = out[0].parse().unwrap();
    assert!((r2 - 1.0).abs() < 1e-6, "R^2 = {}", r2);
}

// ── determinism ──

#[test]
fn test_b5_determinism() {
    let src = r#"
let data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
print(ntile(data, 4));
print(percent_rank(data));
print(cume_dist(data));
"#;
    let out1 = run_mir(src);
    let out2 = run_mir(src);
    assert_eq!(out1, out2);
}
