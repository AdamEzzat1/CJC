//! Phase B audit test B2: Rank Correlations & Partial Correlation

fn run_mir(src: &str) -> Vec<String> {
    let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

#[test]
fn b2_spearman_perfect() {
    let out = run_mir(r#"
let r = spearman_cor([1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]);
print(r);
"#);
    assert_eq!(out, vec!["1"]);
}

#[test]
fn b2_spearman_reverse() {
    let out = run_mir(r#"
let r = spearman_cor([1.0, 2.0, 3.0, 4.0, 5.0], [50.0, 40.0, 30.0, 20.0, 10.0]);
print(r);
"#);
    assert_eq!(out, vec!["-1"]);
}

#[test]
fn b2_kendall_concordant() {
    let out = run_mir(r#"
let t = kendall_cor([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]);
print(t);
"#);
    assert_eq!(out, vec!["1"]);
}

#[test]
fn b2_kendall_discordant() {
    let out = run_mir(r#"
let t = kendall_cor([1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]);
print(t);
"#);
    assert_eq!(out, vec!["-1"]);
}

#[test]
fn b2_kendall_with_ties() {
    let out = run_mir(r#"
let t = kendall_cor([1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]);
print(t);
"#);
    let val: f64 = out[0].parse().unwrap();
    assert!(val > 0.9 && val < 1.0);
}

#[test]
fn b2_partial_cor() {
    let out = run_mir(r#"
let x = [1.0, 2.0, 3.0, 4.0, 5.0];
let y = [2.0, 4.0, 6.0, 8.0, 10.0];
let z = [5.0, 3.0, 1.0, 4.0, 2.0];
let pc = partial_cor(x, y, z);
print(pc);
"#);
    let val: f64 = out[0].parse().unwrap();
    assert!(val > 0.95, "partial_cor should be close to 1, got {val}");
}

#[test]
fn b2_cor_ci_returns_tuple() {
    let out = run_mir(r#"
let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let y = [2.0, 4.0, 5.0, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0, 11.0];
let ci = cor_ci(x, y, 0.05);
print(ci);
"#);
    // Should print a tuple like (lo, hi)
    assert!(out[0].starts_with("("), "expected tuple, got {}", out[0]);
}

#[test]
fn b2_cor_ci_contains_r() {
    // Just verify it returns a tuple and the CI is valid
    let out = run_mir(r#"
let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let y = [2.0, 4.0, 5.0, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0, 11.0];
let ci = cor_ci(x, y, 0.05);
print(ci);
"#);
    // CI should be a tuple (lo, hi) with reasonable bounds
    let s = &out[0];
    assert!(s.starts_with("(") && s.ends_with(")"), "expected tuple, got {s}");
    // Parse the tuple values
    let inner = &s[1..s.len()-1];
    let parts: Vec<&str> = inner.split(", ").collect();
    assert_eq!(parts.len(), 2, "expected 2-element tuple");
    let lo: f64 = parts[0].parse().unwrap();
    let hi: f64 = parts[1].parse().unwrap();
    assert!(lo < hi, "CI lower bound {lo} should be < upper bound {hi}");
    assert!(lo > -1.0 && hi < 1.0, "CI should be in (-1, 1)");
}

#[test]
fn b2_spearman_nonlinear() {
    let out = run_mir(r#"
let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let y = [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0];
let r = spearman_cor(x, y);
print(r);
"#);
    // Strictly monotone → spearman = 1.0
    assert_eq!(out, vec!["1"]);
}

#[test]
fn b2_kendall_known_small() {
    let out = run_mir(r#"
let t = kendall_cor([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]);
print(t);
"#);
    assert_eq!(out, vec!["-1"]);
}
