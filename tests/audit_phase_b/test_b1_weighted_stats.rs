//! Phase B audit test B1: Weighted & Robust Statistics

fn run_mir(src: &str) -> Vec<String> {
    let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

#[test]
fn b1_weighted_mean_uniform() {
    let out = run_mir(r#"
let data = [1.0, 2.0, 3.0, 4.0, 5.0];
let wm = weighted_mean(data, [1.0, 1.0, 1.0, 1.0, 1.0]);
print(wm);
"#);
    assert_eq!(out, vec!["3"]);
}

#[test]
fn b1_weighted_mean_skewed() {
    let out = run_mir(r#"
let wm = weighted_mean([1.0, 2.0, 3.0], [3.0, 0.0, 0.0]);
print(wm);
"#);
    assert_eq!(out, vec!["1"]);
}

#[test]
fn b1_weighted_var() {
    let out = run_mir(r#"
let data = [1.0, 2.0, 3.0];
let wv = weighted_var(data, [1.0, 1.0, 1.0]);
print(wv);
"#);
    // pop_variance of [1,2,3] = ((1-2)^2 + 0 + (3-2)^2) / 3 = 2/3 ≈ 0.6667
    let out_val: f64 = out[0].parse().unwrap();
    assert!((out_val - 2.0 / 3.0).abs() < 1e-10);
}

#[test]
fn b1_trimmed_mean() {
    let out = run_mir(r#"
let data = [-50.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0];
let tm = trimmed_mean(data, 0.1);
print(tm);
"#);
    // Sorted: [-50,2,3,4,5,6,7,8,9,100], trim 1 each → [2,3,4,5,6,7,8,9], mean=5.5
    let out_val: f64 = out[0].parse().unwrap();
    assert!((out_val - 5.5).abs() < 1e-10);
}

#[test]
fn b1_trimmed_mean_zero_proportion() {
    let out = run_mir(r#"
let tm = trimmed_mean([1.0, 2.0, 3.0, 4.0, 5.0], 0.0);
print(tm);
"#);
    assert_eq!(out, vec!["3"]);
}

#[test]
fn b1_winsorize() {
    let out = run_mir(r#"
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let w = winsorize(data, 0.2);
print(w);
"#);
    // Should clip extremes
    assert_eq!(out.len(), 1);
}

#[test]
fn b1_mad() {
    let out = run_mir(r#"
let m = mad([-2.0, -1.0, 0.0, 1.0, 2.0]);
print(m);
"#);
    assert_eq!(out, vec!["1"]);
}

#[test]
fn b1_mad_constant() {
    let out = run_mir(r#"
let m = mad([5.0, 5.0, 5.0]);
print(m);
"#);
    assert_eq!(out, vec!["0"]);
}

#[test]
fn b1_mode() {
    let out = run_mir(r#"
let m = mode([1.0, 2.0, 2.0, 3.0]);
print(m);
"#);
    assert_eq!(out, vec!["2"]);
}

#[test]
fn b1_mode_tie_smallest() {
    let out = run_mir(r#"
let m = mode([2.0, 1.0, 2.0, 1.0]);
print(m);
"#);
    assert_eq!(out, vec!["1"]);
}

#[test]
fn b1_percentile_rank_median() {
    let out = run_mir(r#"
let pr = percentile_rank([1.0, 2.0, 3.0, 4.0, 5.0], 3.0);
print(pr);
"#);
    assert_eq!(out, vec!["0.5"]);
}

#[test]
fn b1_percentile_rank_min() {
    let out = run_mir(r#"
let pr = percentile_rank([1.0, 2.0, 3.0, 4.0, 5.0], 1.0);
print(pr);
"#);
    assert_eq!(out, vec!["0.1"]);
}
