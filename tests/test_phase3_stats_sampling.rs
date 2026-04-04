// CJC Test Suite — Phase 3: Statistical Tests, Effect Sizes, Sampling & CV
// Tests for normality tests, variance tests, effect sizes, and sampling builtins.

fn parse(src: &str) -> cjc_ast::Program {
    let (program, diags) = cjc_parser::parse_source(src);
    if diags.has_errors() {
        panic!("parse error:\n{}", diags.render_all(src, "<test>"));
    }
    program
}

fn run_eval(src: &str) -> Vec<String> {
    let program = parse(src);
    let mut interp = cjc_eval::Interpreter::new(42);
    let _ = interp.exec(&program).unwrap();
    interp.output.clone()
}

fn run_mir(src: &str) -> Vec<String> {
    let program = parse(src);
    let (_, exec) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    exec.output.clone()
}

fn run_parity(src: &str) -> Vec<String> {
    let eval_out = run_eval(src);
    let mir_out = run_mir(src);
    assert_eq!(eval_out, mir_out, "PARITY FAILURE:\n  eval: {:?}\n  mir:  {:?}", eval_out, mir_out);
    eval_out
}

// ════════════════════════════════════════════════════════════════════════
//  Normality Tests
// ════════════════════════════════════════════════���═══════════════════════

#[test]
fn test_jarque_bera_normal_data() {
    // Nearly normal data should have high p-value
    let out = run_parity(r#"
fn main() {
    let data = [-1.2, -0.8, -0.3, 0.1, 0.4, 0.7, 1.1, 1.5, -0.5, 0.2];
    let result = jarque_bera(data);
    print(result.statistic);
    print(result.p_value);
}
"#);
    let stat: f64 = out[0].parse().unwrap();
    let pval: f64 = out[1].parse().unwrap();
    assert!(stat >= 0.0, "JB statistic should be non-negative");
    assert!(pval > 0.05, "Normal-ish data should have p > 0.05, got {}", pval);
}

#[test]
fn test_jarque_bera_skewed_data() {
    // Highly skewed data should have low p-value
    let out = run_parity(r#"
fn main() {
    let data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 10.0, 20.0, 50.0, 100.0];
    let result = jarque_bera(data);
    print(result.p_value);
}
"#);
    let pval: f64 = out[0].parse().unwrap();
    assert!(pval < 0.05, "Skewed data should reject normality, got p={}", pval);
}

#[test]
fn test_anderson_darling() {
    let out = run_parity(r#"
fn main() {
    let data = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, -0.3, 0.3];
    let result = anderson_darling(data);
    print(result.statistic);
    print(result.p_value);
}
"#);
    let stat: f64 = out[0].parse().unwrap();
    assert!(stat >= 0.0, "AD statistic should be non-negative");
}

#[test]
fn test_ks_test() {
    let out = run_parity(r#"
fn main() {
    let data = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -0.8, 0.3, -0.2, 0.7];
    let result = ks_test(data);
    print(result.statistic);
    print(result.p_value);
}
"#);
    let stat: f64 = out[0].parse().unwrap();
    let pval: f64 = out[1].parse().unwrap();
    assert!(stat >= 0.0 && stat <= 1.0, "KS statistic should be in [0,1]");
    assert!(pval >= 0.0 && pval <= 1.0, "p-value should be in [0,1]");
}

// ════════════════════════════════════════════════════════════════════════
//  Effect Sizes
// ═���════════════════════════════════════════════════════��═════════════════

#[test]
fn test_cohens_d_large_effect() {
    let out = run_parity(r#"
fn main() {
    let x = [10.0, 11.0, 12.0, 13.0, 14.0];
    let y = [20.0, 21.0, 22.0, 23.0, 24.0];
    let d = cohens_d(x, y);
    print(d);
}
"#);
    let d: f64 = out[0].parse().unwrap();
    // Groups differ by ~10 with std ~1.58, so |d| should be large
    assert!(d.abs() > 2.0, "Cohen's d should indicate large effect, got {}", d);
}

#[test]
fn test_cohens_d_no_effect() {
    let out = run_parity(r#"
fn main() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [1.5, 2.5, 3.5, 4.5, 5.5];
    let d = cohens_d(x, y);
    print(d);
}
"#);
    let d: f64 = out[0].parse().unwrap();
    assert!(d.abs() < 0.5, "Similar groups should have small |d|, got {}", d);
}

#[test]
fn test_eta_squared() {
    let out = run_parity(r#"
fn main() {
    let g1 = [1.0, 2.0, 3.0];
    let g2 = [10.0, 11.0, 12.0];
    let g3 = [20.0, 21.0, 22.0];
    let es = eta_squared(g1, g2, g3);
    print(es);
}
"#);
    let es: f64 = out[0].parse().unwrap();
    assert!(es > 0.9, "Well-separated groups should have high eta², got {}", es);
}

#[test]
fn test_cramers_v() {
    let out = run_parity(r#"
fn main() {
    // 2x2 contingency table: [[10, 2], [3, 15]]
    let table = [10.0, 2.0, 3.0, 15.0];
    let v = cramers_v(table, 2, 2);
    print(v);
}
"#);
    let v: f64 = out[0].parse().unwrap();
    assert!(v > 0.3, "Strong association should have high V, got {}", v);
    assert!(v <= 1.0, "V should be <= 1.0");
}

// ══════════���════════════════════════════════════════════════════��════════
//  Variance Tests
// ════════════════════��═══════════════════════════════════════════════════

#[test]
fn test_levene_equal_variance() {
    let out = run_parity(r#"
fn main() {
    let g1 = [1.0, 2.0, 3.0, 4.0, 5.0];
    let g2 = [2.0, 3.0, 4.0, 5.0, 6.0];
    let result = levene_test(g1, g2);
    print(result.p_value);
}
"#);
    let pval: f64 = out[0].parse().unwrap();
    assert!(pval > 0.05, "Equal-variance groups should not reject, got p={}", pval);
}

#[test]
fn test_bartlett_test() {
    let out = run_parity(r#"
fn main() {
    let g1 = [1.0, 2.0, 3.0, 4.0, 5.0];
    let g2 = [10.0, 20.0, 30.0, 40.0, 50.0];
    let result = bartlett_test(g1, g2);
    print(result.statistic);
    print(result.p_value);
}
"#);
    let stat: f64 = out[0].parse().unwrap();
    let pval: f64 = out[1].parse().unwrap();
    assert!(stat > 0.0, "Bartlett statistic should be positive");
    assert!(pval < 0.05, "Very different variances should reject, got p={}", pval);
}

// ════��═════════════════════════════════════════════════════════���═════════
//  Sampling & Cross-Validation
// ═════════════════════��══════════════════════════════════════════════════

#[test]
fn test_train_test_split() {
    // train_test_split returns a tuple; print the whole thing
    let out = run_parity(r#"
fn main() {
    let result = train_test_split(100, 0.2, 42);
    print(result);
}
"#);
    // The output is a tuple of two arrays; check it contains arrays
    let s = &out[0];
    assert!(s.starts_with("("), "Should be a tuple: {}", s);
    // Count elements: 80 train + 20 test = 100 total
}

#[test]
fn test_train_test_split_determinism() {
    let src = r#"
fn main() {
    let result = train_test_split(50, 0.3, 123);
    print(result);
}
"#;
    let out1 = run_parity(src);
    let out2 = run_parity(src);
    assert_eq!(out1, out2, "train_test_split must be deterministic");
}

#[test]
fn test_kfold_indices() {
    let out = run_parity(r#"
fn main() {
    let folds = kfold_indices(10, 5, 42);
    print(len(folds));
}
"#);
    assert_eq!(out[0], "5");  // 5 folds
}

#[test]
fn test_kfold_determinism() {
    let src = r#"
fn main() {
    let folds = kfold_indices(20, 4, 99);
    print(folds);
}
"#;
    let out1 = run_parity(src);
    let out2 = run_parity(src);
    assert_eq!(out1, out2, "kfold_indices must be deterministic");
}

#[test]
fn test_latin_hypercube() {
    let out = run_parity(r#"
fn main() {
    let samples = latin_hypercube(5, 2, 42);
    print(samples);
}
"#);
    // Should produce a tensor with 10 values (5 x 2)
    assert!(!out[0].is_empty());
}

#[test]
fn test_sobol_sequence() {
    let out = run_parity(r#"
fn main() {
    let samples = sobol_sequence(4, 2);
    print(samples);
}
"#);
    assert!(!out[0].is_empty());
}

#[test]
fn test_latin_hypercube_determinism() {
    let src = r#"
fn main() {
    let s = latin_hypercube(10, 3, 42);
    print(s);
}
"#;
    let out1 = run_parity(src);
    let out2 = run_parity(src);
    assert_eq!(out1, out2, "latin_hypercube must be deterministic");
}

// ═══════���════════════════════════════════════════════════════════════════
//  Determinism Stress Tests
// ═════════════════════���═════════════════════════════��════════════════════

#[test]
fn test_all_stats_determinism() {
    let src = r#"
fn main() {
    let data = [-1.2, -0.8, -0.3, 0.1, 0.4, 0.7, 1.1, 1.5, -0.5, 0.2];
    let jb = jarque_bera(data);
    print(jb.statistic);
    let ad = anderson_darling(data);
    print(ad.statistic);
    let ks = ks_test(data);
    print(ks.statistic);
    let d = cohens_d(data, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    print(d);
}
"#;
    let out1 = run_parity(src);
    let out2 = run_parity(src);
    assert_eq!(out1, out2, "Statistical tests must be deterministic");
}

// ════════════════════════════════════════════════════════════════════════
//  Bootstrap & Permutation Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_bootstrap_mean() {
    let out = run_parity(r#"
fn main() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let result = bootstrap(data, 1000, 0, 42);
    print(result.point);
    print(result.ci_lower);
    print(result.ci_upper);
    print(result.se);
}
"#);
    let point: f64 = out[0].parse().unwrap();
    let ci_lower: f64 = out[1].parse().unwrap();
    let ci_upper: f64 = out[2].parse().unwrap();
    let se: f64 = out[3].parse().unwrap();
    assert!((point - 5.5).abs() < 0.01, "Bootstrap mean should be ~5.5, got {}", point);
    assert!(ci_lower < point, "CI lower should be below point estimate");
    assert!(ci_upper > point, "CI upper should be above point estimate");
    assert!(se > 0.0, "SE should be positive");
}

#[test]
fn test_bootstrap_median() {
    let out = run_parity(r#"
fn main() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let result = bootstrap(data, 500, 1, 99);
    print(result.point);
    print(result.se);
}
"#);
    let point: f64 = out[0].parse().unwrap();
    let se: f64 = out[1].parse().unwrap();
    assert!((point - 5.5).abs() < 0.01, "Bootstrap median should be ~5.5, got {}", point);
    assert!(se > 0.0, "SE should be positive");
}

#[test]
fn test_bootstrap_determinism() {
    let src = r#"
fn main() {
    let data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let result = bootstrap(data, 200, 0, 77);
    print(result);
}
"#;
    let out1 = run_parity(src);
    let out2 = run_parity(src);
    assert_eq!(out1, out2, "bootstrap must be deterministic");
}

#[test]
fn test_permutation_test_different_groups() {
    let out = run_parity(r#"
fn main() {
    let x = [10.0, 11.0, 12.0, 13.0, 14.0];
    let y = [20.0, 21.0, 22.0, 23.0, 24.0];
    let result = permutation_test(x, y, 1000, 42);
    print(result.observed_diff);
    print(result.p_value);
}
"#);
    let obs: f64 = out[0].parse().unwrap();
    let pval: f64 = out[1].parse().unwrap();
    assert!(obs > 5.0, "Observed diff should be ~10, got {}", obs);
    assert!(pval < 0.05, "Very different groups should have p < 0.05, got {}", pval);
}

#[test]
fn test_permutation_test_similar_groups() {
    let out = run_parity(r#"
fn main() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [1.5, 2.5, 3.5, 4.5, 5.5];
    let result = permutation_test(x, y, 500, 42);
    print(result.p_value);
}
"#);
    let pval: f64 = out[0].parse().unwrap();
    assert!(pval > 0.05, "Similar groups should not reject null, got p={}", pval);
}

#[test]
fn test_permutation_test_determinism() {
    let src = r#"
fn main() {
    let x = [1.0, 3.0, 5.0, 7.0];
    let y = [2.0, 4.0, 6.0, 8.0];
    let result = permutation_test(x, y, 200, 123);
    print(result);
}
"#;
    let out1 = run_parity(src);
    let out2 = run_parity(src);
    assert_eq!(out1, out2, "permutation_test must be deterministic");
}

// ════════════════════════════════════════════════════════════════════════
//  Stratified Sampling
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_stratified_split() {
    let out = run_parity(r#"
fn main() {
    let labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    let result = stratified_split(labels, 0.4, 42);
    print(result);
}
"#);
    let s = &out[0];
    assert!(s.starts_with("("), "Should be a tuple: {}", s);
}

#[test]
fn test_stratified_split_determinism() {
    let src = r#"
fn main() {
    let labels = [0, 0, 0, 1, 1, 1, 2, 2, 2];
    let result = stratified_split(labels, 0.3, 99);
    print(result);
}
"#;
    let out1 = run_parity(src);
    let out2 = run_parity(src);
    assert_eq!(out1, out2, "stratified_split must be deterministic");
}
