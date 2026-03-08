//! Integration tests for Bastion primitive ABI builtins.
//! Validates that nth_element, median_fast, quantile_fast, filter_mask,
//! sample_indices, erf, and erfc are wired through the full CJC pipeline.

fn parse(src: &str) -> cjc_ast::Program {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    program
}

fn mir_output(src: &str) -> Vec<String> {
    let program = parse(src);
    let (_, exec) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    exec.output.clone()
}

fn mir_output_seed(src: &str, seed: u64) -> Vec<String> {
    let program = parse(src);
    let (_, exec) = cjc_mir_exec::run_program_with_executor(&program, seed).unwrap();
    exec.output.clone()
}

// ── nth_element ─────────────────────────────────────────────────

#[test]
fn bastion_nth_element() {
    let out = mir_output(r#"
let data = [5.0, 1.0, 3.0, 4.0, 2.0];
let result = nth_element(data, 2);
print(result);
"#);
    assert_eq!(out[0], "3", "nth_element([5,1,3,4,2], 2) should be 3 (0-indexed)");
}

#[test]
fn bastion_nth_element_min() {
    let out = mir_output(r#"
let data = [10.0, 3.0, 7.0, 1.0];
print(nth_element(data, 0));
"#);
    assert_eq!(out[0], "1", "nth_element(_, 0) should return the minimum");
}

// ── median_fast ─────────────────────────────────────────────────

#[test]
fn bastion_median_fast_odd() {
    let out = mir_output(r#"
let data = [3.0, 1.0, 2.0, 5.0, 4.0];
print(median_fast(data));
"#);
    assert_eq!(out[0], "3", "median of [3,1,2,5,4] = 3");
}

#[test]
fn bastion_median_fast_even() {
    let out = mir_output(r#"
let data = [4.0, 1.0, 3.0, 2.0];
print(median_fast(data));
"#);
    assert_eq!(out[0], "2.5", "median of [4,1,3,2] = 2.5");
}

// ── quantile_fast ───────────────────────────────────────────────

#[test]
fn bastion_quantile_fast() {
    let out = mir_output(r#"
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let q50 = quantile_fast(data, 0.5);
print(q50);
"#);
    assert_eq!(out[0], "5.5", "quantile_fast at p=0.5 of 1..10 = 5.5");
}

// ── filter_mask ─────────────────────────────────────────────────

#[test]
fn bastion_filter_mask() {
    let out = mir_output(r#"
let data = [10.0, 20.0, 30.0, 40.0, 50.0];
let mask = [true, false, true, false, true];
let result = filter_mask(data, mask);
print(result);
"#);
    assert!(out[0].contains("10") && out[0].contains("30") && out[0].contains("50"),
        "filter_mask should keep elements at true positions: {}", out[0]);
}

// ── erf / erfc ──────────────────────────────────────────────────

#[test]
fn bastion_erf_zero() {
    let out = mir_output(r#"
print(erf(0.0));
"#);
    assert_eq!(out[0], "0", "erf(0) = 0 exactly");
}

#[test]
fn bastion_erf_one() {
    let out = mir_output(r#"
let val = erf(1.0);
print(val);
"#);
    // erf(1) ≈ 0.8427007929497149
    let v: f64 = out[0].parse().unwrap();
    assert!((v - 0.8427).abs() < 0.001, "erf(1) ≈ 0.8427, got {}", v);
}

#[test]
fn bastion_erfc_zero() {
    let out = mir_output(r#"
print(erfc(0.0));
"#);
    assert_eq!(out[0], "1", "erfc(0) = 1 exactly");
}

#[test]
fn bastion_erf_erfc_sum() {
    let out = mir_output(r#"
let x = 1.5;
let s = erf(x) + erfc(x);
print(s);
"#);
    assert_eq!(out[0], "1", "erf(x) + erfc(x) = 1 for all x");
}

// ── sample_indices ──────────────────────────────────────────────

#[test]
fn bastion_sample_indices_without_replacement() {
    let out = mir_output(r#"
let idx = sample_indices(10, 5, false, 42);
print(len(idx));
"#);
    assert_eq!(out[0], "5", "sample_indices(10,5,false) should return 5 indices");
}

#[test]
fn bastion_sample_indices_with_replacement() {
    let out = mir_output(r#"
let idx = sample_indices(5, 10, true, 42);
print(len(idx));
"#);
    assert_eq!(out[0], "10", "sample_indices(5,10,true) should return 10 indices");
}

#[test]
fn bastion_sample_indices_determinism() {
    let src = r#"
let idx = sample_indices(100, 10, false, 99);
print(idx);
"#;
    let out1 = mir_output_seed(src, 1);
    let out2 = mir_output_seed(src, 1);
    assert_eq!(out1, out2, "sample_indices with same explicit seed must be deterministic");
}

// ── Parity: median_fast vs median ───────────────────────────────

#[test]
fn bastion_median_fast_parity() {
    let out = mir_output(r#"
let data = [7.0, 2.0, 9.0, 4.0, 5.0, 1.0, 8.0];
let m1 = median(data);
let m2 = median_fast(data);
print(m1);
print(m2);
"#);
    assert_eq!(out[0], out[1], "median and median_fast must agree: {} vs {}", out[0], out[1]);
}
