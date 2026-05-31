//! End-to-end tests for the `locke_*` builtins through both executors.
//!
//! Each test runs the same CJC-Lang snippet under `cjc-eval` (AST tree-walk)
//! and `cjc-mir-exec` (register machine) and asserts byte-identical printed
//! output. Matches the precedent set by `tests/physics_ml/grad_graph_wiring.rs`.

#![allow(clippy::needless_raw_string_hashes)]

#[derive(Clone, Copy, Debug)]
enum Backend {
    Eval,
    Mir,
}

fn run(backend: Backend, body: &str, seed: u64) -> Vec<String> {
    let src = format!("fn main() {{\n{body}\n}}\n");
    let (program, diags) = cjc_parser::parse_source(&src);
    assert!(
        !diags.has_errors(),
        "parse errors:\n{:#?}\nsource:\n{src}",
        diags.diagnostics,
    );
    match backend {
        Backend::Eval => {
            let mut interp = cjc_eval::Interpreter::new(seed);
            interp
                .exec(&program)
                .unwrap_or_else(|e| panic!("eval failed for snippet:\n{src}\nerror: {e:?}"));
            interp.output
        }
        Backend::Mir => {
            let (_v, exec) = cjc_mir_exec::run_program_with_executor(&program, seed)
                .unwrap_or_else(|e| panic!("MIR-exec failed for snippet:\n{src}\nerror: {e:?}"));
            exec.output
        }
    }
}

fn assert_parity(label: &str, body: &str) -> Vec<String> {
    let eval_out = run(Backend::Eval, body, 42);
    let mir_out = run(Backend::Mir, body, 42);
    assert_eq!(
        eval_out, mir_out,
        "[{label}] cjc-eval and cjc-mir-exec disagree.\neval={eval_out:#?}\nmir={mir_out:#?}",
    );
    eval_out
}

#[test]
fn locke_missing_count_through_both_executors() {
    let body = r#"
        let xs = [1.0, 0.0/0.0, 3.0, 0.0/0.0, 5.0];
        let n = locke_missing_count(xs);
        print(n);
    "#;
    let out = assert_parity("missing_count", body);
    assert_eq!(out, vec!["2".to_string()]);
}

#[test]
fn locke_ks_d_disjoint_supports_through_both_executors() {
    let body = r#"
        let train = [1.0, 2.0, 3.0, 4.0, 5.0];
        let test = [100.0, 200.0, 300.0, 400.0, 500.0];
        let d = locke_ks_d(train, test);
        print(d);
    "#;
    let out = assert_parity("ks_d_disjoint", body);
    // KS D of disjoint supports = 1.0
    assert_eq!(out.len(), 1);
    let d: f64 = out[0].parse().expect("KS D should parse as f64");
    assert!((d - 1.0).abs() < 1e-12, "expected 1.0, got {}", d);
}

#[test]
fn locke_psi_identical_distributions_is_zero_through_both_executors() {
    let body = r#"
        let p = [0.25, 0.25, 0.25, 0.25];
        let q = [0.25, 0.25, 0.25, 0.25];
        let s = locke_psi(p, q);
        print(s);
    "#;
    let out = assert_parity("psi_identical", body);
    assert_eq!(out.len(), 1);
    let s: f64 = out[0].parse().expect("PSI should parse as f64");
    assert!(s.abs() < 1e-9, "expected ~0, got {}", s);
}

#[test]
fn locke_sample_score_matches_curve_through_both_executors() {
    let body = r#"
        let s = locke_sample_score(30);
        print(s);
    "#;
    let out = assert_parity("sample_score", body);
    assert_eq!(out.len(), 1);
    let s: f64 = out[0].parse().expect("sample score should parse as f64");
    assert!((s - 0.5).abs() < 0.005, "expected ~0.5, got {}", s);
}

#[test]
fn locke_belief_overall_averages_eight_sub_scores_through_both_executors() {
    let body = r#"
        let s = locke_belief_overall(1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        print(s);
    "#;
    let out = assert_parity("belief_overall", body);
    assert_eq!(out.len(), 1);
    let s: f64 = out[0].parse().expect("belief overall should parse as f64");
    assert!((s - 0.5).abs() < 1e-12, "expected 0.5, got {}", s);
}

#[test]
fn locke_table_handle_validate_through_both_executors() {
    // The headline v0.3 capability: build a DataFrame from .cjcl source,
    // run validate on it, inspect the report. No Value::DataFrame
    // required — handle-based registry.
    let body = r#"
        let h = locke_table_new();
        locke_table_add_float_col(h, "x", [1.0, 0.0/0.0, 3.0, 0.0/0.0, 5.0]);
        let n_rows = locke_table_nrows(h);
        let report = locke_validate(h);
        let n_findings = locke_report_n_findings(report);
        let worst = locke_report_worst_severity(report);
        print(n_rows);
        print(n_findings);
        print(worst);
    "#;
    let out = assert_parity("handle_validate", body);
    assert_eq!(out.len(), 3);
    // 5 rows, at least one finding (E9001 for NaN), worst >= 2 (Warning) because 40% NaN.
    assert_eq!(out[0], "5");
    let n_findings: i64 = out[1].parse().expect("n_findings is int");
    assert!(n_findings >= 1);
    let worst: i64 = out[2].parse().expect("worst is int");
    assert!(worst >= 2, "expected Warning or higher (>=2), got {}", worst);
}

#[test]
fn locke_table_handle_drift_through_both_executors() {
    let body = r#"
        let train = locke_table_new();
        locke_table_add_float_col(train, "x", [1.0, 2.0, 3.0, 4.0, 5.0]);
        let test = locke_table_new();
        locke_table_add_float_col(test, "x", [100.0, 200.0, 300.0, 400.0, 500.0]);
        let report = locke_drift(train, test);
        let worst = locke_drift_worst_severity(report);
        print(worst);
    "#;
    let out = assert_parity("handle_drift", body);
    assert_eq!(out, vec!["3".to_string()]);  // Error severity, disjoint supports
}

#[test]
fn locke_table_overall_score_via_handle() {
    let body = r#"
        let h = locke_table_new();
        locke_table_add_float_col(h, "x", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let report = locke_validate(h);
        let s = locke_report_overall_score(report);
        print(s);
    "#;
    let out = assert_parity("handle_overall_score", body);
    assert_eq!(out.len(), 1);
    let s: f64 = out[0].parse().expect("overall is float");
    assert!(s >= 0.0 && s <= 1.0);
}

#[test]
fn locke_missing_rate_through_both_executors() {
    let body = r#"
        let xs = [0.0/0.0, 0.0/0.0, 0.0/0.0, 1.0];
        let r = locke_missing_rate(xs);
        print(r);
    "#;
    let out = assert_parity("missing_rate", body);
    assert_eq!(out.len(), 1);
    let r: f64 = out[0].parse().expect("missing rate should parse as f64");
    assert!((r - 0.75).abs() < 1e-12, "expected 0.75, got {}", r);
}
