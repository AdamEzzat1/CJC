//! Hardening test H11: Window function integration tests.
//!
//! Tests window_sum, window_mean, window_min, window_max through the
//! MIR executor pipeline. Verifies correctness, edge cases, and
//! determinism of the sliding-window builtins.

/// Helper: parse + MIR-execute a CJC program, return executor output.
fn run_mir(src: &str) -> Vec<String> {
    let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

// ── Basic window function tests ─────────────────────────────────────

#[test]
fn h11_window_sum_basic() {
    let out = run_mir(r#"
let data = [1.0, 2.0, 3.0, 4.0, 5.0];
let result = window_sum(data, 3);
print(len(result));
print(result);
"#);
    assert_eq!(out[0], "3");
    // result should be [6.0, 9.0, 12.0]
    assert!(out[1].contains("6"));
    assert!(out[1].contains("9"));
    assert!(out[1].contains("12"));
}

#[test]
fn h11_window_mean_basic() {
    let out = run_mir(r#"
let data = [1.0, 2.0, 3.0, 4.0, 5.0];
let result = window_mean(data, 3);
print(len(result));
print(result);
"#);
    assert_eq!(out[0], "3");
    // result should be [2.0, 3.0, 4.0]
    assert!(out[1].contains("2"));
    assert!(out[1].contains("3"));
    assert!(out[1].contains("4"));
}

#[test]
fn h11_window_min_basic() {
    let out = run_mir(r#"
let data = [3.0, 1.0, 4.0, 1.0, 5.0];
let result = window_min(data, 3);
print(len(result));
print(result);
"#);
    assert_eq!(out[0], "3");
    // result should be [1.0, 1.0, 1.0]
    assert!(out[1].contains("1"));
}

#[test]
fn h11_window_max_basic() {
    let out = run_mir(r#"
let data = [3.0, 1.0, 4.0, 1.0, 5.0];
let result = window_max(data, 3);
print(len(result));
print(result);
"#);
    assert_eq!(out[0], "3");
    // result should be [4.0, 4.0, 5.0]
    assert!(out[1].contains("4"));
    assert!(out[1].contains("5"));
}

// ── Edge case tests ─────────────────────────────────────────────────

#[test]
fn h11_window_size_equals_data() {
    let out = run_mir(r#"
let data = [1.0, 2.0, 3.0];
let result = window_sum(data, 3);
print(len(result));
"#);
    assert_eq!(out, vec!["1"]);
}

#[test]
fn h11_window_size_one() {
    let out = run_mir(r#"
let data = [10.0, 20.0, 30.0];
let result = window_sum(data, 1);
print(len(result));
"#);
    assert_eq!(out, vec!["3"]);
}

// ── Determinism tests ───────────────────────────────────────────────

#[test]
fn h11_window_determinism() {
    let src = r#"
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let result = window_sum(data, 4);
print(result);
"#;
    let out1 = run_mir(src);
    let out2 = run_mir(src);
    assert_eq!(out1, out2, "window_sum must be deterministic");
}

// ── Effect registry test ────────────────────────────────────────────

#[test]
fn h11_effect_registry_window_classified() {
    use cjc_types::effect_registry;
    assert!(effect_registry::lookup("window_sum").is_some());
    assert!(effect_registry::lookup("window_mean").is_some());
    assert!(effect_registry::lookup("window_min").is_some());
    assert!(effect_registry::lookup("window_max").is_some());
}
