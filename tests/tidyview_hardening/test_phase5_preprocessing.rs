//! Phase 5: Preprocessing builtins tests
//!
//! Tests fillna, is_not_null, interpolate_linear, coalesce, cut, qcut,
//! min_max_scale, robust_scale — including eval/MIR parity.

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn run_eval(src: &str, seed: u64) -> Vec<String> {
    let (program, diag) = cjc_parser::parse_source(src);
    if !diag.diagnostics.is_empty() {
        panic!("Parse errors: {:?}", diag.diagnostics);
    }
    let mut interp = cjc_eval::Interpreter::new(seed);
    match interp.exec(&program) {
        Ok(_) => {}
        Err(e) => panic!("Eval error: {e}"),
    }
    interp.output
}

fn run_mir(src: &str, seed: u64) -> Vec<String> {
    let (program, diag) = cjc_parser::parse_source(src);
    if !diag.diagnostics.is_empty() {
        panic!("MIR parse errors: {:?}", diag.diagnostics);
    }
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, seed)
        .unwrap_or_else(|e| panic!("MIR-exec failed: {e}"));
    executor.output
}

fn assert_parity(src: &str) {
    let eval_out = run_eval(src, 42);
    let mir_out = run_mir(src, 42);
    assert_eq!(
        eval_out, mir_out,
        "Parity failure:\n  eval: {:?}\n  mir:  {:?}",
        eval_out, mir_out
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 5.1  fillna
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fillna_replaces_nan_in_float_array() {
    let src = r#"
let nan = NAN_VAL();
let arr = [1.0, nan, 3.0, nan, 5.0];
let filled = fillna(arr, 0.0);
print(filled);
"#;
    let out = run_eval(src, 42);
    assert_eq!(out.len(), 1);
    assert!(out[0].contains("1"));
    assert!(!out[0].to_lowercase().contains("nan"));
    assert_parity(src);
}

#[test]
fn fillna_replaces_void_in_mixed_array() {
    // Test via Rust API — Void can't be created in CJC array syntax.
    use std::rc::Rc;
    use cjc_runtime::Value;

    let arr = vec![
        Value::Float(1.0),
        Value::Void,
        Value::Float(3.0),
        Value::Void,
    ];
    let args = vec![
        Value::Array(Rc::new(arr)),
        Value::Float(99.0),
    ];
    let result = cjc_runtime::builtins::dispatch_builtin("fillna", &args)
        .expect("fillna failed")
        .expect("fillna returned None");
    match result {
        Value::Array(a) => {
            assert_eq!(a.len(), 4);
            match (&a[0], &a[1], &a[2], &a[3]) {
                (Value::Float(v0), Value::Float(v1), Value::Float(v2), Value::Float(v3)) => {
                    assert!((v0 - 1.0).abs() < 1e-10);
                    assert!((v1 - 99.0).abs() < 1e-10);
                    assert!((v2 - 3.0).abs() < 1e-10);
                    assert!((v3 - 99.0).abs() < 1e-10);
                }
                _ => panic!("expected all Float values"),
            }
        }
        _ => panic!("expected Array"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 5.2  is_not_null
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn is_not_null_returns_correct_booleans() {
    let src = r#"
print(is_not_null(42));
print(is_not_null(3.14));
print(is_not_null("hello"));
let nan = NAN_VAL();
print(is_not_null(nan));
"#;
    let out = run_eval(src, 42);
    assert_eq!(out[0], "true");
    assert_eq!(out[1], "true");
    assert_eq!(out[2], "true");
    assert_eq!(out[3], "false");
    assert_parity(src);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5.3  interpolate_linear
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn interpolate_linear_fills_interior_nan() {
    let src = r#"
let nan = NAN_VAL();
let arr = [1.0, nan, 3.0];
let result = interpolate_linear(arr);
print(result);
"#;
    let out = run_eval(src, 42);
    assert_eq!(out.len(), 1);
    // Interior NaN between 1.0 and 3.0 should become 2.0
    assert!(out[0].contains("2"));
    assert!(!out[0].to_lowercase().contains("nan"));
    assert_parity(src);
}

#[test]
fn interpolate_linear_handles_edge_nan() {
    let src = r#"
let nan = NAN_VAL();
let arr = [nan, 2.0, nan, 4.0, nan];
let result = interpolate_linear(arr);
print(result);
"#;
    let out = run_eval(src, 42);
    assert_eq!(out.len(), 1);
    // Leading NaN -> backward-fill to 2.0
    // Interior NaN between 2.0 and 4.0 -> 3.0
    // Trailing NaN -> forward-fill to 4.0
    assert!(out[0].contains("2"));
    assert!(out[0].contains("3"));
    assert!(out[0].contains("4"));
    assert!(!out[0].to_lowercase().contains("nan"));
    assert_parity(src);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5.4  coalesce
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn coalesce_picks_first_non_null() {
    let src = r#"
let nan = NAN_VAL();
let a = [1.0, nan, 3.0, nan];
let b = [10.0, 20.0, 30.0, 40.0];
let result = coalesce(a, b);
print(result);
"#;
    let out = run_eval(src, 42);
    assert_eq!(out.len(), 1);
    // Position 0: a=1.0 (non-NaN) => 1.0
    // Position 1: a=NaN => b's 20.0
    // Position 2: a=3.0 => 3.0
    // Position 3: a=NaN => b's 40.0
    assert!(out[0].contains("1"));
    assert!(out[0].contains("20"));
    assert!(out[0].contains("3"));
    assert!(out[0].contains("40"));
    assert_parity(src);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5.5  cut
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cut_with_known_breaks() {
    let src = r#"
let data = [5.0, 15.0, 25.0, 35.0];
let breaks = [10.0, 20.0, 30.0];
let result = cut(data, breaks);
print(result);
"#;
    let out = run_eval(src, 42);
    assert_eq!(out.len(), 1);
    assert!(out[0].contains("(-inf,10]"));
    assert!(out[0].contains("(10,20]"));
    assert!(out[0].contains("(20,30]"));
    assert!(out[0].contains("(30,inf)"));
    assert_parity(src);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5.6  qcut
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn qcut_with_4_bins() {
    let src = r#"
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let result = qcut(data, 4);
print(result);
"#;
    let out = run_eval(src, 42);
    assert_eq!(out.len(), 1);
    // Should produce quartile bin labels
    assert!(!out[0].is_empty());
    assert_parity(src);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5.7  min_max_scale
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn min_max_scale_maps_to_0_1() {
    let src = r#"
let data = [10.0, 20.0, 30.0, 40.0, 50.0];
let result = min_max_scale(data, 0.0, 1.0);
print(result);
"#;
    let out = run_eval(src, 42);
    assert_eq!(out.len(), 1);
    // min=10, max=50 => 10->0, 20->0.25, 30->0.5, 40->0.75, 50->1
    assert!(out[0].contains("0.25"));
    assert!(out[0].contains("0.5"));
    assert!(out[0].contains("0.75"));
    assert_parity(src);
}

#[test]
fn min_max_scale_custom_range() {
    // Rust API test for custom range [2, 10]
    use std::rc::Rc;
    use cjc_runtime::Value;

    let data = vec![
        Value::Float(0.0),
        Value::Float(5.0),
        Value::Float(10.0),
    ];
    let args = vec![
        Value::Array(Rc::new(data)),
        Value::Float(2.0),
        Value::Float(10.0),
    ];
    let result = cjc_runtime::builtins::dispatch_builtin("min_max_scale", &args)
        .expect("min_max_scale failed")
        .expect("min_max_scale returned None");
    match result {
        Value::Array(a) => {
            assert_eq!(a.len(), 3);
            match (&a[0], &a[1], &a[2]) {
                (Value::Float(a0), Value::Float(a1), Value::Float(a2)) => {
                    assert!((a0 - 2.0).abs() < 1e-10);
                    assert!((a1 - 6.0).abs() < 1e-10);
                    assert!((a2 - 10.0).abs() < 1e-10);
                }
                _ => panic!("expected floats"),
            }
        }
        _ => panic!("expected Array"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 5.8  robust_scale
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn robust_scale_median_centered() {
    let src = r#"
let data = [1.0, 2.0, 3.0, 4.0, 5.0];
let result = robust_scale(data);
print(result);
"#;
    let out = run_eval(src, 42);
    assert_eq!(out.len(), 1);
    // median=3, IQR=Q3-Q1=4-2=2
    // (1-3)/2=-1, (2-3)/2=-0.5, (3-3)/2=0, (4-3)/2=0.5, (5-3)/2=1
    assert!(out[0].contains("-1"));
    assert!(out[0].contains("-0.5"));
    assert!(out[0].contains("0.5"));
    assert_parity(src);
}

// ═══════════════════════════════════════════════════════════════════════════
// Parity: comprehensive test running all builtins through both executors
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn parity_all_preprocessing_builtins() {
    // fillna
    assert_parity(r#"
let nan = NAN_VAL();
let a = [1.0, nan, 3.0];
print(fillna(a, 0.0));
"#);
    // is_not_null
    assert_parity(r#"
print(is_not_null(1.0));
let nan = NAN_VAL();
print(is_not_null(nan));
"#);
    // interpolate_linear
    assert_parity(r#"
let nan = NAN_VAL();
let a = [1.0, nan, 3.0];
print(interpolate_linear(a));
"#);
    // coalesce
    assert_parity(r#"
let nan = NAN_VAL();
let a = [nan, 2.0];
let b = [10.0, 20.0];
print(coalesce(a, b));
"#);
    // cut
    assert_parity(r#"
print(cut([5.0, 15.0], [10.0]));
"#);
    // qcut
    assert_parity(r#"
print(qcut([1.0, 2.0, 3.0, 4.0], 2));
"#);
    // min_max_scale
    assert_parity(r#"
print(min_max_scale([0.0, 5.0, 10.0], 0.0, 1.0));
"#);
    // robust_scale
    assert_parity(r#"
print(robust_scale([1.0, 2.0, 3.0, 4.0, 5.0]));
"#);
}
