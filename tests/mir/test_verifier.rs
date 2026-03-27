//! Integration tests for the MIR legality verifier — end-to-end from CJC
//! source code.

use cjc_mir::verify::verify_mir_legality;

/// Parse CJC source → MIR, optionally build CFGs, run verifier.
fn verify_source(src: &str, build_cfgs: bool) -> cjc_mir::verify::LegalityReport {
    let (program, diags) = cjc_parser::parse_source(src);
    if diags.has_errors() {
        panic!("CJC parse failed for:\n{}\n", src);
    }
    let mut mir = cjc_mir_exec::lower_to_mir(&program);
    if build_cfgs {
        mir.build_all_cfgs();
    }
    verify_mir_legality(&mir)
}

// ---------------------------------------------------------------------------
// Test 1: Simple program passes verification
// ---------------------------------------------------------------------------
#[test]
fn test_verify_simple_program() {
    let src = "let x: i64 = 42;\nlet y: i64 = x + 1;\n";
    let report = verify_source(src, false);
    assert!(report.is_ok(), "simple program should pass: {:?}", report.errors());
}

// ---------------------------------------------------------------------------
// Test 2: Program with while loop passes (no CFG)
// ---------------------------------------------------------------------------
#[test]
fn test_verify_while_loop_no_cfg() {
    let src = r#"
fn count(n: i64) -> i64 {
    let mut i: i64 = 0;
    while i < n {
        i = i + 1;
    }
    return i;
}
print(count(10));
"#;
    let report = verify_source(src, false);
    assert!(report.is_ok(), "while loop should pass: {:?}", report.errors());
}

// ---------------------------------------------------------------------------
// Test 3: Program with while loop passes (with CFG)
// ---------------------------------------------------------------------------
#[test]
fn test_verify_while_loop_with_cfg() {
    let src = r#"
fn count(n: i64) -> i64 {
    let mut i: i64 = 0;
    while i < n {
        i = i + 1;
    }
    return i;
}
print(count(10));
"#;
    let report = verify_source(src, true);
    assert!(
        report.is_ok(),
        "while loop with CFG should pass: {:?}",
        report.errors()
    );
    assert!(report.checks_total > 0);
    assert_eq!(report.checks_passed, report.checks_total);
}

// ---------------------------------------------------------------------------
// Test 4: Nested control flow passes
// ---------------------------------------------------------------------------
#[test]
fn test_verify_nested_control_flow() {
    let src = r#"
fn complex(n: i64) -> i64 {
    let mut result: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        if i > 5 {
            let mut j: i64 = 0;
            while j < i {
                result = result + j;
                j = j + 1;
            }
        } else {
            result = result + 1;
        }
        i = i + 1;
    }
    return result;
}
print(complex(10));
"#;
    let report = verify_source(src, true);
    assert!(
        report.is_ok(),
        "nested control flow should pass: {:?}",
        report.errors()
    );
}

// ---------------------------------------------------------------------------
// Test 5: Multiple functions pass
// ---------------------------------------------------------------------------
#[test]
fn test_verify_multiple_functions() {
    let src = r#"
fn add(a: i64, b: i64) -> i64 { a + b }
fn mul(a: i64, b: i64) -> i64 { a * b }
let x: i64 = add(3, 4);
let y: i64 = mul(x, 2);
print(y);
"#;
    let report = verify_source(src, true);
    assert!(report.is_ok(), "multiple functions should pass: {:?}", report.errors());
}

// ---------------------------------------------------------------------------
// Test 6: Checks count is correct
// ---------------------------------------------------------------------------
#[test]
fn test_verify_checks_count() {
    let src = r#"
fn f(x: i64) -> i64 { x + 1 }
let result: i64 = f(42);
print(result);
"#;
    let report = verify_source(src, true);
    assert!(report.is_ok());
    // Should have at least: structural bounds for each function + reduction check
    assert!(
        report.checks_total >= 3,
        "should run multiple checks, got {}",
        report.checks_total
    );
}

// ---------------------------------------------------------------------------
// Test 7: Determinism — same source produces identical report
// ---------------------------------------------------------------------------
#[test]
fn test_verify_determinism() {
    let src = r#"
fn compute(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        total = total + i;
        i = i + 1;
    }
    return total;
}
print(compute(5));
"#;
    let report1 = verify_source(src, true);
    let report2 = verify_source(src, true);

    assert_eq!(report1.is_ok(), report2.is_ok());
    assert_eq!(report1.checks_total, report2.checks_total);
    assert_eq!(report1.checks_passed, report2.checks_passed);
    assert_eq!(report1.errors.len(), report2.errors.len());
}

// ---------------------------------------------------------------------------
// Test 8: Program with closures passes
// ---------------------------------------------------------------------------
#[test]
fn test_verify_closure_program() {
    let src = r#"
fn apply(f: Any, x: i64) -> i64 { f(x) }
let double = |x: i64| x * 2;
let result: i64 = apply(double, 21);
print(result);
"#;
    let report = verify_source(src, false);
    assert!(report.is_ok(), "closure program should pass: {:?}", report.errors());
}
