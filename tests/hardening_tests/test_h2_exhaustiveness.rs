//! H-2: Match exhaustiveness is a compile-time error (not just a warning).
//!
//! The type checker emits `Diagnostic::error("E0130", ...)` for non-exhaustive
//! enum matches, and `type_check_program()` / `run_program_type_checked()`
//! surface these as `MirExecError::TypeErrors`.

use cjc_parser::parse_source;
use cjc_types::TypeChecker;

/// Test 1: Exhaustive enum match — no E0130 error.
#[test]
fn test_exhaustive_enum_match_no_error() {
    let src = r#"
enum Dir { North, South, East, West, }
fn go(d: Dir) -> i64 {
    match d {
        North => 1,
        South => 2,
        East  => 3,
        West  => 4,
    }
}
fn main() -> i64 { 0 }
"#;
    let (prog, _) = parse_source(src);
    let mut checker = TypeChecker::new();
    checker.check_program(&prog);
    let exhaustiveness_errors: Vec<_> = checker
        .diagnostics
        .diagnostics
        .iter()
        .filter(|d| d.code == "E0130")
        .collect();
    assert!(
        exhaustiveness_errors.is_empty(),
        "exhaustive match should not emit E0130, got: {:?}",
        exhaustiveness_errors
    );
}

/// Test 2: Non-exhaustive enum match (missing variant) → E0130 error.
#[test]
fn test_nonexhaustive_enum_match_emits_e0130_error() {
    let src = r#"
enum Color { Red, Green, Blue, }
fn name(c: Color) -> i64 {
    match c {
        Red   => 1,
        Green => 2,
        // Blue is missing!
    }
}
fn main() -> i64 { 0 }
"#;
    let (prog, _) = parse_source(src);
    let mut checker = TypeChecker::new();
    checker.check_program(&prog);
    assert!(
        checker.diagnostics.has_errors(),
        "non-exhaustive enum match should produce a type error"
    );
    let e0130 = checker
        .diagnostics
        .diagnostics
        .iter()
        .any(|d| d.code == "E0130");
    assert!(e0130, "non-exhaustive match error code should be E0130");
}

/// Test 3: Wildcard arm suppresses E0130.
#[test]
fn test_wildcard_suppresses_exhaustiveness_error() {
    let src = r#"
enum Status { Ok, Err, Pending, }
fn check(s: Status) -> i64 {
    match s {
        Ok => 1,
        _  => 0,
    }
}
fn main() -> i64 { 0 }
"#;
    let (prog, _) = parse_source(src);
    let mut checker = TypeChecker::new();
    checker.check_program(&prog);
    let e0130 = checker
        .diagnostics
        .diagnostics
        .iter()
        .any(|d| d.code == "E0130");
    assert!(!e0130, "wildcard should suppress E0130");
}

/// Test 4: `run_program_type_checked` rejects non-exhaustive enum match.
#[test]
fn test_run_type_checked_rejects_nonexhaustive_match() {
    use cjc_mir_exec::{run_program_type_checked, MirExecError};
    let src = r#"
enum Shape { Circle, Square, }
fn area(s: Shape) -> i64 {
    match s {
        Circle => 1,
        // Square missing
    }
}
fn main() -> i64 { 0 }
"#;
    let (prog, _) = parse_source(src);
    let result = run_program_type_checked(&prog, 0);
    match result {
        Err(MirExecError::TypeErrors(errs)) => {
            assert!(
                !errs.is_empty(),
                "should have at least one type error for non-exhaustive match"
            );
            // At least one error should mention E0130 or 'non-exhaustive'
            let mentions_exhaustiveness = errs
                .iter()
                .any(|e| e.contains("E0130") || e.contains("non-exhaustive"));
            assert!(
                mentions_exhaustiveness,
                "error should mention E0130 or non-exhaustive, got: {:?}",
                errs
            );
        }
        Ok(_) => panic!("non-exhaustive match should have been rejected"),
        Err(other) => panic!("unexpected error type: {:?}", other),
    }
}

/// Test 5: `type_check_program` returns Ok for exhaustive match.
#[test]
fn test_type_check_program_ok_for_exhaustive_match() {
    use cjc_mir_exec::type_check_program;
    let src = r#"
enum Coin { Heads, Tails, }
fn flip(c: Coin) -> i64 {
    match c {
        Heads => 1,
        Tails => 0,
    }
}
fn main() -> i64 { 0 }
"#;
    let (prog, _) = parse_source(src);
    // Should succeed (no errors)
    let result = type_check_program(&prog);
    assert!(
        result.is_ok(),
        "exhaustive match should pass type_check_program, got: {:?}",
        result
    );
}

/// Test 6: Non-exhaustive match span is non-zero (span plumbing works for E0130).
#[test]
fn test_nonexhaustive_match_error_has_span() {
    let src = r#"
enum Bit { Zero, One, }
fn flip(b: Bit) -> i64 {
    match b { Zero => 0, }
}
fn main() -> i64 { 0 }
"#;
    let (prog, _) = parse_source(src);
    let mut checker = TypeChecker::new();
    checker.check_program(&prog);
    let e0130 = checker
        .diagnostics
        .diagnostics
        .iter()
        .find(|d| d.code == "E0130");
    assert!(e0130.is_some(), "E0130 should be emitted");
    let err = e0130.unwrap();
    // The span end should be >= start (basic sanity)
    assert!(
        err.span.end >= err.span.start,
        "E0130 span should be valid: start={} end={}",
        err.span.start,
        err.span.end
    );
}

/// Test 7: `type_check_program` returns TypeErrors for non-exhaustive match.
#[test]
fn test_type_check_program_returns_type_errors_for_nonexhaustive() {
    use cjc_mir_exec::{type_check_program, MirExecError};
    let src = r#"
enum Light { Red, Yellow, Green, }
fn duration(l: Light) -> i64 {
    match l {
        Red    => 30,
        Yellow => 5,
        // Green missing
    }
}
fn main() -> i64 { 0 }
"#;
    let (prog, _) = parse_source(src);
    let result = type_check_program(&prog);
    assert!(
        result.is_err(),
        "type_check_program should return Err for non-exhaustive match"
    );
    match result {
        Err(MirExecError::TypeErrors(errs)) => {
            assert!(!errs.is_empty());
        }
        _ => panic!("expected TypeErrors"),
    }
}
