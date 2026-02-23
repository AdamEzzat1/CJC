// Milestone 2.6 — Exhaustiveness Checking Tests
//
// Tests for static exhaustiveness checking on match expressions over enum types.
// Uses the TypeChecker directly to verify that non-exhaustive matches are
// detected and exhaustive matches pass without errors.
//
// Error code: E0130 (non-exhaustive match)
//
// CJC uses bare variant names in patterns: `Some(x) =>`, `None =>`.
// Unit variant `None` is special-cased in eval, but the type checker
// understands Variant patterns (with parens) and Binding patterns.

use cjc_types::TypeChecker;

// ---------------------------------------------------------------------------
// Helper: parse source and run type checker, return whether E0130 was emitted
// ---------------------------------------------------------------------------

fn has_exhaustiveness_error(source: &str) -> bool {
    let lexer = cjc_lexer::Lexer::new(source);
    let (tokens, _) = lexer.tokenize();
    let parser = cjc_parser::Parser::new(tokens);
    let (program, _) = parser.parse_program();
    let mut checker = TypeChecker::new();
    checker.check_program(&program);
    checker
        .diagnostics
        .diagnostics
        .iter()
        .any(|d| d.code == "E0130")
}

// ---------------------------------------------------------------------------
// Tests: Non-exhaustive matches should produce errors
// ---------------------------------------------------------------------------

#[test]
fn exhaust_option_missing_none_arm() {
    // Match on Option covering only Some -- missing None
    // Since `None` without parens is parsed as a binding, this test covers
    // the case where only a Variant pattern `Some(v)` is present.
    let src = r#"
        fn main() -> i64 {
            let x = Some(1);
            match x {
                Some(v) => v,
            }
        }
    "#;
    assert!(
        has_exhaustiveness_error(src),
        "match with only Some arm should trigger E0130 (missing None)"
    );
}

#[test]
fn exhaust_user_enum_missing_variant() {
    // User-defined payload enum with 2 variants, match only covers 1
    let src = r#"
        enum Shape { Circle(f64), Rect(f64, f64) }
        fn main() -> f64 {
            let s = Circle(1.0);
            match s {
                Circle(r) => r,
            }
        }
    "#;
    assert!(
        has_exhaustiveness_error(src),
        "match missing Rect variant should trigger E0130"
    );
}

// ---------------------------------------------------------------------------
// Tests: Exhaustive matches should pass
// ---------------------------------------------------------------------------

#[test]
fn exhaust_all_payload_variants_covered() {
    // Both payload variants covered
    let src = r#"
        enum Shape { Circle(f64), Rect(f64, f64) }
        fn main() -> f64 {
            let s = Circle(1.0);
            match s {
                Circle(r) => r,
                Rect(w, h) => w + h,
            }
        }
    "#;
    assert!(
        !has_exhaustiveness_error(src),
        "all variants covered -- should NOT trigger E0130"
    );
}

#[test]
fn exhaust_wildcard_makes_exhaustive() {
    // One variant + wildcard -- should be exhaustive
    let src = r#"
        enum Shape { Circle(f64), Rect(f64, f64) }
        fn main() -> i64 {
            let s = Rect(3.0, 4.0);
            match s {
                Circle(r) => 1,
                _ => 99,
            }
        }
    "#;
    assert!(
        !has_exhaustiveness_error(src),
        "wildcard arm should make match exhaustive"
    );
}

#[test]
fn exhaust_binding_arm_makes_exhaustive() {
    // A bare binding pattern covers all remaining variants
    let src = r#"
        enum Shape { Circle(f64), Rect(f64, f64) }
        fn main() -> i64 {
            let s = Circle(1.0);
            match s {
                Circle(r) => 1,
                other => 42,
            }
        }
    "#;
    assert!(
        !has_exhaustiveness_error(src),
        "binding arm should make match exhaustive"
    );
}

#[test]
fn exhaust_option_both_arms_covered() {
    // Option with both Some and None covered
    let src = r#"
        fn main() -> i64 {
            let x = Some(1);
            match x {
                Some(v) => v,
                None => 0,
            }
        }
    "#;
    assert!(
        !has_exhaustiveness_error(src),
        "both Some and None covered -- should NOT trigger E0130"
    );
}
