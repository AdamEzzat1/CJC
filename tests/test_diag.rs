// CJC Test Suite — cjc-diag (3 tests)
// Source: crates/cjc-diag/src/lib.rs
// These tests are extracted from the inline #[cfg(test)] modules for regression tracking.

use cjc_diag::*;

#[test]
fn test_span_merge() {
    let a = Span::new(5, 10);
    let b = Span::new(8, 15);
    let merged = a.merge(b);
    assert_eq!(merged.start, 5);
    assert_eq!(merged.end, 15);
}

#[test]
fn test_diagnostic_render() {
    let source = "let x = 42 +;\n";
    let diag = Diagnostic::error("E0001", "unexpected token", Span::new(13, 14))
        .with_label(Span::new(13, 14), "expected expression")
        .with_hint("remove the trailing `+` or add an expression after it");

    let renderer = DiagnosticRenderer::new(source, "test.cjc");
    let output = renderer.render(&diag);

    assert!(output.contains("error[E0001]"));
    assert!(output.contains("unexpected token"));
    assert!(output.contains("test.cjc:1:14"));
    assert!(output.contains("expected expression"));
    assert!(output.contains("hint:"));
}

#[test]
fn test_diagnostic_bag() {
    let mut bag = DiagnosticBag::new();
    assert!(!bag.has_errors());

    bag.emit(Diagnostic::error("E0001", "test error", Span::new(0, 1)));
    assert!(bag.has_errors());
    assert_eq!(bag.error_count(), 1);

    bag.emit(Diagnostic::warning("W0001", "test warning", Span::new(0, 1)));
    assert_eq!(bag.error_count(), 1);
}
