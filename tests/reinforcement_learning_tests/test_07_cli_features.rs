/// CLI feature tests — test color diagnostics at the library level.
/// (REPL tests require interactive stdin which we can't do in unit tests,
/// so we test the diagnostic renderer directly.)

#[test]
fn color_diagnostic_contains_ansi() {
    use cjc_diag::{Diagnostic, DiagnosticRenderer, Span};

    let source = "let x = 42 +;\n";
    let diag = Diagnostic::error("E0001", "unexpected token", Span::new(13, 14))
        .with_label(Span::new(13, 14), "expected expression")
        .with_hint("remove the trailing `+` or add an expression after it");

    let renderer = DiagnosticRenderer::new_with_color(source, "test.cjcl", true);
    let output = renderer.render(&diag);

    // Should contain ANSI escape codes
    assert!(output.contains("\x1b["), "color output should contain ANSI escape codes");
    assert!(output.contains("error"), "should contain 'error'");
    assert!(output.contains("E0001"), "should contain error code");
}

#[test]
fn plain_diagnostic_no_ansi() {
    use cjc_diag::{Diagnostic, DiagnosticRenderer, Span};

    let source = "let x = 42 +;\n";
    let diag = Diagnostic::error("E0001", "unexpected token", Span::new(13, 14));

    let renderer = DiagnosticRenderer::new(source, "test.cjcl");
    let output = renderer.render(&diag);

    // Should NOT contain ANSI escape codes
    assert!(!output.contains("\x1b["), "plain output should not contain ANSI escape codes");
    assert!(output.contains("error[E0001]"), "should contain error header");
}

#[test]
fn color_render_all_has_ansi() {
    use cjc_diag::{Diagnostic, DiagnosticBag, Span};

    let mut bag = DiagnosticBag::new();
    bag.emit(Diagnostic::error("E0001", "test error", Span::new(0, 1)));

    let source = "x";
    let output = bag.render_all_color(source, "test.cjcl", true);
    assert!(output.contains("\x1b["), "color render_all should contain ANSI codes");
}

#[test]
fn plain_render_all_no_ansi() {
    use cjc_diag::{Diagnostic, DiagnosticBag, Span};

    let mut bag = DiagnosticBag::new();
    bag.emit(Diagnostic::error("E0001", "test error", Span::new(0, 1)));

    let source = "x";
    let output = bag.render_all(source, "test.cjcl");
    assert!(!output.contains("\x1b["), "plain render_all should not contain ANSI codes");
}
