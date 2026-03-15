//! Cross-file diagnostic tests.

#[test]
fn test_diagnostic_has_filename_field() {
    let diag = cjc_diag::Diagnostic::error("E0001", "test error", cjc_diag::Span::new(0, 1))
        .with_filename("math.cjc");
    assert_eq!(diag.filename.as_deref(), Some("math.cjc"));
}

#[test]
fn test_diagnostic_renderer_uses_filename() {
    let source = "let x = 1;\n";
    let diag = cjc_diag::Diagnostic::error("E2001", "type mismatch", cjc_diag::Span::new(0, 3))
        .with_filename("stats.cjc");

    let renderer = cjc_diag::DiagnosticRenderer::new(source, "main.cjc");
    let output = renderer.render(&diag);
    // Should use stats.cjc, not main.cjc
    assert!(output.contains("stats.cjc"), "expected stats.cjc in: {}", output);
    assert!(!output.contains("main.cjc"), "should not contain main.cjc");
}

#[test]
fn test_type_checker_with_filename() {
    let checker = cjc_types::TypeChecker::new_with_filename("module.cjc");
    assert_eq!(checker.current_filename.as_deref(), Some("module.cjc"));
}

#[test]
fn test_type_checker_default_no_filename() {
    let checker = cjc_types::TypeChecker::new();
    assert!(checker.current_filename.is_none());
}

#[test]
fn test_diagnostic_builder_with_filename() {
    let diag = cjc_diag::DiagnosticBuilder::new(
        cjc_diag::ErrorCode::E1000,
        cjc_diag::Span::new(0, 1),
    )
    .filename("helper.cjc")
    .build();
    assert_eq!(diag.filename.as_deref(), Some("helper.cjc"));
}
