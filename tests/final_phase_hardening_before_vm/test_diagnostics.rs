//! Cross-file diagnostic tests.

#[test]
fn test_diagnostic_has_filename_field() {
    let diag = cjc_diag::Diagnostic::error("E0001", "test error", cjc_diag::Span::new(0, 1))
        .with_filename("math.cjcl");
    assert_eq!(diag.filename.as_deref(), Some("math.cjcl"));
}

#[test]
fn test_diagnostic_renderer_uses_filename() {
    let source = "let x = 1;\n";
    let diag = cjc_diag::Diagnostic::error("E2001", "type mismatch", cjc_diag::Span::new(0, 3))
        .with_filename("stats.cjcl");

    let renderer = cjc_diag::DiagnosticRenderer::new(source, "main.cjcl");
    let output = renderer.render(&diag);
    // Should use stats.cjcl, not main.cjcl
    assert!(output.contains("stats.cjcl"), "expected stats.cjcl in: {}", output);
    assert!(!output.contains("main.cjcl"), "should not contain main.cjcl");
}

#[test]
fn test_type_checker_with_filename() {
    let checker = cjc_types::TypeChecker::new_with_filename("module.cjcl");
    assert_eq!(checker.current_filename.as_deref(), Some("module.cjcl"));
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
    .filename("helper.cjcl")
    .build();
    assert_eq!(diag.filename.as_deref(), Some("helper.cjcl"));
}
