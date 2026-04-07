//! LH01: Diagnostics + Error Taxonomy tests
//!
//! Verifies:
//! - Error code taxonomy is complete and consistent
//! - DiagnosticBuilder produces correct diagnostics
//! - Fix suggestions render properly
//! - Multi-line span rendering
//! - Backward compatibility with existing Diagnostic::error() API

use cjc_diag::{
    Diagnostic, DiagnosticBag, DiagnosticBuilder, ErrorCode, Severity, Span,
};

// ── Error Code Taxonomy ──────────────────────────────────────────────

#[test]
fn test_error_codes_have_consistent_prefixes() {
    // Lexer errors: E0xxx
    assert!(ErrorCode::E0001.code_str().starts_with("E0"));
    assert!(ErrorCode::E0010.code_str().starts_with("E0"));
    // Parser errors: E1xxx
    assert!(ErrorCode::E1000.code_str().starts_with("E1"));
    assert!(ErrorCode::E1013.code_str().starts_with("E1"));
    // Type errors: E2xxx
    assert!(ErrorCode::E2001.code_str().starts_with("E2"));
    assert!(ErrorCode::E2015.code_str().starts_with("E2"));
    // Effect errors: E4xxx
    assert!(ErrorCode::E4001.code_str().starts_with("E4"));
    // Generics/Trait: E6xxx
    assert!(ErrorCode::E6001.code_str().starts_with("E6"));
    // MIR: E7xxx
    assert!(ErrorCode::E7001.code_str().starts_with("E7"));
    // Runtime: E8xxx
    assert!(ErrorCode::E8001.code_str().starts_with("E8"));
    // Module: E9xxx
    assert!(ErrorCode::E9001.code_str().starts_with("E9"));
    // Snap: E06xx
    assert!(ErrorCode::E0601.code_str().starts_with("E06"));
    // Warnings: W0xxx
    assert!(ErrorCode::W0001.code_str().starts_with("W0"));
}

#[test]
fn test_all_errors_are_error_severity() {
    let error_codes = [
        ErrorCode::E0001, ErrorCode::E1000, ErrorCode::E2001,
        ErrorCode::E4001, ErrorCode::E6001, ErrorCode::E7001,
        ErrorCode::E8001, ErrorCode::E9001, ErrorCode::E0601,
    ];
    for code in &error_codes {
        assert_eq!(code.severity(), Severity::Error, "{} should be Error", code);
    }
}

#[test]
fn test_all_warnings_are_warning_severity() {
    let warning_codes = [
        ErrorCode::W0001, ErrorCode::W0002, ErrorCode::W0003,
        ErrorCode::W0004, ErrorCode::W0005,
    ];
    for code in &warning_codes {
        assert_eq!(code.severity(), Severity::Warning, "{} should be Warning", code);
    }
}

#[test]
fn test_error_categories() {
    assert_eq!(ErrorCode::E0001.category(), "lexer");
    assert_eq!(ErrorCode::E1000.category(), "parser");
    assert_eq!(ErrorCode::E2001.category(), "type");
    assert_eq!(ErrorCode::E4001.category(), "effect");
    assert_eq!(ErrorCode::E6001.category(), "generics/trait");
    assert_eq!(ErrorCode::E7001.category(), "mir");
    assert_eq!(ErrorCode::E8001.category(), "runtime");
    assert_eq!(ErrorCode::E9001.category(), "module");
    assert_eq!(ErrorCode::E0601.category(), "snap");
    assert_eq!(ErrorCode::W0001.category(), "warning");
}

#[test]
fn test_error_code_display() {
    assert_eq!(format!("{}", ErrorCode::E0001), "E0001");
    assert_eq!(format!("{}", ErrorCode::E1000), "E1000");
    assert_eq!(format!("{}", ErrorCode::W0001), "W0001");
}

// ── DiagnosticBuilder ────────────────────────────────────────────────

#[test]
fn test_builder_with_default_message() {
    let diag = DiagnosticBuilder::new(ErrorCode::E2001, Span::new(10, 20)).build();
    assert_eq!(diag.code, "E2001");
    assert_eq!(diag.message, "type mismatch");
    assert_eq!(diag.severity, Severity::Error);
    assert_eq!(diag.span, Span::new(10, 20));
    assert!(diag.labels.is_empty());
    assert!(diag.hints.is_empty());
    assert!(diag.fix_suggestions.is_empty());
}

#[test]
fn test_builder_with_custom_message() {
    let diag = DiagnosticBuilder::new(ErrorCode::E2001, Span::new(0, 5))
        .message("expected `i64`, found `str`")
        .build();
    assert_eq!(diag.message, "expected `i64`, found `str`");
}

#[test]
fn test_builder_with_labels_and_hints() {
    let diag = DiagnosticBuilder::new(ErrorCode::E2005, Span::new(0, 10))
        .message("wrong number of arguments: expected 2, found 3")
        .label(Span::new(5, 8), "extra argument here")
        .hint("remove the third argument")
        .build();

    assert_eq!(diag.labels.len(), 1);
    assert_eq!(diag.labels[0].message, "extra argument here");
    assert_eq!(diag.hints.len(), 1);
    assert_eq!(diag.hints[0], "remove the third argument");
}

#[test]
fn test_builder_with_fix_suggestion() {
    let diag = DiagnosticBuilder::new(ErrorCode::E2001, Span::new(7, 10))
        .fix(Span::new(7, 10), "str", "change type to `str`")
        .build();

    assert_eq!(diag.fix_suggestions.len(), 1);
    assert_eq!(diag.fix_suggestions[0].replacement, "str");
    assert_eq!(diag.fix_suggestions[0].message, "change type to `str`");
}

// ── Rendering ────────────────────────────────────────────────────────

#[test]
fn test_render_single_line_error() {
    let source = "let x: i64 = \"hello\";\n";
    let diag = DiagnosticBuilder::new(ErrorCode::E2001, Span::new(13, 20))
        .message("expected `i64`, found `str`")
        .label(Span::new(13, 20), "this is a string")
        .build();

    let renderer = cjc_diag::DiagnosticRenderer::new(source, "test.cjcl");
    let output = renderer.render(&diag);

    assert!(output.contains("error[E2001]"));
    assert!(output.contains("expected `i64`, found `str`"));
    assert!(output.contains("test.cjcl:1:14"));
    assert!(output.contains("^^^^^^^"));
}

#[test]
fn test_render_fix_suggestion() {
    let source = "let x: i32 = 42;\n";
    let diag = Diagnostic::error("E2001", "type mismatch", Span::new(7, 10))
        .with_label(Span::new(7, 10), "expected `i64`")
        .with_fix(Span::new(7, 10), "i64", "change type annotation to `i64`");

    let renderer = cjc_diag::DiagnosticRenderer::new(source, "test.cjcl");
    let output = renderer.render(&diag);

    assert!(output.contains("fix:"));
    assert!(output.contains("change type annotation to `i64`"));
}

#[test]
fn test_render_multiline_span() {
    let source = "fn foo() {\n    let x = 1;\n    let y = 2;\n}\n";
    // Span covering lines 1-4
    let diag = Diagnostic::error("E7001", "internal error in function", Span::new(0, source.len() - 1));

    let renderer = cjc_diag::DiagnosticRenderer::new(source, "test.cjcl");
    let output = renderer.render(&diag);

    assert!(output.contains("error[E7001]"));
    assert!(output.contains("internal error in function"));
    // Should show multiple lines
    assert!(output.contains("fn foo()"));
}

#[test]
fn test_render_warning() {
    let source = "let _unused = 42;\n";
    let diag = DiagnosticBuilder::new(ErrorCode::W0001, Span::new(4, 11))
        .message("unused variable `_unused`")
        .build();

    let renderer = cjc_diag::DiagnosticRenderer::new(source, "test.cjcl");
    let output = renderer.render(&diag);

    assert!(output.contains("warning[W0001]"));
    assert!(output.contains("unused variable"));
}

// ── Backward Compatibility ───────────────────────────────────────────

#[test]
fn test_old_api_still_works() {
    // The existing Diagnostic::error(code_str, msg, span) API must still work
    let diag = Diagnostic::error("E0001", "unexpected char", Span::new(0, 1))
        .with_label(Span::new(0, 1), "here")
        .with_hint("check your syntax");

    assert_eq!(diag.code, "E0001");
    assert_eq!(diag.severity, Severity::Error);
    assert_eq!(diag.labels.len(), 1);
    assert_eq!(diag.hints.len(), 1);
    // fix_suggestions should be empty for old API
    assert!(diag.fix_suggestions.is_empty());
}

#[test]
fn test_diagnostic_bag_emit_coded() {
    let mut bag = DiagnosticBag::new();
    bag.emit_coded(
        DiagnosticBuilder::new(ErrorCode::E4001, Span::new(0, 10))
            .message("GC alloc in nogc block")
            .label(Span::new(0, 10), "this call triggers GC")
    );
    assert!(bag.has_errors());
    assert_eq!(bag.error_count(), 1);
    assert_eq!(bag.diagnostics[0].code, "E4001");
}

// ── Snap Error Codes ─────────────────────────────────────────────────

#[test]
fn test_snap_error_codes() {
    let diag = DiagnosticBuilder::new(ErrorCode::E0601, Span::new(0, 5))
        .message("Snapshot logic hash mismatch. This snapshot was created with Brain v1.2 (hash 0xABC...). You are running Brain v1.3 (hash 0xDEF...). The structure or behavior of this type has changed. Restoring would violate determinism.")
        .build();

    assert_eq!(diag.code, "E0601");
    assert_eq!(diag.severity, Severity::Error);
    assert!(diag.message.contains("logic hash mismatch"));
    assert_eq!(ErrorCode::E0601.category(), "snap");
}

// ── Effect Error Codes ───────────────────────────────────────────────

#[test]
fn test_effect_error_codes() {
    // Verify all effect error codes exist and have correct metadata
    let codes = [
        (ErrorCode::E4001, "GC operation in nogc context"),
        (ErrorCode::E4002, "IO operation in pure context"),
        (ErrorCode::E4003, "nondeterministic operation in deterministic context"),
        (ErrorCode::E4004, "allocation in nogc function"),
        (ErrorCode::E4005, "effect annotation mismatch"),
        (ErrorCode::E4006, "mutation in pure context"),
    ];
    for (code, expected_template) in &codes {
        assert_eq!(code.message_template(), *expected_template, "wrong template for {}", code);
        assert_eq!(code.severity(), Severity::Error);
        assert_eq!(code.category(), "effect");
    }
}

// ── Integration: Parse error rendering ───────────────────────────────

#[test]
fn test_parse_error_uses_diagnostics() {
    let src = "fn foo( { }";
    let (_program, diags) = cjc_parser::parse_source(src);
    assert!(diags.has_errors());

    // Verify rendering doesn't panic
    let output = diags.render_all(src, "test.cjcl");
    assert!(!output.is_empty());
    assert!(output.contains("error"));
}

#[test]
fn test_lexer_error_uses_diagnostics() {
    let src = "let x = \"unterminated";
    let lexer = cjc_lexer::Lexer::new(src);
    let (_tokens, diags) = lexer.tokenize();
    assert!(diags.has_errors());

    let output = diags.render_all(src, "test.cjcl");
    assert!(!output.is_empty());
}
