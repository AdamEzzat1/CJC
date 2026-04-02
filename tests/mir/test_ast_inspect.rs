//! Integration tests for AST inspect/diagnostics — end-to-end from CJC source.

use cjc_ast::inspect;
use cjc_ast::metrics::compute_metrics;
use cjc_ast::validate::validate_ast;
use cjc_ast::Program;

fn parse(src: &str) -> Program {
    let (program, diags) = cjc_parser::parse_source(src);
    if diags.has_errors() {
        panic!("CJC parse failed for:\n{}\n", src);
    }
    program
}

#[test]
fn test_inspect_summary() {
    let src = r#"
fn compute(n: i64) -> i64 {
    return n + 1;
}
print(compute(5));
"#;
    let program = parse(src);
    let text = inspect::dump_ast_summary(&program);
    assert!(text.contains("AstSummary"));
    assert!(text.contains("fn compute"));
}

#[test]
fn test_inspect_metrics() {
    let src = r#"
fn sum(n: i64) -> i64 {
    let mut acc: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        acc = acc + i;
        i = i + 1;
    }
    return acc;
}
print(sum(10));
"#;
    let program = parse(src);
    let metrics = compute_metrics(&program);
    let text = inspect::dump_ast_metrics(&metrics);
    assert!(text.contains("AstMetrics"));
    assert!(text.contains("functions=1"));
    assert!(text.contains("loops=1"));
}

#[test]
fn test_inspect_validation_clean() {
    let src = "let x: i64 = 42;\nprint(x);\n";
    let program = parse(src);
    let report = validate_ast(&program);
    let text = inspect::dump_validation_report(&report);
    assert!(text.contains("ValidationReport"));
}

#[test]
fn test_inspect_determinism() {
    let src = r#"
fn nested(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let mut j: i64 = 0;
        while j < n {
            total = total + 1;
            j = j + 1;
        }
        i = i + 1;
    }
    return total;
}
print(nested(3));
"#;
    let program = parse(src);
    let s1 = inspect::dump_ast_summary(&program);
    let s2 = inspect::dump_ast_summary(&program);
    assert_eq!(s1, s2, "summary dump must be deterministic");

    let m = compute_metrics(&program);
    let m1 = inspect::dump_ast_metrics(&m);
    let m2 = inspect::dump_ast_metrics(&m);
    assert_eq!(m1, m2, "metrics dump must be deterministic");
}

#[test]
fn test_inspect_expr_tree() {
    let src = "let x: i64 = 1 + 2 * 3;\n";
    let program = parse(src);
    // Get the init expression from the let declaration.
    if let cjc_ast::DeclKind::Let(ref l) = program.declarations[0].kind {
        let text = inspect::dump_expr_tree(&l.init);
        assert!(text.contains("Binary"), "should show binary op");
        assert!(text.contains("IntLit"), "should show integer literals");
    } else {
        panic!("expected let declaration");
    }
}
