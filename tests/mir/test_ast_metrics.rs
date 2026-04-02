//! Integration tests for AST metrics — end-to-end from CJC source code.

use cjc_ast::metrics::compute_metrics;
use cjc_ast::Program;

fn parse(src: &str) -> Program {
    let (program, diags) = cjc_parser::parse_source(src);
    if diags.has_errors() {
        panic!("CJC parse failed for:\n{}\n", src);
    }
    program
}

#[test]
fn test_metrics_simple_program() {
    let program = parse("let x: i64 = 42;\nprint(x);\n");
    let m = compute_metrics(&program);
    assert!(m.total_nodes > 0, "should have some nodes");
    assert!(m.expr_count > 0, "should count expressions");
    assert_eq!(m.function_count, 0, "no functions declared");
    assert_eq!(m.loop_count, 0, "no loops");
}

#[test]
fn test_metrics_function_with_loops() {
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
print(nested(5));
"#;
    let program = parse(src);
    let m = compute_metrics(&program);
    assert_eq!(m.function_count, 1);
    assert_eq!(m.loop_count, 2, "two while loops");
    assert!(m.max_stmt_depth >= 2, "nested loops = depth >= 2");
    assert!(m.binary_op_counts.get("+").unwrap_or(&0) >= &3);
    assert!(m.binary_op_counts.get("<").unwrap_or(&0) >= &2);
}

#[test]
fn test_metrics_match_expression() {
    let src = r#"
fn classify(x: i64) -> i64 {
    let result: i64 = match x {
        0 => 0,
        1 => 1,
        _ => 2,
    };
    return result;
}
print(classify(1));
"#;
    let program = parse(src);
    let m = compute_metrics(&program);
    assert_eq!(m.match_count, 1, "one match expression");
    assert!(m.pattern_count >= 3, "at least 3 patterns");
}

#[test]
fn test_metrics_lambda() {
    let src = r#"
let f: Any = |x: i64| x + 1;
print(f(5));
"#;
    let program = parse(src);
    let m = compute_metrics(&program);
    assert_eq!(m.closure_count, 1, "one lambda expression");
}

#[test]
fn test_metrics_determinism() {
    let src = r#"
fn compute(n: i64) -> i64 {
    let mut acc: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        acc = acc + i;
        i = i + 1;
    }
    return acc;
}
print(compute(100));
"#;
    let program = parse(src);
    let m1 = compute_metrics(&program);
    let m2 = compute_metrics(&program);
    assert_eq!(m1.total_nodes, m2.total_nodes);
    assert_eq!(m1.expr_count, m2.expr_count);
    assert_eq!(m1.stmt_count, m2.stmt_count);
    assert_eq!(m1.function_count, m2.function_count);
    assert_eq!(m1.max_expr_depth, m2.max_expr_depth);
}

/// Property-style: visitor node count must equal metrics total_nodes.
#[test]
fn test_metrics_matches_visitor_count() {
    use cjc_ast::visit::{self, AstVisitor};
    use cjc_ast::{Decl, Expr, Pattern, Stmt};

    struct Counter {
        total: u32,
    }
    impl AstVisitor for Counter {
        fn visit_expr(&mut self, expr: &Expr) {
            self.total += 1;
            visit::walk_expr(self, expr);
        }
        fn visit_stmt(&mut self, stmt: &Stmt) {
            self.total += 1;
            visit::walk_stmt(self, stmt);
        }
        fn visit_decl(&mut self, decl: &Decl) {
            self.total += 1;
            visit::walk_decl(self, decl);
        }
        fn visit_pattern(&mut self, pattern: &Pattern) {
            self.total += 1;
            visit::walk_pattern(self, pattern);
        }
    }

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
    let mut counter = Counter { total: 0 };
    counter.visit_program(&program);
    assert_eq!(
        metrics.total_nodes, counter.total,
        "metrics total_nodes must match visitor node count"
    );
}
