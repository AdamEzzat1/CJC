//! Integration tests for AST visitor — end-to-end from CJC source code.

use cjc_ast::visit::{self, AstVisitor};
use cjc_ast::{Expr, Ident, Pattern, Program, Stmt};

fn parse(src: &str) -> Program {
    let (program, diags) = cjc_parser::parse_source(src);
    if diags.has_errors() {
        panic!("CJC parse failed for:\n{}\n", src);
    }
    program
}

struct NodeCounter {
    exprs: u32,
    stmts: u32,
    patterns: u32,
    idents: u32,
}

impl NodeCounter {
    fn new() -> Self {
        Self {
            exprs: 0,
            stmts: 0,
            patterns: 0,
            idents: 0,
        }
    }
}

impl AstVisitor for NodeCounter {
    fn visit_expr(&mut self, expr: &Expr) {
        self.exprs += 1;
        visit::walk_expr(self, expr);
    }
    fn visit_stmt(&mut self, stmt: &Stmt) {
        self.stmts += 1;
        visit::walk_stmt(self, stmt);
    }
    fn visit_pattern(&mut self, pattern: &Pattern) {
        self.patterns += 1;
        visit::walk_pattern(self, pattern);
    }
    fn visit_ident(&mut self, _ident: &Ident) {
        self.idents += 1;
    }
}

#[test]
fn test_visitor_simple_program() {
    let program = parse("let x: i64 = 42;\nprint(x);\n");
    let mut counter = NodeCounter::new();
    counter.visit_program(&program);
    assert!(counter.exprs > 0, "should visit expressions");
    assert!(counter.idents > 0, "should visit identifiers");
}

#[test]
fn test_visitor_function_with_loop() {
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
    let program = parse(src);
    let mut counter = NodeCounter::new();
    counter.visit_program(&program);
    assert!(counter.exprs >= 5, "loop body has multiple expressions");
    assert!(counter.stmts >= 3, "function body has multiple statements");
}

#[test]
fn test_visitor_match_patterns() {
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
    let mut counter = NodeCounter::new();
    counter.visit_program(&program);
    assert!(counter.patterns >= 3, "should visit 3 match arm patterns");
}

#[test]
fn test_visitor_multiple_functions() {
    let src = r#"
fn add(x: i64, y: i64) -> i64 {
    return x + y;
}
fn mul(x: i64, y: i64) -> i64 {
    return x * y;
}
print(add(mul(2, 3), 4));
"#;
    let program = parse(src);
    let mut counter = NodeCounter::new();
    counter.visit_program(&program);
    assert!(counter.idents >= 8, "should visit many identifiers across fns");
}

#[test]
fn test_visitor_struct_decl() {
    let src = r#"
struct Point {
    x: f64,
    y: f64,
}
let p: Point = Point { x: 1.0, y: 2.0 };
print(p.x);
"#;
    let program = parse(src);
    let mut counter = NodeCounter::new();
    counter.visit_program(&program);
    assert!(counter.idents > 0, "should visit struct field identifiers");
}

#[test]
fn test_visitor_determinism() {
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
print(sum(100));
"#;
    let program = parse(src);
    let mut c1 = NodeCounter::new();
    let mut c2 = NodeCounter::new();
    c1.visit_program(&program);
    c2.visit_program(&program);
    assert_eq!(c1.exprs, c2.exprs, "visitor must be deterministic");
    assert_eq!(c1.stmts, c2.stmts);
    assert_eq!(c1.idents, c2.idents);
}
