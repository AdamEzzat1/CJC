// CJC Test Suite — cjc-parser (33 tests)
// Source: crates/cjc-parser/src/lib.rs
// These tests are extracted from the inline #[cfg(test)] modules for regression tracking.

use cjc_parser::parse_source;
use cjc_ast::*;
use cjc_diag::DiagnosticBag;

/// Helper: lex + parse, assert no errors, return program.
fn parse_ok(source: &str) -> Program {
    let (program, diags) = parse_source(source);
    if diags.has_errors() {
        let rendered = diags.render_all(source, "<test>");
        panic!("unexpected parse errors:\n{}", rendered);
    }
    program
}

/// Helper: lex + parse, assert at least one error.
fn parse_err(source: &str) -> DiagnosticBag {
    let (_, diags) = parse_source(source);
    assert!(
        diags.has_errors(),
        "expected parse error but got none for: {}",
        source
    );
    diags
}

// ── Struct parsing ─────────────────────────────────────────────

#[test]
fn test_parse_struct_simple() {
    let prog = parse_ok("struct Point { x: f64, y: f64 }");
    assert_eq!(prog.declarations.len(), 1);
    match &prog.declarations[0].kind {
        DeclKind::Struct(s) => {
            assert_eq!(s.name.name, "Point");
            assert_eq!(s.fields.len(), 2);
            assert_eq!(s.fields[0].name.name, "x");
            assert_eq!(s.fields[1].name.name, "y");
        }
        _ => panic!("expected struct"),
    }
}

#[test]
fn test_parse_struct_generic() {
    let prog = parse_ok("struct Pair<T: Clone, U> { first: T, second: U }");
    match &prog.declarations[0].kind {
        DeclKind::Struct(s) => {
            assert_eq!(s.type_params.len(), 2);
            assert_eq!(s.type_params[0].name.name, "T");
            assert_eq!(s.type_params[0].bounds.len(), 1);
            assert_eq!(s.type_params[1].name.name, "U");
            assert!(s.type_params[1].bounds.is_empty());
        }
        _ => panic!("expected struct"),
    }
}

// ── Class parsing ──────────────────────────────────────────────

#[test]
fn test_parse_class() {
    let prog = parse_ok("class Node<T> { value: T, next: Node<T> }");
    match &prog.declarations[0].kind {
        DeclKind::Class(c) => {
            assert_eq!(c.name.name, "Node");
            assert_eq!(c.type_params.len(), 1);
            assert_eq!(c.fields.len(), 2);
        }
        _ => panic!("expected class"),
    }
}

// ── Function parsing ───────────────────────────────────────────

#[test]
fn test_parse_fn_simple() {
    let prog = parse_ok("fn add(a: i64, b: i64) -> i64 { a + b }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            assert_eq!(f.name.name, "add");
            assert_eq!(f.params.len(), 2);
            assert!(f.return_type.is_some());
            assert!(!f.is_nogc);
            // The body should have a tail expression.
            assert!(f.body.expr.is_some());
        }
        _ => panic!("expected fn"),
    }
}

#[test]
fn test_parse_fn_nogc() {
    let prog = parse_ok("nogc fn fast(x: f64) -> f64 { x }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            assert!(f.is_nogc);
            assert_eq!(f.name.name, "fast");
        }
        _ => panic!("expected fn"),
    }
}

#[test]
fn test_parse_fn_no_return_type() {
    let prog = parse_ok("fn greet(name: String) { name }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            assert!(f.return_type.is_none());
        }
        _ => panic!("expected fn"),
    }
}

// ── Trait parsing ──────────────────────────────────────────────

#[test]
fn test_parse_trait() {
    let prog = parse_ok(
        "trait Numeric: Add + Mul { fn zero() -> Self; fn one() -> Self; }",
    );
    match &prog.declarations[0].kind {
        DeclKind::Trait(t) => {
            assert_eq!(t.name.name, "Numeric");
            assert_eq!(t.super_traits.len(), 2);
            assert_eq!(t.methods.len(), 2);
            assert_eq!(t.methods[0].name.name, "zero");
        }
        _ => panic!("expected trait"),
    }
}

// ── Impl parsing ──────────────────────────────────────────────

#[test]
fn test_parse_impl() {
    let prog = parse_ok(
        "impl<T> Vec<T> : Iterable { fn len(self: Vec<T>) -> i64 { 0 } }",
    );
    match &prog.declarations[0].kind {
        DeclKind::Impl(i) => {
            assert_eq!(i.type_params.len(), 1);
            assert!(i.trait_ref.is_some());
            assert_eq!(i.methods.len(), 1);
        }
        _ => panic!("expected impl"),
    }
}

// ── Import parsing ─────────────────────────────────────────────

#[test]
fn test_parse_import() {
    let prog = parse_ok("import std.io.File as F");
    match &prog.declarations[0].kind {
        DeclKind::Import(i) => {
            assert_eq!(i.path.len(), 3);
            assert_eq!(i.path[0].name, "std");
            assert_eq!(i.path[1].name, "io");
            assert_eq!(i.path[2].name, "File");
            assert_eq!(i.alias.as_ref().unwrap().name, "F");
        }
        _ => panic!("expected import"),
    }
}

#[test]
fn test_parse_import_no_alias() {
    let prog = parse_ok("import math.linalg");
    match &prog.declarations[0].kind {
        DeclKind::Import(i) => {
            assert_eq!(i.path.len(), 2);
            assert!(i.alias.is_none());
        }
        _ => panic!("expected import"),
    }
}

// ── Let statement ──────────────────────────────────────────────

#[test]
fn test_parse_let() {
    let prog = parse_ok("let x: i64 = 42;");
    match &prog.declarations[0].kind {
        DeclKind::Let(l) => {
            assert_eq!(l.name.name, "x");
            assert!(!l.mutable);
            assert!(l.ty.is_some());
        }
        _ => panic!("expected let"),
    }
}

#[test]
fn test_parse_let_mut() {
    let prog = parse_ok("let mut count = 0;");
    match &prog.declarations[0].kind {
        DeclKind::Let(l) => {
            assert!(l.mutable);
            assert!(l.ty.is_none());
        }
        _ => panic!("expected let"),
    }
}

// ── Expression parsing ─────────────────────────────────────────

#[test]
fn test_parse_binary_precedence() {
    // `1 + 2 * 3` should parse as `1 + (2 * 3)`.
    let prog = parse_ok("fn main() { 1 + 2 * 3 }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            let tail = f.body.expr.as_ref().unwrap();
            match &tail.kind {
                ExprKind::Binary { op, left, right } => {
                    assert_eq!(*op, BinOp::Add);
                    // left should be 1.
                    assert!(matches!(left.kind, ExprKind::IntLit(1)));
                    // right should be 2 * 3.
                    match &right.kind {
                        ExprKind::Binary { op, .. } => assert_eq!(*op, BinOp::Mul),
                        _ => panic!("expected binary mul"),
                    }
                }
                _ => panic!("expected binary add"),
            }
        }
        _ => panic!("expected fn"),
    }
}

#[test]
fn test_parse_unary() {
    let prog = parse_ok("fn f() { -x }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            let tail = f.body.expr.as_ref().unwrap();
            match &tail.kind {
                ExprKind::Unary { op, .. } => assert_eq!(*op, UnaryOp::Neg),
                _ => panic!("expected unary"),
            }
        }
        _ => panic!("expected fn"),
    }
}

#[test]
fn test_parse_call_with_named_args() {
    let prog = parse_ok("fn f() { create(width: 10, height: 20) }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            let tail = f.body.expr.as_ref().unwrap();
            match &tail.kind {
                ExprKind::Call { args, .. } => {
                    assert_eq!(args.len(), 2);
                    assert_eq!(args[0].name.as_ref().unwrap().name, "width");
                    assert_eq!(args[1].name.as_ref().unwrap().name, "height");
                }
                _ => panic!("expected call"),
            }
        }
        _ => panic!("expected fn"),
    }
}

#[test]
fn test_parse_field_access_and_method_call() {
    let prog = parse_ok("fn f() { obj.field.method(x) }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            let tail = f.body.expr.as_ref().unwrap();
            // Should be: Call { callee: Field { object: Field { ... }, name: method }, args: [x] }
            match &tail.kind {
                ExprKind::Call { callee, args } => {
                    assert_eq!(args.len(), 1);
                    match &callee.kind {
                        ExprKind::Field { name, .. } => {
                            assert_eq!(name.name, "method");
                        }
                        _ => panic!("expected field access"),
                    }
                }
                _ => panic!("expected call"),
            }
        }
        _ => panic!("expected fn"),
    }
}

#[test]
fn test_parse_index_and_multi_index() {
    let prog = parse_ok("fn f() { a[0]; b[1, 2] }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            // First statement: a[0] — single index.
            match &f.body.stmts[0].kind {
                StmtKind::Expr(e) => match &e.kind {
                    ExprKind::Index { .. } => {}
                    _ => panic!("expected index"),
                },
                _ => panic!("expected expr stmt"),
            }
            // Tail expression: b[1, 2] — multi-index.
            let tail = f.body.expr.as_ref().unwrap();
            match &tail.kind {
                ExprKind::MultiIndex { indices, .. } => {
                    assert_eq!(indices.len(), 2);
                }
                _ => panic!("expected multi-index"),
            }
        }
        _ => panic!("expected fn"),
    }
}

#[test]
fn test_parse_pipe() {
    let prog = parse_ok("fn f() { data |> filter(x) |> map(y) }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            let tail = f.body.expr.as_ref().unwrap();
            // Should be left-associative: (data |> filter(x)) |> map(y)
            match &tail.kind {
                ExprKind::Pipe { right, .. } => {
                    match &right.kind {
                        ExprKind::Call { callee, .. } => match &callee.kind {
                            ExprKind::Ident(id) => assert_eq!(id.name, "map"),
                            _ => panic!("expected ident"),
                        },
                        _ => panic!("expected call"),
                    }
                }
                _ => panic!("expected pipe"),
            }
        }
        _ => panic!("expected fn"),
    }
}

#[test]
fn test_parse_assignment() {
    let prog = parse_ok("fn f() { x = 10; }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => match &f.body.stmts[0].kind {
            StmtKind::Expr(e) => match &e.kind {
                ExprKind::Assign { .. } => {}
                _ => panic!("expected assign"),
            },
            _ => panic!("expected expr stmt"),
        },
        _ => panic!("expected fn"),
    }
}

#[test]
fn test_parse_struct_literal() {
    let prog = parse_ok("fn f() { Point { x: 1, y: 2 } }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            let tail = f.body.expr.as_ref().unwrap();
            match &tail.kind {
                ExprKind::StructLit { name, fields } => {
                    assert_eq!(name.name, "Point");
                    assert_eq!(fields.len(), 2);
                }
                _ => panic!("expected struct lit"),
            }
        }
        _ => panic!("expected fn"),
    }
}

#[test]
fn test_parse_array_literal() {
    let prog = parse_ok("fn f() { [1, 2, 3] }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            let tail = f.body.expr.as_ref().unwrap();
            match &tail.kind {
                ExprKind::ArrayLit(elems) => assert_eq!(elems.len(), 3),
                _ => panic!("expected array lit"),
            }
        }
        _ => panic!("expected fn"),
    }
}

#[test]
fn test_parse_col() {
    let prog = parse_ok(r#"fn f() { col("price") }"#);
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            let tail = f.body.expr.as_ref().unwrap();
            match &tail.kind {
                ExprKind::Col(name) => assert_eq!(name, "price"),
                _ => panic!("expected col"),
            }
        }
        _ => panic!("expected fn"),
    }
}

// ── Control flow ───────────────────────────────────────────────

#[test]
fn test_parse_if_else_if_else() {
    let prog = parse_ok(
        "fn f() { if x { 1; } else if y { 2; } else { 3; } }",
    );
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            assert_eq!(f.body.stmts.len(), 1);
            match &f.body.stmts[0].kind {
                StmtKind::If(if_stmt) => {
                    assert!(if_stmt.else_branch.is_some());
                    match if_stmt.else_branch.as_ref().unwrap() {
                        ElseBranch::ElseIf(elif) => {
                            assert!(elif.else_branch.is_some());
                            match elif.else_branch.as_ref().unwrap() {
                                ElseBranch::Else(_) => {}
                                _ => panic!("expected else block"),
                            }
                        }
                        _ => panic!("expected else-if"),
                    }
                }
                _ => panic!("expected if"),
            }
        }
        _ => panic!("expected fn"),
    }
}

#[test]
fn test_parse_while() {
    let prog = parse_ok("fn f() { while x > 0 { x = x - 1; } }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => match &f.body.stmts[0].kind {
            StmtKind::While(w) => {
                assert!(!w.body.stmts.is_empty());
            }
            _ => panic!("expected while"),
        },
        _ => panic!("expected fn"),
    }
}

#[test]
fn test_parse_return() {
    let prog = parse_ok("fn f() { return 42; }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => match &f.body.stmts[0].kind {
            StmtKind::Return(Some(e)) => {
                assert!(matches!(e.kind, ExprKind::IntLit(42)));
            }
            _ => panic!("expected return"),
        },
        _ => panic!("expected fn"),
    }
}

#[test]
fn test_parse_nogc_block() {
    let prog = parse_ok("fn f() { nogc { x + y; } }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => match &f.body.stmts[0].kind {
            StmtKind::NoGcBlock(block) => {
                assert_eq!(block.stmts.len(), 1);
            }
            _ => panic!("expected nogc block"),
        },
        _ => panic!("expected fn"),
    }
}

// ── Error recovery ─────────────────────────────────────────────

#[test]
fn test_error_recovery_missing_semicolon() {
    // Missing semicolon after let — parser should recover and parse
    // the next declaration.
    let (prog, diags) = parse_source("let x = 1\nfn f() { 0 }");
    assert!(diags.has_errors());
    // Should still have parsed the fn.
    assert!(prog.declarations.iter().any(|d| matches!(&d.kind, DeclKind::Fn(_))));
}

#[test]
fn test_error_recovery_unexpected_token() {
    let diags = parse_err("@@@ fn f() { 0 }");
    assert!(diags.has_errors());
}

#[test]
fn test_error_expected_expression() {
    let diags = parse_err("fn f() { let x = ; }");
    assert!(diags.has_errors());
}

// ── Complex integration test ───────────────────────────────────

#[test]
fn test_parse_full_program() {
    let source = r#"
        import std.math as m

        struct Vec2 {
            x: f64,
            y: f64
        }

        fn dot(a: Vec2, b: Vec2) -> f64 {
            a.x * b.x + a.y * b.y
        }

        trait Shape {
            fn area(self: Self) -> f64;
        }

        impl Vec2 : Shape {
            fn area(self: Vec2) -> f64 {
                self.x * self.y
            }
        }

        let result: f64 = dot(Vec2 { x: 1.0, y: 2.0 }, Vec2 { x: 3.0, y: 4.0 });
    "#;
    let prog = parse_ok(source);
    assert_eq!(prog.declarations.len(), 6);
}

#[test]
fn test_parse_pipe_chain() {
    let source = r#"
        fn pipeline(data: DataFrame) -> DataFrame {
            data
                |> filter(col("age") > 18)
                |> group_by(col("city"))
        }
    "#;
    // Should parse without errors. The pipe chain produces a tail
    // expression.
    let prog = parse_ok(source);
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            assert!(f.body.expr.is_some());
        }
        _ => panic!("expected fn"),
    }
}

// ── Boolean and logical operators ──────────────────────────────

#[test]
fn test_parse_logical_operators() {
    let prog = parse_ok("fn f() { a && b || c }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            let tail = f.body.expr.as_ref().unwrap();
            // `&&` binds tighter than `||`, so: (a && b) || c
            match &tail.kind {
                ExprKind::Binary { op, .. } => assert_eq!(*op, BinOp::Or),
                _ => panic!("expected binary or"),
            }
        }
        _ => panic!("expected fn"),
    }
}

#[test]
fn test_parse_comparison_chain() {
    let prog = parse_ok("fn f() { x == 1 && y != 2 }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            let tail = f.body.expr.as_ref().unwrap();
            match &tail.kind {
                ExprKind::Binary { op, .. } => assert_eq!(*op, BinOp::And),
                _ => panic!("expected and"),
            }
        }
        _ => panic!("expected fn"),
    }
}
