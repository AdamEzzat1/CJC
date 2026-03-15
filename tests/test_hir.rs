// CJC Test Suite — cjc-hir (18 tests)
// Source: crates/cjc-hir/src/lib.rs
// Integration tests for HIR data structures and AST-to-HIR lowering.

use cjc_hir::*;
use cjc_ast::*;

fn span() -> Span {
    Span::dummy()
}

fn ident(name: &str) -> Ident {
    Ident::dummy(name)
}

fn int_expr(v: i64) -> Expr {
    Expr {
        kind: ExprKind::IntLit(v),
        span: span(),
    }
}

fn ident_expr(name: &str) -> Expr {
    Expr {
        kind: ExprKind::Ident(ident(name)),
        span: span(),
    }
}

fn type_expr(name: &str) -> TypeExpr {
    TypeExpr {
        kind: TypeExprKind::Named {
            name: ident(name),
            args: vec![],
        },
        span: span(),
    }
}

#[test]
fn test_lower_int_literal() {
    let mut lowering = AstLowering::new();
    let ast_expr = int_expr(42);
    let hir = lowering.lower_expr(&ast_expr);
    match hir.kind {
        HirExprKind::IntLit(v) => assert_eq!(v, 42),
        _ => panic!("expected IntLit"),
    }
}

#[test]
fn test_lower_variable() {
    let mut lowering = AstLowering::new();
    let ast_expr = ident_expr("x");
    let hir = lowering.lower_expr(&ast_expr);
    match &hir.kind {
        HirExprKind::Var(name) => assert_eq!(name, "x"),
        _ => panic!("expected Var"),
    }
}

#[test]
fn test_lower_binary_op() {
    let mut lowering = AstLowering::new();
    let ast_expr = Expr {
        kind: ExprKind::Binary {
            op: BinOp::Add,
            left: Box::new(int_expr(1)),
            right: Box::new(int_expr(2)),
        },
        span: span(),
    };
    let hir = lowering.lower_expr(&ast_expr);
    match &hir.kind {
        HirExprKind::Binary { op, left, right } => {
            assert_eq!(op, &BinOp::Add);
            assert!(matches!(left.kind, HirExprKind::IntLit(1)));
            assert!(matches!(right.kind, HirExprKind::IntLit(2)));
        }
        _ => panic!("expected Binary"),
    }
}

#[test]
fn test_lower_pipe_desugaring() {
    let mut lowering = AstLowering::new();
    let ast_expr = Expr {
        kind: ExprKind::Pipe {
            left: Box::new(ident_expr("x")),
            right: Box::new(Expr {
                kind: ExprKind::Call {
                    callee: Box::new(ident_expr("f")),
                    args: vec![CallArg {
                        name: None,
                        value: ident_expr("y"),
                        span: span(),
                    }],
                },
                span: span(),
            }),
        },
        span: span(),
    };
    let hir = lowering.lower_expr(&ast_expr);
    match &hir.kind {
        HirExprKind::Call { callee, args } => {
            assert!(matches!(callee.kind, HirExprKind::Var(ref n) if n == "f"));
            assert_eq!(args.len(), 2);
            assert!(matches!(args[0].kind, HirExprKind::Var(ref n) if n == "x"));
            assert!(matches!(args[1].kind, HirExprKind::Var(ref n) if n == "y"));
        }
        _ => panic!("expected Call from pipe desugaring"),
    }
}

#[test]
fn test_lower_pipe_simple_ident() {
    let mut lowering = AstLowering::new();
    let ast_expr = Expr {
        kind: ExprKind::Pipe {
            left: Box::new(ident_expr("x")),
            right: Box::new(ident_expr("f")),
        },
        span: span(),
    };
    let hir = lowering.lower_expr(&ast_expr);
    match &hir.kind {
        HirExprKind::Call { callee, args } => {
            assert!(matches!(callee.kind, HirExprKind::Var(ref n) if n == "f"));
            assert_eq!(args.len(), 1);
            assert!(matches!(args[0].kind, HirExprKind::Var(ref n) if n == "x"));
        }
        _ => panic!("expected Call from pipe desugaring"),
    }
}

#[test]
fn test_lower_fn_decl() {
    let mut lowering = AstLowering::new();
    let fn_decl = FnDecl {
        name: ident("add"),
        type_params: vec![],
        params: vec![
            Param { name: ident("a"), ty: type_expr("i64"), default: None, is_variadic: false, span: span() },
            Param { name: ident("b"), ty: type_expr("i64"), default: None, is_variadic: false, span: span() },
        ],
        return_type: Some(type_expr("i64")),
        body: Block {
            stmts: vec![],
            expr: Some(Box::new(Expr {
                kind: ExprKind::Binary {
                    op: BinOp::Add,
                    left: Box::new(ident_expr("a")),
                    right: Box::new(ident_expr("b")),
                },
                span: span(),
            })),
            span: span(),
        },
        is_nogc: false,
        effect_annotation: None,
        decorators: vec![],
        vis: cjc_ast::Visibility::Private,
    };
    let hir_fn = lowering.lower_fn_decl(&fn_decl);
    assert_eq!(hir_fn.name, "add");
    assert_eq!(hir_fn.params.len(), 2);
    assert_eq!(hir_fn.params[0].name, "a");
    assert_eq!(hir_fn.params[1].name, "b");
    assert_eq!(hir_fn.return_type, Some("i64".to_string()));
    assert!(!hir_fn.is_nogc);
    assert!(hir_fn.body.expr.is_some());
}

#[test]
fn test_lower_struct_literal() {
    let mut lowering = AstLowering::new();
    let ast_expr = Expr {
        kind: ExprKind::StructLit {
            name: ident("Point"),
            fields: vec![
                FieldInit { name: ident("x"), value: int_expr(1), span: span() },
                FieldInit { name: ident("y"), value: int_expr(2), span: span() },
            ],
        },
        span: span(),
    };
    let hir = lowering.lower_expr(&ast_expr);
    match &hir.kind {
        HirExprKind::StructLit { name, fields } => {
            assert_eq!(name, "Point");
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0].0, "x");
            assert_eq!(fields[1].0, "y");
        }
        _ => panic!("expected StructLit"),
    }
}

#[test]
fn test_lower_if_else() {
    let mut lowering = AstLowering::new();
    let if_stmt = IfStmt {
        condition: Expr { kind: ExprKind::BoolLit(true), span: span() },
        then_block: Block {
            stmts: vec![],
            expr: Some(Box::new(int_expr(1))),
            span: span(),
        },
        else_branch: Some(ElseBranch::Else(Block {
            stmts: vec![],
            expr: Some(Box::new(int_expr(2))),
            span: span(),
        })),
    };
    let hir_if = lowering.lower_if(&if_stmt);
    assert!(matches!(hir_if.cond.kind, HirExprKind::BoolLit(true)));
    assert!(hir_if.else_branch.is_some());
}

#[test]
fn test_lower_while() {
    let mut lowering = AstLowering::new();
    let ast_stmt = Stmt {
        kind: StmtKind::While(WhileStmt {
            condition: Expr { kind: ExprKind::BoolLit(true), span: span() },
            body: Block { stmts: vec![], expr: None, span: span() },
        }),
        span: span(),
    };
    let hir_stmt = lowering.lower_stmt(&ast_stmt);
    match &hir_stmt.kind {
        HirStmtKind::While { cond, body } => {
            assert!(matches!(cond.kind, HirExprKind::BoolLit(true)));
            assert!(body.stmts.is_empty());
        }
        _ => panic!("expected While"),
    }
}

#[test]
fn test_lower_array_literal() {
    let mut lowering = AstLowering::new();
    let ast_expr = Expr {
        kind: ExprKind::ArrayLit(vec![int_expr(1), int_expr(2), int_expr(3)]),
        span: span(),
    };
    let hir = lowering.lower_expr(&ast_expr);
    match &hir.kind {
        HirExprKind::ArrayLit(elems) => {
            assert_eq!(elems.len(), 3);
        }
        _ => panic!("expected ArrayLit"),
    }
}

#[test]
fn test_lower_full_program() {
    let mut lowering = AstLowering::new();
    let program = Program {
        declarations: vec![
            Decl {
                kind: DeclKind::Let(LetStmt {
                    name: ident("x"),
                    mutable: false,
                    ty: None,
                    init: Box::new(int_expr(42)),
                }),
                span: span(),
            },
            Decl {
                kind: DeclKind::Fn(FnDecl {
                    name: ident("main"),
                    type_params: vec![],
                    params: vec![],
                    return_type: None,
                    body: Block {
                        stmts: vec![],
                        expr: Some(Box::new(ident_expr("x"))),
                        span: span(),
                    },
                    is_nogc: false,
                    effect_annotation: None,
                    decorators: vec![],
                    vis: cjc_ast::Visibility::Private,
                }),
                span: span(),
            },
        ],
    };
    let hir = lowering.lower_program(&program);
    assert_eq!(hir.items.len(), 2);
    assert!(matches!(hir.items[0], HirItem::Let(_)));
    assert!(matches!(hir.items[1], HirItem::Fn(_)));
}

#[test]
fn test_hir_ids_are_unique() {
    let mut lowering = AstLowering::new();
    let e1 = lowering.lower_expr(&int_expr(1));
    let e2 = lowering.lower_expr(&int_expr(2));
    let e3 = lowering.lower_expr(&int_expr(3));
    assert_ne!(e1.hir_id, e2.hir_id);
    assert_ne!(e2.hir_id, e3.hir_id);
    assert_ne!(e1.hir_id, e3.hir_id);
}

#[test]
fn test_lower_unary_op() {
    let mut lowering = AstLowering::new();
    let ast_expr = Expr {
        kind: ExprKind::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(int_expr(5)),
        },
        span: span(),
    };
    let hir = lowering.lower_expr(&ast_expr);
    match &hir.kind {
        HirExprKind::Unary { op, operand } => {
            assert_eq!(op, &UnaryOp::Neg);
            assert!(matches!(operand.kind, HirExprKind::IntLit(5)));
        }
        _ => panic!("expected Unary"),
    }
}

#[test]
fn test_lower_field_access() {
    let mut lowering = AstLowering::new();
    let ast_expr = Expr {
        kind: ExprKind::Field {
            object: Box::new(ident_expr("point")),
            name: ident("x"),
        },
        span: span(),
    };
    let hir = lowering.lower_expr(&ast_expr);
    match &hir.kind {
        HirExprKind::Field { object, name } => {
            assert!(matches!(object.kind, HirExprKind::Var(ref n) if n == "point"));
            assert_eq!(name, "x");
        }
        _ => panic!("expected Field"),
    }
}

#[test]
fn test_lower_assignment() {
    let mut lowering = AstLowering::new();
    let ast_expr = Expr {
        kind: ExprKind::Assign {
            target: Box::new(ident_expr("x")),
            value: Box::new(int_expr(10)),
        },
        span: span(),
    };
    let hir = lowering.lower_expr(&ast_expr);
    match &hir.kind {
        HirExprKind::Assign { target, value } => {
            assert!(matches!(target.kind, HirExprKind::Var(ref n) if n == "x"));
            assert!(matches!(value.kind, HirExprKind::IntLit(10)));
        }
        _ => panic!("expected Assign"),
    }
}

#[test]
fn test_lower_lambda() {
    let mut lowering = AstLowering::new();
    let ast_expr = Expr {
        kind: ExprKind::Lambda {
            params: vec![Param {
                name: ident("x"),
                ty: type_expr("f64"),
                default: None,
                is_variadic: false,
                span: span(),
            }],
            body: Box::new(ident_expr("x")),
        },
        span: span(),
    };
    let hir = lowering.lower_expr(&ast_expr);
    match &hir.kind {
        HirExprKind::Lambda { params, body } => {
            assert_eq!(params.len(), 1);
            assert_eq!(params[0].name, "x");
            assert_eq!(params[0].ty_name, "f64");
            assert!(matches!(body.kind, HirExprKind::Var(ref n) if n == "x"));
        }
        _ => panic!("expected Lambda"),
    }
}

#[test]
fn test_lower_return_stmt() {
    let mut lowering = AstLowering::new();
    let ast_stmt = Stmt {
        kind: StmtKind::Return(Some(int_expr(42))),
        span: span(),
    };
    let hir_stmt = lowering.lower_stmt(&ast_stmt);
    match &hir_stmt.kind {
        HirStmtKind::Return(Some(expr)) => {
            assert!(matches!(expr.kind, HirExprKind::IntLit(42)));
        }
        _ => panic!("expected Return"),
    }
}

#[test]
fn test_lower_call_with_args() {
    let mut lowering = AstLowering::new();
    let ast_expr = Expr {
        kind: ExprKind::Call {
            callee: Box::new(ident_expr("add")),
            args: vec![
                CallArg { name: None, value: int_expr(1), span: span() },
                CallArg { name: None, value: int_expr(2), span: span() },
            ],
        },
        span: span(),
    };
    let hir = lowering.lower_expr(&ast_expr);
    match &hir.kind {
        HirExprKind::Call { callee, args } => {
            assert!(matches!(callee.kind, HirExprKind::Var(ref n) if n == "add"));
            assert_eq!(args.len(), 2);
        }
        _ => panic!("expected Call"),
    }
}
