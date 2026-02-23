// CJC Test Suite — cjc-mir-exec (8 tests)
// Source: crates/cjc-mir-exec/src/lib.rs
// Integration tests for the MIR executor (full AST -> HIR -> MIR -> Execute pipeline).

use cjc_ast::*;
use cjc_mir_exec::{run_program, run_program_with_executor};
use cjc_runtime::Value;

fn span() -> Span {
    Span::dummy()
}

fn ident(name: &str) -> Ident {
    Ident::dummy(name)
}

fn int_expr(v: i64) -> Expr {
    Expr { kind: ExprKind::IntLit(v), span: span() }
}

fn bool_expr(v: bool) -> Expr {
    Expr { kind: ExprKind::BoolLit(v), span: span() }
}

fn ident_expr(name: &str) -> Expr {
    Expr { kind: ExprKind::Ident(ident(name)), span: span() }
}

fn binary(op: BinOp, left: Expr, right: Expr) -> Expr {
    Expr {
        kind: ExprKind::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        },
        span: span(),
    }
}

fn call(callee: Expr, args: Vec<Expr>) -> Expr {
    let call_args: Vec<CallArg> = args
        .into_iter()
        .map(|value| CallArg { name: None, value, span: span() })
        .collect();
    Expr {
        kind: ExprKind::Call { callee: Box::new(callee), args: call_args },
        span: span(),
    }
}

fn field_expr(object: Expr, name: &str) -> Expr {
    Expr {
        kind: ExprKind::Field { object: Box::new(object), name: ident(name) },
        span: span(),
    }
}

fn pipe_expr(left: Expr, right: Expr) -> Expr {
    Expr {
        kind: ExprKind::Pipe { left: Box::new(left), right: Box::new(right) },
        span: span(),
    }
}

fn array_expr(elems: Vec<Expr>) -> Expr {
    Expr { kind: ExprKind::ArrayLit(elems), span: span() }
}

fn assign_expr(target: Expr, value: Expr) -> Expr {
    Expr {
        kind: ExprKind::Assign { target: Box::new(target), value: Box::new(value) },
        span: span(),
    }
}

fn let_mut_stmt(name: &str, init: Expr) -> Stmt {
    Stmt {
        kind: StmtKind::Let(LetStmt {
            name: ident(name),
            mutable: true,
            ty: None,
            init: Box::new(init),
        }),
        span: span(),
    }
}

fn let_stmt_ast(name: &str, init: Expr) -> Stmt {
    Stmt {
        kind: StmtKind::Let(LetStmt {
            name: ident(name),
            mutable: false,
            ty: None,
            init: Box::new(init),
        }),
        span: span(),
    }
}

fn expr_stmt(expr: Expr) -> Stmt {
    Stmt { kind: StmtKind::Expr(expr), span: span() }
}

fn return_stmt(expr: Option<Expr>) -> Stmt {
    Stmt { kind: StmtKind::Return(expr), span: span() }
}

fn dummy_type_expr() -> TypeExpr {
    TypeExpr {
        kind: TypeExprKind::Named { name: ident("i64"), args: vec![] },
        span: span(),
    }
}

fn make_param(name: &str) -> Param {
    Param { name: ident(name), ty: dummy_type_expr(), span: span() }
}

fn make_fn_decl(name: &str, params: Vec<&str>, body: Block) -> Decl {
    Decl {
        kind: DeclKind::Fn(FnDecl {
            name: ident(name),
            type_params: vec![],
            params: params.into_iter().map(make_param).collect(),
            return_type: None,
            body,
            is_nogc: false,
        }),
        span: span(),
    }
}

fn make_block(stmts: Vec<Stmt>, expr: Option<Expr>) -> Block {
    Block { stmts, expr: expr.map(Box::new), span: span() }
}

// -- Tests ---------------------------------------------------------------

#[test]
fn test_mir_pipeline_arithmetic() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(vec![], Some(binary(BinOp::Add, int_expr(2), int_expr(3)))),
        )],
    };
    let result = run_program(&program, 0).unwrap();
    assert!(matches!(result, Value::Int(5)));
}

#[test]
fn test_mir_pipeline_function_call() {
    let program = Program {
        declarations: vec![
            make_fn_decl(
                "add",
                vec!["a", "b"],
                make_block(
                    vec![],
                    Some(binary(BinOp::Add, ident_expr("a"), ident_expr("b"))),
                ),
            ),
            make_fn_decl(
                "main",
                vec![],
                make_block(
                    vec![],
                    Some(call(ident_expr("add"), vec![int_expr(3), int_expr(4)])),
                ),
            ),
        ],
    };
    let result = run_program(&program, 0).unwrap();
    assert!(matches!(result, Value::Int(7)));
}

#[test]
fn test_mir_pipeline_while_loop() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![
                    let_mut_stmt("i", int_expr(0)),
                    let_mut_stmt("sum", int_expr(0)),
                    Stmt {
                        kind: StmtKind::While(WhileStmt {
                            condition: binary(BinOp::Lt, ident_expr("i"), int_expr(5)),
                            body: make_block(
                                vec![
                                    expr_stmt(assign_expr(
                                        ident_expr("sum"),
                                        binary(BinOp::Add, ident_expr("sum"), ident_expr("i")),
                                    )),
                                    expr_stmt(assign_expr(
                                        ident_expr("i"),
                                        binary(BinOp::Add, ident_expr("i"), int_expr(1)),
                                    )),
                                ],
                                None,
                            ),
                        }),
                        span: span(),
                    },
                ],
                Some(ident_expr("sum")),
            ),
        )],
    };
    let result = run_program(&program, 0).unwrap();
    assert!(matches!(result, Value::Int(10)));
}

#[test]
fn test_mir_pipeline_pipe_desugaring() {
    let program = Program {
        declarations: vec![
            make_fn_decl(
                "double",
                vec!["x"],
                make_block(
                    vec![],
                    Some(binary(BinOp::Mul, ident_expr("x"), int_expr(2))),
                ),
            ),
            make_fn_decl(
                "main",
                vec![],
                make_block(
                    vec![],
                    Some(pipe_expr(int_expr(5), call(ident_expr("double"), vec![]))),
                ),
            ),
        ],
    };
    let result = run_program(&program, 0).unwrap();
    assert!(matches!(result, Value::Int(10)));
}

#[test]
fn test_mir_pipeline_tensor_operations() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![let_stmt_ast(
                    "t",
                    call(
                        field_expr(ident_expr("Tensor"), "zeros"),
                        vec![array_expr(vec![int_expr(2), int_expr(3)])],
                    ),
                )],
                Some(call(
                    field_expr(ident_expr("Tensor"), "zeros"),
                    vec![array_expr(vec![int_expr(2), int_expr(3)])],
                )),
            ),
        )],
    };
    let result = run_program(&program, 0).unwrap();
    match &result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 3]);
            assert_eq!(t.len(), 6);
        }
        _ => panic!("expected Tensor"),
    }
}

#[test]
fn test_mir_pipeline_recursive_factorial() {
    let fact_body = make_block(
        vec![
            Stmt {
                kind: StmtKind::If(IfStmt {
                    condition: binary(BinOp::Le, ident_expr("n"), int_expr(1)),
                    then_block: make_block(vec![return_stmt(Some(int_expr(1)))], None),
                    else_branch: None,
                }),
                span: span(),
            },
            return_stmt(Some(binary(
                BinOp::Mul,
                ident_expr("n"),
                call(
                    ident_expr("factorial"),
                    vec![binary(BinOp::Sub, ident_expr("n"), int_expr(1))],
                ),
            ))),
        ],
        None,
    );

    let program = Program {
        declarations: vec![
            make_fn_decl("factorial", vec!["n"], fact_body),
            make_fn_decl(
                "main",
                vec![],
                make_block(
                    vec![],
                    Some(call(ident_expr("factorial"), vec![int_expr(5)])),
                ),
            ),
        ],
    };
    let result = run_program(&program, 0).unwrap();
    assert!(matches!(result, Value::Int(120)));
}

#[test]
fn test_mir_pipeline_if_else() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![Stmt {
                    kind: StmtKind::If(IfStmt {
                        condition: bool_expr(false),
                        then_block: make_block(vec![], Some(int_expr(1))),
                        else_branch: Some(ElseBranch::Else(
                            make_block(vec![], Some(int_expr(2))),
                        )),
                    }),
                    span: span(),
                }],
                None,
            ),
        )],
    };
    let result = run_program(&program, 0).unwrap();
    assert!(matches!(result, Value::Int(2)));
}

#[test]
fn test_mir_pipeline_print_output() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![expr_stmt(call(
                    ident_expr("print"),
                    vec![Expr {
                        kind: ExprKind::StringLit("hello world".to_string()),
                        span: span(),
                    }],
                ))],
                None,
            ),
        )],
    };
    let (_, executor) = run_program_with_executor(&program, 0).unwrap();
    assert_eq!(executor.output, vec!["hello world"]);
}
