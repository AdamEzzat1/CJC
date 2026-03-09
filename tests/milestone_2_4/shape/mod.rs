// Milestone 2.4 — Shape Metadata and Dimension Check Tests
//
// Ensures that MIR optimizations preserve:
// - Tensor shape metadata
// - Dimension check errors at the same point
// - Shape mismatch still errors correctly

use cjc_ast::*;
use cjc_mir_exec::{run_program, run_program_optimized};
use cjc_runtime::Value;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn span() -> Span {
    Span::dummy()
}

fn ident(name: &str) -> Ident {
    Ident::dummy(name)
}

fn int_expr(v: i64) -> Expr {
    Expr { kind: ExprKind::IntLit(v), span: span() }
}

fn ident_expr(name: &str) -> Expr {
    Expr { kind: ExprKind::Ident(ident(name)), span: span() }
}

fn call(callee: Expr, args: Vec<Expr>) -> Expr {
    let call_args: Vec<CallArg> = args
        .into_iter()
        .map(|value| CallArg { name: None, value, span: span() })
        .collect();
    Expr {
        kind: ExprKind::Call {
            callee: Box::new(callee),
            args: call_args,
        },
        span: span(),
    }
}

fn field_expr(object: Expr, name: &str) -> Expr {
    Expr {
        kind: ExprKind::Field {
            object: Box::new(object),
            name: ident(name),
        },
        span: span(),
    }
}

fn array_expr(elems: Vec<Expr>) -> Expr {
    Expr { kind: ExprKind::ArrayLit(elems), span: span() }
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

fn let_stmt(name: &str, init: Expr) -> Stmt {
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

fn dummy_type_expr() -> TypeExpr {
    TypeExpr {
        kind: TypeExprKind::Named { name: ident("i64"), args: vec![] },
        span: span(),
    }
}

fn make_param(name: &str) -> Param {
    Param { name: ident(name), ty: dummy_type_expr(), default: None, span: span() }
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
            effect_annotation: None,
            decorators: vec![],
        }),
        span: span(),
    }
}

fn make_block(stmts: Vec<Stmt>, expr: Option<Expr>) -> Block {
    Block { stmts, expr: expr.map(Box::new), span: span() }
}

// ===========================================================================
// Shape Preservation Tests
// ===========================================================================

#[test]
fn shape_tensor_zeros_preserved() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![],
                Some(call(
                    field_expr(ident_expr("Tensor"), "zeros"),
                    vec![array_expr(vec![int_expr(3), int_expr(4)])],
                )),
            ),
        )],
    };

    let unopt = run_program(&program, 0).expect("unopt");
    let opt = run_program_optimized(&program, 0).expect("opt");

    match (&unopt, &opt) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            assert_eq!(a.shape(), b.shape(), "shape mismatch");
            assert_eq!(a.shape(), &[3, 4]);
        }
        _ => panic!("expected Tensor"),
    }
}

#[test]
fn shape_tensor_ones_preserved() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![],
                Some(call(
                    field_expr(ident_expr("Tensor"), "ones"),
                    vec![array_expr(vec![int_expr(2), int_expr(5)])],
                )),
            ),
        )],
    };

    let unopt = run_program(&program, 0).expect("unopt");
    let opt = run_program_optimized(&program, 0).expect("opt");

    match (&unopt, &opt) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            assert_eq!(a.shape(), b.shape(), "shape mismatch");
            assert_eq!(a.shape(), &[2, 5]);
        }
        _ => panic!("expected Tensor"),
    }
}

#[test]
fn shape_tensor_add_preserves_shape() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![
                    let_stmt(
                        "a",
                        call(
                            field_expr(ident_expr("Tensor"), "zeros"),
                            vec![array_expr(vec![int_expr(3)])],
                        ),
                    ),
                    let_stmt(
                        "b",
                        call(
                            field_expr(ident_expr("Tensor"), "ones"),
                            vec![array_expr(vec![int_expr(3)])],
                        ),
                    ),
                ],
                Some(binary(BinOp::Add, ident_expr("a"), ident_expr("b"))),
            ),
        )],
    };

    let unopt = run_program(&program, 0).expect("unopt");
    let opt = run_program_optimized(&program, 0).expect("opt");

    match (&unopt, &opt) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            assert_eq!(a.shape(), b.shape(), "shape mismatch after add");
            assert_eq!(a.shape(), &[3]);
        }
        _ => panic!("expected Tensor"),
    }
}

#[test]
fn shape_mismatch_still_errors_unopt() {
    // Adding tensors of different shapes should error.
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![
                    let_stmt(
                        "a",
                        call(
                            field_expr(ident_expr("Tensor"), "zeros"),
                            vec![array_expr(vec![int_expr(3)])],
                        ),
                    ),
                    let_stmt(
                        "b",
                        call(
                            field_expr(ident_expr("Tensor"), "zeros"),
                            vec![array_expr(vec![int_expr(5)])],
                        ),
                    ),
                ],
                Some(binary(BinOp::Add, ident_expr("a"), ident_expr("b"))),
            ),
        )],
    };

    assert!(run_program(&program, 0).is_err());
}

#[test]
fn shape_mismatch_still_errors_opt() {
    // Same shape mismatch must also error in optimized pipeline.
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![
                    let_stmt(
                        "a",
                        call(
                            field_expr(ident_expr("Tensor"), "zeros"),
                            vec![array_expr(vec![int_expr(3)])],
                        ),
                    ),
                    let_stmt(
                        "b",
                        call(
                            field_expr(ident_expr("Tensor"), "zeros"),
                            vec![array_expr(vec![int_expr(5)])],
                        ),
                    ),
                ],
                Some(binary(BinOp::Add, ident_expr("a"), ident_expr("b"))),
            ),
        )],
    };

    assert!(run_program_optimized(&program, 0).is_err());
}

#[test]
fn shape_matmul_dimension_check_preserved() {
    // matmul with compatible shapes should work.
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![
                    let_stmt(
                        "a",
                        call(
                            field_expr(ident_expr("Tensor"), "zeros"),
                            vec![array_expr(vec![int_expr(2), int_expr(3)])],
                        ),
                    ),
                    let_stmt(
                        "b",
                        call(
                            field_expr(ident_expr("Tensor"), "zeros"),
                            vec![array_expr(vec![int_expr(3), int_expr(4)])],
                        ),
                    ),
                ],
                Some(call(ident_expr("matmul"), vec![ident_expr("a"), ident_expr("b")])),
            ),
        )],
    };

    let unopt = run_program(&program, 0).expect("unopt");
    let opt = run_program_optimized(&program, 0).expect("opt");

    match (&unopt, &opt) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            assert_eq!(a.shape(), b.shape(), "matmul result shape mismatch");
            assert_eq!(a.shape(), &[2, 4]);
        }
        _ => panic!("expected Tensor"),
    }
}

#[test]
fn shape_matmul_incompatible_still_errors() {
    // matmul with incompatible inner dimensions should error in both pipelines.
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![
                    let_stmt(
                        "a",
                        call(
                            field_expr(ident_expr("Tensor"), "zeros"),
                            vec![array_expr(vec![int_expr(2), int_expr(3)])],
                        ),
                    ),
                    let_stmt(
                        "b",
                        call(
                            field_expr(ident_expr("Tensor"), "zeros"),
                            vec![array_expr(vec![int_expr(5), int_expr(4)])],
                        ),
                    ),
                ],
                Some(call(ident_expr("matmul"), vec![ident_expr("a"), ident_expr("b")])),
            ),
        )],
    };

    assert!(run_program(&program, 0).is_err());
    assert!(run_program_optimized(&program, 0).is_err());
}

#[test]
fn shape_from_vec_preserves_metadata() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![],
                Some(call(
                    field_expr(ident_expr("Tensor"), "from_vec"),
                    vec![
                        array_expr(vec![
                            Expr { kind: ExprKind::FloatLit(1.0), span: span() },
                            Expr { kind: ExprKind::FloatLit(2.0), span: span() },
                            Expr { kind: ExprKind::FloatLit(3.0), span: span() },
                            Expr { kind: ExprKind::FloatLit(4.0), span: span() },
                            Expr { kind: ExprKind::FloatLit(5.0), span: span() },
                            Expr { kind: ExprKind::FloatLit(6.0), span: span() },
                        ]),
                        array_expr(vec![int_expr(2), int_expr(3)]),
                    ],
                )),
            ),
        )],
    };

    let unopt = run_program(&program, 0).expect("unopt");
    let opt = run_program_optimized(&program, 0).expect("opt");

    match (&unopt, &opt) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            assert_eq!(a.shape(), b.shape(), "from_vec shape mismatch");
            assert_eq!(a.shape(), &[2, 3]);
            // Check data equality
            assert_eq!(a.to_vec(), b.to_vec());
        }
        _ => panic!("expected Tensor"),
    }
}
