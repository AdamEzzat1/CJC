// Milestone 2.4 — Parity Tests (Optimized vs Unoptimized)
//
// Runs the same programs through opt-off and opt-on MIR pipelines.
// Compares results via bit-identical matching:
//   - Integers: exact equality
//   - Floats: to_bits() comparison
//   - Tuples/Arrays/Structs: recursive structural equality
//   - Output strings: exact match

use cjc_ast::*;
use cjc_mir_exec::{run_program, run_program_optimized, run_program_with_executor, run_program_optimized_with_executor};
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

fn float_expr(v: f64) -> Expr {
    Expr { kind: ExprKind::FloatLit(v), span: span() }
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

fn expr_stmt(expr: Expr) -> Stmt {
    Stmt { kind: StmtKind::Expr(expr), span: span() }
}

fn return_stmt(expr: Option<Expr>) -> Stmt {
    Stmt { kind: StmtKind::Return(expr), span: span() }
}

fn assign_expr(target: Expr, value: Expr) -> Expr {
    Expr {
        kind: ExprKind::Assign {
            target: Box::new(target),
            value: Box::new(value),
        },
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
            effect_annotation: None,
        }),
        span: span(),
    }
}

fn make_block(stmts: Vec<Stmt>, expr: Option<Expr>) -> Block {
    Block { stmts, expr: expr.map(Box::new), span: span() }
}

/// Compare two values for bit-identical equality.
fn bit_identical(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => a.to_bits() == b.to_bits(),
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Void, Value::Void) => true,
        (Value::Tuple(a), Value::Tuple(b)) => {
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| bit_identical(x, y))
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| bit_identical(x, y))
        }
        (Value::Tensor(a), Value::Tensor(b)) => {
            a.shape() == b.shape()
                && a.to_vec()
                    .iter()
                    .zip(b.to_vec().iter())
                    .all(|(x, y)| x.to_bits() == y.to_bits())
        }
        _ => false,
    }
}

/// Run a parity check: same program through unoptimized and optimized pipelines.
fn assert_parity(program: &Program, seed: u64) {
    let unopt = run_program(program, seed).expect("unopt failed");
    let opt = run_program_optimized(program, seed).expect("opt failed");
    assert!(
        bit_identical(&unopt, &opt),
        "parity failure: unopt={:?} vs opt={:?}",
        unopt,
        opt
    );
}

/// Run a parity check with output comparison.
fn assert_parity_with_output(program: &Program, seed: u64) {
    let (unopt_val, unopt_exec) =
        run_program_with_executor(program, seed).expect("unopt failed");
    let (opt_val, opt_exec) =
        run_program_optimized_with_executor(program, seed).expect("opt failed");
    assert!(
        bit_identical(&unopt_val, &opt_val),
        "value parity failure: unopt={:?} vs opt={:?}",
        unopt_val,
        opt_val
    );
    assert_eq!(
        unopt_exec.output, opt_exec.output,
        "output parity failure"
    );
}

// ===========================================================================
// Parity Tests
// ===========================================================================

#[test]
fn parity_arithmetic() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![],
                Some(binary(BinOp::Add, int_expr(2), int_expr(3))),
            ),
        )],
    };
    assert_parity(&program, 0);
}

#[test]
fn parity_nested_arithmetic() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![],
                Some(binary(
                    BinOp::Mul,
                    binary(BinOp::Add, int_expr(2), int_expr(3)),
                    binary(BinOp::Sub, int_expr(10), int_expr(4)),
                )),
            ),
        )],
    };
    assert_parity(&program, 0);
}

#[test]
fn parity_float_arithmetic() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![],
                Some(binary(BinOp::Add, float_expr(1.1), float_expr(2.2))),
            ),
        )],
    };
    assert_parity(&program, 0);
}

#[test]
fn parity_mixed_int_float() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![],
                Some(binary(BinOp::Mul, float_expr(3.14), float_expr(2.0))),
            ),
        )],
    };
    assert_parity(&program, 0);
}

#[test]
fn parity_conditional() {
    // Use a non-constant condition to test if/else through both paths.
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![
                    let_stmt("x", int_expr(10)),
                    Stmt {
                        kind: StmtKind::If(IfStmt {
                            condition: binary(BinOp::Gt, ident_expr("x"), int_expr(5)),
                            then_block: make_block(vec![], Some(int_expr(42))),
                            else_branch: Some(ElseBranch::Else(
                                make_block(vec![], Some(int_expr(0))),
                            )),
                        }),
                        span: span(),
                    },
                ],
                None,
            ),
        )],
    };
    assert_parity(&program, 0);
}

#[test]
fn parity_while_loop() {
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
                            condition: binary(BinOp::Lt, ident_expr("i"), int_expr(10)),
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
    assert_parity(&program, 0);
}

#[test]
fn parity_function_call() {
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
                    Some(call(ident_expr("double"), vec![int_expr(21)])),
                ),
            ),
        ],
    };
    assert_parity(&program, 0);
}

#[test]
fn parity_recursive_factorial() {
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
                    Some(call(ident_expr("factorial"), vec![int_expr(10)])),
                ),
            ),
        ],
    };
    assert_parity(&program, 0);
}

#[test]
fn parity_print_output() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![
                    expr_stmt(call(
                        ident_expr("print"),
                        vec![Expr {
                            kind: ExprKind::StringLit("hello".to_string()),
                            span: span(),
                        }],
                    )),
                    expr_stmt(call(
                        ident_expr("print"),
                        vec![int_expr(42)],
                    )),
                ],
                None,
            ),
        )],
    };
    assert_parity_with_output(&program, 0);
}

#[test]
fn parity_tensor_zeros() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![let_stmt(
                    "t",
                    call(
                        field_expr(ident_expr("Tensor"), "zeros"),
                        vec![array_expr(vec![int_expr(3), int_expr(4)])],
                    ),
                )],
                Some(call(
                    field_expr(ident_expr("Tensor"), "zeros"),
                    vec![array_expr(vec![int_expr(2), int_expr(3)])],
                )),
            ),
        )],
    };
    assert_parity(&program, 0);
}

#[test]
fn parity_tensor_randn_seeded() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![],
                Some(call(
                    field_expr(ident_expr("Tensor"), "randn"),
                    vec![array_expr(vec![int_expr(5)])],
                )),
            ),
        )],
    };
    // Same seed => same tensor
    assert_parity(&program, 42);
}

// ---------------------------------------------------------------------------
// Kahan-summation focused tests
// ---------------------------------------------------------------------------

#[test]
fn parity_kahan_alternating_large_small() {
    // Verify that optimizer does not reorder summation.
    // This test creates alternating large+small float values.
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![
                    let_stmt("a", float_expr(1e15)),
                    let_stmt("b", float_expr(1.0)),
                    let_stmt("c", float_expr(-1e15)),
                    let_stmt("d", float_expr(1.0)),
                ],
                Some(binary(
                    BinOp::Add,
                    binary(
                        BinOp::Add,
                        binary(BinOp::Add, ident_expr("a"), ident_expr("b")),
                        ident_expr("c"),
                    ),
                    ident_expr("d"),
                )),
            ),
        )],
    };
    assert_parity(&program, 0);
}

#[test]
fn parity_nan_behavior() {
    // 0.0 / 0.0 should produce NaN in both paths
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![],
                Some(binary(BinOp::Div, float_expr(0.0), float_expr(0.0))),
            ),
        )],
    };
    let unopt = run_program(&program, 0).expect("unopt");
    let opt = run_program_optimized(&program, 0).expect("opt");
    match (&unopt, &opt) {
        (Value::Float(a), Value::Float(b)) => {
            assert!(a.is_nan() && b.is_nan(), "both should be NaN");
            assert_eq!(a.to_bits(), b.to_bits(), "NaN bit pattern mismatch");
        }
        _ => panic!("expected Float"),
    }
}

#[test]
fn parity_negative_zero() {
    // -0.0 should be bit-identical
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![],
                Some(float_expr(-0.0)),
            ),
        )],
    };
    let unopt = run_program(&program, 0).expect("unopt");
    let opt = run_program_optimized(&program, 0).expect("opt");
    match (&unopt, &opt) {
        (Value::Float(a), Value::Float(b)) => {
            assert_eq!(a.to_bits(), b.to_bits(), "-0.0 bit pattern mismatch");
        }
        _ => panic!("expected Float"),
    }
}

#[test]
fn parity_infinity() {
    // 1.0 / 0.0 => inf
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![],
                Some(binary(BinOp::Div, float_expr(1.0), float_expr(0.0))),
            ),
        )],
    };
    assert_parity(&program, 0);
}
