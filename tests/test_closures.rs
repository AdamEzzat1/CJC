// CJC Test Suite — Milestone 2.1: Closures
// Tests for closure parsing, capture analysis, lambda-lifting, and execution.

use cjc_ast::*;
use cjc_hir::{AstLowering, CaptureMode, HirExprKind};
use cjc_mir_exec::{run_program, run_program_with_executor};
use cjc_runtime::Value;

// ── AST helpers ───────────────────────────────────────────────────────

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

#[allow(dead_code)]
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

fn f64_type() -> TypeExpr {
    TypeExpr {
        kind: TypeExprKind::Named { name: ident("f64"), args: vec![] },
        span: span(),
    }
}

fn i64_type() -> TypeExpr {
    TypeExpr {
        kind: TypeExprKind::Named { name: ident("i64"), args: vec![] },
        span: span(),
    }
}

fn make_param(name: &str, ty: TypeExpr) -> Param {
    Param { name: ident(name), ty, default: None, is_variadic: false, span: span() }
}

fn make_fn_decl(name: &str, params: Vec<Param>, body: Block) -> Decl {
    Decl {
        kind: DeclKind::Fn(FnDecl {
            name: ident(name),
            type_params: vec![],
            params,
            return_type: None,
            body,
            is_nogc: false,
            effect_annotation: None,
            decorators: vec![],
            vis: cjc_ast::Visibility::Private,
        }),
        span: span(),
    }
}

fn make_block(stmts: Vec<Stmt>, expr: Option<Expr>) -> Block {
    Block { stmts, expr: expr.map(Box::new), span: span() }
}

/// Create a lambda expression: |params| body
fn lambda(params: Vec<Param>, body: Expr) -> Expr {
    Expr {
        kind: ExprKind::Lambda {
            params,
            body: Box::new(body),
        },
        span: span(),
    }
}

/// Create a nogc block statement wrapping the given block
fn nogc_block(stmts: Vec<Stmt>, expr: Option<Expr>) -> Stmt {
    Stmt {
        kind: StmtKind::NoGcBlock(make_block(stmts, expr)),
        span: span(),
    }
}

// ── Lexer Tests ─────────────────────────────────────────────────────

#[test]
fn test_lexer_pipe_token() {
    use cjc_lexer::{Lexer, TokenKind};
    let lexer = Lexer::new("|x: f64, y: f64| x + y");
    let (tokens, diags) = lexer.tokenize();
    assert!(!diags.has_errors());
    assert_eq!(tokens[0].kind, TokenKind::Pipe);
    // After params, the second | is also Pipe
    let pipe_count = tokens.iter().filter(|t| t.kind == TokenKind::Pipe).count();
    assert_eq!(pipe_count, 2);
}

#[test]
fn test_lexer_pipe_vs_pipepipe_vs_pipegt() {
    use cjc_lexer::{Lexer, TokenKind};
    let lexer = Lexer::new("|x| x || true |> f");
    let (tokens, diags) = lexer.tokenize();
    assert!(!diags.has_errors());
    assert_eq!(tokens[0].kind, TokenKind::Pipe);    // |
    assert_eq!(tokens[2].kind, TokenKind::Pipe);    // |
    assert_eq!(tokens[4].kind, TokenKind::PipePipe); // ||
    assert_eq!(tokens[6].kind, TokenKind::PipeGt);  // |>
}

// ── Parser Tests ────────────────────────────────────────────────────

#[test]
fn test_parse_lambda_single_param() {
    let (program, diags) = cjc_parser::parse_source("let f = |x: f64| x;");
    assert!(!diags.has_errors(), "parse had errors");
    // The program should have a let decl with a Lambda init
    match &program.declarations[0].kind {
        DeclKind::Let(l) => match &l.init.kind {
            ExprKind::Lambda { params, body } => {
                assert_eq!(params.len(), 1);
                assert_eq!(params[0].name.name, "x");
                assert!(matches!(body.kind, ExprKind::Ident(_)));
            }
            _ => panic!("expected Lambda, got {:?}", l.init.kind),
        },
        _ => panic!("expected Let decl"),
    }
}

#[test]
fn test_parse_lambda_multi_param() {
    let (program, diags) = cjc_parser::parse_source("let f = |x: f64, y: f64| x + y;");
    assert!(!diags.has_errors(), "parse had errors");
    match &program.declarations[0].kind {
        DeclKind::Let(l) => match &l.init.kind {
            ExprKind::Lambda { params, .. } => {
                assert_eq!(params.len(), 2);
                assert_eq!(params[0].name.name, "x");
                assert_eq!(params[1].name.name, "y");
            }
            _ => panic!("expected Lambda"),
        },
        _ => panic!("expected Let decl"),
    }
}

#[test]
fn test_parse_lambda_no_params() {
    let (program, diags) = cjc_parser::parse_source("let f = || 42;");
    assert!(!diags.has_errors(), "parse had errors");
    match &program.declarations[0].kind {
        DeclKind::Let(l) => match &l.init.kind {
            ExprKind::Lambda { params, body } => {
                assert_eq!(params.len(), 0);
                assert!(matches!(body.kind, ExprKind::IntLit(42)));
            }
            _ => panic!("expected Lambda"),
        },
        _ => panic!("expected Let decl"),
    }
}

#[test]
fn test_parse_lambda_block_body() {
    let (program, diags) = cjc_parser::parse_source("let f = |x: f64| { let y = x; y };");
    assert!(!diags.has_errors(), "parse had errors");
    match &program.declarations[0].kind {
        DeclKind::Let(l) => match &l.init.kind {
            ExprKind::Lambda { params, body } => {
                assert_eq!(params.len(), 1);
                assert!(matches!(body.kind, ExprKind::Block(_)));
            }
            _ => panic!("expected Lambda with block body"),
        },
        _ => panic!("expected Let decl"),
    }
}

// ── HIR Capture Analysis Tests ──────────────────────────────────────

#[test]
fn test_hir_lambda_no_captures() {
    // |x: f64| x * 2.0  — no captures
    let program = Program {
        declarations: vec![
            Decl {
                kind: DeclKind::Stmt(expr_stmt(lambda(
                    vec![make_param("x", f64_type())],
                    binary(BinOp::Mul, ident_expr("x"), float_expr(2.0)),
                ))),
                span: span(),
            },
        ],
    };
    let mut lowering = AstLowering::new();
    let hir = lowering.lower_program(&program);
    // Should produce a Lambda (not Closure) since no captures
    let stmt = &hir.items[0];
    match stmt {
        cjc_hir::HirItem::Stmt(s) => match &s.kind {
            cjc_hir::HirStmtKind::Expr(e) => {
                assert!(
                    matches!(e.kind, HirExprKind::Lambda { .. }),
                    "expected Lambda (no captures), got {:?}",
                    e.kind
                );
            }
            _ => panic!("expected Expr stmt"),
        },
        _ => panic!("expected Stmt item"),
    }
}

#[test]
fn test_hir_closure_single_capture_ref() {
    // let scale = 2.0; let f = |x: f64| x * scale
    // `scale` should be captured by Ref
    let program = Program {
        declarations: vec![
            Decl {
                kind: DeclKind::Stmt(let_stmt("scale", float_expr(2.0))),
                span: span(),
            },
            Decl {
                kind: DeclKind::Stmt(let_stmt(
                    "f",
                    lambda(
                        vec![make_param("x", f64_type())],
                        binary(BinOp::Mul, ident_expr("x"), ident_expr("scale")),
                    ),
                )),
                span: span(),
            },
        ],
    };
    let mut lowering = AstLowering::new();
    let hir = lowering.lower_program(&program);
    // The second item is a Stmt(Let { init: Closure { captures: [scale/Ref] } })
    match &hir.items[1] {
        cjc_hir::HirItem::Stmt(s) => match &s.kind {
            cjc_hir::HirStmtKind::Let { init, .. } => match &init.kind {
                HirExprKind::Closure { captures, .. } => {
                    assert_eq!(captures.len(), 1);
                    assert_eq!(captures[0].name, "scale");
                    assert_eq!(captures[0].mode, CaptureMode::Ref);
                }
                _ => panic!("expected Closure, got {:?}", init.kind),
            },
            _ => panic!("expected Let stmt"),
        },
        _ => panic!("expected Stmt item"),
    }
}

#[test]
fn test_hir_closure_nogc_capture_clone() {
    // nogc { let bias = 1.0; let f = |x: f64| x + bias }
    // `bias` inside nogc should be captured by Clone
    let program = Program {
        declarations: vec![
            Decl {
                kind: DeclKind::Stmt(nogc_block(
                    vec![
                        let_stmt("bias", float_expr(1.0)),
                        let_stmt(
                            "f",
                            lambda(
                                vec![make_param("x", f64_type())],
                                binary(BinOp::Add, ident_expr("x"), ident_expr("bias")),
                            ),
                        ),
                    ],
                    None,
                )),
                span: span(),
            },
        ],
    };
    let mut lowering = AstLowering::new();
    let hir = lowering.lower_program(&program);
    // Navigate: items[0] -> Stmt(NoGcBlock) -> stmts[1] -> Let { init: Closure }
    match &hir.items[0] {
        cjc_hir::HirItem::Stmt(s) => match &s.kind {
            cjc_hir::HirStmtKind::NoGcBlock(block) => {
                match &block.stmts[1].kind {
                    cjc_hir::HirStmtKind::Let { init, .. } => match &init.kind {
                        HirExprKind::Closure { captures, .. } => {
                            assert_eq!(captures.len(), 1);
                            assert_eq!(captures[0].name, "bias");
                            assert_eq!(captures[0].mode, CaptureMode::Clone);
                        }
                        _ => panic!("expected Closure in nogc, got {:?}", init.kind),
                    },
                    _ => panic!("expected Let"),
                }
            }
            _ => panic!("expected NoGcBlock"),
        },
        _ => panic!("expected Stmt"),
    }
}

#[test]
fn test_hir_closure_multiple_captures() {
    // let a = 1; let b = 2; let f = |x: i64| x + a + b
    let program = Program {
        declarations: vec![
            Decl { kind: DeclKind::Stmt(let_stmt("a", int_expr(1))), span: span() },
            Decl { kind: DeclKind::Stmt(let_stmt("b", int_expr(2))), span: span() },
            Decl {
                kind: DeclKind::Stmt(let_stmt(
                    "f",
                    lambda(
                        vec![make_param("x", i64_type())],
                        binary(
                            BinOp::Add,
                            binary(BinOp::Add, ident_expr("x"), ident_expr("a")),
                            ident_expr("b"),
                        ),
                    ),
                )),
                span: span(),
            },
        ],
    };
    let mut lowering = AstLowering::new();
    let hir = lowering.lower_program(&program);
    match &hir.items[2] {
        cjc_hir::HirItem::Stmt(s) => match &s.kind {
            cjc_hir::HirStmtKind::Let { init, .. } => match &init.kind {
                HirExprKind::Closure { captures, .. } => {
                    assert_eq!(captures.len(), 2);
                    let names: Vec<&str> = captures.iter().map(|c| c.name.as_str()).collect();
                    assert!(names.contains(&"a"));
                    assert!(names.contains(&"b"));
                }
                _ => panic!("expected Closure with 2 captures"),
            },
            _ => panic!("expected Let"),
        },
        _ => panic!("expected Stmt"),
    }
}

#[test]
fn test_hir_closure_skips_known_functions() {
    // fn helper(x: i64) -> i64 { x }
    // let f = |x: i64| helper(x)
    // `helper` should NOT be captured — it's a known function
    let program = Program {
        declarations: vec![
            make_fn_decl(
                "helper",
                vec![make_param("x", i64_type())],
                make_block(vec![], Some(ident_expr("x"))),
            ),
            Decl {
                kind: DeclKind::Stmt(let_stmt(
                    "f",
                    lambda(
                        vec![make_param("x", i64_type())],
                        call(ident_expr("helper"), vec![ident_expr("x")]),
                    ),
                )),
                span: span(),
            },
        ],
    };
    let mut lowering = AstLowering::new();
    let hir = lowering.lower_program(&program);
    match &hir.items[1] {
        cjc_hir::HirItem::Stmt(s) => match &s.kind {
            cjc_hir::HirStmtKind::Let { init, .. } => {
                // Should be Lambda (no captures) since helper is known
                assert!(
                    matches!(init.kind, HirExprKind::Lambda { .. }),
                    "expected Lambda (helper is known function), got {:?}",
                    init.kind
                );
            }
            _ => panic!("expected Let"),
        },
        _ => panic!("expected Stmt"),
    }
}

#[test]
fn test_hir_closure_skips_builtins() {
    // let f = |x: i64| print(x)
    // `print` should NOT be captured
    let program = Program {
        declarations: vec![
            Decl {
                kind: DeclKind::Stmt(let_stmt(
                    "f",
                    lambda(
                        vec![make_param("x", i64_type())],
                        call(ident_expr("print"), vec![ident_expr("x")]),
                    ),
                )),
                span: span(),
            },
        ],
    };
    let mut lowering = AstLowering::new();
    let hir = lowering.lower_program(&program);
    match &hir.items[0] {
        cjc_hir::HirItem::Stmt(s) => match &s.kind {
            cjc_hir::HirStmtKind::Let { init, .. } => {
                assert!(
                    matches!(init.kind, HirExprKind::Lambda { .. }),
                    "expected Lambda (print is builtin), got {:?}",
                    init.kind
                );
            }
            _ => panic!("expected Let"),
        },
        _ => panic!("expected Stmt"),
    }
}

// ── MIR Lambda-Lifting Tests ────────────────────────────────────────

#[test]
fn test_mir_closure_generates_lifted_function() {
    use cjc_mir::MirExprKind;
    // let scale = 2.0; let f = |x: f64| x * scale
    let program = Program {
        declarations: vec![
            Decl { kind: DeclKind::Stmt(let_stmt("scale", float_expr(2.0))), span: span() },
            Decl {
                kind: DeclKind::Stmt(let_stmt(
                    "f",
                    lambda(
                        vec![make_param("x", f64_type())],
                        binary(BinOp::Mul, ident_expr("x"), ident_expr("scale")),
                    ),
                )),
                span: span(),
            },
        ],
    };
    let mut ast_lowering = AstLowering::new();
    let hir = ast_lowering.lower_program(&program);
    let mut hir_to_mir = cjc_mir::HirToMir::new();
    let mir = hir_to_mir.lower_program(&hir);

    // Should have __main + a lifted closure function
    assert!(
        mir.functions.len() >= 2,
        "expected at least 2 functions (main + lifted), got {}",
        mir.functions.len()
    );
    let lifted = mir.functions.iter().find(|f| f.name.starts_with("__closure_"));
    assert!(lifted.is_some(), "expected a __closure_ function");
    let lifted = lifted.unwrap();
    // Lifted function should have capture params + original params
    // capture: scale, original: x => 2 total params
    assert_eq!(lifted.params.len(), 2, "lifted fn should have 2 params (1 capture + 1 original)");
    assert_eq!(lifted.params[0].name, "scale");
    assert_eq!(lifted.params[1].name, "x");

    // The __main function should have a MakeClosure expression
    let main_fn = mir.functions.iter().find(|f| f.name == "__main").unwrap();
    // Check the let stmt init is MakeClosure
    match &main_fn.body.stmts[1] {
        cjc_mir::MirStmt::Let { init, .. } => {
            assert!(
                matches!(init.kind, MirExprKind::MakeClosure { .. }),
                "expected MakeClosure, got {:?}",
                init.kind
            );
        }
        _ => panic!("expected Let stmt"),
    }
}

// ── Full Pipeline Execution Tests ───────────────────────────────────

#[test]
fn test_exec_lambda_no_capture() {
    // fn main() { let f = |x: i64| x * 2; f(5) }
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![let_stmt(
                    "f",
                    lambda(
                        vec![make_param("x", i64_type())],
                        binary(BinOp::Mul, ident_expr("x"), int_expr(2)),
                    ),
                )],
                Some(call(ident_expr("f"), vec![int_expr(5)])),
            ),
        )],
    };
    let result = run_program(&program, 0).unwrap();
    assert!(matches!(result, Value::Int(10)), "expected 10, got {:?}", result);
}

#[test]
fn test_exec_closure_captures_immutable_var() {
    // let scale = 3; fn main() { let f = |x: i64| x * scale; f(4) }
    let program = Program {
        declarations: vec![
            Decl { kind: DeclKind::Stmt(let_stmt("scale", int_expr(3))), span: span() },
            make_fn_decl(
                "main",
                vec![],
                make_block(
                    vec![let_stmt(
                        "f",
                        lambda(
                            vec![make_param("x", i64_type())],
                            binary(BinOp::Mul, ident_expr("x"), ident_expr("scale")),
                        ),
                    )],
                    Some(call(ident_expr("f"), vec![int_expr(4)])),
                ),
            ),
        ],
    };
    let result = run_program(&program, 0).unwrap();
    assert!(matches!(result, Value::Int(12)), "expected 12, got {:?}", result);
}

#[test]
fn test_exec_closure_captures_float() {
    // let bias = 0.5; fn main() { let f = |x: f64| x + bias; f(2.5) }
    let program = Program {
        declarations: vec![
            Decl { kind: DeclKind::Stmt(let_stmt("bias", float_expr(0.5))), span: span() },
            make_fn_decl(
                "main",
                vec![],
                make_block(
                    vec![let_stmt(
                        "f",
                        lambda(
                            vec![make_param("x", f64_type())],
                            binary(BinOp::Add, ident_expr("x"), ident_expr("bias")),
                        ),
                    )],
                    Some(call(ident_expr("f"), vec![float_expr(2.5)])),
                ),
            ),
        ],
    };
    let result = run_program(&program, 0).unwrap();
    match result {
        Value::Float(v) => assert!((v - 3.0).abs() < 1e-10, "expected 3.0, got {v}"),
        other => panic!("expected Float(3.0), got {:?}", other),
    }
}

#[test]
fn test_exec_closure_multiple_captures() {
    // let a = 10; let b = 20; fn main() { let f = |x: i64| x + a + b; f(5) }
    let program = Program {
        declarations: vec![
            Decl { kind: DeclKind::Stmt(let_stmt("a", int_expr(10))), span: span() },
            Decl { kind: DeclKind::Stmt(let_stmt("b", int_expr(20))), span: span() },
            make_fn_decl(
                "main",
                vec![],
                make_block(
                    vec![let_stmt(
                        "f",
                        lambda(
                            vec![make_param("x", i64_type())],
                            binary(
                                BinOp::Add,
                                binary(BinOp::Add, ident_expr("x"), ident_expr("a")),
                                ident_expr("b"),
                            ),
                        ),
                    )],
                    Some(call(ident_expr("f"), vec![int_expr(5)])),
                ),
            ),
        ],
    };
    let result = run_program(&program, 0).unwrap();
    assert!(matches!(result, Value::Int(35)), "expected 35, got {:?}", result);
}

#[test]
fn test_exec_closure_called_multiple_times() {
    // let offset = 100
    // fn main() {
    //   let f = |x: i64| x + offset
    //   let a = f(1)   // 101
    //   let b = f(2)   // 102
    //   a + b           // 203
    // }
    let program = Program {
        declarations: vec![
            Decl { kind: DeclKind::Stmt(let_stmt("offset", int_expr(100))), span: span() },
            make_fn_decl(
                "main",
                vec![],
                make_block(
                    vec![
                        let_stmt(
                            "f",
                            lambda(
                                vec![make_param("x", i64_type())],
                                binary(BinOp::Add, ident_expr("x"), ident_expr("offset")),
                            ),
                        ),
                        let_stmt("a", call(ident_expr("f"), vec![int_expr(1)])),
                        let_stmt("b", call(ident_expr("f"), vec![int_expr(2)])),
                    ],
                    Some(binary(BinOp::Add, ident_expr("a"), ident_expr("b"))),
                ),
            ),
        ],
    };
    let result = run_program(&program, 0).unwrap();
    assert!(matches!(result, Value::Int(203)), "expected 203, got {:?}", result);
}

#[test]
fn test_exec_closure_in_nogc_block() {
    // nogc {
    //   let bias = 1.0
    //   let f = |x: f64| x + bias
    //   f(3.0)
    // }
    // Should work — captures by Clone inside nogc
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![nogc_block(
                    vec![
                        let_stmt("bias", float_expr(1.0)),
                        let_stmt(
                            "f",
                            lambda(
                                vec![make_param("x", f64_type())],
                                binary(BinOp::Add, ident_expr("x"), ident_expr("bias")),
                            ),
                        ),
                    ],
                    Some(call(ident_expr("f"), vec![float_expr(3.0)])),
                )],
                None,
            ),
        )],
    };
    let result = run_program(&program, 0).unwrap();
    match result {
        Value::Float(v) => assert!((v - 4.0).abs() < 1e-10, "expected 4.0, got {v}"),
        other => panic!("expected Float(4.0), got {:?}", other),
    }
}

#[test]
fn test_exec_lambda_zero_params() {
    // let val = 42; fn main() { let f = || val; f() }
    let program = Program {
        declarations: vec![
            Decl { kind: DeclKind::Stmt(let_stmt("val", int_expr(42))), span: span() },
            make_fn_decl(
                "main",
                vec![],
                make_block(
                    vec![let_stmt(
                        "f",
                        lambda(vec![], ident_expr("val")),
                    )],
                    Some(call(ident_expr("f"), vec![])),
                ),
            ),
        ],
    };
    let result = run_program(&program, 0).unwrap();
    assert!(matches!(result, Value::Int(42)), "expected 42, got {:?}", result);
}

#[test]
fn test_exec_closure_captures_function_local() {
    // fn main() {
    //   let multiplier = 5
    //   let f = |x: i64| x * multiplier
    //   f(7)
    // }
    let program = Program {
        declarations: vec![make_fn_decl(
            "main",
            vec![],
            make_block(
                vec![
                    let_stmt("multiplier", int_expr(5)),
                    let_stmt(
                        "f",
                        lambda(
                            vec![make_param("x", i64_type())],
                            binary(BinOp::Mul, ident_expr("x"), ident_expr("multiplier")),
                        ),
                    ),
                ],
                Some(call(ident_expr("f"), vec![int_expr(7)])),
            ),
        )],
    };
    let result = run_program(&program, 0).unwrap();
    assert!(matches!(result, Value::Int(35)), "expected 35, got {:?}", result);
}

#[test]
fn test_exec_closure_print_output() {
    // let greeting = "hello"
    // fn main() {
    //   let f = |x: i64| print(greeting)
    //   f(0)
    // }
    let program = Program {
        declarations: vec![
            Decl {
                kind: DeclKind::Stmt(let_stmt(
                    "greeting",
                    Expr { kind: ExprKind::StringLit("hello".to_string()), span: span() },
                )),
                span: span(),
            },
            make_fn_decl(
                "main",
                vec![],
                make_block(
                    vec![
                        let_stmt(
                            "f",
                            lambda(
                                vec![make_param("x", i64_type())],
                                call(ident_expr("print"), vec![ident_expr("greeting")]),
                            ),
                        ),
                        expr_stmt(call(ident_expr("f"), vec![int_expr(0)])),
                    ],
                    None,
                ),
            ),
        ],
    };
    let (_, executor) = run_program_with_executor(&program, 0).unwrap();
    assert_eq!(executor.output, vec!["hello"]);
}

#[test]
fn test_exec_closure_with_function_call_in_body() {
    // fn double(x: i64) -> i64 { x * 2 }
    // let offset = 10
    // fn main() { let f = |x: i64| double(x) + offset; f(5) }
    // => double(5) + 10 = 10 + 10 = 20
    let program = Program {
        declarations: vec![
            make_fn_decl(
                "double",
                vec![make_param("x", i64_type())],
                make_block(vec![], Some(binary(BinOp::Mul, ident_expr("x"), int_expr(2)))),
            ),
            Decl { kind: DeclKind::Stmt(let_stmt("offset", int_expr(10))), span: span() },
            make_fn_decl(
                "main",
                vec![],
                make_block(
                    vec![let_stmt(
                        "f",
                        lambda(
                            vec![make_param("x", i64_type())],
                            binary(
                                BinOp::Add,
                                call(ident_expr("double"), vec![ident_expr("x")]),
                                ident_expr("offset"),
                            ),
                        ),
                    )],
                    Some(call(ident_expr("f"), vec![int_expr(5)])),
                ),
            ),
        ],
    };
    let result = run_program(&program, 0).unwrap();
    assert!(matches!(result, Value::Int(20)), "expected 20, got {:?}", result);
}

#[test]
fn test_exec_closure_value_type_name() {
    // Verify that a Closure value reports the right type name
    let val = Value::Closure {
        fn_name: "__closure_0".to_string(),
        env: vec![Value::Int(1)],
        arity: 1,
    };
    assert_eq!(val.type_name(), "Closure");
    let display = format!("{val}");
    assert!(display.contains("closure"), "display should contain 'closure': {display}");
}

#[test]
fn test_exec_closure_captures_dont_leak_scope() {
    // The closure should capture at creation time, not at call time
    // (for Ref mode in the interpreter, the value is snapshot at capture eval)
    //
    // let x = 1
    // fn main() {
    //   let f = |y: i64| y + x
    //   f(10)
    // }
    // Result: 11
    let program = Program {
        declarations: vec![
            Decl { kind: DeclKind::Stmt(let_stmt("x", int_expr(1))), span: span() },
            make_fn_decl(
                "main",
                vec![],
                make_block(
                    vec![let_stmt(
                        "f",
                        lambda(
                            vec![make_param("y", i64_type())],
                            binary(BinOp::Add, ident_expr("y"), ident_expr("x")),
                        ),
                    )],
                    Some(call(ident_expr("f"), vec![int_expr(10)])),
                ),
            ),
        ],
    };
    let result = run_program(&program, 0).unwrap();
    assert!(matches!(result, Value::Int(11)), "expected 11, got {:?}", result);
}

// ── Deduplication test (captured variable appears multiple times) ────

#[test]
fn test_hir_closure_deduplicates_captures() {
    // let x = 1; let f = |y: i64| x + x + y
    // `x` should appear only once in captures
    let program = Program {
        declarations: vec![
            Decl { kind: DeclKind::Stmt(let_stmt("x", int_expr(1))), span: span() },
            Decl {
                kind: DeclKind::Stmt(let_stmt(
                    "f",
                    lambda(
                        vec![make_param("y", i64_type())],
                        binary(
                            BinOp::Add,
                            binary(BinOp::Add, ident_expr("x"), ident_expr("x")),
                            ident_expr("y"),
                        ),
                    ),
                )),
                span: span(),
            },
        ],
    };
    let mut lowering = AstLowering::new();
    let hir = lowering.lower_program(&program);
    match &hir.items[1] {
        cjc_hir::HirItem::Stmt(s) => match &s.kind {
            cjc_hir::HirStmtKind::Let { init, .. } => match &init.kind {
                HirExprKind::Closure { captures, .. } => {
                    assert_eq!(captures.len(), 1, "x should only be captured once");
                    assert_eq!(captures[0].name, "x");
                }
                _ => panic!("expected Closure"),
            },
            _ => panic!("expected Let"),
        },
        _ => panic!("expected Stmt"),
    }
}
