// Gate G-8: NoGC Static Verifier Tests
//
// Tests that attempt to smuggle GC into nogc via:
// - Direct GcAlloc in a nogc function
// - Calling a known GC-allocating function from nogc
// - Transitive call chain to GC
// - Calling unknown/external function from nogc
// - Indirect call (closure) in nogc
// - NoGcBlock enforcement in non-nogc functions

use cjc_ast::*;
use cjc_mir::nogc_verify::verify_nogc;
use cjc_mir::{AllocHint, MirBody, MirExpr, MirExprKind, MirFnId, MirFunction, MirProgram, MirStmt};

// ---------------------------------------------------------------------------
// MIR-level helpers
// ---------------------------------------------------------------------------

fn mk_expr(kind: MirExprKind) -> MirExpr {
    MirExpr { kind }
}

fn mk_call(name: &str, args: Vec<MirExpr>) -> MirExpr {
    mk_expr(MirExprKind::Call {
        callee: Box::new(mk_expr(MirExprKind::Var(name.to_string()))),
        args,
    })
}

fn mk_fn(name: &str, is_nogc: bool, stmts: Vec<MirStmt>) -> MirFunction {
    MirFunction {
        id: MirFnId(0),
        name: name.to_string(),
        type_params: vec![],
        params: vec![],
        return_type: None,
        body: MirBody {
            stmts,
            result: None,
        },
        is_nogc,
        cfg_body: None,
    }
}

fn mk_program(functions: Vec<MirFunction>) -> MirProgram {
    MirProgram {
        functions,
        struct_defs: vec![],
        enum_defs: vec![],
        entry: MirFnId(0),
    }
}

// ---------------------------------------------------------------------------
// AST-level helpers (for full pipeline tests)
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

fn expr_stmt(expr: Expr) -> Stmt {
    Stmt { kind: StmtKind::Expr(expr), span: span() }
}

fn make_block(stmts: Vec<Stmt>, expr: Option<Expr>) -> Block {
    Block { stmts, expr: expr.map(Box::new), span: span() }
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

fn make_fn_decl(name: &str, params: Vec<&str>, body: Block, is_nogc: bool) -> Decl {
    Decl {
        kind: DeclKind::Fn(FnDecl {
            name: ident(name),
            type_params: vec![],
            params: params.into_iter().map(make_param).collect(),
            return_type: None,
            body,
            is_nogc,
            effect_annotation: None,
        }),
        span: span(),
    }
}

// ===========================================================================
// G-8 Tests: Smuggle GC into nogc
// ===========================================================================

#[test]
fn g8_direct_gc_alloc_in_nogc_function() {
    let program = mk_program(vec![
        mk_fn("bad_nogc", true, vec![
            MirStmt::Expr(mk_call("gc_alloc", vec![])),
        ]),
    ]);
    let errors = verify_nogc(&program).unwrap_err();
    assert!(errors.iter().any(|e| e.reason.contains("gc_alloc")));
}

#[test]
fn g8_direct_gc_collect_in_nogc_function() {
    let program = mk_program(vec![
        mk_fn("bad_nogc", true, vec![
            MirStmt::Expr(mk_call("gc_collect", vec![])),
        ]),
    ]);
    let errors = verify_nogc(&program).unwrap_err();
    assert!(errors.iter().any(|e| e.reason.contains("gc_collect")));
}

#[test]
fn g8_call_gc_allocating_function_from_nogc() {
    let program = mk_program(vec![
        mk_fn("allocator", false, vec![
            MirStmt::Expr(mk_call("gc_alloc", vec![])),
        ]),
        mk_fn("caller_nogc", true, vec![
            MirStmt::Expr(mk_call("allocator", vec![])),
        ]),
    ]);
    let errors = verify_nogc(&program).unwrap_err();
    assert!(errors.iter().any(|e| e.reason.contains("allocator")));
}

#[test]
fn g8_transitive_gc_chain() {
    // chain: nogc_fn -> middle -> allocator -> gc_alloc
    let program = mk_program(vec![
        mk_fn("allocator", false, vec![
            MirStmt::Expr(mk_call("gc_alloc", vec![])),
        ]),
        mk_fn("middle", false, vec![
            MirStmt::Expr(mk_call("allocator", vec![])),
        ]),
        mk_fn("nogc_fn", true, vec![
            MirStmt::Expr(mk_call("middle", vec![])),
        ]),
    ]);
    let errors = verify_nogc(&program).unwrap_err();
    assert!(errors.iter().any(|e| e.function == "nogc_fn"));
    // Should have a call chain
    assert!(errors.iter().any(|e| !e.call_chain.is_empty()));
}

#[test]
fn g8_unknown_external_function_rejected() {
    let program = mk_program(vec![
        mk_fn("nogc_fn", true, vec![
            MirStmt::Expr(mk_call("mystery_extern_fn", vec![])),
        ]),
    ]);
    let errors = verify_nogc(&program).unwrap_err();
    assert!(!errors.is_empty());
}

#[test]
fn g8_indirect_call_in_nogc_rejected() {
    let program = mk_program(vec![
        mk_fn("nogc_fn", true, vec![
            MirStmt::Expr(mk_expr(MirExprKind::Call {
                callee: Box::new(mk_expr(MirExprKind::Index {
                    object: Box::new(mk_expr(MirExprKind::Var("callbacks".to_string()))),
                    index: Box::new(mk_expr(MirExprKind::IntLit(0))),
                })),
                args: vec![],
            })),
        ]),
    ]);
    let errors = verify_nogc(&program).unwrap_err();
    assert!(errors.iter().any(|e| e.reason.contains("indirect")));
}

#[test]
fn g8_safe_builtin_in_nogc_allowed() {
    let program = mk_program(vec![
        mk_fn("nogc_fn", true, vec![
            MirStmt::Expr(mk_call("print", vec![
                mk_expr(MirExprKind::StringLit("ok".to_string())),
            ])),
        ]),
    ]);
    assert!(verify_nogc(&program).is_ok());
}

#[test]
fn g8_pure_arithmetic_in_nogc_allowed() {
    let program = mk_program(vec![
        mk_fn("pure_math", true, vec![
            MirStmt::Let {
                name: "x".to_string(),
                mutable: false,
                init: mk_expr(MirExprKind::Binary {
                    op: BinOp::Add,
                    left: Box::new(mk_expr(MirExprKind::IntLit(1))),
                    right: Box::new(mk_expr(MirExprKind::IntLit(2))),
                }),
                alloc_hint: None,
            },
        ]),
    ]);
    assert!(verify_nogc(&program).is_ok());
}

#[test]
fn g8_nogc_block_in_non_nogc_function_rejects_gc() {
    let program = mk_program(vec![
        mk_fn("wrapper", false, vec![
            MirStmt::NoGcBlock(MirBody {
                stmts: vec![MirStmt::Expr(mk_call("gc_alloc", vec![]))],
                result: None,
            }),
        ]),
    ]);
    let errors = verify_nogc(&program).unwrap_err();
    assert!(errors.iter().any(|e| e.reason.contains("nogc block")));
}

#[test]
fn g8_nogc_block_allows_safe_ops() {
    let program = mk_program(vec![
        mk_fn("wrapper", false, vec![
            MirStmt::NoGcBlock(MirBody {
                stmts: vec![MirStmt::Expr(mk_call("print", vec![
                    mk_expr(MirExprKind::StringLit("safe".to_string())),
                ]))],
                result: None,
            }),
        ]),
    ]);
    assert!(verify_nogc(&program).is_ok());
}

#[test]
fn g8_nogc_function_calling_known_safe_user_fn() {
    let program = mk_program(vec![
        mk_fn("safe_helper", false, vec![
            MirStmt::Expr(mk_call("print", vec![
                mk_expr(MirExprKind::StringLit("hi".to_string())),
            ])),
        ]),
        mk_fn("nogc_fn", true, vec![
            MirStmt::Expr(mk_call("safe_helper", vec![])),
        ]),
    ]);
    assert!(verify_nogc(&program).is_ok());
}

#[test]
fn g8_full_pipeline_verify_nogc_ast() {
    // Build a clean nogc function via AST and verify it passes.
    let program = Program {
        declarations: vec![
            make_fn_decl(
                "pure_add",
                vec!["a", "b"],
                make_block(
                    vec![],
                    Some(binary(BinOp::Add, ident_expr("a"), ident_expr("b"))),
                ),
                true, // is_nogc = true
            ),
            make_fn_decl(
                "main",
                vec![],
                make_block(
                    vec![],
                    Some(call(ident_expr("pure_add"), vec![int_expr(1), int_expr(2)])),
                ),
                false,
            ),
        ],
    };
    assert!(cjc_mir_exec::verify_nogc(&program).is_ok());
}

#[test]
fn g8_full_pipeline_gc_in_nogc_fails() {
    // Build a nogc function that calls gc_alloc via AST.
    let program = Program {
        declarations: vec![
            make_fn_decl(
                "bad_nogc",
                vec![],
                make_block(
                    vec![expr_stmt(call(ident_expr("gc_alloc"), vec![]))],
                    None,
                ),
                true,
            ),
        ],
    };
    assert!(cjc_mir_exec::verify_nogc(&program).is_err());
}

#[test]
fn g8_deep_transitive_chain_4_levels() {
    // level0 -> level1 -> level2 -> level3 -> gc_alloc
    let program = mk_program(vec![
        mk_fn("level3", false, vec![
            MirStmt::Expr(mk_call("gc_alloc", vec![])),
        ]),
        mk_fn("level2", false, vec![
            MirStmt::Expr(mk_call("level3", vec![])),
        ]),
        mk_fn("level1", false, vec![
            MirStmt::Expr(mk_call("level2", vec![])),
        ]),
        mk_fn("level0", true, vec![
            MirStmt::Expr(mk_call("level1", vec![])),
        ]),
    ]);
    let errors = verify_nogc(&program).unwrap_err();
    assert!(errors.iter().any(|e| e.function == "level0"));
}

#[test]
fn g8_nogc_block_transitive_rejection() {
    let program = mk_program(vec![
        mk_fn("allocator", false, vec![
            MirStmt::Expr(mk_call("gc_alloc", vec![])),
        ]),
        mk_fn("wrapper", false, vec![
            MirStmt::NoGcBlock(MirBody {
                stmts: vec![MirStmt::Expr(mk_call("allocator", vec![]))],
                result: None,
            }),
        ]),
    ]);
    let errors = verify_nogc(&program).unwrap_err();
    assert!(errors.iter().any(|e| e.reason.contains("allocator")));
}

// ===========================================================================
// Step 5: Escape analysis integration tests
// ===========================================================================

#[test]
fn step5_nogc_rejects_returned_string_binding() {
    // A @no_gc function with a string binding that is returned (escapes → Rc).
    // Escape analysis should flag this as Rc, and the verifier should reject it.
    let program = mk_program(vec![
        mk_fn("nogc_returns_string", true, vec![
            MirStmt::Let {
                name: "s".to_string(),
                mutable: false,
                init: mk_expr(MirExprKind::StringLit("hello".to_string())),
                alloc_hint: None,
            },
            MirStmt::Return(Some(mk_expr(MirExprKind::Var("s".to_string())))),
        ]),
    ]);
    let errors = verify_nogc(&program).unwrap_err();
    assert!(errors.iter().any(|e|
        e.function == "nogc_returns_string"
        && e.reason.contains("heap allocation")
    ));
}

#[test]
fn step5_nogc_allows_primitive_only_function() {
    // A @no_gc function with only primitive operations should pass.
    let program = mk_program(vec![
        mk_fn("pure_prims", true, vec![
            MirStmt::Let {
                name: "a".to_string(),
                mutable: false,
                init: mk_expr(MirExprKind::IntLit(42)),
                alloc_hint: None,
            },
            MirStmt::Let {
                name: "b".to_string(),
                mutable: false,
                init: mk_expr(MirExprKind::FloatLit(3.14)),
                alloc_hint: None,
            },
            MirStmt::Let {
                name: "c".to_string(),
                mutable: false,
                init: mk_expr(MirExprKind::BoolLit(true)),
                alloc_hint: None,
            },
            MirStmt::Let {
                name: "d".to_string(),
                mutable: false,
                init: mk_expr(MirExprKind::Binary {
                    op: BinOp::Add,
                    left: Box::new(mk_expr(MirExprKind::IntLit(10))),
                    right: Box::new(mk_expr(MirExprKind::IntLit(20))),
                }),
                alloc_hint: None,
            },
        ]),
    ]);
    assert!(verify_nogc(&program).is_ok());
}

#[test]
fn step5_nogc_allows_non_escaping_string_arena() {
    // A @no_gc function with a string that does NOT escape (never returned,
    // never captured) should be classified as Arena, which is allowed.
    let program = mk_program(vec![
        mk_fn("nogc_local_string", true, vec![
            MirStmt::Let {
                name: "s".to_string(),
                mutable: false,
                init: mk_expr(MirExprKind::StringLit("local".to_string())),
                alloc_hint: None,
            },
            // Only use s in a safe builtin call (print) — no escape.
            MirStmt::Expr(mk_call("print", vec![
                mk_expr(MirExprKind::Var("s".to_string())),
            ])),
        ]),
    ]);
    assert!(verify_nogc(&program).is_ok());
}

#[test]
fn step5_nogc_rejects_mutable_string_binding() {
    // A @no_gc function with a mutable string binding. Mutable → Rc (conservative).
    // The verifier should reject this.
    let program = mk_program(vec![
        mk_fn("nogc_mut_string", true, vec![
            MirStmt::Let {
                name: "s".to_string(),
                mutable: true,
                init: mk_expr(MirExprKind::StringLit("mutable".to_string())),
                alloc_hint: None,
            },
        ]),
    ]);
    let errors = verify_nogc(&program).unwrap_err();
    assert!(errors.iter().any(|e|
        e.function == "nogc_mut_string"
        && e.reason.contains("heap allocation")
    ));
}
