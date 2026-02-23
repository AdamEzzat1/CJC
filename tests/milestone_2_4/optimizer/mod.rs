// Milestone 2.4 — MIR Optimizer Tests (Constant Folding + DCE)
//
// Tests constant folding of pure ops, dead code elimination of unused bindings
// and unreachable blocks, and ensures side-effecting ops are preserved.

use cjc_ast::*;
use cjc_mir::optimize::optimize_program;
use cjc_mir::*;

// ---------------------------------------------------------------------------
// MIR-level helpers
// ---------------------------------------------------------------------------

fn mk_expr(kind: MirExprKind) -> MirExpr {
    MirExpr { kind }
}

fn mk_int(v: i64) -> MirExpr {
    mk_expr(MirExprKind::IntLit(v))
}

fn mk_float(v: f64) -> MirExpr {
    mk_expr(MirExprKind::FloatLit(v))
}

fn mk_bool(v: bool) -> MirExpr {
    mk_expr(MirExprKind::BoolLit(v))
}

fn mk_var(name: &str) -> MirExpr {
    mk_expr(MirExprKind::Var(name.to_string()))
}

fn mk_binary(op: BinOp, left: MirExpr, right: MirExpr) -> MirExpr {
    mk_expr(MirExprKind::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
    })
}

fn mk_unary(op: UnaryOp, operand: MirExpr) -> MirExpr {
    mk_expr(MirExprKind::Unary {
        op,
        operand: Box::new(operand),
    })
}

fn mk_call(name: &str, args: Vec<MirExpr>) -> MirExpr {
    mk_expr(MirExprKind::Call {
        callee: Box::new(mk_expr(MirExprKind::Var(name.to_string()))),
        args,
    })
}

fn mk_fn(name: &str, stmts: Vec<MirStmt>, result: Option<MirExpr>) -> MirFunction {
    MirFunction {
        id: MirFnId(0),
        name: name.to_string(),
        type_params: vec![],
        params: vec![],
        return_type: None,
        body: MirBody {
            stmts,
            result: result.map(Box::new),
        },
        is_nogc: false,
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

// ===========================================================================
// Constant Folding Tests
// ===========================================================================

#[test]
fn cf_int_add() {
    let program = mk_program(vec![mk_fn(
        "__main",
        vec![],
        Some(mk_binary(BinOp::Add, mk_int(10), mk_int(20))),
    )]);
    let opt = optimize_program(&program);
    match &opt.functions[0].body.result.as_ref().unwrap().kind {
        MirExprKind::IntLit(30) => {}
        other => panic!("expected IntLit(30), got {:?}", other),
    }
}

#[test]
fn cf_int_sub() {
    let program = mk_program(vec![mk_fn(
        "__main",
        vec![],
        Some(mk_binary(BinOp::Sub, mk_int(50), mk_int(30))),
    )]);
    let opt = optimize_program(&program);
    match &opt.functions[0].body.result.as_ref().unwrap().kind {
        MirExprKind::IntLit(20) => {}
        other => panic!("expected IntLit(20), got {:?}", other),
    }
}

#[test]
fn cf_int_mul() {
    let program = mk_program(vec![mk_fn(
        "__main",
        vec![],
        Some(mk_binary(BinOp::Mul, mk_int(6), mk_int(7))),
    )]);
    let opt = optimize_program(&program);
    match &opt.functions[0].body.result.as_ref().unwrap().kind {
        MirExprKind::IntLit(42) => {}
        other => panic!("expected IntLit(42), got {:?}", other),
    }
}

#[test]
fn cf_int_div() {
    let program = mk_program(vec![mk_fn(
        "__main",
        vec![],
        Some(mk_binary(BinOp::Div, mk_int(100), mk_int(5))),
    )]);
    let opt = optimize_program(&program);
    match &opt.functions[0].body.result.as_ref().unwrap().kind {
        MirExprKind::IntLit(20) => {}
        other => panic!("expected IntLit(20), got {:?}", other),
    }
}

#[test]
fn cf_int_div_by_zero_not_folded() {
    let program = mk_program(vec![mk_fn(
        "__main",
        vec![],
        Some(mk_binary(BinOp::Div, mk_int(10), mk_int(0))),
    )]);
    let opt = optimize_program(&program);
    // Must NOT be folded — let runtime raise the error.
    match &opt.functions[0].body.result.as_ref().unwrap().kind {
        MirExprKind::Binary { .. } => {}
        other => panic!("expected Binary (unfoldable), got {:?}", other),
    }
}

#[test]
fn cf_int_mod() {
    let program = mk_program(vec![mk_fn(
        "__main",
        vec![],
        Some(mk_binary(BinOp::Mod, mk_int(17), mk_int(5))),
    )]);
    let opt = optimize_program(&program);
    match &opt.functions[0].body.result.as_ref().unwrap().kind {
        MirExprKind::IntLit(2) => {}
        other => panic!("expected IntLit(2), got {:?}", other),
    }
}

#[test]
fn cf_int_comparison() {
    let program = mk_program(vec![mk_fn(
        "__main",
        vec![],
        Some(mk_binary(BinOp::Lt, mk_int(3), mk_int(5))),
    )]);
    let opt = optimize_program(&program);
    match &opt.functions[0].body.result.as_ref().unwrap().kind {
        MirExprKind::BoolLit(true) => {}
        other => panic!("expected BoolLit(true), got {:?}", other),
    }
}

#[test]
fn cf_float_add() {
    let program = mk_program(vec![mk_fn(
        "__main",
        vec![],
        Some(mk_binary(BinOp::Add, mk_float(1.5), mk_float(2.5))),
    )]);
    let opt = optimize_program(&program);
    match &opt.functions[0].body.result.as_ref().unwrap().kind {
        MirExprKind::FloatLit(v) => assert_eq!(*v, 4.0),
        other => panic!("expected FloatLit(4.0), got {:?}", other),
    }
}

#[test]
fn cf_float_comparison() {
    let program = mk_program(vec![mk_fn(
        "__main",
        vec![],
        Some(mk_binary(BinOp::Ge, mk_float(3.14), mk_float(2.71))),
    )]);
    let opt = optimize_program(&program);
    match &opt.functions[0].body.result.as_ref().unwrap().kind {
        MirExprKind::BoolLit(true) => {}
        other => panic!("expected BoolLit(true), got {:?}", other),
    }
}

#[test]
fn cf_bool_and() {
    let program = mk_program(vec![mk_fn(
        "__main",
        vec![],
        Some(mk_binary(BinOp::And, mk_bool(true), mk_bool(false))),
    )]);
    let opt = optimize_program(&program);
    match &opt.functions[0].body.result.as_ref().unwrap().kind {
        MirExprKind::BoolLit(false) => {}
        other => panic!("expected BoolLit(false), got {:?}", other),
    }
}

#[test]
fn cf_string_concat() {
    let program = mk_program(vec![mk_fn(
        "__main",
        vec![],
        Some(mk_binary(
            BinOp::Add,
            mk_expr(MirExprKind::StringLit("hello".to_string())),
            mk_expr(MirExprKind::StringLit(" world".to_string())),
        )),
    )]);
    let opt = optimize_program(&program);
    match &opt.functions[0].body.result.as_ref().unwrap().kind {
        MirExprKind::StringLit(s) => assert_eq!(s, "hello world"),
        other => panic!("expected StringLit, got {:?}", other),
    }
}

#[test]
fn cf_nested_fold() {
    // (2 + 3) * (4 - 1) = 5 * 3 = 15
    let program = mk_program(vec![mk_fn(
        "__main",
        vec![],
        Some(mk_binary(
            BinOp::Mul,
            mk_binary(BinOp::Add, mk_int(2), mk_int(3)),
            mk_binary(BinOp::Sub, mk_int(4), mk_int(1)),
        )),
    )]);
    let opt = optimize_program(&program);
    match &opt.functions[0].body.result.as_ref().unwrap().kind {
        MirExprKind::IntLit(15) => {}
        other => panic!("expected IntLit(15), got {:?}", other),
    }
}

#[test]
fn cf_unary_neg_int() {
    let program = mk_program(vec![mk_fn(
        "__main",
        vec![],
        Some(mk_unary(UnaryOp::Neg, mk_int(42))),
    )]);
    let opt = optimize_program(&program);
    match &opt.functions[0].body.result.as_ref().unwrap().kind {
        MirExprKind::IntLit(-42) => {}
        other => panic!("expected IntLit(-42), got {:?}", other),
    }
}

#[test]
fn cf_unary_not_bool() {
    let program = mk_program(vec![mk_fn(
        "__main",
        vec![],
        Some(mk_unary(UnaryOp::Not, mk_bool(true))),
    )]);
    let opt = optimize_program(&program);
    match &opt.functions[0].body.result.as_ref().unwrap().kind {
        MirExprKind::BoolLit(false) => {}
        other => panic!("expected BoolLit(false), got {:?}", other),
    }
}

#[test]
fn cf_wrapping_overflow() {
    // Ensure wrapping_add semantics match
    let program = mk_program(vec![mk_fn(
        "__main",
        vec![],
        Some(mk_binary(BinOp::Add, mk_int(i64::MAX), mk_int(1))),
    )]);
    let opt = optimize_program(&program);
    match &opt.functions[0].body.result.as_ref().unwrap().kind {
        MirExprKind::IntLit(v) => assert_eq!(*v, i64::MIN),
        other => panic!("expected IntLit wrapping, got {:?}", other),
    }
}

#[test]
fn cf_if_true_eliminated() {
    // if true { 42 } else { 0 } should fold to Block containing 42
    let program = mk_program(vec![mk_fn(
        "__main",
        vec![],
        Some(mk_expr(MirExprKind::If {
            cond: Box::new(mk_bool(true)),
            then_body: MirBody {
                stmts: vec![],
                result: Some(Box::new(mk_int(42))),
            },
            else_body: Some(MirBody {
                stmts: vec![],
                result: Some(Box::new(mk_int(0))),
            }),
        })),
    )]);
    let opt = optimize_program(&program);
    // The if should be eliminated, leaving the then branch
    match &opt.functions[0].body.result.as_ref().unwrap().kind {
        MirExprKind::Block(body) => {
            match &body.result.as_ref().unwrap().kind {
                MirExprKind::IntLit(42) => {}
                other => panic!("expected IntLit(42) in block, got {:?}", other),
            }
        }
        other => panic!("expected Block, got {:?}", other),
    }
}

#[test]
fn cf_no_fold_with_variable_operand() {
    // x + 1 should NOT be folded (x is a variable)
    let program = mk_program(vec![mk_fn(
        "__main",
        vec![],
        Some(mk_binary(BinOp::Add, mk_var("x"), mk_int(1))),
    )]);
    let opt = optimize_program(&program);
    match &opt.functions[0].body.result.as_ref().unwrap().kind {
        MirExprKind::Binary { .. } => {}
        other => panic!("expected Binary (not foldable), got {:?}", other),
    }
}

// ===========================================================================
// Dead Code Elimination Tests
// ===========================================================================

#[test]
fn dce_removes_unused_pure_let() {
    let program = mk_program(vec![mk_fn(
        "__main",
        vec![
            MirStmt::Let {
                name: "unused".to_string(),
                mutable: false,
                init: mk_int(42),
            },
            MirStmt::Let {
                name: "used".to_string(),
                mutable: false,
                init: mk_int(99),
            },
        ],
        Some(mk_var("used")),
    )]);
    let opt = optimize_program(&program);
    // "unused" should be removed
    assert_eq!(opt.functions[0].body.stmts.len(), 1);
    match &opt.functions[0].body.stmts[0] {
        MirStmt::Let { name, .. } => assert_eq!(name, "used"),
        _ => panic!("expected Let"),
    }
}

#[test]
fn dce_preserves_side_effecting_let() {
    // let unused = print("hi") — side effect, must keep
    let program = mk_program(vec![mk_fn(
        "__main",
        vec![MirStmt::Let {
            name: "unused".to_string(),
            mutable: false,
            init: mk_call("print", vec![mk_expr(MirExprKind::StringLit("hi".to_string()))]),
        }],
        None,
    )]);
    let opt = optimize_program(&program);
    assert_eq!(opt.functions[0].body.stmts.len(), 1);
}

#[test]
fn dce_removes_dead_if_false_branch() {
    let program = mk_program(vec![mk_fn(
        "__main",
        vec![MirStmt::If {
            cond: mk_bool(false),
            then_body: MirBody {
                stmts: vec![MirStmt::Expr(mk_int(1))],
                result: None,
            },
            else_body: None,
        }],
        None,
    )]);
    let opt = optimize_program(&program);
    assert!(opt.functions[0].body.stmts.is_empty());
}

#[test]
fn dce_inlines_if_true_branch() {
    let program = mk_program(vec![mk_fn(
        "__main",
        vec![MirStmt::If {
            cond: mk_bool(true),
            then_body: MirBody {
                stmts: vec![MirStmt::Expr(mk_call("print", vec![mk_int(1)]))],
                result: None,
            },
            else_body: None,
        }],
        None,
    )]);
    let opt = optimize_program(&program);
    assert_eq!(opt.functions[0].body.stmts.len(), 1);
    assert!(matches!(opt.functions[0].body.stmts[0], MirStmt::Expr(_)));
}

#[test]
fn dce_removes_dead_while_false() {
    let program = mk_program(vec![mk_fn(
        "__main",
        vec![MirStmt::While {
            cond: mk_bool(false),
            body: MirBody {
                stmts: vec![MirStmt::Expr(mk_int(1))],
                result: None,
            },
        }],
        None,
    )]);
    let opt = optimize_program(&program);
    assert!(opt.functions[0].body.stmts.is_empty());
}

#[test]
fn dce_preserves_while_with_variable_cond() {
    let program = mk_program(vec![mk_fn(
        "__main",
        vec![MirStmt::While {
            cond: mk_var("running"),
            body: MirBody {
                stmts: vec![MirStmt::Expr(mk_int(1))],
                result: None,
            },
        }],
        None,
    )]);
    let opt = optimize_program(&program);
    assert_eq!(opt.functions[0].body.stmts.len(), 1);
}

#[test]
fn dce_combined_with_cf() {
    // let x = 1 + 2   (folds to let x = 3)
    // let unused = 5   (dead, removed)
    // result: x
    let program = mk_program(vec![mk_fn(
        "__main",
        vec![
            MirStmt::Let {
                name: "x".to_string(),
                mutable: false,
                init: mk_binary(BinOp::Add, mk_int(1), mk_int(2)),
            },
            MirStmt::Let {
                name: "unused".to_string(),
                mutable: false,
                init: mk_int(5),
            },
        ],
        Some(mk_var("x")),
    )]);
    let opt = optimize_program(&program);
    assert_eq!(opt.functions[0].body.stmts.len(), 1);
    match &opt.functions[0].body.stmts[0] {
        MirStmt::Let { name, init, .. } => {
            assert_eq!(name, "x");
            assert!(matches!(init.kind, MirExprKind::IntLit(3)));
        }
        _ => panic!("expected Let"),
    }
}
