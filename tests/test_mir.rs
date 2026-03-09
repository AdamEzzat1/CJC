// CJC Test Suite — cjc-mir (6 tests)
// Source: crates/cjc-mir/src/lib.rs
// Integration tests for MIR data structures and HIR-to-MIR lowering.

use cjc_ast::BinOp;
use cjc_hir::*;
use cjc_mir::*;

fn hir_id(n: u32) -> HirId {
    HirId(n)
}

fn hir_int(v: i64) -> HirExpr {
    HirExpr {
        kind: HirExprKind::IntLit(v),
        hir_id: hir_id(0),
    }
}

fn hir_var(name: &str) -> HirExpr {
    HirExpr {
        kind: HirExprKind::Var(name.to_string()),
        hir_id: hir_id(0),
    }
}

#[test]
fn test_lower_hir_literal() {
    let mut lowering = HirToMir::new();
    let hir = hir_int(42);
    let mir = lowering.lower_expr(&hir);
    assert!(matches!(mir.kind, MirExprKind::IntLit(42)));
}

#[test]
fn test_lower_hir_binary() {
    let mut lowering = HirToMir::new();
    let hir = HirExpr {
        kind: HirExprKind::Binary {
            op: BinOp::Add,
            left: Box::new(hir_int(1)),
            right: Box::new(hir_int(2)),
        },
        hir_id: hir_id(0),
    };
    let mir = lowering.lower_expr(&hir);
    match &mir.kind {
        MirExprKind::Binary { op, .. } => assert_eq!(op, &BinOp::Add),
        _ => panic!("expected Binary"),
    }
}

#[test]
fn test_lower_hir_fn() {
    let mut lowering = HirToMir::new();
    let hir_fn = HirFn {
        name: "add".to_string(),
        type_params: vec![],
        params: vec![
            HirParam { name: "a".to_string(), ty_name: "i64".to_string(), default: None, hir_id: hir_id(1) },
            HirParam { name: "b".to_string(), ty_name: "i64".to_string(), default: None, hir_id: hir_id(2) },
        ],
        return_type: Some("i64".to_string()),
        body: HirBlock {
            stmts: vec![],
            expr: Some(Box::new(HirExpr {
                kind: HirExprKind::Binary {
                    op: BinOp::Add,
                    left: Box::new(hir_var("a")),
                    right: Box::new(hir_var("b")),
                },
                hir_id: hir_id(3),
            })),
            hir_id: hir_id(4),
        },
        is_nogc: false,
        hir_id: hir_id(5),
        decorators: vec![],
    };
    let mir_fn = lowering.lower_fn(&hir_fn);
    assert_eq!(mir_fn.name, "add");
    assert_eq!(mir_fn.params.len(), 2);
    assert!(mir_fn.body.result.is_some());
}

#[test]
fn test_lower_hir_program_entry() {
    let mut lowering = HirToMir::new();
    let hir = HirProgram {
        items: vec![
            HirItem::Let(HirLetDecl {
                name: "x".to_string(),
                mutable: false,
                ty_name: None,
                init: hir_int(42),
                hir_id: hir_id(0),
            }),
            HirItem::Fn(HirFn {
                name: "f".to_string(),
                type_params: vec![],
                params: vec![],
                return_type: None,
                body: HirBlock {
                    stmts: vec![],
                    expr: Some(Box::new(hir_var("x"))),
                    hir_id: hir_id(1),
                },
                is_nogc: false,
                hir_id: hir_id(2),
                decorators: vec![],
            }),
        ],
    };
    let mir = lowering.lower_program(&hir);
    assert_eq!(mir.functions.len(), 2);
    let main = mir.functions.iter().find(|f| f.name == "__main").unwrap();
    assert_eq!(main.body.stmts.len(), 1);
    assert_eq!(mir.entry, main.id);
}

#[test]
fn test_lower_hir_if_stmt() {
    let mut lowering = HirToMir::new();
    let hir_if = HirIfExpr {
        cond: Box::new(HirExpr {
            kind: HirExprKind::BoolLit(true),
            hir_id: hir_id(0),
        }),
        then_block: HirBlock {
            stmts: vec![],
            expr: Some(Box::new(hir_int(1))),
            hir_id: hir_id(1),
        },
        else_branch: Some(HirElseBranch::Else(HirBlock {
            stmts: vec![],
            expr: Some(Box::new(hir_int(2))),
            hir_id: hir_id(2),
        })),
        hir_id: hir_id(3),
    };
    let mir_stmt = lowering.lower_if_stmt(&hir_if);
    match &mir_stmt {
        MirStmt::If { then_body, else_body, .. } => {
            assert!(then_body.result.is_some());
            assert!(else_body.is_some());
        }
        _ => panic!("expected If"),
    }
}

#[test]
fn test_lower_struct_def() {
    let mut lowering = HirToMir::new();
    let hir = HirProgram {
        items: vec![HirItem::Struct(HirStructDef {
            name: "Point".to_string(),
            fields: vec![
                ("x".to_string(), "f64".to_string()),
                ("y".to_string(), "f64".to_string()),
            ],
            hir_id: hir_id(0),
        })],
    };
    let mir = lowering.lower_program(&hir);
    assert_eq!(mir.struct_defs.len(), 1);
    assert_eq!(mir.struct_defs[0].name, "Point");
    assert_eq!(mir.struct_defs[0].fields.len(), 2);
}
