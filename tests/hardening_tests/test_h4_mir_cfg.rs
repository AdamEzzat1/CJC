//! H-4: MIR CFG structure — BasicBlock, Terminator, predecessor/successor analysis.

use cjc_mir::cfg::{BasicBlock, CfgBuilder, MirCfg, Terminator};
use cjc_mir::{AllocHint, BlockId, MirBody, MirExpr, MirExprKind, MirStmt};

fn int_expr(v: i64) -> MirExpr {
    MirExpr { kind: MirExprKind::IntLit(v) }
}

fn bool_expr(b: bool) -> MirExpr {
    MirExpr { kind: MirExprKind::BoolLit(b) }
}

fn empty_body() -> MirBody {
    MirBody { stmts: vec![], result: None }
}

/// Test 1: BlockId, BasicBlock, Terminator types are exported correctly.
#[test]
fn test_cfg_types_are_exported() {
    let bb = BasicBlock {
        id: BlockId(0),
        statements: vec![],
        terminator: Terminator::Unreachable,
    };
    assert_eq!(bb.id, BlockId(0));
    assert!(bb.statements.is_empty());
    matches!(bb.terminator, Terminator::Unreachable);
}

/// Test 2: Straight-line body produces exactly one block with Return terminator.
#[test]
fn test_cfg_straight_line_single_block_return() {
    let body = MirBody {
        stmts: vec![
            MirStmt::Let { name: "x".into(), mutable: false, init: int_expr(1), alloc_hint: None },
            MirStmt::Let { name: "y".into(), mutable: false, init: int_expr(2), alloc_hint: None },
        ],
        result: Some(Box::new(int_expr(3))),
    };
    let cfg = CfgBuilder::build(&body);
    assert_eq!(cfg.entry, BlockId(0));
    assert!(!cfg.basic_blocks.is_empty());
    match &cfg.entry_block().terminator {
        Terminator::Return(_) => {}
        other => panic!("expected Return, got {:?}", other),
    }
}

/// Test 3: if/else produces at least 3 blocks with Branch terminator on entry.
#[test]
fn test_cfg_if_else_branch_terminator() {
    let body = MirBody {
        stmts: vec![
            MirStmt::If {
                cond: bool_expr(true),
                then_body: MirBody {
                    stmts: vec![],
                    result: Some(Box::new(int_expr(1))),
                },
                else_body: Some(MirBody {
                    stmts: vec![],
                    result: Some(Box::new(int_expr(2))),
                }),
            },
        ],
        result: None,
    };
    let cfg = CfgBuilder::build(&body);
    assert!(cfg.basic_blocks.len() >= 3, "if/else needs >= 3 blocks");
    match &cfg.entry_block().terminator {
        Terminator::Branch { then_block, else_block, .. } => {
            assert_ne!(then_block, else_block);
        }
        other => panic!("entry should be Branch, got {:?}", other),
    }
}

/// Test 4: while loop produces a back-edge (is_loop_header returns true for header).
#[test]
fn test_cfg_while_loop_header_detected() {
    let body = MirBody {
        stmts: vec![
            MirStmt::While { cond: bool_expr(false), body: empty_body() },
        ],
        result: None,
    };
    let cfg = CfgBuilder::build(&body);
    assert!(cfg.basic_blocks.len() >= 3);
    // Header is block 1 (entry=0 → Goto(1=header) → Branch(body=2, exit=3))
    let header = BlockId(1);
    assert!(cfg.is_loop_header(header), "block 1 should be detected as loop header");
}

/// Test 5: predecessors() — entry block has no predecessors.
#[test]
fn test_cfg_entry_has_no_predecessors() {
    let body = MirBody {
        stmts: vec![],
        result: Some(Box::new(int_expr(0))),
    };
    let cfg = CfgBuilder::build(&body);
    let preds = cfg.predecessors();
    assert_eq!(preds[0].len(), 0, "entry block should have no predecessors");
}

/// Test 6: successors() of a Branch has two entries.
#[test]
fn test_cfg_branch_has_two_successors_via_successors_method() {
    let body = MirBody {
        stmts: vec![
            MirStmt::If {
                cond: bool_expr(true),
                then_body: empty_body(),
                else_body: Some(empty_body()),
            },
        ],
        result: None,
    };
    let cfg = CfgBuilder::build(&body);
    let entry_succs = cfg.successors(cfg.entry);
    assert_eq!(
        entry_succs.len(),
        2,
        "Branch should have 2 successors, got {:?}",
        entry_succs
    );
}

/// Test 7: Return terminator has no successors.
#[test]
fn test_cfg_return_has_no_successors() {
    let term = Terminator::Return(None);
    assert!(term.successors().is_empty());
}

/// Test 8: Goto terminator has exactly one successor.
#[test]
fn test_cfg_goto_has_one_successor() {
    let term = Terminator::Goto(BlockId(7));
    assert_eq!(term.successors(), vec![BlockId(7)]);
}

/// Test 9: MirCfg built from a full CJC program (via lower_to_mir + cfg::build).
#[test]
fn test_cfg_built_from_mir_program() {
    use cjc_parser::parse_source;
    use cjc_mir_exec::lower_to_mir;
    use cjc_mir::cfg::CfgBuilder;

    let src = r#"
fn factorial(n: i64) -> i64 {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}
fn main() -> i64 {
    factorial(5)
}
"#;
    let (prog, parse_diags) = parse_source(src);
    assert!(!parse_diags.has_errors());
    let mir = lower_to_mir(&prog);
    // Build CFG for the `factorial` function
    let factorial_fn = mir.functions.iter().find(|f| f.name == "factorial");
    assert!(factorial_fn.is_some(), "factorial function should be in MIR");
    let cfg = CfgBuilder::build(&factorial_fn.unwrap().body);
    // factorial has an if/else — should produce >= 3 blocks
    assert!(cfg.basic_blocks.len() >= 3, "factorial CFG should have >= 3 blocks");
}

/// Test 10: Determinism — building CFG twice from same body gives identical block count.
#[test]
fn test_cfg_build_is_deterministic() {
    let body = MirBody {
        stmts: vec![
            MirStmt::If {
                cond: bool_expr(true),
                then_body: MirBody {
                    stmts: vec![MirStmt::Let { name: "a".into(), mutable: false, init: int_expr(1), alloc_hint: None }],
                    result: None,
                },
                else_body: Some(MirBody {
                    stmts: vec![MirStmt::Let { name: "b".into(), mutable: false, init: int_expr(2), alloc_hint: None }],
                    result: None,
                }),
            },
        ],
        result: None,
    };
    let cfg1 = CfgBuilder::build(&body);
    let cfg2 = CfgBuilder::build(&body);
    assert_eq!(
        cfg1.basic_blocks.len(),
        cfg2.basic_blocks.len(),
        "CFG builds must be deterministic"
    );
    // Same terminator types on entry block
    let t1 = std::mem::discriminant(&cfg1.entry_block().terminator);
    let t2 = std::mem::discriminant(&cfg2.entry_block().terminator);
    assert_eq!(t1, t2, "terminator types must match across builds");
}
