//! LH06: CFG-Based MIR tests
//!
//! Verifies:
//! - CFG construction from tree-form MIR bodies
//! - Basic block structure (entry, terminators, predecessors)
//! - If/else produces Branch terminators with then/else blocks
//! - While loops produce back-edges and loop headers
//! - Return terminates blocks correctly
//! - build_cfg()/build_all_cfgs() populate cfg_body on MirFunction
//! - Programs lower to CFG and maintain structural correctness

use cjc_mir::cfg::{CfgBuilder, Terminator};
use cjc_mir::BlockId;

// ── CFG from parsed programs ────────────────────────────────────

#[test]
fn test_cfg_simple_fn() {
    let src = r#"
fn add(a: i64, b: i64) -> i64 { a + b }
print(add(1, 2));
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut mir = cjc_mir_exec::lower_to_mir(&program);
    mir.build_all_cfgs();

    for func in &mir.functions {
        let cfg = func.cfg_body.as_ref().expect("cfg_body should be populated");
        assert!(!cfg.basic_blocks.is_empty(), "fn {} should have blocks", func.name);
        assert_eq!(cfg.entry, BlockId(0));
    }
}

#[test]
fn test_cfg_fn_with_if() {
    let src = r#"
fn abs(x: i64) -> i64 {
    if x < 0 { 0 - x } else { x }
}
print(abs(-5));
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut mir = cjc_mir_exec::lower_to_mir(&program);
    mir.build_all_cfgs();

    // Find the `abs` function
    let abs_fn = mir.functions.iter().find(|f| f.name == "abs").unwrap();
    let cfg = abs_fn.cfg_body.as_ref().unwrap();

    // Should have multiple blocks (entry with Branch, then, else, merge)
    assert!(cfg.basic_blocks.len() >= 3, "if/else should create >= 3 blocks, got {}", cfg.basic_blocks.len());

    // Check that at least one block has a Branch terminator
    let has_branch = cfg.basic_blocks.iter().any(|bb| {
        matches!(bb.terminator, Terminator::Branch { .. })
    });
    assert!(has_branch, "if/else should produce a Branch terminator");
}

#[test]
fn test_cfg_fn_with_while() {
    let src = r#"
fn sum_to(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        total = total + i;
        i = i + 1;
    }
    total
}
print(sum_to(5));
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut mir = cjc_mir_exec::lower_to_mir(&program);
    mir.build_all_cfgs();

    let sum_fn = mir.functions.iter().find(|f| f.name == "sum_to").unwrap();
    let cfg = sum_fn.cfg_body.as_ref().unwrap();

    // While loop creates: pre-loop, header, body, exit blocks
    assert!(cfg.basic_blocks.len() >= 3, "while should create >= 3 blocks, got {}", cfg.basic_blocks.len());

    // Check for a loop header (has back-edge)
    let has_loop_header = (0..cfg.basic_blocks.len())
        .any(|i| cfg.is_loop_header(BlockId(i as u32)));
    assert!(has_loop_header, "while loop should create a loop header");
}

#[test]
fn test_cfg_fn_with_return() {
    let src = r#"
fn early_return(x: i64) -> i64 {
    if x > 10 {
        return 10;
    }
    x
}
print(early_return(5));
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut mir = cjc_mir_exec::lower_to_mir(&program);
    mir.build_all_cfgs();

    let func = mir.functions.iter().find(|f| f.name == "early_return").unwrap();
    let cfg = func.cfg_body.as_ref().unwrap();

    // Should have Return terminators
    let return_count = cfg.basic_blocks.iter()
        .filter(|bb| matches!(bb.terminator, Terminator::Return(_)))
        .count();
    assert!(return_count >= 1, "should have at least one Return terminator");
}

// ── Structural properties ───────────────────────────────────────

#[test]
fn test_cfg_entry_has_no_predecessors() {
    let src = r#"
fn id(x: i64) -> i64 { x }
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut mir = cjc_mir_exec::lower_to_mir(&program);
    mir.build_all_cfgs();

    let func = mir.functions.iter().find(|f| f.name == "id").unwrap();
    let cfg = func.cfg_body.as_ref().unwrap();
    let preds = cfg.predecessors();
    assert_eq!(preds[0].len(), 0, "entry block should have no predecessors");
}

#[test]
fn test_cfg_all_blocks_reachable_simple() {
    let src = r#"
fn compute(x: i64) -> i64 {
    let y: i64 = x * 2;
    y + 1
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut mir = cjc_mir_exec::lower_to_mir(&program);
    mir.build_all_cfgs();

    let func = mir.functions.iter().find(|f| f.name == "compute").unwrap();
    let cfg = func.cfg_body.as_ref().unwrap();

    // For straight-line code, there should be exactly 1 block
    // (all statements in the entry block)
    assert_eq!(cfg.basic_blocks.len(), 1,
        "straight-line code should produce exactly 1 block, got {}",
        cfg.basic_blocks.len());
}

// ── build_cfg / build_all_cfgs ──────────────────────────────────

#[test]
fn test_build_cfg_populates_field() {
    let src = "fn f(x: i64) -> i64 { x }";
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut mir = cjc_mir_exec::lower_to_mir(&program);

    // Before build: cfg_body is None
    for func in &mir.functions {
        assert!(func.cfg_body.is_none(), "cfg_body should be None before build");
    }

    // After build: cfg_body is Some
    mir.build_all_cfgs();
    for func in &mir.functions {
        assert!(func.cfg_body.is_some(), "cfg_body should be Some after build for fn {}", func.name);
    }
}

#[test]
fn test_build_cfg_on_single_function() {
    let src = "fn g(x: i64) -> i64 { x * 2 }";
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut mir = cjc_mir_exec::lower_to_mir(&program);

    // Build CFG only on one function
    let func = mir.functions.iter_mut().find(|f| f.name == "g").unwrap();
    func.build_cfg();
    assert!(func.cfg_body.is_some());
}

// ── CFG from tree-form MIR (unit level) ─────────────────────────

#[test]
fn test_cfg_builder_empty_body() {
    use cjc_mir::{MirBody, MirExpr, MirExprKind};
    let body = MirBody {
        stmts: vec![],
        result: Some(Box::new(MirExpr { kind: MirExprKind::IntLit(0) })),
    };
    let cfg = CfgBuilder::build(&body);
    assert_eq!(cfg.basic_blocks.len(), 1);
    assert!(matches!(cfg.entry_block().terminator, Terminator::Return(Some(_))));
}

#[test]
fn test_cfg_builder_let_binding() {
    use cjc_mir::{MirBody, MirStmt, MirExpr, MirExprKind};
    let body = MirBody {
        stmts: vec![
            MirStmt::Let {
                name: "x".into(),
                mutable: false,
                init: MirExpr { kind: MirExprKind::IntLit(42) },
                alloc_hint: None,
            },
        ],
        result: Some(Box::new(MirExpr { kind: MirExprKind::Var("x".into()) })),
    };
    let cfg = CfgBuilder::build(&body);
    assert_eq!(cfg.basic_blocks.len(), 1);
    assert_eq!(cfg.entry_block().statements.len(), 1);
}

// ── Determinism ─────────────────────────────────────────────────

#[test]
fn test_cfg_deterministic() {
    let src = r#"
fn complex(x: i64) -> i64 {
    let mut sum: i64 = 0;
    if x > 0 {
        let mut i: i64 = 0;
        while i < x {
            sum = sum + i;
            i = i + 1;
        }
    } else {
        sum = 0 - x;
    }
    sum
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());

    // Build CFG twice and verify identical structure
    let mir1 = cjc_mir_exec::lower_to_mir(&program);
    let mir2 = cjc_mir_exec::lower_to_mir(&program);

    let func1 = mir1.functions.iter().find(|f| f.name == "complex").unwrap();
    let func2 = mir2.functions.iter().find(|f| f.name == "complex").unwrap();

    let cfg1 = CfgBuilder::build(&func1.body);
    let cfg2 = CfgBuilder::build(&func2.body);

    assert_eq!(cfg1.basic_blocks.len(), cfg2.basic_blocks.len(),
        "CFG should be deterministic: same number of blocks");

    for (b1, b2) in cfg1.basic_blocks.iter().zip(cfg2.basic_blocks.iter()) {
        assert_eq!(b1.id, b2.id, "block IDs should match");
        assert_eq!(b1.statements.len(), b2.statements.len(),
            "block {} statement count should match", b1.id.0);
        assert_eq!(
            format!("{:?}", b1.terminator),
            format!("{:?}", b2.terminator),
            "block {} terminator should match", b1.id.0
        );
    }
}

// ── Successors / Predecessors ───────────────────────────────────

#[test]
fn test_cfg_successor_counts() {
    // Goto: 1 successor, Branch: 2 successors, Return: 0 successors
    let goto = Terminator::Goto(BlockId(1));
    assert_eq!(goto.successors().len(), 1);

    let branch = Terminator::Branch {
        cond: cjc_mir::MirExpr { kind: cjc_mir::MirExprKind::BoolLit(true) },
        then_block: BlockId(2),
        else_block: BlockId(3),
    };
    assert_eq!(branch.successors().len(), 2);

    let ret = Terminator::Return(None);
    assert_eq!(ret.successors().len(), 0);

    let unreach = Terminator::Unreachable;
    assert_eq!(unreach.successors().len(), 0);
}
