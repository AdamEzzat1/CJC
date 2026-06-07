//! End-to-end wiring tests for the CANA Phase-1 passive observer.
//!
//! These tests construct a hand-built `MirProgram` (no parser, no lowering
//! chain) and verify that:
//!
//! - `analyze_program` produces a `CanaReport` with the expected per-function
//!   feature shape.
//! - The JSON report is well-formed and includes every expected field.
//! - The legality gate plumbing works end-to-end (baseline = approved,
//!   strict-reduction violations propagate to the report).

use cjc_cana::{
    analyze_program, CanaReport, DefaultLegalityGate, LegalityGate, PassSequence, ProposedPass,
};
use cjc_mir::{
    MirBody, MirExpr, MirExprKind, MirFnId, MirFunction, MirParam, MirProgram, MirStmt,
};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn ekind(k: MirExprKind) -> MirExpr {
    MirExpr { kind: k }
}

fn fn_named(name: &str) -> MirFunction {
    MirFunction {
        id: MirFnId(0),
        name: name.to_string(),
        type_params: vec![],
        params: vec![],
        return_type: None,
        body: MirBody {
            stmts: vec![],
            result: None,
        },
        is_nogc: false,
        cfg_body: None,
        decorators: vec![],
        vis: cjc_ast::Visibility::Public,
        local_count: 0,
    }
}

fn program_of(fns: Vec<MirFunction>) -> MirProgram {
    let entry = fns.first().map(|f| f.id).unwrap_or(MirFnId(0));
    MirProgram {
        functions: fns,
        struct_defs: vec![],
        enum_defs: vec![],
        entry,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn analyze_program_on_empty_program_returns_well_formed_report() {
    let report: CanaReport = analyze_program(&program_of(vec![]));
    assert_eq!(report.schema_version, 1);
    assert_eq!(report.features.function_count(), 0);
    assert!(report.baseline_verdict.is_approved());

    let json = report.to_json();
    assert!(json.contains("\"schema_version\": 1"));
    assert!(json.contains("\"phase\": \"passive_observer\""));
    assert!(json.contains("\"per_fn\": {"));
}

#[test]
fn analyze_program_indexes_every_function_by_name() {
    let report = analyze_program(&program_of(vec![
        fn_named("__main"),
        fn_named("worker"),
        fn_named("init"),
    ]));
    assert_eq!(report.features.function_count(), 3);
    assert!(report.features.per_fn.contains_key("__main"));
    assert!(report.features.per_fn.contains_key("worker"));
    assert!(report.features.per_fn.contains_key("init"));
}

#[test]
fn json_includes_every_function_field() {
    let mut f = fn_named("__main");
    // Add an if statement so the CFG has at least one branch.
    f.body.stmts.push(MirStmt::If {
        cond: ekind(MirExprKind::BoolLit(true)),
        then_body: MirBody {
            stmts: vec![MirStmt::Expr(ekind(MirExprKind::IntLit(1)))],
            result: None,
        },
        else_body: Some(MirBody {
            stmts: vec![MirStmt::Expr(ekind(MirExprKind::IntLit(2)))],
            result: None,
        }),
    });

    let report = analyze_program(&program_of(vec![f]));
    let json = report.to_json();
    // Function block keys.
    for key in [
        "\"__main\":",
        "\"cfg\":",
        "\"block_count\":",
        "\"branch_count\":",
        "\"memory\":",
        "\"alloc_sites\":",
        "\"reductions\":",
        "\"strict_count\":",
        "\"has_strict_reduction\":",
    ] {
        assert!(
            json.contains(key),
            "JSON missing key {:?}\nFull JSON:\n{}",
            key,
            json
        );
    }
}

#[test]
fn legality_gate_rejects_unknown_function_end_to_end() {
    let report = analyze_program(&program_of(vec![fn_named("__main")]));
    let gate = DefaultLegalityGate::new();

    let mut seq = PassSequence::empty();
    seq.per_function.insert(
        "nonexistent".to_string(),
        vec![ProposedPass::Skip("dce".to_string())],
    );

    let verdict = gate.verify(
        &program_of(vec![fn_named("__main")]),
        &seq,
        &report.features,
    );
    assert!(!verdict.is_approved());
    let violations = verdict.violations().expect("should have violations");
    assert_eq!(violations.len(), 1);
}

#[test]
fn report_for_function_with_parameters_carries_them_through_program_hash() {
    let mut f = fn_named("solve");
    f.params.push(MirParam {
        name: "tol".to_string(),
        ty_name: "f64".to_string(),
        default: None,
        is_variadic: false,
    });
    f.params.push(MirParam {
        name: "max_iter".to_string(),
        ty_name: "i64".to_string(),
        default: None,
        is_variadic: false,
    });

    let r1 = analyze_program(&program_of(vec![f.clone()]));

    // Swapping parameter order changes the hash.
    let mut g = f.clone();
    g.params.swap(0, 1);
    let r2 = analyze_program(&program_of(vec![g]));
    assert_ne!(r1.features.program_hash, r2.features.program_hash);
}

#[test]
fn analyze_program_handles_deeply_nested_control_flow() {
    // Construct: if (cond) { while (cond) { if (cond) { ... } } }
    let inner_if = MirStmt::If {
        cond: ekind(MirExprKind::BoolLit(true)),
        then_body: MirBody {
            stmts: vec![MirStmt::Expr(ekind(MirExprKind::IntLit(1)))],
            result: None,
        },
        else_body: None,
    };
    let while_body = MirBody {
        stmts: vec![inner_if],
        result: None,
    };
    let while_stmt = MirStmt::While {
        cond: ekind(MirExprKind::BoolLit(true)),
        body: while_body,
    };
    let outer_if = MirStmt::If {
        cond: ekind(MirExprKind::BoolLit(true)),
        then_body: MirBody {
            stmts: vec![while_stmt],
            result: None,
        },
        else_body: None,
    };

    let mut f = fn_named("nested");
    f.body.stmts.push(outer_if);

    let report = analyze_program(&program_of(vec![f]));
    let feats = report.features.per_fn.get("nested").unwrap();
    // At least one loop, at least one branch, depth ≥ 1.
    assert!(feats.cfg.loop_count >= 1);
    assert!(feats.cfg.branch_count >= 1);
    assert!(feats.cfg.max_loop_depth >= 1);
}

#[test]
fn tensor_lit_increments_alloc_and_expr_count() {
    let mut f = fn_named("__main");
    f.body.stmts.push(MirStmt::Expr(ekind(MirExprKind::TensorLit {
        rows: vec![
            vec![ekind(MirExprKind::IntLit(1)), ekind(MirExprKind::IntLit(2))],
            vec![ekind(MirExprKind::IntLit(3)), ekind(MirExprKind::IntLit(4))],
        ],
    })));
    let report = analyze_program(&program_of(vec![f]));
    let feats = report.features.per_fn.get("__main").unwrap();
    assert_eq!(feats.memory.alloc_sites, 1);
    // 1 TensorLit + 4 IntLit = 5 expressions.
    assert_eq!(feats.memory.expr_count, 5);
}

#[test]
fn nogc_function_records_decorator_or_attribute() {
    let mut f = fn_named("hot");
    f.is_nogc = true;
    let r = analyze_program(&program_of(vec![f.clone()]));
    let mut g = f;
    g.is_nogc = false;
    let r2 = analyze_program(&program_of(vec![g]));
    // The nogc bit is hashed; changing it must change the program hash.
    assert_ne!(r.features.program_hash, r2.features.program_hash);
}
