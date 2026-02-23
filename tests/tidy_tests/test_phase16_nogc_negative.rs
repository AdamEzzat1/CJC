// Phase 16: NoGC negative tests for Phase 13-16 operations
//
// Confirms that materialising ops (pivot, bind, mutate_across, typed joins)
// are correctly REJECTED inside @nogc, and that view-only ops (relocate,
// drop_cols, group_by_fast) are ACCEPTED.

use cjc_mir::{
    MirBody, MirExpr, MirExprKind, MirFunction, MirFnId, MirProgram, MirStmt,
};
use cjc_mir::nogc_verify::verify_nogc;

fn mk_expr(kind: MirExprKind) -> MirExpr { MirExpr { kind } }
fn mk_call(name: &str) -> MirExpr {
    mk_expr(MirExprKind::Call {
        callee: Box::new(mk_expr(MirExprKind::Var(name.to_string()))),
        args: vec![],
    })
}
fn mk_fn(name: &str, is_nogc: bool, calls: &[&str]) -> MirFunction {
    MirFunction {
        id: MirFnId(0),
        name: name.to_string(),
        type_params: vec![],
        params: vec![],
        return_type: None,
        body: MirBody {
            stmts: calls.iter().map(|c| MirStmt::Expr(mk_call(c))).collect(),
            result: None,
        },
        is_nogc,
    }
}
fn mk_program(fns: Vec<MirFunction>) -> MirProgram {
    MirProgram { functions: fns, struct_defs: vec![], enum_defs: vec![], entry: MirFnId(0) }
}

// ── Materialising ops rejected inside @nogc ───────────────────────────────

#[test]
fn test_nogc_pivot_longer_rejected() {
    let prog = mk_program(vec![mk_fn("f", true, &["tidy_pivot_longer"])]);
    let errs = verify_nogc(&prog).unwrap_err();
    assert!(!errs.is_empty(), "pivot_longer should be rejected in @nogc");
}

#[test]
fn test_nogc_pivot_wider_rejected() {
    let prog = mk_program(vec![mk_fn("f", true, &["tidy_pivot_wider"])]);
    let errs = verify_nogc(&prog).unwrap_err();
    assert!(!errs.is_empty());
}

#[test]
fn test_nogc_bind_rows_rejected() {
    let prog = mk_program(vec![mk_fn("f", true, &["tidy_bind_rows"])]);
    let errs = verify_nogc(&prog).unwrap_err();
    assert!(!errs.is_empty());
}

#[test]
fn test_nogc_bind_cols_rejected() {
    let prog = mk_program(vec![mk_fn("f", true, &["tidy_bind_cols"])]);
    let errs = verify_nogc(&prog).unwrap_err();
    assert!(!errs.is_empty());
}

#[test]
fn test_nogc_mutate_across_rejected() {
    let prog = mk_program(vec![mk_fn("f", true, &["tidy_mutate_across"])]);
    let errs = verify_nogc(&prog).unwrap_err();
    assert!(!errs.is_empty());
}

#[test]
fn test_nogc_right_join_rejected() {
    let prog = mk_program(vec![mk_fn("f", true, &["tidy_right_join"])]);
    let errs = verify_nogc(&prog).unwrap_err();
    assert!(!errs.is_empty());
}

#[test]
fn test_nogc_full_join_rejected() {
    let prog = mk_program(vec![mk_fn("f", true, &["tidy_full_join"])]);
    let errs = verify_nogc(&prog).unwrap_err();
    assert!(!errs.is_empty());
}

#[test]
fn test_nogc_inner_join_typed_rejected() {
    let prog = mk_program(vec![mk_fn("f", true, &["tidy_inner_join_typed"])]);
    let errs = verify_nogc(&prog).unwrap_err();
    assert!(!errs.is_empty());
}

#[test]
fn test_nogc_summarise_across_rejected() {
    let prog = mk_program(vec![mk_fn("f", true, &["tidy_summarise_across"])]);
    let errs = verify_nogc(&prog).unwrap_err();
    assert!(!errs.is_empty());
}

#[test]
fn test_nogc_rename_rejected() {
    // tidy_rename rebuilds base DataFrame — not @nogc safe
    let prog = mk_program(vec![mk_fn("f", true, &["tidy_rename"])]);
    let errs = verify_nogc(&prog).unwrap_err();
    assert!(!errs.is_empty());
}

// ── View-only ops accepted inside @nogc ──────────────────────────────────

#[test]
fn test_nogc_relocate_accepted() {
    let prog = mk_program(vec![mk_fn("f", true, &["tidy_relocate"])]);
    assert!(verify_nogc(&prog).is_ok(), "tidy_relocate should be @nogc safe");
}

#[test]
fn test_nogc_drop_cols_accepted() {
    let prog = mk_program(vec![mk_fn("f", true, &["tidy_drop_cols"])]);
    assert!(verify_nogc(&prog).is_ok(), "tidy_drop_cols should be @nogc safe");
}

#[test]
fn test_nogc_group_by_fast_accepted() {
    let prog = mk_program(vec![mk_fn("f", true, &["tidy_group_by_fast"])]);
    assert!(verify_nogc(&prog).is_ok(), "tidy_group_by_fast should be @nogc safe");
}

#[test]
fn test_nogc_clean_function_still_passes() {
    // Regression: a clean @nogc function with only safe builtins passes
    let prog = mk_program(vec![mk_fn("f", true, &["tidy_filter", "tidy_relocate"])]);
    assert!(verify_nogc(&prog).is_ok());
}
