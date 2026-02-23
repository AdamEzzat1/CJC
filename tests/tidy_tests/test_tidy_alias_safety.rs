// Phase 10 — test_tidy_alias_safety
// Mutating one TidyFrame must not corrupt other views sharing the same base.
use cjc_data::{Column, DataFrame, DBinOp, DExpr};

fn make_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("x".into(), Column::Int(vec![1, 2, 3, 4, 5])),
    ])
    .unwrap()
}

#[test]
fn test_tidy_alias_safety() {
    let df = make_df();
    // Both views share the same Rc<DataFrame> base
    let view_a = df.tidy();
    let view_b = view_a.clone(); // same Rc

    // Materialize view_a into a mutable TidyFrame
    let mut frame_a = view_a.mutate(&[("doubled", DExpr::BinOp {
        op: DBinOp::Mul,
        left: Box::new(DExpr::Col("x".into())),
        right: Box::new(DExpr::LitInt(2)),
    })]).unwrap();

    // Now mutate frame_a in-place (copy-on-write inside TidyFrame)
    frame_a.mutate(&[("tripled", DExpr::BinOp {
        op: DBinOp::Mul,
        left: Box::new(DExpr::Col("x".into())),
        right: Box::new(DExpr::LitInt(3)),
    })]).unwrap();

    // view_b must still see original x values unmodified (base is Rc-shared but immutable)
    let mat_b = view_b.materialize().unwrap();
    if let Column::Int(v) = mat_b.get_column("x").unwrap() {
        assert_eq!(*v, vec![1i64, 2, 3, 4, 5], "view_b base must be unmodified");
    }

    // frame_a has both doubled and tripled
    let b = frame_a.borrow();
    assert!(b.get_column("doubled").is_some());
    assert!(b.get_column("tripled").is_some());
}

#[test]
fn test_tidy_alias_safety_two_frames_independent() {
    let df = make_df();
    let view = df.tidy();

    // Two separate mutated frames from the same view
    let frame_a = view.mutate(&[("a", DExpr::BinOp {
        op: DBinOp::Mul,
        left: Box::new(DExpr::Col("x".into())),
        right: Box::new(DExpr::LitInt(10)),
    })]).unwrap();

    let frame_b = view.mutate(&[("b", DExpr::BinOp {
        op: DBinOp::Mul,
        left: Box::new(DExpr::Col("x".into())),
        right: Box::new(DExpr::LitInt(100)),
    })]).unwrap();

    // frame_a has column "a", not "b"
    let ba = frame_a.borrow();
    assert!(ba.get_column("a").is_some());
    assert!(ba.get_column("b").is_none(), "frame_a must not have frame_b's column");

    // frame_b has column "b", not "a"
    let bb = frame_b.borrow();
    assert!(bb.get_column("b").is_some());
    assert!(bb.get_column("a").is_none(), "frame_b must not have frame_a's column");
}

#[test]
fn test_tidy_alias_safety_clone_then_mutate() {
    // Clone a TidyFrame (shares Rc), then mutate one copy.
    // The clone must be unaffected (CoW triggers deep copy).
    let df = make_df();
    let frame_original = df.tidy_mut();

    // Clone: shares Rc
    let mut frame_copy = frame_original.clone();

    // Mutate frame_copy — should trigger deep copy
    frame_copy.mutate(&[("new_col", DExpr::BinOp {
        op: DBinOp::Add,
        left: Box::new(DExpr::Col("x".into())),
        right: Box::new(DExpr::LitInt(99)),
    })]).unwrap();

    // Original must not have new_col
    let orig = frame_original.borrow();
    assert!(
        orig.get_column("new_col").is_none(),
        "original must not be affected by clone mutation"
    );

    // Copy must have new_col
    let copy = frame_copy.borrow();
    assert!(copy.get_column("new_col").is_some());
}
