// Phase 10 — test_tidy_mutate_ordering
// Multiple assignments in one mutate call use SNAPSHOT semantics:
// each assignment sees columns as they were BEFORE the mutate call,
// not columns created by earlier assignments in the same call.
use cjc_data::{Column, DataFrame, DBinOp, DExpr, TidyError};

fn make_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("a".into(), Column::Int(vec![1, 2, 3])),
        ("b".into(), Column::Int(vec![10, 20, 30])),
    ])
    .unwrap()
}

#[test]
fn test_tidy_mutate_ordering() {
    // Snapshot semantics: mutate(x = a+1, y = x+1) where x is new.
    // y can NOT reference x because x didn't exist at snapshot time.
    let df = make_df();
    let err = df
        .tidy()
        .mutate(&[
            ("x", DExpr::BinOp {
                op: DBinOp::Add,
                left: Box::new(DExpr::Col("a".into())),
                right: Box::new(DExpr::LitInt(1)),
            }),
            ("y", DExpr::BinOp {
                op: DBinOp::Add,
                left: Box::new(DExpr::Col("x".into())), // x is NEW — not in snapshot
                right: Box::new(DExpr::LitInt(1)),
            }),
        ]);
    // Should fail: x not in snapshot
    assert!(
        err.is_err(),
        "should fail because x not in snapshot at call time"
    );
    assert!(matches!(err.unwrap_err(), TidyError::ColumnNotFound(_)));
}

#[test]
fn test_tidy_mutate_ordering_independent_assignments() {
    // Two independent new columns: both reference only original columns.
    let df = make_df();
    let frame = df
        .tidy()
        .mutate(&[
            ("a_plus_1", DExpr::BinOp {
                op: DBinOp::Add,
                left: Box::new(DExpr::Col("a".into())),
                right: Box::new(DExpr::LitInt(1)),
            }),
            ("b_times_2", DExpr::BinOp {
                op: DBinOp::Mul,
                left: Box::new(DExpr::Col("b".into())),
                right: Box::new(DExpr::LitInt(2)),
            }),
        ])
        .unwrap();

    let b = frame.borrow();
    if let Column::Int(v) = b.get_column("a_plus_1").unwrap() {
        assert_eq!(*v, vec![2i64, 3, 4]);
    }
    if let Column::Int(v) = b.get_column("b_times_2").unwrap() {
        assert_eq!(*v, vec![20i64, 40, 60]);
    }
}

#[test]
fn test_tidy_mutate_duplicate_target_rejected() {
    let df = make_df();
    let err = df
        .tidy()
        .mutate(&[
            ("x", DExpr::Col("a".into())),
            ("x", DExpr::Col("b".into())), // duplicate target
        ])
        .unwrap_err();
    assert!(matches!(err, TidyError::DuplicateColumn(_)));
}

#[test]
fn test_tidy_mutate_overwrite_existing_col() {
    // Overwriting an existing column is allowed.
    let df = make_df();
    let frame = df
        .tidy()
        .mutate(&[("a", DExpr::BinOp {
            op: DBinOp::Mul,
            left: Box::new(DExpr::Col("a".into())),
            right: Box::new(DExpr::LitInt(100)),
        })])
        .unwrap();
    let b = frame.borrow();
    if let Column::Int(v) = b.get_column("a").unwrap() {
        assert_eq!(*v, vec![100i64, 200, 300]);
    }
}

#[test]
fn test_tidy_mutate_sequential_via_view_chain() {
    // To simulate sequential evaluation where later step sees earlier result,
    // chain two separate mutate calls (each with its own snapshot).
    let df = make_df();
    let frame1 = df
        .tidy()
        .mutate(&[("x", DExpr::BinOp {
            op: DBinOp::Add,
            left: Box::new(DExpr::Col("a".into())),
            right: Box::new(DExpr::LitInt(1)),
        })])
        .unwrap();

    // Second mutate sees x
    let frame2 = frame1
        .view()
        .mutate(&[("y", DExpr::BinOp {
            op: DBinOp::Add,
            left: Box::new(DExpr::Col("x".into())),
            right: Box::new(DExpr::LitInt(1)),
        })])
        .unwrap();

    let b = frame2.borrow();
    if let Column::Int(v) = b.get_column("y").unwrap() {
        // a=1→x=2→y=3; a=2→x=3→y=4; a=3→x=4→y=5
        assert_eq!(*v, vec![3i64, 4, 5]);
    }
}
