// Phase 10 — test_tidy_mutate_type_promotion
// Type promotion rules: Int + Float → Float column.
use cjc_data::{Column, DataFrame, DBinOp, DExpr};

fn make_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("i".into(), Column::Int(vec![1, 2, 3])),
        ("f".into(), Column::Float(vec![0.5, 1.5, 2.5])),
        ("b".into(), Column::Bool(vec![true, false, true])),
    ])
    .unwrap()
}

#[test]
fn test_tidy_mutate_type_promotion() {
    let df = make_df();
    let view = df.tidy();

    // Int + Float column → should produce Float result
    let frame = view
        .mutate(&[("sum_if", DExpr::BinOp {
            op: DBinOp::Add,
            left: Box::new(DExpr::Col("i".into())),
            right: Box::new(DExpr::Col("f".into())),
        })])
        .unwrap();

    let b = frame.borrow();
    let col = b.get_column("sum_if").unwrap();
    assert!(
        matches!(col, Column::Float(_)),
        "Int+Float must promote to Float, got {:?}", col
    );
    if let Column::Float(v) = col {
        // 1+0.5=1.5, 2+1.5=3.5, 3+2.5=5.5
        let expected = [1.5f64, 3.5, 5.5];
        for (a, b) in v.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "mismatch: {} vs {}",
                a,
                b
            );
        }
    }
}

#[test]
fn test_tidy_mutate_int_plus_int_stays_int() {
    let df = make_df();
    let frame = df
        .tidy()
        .mutate(&[("doubled", DExpr::BinOp {
            op: DBinOp::Add,
            left: Box::new(DExpr::Col("i".into())),
            right: Box::new(DExpr::Col("i".into())),
        })])
        .unwrap();
    let b = frame.borrow();
    let col = b.get_column("doubled").unwrap();
    assert!(matches!(col, Column::Int(_)), "Int+Int must stay Int");
    if let Column::Int(v) = col {
        assert_eq!(*v, vec![2i64, 4, 6]);
    }
}

#[test]
fn test_tidy_mutate_float_plus_literal_int_promotes() {
    // Float col + LitInt → Float (literal ints are eval'd as Int, promoted)
    let df = make_df();
    let frame = df
        .tidy()
        .mutate(&[("fp1", DExpr::BinOp {
            op: DBinOp::Add,
            left: Box::new(DExpr::Col("f".into())),
            right: Box::new(DExpr::LitInt(10)),
        })])
        .unwrap();
    let b = frame.borrow();
    let col = b.get_column("fp1").unwrap();
    // f + 10: 10.5, 11.5, 12.5
    assert!(matches!(col, Column::Float(_)));
    if let Column::Float(v) = col {
        assert!((v[0] - 10.5).abs() < 1e-10);
        assert!((v[1] - 11.5).abs() < 1e-10);
        assert!((v[2] - 12.5).abs() < 1e-10);
    }
}

#[test]
fn test_tidy_mutate_empty_df_type_promotion_no_panic() {
    // Mutate on empty DataFrame must succeed and create correct empty columns.
    let df = DataFrame::from_columns(vec![
        ("i".into(), Column::Int(vec![])),
        ("f".into(), Column::Float(vec![])),
    ])
    .unwrap();

    let frame = df
        .tidy()
        .mutate(&[("sum_if", DExpr::BinOp {
            op: DBinOp::Add,
            left: Box::new(DExpr::Col("i".into())),
            right: Box::new(DExpr::Col("f".into())),
        })])
        .unwrap();
    let b = frame.borrow();
    let col = b.get_column("sum_if").unwrap();
    // Empty frame — column is created but empty (Float default for empty)
    assert_eq!(col.len(), 0);
}
