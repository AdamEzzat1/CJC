// Phase 13: pivot_longer edge-case tests
//
// All tests in tests/tidy_tests/ per the Phase 13-16 spec contract.

use cjc_data::{Column, DataFrame, TidyError};

fn wide_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("id".into(), Column::Int(vec![1, 2, 3])),
        ("a".into(), Column::Float(vec![1.0, 4.0, 7.0])),
        ("b".into(), Column::Float(vec![2.0, 5.0, 8.0])),
        ("c".into(), Column::Float(vec![3.0, 6.0, 9.0])),
    ])
    .unwrap()
}

// ── pivot_longer basic ────────────────────────────────────────────────────

#[test]
fn test_pivot_longer_basic_row_count() {
    let df = wide_df();
    let v = df.tidy();
    let long = v.pivot_longer(&["a", "b", "c"], "name", "value").unwrap();
    let b = long.borrow();
    // 3 source rows × 3 value cols = 9 output rows
    assert_eq!(b.nrows(), 9);
}

#[test]
fn test_pivot_longer_schema_id_name_value() {
    let df = wide_df();
    let v = df.tidy();
    let long = v.pivot_longer(&["a", "b", "c"], "variable", "val").unwrap();
    let b = long.borrow();
    let names = b.column_names();
    // id col first, then "variable", then "val"
    assert_eq!(names, vec!["id", "variable", "val"]);
}

#[test]
fn test_pivot_longer_row_order_stable() {
    let df = wide_df();
    let v = df.tidy();
    let long = v.pivot_longer(&["a", "b", "c"], "name", "value").unwrap();
    let b = long.borrow();
    // Row 0: id=1,name="a",value=1.0  Row 1: id=1,name="b",value=2.0  etc.
    if let Column::Int(ids) = b.get_column("id").unwrap() {
        assert_eq!(ids[0], 1);
        assert_eq!(ids[1], 1);
        assert_eq!(ids[2], 1);
        assert_eq!(ids[3], 2);
        assert_eq!(ids[6], 3);
    } else {
        panic!("expected Int id column");
    }
}

#[test]
fn test_pivot_longer_col_order_within_row() {
    // Columns appear in value_cols list order within each source row
    let df = wide_df();
    let v = df.tidy();
    let long = v.pivot_longer(&["c", "a", "b"], "name", "value").unwrap();
    let b = long.borrow();
    if let Column::Str(names) = b.get_column("name").unwrap() {
        // First 3 rows (source row 0): c, a, b
        assert_eq!(names[0], "c");
        assert_eq!(names[1], "a");
        assert_eq!(names[2], "b");
    } else {
        panic!("expected Str name column");
    }
}

#[test]
fn test_pivot_longer_values_correct() {
    let df = wide_df();
    let v = df.tidy();
    let long = v.pivot_longer(&["a", "b"], "name", "value").unwrap();
    let b = long.borrow();
    // 3 rows × 2 cols = 6 rows; values should be: 1.0, 2.0, 4.0, 5.0, 7.0, 8.0
    if let Column::Float(vals) = b.get_column("value").unwrap() {
        assert_eq!(vals, &[1.0, 2.0, 4.0, 5.0, 7.0, 8.0]);
    } else {
        panic!("expected Float value column");
    }
}

// ── pivot_longer edge cases ───────────────────────────────────────────────

#[test]
fn test_pivot_longer_empty_df_zero_rows() {
    let df = DataFrame::from_columns(vec![
        ("id".into(), Column::Int(vec![])),
        ("a".into(), Column::Float(vec![])),
        ("b".into(), Column::Float(vec![])),
    ])
    .unwrap();
    let v = df.tidy();
    let long = v.pivot_longer(&["a", "b"], "name", "value").unwrap();
    let b = long.borrow();
    assert_eq!(b.nrows(), 0);
    // Schema should still be correct
    assert!(b.get_column("name").is_some());
    assert!(b.get_column("value").is_some());
}

#[test]
fn test_pivot_longer_empty_value_cols_error() {
    let df = wide_df();
    let v = df.tidy();
    let err = v.pivot_longer(&[], "name", "value").unwrap_err();
    assert!(
        matches!(err, TidyError::Internal(ref s) if s.contains("empty selection")),
        "expected empty_selection error, got: {:?}", err
    );
}

#[test]
fn test_pivot_longer_unknown_col_error() {
    let df = wide_df();
    let v = df.tidy();
    let err = v.pivot_longer(&["a", "nonexistent"], "name", "value").unwrap_err();
    assert!(matches!(err, TidyError::ColumnNotFound(_)));
}

#[test]
fn test_pivot_longer_duplicate_col_error() {
    let df = wide_df();
    let v = df.tidy();
    let err = v.pivot_longer(&["a", "a"], "name", "value").unwrap_err();
    assert!(matches!(err, TidyError::DuplicateColumn(_)));
}

#[test]
fn test_pivot_longer_mixed_type_error() {
    // a is Float, id is Int — mixing types should error
    let df = wide_df();
    let v = df.tidy();
    let err = v.pivot_longer(&["a", "id"], "name", "value").unwrap_err();
    assert!(matches!(err, TidyError::TypeMismatch { .. }));
}

#[test]
fn test_pivot_longer_str_values() {
    let df = DataFrame::from_columns(vec![
        ("id".into(), Column::Int(vec![1, 2])),
        ("x".into(), Column::Str(vec!["hello".into(), "world".into()])),
        ("y".into(), Column::Str(vec!["foo".into(), "bar".into()])),
    ])
    .unwrap();
    let v = df.tidy();
    let long = v.pivot_longer(&["x", "y"], "name", "value").unwrap();
    let b = long.borrow();
    assert_eq!(b.nrows(), 4);
    if let Column::Str(vals) = b.get_column("value").unwrap() {
        assert_eq!(vals, &["hello", "foo", "world", "bar"]);
    } else {
        panic!("expected Str value col");
    }
}

#[test]
fn test_pivot_longer_determinism() {
    // Running twice produces identical results
    let df = wide_df();
    let v1 = df.clone().tidy();
    let v2 = df.tidy();
    let r1 = v1.pivot_longer(&["a", "b", "c"], "n", "v").unwrap();
    let r2 = v2.pivot_longer(&["a", "b", "c"], "n", "v").unwrap();
    let b1 = r1.borrow();
    let b2 = r2.borrow();
    assert_eq!(
        b1.get_column("v").unwrap().get_display(0),
        b2.get_column("v").unwrap().get_display(0)
    );
    assert_eq!(b1.nrows(), b2.nrows());
}

#[test]
fn test_pivot_longer_after_filter() {
    // Filter first, then pivot_longer — only visible rows pivoted
    let df = wide_df();
    let v = df.tidy();
    let filtered = v.filter(&cjc_data::DExpr::BinOp {
        op: cjc_data::DBinOp::Gt,
        left: Box::new(cjc_data::DExpr::Col("id".into())),
        right: Box::new(cjc_data::DExpr::LitInt(1)),
    }).unwrap();
    // 2 rows remaining (id=2,3), 3 value cols → 6 output rows
    let long = filtered.pivot_longer(&["a", "b", "c"], "name", "value").unwrap();
    let b = long.borrow();
    assert_eq!(b.nrows(), 6);
}
