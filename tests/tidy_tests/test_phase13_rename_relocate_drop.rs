// Phase 13: rename, relocate, drop_cols edge-case tests

use cjc_data::{Column, DataFrame, RelocatePos, TidyError};

fn make_xyz() -> DataFrame {
    DataFrame::from_columns(vec![
        ("x".into(), Column::Int(vec![1, 2, 3])),
        ("y".into(), Column::Float(vec![1.0, 2.0, 3.0])),
        ("z".into(), Column::Str(vec!["a".into(), "b".into(), "c".into()])),
    ])
    .unwrap()
}

// ── rename basic ──────────────────────────────────────────────────────────

#[test]
fn test_rename_basic() {
    let df = make_xyz();
    let v = df.tidy();
    let renamed = v.rename(&[("x", "new_x")]).unwrap();
    let names = renamed.column_names();
    assert!(names.contains(&"new_x"));
    assert!(!names.contains(&"x"));
}

#[test]
fn test_rename_multiple() {
    let df = make_xyz();
    let v = df.tidy();
    let renamed = v.rename(&[("x", "a"), ("y", "b")]).unwrap();
    let names = renamed.column_names();
    assert_eq!(names, vec!["a", "b", "z"]);
}

#[test]
fn test_rename_noop_same_name() {
    let df = make_xyz();
    let v = df.tidy();
    let renamed = v.rename(&[("x", "x")]).unwrap();
    assert_eq!(renamed.column_names(), vec!["x", "y", "z"]);
}

#[test]
fn test_rename_values_preserved() {
    let df = make_xyz();
    let v = df.tidy();
    let renamed = v.rename(&[("x", "new_x")]).unwrap();
    let mat = renamed.materialize().unwrap();
    if let Column::Int(vals) = mat.get_column("new_x").unwrap() {
        assert_eq!(vals, &[1, 2, 3]);
    } else {
        panic!("expected Int");
    }
}

// ── rename error cases ────────────────────────────────────────────────────

#[test]
fn test_rename_unknown_col_error() {
    let df = make_xyz();
    let v = df.tidy();
    let err = v.rename(&[("nonexistent", "new")]).unwrap_err();
    assert!(matches!(err, TidyError::ColumnNotFound(_)));
}

#[test]
fn test_rename_collision_error() {
    let df = make_xyz();
    let v = df.tidy();
    // Rename "x" to "y" which already exists
    let err = v.rename(&[("x", "y")]).unwrap_err();
    assert!(matches!(err, TidyError::DuplicateColumn(_)));
}

#[test]
fn test_rename_mask_preserved() {
    let df = make_xyz();
    let v = df.tidy();
    let filtered = v.filter(&cjc_data::DExpr::BinOp {
        op: cjc_data::DBinOp::Gt,
        left: Box::new(cjc_data::DExpr::Col("x".into())),
        right: Box::new(cjc_data::DExpr::LitInt(1)),
    }).unwrap();
    let renamed = filtered.rename(&[("x", "xx")]).unwrap();
    // Should still have 2 visible rows
    assert_eq!(renamed.nrows(), 2);
}

// ── relocate ─────────────────────────────────────────────────────────────

#[test]
fn test_relocate_to_front() {
    let df = make_xyz();
    let v = df.tidy();
    let rel = v.relocate(&["z"], RelocatePos::Front).unwrap();
    assert_eq!(rel.column_names(), vec!["z", "x", "y"]);
}

#[test]
fn test_relocate_to_back() {
    let df = make_xyz();
    let v = df.tidy();
    let rel = v.relocate(&["x"], RelocatePos::Back).unwrap();
    assert_eq!(rel.column_names(), vec!["y", "z", "x"]);
}

#[test]
fn test_relocate_before_anchor() {
    let df = make_xyz();
    let v = df.tidy();
    let rel = v.relocate(&["z"], RelocatePos::Before("y")).unwrap();
    assert_eq!(rel.column_names(), vec!["x", "z", "y"]);
}

#[test]
fn test_relocate_after_anchor() {
    let df = make_xyz();
    let v = df.tidy();
    let rel = v.relocate(&["x"], RelocatePos::After("y")).unwrap();
    assert_eq!(rel.column_names(), vec!["y", "x", "z"]);
}

#[test]
fn test_relocate_multiple_cols() {
    let df = make_xyz();
    let v = df.tidy();
    let rel = v.relocate(&["y", "z"], RelocatePos::Front).unwrap();
    assert_eq!(rel.column_names(), vec!["y", "z", "x"]);
}

#[test]
fn test_relocate_unknown_col_error() {
    let df = make_xyz();
    let v = df.tidy();
    let err = v.relocate(&["nonexistent"], RelocatePos::Front).unwrap_err();
    assert!(matches!(err, TidyError::ColumnNotFound(_)));
}

#[test]
fn test_relocate_unknown_anchor_error() {
    let df = make_xyz();
    let v = df.tidy();
    let err = v.relocate(&["x"], RelocatePos::Before("nonexistent")).unwrap_err();
    assert!(matches!(err, TidyError::ColumnNotFound(_)));
}

#[test]
fn test_relocate_values_preserved() {
    let df = make_xyz();
    let v = df.tidy();
    let rel = v.relocate(&["z"], RelocatePos::Front).unwrap();
    let mat = rel.materialize().unwrap();
    // z should have its original values
    if let Column::Str(vals) = mat.get_column("z").unwrap() {
        assert_eq!(vals, &["a", "b", "c"]);
    }
}

// ── drop_cols ─────────────────────────────────────────────────────────────

#[test]
fn test_drop_cols_basic() {
    let df = make_xyz();
    let v = df.tidy();
    let dropped = v.drop_cols(&["y"]).unwrap();
    let names = dropped.column_names();
    assert!(!names.contains(&"y"));
    assert!(names.contains(&"x"));
    assert!(names.contains(&"z"));
}

#[test]
fn test_drop_cols_multiple() {
    let df = make_xyz();
    let v = df.tidy();
    let dropped = v.drop_cols(&["x", "y"]).unwrap();
    assert_eq!(dropped.column_names(), vec!["z"]);
}

#[test]
fn test_drop_cols_all_cols() {
    let df = make_xyz();
    let v = df.tidy();
    let dropped = v.drop_cols(&["x", "y", "z"]).unwrap();
    assert_eq!(dropped.ncols(), 0);
    assert_eq!(dropped.nrows(), 3);
}

#[test]
fn test_drop_cols_unknown_error() {
    let df = make_xyz();
    let v = df.tidy();
    let err = v.drop_cols(&["nonexistent"]).unwrap_err();
    assert!(matches!(err, TidyError::ColumnNotFound(_)));
}

#[test]
fn test_drop_cols_preserves_row_count() {
    let df = make_xyz();
    let v = df.tidy().filter(&cjc_data::DExpr::BinOp {
        op: cjc_data::DBinOp::Gt,
        left: Box::new(cjc_data::DExpr::Col("x".into())),
        right: Box::new(cjc_data::DExpr::LitInt(1)),
    }).unwrap();
    let dropped = v.drop_cols(&["y"]).unwrap();
    assert_eq!(dropped.nrows(), 2);
}
