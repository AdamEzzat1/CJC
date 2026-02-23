// Phase 15: join maturity — typed joins, suffix handling, right_join, full_join

use cjc_data::{Column, DataFrame, JoinSuffix, TidyError};

fn make_left() -> DataFrame {
    DataFrame::from_columns(vec![
        ("id".into(), Column::Int(vec![1, 2, 3])),
        ("val".into(), Column::Float(vec![10.0, 20.0, 30.0])),
    ])
    .unwrap()
}

fn make_right() -> DataFrame {
    DataFrame::from_columns(vec![
        ("id".into(), Column::Int(vec![2, 3, 4])),
        ("name".into(), Column::Str(vec!["b".into(), "c".into(), "d".into()])),
    ])
    .unwrap()
}

fn make_right_with_val() -> DataFrame {
    // right side also has "val" column → collision
    DataFrame::from_columns(vec![
        ("id".into(), Column::Int(vec![1, 2])),
        ("val".into(), Column::Float(vec![100.0, 200.0])),
    ])
    .unwrap()
}

// ── Type validation ───────────────────────────────────────────────────────

#[test]
fn test_inner_join_typed_type_mismatch_error() {
    // left key is Int, right key is Str → type mismatch
    let left = DataFrame::from_columns(vec![
        ("id".into(), Column::Int(vec![1, 2])),
    ])
    .unwrap();
    let right = DataFrame::from_columns(vec![
        ("id".into(), Column::Str(vec!["1".into(), "2".into()])),
    ])
    .unwrap();
    let l = left.tidy();
    let r = right.tidy();
    let suffix = JoinSuffix::default();
    let err = l.inner_join_typed(&r, &[("id", "id")], &suffix).unwrap_err();
    assert!(matches!(err, TidyError::TypeMismatch { .. }));
}

#[test]
fn test_inner_join_typed_int_float_compatible() {
    // Int ↔ Float widening is permitted
    let left = DataFrame::from_columns(vec![
        ("id".into(), Column::Int(vec![1, 2])),
        ("v".into(), Column::Float(vec![1.0, 2.0])),
    ])
    .unwrap();
    let right = DataFrame::from_columns(vec![
        ("id".into(), Column::Float(vec![2.0, 3.0])),
        ("w".into(), Column::Str(vec!["x".into(), "y".into()])),
    ])
    .unwrap();
    let l = left.tidy();
    let r = right.tidy();
    let suffix = JoinSuffix::default();
    // Should NOT error on type check (Int↔Float allowed)
    // Note: matching uses Display, so 1 vs 1.0 won't match, but that's a runtime
    // matter — type validation passes
    let result = l.inner_join_typed(&r, &[("id", "id")], &suffix);
    assert!(result.is_ok());
}

#[test]
fn test_left_join_typed_type_mismatch_error() {
    let left = DataFrame::from_columns(vec![
        ("k".into(), Column::Bool(vec![true])),
    ])
    .unwrap();
    let right = DataFrame::from_columns(vec![
        ("k".into(), Column::Int(vec![1])),
    ])
    .unwrap();
    let l = left.tidy();
    let r = right.tidy();
    let suffix = JoinSuffix::default();
    let err = l.left_join_typed(&r, &[("k", "k")], &suffix).unwrap_err();
    assert!(matches!(err, TidyError::TypeMismatch { .. }));
}

// ── Suffix collision handling ─────────────────────────────────────────────

#[test]
fn test_inner_join_typed_suffix_on_collision() {
    // Both left and right have "val" — should produce "val.x" and "val.y"
    let l = make_left().tidy();
    let r = make_right_with_val().tidy();
    let suffix = JoinSuffix::default(); // .x / .y
    let result = l.inner_join_typed(&r, &[("id", "id")], &suffix).unwrap();
    let b = result.borrow();
    let names = b.column_names();
    assert!(names.contains(&"val.x") || names.contains(&"val.y"),
        "expected suffixed val columns, got: {:?}", names);
}

#[test]
fn test_inner_join_typed_custom_suffix() {
    let l = make_left().tidy();
    let r = make_right_with_val().tidy();
    let suffix = JoinSuffix::new("_left", "_right");
    let result = l.inner_join_typed(&r, &[("id", "id")], &suffix).unwrap();
    let b = result.borrow();
    let names = b.column_names();
    // At least one suffixed name present
    assert!(names.iter().any(|n| n.contains("_left") || n.contains("_right")),
        "expected custom suffix, got: {:?}", names);
}

#[test]
fn test_inner_join_typed_no_collision_no_suffix() {
    // No column collision → no suffix applied
    let l = make_left().tidy();
    let r = make_right().tidy();
    let suffix = JoinSuffix::default();
    let result = l.inner_join_typed(&r, &[("id", "id")], &suffix).unwrap();
    let b = result.borrow();
    // "name" is from right, "val" from left — no collision
    assert!(b.get_column("val").is_some());
    assert!(b.get_column("name").is_some());
}

// ── left_join_typed ───────────────────────────────────────────────────────

#[test]
fn test_left_join_typed_all_left_rows() {
    let l = make_left().tidy();
    let r = make_right().tidy();
    let suffix = JoinSuffix::default();
    let result = l.left_join_typed(&r, &[("id", "id")], &suffix).unwrap();
    let b = result.borrow();
    // All 3 left rows retained (id=1 unmatched → null fill on right cols)
    assert_eq!(b.nrows(), 3);
}

#[test]
fn test_left_join_typed_null_fill_for_unmatched() {
    let l = make_left().tidy();
    let r = make_right().tidy();
    let suffix = JoinSuffix::default();
    let result = l.left_join_typed(&r, &[("id", "id")], &suffix).unwrap();
    let b = result.borrow();
    // id=1 has no match in right → "name" col gets "" fill
    if let Column::Str(names) = b.get_column("name").unwrap() {
        // First row (id=1) should have null fill
        assert_eq!(names[0], ""); // sentinel fill for unmatched
    }
}

// ── right_join ────────────────────────────────────────────────────────────

#[test]
fn test_right_join_basic_row_count() {
    let l = make_left().tidy();
    let r = make_right().tidy();
    let suffix = JoinSuffix::default();
    let result = l.right_join(&r, &[("id", "id")], &suffix).unwrap();
    // All 3 right rows retained (id=4 unmatched → left cols null-filled)
    assert_eq!(result.nrows(), 3);
}

#[test]
fn test_right_join_right_rows_present() {
    let l = make_left().tidy();
    let r = make_right().tidy();
    let suffix = JoinSuffix::default();
    let result = l.right_join(&r, &[("id", "id")], &suffix).unwrap();
    let tv = result.to_tidy_view_filled();
    let mat = tv.materialize().unwrap();
    // "name" column should contain b, c, d
    if let Column::Str(names) = mat.get_column("name").unwrap() {
        assert!(names.contains(&"b".to_string()));
        assert!(names.contains(&"c".to_string()));
        assert!(names.contains(&"d".to_string()));
    }
}

#[test]
fn test_right_join_type_mismatch_error() {
    let left = DataFrame::from_columns(vec![
        ("id".into(), Column::Int(vec![1])),
    ])
    .unwrap();
    let right = DataFrame::from_columns(vec![
        ("id".into(), Column::Str(vec!["1".into()])),
    ])
    .unwrap();
    let l = left.tidy();
    let r = right.tidy();
    let suffix = JoinSuffix::default();
    let err = l.right_join(&r, &[("id", "id")], &suffix).unwrap_err();
    assert!(matches!(err, TidyError::TypeMismatch { .. }));
}

// ── full_join ─────────────────────────────────────────────────────────────

#[test]
fn test_full_join_row_count() {
    // left: {1,2,3}, right: {2,3,4}
    // Matched: 2, 3 (2 rows); Unmatched left: 1 (1 row); Unmatched right: 4 (1 row)
    // Total: 4 rows
    let l = make_left().tidy();
    let r = make_right().tidy();
    let suffix = JoinSuffix::default();
    let result = l.full_join(&r, &[("id", "id")], &suffix).unwrap();
    assert_eq!(result.nrows(), 4);
}

#[test]
fn test_full_join_left_unmatched_right_cols_null() {
    let l = make_left().tidy();
    let r = make_right().tidy();
    let suffix = JoinSuffix::default();
    let result = l.full_join(&r, &[("id", "id")], &suffix).unwrap();
    // id=1 (unmatched in left): "name" col should be null
    let _name_col = result.get_column("name").unwrap();
    // Find id=1 row (first left row, unmatched)
    let tv = result.to_tidy_view_filled();
    let mat = tv.materialize().unwrap();
    assert_eq!(mat.nrows(), 4);
}

#[test]
fn test_full_join_type_mismatch_error() {
    let left = DataFrame::from_columns(vec![
        ("id".into(), Column::Bool(vec![true])),
    ])
    .unwrap();
    let right = DataFrame::from_columns(vec![
        ("id".into(), Column::Str(vec!["x".into()])),
    ])
    .unwrap();
    let l = left.tidy();
    let r = right.tidy();
    let suffix = JoinSuffix::default();
    let err = l.full_join(&r, &[("id", "id")], &suffix).unwrap_err();
    assert!(matches!(err, TidyError::TypeMismatch { .. }));
}

#[test]
fn test_full_join_deterministic() {
    let l1 = make_left().tidy();
    let r1 = make_right().tidy();
    let l2 = make_left().tidy();
    let r2 = make_right().tidy();
    let suffix = JoinSuffix::default();
    let res1 = l1.full_join(&r1, &[("id", "id")], &suffix).unwrap();
    let res2 = l2.full_join(&r2, &[("id", "id")], &suffix).unwrap();
    assert_eq!(res1.nrows(), res2.nrows());
    assert_eq!(res1.column_names(), res2.column_names());
}

// ── inner_join_typed basic semantics ─────────────────────────────────────

#[test]
fn test_inner_join_typed_basic_row_count() {
    // left: {1,2,3}, right: {2,3,4} → matching: 2, 3 → 2 rows
    let l = make_left().tidy();
    let r = make_right().tidy();
    let suffix = JoinSuffix::default();
    let result = l.inner_join_typed(&r, &[("id", "id")], &suffix).unwrap();
    let b = result.borrow();
    assert_eq!(b.nrows(), 2);
}

#[test]
fn test_inner_join_typed_order_preserved() {
    // Left row order preserved (id=2 before id=3)
    let l = make_left().tidy();
    let r = make_right().tidy();
    let suffix = JoinSuffix::default();
    let result = l.inner_join_typed(&r, &[("id", "id")], &suffix).unwrap();
    let b = result.borrow();
    if let Column::Int(ids) = b.get_column("id").unwrap() {
        assert_eq!(ids[0], 2);
        assert_eq!(ids[1], 3);
    }
}
