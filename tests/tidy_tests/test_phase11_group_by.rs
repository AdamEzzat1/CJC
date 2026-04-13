// Phase 11 — group_by, ungroup
// Tests: empty df, single key, multi key, ordering, after filter, after arrange, ungroup
use cjc_data::{Column, DataFrame, DBinOp, DExpr, GroupKey, TidyError};

fn make_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("dept".into(), Column::Str(vec!["eng".into(), "hr".into(), "eng".into(), "hr".into(), "eng".into()])),
        ("level".into(), Column::Str(vec!["senior".into(), "junior".into(), "junior".into(), "senior".into(), "junior".into()])),
        ("salary".into(), Column::Int(vec![100, 60, 80, 70, 75])),
    ])
    .unwrap()
}

#[test]
fn test_group_by_empty_df() {
    let df = DataFrame::from_columns(vec![
        ("x".into(), Column::Int(vec![])),
    ])
    .unwrap();
    let gv = df.tidy().group_by(&["x"]).unwrap();
    assert_eq!(gv.ngroups(), 0, "empty df → 0 groups");
}

#[test]
fn test_group_by_single_key() {
    let df = make_df();
    let gv = df.tidy().group_by(&["dept"]).unwrap();
    assert_eq!(gv.ngroups(), 2);
    // First-occurrence order: "eng" appears first
    assert_eq!(gv.group_index().groups[0].key_values[0], GroupKey::Str("eng".into()));
    assert_eq!(gv.group_index().groups[1].key_values[0], GroupKey::Str("hr".into()));
}

#[test]
fn test_group_by_multi_key() {
    let df = make_df();
    let gv = df.tidy().group_by(&["dept", "level"]).unwrap();
    // Unique (dept, level) combos in first-occurrence order:
    // ("eng","senior"), ("hr","junior"), ("eng","junior"), ("hr","senior")
    assert_eq!(gv.ngroups(), 4);
    let g0 = &gv.group_index().groups[0].key_values;
    assert_eq!(g0, &[GroupKey::Str("eng".into()), GroupKey::Str("senior".into())]);
    let g1 = &gv.group_index().groups[1].key_values;
    assert_eq!(g1, &[GroupKey::Str("hr".into()), GroupKey::Str("junior".into())]);
}

#[test]
fn test_group_by_ordering_is_first_occurrence() {
    // With a DataFrame where row order is reversed, group order must track first occurrence
    let df = DataFrame::from_columns(vec![
        ("cat".into(), Column::Str(vec!["z".into(), "a".into(), "z".into(), "a".into()])),
    ])
    .unwrap();
    let gv = df.tidy().group_by(&["cat"]).unwrap();
    // "z" appears first
    assert_eq!(gv.group_index().groups[0].key_values[0], GroupKey::Str("z".into()));
    assert_eq!(gv.group_index().groups[1].key_values[0], GroupKey::Str("a".into()));
}

#[test]
fn test_group_by_row_assignments() {
    let df = make_df();
    let gv = df.tidy().group_by(&["dept"]).unwrap();
    // eng rows: base indices 0,2,4; hr rows: 1,3
    let eng_rows = &gv.group_index().groups[0].row_indices;
    assert_eq!(*eng_rows, vec![0usize, 2, 4]);
    let hr_rows = &gv.group_index().groups[1].row_indices;
    assert_eq!(*hr_rows, vec![1usize, 3]);
}

#[test]
fn test_group_by_unknown_key_errors() {
    let df = make_df();
    let err = df.tidy().group_by(&["nonexistent"]).unwrap_err();
    assert!(matches!(err, TidyError::ColumnNotFound(_)));
}

#[test]
fn test_group_by_after_filter() {
    let df = make_df();
    // Filter: salary > 70 → rows 0(100), 2(80), 3(70 NO), 4(75 YES) → rows 0,2,4 → salaries 100,80,75
    // Actually >70 means 75>70 yes. So rows 0(100),2(80),4(75) → dept: eng,eng,eng → 1 group
    let view = df
        .tidy()
        .filter(&DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("salary".into())),
            right: Box::new(DExpr::LitInt(70)),
        })
        .unwrap();
    let gv = view.group_by(&["dept"]).unwrap();
    // Only eng rows pass the filter
    assert_eq!(gv.ngroups(), 1);
    assert_eq!(gv.group_index().groups[0].key_values[0], GroupKey::Str("eng".into()));
}

#[test]
fn test_ungroup_restores_view() {
    let df = make_df();
    let original_nrows = df.nrows();
    let view = df.tidy();
    let gv = view.group_by(&["dept"]).unwrap();
    let restored = gv.ungroup();
    // All rows still visible
    assert_eq!(restored.nrows(), original_nrows);
    assert_eq!(restored.ncols(), 3);
}

#[test]
fn test_group_by_int_key() {
    let df = DataFrame::from_columns(vec![
        ("bucket".into(), Column::Int(vec![1, 2, 1, 3, 2, 1])),
        ("val".into(), Column::Float(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
    ])
    .unwrap();
    let gv = df.tidy().group_by(&["bucket"]).unwrap();
    // First occurrence order: 1, 2, 3
    assert_eq!(gv.ngroups(), 3);
    assert_eq!(gv.group_index().groups[0].key_values[0], GroupKey::Int(1));
    assert_eq!(gv.group_index().groups[1].key_values[0], GroupKey::Int(2));
    assert_eq!(gv.group_index().groups[2].key_values[0], GroupKey::Int(3));
}

#[test]
fn test_group_by_bool_key() {
    let df = DataFrame::from_columns(vec![
        ("flag".into(), Column::Bool(vec![true, false, true, false, true])),
        ("x".into(), Column::Int(vec![1, 2, 3, 4, 5])),
    ])
    .unwrap();
    let gv = df.tidy().group_by(&["flag"]).unwrap();
    assert_eq!(gv.ngroups(), 2);
    // true appears first
    assert_eq!(gv.group_index().groups[0].key_values[0], GroupKey::Bool(true));
}
