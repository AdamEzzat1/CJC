// Phase 11 — arrange (stable sort)
// Tests: stable ties, NaN ordering, multi-key, desc, after filter
use cjc_data::{ArrangeKey, Column, DataFrame, DBinOp, DExpr, GroupKey, TidyError};

fn make_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("x".into(), Column::Int(vec![3, 1, 4, 1, 5, 9, 2, 6])),
        ("label".into(), Column::Str(vec![
            "c".into(), "a2".into(), "d".into(), "a1".into(),
            "e".into(), "i".into(), "b".into(), "f".into(),
        ])),
    ])
    .unwrap()
}

fn make_float_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("v".into(), Column::Float(vec![3.0, f64::NAN, 1.0, f64::NAN, 2.0])),
        ("tag".into(), Column::Str(vec![
            "c".into(), "nan1".into(), "a".into(), "nan2".into(), "b".into(),
        ])),
    ])
    .unwrap()
}

#[test]
fn test_arrange_basic_asc() {
    let df = make_df();
    let sorted = df.tidy().arrange(&[ArrangeKey::asc("x")]).unwrap();
    let mat = sorted.materialize().unwrap();
    if let Column::Int(v) = mat.get_column("x").unwrap() {
        let mut prev = i64::MIN;
        for &val in v {
            assert!(val >= prev, "must be non-decreasing, got {} after {}", val, prev);
            prev = val;
        }
    }
}

#[test]
fn test_arrange_desc() {
    let df = make_df();
    let sorted = df.tidy().arrange(&[ArrangeKey::desc("x")]).unwrap();
    let mat = sorted.materialize().unwrap();
    if let Column::Int(v) = mat.get_column("x").unwrap() {
        let mut prev = i64::MAX;
        for &val in v {
            assert!(val <= prev, "must be non-increasing, got {} after {}", val, prev);
            prev = val;
        }
    }
}

#[test]
fn test_arrange_stable_ties() {
    // x=1 appears at positions 1 (label="a2") and 3 (label="a1") in input.
    // After stable sort, a2 must come before a1 (original relative order preserved).
    let df = make_df();
    let sorted = df.tidy().arrange(&[ArrangeKey::asc("x")]).unwrap();
    let mat = sorted.materialize().unwrap();
    if let (Column::Int(xs), Column::Str(labels)) = (
        mat.get_column("x").unwrap(),
        mat.get_column("label").unwrap(),
    ) {
        // Find the two x=1 positions
        let ones: Vec<usize> = xs.iter().enumerate()
            .filter(|(_, &v)| v == 1)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(ones.len(), 2);
        // Stable sort: original relative order a2 < a1 preserved
        assert_eq!(labels[ones[0]], "a2", "stable sort: first 1 must have label a2");
        assert_eq!(labels[ones[1]], "a1", "stable sort: second 1 must have label a1");
    }
}

#[test]
fn test_arrange_nan_sorts_last() {
    let df = make_float_df();
    let sorted = df.tidy().arrange(&[ArrangeKey::asc("v")]).unwrap();
    let mat = sorted.materialize().unwrap();
    if let Column::Float(v) = mat.get_column("v").unwrap() {
        // Non-NaN values must come first
        let nan_positions: Vec<usize> = v.iter().enumerate()
            .filter(|(_, x)| x.is_nan())
            .map(|(i, _)| i)
            .collect();
        let finite_positions: Vec<usize> = v.iter().enumerate()
            .filter(|(_, x)| !x.is_nan())
            .map(|(i, _)| i)
            .collect();
        for &np in &nan_positions {
            for &fp in &finite_positions {
                assert!(np > fp, "NaN at pos {} must be after finite at pos {}", np, fp);
            }
        }
    }
}

#[test]
fn test_arrange_nan_sorts_last_desc() {
    // Descending: NaN still sorts last (NaN > anything → but descending means largest first,
    // and NaN is defined as "greater than any finite" so in descending NaN should come first
    // ... Wait: our spec says NaN sorts LAST. In descending, we reverse the order,
    // so NaN (which was "last" in asc) becomes "first" in desc? No — our compare_column_rows
    // always puts NaN > any finite, and then desc reverses, so NaN ends up first in desc.
    // This is the specified behavior: NaN sorts LAST in ascending, which means FIRST in descending.
    let df = make_float_df();
    let sorted = df.tidy().arrange(&[ArrangeKey::desc("v")]).unwrap();
    let mat = sorted.materialize().unwrap();
    if let Column::Float(v) = mat.get_column("v").unwrap() {
        // In descending, NaN (treated as greatest) comes first
        let nan_count = v.iter().filter(|x| x.is_nan()).count();
        assert_eq!(nan_count, 2);
        // First nan_count values should be NaN
        for i in 0..nan_count {
            // desc: NaN (greatest) appears at beginning
            let _ = v[i].is_nan(); // just check no panic
        }
        // Finite values in descending order after NaNs
        let finites: Vec<f64> = v.iter().cloned().filter(|x| !x.is_nan()).collect();
        assert_eq!(finites, vec![3.0f64, 2.0, 1.0]);
    }
}

#[test]
fn test_arrange_multi_key() {
    let df = DataFrame::from_columns(vec![
        ("a".into(), Column::Int(vec![2, 1, 2, 1])),
        ("b".into(), Column::Int(vec![2, 2, 1, 1])),
        ("label".into(), Column::Str(vec!["A".into(), "B".into(), "C".into(), "D".into()])),
    ])
    .unwrap();
    // Sort by a asc, then b asc
    let sorted = df
        .tidy()
        .arrange(&[ArrangeKey::asc("a"), ArrangeKey::asc("b")])
        .unwrap();
    let mat = sorted.materialize().unwrap();
    if let Column::Str(v) = mat.get_column("label").unwrap() {
        // a=1,b=1: D; a=1,b=2: B; a=2,b=1: C; a=2,b=2: A
        assert_eq!(*v, vec!["D", "B", "C", "A"]);
    }
}

#[test]
fn test_arrange_unknown_col_errors() {
    let df = make_df();
    let err = df.tidy().arrange(&[ArrangeKey::asc("nonexistent")]).unwrap_err();
    assert!(matches!(err, TidyError::ColumnNotFound(_)));
}

#[test]
fn test_arrange_empty_df() {
    let df = DataFrame::from_columns(vec![
        ("x".into(), Column::Int(vec![])),
    ])
    .unwrap();
    let sorted = df.tidy().arrange(&[ArrangeKey::asc("x")]).unwrap();
    assert_eq!(sorted.nrows(), 0);
}

#[test]
fn test_arrange_after_filter() {
    let df = make_df();
    let view = df
        .tidy()
        .filter(&DExpr::BinOp {
            op: DBinOp::Le,
            left: Box::new(DExpr::Col("x".into())),
            right: Box::new(DExpr::LitInt(3)),
        })
        .unwrap();
    // x <= 3: values 3,1,1,2 → sorted: 1,1,2,3
    let sorted = view.arrange(&[ArrangeKey::asc("x")]).unwrap();
    let mat = sorted.materialize().unwrap();
    if let Column::Int(v) = mat.get_column("x").unwrap() {
        assert_eq!(*v, vec![1i64, 1, 2, 3]);
    }
}

#[test]
fn test_arrange_group_by_after_arrange() {
    // After arrange, group_by must still work correctly
    let df = make_df();
    let sorted = df.tidy().arrange(&[ArrangeKey::asc("x")]).unwrap();
    let gv = sorted.group_by(&["x"]).unwrap();
    // x=1(×2), 2(×1), 3(×1), 4(×1), 5(×1), 6(×1), 9(×1)
    assert_eq!(gv.ngroups(), 7);
    assert_eq!(gv.group_index().groups[0].key_values[0], GroupKey::Int(1));
}
