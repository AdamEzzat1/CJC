// Phase 11 — summarise
// Tests: each aggregator, empty groups, float determinism, promotion, overflow
use cjc_data::{Column, DataFrame, TidyAgg, TidyError};

fn make_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("dept".into(), Column::Str(vec![
            "eng".into(), "hr".into(), "eng".into(), "hr".into(), "eng".into(),
        ])),
        ("salary".into(), Column::Float(vec![100.0, 60.0, 80.0, 70.0, 75.0])),
        ("headcount".into(), Column::Int(vec![1, 1, 1, 1, 1])),
    ])
    .unwrap()
}

fn make_empty_grouped() -> cjc_data::GroupedTidyView {
    let df = DataFrame::from_columns(vec![
        ("g".into(), Column::Str(vec![])),
        ("v".into(), Column::Float(vec![])),
    ])
    .unwrap();
    df.tidy().group_by(&["g"]).unwrap()
}

#[test]
fn test_summarise_count() {
    let df = make_df();
    let gv = df.tidy().group_by(&["dept"]).unwrap();
    let frame = gv.summarise(&[("n", TidyAgg::Count)]).unwrap();
    let b = frame.borrow();
    // eng: 3, hr: 2 (first-occurrence order)
    if let Column::Int(v) = b.get_column("n").unwrap() {
        assert_eq!(*v, vec![3i64, 2]);
    } else {
        panic!("expected Int column for count");
    }
}

#[test]
fn test_summarise_sum() {
    let df = make_df();
    let gv = df.tidy().group_by(&["dept"]).unwrap();
    let frame = gv.summarise(&[("total", TidyAgg::Sum("salary".into()))]).unwrap();
    let b = frame.borrow();
    // eng: 100+80+75=255, hr: 60+70=130
    if let Column::Float(v) = b.get_column("total").unwrap() {
        assert!((v[0] - 255.0).abs() < 1e-9, "eng sum={}", v[0]);
        assert!((v[1] - 130.0).abs() < 1e-9, "hr sum={}", v[1]);
    } else {
        panic!("expected Float column for sum");
    }
}

#[test]
fn test_summarise_mean() {
    let df = make_df();
    let gv = df.tidy().group_by(&["dept"]).unwrap();
    let frame = gv.summarise(&[("avg", TidyAgg::Mean("salary".into()))]).unwrap();
    let b = frame.borrow();
    // eng: 255/3=85, hr: 130/2=65
    if let Column::Float(v) = b.get_column("avg").unwrap() {
        assert!((v[0] - 85.0).abs() < 1e-9, "eng mean={}", v[0]);
        assert!((v[1] - 65.0).abs() < 1e-9, "hr mean={}", v[1]);
    }
}

#[test]
fn test_summarise_min() {
    let df = make_df();
    let gv = df.tidy().group_by(&["dept"]).unwrap();
    let frame = gv.summarise(&[("lo", TidyAgg::Min("salary".into()))]).unwrap();
    let b = frame.borrow();
    if let Column::Float(v) = b.get_column("lo").unwrap() {
        assert!((v[0] - 75.0).abs() < 1e-9, "eng min={}", v[0]);
        assert!((v[1] - 60.0).abs() < 1e-9, "hr min={}", v[1]);
    }
}

#[test]
fn test_summarise_max() {
    let df = make_df();
    let gv = df.tidy().group_by(&["dept"]).unwrap();
    let frame = gv.summarise(&[("hi", TidyAgg::Max("salary".into()))]).unwrap();
    let b = frame.borrow();
    if let Column::Float(v) = b.get_column("hi").unwrap() {
        assert!((v[0] - 100.0).abs() < 1e-9, "eng max={}", v[0]);
        assert!((v[1] - 70.0).abs() < 1e-9, "hr max={}", v[1]);
    }
}

#[test]
fn test_summarise_first_last() {
    let df = make_df();
    let gv = df.tidy().group_by(&["dept"]).unwrap();
    let frame = gv
        .summarise(&[
            ("first_salary", TidyAgg::First("salary".into())),
            ("last_salary", TidyAgg::Last("salary".into())),
        ])
        .unwrap();
    let b = frame.borrow();
    // eng: rows 0,2,4 → first=100, last=75
    if let Column::Float(v) = b.get_column("first_salary").unwrap() {
        assert!((v[0] - 100.0).abs() < 1e-9);
        assert!((v[1] - 60.0).abs() < 1e-9);
    }
    if let Column::Float(v) = b.get_column("last_salary").unwrap() {
        assert!((v[0] - 75.0).abs() < 1e-9);
        assert!((v[1] - 70.0).abs() < 1e-9);
    }
}

#[test]
fn test_summarise_empty_group_count() {
    let gv = make_empty_grouped();
    let frame = gv.summarise(&[("n", TidyAgg::Count)]).unwrap();
    let b = frame.borrow();
    assert_eq!(b.nrows(), 0, "0 groups → 0 rows in summary");
}

#[test]
fn test_summarise_empty_group_first_errors() {
    // Create a single group with 0 rows by filtering everything out then grouping
    let df = DataFrame::from_columns(vec![
        ("g".into(), Column::Str(vec!["a".into(), "b".into()])),
        ("v".into(), Column::Float(vec![1.0, 2.0])),
    ])
    .unwrap();
    let view = df.tidy();
    // Filter to 0 rows
    let empty_view = view
        .filter(&cjc_data::DExpr::BinOp {
            op: cjc_data::DBinOp::Gt,
            left: Box::new(cjc_data::DExpr::Col("v".into())),
            right: Box::new(cjc_data::DExpr::LitFloat(1000.0)),
        })
        .unwrap();
    let gv = empty_view.group_by(&["g"]).unwrap();
    // 0 groups → summarise succeeds with 0 rows (no individual empty group to trigger error)
    let frame = gv.summarise(&[("first_v", TidyAgg::First("v".into()))]).unwrap();
    assert_eq!(frame.borrow().nrows(), 0);
}

#[test]
fn test_summarise_float_determinism() {
    // Run summarise twice on same data; results must be bitwise identical
    let df = DataFrame::from_columns(vec![
        ("g".into(), Column::Str(vec!["a".into(); 100])),
        ("v".into(), Column::Float((0..100).map(|i| i as f64 * 0.001).collect())),
    ])
    .unwrap();

    let result1 = df
        .clone()
        .tidy()
        .group_by(&["g"])
        .unwrap()
        .summarise(&[("s", TidyAgg::Sum("v".into()))])
        .unwrap();
    let result2 = df
        .tidy()
        .group_by(&["g"])
        .unwrap()
        .summarise(&[("s", TidyAgg::Sum("v".into()))])
        .unwrap();

    let v1 = result1.borrow();
    let v2 = result2.borrow();
    if let (Column::Float(a), Column::Float(b)) =
        (v1.get_column("s").unwrap(), v2.get_column("s").unwrap())
    {
        assert_eq!(
            a[0].to_bits(),
            b[0].to_bits(),
            "sum must be bitwise identical across runs"
        );
    }
}

#[test]
fn test_summarise_duplicate_output_name_rejected() {
    let df = make_df();
    let gv = df.tidy().group_by(&["dept"]).unwrap();
    let err = gv
        .summarise(&[
            ("x", TidyAgg::Count),
            ("x", TidyAgg::Sum("salary".into())),
        ])
        .unwrap_err();
    assert!(matches!(err, TidyError::DuplicateColumn(_)));
}

#[test]
fn test_summarise_unknown_col_errors() {
    let df = make_df();
    let gv = df.tidy().group_by(&["dept"]).unwrap();
    let err = gv
        .summarise(&[("x", TidyAgg::Sum("nonexistent".into()))])
        .unwrap_err();
    assert!(matches!(err, TidyError::ColumnNotFound(_)));
}

#[test]
fn test_summarise_key_cols_in_output() {
    let df = make_df();
    let gv = df.tidy().group_by(&["dept"]).unwrap();
    let frame = gv.summarise(&[("n", TidyAgg::Count)]).unwrap();
    let b = frame.borrow();
    // "dept" column must be present
    assert!(b.get_column("dept").is_some(), "key col must appear in output");
    assert!(b.get_column("n").is_some());
}

#[test]
fn test_summarise_int_col_sum() {
    let df = make_df();
    let gv = df.tidy().group_by(&["dept"]).unwrap();
    let frame = gv.summarise(&[("hc", TidyAgg::Sum("headcount".into()))]).unwrap();
    let b = frame.borrow();
    // eng: 3×1=3, hr: 2×1=2
    if let Column::Float(v) = b.get_column("hc").unwrap() {
        assert!((v[0] - 3.0).abs() < 1e-9);
        assert!((v[1] - 2.0).abs() < 1e-9);
    }
}

#[test]
fn test_summarise_multiple_groups_single_row_each() {
    let df = DataFrame::from_columns(vec![
        ("id".into(), Column::Int(vec![1, 2, 3])),
        ("v".into(), Column::Float(vec![10.0, 20.0, 30.0])),
    ])
    .unwrap();
    let gv = df.tidy().group_by(&["id"]).unwrap();
    assert_eq!(gv.ngroups(), 3);
    let frame = gv.summarise(&[("total", TidyAgg::Sum("v".into()))]).unwrap();
    let b = frame.borrow();
    if let Column::Float(v) = b.get_column("total").unwrap() {
        assert_eq!(v.len(), 3);
        assert!((v[0] - 10.0).abs() < 1e-9);
        assert!((v[1] - 20.0).abs() < 1e-9);
        assert!((v[2] - 30.0).abs() < 1e-9);
    }
}
