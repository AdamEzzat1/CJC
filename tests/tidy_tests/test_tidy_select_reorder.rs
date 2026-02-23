// Phase 10 — test_tidy_select_reorder
// Reordering columns via select preserves correctness in materialize/to_tensor.
use cjc_data::{Column, DataFrame};

fn make_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("a".into(), Column::Int(vec![10, 20, 30])),
        ("b".into(), Column::Float(vec![1.1, 2.2, 3.3])),
        ("c".into(), Column::Str(vec!["x".into(), "y".into(), "z".into()])),
    ])
    .unwrap()
}

#[test]
fn test_tidy_select_reorder() {
    let df = make_df();
    let view = df.tidy();

    // Reorder: c, a (drop b)
    let selected = view.select(&["c", "a"]).unwrap();
    assert_eq!(selected.column_names(), vec!["c", "a"]);
    assert_eq!(selected.ncols(), 2);
    assert_eq!(selected.nrows(), 3);

    let mat = selected.materialize().unwrap();
    assert_eq!(mat.column_names(), vec!["c", "a"]);

    // Values must be in the reordered column order
    let c_col = mat.get_column("c").unwrap();
    let a_col = mat.get_column("a").unwrap();
    if let Column::Str(cv) = c_col {
        assert_eq!(*cv, vec!["x".to_string(), "y".to_string(), "z".to_string()]);
    } else {
        panic!("expected Str");
    }
    if let Column::Int(av) = a_col {
        assert_eq!(*av, vec![10i64, 20, 30]);
    } else {
        panic!("expected Int");
    }
}

#[test]
fn test_tidy_select_reorder_to_tensor() {
    let df = make_df();
    let view = df.tidy();

    // Select numeric cols in reverse order: b, a
    let selected = view.select(&["b", "a"]).unwrap();
    let tensor = selected.to_tensor(&["b", "a"]).unwrap();

    // Shape: [3 rows, 2 cols]
    assert_eq!(tensor.shape(), &[3, 2]);

    // Row 0: b=1.1, a=10
    // Row 1: b=2.2, a=20
    // Row 2: b=3.3, a=30
    // to_tensor reads in row-major order
}

#[test]
fn test_tidy_select_single_col() {
    let df = make_df();
    let selected = df.tidy().select(&["a"]).unwrap();
    assert_eq!(selected.ncols(), 1);
    assert_eq!(selected.column_names(), vec!["a"]);
    let mat = selected.materialize().unwrap();
    assert_eq!(mat.ncols(), 1);
}

#[test]
fn test_tidy_select_all_cols_original_order() {
    let df = make_df();
    let selected = df.tidy().select(&["a", "b", "c"]).unwrap();
    assert_eq!(selected.column_names(), vec!["a", "b", "c"]);
}

#[test]
fn test_tidy_select_reorder_after_filter() {
    let df = make_df();
    // Filter to rows where a > 10, then reorder cols
    let view = df
        .tidy()
        .filter(&cjc_data::DExpr::BinOp {
            op: cjc_data::DBinOp::Gt,
            left: Box::new(cjc_data::DExpr::Col("a".into())),
            right: Box::new(cjc_data::DExpr::LitInt(10)),
        })
        .unwrap()
        .select(&["c", "a"])
        .unwrap();

    assert_eq!(view.nrows(), 2); // rows 1,2 (a=20,30)
    assert_eq!(view.column_names(), vec!["c", "a"]);

    let mat = view.materialize().unwrap();
    if let Column::Str(cv) = mat.get_column("c").unwrap() {
        assert_eq!(*cv, vec!["y".to_string(), "z".to_string()]);
    }
}
