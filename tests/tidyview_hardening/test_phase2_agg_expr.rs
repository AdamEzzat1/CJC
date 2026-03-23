//! Phase 2 TidyView hardening tests: new TidyAgg variants and DExpr extensions.

use cjc_data::{Column, DExpr, DataFrame, TidyAgg};

/// Build a small grouped test DataFrame:
///   group: ["a","a","a","b","b","b"]
///   val:   [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]
fn make_grouped() -> cjc_data::GroupedTidyView {
    let df = DataFrame::from_columns(vec![
        ("group".into(), Column::Str(vec!["a".into(), "a".into(), "a".into(), "b".into(), "b".into(), "b".into()])),
        ("val".into(), Column::Float(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0])),
    ])
    .unwrap();
    let view = df.tidy();
    view.group_by(&["group"]).unwrap()
}

#[test]
fn summarise_median_per_group() {
    let grouped = make_grouped();
    let result = grouped
        .summarise(&[("med", TidyAgg::Median("val".into()))])
        .unwrap();
    let df = result.borrow();
    let col = df.get_column("med").unwrap();
    if let Column::Float(vals) = col {
        // group a: median of [1,2,3] = 2.0
        // group b: median of [10,20,30] = 20.0
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 2.0).abs() < 1e-12);
        assert!((vals[1] - 20.0).abs() < 1e-12);
    } else {
        panic!("expected Float column for median");
    }
}

#[test]
fn summarise_sd_per_group() {
    let grouped = make_grouped();
    let result = grouped
        .summarise(&[("sd_val", TidyAgg::Sd("val".into()))])
        .unwrap();
    let df = result.borrow();
    let col = df.get_column("sd_val").unwrap();
    if let Column::Float(vals) = col {
        // group a: sd of [1,2,3] = 1.0 (sample sd)
        // group b: sd of [10,20,30] = 10.0
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 1.0).abs() < 1e-12);
        assert!((vals[1] - 10.0).abs() < 1e-12);
    } else {
        panic!("expected Float column for sd");
    }
}

#[test]
fn summarise_var_per_group() {
    let grouped = make_grouped();
    let result = grouped
        .summarise(&[("var_val", TidyAgg::Var("val".into()))])
        .unwrap();
    let df = result.borrow();
    let col = df.get_column("var_val").unwrap();
    if let Column::Float(vals) = col {
        // group a: var of [1,2,3] = 1.0 (sample variance)
        // group b: var of [10,20,30] = 100.0
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 1.0).abs() < 1e-12);
        assert!((vals[1] - 100.0).abs() < 1e-12);
    } else {
        panic!("expected Float column for var");
    }
}

#[test]
fn summarise_n_distinct_per_group() {
    let df = DataFrame::from_columns(vec![
        ("group".into(), Column::Str(vec!["a".into(), "a".into(), "a".into(), "b".into(), "b".into()])),
        ("val".into(), Column::Float(vec![1.0, 1.0, 2.0, 5.0, 5.0])),
    ])
    .unwrap();
    let view = df.tidy();
    let grouped = view.group_by(&["group"]).unwrap();
    let result = grouped
        .summarise(&[("nd", TidyAgg::NDistinct("val".into()))])
        .unwrap();
    let mat = result.borrow();
    let col = mat.get_column("nd").unwrap();
    if let Column::Float(vals) = col {
        // group a: 2 distinct (1.0, 2.0)
        // group b: 1 distinct (5.0)
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 2.0).abs() < 1e-12);
        assert!((vals[1] - 1.0).abs() < 1e-12);
    } else {
        panic!("expected Float column for n_distinct");
    }
}

#[test]
fn summarise_quantile_per_group() {
    let grouped = make_grouped();
    let result = grouped
        .summarise(&[("q50", TidyAgg::Quantile("val".into(), 0.5))])
        .unwrap();
    let df = result.borrow();
    let col = df.get_column("q50").unwrap();
    if let Column::Float(vals) = col {
        // quantile 0.5 of [1,2,3] = 2.0
        // quantile 0.5 of [10,20,30] = 20.0
        assert!((vals[0] - 2.0).abs() < 1e-12);
        assert!((vals[1] - 20.0).abs() < 1e-12);
    } else {
        panic!("expected Float column for quantile");
    }
}

#[test]
fn summarise_iqr_per_group() {
    let grouped = make_grouped();
    let result = grouped
        .summarise(&[("iqr_val", TidyAgg::Iqr("val".into()))])
        .unwrap();
    let df = result.borrow();
    let col = df.get_column("iqr_val").unwrap();
    if let Column::Float(vals) = col {
        // IQR of [1,2,3]: Q3=2.5, Q1=1.5, IQR=1.0
        // IQR of [10,20,30]: Q3=25.0, Q1=15.0, IQR=10.0
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 1.0).abs() < 1e-12);
        assert!((vals[1] - 10.0).abs() < 1e-12);
    } else {
        panic!("expected Float column for iqr");
    }
}

#[test]
fn dexpr_fncall_log_in_mutate() {
    let df = DataFrame::from_columns(vec![
        ("x".into(), Column::Float(vec![1.0, std::f64::consts::E, std::f64::consts::E * std::f64::consts::E])),
    ])
    .unwrap();
    let view = df.tidy();
    let result = view
        .mutate(&[("log_x", DExpr::FnCall("log".into(), vec![DExpr::Col("x".into())]))])
        .unwrap();
    let mat = result.borrow();
    let col = mat.get_column("log_x").unwrap();
    if let Column::Float(vals) = col {
        assert!((vals[0] - 0.0).abs() < 1e-12); // ln(1) = 0
        assert!((vals[1] - 1.0).abs() < 1e-12); // ln(e) = 1
        assert!((vals[2] - 2.0).abs() < 1e-12); // ln(e^2) = 2
    } else {
        panic!("expected Float column");
    }
}

#[test]
fn dexpr_cumsum_in_mutate() {
    let df = DataFrame::from_columns(vec![
        ("x".into(), Column::Float(vec![1.0, 2.0, 3.0, 4.0])),
    ])
    .unwrap();
    let view = df.tidy();
    let result = view
        .mutate(&[("cs", DExpr::CumSum(Box::new(DExpr::Col("x".into()))))])
        .unwrap();
    let mat = result.borrow();
    let col = mat.get_column("cs").unwrap();
    if let Column::Float(vals) = col {
        assert!((vals[0] - 1.0).abs() < 1e-12);
        assert!((vals[1] - 3.0).abs() < 1e-12);
        assert!((vals[2] - 6.0).abs() < 1e-12);
        assert!((vals[3] - 10.0).abs() < 1e-12);
    } else {
        panic!("expected Float column");
    }
}

#[test]
fn dexpr_lag_in_mutate() {
    let df = DataFrame::from_columns(vec![
        ("x".into(), Column::Float(vec![10.0, 20.0, 30.0, 40.0])),
    ])
    .unwrap();
    let view = df.tidy();
    let result = view
        .mutate(&[("lagged", DExpr::Lag(Box::new(DExpr::Col("x".into())), 1))])
        .unwrap();
    let mat = result.borrow();
    let col = mat.get_column("lagged").unwrap();
    if let Column::Float(vals) = col {
        assert!(vals[0].is_nan()); // no previous row
        assert!((vals[1] - 10.0).abs() < 1e-12);
        assert!((vals[2] - 20.0).abs() < 1e-12);
        assert!((vals[3] - 30.0).abs() < 1e-12);
    } else {
        panic!("expected Float column");
    }
}

#[test]
fn dexpr_row_number_in_mutate() {
    let df = DataFrame::from_columns(vec![
        ("x".into(), Column::Float(vec![100.0, 200.0, 300.0])),
    ])
    .unwrap();
    let view = df.tidy();
    let result = view
        .mutate(&[("rn", DExpr::RowNumber)])
        .unwrap();
    let mat = result.borrow();
    let col = mat.get_column("rn").unwrap();
    if let Column::Int(vals) = col {
        assert_eq!(vals, &[1, 2, 3]);
    } else {
        panic!("expected Int column for RowNumber");
    }
}

#[test]
fn dexpr_rank_in_mutate() {
    let df = DataFrame::from_columns(vec![
        ("x".into(), Column::Float(vec![30.0, 10.0, 20.0, 10.0])),
    ])
    .unwrap();
    let view = df.tidy();
    let result = view
        .mutate(&[("r", DExpr::Rank(Box::new(DExpr::Col("x".into()))))])
        .unwrap();
    let mat = result.borrow();
    let col = mat.get_column("r").unwrap();
    if let Column::Float(vals) = col {
        // sorted: 10,10,20,30 -> ranks: 1.5,1.5,3,4
        // original order: 30->4, 10->1.5, 20->3, 10->1.5
        assert!((vals[0] - 4.0).abs() < 1e-12);
        assert!((vals[1] - 1.5).abs() < 1e-12);
        assert!((vals[2] - 3.0).abs() < 1e-12);
        assert!((vals[3] - 1.5).abs() < 1e-12);
    } else {
        panic!("expected Float column for rank");
    }
}

#[test]
fn dexpr_dense_rank_in_mutate() {
    let df = DataFrame::from_columns(vec![
        ("x".into(), Column::Float(vec![30.0, 10.0, 20.0, 10.0])),
    ])
    .unwrap();
    let view = df.tidy();
    let result = view
        .mutate(&[("dr", DExpr::DenseRank(Box::new(DExpr::Col("x".into()))))])
        .unwrap();
    let mat = result.borrow();
    let col = mat.get_column("dr").unwrap();
    if let Column::Int(vals) = col {
        // sorted: 10,10,20,30 -> dense ranks: 1,1,2,3
        // original order: 30->3, 10->1, 20->2, 10->1
        assert_eq!(vals[0], 3);
        assert_eq!(vals[1], 1);
        assert_eq!(vals[2], 2);
        assert_eq!(vals[3], 1);
    } else {
        panic!("expected Int column for dense_rank");
    }
}

#[test]
fn determinism_three_runs_identical() {
    // Run all aggregations 3 times and verify identical results
    for _ in 0..3 {
        let grouped = make_grouped();
        let result = grouped
            .summarise(&[
                ("med", TidyAgg::Median("val".into())),
                ("sd_val", TidyAgg::Sd("val".into())),
                ("var_val", TidyAgg::Var("val".into())),
                ("nd", TidyAgg::NDistinct("val".into())),
                ("iqr_val", TidyAgg::Iqr("val".into())),
                ("q25", TidyAgg::Quantile("val".into(), 0.25)),
            ])
            .unwrap();
        let df = result.borrow();

        // Check median
        if let Column::Float(vals) = df.get_column("med").unwrap() {
            assert_eq!(vals[0].to_bits(), (2.0_f64).to_bits());
            assert_eq!(vals[1].to_bits(), (20.0_f64).to_bits());
        }
        // Check sd
        if let Column::Float(vals) = df.get_column("sd_val").unwrap() {
            assert_eq!(vals[0].to_bits(), (1.0_f64).to_bits());
            assert_eq!(vals[1].to_bits(), (10.0_f64).to_bits());
        }
        // Check var
        if let Column::Float(vals) = df.get_column("var_val").unwrap() {
            assert_eq!(vals[0].to_bits(), (1.0_f64).to_bits());
            assert_eq!(vals[1].to_bits(), (100.0_f64).to_bits());
        }
    }
}

#[test]
fn dexpr_fncall_sqrt_abs() {
    let df = DataFrame::from_columns(vec![
        ("x".into(), Column::Float(vec![4.0, 9.0, -16.0])),
    ])
    .unwrap();
    let view = df.tidy();
    let result = view
        .mutate(&[
            ("sqrt_x", DExpr::FnCall("sqrt".into(), vec![DExpr::Col("x".into())])),
            ("abs_x", DExpr::FnCall("abs".into(), vec![DExpr::Col("x".into())])),
        ])
        .unwrap();
    let mat = result.borrow();

    if let Column::Float(vals) = mat.get_column("sqrt_x").unwrap() {
        assert!((vals[0] - 2.0).abs() < 1e-12);
        assert!((vals[1] - 3.0).abs() < 1e-12);
        assert!(vals[2].is_nan()); // sqrt(-16) = NaN
    }
    if let Column::Float(vals) = mat.get_column("abs_x").unwrap() {
        assert!((vals[0] - 4.0).abs() < 1e-12);
        assert!((vals[1] - 9.0).abs() < 1e-12);
        assert!((vals[2] - 16.0).abs() < 1e-12);
    }
}
