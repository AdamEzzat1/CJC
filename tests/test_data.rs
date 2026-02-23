// CJC Test Suite — cjc-data (10 tests)
// Source: crates/cjc-data/src/lib.rs
// These tests are extracted from the inline #[cfg(test)] modules for regression tracking.

use cjc_data::*;

fn sample_df() -> DataFrame {
    DataFrame::from_columns(vec![
        (
            "name".into(),
            Column::Str(vec![
                "Alice".into(),
                "Bob".into(),
                "Carol".into(),
                "Dave".into(),
                "Eve".into(),
                "Frank".into(),
            ]),
        ),
        (
            "dept".into(),
            Column::Str(vec![
                "eng".into(),
                "eng".into(),
                "sales".into(),
                "eng".into(),
                "sales".into(),
                "eng".into(),
            ]),
        ),
        (
            "salary".into(),
            Column::Float(vec![95000.0, 102000.0, 78000.0, 110000.0, 82000.0, 98000.0]),
        ),
        (
            "tenure".into(),
            Column::Int(vec![3, 7, 2, 10, 1, 5]),
        ),
    ])
    .unwrap()
}

#[test]
fn test_dataframe_creation() {
    let df = sample_df();
    assert_eq!(df.nrows(), 6);
    assert_eq!(df.ncols(), 4);
    assert_eq!(
        df.column_names(),
        vec!["name", "dept", "salary", "tenure"]
    );
}

#[test]
fn test_filter() {
    let df = sample_df();

    // Filter tenure > 2
    let result = Pipeline::scan(df)
        .filter(DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("tenure".into())),
            right: Box::new(DExpr::LitInt(2)),
        })
        .collect()
        .unwrap();

    assert_eq!(result.nrows(), 4); // Alice(3), Bob(7), Dave(10), Frank(5)
}

#[test]
fn test_group_by_summarize() {
    let df = sample_df();

    let result = Pipeline::scan(df)
        .summarize(
            vec!["dept".into()],
            vec![
                (
                    "avg_salary".into(),
                    DExpr::Agg(AggFunc::Mean, Box::new(DExpr::Col("salary".into()))),
                ),
                ("headcount".into(), DExpr::Count),
            ],
        )
        .collect()
        .unwrap();

    assert_eq!(result.nrows(), 2); // eng, sales

    // Find eng row
    let dept_col = result.get_column("dept").unwrap();
    let avg_col = result.get_column("avg_salary").unwrap();
    let count_col = result.get_column("headcount").unwrap();

    if let (Column::Str(depts), Column::Float(avgs), Column::Float(counts)) =
        (dept_col, avg_col, count_col)
    {
        let eng_idx = depts.iter().position(|d| d == "eng").unwrap();
        let sales_idx = depts.iter().position(|d| d == "sales").unwrap();

        // eng: (95000 + 102000 + 110000 + 98000) / 4 = 101250
        assert!((avgs[eng_idx] - 101250.0).abs() < 0.01);
        assert!((counts[eng_idx] - 4.0).abs() < 0.01);

        // sales: (78000 + 82000) / 2 = 80000
        assert!((avgs[sales_idx] - 80000.0).abs() < 0.01);
        assert!((counts[sales_idx] - 2.0).abs() < 0.01);
    } else {
        panic!("unexpected column types");
    }
}

#[test]
fn test_filter_then_aggregate() {
    let df = sample_df();

    // Filter tenure > 2, then aggregate by dept
    let result = Pipeline::scan(df)
        .filter(DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("tenure".into())),
            right: Box::new(DExpr::LitInt(2)),
        })
        .summarize(
            vec!["dept".into()],
            vec![
                (
                    "avg_salary".into(),
                    DExpr::Agg(AggFunc::Mean, Box::new(DExpr::Col("salary".into()))),
                ),
                (
                    "max_tenure".into(),
                    DExpr::Agg(AggFunc::Max, Box::new(DExpr::Col("tenure".into()))),
                ),
                ("headcount".into(), DExpr::Count),
            ],
        )
        .collect()
        .unwrap();

    // After filter: Alice(3,eng), Bob(7,eng), Dave(10,eng), Frank(5,eng)
    // Only eng remains
    assert_eq!(result.nrows(), 1);

    if let Column::Float(avgs) = result.get_column("avg_salary").unwrap() {
        // (95000 + 102000 + 110000 + 98000) / 4 = 101250
        assert!((avgs[0] - 101250.0).abs() < 0.01);
    }
    if let Column::Float(maxes) = result.get_column("max_tenure").unwrap() {
        assert!((maxes[0] - 10.0).abs() < 0.01);
    }
}

#[test]
fn test_to_tensor_data() {
    let df = DataFrame::from_columns(vec![
        ("x".into(), Column::Float(vec![1.0, 2.0, 3.0])),
        ("y".into(), Column::Float(vec![4.0, 5.0, 6.0])),
    ])
    .unwrap();

    let (data, shape) = df.to_tensor_data(&["x", "y"]).unwrap();
    assert_eq!(shape, vec![3, 2]);
    assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_display() {
    let df = DataFrame::from_columns(vec![
        ("x".into(), Column::Int(vec![1, 2, 3])),
        ("y".into(), Column::Float(vec![4.5, 5.5, 6.5])),
    ])
    .unwrap();

    let output = format!("{}", df);
    assert!(output.contains("x"));
    assert!(output.contains("y"));
    assert!(output.contains("4.5"));
}

#[test]
fn test_column_not_found() {
    let df = sample_df();
    let result = Pipeline::scan(df)
        .filter(DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("nonexistent".into())),
            right: Box::new(DExpr::LitInt(0)),
        })
        .collect();

    assert!(result.is_err());
}

#[test]
fn test_aggregation_functions() {
    let df = DataFrame::from_columns(vec![
        ("group".into(), Column::Str(vec!["a".into(), "a".into(), "a".into()])),
        ("val".into(), Column::Float(vec![10.0, 20.0, 30.0])),
    ])
    .unwrap();

    let result = Pipeline::scan(df)
        .summarize(
            vec!["group".into()],
            vec![
                ("total".into(), DExpr::Agg(AggFunc::Sum, Box::new(DExpr::Col("val".into())))),
                ("avg".into(), DExpr::Agg(AggFunc::Mean, Box::new(DExpr::Col("val".into())))),
                ("lo".into(), DExpr::Agg(AggFunc::Min, Box::new(DExpr::Col("val".into())))),
                ("hi".into(), DExpr::Agg(AggFunc::Max, Box::new(DExpr::Col("val".into())))),
                ("n".into(), DExpr::Count),
            ],
        )
        .collect()
        .unwrap();

    if let Column::Float(totals) = result.get_column("total").unwrap() {
        assert!((totals[0] - 60.0).abs() < 0.01);
    }
    if let Column::Float(avgs) = result.get_column("avg").unwrap() {
        assert!((avgs[0] - 20.0).abs() < 0.01);
    }
    if let Column::Float(mins) = result.get_column("lo").unwrap() {
        assert!((mins[0] - 10.0).abs() < 0.01);
    }
    if let Column::Float(maxs) = result.get_column("hi").unwrap() {
        assert!((maxs[0] - 30.0).abs() < 0.01);
    }
    if let Column::Float(counts) = result.get_column("n").unwrap() {
        assert!((counts[0] - 3.0).abs() < 0.01);
    }
}

#[test]
fn test_empty_dataframe() {
    let df = DataFrame::new();
    assert_eq!(df.nrows(), 0);
    assert_eq!(df.ncols(), 0);
}

#[test]
fn test_expr_display() {
    let expr = DExpr::BinOp {
        op: DBinOp::Gt,
        left: Box::new(DExpr::Col("age".into())),
        right: Box::new(DExpr::LitInt(18)),
    };
    assert_eq!(format!("{}", expr), "(col(\"age\") > 18)");
}
