// Phase 14: across() scoped transforms edge-case tests

use cjc_data::{AcrossSpec, AcrossTransform, Column, DataFrame, TidyError};

fn make_nums() -> DataFrame {
    DataFrame::from_columns(vec![
        ("a".into(), Column::Int(vec![1, 2, 3])),
        ("b".into(), Column::Int(vec![4, 5, 6])),
        ("c".into(), Column::Float(vec![0.1, 0.2, 0.3])),
    ])
    .unwrap()
}

fn double_transform() -> AcrossTransform {
    AcrossTransform::new("double", |_name, col| match col {
        Column::Int(v) => Ok(Column::Int(v.iter().map(|x| x * 2).collect())),
        Column::Float(v) => Ok(Column::Float(v.iter().map(|x| x * 2.0).collect())),
        _ => Err(TidyError::TypeMismatch {
            expected: "Int or Float".into(),
            got: col.type_name().into(),
        }),
    })
}

fn sum_transform() -> AcrossTransform {
    AcrossTransform::new("sum", |_name, col| match col {
        Column::Int(v) => Ok(Column::Float(vec![v.iter().map(|&x| x as f64).sum::<f64>()])),
        Column::Float(v) => Ok(Column::Float(vec![v.iter().sum::<f64>()])),
        _ => Err(TidyError::TypeMismatch {
            expected: "numeric".into(),
            got: col.type_name().into(),
        }),
    })
}

// ── mutate_across basic ────────────────────────────────────────────────────

#[test]
fn test_mutate_across_basic_generates_cols() {
    let df = make_nums();
    let v = df.tidy();
    let spec = AcrossSpec::new(vec!["a", "b"], double_transform());
    let result = v.mutate_across(&[spec]).unwrap();
    let b = result.borrow();
    // Should generate a_double and b_double
    assert!(b.get_column("a_double").is_some());
    assert!(b.get_column("b_double").is_some());
}

#[test]
fn test_mutate_across_values_correct() {
    let df = make_nums();
    let v = df.tidy();
    let spec = AcrossSpec::new(vec!["a"], double_transform());
    let result = v.mutate_across(&[spec]).unwrap();
    let b = result.borrow();
    if let Column::Int(vals) = b.get_column("a_double").unwrap() {
        assert_eq!(vals, &[2, 4, 6]);
    } else {
        panic!("expected Int a_double");
    }
}

#[test]
fn test_mutate_across_preserves_original_cols() {
    let df = make_nums();
    let v = df.tidy();
    let spec = AcrossSpec::new(vec!["a"], double_transform());
    let result = v.mutate_across(&[spec]).unwrap();
    let b = result.borrow();
    // Original 'a' column should still be present
    assert!(b.get_column("a").is_some());
}

#[test]
fn test_mutate_across_custom_name_template() {
    let df = make_nums();
    let v = df.tidy();
    let spec = AcrossSpec::new(vec!["a", "b"], double_transform())
        .with_template("x_{col}");
    let result = v.mutate_across(&[spec]).unwrap();
    let b = result.borrow();
    assert!(b.get_column("x_a").is_some());
    assert!(b.get_column("x_b").is_some());
}

#[test]
fn test_mutate_across_empty_cols_noop() {
    let df = make_nums();
    let v = df.tidy();
    let spec = AcrossSpec::new(Vec::<&str>::new(), double_transform());
    let result = v.mutate_across(&[spec]).unwrap();
    let b = result.borrow();
    // Same columns as original (a, b, c)
    assert_eq!(b.ncols(), 3);
}

#[test]
fn test_mutate_across_unknown_col_error() {
    let df = make_nums();
    let v = df.tidy();
    let spec = AcrossSpec::new(vec!["nonexistent"], double_transform());
    let err = v.mutate_across(&[spec]).unwrap_err();
    assert!(matches!(err, TidyError::ColumnNotFound(_)));
}

#[test]
fn test_mutate_across_expansion_order_stable() {
    // a_double should come before b_double in output
    let df = make_nums();
    let v = df.tidy();
    let spec = AcrossSpec::new(vec!["a", "b"], double_transform());
    let result = v.mutate_across(&[spec]).unwrap();
    let b = result.borrow();
    let names = b.column_names();
    let pos_a = names.iter().position(|&n| n == "a_double").unwrap();
    let pos_b = names.iter().position(|&n| n == "b_double").unwrap();
    assert!(pos_a < pos_b);
}

// ── summarise_across ───────────────────────────────────────────────────────

#[test]
fn test_summarise_across_basic() {
    let df = DataFrame::from_columns(vec![
        ("grp".into(), Column::Str(vec!["A".into(), "A".into(), "B".into(), "B".into()])),
        ("val".into(), Column::Int(vec![1, 2, 3, 4])),
    ])
    .unwrap();
    let v = df.tidy();
    let grouped = v.group_by(&["grp"]).unwrap();
    let spec = AcrossSpec::new(vec!["val"], sum_transform());
    let result = grouped.summarise_across(&[spec]).unwrap();
    let b = result.borrow();
    // 2 groups × 1 agg col = 2 rows, cols: grp, val_sum
    assert_eq!(b.nrows(), 2);
    assert!(b.get_column("val_sum").is_some());
}

#[test]
fn test_summarise_across_values_correct() {
    let df = DataFrame::from_columns(vec![
        ("grp".into(), Column::Str(vec!["A".into(), "A".into(), "B".into()])),
        ("x".into(), Column::Int(vec![10, 20, 30])),
    ])
    .unwrap();
    let v = df.tidy();
    let grouped = v.group_by(&["grp"]).unwrap();
    let spec = AcrossSpec::new(vec!["x"], sum_transform());
    let result = grouped.summarise_across(&[spec]).unwrap();
    let b = result.borrow();
    if let Column::Float(sums) = b.get_column("x_sum").unwrap() {
        // Group A: 10+20=30, Group B: 30
        assert_eq!(sums[0], 30.0);
        assert_eq!(sums[1], 30.0);
    } else {
        panic!("expected Float x_sum");
    }
}

#[test]
fn test_summarise_across_duplicate_output_error() {
    let df = make_nums();
    let v = df.tidy();
    let grouped = v.group_by(&[]).unwrap();
    // Two specs producing the same output name
    let spec1 = AcrossSpec::new(vec!["a"], sum_transform());
    let spec2 = AcrossSpec::new(vec!["a"], sum_transform());
    let err = grouped.summarise_across(&[spec1, spec2]).unwrap_err();
    assert!(matches!(err, TidyError::DuplicateColumn(_)));
}

#[test]
fn test_summarise_across_unknown_col_error() {
    let df = make_nums();
    let v = df.tidy();
    let grouped = v.group_by(&[]).unwrap();
    let spec = AcrossSpec::new(vec!["nonexistent"], sum_transform());
    let err = grouped.summarise_across(&[spec]).unwrap_err();
    assert!(matches!(err, TidyError::ColumnNotFound(_)));
}

#[test]
fn test_across_deterministic_two_runs() {
    let df = make_nums();
    let v1 = df.clone().tidy();
    let v2 = df.tidy();
    let spec1 = AcrossSpec::new(vec!["a", "b"], double_transform());
    let spec2 = AcrossSpec::new(vec!["a", "b"], double_transform());
    let r1 = v1.mutate_across(&[spec1]).unwrap();
    let r2 = v2.mutate_across(&[spec2]).unwrap();
    let b1 = r1.borrow();
    let b2 = r2.borrow();
    assert_eq!(
        b1.get_column("a_double").unwrap().get_display(0),
        b2.get_column("a_double").unwrap().get_display(0),
    );
}

#[test]
fn test_mutate_across_ungrouped_vs_grouped_same_result() {
    // mutate_across on grouped view should give same row-wise result as ungrouped
    let df = make_nums();
    let v = df.clone().tidy();
    let grouped = df.tidy().group_by(&[]).unwrap();
    let spec_v = AcrossSpec::new(vec!["a"], double_transform());
    let spec_g = AcrossSpec::new(vec!["a"], double_transform());
    let r_v = v.mutate_across(&[spec_v]).unwrap();
    let r_g = grouped.mutate_across(&[spec_g]).unwrap();
    let b_v = r_v.borrow();
    let b_g = r_g.borrow();
    if let (Column::Int(v1), Column::Int(v2)) = (
        b_v.get_column("a_double").unwrap(),
        b_g.get_column("a_double").unwrap(),
    ) {
        assert_eq!(v1, v2);
    }
}
