// Milestone 2.5 — Data Join Tests
//
// Tests for the cjc_data Pipeline join operations:
// - inner_join: matching rows only
// - left_join: all left rows, NaN fill for non-matching right
// - cross_join: cartesian product
// - Join key column deduplication
// - Empty join result

use cjc_data::{Column, DataFrame, Pipeline, DExpr, DBinOp};

fn employees_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("id".into(), Column::Int(vec![1, 2, 3, 4])),
        (
            "name".into(),
            Column::Str(vec![
                "Alice".into(),
                "Bob".into(),
                "Carol".into(),
                "Dave".into(),
            ]),
        ),
        ("dept_id".into(), Column::Int(vec![10, 20, 10, 30])),
    ])
    .unwrap()
}

fn departments_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("dept_id".into(), Column::Int(vec![10, 20, 40])),
        (
            "dept_name".into(),
            Column::Str(vec!["Engineering".into(), "Sales".into(), "Marketing".into()]),
        ),
    ])
    .unwrap()
}

#[test]
fn join_inner_basic() {
    let emp = employees_df();
    let dept = departments_df();

    let result = Pipeline::scan(emp)
        .inner_join(dept, "dept_id", "dept_id")
        .collect()
        .unwrap();

    // Only employees with matching dept_id: Alice(10), Bob(20), Carol(10)
    // Dave(30) has no match in departments
    assert_eq!(result.nrows(), 3);

    // Check that dept_name column exists
    assert!(result.get_column("dept_name").is_some());

    // Check that the join key column from the right is not duplicated
    // (the left "dept_id" is kept, the right "dept_id" is dropped)
    let col_names = result.column_names();
    let dept_id_count = col_names.iter().filter(|&&n| n == "dept_id").count();
    assert_eq!(dept_id_count, 1, "join key should not be duplicated");
}

#[test]
fn join_left_preserves_all_left_rows() {
    let emp = employees_df();
    let dept = departments_df();

    let result = Pipeline::scan(emp)
        .left_join(dept, "dept_id", "dept_id")
        .collect()
        .unwrap();

    // All 4 employees should appear
    assert_eq!(result.nrows(), 4);

    // Dave (dept_id=30) has no match, so dept_name should be empty string
    if let Column::Str(names) = result.get_column("dept_name").unwrap() {
        // Find Dave's row -- he's index 3
        // After left join, the order follows left table order
        // Dave's dept_name should be "" (null fill for strings)
        let dave_dept = &names[3];
        assert_eq!(dave_dept, "", "non-matching left join row should have empty string");
    } else {
        panic!("expected Str column for dept_name");
    }
}

#[test]
fn join_cross_cartesian_product() {
    let left = DataFrame::from_columns(vec![
        ("a".into(), Column::Int(vec![1, 2])),
    ])
    .unwrap();

    let right = DataFrame::from_columns(vec![
        ("b".into(), Column::Int(vec![10, 20, 30])),
    ])
    .unwrap();

    let result = Pipeline::scan(left)
        .cross_join(right)
        .collect()
        .unwrap();

    // 2 * 3 = 6 rows
    assert_eq!(result.nrows(), 6);
    assert_eq!(result.ncols(), 2);
}

#[test]
fn join_inner_no_matches() {
    let left = DataFrame::from_columns(vec![
        ("key".into(), Column::Int(vec![1, 2, 3])),
    ])
    .unwrap();

    let right = DataFrame::from_columns(vec![
        ("key".into(), Column::Int(vec![4, 5, 6])),
        ("val".into(), Column::Float(vec![1.0, 2.0, 3.0])),
    ])
    .unwrap();

    let result = Pipeline::scan(left)
        .inner_join(right, "key", "key")
        .collect()
        .unwrap();

    assert_eq!(result.nrows(), 0, "no matching keys should give empty result");
}

#[test]
fn join_inner_then_filter() {
    let emp = employees_df();
    let dept = departments_df();

    // Join then filter to only Engineering
    let result = Pipeline::scan(emp)
        .inner_join(dept, "dept_id", "dept_id")
        .filter(DExpr::BinOp {
            op: DBinOp::Eq,
            left: Box::new(DExpr::Col("dept_name".into())),
            right: Box::new(DExpr::LitStr("Engineering".into())),
        })
        .collect()
        .unwrap();

    // Only Alice and Carol are in Engineering
    assert_eq!(result.nrows(), 2);
}
