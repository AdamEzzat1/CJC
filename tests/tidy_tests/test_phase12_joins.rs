// Phase 12 — semi_join, anti_join, inner_join, left_join
// Tests: 1-1, 1-many, many-1, many-many, unknown col, null-equivalent, after filter/arrange
use cjc_data::{ArrangeKey, Column, DataFrame, DBinOp, DExpr, TidyError};

fn employees() -> DataFrame {
    DataFrame::from_columns(vec![
        ("id".into(), Column::Int(vec![1, 2, 3, 4, 5])),
        ("dept".into(), Column::Str(vec![
            "eng".into(), "hr".into(), "eng".into(), "mkt".into(), "hr".into(),
        ])),
        ("salary".into(), Column::Float(vec![100.0, 80.0, 90.0, 70.0, 75.0])),
    ])
    .unwrap()
}

fn departments() -> DataFrame {
    DataFrame::from_columns(vec![
        ("dept".into(), Column::Str(vec!["eng".into(), "hr".into(), "fin".into()])),
        ("budget".into(), Column::Float(vec![500.0, 200.0, 300.0])),
    ])
    .unwrap()
}

fn left_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("k".into(), Column::Int(vec![1, 2, 3, 2])),
        ("lv".into(), Column::Float(vec![10.0, 20.0, 30.0, 40.0])),
    ])
    .unwrap()
}

fn right_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("k".into(), Column::Int(vec![2, 3, 2, 4])),
        ("rv".into(), Column::Float(vec![200.0, 300.0, 400.0, 0.0])),
    ])
    .unwrap()
}

// ── semi_join ────────────────────────────────────────────────────────────────

#[test]
fn test_semi_join_basic() {
    let emp = employees();
    let dept = departments();
    let result = emp.clone().tidy()
        .semi_join(&dept.tidy(), &[("dept", "dept")])
        .unwrap();
    // emp depts: eng, hr, eng, mkt, hr → depts in departments: eng, hr → rows 0,1,2,4
    assert_eq!(result.nrows(), 4);
    // No right-side columns
    assert_eq!(result.ncols(), emp.ncols());
}

#[test]
fn test_semi_join_no_matches() {
    let emp = employees();
    let other = DataFrame::from_columns(vec![
        ("dept".into(), Column::Str(vec!["legal".into()])),
    ])
    .unwrap();
    let result = emp.tidy()
        .semi_join(&other.tidy(), &[("dept", "dept")])
        .unwrap();
    assert_eq!(result.nrows(), 0);
}

#[test]
fn test_semi_join_all_match() {
    let emp = employees();
    let all_depts = DataFrame::from_columns(vec![
        ("dept".into(), Column::Str(vec!["eng".into(), "hr".into(), "mkt".into()])),
    ])
    .unwrap();
    let result = emp.tidy()
        .semi_join(&all_depts.tidy(), &[("dept", "dept")])
        .unwrap();
    assert_eq!(result.nrows(), 5); // all employees match
}

#[test]
fn test_semi_join_unknown_left_col_errors() {
    let emp = employees();
    let dept = departments();
    let err = emp.tidy()
        .semi_join(&dept.tidy(), &[("nonexistent", "dept")])
        .unwrap_err();
    assert!(matches!(err, TidyError::ColumnNotFound(_)));
}

#[test]
fn test_semi_join_unknown_right_col_errors() {
    let emp = employees();
    let dept = departments();
    let err = emp.tidy()
        .semi_join(&dept.tidy(), &[("dept", "nonexistent")])
        .unwrap_err();
    assert!(matches!(err, TidyError::ColumnNotFound(_)));
}

#[test]
fn test_semi_join_preserves_left_order() {
    let left = left_df();
    let right = right_df();
    let result = left.tidy().semi_join(&right.tidy(), &[("k", "k")]).unwrap();
    // left k=1,2,3,2; right has k=2,3 → left rows with k=2(row1),3(row2),2(row3) match
    // Preserved left order: rows 1,2,3
    let mat = result.materialize().unwrap();
    if let Column::Int(v) = mat.get_column("k").unwrap() {
        assert_eq!(*v, vec![2i64, 3, 2]);
    }
}

// ── anti_join ────────────────────────────────────────────────────────────────

#[test]
fn test_anti_join_basic() {
    let emp = employees();
    let dept = departments();
    let result = emp.tidy()
        .anti_join(&dept.tidy(), &[("dept", "dept")])
        .unwrap();
    // mkt (row3) has no match in departments
    assert_eq!(result.nrows(), 1);
    let mat = result.materialize().unwrap();
    if let Column::Str(v) = mat.get_column("dept").unwrap() {
        assert_eq!(v[0], "mkt");
    }
}

#[test]
fn test_anti_join_no_matches_keeps_all() {
    let emp = employees();
    let other = DataFrame::from_columns(vec![
        ("dept".into(), Column::Str(vec!["legal".into()])),
    ])
    .unwrap();
    let result = emp.tidy()
        .anti_join(&other.tidy(), &[("dept", "dept")])
        .unwrap();
    assert_eq!(result.nrows(), 5); // all left rows kept
}

#[test]
fn test_anti_join_all_match_returns_empty() {
    let emp = employees();
    let all_depts = DataFrame::from_columns(vec![
        ("dept".into(), Column::Str(vec!["eng".into(), "hr".into(), "mkt".into()])),
    ])
    .unwrap();
    let result = emp.tidy()
        .anti_join(&all_depts.tidy(), &[("dept", "dept")])
        .unwrap();
    assert_eq!(result.nrows(), 0);
}

// ── inner_join ───────────────────────────────────────────────────────────────

#[test]
fn test_inner_join_one_to_one() {
    let emp = employees();
    let dept = departments();
    let frame = emp.tidy()
        .inner_join(&dept.tidy(), &[("dept", "dept")])
        .unwrap();
    let b = frame.borrow();
    // 4 employees match (eng×2, hr×2); mkt does not match
    assert_eq!(b.nrows(), 4);
    // budget column from right must be present
    assert!(b.get_column("budget").is_some());
    // dept column should appear once (from left), not duplicated
    assert_eq!(
        b.columns.iter().filter(|(n, _)| n == "dept").count(),
        1,
        "join key should not be duplicated"
    );
}

#[test]
fn test_inner_join_many_to_one() {
    let left = left_df();  // k: 1,2,3,2
    let right = right_df(); // k: 2,3,2,4
    let frame = left.tidy()
        .inner_join(&right.tidy(), &[("k", "k")])
        .unwrap();
    let b = frame.borrow();
    // Left k=2 matches right k=2(row0) and k=2(row2) → 2 matches
    // Left k=3 matches right k=3(row1) → 1 match
    // Left k=2(row3) matches right k=2(row0,row2) → 2 matches
    // Left k=1 → no match
    // Total: 2+1+2 = 5 rows
    assert_eq!(b.nrows(), 5);
}

#[test]
fn test_inner_join_ordering_deterministic() {
    // Run twice, verify identical row order
    let left = left_df();
    let right = right_df();
    let frame1 = left.clone().tidy().inner_join(&right.clone().tidy(), &[("k", "k")]).unwrap();
    let frame2 = left.tidy().inner_join(&right.tidy(), &[("k", "k")]).unwrap();
    let b1 = frame1.borrow();
    let b2 = frame2.borrow();
    assert_eq!(b1.nrows(), b2.nrows());
    if let (Column::Float(v1), Column::Float(v2)) = (
        b1.get_column("lv").unwrap(),
        b2.get_column("lv").unwrap(),
    ) {
        assert_eq!(
            v1.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            v2.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "inner_join must be bit-deterministic"
        );
    }
}

#[test]
fn test_inner_join_unknown_left_key_errors() {
    let emp = employees();
    let dept = departments();
    let err = emp.tidy()
        .inner_join(&dept.tidy(), &[("bad_col", "dept")])
        .unwrap_err();
    assert!(matches!(err, TidyError::ColumnNotFound(_)));
}

#[test]
fn test_inner_join_empty_result() {
    let left = DataFrame::from_columns(vec![
        ("k".into(), Column::Int(vec![1, 2])),
    ])
    .unwrap();
    let right = DataFrame::from_columns(vec![
        ("k".into(), Column::Int(vec![9, 10])),
        ("v".into(), Column::Float(vec![1.0, 2.0])),
    ])
    .unwrap();
    let frame = left.tidy().inner_join(&right.tidy(), &[("k", "k")]).unwrap();
    assert_eq!(frame.borrow().nrows(), 0);
}

#[test]
fn test_inner_join_preserves_left_row_order() {
    let emp = employees();
    let dept = departments();
    let frame = emp.tidy().inner_join(&dept.tidy(), &[("dept", "dept")]).unwrap();
    let b = frame.borrow();
    if let Column::Int(ids) = b.get_column("id").unwrap() {
        // Left row order: eng(id=1), hr(id=2), eng(id=3), hr(id=5) → ids 1,2,3,5
        assert_eq!(*ids, vec![1i64, 2, 3, 5]);
    }
}

// ── left_join ────────────────────────────────────────────────────────────────

#[test]
fn test_left_join_basic() {
    let emp = employees();
    let dept = departments();
    let frame = emp.tidy()
        .left_join(&dept.tidy(), &[("dept", "dept")])
        .unwrap();
    let b = frame.borrow();
    // All 5 left rows retained; mkt gets NaN budget
    assert_eq!(b.nrows(), 5);
    if let Column::Float(budgets) = b.get_column("budget").unwrap() {
        // Row 3 (mkt) has no match → NAN
        assert!(budgets[3].is_nan(), "unmatched left row must have NaN budget, got {}", budgets[3]);
        // Others have values
        assert!(budgets[0].is_finite());
    }
}

#[test]
fn test_left_join_all_match() {
    let left = left_df(); // k: 1,2,3,2 — all k in right except k=1
    let right = DataFrame::from_columns(vec![
        ("k".into(), Column::Int(vec![1, 2, 3])),
        ("rv".into(), Column::Float(vec![10.0, 20.0, 30.0])),
    ])
    .unwrap();
    let frame = left.tidy().left_join(&right.tidy(), &[("k", "k")]).unwrap();
    let b = frame.borrow();
    // k=1(row0)→match, k=2(row1)→match, k=3(row2)→match, k=2(row3)→match → 4 rows, all matched
    assert_eq!(b.nrows(), 4);
    if let Column::Float(rv) = b.get_column("rv").unwrap() {
        assert!(rv.iter().all(|x| x.is_finite()), "all matched: all rv should be finite");
    }
}

#[test]
fn test_left_join_no_matches_fills_null() {
    let left = DataFrame::from_columns(vec![
        ("k".into(), Column::Int(vec![9, 10])),
    ])
    .unwrap();
    let right = DataFrame::from_columns(vec![
        ("k".into(), Column::Int(vec![1, 2])),
        ("v".into(), Column::Float(vec![1.0, 2.0])),
    ])
    .unwrap();
    let frame = left.tidy().left_join(&right.tidy(), &[("k", "k")]).unwrap();
    let b = frame.borrow();
    assert_eq!(b.nrows(), 2); // both left rows retained
    if let Column::Float(v) = b.get_column("v").unwrap() {
        assert!(v[0].is_nan());
        assert!(v[1].is_nan());
    }
}

#[test]
fn test_left_join_one_to_many_explosion() {
    // left has k=2 once, right has k=2 twice → left row explodes to 2 output rows
    let left = DataFrame::from_columns(vec![
        ("k".into(), Column::Int(vec![1, 2, 3])),
        ("lv".into(), Column::Float(vec![10.0, 20.0, 30.0])),
    ])
    .unwrap();
    let right = DataFrame::from_columns(vec![
        ("k".into(), Column::Int(vec![2, 2])),
        ("rv".into(), Column::Float(vec![200.0, 201.0])),
    ])
    .unwrap();
    let frame = left.tidy().left_join(&right.tidy(), &[("k", "k")]).unwrap();
    let b = frame.borrow();
    // k=1: no match (1 row), k=2: 2 matches (2 rows), k=3: no match (1 row) → 4 total
    assert_eq!(b.nrows(), 4);
}

#[test]
fn test_left_join_after_filter_and_arrange() {
    let emp = employees();
    let dept = departments();
    // Filter + arrange on left, then join
    let left_view = emp
        .tidy()
        .filter(&DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("salary".into())),
            right: Box::new(DExpr::LitFloat(75.0)),
        })
        .unwrap()
        .arrange(&[ArrangeKey::asc("salary")])
        .unwrap();
    let frame = left_view.left_join(&dept.tidy(), &[("dept", "dept")]).unwrap();
    let b = frame.borrow();
    // salary > 75: row0(100,eng), row1(80,hr), row2(90,eng) → sorted asc: 80,90,100
    assert_eq!(b.nrows(), 3);
    if let Column::Float(sals) = b.get_column("salary").unwrap() {
        assert_eq!(*sals, vec![80.0f64, 90.0, 100.0]);
    }
}

#[test]
fn test_inner_join_then_to_tensor() {
    let emp = employees();
    let dept = departments();
    let frame = emp.tidy().inner_join(&dept.tidy(), &[("dept", "dept")]).unwrap();
    // Result has salary (Float) and budget (Float)
    let view = frame.view();
    let tensor = view.to_tensor(&["salary", "budget"]).unwrap();
    assert_eq!(tensor.shape()[0], 4); // 4 matched rows
    assert_eq!(tensor.shape()[1], 2); // 2 columns
}

#[test]
fn test_many_many_explosion_order_deterministic() {
    // Many-to-many: left k=2×2, right k=2×2 → 4 result rows
    // Order must be deterministic: left outer loop, right sorted ascending
    let left = DataFrame::from_columns(vec![
        ("k".into(), Column::Int(vec![2, 2])),
        ("li".into(), Column::Int(vec![1, 2])),
    ])
    .unwrap();
    let right = DataFrame::from_columns(vec![
        ("k".into(), Column::Int(vec![2, 2])),
        ("ri".into(), Column::Int(vec![10, 20])),
    ])
    .unwrap();
    let frame1 = left.clone().tidy().inner_join(&right.clone().tidy(), &[("k", "k")]).unwrap();
    let frame2 = left.tidy().inner_join(&right.tidy(), &[("k", "k")]).unwrap();
    let b1 = frame1.borrow();
    let b2 = frame2.borrow();
    assert_eq!(b1.nrows(), b2.nrows());
    if let (Column::Int(li1), Column::Int(li2)) = (
        b1.get_column("li").unwrap(),
        b2.get_column("li").unwrap(),
    ) {
        assert_eq!(li1, li2, "many-many must produce identical order across runs");
    }
}
