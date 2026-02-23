// Phase 11 — slice_*, slice_sample
// Tests: empty, bounds, head, tail, range, sample determinism, n > nrows
use cjc_data::{Column, DataFrame};

fn make_df(n: usize) -> DataFrame {
    DataFrame::from_columns(vec![
        ("x".into(), Column::Int((0..n as i64).collect())),
    ])
    .unwrap()
}

fn col_int(df: &cjc_data::DataFrame, name: &str) -> Vec<i64> {
    if let Column::Int(v) = df.get_column(name).unwrap() {
        v.clone()
    } else {
        panic!("expected Int");
    }
}

#[test]
fn test_slice_basic_range() {
    let df = make_df(10);
    let view = df.tidy().slice(2, 5);
    let mat = view.materialize().unwrap();
    assert_eq!(col_int(&mat, "x"), vec![2i64, 3, 4]);
}

#[test]
fn test_slice_empty_range() {
    let df = make_df(10);
    let view = df.tidy().slice(5, 5);
    assert_eq!(view.nrows(), 0);
}

#[test]
fn test_slice_out_of_bounds_clamped() {
    let df = make_df(5);
    // end > nrows: clamped to nrows
    let view = df.tidy().slice(3, 1000);
    let mat = view.materialize().unwrap();
    assert_eq!(col_int(&mat, "x"), vec![3i64, 4]);
}

#[test]
fn test_slice_empty_df() {
    let df = make_df(0);
    let view = df.tidy().slice(0, 10);
    assert_eq!(view.nrows(), 0);
}

#[test]
fn test_slice_head() {
    let df = make_df(10);
    let view = df.tidy().slice_head(3);
    let mat = view.materialize().unwrap();
    assert_eq!(col_int(&mat, "x"), vec![0i64, 1, 2]);
}

#[test]
fn test_slice_head_larger_than_nrows() {
    let df = make_df(5);
    let view = df.tidy().slice_head(1000);
    assert_eq!(view.nrows(), 5);
}

#[test]
fn test_slice_head_zero() {
    let df = make_df(5);
    let view = df.tidy().slice_head(0);
    assert_eq!(view.nrows(), 0);
}

#[test]
fn test_slice_tail() {
    let df = make_df(10);
    let view = df.tidy().slice_tail(3);
    let mat = view.materialize().unwrap();
    assert_eq!(col_int(&mat, "x"), vec![7i64, 8, 9]);
}

#[test]
fn test_slice_tail_larger_than_nrows() {
    let df = make_df(5);
    let view = df.tidy().slice_tail(1000);
    assert_eq!(view.nrows(), 5);
}

#[test]
fn test_slice_tail_zero() {
    let df = make_df(5);
    let view = df.tidy().slice_tail(0);
    assert_eq!(view.nrows(), 0);
}

#[test]
fn test_slice_sample_deterministic() {
    // Same seed → same result every time
    let df = make_df(100);
    let result1 = df.clone().tidy().slice_sample(10, 42);
    let result2 = df.tidy().slice_sample(10, 42);
    let m1 = result1.materialize().unwrap();
    let m2 = result2.materialize().unwrap();
    assert_eq!(col_int(&m1, "x"), col_int(&m2, "x"), "same seed → same sample");
}

#[test]
fn test_slice_sample_different_seeds_differ() {
    // Different seeds should (very likely) produce different results for large N
    let df = make_df(100);
    let r1 = df.clone().tidy().slice_sample(10, 1).materialize().unwrap();
    let r2 = df.tidy().slice_sample(10, 999999).materialize().unwrap();
    // This test is probabilistic — with 100 rows and 10 samples, collision is ~1/(C(100,10))
    // Extremely unlikely to match unless LCG is broken
    let v1 = col_int(&r1, "x");
    let v2 = col_int(&r2, "x");
    assert_ne!(v1, v2, "different seeds should (almost certainly) produce different samples");
}

#[test]
fn test_slice_sample_n_gt_nrows_clamped() {
    let df = make_df(5);
    let view = df.tidy().slice_sample(1000, 42);
    // Should return all 5 rows (clamped, no error)
    assert_eq!(view.nrows(), 5);
}

#[test]
fn test_slice_sample_n_zero() {
    let df = make_df(10);
    let view = df.tidy().slice_sample(0, 42);
    assert_eq!(view.nrows(), 0);
}

#[test]
fn test_slice_sample_preserves_ascending_order() {
    // slice_sample sorts selected indices → output in ascending row order
    let df = make_df(100);
    let view = df.tidy().slice_sample(10, 12345);
    let mat = view.materialize().unwrap();
    let v = col_int(&mat, "x");
    let mut sorted = v.clone();
    sorted.sort();
    assert_eq!(v, sorted, "slice_sample output must be in ascending row order");
}

#[test]
fn test_slice_after_filter() {
    let df = make_df(10);
    let view = df
        .tidy()
        .filter(&cjc_data::DExpr::BinOp {
            op: cjc_data::DBinOp::Ge,
            left: Box::new(cjc_data::DExpr::Col("x".into())),
            right: Box::new(cjc_data::DExpr::LitInt(5)),
        })
        .unwrap(); // x in {5,6,7,8,9}
    let sliced = view.slice_head(3);
    let mat = sliced.materialize().unwrap();
    assert_eq!(col_int(&mat, "x"), vec![5i64, 6, 7]);
}
