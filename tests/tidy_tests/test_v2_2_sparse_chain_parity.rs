// Adaptive TidyView Engine v2.2 — sparse-chain filter parity tests
//
// In v2.2 the predicate bytecode interpreter has two paths: the
// sequential column-scan `interpret` (used when the existing selection
// retains ≥ 25% of rows) and the random-access gather `interpret_sparse`
// (used when the parent selection has already narrowed below 25%). For
// the sparse path to be safe to enable in production, every chained
// `df.filter(p1).filter(p2)` must produce the same masked-in row indices
// as a single equivalent `df.filter(p1 AND p2)`, and as a row-by-row
// scalar reference.
//
// This file pins both equivalences across the chain → AND-collapse →
// scalar-reference triangle for chains where the second filter is
// guaranteed to land on the sparse path.

use cjc_data::{Column, DBinOp, DExpr, DataFrame};

fn col_int(name: &str, xs: Vec<i64>) -> DataFrame {
    DataFrame::from_columns(vec![(name.into(), Column::Int(xs))]).unwrap()
}

fn binop(op: DBinOp, l: DExpr, r: DExpr) -> DExpr {
    DExpr::BinOp {
        op,
        left: Box::new(l),
        right: Box::new(r),
    }
}

fn col(name: &str) -> DExpr {
    DExpr::Col(name.into())
}

fn lit(v: i64) -> DExpr {
    DExpr::LitInt(v)
}

/// Visible row indices after applying a chain of predicates.
fn chain(df: DataFrame, predicates: &[DExpr]) -> Vec<usize> {
    let mut view = df.tidy();
    for p in predicates {
        view = view.filter(p).unwrap();
    }
    view.selection().iter_indices().collect()
}

#[test]
fn sparse_chain_matches_anded_single_filter() {
    // After `x < 100` over 0..1M, the parent retains 100/1M = 0.01% — far
    // below the 25% threshold, so the second filter takes the sparse path.
    let xs: Vec<i64> = (0..1_000_000).collect();
    let p1 = binop(DBinOp::Lt, col("x"), lit(100));
    let p2 = binop(DBinOp::Ge, col("x"), lit(50));

    let chained = chain(col_int("x", xs.clone()), &[p1.clone(), p2.clone()]);
    let single = chain(
        col_int("x", xs.clone()),
        &[binop(DBinOp::And, p1, p2)],
    );

    assert_eq!(chained, single);
    // Ground truth: 50..100
    let expected: Vec<usize> = (50..100).collect();
    assert_eq!(chained, expected);
}

#[test]
fn sparse_chain_with_or_matches_anded_single_filter() {
    // Chain narrows to 200 rows over 1M (sparse band), then OR-predicate
    // — exercises the sparse-OR-monotone path through the live filter API.
    let xs: Vec<i64> = (0..1_000_000).collect();
    let p1 = binop(DBinOp::Lt, col("x"), lit(200));
    let p2 = binop(
        DBinOp::Or,
        binop(DBinOp::Lt, col("x"), lit(50)),
        binop(DBinOp::Ge, col("x"), lit(150)),
    );

    let chained = chain(col_int("x", xs.clone()), &[p1.clone(), p2.clone()]);
    let single = chain(
        col_int("x", xs.clone()),
        &[binop(DBinOp::And, p1, p2)],
    );

    assert_eq!(chained, single);
    let mut expected: Vec<usize> = (0..50).collect();
    expected.extend(150..200);
    assert_eq!(chained, expected);
}

#[test]
fn three_filter_chain_matches_anded_single_filter() {
    // Three-step chain: 0..1M → x<10000 (1%) → x≥5000 → x<7500.
    // Steps 2 and 3 both land on the sparse path.
    let xs: Vec<i64> = (0..1_000_000).collect();
    let p1 = binop(DBinOp::Lt, col("x"), lit(10_000));
    let p2 = binop(DBinOp::Ge, col("x"), lit(5_000));
    let p3 = binop(DBinOp::Lt, col("x"), lit(7_500));

    let chained = chain(col_int("x", xs.clone()), &[p1.clone(), p2.clone(), p3.clone()]);
    let single = chain(
        col_int("x", xs.clone()),
        &[binop(
            DBinOp::And,
            binop(DBinOp::And, p1, p2),
            p3,
        )],
    );

    assert_eq!(chained, single);
    let expected: Vec<usize> = (5000..7500).collect();
    assert_eq!(chained, expected);
}

#[test]
fn sparse_chain_empty_intermediate_still_correct() {
    // First filter retains zero rows → second filter sees Empty selection.
    // Sparse path with empty existing_indices must produce empty output
    // without touching the column buffer.
    let xs: Vec<i64> = (0..1000).collect();
    let p_impossible = binop(DBinOp::Lt, col("x"), lit(-1));
    let p_irrelevant = binop(DBinOp::Lt, col("x"), lit(500));

    let chained = chain(col_int("x", xs), &[p_impossible, p_irrelevant]);
    assert_eq!(chained, Vec::<usize>::new());
}

#[test]
fn dense_to_sparse_transition_via_chain() {
    // Filter 1 keeps 50% (dense path); filter 2 narrows to 1% (sparse
    // path triggers on second call). Combined behavior must still match
    // the AND-collapsed single filter.
    let xs: Vec<i64> = (0..200_000).collect();
    let p1 = binop(DBinOp::Lt, col("x"), lit(100_000));      // 50% — dense parent
    let p2 = binop(DBinOp::Lt, col("x"), lit(2_000));        // 2% absolute — sparse on second call

    let chained = chain(col_int("x", xs.clone()), &[p1.clone(), p2.clone()]);
    let single = chain(
        col_int("x", xs.clone()),
        &[binop(DBinOp::And, p1, p2)],
    );

    assert_eq!(chained, single);
    let expected: Vec<usize> = (0..2_000).collect();
    assert_eq!(chained, expected);
}
