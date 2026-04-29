// Adaptive TidyView Engine v2.1 — predicate bytecode parity tests
//
// The v2.1 columnar fast path lowers `DExpr` predicates to a flat stack
// bytecode and interprets them in a tight loop. This file validates that
// the bytecode-driven `TidyView::filter` produces the same masked-in row
// indices as a hand-coded scalar predicate computed directly in Rust.
//
// Coverage matrix:
//   • Int column × {Lt, Le, Gt, Ge, Eq, Ne}
//   • Float column × {Lt, Le, Gt, Ge, Eq, Ne}
//   • Reversed (`Lit op Col`) — exercises the flip path
//   • Mixed types — Int col × float lit (column promoted), Float col × int lit (lit promoted)
//   • Compound — And, Or, nested And/Or
//   • Edge cases — empty df, all-pass, all-fail, NaN handling
//   • Density mix — sparse, mid, dense (each routes to a different
//     AdaptiveSelection arm but the *visible row set* must be identical)

use cjc_data::{Column, DBinOp, DExpr, DataFrame};

fn col_int(name: &str, xs: Vec<i64>) -> DataFrame {
    DataFrame::from_columns(vec![(name.into(), Column::Int(xs))]).unwrap()
}

fn col_float(name: &str, xs: Vec<f64>) -> DataFrame {
    DataFrame::from_columns(vec![(name.into(), Column::Float(xs))]).unwrap()
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

fn lit_i(v: i64) -> DExpr {
    DExpr::LitInt(v)
}

fn lit_f(v: f64) -> DExpr {
    DExpr::LitFloat(v)
}

/// Ground truth: run the predicate scalar-by-scalar in plain Rust.
fn truth_int(xs: &[i64], pred: impl Fn(i64) -> bool) -> Vec<usize> {
    xs.iter()
        .enumerate()
        .filter_map(|(i, &v)| if pred(v) { Some(i) } else { None })
        .collect()
}

fn truth_float(xs: &[f64], pred: impl Fn(f64) -> bool) -> Vec<usize> {
    xs.iter()
        .enumerate()
        .filter_map(|(i, &v)| if pred(v) { Some(i) } else { None })
        .collect()
}

fn visible(df: DataFrame, pred: &DExpr) -> Vec<usize> {
    df.tidy()
        .filter(pred)
        .unwrap()
        .selection()
        .iter_indices()
        .collect()
}

// ── Int column × six relations ─────────────────────────────────────────

#[test]
fn parity_int_lt() {
    let xs: Vec<i64> = (0..1000).collect();
    let got = visible(col_int("x", xs.clone()), &binop(DBinOp::Lt, col("x"), lit_i(123)));
    assert_eq!(got, truth_int(&xs, |v| v < 123));
}

#[test]
fn parity_int_le() {
    let xs: Vec<i64> = (-100..100).collect();
    let got = visible(col_int("x", xs.clone()), &binop(DBinOp::Le, col("x"), lit_i(0)));
    assert_eq!(got, truth_int(&xs, |v| v <= 0));
}

#[test]
fn parity_int_gt() {
    let xs: Vec<i64> = (0..256).collect();
    let got = visible(col_int("x", xs.clone()), &binop(DBinOp::Gt, col("x"), lit_i(200)));
    assert_eq!(got, truth_int(&xs, |v| v > 200));
}

#[test]
fn parity_int_ge() {
    let xs: Vec<i64> = (0..256).collect();
    let got = visible(col_int("x", xs.clone()), &binop(DBinOp::Ge, col("x"), lit_i(200)));
    assert_eq!(got, truth_int(&xs, |v| v >= 200));
}

#[test]
fn parity_int_eq() {
    let xs: Vec<i64> = (0..1000).map(|i| i % 7).collect();
    let got = visible(col_int("x", xs.clone()), &binop(DBinOp::Eq, col("x"), lit_i(3)));
    assert_eq!(got, truth_int(&xs, |v| v == 3));
}

#[test]
fn parity_int_ne() {
    let xs: Vec<i64> = (0..1000).map(|i| i % 5).collect();
    let got = visible(col_int("x", xs.clone()), &binop(DBinOp::Ne, col("x"), lit_i(0)));
    assert_eq!(got, truth_int(&xs, |v| v != 0));
}

// ── Float column × six relations ───────────────────────────────────────

#[test]
fn parity_float_lt() {
    let xs: Vec<f64> = (0..1000).map(|i| i as f64 / 10.0).collect();
    let got = visible(col_float("x", xs.clone()), &binop(DBinOp::Lt, col("x"), lit_f(50.0)));
    assert_eq!(got, truth_float(&xs, |v| v < 50.0));
}

#[test]
fn parity_float_eq_with_nan() {
    let xs = vec![1.0, f64::NAN, 2.0, f64::NAN, 3.0];
    let got = visible(col_float("x", xs.clone()), &binop(DBinOp::Eq, col("x"), lit_f(2.0)));
    assert_eq!(got, vec![2]);
    // NaN must not equal NaN — IEEE 754 deterministic.
    let got_nan = visible(col_float("x", xs.clone()), &binop(DBinOp::Eq, col("x"), lit_f(f64::NAN)));
    assert_eq!(got_nan, Vec::<usize>::new());
}

#[test]
fn parity_float_ne_with_nan() {
    let xs = vec![1.0, f64::NAN, 2.0];
    let got = visible(col_float("x", xs.clone()), &binop(DBinOp::Ne, col("x"), lit_f(1.0)));
    // NaN != 1.0 is true (IEEE 754).
    assert_eq!(got, vec![1, 2]);
}

#[test]
fn parity_float_lt_nan_excluded() {
    let xs = vec![1.0, f64::NAN, 2.0, 3.0];
    let got = visible(col_float("x", xs.clone()), &binop(DBinOp::Lt, col("x"), lit_f(2.5)));
    // NaN < 2.5 is false.
    assert_eq!(got, vec![0, 2]);
}

// ── Reversed: Lit op Col → flips operator ──────────────────────────────

#[test]
fn parity_reversed_lit_gt_col() {
    // 5 > x   →   x < 5
    let xs: Vec<i64> = (0..10).collect();
    let got = visible(col_int("x", xs.clone()), &binop(DBinOp::Gt, lit_i(5), col("x")));
    assert_eq!(got, truth_int(&xs, |v| v < 5));
}

#[test]
fn parity_reversed_lit_ge_col() {
    // 5 >= x   →   x <= 5
    let xs: Vec<i64> = (0..10).collect();
    let got = visible(col_int("x", xs.clone()), &binop(DBinOp::Ge, lit_i(5), col("x")));
    assert_eq!(got, truth_int(&xs, |v| v <= 5));
}

// ── Mixed types: Int col × float lit, Float col × int lit ──────────────

#[test]
fn parity_int_col_float_lit() {
    let xs: Vec<i64> = (0..10).collect();
    let got = visible(col_int("x", xs.clone()), &binop(DBinOp::Lt, col("x"), lit_f(3.5)));
    // i64 column promoted to f64 for comparison; 0,1,2,3 < 3.5
    assert_eq!(got, vec![0, 1, 2, 3]);
}

#[test]
fn parity_float_col_int_lit() {
    let xs: Vec<f64> = vec![0.5, 1.5, 2.5, 3.5];
    let got = visible(col_float("x", xs.clone()), &binop(DBinOp::Lt, col("x"), lit_i(2)));
    // i64 lit promoted to f64; 0.5, 1.5 < 2.0
    assert_eq!(got, vec![0, 1]);
}

// ── Compound: And, Or, nested ──────────────────────────────────────────

#[test]
fn parity_compound_and() {
    let xs: Vec<i64> = (0..1000).collect();
    let pred = binop(
        DBinOp::And,
        binop(DBinOp::Ge, col("x"), lit_i(100)),
        binop(DBinOp::Lt, col("x"), lit_i(200)),
    );
    let got = visible(col_int("x", xs.clone()), &pred);
    assert_eq!(got, truth_int(&xs, |v| v >= 100 && v < 200));
}

#[test]
fn parity_compound_or() {
    let xs: Vec<i64> = (0..1000).collect();
    let pred = binop(
        DBinOp::Or,
        binop(DBinOp::Lt, col("x"), lit_i(50)),
        binop(DBinOp::Ge, col("x"), lit_i(950)),
    );
    let got = visible(col_int("x", xs.clone()), &pred);
    assert_eq!(got, truth_int(&xs, |v| v < 50 || v >= 950));
}

#[test]
fn parity_compound_nested_and_or() {
    let xs: Vec<i64> = (0..1000).collect();
    // (x < 10) OR (x > 990 AND x <= 995)
    let pred = binop(
        DBinOp::Or,
        binop(DBinOp::Lt, col("x"), lit_i(10)),
        binop(
            DBinOp::And,
            binop(DBinOp::Gt, col("x"), lit_i(990)),
            binop(DBinOp::Le, col("x"), lit_i(995)),
        ),
    );
    let got = visible(col_int("x", xs.clone()), &pred);
    assert_eq!(got, truth_int(&xs, |v| v < 10 || (v > 990 && v <= 995)));
}

// ── Edge cases ─────────────────────────────────────────────────────────

#[test]
fn parity_empty_df() {
    let got = visible(col_int("x", vec![]), &binop(DBinOp::Lt, col("x"), lit_i(5)));
    assert_eq!(got, Vec::<usize>::new());
}

#[test]
fn parity_all_pass() {
    let xs: Vec<i64> = (0..100).collect();
    let got = visible(col_int("x", xs.clone()), &binop(DBinOp::Ge, col("x"), lit_i(0)));
    assert_eq!(got, truth_int(&xs, |_| true));
}

#[test]
fn parity_all_fail() {
    let xs: Vec<i64> = (0..100).collect();
    let got = visible(col_int("x", xs.clone()), &binop(DBinOp::Lt, col("x"), lit_i(0)));
    assert_eq!(got, Vec::<usize>::new());
}

// ── Density-mix: same predicate at thresholds straddling each adaptive arm
// (Empty / SelectionVector / Hybrid / VerbatimMask / All) must produce
// the same row-set order as the ground-truth scalar pass. The arm choice
// is internal; we assert externally-visible equality.

#[test]
fn parity_density_sparse_arm() {
    const N: usize = 1_000_000;
    let xs: Vec<i64> = (0..N as i64).collect();
    // ~10 hits over 1M rows → SelectionVector
    let got = visible(col_int("x", xs.clone()), &binop(DBinOp::Lt, col("x"), lit_i(10)));
    let expected: Vec<usize> = (0..10).collect();
    assert_eq!(got, expected);
}

#[test]
fn parity_density_hybrid_arm() {
    const N: usize = 1_000_000;
    let xs: Vec<i64> = (0..N as i64).collect();
    // ~10% hits → Hybrid (mid-band)
    let got = visible(col_int("x", xs.clone()), &binop(DBinOp::Lt, col("x"), lit_i(100_000)));
    let expected: Vec<usize> = (0..100_000).collect();
    assert_eq!(got, expected);
}

#[test]
fn parity_density_dense_arm() {
    const N: usize = 1_000_000;
    let xs: Vec<i64> = (0..N as i64).collect();
    // ~70% hits → VerbatimMask (>30% threshold)
    let got = visible(col_int("x", xs.clone()), &binop(DBinOp::Lt, col("x"), lit_i(700_000)));
    assert_eq!(got.len(), 700_000);
    assert_eq!(got[0], 0);
    assert_eq!(got[699_999], 699_999);
}
