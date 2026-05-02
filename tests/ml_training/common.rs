// Shared fixtures for Phase 1 DatasetPlan tests.
//
// Keep these tiny and deterministic — every test should be reproducible
// from a literal vector. No file I/O, no environment dependencies.

use cjc_data::{Column, DataFrame};

/// Small mixed-type table — 8 rows, three feature-y columns and a
/// label-y column. Useful baseline for unit tests.
///
/// ```text
///  x: f64       y: i64    cat: Str   label: f64
///  0.0          0         "a"        0.0
///  1.0          10        "b"        1.0
///  2.0          20        "a"        0.0
///  3.0          30        "c"        1.0
///  4.0          40        "b"        0.0
///  5.0          50        "a"        1.0
///  6.0          60        "c"        0.0
///  7.0          70        "b"        1.0
/// ```
pub fn small_mixed_df() -> DataFrame {
    let xs: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let ys: Vec<i64> = (0..8).map(|i| i as i64 * 10).collect();
    let cats: Vec<String> = ["a", "b", "a", "c", "b", "a", "c", "b"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let labels: Vec<f64> = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    DataFrame::from_columns(vec![
        ("x".into(), Column::Float(xs)),
        ("y".into(), Column::Int(ys)),
        ("cat".into(), Column::Str(cats)),
        ("label".into(), Column::Float(labels)),
    ])
    .unwrap()
}

/// Rectangular table with `n` rows; column `idx` is `0..n`,
/// column `even` is true iff `idx % 2 == 0`. Useful for split coverage.
pub fn idx_df(n: usize) -> DataFrame {
    let idx: Vec<i64> = (0..n as i64).collect();
    let even: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    DataFrame::from_columns(vec![
        ("idx".into(), Column::Int(idx)),
        ("even".into(), Column::Bool(even)),
    ])
    .unwrap()
}
