//! The vectorized `mutate` fast path must compute `log`/`exp`/`sin`/`cos`/`tan`
//! with `cjc_repro::dmath`, exactly like the row-by-row evaluator.
//!
//! It used to call `f64::ln`/`exp`/`sin`/`cos`/`tan`, which are platform libm.
//! The same DataFrame expression then gave different bits depending on whether
//! the fast path or the row fallback ran, and different bits on different OSes.
//!
//! Oracle: the dmath functions called directly. dmath uses only IEEE
//! `+ - * / sqrt`, so it gives the same bits on every platform; it is the
//! definition the row evaluator already uses.

use cjc_data::{Column, DExpr, DataFrame};
use cjc_repro::dmath::DetMath;

/// Deterministic inputs (SplitMix64) mapped into `[lo, hi)`.
fn inputs(seed: u64, n: usize, lo: f64, hi: f64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let u = (z >> 11) as f64 / (1u64 << 53) as f64;
            lo + (hi - lo) * u
        })
        .collect()
}

fn mutate_fn(name: &str, xs: &[f64]) -> Vec<f64> {
    let df = DataFrame::from_columns(vec![("x".into(), Column::Float(xs.to_vec()))]).unwrap();
    let out = df
        .tidy()
        .mutate(&[("y", DExpr::FnCall(name.into(), vec![DExpr::Col("x".into())]))])
        .unwrap();
    let mat = out.borrow();
    match mat.get_column("y").unwrap() {
        Column::Float(v) => v.clone(),
        other => panic!("{name}: expected Float column, got {other:?}"),
    }
}

/// Every element must have the same bits as the dmath oracle.
fn check(name: &str, xs: &[f64], oracle: fn(f64) -> f64) {
    let got = mutate_fn(name, xs);
    assert_eq!(got.len(), xs.len());
    let bad: Vec<(f64, f64, f64)> = xs
        .iter()
        .zip(got.iter())
        .filter(|(x, g)| g.to_bits() != oracle(**x).to_bits())
        .map(|(x, g)| (*x, *g, oracle(*x)))
        .collect();
    assert!(
        bad.is_empty(),
        "{name}: {} of {} elements differ from dmath; first: x={:?} got={:?} dmath={:?}",
        bad.len(),
        xs.len(),
        bad[0].0,
        bad[0].1,
        bad[0].2
    );
}

#[test]
fn mutate_log_uses_dmath() {
    let mut xs = inputs(1, 4000, 0.5, 2.0);
    xs.extend(inputs(2, 4000, 1e-300, 1e300));
    check("log", &xs, |x| x.det_ln());
}

#[test]
fn mutate_exp_uses_dmath() {
    let mut xs = inputs(3, 4000, -1.0, 1.0);
    xs.extend(inputs(4, 4000, -700.0, 700.0));
    check("exp", &xs, |x| x.det_exp());
}

#[test]
fn mutate_sin_uses_dmath() {
    let mut xs = inputs(5, 4000, -10.0, 10.0);
    xs.extend(inputs(6, 4000, -1e6, 1e6));
    check("sin", &xs, |x| x.det_sin());
}

#[test]
fn mutate_cos_uses_dmath() {
    let mut xs = inputs(7, 4000, -10.0, 10.0);
    xs.extend(inputs(8, 4000, -1e6, 1e6));
    check("cos", &xs, |x| x.det_cos());
}

#[test]
fn mutate_tan_uses_dmath() {
    let mut xs = inputs(9, 4000, -1.5, 1.5);
    xs.extend(inputs(10, 4000, -1e6, 1e6));
    check("tan", &xs, |x| x.det_tan());
}

/// Int columns are promoted to f64 and go through the same functions.
#[test]
fn mutate_int_column_uses_dmath() {
    let ints: Vec<i64> = (1..=2000).map(|i| i * 7919).collect();
    let df = DataFrame::from_columns(vec![("x".into(), Column::Int(ints.clone()))]).unwrap();
    let out = df
        .tidy()
        .mutate(&[("y", DExpr::FnCall("log".into(), vec![DExpr::Col("x".into())]))])
        .unwrap();
    let mat = out.borrow();
    let Column::Float(got) = mat.get_column("y").unwrap() else { panic!("Float expected") };
    for (i, g) in ints.iter().zip(got.iter()) {
        assert_eq!(g.to_bits(), (*i as f64).det_ln().to_bits(), "log({i})");
    }
}

/// Special values match dmath too (NaN payloads aside: compared by is_nan).
#[test]
fn mutate_special_values_match_dmath() {
    let xs = [0.0, -0.0, 1.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, f64::MIN_POSITIVE, 5e-324];
    for (name, oracle) in [
        ("log", (|x: f64| x.det_ln()) as fn(f64) -> f64),
        ("exp", |x: f64| x.det_exp()),
        ("sin", |x: f64| x.det_sin()),
        ("cos", |x: f64| x.det_cos()),
        ("tan", |x: f64| x.det_tan()),
    ] {
        let got = mutate_fn(name, &xs);
        for (x, g) in xs.iter().zip(got.iter()) {
            let w = oracle(*x);
            let same = (g.is_nan() && w.is_nan()) || g.to_bits() == w.to_bits();
            assert!(same, "{name}({x:?}): got {g:?}, dmath {w:?}");
        }
    }
}

#[test]
fn mutate_fn_is_deterministic() {
    let xs = inputs(11, 2000, -50.0, 50.0);
    let first: Vec<u64> = mutate_fn("sin", &xs).iter().map(|v| v.to_bits()).collect();
    for _ in 0..10 {
        let again: Vec<u64> = mutate_fn("sin", &xs).iter().map(|v| v.to_bits()).collect();
        assert_eq!(again, first);
    }
}
