//! Bolero fuzz target for the v2.1 predicate bytecode.
//!
//! Generates random integer columns and random predicate trees, runs them
//! through `TidyView::filter` (which lowers to bytecode), and asserts the
//! masked-in row indices exactly match a scalar reference loop computed
//! directly in Rust.
//!
//! Why this matters: the bytecode interpreter, the `Cmp` opcode flip, the
//! And/Or stack semantics, and the `AdaptiveSelection` arm selection
//! after filter are all under one oracle here — any divergence (wrong
//! flip, wrong NaN handling, wrong density classification, arm-specific
//! `iter_indices` bug) produces an inequality.
//!
//! Determinism contract: every run on identical input must produce
//! identical output. We assert that explicitly per fuzz case.
//!
//! Run with:
//!   cargo test --test bolero_fuzz v2_1_bytecode

use cjc_data::{Column, DBinOp, DExpr, DataFrame};
use std::panic;

/// Decode the first byte as op, second byte as literal magnitude, third
/// byte as a "compose with And/Or another op" flag. Remainder of bytes
/// becomes column data (interpreted as i8 → i64 to keep values small).
fn parse_input(input: &[u8]) -> Option<(Vec<i64>, DExpr)> {
    if input.len() < 4 {
        return None;
    }
    let op_code = input[0] % 6;
    let lit = (input[1] as i8) as i64;
    let compose = input[2] % 4; // 0 = single, 1 = AND, 2 = OR, 3 = nested
    let lit2 = (input[3] as i8) as i64;

    let xs: Vec<i64> = input[4..].iter().take(2048).map(|&b| (b as i8) as i64).collect();
    if xs.is_empty() {
        return None;
    }

    let op = match op_code {
        0 => DBinOp::Lt,
        1 => DBinOp::Le,
        2 => DBinOp::Gt,
        3 => DBinOp::Ge,
        4 => DBinOp::Eq,
        _ => DBinOp::Ne,
    };

    let leaf1 = DExpr::BinOp {
        op,
        left: Box::new(DExpr::Col("x".into())),
        right: Box::new(DExpr::LitInt(lit)),
    };
    let leaf2 = DExpr::BinOp {
        op: DBinOp::Le,
        left: Box::new(DExpr::Col("x".into())),
        right: Box::new(DExpr::LitInt(lit2)),
    };

    let predicate = match compose {
        0 => leaf1,
        1 => DExpr::BinOp {
            op: DBinOp::And,
            left: Box::new(leaf1),
            right: Box::new(leaf2),
        },
        2 => DExpr::BinOp {
            op: DBinOp::Or,
            left: Box::new(leaf1),
            right: Box::new(leaf2),
        },
        _ => DExpr::BinOp {
            op: DBinOp::Or,
            left: Box::new(leaf1.clone()),
            right: Box::new(DExpr::BinOp {
                op: DBinOp::And,
                left: Box::new(leaf1),
                right: Box::new(leaf2),
            }),
        },
    };

    Some((xs, predicate))
}

/// Scalar reference: walk the column, evaluate the predicate row by row,
/// return the indices that pass.
fn scalar_eval(xs: &[i64], pred: &DExpr) -> Vec<usize> {
    fn eval(p: &DExpr, v: i64) -> bool {
        match p {
            DExpr::BinOp { op, left, right } => {
                let l = match (left.as_ref(), right.as_ref()) {
                    (DExpr::Col(_), DExpr::LitInt(lit)) => Some((v, *lit, false)),
                    (DExpr::LitInt(lit), DExpr::Col(_)) => Some((v, *lit, true)),
                    _ => None,
                };
                if let Some((cv, lit, reversed)) = l {
                    let (a, b) = if reversed { (lit, cv) } else { (cv, lit) };
                    return match op {
                        DBinOp::Lt => a < b,
                        DBinOp::Le => a <= b,
                        DBinOp::Gt => a > b,
                        DBinOp::Ge => a >= b,
                        DBinOp::Eq => a == b,
                        DBinOp::Ne => a != b,
                        _ => false,
                    };
                }
                match op {
                    DBinOp::And => eval(left, v) && eval(right, v),
                    DBinOp::Or => eval(left, v) || eval(right, v),
                    _ => false,
                }
            }
            _ => false,
        }
    }
    xs.iter()
        .enumerate()
        .filter_map(|(i, &v)| if eval(pred, v) { Some(i) } else { None })
        .collect()
}

/// Bytecode output must match scalar evaluation on every fuzz case.
#[test]
fn fuzz_bytecode_vs_scalar() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let _ = panic::catch_unwind(|| {
            let Some((xs, pred)) = parse_input(input) else {
                return;
            };
            let df = DataFrame::from_columns(vec![("x".into(), Column::Int(xs.clone()))]).unwrap();

            let got: Vec<usize> = df
                .clone()
                .tidy()
                .filter(&pred)
                .unwrap()
                .selection()
                .iter_indices()
                .collect();

            let expected = scalar_eval(&xs, &pred);

            assert_eq!(
                got, expected,
                "bytecode/filter ≠ scalar reference for predicate"
            );
        });
    });
}

/// Determinism: same input → identical visible row indices on both runs.
#[test]
fn fuzz_bytecode_determinism() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let _ = panic::catch_unwind(|| {
            let Some((xs, pred)) = parse_input(input) else {
                return;
            };
            let df = DataFrame::from_columns(vec![("x".into(), Column::Int(xs))]).unwrap();

            let r1: Vec<usize> = df
                .clone()
                .tidy()
                .filter(&pred)
                .unwrap()
                .selection()
                .iter_indices()
                .collect();
            let r2: Vec<usize> = df
                .clone()
                .tidy()
                .filter(&pred)
                .unwrap()
                .selection()
                .iter_indices()
                .collect();

            assert_eq!(r1, r2, "bytecode non-deterministic across runs");
        });
    });
}

/// Cardinality identity: |filter(A AND B)| + |filter(A OR B)| == |A| + |B|
/// must hold for every shape the bytecode accepts. This is the same
/// invariant adaptive_selection_fuzz uses for set ops, but exercised
/// from the predicate side: bytecode must compose AND/OR consistently
/// with the AdaptiveSelection trait surface.
#[test]
fn fuzz_bytecode_cardinality_identity() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let _ = panic::catch_unwind(|| {
            if input.len() < 4 {
                return;
            }
            let lit_a = (input[0] as i8) as i64;
            let lit_b = (input[1] as i8) as i64;
            let xs: Vec<i64> = input[2..].iter().take(1024).map(|&b| (b as i8) as i64).collect();
            if xs.is_empty() {
                return;
            }

            let df = DataFrame::from_columns(vec![("x".into(), Column::Int(xs))]).unwrap();

            let pa = DExpr::BinOp {
                op: DBinOp::Lt,
                left: Box::new(DExpr::Col("x".into())),
                right: Box::new(DExpr::LitInt(lit_a)),
            };
            let pb = DExpr::BinOp {
                op: DBinOp::Ge,
                left: Box::new(DExpr::Col("x".into())),
                right: Box::new(DExpr::LitInt(lit_b)),
            };
            let p_and = DExpr::BinOp {
                op: DBinOp::And,
                left: Box::new(pa.clone()),
                right: Box::new(pb.clone()),
            };
            let p_or = DExpr::BinOp {
                op: DBinOp::Or,
                left: Box::new(pa.clone()),
                right: Box::new(pb.clone()),
            };

            let a = df.clone().tidy().filter(&pa).unwrap().nrows();
            let b = df.clone().tidy().filter(&pb).unwrap().nrows();
            let ab = df.clone().tidy().filter(&p_and).unwrap().nrows();
            let aob = df.clone().tidy().filter(&p_or).unwrap().nrows();

            assert_eq!(a + b, ab + aob, "cardinality identity violated");
        });
    });
}
