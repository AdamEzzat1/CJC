//! Proptest property tests for TidyView performance optimizations (P1-P6).
//!
//! These tests verify invariants across random inputs:
//! - arrange preserves row count
//! - stable sort preserves tie order
//! - group_by + summarise(Sum) is Kahan-deterministic
//! - filter-then-arrange produces sorted filtered output

use cjc_data::{ArrangeKey, Column, DBinOp, DExpr, DataFrame, TidyAgg};
use proptest::prelude::*;

// ── Helpers ────────────────────────────────────────────────────────────────

fn make_random_df(ints: Vec<i64>, floats: Vec<f64>) -> DataFrame {
    let n = ints.len().min(floats.len());
    let ints = ints[..n].to_vec();
    let floats = floats[..n].to_vec();
    DataFrame::from_columns(vec![
        ("id".into(), Column::Int(ints)),
        ("val".into(), Column::Float(floats)),
    ])
    .unwrap()
}

// ── Properties ─────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_arrange_preserves_row_count(
        data in prop::collection::vec(-1000i64..1000, 10..2000),
    ) {
        let n = data.len();
        let floats: Vec<f64> = data.iter().map(|&x| x as f64 * 0.1).collect();
        let df = make_random_df(data, floats);
        let view = df.tidy();
        let sorted = view.arrange(&[ArrangeKey::asc("id")]).unwrap();
        let frame = sorted.materialize().unwrap();
        prop_assert_eq!(frame.nrows(), n);
    }

    #[test]
    fn prop_arrange_actually_sorts(
        data in prop::collection::vec(-1000i64..1000, 10..2000),
    ) {
        let floats: Vec<f64> = data.iter().map(|&x| x as f64).collect();
        let df = make_random_df(data, floats);
        let view = df.tidy();
        let sorted = view.arrange(&[ArrangeKey::asc("id")]).unwrap();
        let frame = sorted.materialize().unwrap();
        if let Column::Int(v) = frame.get_column("id").unwrap() {
            for w in v.windows(2) {
                prop_assert!(w[0] <= w[1], "not sorted: {} > {}", w[0], w[1]);
            }
        }
    }

    #[test]
    fn prop_arrange_stable_sort_ties(
        groups in prop::collection::vec(0i64..5, 20..500),
    ) {
        // When sorted by group, rows with the same group key keep original order
        let n = groups.len();
        let order: Vec<i64> = (0..n as i64).collect();
        let floats: Vec<f64> = order.iter().map(|&x| x as f64).collect();
        let df = DataFrame::from_columns(vec![
            ("grp".into(), Column::Int(groups)),
            ("order".into(), Column::Int(order)),
            ("val".into(), Column::Float(floats)),
        ]).unwrap();
        let sorted = df.tidy().arrange(&[ArrangeKey::asc("grp")]).unwrap();
        let frame = sorted.materialize().unwrap();
        if let (Column::Int(grps), Column::Int(ords)) = (
            frame.get_column("grp").unwrap(),
            frame.get_column("order").unwrap(),
        ) {
            // Within each group, order values should be ascending (stable)
            for w in grps.windows(2).zip(ords.windows(2)) {
                let (g, o) = w;
                if g[0] == g[1] {
                    prop_assert!(o[0] < o[1], "unstable: same group {} but order {} >= {}", g[0], o[0], o[1]);
                }
            }
        }
    }

    #[test]
    fn prop_group_summarise_sum_kahan_deterministic(
        n in 100usize..5000,
        n_groups in 2usize..50,
    ) {
        let groups: Vec<i64> = (0..n).map(|i| (i % n_groups) as i64).collect();
        let vals: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001 + 1e-15).collect();
        let df = DataFrame::from_columns(vec![
            ("grp".into(), Column::Int(groups)),
            ("val".into(), Column::Float(vals)),
        ]).unwrap();

        let g1 = df.clone().tidy().group_by(&["grp"]).unwrap();
        let r1 = g1.summarise(&[("s", TidyAgg::Sum("val".into()))]).unwrap();
        let b1: Vec<u64> = if let Column::Float(v) = r1.borrow().get_column("s").unwrap() {
            v.iter().map(|f| f.to_bits()).collect()
        } else { panic!("") };

        // Second run must be bit-identical
        let g2 = df.tidy().group_by(&["grp"]).unwrap();
        let r2 = g2.summarise(&[("s", TidyAgg::Sum("val".into()))]).unwrap();
        let b2: Vec<u64> = if let Column::Float(v) = r2.borrow().get_column("s").unwrap() {
            v.iter().map(|f| f.to_bits()).collect()
        } else { panic!("") };

        prop_assert_eq!(b1, b2, "Kahan sums differ between runs");
    }

    #[test]
    fn prop_filter_arrange_produces_sorted_subset(
        data in prop::collection::vec(0i64..100, 50..1000),
        threshold in 10i64..90,
    ) {
        let floats: Vec<f64> = data.iter().map(|&x| x as f64).collect();
        let df = make_random_df(data, floats);
        let view = df.tidy();
        let filtered = view.filter(&DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("id".into())),
            right: Box::new(DExpr::LitInt(threshold)),
        }).unwrap();
        let sorted = filtered.arrange(&[ArrangeKey::asc("id")]).unwrap();
        let frame = sorted.materialize().unwrap();
        if let Column::Int(v) = frame.get_column("id").unwrap() {
            for val in v {
                prop_assert!(*val > threshold, "filter leaked: {} <= {}", val, threshold);
            }
            for w in v.windows(2) {
                prop_assert!(w[0] <= w[1], "not sorted after filter+arrange");
            }
        }
    }
}
