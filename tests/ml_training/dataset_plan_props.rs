//! Phase 1 — DatasetPlan property tests (proptest, 256 cases each).
//!
//! Properties:
//!   1. **Sequential split coverage** — train ∪ val ∪ test partitions are
//!      disjoint and sum to ≤ nrows.
//!   2. **No-shuffle iteration** — when `shuffle = None`, batched row IDs
//!      are exactly the ascending split row IDs.
//!   3. **Shuffle is a permutation** — when `shuffle = Some(seed)`, the
//!      visited rows are a permutation of the split rows (same multiset).
//!   4. **Shuffle determinism** — two iterations with the same seed yield
//!      identical row orders.
//!   5. **Hashed split disjointness** — Train/Val/Test are pairwise
//!      disjoint and ascending.
//!   6. **Categorical encoding determinism** — same source rows + same
//!      ordering produce byte-identical feature tensors.

use proptest::prelude::*;

use cjc_data::byte_dict::CategoryOrdering;
use cjc_data::{
    BatchSpec, Column, DataFrame, DatasetPlan, EncodingSpec, Split, SplitSpec,
};

fn idx_df(n: usize) -> DataFrame {
    let idx: Vec<i64> = (0..n as i64).collect();
    DataFrame::from_columns(vec![("idx".into(), Column::Int(idx))]).unwrap()
}

fn cat_df(values: Vec<String>) -> DataFrame {
    DataFrame::from_columns(vec![("c".into(), Column::Str(values))]).unwrap()
}

fn frac_strategy() -> impl Strategy<Value = (f64, f64, f64)> {
    // Three positive fractions summing to ≤ 1.0. Build by sampling two
    // cut-points on [0, 1] and taking widths; clamp the residual to test.
    (0u32..1000, 0u32..1000).prop_map(|(a, b)| {
        let mut cuts = [a as f64 / 1000.0, b as f64 / 1000.0];
        cuts.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let train = cuts[0];
        let val = cuts[1] - cuts[0];
        let test = 1.0 - cuts[1];
        (train, val, test)
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn prop_sequential_split_disjoint_and_within_bounds(
        n in 1usize..2_000,
        (train, val, test) in frac_strategy(),
    ) {
        let plan = DatasetPlan::from_dataframe(idx_df(n))
            .with_features(vec!["idx".into()])
            .with_split(SplitSpec::Sequential { train, val, test });
        let tr = plan.split_rows(Split::Train).unwrap();
        let va = plan.split_rows(Split::Val).unwrap();
        let te = plan.split_rows(Split::Test).unwrap();

        // Each split is ascending.
        prop_assert!(tr.windows(2).all(|w| w[0] < w[1]));
        prop_assert!(va.windows(2).all(|w| w[0] < w[1]));
        prop_assert!(te.windows(2).all(|w| w[0] < w[1]));

        // Union is contiguous and total length ≤ n.
        let total = tr.len() + va.len() + te.len();
        prop_assert!(total <= n);

        // Disjoint: max(train) < min(val) < max(val) < min(test).
        if let (Some(&tr_max), Some(&va_min)) = (tr.last(), va.first()) {
            prop_assert!(tr_max < va_min);
        }
        if let (Some(&va_max), Some(&te_min)) = (va.last(), te.first()) {
            prop_assert!(va_max < te_min);
        }
    }

    #[test]
    fn prop_no_shuffle_iter_matches_split_ascending(
        n in 1usize..500,
        batch_size in 1usize..=64,
        (train, val, test) in frac_strategy(),
    ) {
        let plan = DatasetPlan::from_dataframe(idx_df(n))
            .with_features(vec!["idx".into()])
            .with_encoding("idx".into(), EncodingSpec::IntAsFloat)
            .with_split(SplitSpec::Sequential { train, val, test })
            .with_batch(BatchSpec::new(batch_size)); // no shuffle

        let split_rows = plan.split_rows(Split::Train).unwrap();
        let visited: Vec<u32> = plan.iter_batches(Split::Train).unwrap()
            .flat_map(|b| b.unwrap().row_ids)
            .collect();
        prop_assert_eq!(visited, split_rows);
    }

    #[test]
    fn prop_shuffle_is_permutation_and_deterministic(
        n in 1usize..500,
        batch_size in 1usize..=64,
        seed in any::<u64>(),
    ) {
        let mk = || DatasetPlan::from_dataframe(idx_df(n))
            .with_features(vec!["idx".into()])
            .with_encoding("idx".into(), EncodingSpec::IntAsFloat)
            .with_batch(BatchSpec::new(batch_size).with_shuffle(seed));

        let a: Vec<u32> = mk().iter_batches(Split::Full).unwrap()
            .flat_map(|b| b.unwrap().row_ids).collect();
        let b: Vec<u32> = mk().iter_batches(Split::Full).unwrap()
            .flat_map(|b| b.unwrap().row_ids).collect();

        // Same seed → same order.
        prop_assert_eq!(&a, &b);

        // Permutation of 0..n.
        let mut sorted = a;
        sorted.sort();
        prop_assert_eq!(sorted, (0u32..n as u32).collect::<Vec<_>>());
    }

    #[test]
    fn prop_hashed_split_pairwise_disjoint(
        n in 1usize..1_000,
        seed in any::<u64>(),
        (train, val, test) in frac_strategy(),
    ) {
        let plan = DatasetPlan::from_dataframe(idx_df(n))
            .with_features(vec!["idx".into()])
            .with_split(SplitSpec::Hashed { seed, train, val, test });
        let tr = plan.split_rows(Split::Train).unwrap();
        let va = plan.split_rows(Split::Val).unwrap();
        let te = plan.split_rows(Split::Test).unwrap();

        // Each ascending.
        prop_assert!(tr.windows(2).all(|w| w[0] < w[1]));
        prop_assert!(va.windows(2).all(|w| w[0] < w[1]));
        prop_assert!(te.windows(2).all(|w| w[0] < w[1]));

        // Pairwise disjoint via BTreeSet intersection (BTreeSet is
        // deterministic, no HashSet allowed by determinism rules).
        use std::collections::BTreeSet;
        let s_tr: BTreeSet<u32> = tr.iter().copied().collect();
        let s_va: BTreeSet<u32> = va.iter().copied().collect();
        let s_te: BTreeSet<u32> = te.iter().copied().collect();
        prop_assert!(s_tr.is_disjoint(&s_va));
        prop_assert!(s_tr.is_disjoint(&s_te));
        prop_assert!(s_va.is_disjoint(&s_te));
    }

    #[test]
    fn prop_categorical_encoding_is_deterministic(
        // Random small categorical column from a fixed alphabet so the
        // dictionary stays small and FirstSeen is well-defined.
        values in proptest::collection::vec(prop_oneof![
            Just("alpha".to_string()),
            Just("beta".to_string()),
            Just("gamma".to_string()),
            Just("delta".to_string()),
        ], 1..50),
    ) {
        let mk = || DatasetPlan::from_dataframe(cat_df(values.clone()))
            .with_features(vec!["c".into()])
            .with_encoding("c".into(), EncodingSpec::Categorical {
                ordering: CategoryOrdering::FirstSeen,
            })
            .with_batch(BatchSpec::new(values.len().max(1)));

        let a: Vec<f64> = mk().iter_batches(Split::Full).unwrap()
            .flat_map(|b| b.unwrap().features.to_vec()).collect();
        let b: Vec<f64> = mk().iter_batches(Split::Full).unwrap()
            .flat_map(|b| b.unwrap().features.to_vec()).collect();

        // Bit-identical.
        prop_assert_eq!(a.len(), values.len());
        prop_assert_eq!(
            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn prop_drop_last_yields_full_batches_only(
        n in 1usize..200,
        batch_size in 1usize..=32,
    ) {
        let plan = DatasetPlan::from_dataframe(idx_df(n))
            .with_features(vec!["idx".into()])
            .with_encoding("idx".into(), EncodingSpec::IntAsFloat)
            .with_batch(BatchSpec::new(batch_size).with_drop_last(true));
        for batch in plan.iter_batches(Split::Full).unwrap() {
            let b = batch.unwrap();
            prop_assert_eq!(b.row_ids.len(), batch_size);
            prop_assert_eq!(b.features.shape(), &[batch_size, 1]);
        }
    }
}
