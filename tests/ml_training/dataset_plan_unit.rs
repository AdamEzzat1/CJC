//! Phase 1 — DatasetPlan unit tests.
//!
//! Covers: construction, validation errors, sequential / hashed / full
//! splits, batch iteration with and without `drop_last`, shuffle
//! determinism, all four `EncodingSpec` variants, and the
//! `MaterializedBatch` shape contract.

use cjc_data::byte_dict::CategoryOrdering;
use cjc_data::{
    BatchSpec, DatasetError, DatasetPlan, EncodingSpec, Split, SplitSpec,
};

use super::common::{idx_df, small_mixed_df};

// ── Construction & validation ───────────────────────────────────────

#[test]
fn validate_rejects_empty_features() {
    let plan = DatasetPlan::from_dataframe(small_mixed_df());
    assert!(matches!(plan.validate(), Err(DatasetError::NoFeatures)));
}

#[test]
fn validate_rejects_unknown_feature_column() {
    let plan = DatasetPlan::from_dataframe(small_mixed_df())
        .with_features(vec!["does_not_exist".into()]);
    match plan.validate() {
        Err(DatasetError::UnknownColumn(c)) => assert_eq!(c, "does_not_exist"),
        other => panic!("expected UnknownColumn, got {other:?}"),
    }
}

#[test]
fn validate_rejects_orphan_encoding() {
    let plan = DatasetPlan::from_dataframe(small_mixed_df())
        .with_features(vec!["x".into()])
        .with_encoding("y".into(), EncodingSpec::IntAsFloat); // y is neither feature nor label
    match plan.validate() {
        Err(DatasetError::OrphanEncoding(c)) => assert_eq!(c, "y"),
        other => panic!("expected OrphanEncoding, got {other:?}"),
    }
}

#[test]
fn validate_rejects_invalid_fractions() {
    let plan = DatasetPlan::from_dataframe(small_mixed_df())
        .with_features(vec!["x".into()])
        .with_split(SplitSpec::Sequential {
            train: 0.6,
            val: 0.5,
            test: 0.5,
        });
    assert!(matches!(
        plan.validate(),
        Err(DatasetError::InvalidFractions { .. })
    ));
}

#[test]
fn validate_rejects_zero_batch_size() {
    let plan = DatasetPlan::from_dataframe(small_mixed_df())
        .with_features(vec!["x".into()])
        .with_batch(BatchSpec {
            batch_size: 0,
            drop_last: false,
            shuffle: None,
        });
    assert!(matches!(plan.validate(), Err(DatasetError::BadBatchSize(0))));
}

// ── Splits ───────────────────────────────────────────────────────────

#[test]
fn split_full_returns_all_rows_ascending() {
    let plan = DatasetPlan::from_dataframe(small_mixed_df())
        .with_features(vec!["x".into()]);
    let rows = plan.split_rows(Split::Full).unwrap();
    assert_eq!(rows, (0u32..8).collect::<Vec<_>>());
    assert!(plan.split_rows(Split::Train).unwrap().is_empty());
}

#[test]
fn split_sequential_partitions_ascending_disjoint() {
    let n = 100usize;
    let plan = DatasetPlan::from_dataframe(idx_df(n))
        .with_features(vec!["idx".into()])
        .with_split(SplitSpec::Sequential {
            train: 0.7,
            val: 0.2,
            test: 0.1,
        });
    let train = plan.split_rows(Split::Train).unwrap();
    let val = plan.split_rows(Split::Val).unwrap();
    let test = plan.split_rows(Split::Test).unwrap();
    assert_eq!(train, (0u32..70).collect::<Vec<_>>());
    assert_eq!(val, (70u32..90).collect::<Vec<_>>());
    assert_eq!(test, (90u32..100).collect::<Vec<_>>());

    // Disjoint + exhaustive when fractions sum to 1.0.
    let mut all = train;
    all.extend(val);
    all.extend(test);
    assert_eq!(all, (0u32..100).collect::<Vec<_>>());
}

#[test]
fn split_sequential_with_unallocated_remainder_excludes_tail() {
    // 0.5 + 0.3 + 0.1 = 0.9 → 10% of rows excluded.
    let n = 100usize;
    let plan = DatasetPlan::from_dataframe(idx_df(n))
        .with_features(vec!["idx".into()])
        .with_split(SplitSpec::Sequential {
            train: 0.5,
            val: 0.3,
            test: 0.1,
        });
    let train = plan.split_rows(Split::Train).unwrap();
    let val = plan.split_rows(Split::Val).unwrap();
    let test = plan.split_rows(Split::Test).unwrap();
    assert_eq!(train.len() + val.len() + test.len(), 90);
    assert_eq!(train.last().copied(), Some(49));
    assert_eq!(val.first().copied(), Some(50));
    assert_eq!(test.last().copied(), Some(89));
}

#[test]
fn split_hashed_is_deterministic_same_seed() {
    let n = 1_000usize;
    let plan = |seed: u64| {
        DatasetPlan::from_dataframe(idx_df(n))
            .with_features(vec!["idx".into()])
            .with_split(SplitSpec::Hashed {
                seed,
                train: 0.7,
                val: 0.15,
                test: 0.15,
            })
    };
    let a = plan(42).split_rows(Split::Train).unwrap();
    let b = plan(42).split_rows(Split::Train).unwrap();
    assert_eq!(a, b, "same seed must produce identical hashed split");
}

#[test]
fn split_hashed_seeds_differ() {
    let n = 1_000usize;
    let plan = |seed: u64| {
        DatasetPlan::from_dataframe(idx_df(n))
            .with_features(vec!["idx".into()])
            .with_split(SplitSpec::Hashed {
                seed,
                train: 0.7,
                val: 0.15,
                test: 0.15,
            })
    };
    let a = plan(1).split_rows(Split::Train).unwrap();
    let b = plan(2).split_rows(Split::Train).unwrap();
    assert_ne!(a, b, "different seeds should produce different hashed splits");
}

#[test]
fn split_hashed_returns_ascending_within_split() {
    let plan = DatasetPlan::from_dataframe(idx_df(500))
        .with_features(vec!["idx".into()])
        .with_split(SplitSpec::Hashed {
            seed: 17,
            train: 0.6,
            val: 0.2,
            test: 0.2,
        });
    let train = plan.split_rows(Split::Train).unwrap();
    assert!(
        train.windows(2).all(|w| w[0] < w[1]),
        "hashed-split row IDs must be strictly ascending"
    );
}

// ── Batch iteration ──────────────────────────────────────────────────

#[test]
fn iter_batches_full_no_shuffle_yields_ascending_rows() {
    let plan = DatasetPlan::from_dataframe(small_mixed_df())
        .with_features(vec!["x".into(), "y".into()])
        .with_encoding("x".into(), EncodingSpec::Float)
        .with_encoding("y".into(), EncodingSpec::IntAsFloat)
        .with_batch(BatchSpec::new(3));
    let batches: Vec<_> = plan
        .iter_batches(Split::Full)
        .unwrap()
        .map(|b| b.unwrap())
        .collect();
    assert_eq!(batches.len(), 3); // 8 rows / batch_size 3 = 3 + 3 + 2
    assert_eq!(batches[0].row_ids, vec![0, 1, 2]);
    assert_eq!(batches[1].row_ids, vec![3, 4, 5]);
    assert_eq!(batches[2].row_ids, vec![6, 7]);
}

#[test]
fn iter_batches_drop_last_drops_short_tail() {
    let plan = DatasetPlan::from_dataframe(small_mixed_df())
        .with_features(vec!["x".into()])
        .with_encoding("x".into(), EncodingSpec::Float)
        .with_batch(BatchSpec::new(3).with_drop_last(true));
    let batches: Vec<_> = plan
        .iter_batches(Split::Full)
        .unwrap()
        .map(|b| b.unwrap())
        .collect();
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].row_ids, vec![0, 1, 2]);
    assert_eq!(batches[1].row_ids, vec![3, 4, 5]);
}

#[test]
fn iter_batches_shuffle_same_seed_is_deterministic() {
    let mk = || {
        DatasetPlan::from_dataframe(small_mixed_df())
            .with_features(vec!["x".into()])
            .with_encoding("x".into(), EncodingSpec::Float)
            .with_batch(BatchSpec::new(2).with_shuffle(7))
    };
    let a: Vec<u32> = mk()
        .iter_batches(Split::Full)
        .unwrap()
        .flat_map(|b| b.unwrap().row_ids)
        .collect();
    let b: Vec<u32> = mk()
        .iter_batches(Split::Full)
        .unwrap()
        .flat_map(|b| b.unwrap().row_ids)
        .collect();
    assert_eq!(a, b, "shuffle with same seed must be deterministic");
    // Permutation must visit every row exactly once.
    let mut sorted = a.clone();
    sorted.sort();
    assert_eq!(sorted, (0u32..8).collect::<Vec<_>>());
}

#[test]
fn iter_batches_shuffle_different_seed_differs() {
    let mk = |seed| {
        DatasetPlan::from_dataframe(idx_df(50))
            .with_features(vec!["idx".into()])
            .with_encoding("idx".into(), EncodingSpec::IntAsFloat)
            .with_batch(BatchSpec::new(5).with_shuffle(seed))
    };
    let a: Vec<u32> = mk(1)
        .iter_batches(Split::Full)
        .unwrap()
        .flat_map(|b| b.unwrap().row_ids)
        .collect();
    let b: Vec<u32> = mk(2)
        .iter_batches(Split::Full)
        .unwrap()
        .flat_map(|b| b.unwrap().row_ids)
        .collect();
    assert_ne!(a, b);
}

// ── Tensor materialization ───────────────────────────────────────────

#[test]
fn materialize_features_shape_and_values() {
    let plan = DatasetPlan::from_dataframe(small_mixed_df())
        .with_features(vec!["x".into(), "y".into()])
        .with_label("label".into())
        .with_encoding("x".into(), EncodingSpec::Float)
        .with_encoding("y".into(), EncodingSpec::IntAsFloat)
        .with_encoding("label".into(), EncodingSpec::Float)
        .with_batch(BatchSpec::new(4));
    let mut iter = plan.iter_batches(Split::Full).unwrap();
    let batch = iter.next().unwrap().unwrap();
    assert_eq!(batch.features.shape(), &[4, 2]);
    assert_eq!(
        batch.features.to_vec(),
        vec![0.0, 0.0, 1.0, 10.0, 2.0, 20.0, 3.0, 30.0]
    );
    let labels = batch.labels.unwrap();
    assert_eq!(labels.shape(), &[4]);
    assert_eq!(labels.to_vec(), vec![0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn materialize_no_label_yields_none() {
    let plan = DatasetPlan::from_dataframe(small_mixed_df())
        .with_features(vec!["x".into()])
        .with_encoding("x".into(), EncodingSpec::Float)
        .with_batch(BatchSpec::new(8));
    let batch = plan
        .iter_batches(Split::Full)
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    assert!(batch.labels.is_none());
}

#[test]
fn materialize_categorical_encoding_emits_codes() {
    // FirstSeen: order of first appearance in source rows: a, b, c.
    let plan = DatasetPlan::from_dataframe(small_mixed_df())
        .with_features(vec!["cat".into()])
        .with_encoding(
            "cat".into(),
            EncodingSpec::Categorical {
                ordering: CategoryOrdering::FirstSeen,
            },
        )
        .with_batch(BatchSpec::new(8));
    let batch = plan
        .iter_batches(Split::Full)
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    // cats = [a, b, a, c, b, a, c, b] ; FirstSeen → a=0, b=1, c=2
    assert_eq!(
        batch.features.to_vec(),
        vec![0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0]
    );
}

#[test]
fn materialize_bool_encoding() {
    let plan = DatasetPlan::from_dataframe(idx_df(4))
        .with_features(vec!["even".into()])
        .with_encoding("even".into(), EncodingSpec::BoolAsFloat)
        .with_batch(BatchSpec::new(4));
    let batch = plan
        .iter_batches(Split::Full)
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    assert_eq!(batch.features.to_vec(), vec![1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn encoding_mismatch_returns_error() {
    // Float-encoding on an Int column.
    let plan = DatasetPlan::from_dataframe(small_mixed_df())
        .with_features(vec!["y".into()])
        .with_encoding("y".into(), EncodingSpec::Float)
        .with_batch(BatchSpec::new(2));
    let mut iter = plan.iter_batches(Split::Full).unwrap();
    let first = iter.next().unwrap();
    match first {
        Err(DatasetError::EncodingMismatch { column, .. }) => assert_eq!(column, "y"),
        other => panic!("expected EncodingMismatch, got {other:?}"),
    }
}

#[test]
fn plan_hash_is_none_in_phase_1() {
    let plan = DatasetPlan::from_dataframe(small_mixed_df())
        .with_features(vec!["x".into()]);
    assert!(plan.plan_hash().is_none(),
        "plan_hash field is reserved for Phase 6");
}

#[test]
fn split_then_iter_batches_visits_split_rows_only() {
    let plan = DatasetPlan::from_dataframe(idx_df(20))
        .with_features(vec!["idx".into()])
        .with_encoding("idx".into(), EncodingSpec::IntAsFloat)
        .with_split(SplitSpec::Sequential {
            train: 0.5,
            val: 0.25,
            test: 0.25,
        })
        .with_batch(BatchSpec::new(3));
    let train_rows: Vec<u32> = plan
        .iter_batches(Split::Train)
        .unwrap()
        .flat_map(|b| b.unwrap().row_ids)
        .collect();
    assert_eq!(train_rows, (0u32..10).collect::<Vec<_>>());

    let val_rows: Vec<u32> = plan
        .iter_batches(Split::Val)
        .unwrap()
        .flat_map(|b| b.unwrap().row_ids)
        .collect();
    assert_eq!(val_rows, (10u32..15).collect::<Vec<_>>());
}
