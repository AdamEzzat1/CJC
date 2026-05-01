//! Phase 1 — DatasetPlan determinism / parity gates.
//!
//! Phase 1 is a Rust-only library; `DatasetPlan` is not yet exposed to
//! `.cjcl` (that's Phase 3). True cross-executor parity (eval vs MIR
//! byte-equal materialized batches) therefore lives in the upstream
//! `DataFrame` + `Tensor` code paths and is already gated by the
//! existing tidy-test and physics_ml-parity suites.
//!
//! What we *can* lock in here at Phase 1 is the **stronger** invariant
//! that any DatasetPlan run is byte-deterministic regardless of:
//!
//!   1. Plan-instance identity         (re-construct → same bytes)
//!   2. `iter_batches` invocation count (idempotent re-iteration)
//!   3. Builder method order            (`with_features → with_label`
//!                                       vs. `with_label → with_features`)
//!   4. Source DataFrame instance       (clone → same bytes)
//!
//! Each gate compares full feature/label tensors via `f64::to_bits`,
//! never via `==`, so NaN-vs-NaN is a real assert and any reordering
//! that touches the low bit of a mantissa fails loudly.

use cjc_data::byte_dict::CategoryOrdering;
use cjc_data::{
    BatchSpec, Column, DataFrame, DatasetPlan, EncodingSpec, Split, SplitSpec,
};

/// Canonical mixed-type fixture: 16 rows × 4 columns, two of which
/// require non-trivial encoding (Int and categorical Str).
fn fixture_df() -> DataFrame {
    let n = 16;
    let xs: Vec<f64> = (0..n).map(|i| (i as f64) * 0.125).collect();
    let ys: Vec<i64> = (0..n).map(|i| (i as i64) * 7 - 3).collect();
    let cats: Vec<String> = (0..n)
        .map(|i| ["alpha", "beta", "gamma"][(i % 3) as usize].to_string())
        .collect();
    let labels: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
    DataFrame::from_columns(vec![
        ("x".into(), Column::Float(xs)),
        ("y".into(), Column::Int(ys)),
        ("cat".into(), Column::Str(cats)),
        ("label".into(), Column::Float(labels)),
    ])
    .unwrap()
}

fn collect_bits(plan: &DatasetPlan, which: Split) -> (Vec<u32>, Vec<u64>, Vec<u64>) {
    let mut row_ids = Vec::new();
    let mut feat_bits = Vec::new();
    let mut label_bits = Vec::new();
    for batch in plan.iter_batches(which).unwrap() {
        let b = batch.unwrap();
        row_ids.extend_from_slice(&b.row_ids);
        feat_bits.extend(b.features.to_vec().iter().map(|x| x.to_bits()));
        if let Some(l) = b.labels {
            label_bits.extend(l.to_vec().iter().map(|x| x.to_bits()));
        }
    }
    (row_ids, feat_bits, label_bits)
}

fn canonical_plan(df: DataFrame) -> DatasetPlan {
    DatasetPlan::from_dataframe(df)
        .with_features(vec!["x".into(), "y".into(), "cat".into()])
        .with_label("label".into())
        .with_encoding("x".into(), EncodingSpec::Float)
        .with_encoding("y".into(), EncodingSpec::IntAsFloat)
        .with_encoding(
            "cat".into(),
            EncodingSpec::Categorical {
                ordering: CategoryOrdering::FirstSeen,
            },
        )
        .with_encoding("label".into(), EncodingSpec::Float)
        .with_split(SplitSpec::Sequential {
            train: 0.5,
            val: 0.25,
            test: 0.25,
        })
        .with_batch(BatchSpec::new(4))
}

// ─── Gate 1: independent plan instances → identical bytes ───────────

#[test]
fn parity_independent_instances_yield_byte_equal_batches() {
    let p1 = canonical_plan(fixture_df());
    let p2 = canonical_plan(fixture_df());
    for which in [Split::Train, Split::Val, Split::Test] {
        let a = collect_bits(&p1, which);
        let b = collect_bits(&p2, which);
        assert_eq!(a, b, "split {which:?} differs across plan instances");
    }
}

// ─── Gate 2: re-iteration is idempotent ─────────────────────────────

#[test]
fn parity_repeated_iter_is_idempotent() {
    let plan = canonical_plan(fixture_df());
    let a = collect_bits(&plan, Split::Train);
    let b = collect_bits(&plan, Split::Train);
    let c = collect_bits(&plan, Split::Train);
    assert_eq!(a, b);
    assert_eq!(b, c);
}

// ─── Gate 3: builder method order does not affect output ────────────

#[test]
fn parity_builder_order_independent() {
    let df = fixture_df();
    // Order A: features → label → encodings → split → batch
    let order_a = DatasetPlan::from_dataframe(df.clone())
        .with_features(vec!["x".into(), "y".into(), "cat".into()])
        .with_label("label".into())
        .with_encoding("x".into(), EncodingSpec::Float)
        .with_encoding("y".into(), EncodingSpec::IntAsFloat)
        .with_encoding(
            "cat".into(),
            EncodingSpec::Categorical {
                ordering: CategoryOrdering::FirstSeen,
            },
        )
        .with_encoding("label".into(), EncodingSpec::Float)
        .with_split(SplitSpec::Sequential {
            train: 0.5,
            val: 0.25,
            test: 0.25,
        })
        .with_batch(BatchSpec::new(4));

    // Order B: split → batch → label → encodings (reverse) → features last
    let order_b = DatasetPlan::from_dataframe(df.clone())
        .with_split(SplitSpec::Sequential {
            train: 0.5,
            val: 0.25,
            test: 0.25,
        })
        .with_batch(BatchSpec::new(4))
        .with_label("label".into())
        .with_encoding("label".into(), EncodingSpec::Float)
        .with_encoding(
            "cat".into(),
            EncodingSpec::Categorical {
                ordering: CategoryOrdering::FirstSeen,
            },
        )
        .with_encoding("y".into(), EncodingSpec::IntAsFloat)
        .with_encoding("x".into(), EncodingSpec::Float)
        .with_features(vec!["x".into(), "y".into(), "cat".into()]);

    for which in [Split::Train, Split::Val, Split::Test] {
        let a = collect_bits(&order_a, which);
        let b = collect_bits(&order_b, which);
        assert_eq!(
            a, b,
            "builder method order changed output bytes for {which:?}"
        );
    }
}

// ─── Gate 4: cloning the source DataFrame does not perturb output ───

#[test]
fn parity_dataframe_clone_yields_byte_equal_batches() {
    let df = fixture_df();
    let p1 = canonical_plan(df.clone());
    let p2 = canonical_plan(df.clone());
    let a = collect_bits(&p1, Split::Full);
    let b = collect_bits(&p2, Split::Full);
    assert_eq!(a, b);
}

// ─── Shuffle determinism across plan instances ──────────────────────

#[test]
fn parity_shuffled_iteration_is_seed_deterministic_across_instances() {
    let mk = || {
        DatasetPlan::from_dataframe(fixture_df())
            .with_features(vec!["x".into()])
            .with_encoding("x".into(), EncodingSpec::Float)
            .with_batch(BatchSpec::new(3).with_shuffle(0xDEADBEEF))
    };
    let a = collect_bits(&mk(), Split::Full);
    let b = collect_bits(&mk(), Split::Full);
    assert_eq!(
        a, b,
        "same shuffle seed must produce byte-identical batches \
         across independent plan constructions"
    );
}

// ─── Hashed-split cross-instance determinism ────────────────────────

#[test]
fn parity_hashed_split_is_seed_deterministic_across_instances() {
    let mk = || {
        DatasetPlan::from_dataframe(fixture_df())
            .with_features(vec!["x".into()])
            .with_encoding("x".into(), EncodingSpec::Float)
            .with_split(SplitSpec::Hashed {
                seed: 0xCAFEBABE,
                train: 0.6,
                val: 0.2,
                test: 0.2,
            })
            .with_batch(BatchSpec::new(2))
    };
    let a = collect_bits(&mk(), Split::Train);
    let b = collect_bits(&mk(), Split::Train);
    assert_eq!(a, b);
}

// ─── Categorical dictionary stability across plan instances ─────────

#[test]
fn parity_categorical_codes_stable_across_instances() {
    // FirstSeen ordering means the codes depend on row iteration order.
    // Two independent plans seeing the same DataFrame must agree on the
    // u64 → string mapping, byte-for-byte in the materialized features.
    let mk = || {
        DatasetPlan::from_dataframe(fixture_df())
            .with_features(vec!["cat".into()])
            .with_encoding(
                "cat".into(),
                EncodingSpec::Categorical {
                    ordering: CategoryOrdering::FirstSeen,
                },
            )
            .with_batch(BatchSpec::new(16))
    };
    let a = collect_bits(&mk(), Split::Full);
    let b = collect_bits(&mk(), Split::Full);
    assert_eq!(a, b);
    // alpha=0, beta=1, gamma=2 by FirstSeen, repeating every 3 rows.
    let expected_pattern: Vec<u64> = (0..16)
        .map(|i| (i % 3) as f64)
        .map(|x: f64| x.to_bits())
        .collect();
    assert_eq!(a.1, expected_pattern);
}
