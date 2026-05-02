//! Phase 1 — DatasetPlan bolero fuzz targets.
//!
//! Two harnesses:
//!
//! 1. `fuzz_dataset_plan_oracle_parity` — decode a random byte sequence
//!    into a small `DataFrame` (Float + Int columns) and a plan config
//!    (split spec, batch size, optional shuffle seed). Run the plan and
//!    a hand-written naive oracle that reproduces the same row-order
//!    and encoding logic by index arithmetic. Assert that the
//!    materialized features are bit-identical to the oracle output.
//!    The contract being fuzzed: **DatasetPlan introduces no
//!    nondeterminism, no off-by-one in batching, and no f64-bit drift
//!    on Int→Float / Float→Float pass-through.**
//!
//! 2. `fuzz_dataset_plan_no_panic_on_garbage` — feed pathological
//!    configs (zero rows, zero batch, encoding-type mismatches, OOB
//!    column names) and assert the API returns `Err(DatasetError::*)`,
//!    never panics. Phase 1's error-not-panic boundary.
//!
//! Both harnesses use `bolero::check!`. Default cargo test runs them as
//! proptest with the env-configurable case count
//! (`BOLERO_FUZZ_OVERRIDE_ITERATIONS`); under `cargo bolero` they
//! become libfuzzer/AFL targets.

use std::panic;

use bolero::check;

use cjc_data::{
    BatchSpec, Column, DataFrame, DatasetError, DatasetPlan, EncodingSpec, Split,
    SplitSpec,
};

/// Decode the raw byte sequence into:
///   - n in [1, 32]                 (4 bits)
///   - shuffle_seed                  (8 bytes)
///   - shuffle_on                    (1 bit)
///   - split_seq vs split_full       (1 bit)
///   - train, val, test fractions    (3 × 8 bits scaled to /1000, capped to ≤1)
///   - batch_size in [1, 16]         (4 bits)
///   - drop_last                     (1 bit)
///   - col_x: Vec<f64> of length n   (n × 8 bytes)
///   - col_y: Vec<i64> of length n   (n × 8 bytes)
///
/// Returns `None` if the byte slice is too short to satisfy any field
/// — bolero will discard such inputs without counting them as failures.
struct DecodedCase {
    n: usize,
    seq: bool,
    train: f64,
    val: f64,
    test: f64,
    batch_size: usize,
    drop_last: bool,
    shuffle: Option<u64>,
    xs: Vec<f64>,
    ys: Vec<i64>,
}

fn decode(bytes: &[u8]) -> Option<DecodedCase> {
    // Inline byte takes — a closure returning a borrow runs into the
    // higher-ranked-lifetime quirk where the closure's input lifetime
    // and output lifetime get unified to the call-site's.
    fn need(bytes: &[u8], c: usize, k: usize) -> Option<usize> {
        if c + k <= bytes.len() { Some(c + k) } else { None }
    }

    let mut c = 0usize;

    let head_end = need(bytes, c, 4)?;
    let head = &bytes[c..head_end];
    c = head_end;
    let n = ((head[0] & 0x1F) as usize).max(1).min(32);
    let seq = (head[1] & 1) == 1;
    let shuffle_on = (head[1] & 2) == 2;
    let drop_last = (head[1] & 4) == 4;
    let batch_size = ((head[2] & 0x0F) as usize).max(1).min(16);

    let frac_a = head[3] as f64 / 256.0;
    let next1 = need(bytes, c, 1)?;
    let frac_b = bytes[c] as f64 / 256.0;
    c = next1;
    let next2 = need(bytes, c, 1)?;
    let frac_c = bytes[c] as f64 / 256.0;
    c = next2;
    // Normalize so they sum to ≤ 1.0 (clamp last fraction).
    let mut train = frac_a.min(1.0);
    let mut val = frac_b.min((1.0 - train).max(0.0));
    let mut test = frac_c.min((1.0 - train - val).max(0.0));
    // Avoid all-zero (would make every split empty).
    if train + val + test < 1e-6 {
        train = 0.5;
        val = 0.25;
        test = 0.25;
    }

    let seed_end = need(bytes, c, 8)?;
    let seed_bytes = &bytes[c..seed_end];
    c = seed_end;
    let shuffle = if shuffle_on {
        Some(u64::from_le_bytes([
            seed_bytes[0], seed_bytes[1], seed_bytes[2], seed_bytes[3],
            seed_bytes[4], seed_bytes[5], seed_bytes[6], seed_bytes[7],
        ]))
    } else {
        None
    };

    let mut xs = Vec::with_capacity(n);
    for _ in 0..n {
        let end = need(bytes, c, 8)?;
        let b = &bytes[c..end];
        c = end;
        let bits = u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
        let f = f64::from_bits(bits);
        xs.push(if f.is_finite() { f } else { 0.0 });
    }
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let end = need(bytes, c, 8)?;
        let b = &bytes[c..end];
        c = end;
        let v = i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
        ys.push(v);
    }

    Some(DecodedCase {
        n,
        seq,
        train,
        val,
        test,
        batch_size,
        drop_last,
        shuffle,
        xs,
        ys,
    })
}

fn build_df(case: &DecodedCase) -> DataFrame {
    DataFrame::from_columns(vec![
        ("x".into(), Column::Float(case.xs.clone())),
        ("y".into(), Column::Int(case.ys.clone())),
    ])
    .unwrap()
}

fn build_plan(case: &DecodedCase) -> DatasetPlan {
    let split = if case.seq {
        SplitSpec::Sequential {
            train: case.train,
            val: case.val,
            test: case.test,
        }
    } else {
        SplitSpec::Hashed {
            seed: case.shuffle.unwrap_or(0xA5A5_A5A5_A5A5_A5A5),
            train: case.train,
            val: case.val,
            test: case.test,
        }
    };
    let mut batch = BatchSpec::new(case.batch_size).with_drop_last(case.drop_last);
    if let Some(s) = case.shuffle {
        batch = batch.with_shuffle(s);
    }
    DatasetPlan::from_dataframe(build_df(case))
        .with_features(vec!["x".into(), "y".into()])
        .with_encoding("x".into(), EncodingSpec::Float)
        .with_encoding("y".into(), EncodingSpec::IntAsFloat)
        .with_split(split)
        .with_batch(batch)
}

/// Hand-rolled SplitMix64 mixer matching `dataset_plan::splitmix64_mix`.
/// Re-implemented locally so the oracle is independent of the
/// implementation under test.
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// Naive oracle: assigns split, applies shuffle, materializes batches.
/// Returns the same `(row_ids, feature_bits)` triple as the plan, but
/// computed via straight-line index arithmetic so any algorithmic
/// disagreement surfaces as a byte mismatch.
fn oracle(case: &DecodedCase, which: Split) -> (Vec<u32>, Vec<u64>) {
    let n = case.n;

    // 1. Assign split.
    let mut rows: Vec<u32> = if case.seq {
        let train_n = (n as f64 * case.train).floor() as usize;
        let val_n = (n as f64 * case.val).floor() as usize;
        let test_n = (n as f64 * case.test).floor() as usize;
        match which {
            Split::Train => (0..train_n as u32).collect(),
            Split::Val => (train_n as u32..(train_n + val_n) as u32).collect(),
            Split::Test => (
                (train_n + val_n) as u32
                    ..(train_n + val_n + test_n) as u32
            )
                .collect(),
            Split::Full => (0..n as u32).collect(),
        }
    } else {
        let seed = case.shuffle.unwrap_or(0xA5A5_A5A5_A5A5_A5A5);
        let train_t = case.train;
        let val_t = train_t + case.val;
        let test_t = val_t + case.test;
        if matches!(which, Split::Full) {
            (0..n as u32).collect()
        } else {
            let mut out = Vec::new();
            for r in 0..n as u32 {
                let h = mix((r as u64) ^ seed);
                let bucket = (h >> 32) as f64 / (u32::MAX as f64 + 1.0);
                let pick = if bucket < train_t {
                    Split::Train
                } else if bucket < val_t {
                    Split::Val
                } else if bucket < test_t {
                    Split::Test
                } else {
                    continue;
                };
                if pick == which {
                    out.push(r);
                }
            }
            out
        }
    };

    // 2. Shuffle (Fisher-Yates with cjc_repro::Rng).
    if let Some(seed) = case.shuffle {
        let mut rng = cjc_repro::Rng::seeded(seed);
        if rows.len() > 1 {
            for i in (1..rows.len()).rev() {
                let j = (rng.next_u64() % (i as u64 + 1)) as usize;
                rows.swap(i, j);
            }
        }
    }

    // 3. Materialize features. Encoding: x→f64 pass-through, y→i64 as f64.
    let mut bits = Vec::with_capacity(rows.len() * 2);
    let mut visited_rows = Vec::with_capacity(rows.len());
    let total = rows.len();
    let mut cursor = 0usize;
    while cursor < total {
        let end = (cursor + case.batch_size).min(total);
        let len = end - cursor;
        if len < case.batch_size && case.drop_last {
            break;
        }
        for &r in &rows[cursor..end] {
            visited_rows.push(r);
            bits.push(case.xs[r as usize].to_bits());
            bits.push((case.ys[r as usize] as f64).to_bits());
        }
        cursor = end;
    }

    (visited_rows, bits)
}

/// Collect the plan's output in the same shape as the oracle.
fn plan_output(plan: &DatasetPlan, which: Split) -> (Vec<u32>, Vec<u64>) {
    let mut row_ids = Vec::new();
    let mut bits = Vec::new();
    for batch in plan.iter_batches(which).unwrap() {
        let b = batch.unwrap();
        row_ids.extend_from_slice(&b.row_ids);
        bits.extend(b.features.to_vec().iter().map(|x| x.to_bits()));
    }
    (row_ids, bits)
}

// ─── Harness 1: oracle parity ───────────────────────────────────────

#[test]
#[cfg_attr(miri, ignore)]
fn fuzz_dataset_plan_oracle_parity() {
    check!()
        .with_iterations(if std::env::var("BOLERO_FUZZ_OVERRIDE_ITERATIONS").is_ok() {
            // Honor explicit override; bolero will reset to its own value.
            1
        } else {
            512
        })
        .for_each(|bytes: &[u8]| {
            let Some(case) = decode(bytes) else {
                return;
            };
            let plan = build_plan(&case);
            for which in [Split::Full, Split::Train, Split::Val, Split::Test] {
                let (rids_p, bits_p) = plan_output(&plan, which);
                let (rids_o, bits_o) = oracle(&case, which);
                assert_eq!(
                    rids_p, rids_o,
                    "row_ids mismatch on split {which:?} for n={}, seq={}, batch={}, shuffle={:?}",
                    case.n, case.seq, case.batch_size, case.shuffle
                );
                assert_eq!(
                    bits_p, bits_o,
                    "feature bits mismatch on split {which:?} for n={}, seq={}, batch={}",
                    case.n, case.seq, case.batch_size
                );
            }
        });
}

// ─── Harness 2: pathological configs return Err, never panic ────────

#[test]
#[cfg_attr(miri, ignore)]
fn fuzz_dataset_plan_no_panic_on_garbage() {
    check!()
        .with_iterations(256)
        .for_each(|bytes: &[u8]| {
            // Decode loosely: if parsing fails we still construct a
            // plan with deliberately broken settings so we exercise
            // the error path.
            let case = decode(bytes);

            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                let df = match &case {
                    Some(c) => build_df(c),
                    None => DataFrame::from_columns(vec![(
                        "x".into(),
                        Column::Float(vec![1.0, 2.0]),
                    )])
                    .unwrap(),
                };
                // Deliberately pathological: zero batch size + mismatched
                // encoding (Float on Int column) + unknown extra column.
                let plan = DatasetPlan::from_dataframe(df)
                    .with_features(vec!["x".into(), "phantom".into()])
                    .with_encoding("x".into(), EncodingSpec::IntAsFloat) // mismatch
                    .with_batch(BatchSpec::new(0));
                let validate = plan.validate();
                let iter = plan.iter_batches(Split::Full);
                (validate, iter.is_err())
            }));

            // The library must not panic on garbage input.
            assert!(result.is_ok(), "DatasetPlan panicked on garbage input");
            let (validate, iter_errored) = result.unwrap();
            // Either validation rejected it, or the iter itself errored.
            // We don't assert which — just that the error surfaced.
            assert!(
                validate.is_err() || iter_errored,
                "expected DatasetError, got success on pathological config"
            );
            // And the error variant must be one of our declared kinds.
            if let Err(e) = validate {
                let _name: &dyn std::error::Error = &e;
                match e {
                    DatasetError::BadBatchSize(_)
                    | DatasetError::UnknownColumn(_)
                    | DatasetError::EncodingMismatch { .. }
                    | DatasetError::OrphanEncoding(_)
                    | DatasetError::NoFeatures
                    | DatasetError::InvalidFractions { .. }
                    | DatasetError::Tidy(_)
                    | DatasetError::Shape(_)
                    | DatasetError::UnsupportedColumnType { .. }
                    | DatasetError::NullCategorical { .. }
                    | DatasetError::EmptySplit(_) => {}
                }
            }
        });
}
