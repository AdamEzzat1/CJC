//! Phase 0.9.5 COMMIT 3 — proptest property suite for the deterministic
//! categorical subsystem ([`cjc_abng::categorical`]).
//!
//! COMMIT 1 + 2 shipped fixed-case unit tests; this module generalises
//! them into the determinism contracts of `docs/abng/PHASE_0_9_5_HANDOFF.md`
//! §3.7. The four §3.7-named contracts:
//!
//! * **row-order invariance** — shuffling the training rows cannot change
//!   the vocabulary, its canonical bytes, or its hash.
//! * **train-only** — a frozen dictionary is immune to transform-time
//!   activity: arbitrary `encode` calls (the inference / test-split
//!   traffic) never mutate it. This is the foundation of the leakage
//!   guard (handoff directive 6); COMMIT 4 extends it to the real split.
//! * **unknown safety** — `encode` is total: any string, however
//!   pathological, lands on a valid code, and a never-trained category
//!   yields `CODE_UNKNOWN`.
//! * **rare determinism** — the rare-fold set is a pure function of the
//!   training counts and the `RarePolicy`.
//!
//! The remaining properties extend coverage to the COMMIT 2 encoding
//! surface (`OneHotEncoder`, `route_bucket`, `SchemaSnapshot`), whose
//! width / slot / bucket bounds are determinism-critical safety
//! invariants. Each property runs 256 generated cases.

use std::collections::BTreeMap;

use proptest::prelude::*;

use cjc_abng::categorical::{
    route_bucket, CategoricalTransform, CategoryDictionary, CategoryDictionaryBuilder, ColumnRole,
    OneHotEncoder, RarePolicy, Schema, SchemaSnapshot, TransformConfig, CODE_MISSING, CODE_RARE,
    CODE_UNKNOWN, FIRST_REAL_CODE,
};

/// SplitMix64 — the same mixer as `cjc_repro::Rng`, used here to drive a
/// deterministic shuffle from a single `u64` seed, so the row-order
/// property does not depend on proptest's own shuffle strategy.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Deterministic Fisher–Yates shuffle of `items` under `seed`.
fn shuffle<T: Clone>(items: &[T], seed: u64) -> Vec<T> {
    let mut out = items.to_vec();
    let mut state = seed;
    for i in (1..out.len()).rev() {
        let j = (splitmix64(&mut state) % (i as u64 + 1)) as usize;
        out.swap(i, j);
    }
    out
}

/// Build a frozen dictionary from a list of training cell values.
fn build(values: &[String], markers: &[&str], policy: RarePolicy) -> CategoryDictionary {
    let mut b = CategoryDictionaryBuilder::new(markers);
    b.observe_all(values.iter());
    b.build(policy)
}

/// Map generated category ids to stable label strings.
fn labels(ids: &[u8]) -> Vec<String> {
    ids.iter().map(|b| format!("cat{b}")).collect()
}

/// Fixed schema for the COMMIT 4 `CategoricalTransform` properties — one
/// column of each fittable role.
fn tf_schema() -> Schema {
    Schema::new(vec![
        ("cat".to_string(), ColumnRole::Categorical),
        ("num".to_string(), ColumnRole::Numeric),
        ("lab".to_string(), ColumnRole::Target),
    ])
}

/// Transform config with `KEEP_ALL` folding so the small generated
/// fixtures do not collapse entirely into the RARE slot.
fn tf_config() -> TransformConfig {
    TransformConfig {
        rare_policy: RarePolicy::KEEP_ALL,
        target_positives: vec!["yes".to_string()],
        ..TransformConfig::default()
    }
}

/// Map generated `(cat_id, num_value, positive)` triples to raw string
/// rows matching [`tf_schema`].
fn tf_rows(spec: &[(u8, u8, bool)]) -> Vec<Vec<String>> {
    spec.iter()
        .map(|&(c, n, pos)| {
            vec![
                format!("cat{c}"),
                format!("{n}"),
                if pos { "yes".to_string() } else { "no".to_string() },
            ]
        })
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    // §3.7 row-order invariance — the vocabulary is frequency+lexically
    // sorted, so any permutation of the training rows yields a
    // byte-identical dictionary.
    #[test]
    fn prop_row_order_invariance(
        ids in prop::collection::vec(0u8..24, 0..256),
        shuffle_seed in any::<u64>(),
    ) {
        let cats = labels(&ids);
        let shuffled = shuffle(&cats, shuffle_seed);
        let a = build(&cats, &["?"], RarePolicy::KEEP_ALL);
        let b = build(&shuffled, &["?"], RarePolicy::KEEP_ALL);
        prop_assert_eq!(a.canonical_bytes(), b.canonical_bytes());
        prop_assert_eq!(a.vocab_hash(), b.vocab_hash());
        prop_assert_eq!(a.n_real(), b.n_real());
        prop_assert_eq!(a.total_observed(), b.total_observed());
    }

    // §3.7 train-only — a frozen dictionary is a pure function of the
    // training split. Arbitrary `encode` traffic (inference / the test
    // split) cannot mutate the vocabulary or its hash. `encode` takes
    // `&self`, so this is compiler-guaranteed; the property pins the
    // *observable* contract that a schema replay depends on.
    #[test]
    fn prop_frozen_vocabulary_immune_to_encoding(
        ids in prop::collection::vec(0u8..24, 1..256),
        probes in prop::collection::vec(0u8..64, 0..128),
    ) {
        let dict = build(&labels(&ids), &["?"], RarePolicy::DEFAULT);
        let before_bytes = dict.canonical_bytes();
        let before_hash = dict.vocab_hash();
        for p in &probes {
            // Alphabet 0..64 vs trained 0..24 -> many never-seen probes.
            let _ = dict.encode(&format!("cat{p}"));
            let _ = dict.encode("?");
            let _ = dict.encode("");
        }
        prop_assert_eq!(dict.canonical_bytes(), before_bytes);
        prop_assert_eq!(dict.vocab_hash(), before_hash);
    }

    // §3.7 unknown safety — `encode` is total. An arbitrary probe string
    // never panics and always lands on a valid code; a probe that was
    // never trained (and is not a missing-marker) is `CODE_UNKNOWN`.
    #[test]
    fn prop_unknown_category_safety(
        ids in prop::collection::vec(0u8..16, 0..128),
        probe in ".*",
    ) {
        let cats = labels(&ids);
        let dict = build(&cats, &["?"], RarePolicy::KEEP_ALL);
        let code = dict.encode(&probe);
        // Total: the code is always within the vocabulary's code space.
        prop_assert!(code < dict.vocab_size());
        if probe == "?" {
            prop_assert_eq!(code, CODE_MISSING);
        } else if !cats.iter().any(|c| c == &probe) {
            prop_assert_eq!(code, CODE_UNKNOWN);
        }
    }

    // §3.7 rare determinism — the rare-fold set is a pure function of the
    // training counts and the policy; row order cannot change it.
    #[test]
    fn prop_rare_fold_determinism(
        ids in prop::collection::vec(0u8..40, 0..400),
        shuffle_seed in any::<u64>(),
        min_count in 0u64..50,
        min_frac in 0.0f64..0.2,
    ) {
        let policy = RarePolicy { min_count, min_frac };
        let cats = labels(&ids);
        let shuffled = shuffle(&cats, shuffle_seed);
        let a = build(&cats, &[], policy);
        let b = build(&shuffled, &[], policy);
        prop_assert_eq!(a.n_rare_folded(), b.n_rare_folded());
        prop_assert_eq!(a.rare_count(), b.rare_count());
        prop_assert_eq!(a.canonical_bytes(), b.canonical_bytes());
    }

    // Rare-fold rule (handoff §3.4) — a training category folds iff it
    // fails *either* floor: `count < min_count` or
    // `count < min_frac * total`. Re-derived independently of the impl.
    #[test]
    fn prop_rare_fold_matches_either_floor_rule(
        ids in prop::collection::vec(0u8..40, 1..400),
        min_count in 0u64..60,
        min_frac in 0.0f64..0.15,
    ) {
        let policy = RarePolicy { min_count, min_frac };
        let cats = labels(&ids);
        let dict = build(&cats, &[], policy);
        let total = dict.total_observed();
        let frac_floor = min_frac * (total as f64);
        let mut counts: BTreeMap<&str, u64> = BTreeMap::new();
        for c in &cats {
            *counts.entry(c.as_str()).or_insert(0) += 1;
        }
        for (label, count) in counts {
            let folded = dict.encode(label) == CODE_RARE;
            let expect = count < min_count || (count as f64) < frac_floor;
            prop_assert_eq!(
                folded, expect,
                "label {:?}: count {}, total {}, policy {:?}",
                label, count, total, policy
            );
        }
    }

    // COMMIT 2 surface — `OneHotEncoder::slot` is always a valid index
    // into a `width`-long one-hot vector, for every cap and every raw
    // input (the predictive-side bounds guard).
    #[test]
    fn prop_one_hot_slot_within_width(
        ids in prop::collection::vec(0u8..32, 0..256),
        cap in prop::option::of(0u32..40),
        probe in ".*",
    ) {
        let cats = labels(&ids);
        let dict = build(&cats, &["?"], RarePolicy::KEEP_ALL);
        let enc = match cap {
            Some(m) => OneHotEncoder::with_max_real(m),
            None => OneHotEncoder::new(),
        };
        let width = enc.width(&dict);
        prop_assert!(width >= FIRST_REAL_CODE as usize);
        let mut raws: Vec<&str> = cats.iter().map(String::as_str).collect();
        raws.push("?");
        raws.push("");
        raws.push(probe.as_str());
        for &raw in &raws {
            prop_assert!(enc.slot(&dict, raw) < width, "slot OOB for {raw:?}");
        }
    }

    // COMMIT 2 surface — `encode_into` produces a genuine one-hot vector:
    // length `width`, exactly one `1.0`, that `1.0` at `slot`. The caller
    // buffer starts garbage of the wrong length to exercise the resize.
    #[test]
    fn prop_one_hot_encode_into_is_one_hot(
        ids in prop::collection::vec(0u8..32, 1..200),
        cap in prop::option::of(0u32..40),
        probe in ".*",
    ) {
        let cats = labels(&ids);
        let dict = build(&cats, &["?"], RarePolicy::KEEP_ALL);
        let enc = match cap {
            Some(m) => OneHotEncoder::with_max_real(m),
            None => OneHotEncoder::new(),
        };
        let mut buf = vec![7.0; 3];
        for raw in [cats[0].as_str(), "?", probe.as_str()] {
            enc.encode_into(&dict, raw, &mut buf);
            prop_assert_eq!(buf.len(), enc.width(&dict));
            prop_assert_eq!(buf.iter().sum::<f64>(), 1.0);
            prop_assert_eq!(buf.iter().filter(|&&x| x == 1.0).count(), 1);
            prop_assert_eq!(buf[enc.slot(&dict, raw)], 1.0);
        }
    }

    // COMMIT 2 surface — `route_bucket` clamps any code into
    // `0..route_bins` (the routing-side explosion guard); below the last
    // bucket it is identity, at/above it saturates to the last bucket.
    #[test]
    fn prop_route_bucket_bounded(code in any::<u32>(), bins in 0u8..=64) {
        let bucket = route_bucket(code, bins);
        if bins == 0 {
            // Degenerate config — must clamp to 0, never panic.
            prop_assert_eq!(bucket, 0);
        } else {
            prop_assert!((bucket as u32) < bins as u32);
            let last = bins as u32 - 1;
            prop_assert_eq!(bucket as u32, code.min(last));
        }
    }

    // COMMIT 2 surface — `SchemaSnapshot::snapshot_hash` is deterministic
    // (same fields -> same hash) and sensitive (any single-field change
    // -> a different hash).
    #[test]
    fn prop_schema_snapshot_determinism_and_sensitivity(
        raw in prop::array::uniform32(any::<u8>()),
        schema in prop::array::uniform32(any::<u8>()),
        vocab in prop::array::uniform32(any::<u8>()),
        numeric in prop::array::uniform32(any::<u8>()),
        version in any::<u32>(),
        seed in any::<u64>(),
        rows in any::<u64>(),
        target in ".*",
    ) {
        let base = SchemaSnapshot {
            raw_dataset_hash: raw,
            schema_hash: schema,
            categorical_vocab_hash: vocab,
            numeric_standardization_hash: numeric,
            feature_transform_version: version,
            split_seed: seed,
            row_count: rows,
            target_definition: target,
        };
        prop_assert_eq!(base.snapshot_hash(), base.clone().snapshot_hash());
        let h = base.snapshot_hash();

        let mut v = base.clone();
        v.raw_dataset_hash[0] ^= 1;
        prop_assert_ne!(v.snapshot_hash(), h);
        v = base.clone();
        v.schema_hash[0] ^= 1;
        prop_assert_ne!(v.snapshot_hash(), h);
        v = base.clone();
        v.categorical_vocab_hash[0] ^= 1;
        prop_assert_ne!(v.snapshot_hash(), h);
        v = base.clone();
        v.numeric_standardization_hash[0] ^= 1;
        prop_assert_ne!(v.snapshot_hash(), h);
        v = base.clone();
        v.feature_transform_version = version.wrapping_add(1);
        prop_assert_ne!(v.snapshot_hash(), h);
        v = base.clone();
        v.split_seed = seed.wrapping_add(1);
        prop_assert_ne!(v.snapshot_hash(), h);
        v = base.clone();
        v.row_count = rows.wrapping_add(1);
        prop_assert_ne!(v.snapshot_hash(), h);
        v = base.clone();
        v.target_definition.push('\u{1}');
        prop_assert_ne!(v.snapshot_hash(), h);
    }

    // COMMIT 4 — `CategoricalTransform::fit` is invariant to training-row
    // order: a permutation of the train rows yields a byte-identical
    // snapshot, routing selection, and per-row transform output. This is
    // the directive-1 determinism contract over the whole transform.
    #[test]
    fn prop_transform_fit_is_row_order_invariant(
        spec in prop::collection::vec((0u8..6, 0u8..20, any::<bool>()), 1..200),
        shuffle_seed in any::<u64>(),
    ) {
        let schema = tf_schema();
        let config = tf_config();
        let forward = tf_rows(&spec);
        let shuffled = shuffle(&forward, shuffle_seed);
        let a = CategoricalTransform::fit(&schema, &forward, &config).unwrap();
        let b = CategoricalTransform::fit(&schema, &shuffled, &config).unwrap();
        prop_assert_eq!(a.snapshot().snapshot_hash(), b.snapshot().snapshot_hash());
        prop_assert_eq!(a.routing_feature_columns(), b.routing_feature_columns());
        prop_assert_eq!(a.phi_width(), b.phi_width());
        for r in &forward {
            prop_assert_eq!(a.transform(r).unwrap(), b.transform(r).unwrap());
        }
    }

    // COMMIT 4 — every transformed row is well-formed: `x` holds one
    // valid route bucket per routing feature, `phi` has the fixed
    // `phi_width`, and `y` is a `0.0` / `1.0` label.
    #[test]
    fn prop_transform_outputs_well_formed(
        spec in prop::collection::vec((0u8..6, 0u8..20, any::<bool>()), 1..200),
    ) {
        let t = CategoricalTransform::fit(&tf_schema(), &tf_rows(&spec), &tf_config()).unwrap();
        let route_bins = t.route_bins() as f64;
        for r in &tf_rows(&spec) {
            let (x, phi, y) = t.transform(r).unwrap();
            prop_assert_eq!(x.len(), t.n_routing_features());
            for &xi in &x {
                prop_assert!(xi >= 0.0 && xi < route_bins);
            }
            prop_assert_eq!(phi.len(), t.phi_width());
            prop_assert!(y == 0.0 || y == 1.0);
        }
    }
}
