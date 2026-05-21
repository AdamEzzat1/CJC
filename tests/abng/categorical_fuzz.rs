//! Phase 0.9.5 COMMIT 3 — bolero fuzz targets for the deterministic
//! categorical subsystem ([`cjc_abng::categorical`]).
//!
//! `docs/abng/PHASE_0_9_5_HANDOFF.md` §7 names four fuzz angles for
//! COMMIT 3: malformed CSV, huge / Unicode labels, mixed columns, and
//! missing target. COMMIT 3's subsystem surface is the COMMIT 1 + 2 API
//! — the CSV reader and `CategoricalTransform` land in COMMIT 4-6 — so
//! each angle is exercised against that surface:
//!
//! * **malformed CSV** — arbitrary byte streams split into cell values
//!   and fed to `CategoryDictionaryBuilder`; to this layer a malformed
//!   CSV is exactly a stream of garbage cells.
//! * **huge / Unicode labels** — labels decoded lossily from arbitrary
//!   bytes, plus a kilobyte-scale repeated label and a Unicode-laden
//!   `SchemaSnapshot` target definition.
//! * **mixed columns** — several per-feature dictionaries combined via
//!   `hash_vocabularies`, pinning feature-order independence.
//! * **missing target** — arbitrary missing-marker sets, all-missing
//!   columns, and the missing-count accounting.
//!
//! The subsystem's contract is that `encode` never panics and the
//! builder / hash paths are total. Unlike the older
//! `tests/bolero_fuzz/abng_decision_fuzz.rs` targets, these do **not**
//! wrap the body in `catch_unwind`: a swallowed panic would defeat the
//! purpose — any panic here is a real defect and must reach bolero.
//!
//! `bolero::check!` compiles to proptest on Windows / macOS and to
//! libfuzzer / AFL under `cargo bolero` on Linux.

use bolero::check;

use cjc_abng::categorical::{
    hash_vocabularies, CategoryDictionary, CategoryDictionaryBuilder, RarePolicy, SchemaSnapshot,
    CODE_MISSING, CODE_UNKNOWN, FEATURE_TRANSFORM_VERSION, FIRST_REAL_CODE,
};

/// Split a fuzz byte stream into category cell values on the NUL byte.
/// Each run of non-NUL bytes is one cell, decoded lossily so arbitrary
/// (including invalid-UTF-8) input still yields a `String`. Capped so a
/// pathological input cannot make a single case unbounded.
fn cells(bytes: &[u8]) -> Vec<String> {
    bytes
        .split(|&b| b == 0)
        .map(|chunk| String::from_utf8_lossy(chunk).into_owned())
        .take(512)
        .collect()
}

/// Malformed-CSV angle: arbitrary cell streams must build a dictionary
/// without panicking; `encode` stays total and consistent, and an
/// independent re-build from the same cells is byte-identical.
#[test]
fn fuzz_malformed_cell_streams() {
    check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let values = cells(input);
        let markers = ["?", ""];
        let mut builder = CategoryDictionaryBuilder::new(&markers);
        builder.observe_all(values.iter());
        let dict = builder.build(RarePolicy::DEFAULT);

        // Every observed row is counted exactly once.
        assert_eq!(dict.total_observed(), values.len() as u64);

        for v in &values {
            let code = dict.encode(v);
            // `encode` is total — every code is within the code space.
            assert!(code < dict.vocab_size(), "encode code {code} >= vocab_size");
            // An observed, non-marker value is never UNKNOWN.
            if !markers.contains(&v.as_str()) {
                assert_ne!(code, CODE_UNKNOWN, "observed {v:?} encoded UNKNOWN");
            }
        }

        // Determinism: an independent re-build is byte-identical.
        let mut rebuild = CategoryDictionaryBuilder::new(&markers);
        rebuild.observe_all(values.iter());
        let dict2 = rebuild.build(RarePolicy::DEFAULT);
        assert_eq!(dict.canonical_bytes(), dict2.canonical_bytes());
        assert_eq!(dict.vocab_hash(), dict2.vocab_hash());
    });
}

/// Huge / Unicode-label angle: pathological labels (lossy-decoded
/// arbitrary bytes, a kilobyte-scale repeat, the empty string) must
/// encode, label, and hash without panicking — including a
/// Unicode-laden `SchemaSnapshot` target definition.
#[test]
fn fuzz_huge_and_unicode_labels() {
    check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let base = String::from_utf8_lossy(input).into_owned();
        let reps = if base.is_empty() { 0 } else { 8192 / base.len() + 1 };
        let huge = base.repeat(reps);
        let labels = [base.clone(), huge, String::new(), "?".to_string()];

        let mut builder = CategoryDictionaryBuilder::new(&["?"]);
        for l in &labels {
            builder.observe(l);
            builder.observe(l);
        }
        let dict = builder.build(RarePolicy::KEEP_ALL);

        for l in &labels {
            let code = dict.encode(l);
            assert!(code < dict.vocab_size());
            let _ = dict.label(code); // total for any code
        }
        let vocab_hash = dict.vocab_hash();

        // A Unicode-laden target definition must hash deterministically.
        let snap = SchemaSnapshot {
            raw_dataset_hash: [0u8; 32],
            schema_hash: [0u8; 32],
            categorical_vocab_hash: vocab_hash,
            numeric_standardization_hash: [0u8; 32],
            feature_transform_version: FEATURE_TRANSFORM_VERSION,
            split_seed: 0,
            row_count: 0,
            target_definition: base,
        };
        assert_eq!(snap.snapshot_hash(), snap.clone().snapshot_hash());
    });
}

/// Mixed-columns angle: cells round-robined into up to 8 per-feature
/// dictionaries; `hash_vocabularies` sorts by feature name, so the
/// combined hash is independent of the order features are supplied in.
#[test]
fn fuzz_mixed_columns_hash_order_independent() {
    check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let n_cols = (input.first().copied().unwrap_or(0) as usize % 8) + 1;
        let body: &[u8] = match input.split_first() {
            Some((_, rest)) => rest,
            None => &[],
        };
        let mut columns: Vec<Vec<String>> = vec![Vec::new(); n_cols];
        for (i, c) in cells(body).into_iter().enumerate() {
            columns[i % n_cols].push(c);
        }
        let dicts: Vec<CategoryDictionary> = columns
            .iter()
            .map(|col| {
                let mut b = CategoryDictionaryBuilder::new(&["?"]);
                b.observe_all(col.iter());
                b.build(RarePolicy::DEFAULT)
            })
            .collect();
        let names: Vec<String> = (0..n_cols).map(|i| format!("feature_{i}")).collect();

        let forward: Vec<(&str, &CategoryDictionary)> =
            names.iter().map(String::as_str).zip(dicts.iter()).collect();
        let mut reversed = forward.clone();
        reversed.reverse();

        assert_eq!(hash_vocabularies(&forward), hash_vocabularies(&reversed));
        // And a re-hash on identical input is stable.
        assert_eq!(hash_vocabularies(&forward), hash_vocabularies(&forward));
    });
}

/// Missing-target angle: arbitrary marker sets + cell streams. Every
/// marker encodes to `CODE_MISSING`, the missing count matches an
/// independent recount, and an all-missing column has zero real
/// categories.
#[test]
fn fuzz_missing_marker_handling() {
    check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let mid = input.len() / 2;
        let (marker_bytes, cell_bytes) = input.split_at(mid);
        let marker_strings = cells(marker_bytes);
        let markers: Vec<&str> = marker_strings.iter().map(String::as_str).collect();
        let values = cells(cell_bytes);

        let mut builder = CategoryDictionaryBuilder::new(&markers);
        builder.observe_all(values.iter());
        let dict = builder.build(RarePolicy::DEFAULT);

        for m in &markers {
            assert_eq!(dict.encode(m), CODE_MISSING, "marker {m:?} not MISSING");
        }
        let expect_missing = values
            .iter()
            .filter(|v| markers.contains(&v.as_str()))
            .count() as u64;
        assert_eq!(dict.missing_count(), expect_missing);
        assert_eq!(dict.total_observed(), values.len() as u64);
        assert!(dict.vocab_size() >= FIRST_REAL_CODE);

        if !values.is_empty() && values.iter().all(|v| markers.contains(&v.as_str())) {
            assert_eq!(dict.n_real(), 0, "all-missing column has a real category");
        }
    });
}
