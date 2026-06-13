//! Property tests for the compression layer.
//!
//! Properties asserted:
//!
//! - **Round-trip preserves lossless input.** Random byte streams
//!   compress + decompress to themselves exactly (RLE and motif-dict
//!   schemes).
//! - **Hash stability under repeated construction.** Identical inputs
//!   produce identical canonical hashes across N constructions.
//! - **Ranking is stable under sorted-equivalent input order.** Same
//!   set of energy components → same ranking regardless of input
//!   permutation.
//! - **Pressure values stay finite and clamped.** No NaN/Inf leaks
//!   from the energy ranker.
//! - **Semantic-critical lossy combination is always rejected.** The
//!   constructor's invariant holds across all generated lossy kinds
//!   + non-empty payloads.
//! - **Reconstruction error is bounded.** Advisory compressors report
//!   `frobenius_error ≤ 1.0` (after clamping) and `≥ 0.0`.

use cjc_cana_compress::lossless_trace::{lossless_compress_bytes, lossless_decompress_bytes};
use cjc_cana_compress::profile_db::{CompilationProfile, FnProfile, PROFILE_SCHEMA_VERSION};
use cjc_cana_compress::{
    compress_low_rank, compress_motif_dictionary, compress_pass_history, compress_tensor_train,
    decompress_motif_dictionary, decompress_pass_history, CandidateId, CompressionCandidate,
    CompressionKind, Criticality, EnergyComponents, EnergyRanker,
};

use proptest::collection::vec;
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

fn arb_nonempty_bytes(max_len: usize) -> impl Strategy<Value = Vec<u8>> {
    vec(any::<u8>(), 1..=max_len)
}

fn arb_small_matrix(max_dim: usize) -> impl Strategy<Value = (Vec<f64>, usize, usize)> {
    (2..=max_dim, 2..=max_dim).prop_flat_map(|(rows, cols)| {
        let len = rows * cols;
        // Bounded f64 entries to keep the SVD well-conditioned.
        vec(-1000.0f64..1000.0f64, len..=len).prop_map(move |m| (m, rows, cols))
    })
}

fn arb_small_tensor() -> impl Strategy<Value = (Vec<f64>, Vec<usize>)> {
    // Limited to small 3D tensors to keep the proptest runtime sane.
    (1usize..=4, 1usize..=4, 1usize..=4).prop_flat_map(|(d0, d1, d2)| {
        let len = d0 * d1 * d2;
        vec(-100.0f64..100.0f64, len..=len).prop_map(move |t| (t, vec![d0, d1, d2]))
    })
}

fn arb_energy_component() -> impl Strategy<Value = EnergyComponents> {
    // Nested tuples keep each group under proptest's tuple-arity limit
    // now that EnergyComponents has 11 fields (7 costs + 4 rewards).
    (
        (
            0.0f64..1000.0, // runtime_cost
            0.0f64..1.0,    // memory_pressure
            0.0f64..1.0,    // thermal_pressure
            0.0f64..1.0,    // bandwidth_pressure
            0.0f64..1000.0, // code_size_pressure
            0.0f64..1.0,    // reconstruction_risk
            0.0f64..1.0,    // verifier_risk_penalty
        ),
        (
            0.0f64..500.0, // fusion_reward
            0.0f64..500.0, // reuse_reward
            0.0f64..500.0, // compression_reward
            0.0f64..500.0, // locality_reward
        ),
    )
        .prop_map(|(c, r)| {
            EnergyComponents::new(c.0, c.1, c.2, c.3, c.4, c.5, c.6, r.0, r.1, r.2, r.3)
                .expect("all components are non-negative and finite by strategy")
        })
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        ..ProptestConfig::default()
    })]

    /// **Lossless round-trip preserves bytes exactly.** RLE codec.
    #[test]
    fn lossless_rle_round_trips(bytes in arb_nonempty_bytes(2048)) {
        let payload = lossless_compress_bytes(&bytes);
        let back = lossless_decompress_bytes(&payload.bytes).expect("decode");
        prop_assert_eq!(back, bytes);
    }

    /// **Lossless round-trip preserves bytes exactly.** Motif codec.
    #[test]
    fn lossless_motif_round_trips(bytes in arb_nonempty_bytes(2048)) {
        let payload = compress_motif_dictionary(&bytes);
        let back = decompress_motif_dictionary(&payload.bytes).expect("decode");
        prop_assert_eq!(back, bytes);
    }

    /// **Compressing twice produces byte-identical outputs.** RLE.
    #[test]
    fn lossless_rle_is_deterministic(bytes in arb_nonempty_bytes(1024)) {
        let p1 = lossless_compress_bytes(&bytes);
        let p2 = lossless_compress_bytes(&bytes);
        prop_assert_eq!(p1.bytes, p2.bytes);
        prop_assert_eq!(p1.input_hash, p2.input_hash);
        prop_assert_eq!(p1.compressed_hash, p2.compressed_hash);
    }

    /// **Compressing twice produces byte-identical outputs.** Motif.
    #[test]
    fn lossless_motif_is_deterministic(bytes in arb_nonempty_bytes(1024)) {
        let p1 = compress_motif_dictionary(&bytes);
        let p2 = compress_motif_dictionary(&bytes);
        prop_assert_eq!(p1.bytes, p2.bytes);
        prop_assert_eq!(p1.input_hash, p2.input_hash);
        prop_assert_eq!(p1.compressed_hash, p2.compressed_hash);
    }

    /// **Semantic-critical lossy combination is always rejected.**
    #[test]
    fn semantic_critical_lossy_always_rejected(
        bytes in arb_nonempty_bytes(256),
        id in any::<u64>(),
        lossy_kind_idx in 0u8..2
    ) {
        let kind = match lossy_kind_idx {
            0 => CompressionKind::LowRankAdvisory,
            _ => CompressionKind::TensorTrainAdvisory,
        };
        let r = CompressionCandidate::new(
            CandidateId(id),
            kind,
            Criticality::SemanticCritical,
            bytes,
            "p",
        );
        prop_assert!(r.is_err());
    }

    /// **Pass-history round-trip is exact.**
    #[test]
    fn pass_history_round_trips(
        names in vec("[a-z_]{1,12}", 1..16),
        hashes in vec(any::<u64>(), 1..32),
    ) {
        use cjc_cana::hash::ProgramHash;
        use cjc_cana::pass_history::{PassHistory, PassOutcome, PassRecord};
        let mut h = PassHistory::with_capacity(64);
        let n = names.len().min(hashes.len()).max(1);
        for i in 0..n {
            h.record(PassRecord {
                pass_name: names[i % names.len()].clone(),
                input_hash: ProgramHash(hashes[i % hashes.len()]),
                output_hash: ProgramHash(hashes[(i + 1) % hashes.len()]),
                outcome: PassOutcome::Changed,
            });
        }
        let payload = compress_pass_history(&h);
        let back = decompress_pass_history(&payload).expect("decode");
        let original: Vec<_> = h.iter().cloned().collect();
        let restored: Vec<_> = back.iter().cloned().collect();
        prop_assert_eq!(original, restored);
    }

    /// **Low-rank reconstruction error stays in `[0, 1]`.**
    #[test]
    fn low_rank_error_in_unit_interval(t in arb_small_matrix(6)) {
        let (m, rows, cols) = t;
        let max_rank = rows.min(cols);
        let p = compress_low_rank(&m, rows, cols, max_rank);
        if let Ok(p) = p {
            prop_assert!(p.frobenius_error.is_finite());
            prop_assert!(p.frobenius_error >= 0.0);
            prop_assert!(p.frobenius_error <= 1.0);
        }
    }

    /// **Low-rank summary is deterministic.**
    #[test]
    fn low_rank_deterministic(t in arb_small_matrix(6)) {
        let (m, rows, cols) = t;
        let max_rank = rows.min(cols);
        let p1 = compress_low_rank(&m, rows, cols, max_rank);
        let p2 = compress_low_rank(&m, rows, cols, max_rank);
        match (p1, p2) {
            (Ok(p1), Ok(p2)) => {
                prop_assert_eq!(p1.summary_hash, p2.summary_hash);
                prop_assert_eq!(p1.singular_values, p2.singular_values);
                prop_assert_eq!(p1.frobenius_error.to_bits(), p2.frobenius_error.to_bits());
            }
            (Err(_), Err(_)) => {}
            _ => prop_assert!(false, "non-deterministic Ok/Err outcome"),
        }
    }

    /// **Tensor-train summary is deterministic + error bounded.**
    #[test]
    fn tensor_train_deterministic_and_bounded(t in arb_small_tensor()) {
        let (tensor, shape) = t;
        let p1 = compress_tensor_train(&tensor, &shape, 4, 1e-12);
        let p2 = compress_tensor_train(&tensor, &shape, 4, 1e-12);
        match (p1, p2) {
            (Ok(p1), Ok(p2)) => {
                prop_assert_eq!(p1.summary_hash, p2.summary_hash);
                prop_assert!(p1.frobenius_error.is_finite());
                prop_assert!(p1.frobenius_error >= 0.0);
                prop_assert!(p1.frobenius_error <= 1.0);
            }
            (Err(_), Err(_)) => {}
            _ => prop_assert!(false, "non-deterministic Ok/Err outcome"),
        }
    }

    /// **Energy ranker is stable under input shuffle.**
    #[test]
    fn energy_ranker_stable_under_shuffle(
        components in vec(arb_energy_component(), 1..16),
    ) {
        let inputs_a: Vec<_> = components.iter().enumerate()
            .map(|(i, c)| (CandidateId(i as u64), *c))
            .collect();
        let mut inputs_b = inputs_a.clone();
        inputs_b.reverse();
        let r = EnergyRanker::new();
        let ranking_a = r.rank(inputs_a);
        let ranking_b = r.rank(inputs_b);
        let ids_a: Vec<u64> = ranking_a.ordered.iter().map(|x| x.id.0).collect();
        let ids_b: Vec<u64> = ranking_b.ordered.iter().map(|x| x.id.0).collect();
        prop_assert_eq!(ids_a, ids_b);
    }

    /// **Energy ranker produces finite totals.**
    #[test]
    fn energy_ranker_totals_are_finite(
        components in vec(arb_energy_component(), 1..16),
    ) {
        let inputs: Vec<_> = components.iter().enumerate()
            .map(|(i, c)| (CandidateId(i as u64), *c))
            .collect();
        let r = EnergyRanker::new();
        let ranking = r.rank(inputs);
        for rc in &ranking.ordered {
            prop_assert!(rc.score.total.is_finite());
            prop_assert!(!rc.score.total.is_nan());
        }
    }

    /// **Schema-v3 profile rows round-trip for any per-function map.**
    /// (Phase A items 2+3: the nested `FnProfile` records — arbitrary
    /// names, counters, and finite labels — survive the canonical
    /// codec exactly.)
    #[test]
    fn profile_row_roundtrips_any_per_function(
        fns in vec(arb_fn_profile_entry(), 0..10),
        flops in any::<u64>(),
        score in -1.0e9f64..1.0e9,
    ) {
        let mut row = base_profile_row();
        row.per_function = fns;
        row.estimated_flops = flops;
        row.score = score;
        let back = CompilationProfile::from_canonical_bytes(&row.canonical_bytes())
            .expect("canonical round-trip");
        prop_assert_eq!(row, back);
    }

    /// **CPB1 energy bundles round-trip for any pass vocabulary and
    /// finite weights.** (Phase B: the variable-length basis — the
    /// part CPB0 didn't have — survives the codec exactly.)
    #[test]
    fn energy_bundle_roundtrips_any_vocabulary(
        names in vec("[a-z_]{1,16}", 0..12),
        seed in -100.0f64..100.0,
    ) {
        use cjc_cana::pinn_energy_v1::{PinnEnergyV1, ENERGY_TAIL_FEATURES, ENERGY_WORKLOAD_FEATURES};
        use cjc_cana_compress::energy_bundle::EnergyBundle;
        // Dedup: the basis assumes distinct pass names.
        let mut names = names;
        names.sort();
        names.dedup();
        let n = ENERGY_WORKLOAD_FEATURES + names.len() + ENERGY_TAIL_FEATURES;
        let bundle = EnergyBundle {
            model_id: "pinn_energy_v1".to_string(),
            model_version: 1,
            head: PinnEnergyV1 {
                pass_names: names,
                feature_means: (0..n).map(|i| seed + i as f64).collect(),
                feature_stds: (0..n).map(|i| 1.0 + (i as f64) * 0.25).collect(),
                coefficients: (0..n).map(|i| seed * 0.01 - i as f64 * 0.001).collect(),
                intercept: seed * 0.5,
            },
        };
        let back = EnergyBundle::from_canonical_bytes(&bundle.canonical_bytes())
            .expect("canonical round-trip");
        prop_assert_eq!(bundle, back);
    }
}

// ---------------------------------------------------------------------------
// Schema-v3 profile-row strategies (Phase A items 2+3)
// ---------------------------------------------------------------------------

fn arb_fn_profile_entry() -> impl Strategy<Value = (String, FnProfile)> {
    (
        "[a-z_][a-z0-9_]{0,12}",
        any::<u64>(),
        any::<u64>(),
        any::<u64>(),
        any::<u64>(),
        any::<u64>(),
        any::<u64>(),
        any::<u32>(),
        any::<u32>(),
        // Finite labels: NaN breaks PartialEq-based round-trip asserts
        // and the writer never produces it (labels are clamped [0,1]).
        -1.0e6f64..1.0e6,
        -1.0e6f64..1.0e6,
        -1.0e6f64..1.0e6,
    )
        .prop_map(
            |(name, fl, br, bw, ab, ws, fo, clc, mld, cpu, mem, thermal)| {
                (
                    name,
                    FnProfile {
                        flops: fl,
                        bytes_read: br,
                        bytes_written: bw,
                        alloc_bytes: ab,
                        working_set: ws,
                        float_ops: fo,
                        // Schema v4: derived deterministically from the
                        // generated fields so the strategy tuple stays
                        // within proptest's arity limit.
                        creation_alloc: ab.rotate_left(7) ^ fo,
                        countable_loop_count: clc,
                        max_loop_depth: mld,
                        nss_cpu: cpu,
                        nss_memory: mem,
                        nss_thermal: thermal,
                    },
                )
            },
        )
}

fn base_profile_row() -> CompilationProfile {
    CompilationProfile {
        schema_version: PROFILE_SCHEMA_VERSION,
        program_name: "prop".to_string(),
        program_hash: 1,
        feature_hash: 2,
        sidecar_bundle_hash: 0,
        config_id: "baseline".to_string(),
        cost_model_id: "linear_v1".to_string(),
        cost_model_version: 1,
        pass_sequence: vec![("f".to_string(), vec!["dce".to_string()])],
        per_function: vec![],
        estimated_flops: 0,
        estimated_bytes_read: 0,
        estimated_bytes_written: 0,
        estimated_alloc_bytes: 0,
        estimated_working_set: 0,
        estimated_float_ops: 0,
        estimated_creation_alloc_bytes: 0,
        nss_predicted_cpu_max: 0.0,
        nss_predicted_memory_max: 0.0,
        nss_predicted_thermal_max: 0.0,
        pinn_predicted_energy_max: 0.0,
        pinn_predicted_thermal_max: 0.0,
        pinn_predicted_bandwidth_max: 0.0,
        mir_nodes_before: 0,
        mir_nodes_after: 0,
        recommended_count: 0,
        dropped_count: 0,
        legality_approved: true,
        legality_violation_count: 0,
        parity_match: Some(true),
        compile_wall_micros: 0,
        score: 1.0,
    }
}
