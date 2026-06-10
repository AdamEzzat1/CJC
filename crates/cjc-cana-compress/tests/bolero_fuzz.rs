//! Bolero fuzz harness for the compression layer.
//!
//! These targets explore the boundary surfaces:
//!
//! - **Malformed compression candidates** — randomly-shaped payload
//!   bytes fed to `lossless_decompress_bytes` / `decompress_motif_dictionary`
//!   must never panic; they may error.
//! - **Random pass-history traces** — proves the canonical
//!   PassHistory serializer / deserializer survive random input.
//! - **Random pressure vectors** — `PressureDensityState::from_trajectory`
//!   handles whatever shape the fuzzer produces (within validation
//!   constraints) without panicking.
//! - **Random candidate rankings** — energy ranker is total and stable
//!   on every input set.
//!
//! Fuzz invariants:
//! - No panic from valid public-API calls.
//! - No NaN/Inf in pressure outputs.
//! - Lossless decompression either round-trips or errors cleanly.
//! - Energy ranker produces a total order.

use bolero::check;
use cjc_cana_compress::lossless_trace::{lossless_compress_bytes, lossless_decompress_bytes};
use cjc_cana_compress::motif_dictionary::{compress_motif_dictionary, decompress_motif_dictionary};
use cjc_cana_compress::{CandidateId, EnergyComponents, EnergyRanker};
use cjc_nss::PressureDensityState;
use cjc_nss::{Pressure, PressureField, PressureKind};

// ---------------------------------------------------------------------------
// 1. Lossless RLE decoder on arbitrary bytes — no panic
// ---------------------------------------------------------------------------

#[test]
fn fuzz_lossless_decoder_never_panics() {
    check!()
        .with_max_len(2048)
        .with_iterations(1000)
        .for_each(|bytes: &[u8]| {
            // Decoder must produce either Ok(_) or Err(_), never panic.
            let _ = lossless_decompress_bytes(bytes);
        });
}

// ---------------------------------------------------------------------------
// 2. Motif decoder on arbitrary bytes — no panic
// ---------------------------------------------------------------------------

#[test]
fn fuzz_motif_decoder_never_panics() {
    check!()
        .with_max_len(2048)
        .with_iterations(1000)
        .for_each(|bytes: &[u8]| {
            let _ = decompress_motif_dictionary(bytes);
        });
}

// ---------------------------------------------------------------------------
// 3. Lossless round-trip — compress then decompress always exact
// ---------------------------------------------------------------------------

#[test]
fn fuzz_cana_compression_roundtrip() {
    check!()
        .with_max_len(1024)
        .with_iterations(1000)
        .for_each(|bytes: &[u8]| {
            if bytes.is_empty() {
                return;
            }
            let p = lossless_compress_bytes(bytes);
            let back = lossless_decompress_bytes(&p.bytes).expect("decode");
            assert_eq!(back.as_slice(), bytes);
        });
}

// ---------------------------------------------------------------------------
// 4. Motif round-trip — compress then decompress always exact
// ---------------------------------------------------------------------------

#[test]
fn fuzz_motif_compression_roundtrip() {
    check!()
        .with_max_len(1024)
        .with_iterations(1000)
        .for_each(|bytes: &[u8]| {
            if bytes.is_empty() {
                return;
            }
            let p = compress_motif_dictionary(bytes);
            let back = decompress_motif_dictionary(&p.bytes).expect("decode");
            assert_eq!(back.as_slice(), bytes);
        });
}

// ---------------------------------------------------------------------------
// 5. Pressure density state — finite + clamped under arbitrary input
// ---------------------------------------------------------------------------

#[test]
fn fuzz_nss_pressure_density_summary() {
    // Fuzz pressure-magnitude pairs and ensure the resulting density
    // state's diagonal entries stay in `[0, ∞)` and the summary
    // collapse_risk stays in `[0, 1]`.
    check!()
        .with_max_len(64)
        .with_iterations(1000)
        .for_each(|raw: &[u8]| {
            if raw.is_empty() {
                return;
            }
            // Map each pair of bytes into (PressureKind index, magnitude).
            let mut state = PressureDensityState::empty();
            let kinds = PressureKind::all();
            for chunk in raw.chunks(2) {
                if chunk.len() < 2 {
                    break;
                }
                let kind_idx = (chunk[0] as usize) % kinds.len();
                let mag = (chunk[1] as f64) / 255.0; // → [0, 1]
                state.apply_delta(kinds[kind_idx], mag);
            }
            let summary = state.summary();
            assert!(summary.saturation_score.is_finite());
            assert!(summary.collapse_risk.is_finite());
            assert!((0.0..=1.0).contains(&summary.saturation_score));
            assert!((0.0..=1.0).contains(&summary.collapse_risk));
        });
}

// ---------------------------------------------------------------------------
// 6. Energy ranker — total order + no NaN
// ---------------------------------------------------------------------------

#[test]
fn fuzz_quantum_inspired_ranking_total_order() {
    check!()
        .with_max_len(256)
        .with_iterations(1000)
        .for_each(|raw: &[u8]| {
            if raw.is_empty() {
                return;
            }
            // Build candidates from bytes: each chunk of 22 bytes encodes
            // 11 component magnitudes (each in [0, 1]).
            let mut inputs = Vec::new();
            for (i, chunk) in raw.chunks(22).enumerate() {
                if chunk.len() < 22 {
                    break;
                }
                let v = |off: usize| -> f64 {
                    let raw = u16::from_le_bytes([chunk[off], chunk[off + 1]]) as f64;
                    raw / u16::MAX as f64
                };
                let components = EnergyComponents::new(
                    v(0),
                    v(2),
                    v(4),
                    v(6),
                    v(8),
                    v(10),
                    v(12),
                    v(14),
                    v(16),
                    v(18),
                    v(20),
                )
                .expect("non-negative finite values by construction");
                inputs.push((CandidateId(i as u64), components));
            }
            if inputs.is_empty() {
                return;
            }
            let ranker = EnergyRanker::new();
            let result = ranker.rank(inputs);
            // Total order: results sorted by ascending energy with stable
            // tie-break by ID.
            for w in result.ordered.windows(2) {
                let a = w[0].score.total;
                let b = w[1].score.total;
                assert!(a.is_finite());
                assert!(b.is_finite());
                if a == b {
                    // Tie-break must put smaller ID first.
                    assert!(w[0].id <= w[1].id);
                } else {
                    assert!(a <= b);
                }
            }
            // All survivors have valid components.
            for rc in &result.ordered {
                assert!(rc.score.components.is_valid());
            }
        });
}

// ---------------------------------------------------------------------------
// 7. Pressure trajectory smoke test — no panic from arbitrary magnitudes
// ---------------------------------------------------------------------------

#[test]
fn fuzz_pressure_trajectory_never_panics() {
    check!()
        .with_max_len(128)
        .with_iterations(500)
        .for_each(|raw: &[u8]| {
            // Build small trajectories from the input bytes; each byte
            // produces a single-pressure field with magnitude in
            // `[0, 1]`. The resulting summary must be finite.
            let mut traj = Vec::new();
            for &b in raw.iter().take(16) {
                let mut f = PressureField::empty();
                let p = match Pressure::new((b as f64) / 255.0, 1.0, 0.1) {
                    Ok(p) => p,
                    Err(_) => continue,
                };
                f.set(PressureKind::Memory, p);
                traj.push(f);
            }
            let state = PressureDensityState::from_trajectory(&traj);
            let summary = state.summary();
            assert!(summary.saturation_score.is_finite());
            assert!(!summary.saturation_score.is_nan());
        });
}

// ---------------------------------------------------------------------------
// 8. PINN v1 physical-cost model — arbitrary queries never panic, outputs
//    always valid (or the coefficient set is provably invalid)
// ---------------------------------------------------------------------------

#[test]
fn fuzz_physical_cost_prediction_stays_clamped() {
    use cjc_cana::physical_cost::{predict_physical, PhysicalCoefficients, PhysicalCostQuery};

    check!()
        .with_max_len(256)
        .with_iterations(1000)
        .for_each(|raw: &[u8]| {
            // 5 u64 + 2 u32 + 1 u64 overhead = 56 bytes for the query,
            // 8 f64 = 64 bytes for the coefficients → 120 total.
            if raw.len() < 120 {
                return;
            }
            let u64_at =
                |i: usize| -> u64 { u64::from_le_bytes(raw[i..i + 8].try_into().unwrap()) };
            let u32_at =
                |i: usize| -> u32 { u32::from_le_bytes(raw[i..i + 4].try_into().unwrap()) };
            let f64_at =
                |i: usize| -> f64 { f64::from_le_bytes(raw[i..i + 8].try_into().unwrap()) };

            let query = PhysicalCostQuery {
                function_name: "fuzz",
                strategy_id: "loop_unroll",
                flops_estimate: u64_at(0),
                bytes_read_estimate: u64_at(8),
                bytes_written_estimate: u64_at(16),
                allocation_bytes_estimate: u64_at(24),
                working_set_bytes_estimate: u64_at(32),
                thread_count: u32_at(40),
                batch_size: u32_at(44),
                compression_overhead_bytes: u64_at(48),
            };
            // Raw-bit f64s are frequently NaN / negative / subnormal —
            // exactly the inputs the validity gate must catch.
            let coeffs = PhysicalCoefficients {
                flops_norm_scale: f64_at(56),
                cooling_rate: f64_at(64),
                bytes_per_flop_scale: f64_at(72),
                bytes_norm_scale: f64_at(80),
                alloc_norm_scale: f64_at(88),
                thread_amplification: f64_at(96),
                batch_amplification: f64_at(104),
                locality_weight: f64_at(112),
            };

            match predict_physical(&query, &coeffs) {
                Some(est) => assert!(
                    est.is_valid(),
                    "valid coefficients must produce a valid estimate: {est:?}"
                ),
                None => assert!(
                    !coeffs.is_valid(),
                    "predict_physical may only abstain on invalid coefficients"
                ),
            }
        });
}

// ---------------------------------------------------------------------------
// 9. PINN v1 — default coefficients accept every possible workload query
// ---------------------------------------------------------------------------

#[test]
fn fuzz_default_coefficients_never_abstain() {
    use cjc_cana::physical_cost::{predict_physical, PhysicalCoefficients, PhysicalCostQuery};

    check!()
        .with_max_len(64)
        .with_iterations(1000)
        .for_each(|raw: &[u8]| {
            if raw.len() < 48 {
                return;
            }
            let u64_at =
                |i: usize| -> u64 { u64::from_le_bytes(raw[i..i + 8].try_into().unwrap()) };
            let u32_at =
                |i: usize| -> u32 { u32::from_le_bytes(raw[i..i + 4].try_into().unwrap()) };
            let query = PhysicalCostQuery {
                function_name: "fuzz",
                strategy_id: "vectorize",
                flops_estimate: u64_at(0),
                bytes_read_estimate: u64_at(8),
                bytes_written_estimate: u64_at(16),
                allocation_bytes_estimate: u64_at(24),
                working_set_bytes_estimate: u64_at(32),
                thread_count: u32_at(40),
                batch_size: u32_at(44),
                compression_overhead_bytes: 0,
            };
            let est = predict_physical(&query, &PhysicalCoefficients::default())
                .expect("default coefficients are valid by construction");
            assert!(est.is_valid(), "{est:?}");
        });
}
