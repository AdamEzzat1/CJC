//! Determinism regression: same seed/input → byte-identical output.
//!
//! These are the canary tests for the "determinism contract" the
//! design doc promises. They run the entire stack twice and assert
//! every observable artifact is bit-equal:
//!
//! - CompressionReport hashes
//! - PressureDensityState canonical bytes + stable hash
//! - PressureCorrelationSummary fields
//! - EnergyRanking IDs and totals
//!
//! Tests are intentionally redundant with per-module determinism tests
//! — the redundancy guards against a regression where one layer keeps
//! determinism but a wiring layer above it adds nondeterminism.

use cjc_cana_compress::{
    compression_pressure_delta, encode_low_rank_payload, encode_tensor_train_payload,
    BridgeCoefficients, CandidateId, CompressionCandidate, CompressionKind, CompressionPlan,
    Criticality, EnergyComponents, EnergyRanker,
};
use cjc_nss::{PressureDensityState, PressureKind};

fn build_baseline_density() -> PressureDensityState {
    let mut s = PressureDensityState::empty();
    s.apply_delta(PressureKind::Memory, 0.6);
    s.apply_delta(PressureKind::Thermal, 0.3);
    s.apply_delta(PressureKind::Cpu, 0.45);
    s.apply_delta(PressureKind::Io, 0.10);
    s
}

fn build_plan() -> CompressionPlan {
    let lossless = CompressionCandidate::new(
        CandidateId(10),
        CompressionKind::LosslessTrace,
        Criticality::SemanticCritical,
        b"AAAAAAAA".repeat(20),
        "hist.a",
    )
    .unwrap();
    let motif = CompressionCandidate::new(
        CandidateId(5),
        CompressionKind::MotifDictionary,
        Criticality::SemanticCritical,
        b"xyzxyzxyzxyzxyzxyz".to_vec(),
        "motif.b",
    )
    .unwrap();
    let low_rank = {
        let m: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let payload = encode_low_rank_payload(&m, 3, 4, 2);
        CompressionCandidate::new(
            CandidateId(7),
            CompressionKind::LowRankAdvisory,
            Criticality::AdvisoryOnly { tolerance_f: 0.5 },
            payload,
            "feat.c",
        )
        .unwrap()
    };
    let tt = {
        let t: Vec<f64> = (1..=8).map(|x| x as f64 * 0.25).collect();
        let payload = encode_tensor_train_payload(&t, &[2, 2, 2], 4);
        CompressionCandidate::new(
            CandidateId(2),
            CompressionKind::TensorTrainAdvisory,
            Criticality::AdvisoryOnly { tolerance_f: 1e-6 },
            payload,
            "tt.d",
        )
        .unwrap()
    };
    CompressionPlan::new(vec![lossless, motif, low_rank, tt])
}

// ---------------------------------------------------------------------------
// CompressionReport determinism
// ---------------------------------------------------------------------------

#[test]
fn report_hash_is_byte_identical_across_runs() {
    let plan_a = build_plan();
    let plan_b = build_plan();
    let report_a = plan_a.execute();
    let report_b = plan_b.execute();
    assert_eq!(report_a.report_hash(), report_b.report_hash());
    assert_eq!(report_a.canonical_bytes(), report_b.canonical_bytes());
}

#[test]
fn report_hash_stable_across_n_iterations() {
    let baseline = build_plan().execute();
    let hash = baseline.report_hash();
    for _ in 0..32 {
        let r = build_plan().execute();
        assert_eq!(r.report_hash(), hash);
    }
}

#[test]
fn report_json_is_byte_identical_across_runs() {
    let plan_a = build_plan();
    let plan_b = build_plan();
    let json_a = plan_a.execute().to_json();
    let json_b = plan_b.execute().to_json();
    assert_eq!(json_a, json_b);
}

// ---------------------------------------------------------------------------
// PressureDensityState determinism
// ---------------------------------------------------------------------------

#[test]
fn pressure_density_canonical_bytes_stable_across_runs() {
    let a = build_baseline_density();
    let b = build_baseline_density();
    assert_eq!(a.canonical_bytes(), b.canonical_bytes());
    assert_eq!(a.stable_hash(), b.stable_hash());
}

#[test]
fn pressure_density_summary_stable_across_runs() {
    let a = build_baseline_density();
    let b = build_baseline_density();
    assert_eq!(a.summary(), b.summary());
}

// ---------------------------------------------------------------------------
// Bridge determinism — same input flows produce identical post-delta states
// ---------------------------------------------------------------------------

#[test]
fn bridge_pressure_delta_is_byte_identical_across_runs() {
    let report_a = build_plan().execute();
    let report_b = build_plan().execute();
    let result_a = compression_pressure_delta(
        build_baseline_density(),
        &report_a,
        BridgeCoefficients::default(),
    );
    let result_b = compression_pressure_delta(
        build_baseline_density(),
        &report_b,
        BridgeCoefficients::default(),
    );
    assert_eq!(
        result_a.updated.stable_hash(),
        result_b.updated.stable_hash()
    );
    assert_eq!(
        result_a.delta_memory.to_bits(),
        result_b.delta_memory.to_bits()
    );
    assert_eq!(
        result_a.delta_thermal.to_bits(),
        result_b.delta_thermal.to_bits()
    );
    assert_eq!(
        result_a.delta_throughput.to_bits(),
        result_b.delta_throughput.to_bits()
    );
    assert_eq!(result_a.rewarded_entries, result_b.rewarded_entries);
    assert_eq!(result_a.penalised_entries, result_b.penalised_entries);
}

// ---------------------------------------------------------------------------
// EnergyRanker determinism
// ---------------------------------------------------------------------------

#[test]
fn energy_ranker_order_is_byte_identical_across_runs() {
    let inputs = vec![
        (
            CandidateId(7),
            EnergyComponents::builder()
                .runtime_cost(0.3)
                .compression_reward(0.5)
                .build()
                .unwrap(),
        ),
        (
            CandidateId(2),
            EnergyComponents::builder()
                .runtime_cost(0.6)
                .compression_reward(0.1)
                .build()
                .unwrap(),
        ),
        (
            CandidateId(5),
            EnergyComponents::builder()
                .thermal_pressure(0.9)
                .build()
                .unwrap(),
        ),
    ];
    let ranker = EnergyRanker::new();
    let a = ranker.rank(inputs.clone());
    let b = ranker.rank(inputs);
    let ids_a: Vec<u64> = a.ordered.iter().map(|x| x.id.0).collect();
    let ids_b: Vec<u64> = b.ordered.iter().map(|x| x.id.0).collect();
    assert_eq!(ids_a, ids_b);
    for (x, y) in a.ordered.iter().zip(b.ordered.iter()) {
        assert_eq!(x.score.total.to_bits(), y.score.total.to_bits());
        assert_eq!(x.score.components, y.score.components);
    }
}

// ---------------------------------------------------------------------------
// End-to-end pipeline determinism
// ---------------------------------------------------------------------------

#[test]
fn end_to_end_pipeline_double_run() {
    let run = || {
        let plan = build_plan();
        let report = plan.execute();
        let delta = compression_pressure_delta(
            build_baseline_density(),
            &report,
            BridgeCoefficients::default(),
        );
        let summary = delta.summary.clone();
        let json = report.to_json();
        let density_bytes = delta.updated.canonical_bytes();
        (
            report.report_hash(),
            summary.stable_hash,
            summary.saturation_score.to_bits(),
            summary.collapse_risk.to_bits(),
            json,
            density_bytes,
        )
    };
    let a = run();
    let b = run();
    assert_eq!(a.0, b.0);
    assert_eq!(a.1, b.1);
    assert_eq!(a.2, b.2);
    assert_eq!(a.3, b.3);
    assert_eq!(a.4, b.4);
    assert_eq!(a.5, b.5);
}
