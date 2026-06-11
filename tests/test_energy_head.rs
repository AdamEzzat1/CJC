//! Phase B — energy-head WIRING tests.
//!
//! Locks the trained artifact end-to-end: the COMMITTED CPB1 bundle
//! loads through the real loader, carries the right identity, and
//! produces bit-deterministic predictions through the shared basis
//! definition. (Codec corruption/round-trip behavior is unit- and
//! fuzz-tested in `cjc-cana-compress`; these tests prove the shipped
//! artifact + API surface Phase C will consume.)

use std::path::Path;

use cjc_cana::pinn_energy_v1::{EnergyQuery, PINN_ENERGY_V1_MODEL_ID};
use cjc_cana_compress::energy_bundle::read_energy_bundle;

const BUNDLE_PATH: &str = "bench_results/cana_train_pinn/pinn_energy_v1.cpb";

fn committed_bundle() -> cjc_cana_compress::energy_bundle::EnergyBundle {
    read_energy_bundle(Path::new(BUNDLE_PATH))
        .expect("committed CPB1 bundle must load — run cana-train-pinn -- train-energy if missing")
}

fn sample_query(head: &cjc_cana::pinn_energy_v1::PinnEnergyV1, passes: &[&str]) -> EnergyQuery {
    EnergyQuery {
        flops_estimate: 1_500,
        bytes_read_estimate: 12_000,
        bytes_written_estimate: 2_600,
        allocation_bytes_estimate: 192,
        working_set_bytes_estimate: 2_240,
        float_ops_estimate: 640,
        mir_nodes_before: 150,
        recommended_count: 4,
        dropped_count: 0,
        pass_counts: head.pass_counts(passes.iter().copied()),
        countable_loop_count: 3,
        max_loop_depth: 2,
        mir_nodes_after: 130,
    }
}

#[test]
fn committed_bundle_loads_with_correct_identity() {
    let b = committed_bundle();
    assert_eq!(b.model_id, PINN_ENERGY_V1_MODEL_ID);
    assert!(b.head.is_valid());
    // The vocabulary is part of the artifact; it must be non-empty and
    // sorted (BTreeSet order at train time) — alignment depends on it.
    assert!(!b.head.pass_names.is_empty());
    let mut sorted = b.head.pass_names.clone();
    sorted.sort();
    assert_eq!(sorted, b.head.pass_names, "vocabulary must be sorted");
}

#[test]
fn predictions_are_bit_deterministic() {
    let b = committed_bundle();
    let q = sample_query(&b.head, &["dce", "licm", "loop_unroll"]);
    let first = b.head.predict_ln_score(&q).to_bits();
    for _ in 0..50 {
        assert_eq!(first, b.head.predict_ln_score(&q).to_bits());
    }
}

#[test]
fn predictions_are_finite_across_plan_space() {
    // Every subset of the head's own vocabulary must predict a finite
    // ln-score (the selector will sweep candidate plans).
    let b = committed_bundle();
    let names: Vec<&str> = b.head.pass_names.iter().map(|s| s.as_str()).collect();
    for k in 0..=names.len() {
        let plan: Vec<&str> = names.iter().take(k).copied().collect();
        let q = sample_query(&b.head, &plan);
        let p = b.head.predict_ln_score(&q);
        assert!(p.is_finite(), "non-finite prediction for plan {plan:?}");
    }
}

#[test]
fn plan_contents_change_the_prediction() {
    // The head must actually discriminate between plans — an empty
    // plan and a full canonical plan on the same workload should not
    // predict identically (otherwise the selector criterion is inert).
    let b = committed_bundle();
    let names: Vec<&str> = b.head.pass_names.iter().map(|s| s.as_str()).collect();
    let q_none = sample_query(&b.head, &[]);
    let q_all = sample_query(&b.head, &names);
    assert_ne!(
        b.head.predict_ln_score(&q_none).to_bits(),
        b.head.predict_ln_score(&q_all).to_bits(),
        "head is plan-blind"
    );
}

#[test]
fn unknown_passes_do_not_perturb_alignment() {
    let b = committed_bundle();
    let q_known = sample_query(&b.head, &["dce"]);
    let q_with_unknown = sample_query(&b.head, &["dce", "not_a_real_pass"]);
    assert_eq!(
        b.head.predict_ln_score(&q_known).to_bits(),
        b.head.predict_ln_score(&q_with_unknown).to_bits(),
        "unknown pass names must be ignored, not shift counts"
    );
}
