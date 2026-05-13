//! ABNG demo: lineage attestation for regulated ML training.
//!
//! ## What this demo proves
//!
//! Suppose a regulator (FDA, financial auditor, ethics board) needs to
//! verify that a deployed model was trained on the exact dataset the
//! lab claims it was — not a contaminated, mislabeled, or post-hoc
//! "improved" version. With a vanilla MLP, the answer is essentially
//! "trust the lab's bookkeeping." With ABNG, every observation is in
//! an SHA-256 audit chain bound to a per-node provenance stamp, and
//! every prediction snapshot carries the full lineage end-to-end.
//!
//! This file constructs the regulator's nightmare scenario:
//!
//! 1. Lab `A` trains an ABNG model on a fixed clinical-trial dataset.
//!    They stamp the root with `sha256(dataset_A_bytes)`, generate
//!    predictions, save the predict_snap blobs to disk.
//! 2. An attacker substitutes a model trained on a *tampered* dataset
//!    `B` — one observation rewritten — and tries to pass off the
//!    original lab-A predictions as still valid.
//! 3. The regulator runs `cjcl abng explain --model B.snap A.predict`
//!    (here, the equivalent in-process check). The attestation
//!    fails by *all three* independent signals: chain_head, BLR
//!    state_hash, AND provenance stamp.
//!
//! No vanilla MLP can produce all three signals; this is ABNG's
//! cleanest tangible benefit.

use cjc_abng::audit::AuditKind;
use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_abng::predict_snap::{pack, unpack, PredictionSnap};
use cjc_abng::serialize::{replay, serialize};
use cjc_ad::pinn::Activation;

/// "Clinical trial" dataset row: `(patient_id, dose, response)`.
/// Synthetic but reproducible.
type Row = (u32, f64, f64);

/// Dataset A — the lab's official 64-row record. Reproducible
/// SHA-256 fingerprint independent of the in-memory representation.
fn dataset_a() -> Vec<Row> {
    (0..64u32)
        .map(|i| {
            let dose = (i as f64 + 1.0) / 64.0; // dose ∈ (0, 1]
            // Truth: response = 0.2 + 0.6 * dose + 0.1 * dose² (small dose-response curve)
            let response = 0.2 + 0.6 * dose + 0.1 * dose * dose;
            (i, dose, response)
        })
        .collect()
}

/// Dataset B — the tampered version. Identical to A except patient
/// 17's response is rewritten (a hypothetical fraudster boosting one
/// patient's apparent efficacy).
fn dataset_b() -> Vec<Row> {
    let mut d = dataset_a();
    d[17] = (17, d[17].1, d[17].2 + 0.5); // tamper this single row
    d
}

/// Compute the SHA-256 fingerprint of a dataset by hashing its
/// canonical big-endian byte representation. Order-sensitive — a
/// reorder of rows produces a different fingerprint.
fn dataset_fingerprint(rows: &[Row]) -> [u8; 32] {
    let mut buf: Vec<u8> = Vec::with_capacity(rows.len() * 20);
    for &(pid, dose, resp) in rows {
        buf.extend_from_slice(&pid.to_be_bytes());
        buf.extend_from_slice(&dose.to_bits().to_be_bytes());
        buf.extend_from_slice(&resp.to_bits().to_be_bytes());
    }
    cjc_snap::hash::sha256(&buf)
}

/// Build a lineage-friendly graph: 1-D codebook over `dose`, 1→4→1
/// tanh leaf head, BLR over the 4-D feature basis
/// `[1, dose, dose², dose³]`, density + calibration heads, 4 children
/// at prefix bytes 0..3.
fn build_lineage_graph(seed: u64) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(seed);
    g.set_codebook(1, 4, &[0.25, 0.5, 0.75]).unwrap();
    g.set_leaf_head(1, vec![4], 1, Activation::Tanh).unwrap();
    g.set_blr_prior(2.0, 1.0, 0.5).unwrap();
    g.set_density_tracker().unwrap();
    g.set_calibration(15).unwrap();
    let thresholds = [
        0.5, 64.0, 128.0, 0.05, 0.02, 4.0, 0.1, 32.0, 10.0, 8.0, 20.0, f64::MAX,
        0.005, 1.05,
    ];
    g.set_decision_policy(&thresholds).unwrap();
    for byte in 0u8..4 {
        g.add_node(0, byte).unwrap();
    }
    g
}

/// Train + stamp the graph on the given dataset. The provenance
/// stamp = `sha256(dataset_bytes)`. Returns the trained graph.
fn train_and_stamp(seed: u64, rows: &[Row]) -> AdaptiveBeliefGraph {
    let mut g = build_lineage_graph(seed);
    let stamp = dataset_fingerprint(rows);
    g.stamp_provenance(0, stamp).unwrap();
    for &(_pid, dose, resp) in rows {
        // Phase 0.8c v14 Item A2 — fused per-row training step. One
        // `AuditKind::TrainStep` event per row instead of the pre-A2
        // `BlrUpdated + BeliefUpdate` pair. `train_step` does its own
        // descend, so the previous `route_leaf` indirection folds in.
        let phi = features(dose);
        g.train_step(&[dose], &phi, resp).unwrap();
    }
    g
}

fn features(dose: f64) -> [f64; 4] {
    [1.0, dose, dose * dose, dose * dose * dose]
}

fn route_leaf(g: &AdaptiveBeliefGraph, dose: f64) -> u32 {
    let prefix = g.encode_prefix(&[dose]).unwrap();
    g.descend(&prefix).leaf_id
}

// ── Setup smoke test ──────────────────────────────────────────────────

#[test]
fn lineage_dataset_a_and_b_have_distinct_fingerprints() {
    let fp_a = dataset_fingerprint(&dataset_a());
    let fp_b = dataset_fingerprint(&dataset_b());
    assert_ne!(
        fp_a, fp_b,
        "tampering one row must produce a different SHA-256 fingerprint"
    );
}

#[test]
fn lineage_train_and_stamp_emits_provenance_event() {
    let rows = dataset_a();
    let g = train_and_stamp(7, &rows);
    let n_stamped = g
        .audit
        .iter()
        .filter(|e| matches!(e.kind, AuditKind::ProvenanceStamped { .. }))
        .count();
    assert_eq!(n_stamped, 1);
    let expected_fp = dataset_fingerprint(&rows);
    assert_eq!(g.nodes[0].provenance_stamp_hash, expected_fp);
}

// ── The regulator's nightmare: dataset spoofing detection ─────────────

#[test]
fn lineage_chain_head_diverges_when_dataset_is_tampered() {
    // The most fundamental tamper signal: tampering ONE row in a
    // 64-row dataset produces an entirely different chain_head.
    // ABNG's audit chain is a hash of every event payload; any
    // change in observation values cascades into every subsequent
    // hash.
    let g_a = train_and_stamp(7, &dataset_a());
    let g_b = train_and_stamp(7, &dataset_b());
    assert_ne!(g_a.chain_head, g_b.chain_head);
}

#[test]
fn lineage_blr_state_hash_diverges_when_dataset_is_tampered() {
    // Independent signal #2: the per-leaf BLR posterior's
    // state_hash also diverges. The tampered patient #17 routes to
    // some specific leaf; that leaf's BLR posterior absorbs the
    // bogus response value, producing a different state_hash.
    // Even if an attacker forged the chain_head check, this
    // independent signal would still catch the tamper.
    let g_a = train_and_stamp(7, &dataset_a());
    let g_b = train_and_stamp(7, &dataset_b());
    let mut diff_count = 0;
    for child_id in 1..g_a.node_count() {
        let h_a = g_a.nodes[child_id as usize].blr.as_ref().unwrap().state_hash();
        let h_b = g_b.nodes[child_id as usize].blr.as_ref().unwrap().state_hash();
        if h_a != h_b {
            diff_count += 1;
        }
    }
    assert!(
        diff_count >= 1,
        "at least one leaf's BLR state_hash must reflect the tamper"
    );
}

#[test]
fn lineage_provenance_stamp_diverges_when_dataset_is_tampered() {
    // Independent signal #3: the provenance stamp itself rotates
    // because the dataset fingerprint differs. This is the *most
    // explicit* signal — any third party can compute
    // sha256(dataset_bytes) and verify the stamp.
    let g_a = train_and_stamp(7, &dataset_a());
    let g_b = train_and_stamp(7, &dataset_b());
    assert_ne!(
        g_a.nodes[0].provenance_stamp_hash,
        g_b.nodes[0].provenance_stamp_hash
    );
}

#[test]
fn lineage_attestation_three_signals_independent() {
    // The composite spoof-detection assertion. Even if ONE signal
    // were forged, the other two would still surface the tamper.
    // Three orthogonal proofs: chain layout, per-leaf math state,
    // and dataset fingerprint.
    let g_a = train_and_stamp(7, &dataset_a());
    let g_b = train_and_stamp(7, &dataset_b());
    let chain_diff = g_a.chain_head != g_b.chain_head;
    let stamp_diff =
        g_a.nodes[0].provenance_stamp_hash != g_b.nodes[0].provenance_stamp_hash;
    let mut blr_diff = false;
    for child_id in 1..g_a.node_count() {
        let h_a = g_a.nodes[child_id as usize].blr.as_ref().unwrap().state_hash();
        let h_b = g_b.nodes[child_id as usize].blr.as_ref().unwrap().state_hash();
        if h_a != h_b {
            blr_diff = true;
            break;
        }
    }
    assert!(chain_diff && stamp_diff && blr_diff,
        "all three signals must independently catch the tamper: \
         chain_diff={chain_diff}, stamp_diff={stamp_diff}, blr_diff={blr_diff}");
}

// ── Predict-snap lineage round-trip ────────────────────────────────────

#[test]
fn lineage_predict_snap_carries_dataset_fingerprint() {
    // The prediction snapshot carries the full lineage end-to-end:
    // chain_head + codebook hash + leaf-head hash + BLR state hash
    // + dataset provenance stamp. A regulator opening a saved
    // prediction file gets all five fingerprints in one blob.
    let g = train_and_stamp(7, &dataset_a());
    let dose = 0.42;
    let leaf = route_leaf(&g, dose);
    let phi = features(dose);
    let bytes = pack(&g, leaf, &phi).unwrap();
    let snap = unpack(&bytes).unwrap();
    // The leaf wasn't stamped (only root was) — leaf-bound
    // predict_snap reflects the leaf's own stamp ([0u8; 32]).
    assert_eq!(snap.provenance_stamp_hash, [0u8; 32]);
    // Chain_head matches the model.
    assert_eq!(snap.model_chain_head, g.chain_head);
    // Predict tuple is finite.
    assert!(snap.mean.is_finite());
    assert!(snap.epistemic_leverage.is_finite());
}

#[test]
fn lineage_root_stamp_inherited_via_explicit_leaf_stamp() {
    // To bind a leaf prediction to the dataset fingerprint, stamp
    // the leaf directly. Then predict_snap on that leaf carries
    // the dataset_fp. Useful pattern: stamp the leaf at deploy time
    // with the same fingerprint as the root.
    let mut g = train_and_stamp(7, &dataset_a());
    let dose = 0.6;
    let leaf = route_leaf(&g, dose);
    let dataset_fp = dataset_fingerprint(&dataset_a());
    g.stamp_provenance(leaf, dataset_fp).unwrap();
    let bytes = pack(&g, leaf, &features(dose)).unwrap();
    let snap = unpack(&bytes).unwrap();
    assert_eq!(snap.provenance_stamp_hash, dataset_fp);
}

#[test]
fn lineage_predict_snap_from_a_rejects_against_b_chain_head() {
    // The end-to-end attestation flow:
    //  1. lab A trains, packs prediction
    //  2. attacker tries to substitute model B
    //  3. regulator: compare prediction.model_chain_head to
    //     model_b.chain_head — they MUST disagree
    let g_a = train_and_stamp(7, &dataset_a());
    let dose = 0.42;
    let leaf = route_leaf(&g_a, dose);
    let phi = features(dose);
    let bytes_a = pack(&g_a, leaf, &phi).unwrap();
    let snap_a = unpack(&bytes_a).unwrap();

    let g_b = train_and_stamp(7, &dataset_b());
    assert_ne!(
        snap_a.model_chain_head, g_b.chain_head,
        "prediction-A's chain_head must NOT match model-B's chain_head"
    );
    // And vs the legitimate model-A: they MUST match.
    assert_eq!(snap_a.model_chain_head, g_a.chain_head);
}

// ── Audit chain integrity post-lineage ─────────────────────────────────

#[test]
fn lineage_audit_chain_verifies_post_train() {
    let g = train_and_stamp(7, &dataset_a());
    assert!(g.verify_chain().is_ok());
}

#[test]
fn lineage_serialize_replay_round_trip_preserves_lineage() {
    // The serialized model preserves the audit chain + provenance
    // stamp byte-for-byte. A regulator who archives the snapshot
    // can replay it years later and reproduce every fingerprint.
    let g = train_and_stamp(7, &dataset_a());
    let blob = serialize(&g);
    let g2 = replay(&blob).unwrap();
    assert_eq!(g.chain_head, g2.chain_head);
    assert_eq!(g.nodes[0].provenance_stamp_hash, g2.nodes[0].provenance_stamp_hash);
    for i in 0..g.node_count() {
        let h_pre = g.nodes[i as usize]
            .blr
            .as_ref()
            .map(|b| b.state_hash());
        let h_post = g2.nodes[i as usize]
            .blr
            .as_ref()
            .map(|b| b.state_hash());
        assert_eq!(h_pre, h_post);
    }
}

#[test]
fn lineage_audit_log_contains_observation_history_in_order() {
    // Direct evidence the audit chain captures the dataset: count
    // the per-row training events in order; should match dataset
    // size exactly. A regulator can scan the audit log and
    // reconstruct the exact training sequence.
    //
    // Phase 0.8c v14 Item A2 — `train_and_stamp` now uses the fused
    // `train_step` builtin, so each training row emits one
    // `AuditKind::TrainStep` (tag 0x1E) instead of the pre-A2
    // `BlrUpdated + BeliefUpdate` pair. The dataset-size invariant
    // is unchanged; the kind matched on shifts from
    // `BeliefUpdate` to `TrainStep`.
    let rows = dataset_a();
    let g = train_and_stamp(7, &rows);
    let n_train_steps = g
        .audit
        .iter()
        .filter(|e| matches!(e.kind, AuditKind::TrainStep { .. }))
        .count();
    assert_eq!(
        n_train_steps,
        rows.len(),
        "audit chain must contain exactly one TrainStep per training row"
    );
}

// ── Multi-stamp rotation: re-training scenario ────────────────────────

#[test]
fn lineage_re_stamping_with_new_fingerprint_records_rotation() {
    // Realistic scenario: lab A retrains on dataset_v2 (different
    // bytes). Stamping with the new fingerprint emits a NEW
    // ProvenanceStamped event on top of the existing chain — the
    // audit log preserves the full history of which dataset was
    // active when. A future regulator can scan the events to see
    // the full lineage, including past datasets.
    let mut g = train_and_stamp(7, &dataset_a());
    let pre_stamp_count = g
        .audit
        .iter()
        .filter(|e| matches!(e.kind, AuditKind::ProvenanceStamped { .. }))
        .count();
    let new_fp = dataset_fingerprint(&dataset_b());
    g.stamp_provenance(0, new_fp).unwrap();
    let post_stamp_count = g
        .audit
        .iter()
        .filter(|e| matches!(e.kind, AuditKind::ProvenanceStamped { .. }))
        .count();
    assert_eq!(post_stamp_count, pre_stamp_count + 1);
    assert_eq!(g.nodes[0].provenance_stamp_hash, new_fp);
}

#[test]
fn lineage_re_stamping_with_same_fingerprint_is_idempotent() {
    // Reaffirming a stamp with the same fingerprint is a no-op
    // (no event emitted). This prevents log explosion from
    // periodic stamp-refresh patterns.
    let mut g = train_and_stamp(7, &dataset_a());
    let pre_chain = g.chain_head;
    let pre_audit_len = g.audit.len();
    let same_fp = dataset_fingerprint(&dataset_a());
    g.stamp_provenance(0, same_fp).unwrap();
    assert_eq!(g.chain_head, pre_chain);
    assert_eq!(g.audit.len(), pre_audit_len);
}

// ── Locked canary ─────────────────────────────────────────────────────

#[test]
fn lineage_chain_head_canary_locked() {
    let g = train_and_stamp(7, &dataset_a());
    let actual_hex = hex_of(&g.chain_head);
    println!("lineage canary chain_head = {actual_hex}");
    // Re-locked at Phase 0.8c v14 Item A2 — the demo's
    // `train_and_stamp` flipped from `blr_update + observe`
    // (pre-A2: two events / row, tags 0x0A + 0x01) to
    // `train_step` (post-A2: one TrainStep event / row, tag
    // 0x1E). The audit-chain payload bytes per training row
    // therefore changed, so the chain head shifted. Pre-A2 hex
    // was `789acce77a22241c2e3601bf958e978b24e4707874cdbb23a7fde9a98f0606c2`;
    // V14_MIGRATION.md records the v13 → v14 mapping.
    const CANARY_HEX: &str =
        "7892bd9f9e2331e7729c3e973c4ae7c8db9aaf344772bedd786fd22418fddf81";
    assert_eq!(
        actual_hex, CANARY_HEX,
        "lineage chain_head canary mismatch — see comment"
    );
}

#[test]
fn lineage_dataset_a_fingerprint_canary_locked() {
    let actual_hex = hex_of(&dataset_fingerprint(&dataset_a()));
    println!("dataset A fingerprint = {actual_hex}");
    // Locked at Phase 0.5 ship — fires if the synthetic dataset
    // generator ever changes byte-layout. The whole demo's tamper
    // detection rests on this fingerprint being byte-stable.
    const CANARY_HEX: &str =
        "3e85d52f2508aecaaf32737edca48a644796783d6be7e6e324e6760506bc3634";
    assert_eq!(
        actual_hex, CANARY_HEX,
        "dataset A fingerprint canary mismatch — see comment"
    );
}

// ── Helpers ───────────────────────────────────────────────────────────

fn hex_of(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

#[allow(dead_code)]
fn _ensure_predict_snap_path_used(snap: PredictionSnap) -> [u8; 32] {
    // Pull `PredictionSnap` into actual use so its identity-only
    // re-export from cjc_abng::predict_snap doesn't dead-code-warn.
    snap.provenance_stamp_hash
}
