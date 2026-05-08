//! Phase 0.4 Track C-2.3.2 — boundary validation: reject non-finite f64
//! inputs at every public observe / update boundary.
//!
//! Without these checks a single NaN or Inf observation poisons Welford
//! state forever, the audit chain hashes to a stable but-poisoned value,
//! and replay silently passes — a silent corruption mode that survives
//! across runs and is invisible to existing determinism gates.
//!
//! This file pins the contract for the four affected boundaries:
//!
//! - `AdaptiveBeliefGraph::observe`             → `GraphError::ObserveNonFinite`
//! - `AdaptiveBeliefGraph::density_observe`     → `DensityError::NonFiniteInput`
//! - `AdaptiveBeliefGraph::blr_update`          → `BlrError::NonFiniteInput`
//! - `AdaptiveBeliefGraph::calibration_observe` → `CalibrationError::InvalidProbability`
//!   (already enforced pre-Phase-0.4; tests here pin the contract)
//!
//! Each rejected call must leave `audit_len`, `chain_head`, and
//! `verify_chain()` identical to pre-call so the chain can never witness
//! a partially-applied non-finite observation.

use cjc_abng::audit::AuditKind;
use cjc_abng::blr::BlrError;
use cjc_abng::calibration::CalibrationError;
use cjc_abng::density::DensityError;
use cjc_abng::graph::{AdaptiveBeliefGraph, GraphError};
use cjc_abng::serialize::{replay, serialize};
use cjc_ad::pinn::Activation;

const NAN: f64 = f64::NAN;
const POS_INF: f64 = f64::INFINITY;
const NEG_INF: f64 = f64::NEG_INFINITY;

fn build_full_stack() -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(7);
    g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
    g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
    g.set_density_tracker().unwrap();
    g.set_calibration(15).unwrap();
    g
}

/// Snapshot of state expected to be unchanged across a rejected
/// boundary call.
struct PreCallSnapshot {
    audit_len: u64,
    chain_head: [u8; 32],
}

impl PreCallSnapshot {
    fn capture(g: &AdaptiveBeliefGraph) -> Self {
        Self {
            audit_len: g.audit_len(),
            chain_head: g.chain_head,
        }
    }

    fn assert_unchanged(&self, g: &AdaptiveBeliefGraph) {
        assert_eq!(g.audit_len(), self.audit_len, "audit_len changed");
        assert_eq!(g.chain_head, self.chain_head, "chain_head changed");
        g.verify_chain()
            .expect("chain still verifies after rejected call");
    }
}

// ── observe ────────────────────────────────────────────────────────────

#[test]
fn observe_rejects_nan_unchanged_state() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let pre = PreCallSnapshot::capture(&g);
    let err = g.observe(0, NAN).unwrap_err();
    assert!(
        matches!(err, GraphError::ObserveNonFinite { value } if value.is_nan()),
        "expected ObserveNonFinite(NaN), got {err:?}"
    );
    pre.assert_unchanged(&g);
}

#[test]
fn observe_rejects_pos_inf_unchanged_state() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let pre = PreCallSnapshot::capture(&g);
    let err = g.observe(0, POS_INF).unwrap_err();
    assert_eq!(err, GraphError::ObserveNonFinite { value: POS_INF });
    pre.assert_unchanged(&g);
}

#[test]
fn observe_rejects_neg_inf_unchanged_state() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let pre = PreCallSnapshot::capture(&g);
    let err = g.observe(0, NEG_INF).unwrap_err();
    assert_eq!(err, GraphError::ObserveNonFinite { value: NEG_INF });
    pre.assert_unchanged(&g);
}

#[test]
fn observe_rejects_check_runs_after_node_id_check() {
    // node_id 99 is out of range; even with a NaN value the bad-id error
    // surfaces first. Pins the validation order documented on
    // `AdaptiveBeliefGraph::observe`.
    let mut g = AdaptiveBeliefGraph::new(0);
    let err = g.observe(99, NAN).unwrap_err();
    assert!(matches!(err, GraphError::NodeOutOfRange { .. }));
}

#[test]
fn observe_slice_rejects_nan_mid_batch_partial_apply() {
    // observe_slice loops over observe(); the good prefix applies, then
    // the NaN rejects. Audit grows by exactly the count of finite
    // values that came before the first non-finite one.
    let mut g = AdaptiveBeliefGraph::new(0);
    let audit_pre = g.audit_len();
    let err = g.observe_slice(0, &[1.0, 2.0, NAN, 5.0]).unwrap_err();
    assert!(matches!(err, GraphError::ObserveNonFinite { .. }));
    assert_eq!(g.audit_len(), audit_pre + 2, "two good observations applied");
    g.verify_chain().expect("chain verifies after partial slice");
}

// ── density_observe ────────────────────────────────────────────────────

#[test]
fn density_observe_rejects_nan_unchanged_state() {
    let mut g = build_full_stack();
    let pre = PreCallSnapshot::capture(&g);
    let err = g.density_observe(0, &[NAN, 1.0]).unwrap_err();
    assert!(
        matches!(
            err,
            GraphError::Density(DensityError::NonFiniteInput { value }) if value.is_nan()
        ),
        "expected Density(NonFiniteInput(NaN)), got {err:?}"
    );
    pre.assert_unchanged(&g);
}

#[test]
fn density_observe_rejects_pos_inf_unchanged_state() {
    let mut g = build_full_stack();
    let pre = PreCallSnapshot::capture(&g);
    let err = g.density_observe(0, &[POS_INF, 1.0]).unwrap_err();
    assert_eq!(
        err,
        GraphError::Density(DensityError::NonFiniteInput { value: POS_INF })
    );
    pre.assert_unchanged(&g);
}

#[test]
fn density_observe_rejects_neg_inf_unchanged_state() {
    let mut g = build_full_stack();
    let pre = PreCallSnapshot::capture(&g);
    let err = g.density_observe(0, &[1.0, NEG_INF]).unwrap_err();
    assert_eq!(
        err,
        GraphError::Density(DensityError::NonFiniteInput { value: NEG_INF })
    );
    pre.assert_unchanged(&g);
}

#[test]
fn density_observe_rejects_nan_after_clean_batch_unchanged_state() {
    // Confirm rejection works after legitimate data has accrued —
    // n_seen and Welford state are not touched by the rejected call.
    let mut g = build_full_stack();
    g.density_observe(0, &[1.0, 1.0, 2.0, 2.0]).unwrap();
    let n_before = g.density_n_seen(0).unwrap();
    let pre = PreCallSnapshot::capture(&g);
    let err = g.density_observe(0, &[3.0, 3.0, NAN, 4.0]).unwrap_err();
    assert!(matches!(
        err,
        GraphError::Density(DensityError::NonFiniteInput { .. })
    ));
    assert_eq!(
        g.density_n_seen(0).unwrap(),
        n_before,
        "n_seen unchanged after rejection"
    );
    pre.assert_unchanged(&g);
}

// ── calibration_observe (already protected — pin the existing contract) ─

#[test]
fn calibration_observe_rejects_nan_unchanged_state() {
    let mut g = build_full_stack();
    let pre = PreCallSnapshot::capture(&g);
    let err = g.calibration_observe(0, NAN, true).unwrap_err();
    assert!(
        matches!(
            err,
            GraphError::Calibration(CalibrationError::InvalidProbability(p)) if p.is_nan()
        ),
        "expected Calibration(InvalidProbability(NaN)), got {err:?}"
    );
    pre.assert_unchanged(&g);
}

#[test]
fn calibration_observe_rejects_pos_inf_unchanged_state() {
    let mut g = build_full_stack();
    let pre = PreCallSnapshot::capture(&g);
    let err = g.calibration_observe(0, POS_INF, true).unwrap_err();
    assert_eq!(
        err,
        GraphError::Calibration(CalibrationError::InvalidProbability(POS_INF))
    );
    pre.assert_unchanged(&g);
}

#[test]
fn calibration_observe_rejects_neg_inf_unchanged_state() {
    let mut g = build_full_stack();
    let pre = PreCallSnapshot::capture(&g);
    let err = g.calibration_observe(0, NEG_INF, false).unwrap_err();
    assert_eq!(
        err,
        GraphError::Calibration(CalibrationError::InvalidProbability(NEG_INF))
    );
    pre.assert_unchanged(&g);
}

// ── blr_update ─────────────────────────────────────────────────────────

#[test]
fn blr_update_rejects_nan_in_features_unchanged_state() {
    let mut g = build_full_stack();
    let pre = PreCallSnapshot::capture(&g);
    let err = g.blr_update(0, &[NAN, 0.5], &[1.0]).unwrap_err();
    assert!(
        matches!(
            err,
            GraphError::Blr(BlrError::NonFiniteInput { value }) if value.is_nan()
        ),
        "expected Blr(NonFiniteInput(NaN)), got {err:?}"
    );
    pre.assert_unchanged(&g);
}

#[test]
fn blr_update_rejects_pos_inf_in_features_unchanged_state() {
    let mut g = build_full_stack();
    let pre = PreCallSnapshot::capture(&g);
    let err = g.blr_update(0, &[POS_INF, 0.5], &[1.0]).unwrap_err();
    assert_eq!(
        err,
        GraphError::Blr(BlrError::NonFiniteInput { value: POS_INF })
    );
    pre.assert_unchanged(&g);
}

#[test]
fn blr_update_rejects_neg_inf_in_features_unchanged_state() {
    let mut g = build_full_stack();
    let pre = PreCallSnapshot::capture(&g);
    let err = g.blr_update(0, &[0.5, NEG_INF], &[1.0]).unwrap_err();
    assert_eq!(
        err,
        GraphError::Blr(BlrError::NonFiniteInput { value: NEG_INF })
    );
    pre.assert_unchanged(&g);
}

#[test]
fn blr_update_rejects_nan_in_y_unchanged_state() {
    let mut g = build_full_stack();
    let pre = PreCallSnapshot::capture(&g);
    let err = g.blr_update(0, &[1.0, 2.0], &[NAN]).unwrap_err();
    assert!(matches!(
        err,
        GraphError::Blr(BlrError::NonFiniteInput { value }) if value.is_nan()
    ));
    pre.assert_unchanged(&g);
}

#[test]
fn blr_update_rejects_pos_inf_in_y_unchanged_state() {
    let mut g = build_full_stack();
    let pre = PreCallSnapshot::capture(&g);
    let err = g.blr_update(0, &[1.0, 2.0], &[POS_INF]).unwrap_err();
    assert_eq!(
        err,
        GraphError::Blr(BlrError::NonFiniteInput { value: POS_INF })
    );
    pre.assert_unchanged(&g);
}

#[test]
fn blr_update_rejects_neg_inf_in_y_unchanged_state() {
    let mut g = build_full_stack();
    let pre = PreCallSnapshot::capture(&g);
    let err = g.blr_update(0, &[1.0, 2.0], &[NEG_INF]).unwrap_err();
    assert_eq!(
        err,
        GraphError::Blr(BlrError::NonFiniteInput { value: NEG_INF })
    );
    pre.assert_unchanged(&g);
}

#[test]
fn blr_update_rejects_nan_after_clean_batch_unchanged_state() {
    // n_seen advances by the clean batch only; the rejected call
    // preserves n_seen and the chain head.
    let mut g = build_full_stack();
    g.blr_update(0, &[1.0, 0.5, 0.5, 1.0], &[1.0, 1.0]).unwrap();
    let n_before = g.blr_n_seen(0).unwrap();
    let pre = PreCallSnapshot::capture(&g);
    let err = g
        .blr_update(0, &[1.0, 0.5, NAN, 1.0], &[2.0, 2.0])
        .unwrap_err();
    assert!(matches!(
        err,
        GraphError::Blr(BlrError::NonFiniteInput { .. })
    ));
    assert_eq!(
        g.blr_n_seen(0).unwrap(),
        n_before,
        "n_seen unchanged after rejection"
    );
    pre.assert_unchanged(&g);
}

// ── replay sanity: rejected calls don't leak into snapshots ────────────

#[test]
fn rejected_calls_do_not_pollute_snapshot_replay() {
    // The most important integration test: pepper a training session
    // with a mix of valid and non-finite calls, then snapshot+replay.
    // The replayed graph's chain head must match the live graph's,
    // proving the rejected calls never participated in chain hashing.
    let mut g = build_full_stack();
    g.observe(0, 1.0).unwrap();

    // Each of these must Err and leave the chain alone.
    assert!(g.observe(0, NAN).is_err());
    assert!(g.observe(0, POS_INF).is_err());
    assert!(g.density_observe(0, &[NAN, 1.0]).is_err());
    assert!(g.density_observe(0, &[2.0, NEG_INF]).is_err());
    assert!(g.blr_update(0, &[1.0, NAN], &[1.0]).is_err());
    assert!(g.blr_update(0, &[1.0, 2.0], &[NAN]).is_err());
    assert!(g.calibration_observe(0, NAN, true).is_err());

    // Sandwich the rejected calls between two valid observes.
    g.observe(0, 2.0).unwrap();

    let head_before = g.chain_head;
    let audit_before = g.audit_len();

    let bytes = serialize(&g);
    let g2 = replay(&bytes).unwrap();

    assert_eq!(g2.chain_head, head_before, "replay chain head matches");
    assert_eq!(g2.audit_len(), audit_before, "audit length matches");
    g2.verify_chain().expect("replayed chain verifies");

    // Exactly two BeliefUpdate events — the rejected ones never
    // touched the audit log.
    let n_belief = g
        .audit
        .iter()
        .filter(|e| matches!(e.kind, AuditKind::BeliefUpdate { .. }))
        .count();
    assert_eq!(n_belief, 2, "rejected observes are absent from audit");
}
