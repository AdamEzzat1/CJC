//! Content-addressed [`FingerprintId`] for an [`EffectEstimate`].
//!
//! Per ADR-0043 §4, the identifier is a SplitMix64-derived 64-bit fingerprint
//! over the canonical byte representation of:
//!
//! 1. Estimator type label (e.g. `"propensity_score_matcher"`)
//! 2. Treatment column name
//! 3. Outcome column name
//! 4. Covariate column names sorted ascending
//! 5. Assumptions sorted ascending (by discriminant ordering)
//! 6. Caller-supplied seed
//! 7. Point estimate bits ([`f64::to_bits`])
//! 8. Std error bits
//!
//! Two runs that produce the same canonical inputs produce byte-identical
//! identifiers. This is the publishable reproducibility claim.

use crate::assumption::IdentificationAssumption;
use cjc_locke::id::{fingerprint, fingerprint_compose, fingerprint_str, FingerprintId, IdDomain};

/// Compute the content-addressed identifier for an effect estimate.
///
/// The canonicalisation rule sorts `covariates` and `assumptions` ascending
/// so that two callers who supply the same multi-set get the same ID
/// regardless of order. Sorting `assumptions` uses `IdentificationAssumption`'s
/// derived `Ord` (variant discriminant order), which is deterministic across
/// runs and platforms.
pub fn compute_identifier(
    estimator_label: &str,
    treatment: &str,
    outcome: &str,
    covariates: &[&str],
    assumptions: &[IdentificationAssumption],
    seed: u64,
    point: f64,
    std_error: f64,
) -> FingerprintId {
    let mut sorted_covs: Vec<&str> = covariates.to_vec();
    sorted_covs.sort();

    let mut sorted_assumps: Vec<IdentificationAssumption> = assumptions.to_vec();
    sorted_assumps.sort();

    let mut parts: Vec<FingerprintId> = Vec::with_capacity(8 + sorted_covs.len() + sorted_assumps.len());
    parts.push(fingerprint_str(IdDomain::CausalClaim, estimator_label));
    parts.push(fingerprint_str(IdDomain::CausalClaim, treatment));
    parts.push(fingerprint_str(IdDomain::CausalClaim, outcome));
    for c in &sorted_covs {
        parts.push(fingerprint_str(IdDomain::CausalClaim, c));
    }
    for a in &sorted_assumps {
        parts.push(fingerprint(IdDomain::CausalClaim, &[*a as u8]));
    }
    parts.push(fingerprint(IdDomain::CausalClaim, &seed.to_le_bytes()));
    parts.push(fingerprint(IdDomain::CausalClaim, &point.to_bits().to_le_bytes()));
    parts.push(fingerprint(IdDomain::CausalClaim, &std_error.to_bits().to_le_bytes()));

    fingerprint_compose(IdDomain::CausalClaim, "effect_estimate", &parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (String, String, String, Vec<&'static str>, Vec<IdentificationAssumption>, u64, f64, f64) {
        (
            "propensity_score_matcher".to_string(),
            "treatment".to_string(),
            "outcome".to_string(),
            vec!["age", "income"],
            vec![IdentificationAssumption::Unconfoundedness, IdentificationAssumption::Positivity],
            42,
            0.5,
            0.1,
        )
    }

    #[test]
    fn identical_inputs_produce_identical_id() {
        let (e, t, o, c, a, s, p, se) = fixture();
        let id1 = compute_identifier(&e, &t, &o, &c, &a, s, p, se);
        let id2 = compute_identifier(&e, &t, &o, &c, &a, s, p, se);
        assert_eq!(id1, id2, "same inputs must produce same identifier");
    }

    #[test]
    fn different_seed_produces_different_id() {
        let (e, t, o, c, a, _s, p, se) = fixture();
        let id1 = compute_identifier(&e, &t, &o, &c, &a, 42, p, se);
        let id2 = compute_identifier(&e, &t, &o, &c, &a, 43, p, se);
        assert_ne!(id1, id2);
    }

    #[test]
    fn different_point_produces_different_id() {
        let (e, t, o, c, a, s, _p, se) = fixture();
        let id1 = compute_identifier(&e, &t, &o, &c, &a, s, 0.5, se);
        let id2 = compute_identifier(&e, &t, &o, &c, &a, s, 0.6, se);
        assert_ne!(id1, id2);
    }

    #[test]
    fn different_std_error_produces_different_id() {
        let (e, t, o, c, a, s, p, _se) = fixture();
        let id1 = compute_identifier(&e, &t, &o, &c, &a, s, p, 0.1);
        let id2 = compute_identifier(&e, &t, &o, &c, &a, s, p, 0.2);
        assert_ne!(id1, id2);
    }

    #[test]
    fn covariate_order_is_canonicalised() {
        let (e, t, o, _c, a, s, p, se) = fixture();
        let id_ab = compute_identifier(&e, &t, &o, &["age", "income"], &a, s, p, se);
        let id_ba = compute_identifier(&e, &t, &o, &["income", "age"], &a, s, p, se);
        assert_eq!(id_ab, id_ba, "covariate order must not change identifier");
    }

    #[test]
    fn assumption_order_is_canonicalised() {
        let (e, t, o, c, _a, s, p, se) = fixture();
        let id_ab = compute_identifier(
            &e, &t, &o, &c,
            &[IdentificationAssumption::Unconfoundedness, IdentificationAssumption::Positivity],
            s, p, se,
        );
        let id_ba = compute_identifier(
            &e, &t, &o, &c,
            &[IdentificationAssumption::Positivity, IdentificationAssumption::Unconfoundedness],
            s, p, se,
        );
        assert_eq!(id_ab, id_ba, "assumption order must not change identifier");
    }

    #[test]
    fn different_estimator_label_produces_different_id() {
        let (_e, t, o, c, a, s, p, se) = fixture();
        let id1 = compute_identifier("propensity_score_matcher", &t, &o, &c, &a, s, p, se);
        let id2 = compute_identifier("iv_regression", &t, &o, &c, &a, s, p, se);
        assert_ne!(id1, id2);
    }

    #[test]
    fn nan_point_is_a_stable_input() {
        let (e, t, o, c, a, s, _p, se) = fixture();
        // f64::NAN.to_bits() is a fixed bit pattern, so the hash is reproducible.
        let id1 = compute_identifier(&e, &t, &o, &c, &a, s, f64::NAN, se);
        let id2 = compute_identifier(&e, &t, &o, &c, &a, s, f64::NAN, se);
        assert_eq!(id1, id2, "NaN inputs must still hash deterministically");
    }
}
