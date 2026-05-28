//! Formal composition algebra over [`BeliefScore`] (v0.7).
//!
//! The v0.5 blog post and ADRs 0033 / 0034 document `(BeliefScore, meet, ⊤)`
//! as a commutative idempotent monoid with identity — i.e. a meet-semilattice
//! under component-wise `min`. This module formalises that claim in code:
//!
//! * The default composition is component-wise `min` on every axis.
//! * A `BeliefAxisRules` struct lets callers override the per-axis composition
//!   for use cases where `min` isn't the right semantics (e.g. when combining
//!   sub-leaf samples, `ArithmeticMean` over `sample_score` is more faithful
//!   than `min`).
//! * Proptest-locked laws (see `tests/locke/locke_proptest.rs`) verify the
//!   meet-semilattice properties hold under the default rules and that
//!   alternative rules satisfy their weaker-but-stated guarantees.
//!
//! ## Laws (default rules — all `Min`)
//!
//! For any belief scores `a`, `b`, `c` and the identity `⊤ = [1, 1, …, 1]`:
//!
//! | Law | Statement |
//! |---|---|
//! | Identity | `compose(b, ⊤) = b` |
//! | Idempotence | `compose(b, b) = b` |
//! | Commutativity | `compose(b, c) = compose(c, b)` |
//! | Associativity | `compose(compose(a, b), c) = compose(a, compose(b, c))` |
//! | Monotonicity | `compose(b, c) ≤ b` component-wise |
//!
//! Together these make `(BeliefScore, compose, ⊤)` a meet-semilattice.
//!
//! ## Non-default rules
//!
//! Other rules satisfy strictly weaker law sets — documented per variant.
//! In particular, `GeometricMean` and `ArithmeticMean` are *not idempotent*
//! except at boundary values `{0, 1}`, so they break the meet-semilattice
//! structure. That's fine — they're for use cases (aggregating sub-views,
//! weighted leaf combinations) where the meet semantics is the wrong model.

use crate::belief::{BeliefScore, BeliefWeights};

// ─── Composition rule per axis ────────────────────────────────────────────

/// How two axis values should be combined when composing belief scores.
///
/// `Min` is the default; the other variants are available for callers
/// who need different semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompositionRule {
    /// Component-wise `min`. Satisfies: identity (1), idempotence, commutativity,
    /// associativity, monotonicity. The meet-semilattice operator.
    Min,
    /// Component-wise `max`. Satisfies: identity (0), idempotence, commutativity,
    /// associativity. Breaks the meet-monotonicity (`compose(b, c) ≥ b`); the
    /// dual semilattice (join). Useful as a diagnostic — "if I take the best
    /// of each axis, what would the score be?"
    Max,
    /// Geometric mean: `sqrt(a * b)`. Satisfies: identity (1, vacuously since
    /// `sqrt(a)` ≠ `a`), commutativity, associativity. NOT idempotent except
    /// at `{0, 1}`. Useful for compositions where independent evidence units
    /// should multiply down.
    GeometricMean,
    /// Arithmetic mean: `(a + b) / 2`. Satisfies: idempotence, commutativity.
    /// NOT associative for 3+ inputs unless they're equal (the 3-input mean
    /// `((a+b)/2 + c)/2` ≠ `(a + (b+c)/2)/2` in general). NOT a true identity
    /// element. Useful for "average quality across N sub-views"; for that
    /// case prefer [`compose_many_arithmetic`] which averages all inputs at
    /// once rather than chaining pairwise.
    ArithmeticMean,
}

impl CompositionRule {
    /// Apply the rule to a single pair of axis values, both expected in `[0, 1]`.
    /// Inputs are clamped to `[0, 1]` defensively; outputs are guaranteed in `[0, 1]`.
    #[inline]
    pub fn apply(self, a: f64, b: f64) -> f64 {
        let a = a.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);
        let v = match self {
            CompositionRule::Min => a.min(b),
            CompositionRule::Max => a.max(b),
            CompositionRule::GeometricMean => (a * b).sqrt(),
            CompositionRule::ArithmeticMean => 0.5 * (a + b),
        };
        // Defensive clamp — sqrt of clamped inputs cannot escape [0, 1] but
        // floating arithmetic on extreme inputs may produce subnormal noise.
        v.clamp(0.0, 1.0)
    }

    /// True iff this rule is **idempotent** (`apply(x, x) = x`).
    /// `Min`, `Max` always; `GeometricMean` only at `{0, 1}`; `ArithmeticMean` always
    /// (since `(x + x)/2 = x`). Used by the meet-semilattice law tests to
    /// scope what gets asserted under each rule.
    #[inline]
    pub fn is_idempotent_everywhere(self) -> bool {
        matches!(self, CompositionRule::Min | CompositionRule::Max | CompositionRule::ArithmeticMean)
    }

    /// True iff this rule is **monotonically downward** (`apply(a, b) ≤ a`).
    /// Only `Min` and `GeometricMean` (for `b ≤ 1`).
    #[inline]
    pub fn is_monotonic_down(self) -> bool {
        matches!(self, CompositionRule::Min | CompositionRule::GeometricMean)
    }

    /// True iff this rule is **associative** for arbitrary inputs.
    /// `Min`, `Max`, `GeometricMean` all are; `ArithmeticMean` is NOT
    /// (chaining gives different results than averaging-all-at-once).
    #[inline]
    pub fn is_associative(self) -> bool {
        matches!(
            self,
            CompositionRule::Min | CompositionRule::Max | CompositionRule::GeometricMean
        )
    }
}

/// Per-axis composition rules for [`BeliefScore`].
///
/// `Default::default()` returns all-`Min`, which preserves the documented
/// meet-semilattice algebra. Override per-axis for non-standard combinators.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BeliefAxisRules {
    pub schema: CompositionRule,
    pub missingness: CompositionRule,
    pub drift: CompositionRule,
    pub leakage: CompositionRule,
    pub lineage: CompositionRule,
    pub sample: CompositionRule,
    pub duplication: CompositionRule,
    pub constraint: CompositionRule,
}

impl Default for BeliefAxisRules {
    fn default() -> Self {
        Self::all_min()
    }
}

impl BeliefAxisRules {
    /// All axes use `Min`. Equivalent to the documented meet-semilattice algebra.
    pub fn all_min() -> Self {
        Self {
            schema: CompositionRule::Min,
            missingness: CompositionRule::Min,
            drift: CompositionRule::Min,
            leakage: CompositionRule::Min,
            lineage: CompositionRule::Min,
            sample: CompositionRule::Min,
            duplication: CompositionRule::Min,
            constraint: CompositionRule::Min,
        }
    }

    /// All axes use `Max` — the dual semilattice (join). Useful as a diagnostic.
    pub fn all_max() -> Self {
        Self {
            schema: CompositionRule::Max,
            missingness: CompositionRule::Max,
            drift: CompositionRule::Max,
            leakage: CompositionRule::Max,
            lineage: CompositionRule::Max,
            sample: CompositionRule::Max,
            duplication: CompositionRule::Max,
            constraint: CompositionRule::Max,
        }
    }

    /// All axes use `ArithmeticMean`. Useful for "average per-leaf belief".
    pub fn all_arithmetic_mean() -> Self {
        Self {
            schema: CompositionRule::ArithmeticMean,
            missingness: CompositionRule::ArithmeticMean,
            drift: CompositionRule::ArithmeticMean,
            leakage: CompositionRule::ArithmeticMean,
            lineage: CompositionRule::ArithmeticMean,
            sample: CompositionRule::ArithmeticMean,
            duplication: CompositionRule::ArithmeticMean,
            constraint: CompositionRule::ArithmeticMean,
        }
    }

    /// True iff every axis uses an idempotent rule. Required for the
    /// idempotence law of the meet-semilattice.
    pub fn is_idempotent_everywhere(&self) -> bool {
        self.schema.is_idempotent_everywhere()
            && self.missingness.is_idempotent_everywhere()
            && self.drift.is_idempotent_everywhere()
            && self.leakage.is_idempotent_everywhere()
            && self.lineage.is_idempotent_everywhere()
            && self.sample.is_idempotent_everywhere()
            && self.duplication.is_idempotent_everywhere()
            && self.constraint.is_idempotent_everywhere()
    }

    /// True iff every axis uses an associative rule. Required for the
    /// associativity law.
    pub fn is_associative(&self) -> bool {
        self.schema.is_associative()
            && self.missingness.is_associative()
            && self.drift.is_associative()
            && self.leakage.is_associative()
            && self.lineage.is_associative()
            && self.sample.is_associative()
            && self.duplication.is_associative()
            && self.constraint.is_associative()
    }
}

// ─── Identity element ────────────────────────────────────────────────────

/// The identity element `⊤ = [1, 1, 1, 1, 1, 1, 1, 1]` for `Min`-composition.
/// `compose(b, top()) = b` for every `b` under `BeliefAxisRules::all_min()`.
pub fn top() -> BeliefScore {
    BeliefScore::from_dimensions(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
}

/// The dual identity `⊥ = [0, 0, 0, 0, 0, 0, 0, 0]` for `Max`-composition.
pub fn bottom() -> BeliefScore {
    BeliefScore::from_dimensions(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
}

// ─── Composition ─────────────────────────────────────────────────────────

/// Compose two belief scores axis-by-axis under the supplied rules.
/// The `overall` of the result is recomputed via `BeliefScore::from_dimensions`
/// (default equal-weight mean) so it stays consistent with the per-axis values.
pub fn compose(a: &BeliefScore, b: &BeliefScore, rules: &BeliefAxisRules) -> BeliefScore {
    BeliefScore::from_dimensions(
        rules.schema.apply(a.schema_score, b.schema_score),
        rules.missingness.apply(a.missingness_score, b.missingness_score),
        rules.drift.apply(a.drift_score, b.drift_score),
        rules.leakage.apply(a.leakage_score, b.leakage_score),
        rules.lineage.apply(a.lineage_score, b.lineage_score),
        rules.sample.apply(a.sample_score, b.sample_score),
        rules.duplication.apply(a.duplication_score, b.duplication_score),
        rules.constraint.apply(a.constraint_score, b.constraint_score),
    )
}

/// Compose a slice of belief scores by chained pairwise reduction:
/// `compose(... compose(s[0], s[1]) ..., s[n-1])`. Returns `None` for
/// an empty slice. Note: for `ArithmeticMean` rules, chained pairwise is
/// NOT the same as a single all-at-once average — see [`compose_many_arithmetic`]
/// for the alternative.
pub fn compose_many(scores: &[BeliefScore], rules: &BeliefAxisRules) -> Option<BeliefScore> {
    if scores.is_empty() {
        return None;
    }
    let mut acc = scores[0].clone();
    for s in &scores[1..] {
        acc = compose(&acc, s, rules);
    }
    Some(acc)
}

/// Single-pass arithmetic-mean composition across `n` scores. Equivalent
/// to `BeliefScore::from_dimensions(mean(axis_0), mean(axis_1), …)`.
/// Use this when you want a true "average of N belief scores" and not
/// the pairwise chain that `compose_many` with `ArithmeticMean` would
/// produce.
pub fn compose_many_arithmetic(scores: &[BeliefScore]) -> Option<BeliefScore> {
    if scores.is_empty() {
        return None;
    }
    let n = scores.len() as f64;
    let mut schema = 0.0;
    let mut missing = 0.0;
    let mut drift = 0.0;
    let mut leakage = 0.0;
    let mut lineage = 0.0;
    let mut sample = 0.0;
    let mut duplication = 0.0;
    let mut constraint = 0.0;
    for s in scores {
        schema += s.schema_score.clamp(0.0, 1.0);
        missing += s.missingness_score.clamp(0.0, 1.0);
        drift += s.drift_score.clamp(0.0, 1.0);
        leakage += s.leakage_score.clamp(0.0, 1.0);
        lineage += s.lineage_score.clamp(0.0, 1.0);
        sample += s.sample_score.clamp(0.0, 1.0);
        duplication += s.duplication_score.clamp(0.0, 1.0);
        constraint += s.constraint_score.clamp(0.0, 1.0);
    }
    Some(BeliefScore::from_dimensions(
        schema / n,
        missing / n,
        drift / n,
        leakage / n,
        lineage / n,
        sample / n,
        duplication / n,
        constraint / n,
    ))
}

/// Weighted arithmetic mean over `n` scores with per-score weights (e.g.
/// row counts per leaf). Returns `None` for an empty slice or when all
/// weights are zero. Negative or NaN weights are clamped to 0.
pub fn compose_weighted(
    scores: &[BeliefScore],
    weights: &[f64],
) -> Option<BeliefScore> {
    if scores.is_empty() || scores.len() != weights.len() {
        return None;
    }
    let mut total_w = 0.0;
    let mut schema = 0.0;
    let mut missing = 0.0;
    let mut drift = 0.0;
    let mut leakage = 0.0;
    let mut lineage = 0.0;
    let mut sample = 0.0;
    let mut duplication = 0.0;
    let mut constraint = 0.0;
    for (s, w) in scores.iter().zip(weights.iter()) {
        let w = if w.is_finite() && *w > 0.0 { *w } else { 0.0 };
        if w == 0.0 {
            continue;
        }
        total_w += w;
        schema += w * s.schema_score.clamp(0.0, 1.0);
        missing += w * s.missingness_score.clamp(0.0, 1.0);
        drift += w * s.drift_score.clamp(0.0, 1.0);
        leakage += w * s.leakage_score.clamp(0.0, 1.0);
        lineage += w * s.lineage_score.clamp(0.0, 1.0);
        sample += w * s.sample_score.clamp(0.0, 1.0);
        duplication += w * s.duplication_score.clamp(0.0, 1.0);
        constraint += w * s.constraint_score.clamp(0.0, 1.0);
    }
    if total_w == 0.0 {
        return None;
    }
    Some(BeliefScore::from_dimensions(
        schema / total_w,
        missing / total_w,
        drift / total_w,
        leakage / total_w,
        lineage / total_w,
        sample / total_w,
        duplication / total_w,
        constraint / total_w,
    ))
}

// ─── Order relation (meet-semilattice) ───────────────────────────────────

/// Component-wise partial order on belief scores: `a ≤ b` iff every axis
/// of `a` is `≤` the corresponding axis of `b` (within `eps` for floating
/// tolerance). Forms the partial order under which `Min` is a meet.
pub fn le_componentwise(a: &BeliefScore, b: &BeliefScore, eps: f64) -> bool {
    a.schema_score <= b.schema_score + eps
        && a.missingness_score <= b.missingness_score + eps
        && a.drift_score <= b.drift_score + eps
        && a.leakage_score <= b.leakage_score + eps
        && a.lineage_score <= b.lineage_score + eps
        && a.sample_score <= b.sample_score + eps
        && a.duplication_score <= b.duplication_score + eps
        && a.constraint_score <= b.constraint_score + eps
}

/// True iff `a` and `b` are component-wise equal within `eps`. The
/// equality relation of the meet-semilattice.
pub fn eq_componentwise(a: &BeliefScore, b: &BeliefScore, eps: f64) -> bool {
    (a.schema_score - b.schema_score).abs() <= eps
        && (a.missingness_score - b.missingness_score).abs() <= eps
        && (a.drift_score - b.drift_score).abs() <= eps
        && (a.leakage_score - b.leakage_score).abs() <= eps
        && (a.lineage_score - b.lineage_score).abs() <= eps
        && (a.sample_score - b.sample_score).abs() <= eps
        && (a.duplication_score - b.duplication_score).abs() <= eps
        && (a.constraint_score - b.constraint_score).abs() <= eps
}

// ─── Re-export for ergonomic test access ─────────────────────────────────

pub use crate::belief::BeliefScore as _BeliefScore;
pub use crate::belief::BeliefWeights as _BeliefWeights;

// ─── Unit tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn bs(s: f64, m: f64, d: f64, l: f64, li: f64, sa: f64, du: f64, c: f64) -> BeliefScore {
        BeliefScore::from_dimensions(s, m, d, l, li, sa, du, c)
    }

    // ── CompositionRule.apply ──────────────────────────────────────────

    #[test]
    fn min_applies_componentwise() {
        assert_eq!(CompositionRule::Min.apply(0.3, 0.7), 0.3);
        assert_eq!(CompositionRule::Min.apply(0.7, 0.3), 0.3);
        assert_eq!(CompositionRule::Min.apply(0.5, 0.5), 0.5);
    }

    #[test]
    fn max_applies_componentwise() {
        assert_eq!(CompositionRule::Max.apply(0.3, 0.7), 0.7);
        assert_eq!(CompositionRule::Max.apply(0.7, 0.3), 0.7);
    }

    #[test]
    fn geometric_mean_works() {
        let v = CompositionRule::GeometricMean.apply(0.25, 0.64);
        assert!((v - 0.4).abs() < 1e-9, "got {}", v);
    }

    #[test]
    fn arithmetic_mean_works() {
        let v = CompositionRule::ArithmeticMean.apply(0.2, 0.8);
        assert!((v - 0.5).abs() < 1e-9, "got {}", v);
    }

    #[test]
    fn apply_clamps_inputs_to_unit_interval() {
        assert_eq!(CompositionRule::Min.apply(-0.5, 1.5), 0.0);
        assert_eq!(CompositionRule::Max.apply(-0.5, 1.5), 1.0);
    }

    // ── Meet-semilattice laws under default (all-Min) rules ────────────

    #[test]
    fn identity_with_top_returns_b() {
        let rules = BeliefAxisRules::default();
        let b = bs(0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.3, 0.2);
        let c = compose(&b, &top(), &rules);
        assert!(eq_componentwise(&b, &c, 1e-12));
    }

    #[test]
    fn idempotence_compose_b_b_equals_b() {
        let rules = BeliefAxisRules::default();
        let b = bs(0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.3, 0.2);
        let c = compose(&b, &b, &rules);
        assert!(eq_componentwise(&b, &c, 1e-12));
    }

    #[test]
    fn commutativity_compose_a_b_equals_compose_b_a() {
        let rules = BeliefAxisRules::default();
        let a = bs(0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.3, 0.2);
        let b = bs(0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7);
        let ab = compose(&a, &b, &rules);
        let ba = compose(&b, &a, &rules);
        assert!(eq_componentwise(&ab, &ba, 1e-12));
    }

    #[test]
    fn associativity_compose_chain_independent_of_grouping() {
        let rules = BeliefAxisRules::default();
        let a = bs(0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.3, 0.2);
        let b = bs(0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7);
        let c = bs(0.1, 0.9, 0.5, 0.5, 0.7, 0.2, 0.4, 0.6);
        let ab_then_c = compose(&compose(&a, &b, &rules), &c, &rules);
        let a_then_bc = compose(&a, &compose(&b, &c, &rules), &rules);
        assert!(eq_componentwise(&ab_then_c, &a_then_bc, 1e-12));
    }

    #[test]
    fn monotonicity_compose_b_c_le_b_componentwise() {
        let rules = BeliefAxisRules::default();
        let b = bs(0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.3, 0.2);
        let c = bs(0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7);
        let bc = compose(&b, &c, &rules);
        assert!(le_componentwise(&bc, &b, 1e-12));
        assert!(le_componentwise(&bc, &c, 1e-12));
    }

    // ── Alternative rule guarantees ────────────────────────────────────

    #[test]
    fn max_is_idempotent_and_associative() {
        let rules = BeliefAxisRules::all_max();
        let b = bs(0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.3, 0.2);
        assert!(eq_componentwise(&compose(&b, &b, &rules), &b, 1e-12));
        let c = bs(0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7);
        let d = bs(0.1, 0.9, 0.5, 0.5, 0.7, 0.2, 0.4, 0.6);
        let abc1 = compose(&compose(&b, &c, &rules), &d, &rules);
        let abc2 = compose(&b, &compose(&c, &d, &rules), &rules);
        assert!(eq_componentwise(&abc1, &abc2, 1e-12));
    }

    #[test]
    fn arithmetic_mean_not_associative_in_general() {
        let rules = BeliefAxisRules::all_arithmetic_mean();
        let a = bs(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let b = bs(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        let c = bs(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        let ab_then_c = compose(&compose(&a, &b, &rules), &c, &rules);
        let a_then_bc = compose(&a, &compose(&b, &c, &rules), &rules);
        // (((0+1)/2) + 1)/2 = 0.75 vs (0 + (1+1)/2)/2 = 0.5 — divergent
        assert!(!eq_componentwise(&ab_then_c, &a_then_bc, 1e-3));
    }

    // ── compose_many / compose_many_arithmetic / compose_weighted ──────

    #[test]
    fn compose_many_returns_none_on_empty() {
        let r = BeliefAxisRules::default();
        assert!(compose_many(&[], &r).is_none());
    }

    #[test]
    fn compose_many_equals_pairwise_chain() {
        let r = BeliefAxisRules::default();
        let a = bs(0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.3, 0.2);
        let b = bs(0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7);
        let c = bs(0.1, 0.9, 0.5, 0.5, 0.7, 0.2, 0.4, 0.6);
        let chained = compose(&compose(&a, &b, &r), &c, &r);
        let many = compose_many(&[a, b, c], &r).unwrap();
        assert!(eq_componentwise(&chained, &many, 1e-12));
    }

    #[test]
    fn compose_many_arithmetic_averages_all_at_once() {
        let a = bs(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let b = bs(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        let c = bs(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        let m = compose_many_arithmetic(&[a, b, c]).unwrap();
        // Mean of [0, 1, 1] = 2/3.
        assert!((m.schema_score - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn compose_weighted_falls_back_to_arithmetic_when_weights_equal() {
        let a = bs(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9);
        let b = bs(0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1);
        let unweighted = compose_many_arithmetic(&[a.clone(), b.clone()]).unwrap();
        let weighted = compose_weighted(&[a, b], &[5.0, 5.0]).unwrap();
        assert!(eq_componentwise(&unweighted, &weighted, 1e-12));
    }

    #[test]
    fn compose_weighted_handles_negative_and_nan_weights() {
        let a = bs(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9);
        let b = bs(0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1);
        // Negative weight on b → b ignored → result equals a.
        let w = compose_weighted(&[a.clone(), b], &[1.0, -2.0]).unwrap();
        assert!(eq_componentwise(&w, &a, 1e-12));
        // All weights zero / non-finite → None.
        let z = compose_weighted(&[bs(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)], &[f64::NAN]);
        assert!(z.is_none());
    }

    // ── Order relations ───────────────────────────────────────────────

    #[test]
    fn le_componentwise_handles_eps() {
        let a = bs(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5);
        let b = bs(0.5 - 1e-13, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5);
        // b's first axis is slightly less than a's, within eps.
        assert!(le_componentwise(&b, &a, 1e-12));
    }

    // ── Identity / dual identity ───────────────────────────────────────

    #[test]
    fn top_is_identity_for_all_min() {
        let r = BeliefAxisRules::default();
        let b = bs(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5);
        let c = compose(&b, &top(), &r);
        assert!(eq_componentwise(&b, &c, 1e-12));
    }

    #[test]
    fn bottom_is_identity_for_all_max() {
        let r = BeliefAxisRules::all_max();
        let b = bs(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5);
        let c = compose(&b, &bottom(), &r);
        assert!(eq_componentwise(&b, &c, 1e-12));
    }

    // ── Determinism ───────────────────────────────────────────────────

    #[test]
    fn compose_is_deterministic_across_runs() {
        let r = BeliefAxisRules::default();
        let a = bs(0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.3, 0.2);
        let b = bs(0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7);
        let r1 = compose(&a, &b, &r);
        let r2 = compose(&a, &b, &r);
        assert!(eq_componentwise(&r1, &r2, 0.0)); // bit-equal
    }
}
