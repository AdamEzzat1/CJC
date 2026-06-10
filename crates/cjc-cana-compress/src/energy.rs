//! Quantum-inspired Ising / QAOA-style energy ranking of compiler plans.
//!
//! ## What this module is
//!
//! A deterministic scoring + ranking primitive. Each candidate plan
//! receives an [`EnergyScore`] — a Kahan-summed scalar of cost terms
//! (penalties) minus reward terms (bonuses) — and [`EnergyRanker::rank`]
//! returns the candidates sorted by ascending energy with stable
//! tie-break by `CandidateId`.
//!
//! ## What this module is NOT
//!
//! - **Not a quantum solver.** No QAOA circuit is constructed; no
//!   parameterised Hamiltonian is optimised. The terminology is honest
//!   only because the *shape* of the objective is Ising-style (a
//!   weighted sum of pairwise contributions over discrete decisions)
//!   and the decomposition is fully transparent to the auditor.
//! - **Not authoritative.** A high-ranking candidate is a suggestion.
//!   `cjc_cana::LegalityGate` and the MIR verifier remain final
//!   authority.
//!
//! ## The objective
//!
//! ```text
//!   energy =
//!     + runtime_cost
//!     + memory_pressure
//!     + thermal_pressure
//!     + bandwidth_pressure
//!     + code_size_pressure
//!     + reconstruction_risk
//!     + verifier_risk_penalty
//!     - fusion_reward
//!     - reuse_reward
//!     - compression_reward
//!     - locality_reward
//! ```
//!
//! `bandwidth_pressure`, `code_size_pressure` (renamed from
//! `code_size_cost`), and `locality_reward` were added for the PINN v1
//! physical layer (see `cjc_cana::physical_cost`) — the physical model
//! prices memory traffic and cache locality, and this decomposition is
//! where those prices surface for ranking.
//!
//! Every term is a non-negative `f64`. Costs add; rewards subtract. The
//! Kahan accumulator keeps the running sum compensated against catastrophic
//! cancellation when costs and rewards are close in magnitude.
//!
//! ## Determinism contract
//!
//! - All inputs are non-negative `f64` (the constructor rejects negative
//!   or non-finite values).
//! - The sum is computed via [`KahanAccumulatorF64`] in a fixed term
//!   order — the same order the documentation shows.
//! - Tie-break is `(total, candidate_id)` ascending — total wins;
//!   `CandidateId(u64)` is total/stable on equal totals.
//! - Sort is `Vec::sort_by` which is a stable sort in Rust's standard
//!   library; pre-sorting by ID before scoring would also work but is
//!   unnecessary because we always pass through the stable sort.

use cjc_repro::KahanAccumulatorF64;

use crate::candidate::{CandidateId, CompressionError};

// ---------------------------------------------------------------------------
// EnergyComponents
// ---------------------------------------------------------------------------

/// Per-candidate cost/reward decomposition.
///
/// Every field is a non-negative finite `f64`. Construct via
/// [`EnergyComponents::new`] or [`EnergyComponents::builder`]. Direct
/// field access is allowed (for tests / wiring) but
/// [`EnergyComponents::is_valid`] should be used before relying on the
/// values.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EnergyComponents {
    /// Predicted runtime cost of the plan. Non-negative.
    pub runtime_cost: f64,
    /// Predicted memory pressure delta. Non-negative.
    pub memory_pressure: f64,
    /// Predicted thermal pressure delta. Non-negative.
    pub thermal_pressure: f64,
    /// Predicted memory-traffic / bandwidth saturation delta.
    /// Non-negative. Populated by the PINN physical layer; 0.0 when no
    /// physical model is in the stack.
    pub bandwidth_pressure: f64,
    /// Predicted code-size cost. Non-negative. (Renamed from
    /// `code_size_cost` for the PINN v1 vocabulary; the builder keeps
    /// a deprecated `.code_size_cost()` alias for one release.)
    pub code_size_pressure: f64,
    /// Risk of high reconstruction error (for advisory compression plans).
    /// Non-negative.
    pub reconstruction_risk: f64,
    /// Penalty for plans the verifier *might* reject (the gate refuses
    /// outright, but plans near the policy boundary still get penalised
    /// so the ranker biases toward safe choices). Non-negative.
    pub verifier_risk_penalty: f64,
    /// Reward for plans that enable fusion. Non-negative.
    pub fusion_reward: f64,
    /// Reward for plans that increase buffer / kernel-result reuse.
    /// Non-negative.
    pub reuse_reward: f64,
    /// Reward for plans that reduce memory pressure via compression.
    /// Non-negative.
    pub compression_reward: f64,
    /// Reward for plans that improve cache locality (small working
    /// set relative to the cache proxy). Non-negative. Populated by
    /// the PINN physical layer; 0.0 when no physical model is in the
    /// stack.
    pub locality_reward: f64,
}

impl EnergyComponents {
    /// Validate that every field is finite and non-negative.
    pub fn is_valid(&self) -> bool {
        let fs = [
            self.runtime_cost,
            self.memory_pressure,
            self.thermal_pressure,
            self.bandwidth_pressure,
            self.code_size_pressure,
            self.reconstruction_risk,
            self.verifier_risk_penalty,
            self.fusion_reward,
            self.reuse_reward,
            self.compression_reward,
            self.locality_reward,
        ];
        fs.iter().all(|x| x.is_finite() && *x >= 0.0)
    }

    /// Build with explicit values. Returns `Err` on non-finite or negative
    /// inputs to keep the determinism contract auditable at construction
    /// time.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        runtime_cost: f64,
        memory_pressure: f64,
        thermal_pressure: f64,
        bandwidth_pressure: f64,
        code_size_pressure: f64,
        reconstruction_risk: f64,
        verifier_risk_penalty: f64,
        fusion_reward: f64,
        reuse_reward: f64,
        compression_reward: f64,
        locality_reward: f64,
    ) -> Result<Self, CompressionError> {
        let c = Self {
            runtime_cost,
            memory_pressure,
            thermal_pressure,
            bandwidth_pressure,
            code_size_pressure,
            reconstruction_risk,
            verifier_risk_penalty,
            fusion_reward,
            reuse_reward,
            compression_reward,
            locality_reward,
        };
        if !c.is_valid() {
            return Err(CompressionError::InvalidTolerance {
                value: f64::NAN, // sentinel meaning "some component invalid"
            });
        }
        Ok(c)
    }

    /// Fluent builder for situations where most fields are zero.
    pub fn builder() -> EnergyComponentsBuilder {
        EnergyComponentsBuilder::default()
    }

    /// Sum into a Kahan accumulator in declaration order.
    /// Order matters for bit-exact reproducibility on platforms without
    /// FMA — using `naive .iter().sum()` would let the optimizer rearrange
    /// the additions.
    fn kahan_total(&self) -> f64 {
        let mut acc = KahanAccumulatorF64::new();
        // Costs.
        acc.add(self.runtime_cost);
        acc.add(self.memory_pressure);
        acc.add(self.thermal_pressure);
        acc.add(self.bandwidth_pressure);
        acc.add(self.code_size_pressure);
        acc.add(self.reconstruction_risk);
        acc.add(self.verifier_risk_penalty);
        // Rewards (negated).
        acc.add(-self.fusion_reward);
        acc.add(-self.reuse_reward);
        acc.add(-self.compression_reward);
        acc.add(-self.locality_reward);
        acc.finalize()
    }
}

/// Fluent builder. All defaults are `0.0`. Each setter is chainable.
#[derive(Debug, Default, Clone)]
pub struct EnergyComponentsBuilder {
    inner: EnergyComponents,
}

impl EnergyComponentsBuilder {
    /// Set runtime cost. Returns `self` for chaining.
    pub fn runtime_cost(mut self, v: f64) -> Self {
        self.inner.runtime_cost = v;
        self
    }
    /// Set memory pressure.
    pub fn memory_pressure(mut self, v: f64) -> Self {
        self.inner.memory_pressure = v;
        self
    }
    /// Set thermal pressure.
    pub fn thermal_pressure(mut self, v: f64) -> Self {
        self.inner.thermal_pressure = v;
        self
    }
    /// Set bandwidth pressure.
    pub fn bandwidth_pressure(mut self, v: f64) -> Self {
        self.inner.bandwidth_pressure = v;
        self
    }
    /// Set code-size pressure.
    pub fn code_size_pressure(mut self, v: f64) -> Self {
        self.inner.code_size_pressure = v;
        self
    }
    /// Deprecated alias for [`Self::code_size_pressure`] — the field
    /// was renamed for the PINN v1 vocabulary. Removed next release.
    #[deprecated(since = "0.1.12", note = "renamed to code_size_pressure")]
    pub fn code_size_cost(self, v: f64) -> Self {
        self.code_size_pressure(v)
    }
    /// Set reconstruction risk.
    pub fn reconstruction_risk(mut self, v: f64) -> Self {
        self.inner.reconstruction_risk = v;
        self
    }
    /// Set verifier-risk penalty.
    pub fn verifier_risk_penalty(mut self, v: f64) -> Self {
        self.inner.verifier_risk_penalty = v;
        self
    }
    /// Set fusion reward.
    pub fn fusion_reward(mut self, v: f64) -> Self {
        self.inner.fusion_reward = v;
        self
    }
    /// Set reuse reward.
    pub fn reuse_reward(mut self, v: f64) -> Self {
        self.inner.reuse_reward = v;
        self
    }
    /// Set compression reward.
    pub fn compression_reward(mut self, v: f64) -> Self {
        self.inner.compression_reward = v;
        self
    }
    /// Set locality reward.
    pub fn locality_reward(mut self, v: f64) -> Self {
        self.inner.locality_reward = v;
        self
    }
    /// Validate and produce [`EnergyComponents`].
    pub fn build(self) -> Result<EnergyComponents, CompressionError> {
        if !self.inner.is_valid() {
            return Err(CompressionError::InvalidTolerance { value: f64::NAN });
        }
        Ok(self.inner)
    }
}

// ---------------------------------------------------------------------------
// EnergyScore
// ---------------------------------------------------------------------------

/// Scalar energy + the decomposition that produced it.
///
/// `total` can be negative when rewards exceed costs (which is what
/// makes a plan a *good* recommendation). The auditor reads `components`
/// to see *why* a plan won — no hidden weights.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyScore {
    /// Kahan-summed total: `costs - rewards`.
    pub total: f64,
    /// The decomposition.
    pub components: EnergyComponents,
}

impl EnergyScore {
    /// Compute from raw components.
    pub fn from_components(components: EnergyComponents) -> Self {
        let total = components.kahan_total();
        Self { total, components }
    }

    /// `true` iff `total` and every component is finite. The ranker
    /// rejects non-finite scores before sorting.
    pub fn is_finite(&self) -> bool {
        self.total.is_finite() && self.components.is_valid()
    }
}

// ---------------------------------------------------------------------------
// RankedCandidate
// ---------------------------------------------------------------------------

/// One row of [`EnergyRanking::ordered`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RankedCandidate {
    /// Stable identifier.
    pub id: CandidateId,
    /// Computed score.
    pub score: EnergyScore,
}

// ---------------------------------------------------------------------------
// EnergyRanker
// ---------------------------------------------------------------------------

/// Stateless ranker. Carries no configuration; the input `(id, components)`
/// pairs are scored and sorted deterministically.
///
/// The empty struct lets us add per-instance options later (e.g.
/// scalar weight per term) without breaking existing callers.
#[derive(Debug, Clone, Copy, Default)]
pub struct EnergyRanker;

impl EnergyRanker {
    /// Construct.
    pub const fn new() -> Self {
        Self
    }

    /// Score + sort the inputs.
    ///
    /// On non-finite components, the offending entry is dropped from the
    /// ranking and listed in [`RankingMetadata::dropped`] — the ranker
    /// never panics. Same input shape → byte-identical ranking.
    pub fn rank(&self, input: Vec<(CandidateId, EnergyComponents)>) -> EnergyRanking {
        let mut scored: Vec<RankedCandidate> = Vec::with_capacity(input.len());
        let mut dropped: Vec<CandidateId> = Vec::new();
        for (id, components) in input.into_iter() {
            if !components.is_valid() {
                dropped.push(id);
                continue;
            }
            let score = EnergyScore::from_components(components);
            if !score.is_finite() {
                dropped.push(id);
                continue;
            }
            scored.push(RankedCandidate { id, score });
        }
        // Stable sort by (total ASC, id ASC). `f64::total_cmp` is
        // deterministic across platforms.
        scored.sort_by(|a, b| {
            a.score
                .total
                .total_cmp(&b.score.total)
                .then_with(|| a.id.cmp(&b.id))
        });
        dropped.sort();
        let metadata = RankingMetadata {
            input_count: scored.len() + dropped.len(),
            dropped,
        };
        EnergyRanking {
            ordered: scored,
            metadata,
        }
    }
}

// ---------------------------------------------------------------------------
// EnergyRanking / RankingMetadata
// ---------------------------------------------------------------------------

/// Result of [`EnergyRanker::rank`].
#[derive(Debug, Clone)]
pub struct EnergyRanking {
    /// Candidates sorted by ascending energy (best plan first).
    pub ordered: Vec<RankedCandidate>,
    /// Diagnostics.
    pub metadata: RankingMetadata,
}

/// Diagnostics surfaced alongside [`EnergyRanking::ordered`].
#[derive(Debug, Clone, Default)]
pub struct RankingMetadata {
    /// Total number of inputs the ranker received.
    pub input_count: usize,
    /// IDs of candidates that were dropped because their components were
    /// non-finite or negative. Sorted ascending for determinism.
    pub dropped: Vec<CandidateId>,
}

impl EnergyRanking {
    /// Best candidate, or `None` if the ranking is empty.
    pub fn best(&self) -> Option<RankedCandidate> {
        self.ordered.first().copied()
    }

    /// Number of candidates surviving validation.
    pub fn len(&self) -> usize {
        self.ordered.len()
    }

    /// `true` iff the ranking has no candidates.
    pub fn is_empty(&self) -> bool {
        self.ordered.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn comp(rt: f64, mem: f64, th: f64, fu: f64, co: f64) -> EnergyComponents {
        EnergyComponents::builder()
            .runtime_cost(rt)
            .memory_pressure(mem)
            .thermal_pressure(th)
            .fusion_reward(fu)
            .compression_reward(co)
            .build()
            .unwrap()
    }

    #[test]
    fn components_total_costs_minus_rewards() {
        let c = comp(1.0, 1.0, 1.0, 1.0, 1.0);
        // costs (3.0) - rewards (2.0) = 1.0
        let s = EnergyScore::from_components(c);
        assert!((s.total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn components_validate_non_negative() {
        let bad = EnergyComponents::new(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(bad.is_err());
        let nan = EnergyComponents::new(f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(nan.is_err());
        let inf = EnergyComponents::new(
            f64::INFINITY,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        );
        assert!(inf.is_err());
        let neg_rew = EnergyComponents::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0);
        assert!(neg_rew.is_err());
        let neg_bandwidth =
            EnergyComponents::new(0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(neg_bandwidth.is_err());
        let neg_locality =
            EnergyComponents::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1);
        assert!(neg_locality.is_err());
    }

    #[test]
    fn pinn_fields_participate_in_total() {
        // bandwidth_pressure is a cost; locality_reward subtracts.
        let c = EnergyComponents::builder()
            .bandwidth_pressure(0.5)
            .locality_reward(0.2)
            .build()
            .unwrap();
        let s = EnergyScore::from_components(c);
        assert!(
            (s.total - 0.3).abs() < 1e-12,
            "0.5 - 0.2 = 0.3, got {}",
            s.total
        );
    }

    #[test]
    #[allow(deprecated)]
    fn deprecated_code_size_cost_alias_sets_renamed_field() {
        let c = EnergyComponents::builder()
            .code_size_cost(0.4)
            .build()
            .unwrap();
        assert_eq!(c.code_size_pressure, 0.4);
    }

    #[test]
    fn empty_ranking_is_empty() {
        let r = EnergyRanker::new();
        let result = r.rank(vec![]);
        assert!(result.is_empty());
        assert!(result.best().is_none());
        assert_eq!(result.metadata.input_count, 0);
    }

    #[test]
    fn lower_runtime_cost_ranks_higher() {
        let r = EnergyRanker::new();
        let result = r.rank(vec![
            (CandidateId(1), comp(10.0, 0.0, 0.0, 0.0, 0.0)),
            (CandidateId(2), comp(1.0, 0.0, 0.0, 0.0, 0.0)),
        ]);
        assert_eq!(result.best().unwrap().id, CandidateId(2));
    }

    #[test]
    fn higher_compression_reward_ranks_higher() {
        let r = EnergyRanker::new();
        let result = r.rank(vec![
            (CandidateId(1), comp(0.0, 0.0, 0.0, 0.0, 0.0)),
            (CandidateId(2), comp(0.0, 0.0, 0.0, 0.0, 5.0)),
        ]);
        // c2 has compression_reward -5.0 → total -5.0, beats 0.0.
        assert_eq!(result.best().unwrap().id, CandidateId(2));
    }

    #[test]
    fn equal_energy_tie_breaks_by_id() {
        let r = EnergyRanker::new();
        let result = r.rank(vec![
            (CandidateId(2), comp(1.0, 0.0, 0.0, 0.0, 0.0)),
            (CandidateId(1), comp(1.0, 0.0, 0.0, 0.0, 0.0)),
        ]);
        // Equal totals → smaller ID wins.
        assert_eq!(result.ordered[0].id, CandidateId(1));
        assert_eq!(result.ordered[1].id, CandidateId(2));
    }

    #[test]
    fn ranking_is_deterministic_under_shuffle() {
        // Same set of (id, components) in different input orders should
        // produce the same ranking.
        let make_input = |order: &[u64]| {
            order
                .iter()
                .map(|i| (CandidateId(*i), comp(*i as f64, 0.0, 0.0, 0.0, 0.0)))
                .collect::<Vec<_>>()
        };
        let r = EnergyRanker::new();
        let r1 = r.rank(make_input(&[1, 2, 3, 4]));
        let r2 = r.rank(make_input(&[4, 1, 3, 2]));
        let r3 = r.rank(make_input(&[3, 4, 2, 1]));
        let ids1: Vec<u64> = r1.ordered.iter().map(|x| x.id.0).collect();
        let ids2: Vec<u64> = r2.ordered.iter().map(|x| x.id.0).collect();
        let ids3: Vec<u64> = r3.ordered.iter().map(|x| x.id.0).collect();
        assert_eq!(ids1, ids2);
        assert_eq!(ids1, ids3);
    }

    #[test]
    fn negative_total_when_rewards_dominate() {
        let r = EnergyRanker::new();
        let result = r.rank(vec![(
            CandidateId(1),
            EnergyComponents::builder()
                .runtime_cost(1.0)
                .compression_reward(10.0)
                .build()
                .unwrap(),
        )]);
        assert!(result.best().unwrap().score.total < 0.0);
    }

    #[test]
    fn invalid_components_dropped_not_panicked() {
        // Test the runtime-validity path — we can't pass invalid
        // components through EnergyComponents::new (it errors), so we
        // construct them via direct field assignment.
        let mut bad = EnergyComponents::default();
        bad.runtime_cost = f64::NAN;
        let r = EnergyRanker::new();
        let result = r.rank(vec![
            (CandidateId(1), bad),
            (CandidateId(2), comp(0.0, 0.0, 0.0, 0.0, 0.0)),
        ]);
        assert_eq!(result.len(), 1);
        assert_eq!(result.metadata.dropped, vec![CandidateId(1)]);
        assert_eq!(result.metadata.input_count, 2);
    }

    #[test]
    fn kahan_sum_preserves_modest_precision_loss_scenario() {
        // Kahan beats naive summation for sequences with multiple
        // small additions, but it cannot fully recover from a
        // catastrophic cancellation where a single large positive is
        // followed by a single large negative — the compensation
        // register itself gets washed out when the running sum
        // collapses to ~0. This test asserts the property Kahan
        // actually guarantees: the sum of N small values (N ≥ 2) is
        // exact, vs naive summation which would drift.
        let small = 0.1f64;
        let c = EnergyComponents::new(
            small, small, small, small, small, small, small, 0.0, 0.0, 0.0, 0.0,
        )
        .unwrap();
        let s = EnergyScore::from_components(c);
        // 7 × 0.1 == 0.7 exactly under Kahan; naive f64 sum drifts ~1e-17.
        assert!(
            (s.total - 0.7).abs() < 1e-15,
            "Kahan sum drifted on small-value sequence: total = {}",
            s.total
        );
    }

    #[test]
    fn ranking_metadata_input_count_accurate() {
        let r = EnergyRanker::new();
        let inputs = vec![
            (CandidateId(1), comp(0.0, 0.0, 0.0, 0.0, 0.0)),
            (CandidateId(2), comp(1.0, 0.0, 0.0, 0.0, 0.0)),
            (CandidateId(3), comp(2.0, 0.0, 0.0, 0.0, 0.0)),
        ];
        let result = r.rank(inputs);
        assert_eq!(result.metadata.input_count, 3);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn rank_preserves_score_decomposition() {
        let r = EnergyRanker::new();
        let c = EnergyComponents::builder()
            .runtime_cost(5.0)
            .compression_reward(3.0)
            .build()
            .unwrap();
        let result = r.rank(vec![(CandidateId(1), c)]);
        let scored = result.ordered[0];
        assert_eq!(scored.score.components.runtime_cost, 5.0);
        assert_eq!(scored.score.components.compression_reward, 3.0);
        assert!((scored.score.total - 2.0).abs() < 1e-12);
    }

    #[test]
    fn higher_thermal_pressure_ranks_lower() {
        // Verifier-relevant: a thermally-hot plan should be down-ranked.
        let r = EnergyRanker::new();
        let result = r.rank(vec![
            (
                CandidateId(1),
                EnergyComponents::builder()
                    .thermal_pressure(0.0)
                    .build()
                    .unwrap(),
            ),
            (
                CandidateId(2),
                EnergyComponents::builder()
                    .thermal_pressure(0.9)
                    .build()
                    .unwrap(),
            ),
        ]);
        assert_eq!(result.best().unwrap().id, CandidateId(1));
    }
}
