//! Phase A1 — `EnergyAwarePassRanker`: opt-in energy re-ranking wrapper
//! around any `cjc_cana::PassRanker`.
//!
//! ## What this does
//!
//! Wraps a base [`cjc_cana::PassRanker`] and re-orders its
//! per-function `recommended` list using the [`EnergyRanker`]. The base
//! ranker still:
//!
//! 1. Decides which passes to keep vs drop (skip-threshold logic).
//! 2. Filters by legality gate.
//! 3. Injects `fusion_rewrite` when fusion candidates exist.
//! 4. Produces the final `RankingReport.verdict`.
//!
//! The adapter changes **only the *order* of already-approved passes**
//! within each function's `recommended` list. It cannot:
//!
//! - introduce a recommendation the base ranker dropped,
//! - remove a recommendation the base ranker kept (the count is
//!   structurally preserved),
//! - alter the legality verdict (no new passes proposed; the gate has
//!   already run).
//!
//! This makes the adapter **safe-by-construction** for production use:
//! the worst it can do is re-order recommendations within a function in
//! a way that performs slightly worse than the base ranker, which is
//! catchable by the existing parity gate.
//!
//! ## Why it lives in `cjc-cana-compress` (not `cjc-cana`)
//!
//! `EnergyRanker` already lives in this crate. Putting the adapter
//! here keeps the cjc-quantum dependency out of `cjc-cana`'s tree —
//! same isolation pattern as `cjc-cana-nss::NssPressurePredictor`.
//!
//! ## Determinism contract
//!
//! Same `(program, features, base_ranker, predictor, config)` produces
//! byte-identical [`cjc_cana::RankingReport`] across runs and
//! platforms. Backed by:
//!
//! - [`BTreeMap`] iteration over functions (sorted).
//! - [`EnergyRanker`]'s `(total, candidate_id)` tie-break.
//! - Deterministic `PressurePredictor` outputs (the trait contract
//!   already requires this).
//! - All `EnergyComponents` derivation uses fixed scaling factors from
//!   [`EnergyComponentsConfig`] — no wall-clock, no RNG.

use std::collections::BTreeMap;

use cjc_cana::cost_model::CostModel;
use cjc_cana::features::{CanaFeatures, FnFeatures};
use cjc_cana::legality::{pass_safety_tier, LegalityGate, PassSafetyTier, ProposedPass};
use cjc_cana::pass_ranker::{PassRanker, PassRecommendation, RankingReport};
use cjc_cana::pressure::PressurePredictor;
use cjc_mir::MirProgram;

use crate::candidate::CandidateId;
use crate::energy::{EnergyComponents, EnergyRanker};

// ---------------------------------------------------------------------------
// EnergyComponentsConfig
// ---------------------------------------------------------------------------

/// Scaling factors that map the base ranker's outputs and the NSS
/// predictor's outputs into the 9 [`EnergyComponents`] fields.
///
/// All fields are non-negative, finite `f64`. The defaults are
/// hand-picked conservative weights matched against the existing
/// `cjc-cana::linear_cost_model` coefficient scale. Tune the individual
/// factors via the fluent setters before passing into
/// [`EnergyAwarePassRanker::with_config`].
#[derive(Debug, Clone, Copy)]
pub struct EnergyComponentsConfig {
    /// Multiplier on the base ranker's `predicted_compile_cost`. Default
    /// `1.0` — the compile-cost term enters the energy in its native
    /// (normalized) units.
    pub runtime_cost_scale: f64,
    /// Multiplier on the NSS predictor's thermal pressure. Default
    /// `1.0`. Larger values down-rank passes on hot functions more
    /// aggressively.
    pub thermal_pressure_scale: f64,
    /// Multiplier on the NSS predictor's memory-peak pressure. Default
    /// `1.0`.
    pub memory_pressure_scale: f64,
    /// Multiplier on the NSS predictor's CPU saturation, mapped into
    /// the `runtime_cost` term (a saturated CPU implies that the pass's
    /// compile-time cost is more painful). Default `0.5`.
    pub cpu_pressure_to_runtime_scale: f64,
    /// Code-size cost coefficient for code-expanding passes.
    /// Multiplies the per-pass `code_size_factor` from
    /// [`pass_code_size_factor`]. Default `0.1` — a 0.3 factor × 0.1 =
    /// 0.03 contribution for `loop_unroll`.
    pub code_size_scale: f64,
    /// Constant penalty added to `verifier_risk_penalty` for any
    /// recommendation whose pass is [`PassSafetyTier::NoStrictReductions`].
    /// Default `0.02` — small enough to break ties but not to flip
    /// strong winners.
    pub no_strict_reductions_risk: f64,
    /// Multiplier on the base ranker's `predicted_benefit` when routed
    /// into the appropriate reward channel (see
    /// [`pass_benefit_channel`]). Default `1.0`.
    pub benefit_reward_scale: f64,
    /// Additional confidence weight: `effective_benefit = benefit *
    /// (confidence ^ confidence_exponent)`. Default `1.0` (linear);
    /// raise to penalise low-confidence predictions more strongly.
    pub confidence_exponent: f64,
}

impl Default for EnergyComponentsConfig {
    fn default() -> Self {
        Self {
            runtime_cost_scale: 1.0,
            thermal_pressure_scale: 1.0,
            memory_pressure_scale: 1.0,
            cpu_pressure_to_runtime_scale: 0.5,
            code_size_scale: 0.1,
            no_strict_reductions_risk: 0.02,
            benefit_reward_scale: 1.0,
            confidence_exponent: 1.0,
        }
    }
}

impl EnergyComponentsConfig {
    /// All-zero config — useful for tests that want to isolate
    /// individual terms.
    pub fn zero() -> Self {
        Self {
            runtime_cost_scale: 0.0,
            thermal_pressure_scale: 0.0,
            memory_pressure_scale: 0.0,
            cpu_pressure_to_runtime_scale: 0.0,
            code_size_scale: 0.0,
            no_strict_reductions_risk: 0.0,
            benefit_reward_scale: 0.0,
            confidence_exponent: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-pass classification helpers
// ---------------------------------------------------------------------------

/// Which reward channel the base ranker's `predicted_benefit` flows
/// into. The energy model has three reward channels
/// (`fusion_reward`, `reuse_reward`, `compression_reward`), and each
/// pass's benefit naturally fits one of them:
///
/// - **`Reuse`**: passes that reduce allocations / improve buffer
///   reuse (`dce`, `cse`, `licm`).
/// - **`Fusion`**: passes that enable downstream fusion or specialise
///   code (`loop_unroll`, `vectorize`, `specialize`, `monomorphize`,
///   `fusion_rewrite`).
/// - **`Fusion` (default)**: everything else (`constant_fold`,
///   `cf_round_2`, `strength_reduce`) — these all simplify
///   expressions so downstream fusion has cleaner shapes to work with.
///
/// `Compression` is not used for pass ranking — it's reserved for the
/// CANA compression layer's energy ranker calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenefitChannel {
    /// Route benefit to `EnergyComponents::reuse_reward`.
    Reuse,
    /// Route benefit to `EnergyComponents::fusion_reward`.
    Fusion,
}

/// Pure mapping from pass name to benefit channel. See
/// [`BenefitChannel`] for the rationale.
pub fn pass_benefit_channel(pass_name: &str) -> BenefitChannel {
    match pass_name {
        "dce" | "dead_code_elimination" => BenefitChannel::Reuse,
        "cse" | "common_subexpression_elimination" => BenefitChannel::Reuse,
        "licm" | "loop_invariant_code_motion" => BenefitChannel::Reuse,
        _ => BenefitChannel::Fusion,
    }
}

/// Code-size cost factor in `[0.0, 1.0]` for each canonical pass.
/// Multiplied by [`EnergyComponentsConfig::code_size_scale`] to produce
/// the final `code_size_cost` term.
///
/// - `loop_unroll` → 0.3 (replicates body N times in sequence)
/// - `vectorize` → 0.15 (lane variants)
/// - `specialize` / `monomorphize` → 0.2 (instantiation per type)
/// - everything else → 0.0
pub fn pass_code_size_factor(pass_name: &str) -> f64 {
    match pass_name {
        "loop_unroll" | "unroll" => 0.3,
        "vectorize" => 0.15,
        "specialize" => 0.2,
        "monomorphize" => 0.2,
        _ => 0.0,
    }
}

// ---------------------------------------------------------------------------
// derive_energy_components
// ---------------------------------------------------------------------------

/// Derive [`EnergyComponents`] for one [`PassRecommendation`] given
/// the function's features and NSS pressure predictions.
///
/// Pure: same inputs → same output. The component values are clamped
/// non-negative and finite by construction.
///
/// ## Mapping
///
/// | Component | Source |
/// |---|---|
/// | `runtime_cost` | `predicted_compile_cost * runtime_cost_scale + cpu_saturation * cpu_pressure_to_runtime_scale` |
/// | `memory_pressure` | `nss_memory_peak * memory_pressure_scale` |
/// | `thermal_pressure` | `nss_thermal * thermal_pressure_scale` |
/// | `code_size_cost` | `pass_code_size_factor * code_size_scale` |
/// | `reconstruction_risk` | 0 (not used for pass ranking) |
/// | `verifier_risk_penalty` | `no_strict_reductions_risk` if `pass_safety_tier == NoStrictReductions` |
/// | `fusion_reward` | `predicted_benefit * confidence^exp * benefit_reward_scale` if channel is Fusion |
/// | `reuse_reward` | `predicted_benefit * confidence^exp * benefit_reward_scale` if channel is Reuse |
/// | `compression_reward` | 0 (not used for pass ranking) |
pub fn derive_energy_components(
    recommendation: &PassRecommendation,
    _features: &FnFeatures,
    thermal: f64,
    memory: f64,
    cpu: f64,
    config: &EnergyComponentsConfig,
) -> EnergyComponents {
    let pass_name = &recommendation.pass_name;
    let tier = pass_safety_tier(pass_name);
    let channel = pass_benefit_channel(pass_name);

    // Clamp all NSS predictions to `[0, 1]` regardless of what the
    // predictor returned. The PressurePredictor trait contract says
    // the output should be in `[0, 1]`, but defensive clamping makes
    // the energy values bounded by construction.
    let thermal_norm = clamp_unit(thermal);
    let memory_norm = clamp_unit(memory);
    let cpu_norm = clamp_unit(cpu);

    // Components flowing into the energy total.
    let runtime_cost = (recommendation.predicted_compile_cost * config.runtime_cost_scale)
        + (cpu_norm * config.cpu_pressure_to_runtime_scale);
    let thermal_pressure = thermal_norm * config.thermal_pressure_scale;
    let memory_pressure = memory_norm * config.memory_pressure_scale;
    let code_size_cost = pass_code_size_factor(pass_name) * config.code_size_scale;
    let verifier_risk_penalty = match tier {
        PassSafetyTier::NoStrictReductions => config.no_strict_reductions_risk,
        PassSafetyTier::Universal => 0.0,
    };

    // Confidence-weighted benefit reward.
    let confidence_weighted = recommendation.predicted_benefit
        * confidence_exp_safe(recommendation.confidence, config.confidence_exponent);
    let reward_value = confidence_weighted.max(0.0) * config.benefit_reward_scale;

    let (fusion_reward, reuse_reward) = match channel {
        BenefitChannel::Fusion => (reward_value, 0.0),
        BenefitChannel::Reuse => (0.0, reward_value),
    };

    // Build via `EnergyComponents::new` so the constructor's
    // non-negative-finite guard runs. On any failure (shouldn't happen
    // because every input was clamped), fall back to all-zero
    // components — that gives the recommendation a neutral energy of 0,
    // which is the most conservative possible re-ranking signal.
    EnergyComponents::new(
        non_neg_finite(runtime_cost),
        non_neg_finite(memory_pressure),
        non_neg_finite(thermal_pressure),
        non_neg_finite(code_size_cost),
        0.0, // reconstruction_risk
        non_neg_finite(verifier_risk_penalty),
        non_neg_finite(fusion_reward),
        non_neg_finite(reuse_reward),
        0.0, // compression_reward
    )
    .unwrap_or_default()
}

fn clamp_unit(x: f64) -> f64 {
    if !x.is_finite() {
        0.0
    } else {
        x.clamp(0.0, 1.0)
    }
}

fn non_neg_finite(x: f64) -> f64 {
    if !x.is_finite() {
        0.0
    } else {
        x.max(0.0)
    }
}

/// `confidence^exp` with safe handling: confidence is clamped to
/// `[0, 1]`, exp must be finite and non-negative. Negative or
/// non-finite exponents would cause undefined behaviour at confidence
/// 0; we fall back to `1.0` (linear) on any unsoundness.
fn confidence_exp_safe(confidence: f64, exp: f64) -> f64 {
    let c = clamp_unit(confidence);
    if !exp.is_finite() || exp < 0.0 {
        return c;
    }
    if c == 0.0 {
        return 0.0;
    }
    c.powf(exp)
}

// ---------------------------------------------------------------------------
// EnergyAwarePassRanker
// ---------------------------------------------------------------------------

/// Wraps a [`cjc_cana::PassRanker`] with energy-aware re-ordering.
///
/// Construct via [`EnergyAwarePassRanker::new`]; configure scaling via
/// [`EnergyAwarePassRanker::with_config`]. Run via [`Self::rank`] —
/// returns the same [`RankingReport`] shape the base ranker emits, with
/// per-function `recommended` lists re-ordered by ascending energy.
pub struct EnergyAwarePassRanker<M: CostModel, G: LegalityGate> {
    base: PassRanker<M, G>,
    predictor: Box<dyn PressurePredictor>,
    config: EnergyComponentsConfig,
}

impl<M: CostModel, G: LegalityGate> EnergyAwarePassRanker<M, G> {
    /// Construct an adapter wrapping `base` and using `predictor` for
    /// pressure forecasts.
    pub fn new(base: PassRanker<M, G>, predictor: Box<dyn PressurePredictor>) -> Self {
        Self {
            base,
            predictor,
            config: EnergyComponentsConfig::default(),
        }
    }

    /// Replace the scaling config.
    pub fn with_config(mut self, config: EnergyComponentsConfig) -> Self {
        self.config = config;
        self
    }

    /// Borrow the wrapped base ranker. Useful for testing or for code
    /// that wants to query the underlying cost model.
    pub fn base(&self) -> &PassRanker<M, G> {
        &self.base
    }

    /// Borrow the wrapped predictor.
    pub fn predictor(&self) -> &dyn PressurePredictor {
        &*self.predictor
    }

    /// Borrow the config.
    pub fn config(&self) -> &EnergyComponentsConfig {
        &self.config
    }

    /// Produce a re-ranked report. Same shape as `PassRanker::rank`:
    ///
    /// 1. Run the base ranker, get its `RankingReport`.
    /// 2. Query the predictor once for thermal / memory / CPU per
    ///    function.
    /// 3. For each function, derive [`EnergyComponents`] for every
    ///    recommendation, sort by ascending energy via
    ///    [`EnergyRanker`].
    /// 4. Re-write `RankingReport.sequence.per_function` to reflect the
    ///    new order.
    /// 5. Verdict is unchanged — no new recommendations were added; the
    ///    gate has already approved everything in `recommended`.
    pub fn rank(&self, program: &MirProgram, features: &CanaFeatures) -> RankingReport {
        let mut report = self.base.rank(program, features);

        let thermal = self.predictor.predict_thermal(program, features);
        let memory = self.predictor.predict_memory_peak(program, features);
        let cpu = self.predictor.predict_cpu_saturation(program, features);

        // Collect function names up front so we can borrow `report`
        // mutably inside the loop.
        let fn_names: Vec<String> = report.per_fn.keys().cloned().collect();

        for fn_name in &fn_names {
            // Skip if the function has no features (defensive — should
            // never happen because per_fn comes from features.per_fn).
            let Some(fn_features) = features.per_fn.get(fn_name) else {
                continue;
            };

            let fn_thermal = thermal.get(fn_name).copied().unwrap_or(0.0);
            let fn_memory = memory.get(fn_name).copied().unwrap_or(0.0);
            let fn_cpu = cpu.get(fn_name).copied().unwrap_or(0.0);

            let ranking = match report.per_fn.get(fn_name) {
                Some(r) => r,
                None => continue,
            };

            if ranking.recommended.is_empty() {
                continue;
            }

            // Derive components per recommendation and rank.
            let energy_inputs: Vec<(CandidateId, EnergyComponents)> = ranking
                .recommended
                .iter()
                .enumerate()
                .map(|(slot, rec)| {
                    let components = derive_energy_components(
                        rec,
                        fn_features,
                        fn_thermal,
                        fn_memory,
                        fn_cpu,
                        &self.config,
                    );
                    (CandidateId(slot as u64), components)
                })
                .collect();

            let energy_ranker = EnergyRanker::new();
            let energy_ranking = energy_ranker.rank(energy_inputs);

            // Re-order by energy. The original list count must be
            // preserved — any recommendation `EnergyRanker` dropped
            // (e.g. non-finite components, shouldn't happen) is
            // appended in its original position.
            let total = ranking.recommended.len();
            let mut reordered: Vec<PassRecommendation> = Vec::with_capacity(total);
            for ranked in &energy_ranking.ordered {
                let slot = ranked.id.0 as usize;
                if let Some(rec) = ranking.recommended.get(slot) {
                    reordered.push(rec.clone());
                }
            }
            // Append any dropped (defensive).
            for (slot, rec) in ranking.recommended.iter().enumerate() {
                let kept = energy_ranking
                    .ordered
                    .iter()
                    .any(|r| r.id.0 as usize == slot);
                if !kept {
                    reordered.push(rec.clone());
                }
            }
            debug_assert_eq!(
                reordered.len(),
                total,
                "EnergyAwarePassRanker must not change recommendation count"
            );

            // Apply the reorder.
            if let Some(ranking_mut) = report.per_fn.get_mut(fn_name) {
                ranking_mut.recommended = reordered;
            }

            // Rebuild this function's slot in the bundled sequence.
            // The base ranker's `verdict` was computed before our
            // reorder; since we never added a new pass, the verdict
            // remains valid (the gate would approve the same set in
            // any order).
            let proposed: Vec<ProposedPass> = report
                .per_fn
                .get(fn_name)
                .map(|r| {
                    r.recommended
                        .iter()
                        .map(|x| ProposedPass::Run(x.pass_name.clone()))
                        .collect()
                })
                .unwrap_or_default();
            if !proposed.is_empty() {
                report
                    .sequence
                    .per_function
                    .insert(fn_name.clone(), proposed);
            } else {
                report.sequence.per_function.remove(fn_name);
            }
        }

        report
    }
}

impl<M: CostModel + std::fmt::Debug, G: LegalityGate> std::fmt::Debug
    for EnergyAwarePassRanker<M, G>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnergyAwarePassRanker")
            .field("predictor_name", &self.predictor.name())
            .field("predictor_version", &self.predictor.version())
            .field("config", &self.config)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Convenience: aggregate per-function energy summary in a report-shaped form
// ---------------------------------------------------------------------------

/// Per-function summary of the energy decisions the adapter made.
///
/// Useful for audit / debugging: an operator can see which passes were
/// re-ordered and by how much. Produced by
/// [`EnergyAwarePassRanker::audit`].
#[derive(Debug, Clone, Default)]
pub struct EnergyAuditEntry {
    /// Function name.
    pub function: String,
    /// Ordered list of `(pass_name, energy_total)` after re-ranking.
    pub energy_ordered_passes: Vec<(String, f64)>,
}

impl<M: CostModel, G: LegalityGate> EnergyAwarePassRanker<M, G> {
    /// Run [`Self::rank`] and additionally collect per-function energy
    /// totals for audit purposes. Returns `(report, audit)`.
    pub fn audit(
        &self,
        program: &MirProgram,
        features: &CanaFeatures,
    ) -> (RankingReport, BTreeMap<String, EnergyAuditEntry>) {
        let report = self.rank(program, features);
        let thermal = self.predictor.predict_thermal(program, features);
        let memory = self.predictor.predict_memory_peak(program, features);
        let cpu = self.predictor.predict_cpu_saturation(program, features);

        let mut audit = BTreeMap::new();
        for (fn_name, ranking) in &report.per_fn {
            let Some(fn_features) = features.per_fn.get(fn_name) else {
                continue;
            };
            let fn_thermal = thermal.get(fn_name).copied().unwrap_or(0.0);
            let fn_memory = memory.get(fn_name).copied().unwrap_or(0.0);
            let fn_cpu = cpu.get(fn_name).copied().unwrap_or(0.0);

            let mut pairs = Vec::with_capacity(ranking.recommended.len());
            for rec in &ranking.recommended {
                let components = derive_energy_components(
                    rec,
                    fn_features,
                    fn_thermal,
                    fn_memory,
                    fn_cpu,
                    &self.config,
                );
                let score = crate::energy::EnergyScore::from_components(components);
                pairs.push((rec.pass_name.clone(), score.total));
            }

            audit.insert(
                fn_name.clone(),
                EnergyAuditEntry {
                    function: fn_name.clone(),
                    energy_ordered_passes: pairs,
                },
            );
        }
        (report, audit)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_cana::cost_model::{CostEstimate, CostQuery};
    use cjc_cana::features::extract;
    use cjc_cana::legality::PerPassLegalityGate;
    use cjc_cana::pass_ranker::RankingRationale;
    use cjc_mir::{MirBody, MirFnId, MirFunction, MirProgram};

    fn empty_program(fn_names: &[&str]) -> MirProgram {
        MirProgram {
            functions: fn_names
                .iter()
                .enumerate()
                .map(|(i, n)| MirFunction {
                    id: MirFnId(i as u32),
                    name: n.to_string(),
                    type_params: vec![],
                    params: vec![],
                    return_type: None,
                    body: MirBody {
                        stmts: vec![],
                        result: None,
                    },
                    is_nogc: false,
                    cfg_body: None,
                    decorators: vec![],
                    vis: cjc_ast::Visibility::Public,
                    local_count: 0,
                })
                .collect(),
            struct_defs: vec![],
            enum_defs: vec![],
            entry: MirFnId(0),
        }
    }

    /// A stub cost model that returns a per-pass benefit table.
    #[derive(Debug)]
    struct StubCostModel {
        benefits: BTreeMap<String, f64>,
        compile_costs: BTreeMap<String, f64>,
    }
    impl CostModel for StubCostModel {
        fn query(
            &self,
            _program: &MirProgram,
            _features: &CanaFeatures,
            query: &CostQuery<'_>,
        ) -> CostEstimate {
            match query {
                CostQuery::PassBenefit { pass_name, .. } => self
                    .benefits
                    .get(*pass_name)
                    .map(|&v| CostEstimate::Estimated {
                        value: v,
                        confidence: 0.8,
                    })
                    .unwrap_or(CostEstimate::Unknown),
                CostQuery::PassRuntime { pass_name, .. } => self
                    .compile_costs
                    .get(*pass_name)
                    .map(|&v| CostEstimate::Estimated {
                        value: v,
                        confidence: 0.8,
                    })
                    .unwrap_or(CostEstimate::Unknown),
                _ => CostEstimate::Unknown,
            }
        }
        fn name(&self) -> &'static str {
            "stub"
        }
    }

    /// A stub predictor with per-function thermal/memory/CPU values.
    #[derive(Debug, Clone)]
    struct StubPredictor {
        thermal: BTreeMap<String, f64>,
        memory: BTreeMap<String, f64>,
        cpu: BTreeMap<String, f64>,
    }
    impl PressurePredictor for StubPredictor {
        fn predict_thermal(
            &self,
            _program: &MirProgram,
            _features: &CanaFeatures,
        ) -> BTreeMap<String, f64> {
            self.thermal.clone()
        }
        fn predict_memory_peak(
            &self,
            _program: &MirProgram,
            _features: &CanaFeatures,
        ) -> BTreeMap<String, f64> {
            self.memory.clone()
        }
        fn predict_cpu_saturation(
            &self,
            _program: &MirProgram,
            _features: &CanaFeatures,
        ) -> BTreeMap<String, f64> {
            self.cpu.clone()
        }
        fn identify_structural_hot_kernels(
            &self,
            _program: &MirProgram,
            _features: &CanaFeatures,
        ) -> Vec<String> {
            Vec::new()
        }
        fn name(&self) -> &'static str {
            "stub"
        }
        fn version(&self) -> u32 {
            1
        }
    }

    fn fake_rec(pass: &str, benefit: f64, compile_cost: f64) -> PassRecommendation {
        PassRecommendation {
            pass_name: pass.to_string(),
            predicted_benefit: benefit,
            predicted_compile_cost: compile_cost,
            confidence: 0.9,
            rationale: RankingRationale::BenefitAboveThreshold,
        }
    }

    // ----- BenefitChannel + pass_code_size_factor ----------------------

    #[test]
    fn benefit_channel_mapping_is_total() {
        assert_eq!(pass_benefit_channel("dce"), BenefitChannel::Reuse);
        assert_eq!(
            pass_benefit_channel("dead_code_elimination"),
            BenefitChannel::Reuse
        );
        assert_eq!(pass_benefit_channel("cse"), BenefitChannel::Reuse);
        assert_eq!(pass_benefit_channel("licm"), BenefitChannel::Reuse);
        assert_eq!(pass_benefit_channel("loop_unroll"), BenefitChannel::Fusion);
        assert_eq!(pass_benefit_channel("vectorize"), BenefitChannel::Fusion);
        assert_eq!(
            pass_benefit_channel("constant_fold"),
            BenefitChannel::Fusion
        );
        assert_eq!(
            pass_benefit_channel("strength_reduce"),
            BenefitChannel::Fusion
        );
        // Unknown passes default to Fusion (no soft information).
        assert_eq!(
            pass_benefit_channel("hypothetical_future_pass"),
            BenefitChannel::Fusion
        );
    }

    #[test]
    fn code_size_factor_is_non_negative_and_finite() {
        for pass in [
            "constant_fold",
            "strength_reduce",
            "dce",
            "cse",
            "licm",
            "loop_unroll",
            "vectorize",
            "specialize",
            "monomorphize",
            "cf_round_2",
            "unknown_pass",
        ] {
            let f = pass_code_size_factor(pass);
            assert!(f.is_finite() && f >= 0.0, "{}: {}", pass, f);
        }
    }

    // ----- derive_energy_components ------------------------------------

    #[test]
    fn derive_components_are_always_finite_and_valid() {
        let program = empty_program(&["f"]);
        let features = extract(&program);
        let fn_features = features.per_fn.get("f").unwrap();
        let rec = fake_rec("dce", 0.1, 0.05);
        let config = EnergyComponentsConfig::default();

        // Sweep across plausible NSS predictions.
        for thermal in [0.0, 0.5, 0.9, 1.0, 1.5, f64::NAN, f64::INFINITY] {
            for memory in [0.0, 0.5, 1.0] {
                for cpu in [0.0, 0.5, 1.0] {
                    let c =
                        derive_energy_components(&rec, fn_features, thermal, memory, cpu, &config);
                    assert!(c.is_valid(), "derived components must be valid: {:?}", c);
                }
            }
        }
    }

    #[test]
    fn derive_components_routes_dce_to_reuse() {
        let program = empty_program(&["f"]);
        let features = extract(&program);
        let fn_features = features.per_fn.get("f").unwrap();
        let rec = fake_rec("dce", 0.5, 0.05);
        let config = EnergyComponentsConfig::default();
        let c = derive_energy_components(&rec, fn_features, 0.0, 0.0, 0.0, &config);
        assert!(c.reuse_reward > 0.0);
        assert_eq!(c.fusion_reward, 0.0);
    }

    #[test]
    fn derive_components_routes_loop_unroll_to_fusion() {
        let program = empty_program(&["f"]);
        let features = extract(&program);
        let fn_features = features.per_fn.get("f").unwrap();
        let rec = fake_rec("loop_unroll", 0.4, 0.1);
        let config = EnergyComponentsConfig::default();
        let c = derive_energy_components(&rec, fn_features, 0.0, 0.0, 0.0, &config);
        assert!(c.fusion_reward > 0.0);
        assert_eq!(c.reuse_reward, 0.0);
        // Code-size cost is non-zero for loop_unroll.
        assert!(c.code_size_cost > 0.0);
    }

    #[test]
    fn derive_components_thermal_pressure_propagates() {
        let program = empty_program(&["f"]);
        let features = extract(&program);
        let fn_features = features.per_fn.get("f").unwrap();
        let rec = fake_rec("cse", 0.2, 0.05);
        let config = EnergyComponentsConfig::default();
        let c_cold = derive_energy_components(&rec, fn_features, 0.0, 0.0, 0.0, &config);
        let c_hot = derive_energy_components(&rec, fn_features, 0.9, 0.0, 0.0, &config);
        assert!(c_hot.thermal_pressure > c_cold.thermal_pressure);
    }

    #[test]
    fn derive_components_clamps_oob_predictions() {
        let program = empty_program(&["f"]);
        let features = extract(&program);
        let fn_features = features.per_fn.get("f").unwrap();
        let rec = fake_rec("cse", 0.2, 0.05);
        let config = EnergyComponentsConfig::default();
        let c = derive_energy_components(&rec, fn_features, 5.0, -1.0, f64::NAN, &config);
        assert!(c.thermal_pressure <= 1.0 * config.thermal_pressure_scale);
        assert!(c.memory_pressure >= 0.0);
        assert!(c.is_valid());
    }

    #[test]
    fn derive_components_no_strict_reductions_tier_adds_penalty() {
        let program = empty_program(&["f"]);
        let features = extract(&program);
        let fn_features = features.per_fn.get("f").unwrap();
        let config = EnergyComponentsConfig::default();
        // CSE is NoStrictReductions tier.
        let rec_cse = fake_rec("cse", 0.2, 0.05);
        let c_cse = derive_energy_components(&rec_cse, fn_features, 0.0, 0.0, 0.0, &config);
        // DCE is Universal tier.
        let rec_dce = fake_rec("dce", 0.2, 0.05);
        let c_dce = derive_energy_components(&rec_dce, fn_features, 0.0, 0.0, 0.0, &config);
        assert!(c_cse.verifier_risk_penalty > 0.0);
        assert_eq!(c_dce.verifier_risk_penalty, 0.0);
    }

    #[test]
    fn derive_components_zero_config_produces_zero_energy() {
        let program = empty_program(&["f"]);
        let features = extract(&program);
        let fn_features = features.per_fn.get("f").unwrap();
        let rec = fake_rec("loop_unroll", 0.4, 0.1);
        let c = derive_energy_components(
            &rec,
            fn_features,
            0.5,
            0.5,
            0.5,
            &EnergyComponentsConfig::zero(),
        );
        // Every term must be 0 under zero config.
        assert_eq!(c.runtime_cost, 0.0);
        assert_eq!(c.memory_pressure, 0.0);
        assert_eq!(c.thermal_pressure, 0.0);
        assert_eq!(c.code_size_cost, 0.0);
        assert_eq!(c.verifier_risk_penalty, 0.0);
        assert_eq!(c.fusion_reward, 0.0);
        assert_eq!(c.reuse_reward, 0.0);
    }

    // ----- EnergyAwarePassRanker ---------------------------------------

    fn stub_ranker_setup() -> (
        MirProgram,
        CanaFeatures,
        StubCostModel,
        PerPassLegalityGate,
        StubPredictor,
    ) {
        let program = empty_program(&["hot_fn"]);
        let features = extract(&program);
        let mut benefits = BTreeMap::new();
        benefits.insert("constant_fold".to_string(), 0.10);
        benefits.insert("strength_reduce".to_string(), 0.08);
        benefits.insert("dce".to_string(), 0.06);
        benefits.insert("cse".to_string(), 0.05);
        benefits.insert("licm".to_string(), 0.04);
        benefits.insert("loop_unroll".to_string(), 0.03);
        let mut compile_costs = BTreeMap::new();
        for k in [
            "constant_fold",
            "strength_reduce",
            "dce",
            "cse",
            "licm",
            "loop_unroll",
        ] {
            compile_costs.insert(k.to_string(), 0.05);
        }
        let cost_model = StubCostModel {
            benefits,
            compile_costs,
        };
        let gate = PerPassLegalityGate::new();
        let mut thermal = BTreeMap::new();
        thermal.insert("hot_fn".to_string(), 0.8);
        let mut memory = BTreeMap::new();
        memory.insert("hot_fn".to_string(), 0.2);
        let mut cpu = BTreeMap::new();
        cpu.insert("hot_fn".to_string(), 0.5);
        let predictor = StubPredictor {
            thermal,
            memory,
            cpu,
        };
        (program, features, cost_model, gate, predictor)
    }

    #[test]
    fn adapter_preserves_recommendation_count() {
        let (program, features, cost_model, gate, predictor) = stub_ranker_setup();
        let base = PassRanker::new(cost_model, gate);
        let base_count = base
            .rank(&program, &features)
            .per_fn
            .values()
            .map(|r| r.recommended.len())
            .sum::<usize>();
        let adapter = EnergyAwarePassRanker::new(base, Box::new(predictor));
        let after = adapter
            .rank(&program, &features)
            .per_fn
            .values()
            .map(|r| r.recommended.len())
            .sum::<usize>();
        assert_eq!(
            after, base_count,
            "adapter must not change recommendation count"
        );
    }

    #[test]
    fn adapter_preserves_legality_verdict() {
        let (program, features, cost_model, gate, predictor) = stub_ranker_setup();
        let base = PassRanker::new(cost_model, gate);
        let base_verdict = base.rank(&program, &features).verdict.clone();
        let (cost_model2, gate2) = {
            let (_, _, cm, g, _) = stub_ranker_setup();
            (cm, g)
        };
        let adapter =
            EnergyAwarePassRanker::new(PassRanker::new(cost_model2, gate2), Box::new(predictor));
        let energy_verdict = adapter.rank(&program, &features).verdict;
        assert_eq!(base_verdict, energy_verdict);
    }

    #[test]
    fn adapter_is_deterministic() {
        let (program, features, cost_model, gate, predictor) = stub_ranker_setup();
        let adapter =
            EnergyAwarePassRanker::new(PassRanker::new(cost_model, gate), Box::new(predictor));
        let report_a = adapter.rank(&program, &features);
        let report_b = adapter.rank(&program, &features);
        let order_a: Vec<Vec<String>> = report_a
            .per_fn
            .values()
            .map(|r| r.recommended.iter().map(|x| x.pass_name.clone()).collect())
            .collect();
        let order_b: Vec<Vec<String>> = report_b
            .per_fn
            .values()
            .map(|r| r.recommended.iter().map(|x| x.pass_name.clone()).collect())
            .collect();
        assert_eq!(order_a, order_b);
    }

    #[test]
    fn adapter_with_zero_config_matches_base_ordering_modulo_stable_sort() {
        // With all scales zero, every energy total is 0 → ranker
        // tie-breaks by CandidateId. Slot index in `recommended`
        // becomes the CandidateId. So the resulting order is
        // [slot 0, slot 1, ..., slot n-1] — i.e., unchanged from base.
        let (program, features, cost_model, gate, predictor) = stub_ranker_setup();
        let adapter =
            EnergyAwarePassRanker::new(PassRanker::new(cost_model, gate), Box::new(predictor))
                .with_config(EnergyComponentsConfig::zero());
        let report = adapter.rank(&program, &features);
        // The base ranker sorts by descending benefit. With zero
        // config, the energy adapter preserves that order.
        for ranking in report.per_fn.values() {
            for w in ranking.recommended.windows(2) {
                assert!(
                    w[0].predicted_benefit + 1e-12 >= w[1].predicted_benefit,
                    "zero-config ordering should preserve base descending-benefit order"
                );
            }
        }
    }

    #[test]
    fn adapter_higher_thermal_demotes_thermally_aggressive_passes() {
        // Two predictors: one cold, one hot. The hot version should
        // push loop_unroll (high thermal cost & code_size_cost) lower
        // in the ranking compared to the cold version, all else equal.
        let (program, features, cost_model_cold, gate_cold, _) = stub_ranker_setup();
        let (_, _, cost_model_hot, gate_hot, _) = stub_ranker_setup();
        let mut cold = BTreeMap::new();
        cold.insert("hot_fn".to_string(), 0.0);
        let cold_pred = StubPredictor {
            thermal: cold.clone(),
            memory: cold.clone(),
            cpu: cold,
        };
        let mut hot = BTreeMap::new();
        hot.insert("hot_fn".to_string(), 0.99);
        let hot_pred = StubPredictor {
            thermal: hot.clone(),
            memory: BTreeMap::new(),
            cpu: BTreeMap::new(),
        };

        let cold_adapter = EnergyAwarePassRanker::new(
            PassRanker::new(cost_model_cold, gate_cold),
            Box::new(cold_pred),
        );
        let hot_adapter = EnergyAwarePassRanker::new(
            PassRanker::new(cost_model_hot, gate_hot),
            Box::new(hot_pred),
        );

        let cold_report = cold_adapter.rank(&program, &features);
        let hot_report = hot_adapter.rank(&program, &features);

        let cold_loop_unroll = position_of(&cold_report, "hot_fn", "loop_unroll");
        let hot_loop_unroll = position_of(&hot_report, "hot_fn", "loop_unroll");

        if let (Some(c), Some(h)) = (cold_loop_unroll, hot_loop_unroll) {
            assert!(
                h >= c,
                "hot scenario should demote (or tie) loop_unroll vs cold; cold={}, hot={}",
                c,
                h
            );
        }
    }

    fn position_of(report: &RankingReport, fn_name: &str, pass: &str) -> Option<usize> {
        report
            .per_fn
            .get(fn_name)?
            .recommended
            .iter()
            .position(|r| r.pass_name == pass)
    }

    #[test]
    fn adapter_with_null_predictor_still_runs() {
        use cjc_cana::pressure::NullPressurePredictor;
        let (program, features, cost_model, gate, _) = stub_ranker_setup();
        let adapter = EnergyAwarePassRanker::new(
            PassRanker::new(cost_model, gate),
            Box::new(NullPressurePredictor),
        );
        // Should not panic.
        let _ = adapter.rank(&program, &features);
    }

    #[test]
    fn adapter_audit_returns_per_function_energy_totals() {
        let (program, features, cost_model, gate, predictor) = stub_ranker_setup();
        let adapter =
            EnergyAwarePassRanker::new(PassRanker::new(cost_model, gate), Box::new(predictor));
        let (_report, audit) = adapter.audit(&program, &features);
        let entry = audit.get("hot_fn").expect("hot_fn audit entry");
        // Each audit entry must have the same length as `recommended`.
        assert!(!entry.energy_ordered_passes.is_empty());
        // Audit pairs are ordered the same as the report.
        // (Re-running rank() and comparing is the simplest check.)
        let report2 = adapter.rank(&program, &features);
        let report_passes: Vec<&str> = report2
            .per_fn
            .get("hot_fn")
            .unwrap()
            .recommended
            .iter()
            .map(|r| r.pass_name.as_str())
            .collect();
        let audit_passes: Vec<&str> = entry
            .energy_ordered_passes
            .iter()
            .map(|(p, _)| p.as_str())
            .collect();
        assert_eq!(report_passes, audit_passes);
    }
}
