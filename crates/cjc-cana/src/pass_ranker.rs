//! Phase 2 — `PassRanker`: produces a per-function ordered pass sequence
//! from features + cost model + legality gate.
//!
//! ## What this does
//!
//! For each function in a `MirProgram`, the ranker queries the cost model
//! for the predicted benefit of every pass, sorts passes by descending
//! benefit, drops passes whose predicted benefit is below a threshold (an
//! S2 "pass skipping" decision), and emits a `PassRecommendation` per
//! function. The recommendations bundle into a `PassSequence` that the
//! legality gate vets before any compiler decision is made.
//!
//! ## What this does NOT do
//!
//! - Apply any optimization. The ranker emits *recommendations*; an
//!   external driver (`cjc-mir::optimize_program` extension, deferred to
//!   a follow-up commit) is responsible for consuming them.
//! - Override the legality gate. Even if the cost model predicts a huge
//!   win from reordering a `StrictFold`, the gate rejects it and the
//!   ranker drops that recommendation.
//! - Train. The cost model is a `Box<dyn CostModel>` injected by the
//!   caller; for Phase 2 we use `LinearCostModel` (hand-tuned), but the
//!   trait allows swap-in of trained models in Phase 5.
//!
//! ## Determinism contract
//!
//! Same `(program, features, cost_model)` → same `RankingReport`,
//! byte-for-byte. The implementation:
//! - Iterates the `BTreeMap<String, FnFeatures>` (sorted by name)
//! - Sorts passes by `(neg_benefit, name)` using a stable comparator —
//!   ties broken alphabetically so two passes with identical benefits
//!   always emit in the same order

use std::collections::BTreeMap;

use cjc_mir::MirProgram;

use crate::cost_model::{CostEstimate, CostModel, CostQuery};
use crate::features::CanaFeatures;
use crate::legality::{LegalityGate, LegalityVerdict, PassSequence, ProposedPass};

// ---------------------------------------------------------------------------
// The canonical pass set for Phase 2
// ---------------------------------------------------------------------------

/// The fixed pass vocabulary the ranker draws from. Aligns 1:1 with
/// `cjc_mir::optimize::optimize_program`'s 6-pass sequence (with the second
/// `constant_fold` run handled as a separate `cf_round_2` slot).
///
/// Order in this array is the *default* pipeline order; the ranker may
/// reorder them per function based on predicted benefit.
pub const CANONICAL_PASSES: &[&str] = &[
    "constant_fold",
    "strength_reduce",
    "dce",
    "cse",
    "licm",
    "cf_round_2",
];

/// Default benefit threshold for the ranker's S2 "skip" decision.
/// Passes whose predicted benefit (from the cost model) falls below this
/// threshold are dropped from the recommendation. The threshold is
/// deliberately conservative: 0.005 = "saves less than 0.5% of expected
/// runtime." Below that, the compile-time cost almost certainly outweighs
/// the runtime saving.
pub const DEFAULT_SKIP_THRESHOLD: f64 = 0.005;

/// Pass name the ranker injects when fusion candidates are present.
/// Matches the case accepted by [`cjc_mir::optimize::apply_pass`].
pub const FUSION_REWRITE_PASS_NAME: &str = "fusion_rewrite";

/// Hardcoded predicted benefit for fusion_rewrite. Bypasses the cost
/// model because the linear cost model can't yet predict the per-call
/// allocation savings from eliminating intermediate Tensor wrappers.
/// 0.10 = "saves ~10% of expected runtime on a chain-heavy function" —
/// a deliberately conservative estimate.
pub const FUSION_REWRITE_BENEFIT: f64 = 0.10;

/// Hardcoded predicted compile cost for fusion_rewrite. Very cheap: a
/// single walk of each body with use-count map + pattern match. The
/// 0.02 figure reflects "less than 2% of the canonical 6-pass build
/// time" — should be well-amortized.
pub const FUSION_REWRITE_COMPILE_COST: f64 = 0.02;

// ---------------------------------------------------------------------------
// Recommendation types
// ---------------------------------------------------------------------------

/// A single ranked pass recommendation for one function.
#[derive(Debug, Clone, PartialEq)]
pub struct PassRecommendation {
    /// Pass name (one of [`CANONICAL_PASSES`]).
    pub pass_name: String,
    /// Predicted normalized runtime benefit in [0, 0.5].
    pub predicted_benefit: f64,
    /// Predicted normalized compile cost in [0.01, 1.0].
    pub predicted_compile_cost: f64,
    /// Cost-model confidence in [0, 1] for this recommendation.
    pub confidence: f64,
    /// Why the ranker made this recommendation.
    pub rationale: RankingRationale,
}

/// Structured explanation for why a pass was (or wasn't) recommended.
/// Used in audit reports so a reviewer can understand each decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankingRationale {
    /// Predicted benefit above threshold; pass is kept.
    BenefitAboveThreshold,
    /// Pass kept because it appears in the canonical pipeline even though
    /// its predicted benefit is unknown (cost model returned `Unknown`).
    /// This is the conservative default — we'd rather over-run optimization
    /// than skip a pass we can't predict for.
    UnknownButKeptConservatively,
    /// Pass dropped because predicted benefit is below the skip threshold.
    BelowSkipThreshold,
    /// Pass dropped because the legality gate would reject it.
    LegalityGateRejected,
    /// Phase 3.5c+ — pass kept because the function has fusion-worthy
    /// candidates (length ≥ 2 chain of native primitives) identified by
    /// [`crate::fusion::identify_fusion_candidates`]. Bypasses the cost
    /// model: fusion eliminates intermediate-tensor allocations, a benefit
    /// the linear cost model can't yet quantify.
    FusionCandidatesIdentified,
}

/// Per-function ranked recommendation set.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FunctionRanking {
    /// Recommendations in ranked order (highest benefit first).
    pub recommended: Vec<PassRecommendation>,
    /// Recommendations dropped from the sequence, with rationale.
    pub dropped: Vec<PassRecommendation>,
}

impl FunctionRanking {
    pub fn recommendation_count(&self) -> usize {
        self.recommended.len()
    }

    pub fn dropped_count(&self) -> usize {
        self.dropped.len()
    }
}

/// Full program-level ranking output.
#[derive(Debug, Clone, Default)]
pub struct RankingReport {
    /// Per-function ranking, keyed by function name.
    pub per_fn: BTreeMap<String, FunctionRanking>,
    /// The bundled `PassSequence` ready to feed the legality gate.
    pub sequence: PassSequence,
    /// The legality verdict over `sequence`. Always `Approved` when the
    /// ranker is configured with the default legality gate, because the
    /// ranker drops illegal recommendations *before* assembling the
    /// sequence. Included anyway as a defensive sanity check.
    pub verdict: LegalityVerdict,
}

impl RankingReport {
    /// Total number of recommended passes across all functions.
    pub fn total_recommended(&self) -> usize {
        self.per_fn.values().map(|r| r.recommended.len()).sum()
    }

    /// Total number of dropped passes across all functions.
    pub fn total_dropped(&self) -> usize {
        self.per_fn.values().map(|r| r.dropped.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// PassRanker
// ---------------------------------------------------------------------------

/// The ranker.
///
/// Inject a cost model and a legality gate at construction time. The ranker
/// is otherwise stateless — `rank()` is a pure function of its inputs.
pub struct PassRanker<M: CostModel, G: LegalityGate> {
    cost_model: M,
    legality_gate: G,
    skip_threshold: f64,
}

impl<M: CostModel, G: LegalityGate> PassRanker<M, G> {
    pub fn new(cost_model: M, legality_gate: G) -> Self {
        Self {
            cost_model,
            legality_gate,
            skip_threshold: DEFAULT_SKIP_THRESHOLD,
        }
    }

    /// Override the skip threshold (default `DEFAULT_SKIP_THRESHOLD`).
    pub fn with_skip_threshold(mut self, threshold: f64) -> Self {
        self.skip_threshold = threshold;
        self
    }

    /// Borrow the wrapped cost model (e.g. for auditability surfaces).
    pub fn cost_model(&self) -> &M {
        &self.cost_model
    }

    /// Produce a ranking report for every function in the program.
    pub fn rank(&self, program: &MirProgram, features: &CanaFeatures) -> RankingReport {
        let mut per_fn: BTreeMap<String, FunctionRanking> = BTreeMap::new();
        let mut sequence = PassSequence::empty();

        // CANA Phase 3.5c+: pre-compute fusion candidates once for the
        // whole program. Each function-level rank inspects this to decide
        // whether to inject the `fusion_rewrite` pass.
        let fusion_plan = crate::fusion::identify_fusion_candidates(program);
        let fn_has_fusion: BTreeMap<&str, bool> = program
            .functions
            .iter()
            .map(|f| {
                let any = fusion_plan
                    .fusion_worthy()
                    .any(|c| c.function_name == f.name);
                (f.name.as_str(), any)
            })
            .collect();

        // Iteration is in BTreeMap key order → deterministic per function.
        for fn_name in features.per_fn.keys() {
            let mut ranking = self.rank_function(program, features, fn_name);

            // CANA Phase 3.5c+ — auto-inject fusion_rewrite when the
            // function has fusion-worthy chains. We append (rather than
            // sort-into-place) because fusion_rewrite is semantics-
            // preserving and runs AFTER the canonical optimization passes
            // — those passes simplify expressions that may expose new
            // fusion candidates, so fusion runs last by convention.
            if fn_has_fusion.get(fn_name.as_str()).copied().unwrap_or(false) {
                let rec = PassRecommendation {
                    pass_name: FUSION_REWRITE_PASS_NAME.to_string(),
                    predicted_benefit: FUSION_REWRITE_BENEFIT,
                    predicted_compile_cost: FUSION_REWRITE_COMPILE_COST,
                    confidence: 1.0,
                    rationale: RankingRationale::FusionCandidatesIdentified,
                };
                // Legality gate check, identical pattern to rank_function.
                let mut single = PassSequence::empty();
                single.per_function.insert(
                    fn_name.clone(),
                    vec![ProposedPass::Run(rec.pass_name.clone())],
                );
                let verdict = self.legality_gate.verify(program, &single, features);
                match verdict {
                    LegalityVerdict::Approved => {
                        ranking.recommended.push(rec);
                    }
                    LegalityVerdict::Rejected(_) => {
                        let mut rejected = rec;
                        rejected.rationale = RankingRationale::LegalityGateRejected;
                        ranking.dropped.push(rejected);
                    }
                }
            }

            // Populate the proposed `PassSequence` with the kept passes,
            // in their ranked order, as `ProposedPass::Run`.
            let proposed_passes: Vec<ProposedPass> = ranking
                .recommended
                .iter()
                .map(|r| ProposedPass::Run(r.pass_name.clone()))
                .collect();
            if !proposed_passes.is_empty() {
                sequence
                    .per_function
                    .insert(fn_name.clone(), proposed_passes);
            }
            per_fn.insert(fn_name.clone(), ranking);
        }

        // Defensive verdict: the ranker already filters illegal recommendations,
        // so this should always be Approved. We include it so callers can
        // detect a regression in the ranker's own filtering logic.
        let verdict = self.legality_gate.verify(program, &sequence, features);

        RankingReport {
            per_fn,
            sequence,
            verdict,
        }
    }

    /// Rank passes for a single function. Returns kept + dropped lists.
    fn rank_function(
        &self,
        program: &MirProgram,
        features: &CanaFeatures,
        fn_name: &str,
    ) -> FunctionRanking {
        let mut candidates: Vec<(PassRecommendation, bool)> =
            Vec::with_capacity(CANONICAL_PASSES.len());

        for &pass in CANONICAL_PASSES {
            // Phase 2 alias: cf_round_2 has the same cost shape as
            // constant_fold but runs second in the default pipeline. We
            // query the model with "constant_fold" and tag the alias on
            // the recommendation.
            let model_key = if pass == "cf_round_2" {
                "constant_fold"
            } else {
                pass
            };

            let benefit_est = self.cost_model.query(
                program,
                features,
                &CostQuery::PassBenefit {
                    function_name: fn_name,
                    pass_name: model_key,
                },
            );
            let compile_est = self.cost_model.query(
                program,
                features,
                &CostQuery::PassRuntime {
                    function_name: fn_name,
                    pass_name: model_key,
                },
            );

            let (benefit, confidence, rationale_if_kept) = match benefit_est {
                CostEstimate::Estimated { value, confidence } => {
                    if value >= self.skip_threshold {
                        (value, confidence, RankingRationale::BenefitAboveThreshold)
                    } else {
                        // Below threshold — drop, but record the prediction.
                        let drop_rec = PassRecommendation {
                            pass_name: pass.to_string(),
                            predicted_benefit: value,
                            predicted_compile_cost: compile_est.value().unwrap_or(0.0),
                            confidence,
                            rationale: RankingRationale::BelowSkipThreshold,
                        };
                        candidates.push((drop_rec, false));
                        continue;
                    }
                }
                CostEstimate::Unknown => {
                    // Conservative default: keep the pass at low benefit
                    // priority so it still runs but ranks last.
                    (0.001, 0.0, RankingRationale::UnknownButKeptConservatively)
                }
            };

            let rec = PassRecommendation {
                pass_name: pass.to_string(),
                predicted_benefit: benefit,
                predicted_compile_cost: compile_est.value().unwrap_or(0.05),
                confidence,
                rationale: rationale_if_kept,
            };
            candidates.push((rec, true));
        }

        // Sort kept recommendations by (descending benefit, ascending name)
        // for deterministic ordering when benefits tie.
        let (mut kept, dropped): (Vec<_>, Vec<_>) = candidates.into_iter().partition(|(_, k)| *k);
        kept.sort_by(|a, b| {
            b.0.predicted_benefit
                .partial_cmp(&a.0.predicted_benefit)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.pass_name.cmp(&b.0.pass_name))
        });

        // Now apply legality gating per recommendation. Build a single-pass
        // sequence for each candidate and check it; if rejected, move to
        // dropped with `LegalityGateRejected`.
        let mut final_kept = Vec::with_capacity(kept.len());
        let mut final_dropped = dropped.into_iter().map(|(r, _)| r).collect::<Vec<_>>();
        for (rec, _) in kept {
            let mut single = PassSequence::empty();
            single.per_function.insert(
                fn_name.to_string(),
                vec![ProposedPass::Run(rec.pass_name.clone())],
            );
            let verdict = self.legality_gate.verify(program, &single, features);
            match verdict {
                LegalityVerdict::Approved => final_kept.push(rec),
                LegalityVerdict::Rejected(_) => {
                    let mut rejected = rec;
                    rejected.rationale = RankingRationale::LegalityGateRejected;
                    final_dropped.push(rejected);
                }
            }
        }

        FunctionRanking {
            recommended: final_kept,
            dropped: final_dropped,
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience builder using the default Phase 2 components
// ---------------------------------------------------------------------------

/// Build a ranker with Phase 2 defaults: `LinearCostModel` +
/// `DefaultLegalityGate`. The standard entry point for callers that just
/// want recommendations without dependency injection.
pub fn default_ranker() -> PassRanker<crate::linear_cost_model::LinearCostModel, crate::legality::DefaultLegalityGate>
{
    PassRanker::new(
        crate::linear_cost_model::LinearCostModel::new(),
        crate::legality::DefaultLegalityGate::new(),
    )
}

// ---------------------------------------------------------------------------
// Conversion: CANA's PassSequence → cjc-mir's PassPlan
// ---------------------------------------------------------------------------

/// Convert a CANA `PassSequence` into a `cjc_mir::optimize::PassPlan`.
///
/// This is the dependency-direction-safe bridge between CANA's
/// recommendation surface and the MIR optimizer. CANA's `PassSequence`
/// holds `ProposedPass::Run("name")` entries; `cjc_mir::PassPlan` holds
/// plain pass-name strings. We unpack the run/skip distinction (Phase 2
/// only emits `Run`; future `Skip` entries can be modeled by simply
/// omitting from the plan).
///
/// Functions with no entry in `sequence` are absent from the returned
/// plan, which means `cjc_mir::optimize_program_with_plan` will run the
/// default 6-pass sequence on them. This is the safe fallback.
///
/// # Determinism
///
/// Output ordering inherits from `PassSequence::per_function`'s
/// `BTreeMap` iteration; the per-function pass vec preserves CANA's
/// recommended order verbatim.
pub fn pass_plan_from(sequence: &PassSequence) -> cjc_mir::optimize::PassPlan {
    let mut plan = cjc_mir::optimize::PassPlan::empty();
    for (fn_name, proposed_passes) in &sequence.per_function {
        let run_only: Vec<String> = proposed_passes
            .iter()
            .filter_map(|p| match p {
                ProposedPass::Run(name) => Some(name.clone()),
                ProposedPass::Skip(_) => None, // Phase 2 ignores Skip
            })
            .collect();
        if !run_only.is_empty() {
            plan.per_function.insert(fn_name.clone(), run_only);
        }
    }
    plan
}

/// Convenience: run the default ranker and convert the resulting
/// sequence to a `cjc_mir::PassPlan` in one step.
///
/// This is the entry point `cjc-mir-exec` calls when `--mir-opt` is set.
pub fn recommend_pass_plan(
    program: &cjc_mir::MirProgram,
) -> (RankingReport, cjc_mir::optimize::PassPlan) {
    let features = crate::analyze_program(program).features;
    let ranker = default_ranker();
    let report = ranker.rank(program, &features);
    let plan = pass_plan_from(&report.sequence);
    (report, plan)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::extract;
    use cjc_mir::{MirBody, MirExpr, MirExprKind, MirFnId, MirFunction, MirProgram, MirStmt};

    fn program_with_loops() -> MirProgram {
        MirProgram {
            functions: vec![MirFunction {
                id: MirFnId(0),
                name: "with_loops".to_string(),
                type_params: vec![],
                params: vec![],
                return_type: None,
                body: MirBody {
                    stmts: (0..6)
                        .map(|i| MirStmt::Expr(MirExpr { kind: MirExprKind::IntLit(i) }))
                        .chain(std::iter::once(MirStmt::While {
                            cond: MirExpr { kind: MirExprKind::BoolLit(true) },
                            body: MirBody {
                                stmts: (0..6)
                                    .map(|i| MirStmt::Expr(MirExpr { kind: MirExprKind::IntLit(i * 10) }))
                                    .collect(),
                                result: None,
                            },
                        }))
                        .collect(),
                    result: None,
                },
                is_nogc: false,
                cfg_body: None,
                decorators: vec![],
                vis: cjc_ast::Visibility::Public,
                local_count: 0,
            }],
            struct_defs: vec![],
            enum_defs: vec![],
            entry: MirFnId(0),
        }
    }

    fn empty_program() -> MirProgram {
        MirProgram {
            functions: vec![MirFunction {
                id: MirFnId(0),
                name: "empty".to_string(),
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
            }],
            struct_defs: vec![],
            enum_defs: vec![],
            entry: MirFnId(0),
        }
    }

    #[test]
    fn ranking_is_deterministic_across_runs() {
        let p = program_with_loops();
        let f = extract(&p);
        let ranker = default_ranker();
        let first = ranker.rank(&p, &f);
        for _ in 0..50 {
            let again = ranker.rank(&p, &f);
            assert_eq!(again.per_fn, first.per_fn);
            assert_eq!(again.sequence, first.sequence);
        }
    }

    #[test]
    fn ranking_for_empty_program_runs_passes_at_low_priority() {
        // Empty function has nothing to optimize, so most passes should
        // either drop below threshold or land at conservative-low benefit.
        let p = empty_program();
        let f = extract(&p);
        let ranker = default_ranker();
        let report = ranker.rank(&p, &f);
        let r = report.per_fn.get("empty").expect("function ranked");
        // Total of kept + dropped equals canonical pass count (6).
        assert_eq!(r.recommended.len() + r.dropped.len(), CANONICAL_PASSES.len());
    }

    #[test]
    fn licm_appears_in_loop_program_recommendations() {
        let p = program_with_loops();
        let f = extract(&p);
        let ranker = default_ranker();
        let report = ranker.rank(&p, &f);
        let r = report.per_fn.get("with_loops").expect("function ranked");
        let licm_present = r.recommended.iter().any(|rec| rec.pass_name == "licm");
        assert!(
            licm_present,
            "licm should be recommended for programs with loops; got: {:?}",
            r.recommended.iter().map(|rec| &rec.pass_name).collect::<Vec<_>>()
        );
    }

    #[test]
    fn recommended_passes_are_in_descending_benefit_order() {
        let p = program_with_loops();
        let f = extract(&p);
        let ranker = default_ranker();
        let report = ranker.rank(&p, &f);
        for ranking in report.per_fn.values() {
            for window in ranking.recommended.windows(2) {
                assert!(
                    window[0].predicted_benefit >= window[1].predicted_benefit,
                    "ranking out of order: {:?} then {:?}",
                    window[0],
                    window[1]
                );
            }
        }
    }

    #[test]
    fn ranker_verdict_is_approved_under_default_gate() {
        // The ranker filters illegal recommendations before building the
        // sequence, so the final verdict should always be Approved.
        let p = program_with_loops();
        let f = extract(&p);
        let ranker = default_ranker();
        let report = ranker.rank(&p, &f);
        assert!(
            report.verdict.is_approved(),
            "ranker's filtered sequence should always pass legality: {:?}",
            report.verdict
        );
    }

    #[test]
    fn skip_threshold_controls_drop_decisions() {
        let p = program_with_loops();
        let f = extract(&p);

        // A very high threshold should drop most passes.
        let strict = PassRanker::new(
            crate::linear_cost_model::LinearCostModel::new(),
            crate::legality::DefaultLegalityGate::new(),
        )
        .with_skip_threshold(0.5);
        let strict_report = strict.rank(&p, &f);
        let strict_kept = strict_report.total_recommended();

        // A zero threshold should keep more (or equal) passes.
        let lax = PassRanker::new(
            crate::linear_cost_model::LinearCostModel::new(),
            crate::legality::DefaultLegalityGate::new(),
        )
        .with_skip_threshold(0.0);
        let lax_report = lax.rank(&p, &f);
        let lax_kept = lax_report.total_recommended();

        assert!(
            lax_kept >= strict_kept,
            "lax threshold should keep ≥ strict's count ({} vs {})",
            lax_kept,
            strict_kept
        );
    }

    #[test]
    fn dropped_recommendations_carry_rationale() {
        let p = program_with_loops();
        let f = extract(&p);
        // Force every pass to be dropped via an absurd threshold.
        let ranker = PassRanker::new(
            crate::linear_cost_model::LinearCostModel::new(),
            crate::legality::DefaultLegalityGate::new(),
        )
        .with_skip_threshold(10.0);
        let report = ranker.rank(&p, &f);
        for ranking in report.per_fn.values() {
            for dropped in &ranking.dropped {
                assert!(matches!(
                    dropped.rationale,
                    RankingRationale::BelowSkipThreshold | RankingRationale::LegalityGateRejected
                ));
            }
        }
    }

    #[test]
    fn canonical_pass_count_is_six() {
        assert_eq!(CANONICAL_PASSES.len(), 6);
    }

    // ---------------------------------------------------------------
    // Conversion tests: PassSequence -> cjc-mir PassPlan
    // ---------------------------------------------------------------

    #[test]
    fn pass_plan_from_empty_sequence_is_empty() {
        let seq = PassSequence::empty();
        let plan = pass_plan_from(&seq);
        assert!(plan.per_function.is_empty());
    }

    #[test]
    fn pass_plan_from_translates_run_entries_verbatim() {
        let mut seq = PassSequence::empty();
        seq.per_function.insert(
            "main".to_string(),
            vec![
                ProposedPass::Run("constant_fold".to_string()),
                ProposedPass::Run("licm".to_string()),
                ProposedPass::Run("dce".to_string()),
            ],
        );
        let plan = pass_plan_from(&seq);
        let entries = plan.per_function.get("main").expect("main present");
        assert_eq!(entries, &vec!["constant_fold", "licm", "dce"]);
    }

    #[test]
    fn pass_plan_from_drops_skip_entries() {
        let mut seq = PassSequence::empty();
        seq.per_function.insert(
            "main".to_string(),
            vec![
                ProposedPass::Run("cf".to_string()),
                ProposedPass::Skip("cse".to_string()),
                ProposedPass::Run("dce".to_string()),
            ],
        );
        let plan = pass_plan_from(&seq);
        let entries = plan.per_function.get("main").expect("main present");
        // Skip is dropped; Run entries are preserved in order.
        assert_eq!(entries, &vec!["cf", "dce"]);
    }

    #[test]
    fn pass_plan_from_omits_function_with_only_skips() {
        let mut seq = PassSequence::empty();
        seq.per_function.insert(
            "skip_only".to_string(),
            vec![ProposedPass::Skip("cf".to_string())],
        );
        let plan = pass_plan_from(&seq);
        // A function with only Skip entries → no entry in the plan,
        // meaning the MIR optimizer will fall back to default. This is
        // the safe fallback.
        assert!(plan.per_function.is_empty());
    }

    #[test]
    fn pass_plan_from_preserves_btreemap_ordering() {
        let mut seq = PassSequence::empty();
        for name in ["zebra", "alpha", "middle"].iter() {
            seq.per_function.insert(
                name.to_string(),
                vec![ProposedPass::Run("cf".to_string())],
            );
        }
        let plan = pass_plan_from(&seq);
        let keys: Vec<&str> = plan.per_function.keys().map(|s| s.as_str()).collect();
        // BTreeMap → sorted iteration → "alpha", "middle", "zebra"
        assert_eq!(keys, vec!["alpha", "middle", "zebra"]);
    }

    #[test]
    fn recommend_pass_plan_end_to_end_returns_plan() {
        let p = program_with_loops();
        let (report, plan) = recommend_pass_plan(&p);
        // The plan should have an entry for the function (which has loops
        // so LICM gets recommended).
        assert!(plan.per_function.contains_key("with_loops"));
        // And the verdict should be Approved.
        assert!(report.verdict.is_approved());
    }

    #[test]
    fn recommend_pass_plan_is_deterministic() {
        let p = program_with_loops();
        let (_, first) = recommend_pass_plan(&p);
        for _ in 0..50 {
            let (_, again) = recommend_pass_plan(&p);
            assert_eq!(again, first);
        }
    }

    // -----------------------------------------------------------------------
    // CANA Phase 3.5c+ — auto-inject fusion_rewrite tests
    // -----------------------------------------------------------------------

    fn var(n: &str) -> MirExpr {
        MirExpr { kind: MirExprKind::Var(n.to_string()) }
    }

    fn call(callee: &str, args: Vec<MirExpr>) -> MirExpr {
        MirExpr {
            kind: MirExprKind::Call {
                callee: Box::new(var(callee)),
                args,
            },
        }
    }

    fn let_stmt(name: &str, init: MirExpr) -> MirStmt {
        MirStmt::Let {
            name: name.to_string(),
            mutable: false,
            init,
            alloc_hint: None,
            slot: None,
        }
    }

    fn program_with_matmul_norm_chain() -> MirProgram {
        MirProgram {
            functions: vec![MirFunction {
                id: MirFnId(0),
                name: "f_with_chain".to_string(),
                type_params: vec![],
                params: vec![],
                return_type: None,
                body: MirBody {
                    stmts: vec![
                        let_stmt("h", call("matmul", vec![var("a"), var("w")])),
                        let_stmt("n", call("norm", vec![var("h")])),
                    ],
                    result: None,
                },
                is_nogc: false,
                cfg_body: None,
                decorators: vec![],
                vis: cjc_ast::Visibility::Public,
                local_count: 0,
            }],
            struct_defs: vec![],
            enum_defs: vec![],
            entry: MirFnId(0),
        }
    }

    #[test]
    fn fusion_rewrite_injected_when_chain_present() {
        let p = program_with_matmul_norm_chain();
        let f = extract(&p);
        let ranker = default_ranker();
        let report = ranker.rank(&p, &f);

        let r = report.per_fn.get("f_with_chain").expect("function ranked");
        assert!(
            r.recommended.iter().any(|rec| rec.pass_name == FUSION_REWRITE_PASS_NAME),
            "fusion_rewrite should be auto-injected; got: {:?}",
            r.recommended.iter().map(|r| &r.pass_name).collect::<Vec<_>>()
        );

        // The sequence also carries it.
        let seq = report
            .sequence
            .per_function
            .get("f_with_chain")
            .expect("sequence has the function");
        assert!(
            seq.iter().any(|p| matches!(p, ProposedPass::Run(n) if n == FUSION_REWRITE_PASS_NAME)),
            "PassSequence should include fusion_rewrite"
        );
    }

    #[test]
    fn fusion_rewrite_carries_correct_rationale() {
        let p = program_with_matmul_norm_chain();
        let f = extract(&p);
        let ranker = default_ranker();
        let report = ranker.rank(&p, &f);
        let r = report.per_fn.get("f_with_chain").expect("function ranked");
        let rec = r
            .recommended
            .iter()
            .find(|rec| rec.pass_name == FUSION_REWRITE_PASS_NAME)
            .expect("fusion_rewrite present");
        assert_eq!(rec.rationale, RankingRationale::FusionCandidatesIdentified);
        assert!((rec.predicted_benefit - FUSION_REWRITE_BENEFIT).abs() < 1e-9);
        assert!((rec.predicted_compile_cost - FUSION_REWRITE_COMPILE_COST).abs() < 1e-9);
    }

    #[test]
    fn fusion_rewrite_not_injected_without_candidates() {
        // Empty program has no fusion candidates → no fusion_rewrite.
        let p = empty_program();
        let f = extract(&p);
        let ranker = default_ranker();
        let report = ranker.rank(&p, &f);
        let r = report.per_fn.get("empty").expect("function ranked");
        assert!(
            !r.recommended.iter().any(|rec| rec.pass_name == FUSION_REWRITE_PASS_NAME),
            "fusion_rewrite must NOT be injected for a function with no candidates"
        );
    }

    #[test]
    fn fusion_rewrite_pass_plan_includes_the_pass() {
        let p = program_with_matmul_norm_chain();
        let (_, plan) = recommend_pass_plan(&p);
        let passes = plan
            .per_function
            .get("f_with_chain")
            .expect("function in plan");
        assert!(
            passes.iter().any(|p| p == FUSION_REWRITE_PASS_NAME),
            "PassPlan should propagate fusion_rewrite to cjc-mir; got: {passes:?}"
        );
    }

    #[test]
    fn fusion_rewrite_injection_is_deterministic() {
        // Same program, run 50 times, fusion_rewrite is always either
        // present-everywhere or absent-everywhere across runs.
        let p = program_with_matmul_norm_chain();
        let f = extract(&p);
        let ranker = default_ranker();
        let first = ranker.rank(&p, &f);
        for _ in 0..50 {
            let again = ranker.rank(&p, &f);
            assert_eq!(again.per_fn, first.per_fn);
            assert_eq!(again.sequence, first.sequence);
        }
    }

    #[test]
    fn fusion_rewrite_runs_last_after_canonical_passes() {
        // Convention: fusion runs AFTER the canonical optimization passes
        // so CF/DCE/CSE can simplify what fusion will rewrite.
        let p = program_with_matmul_norm_chain();
        let f = extract(&p);
        let ranker = default_ranker();
        let report = ranker.rank(&p, &f);
        let seq = report
            .sequence
            .per_function
            .get("f_with_chain")
            .expect("function in sequence");
        let last = seq.last().expect("sequence non-empty");
        match last {
            ProposedPass::Run(name) => assert_eq!(name, FUSION_REWRITE_PASS_NAME),
            other => panic!("expected fusion_rewrite last, got {other:?}"),
        }
    }
}
