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

        // Iteration is in BTreeMap key order → deterministic per function.
        for fn_name in features.per_fn.keys() {
            let ranking = self.rank_function(program, features, fn_name);

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
}
