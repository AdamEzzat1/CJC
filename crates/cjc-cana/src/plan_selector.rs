//! Phase C — `PassPlanSelector`: choose-among-plans, energy-criterion.
//!
//! ## Architecture position (research doc §1)
//!
//! The ranker DECIDES a recommended pass set per function; the selector
//! CHOOSES among a deterministic candidate set of whole plans — the two
//! concerns stay separate (this wraps a ranked plan, it does not live
//! inside `EnergyAwarePassRanker`). Until Phase C the energy layer had
//! a one-candidate space, which is why every prior ablation measured
//! ZERO outcome effect from it; this module is the designed fix.
//!
//! ## Candidate set (10 per function, fixed IDs, fixed order)
//!
//! | id | plan |
//! |----|------|
//! | 0 | the RANKED plan for this function (explicit `DEFAULT_PASS_SEQUENCE` when the ranked plan has no entry — absence means "default", never "nothing") |
//! | 1 | none (empty plan) |
//! | 2 | all of [`CANONICAL_PASSES`] |
//! | 3–9 | each canonical pass as a singleton |
//!
//! Every candidate is legality-filtered per `(function, pass)` through
//! the SAME gate the forced-plan ablation configs use — the gate
//! retains final authority; rejected pairs are dropped and counted.
//!
//! ## Scoring
//!
//! Each candidate is scored by the TRAINED energy head
//! ([`PinnEnergyV1`], CPB1 bundle, shadow gate PROMOTE) over a
//! per-function [`EnergyQuery`]: neutral-pass workload estimates,
//! plan pass-counts, loop features, and the post-plan node count
//! obtained by ACTUALLY applying the candidate to the function's MIR
//! (deterministic, compile-time). The winner is the
//! `(predicted, candidate_id)` minimum under `f64::total_cmp` —
//! a total order, so ties break to the lower id, deterministically.
//!
//! ## Granularity note (mirrors the thermal head's)
//!
//! The head trained on PROGRAM-level rows; selection queries are
//! per-FUNCTION. Log-magnitude features shift by the aggregation
//! factor and the density/count features are scale-invariant; the
//! ablation harness's `selector_rec` config measures the consequence
//! against real labels — exactly the shadow-before-trust pattern.
//!
//! ## Determinism
//!
//! Pure function of `(head, MIR, features, ranked plan, gate)`:
//! BTreeMap iteration, fixed candidate order, `total_cmp` argmin, no
//! RNG, no clocks. Selector identity ([`ENERGY_SELECTOR_ID`],
//! [`ENERGY_SELECTOR_VERSION`]) is exposed for report-hash joining.

use std::collections::BTreeMap;

use cjc_mir::optimize::{optimize_function_with_passes, PassPlan, DEFAULT_PASS_SEQUENCE};
use cjc_mir::MirProgram;

use crate::features::CanaFeatures;
use crate::memory_proxy::MemoryProxy;
use crate::legality::{LegalityGate, LegalityVerdict, PassSequence, PerPassLegalityGate, ProposedPass};
use crate::pass_ranker::CANONICAL_PASSES;
use crate::physical_cost::build_physical_query;
use crate::pinn_energy_v1::{EnergyQuery, PinnEnergyV1};

/// Stable selector identifier; joins report identity wherever the
/// selector drives plan choice (precedent: `PINN_V2_MODEL_ID`).
pub const ENERGY_SELECTOR_ID: &str = "energy_selector_v1";

/// Monotonic selector version; bump on any change to the candidate
/// set, the scoring query construction, or the tie-break rule.
pub const ENERGY_SELECTOR_VERSION: u32 = 1;

/// One scored candidate for one function.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredCandidate {
    /// Stable candidate id (0 = ranked … 9 = last singleton).
    pub id: u32,
    /// The legality-filtered pass list this candidate would run.
    pub passes: Vec<String>,
    /// Passes the legality gate removed from this candidate.
    pub dropped: u32,
    /// The head's predicted `ln(score)` (lower = cheaper).
    pub predicted_ln_score: f64,
}

/// Per-function selection outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionSelection {
    /// The winning candidate.
    pub chosen: ScoredCandidate,
    /// Predicted `ln(score)` of candidate 0 (the ranked plan) — the
    /// never-worse-than-ranked gate compares against this.
    pub ranked_predicted: f64,
    /// Number of candidates scored (always 10 today; recorded so the
    /// report stays honest if the set ever changes).
    pub candidates_scored: u32,
}

/// Whole-program selection report.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionReport {
    /// The selected plan. Contains an EXPLICIT entry for every
    /// featurized function — never relies on the absence-means-default
    /// fallback (`cjc-mir/src/optimize.rs` plan semantics).
    pub plan: PassPlan,
    /// Per-function outcomes, keyed by function name.
    pub per_function: BTreeMap<String, FunctionSelection>,
}

/// Deterministic argmin over `(predicted, candidate_id)` pairs under
/// `f64::total_cmp` — a TOTAL order (NaN included), so exactly one
/// winner exists for any non-empty input and the winner is invariant
/// under permutation. Public so the property tests can hammer it
/// directly.
pub fn select_argmin(scored: &[(f64, u32)]) -> Option<u32> {
    scored
        .iter()
        .min_by(|(pa, ia), (pb, ib)| pa.total_cmp(pb).then(ia.cmp(ib)))
        .map(|(_, id)| *id)
}

/// The selector: a trained energy head + the legality gate it defers to.
#[derive(Debug, Clone)]
pub struct PassPlanSelector {
    head: PinnEnergyV1,
}

impl PassPlanSelector {
    /// Build a selector around a trained (validated) energy head.
    /// Invalid heads are refused — the caller must fall back to the
    /// ranked plan rather than select with garbage weights.
    pub fn new(head: PinnEnergyV1) -> Option<Self> {
        if head.is_valid() {
            Some(Self { head })
        } else {
            None
        }
    }

    /// Access the head (read-only; identity feeds report hashes).
    pub fn head(&self) -> &PinnEnergyV1 {
        &self.head
    }

    /// Select a plan for every featurized function of `mir`.
    ///
    /// `ranked_plan` is candidate 0's source: functions absent there
    /// are treated as `DEFAULT_PASS_SEQUENCE` (matching the executor's
    /// absence semantics), made EXPLICIT before scoring.
    pub fn select(
        &self,
        mir: &MirProgram,
        features: &CanaFeatures,
        ranked_plan: &PassPlan,
        gate: &PerPassLegalityGate,
    ) -> SelectionReport {
        let mut plan = PassPlan::empty();
        let mut per_function = BTreeMap::new();

        for (fn_name, ff) in &features.per_fn {
            // -- Candidate pass lists (pre-legality), fixed order ----------
            let ranked: Vec<String> = ranked_plan
                .per_function
                .get(fn_name)
                .cloned()
                .unwrap_or_else(|| {
                    DEFAULT_PASS_SEQUENCE.iter().map(|s| s.to_string()).collect()
                });
            let mut raw_candidates: Vec<Vec<String>> = Vec::with_capacity(10);
            raw_candidates.push(ranked);
            raw_candidates.push(Vec::new());
            raw_candidates.push(CANONICAL_PASSES.iter().map(|s| s.to_string()).collect());
            for p in CANONICAL_PASSES {
                raw_candidates.push(vec![p.to_string()]);
            }

            // -- Legality filter + scoring, with a per-distinct-plan cache
            //    (a filtered singleton can collapse into "none"; no need
            //    to optimize the same plan twice) ------------------------
            let mut cache: BTreeMap<Vec<String>, (f64, u64)> = BTreeMap::new();
            let mut scored: Vec<ScoredCandidate> = Vec::with_capacity(raw_candidates.len());
            for (id, raw) in raw_candidates.into_iter().enumerate() {
                let mut passes: Vec<String> = Vec::with_capacity(raw.len());
                let mut dropped = 0u32;
                for p in raw {
                    let mut seq = PassSequence::default();
                    seq.per_function
                        .insert(fn_name.clone(), vec![ProposedPass::Run(p.clone())]);
                    match gate.verify(mir, &seq, features) {
                        LegalityVerdict::Approved => passes.push(p),
                        LegalityVerdict::Rejected(_) => dropped = dropped.saturating_add(1),
                    }
                }

                let (predicted, _nodes_after) = match cache.get(&passes) {
                    Some(&hit) => hit,
                    None => {
                        let nodes_after = self.post_plan_node_count(mir, fn_name, &passes);
                        let q = self.query_for(fn_name, ff, &passes, dropped, nodes_after);
                        let predicted = self.head.predict_ln_score(&q);
                        cache.insert(passes.clone(), (predicted, nodes_after));
                        (predicted, nodes_after)
                    }
                };
                scored.push(ScoredCandidate {
                    id: id as u32,
                    passes,
                    dropped,
                    predicted_ln_score: predicted,
                });
            }

            let ranked_predicted = scored[0].predicted_ln_score;
            let pairs: Vec<(f64, u32)> = scored
                .iter()
                .map(|c| (c.predicted_ln_score, c.id))
                .collect();
            let winner_id = select_argmin(&pairs).expect("candidate set is non-empty");
            let chosen = scored
                .into_iter()
                .find(|c| c.id == winner_id)
                .expect("winner id comes from the same set");

            // EXPLICIT entry — the absence-means-default trap.
            plan.per_function
                .insert(fn_name.clone(), chosen.passes.clone());
            per_function.insert(
                fn_name.clone(),
                FunctionSelection {
                    chosen,
                    ranked_predicted,
                    candidates_scored: 10,
                },
            );
        }

        SelectionReport { plan, per_function }
    }

    /// Post-plan node count for ONE function: optimize a lone clone of
    /// THAT function under `passes`
    /// ([`optimize_function_with_passes`]) and count its expression
    /// nodes. Deterministic and compile-time.
    ///
    /// ## Memory note (Phase D diagnostics §3.1)
    ///
    /// This probe originally went through
    /// `optimize_program_with_plan` — a plan running `passes` on
    /// `fn_name` and nothing anywhere else (explicit empty entries) —
    /// which clones the WHOLE program once per scored candidate. On a
    /// real workload (`examples/08_pinn_heat_equation.cjcl`) that put
    /// planning-time peak RSS at 1.63 GB vs 206 MB for the baseline
    /// arm. The per-function entry point produces a byte-identical
    /// optimized function (locked by
    /// `post_plan_node_count_matches_whole_program_probe` below and
    /// the equivalence test in `cjc-mir/src/optimize.rs`), so node
    /// counts — and therefore queries, predictions, and selected
    /// plans — are unchanged; only the footprint drops to one
    /// function clone per candidate.
    ///
    /// `expr_count` comes from [`MemoryProxy::from_function`], a pure
    /// per-function walk — exactly the value whole-program
    /// `features::extract` reported for this function. The reverse
    /// find mirrors `extract`'s last-insert-wins keying should two
    /// functions ever share a name; a missing function still counts
    /// as 0, matching the old `.get(fn_name)` fallback.
    fn post_plan_node_count(&self, mir: &MirProgram, fn_name: &str, passes: &[String]) -> u64 {
        mir.functions
            .iter()
            .rev()
            .find(|f| f.name == fn_name)
            .map(|f| {
                let optimized = optimize_function_with_passes(f, passes);
                MemoryProxy::from_function(&optimized).expr_count as u64
            })
            .unwrap_or(0)
    }

    /// Build the per-function energy query for one candidate. Workload
    /// estimates use the neutral pass (mirroring how training rows were
    /// constructed); the plan enters through pass counts, the dropped
    /// count, and the post-plan node features.
    fn query_for(
        &self,
        fn_name: &str,
        ff: &crate::features::FnFeatures,
        passes: &[String],
        dropped: u32,
        nodes_after: u64,
    ) -> EnergyQuery {
        let q = build_physical_query(fn_name, "dce", ff);
        EnergyQuery {
            flops_estimate: q.flops_estimate,
            bytes_read_estimate: q.bytes_read_estimate,
            bytes_written_estimate: q.bytes_written_estimate,
            allocation_bytes_estimate: q.allocation_bytes_estimate,
            working_set_bytes_estimate: q.working_set_bytes_estimate,
            float_ops_estimate: q.float_ops_estimate,
            mir_nodes_before: ff.memory.expr_count as u64,
            recommended_count: passes.len() as u32,
            dropped_count: dropped,
            pass_counts: self.head.pass_counts(passes.iter().map(|s| s.as_str())),
            countable_loop_count: ff.cfg.countable_loop_count as u64,
            max_loop_depth: ff.cfg.max_loop_depth,
            mir_nodes_after: nodes_after,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pinn_energy_v1::{ENERGY_TAIL_FEATURES, ENERGY_WORKLOAD_FEATURES};

    fn neutral_head() -> PinnEnergyV1 {
        let pass_names: Vec<String> = CANONICAL_PASSES.iter().map(|s| s.to_string()).collect();
        let n = ENERGY_WORKLOAD_FEATURES + pass_names.len() + ENERGY_TAIL_FEATURES;
        PinnEnergyV1 {
            pass_names,
            feature_means: vec![0.0; n],
            feature_stds: vec![1.0; n],
            coefficients: vec![0.0; n],
            intercept: 0.0,
        }
    }

    /// A head that prefers SMALLER post-plan node counts (weight on
    /// the ln(1+nodes_after) feature) — gives selection a direction.
    fn nodes_after_head() -> PinnEnergyV1 {
        let mut head = neutral_head();
        let idx = head.feature_count() - 2; // ln(1+nodes_after)
        head.coefficients[idx] = 1.0;
        head
    }

    fn lower(src: &str) -> (MirProgram, CanaFeatures) {
        let (ast, diags) = cjc_parser::parse_source(src);
        assert!(!diags.has_errors(), "{:?}", diags.diagnostics);
        let mut al = cjc_hir::AstLowering::new();
        let hir = al.lower_program(&ast);
        let mut h2m = cjc_mir::HirToMir::new();
        let mir = h2m.lower_program(&hir);
        let features = crate::analyze_program(&mir).features;
        (mir, features)
    }

    const SRC: &str = r#"
fn work(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        total = total + i * 2;
        i = i + 1;
    }
    return total;
}
print(work(100));
"#;

    #[test]
    fn invalid_head_is_refused() {
        let mut head = neutral_head();
        head.feature_stds[0] = 0.0;
        assert!(PassPlanSelector::new(head).is_none());
    }

    #[test]
    fn every_featurized_function_gets_an_explicit_entry() {
        let (mir, features) = lower(SRC);
        let selector = PassPlanSelector::new(neutral_head()).unwrap();
        let report = selector.select(&mir, &features, &PassPlan::empty(), &PerPassLegalityGate::new());
        for fn_name in features.per_fn.keys() {
            assert!(
                report.plan.per_function.contains_key(fn_name),
                "{fn_name} missing an explicit plan entry (absence-means-default trap)"
            );
        }
    }

    #[test]
    fn neutral_head_ties_break_to_ranked_candidate() {
        // All candidates predict identically under the zero head; the
        // (predicted, id) tie-break must pick id 0 — the ranked plan.
        let (mir, features) = lower(SRC);
        let selector = PassPlanSelector::new(neutral_head()).unwrap();
        let report = selector.select(&mir, &features, &PassPlan::empty(), &PerPassLegalityGate::new());
        for (f, sel) in &report.per_function {
            assert_eq!(sel.chosen.id, 0, "{f}: neutral head must keep the ranked plan");
        }
    }

    #[test]
    fn ranked_absence_means_default_sequence_not_nothing() {
        let (mir, features) = lower(SRC);
        let selector = PassPlanSelector::new(neutral_head()).unwrap();
        // Empty ranked plan → candidate 0 must be the DEFAULT sequence.
        let report = selector.select(&mir, &features, &PassPlan::empty(), &PerPassLegalityGate::new());
        let work = &report.per_function["work"];
        assert!(
            !work.chosen.passes.is_empty(),
            "candidate 0 from an absent ranked entry must be the default sequence"
        );
    }

    #[test]
    fn selection_is_deterministic_across_runs() {
        let (mir, features) = lower(SRC);
        let selector = PassPlanSelector::new(nodes_after_head()).unwrap();
        let gate = PerPassLegalityGate::new();
        let first = selector.select(&mir, &features, &PassPlan::empty(), &gate);
        for _ in 0..5 {
            let again = selector.select(&mir, &features, &PassPlan::empty(), &gate);
            assert_eq!(first, again);
        }
    }

    #[test]
    fn chosen_predicted_never_exceeds_ranked_predicted() {
        let (mir, features) = lower(SRC);
        let selector = PassPlanSelector::new(nodes_after_head()).unwrap();
        let report =
            selector.select(&mir, &features, &PassPlan::empty(), &PerPassLegalityGate::new());
        for (f, sel) in &report.per_function {
            assert!(
                sel.chosen.predicted_ln_score <= sel.ranked_predicted,
                "{f}: argmin over a set containing the ranked candidate can never be worse"
            );
        }
    }

    /// Two-function source (plus synthetic `__main`) covering loop and
    /// straight-line shapes — exercises passes that rewrite (CF/SR),
    /// shrink (DCE), and restructure (LICM/unroll).
    const SRC_MULTI: &str = r#"
fn hot(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let waste: i64 = 3 * 4 + 5;
        total = total + i * 2 + waste;
        i = i + 1;
    }
    return total;
}
fn flat(x: i64) -> i64 {
    let a: i64 = 10 * 5 + 2;
    let b: i64 = a + x;
    return b + a;
}
print(hot(100) + flat(7));
"#;

    /// Plan-identity proof at the probe level: the per-function probe
    /// must return EXACTLY what the original whole-program probe
    /// (optimize_program_with_plan over explicit-empty entries for
    /// every other function, then whole-program feature extraction)
    /// returned — for every function and every candidate-shaped pass
    /// list. Identical node counts → identical queries → identical
    /// predictions → identical selected plans; the corpus-level
    /// consequence is locked by the `selector_rec` gate in
    /// `bench/cana_ablation`.
    #[test]
    fn post_plan_node_count_matches_whole_program_probe() {
        use cjc_mir::optimize::optimize_program_with_plan;

        let (mir, features) = lower(SRC_MULTI);
        let selector = PassPlanSelector::new(neutral_head()).unwrap();

        let mut candidate_lists: Vec<Vec<String>> = vec![
            Vec::new(),
            DEFAULT_PASS_SEQUENCE.iter().map(|s| s.to_string()).collect(),
            CANONICAL_PASSES.iter().map(|s| s.to_string()).collect(),
        ];
        for p in CANONICAL_PASSES {
            candidate_lists.push(vec![p.to_string()]);
        }

        for fn_name in features.per_fn.keys() {
            for passes in &candidate_lists {
                let per_function = selector.post_plan_node_count(&mir, fn_name, passes);

                // The original whole-program probe, verbatim.
                let mut probe_plan = PassPlan::empty();
                for other in features.per_fn.keys() {
                    probe_plan.per_function.insert(other.clone(), Vec::new());
                }
                probe_plan
                    .per_function
                    .insert(fn_name.clone(), passes.clone());
                let optimized = optimize_program_with_plan(&mir, &probe_plan);
                let post = crate::features::extract(&optimized);
                let whole_program = post
                    .per_fn
                    .get(fn_name)
                    .map(|f| f.memory.expr_count as u64)
                    .unwrap_or(0);

                assert_eq!(
                    per_function, whole_program,
                    "{fn_name} under {passes:?}: probe paths disagree"
                );
            }
        }
    }

    #[test]
    fn argmin_total_order_and_permutation_invariance() {
        let scored = vec![(0.5, 3), (0.5, 1), (-0.2, 7), (f64::NAN, 2), (-0.2, 4)];
        let winner = select_argmin(&scored).unwrap();
        // total_cmp orders NaN above all finite values; -0.2 wins, and
        // among the two -0.2 entries the lower id (4) wins.
        assert_eq!(winner, 4);
        let mut shuffled = scored.clone();
        shuffled.reverse();
        assert_eq!(select_argmin(&shuffled).unwrap(), winner);
        assert_eq!(select_argmin(&[]), None);
    }
}
