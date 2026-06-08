//! Phase 2 — `LinearCostModel`: hand-tuned deterministic cost predictor.
//!
//! This is the first non-null implementation of the `CostModel` trait. It
//! predicts the per-pass *runtime gain* and per-pass *compile cost* using a
//! small linear function over `CanaFeatures`. The coefficients are
//! hand-tuned (not trained) to capture qualitative behaviour:
//!
//! - Pass `licm` matters when the function has loops; not at all otherwise
//! - Pass `cse` matters when there are many shared sub-expressions, which
//!   we approximate by `expr_count / block_count` (expressions per block —
//!   high values mean dense expression trees with reuse potential)
//! - Pass `dce` matters when there are many statements but few outputs;
//!   we approximate via `expr_count` and presence of unreachable blocks
//! - Pass `constant_fold` matters in proportion to `expr_count` (every
//!   expression is a potential folding site)
//! - Pass `strength_reduce` matters when there are binary operations
//!
//! ## Why hand-tuned, not trained
//!
//! Phase 2's primary goal is to *demonstrate that the architecture works
//! end-to-end* — featurizer → cost model → ranker → legality gate →
//! recommendations. Replacing hand-tuned coefficients with a trained model
//! is a Phase 5 task once we have a benchmark corpus.
//!
//! The hand-tuned coefficients are also **trivially auditable**: every
//! coefficient is a constant in this file with a comment explaining why it
//! has the value it does. A trained model is a black box.
//!
//! ## Determinism contract
//!
//! - All coefficients are `f64` literals — no `random()`, no clock reads.
//! - The model uses Kahan summation if/when sums exceed 4 terms (per
//!   CLAUDE.md's deterministic-reduction rule). For Phase 2 the sums are
//!   small enough that naive summation is bit-exact.
//! - `query()` is a pure function of `(program, features, query)`. Same
//!   inputs → same outputs across runs, OS, and CPU architecture.

use cjc_mir::MirProgram;

use crate::cost_model::{CostEstimate, CostModel, CostQuery};
use crate::features::{CanaFeatures, FnFeatures};

// ---------------------------------------------------------------------------
// LinearCostModel
// ---------------------------------------------------------------------------

/// Source of per-pass coefficients in a [`LinearCostModel`].
///
/// `Default` uses the hand-tuned Phase 2 coefficients in this file.
/// `Trained` uses the values fit by `bench/cana_train_cost_model/` from
/// wall-clock measurements over an 18-program corpus.
///
/// Trained coefficients exist to demonstrate the data-driven pipeline
/// works end-to-end. They are NOT necessarily better than hand-tuned on
/// every program — the corpus is small and microsecond-level wall-clock
/// noise dominates the signal for some passes. See the file-level docs
/// for the noise-floor discussion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoefficientSource {
    /// Hand-tuned, audited per-pass coefficients (the Phase 2 default).
    Default,
    /// Values fit by gradient-descent OLS over the
    /// `bench/cana_train_cost_model/` corpus. Regenerable.
    Trained,
}

/// A linear cost model over `CanaFeatures`.
///
/// Predicts normalized [0, 1] runtime cost / benefit / memory peak per
/// query. The "linear" name is the architecture, not the inputs — each
/// feature is fed through a small per-pass coefficient table, then summed.
///
/// The model has two coefficient sources: hand-tuned ([`Self::new`]) and
/// trained ([`Self::trained`]). Choose at construction time via
/// [`CoefficientSource`].
#[derive(Debug, Clone, Copy)]
pub struct LinearCostModel {
    source: CoefficientSource,
}

impl Default for LinearCostModel {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearCostModel {
    /// Construct the default Phase 2 model with hand-tuned coefficients.
    /// This is what `default_ranker()` plugs in by default.
    pub const fn new() -> Self {
        Self { source: CoefficientSource::Default }
    }

    /// Construct a model using trained coefficients fit from
    /// `bench/cana_train_cost_model/`'s 18-program corpus via
    /// gradient-descent OLS.
    ///
    /// ## Caveat
    ///
    /// The training signal is wall-clock run_us, measured at microsecond
    /// scale where OS scheduler + cache noise can exceed the per-pass
    /// benefit signal. The fitted coefficients reflect that noise — some
    /// sign-flips visible in the table below (e.g. negative
    /// `w_loop_depth` for `licm`) are statistical artifacts, not
    /// indications that LICM hurts loops.
    ///
    /// Confidence values are derived from per-pass training RMSE, so
    /// passes that fit cleanly get higher confidence and passes that
    /// fit noisily get lower confidence — the ranker downweights
    /// noisy-fit passes naturally.
    ///
    /// To regenerate: `cargo run --release -p cana-train-cost-model`
    /// then paste the emitted Rust source over `trained_pass_coefficients`.
    pub const fn trained() -> Self {
        Self { source: CoefficientSource::Trained }
    }

    /// Which coefficient source this model uses.
    pub const fn coefficient_source(&self) -> CoefficientSource {
        self.source
    }

    /// Internal: look up the per-pass coefficients for a given pass name.
    /// Returns `None` for unknown passes (the model abstains rather than
    /// guesses, matching the trait contract).
    fn pass_coefficients(&self, pass_name: &str) -> Option<PassCoefficients> {
        match self.source {
            CoefficientSource::Default => default_pass_coefficients(pass_name),
            CoefficientSource::Trained => trained_pass_coefficients(pass_name),
        }
    }
}

// Default (hand-tuned) per-pass table.
fn default_pass_coefficients(pass_name: &str) -> Option<PassCoefficients> {
    match pass_name {
            "constant_fold" | "cf" => Some(PassCoefficients {
                // Constant folding: linear in expression count. Confidence is
                // high because CF is a well-understood transform.
                w_expr_count: 0.0008,
                w_loop_depth: 0.0,
                w_branch_count: 0.0,
                w_alloc_sites: 0.0,
                base_compile_cost: 0.05,
                confidence: 0.75,
            }),
            "strength_reduce" | "sr" => Some(PassCoefficients {
                // Strength reduction: helps when there are many binary ops.
                // We approximate via expr_count — every binary is in there.
                w_expr_count: 0.0003,
                w_loop_depth: 0.05, // Doubly helpful inside loops
                w_branch_count: 0.0,
                w_alloc_sites: 0.0,
                base_compile_cost: 0.03,
                confidence: 0.5,
            }),
            "dce" | "dead_code_elimination" => Some(PassCoefficients {
                // DCE: helps proportionally to number of statements. Bonus for
                // programs with many allocation sites (more chances for
                // unused-result temporaries).
                w_expr_count: 0.0005,
                w_loop_depth: 0.0,
                w_branch_count: 0.01,
                w_alloc_sites: 0.02,
                base_compile_cost: 0.04,
                confidence: 0.7,
            }),
            "cse" | "common_subexpression_elimination" => Some(PassCoefficients {
                // CSE: helps when expressions per block is high (dense trees).
                // We model "expr density" as expr_count / max(block_count, 1)
                // in the runtime helper below — here we treat the raw count
                // as a proxy and rely on the ranker to combine.
                w_expr_count: 0.0006,
                w_loop_depth: 0.02,
                w_branch_count: -0.005, // CSE less useful when control flow is fragmented
                w_alloc_sites: 0.0,
                base_compile_cost: 0.08,
                confidence: 0.55,
            }),
            "licm" | "loop_invariant_code_motion" => Some(PassCoefficients {
                // LICM: critical when loops exist; useless otherwise. The
                // strong loop-depth coefficient captures this.
                w_expr_count: 0.0001,
                w_loop_depth: 0.15, // The dominant signal
                w_branch_count: 0.0,
                w_alloc_sites: 0.0,
                base_compile_cost: 0.07,
                confidence: 0.85,
            }),
            _ => None,
    }
}

// ---------------------------------------------------------------------------
// Trained (fit-from-data) per-pass table
// ---------------------------------------------------------------------------
//
// Generated by `cargo run --release -p cana-train-cost-model` from the
// 18-program corpus in `bench/cana_train_cost_model/programs.rs`.
//
// Each per-pass fit reports `train_rmse` for sanity-checking. RMSE
// 0.13-0.18 reflects microsecond-scale wall-clock noise floor exceeding
// the per-pass benefit signal on small programs. Negative weights and
// confidence values below 0.5 are honest reflections of that noise —
// not editorial fudges.
//
// Regenerate after corpus changes; do not hand-edit.
fn trained_pass_coefficients(pass_name: &str) -> Option<PassCoefficients> {
    match pass_name {
        "constant_fold" | "cf" => Some(PassCoefficients {
            w_expr_count: 8.509625e-3,   // train_rmse=0.1817, mean_benefit=0.1229
            w_loop_depth: -2.637812e-2,
            w_branch_count: -6.372325e-2,
            w_alloc_sites: 0.000000e0,
            base_compile_cost: 0.0500,
            confidence: 0.1000,
        }),
        "cse" | "common_subexpression_elimination" => Some(PassCoefficients {
            w_expr_count: 3.910670e-3,   // train_rmse=0.1369, mean_benefit=0.0609
            w_loop_depth: 5.012836e-4,
            w_branch_count: -2.858385e-2,
            w_alloc_sites: 0.000000e0,
            base_compile_cost: 0.0800,
            confidence: 0.2860,
        }),
        "dce" | "dead_code_elimination" => Some(PassCoefficients {
            w_expr_count: 8.110961e-3,   // train_rmse=0.1663, mean_benefit=0.1049
            w_loop_depth: 6.907174e-3,
            w_branch_count: -5.895890e-2,
            w_alloc_sites: 0.000000e0,
            base_compile_cost: 0.0400,
            confidence: 0.1552,
        }),
        "licm" | "loop_invariant_code_motion" => Some(PassCoefficients {
            w_expr_count: 9.198581e-3,   // train_rmse=0.1687, mean_benefit=0.1237
            w_loop_depth: -3.808031e-2,
            w_branch_count: -5.822055e-2,
            w_alloc_sites: 0.000000e0,
            base_compile_cost: 0.0700,
            confidence: 0.1447,
        }),
        "strength_reduce" | "sr" => Some(PassCoefficients {
            w_expr_count: 6.673866e-3,   // train_rmse=0.1295, mean_benefit=0.0728
            w_loop_depth: -5.350112e-2,
            w_branch_count: -3.017087e-2,
            w_alloc_sites: 0.000000e0,
            base_compile_cost: 0.0300,
            confidence: 0.3188,
        }),
        _ => None,
    }
}

impl LinearCostModel {
    /// Compute the predicted runtime *gain* (savings) of a pass on a
    /// specific function. Returns a value in [0, 1] where 0 = no benefit,
    /// 1 = halves runtime (approximate upper bound).
    fn predict_pass_gain(coefs: &PassCoefficients, ff: &FnFeatures) -> f64 {
        let raw = coefs.w_expr_count * ff.memory.expr_count as f64
            + coefs.w_loop_depth * ff.cfg.max_loop_depth as f64
            + coefs.w_branch_count * ff.cfg.branch_count as f64
            + coefs.w_alloc_sites * ff.memory.alloc_sites as f64;
        // Clamp to [0, 0.5] — even the most beneficial pass realistically
        // halves runtime at most. Going higher would over-promise.
        raw.clamp(0.0, 0.5)
    }

    /// Compute the predicted compile-time *cost* of a pass on a function.
    /// Returns a value in [0, 1] normalized to the slowest pass.
    fn predict_pass_compile_cost(coefs: &PassCoefficients, ff: &FnFeatures) -> f64 {
        // Compile cost grows with program size; the per-pass base cost is
        // multiplied by a size factor.
        let size_factor = 1.0 + (ff.memory.expr_count as f64).log10().max(0.0) * 0.3;
        (coefs.base_compile_cost * size_factor).clamp(0.01, 1.0)
    }

    /// Compute the predicted peak memory usage of a function (Phase 2's
    /// answer is intentionally simple — Phase 4 will refine this with
    /// allocation-lifetime predictions). Returns a normalized value where
    /// 1.0 ≈ "every allocation site materializes simultaneously."
    fn predict_peak_memory(ff: &FnFeatures) -> f64 {
        let baseline = ff.memory.alloc_sites as f64 * 0.05
            + ff.memory.tensor_heavy_ops as f64 * 0.1
            + ff.memory.cow_write_sites as f64 * 0.02;
        // Discount when there's only one block (no live-range overlap).
        let block_factor = if ff.cfg.block_count <= 1 { 0.5 } else { 1.0 };
        (baseline * block_factor).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Per-pass coefficient table
// ---------------------------------------------------------------------------

/// Hand-tuned coefficient set for one optimization pass.
#[derive(Debug, Clone, Copy)]
struct PassCoefficients {
    /// Per-expression coefficient (programs with many exprs benefit more)
    w_expr_count: f64,
    /// Per-unit loop-depth coefficient (passes targeting loops weight this)
    w_loop_depth: f64,
    /// Per-branch coefficient (control-flow-aware passes weight this)
    w_branch_count: f64,
    /// Per-allocation-site coefficient
    w_alloc_sites: f64,
    /// Base compile-time cost (in normalized units; 1.0 = slowest pass)
    base_compile_cost: f64,
    /// Model confidence in [0, 1] for this pass's prediction
    confidence: f64,
}

// ---------------------------------------------------------------------------
// CostModel implementation
// ---------------------------------------------------------------------------

impl CostModel for LinearCostModel {
    fn name(&self) -> &'static str {
        "linear_v1"
    }

    fn version(&self) -> u32 {
        // Bump whenever any coefficient in this file changes — that's the
        // signal to invalidate any cached predictions downstream (e.g. in
        // Phase 5's `PassHistory`).
        1
    }

    fn query<'a>(
        &self,
        _program: &MirProgram,
        features: &CanaFeatures,
        query: &CostQuery<'a>,
    ) -> CostEstimate {
        match query {
            CostQuery::PassRuntime {
                function_name,
                pass_name,
            } => {
                let Some(coefs) = self.pass_coefficients(pass_name) else {
                    return CostEstimate::Unknown;
                };
                let Some(ff) = features.per_fn.get(*function_name) else {
                    return CostEstimate::Unknown;
                };
                CostEstimate::Estimated {
                    value: Self::predict_pass_compile_cost(&coefs, ff),
                    confidence: coefs.confidence,
                }
            }
            CostQuery::PassBenefit {
                function_name,
                pass_name,
            } => {
                let Some(coefs) = self.pass_coefficients(pass_name) else {
                    return CostEstimate::Unknown;
                };
                let Some(ff) = features.per_fn.get(*function_name) else {
                    return CostEstimate::Unknown;
                };
                CostEstimate::Estimated {
                    value: Self::predict_pass_gain(&coefs, ff),
                    confidence: coefs.confidence,
                }
            }
            CostQuery::PeakMemory { function_name } => {
                let Some(ff) = features.per_fn.get(*function_name) else {
                    return CostEstimate::Unknown;
                };
                CostEstimate::Estimated {
                    value: Self::predict_peak_memory(ff),
                    confidence: 0.6,
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::extract;
    use cjc_mir::{MirBody, MirExpr, MirExprKind, MirFnId, MirFunction, MirProgram, MirStmt};

    fn build_simple_program() -> MirProgram {
        let body = MirBody {
            stmts: vec![
                MirStmt::Expr(MirExpr {
                    kind: MirExprKind::IntLit(1),
                }),
                MirStmt::Expr(MirExpr {
                    kind: MirExprKind::IntLit(2),
                }),
                MirStmt::While {
                    cond: MirExpr {
                        kind: MirExprKind::BoolLit(true),
                    },
                    body: MirBody {
                        stmts: vec![MirStmt::Expr(MirExpr {
                            kind: MirExprKind::IntLit(3),
                        })],
                        result: None,
                    },
                },
            ],
            result: None,
        };
        MirProgram {
            functions: vec![MirFunction {
                id: MirFnId(0),
                name: "demo".to_string(),
                type_params: vec![],
                params: vec![],
                return_type: None,
                body,
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
    fn name_and_version_are_stable() {
        let m = LinearCostModel::new();
        assert_eq!(m.name(), "linear_v1");
        assert_eq!(m.version(), 1);
    }

    #[test]
    fn unknown_pass_returns_unknown_not_zero() {
        let p = build_simple_program();
        let f = extract(&p);
        let m = LinearCostModel::new();
        let est = m.query(
            &p,
            &f,
            &CostQuery::PassBenefit {
                function_name: "demo",
                pass_name: "nonexistent_pass",
            },
        );
        assert!(est.is_unknown());
    }

    #[test]
    fn unknown_function_returns_unknown() {
        let p = build_simple_program();
        let f = extract(&p);
        let m = LinearCostModel::new();
        let est = m.query(
            &p,
            &f,
            &CostQuery::PassBenefit {
                function_name: "ghost",
                pass_name: "dce",
            },
        );
        assert!(est.is_unknown());
    }

    #[test]
    fn licm_predicts_more_benefit_with_loops_than_without() {
        let p_loop = build_simple_program(); // has a while loop
        let p_no_loop = empty_program();
        let f_loop = extract(&p_loop);
        let f_no_loop = extract(&p_no_loop);
        let m = LinearCostModel::new();

        let benefit_with_loop = m
            .query(
                &p_loop,
                &f_loop,
                &CostQuery::PassBenefit {
                    function_name: "demo",
                    pass_name: "licm",
                },
            )
            .value()
            .expect("loop program should produce an estimate for licm");

        let benefit_no_loop = m
            .query(
                &p_no_loop,
                &f_no_loop,
                &CostQuery::PassBenefit {
                    function_name: "empty",
                    pass_name: "licm",
                },
            )
            .value()
            .expect("loop-free program should still produce an estimate");

        assert!(
            benefit_with_loop > benefit_no_loop,
            "licm should benefit loop programs more (got {} vs {})",
            benefit_with_loop,
            benefit_no_loop
        );
    }

    #[test]
    fn predictions_are_deterministic_across_runs() {
        let p = build_simple_program();
        let f = extract(&p);
        let m = LinearCostModel::new();
        let q = CostQuery::PassBenefit {
            function_name: "demo",
            pass_name: "licm",
        };
        let first = m.query(&p, &f, &q);
        for _ in 0..100 {
            let again = m.query(&p, &f, &q);
            assert_eq!(again, first);
        }
    }

    #[test]
    fn predictions_are_in_normalized_range() {
        let p = build_simple_program();
        let f = extract(&p);
        let m = LinearCostModel::new();
        for pass in &["constant_fold", "strength_reduce", "dce", "cse", "licm"] {
            let est = m.query(
                &p,
                &f,
                &CostQuery::PassBenefit {
                    function_name: "demo",
                    pass_name: pass,
                },
            );
            let v = est.value().expect("known pass + known fn should produce estimate");
            assert!(
                (0.0..=0.5).contains(&v),
                "pass {} produced out-of-range benefit {}",
                pass,
                v
            );
        }
    }

    #[test]
    fn compile_cost_is_positive_and_bounded() {
        let p = build_simple_program();
        let f = extract(&p);
        let m = LinearCostModel::new();
        for pass in &["constant_fold", "strength_reduce", "dce", "cse", "licm"] {
            let est = m.query(
                &p,
                &f,
                &CostQuery::PassRuntime {
                    function_name: "demo",
                    pass_name: pass,
                },
            );
            let v = est.value().expect("known pass should produce compile-cost estimate");
            assert!(
                (0.01..=1.0).contains(&v),
                "pass {} compile cost {} out of [0.01, 1.0]",
                pass,
                v
            );
        }
    }

    #[test]
    fn peak_memory_estimate_for_empty_program_is_low() {
        let p = empty_program();
        let f = extract(&p);
        let m = LinearCostModel::new();
        let est = m.query(
            &p,
            &f,
            &CostQuery::PeakMemory {
                function_name: "empty",
            },
        );
        let v = est.value().expect("any function gets a memory estimate");
        // Empty function has no allocs → expect a very small value (allowing
        // for the block_factor floor).
        assert!(v < 0.1, "empty program should not predict heavy memory ({})", v);
    }

    #[test]
    fn confidence_is_in_unit_interval() {
        let p = build_simple_program();
        let f = extract(&p);
        let m = LinearCostModel::new();
        for pass in &["constant_fold", "dce", "cse", "licm"] {
            let est = m.query(
                &p,
                &f,
                &CostQuery::PassBenefit {
                    function_name: "demo",
                    pass_name: pass,
                },
            );
            let c = est.confidence().expect("should report confidence");
            assert!(
                (0.0..=1.0).contains(&c),
                "pass {} confidence {} not in [0,1]",
                pass,
                c
            );
        }
    }

    // -----------------------------------------------------------------------
    // Trained model tests
    // -----------------------------------------------------------------------

    #[test]
    fn trained_constructor_returns_distinct_source() {
        let default = LinearCostModel::new();
        let trained = LinearCostModel::trained();
        assert_eq!(default.coefficient_source(), CoefficientSource::Default);
        assert_eq!(trained.coefficient_source(), CoefficientSource::Trained);
    }

    #[test]
    fn trained_model_resolves_known_passes() {
        let p = build_simple_program();
        let f = extract(&p);
        let m = LinearCostModel::trained();
        for pass in &["constant_fold", "strength_reduce", "dce", "cse", "licm"] {
            let est = m.query(
                &p,
                &f,
                &CostQuery::PassBenefit {
                    function_name: "demo",
                    pass_name: pass,
                },
            );
            assert!(
                est.value().is_some(),
                "trained model should resolve known pass {}",
                pass
            );
        }
    }

    #[test]
    fn trained_model_rejects_unknown_passes() {
        let p = build_simple_program();
        let f = extract(&p);
        let m = LinearCostModel::trained();
        let est = m.query(
            &p,
            &f,
            &CostQuery::PassBenefit {
                function_name: "demo",
                pass_name: "definitely_not_a_real_pass",
            },
        );
        assert!(est.is_unknown());
    }

    #[test]
    fn trained_predictions_stay_in_normalized_range() {
        // Even with negative-sign weights from noisy fits, the clamp in
        // predict_pass_gain ensures predictions stay in [0, 0.5].
        let p = build_simple_program();
        let f = extract(&p);
        let m = LinearCostModel::trained();
        for pass in &["constant_fold", "strength_reduce", "dce", "cse", "licm"] {
            let est = m.query(
                &p,
                &f,
                &CostQuery::PassBenefit {
                    function_name: "demo",
                    pass_name: pass,
                },
            );
            let v = est.value().expect("trained should produce estimate");
            assert!(
                (0.0..=0.5).contains(&v),
                "trained pass {} predicted out-of-range benefit {}",
                pass,
                v
            );
        }
    }

    #[test]
    fn trained_predictions_are_deterministic() {
        let p = build_simple_program();
        let f = extract(&p);
        let m = LinearCostModel::trained();
        let q = CostQuery::PassBenefit {
            function_name: "demo",
            pass_name: "licm",
        };
        let first = m.query(&p, &f, &q);
        for _ in 0..100 {
            let again = m.query(&p, &f, &q);
            assert_eq!(again, first);
        }
    }

    #[test]
    fn trained_and_default_differ_on_at_least_one_query() {
        // The two models share the SAME architecture (predict_pass_gain
        // logic) but DIFFERENT coefficients. They should produce
        // different numeric predictions for at least one query — otherwise
        // the "trained" constructor isn't actually distinct from default.
        let p = build_simple_program();
        let f = extract(&p);
        let default = LinearCostModel::new();
        let trained = LinearCostModel::trained();
        let mut any_diff = false;
        for pass in &["constant_fold", "strength_reduce", "dce", "cse", "licm"] {
            let q = CostQuery::PassBenefit {
                function_name: "demo",
                pass_name: pass,
            };
            let d = default.query(&p, &f, &q).value().unwrap();
            let t = trained.query(&p, &f, &q).value().unwrap();
            if (d - t).abs() > 1e-9 {
                any_diff = true;
            }
        }
        assert!(any_diff, "trained model should differ from default on at least one pass");
    }

    #[test]
    fn trained_confidence_reflects_data_quality() {
        // Trained confidence values come from per-pass RMSE. They should
        // all be in [0.1, 0.95] per the trained_confidence function in
        // the training binary's emit_rust_source.
        let p = build_simple_program();
        let f = extract(&p);
        let m = LinearCostModel::trained();
        for pass in &["constant_fold", "strength_reduce", "dce", "cse", "licm"] {
            let est = m.query(
                &p,
                &f,
                &CostQuery::PassBenefit {
                    function_name: "demo",
                    pass_name: pass,
                },
            );
            let c = est.confidence().expect("trained reports confidence");
            assert!(
                (0.1..=0.95).contains(&c),
                "trained pass {} confidence {} out of [0.1, 0.95]",
                pass,
                c
            );
        }
    }
}
