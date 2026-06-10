//! PINN v1 — `PinnPhysicalCostModel`, the physics-informed cost-model
//! wrapper.
//!
//! Composes a base [`CostModel`] + a [`PressurePredictor`] + the
//! deterministic v1 physical-cost calculation from
//! [`crate::physical_cost`] into a [`CostModel`] implementation that:
//!
//! 1. Delegates every query to the base model first.
//! 2. For `PassBenefit` queries only, derives an integer
//!    [`PhysicalCostQuery`] from the function's MIR features, runs
//!    [`predict_physical`], and:
//!    - **hard-rejects** any strategy whose estimate violates
//!      [`PhysicalConstraints`], by returning
//!      `Estimated { value: 0.0, confidence: 1.0 }` — a confident
//!      zero-benefit prediction the ranker drops below its skip
//!      threshold (`BelowSkipThreshold`, recorded in the audit trail).
//!      NOT `CostEstimate::Unknown`: this codebase's `PassRanker`
//!      treats `Unknown` as *keep conservatively*
//!      (`UnknownButKeptConservatively`), which would make hard limits
//!      a no-op. The Phase-A handoff's "Unknown = do nothing"
//!      assumption was wrong about the ranker's actual semantics.
//!    - otherwise **soft-blends**: subtracts a margin-scaled physical
//!      penalty from the base benefit and multiplies confidence by the
//!      estimate's confidence.
//! 3. Passes `PassRuntime` / `PeakMemory` queries through unchanged —
//!    physical pressure is a runtime phenomenon, not a compile cost.
//!
//! ## Stack placement — do NOT double-wrap thermal
//!
//! ```text
//! recommended:  LinearCostModel → PinnPhysicalCostModel
//! NOT:          LinearCostModel → ThermalAwareCostModel → PinnPhysicalCostModel
//! ```
//!
//! [`crate::thermal_cost_model::ThermalAwareCostModel`] and this
//! wrapper both penalize thermally-aggressive passes on `PassBenefit`
//! queries; stacking them applies thermal influence twice. This
//! wrapper *supersedes* the single-axis thermal wrapper with a
//! multi-axis physical model (thermal + memory + bandwidth + locality
//! + energy). The `stacking_double_penalizes_thermal` test locks this
//! documentation claim.
//!
//! ## NSS blend
//!
//! The `P: PressurePredictor` parameter folds NSS-predicted thermal in
//! conservatively: `thermal = max(closed_form, nss_predicted)`. Taking
//! the max never *hides* heat the formula sees, and never invents
//! precision the predictor doesn't have.
//!
//! ## Authority
//!
//! Advisory only. This model can withhold a recommendation
//! (`Unknown`); it cannot approve anything —
//! [`crate::legality::LegalityGate`] and the MIR verifier retain final
//! authority (determinism invariant #12).
//!
//! ## v2 forward-compatibility
//!
//! v2 swaps [`PhysicalCoefficients`] for a trained MLP behind this
//! same struct + trait surface (handoff §4.3). `model_id` /
//! `model_version` flow into `CanaReport` hashing so two compilations
//! with different coefficient sets produce different report hashes
//! (determinism invariant #8).

use cjc_mir::MirProgram;

use crate::cost_model::{CostEstimate, CostModel, CostQuery};
use crate::features::CanaFeatures;
use crate::physical_cost::{
    build_physical_query, predict_physical, PhysicalCoefficients, PhysicalConstraints,
    PhysicalCostEstimate,
};
use crate::pressure::PressurePredictor;

// ---------------------------------------------------------------------------
// The wrapper
// ---------------------------------------------------------------------------

/// Physics-informed cost-model wrapper. See module docs.
///
/// Construction:
///
/// ```ignore
/// let model = PinnPhysicalCostModel::new(LinearCostModel::new(), NullPressurePredictor);
/// // or with custom physics:
/// let model = PinnPhysicalCostModel::new(base, predictor)
///     .with_coefficients(my_coeffs)
///     .with_constraints(my_limits);
/// ```
pub struct PinnPhysicalCostModel<M: CostModel, P: PressurePredictor> {
    pub base_model: M,
    pub pressure_predictor: P,
    pub physical_coeffs: PhysicalCoefficients,
    pub thresholds: PhysicalConstraints,
}

/// Stable model identifier; flows into report hashes via
/// [`CostModel::name`]. v2's trained model will use a different id.
pub const PINN_V1_MODEL_ID: &str = "pinn_coeffs_v1";

/// Monotonic model version; bump whenever the prediction formulas or
/// default coefficients change.
pub const PINN_V1_MODEL_VERSION: u32 = 1;

impl<M: CostModel, P: PressurePredictor> PinnPhysicalCostModel<M, P> {
    pub fn new(base_model: M, pressure_predictor: P) -> Self {
        Self {
            base_model,
            pressure_predictor,
            physical_coeffs: PhysicalCoefficients::default(),
            thresholds: PhysicalConstraints::default(),
        }
    }

    /// Override the physical coefficients.
    pub fn with_coefficients(mut self, coeffs: PhysicalCoefficients) -> Self {
        self.physical_coeffs = coeffs;
        self
    }

    /// Override the constraint thresholds.
    pub fn with_constraints(mut self, constraints: PhysicalConstraints) -> Self {
        self.thresholds = constraints;
        self
    }

    /// Run the physical prediction for one `(function, pass)` pair,
    /// blending in the NSS thermal signal. Returns `None` when the
    /// function has no feature record or the coefficients are invalid
    /// — callers map `None` to "leave the base estimate alone" (the
    /// physical layer has no opinion), NOT to rejection.
    fn physical_estimate(
        &self,
        program: &MirProgram,
        features: &CanaFeatures,
        function_name: &str,
        pass_name: &str,
    ) -> Option<PhysicalCostEstimate> {
        let fn_features = features.per_fn.get(function_name)?;
        let query = build_physical_query(function_name, pass_name, fn_features);
        let mut est = predict_physical(&query, &self.physical_coeffs)?;

        // Conservative NSS blend: never report less heat than the
        // predictor sees. f64::max is exact (no rounding), so this
        // preserves bit-determinism of whichever source wins.
        let nss_thermal = self
            .pressure_predictor
            .predict_thermal(program, features)
            .get(function_name)
            .copied()
            .unwrap_or(0.0);
        if nss_thermal.is_finite() {
            est.thermal_pressure = est.thermal_pressure.max(nss_thermal.clamp(0.0, 1.0));
        }
        Some(est)
    }
}

impl<M: CostModel, P: PressurePredictor> std::fmt::Debug for PinnPhysicalCostModel<M, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnPhysicalCostModel")
            .field("base_model", &self.base_model.name())
            .field("pressure_predictor", &self.pressure_predictor)
            .field("physical_coeffs", &self.physical_coeffs)
            .field("thresholds", &self.thresholds)
            .finish()
    }
}

impl<M: CostModel, P: PressurePredictor> CostModel for PinnPhysicalCostModel<M, P> {
    fn query(
        &self,
        program: &MirProgram,
        features: &CanaFeatures,
        query: &CostQuery,
    ) -> CostEstimate {
        let base = self.base_model.query(program, features, query);

        let (function_name, pass_name) = match query {
            CostQuery::PassBenefit {
                function_name,
                pass_name,
            } => (*function_name, *pass_name),
            // PassRuntime is compile cost; PeakMemory is the base
            // model's own estimate. Physical pressure adjusts neither.
            _ => return base,
        };

        // Never invent an estimate the base didn't make — Unknown
        // stays Unknown regardless of physics.
        let CostEstimate::Estimated { value, confidence } = base else {
            return CostEstimate::Unknown;
        };

        let Some(est) = self.physical_estimate(program, features, function_name, pass_name) else {
            // No feature record / invalid coefficients: the physical
            // layer abstains; the base estimate stands.
            return base;
        };

        // Hard rejection: constraint violation withholds the
        // recommendation by predicting zero benefit with full
        // confidence — the ranker drops it below the skip threshold
        // and records the drop. (`Unknown` would be KEPT
        // conservatively by the ranker; see module docs.)
        if !est.is_valid() || self.thresholds.rejects(&est) {
            return CostEstimate::Estimated {
                value: 0.0,
                confidence: 1.0,
            };
        }

        // Soft blend: margin-scaled penalty, floored at zero benefit.
        // No FMA: each product binds to a named intermediate.
        let pressure_partial = est.thermal_pressure + est.memory_pressure;
        let pressure_sum = pressure_partial + est.bandwidth_pressure;
        let physical_penalty = self.thresholds.prefer_cooler_plan_margin * pressure_sum;
        let blended_value = (value - physical_penalty).max(0.0);
        let blended_confidence_raw = confidence * est.confidence;
        let blended_confidence = blended_confidence_raw.clamp(0.0, 1.0);

        CostEstimate::Estimated {
            value: blended_value,
            confidence: blended_confidence,
        }
    }

    fn name(&self) -> &'static str {
        PINN_V1_MODEL_ID
    }

    fn version(&self) -> u32 {
        PINN_V1_MODEL_VERSION
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost_model::NullCostModel;
    use crate::features::extract;
    use crate::pressure::NullPressurePredictor;
    use crate::thermal_cost_model::ThermalAwareCostModel;
    use cjc_mir::{MirBody, MirExpr, MirExprKind, MirFnId, MirFunction, MirProgram, MirStmt};

    use std::collections::BTreeMap;

    /// A function whose body is `n` integer-literal expression
    /// statements — gives nonzero expr_count so the physical query is
    /// non-trivial.
    fn fn_with_exprs(name: &str, n: usize) -> MirFunction {
        let stmts = (0..n)
            .map(|i| {
                MirStmt::Expr(MirExpr {
                    kind: MirExprKind::IntLit(i as i64),
                })
            })
            .collect();
        MirFunction {
            id: MirFnId(0),
            name: name.to_string(),
            type_params: vec![],
            params: vec![],
            return_type: None,
            body: MirBody {
                stmts,
                result: None,
            },
            is_nogc: false,
            cfg_body: None,
            decorators: vec![],
            vis: cjc_ast::Visibility::Public,
            local_count: 0,
        }
    }

    fn program_with(name: &str, exprs: usize) -> MirProgram {
        MirProgram {
            functions: vec![fn_with_exprs(name, exprs)],
            struct_defs: vec![],
            enum_defs: vec![],
            entry: MirFnId(0),
        }
    }

    /// Constant benefit estimate, isolating the blend math.
    #[derive(Debug)]
    struct ConstantCostModel {
        value: f64,
        confidence: f64,
    }
    impl CostModel for ConstantCostModel {
        fn query(&self, _p: &MirProgram, _f: &CanaFeatures, _q: &CostQuery) -> CostEstimate {
            CostEstimate::Estimated {
                value: self.value,
                confidence: self.confidence,
            }
        }
        fn name(&self) -> &'static str {
            "constant"
        }
        fn version(&self) -> u32 {
            1
        }
    }

    /// Fixed thermal predictor for the NSS-blend tests.
    #[derive(Debug)]
    struct FixedThermalPredictor {
        thermal: f64,
    }
    impl PressurePredictor for FixedThermalPredictor {
        fn predict_thermal(&self, p: &MirProgram, _f: &CanaFeatures) -> BTreeMap<String, f64> {
            p.functions
                .iter()
                .map(|f| (f.name.clone(), self.thermal))
                .collect()
        }
        fn predict_memory_peak(&self, p: &MirProgram, _f: &CanaFeatures) -> BTreeMap<String, f64> {
            p.functions.iter().map(|f| (f.name.clone(), 0.0)).collect()
        }
        fn predict_cpu_saturation(
            &self,
            p: &MirProgram,
            _f: &CanaFeatures,
        ) -> BTreeMap<String, f64> {
            p.functions.iter().map(|f| (f.name.clone(), 0.0)).collect()
        }
        fn identify_structural_hot_kernels(
            &self,
            _p: &MirProgram,
            _f: &CanaFeatures,
        ) -> Vec<String> {
            Vec::new()
        }
        fn name(&self) -> &'static str {
            "fixed_thermal"
        }
        fn version(&self) -> u32 {
            1
        }
    }

    fn benefit_query<'a>(function_name: &'a str, pass_name: &'a str) -> CostQuery<'a> {
        CostQuery::PassBenefit {
            function_name,
            pass_name,
        }
    }

    #[test]
    fn unknown_base_stays_unknown() {
        let p = program_with("main", 10);
        let f = extract(&p);
        let model = PinnPhysicalCostModel::new(NullCostModel, NullPressurePredictor);
        let est = model.query(&p, &f, &benefit_query("main", "loop_unroll"));
        assert!(est.is_unknown(), "physics never invents estimates");
    }

    #[test]
    fn small_function_gets_negligible_penalty() {
        let p = program_with("main", 5);
        let f = extract(&p);
        let model = PinnPhysicalCostModel::new(
            ConstantCostModel {
                value: 0.5,
                confidence: 0.8,
            },
            NullPressurePredictor,
        );
        match model.query(&p, &f, &benefit_query("main", "dce")) {
            CostEstimate::Estimated { value, confidence } => {
                // 5 exprs vs 1e7 scale: pressures ~0; penalty ~0.
                assert!((value - 0.5).abs() < 1e-3, "value {value}");
                assert!((confidence - 0.8).abs() < 1e-3, "confidence {confidence}");
            }
            other => panic!("expected Estimated, got {other:?}"),
        }
    }

    #[test]
    fn hard_thermal_limit_rejects_via_nss_blend() {
        let p = program_with("main", 10);
        let f = extract(&p);
        // NSS says 0.99 thermal — above the default 0.95 hard limit.
        let model = PinnPhysicalCostModel::new(
            ConstantCostModel {
                value: 0.5,
                confidence: 0.8,
            },
            FixedThermalPredictor { thermal: 0.99 },
        );
        match model.query(&p, &f, &benefit_query("main", "loop_unroll")) {
            CostEstimate::Estimated { value, confidence } => {
                assert_eq!(value, 0.0, "rejection = confident zero benefit");
                assert_eq!(confidence, 1.0);
            }
            other => panic!(
                "rejection must be Estimated zero (Unknown would be \
                 kept conservatively by the ranker), got {other:?}"
            ),
        }
    }

    #[test]
    fn hotter_strategy_is_demoted_below_equivalent_cooler_one() {
        let p = program_with("main", 10);
        let f = extract(&p);
        let cool = PinnPhysicalCostModel::new(
            ConstantCostModel {
                value: 0.5,
                confidence: 0.8,
            },
            FixedThermalPredictor { thermal: 0.1 },
        );
        let hot = PinnPhysicalCostModel::new(
            ConstantCostModel {
                value: 0.5,
                confidence: 0.8,
            },
            FixedThermalPredictor { thermal: 0.9 },
        );
        let q = benefit_query("main", "loop_unroll");
        let (cool_v, hot_v) = match (cool.query(&p, &f, &q), hot.query(&p, &f, &q)) {
            (
                CostEstimate::Estimated { value: cv, .. },
                CostEstimate::Estimated { value: hv, .. },
            ) => (cv, hv),
            other => panic!("expected two estimates, got {other:?}"),
        };
        assert!(
            hot_v < cool_v,
            "equivalent strategies: hotter ({hot_v}) must rank below cooler ({cool_v})"
        );
    }

    #[test]
    fn pass_runtime_and_peak_memory_pass_through() {
        let p = program_with("main", 10);
        let f = extract(&p);
        let model = PinnPhysicalCostModel::new(
            ConstantCostModel {
                value: 0.5,
                confidence: 0.8,
            },
            FixedThermalPredictor { thermal: 0.99 },
        );
        for q in [
            CostQuery::PassRuntime {
                function_name: "main",
                pass_name: "loop_unroll",
            },
            CostQuery::PeakMemory {
                function_name: "main",
            },
        ] {
            match model.query(&p, &f, &q) {
                CostEstimate::Estimated { value, .. } => {
                    assert_eq!(value, 0.5, "{q:?} must pass through unchanged");
                }
                other => panic!("expected Estimated for {q:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn missing_function_features_leaves_base_estimate_alone() {
        let p = program_with("main", 10);
        let f = extract(&p);
        let model = PinnPhysicalCostModel::new(
            ConstantCostModel {
                value: 0.5,
                confidence: 0.8,
            },
            NullPressurePredictor,
        );
        // Query a function that doesn't exist in the program.
        match model.query(&p, &f, &benefit_query("ghost", "dce")) {
            CostEstimate::Estimated { value, confidence } => {
                assert_eq!(value, 0.5);
                assert_eq!(confidence, 0.8);
            }
            other => panic!("abstention must preserve the base, got {other:?}"),
        }
    }

    #[test]
    fn invalid_coefficients_abstain_rather_than_reject() {
        let p = program_with("main", 10);
        let f = extract(&p);
        let model = PinnPhysicalCostModel::new(
            ConstantCostModel {
                value: 0.5,
                confidence: 0.8,
            },
            NullPressurePredictor,
        )
        .with_coefficients(PhysicalCoefficients {
            flops_norm_scale: f64::NAN,
            ..PhysicalCoefficients::default()
        });
        match model.query(&p, &f, &benefit_query("main", "dce")) {
            CostEstimate::Estimated { value, .. } => assert_eq!(value, 0.5),
            other => panic!("invalid physics must abstain, got {other:?}"),
        }
    }

    #[test]
    fn tight_energy_constraint_rejects() {
        let p = program_with("main", 200);
        let f = extract(&p);
        let model = PinnPhysicalCostModel::new(
            ConstantCostModel {
                value: 0.5,
                confidence: 0.8,
            },
            NullPressurePredictor,
        )
        .with_constraints(PhysicalConstraints {
            max_energy_estimate: Some(0.0),
            ..PhysicalConstraints::default()
        });
        // 200 exprs → nonzero energy proxy > 0.0 limit.
        match model.query(&p, &f, &benefit_query("main", "loop_unroll")) {
            CostEstimate::Estimated { value, .. } => assert_eq!(value, 0.0),
            other => panic!("expected zero-benefit rejection, got {other:?}"),
        }
    }

    #[test]
    fn stacking_double_penalizes_thermal() {
        // Locks the module-docs claim: Pinn over ThermalAware applies
        // thermal influence twice; the recommended stack is Pinn over
        // the base directly.
        let p = program_with("main", 10);
        let f = extract(&p);
        let q = benefit_query("main", "loop_unroll");
        // 0.9 thermal: above ThermalAware's 0.80 penalty threshold,
        // below Pinn's 0.95 hard limit.
        let recommended = PinnPhysicalCostModel::new(
            ConstantCostModel {
                value: 0.5,
                confidence: 0.8,
            },
            FixedThermalPredictor { thermal: 0.9 },
        );
        let double_stacked = PinnPhysicalCostModel::new(
            ThermalAwareCostModel::new(
                ConstantCostModel {
                    value: 0.5,
                    confidence: 0.8,
                },
                FixedThermalPredictor { thermal: 0.9 },
            ),
            FixedThermalPredictor { thermal: 0.9 },
        );
        let (rec_v, dbl_v) = match (
            recommended.query(&p, &f, &q),
            double_stacked.query(&p, &f, &q),
        ) {
            (
                CostEstimate::Estimated { value: rv, .. },
                CostEstimate::Estimated { value: dv, .. },
            ) => (rv, dv),
            other => panic!("expected two estimates, got {other:?}"),
        };
        assert!(
            dbl_v < rec_v,
            "double-stacked ({dbl_v}) penalizes more than recommended ({rec_v}) — \
             which is why the stack docs say don't do it"
        );
    }

    #[test]
    fn model_identifies_itself_for_report_hashing() {
        let model = PinnPhysicalCostModel::new(NullCostModel, NullPressurePredictor);
        assert_eq!(model.name(), "pinn_coeffs_v1");
        assert_eq!(model.version(), 1);
    }

    #[test]
    fn output_is_deterministic() {
        let p = program_with("main", 50);
        let f = extract(&p);
        let model = PinnPhysicalCostModel::new(
            ConstantCostModel {
                value: 0.5,
                confidence: 0.8,
            },
            FixedThermalPredictor { thermal: 0.4 },
        );
        let q = benefit_query("main", "loop_unroll");
        let first = format!("{:?}", model.query(&p, &f, &q));
        for _ in 0..50 {
            assert_eq!(first, format!("{:?}", model.query(&p, &f, &q)));
        }
    }
}
