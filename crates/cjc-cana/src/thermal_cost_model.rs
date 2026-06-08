//! Phase 4 (NSS) — `ThermalAwareCostModel` scaffolding.
//!
//! Composes a base [`CostModel`] with a [`PressurePredictor`] so the cost
//! model's pass-benefit estimates can be down-weighted on thermally-
//! aggressive functions.
//!
//! ## Why this layer exists
//!
//! The base cost model (`LinearCostModel`) predicts pass benefits as a
//! function of static MIR features — block count, reduction count, etc.
//! It has no concept of *thermal* cost.
//!
//! Phase 4 closes that gap. When a [`PressurePredictor`] (typically NSS-
//! backed) flags a function as already hot, this wrapper halves the
//! predicted benefit of passes that increase thermal load — loop unrolling,
//! vectorization, monomorphization. The ranker then naturally deprioritizes
//! those passes for that function.
//!
//! ## What this file is and is NOT
//!
//! IS: the wrapper, fully implemented against the [`CostModel`] trait,
//! testable against [`NullPressurePredictor`] today.
//!
//! IS NOT: an NSS bridge. The actual NSS predictor lives in
//! `crates/cjc-cana-nss` (when NSS lands).
//!
//! See `docs/cana/CANA_PHASE_4_NSS_INTEGRATION_DESIGN.md` §3.3 for the
//! design rationale.

use cjc_mir::MirProgram;

use crate::cost_model::{CostEstimate, CostModel, CostQuery};
use crate::features::CanaFeatures;
use crate::pressure::PressurePredictor;

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/// Default thermal threshold above which passes get penalized. A predicted
/// thermal pressure ≥ this value causes the wrapper to halve the benefit of
/// thermally-aggressive passes.
///
/// 0.80 ≈ "80% of the way to thermal trip" — leaves room for cooler kernels
/// to still get aggressive optimization. Tunable per use case.
pub const DEFAULT_THERMAL_THRESHOLD: f64 = 0.80;

/// Factor applied to predicted benefit when a thermally-aggressive pass
/// targets an above-threshold function. 0.5 = halve the benefit.
pub const THERMAL_PENALTY_FACTOR: f64 = 0.5;

/// Pass names considered "thermally aggressive" — likely to raise sustained
/// CPU/cache pressure when applied. The cost model penalizes these on
/// already-hot functions.
///
/// Order: kept stable for determinism (no hashing or randomized iteration
/// in callers that consume this).
pub const THERMALLY_AGGRESSIVE_PASSES: &[&str] = &[
    "loop_unroll",
    "vectorize",
    "specialize",
    "monomorphize",
];

// ---------------------------------------------------------------------------
// The wrapper
// ---------------------------------------------------------------------------

/// Cost model that wraps a base estimator and applies a thermal-pressure
/// penalty using a [`PressurePredictor`].
///
/// Construction:
///
/// ```ignore
/// let base = LinearCostModel::new();
/// let pp = NullPressurePredictor;  // or NssPressurePredictor::from_seed(42)
/// let model = ThermalAwareCostModel::new(base, pp);
/// ```
///
/// The wrapper is generic over both the base model and the predictor so
/// either can be swapped without re-implementing the trait.
pub struct ThermalAwareCostModel<M: CostModel, P: PressurePredictor> {
    pub base_model: M,
    pub pressure: P,
    pub thermal_threshold: f64,
}

impl<M: CostModel, P: PressurePredictor> ThermalAwareCostModel<M, P> {
    pub fn new(base_model: M, pressure: P) -> Self {
        Self {
            base_model,
            pressure,
            thermal_threshold: DEFAULT_THERMAL_THRESHOLD,
        }
    }

    /// Override the thermal threshold. Use a lower value to be more
    /// conservative (penalize more passes), higher to be more aggressive.
    pub fn with_thermal_threshold(mut self, t: f64) -> Self {
        self.thermal_threshold = t;
        self
    }
}

impl<M: CostModel, P: PressurePredictor> std::fmt::Debug for ThermalAwareCostModel<M, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThermalAwareCostModel")
            .field("base_model", &"<dyn CostModel>")
            .field("pressure", &self.pressure)
            .field("thermal_threshold", &self.thermal_threshold)
            .finish()
    }
}

impl<M: CostModel, P: PressurePredictor> CostModel for ThermalAwareCostModel<M, P> {
    fn query(
        &self,
        program: &MirProgram,
        features: &CanaFeatures,
        query: &CostQuery,
    ) -> CostEstimate {
        let base = self.base_model.query(program, features, query);

        match query {
            CostQuery::PassBenefit { function_name, pass_name } => {
                // Only adjust for thermally-aggressive passes.
                if !THERMALLY_AGGRESSIVE_PASSES.contains(pass_name) {
                    return base;
                }
                let thermal_map = self.pressure.predict_thermal(program, features);
                let predicted_thermal =
                    thermal_map.get(*function_name).copied().unwrap_or(0.0);
                if predicted_thermal < self.thermal_threshold {
                    return base;
                }
                // Penalize. Preserve confidence; only scale the predicted value.
                match base {
                    CostEstimate::Estimated { value, confidence } => {
                        CostEstimate::Estimated {
                            value: value * THERMAL_PENALTY_FACTOR,
                            confidence,
                        }
                    }
                    CostEstimate::Unknown => CostEstimate::Unknown,
                }
            }
            // Other query kinds (PassRuntime, ProgramFeatures, etc.) are
            // not currently adjusted for thermal pressure — only the
            // benefit estimate. PassRuntime is a compile-cost, not a
            // runtime-pressure signal, so the wrapper passes it through.
            _ => base,
        }
    }

    fn name(&self) -> &'static str {
        "thermal_aware"
    }

    fn version(&self) -> u32 {
        1
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
    use crate::pressure::{NullPressurePredictor, PressurePredictor};
    use cjc_mir::{MirBody, MirFnId, MirFunction, MirProgram};

    use std::collections::BTreeMap;

    fn empty_program() -> MirProgram {
        MirProgram {
            functions: vec![MirFunction {
                id: MirFnId(0),
                name: "main".to_string(),
                type_params: vec![],
                params: vec![],
                return_type: None,
                body: MirBody { stmts: vec![], result: None },
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

    /// Constant cost-model that always returns `Estimated { value: 0.5, confidence: 0.7 }`.
    /// Lets us isolate the thermal-penalty math from the linear model's logic.
    #[derive(Debug)]
    struct ConstantCostModel;
    impl CostModel for ConstantCostModel {
        fn query(&self, _p: &MirProgram, _f: &CanaFeatures, _q: &CostQuery) -> CostEstimate {
            CostEstimate::Estimated { value: 0.5, confidence: 0.7 }
        }
        fn name(&self) -> &'static str { "constant" }
        fn version(&self) -> u32 { 1 }
    }

    /// A predictor that returns a configurable thermal value for "main".
    #[derive(Debug)]
    struct FixedThermalPredictor {
        thermal: f64,
    }
    impl PressurePredictor for FixedThermalPredictor {
        fn predict_thermal(
            &self,
            program: &MirProgram,
            _f: &CanaFeatures,
        ) -> BTreeMap<String, f64> {
            program
                .functions
                .iter()
                .map(|f| (f.name.clone(), self.thermal))
                .collect()
        }
        fn predict_memory_peak(&self, p: &MirProgram, _f: &CanaFeatures) -> BTreeMap<String, f64> {
            p.functions.iter().map(|f| (f.name.clone(), 0.0)).collect()
        }
        fn predict_cpu_saturation(&self, p: &MirProgram, _f: &CanaFeatures) -> BTreeMap<String, f64> {
            p.functions.iter().map(|f| (f.name.clone(), 0.0)).collect()
        }
        fn identify_structural_hot_kernels(&self, _p: &MirProgram, _f: &CanaFeatures) -> Vec<String> {
            Vec::new()
        }
        fn name(&self) -> &'static str { "fixed_thermal" }
        fn version(&self) -> u32 { 1 }
    }

    #[test]
    fn wrapper_passes_through_when_thermal_is_low() {
        let p = empty_program();
        let f = extract(&p);
        let model = ThermalAwareCostModel::new(ConstantCostModel, FixedThermalPredictor {
            thermal: 0.0,
        });
        let q = CostQuery::PassBenefit {
            function_name: "main",
            pass_name: "loop_unroll",
        };
        match model.query(&p, &f, &q) {
            CostEstimate::Estimated { value, confidence } => {
                assert_eq!(value, 0.5, "thermal=0 → no penalty, full benefit");
                assert_eq!(confidence, 0.7);
            }
            other => panic!("expected Estimated, got {other:?}"),
        }
    }

    #[test]
    fn wrapper_halves_benefit_when_thermal_exceeds_threshold() {
        let p = empty_program();
        let f = extract(&p);
        let model = ThermalAwareCostModel::new(ConstantCostModel, FixedThermalPredictor {
            thermal: 0.9,  // above 0.80 default threshold
        });
        let q = CostQuery::PassBenefit {
            function_name: "main",
            pass_name: "loop_unroll",
        };
        match model.query(&p, &f, &q) {
            CostEstimate::Estimated { value, confidence } => {
                assert!((value - 0.25).abs() < 1e-12, "thermal=0.9 → halve to 0.25");
                assert_eq!(confidence, 0.7, "confidence is preserved");
            }
            other => panic!("expected Estimated, got {other:?}"),
        }
    }

    #[test]
    fn wrapper_does_not_penalize_non_thermally_aggressive_passes() {
        let p = empty_program();
        let f = extract(&p);
        let model = ThermalAwareCostModel::new(ConstantCostModel, FixedThermalPredictor {
            thermal: 0.95,
        });
        // dce is not in THERMALLY_AGGRESSIVE_PASSES.
        let q = CostQuery::PassBenefit {
            function_name: "main",
            pass_name: "dce",
        };
        match model.query(&p, &f, &q) {
            CostEstimate::Estimated { value, .. } => {
                assert_eq!(value, 0.5, "dce should not get a thermal penalty");
            }
            _ => panic!(),
        }
    }

    #[test]
    fn wrapper_passes_pass_runtime_queries_through_unchanged() {
        // Thermal pressure affects benefit predictions but NOT compile-cost
        // (PassRuntime) — a hot kernel doesn't change how long the
        // compiler takes to optimize it.
        let p = empty_program();
        let f = extract(&p);
        let model = ThermalAwareCostModel::new(ConstantCostModel, FixedThermalPredictor {
            thermal: 0.99,
        });
        let q = CostQuery::PassRuntime {
            function_name: "main",
            pass_name: "loop_unroll",
        };
        match model.query(&p, &f, &q) {
            CostEstimate::Estimated { value, .. } => {
                assert_eq!(value, 0.5, "PassRuntime queries pass through");
            }
            _ => panic!(),
        }
    }

    #[test]
    fn wrapper_with_null_predictor_acts_like_base_model() {
        // NullPressurePredictor returns 0.0 everywhere, so the wrapper
        // should never apply a penalty — output is identical to the base.
        let p = empty_program();
        let f = extract(&p);
        let model = ThermalAwareCostModel::new(NullCostModel, NullPressurePredictor);
        let q = CostQuery::PassBenefit {
            function_name: "main",
            pass_name: "loop_unroll",
        };
        let wrapped = model.query(&p, &f, &q);
        let base = NullCostModel.query(&p, &f, &q);
        assert_eq!(format!("{wrapped:?}"), format!("{base:?}"));
    }

    #[test]
    fn wrapper_threshold_is_tunable() {
        let p = empty_program();
        let f = extract(&p);
        // Use a low threshold (0.5) so a thermal=0.6 reading triggers
        // penalty, even though it wouldn't under the default 0.80.
        let model = ThermalAwareCostModel::new(ConstantCostModel, FixedThermalPredictor {
            thermal: 0.6,
        })
        .with_thermal_threshold(0.5);
        let q = CostQuery::PassBenefit {
            function_name: "main",
            pass_name: "loop_unroll",
        };
        match model.query(&p, &f, &q) {
            CostEstimate::Estimated { value, .. } => {
                assert!((value - 0.25).abs() < 1e-12, "threshold=0.5 → penalty applied");
            }
            _ => panic!(),
        }
    }

    #[test]
    fn wrapper_identifies_itself_for_audit() {
        let model = ThermalAwareCostModel::new(NullCostModel, NullPressurePredictor);
        assert_eq!(model.name(), "thermal_aware");
        assert_eq!(model.version(), 1);
    }

    #[test]
    fn wrapper_output_is_deterministic() {
        let p = empty_program();
        let f = extract(&p);
        let model = ThermalAwareCostModel::new(ConstantCostModel, FixedThermalPredictor {
            thermal: 0.9,
        });
        let q = CostQuery::PassBenefit {
            function_name: "main",
            pass_name: "loop_unroll",
        };
        let first = format!("{:?}", model.query(&p, &f, &q));
        for _ in 0..50 {
            let again = format!("{:?}", model.query(&p, &f, &q));
            assert_eq!(first, again);
        }
    }
}
