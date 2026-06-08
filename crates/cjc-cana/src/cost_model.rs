//! Cost-model trait surface + Phase-1 `NullCostModel`.
//!
//! The cost model answers questions like:
//!
//! > *"How much wall-clock will running pass `X` on function `F` save vs.
//! > skipping it?"*
//!
//! Phase 1 ships the **trait** (so Phase 2 has a stable seam to plug into)
//! and a single implementation — [`NullCostModel`] — that returns
//! [`CostEstimate::Unknown`] for every query. This lets the rest of the
//! pipeline operate end-to-end today without any actual prediction.
//!
//! Phase 2 will ship a tiny deterministic linear regression over the
//! [`CanaFeatures`] struct as the first non-null implementation.

use cjc_mir::MirProgram;

use crate::features::CanaFeatures;

// ---------------------------------------------------------------------------
// CostEstimate — the three-valued return type
// ---------------------------------------------------------------------------

/// A cost-model prediction.
///
/// Three-valued so callers always distinguish "I don't know" from
/// "I predict zero" — the difference matters for any advisor's decision
/// logic. Phase 1's `NullCostModel` returns `Unknown` for every query;
/// future models return `Estimated`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CostEstimate {
    /// Model has no opinion. Callers MUST treat this as "do nothing" — never
    /// fall back to assumed zero / assumed worst case.
    Unknown,
    /// Predicted value in unspecified units (typically normalized [0, 1] or
    /// nanoseconds for runtime queries). The unit is documented on the
    /// `CostModel` method.
    ///
    /// `confidence` is a hint in [0.0, 1.0]; 0.0 means "barely better than
    /// guessing", 1.0 means "high confidence". Phase-2 models will populate
    /// this from training data variance.
    Estimated { value: f64, confidence: f64 },
}

impl CostEstimate {
    /// Convenience predicate.
    pub fn is_unknown(&self) -> bool {
        matches!(self, CostEstimate::Unknown)
    }

    /// Extract the value, returning `None` for `Unknown`.
    pub fn value(&self) -> Option<f64> {
        match self {
            CostEstimate::Estimated { value, .. } => Some(*value),
            CostEstimate::Unknown => None,
        }
    }

    /// Extract the confidence, returning `None` for `Unknown`.
    pub fn confidence(&self) -> Option<f64> {
        match self {
            CostEstimate::Estimated { confidence, .. } => Some(*confidence),
            CostEstimate::Unknown => None,
        }
    }
}

// ---------------------------------------------------------------------------
// CostQuery — what callers ask about
// ---------------------------------------------------------------------------

/// A single cost-model query.
///
/// Kept small on purpose — the trait is mostly forward-compatible by adding
/// variants here, not by adding methods to `CostModel`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CostQuery<'a> {
    /// Predict the runtime cost (normalized) of running `pass_name` on
    /// `function_name`. Returned `value` is in [0, 1]: 0 = effectively free,
    /// 1 = the slowest pass in the pipeline on the largest function.
    PassRuntime {
        function_name: &'a str,
        pass_name: &'a str,
    },
    /// Predict the runtime *benefit* (savings) of running `pass_name` on
    /// `function_name`. Returned `value` is in [0, 1]: 0 = no improvement,
    /// 1 = halves runtime. Negative would be modeled as `Estimated { value:
    /// negative }` but Phase 1 doesn't go there.
    PassBenefit {
        function_name: &'a str,
        pass_name: &'a str,
    },
    /// Predict peak memory use of `function_name` after running the default
    /// pipeline. Returned `value` is in normalized bytes; the unit
    /// interpretation is up to the caller.
    PeakMemory { function_name: &'a str },
}

// ---------------------------------------------------------------------------
// Trait surface
// ---------------------------------------------------------------------------

/// A deterministic predictor over MIR features.
///
/// Implementations MUST be:
/// - Deterministic: same `(program, features, query)` → same `CostEstimate`
/// - Pure: no RNG that isn't seed-threaded, no time, no IO
/// - Total: never panic, return `Unknown` rather than guessing under
///   distribution shift
pub trait CostModel {
    /// Answer a cost query.
    ///
    /// The default implementation returns `Unknown` for every query — this
    /// makes the trait safe to call from anywhere even when no real model is
    /// plugged in.
    fn query<'a>(
        &self,
        program: &MirProgram,
        features: &CanaFeatures,
        query: &CostQuery<'a>,
    ) -> CostEstimate {
        let _ = (program, features, query);
        CostEstimate::Unknown
    }

    /// Human-readable model name, used in audit traces.
    fn name(&self) -> &'static str;

    /// Monotonic model version, used in content-addressing of any prediction
    /// the model produces. Phase-1 `NullCostModel` is `0`; future models
    /// MUST bump this every time their predictions could change.
    fn version(&self) -> u32 {
        0
    }
}

// ---------------------------------------------------------------------------
// NullCostModel — Phase-1 default
// ---------------------------------------------------------------------------

/// A cost model that returns [`CostEstimate::Unknown`] for every query.
///
/// Phase 1's only implementation. Lets the rest of the pipeline operate
/// end-to-end without any actual prediction — advisors that wrap the cost
/// model receive `Unknown` and naturally short-circuit.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullCostModel;

impl NullCostModel {
    pub fn new() -> Self {
        Self
    }
}

impl CostModel for NullCostModel {
    fn name(&self) -> &'static str {
        "null"
    }
    // version() defaults to 0 — appropriate; this model never changes.
    // query() defaults to Unknown — exactly what we want.
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::extract;
    use cjc_mir::{MirBody, MirFnId, MirFunction, MirProgram};

    fn empty_program() -> MirProgram {
        MirProgram {
            functions: vec![MirFunction {
                id: MirFnId(0),
                name: "__main".to_string(),
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
    fn null_cost_model_returns_unknown_for_every_query() {
        let program = empty_program();
        let features = extract(&program);
        let model = NullCostModel::new();

        let queries = [
            CostQuery::PassRuntime {
                function_name: "__main",
                pass_name: "dce",
            },
            CostQuery::PassBenefit {
                function_name: "__main",
                pass_name: "licm",
            },
            CostQuery::PeakMemory {
                function_name: "__main",
            },
        ];

        for q in &queries {
            let est = model.query(&program, &features, q);
            assert!(est.is_unknown(), "got {:?} for {:?}", est, q);
            assert!(est.value().is_none());
            assert!(est.confidence().is_none());
        }
    }

    #[test]
    fn null_cost_model_name_and_version() {
        let m = NullCostModel::new();
        assert_eq!(m.name(), "null");
        assert_eq!(m.version(), 0);
    }

    #[test]
    fn estimated_extracts_value_and_confidence() {
        let est = CostEstimate::Estimated {
            value: 0.42,
            confidence: 0.9,
        };
        assert_eq!(est.value(), Some(0.42));
        assert_eq!(est.confidence(), Some(0.9));
        assert!(!est.is_unknown());
    }
}
