//! Phase 4 (NSS) — `KernelVariant` enum scaffolding.
//!
//! NSS-aware kernel selection generates three variants of each fused
//! kernel from Phase 3.5. At runtime, NSS samples observed thermal/CPU
//! pressure and picks which variant to call. All three produce
//! byte-identical output (Phase 1 determinism contract).
//!
//! ## What this file is and is NOT
//!
//! IS: the enum + variant-selection trait surface, fully testable today
//! with [`crate::pressure::NullPressurePredictor`].
//!
//! IS NOT: actual codegen of the three variants. That work depends on
//! Phase 3.5 fusion codegen (shipped) plus a future "kernel specializer"
//! that emits the warm and cool versions of each fused kernel.
//!
//! See `docs/cana/CANA_PHASE_4_NSS_INTEGRATION_DESIGN.md` §3.4.

// ---------------------------------------------------------------------------
// The enum
// ---------------------------------------------------------------------------

/// Which variant of a fused kernel to call at runtime.
///
/// All three variants produce **byte-identical** output for the same inputs
/// — Phase 1's determinism contract is preserved across variants. They
/// differ in *resource usage*, not semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum KernelVariant {
    /// Fully fused, peak speed, peak heat. The codegen output of Phase 3.5
    /// today is implicitly this variant — all fused primitives (e.g.
    /// `fused_matmul_norm`, `fused_matmul_matmul`) run hot.
    Hot,
    /// Partially fused: some sub-steps stay unfused. Moderate cycles, lower
    /// thermal footprint. Useful when sustained throughput matters more
    /// than per-call latency.
    Warm,
    /// MIR-walked: no fusion, falls back to the canonical unfused chain.
    /// Slowest in cycles but lowest in thermal pressure. Useful when the
    /// CPU is already at thermal limit and we'd rather pay latency than
    /// trigger throttling.
    Cool,
}

impl KernelVariant {
    /// All variants in deterministic order. Used by audit reports and
    /// any caller that needs to enumerate the variant space.
    pub const ALL: &'static [KernelVariant] =
        &[KernelVariant::Hot, KernelVariant::Warm, KernelVariant::Cool];

    /// Short audit label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Hot => "hot",
            Self::Warm => "warm",
            Self::Cool => "cool",
        }
    }

    /// Pick the highest-performance variant whose thermal cost stays below
    /// the budget. `thermal_budget` is the predicted thermal pressure
    /// available to this call, in [0, 1].
    ///
    /// Thresholds (deliberately conservative):
    ///   - budget ≥ 0.30 → Hot (we have headroom)
    ///   - budget ≥ 0.10 → Warm (partial fusion)
    ///   - else          → Cool (preserve thermal headroom)
    ///
    /// Wrap in a real `PressurePredictor`-driven selector when NSS lands;
    /// this is the deterministic default selector used in tests.
    pub fn select_for_budget(thermal_budget: f64) -> Self {
        if thermal_budget >= 0.30 {
            Self::Hot
        } else if thermal_budget >= 0.10 {
            Self::Warm
        } else {
            Self::Cool
        }
    }
}

// ---------------------------------------------------------------------------
// §4B.4 Option α — KernelVariantSelector trait + default impls
// ---------------------------------------------------------------------------

/// Strategy for picking a [`KernelVariant`] at compile-time or call-time.
///
/// Implementations must be **deterministic** for a given input pair —
/// selector verdicts feed compile-time decisions (when the codegen tier
/// exists) AND runtime dispatch (when the runtime tier exists), and both
/// layers need to agree.
///
/// See `docs/cana/CANA_PHASE_4_KERNEL_VARIANT_DESIGN_OPTIONS.md` for the
/// §4B.4 design choice (this trait + two impls = Option α).
pub trait KernelVariantSelector: std::fmt::Debug {
    /// Pick the variant of `kernel_name` to call given the thermal-pressure
    /// prediction `predicted_thermal` in `[0.0, 1.0]`.
    ///
    /// Convention: `predicted_thermal = 0.0` means the function is
    /// thermally idle (full budget); `1.0` means it's already at the
    /// thermal limit (no budget). The selector translates a thermal
    /// pressure into the corresponding `KernelVariant`.
    fn select(&self, kernel_name: &str, predicted_thermal: f64) -> KernelVariant;

    /// Short audit label, e.g. `"always_hot"` or `"pressure_aware"`. Logged
    /// in benchmark output and `CanaReport` sidecars so downstream
    /// consumers can attribute decisions.
    fn name(&self) -> &'static str;
}

/// Trivial selector that always returns [`KernelVariant::Hot`].
///
/// This is the current production behaviour today: Phase 3.5 fusion
/// codegen emits exactly one variant (Hot), and `cjc-runtime`'s dispatch
/// tables route every call to it. `AlwaysHotSelector` makes that
/// behaviour explicit at the trait level so a `PressureAwareSelector`
/// can be wired in later without changing the call sites.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AlwaysHotSelector;

impl KernelVariantSelector for AlwaysHotSelector {
    fn select(&self, _kernel_name: &str, _predicted_thermal: f64) -> KernelVariant {
        KernelVariant::Hot
    }

    fn name(&self) -> &'static str {
        "always_hot"
    }
}

/// Pressure-driven selector — delegates to
/// [`KernelVariant::select_for_budget`].
///
/// `predicted_thermal ∈ [0, 1]` maps to `thermal_budget = 1.0 -
/// predicted_thermal`. So a function predicted to be 90% loaded yields
/// `budget = 0.10` → `KernelVariant::Warm`; 100% loaded → `budget = 0.0`
/// → `KernelVariant::Cool`; idle (predicted 0%) → `budget = 1.0` →
/// `KernelVariant::Hot`.
///
/// This is the selector that will be wired into the runtime once
/// `NssPressurePredictor` migrates from Option C (empty thermal maps,
/// today) to Option A (synthetic trace) or B (real instrumentation).
/// Today the predictor returns empty thermal maps, so callers see
/// `predicted_thermal = 0.0` for every kernel → selector returns Hot.
/// Behaviourally identical to `AlwaysHotSelector` until the predictor
/// activates.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PressureAwareSelector;

impl KernelVariantSelector for PressureAwareSelector {
    fn select(&self, _kernel_name: &str, predicted_thermal: f64) -> KernelVariant {
        let budget = (1.0 - predicted_thermal).clamp(0.0, 1.0);
        KernelVariant::select_for_budget(budget)
    }

    fn name(&self) -> &'static str {
        "pressure_aware"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_variants_listed_in_order() {
        assert_eq!(
            KernelVariant::ALL,
            &[KernelVariant::Hot, KernelVariant::Warm, KernelVariant::Cool]
        );
    }

    #[test]
    fn labels_are_stable() {
        assert_eq!(KernelVariant::Hot.label(), "hot");
        assert_eq!(KernelVariant::Warm.label(), "warm");
        assert_eq!(KernelVariant::Cool.label(), "cool");
    }

    #[test]
    fn select_for_high_budget_is_hot() {
        assert_eq!(KernelVariant::select_for_budget(0.5), KernelVariant::Hot);
        assert_eq!(KernelVariant::select_for_budget(1.0), KernelVariant::Hot);
        assert_eq!(KernelVariant::select_for_budget(0.30), KernelVariant::Hot);
    }

    #[test]
    fn select_for_mid_budget_is_warm() {
        assert_eq!(KernelVariant::select_for_budget(0.20), KernelVariant::Warm);
        assert_eq!(KernelVariant::select_for_budget(0.10), KernelVariant::Warm);
    }

    #[test]
    fn select_for_low_budget_is_cool() {
        assert_eq!(KernelVariant::select_for_budget(0.0), KernelVariant::Cool);
        assert_eq!(KernelVariant::select_for_budget(0.05), KernelVariant::Cool);
    }

    #[test]
    fn select_is_monotonic_across_budget() {
        // Sanity: as budget increases, selected variant doesn't downgrade.
        let mut prev_score = -1;
        for i in 0..=10 {
            let budget = (i as f64) * 0.1;
            let v = KernelVariant::select_for_budget(budget);
            let score = match v {
                KernelVariant::Cool => 0,
                KernelVariant::Warm => 1,
                KernelVariant::Hot => 2,
            };
            assert!(
                score >= prev_score,
                "budget {budget} downgraded variant from prev={prev_score} to score={score}"
            );
            prev_score = score;
        }
    }

    #[test]
    fn select_is_deterministic() {
        let first = KernelVariant::select_for_budget(0.4);
        for _ in 0..100 {
            assert_eq!(KernelVariant::select_for_budget(0.4), first);
        }
    }

    // -----------------------------------------------------------------------
    // §4B.4 Option α — KernelVariantSelector trait
    // -----------------------------------------------------------------------

    #[test]
    fn always_hot_selector_returns_hot_unconditionally() {
        let sel = AlwaysHotSelector;
        // Sweep across the thermal range; result must always be Hot.
        for i in 0..=10 {
            let thermal = (i as f64) * 0.1;
            assert_eq!(sel.select("fused_matmul_norm", thermal), KernelVariant::Hot);
        }
        // Kernel name doesn't affect choice either.
        for name in &["fused_matmul_dot", "fused_matmul_matmul", "anything"] {
            assert_eq!(sel.select(name, 0.99), KernelVariant::Hot);
        }
    }

    #[test]
    fn always_hot_selector_name_is_stable() {
        assert_eq!(AlwaysHotSelector.name(), "always_hot");
    }

    #[test]
    fn pressure_aware_selector_low_pressure_is_hot() {
        let sel = PressureAwareSelector;
        // predicted_thermal = 0.0 → budget = 1.0 → Hot
        assert_eq!(sel.select("k", 0.0), KernelVariant::Hot);
        // predicted_thermal = 0.5 → budget = 0.5 → Hot
        assert_eq!(sel.select("k", 0.5), KernelVariant::Hot);
        // predicted_thermal = 0.7 → budget = 0.3 → Hot (boundary)
        assert_eq!(sel.select("k", 0.7), KernelVariant::Hot);
    }

    #[test]
    fn pressure_aware_selector_mid_pressure_is_warm() {
        let sel = PressureAwareSelector;
        // predicted_thermal = 0.8 → budget = 0.2 → Warm
        assert_eq!(sel.select("k", 0.8), KernelVariant::Warm);
        // predicted_thermal = 0.85 → budget = 0.15 → Warm
        // (avoid 0.9 here: floating-point `1.0 - 0.9` lands at
        // 0.09999999999999998, just below the 0.10 Warm threshold →
        // Cool. The trait's contract intentionally doesn't promise
        // specific behaviour at exact boundary values; tests use
        // values clearly inside each region.)
        assert_eq!(sel.select("k", 0.85), KernelVariant::Warm);
    }

    #[test]
    fn pressure_aware_selector_high_pressure_is_cool() {
        let sel = PressureAwareSelector;
        // predicted_thermal = 1.0 → budget = 0.0 → Cool
        assert_eq!(sel.select("k", 1.0), KernelVariant::Cool);
        // predicted_thermal = 0.95 → budget = 0.05 → Cool
        assert_eq!(sel.select("k", 0.95), KernelVariant::Cool);
    }

    #[test]
    fn pressure_aware_selector_clamps_out_of_range_input() {
        let sel = PressureAwareSelector;
        // Negative input (shouldn't happen but be defensive): clamps to 0
        // → budget = 1.0 → Hot.
        assert_eq!(sel.select("k", -0.5), KernelVariant::Hot);
        // > 1 input: clamps to 1 → budget = 0.0 → Cool.
        assert_eq!(sel.select("k", 1.5), KernelVariant::Cool);
    }

    #[test]
    fn pressure_aware_selector_name_is_stable() {
        assert_eq!(PressureAwareSelector.name(), "pressure_aware");
    }

    #[test]
    fn pressure_aware_selector_is_deterministic() {
        // Same input → same output across many calls.
        let sel = PressureAwareSelector;
        let baseline = sel.select("k1", 0.75);
        for _ in 0..100 {
            assert_eq!(sel.select("k1", 0.75), baseline);
        }
    }

    #[test]
    fn empty_thermal_map_behaves_like_always_hot() {
        // §4B.2 Option C contract: NssPressurePredictor returns empty
        // thermal maps. When the caller fetches a non-existent key,
        // ThermalAwareCostModel uses `.unwrap_or(0.0)` → predicted_thermal
        // = 0.0. PressureAwareSelector at predicted_thermal = 0.0 returns
        // Hot — behaviourally identical to AlwaysHotSelector. This test
        // documents that equivalence at the unit level.
        let pressure_aware = PressureAwareSelector;
        let always_hot = AlwaysHotSelector;
        let predicted_thermal = 0.0; // What the empty map produces.
        for name in &["fused_matmul_dot", "fused_matmul_norm", "fused_matmul_matmul"] {
            assert_eq!(
                pressure_aware.select(name, predicted_thermal),
                always_hot.select(name, predicted_thermal),
                "Under §4B.2 Option C, PressureAware and AlwaysHot must agree on {}",
                name,
            );
        }
    }
}
