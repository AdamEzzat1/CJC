//! PINN v1 — deterministic physical-cost primitives.
//!
//! ## What this file is
//!
//! The four data types of the physics-informed cost layer, plus the
//! closed-form v1 prediction:
//!
//! - [`PhysicalCostQuery`] — integer workload estimates for one
//!   `(function, pass)` candidate, derived from [`FnFeatures`] by
//!   [`build_physical_query`].
//! - [`PhysicalCostEstimate`] — the predicted pressures, every float
//!   finite and clamped to `[0, 1]` (except `energy_estimate`, an
//!   unbounded additive Joule-proxy).
//! - [`PhysicalCoefficients`] — the hand-tuned v1 knobs. v2 replaces
//!   this table with a trained MLP behind the same call surface.
//! - [`PhysicalConstraints`] — hard `max_*` rejection limits + the
//!   soft `prefer_cooler_plan_margin` preference.
//!
//! ## What this file is NOT
//!
//! - Not a neural net. v1 is a deterministic coefficient model — zero
//!   training data, zero new dependencies, deterministic by
//!   construction (no shadow mode required).
//! - Not a pressure *measurement*. Inputs are MIR-shape proxies (see
//!   [`crate::memory_proxy`]); Phase A6's profile DB exists to replace
//!   the proxy scales with measured values offline.
//! - Not authoritative. Consumers ([`crate::pinn_cost_model`]) can only
//!   *withhold* recommendations on constraint violation — the
//!   [`crate::legality::LegalityGate`] retains final authority.
//!
//! ## Determinism contract
//!
//! `predict_physical` is a pure function of `(query, coefficients)`:
//! integer inputs reduce through saturating arithmetic, convert to
//! `f64` once, and combine through a fixed sequence of named
//! intermediates (no FMA contraction, no accumulation loops — so no
//! Kahan needed). Same inputs → bit-identical outputs across runs,
//! OS, and CPU.

use crate::features::FnFeatures;

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/// Cache-size proxy for the locality term: 2 MiB ≈ a per-core L2.
/// Working sets near/above this are flagged as locality risks.
pub const CACHE_PROXY_BYTES: u64 = 1 << 21;

/// Per-loop-depth amplification step for workload estimates.
/// `loop_amp = 1 + LOOP_AMP_STEP * min(max_loop_depth, LOOP_AMP_CAP)`.
pub const LOOP_AMP_STEP: u64 = 7;

/// Loop depths beyond this contribute no further amplification —
/// deeper static nesting tells us little extra about dynamic cost.
pub const LOOP_AMP_CAP: u32 = 4;

// Proxy byte-scales. Documented guesses, not measurements; the names
// say what they price, the values keep the derived pressures in a
// useful dynamic range for MIR-shaped inputs (expr counts 1e1..1e5).
const BYTES_PER_TENSOR_OP_READ: u64 = 256;
const BYTES_PER_EXPR_READ: u64 = 8;
const BYTES_PER_COW_WRITE: u64 = 256;
const BYTES_PER_ALLOC_SITE: u64 = 64;
const WORKING_SET_PER_TENSOR_OP: u64 = 1024;

// ---------------------------------------------------------------------------
// PhysicalCostQuery
// ---------------------------------------------------------------------------

/// Input query for a physical-cost prediction.
///
/// All counters are integer estimates derived from MIR shape + a
/// candidate pass; no wall-clock, no OS sensors (determinism
/// invariant #7). Construct via [`build_physical_query`] in production;
/// literal construction is fine in tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalCostQuery<'a> {
    pub function_name: &'a str,
    /// The candidate strategy — for v1 this is the pass name from
    /// `CostQuery::PassBenefit`.
    pub strategy_id: &'a str,
    pub flops_estimate: u64,
    pub bytes_read_estimate: u64,
    pub bytes_written_estimate: u64,
    pub allocation_bytes_estimate: u64,
    pub working_set_bytes_estimate: u64,
    /// CJC executes single-threaded deterministically today; kept in
    /// the schema so v2 training rows are forward-compatible.
    pub thread_count: u32,
    pub batch_size: u32,
}

// ---------------------------------------------------------------------------
// PhysicalCostEstimate
// ---------------------------------------------------------------------------

/// Output of a physical-cost prediction.
///
/// Every pressure is finite and clamped to `[0, 1]`; `energy_estimate`
/// is finite but unbounded (it's an additive proxy, not a pressure).
/// `confidence` weights downstream blending and is also in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicalCostEstimate {
    pub thermal_pressure: f64,
    pub memory_pressure: f64,
    pub bandwidth_pressure: f64,
    pub energy_estimate: f64,
    pub locality_risk: f64,
    pub confidence: f64,
}

impl PhysicalCostEstimate {
    /// `true` iff every field is finite, pressures/confidence are in
    /// `[0, 1]`, and the energy proxy is non-negative.
    pub fn is_valid(&self) -> bool {
        let unit_bounded = [
            self.thermal_pressure,
            self.memory_pressure,
            self.bandwidth_pressure,
            self.locality_risk,
            self.confidence,
        ];
        unit_bounded
            .iter()
            .all(|x| x.is_finite() && (0.0..=1.0).contains(x))
            && self.energy_estimate.is_finite()
            && self.energy_estimate >= 0.0
    }
}

// ---------------------------------------------------------------------------
// PhysicalCoefficients
// ---------------------------------------------------------------------------

/// Coefficients of the deterministic physical-cost model.
///
/// Hand-tuned for v1; trained offline for v2. The handoff's
/// per-unit coefficients map onto normalization scales here:
/// `heat_per_flop` → [`Self::flops_norm_scale`], `bandwidth_per_byte`
/// → [`Self::bytes_norm_scale`], `alloc_churn_per_byte` →
/// [`Self::alloc_norm_scale`] (each `norm(x, s) = x / (x + s)` term is
/// a smooth, monotone, overflow-free stand-in for `coeff * x` with
/// built-in saturation at 1.0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicalCoefficients {
    /// FLOPs at which heat accumulation reaches 0.5. Lower = hotter
    /// model.
    pub flops_norm_scale: f64,
    /// Cooling / dissipation fraction subtracted from heat
    /// accumulation. `[0, 1)`.
    pub cooling_rate: f64,
    /// Weight of bytes-moved (relative to allocation pressure) in the
    /// memory-pressure term.
    pub bytes_per_flop_scale: f64,
    /// Bytes moved at which bandwidth pressure reaches 0.5.
    pub bytes_norm_scale: f64,
    /// Allocated bytes at which allocation churn reaches 0.5.
    pub alloc_norm_scale: f64,
    /// Thermal amplification per extra thread beyond the first.
    pub thread_amplification: f64,
    /// Thermal amplification per extra batch element beyond the first.
    pub batch_amplification: f64,
    /// Weight of the working-set / cache-proxy ratio in locality risk.
    pub locality_weight: f64,
}

impl Default for PhysicalCoefficients {
    fn default() -> Self {
        Self {
            flops_norm_scale: 1e7,
            cooling_rate: 0.05,
            bytes_per_flop_scale: 0.1,
            bytes_norm_scale: 1e8,
            alloc_norm_scale: 1e6,
            thread_amplification: 0.1,
            batch_amplification: 0.05,
            locality_weight: 0.3,
        }
    }
}

impl PhysicalCoefficients {
    /// `true` iff every knob is finite, non-negative, scales are
    /// strictly positive, and `cooling_rate < 1` (a cooling rate ≥ 1
    /// would zero or negate all heat, making the thermal term
    /// vacuous).
    pub fn is_valid(&self) -> bool {
        let non_negative = [
            self.cooling_rate,
            self.bytes_per_flop_scale,
            self.thread_amplification,
            self.batch_amplification,
            self.locality_weight,
        ];
        let positive_scales = [
            self.flops_norm_scale,
            self.bytes_norm_scale,
            self.alloc_norm_scale,
        ];
        non_negative.iter().all(|x| x.is_finite() && *x >= 0.0)
            && positive_scales.iter().all(|x| x.is_finite() && *x > 0.0)
            && self.cooling_rate < 1.0
    }
}

// ---------------------------------------------------------------------------
// PhysicalConstraints
// ---------------------------------------------------------------------------

/// Hard upper bounds and soft preferences.
///
/// Consumers reject (return `CostEstimate::Unknown` for) strategies
/// whose estimate exceeds any hard `max_*`; the soft margin scales the
/// blend penalty so v1 bias-orders rather than dominates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicalConstraints {
    /// Hard limit; an estimate above this is rejected outright.
    pub max_thermal_pressure: f64,
    pub max_memory_pressure: f64,
    pub max_bandwidth_pressure: f64,
    /// `None` = unbounded.
    pub max_energy_estimate: Option<f64>,
    /// Soft preference: scales the physical penalty subtracted from
    /// base benefit estimates. Default 0.1 = "prefer plans with 10%
    /// headroom" — bias-ordering, not domination.
    pub prefer_cooler_plan_margin: f64,
}

impl Default for PhysicalConstraints {
    fn default() -> Self {
        Self {
            max_thermal_pressure: 0.95,
            max_memory_pressure: 0.95,
            max_bandwidth_pressure: 0.95,
            max_energy_estimate: None,
            prefer_cooler_plan_margin: 0.1,
        }
    }
}

impl PhysicalConstraints {
    /// `true` iff every limit is finite and in a sane range.
    pub fn is_valid(&self) -> bool {
        let limits = [
            self.max_thermal_pressure,
            self.max_memory_pressure,
            self.max_bandwidth_pressure,
            self.prefer_cooler_plan_margin,
        ];
        limits.iter().all(|x| x.is_finite() && *x >= 0.0)
            && match self.max_energy_estimate {
                Some(e) => e.is_finite() && e >= 0.0,
                None => true,
            }
    }

    /// `true` iff `est` violates any hard limit.
    pub fn rejects(&self, est: &PhysicalCostEstimate) -> bool {
        if est.thermal_pressure > self.max_thermal_pressure {
            return true;
        }
        if est.memory_pressure > self.max_memory_pressure {
            return true;
        }
        if est.bandwidth_pressure > self.max_bandwidth_pressure {
            return true;
        }
        if let Some(max_e) = self.max_energy_estimate {
            if est.energy_estimate > max_e {
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// build_physical_query — FnFeatures → integer workload estimates
// ---------------------------------------------------------------------------

/// Per-pass amplification of the workload estimates.
///
/// Mirrors the spirit of the energy ranker's `pass_code_size_factor`:
/// a pass that duplicates loop bodies multiplies the FLOPs/bytes the
/// post-pass function will execute. Identity for passes with no
/// physical footprint (dce, cse, const_fold shrink or keep work — we
/// conservatively leave their estimates unamplified rather than
/// crediting reductions v1 can't verify).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PassPhysicalAmp {
    flops: u64,
    bytes: u64,
}

fn pass_physical_amp(pass_name: &str) -> PassPhysicalAmp {
    match pass_name {
        // Unrolling duplicates body work per iteration window.
        "loop_unroll" => PassPhysicalAmp { flops: 2, bytes: 2 },
        // Vectorization widens memory traffic per retired op.
        "vectorize" => PassPhysicalAmp { flops: 1, bytes: 2 },
        // Specialization / monomorphization duplicate code, not
        // dynamic work — code size is priced by the energy ranker's
        // code_size factor, not here.
        _ => PassPhysicalAmp { flops: 1, bytes: 1 },
    }
}

/// Derive an integer workload query from one function's MIR-shape
/// features + a candidate pass.
///
/// Pure saturating-integer arithmetic; deterministic by construction.
pub fn build_physical_query<'a>(
    function_name: &'a str,
    pass_name: &'a str,
    features: &FnFeatures,
) -> PhysicalCostQuery<'a> {
    let amp = pass_physical_amp(pass_name);

    let depth = features.cfg.max_loop_depth.min(LOOP_AMP_CAP) as u64;
    let loop_amp = 1u64.saturating_add(LOOP_AMP_STEP.saturating_mul(depth));

    let expr = features.memory.expr_count as u64;
    let tensor = features.memory.tensor_heavy_ops as u64;
    let cow = features.memory.cow_write_sites as u64;
    let alloc = features.memory.alloc_sites as u64;

    let flops_base = expr.saturating_mul(loop_amp);
    let flops_estimate = flops_base.saturating_mul(amp.flops);

    let read_base = tensor
        .saturating_mul(BYTES_PER_TENSOR_OP_READ)
        .saturating_add(expr.saturating_mul(BYTES_PER_EXPR_READ));
    let bytes_read_estimate = read_base.saturating_mul(loop_amp).saturating_mul(amp.bytes);

    let write_base = cow
        .saturating_mul(BYTES_PER_COW_WRITE)
        .saturating_add(alloc.saturating_mul(BYTES_PER_ALLOC_SITE));
    let bytes_written_estimate = write_base
        .saturating_mul(loop_amp)
        .saturating_mul(amp.bytes);

    let allocation_bytes_estimate = alloc.saturating_mul(BYTES_PER_ALLOC_SITE);

    let working_set_bytes_estimate = alloc
        .saturating_mul(BYTES_PER_ALLOC_SITE)
        .saturating_add(tensor.saturating_mul(WORKING_SET_PER_TENSOR_OP));

    PhysicalCostQuery {
        function_name,
        strategy_id: pass_name,
        flops_estimate,
        bytes_read_estimate,
        bytes_written_estimate,
        allocation_bytes_estimate,
        working_set_bytes_estimate,
        thread_count: 1,
        batch_size: 1,
    }
}

// ---------------------------------------------------------------------------
// predict_physical — the closed-form v1 model
// ---------------------------------------------------------------------------

/// Clamp to `[0, 1]`, mapping NaN to 0 (NaN never survives into an
/// estimate — defensive only; valid coefficients can't produce it).
fn clamp01(x: f64) -> f64 {
    if x.is_nan() {
        return 0.0;
    }
    x.clamp(0.0, 1.0)
}

/// Rational normalization `x / (x + scale)`: smooth, monotone in `x`,
/// always in `[0, 1)` for `x ≥ 0, scale > 0`, no overflow after the
/// one-time `u64 → f64` conversion.
fn norm(x: u64, scale: f64) -> f64 {
    let xf = x as f64;
    xf / (xf + scale)
}

/// Run the deterministic v1 physical-cost prediction.
///
/// Pure function of `(query, coeffs)`. If `coeffs` is invalid
/// (non-finite / non-positive scales), returns `None` — callers map
/// that to `CostEstimate::Unknown`, never a panic.
///
/// No FMA: every product is bound to a named intermediate before any
/// addition (determinism invariant #3).
pub fn predict_physical(
    query: &PhysicalCostQuery<'_>,
    coeffs: &PhysicalCoefficients,
) -> Option<PhysicalCostEstimate> {
    if !coeffs.is_valid() {
        return None;
    }

    let extra_threads = query.thread_count.saturating_sub(1) as f64;
    let extra_batch = query.batch_size.saturating_sub(1) as f64;

    // Thermal: normalized FLOPs, amplified by parallelism, damped by
    // cooling.
    let heat_base = norm(query.flops_estimate, coeffs.flops_norm_scale);
    let thread_term = coeffs.thread_amplification * extra_threads;
    let batch_term = coeffs.batch_amplification * extra_batch;
    let thread_factor = 1.0 + thread_term;
    let batch_factor = 1.0 + batch_term;
    let heat_amplified = heat_base * thread_factor;
    let heat_accumulation = heat_amplified * batch_factor;
    let cooling_keep = 1.0 - coeffs.cooling_rate;
    let thermal_pressure = clamp01(heat_accumulation * cooling_keep);

    // Memory: allocation pressure + weighted traffic pressure.
    let bytes_moved = query
        .bytes_read_estimate
        .saturating_add(query.bytes_written_estimate);
    let alloc_pressure = norm(query.allocation_bytes_estimate, coeffs.alloc_norm_scale);
    let traffic_norm = norm(bytes_moved, coeffs.bytes_norm_scale);
    let traffic_term = coeffs.bytes_per_flop_scale * traffic_norm;
    let memory_pressure = clamp01(alloc_pressure + traffic_term);

    // Bandwidth: traffic alone.
    let bandwidth_pressure = clamp01(traffic_norm);

    // Allocation churn feeds energy + confidence.
    let alloc_churn = clamp01(alloc_pressure);

    // Energy: additive Joule-proxy (unbounded, non-negative).
    let energy_partial = heat_accumulation + bandwidth_pressure;
    let energy_estimate = energy_partial + alloc_churn;

    // Locality: working set vs cache proxy.
    let ws_ratio = norm(query.working_set_bytes_estimate, CACHE_PROXY_BYTES as f64);
    let locality_risk = clamp01(coeffs.locality_weight * ws_ratio);

    // Confidence: erodes with churn (proxy quality drops on
    // allocation-heavy code) and with parallelism (v1 has no
    // contention model).
    let churn_half = 0.5 * alloc_churn;
    let thread_penalty = coeffs.thread_amplification * extra_threads;
    let confidence_raw = 1.0 - churn_half - thread_penalty;
    let confidence = clamp01(confidence_raw);

    Some(PhysicalCostEstimate {
        thermal_pressure,
        memory_pressure,
        bandwidth_pressure,
        energy_estimate,
        locality_risk,
        confidence,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg_metrics::CfgMetrics;
    use crate::hash::CfgHash;
    use crate::memory_proxy::MemoryProxy;
    use crate::reduction_axes::ReductionAxes;

    fn zero_query() -> PhysicalCostQuery<'static> {
        PhysicalCostQuery {
            function_name: "f",
            strategy_id: "dce",
            flops_estimate: 0,
            bytes_read_estimate: 0,
            bytes_written_estimate: 0,
            allocation_bytes_estimate: 0,
            working_set_bytes_estimate: 0,
            thread_count: 1,
            batch_size: 1,
        }
    }

    fn synthetic_query() -> PhysicalCostQuery<'static> {
        PhysicalCostQuery {
            function_name: "hot_kernel",
            strategy_id: "loop_unroll",
            flops_estimate: 5_000_000,
            bytes_read_estimate: 40_000_000,
            bytes_written_estimate: 10_000_000,
            allocation_bytes_estimate: 500_000,
            working_set_bytes_estimate: 1_000_000,
            thread_count: 1,
            batch_size: 1,
        }
    }

    fn fn_features(
        expr_count: u32,
        tensor_heavy_ops: u32,
        cow_write_sites: u32,
        alloc_sites: u32,
        max_loop_depth: u32,
    ) -> FnFeatures {
        FnFeatures {
            cfg: CfgMetrics {
                block_count: 1,
                edge_count: 0,
                branch_count: 0,
                return_count: 1,
                unreachable_count: 0,
                goto_count: 0,
                max_branch_factor: 0,
                loop_count: if max_loop_depth > 0 { 1 } else { 0 },
                max_loop_depth,
                back_edge_count: if max_loop_depth > 0 { 1 } else { 0 },
                countable_loop_count: 0,
                cfg_hash: CfgHash(0),
            },
            memory: MemoryProxy {
                alloc_sites,
                cow_write_sites,
                tensor_heavy_ops,
                expr_count,
            },
            reductions: ReductionAxes::default(),
        }
    }

    #[test]
    fn zero_query_produces_zero_pressures_full_confidence() {
        let est = predict_physical(&zero_query(), &PhysicalCoefficients::default()).unwrap();
        assert_eq!(est.thermal_pressure, 0.0);
        assert_eq!(est.memory_pressure, 0.0);
        assert_eq!(est.bandwidth_pressure, 0.0);
        assert_eq!(est.energy_estimate, 0.0);
        assert_eq!(est.locality_risk, 0.0);
        assert_eq!(est.confidence, 1.0);
        assert!(est.is_valid());
    }

    #[test]
    fn default_coefficients_give_sensible_synthetic_outputs() {
        let est = predict_physical(&synthetic_query(), &PhysicalCoefficients::default()).unwrap();
        assert!(est.is_valid(), "estimate must validate: {est:?}");
        // 5M flops vs 1e7 scale → heat_base = 1/3; cooled → ~0.317.
        assert!(
            (est.thermal_pressure - (5.0 / 15.0) * 0.95).abs() < 1e-12,
            "thermal {}",
            est.thermal_pressure
        );
        // Nonzero but unsaturated everywhere.
        assert!(est.memory_pressure > 0.0 && est.memory_pressure < 1.0);
        assert!(est.bandwidth_pressure > 0.0 && est.bandwidth_pressure < 1.0);
        assert!(est.locality_risk > 0.0 && est.locality_risk < 1.0);
        assert!(
            est.confidence > 0.5,
            "single-thread low-churn stays confident"
        );
    }

    #[test]
    fn outputs_finite_and_clamped_over_coefficient_grid() {
        let scales = [1.0, 1e3, 1e7, 1e12];
        let rates = [0.0, 0.05, 0.5, 0.99];
        let weights = [0.0, 0.1, 1.0, 10.0];
        let queries = [zero_query(), synthetic_query(), {
            let mut q = synthetic_query();
            q.flops_estimate = u64::MAX;
            q.bytes_read_estimate = u64::MAX;
            q.bytes_written_estimate = u64::MAX;
            q.allocation_bytes_estimate = u64::MAX;
            q.working_set_bytes_estimate = u64::MAX;
            q.thread_count = u32::MAX;
            q.batch_size = u32::MAX;
            q
        }];
        for &s in &scales {
            for &r in &rates {
                for &w in &weights {
                    let coeffs = PhysicalCoefficients {
                        flops_norm_scale: s,
                        cooling_rate: r,
                        bytes_per_flop_scale: w,
                        bytes_norm_scale: s,
                        alloc_norm_scale: s,
                        thread_amplification: w,
                        batch_amplification: w,
                        locality_weight: w,
                    };
                    assert!(coeffs.is_valid());
                    for q in &queries {
                        let est = predict_physical(q, &coeffs).unwrap();
                        assert!(
                            est.is_valid(),
                            "invalid estimate for scale={s} rate={r} weight={w}: {est:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn invalid_coefficients_yield_none_not_panic() {
        let bad_cases = [
            PhysicalCoefficients {
                flops_norm_scale: 0.0, // scale must be > 0
                ..PhysicalCoefficients::default()
            },
            PhysicalCoefficients {
                flops_norm_scale: f64::NAN,
                ..PhysicalCoefficients::default()
            },
            PhysicalCoefficients {
                cooling_rate: 1.0, // must be < 1
                ..PhysicalCoefficients::default()
            },
            PhysicalCoefficients {
                cooling_rate: -0.1,
                ..PhysicalCoefficients::default()
            },
            PhysicalCoefficients {
                locality_weight: f64::INFINITY,
                ..PhysicalCoefficients::default()
            },
        ];
        for coeffs in &bad_cases {
            assert!(!coeffs.is_valid());
            assert!(predict_physical(&synthetic_query(), coeffs).is_none());
        }
    }

    #[test]
    fn more_flops_never_lowers_thermal() {
        let coeffs = PhysicalCoefficients::default();
        let mut last = -1.0;
        for flops in [0u64, 10, 1_000, 100_000, 10_000_000, 1_000_000_000] {
            let mut q = zero_query();
            q.flops_estimate = flops;
            let est = predict_physical(&q, &coeffs).unwrap();
            assert!(
                est.thermal_pressure >= last,
                "thermal must be monotone in flops"
            );
            last = est.thermal_pressure;
        }
    }

    #[test]
    fn saturated_query_stays_valid() {
        let mut q = zero_query();
        q.flops_estimate = u64::MAX;
        q.bytes_read_estimate = u64::MAX;
        q.bytes_written_estimate = u64::MAX; // read + write saturates
        q.allocation_bytes_estimate = u64::MAX;
        q.working_set_bytes_estimate = u64::MAX;
        let est = predict_physical(&q, &PhysicalCoefficients::default()).unwrap();
        assert!(est.is_valid(), "{est:?}");
        // norm() approaches but never reaches 1.0 pre-clamp; the clamp
        // guarantees the contract either way.
        assert!(est.thermal_pressure <= 1.0);
        assert!(est.bandwidth_pressure <= 1.0);
    }

    #[test]
    fn prediction_is_deterministic() {
        let q = synthetic_query();
        let coeffs = PhysicalCoefficients::default();
        let first = predict_physical(&q, &coeffs).unwrap();
        for _ in 0..50 {
            let again = predict_physical(&q, &coeffs).unwrap();
            assert_eq!(first, again);
        }
    }

    #[test]
    fn constraints_default_rejects_only_above_95() {
        let c = PhysicalConstraints::default();
        assert!(c.is_valid());
        let mut est = predict_physical(&zero_query(), &PhysicalCoefficients::default()).unwrap();
        assert!(!c.rejects(&est));
        est.thermal_pressure = 0.96;
        assert!(c.rejects(&est));
        est.thermal_pressure = 0.5;
        est.memory_pressure = 0.99;
        assert!(c.rejects(&est));
        est.memory_pressure = 0.5;
        est.bandwidth_pressure = 1.0;
        assert!(c.rejects(&est));
    }

    #[test]
    fn energy_constraint_applies_only_when_set() {
        let mut est =
            predict_physical(&synthetic_query(), &PhysicalCoefficients::default()).unwrap();
        est.energy_estimate = 1e9;
        let unbounded = PhysicalConstraints::default();
        assert!(!unbounded.rejects(&est));
        let bounded = PhysicalConstraints {
            max_energy_estimate: Some(10.0),
            ..PhysicalConstraints::default()
        };
        assert!(bounded.rejects(&est));
    }

    #[test]
    fn build_query_amplifies_by_loop_depth_and_pass() {
        let flat = fn_features(100, 4, 2, 3, 0);
        let nested = fn_features(100, 4, 2, 3, 2);

        let q_flat = build_physical_query("f", "dce", &flat);
        let q_nested = build_physical_query("f", "dce", &nested);
        assert_eq!(q_flat.flops_estimate, 100);
        assert_eq!(q_nested.flops_estimate, 100 * 15); // 1 + 7*2

        let q_unroll = build_physical_query("f", "loop_unroll", &nested);
        assert_eq!(q_unroll.flops_estimate, q_nested.flops_estimate * 2);
        assert_eq!(
            q_unroll.bytes_read_estimate,
            q_nested.bytes_read_estimate * 2
        );

        // Allocation + working set are not loop-amplified (a loop
        // reuses its buffers under COW; sites are static).
        assert_eq!(q_flat.allocation_bytes_estimate, 3 * 64);
        assert_eq!(q_nested.allocation_bytes_estimate, 3 * 64);
        assert_eq!(q_flat.working_set_bytes_estimate, 3 * 64 + 4 * 1024);
    }

    #[test]
    fn build_query_loop_depth_caps() {
        let deep = fn_features(10, 0, 0, 0, 9);
        let q = build_physical_query("f", "dce", &deep);
        // depth capped at 4 → amp = 1 + 7*4 = 29
        assert_eq!(q.flops_estimate, 10 * 29);
    }

    #[test]
    fn build_query_saturates_instead_of_overflowing() {
        let huge = fn_features(u32::MAX, u32::MAX, u32::MAX, u32::MAX, 4);
        let q = build_physical_query("f", "loop_unroll", &huge);
        // Must not panic; estimates land at or below u64::MAX.
        assert!(q.bytes_read_estimate <= u64::MAX);
        let est = predict_physical(&q, &PhysicalCoefficients::default()).unwrap();
        assert!(est.is_valid());
    }

    #[test]
    fn vectorize_widens_bytes_not_flops() {
        let f = fn_features(100, 4, 2, 3, 1);
        let base = build_physical_query("f", "dce", &f);
        let vec = build_physical_query("f", "vectorize", &f);
        assert_eq!(vec.flops_estimate, base.flops_estimate);
        assert_eq!(vec.bytes_read_estimate, base.bytes_read_estimate * 2);
        assert_eq!(vec.bytes_written_estimate, base.bytes_written_estimate * 2);
    }
}
