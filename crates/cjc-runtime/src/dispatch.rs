//! Hybrid Summation Dispatch — Routes reductions to the appropriate accumulator.
//!
//! # Dispatch Rules
//!
//! | Condition                          | Strategy              |
//! |------------------------------------|-----------------------|
//! | `ExecMode::Parallel`               | BinnedAccumulator     |
//! | `@nogc` context                    | BinnedAccumulator     |
//! | `ReproMode::Strict`                | BinnedAccumulator     |
//! | Reduction inside `LinalgOp`        | BinnedAccumulator     |
//! | Serial + `ReproMode::On`           | Kahan Summation       |
//! | Serial + no vectorization          | Kahan Summation       |
//! | Not forced strict                  | Kahan Summation       |
//!
//! The dispatch path is deterministic and unit-tested.

use crate::accumulator::{binned_sum_f64, BinnedAccumulatorF64};
use cjc_repro::kahan_sum_f64;

// ---------------------------------------------------------------------------
// Execution context for dispatch decisions
// ---------------------------------------------------------------------------

/// Execution mode for the current context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecMode {
    /// Serial, single-threaded execution.
    Serial,
    /// Parallel / multi-threaded execution.
    Parallel,
}

/// Reproducibility mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReproMode {
    /// Reproducibility disabled — fastest path.
    Off,
    /// Reproducibility enabled — Kahan for serial, Binned for parallel.
    On,
    /// Strict reproducibility — always Binned, regardless of exec mode.
    Strict,
}

/// Reduction context passed to the dispatch logic.
#[derive(Debug, Clone, Copy)]
pub struct ReductionContext {
    /// Current execution mode.
    pub exec_mode: ExecMode,
    /// Reproducibility mode.
    pub repro_mode: ReproMode,
    /// Whether we are inside a @nogc function.
    pub in_nogc: bool,
    /// Whether this is a linalg operation (matmul, etc.).
    pub is_linalg: bool,
}

impl ReductionContext {
    /// Default context: serial, repro on, not in nogc, not linalg.
    pub fn default_serial() -> Self {
        ReductionContext {
            exec_mode: ExecMode::Serial,
            repro_mode: ReproMode::On,
            in_nogc: false,
            is_linalg: false,
        }
    }

    /// Context for @nogc zones.
    pub fn nogc() -> Self {
        ReductionContext {
            exec_mode: ExecMode::Serial,
            repro_mode: ReproMode::Strict,
            in_nogc: true,
            is_linalg: false,
        }
    }

    /// Context for linalg operations.
    pub fn linalg() -> Self {
        ReductionContext {
            exec_mode: ExecMode::Serial,
            repro_mode: ReproMode::On,
            in_nogc: false,
            is_linalg: true,
        }
    }

    /// Context for strict parallel.
    pub fn strict_parallel() -> Self {
        ReductionContext {
            exec_mode: ExecMode::Parallel,
            repro_mode: ReproMode::Strict,
            in_nogc: false,
            is_linalg: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Strategy selection
// ---------------------------------------------------------------------------

/// Which summation strategy to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SumStrategy {
    /// Kahan compensated summation (serial, order-dependent).
    Kahan,
    /// Binned superaccumulator (order-invariant, deterministic).
    Binned,
}

/// Determine the appropriate summation strategy for the given context.
///
/// # Rules (in priority order)
///
/// 1. `ExecMode::Parallel` → Binned
/// 2. `@nogc` context → Binned
/// 3. `ReproMode::Strict` → Binned
/// 4. Linalg operation → Binned
/// 5. Otherwise → Kahan
pub fn select_strategy(ctx: &ReductionContext) -> SumStrategy {
    if ctx.exec_mode == ExecMode::Parallel {
        return SumStrategy::Binned;
    }
    if ctx.in_nogc {
        return SumStrategy::Binned;
    }
    if ctx.repro_mode == ReproMode::Strict {
        return SumStrategy::Binned;
    }
    if ctx.is_linalg {
        return SumStrategy::Binned;
    }
    SumStrategy::Kahan
}

// ---------------------------------------------------------------------------
// Dispatched summation functions
// ---------------------------------------------------------------------------

/// Sum f64 values using the strategy appropriate for the given context.
///
/// This is the primary entry point for all reductions in the CJC runtime.
#[inline]
pub fn dispatch_sum_f64(values: &[f64], ctx: &ReductionContext) -> f64 {
    match select_strategy(ctx) {
        SumStrategy::Kahan => kahan_sum_f64(values),
        SumStrategy::Binned => binned_sum_f64(values),
    }
}

/// Dot product of two equal-length f64 slices using dispatched summation.
///
/// Computes element-wise products, then sums with the selected strategy.
/// For Binned strategy, the products Vec is collected on the stack (via Vec)
/// before passing to the accumulator. This is acceptable because the Vec
/// is for the products array, not the accumulator itself.
#[inline]
pub fn dispatch_dot_f64(a: &[f64], b: &[f64], ctx: &ReductionContext) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    match select_strategy(ctx) {
        SumStrategy::Kahan => {
            let products: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect();
            kahan_sum_f64(&products)
        }
        SumStrategy::Binned => {
            let mut acc = BinnedAccumulatorF64::new();
            for (&x, &y) in a.iter().zip(b.iter()) {
                acc.add(x * y);
            }
            acc.finalize()
        }
    }
}

// ---------------------------------------------------------------------------
// Inline tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serial_on_uses_kahan() {
        let ctx = ReductionContext::default_serial();
        assert_eq!(select_strategy(&ctx), SumStrategy::Kahan);
    }

    #[test]
    fn test_parallel_uses_binned() {
        let ctx = ReductionContext {
            exec_mode: ExecMode::Parallel,
            repro_mode: ReproMode::On,
            in_nogc: false,
            is_linalg: false,
        };
        assert_eq!(select_strategy(&ctx), SumStrategy::Binned);
    }

    #[test]
    fn test_nogc_uses_binned() {
        let ctx = ReductionContext::nogc();
        assert_eq!(select_strategy(&ctx), SumStrategy::Binned);
    }

    #[test]
    fn test_strict_uses_binned() {
        let ctx = ReductionContext {
            exec_mode: ExecMode::Serial,
            repro_mode: ReproMode::Strict,
            in_nogc: false,
            is_linalg: false,
        };
        assert_eq!(select_strategy(&ctx), SumStrategy::Binned);
    }

    #[test]
    fn test_linalg_uses_binned() {
        let ctx = ReductionContext::linalg();
        assert_eq!(select_strategy(&ctx), SumStrategy::Binned);
    }

    #[test]
    fn test_off_serial_uses_kahan() {
        let ctx = ReductionContext {
            exec_mode: ExecMode::Serial,
            repro_mode: ReproMode::Off,
            in_nogc: false,
            is_linalg: false,
        };
        assert_eq!(select_strategy(&ctx), SumStrategy::Kahan);
    }

    #[test]
    fn test_dispatch_sum_kahan() {
        let ctx = ReductionContext::default_serial();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(dispatch_sum_f64(&values, &ctx), 15.0);
    }

    #[test]
    fn test_dispatch_sum_binned() {
        let ctx = ReductionContext::strict_parallel();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(dispatch_sum_f64(&values, &ctx), 15.0);
    }

    #[test]
    fn test_dispatch_dot_kahan() {
        let ctx = ReductionContext::default_serial();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_eq!(dispatch_dot_f64(&a, &b, &ctx), 32.0);
    }

    #[test]
    fn test_dispatch_dot_binned() {
        let ctx = ReductionContext::strict_parallel();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_eq!(dispatch_dot_f64(&a, &b, &ctx), 32.0);
    }

    #[test]
    fn test_dispatch_strategies_agree_on_simple() {
        let values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let kahan_ctx = ReductionContext::default_serial();
        let binned_ctx = ReductionContext::strict_parallel();

        let kahan_result = dispatch_sum_f64(&values, &kahan_ctx);
        let binned_result = dispatch_sum_f64(&values, &binned_ctx);

        assert_eq!(kahan_result, 5050.0);
        assert_eq!(binned_result, 5050.0);
    }
}
