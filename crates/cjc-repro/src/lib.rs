//! Deterministic computation primitives for CJC.
//!
//! This crate provides the foundational building blocks that guarantee
//! **bit-identical** results across runs, platforms, and thread counts:
//!
//! - [`Rng`] -- a SplitMix64 PRNG with explicit seed threading.  Same seed
//!   produces the identical sequence on every platform.
//! - [`KahanAccumulatorF64`] / [`KahanAccumulatorF32`] -- incremental
//!   compensated-summation accumulators (re-exported from the [`kahan`] module).
//! - [`kahan_sum_f64`] / [`kahan_sum_f32`] -- one-shot compensated summation
//!   over slices.
//! - [`pairwise_sum_f64`] -- recursive pairwise summation that falls back to
//!   Kahan summation for leaves of 32 elements or fewer.
//! - [`ReproConfig`] -- a lightweight toggle that carries the reproducibility
//!   seed through the compiler pipeline.
//!
//! # Determinism contract
//!
//! All primitives in this crate are **serial and deterministic**.  When the
//! same inputs are provided in the same order, the output is bit-for-bit
//! identical regardless of the host platform, compiler version, or OS.
//!
//! No `HashMap`, no FMA, no non-deterministic SIMD reductions.

pub mod kahan;
pub use kahan::{KahanAccumulatorF32, KahanAccumulatorF64};

/// Deterministic pseudo-random number generator using the SplitMix64 algorithm.
///
/// Guarantees identical sequences for the same seed across all platforms.
/// SplitMix64 has a period of 2^64 and passes BigCrush statistical tests.
///
/// # Determinism
///
/// Two [`Rng`] instances created with the same seed will always produce the
/// exact same sequence of values, regardless of the host OS or architecture.
/// This is the backbone of CJC's reproducible computation model.
///
/// # Examples
///
/// ```
/// use cjc_repro::Rng;
///
/// let mut rng = Rng::seeded(42);
/// let a = rng.next_f64(); // deterministic value in [0, 1)
/// let b = rng.next_u64(); // deterministic u64
/// ```
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Creates a new [`Rng`] initialized with the given seed.
    ///
    /// # Arguments
    ///
    /// * `seed` -- The initial state.  Seed `0` is valid and produces a
    ///   well-defined sequence.
    ///
    /// # Examples
    ///
    /// ```
    /// use cjc_repro::Rng;
    /// let mut rng = Rng::seeded(0);
    /// assert_eq!(rng.next_u64(), Rng::seeded(0).next_u64());
    /// ```
    pub fn seeded(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generates the next `u64` using the SplitMix64 mixing function.
    ///
    /// Advances the internal state by one step and returns a uniformly
    /// distributed 64-bit value.
    ///
    /// # Returns
    ///
    /// A deterministic `u64` drawn from the full `0..=u64::MAX` range.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Generates a uniformly distributed `f64` in the half-open interval `[0, 1)`.
    ///
    /// Uses the upper 53 bits of [`next_u64`](Self::next_u64) to fill the
    /// 53-bit mantissa of an IEEE-754 double, then divides by 2^53.
    ///
    /// # Returns
    ///
    /// A deterministic `f64` satisfying `0.0 <= value < 1.0`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Generates a uniformly distributed `f32` in the half-open interval `[0, 1)`.
    ///
    /// Uses the upper 24 bits of [`next_u64`](Self::next_u64) to fill the
    /// 24-bit mantissa of an IEEE-754 single, then divides by 2^24.
    ///
    /// # Returns
    ///
    /// A deterministic `f32` satisfying `0.0 <= value < 1.0`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Generates a sample from the standard normal distribution (mean 0, variance 1)
    /// using the Box-Muller transform.
    ///
    /// Consumes **two** uniform samples from [`next_f64`](Self::next_f64) per call.
    /// The transform is: `sqrt(-2 ln(u1)) * cos(2 pi u2)`.
    ///
    /// # Returns
    ///
    /// A deterministic `f64` drawn from N(0, 1).
    pub fn next_normal_f64(&mut self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Generates a standard-normal `f32` sample.
    ///
    /// Delegates to [`next_normal_f64`](Self::next_normal_f64) and narrows the
    /// result to `f32`.  Consumes two uniform draws from the underlying state.
    ///
    /// # Returns
    ///
    /// A deterministic `f32` drawn from N(0, 1).
    pub fn next_normal_f32(&mut self) -> f32 {
        self.next_normal_f64() as f32
    }

    /// Forks the RNG into an independent sub-stream.
    ///
    /// The returned [`Rng`] is seeded with the next `u64` drawn from `self`,
    /// so both the parent and the child advance deterministically.  This is
    /// the standard mechanism for giving each CJC closure or parallel lane
    /// its own reproducible random stream.
    ///
    /// # Returns
    ///
    /// A new [`Rng`] whose state is derived from the current generator.
    ///
    /// # Examples
    ///
    /// ```
    /// use cjc_repro::Rng;
    /// let mut parent = Rng::seeded(7);
    /// let mut child = parent.fork();
    /// // parent and child now produce independent but deterministic sequences
    /// let _ = child.next_f64();
    /// ```
    pub fn fork(&mut self) -> Rng {
        Rng {
            state: self.next_u64(),
        }
    }
}

/// Computes the sum of a slice of `f64` values using Kahan compensated summation.
///
/// Achieves an error bound of O(epsilon) for *n* summands, compared to O(*n* * epsilon)
/// for naive left-to-right addition.  Uses only two scalar registers (sum and
/// compensation) with no heap allocation.
///
/// # Arguments
///
/// * `values` -- The slice of `f64` values to sum.
///
/// # Returns
///
/// The compensated sum as `f64`.
///
/// # Determinism
///
/// The result is deterministic for a given input slice.  Different orderings of
/// the same values may yield different (but equally stable) results.
///
/// # Examples
///
/// ```
/// use cjc_repro::kahan_sum_f64;
/// let vals: Vec<f64> = (0..10_000).map(|_| 0.0001).collect();
/// let sum = kahan_sum_f64(&vals);
/// assert!((sum - 1.0).abs() < 1e-10);
/// ```
pub fn kahan_sum_f64(values: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    let mut compensation = 0.0f64;
    for &val in values {
        let y = val - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Computes the sum of a slice of `f32` values using Kahan compensated summation.
///
/// This is the single-precision counterpart to [`kahan_sum_f64`].  The error
/// bound is O(epsilon) relative to `f32` machine epsilon, with no heap
/// allocation.
///
/// # Arguments
///
/// * `values` -- The slice of `f32` values to sum.
///
/// # Returns
///
/// The compensated sum as `f32`.
///
/// # Determinism
///
/// Deterministic for a given input slice ordering.
pub fn kahan_sum_f32(values: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut compensation = 0.0f32;
    for &val in values {
        let y = val - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Computes the sum of a slice of `f64` values using recursive pairwise summation.
///
/// Recursively splits the slice in half and sums each half independently.
/// Leaves of 32 elements or fewer are reduced with [`kahan_sum_f64`].  This
/// yields an error bound of O(epsilon * log2(*n*)) with good cache locality.
///
/// # Arguments
///
/// * `values` -- The slice of `f64` values to sum.
///
/// # Returns
///
/// The pairwise-compensated sum as `f64`.
///
/// # Determinism
///
/// Deterministic for a given input slice.  The recursive split point is always
/// `len / 2`, so the tree structure is fully determined by the length.
///
/// # Examples
///
/// ```
/// use cjc_repro::pairwise_sum_f64;
/// let vals: Vec<f64> = (0..10_000).map(|_| 0.0001).collect();
/// let sum = pairwise_sum_f64(&vals);
/// assert!((sum - 1.0).abs() < 1e-10);
/// ```
pub fn pairwise_sum_f64(values: &[f64]) -> f64 {
    if values.len() <= 32 {
        return kahan_sum_f64(values);
    }
    let mid = values.len() / 2;
    pairwise_sum_f64(&values[..mid]) + pairwise_sum_f64(&values[mid..])
}

/// Configuration that controls whether deterministic reproducibility is active.
///
/// When `enabled` is `true`, the runtime seeds all [`Rng`] instances from
/// [`seed`](ReproConfig::seed) and enforces deterministic reduction ordering.
/// When `enabled` is `false`, the seed field is ignored and the runtime may
/// use a non-deterministic source.
///
/// # Examples
///
/// ```
/// use cjc_repro::ReproConfig;
///
/// let cfg = ReproConfig::enabled(42);
/// assert!(cfg.enabled);
/// assert_eq!(cfg.seed, 42);
///
/// let off = ReproConfig::disabled();
/// assert!(!off.enabled);
/// ```
#[derive(Debug, Clone)]
pub struct ReproConfig {
    /// Whether reproducibility mode is active.
    pub enabled: bool,
    /// The global seed used to initialize all [`Rng`] instances when
    /// reproducibility is enabled.
    pub seed: u64,
}

impl ReproConfig {
    /// Creates a [`ReproConfig`] with reproducibility **disabled**.
    ///
    /// The seed is set to `0` but will not be used by the runtime.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            seed: 0,
        }
    }

    /// Creates a [`ReproConfig`] with reproducibility **enabled** using the
    /// given `seed`.
    ///
    /// # Arguments
    ///
    /// * `seed` -- The global seed that will be threaded through the runtime.
    pub fn enabled(seed: u64) -> Self {
        Self {
            enabled: true,
            seed,
        }
    }
}

impl Default for ReproConfig {
    /// Returns [`ReproConfig::disabled()`].
    fn default() -> Self {
        Self::disabled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_deterministic() {
        let mut rng1 = Rng::seeded(42);
        let mut rng2 = Rng::seeded(42);

        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_rng_f64_range() {
        let mut rng = Rng::seeded(123);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_rng_fork_deterministic() {
        let mut rng1 = Rng::seeded(42);
        let mut rng2 = Rng::seeded(42);

        let mut fork1 = rng1.fork();
        let mut fork2 = rng2.fork();

        for _ in 0..50 {
            assert_eq!(fork1.next_u64(), fork2.next_u64());
        }
    }

    #[test]
    fn test_kahan_sum() {
        // Sum of many small values where naive sum would lose precision
        let values: Vec<f64> = (0..10000).map(|_| 0.0001).collect();
        let result = kahan_sum_f64(&values);
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kahan_sum_f32() {
        let values: Vec<f32> = (0..10000).map(|_| 0.0001f32).collect();
        let result = kahan_sum_f32(&values);
        assert!((result - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_pairwise_sum() {
        let values: Vec<f64> = (0..10000).map(|_| 0.0001).collect();
        let result = pairwise_sum_f64(&values);
        assert!((result - 1.0).abs() < 1e-10);
    }
}
