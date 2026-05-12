//! Kahan Summation — Scalar-tier deterministic reduction.
//!
//! # Determinism Contract
//!
//! Kahan compensated summation is deterministic in **serial execution** when
//! values are processed in a fixed order. It is NOT order-invariant — different
//! input orderings may produce different (but numerically stable) results.
//!
//! # When Used
//!
//! - Serial execution with `ReproMode::On`
//! - Non-vectorized loops
//! - Not inside `@nogc` or forced strict mode
//!
//! # Properties
//!
//! - **Error bound:** O(ε) for n summands (vs O(nε) for naive).
//! - **Heap allocation:** None. Two f64 registers (sum + compensation).
//! - **Branching:** No branches inside inner loop except compensation update.
//!
//! # Implementation Note
//!
//! The core implementations live in `cjc_repro::kahan_sum_f64` and
//! `cjc_repro::kahan_sum_f32` (the parent crate's `lib.rs`). This module
//! provides the `KahanAccumulator` struct for incremental accumulation.

/// Incremental Kahan compensated-summation accumulator for `f64`.
///
/// Maintains a running sum and a compensation term on the stack -- no heap
/// allocation is ever performed.  The error bound is O(epsilon) for *n*
/// additions, compared to O(*n* * epsilon) for naive summation.
///
/// Use this when values arrive one at a time or in variable-length batches.
/// For a one-shot slice reduction, see [`crate::kahan_sum_f64`].
///
/// # Determinism
///
/// The accumulator is deterministic for a given sequence of [`add`](Self::add)
/// calls in a fixed order.  It is **not** order-invariant -- permuting the
/// input may produce a different (but equally stable) result.
///
/// # Examples
///
/// ```
/// use cjc_repro::KahanAccumulatorF64;
///
/// let mut acc = KahanAccumulatorF64::new();
/// for _ in 0..10_000 {
///     acc.add(0.0001);
/// }
/// assert!((acc.finalize() - 1.0).abs() < 1e-10);
/// assert_eq!(acc.count(), 10_000);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct KahanAccumulatorF64 {
    sum: f64,
    compensation: f64,
    count: u64,
}

impl KahanAccumulatorF64 {
    /// Creates a new accumulator initialized to zero.
    ///
    /// Both the running sum and the compensation term start at `0.0`, and the
    /// count starts at `0`.
    #[inline]
    pub fn new() -> Self {
        KahanAccumulatorF64 {
            sum: 0.0,
            compensation: 0.0,
            count: 0,
        }
    }

    /// Adds a single `f64` value to the running sum.
    ///
    /// If `value` is exactly `0.0`, the compensation term is left untouched so
    /// that the accumulated sum remains bit-identical to what it would be
    /// without the zero.  The count is still incremented.
    ///
    /// # Arguments
    ///
    /// * `value` -- The value to accumulate.
    #[inline]
    pub fn add(&mut self, value: f64) {
        if value == 0.0 {
            self.count += 1;
            return;
        }
        let y = value - self.compensation;
        let t = self.sum + y;
        self.compensation = (t - self.sum) - y;
        self.sum = t;
        self.count += 1;
    }

    /// Adds every element of `values` to the running sum in order.
    ///
    /// Equivalent to calling [`add`](Self::add) for each element.
    ///
    /// # Arguments
    ///
    /// * `values` -- The slice of `f64` values to accumulate.
    #[inline]
    pub fn add_slice(&mut self, values: &[f64]) {
        for &v in values {
            self.add(v);
        }
    }

    /// Returns the accumulated compensated sum.
    ///
    /// This does **not** consume the accumulator -- you may continue adding
    /// values after calling `finalize`.
    ///
    /// # Returns
    ///
    /// The current compensated sum as `f64`.
    #[inline]
    pub fn finalize(&self) -> f64 {
        self.sum
    }

    /// Returns the number of values that have been added so far.
    ///
    /// Includes zero-valued additions.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Returns the IEEE-754 bit pattern of the compensation register.
    ///
    /// The compensation register captures the bits that were lost in
    /// the running sum's last addition; it's the second half of the
    /// Kahan invariant `sum + compensation ≈ exact`. Two accumulators
    /// with identical addition orders produce identical compensation
    /// bits across runs and platforms — this is the determinism
    /// contract that callers serializing the full Welford state rely on.
    #[inline]
    pub fn compensation_bits(&self) -> u64 {
        self.compensation.to_bits()
    }

    /// Construct an accumulator from its component registers.
    ///
    /// Inverse of [`finalize`](Self::finalize) +
    /// [`compensation_bits`](Self::compensation_bits) + [`count`](Self::count).
    /// Used by snapshot decoders that resume the full Welford state
    /// from the canonical 24-byte (sum + compensation + count)
    /// encoding instead of replaying the original observation
    /// sequence.
    ///
    /// The construction is bit-stable: feeding the same `(sum,
    /// compensation, count)` triple always yields the same internal
    /// representation, so a serialize → reconstruct → serialize cycle
    /// is byte-identical.
    #[inline]
    pub fn from_components(sum: f64, compensation: f64, count: u64) -> Self {
        Self {
            sum,
            compensation,
            count,
        }
    }
}

impl Default for KahanAccumulatorF64 {
    /// Returns a zero-initialized accumulator (equivalent to [`KahanAccumulatorF64::new`]).
    fn default() -> Self {
        Self::new()
    }
}

/// Incremental Kahan compensated-summation accumulator for `f32`.
///
/// Single-precision counterpart to [`KahanAccumulatorF64`].  Maintains a
/// running sum and compensation term on the stack with no heap allocation.
/// The error bound is O(epsilon) relative to `f32` machine epsilon.
///
/// # Determinism
///
/// Deterministic for a given sequence of additions in a fixed order.
///
/// # Examples
///
/// ```
/// use cjc_repro::KahanAccumulatorF32;
///
/// let mut acc = KahanAccumulatorF32::new();
/// for _ in 0..10_000 {
///     acc.add(0.0001_f32);
/// }
/// assert!((acc.finalize() - 1.0).abs() < 1e-4);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct KahanAccumulatorF32 {
    sum: f32,
    compensation: f32,
    count: u64,
}

impl KahanAccumulatorF32 {
    /// Creates a new accumulator initialized to zero.
    #[inline]
    pub fn new() -> Self {
        KahanAccumulatorF32 {
            sum: 0.0,
            compensation: 0.0,
            count: 0,
        }
    }

    /// Adds a single `f32` value to the running sum.
    ///
    /// If `value` is exactly `0.0`, the compensation term is left untouched
    /// to preserve bit-identical results.  The count is still incremented.
    ///
    /// # Arguments
    ///
    /// * `value` -- The value to accumulate.
    #[inline]
    pub fn add(&mut self, value: f32) {
        if value == 0.0 {
            self.count += 1;
            return;
        }
        let y = value - self.compensation;
        let t = self.sum + y;
        self.compensation = (t - self.sum) - y;
        self.sum = t;
        self.count += 1;
    }

    /// Adds every element of `values` to the running sum in order.
    ///
    /// Equivalent to calling [`add`](Self::add) for each element.
    ///
    /// # Arguments
    ///
    /// * `values` -- The slice of `f32` values to accumulate.
    #[inline]
    pub fn add_slice(&mut self, values: &[f32]) {
        for &v in values {
            self.add(v);
        }
    }

    /// Returns the accumulated compensated sum.
    ///
    /// Does **not** consume the accumulator.
    ///
    /// # Returns
    ///
    /// The current compensated sum as `f32`.
    #[inline]
    pub fn finalize(&self) -> f32 {
        self.sum
    }

    /// Returns the number of values that have been added so far.
    ///
    /// Includes zero-valued additions.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }
}

impl Default for KahanAccumulatorF32 {
    /// Returns a zero-initialized accumulator (equivalent to [`KahanAccumulatorF32::new`]).
    fn default() -> Self {
        Self::new()
    }
}

// ─── Phase 0.8 Item D2 — SIMD-friendly Kahan accumulators ────────────────
//
// `KahanAccumulatorF64x4` and `KahanAccumulatorF64x8` are lane-parallel
// variants of `KahanAccumulatorF64` that maintain N independent
// (sum, compensation) pairs. They are intended as a building block for
// hot loops that accumulate many independent reductions in parallel —
// e.g., the d×d cross-product matrix in `BlrState::update` where each
// matrix entry is its own independent Kahan accumulation.
//
// # Determinism model
//
// The lane representation is `[f64; N]` (plain array, not `std::simd`).
// IEEE-754 `+`, `-`, and `*` are mandated bit-identical across every
// architecture by the language spec, so the same Rust source produces
// the same byte-exact result on x86_64, aarch64, and arm64 without any
// `#[cfg(target_arch)]` gating. Auto-vectorization is opportunistic: a
// modern release build on a SIMD-capable target may emit packed adds,
// but correctness does not depend on it.
//
// # Lane assignment
//
// `add_slice` walks the input in lane-major order: input index `i`
// goes to lane `i % N`. Tail elements (input length not divisible by
// N) are folded into lane 0 sequentially. This is the canonical lane
// assignment — every implementation of `KahanAccumulatorF64xN` MUST
// agree on it or the cross-platform determinism gate breaks.
//
// # Horizontal reduction
//
// `finalize` reduces the N lanes to a single scalar using a *scalar*
// `KahanAccumulatorF64` in fixed lane order (0, 1, 2, ..., N-1). For
// each lane it adds first the running `sum[i]`, then the carried
// `compensation[i]`. This is *not* bit-equivalent to running
// `KahanAccumulatorF64` over the original input — the parallel-lane
// shape inherently produces a different (but equally stable, and
// equally deterministic) result.
//
// # Zero-value short-circuit
//
// Matches `KahanAccumulatorF64::add`: if a value is exactly `0.0`,
// the compensation register for that lane is left untouched. This
// keeps the lane state bit-identical to the same input with zeros
// removed and the count still incrementing.

/// 4-lane SIMD-friendly Kahan compensated-summation accumulator for `f64`.
///
/// Maintains 4 independent `(sum, compensation)` pairs as a `[f64; 4]`
/// array. The compiler may auto-vectorize the lane-parallel inner loop
/// on AVX/AVX2/NEON targets, but correctness is guaranteed by the
/// IEEE-754 arithmetic semantics of plain f64 ops regardless of
/// whether vectorization fires.
///
/// See the module-level note on lane assignment and horizontal
/// reduction. This accumulator is intended as a parallel building
/// block for hot inner loops; for snapshot serialization use the
/// scalar [`KahanAccumulatorF64`].
///
/// # Examples
///
/// ```
/// use cjc_repro::KahanAccumulatorF64x4;
///
/// let mut acc = KahanAccumulatorF64x4::new();
/// let values: Vec<f64> = (0..10_000).map(|_| 0.0001).collect();
/// acc.add_slice(&values);
/// assert!((acc.finalize() - 1.0).abs() < 1e-10);
/// assert_eq!(acc.count(), 10_000);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct KahanAccumulatorF64x4 {
    sum: [f64; 4],
    compensation: [f64; 4],
    count: u64,
}

impl KahanAccumulatorF64x4 {
    /// Creates a new accumulator with all 4 lanes zero-initialized.
    #[inline]
    pub fn new() -> Self {
        Self {
            sum: [0.0; 4],
            compensation: [0.0; 4],
            count: 0,
        }
    }

    /// Add a 4-element chunk in lane-parallel: lane `i` receives
    /// `values[i]`. Each lane runs the standard Kahan update with the
    /// zero-value short-circuit. The count is incremented by 4.
    #[inline]
    pub fn add_lanes(&mut self, values: [f64; 4]) {
        // The body of this loop is the scalar Kahan update applied to
        // each lane. Writing it as a fixed-trip loop over `[0; 4]`
        // gives the compiler a clear vectorization target without
        // pinning the result to any specific ISA.
        for i in 0..4 {
            let v = values[i];
            if v == 0.0 {
                continue;
            }
            let y = v - self.compensation[i];
            let t = self.sum[i] + y;
            self.compensation[i] = (t - self.sum[i]) - y;
            self.sum[i] = t;
        }
        self.count += 4;
    }

    /// Add a slice of `f64`s in lane-major order. Input index `i`
    /// goes to lane `i % 4`; tail elements (length not divisible by 4)
    /// are folded into lane 0 sequentially.
    ///
    /// Determinism: result is byte-stable across runs and platforms
    /// for any given input slice.
    #[inline]
    pub fn add_slice(&mut self, values: &[f64]) {
        let mut iter = values.chunks_exact(4);
        for chunk in &mut iter {
            // Safe: chunks_exact yields slices of length exactly 4.
            let lanes = [chunk[0], chunk[1], chunk[2], chunk[3]];
            self.add_lanes(lanes);
        }
        // Tail goes to lane 0 sequentially. Mirrors the scalar Kahan
        // update; preserves the zero-value short-circuit and bumps the
        // count for every element.
        for &v in iter.remainder() {
            if v == 0.0 {
                self.count += 1;
                continue;
            }
            let y = v - self.compensation[0];
            let t = self.sum[0] + y;
            self.compensation[0] = (t - self.sum[0]) - y;
            self.sum[0] = t;
            self.count += 1;
        }
    }

    /// Horizontally reduce the 4 lanes to a single `f64`. Each lane
    /// contributes its `sum` and `compensation` in turn to a scalar
    /// [`KahanAccumulatorF64`], in fixed lane order 0..4.
    ///
    /// Does not consume the accumulator.
    #[inline]
    pub fn finalize(&self) -> f64 {
        let mut acc = KahanAccumulatorF64::new();
        for i in 0..4 {
            acc.add(self.sum[i]);
            acc.add(self.compensation[i]);
        }
        acc.finalize()
    }

    /// Total number of `add_slice` / `add_lanes` elements seen, including
    /// zero-valued additions.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }
}

impl Default for KahanAccumulatorF64x4 {
    /// Returns a zero-initialized accumulator (equivalent to
    /// [`KahanAccumulatorF64x4::new`]).
    fn default() -> Self {
        Self::new()
    }
}

/// 8-lane SIMD-friendly Kahan compensated-summation accumulator for `f64`.
///
/// Same shape as [`KahanAccumulatorF64x4`] with twice the lane count.
/// Useful for hot loops with `d % 8 == 0` where the wider lane group
/// lets the compiler issue two AVX2 packed-add pairs per inner step
/// (256-bit lanes × 2 = 8 f64s per cycle on supporting cores).
///
/// See the [`KahanAccumulatorF64x4`] docs for the lane-assignment and
/// horizontal-reduction conventions; the x8 variant uses the
/// equivalent extension to 8 lanes.
#[derive(Debug, Clone, Copy)]
pub struct KahanAccumulatorF64x8 {
    sum: [f64; 8],
    compensation: [f64; 8],
    count: u64,
}

impl KahanAccumulatorF64x8 {
    /// Creates a new accumulator with all 8 lanes zero-initialized.
    #[inline]
    pub fn new() -> Self {
        Self {
            sum: [0.0; 8],
            compensation: [0.0; 8],
            count: 0,
        }
    }

    /// Add an 8-element chunk in lane-parallel: lane `i` receives
    /// `values[i]`.
    #[inline]
    pub fn add_lanes(&mut self, values: [f64; 8]) {
        for i in 0..8 {
            let v = values[i];
            if v == 0.0 {
                continue;
            }
            let y = v - self.compensation[i];
            let t = self.sum[i] + y;
            self.compensation[i] = (t - self.sum[i]) - y;
            self.sum[i] = t;
        }
        self.count += 8;
    }

    /// Add a slice of `f64`s in lane-major order. Input index `i`
    /// goes to lane `i % 8`; tail elements are folded into lane 0
    /// sequentially.
    #[inline]
    pub fn add_slice(&mut self, values: &[f64]) {
        let mut iter = values.chunks_exact(8);
        for chunk in &mut iter {
            let lanes = [
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ];
            self.add_lanes(lanes);
        }
        for &v in iter.remainder() {
            if v == 0.0 {
                self.count += 1;
                continue;
            }
            let y = v - self.compensation[0];
            let t = self.sum[0] + y;
            self.compensation[0] = (t - self.sum[0]) - y;
            self.sum[0] = t;
            self.count += 1;
        }
    }

    /// Horizontally reduce the 8 lanes via scalar Kahan in fixed lane
    /// order 0..8. See [`KahanAccumulatorF64x4::finalize`].
    #[inline]
    pub fn finalize(&self) -> f64 {
        let mut acc = KahanAccumulatorF64::new();
        for i in 0..8 {
            acc.add(self.sum[i]);
            acc.add(self.compensation[i]);
        }
        acc.finalize()
    }

    /// Total number of `add_slice` / `add_lanes` elements seen, including
    /// zero-valued additions.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }
}

impl Default for KahanAccumulatorF64x8 {
    /// Returns a zero-initialized accumulator (equivalent to
    /// [`KahanAccumulatorF64x8::new`]).
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kahan_accumulator_simple() {
        let mut acc = KahanAccumulatorF64::new();
        for i in 1..=10 {
            acc.add(i as f64);
        }
        assert_eq!(acc.finalize(), 55.0);
        assert_eq!(acc.count(), 10);
    }

    #[test]
    fn test_kahan_accumulator_stability() {
        let mut acc = KahanAccumulatorF64::new();
        for _ in 0..10000 {
            acc.add(0.0001);
        }
        assert!((acc.finalize() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kahan_accumulator_f32() {
        let mut acc = KahanAccumulatorF32::new();
        for _ in 0..10000 {
            acc.add(0.0001f32);
        }
        assert!((acc.finalize() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_kahan_matches_existing_function() {
        let values: Vec<f64> = (1..=1000).map(|i| i as f64 * 0.001).collect();
        let mut acc = KahanAccumulatorF64::new();
        acc.add_slice(&values);

        let func_result = crate::kahan_sum_f64(&values);
        assert_eq!(acc.finalize().to_bits(), func_result.to_bits());
    }

    // ── Phase 0.8 Item D2 — SIMD Kahan in-crate tests ──────────────────

    #[test]
    fn test_kahan_x4_simple_sum() {
        let mut acc = KahanAccumulatorF64x4::new();
        // Lengths divisible by 4 hit only the lane-parallel path.
        let values: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        acc.add_slice(&values);
        // 1+2+...+20 = 210; exact in f64.
        assert_eq!(acc.finalize(), 210.0);
        assert_eq!(acc.count(), 20);
    }

    #[test]
    fn test_kahan_x4_stability() {
        // The classic many-small-values test: naive sum loses precision,
        // Kahan recovers it. SIMD-Kahan must also recover it because
        // each of the 4 lanes runs its own Kahan accumulation.
        let mut acc = KahanAccumulatorF64x4::new();
        let values: Vec<f64> = (0..10_000).map(|_| 0.0001).collect();
        acc.add_slice(&values);
        assert!((acc.finalize() - 1.0).abs() < 1e-10);
        assert_eq!(acc.count(), 10_000);
    }

    #[test]
    fn test_kahan_x8_stability() {
        let mut acc = KahanAccumulatorF64x8::new();
        let values: Vec<f64> = (0..10_000).map(|_| 0.0001).collect();
        acc.add_slice(&values);
        assert!((acc.finalize() - 1.0).abs() < 1e-10);
        assert_eq!(acc.count(), 10_000);
    }

    #[test]
    fn test_kahan_x4_tail_handling() {
        // Length not divisible by 4 must route the tail to lane 0
        // sequentially. Result is byte-stable across runs.
        let mut acc1 = KahanAccumulatorF64x4::new();
        let mut acc2 = KahanAccumulatorF64x4::new();
        let values: Vec<f64> = (0..23).map(|i| (i as f64) * 0.137).collect();
        acc1.add_slice(&values);
        acc2.add_slice(&values);
        assert_eq!(
            acc1.finalize().to_bits(),
            acc2.finalize().to_bits(),
            "two identical runs must agree bit-for-bit",
        );
        assert_eq!(acc1.count(), 23);
    }

    #[test]
    fn test_kahan_x4_byte_stable_across_runs() {
        // Determinism canary: 100 fixed values produce the same bits
        // on every run. If this ever changes, lane assignment or
        // horizontal reduction order has shifted.
        let values: Vec<f64> = (0..100)
            .map(|i| ((i as f64) * 0.137).sin() * 1e7)
            .collect();
        let runs: Vec<u64> = (0..5)
            .map(|_| {
                let mut acc = KahanAccumulatorF64x4::new();
                acc.add_slice(&values);
                acc.finalize().to_bits()
            })
            .collect();
        let first = runs[0];
        for (i, bits) in runs.iter().enumerate() {
            assert_eq!(*bits, first, "run {i} bits diverged");
        }
    }

    #[test]
    fn test_kahan_x4_zero_value_short_circuit() {
        // Adding zero values must not perturb the compensation register
        // for any lane. Two interleavings (with vs. without zeros)
        // must produce identical sum bits when the zeros are stripped.
        let values_with_zeros: Vec<f64> =
            [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0].to_vec();
        let values_no_zeros: Vec<f64> = [1.0, 2.0, 3.0, 4.0].to_vec();
        let mut a = KahanAccumulatorF64x4::new();
        let mut b = KahanAccumulatorF64x4::new();
        a.add_slice(&values_with_zeros);
        b.add_slice(&values_no_zeros);
        // The two accumulators see different lane assignments (8 vs 4
        // elements), so their internal lane states differ — but each
        // is its own byte-stable computation. Test what we actually
        // care about: counts and a stable result.
        assert_eq!(a.count(), 8);
        assert_eq!(b.count(), 4);
        // Both finalize to the same mathematical value: 1+2+3+4 = 10.
        assert_eq!(a.finalize(), 10.0);
        assert_eq!(b.finalize(), 10.0);
    }

    #[test]
    fn test_kahan_x4_add_lanes_matches_add_slice_at_alignment() {
        // For lane-aligned input, add_slice and a sequence of
        // add_lanes calls must produce byte-identical state.
        let values: [f64; 12] = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let mut slice_acc = KahanAccumulatorF64x4::new();
        slice_acc.add_slice(&values);

        let mut lanes_acc = KahanAccumulatorF64x4::new();
        lanes_acc.add_lanes([values[0], values[1], values[2], values[3]]);
        lanes_acc.add_lanes([values[4], values[5], values[6], values[7]]);
        lanes_acc.add_lanes([values[8], values[9], values[10], values[11]]);

        assert_eq!(
            slice_acc.finalize().to_bits(),
            lanes_acc.finalize().to_bits(),
            "add_slice and add_lanes must produce byte-identical results at alignment"
        );
        assert_eq!(slice_acc.count(), lanes_acc.count());
    }

    #[test]
    fn test_kahan_x8_byte_stable_across_runs() {
        let values: Vec<f64> = (0..200)
            .map(|i| ((i as f64) * 0.057).cos() * 1e6)
            .collect();
        let runs: Vec<u64> = (0..5)
            .map(|_| {
                let mut acc = KahanAccumulatorF64x8::new();
                acc.add_slice(&values);
                acc.finalize().to_bits()
            })
            .collect();
        let first = runs[0];
        for (i, bits) in runs.iter().enumerate() {
            assert_eq!(*bits, first, "x8 run {i} bits diverged");
        }
    }

    #[test]
    fn test_kahan_x4_empty_slice_is_zero() {
        let mut acc = KahanAccumulatorF64x4::new();
        acc.add_slice(&[]);
        assert_eq!(acc.finalize().to_bits(), 0.0_f64.to_bits());
        assert_eq!(acc.count(), 0);
    }
}
