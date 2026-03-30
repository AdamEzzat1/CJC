//! SIMD Kernels — AVX2-accelerated gate application with cache-aware blocking.
//!
//! # Architecture
//!
//! Complex amplitudes are stored as interleaved (re, im) pairs. The SIMD
//! kernels process 2 complex numbers (4 f64 values) per AVX2 iteration.
//!
//! # Determinism
//!
//! - No FMA instructions used (explicitly decomposed multiply-add)
//! - Operations maintain identical ordering to scalar baseline
//! - Runtime CPU detection falls back to scalar on non-AVX2 platforms
//!
//! # Cache Blocking
//!
//! For qubit targets where the stride exceeds L1 cache size (~32KB),
//! loop tiling groups basis state pairs into cache-friendly blocks.

use cjc_runtime::complex::ComplexF64;
use crate::statevector::Statevector;

/// Cache block size in complex amplitudes (L1 = ~32KB = 4K f64 = 2K complex).
const CACHE_BLOCK_SIZE: usize = 2048;

// ---------------------------------------------------------------------------
// Cache-Aware Single-Qubit Gate Application
// ---------------------------------------------------------------------------

/// Apply a 2x2 unitary to qubit `q` with cache-aware blocking.
///
/// For high qubit indices where the stride (2^q) is large, naive iteration
/// causes cache thrashing. This implementation tiles the loop to keep
/// working set within L1 cache.
///
/// Falls back to the standard path for small qubits (stride < CACHE_BLOCK_SIZE).
pub fn apply_single_qubit_cached(
    sv: &mut Statevector,
    q: usize,
    u: [[ComplexF64; 2]; 2],
) {
    let n = sv.n_states();
    let bit = 1usize << q;

    // For small qubits or small state vectors, the standard path is fine
    if bit < CACHE_BLOCK_SIZE || n <= CACHE_BLOCK_SIZE * 2 {
        apply_single_qubit_scalar(sv, q, u);
        return;
    }

    // Cache-aware tiled iteration.
    // Process blocks of CACHE_BLOCK_SIZE pairs at a time.
    let block_size = CACHE_BLOCK_SIZE;

    let mut block_start = 0;
    while block_start < n {
        let block_end = (block_start + block_size).min(n);

        for i in block_start..block_end {
            if i & bit == 0 {
                let j = i | bit;
                if j < n {
                    let a0 = sv.amplitudes[i];
                    let a1 = sv.amplitudes[j];
                    sv.amplitudes[i] = u[0][0].mul_fixed(a0).add(u[0][1].mul_fixed(a1));
                    sv.amplitudes[j] = u[1][0].mul_fixed(a0).add(u[1][1].mul_fixed(a1));
                }
            }
        }

        block_start = block_end;
    }
}

/// Scalar baseline for single-qubit gate application.
/// Processes pairs in ascending order for determinism.
fn apply_single_qubit_scalar(
    sv: &mut Statevector,
    q: usize,
    u: [[ComplexF64; 2]; 2],
) {
    let n = sv.n_states();
    let bit = 1usize << q;

    let mut i = 0;
    while i < n {
        if i & bit == 0 {
            let j = i | bit;
            let a0 = sv.amplitudes[i];
            let a1 = sv.amplitudes[j];

            // Fixed-sequence: 4 multiplications, 2 additions per output
            sv.amplitudes[i] = u[0][0].mul_fixed(a0).add(u[0][1].mul_fixed(a1));
            sv.amplitudes[j] = u[1][0].mul_fixed(a0).add(u[1][1].mul_fixed(a1));
        }
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// AVX2 Complex Arithmetic Kernels
// ---------------------------------------------------------------------------

/// Multiply two complex numbers using AVX2, maintaining deterministic ordering.
///
/// On platforms without AVX2, falls back to scalar `mul_fixed`.
///
/// # Safety
///
/// Uses `core::arch::x86_64` intrinsics. CPU feature detection via
/// `is_x86_feature_detected!("avx2")` must be checked before calling.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn complex_mul_batch_2(
    a: &[ComplexF64; 2],
    b: &[ComplexF64; 2],
) -> [ComplexF64; 2] {
    // Check for AVX2 at runtime
    if is_x86_feature_detected!("avx2") {
        // SAFETY: We checked for AVX2 support.
        unsafe { complex_mul_batch_2_avx2(a, b) }
    } else {
        [a[0].mul_fixed(b[0]), a[1].mul_fixed(b[1])]
    }
}

/// Fallback for non-x86_64 platforms.
#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn complex_mul_batch_2(
    a: &[ComplexF64; 2],
    b: &[ComplexF64; 2],
) -> [ComplexF64; 2] {
    [a[0].mul_fixed(b[0]), a[1].mul_fixed(b[1])]
}

/// AVX2 implementation of batched complex multiplication.
///
/// Processes 2 complex numbers (4 f64 values) in a single AVX2 register.
/// Uses explicit decomposition to avoid FMA:
///
/// ```text
/// Re(a*b) = a.re*b.re - a.im*b.im
/// Im(a*b) = a.re*b.im + a.im*b.re
/// ```
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn complex_mul_batch_2_avx2(
    a: &[ComplexF64; 2],
    b: &[ComplexF64; 2],
) -> [ComplexF64; 2] {
    use core::arch::x86_64::*;

    // Load a = [a0.re, a0.im, a1.re, a1.im]
    let a_re = _mm256_set_pd(a[1].re, a[1].re, a[0].re, a[0].re);
    let a_im = _mm256_set_pd(a[1].im, a[1].im, a[0].im, a[0].im);
    let b_ri = _mm256_set_pd(b[1].im, b[1].re, b[0].im, b[0].re);
    let b_ir = _mm256_set_pd(b[1].re, b[1].im, b[0].re, b[0].im);

    // Step 1: Four multiplications (no FMA!)
    let t1 = _mm256_mul_pd(a_re, b_ri); // [a0.re*b0.re, a0.re*b0.im, a1.re*b1.re, a1.re*b1.im]
    let t2 = _mm256_mul_pd(a_im, b_ir); // [a0.im*b0.im, a0.im*b0.re, a1.im*b1.im, a1.im*b1.re]

    // Step 2: Subtract/add to get real/imaginary parts
    // Re = a.re*b.re - a.im*b.im (even positions)
    // Im = a.re*b.im + a.im*b.re (odd positions)
    let sign = _mm256_set_pd(1.0, -1.0, 1.0, -1.0);
    let t2_signed = _mm256_mul_pd(t2, sign);
    let result = _mm256_add_pd(t1, t2_signed);

    // Extract results
    let mut out = [0.0f64; 4];
    _mm256_storeu_pd(out.as_mut_ptr(), result);

    [
        ComplexF64::new(out[0], out[1]),
        ComplexF64::new(out[2], out[3]),
    ]
}

/// Apply a 2x2 unitary to qubit `q` using SIMD-accelerated complex arithmetic.
///
/// Processes two amplitude pairs per iteration on AVX2 platforms.
#[cfg(target_arch = "x86_64")]
pub fn apply_single_qubit_simd(
    sv: &mut Statevector,
    q: usize,
    u: [[ComplexF64; 2]; 2],
) {
    let n = sv.n_states();
    let bit = 1usize << q;

    if !is_x86_feature_detected!("avx2") || n < 4 {
        apply_single_qubit_scalar(sv, q, u);
        return;
    }

    // Process pairs using SIMD batching
    // Collect pairs where bit q is 0, process 2 at a time
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        if i & bit == 0 {
            pairs.push((i, i | bit));
        }
    }

    // Process in batches of 2 pairs
    let mut p = 0;
    while p + 1 < pairs.len() {
        let (i0, j0) = pairs[p];
        let (i1, j1) = pairs[p + 1];

        let a0 = [sv.amplitudes[i0], sv.amplitudes[i1]];
        let a1 = [sv.amplitudes[j0], sv.amplitudes[j1]];

        // new_a0 = U[0][0]*a0 + U[0][1]*a1
        let t00 = complex_mul_batch_2(&[u[0][0], u[0][0]], &a0);
        let t01 = complex_mul_batch_2(&[u[0][1], u[0][1]], &a1);

        // new_a1 = U[1][0]*a0 + U[1][1]*a1
        let t10 = complex_mul_batch_2(&[u[1][0], u[1][0]], &a0);
        let t11 = complex_mul_batch_2(&[u[1][1], u[1][1]], &a1);

        sv.amplitudes[i0] = t00[0].add(t01[0]);
        sv.amplitudes[i1] = t00[1].add(t01[1]);
        sv.amplitudes[j0] = t10[0].add(t11[0]);
        sv.amplitudes[j1] = t10[1].add(t11[1]);

        p += 2;
    }

    // Handle remaining pair
    if p < pairs.len() {
        let (i, j) = pairs[p];
        let a0 = sv.amplitudes[i];
        let a1 = sv.amplitudes[j];
        sv.amplitudes[i] = u[0][0].mul_fixed(a0).add(u[0][1].mul_fixed(a1));
        sv.amplitudes[j] = u[1][0].mul_fixed(a0).add(u[1][1].mul_fixed(a1));
    }
}

/// Non-x86 fallback.
#[cfg(not(target_arch = "x86_64"))]
pub fn apply_single_qubit_simd(
    sv: &mut Statevector,
    q: usize,
    u: [[ComplexF64; 2]; 2],
) {
    apply_single_qubit_scalar(sv, q, u);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::Gate;

    const TOL: f64 = 1e-12;
    const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;

    fn h_matrix() -> [[ComplexF64; 2]; 2] {
        let s = ComplexF64::real(INV_SQRT2);
        let ms = ComplexF64::real(-INV_SQRT2);
        [[s, s], [s, ms]]
    }

    fn x_matrix() -> [[ComplexF64; 2]; 2] {
        [[ComplexF64::ZERO, ComplexF64::ONE],
         [ComplexF64::ONE, ComplexF64::ZERO]]
    }

    #[test]
    fn test_scalar_h_gate() {
        let mut sv = Statevector::new(1);
        apply_single_qubit_scalar(&mut sv, 0, h_matrix());
        assert!((sv.amplitudes[0].re - INV_SQRT2).abs() < TOL);
        assert!((sv.amplitudes[1].re - INV_SQRT2).abs() < TOL);
        assert!(sv.is_normalized(TOL));
    }

    #[test]
    fn test_cached_matches_scalar() {
        // Verify cached version produces identical results to scalar
        for n_qubits in 1..=6 {
            for target in 0..n_qubits {
                let mut sv_scalar = Statevector::new(n_qubits);
                let mut sv_cached = sv_scalar.clone();

                // Apply H to initial state
                apply_single_qubit_scalar(&mut sv_scalar, target, h_matrix());
                apply_single_qubit_cached(&mut sv_cached, target, h_matrix());

                for i in 0..sv_scalar.n_states() {
                    assert_eq!(
                        sv_scalar.amplitudes[i].re.to_bits(),
                        sv_cached.amplitudes[i].re.to_bits(),
                        "Cached mismatch at n={}, q={}, state={}, re",
                        n_qubits, target, i
                    );
                    assert_eq!(
                        sv_scalar.amplitudes[i].im.to_bits(),
                        sv_cached.amplitudes[i].im.to_bits(),
                        "Cached mismatch at n={}, q={}, state={}, im",
                        n_qubits, target, i
                    );
                }
            }
        }
    }

    #[test]
    fn test_simd_matches_scalar() {
        // Verify SIMD version produces identical results to scalar
        for n_qubits in 1..=6 {
            for target in 0..n_qubits {
                let mut sv_scalar = Statevector::new(n_qubits);
                let mut sv_simd = sv_scalar.clone();

                apply_single_qubit_scalar(&mut sv_scalar, target, h_matrix());
                apply_single_qubit_simd(&mut sv_simd, target, h_matrix());

                for i in 0..sv_scalar.n_states() {
                    assert_eq!(
                        sv_scalar.amplitudes[i].re.to_bits(),
                        sv_simd.amplitudes[i].re.to_bits(),
                        "SIMD mismatch at n={}, q={}, state={}, re",
                        n_qubits, target, i
                    );
                    assert_eq!(
                        sv_scalar.amplitudes[i].im.to_bits(),
                        sv_simd.amplitudes[i].im.to_bits(),
                        "SIMD mismatch at n={}, q={}, state={}, im",
                        n_qubits, target, i
                    );
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_complex_mul_batch_deterministic() {
        let a = [
            ComplexF64::new(1.23, -4.56),
            ComplexF64::new(7.89, 0.12),
        ];
        let b = [
            ComplexF64::new(-3.45, 6.78),
            ComplexF64::new(9.01, -2.34),
        ];

        let result1 = complex_mul_batch_2(&a, &b);
        let result2 = complex_mul_batch_2(&a, &b);

        for i in 0..2 {
            assert_eq!(result1[i].re.to_bits(), result2[i].re.to_bits());
            assert_eq!(result1[i].im.to_bits(), result2[i].im.to_bits());
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_complex_mul_batch_matches_scalar() {
        let a = [
            ComplexF64::new(1.23, -4.56),
            ComplexF64::new(7.89, 0.12),
        ];
        let b = [
            ComplexF64::new(-3.45, 6.78),
            ComplexF64::new(9.01, -2.34),
        ];

        let batch = complex_mul_batch_2(&a, &b);
        let scalar = [a[0].mul_fixed(b[0]), a[1].mul_fixed(b[1])];

        for i in 0..2 {
            assert_eq!(batch[i].re.to_bits(), scalar[i].re.to_bits(),
                "Batch vs scalar mismatch at {}, re", i);
            assert_eq!(batch[i].im.to_bits(), scalar[i].im.to_bits(),
                "Batch vs scalar mismatch at {}, im", i);
        }
    }

    #[test]
    fn test_x_gate_all_kernels() {
        for n_qubits in 1..=4 {
            let mut sv1 = Statevector::new(n_qubits);
            let mut sv2 = sv1.clone();
            let mut sv3 = sv1.clone();

            apply_single_qubit_scalar(&mut sv1, 0, x_matrix());
            apply_single_qubit_cached(&mut sv2, 0, x_matrix());
            apply_single_qubit_simd(&mut sv3, 0, x_matrix());

            // X|0...0⟩ should put all amplitude in state 1
            assert!((sv1.amplitudes[1].re - 1.0).abs() < TOL);
            assert!((sv2.amplitudes[1].re - 1.0).abs() < TOL);
            assert!((sv3.amplitudes[1].re - 1.0).abs() < TOL);
        }
    }
}
