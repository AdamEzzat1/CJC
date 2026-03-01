//! FFT — Cooley-Tukey radix-2 FFT, inverse FFT, real FFT, PSD.
//!
//! # Determinism Contract
//! - Bit-reversal permutation is deterministic.
//! - Butterfly operations in fixed order.
//! - Zero-padding to next power of 2 is deterministic.

use std::f64::consts::PI;

/// Cooley-Tukey radix-2 FFT. Input length must be power of 2.
/// Returns Vec of (re, im) pairs.
pub fn fft(data: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = data.len();
    assert!(n.is_power_of_two(), "FFT: input length must be power of 2, got {n}");

    let mut buf = data.to_vec();

    // Bit-reversal permutation
    let bits = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = bit_reverse(i, bits);
        if i < j {
            buf.swap(i, j);
        }
    }

    // Butterfly stages
    let mut size = 2;
    while size <= n {
        let half = size / 2;
        let angle = -2.0 * PI / size as f64;
        let w_base = (angle.cos(), angle.sin());
        for start in (0..n).step_by(size) {
            let mut w = (1.0, 0.0);
            for k in 0..half {
                let i = start + k;
                let j = start + k + half;
                let t = complex_mul(w, buf[j]);
                buf[j] = (buf[i].0 - t.0, buf[i].1 - t.1);
                buf[i] = (buf[i].0 + t.0, buf[i].1 + t.1);
                w = complex_mul(w, w_base);
            }
        }
        size *= 2;
    }
    buf
}

/// Inverse FFT.
pub fn ifft(data: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = data.len();
    // Conjugate input
    let conjugated: Vec<(f64, f64)> = data.iter().map(|&(r, i)| (r, -i)).collect();
    // Forward FFT
    let mut result = fft(&conjugated);
    // Conjugate and scale
    let scale = 1.0 / n as f64;
    for v in &mut result {
        v.0 *= scale;
        v.1 = -v.1 * scale;
    }
    result
}

/// Real-valued FFT: input is real, output is complex.
/// Zero-pads to next power of 2 if needed.
pub fn rfft(data: &[f64]) -> Vec<(f64, f64)> {
    let n = next_power_of_2(data.len());
    let mut complex_data: Vec<(f64, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        let val = if i < data.len() { data[i] } else { 0.0 };
        complex_data.push((val, 0.0));
    }
    fft(&complex_data)
}

/// Power spectral density: |FFT(x)|^2.
pub fn psd(data: &[f64]) -> Vec<f64> {
    let spectrum = rfft(data);
    spectrum.iter().map(|&(r, i)| r * r + i * i).collect()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

fn complex_mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn next_power_of_2(n: usize) -> usize {
    let mut p = 1;
    while p < n { p <<= 1; }
    p
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_constant() {
        // FFT of [1,1,1,1] = [4, 0, 0, 0]
        let data = vec![(1.0, 0.0); 4];
        let result = fft(&data);
        assert!((result[0].0 - 4.0).abs() < 1e-12);
        for i in 1..4 {
            assert!(result[i].0.abs() < 1e-12);
            assert!(result[i].1.abs() < 1e-12);
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        let data = vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)];
        let spectrum = fft(&data);
        let recovered = ifft(&spectrum);
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!((orig.0 - rec.0).abs() < 1e-10);
            assert!((orig.1 - rec.1).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rfft_basic() {
        let data = [1.0, 0.0, 0.0, 0.0];
        let result = rfft(&data);
        // All ones for impulse
        for &(re, im) in &result {
            assert!((re - 1.0).abs() < 1e-12);
            assert!(im.abs() < 1e-12);
        }
    }

    #[test]
    fn test_psd_basic() {
        let data = [1.0, 0.0, 0.0, 0.0];
        let power = psd(&data);
        // All 1.0 for impulse
        for &p in &power {
            assert!((p - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_determinism() {
        let data = vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)];
        let r1 = fft(&data);
        let r2 = fft(&data);
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.0.to_bits(), b.0.to_bits());
            assert_eq!(a.1.to_bits(), b.1.to_bits());
        }
    }
}
