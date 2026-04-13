//! FFT — Cooley-Tukey radix-2 FFT, inverse FFT, real FFT, PSD.
//!
//! # Determinism Contract
//! - Bit-reversal permutation is deterministic.
//! - Butterfly operations in fixed order.
//! - Zero-padding to next power of 2 is deterministic.

use std::f64::consts::PI;

/// Compute the Cooley-Tukey radix-2 FFT in-place.
///
/// # Arguments
///
/// * `data` - Input signal as `(re, im)` pairs. Length **must** be a power of 2.
///
/// # Returns
///
/// Frequency-domain representation as `Vec<(f64, f64)>` of `(re, im)` pairs.
///
/// # Panics
///
/// Panics if `data.len()` is not a power of 2.
///
/// # Algorithm
///
/// Iterative radix-2 decimation-in-time with bit-reversal permutation followed
/// by butterfly stages. All twiddle factors are computed in fixed order.
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

/// Compute the inverse FFT by conjugating, applying [`fft`], then conjugating
/// and scaling by `1/N`.
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

/// Compute the FFT of a real-valued signal.
///
/// Automatically zero-pads to the next power of 2 if `data.len()` is not
/// already a power of 2.
pub fn rfft(data: &[f64]) -> Vec<(f64, f64)> {
    let n = next_power_of_2(data.len());
    let mut complex_data: Vec<(f64, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        let val = if i < data.len() { data[i] } else { 0.0 };
        complex_data.push((val, 0.0));
    }
    fft(&complex_data)
}

/// Compute the power spectral density: `|FFT(x)|^2` for each frequency bin.
pub fn psd(data: &[f64]) -> Vec<f64> {
    let spectrum = rfft(data);
    spectrum.iter().map(|&(r, i)| r * r + i * i).collect()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Reverse the lowest `bits` bits of `x`.
fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// Multiply two complex numbers represented as `(re, im)` tuples.
fn complex_mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// Return the smallest power of 2 that is >= `n`.
fn next_power_of_2(n: usize) -> usize {
    let mut p = 1;
    while p < n { p <<= 1; }
    p
}

// ---------------------------------------------------------------------------
// Phase B6: Window functions
// ---------------------------------------------------------------------------

/// Hann window: w[k] = 0.5 * (1 - cos(2*pi*k / (N-1))).
/// For N=1, returns [1.0].
pub fn hann_window(n: usize) -> Vec<f64> {
    if n <= 1 { return vec![1.0; n]; }
    (0..n).map(|k| 0.5 * (1.0 - (2.0 * PI * k as f64 / (n - 1) as f64).cos())).collect()
}

/// Hamming window: w[k] = 0.54 - 0.46 * cos(2*pi*k / (N-1)).
/// For N=1, returns [1.0].
pub fn hamming_window(n: usize) -> Vec<f64> {
    if n <= 1 { return vec![1.0; n]; }
    (0..n).map(|k| 0.54 - 0.46 * (2.0 * PI * k as f64 / (n - 1) as f64).cos()).collect()
}

/// Blackman window: w[k] = 0.42 - 0.5*cos(2*pi*k/(N-1)) + 0.08*cos(4*pi*k/(N-1)).
/// For N=1, returns [1.0].
pub fn blackman_window(n: usize) -> Vec<f64> {
    if n <= 1 { return vec![1.0; n]; }
    (0..n).map(|k| {
        let frac = k as f64 / (n - 1) as f64;
        0.42 - 0.5 * (2.0 * PI * frac).cos() + 0.08 * (4.0 * PI * frac).cos()
    }).collect()
}

// ---------------------------------------------------------------------------
// Phase B6: Arbitrary-length FFT (Bluestein's algorithm)
// ---------------------------------------------------------------------------

/// Compute the FFT for an arbitrary-length signal using Bluestein's chirp-z
/// algorithm.
///
/// Delegates to [`fft`] when the input length is already a power of 2.
/// For non-power-of-2 lengths, the signal is convolved with a chirp sequence
/// via zero-padded radix-2 FFTs.
pub fn fft_arbitrary(data: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = data.len();
    if n == 0 { return vec![]; }
    if n == 1 { return data.to_vec(); }

    // If already power of 2, delegate to radix-2
    if n.is_power_of_two() {
        return fft(data);
    }

    // Chirp sequence: w[k] = exp(-i * pi * k^2 / N)
    let chirp: Vec<(f64, f64)> = (0..n).map(|k| {
        let angle = -PI * (k * k) as f64 / n as f64;
        (angle.cos(), angle.sin())
    }).collect();

    // Multiply input by chirp: a[k] = x[k] * chirp[k]
    let a: Vec<(f64, f64)> = data.iter().zip(chirp.iter()).map(|(&x, &w)| complex_mul(x, w)).collect();

    // Convolution sequence: b[k] = conj(chirp[k])
    // We need b extended for circular convolution
    let m = next_power_of_2(2 * n - 1);

    // Zero-pad a to length m
    let mut a_padded = vec![(0.0, 0.0); m];
    for (i, &v) in a.iter().enumerate() {
        a_padded[i] = v;
    }

    // Build b_padded: b[0..n] = conj(chirp[0..n]), b[m-n+1..m] = conj(chirp[n-1..1])
    let mut b_padded = vec![(0.0, 0.0); m];
    for i in 0..n {
        b_padded[i] = (chirp[i].0, -chirp[i].1); // conj
    }
    for i in 1..n {
        b_padded[m - i] = (chirp[i].0, -chirp[i].1); // conj
    }

    // Convolve via FFT
    let a_fft = fft(&a_padded);
    let b_fft = fft(&b_padded);
    let c_fft: Vec<(f64, f64)> = a_fft.iter().zip(b_fft.iter()).map(|(&a, &b)| complex_mul(a, b)).collect();
    let c = ifft(&c_fft);

    // Extract and multiply by chirp
    (0..n).map(|k| complex_mul(chirp[k], c[k])).collect()
}

// ---------------------------------------------------------------------------
// Phase B6: 2D FFT
// ---------------------------------------------------------------------------

/// Compute the 2-D FFT by applying 1-D [`fft`] along rows then along columns.
///
/// Both `rows` and `cols` must be powers of 2.
///
/// # Errors
///
/// Returns `Err` if `data.len() != rows * cols` or dimensions are not powers
/// of 2.
pub fn fft_2d(data: &[(f64, f64)], rows: usize, cols: usize) -> Result<Vec<(f64, f64)>, String> {
    if data.len() != rows * cols {
        return Err(format!("fft_2d: expected {} elements, got {}", rows * cols, data.len()));
    }
    if !rows.is_power_of_two() || !cols.is_power_of_two() {
        return Err("fft_2d: rows and cols must be powers of 2".into());
    }

    // FFT along rows
    let mut result = data.to_vec();
    for r in 0..rows {
        let row: Vec<(f64, f64)> = result[r * cols..(r + 1) * cols].to_vec();
        let fft_row = fft(&row);
        result[r * cols..(r + 1) * cols].copy_from_slice(&fft_row);
    }

    // FFT along columns
    for c in 0..cols {
        let col: Vec<(f64, f64)> = (0..rows).map(|r| result[r * cols + c]).collect();
        let fft_col = fft(&col);
        for r in 0..rows {
            result[r * cols + c] = fft_col[r];
        }
    }

    Ok(result)
}

/// Compute the 2-D inverse FFT by applying 1-D [`ifft`] along rows then columns.
pub fn ifft_2d(data: &[(f64, f64)], rows: usize, cols: usize) -> Result<Vec<(f64, f64)>, String> {
    if data.len() != rows * cols {
        return Err(format!("ifft_2d: expected {} elements, got {}", rows * cols, data.len()));
    }
    if !rows.is_power_of_two() || !cols.is_power_of_two() {
        return Err("ifft_2d: rows and cols must be powers of 2".into());
    }

    // IFFT along rows
    let mut result = data.to_vec();
    for r in 0..rows {
        let row: Vec<(f64, f64)> = result[r * cols..(r + 1) * cols].to_vec();
        let ifft_row = ifft(&row);
        result[r * cols..(r + 1) * cols].copy_from_slice(&ifft_row);
    }

    // IFFT along columns
    for c in 0..cols {
        let col: Vec<(f64, f64)> = (0..rows).map(|r| result[r * cols + c]).collect();
        let ifft_col = ifft(&col);
        for r in 0..rows {
            result[r * cols + c] = ifft_col[r];
        }
    }

    Ok(result)
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

    // --- B6: Window functions ---

    #[test]
    fn test_hann_endpoints() {
        let w = hann_window(8);
        assert!(w[0].abs() < 1e-12, "hann[0] = {}", w[0]);
        assert!(w[7].abs() < 1e-12, "hann[N-1] = {}", w[7]);
    }

    #[test]
    fn test_hann_midpoint() {
        let w = hann_window(9); // odd, so exact midpoint
        assert!((w[4] - 1.0).abs() < 1e-12, "hann[4] = {}", w[4]);
    }

    #[test]
    fn test_hann_symmetry() {
        let w = hann_window(16);
        for k in 0..8 {
            assert!((w[k] - w[15 - k]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_hamming_endpoints() {
        let w = hamming_window(8);
        assert!((w[0] - 0.08).abs() < 1e-12, "hamming[0] = {}", w[0]);
        assert!((w[7] - 0.08).abs() < 1e-12, "hamming[N-1] = {}", w[7]);
    }

    #[test]
    fn test_blackman_endpoints() {
        let w = blackman_window(16);
        assert!(w[0].abs() < 1e-12, "blackman[0] = {}", w[0]);
        assert!(w[15].abs() < 1e-12, "blackman[N-1] = {}", w[15]);
    }

    // --- B6: Arbitrary FFT ---

    #[test]
    fn test_fft_arbitrary_prime() {
        // 7-element signal, brute-force DFT
        let n = 7;
        let data: Vec<(f64, f64)> = (0..n).map(|k| ((k + 1) as f64, 0.0)).collect();
        let result = fft_arbitrary(&data);

        // Brute-force DFT for comparison
        for k in 0..n {
            let mut re = 0.0;
            let mut im = 0.0;
            for j in 0..n {
                let angle = -2.0 * PI * (k * j) as f64 / n as f64;
                re += data[j].0 * angle.cos() - data[j].1 * angle.sin();
                im += data[j].0 * angle.sin() + data[j].1 * angle.cos();
            }
            assert!((result[k].0 - re).abs() < 1e-8, "re[{k}]: got {} expected {re}", result[k].0);
            assert!((result[k].1 - im).abs() < 1e-8, "im[{k}]: got {} expected {im}", result[k].1);
        }
    }

    #[test]
    fn test_fft_arbitrary_matches_radix2() {
        let data: Vec<(f64, f64)> = vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)];
        let r_radix2 = fft(&data);
        let r_arb = fft_arbitrary(&data);
        for (a, b) in r_radix2.iter().zip(r_arb.iter()) {
            assert!((a.0 - b.0).abs() < 1e-10);
            assert!((a.1 - b.1).abs() < 1e-10);
        }
    }

    #[test]
    fn test_fft_arbitrary_parseval() {
        let data: Vec<(f64, f64)> = vec![(1.0, 0.0), (2.0, 1.0), (3.0, -1.0), (0.5, 0.5), (4.0, 0.0)];
        let n = data.len();
        let time_energy: f64 = data.iter().map(|&(r, i)| r * r + i * i).sum();
        let freq = fft_arbitrary(&data);
        let freq_energy: f64 = freq.iter().map(|&(r, i)| r * r + i * i).sum::<f64>() / n as f64;
        assert!((time_energy - freq_energy).abs() < 1e-8, "time={time_energy} freq={freq_energy}");
    }

    // --- B6: 2D FFT ---

    #[test]
    fn test_fft_2d_constant() {
        let data = vec![(1.0, 0.0); 4]; // 2x2 constant
        let result = fft_2d(&data, 2, 2).unwrap();
        // DC component should be N*M = 4
        assert!((result[0].0 - 4.0).abs() < 1e-10);
        for i in 1..4 {
            assert!(result[i].0.abs() < 1e-10);
            assert!(result[i].1.abs() < 1e-10);
        }
    }

    #[test]
    fn test_fft_2d_roundtrip() {
        let data: Vec<(f64, f64)> = vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]; // 2x2
        let freq = fft_2d(&data, 2, 2).unwrap();
        let recovered = ifft_2d(&freq, 2, 2).unwrap();
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!((orig.0 - rec.0).abs() < 1e-10);
            assert!((orig.1 - rec.1).abs() < 1e-10);
        }
    }

    #[test]
    fn test_b6_fft_determinism() {
        let data: Vec<(f64, f64)> = vec![(1.0, 2.0), (3.0, 0.0), (5.0, -1.0)];
        let r1 = fft_arbitrary(&data);
        let r2 = fft_arbitrary(&data);
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.0.to_bits(), b.0.to_bits());
            assert_eq!(a.1.to_bits(), b.1.to_bits());
        }
    }
}
