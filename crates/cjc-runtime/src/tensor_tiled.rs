//! Tensor Tiling — L2-friendly tiled matrix multiplication.
//!
//! Provides a tiled matmul implementation that operates on tiles that fit
//! within the L2 cache, improving locality for large matrices.
//!
//! # Determinism
//!
//! - Tile iteration order is deterministic (row-major over tiles).
//! - The summation within each tile uses the same accumulation order.
//! - Same inputs → bit-identical outputs on the same platform.
//!
//! # Tile Size
//!
//! Default tile size is 64×64 (32 KB per tile at f64, fits in most L2 caches).
//! Configurable via `TiledMatmul::with_tile_size()`.

use crate::tensor_simd;

/// Default tile dimension. 64×64 doubles = 32 KB per tile.
const DEFAULT_TILE_SIZE: usize = 64;

/// Tiled matrix multiplication engine.
pub struct TiledMatmul {
    /// Tile dimension (square tiles).
    pub tile_size: usize,
}

impl TiledMatmul {
    /// Create with default tile size (64).
    pub fn new() -> Self {
        TiledMatmul {
            tile_size: DEFAULT_TILE_SIZE,
        }
    }

    /// Create with a custom tile size.
    pub fn with_tile_size(tile_size: usize) -> Self {
        let ts = if tile_size == 0 { DEFAULT_TILE_SIZE } else { tile_size };
        TiledMatmul { tile_size: ts }
    }

    /// Compute C = A × B using tiled iteration.
    ///
    /// - `a`: row-major matrix [m × k]
    /// - `b`: row-major matrix [k × n]
    /// - Returns: row-major matrix [m × n]
    ///
    /// Panics if inner dimensions don't match.
    pub fn matmul(
        &self,
        a: &[f64],
        m: usize,
        k: usize,
        b: &[f64],
        n: usize,
    ) -> Vec<f64> {
        assert_eq!(a.len(), m * k, "a dimensions mismatch");
        assert_eq!(b.len(), k * n, "b dimensions mismatch");

        let mut c = vec![0.0f64; m * n];
        let ts = self.tile_size;

        // Tile over all three dimensions: i, j, p (deterministic order).
        let mut ii = 0;
        while ii < m {
            let i_end = (ii + ts).min(m);
            let mut jj = 0;
            while jj < n {
                let j_end = (jj + ts).min(n);
                let mut pp = 0;
                while pp < k {
                    let p_end = (pp + ts).min(k);

                    // Micro-kernel: accumulate tile contribution.
                    // Uses SIMD-accelerated AXPY for the inner j-loop
                    // (4-wide AVX2 when available, scalar fallback otherwise).
                    let j_len = j_end - jj;
                    for i in ii..i_end {
                        for p in pp..p_end {
                            let a_ip = a[i * k + p];
                            let c_slice = &mut c[i * n + jj .. i * n + j_end];
                            let b_slice = &b[p * n + jj .. p * n + j_end];
                            tensor_simd::simd_axpy(c_slice, b_slice, a_ip, j_len);
                        }
                    }

                    pp += ts;
                }
                jj += ts;
            }
            ii += ts;
        }

        c
    }

    /// Compute C = A × B^T using tiled iteration (useful when B is stored
    /// in row-major but you need A × B^T).
    ///
    /// - `a`: row-major matrix [m × k]
    /// - `b`: row-major matrix [n × k] (transposed: each row of b is a column of B)
    /// - Returns: row-major matrix [m × n]
    pub fn matmul_transposed_b(
        &self,
        a: &[f64],
        m: usize,
        k: usize,
        b: &[f64],
        n: usize,
    ) -> Vec<f64> {
        assert_eq!(a.len(), m * k, "a dimensions mismatch");
        assert_eq!(b.len(), n * k, "b dimensions mismatch (n × k expected)");

        let mut c = vec![0.0f64; m * n];
        let ts = self.tile_size;

        let mut ii = 0;
        while ii < m {
            let i_end = (ii + ts).min(m);
            let mut jj = 0;
            while jj < n {
                let j_end = (jj + ts).min(n);

                for i in ii..i_end {
                    for j in jj..j_end {
                        let mut sum = 0.0f64;
                        for p in 0..k {
                            sum += a[i * k + p] * b[j * k + p];
                        }
                        c[i * n + j] = sum;
                    }
                }

                jj += ts;
            }
            ii += ts;
        }

        c
    }
}

impl Default for TiledMatmul {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiled_matmul_2x2() {
        let engine = TiledMatmul::new();
        // [1 2] × [5 6] = [19 22]
        // [3 4]   [7 8]   [43 50]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = engine.matmul(&a, 2, 2, &b, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_tiled_matmul_nonsquare() {
        let engine = TiledMatmul::new();
        // [2 3] × [1 0] = [2+12 0+15]   = [14 15]
        //         [4 5]
        let a = vec![2.0, 3.0];
        let b = vec![1.0, 0.0, 4.0, 5.0];
        let c = engine.matmul(&a, 1, 2, &b, 2);
        assert_eq!(c, vec![14.0, 15.0]);
    }

    #[test]
    fn test_tiled_matmul_identity() {
        let engine = TiledMatmul::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let eye = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let c = engine.matmul(&a, 3, 3, &eye, 3);
        assert_eq!(c, a);
    }

    #[test]
    fn test_tiled_with_small_tile() {
        // Use tile_size=2 to force tiling on a 4×4 matrix.
        let engine = TiledMatmul::with_tile_size(2);
        let a = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let b = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let c = engine.matmul(&a, 4, 4, &b, 4);
        assert_eq!(c, a, "A × I = A with tiling");
    }

    #[test]
    fn test_tiled_deterministic() {
        let e1 = TiledMatmul::with_tile_size(3);
        let e2 = TiledMatmul::with_tile_size(3);

        let a: Vec<f64> = (0..25).map(|i| i as f64 * 0.1).collect();
        let b: Vec<f64> = (0..25).map(|i| (25 - i) as f64 * 0.1).collect();

        let c1 = e1.matmul(&a, 5, 5, &b, 5);
        let c2 = e2.matmul(&a, 5, 5, &b, 5);

        assert_eq!(c1, c2, "deterministic tiled matmul");
    }

    #[test]
    fn test_tiled_matches_naive() {
        let engine = TiledMatmul::with_tile_size(2);
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let tiled = engine.matmul(&a, 2, 3, &b, 2);

        // Naive computation.
        let expected = naive_matmul(&a, 2, 3, &b, 2);

        for (i, (t, e)) in tiled.iter().zip(expected.iter()).enumerate() {
            assert!(
                (t - e).abs() < 1e-12,
                "mismatch at index {i}: tiled={t}, naive={e}"
            );
        }
    }

    #[test]
    fn test_transposed_b_matmul() {
        let engine = TiledMatmul::new();
        // A = [1 2]   B^T stored as [5 7] (row 0 of B^T = col 0 of B)
        //     [3 4]                  [6 8] (row 1 of B^T = col 1 of B)
        // A × B = A × (B^T)^T
        // where B^T is [5 7; 6 8], so B = [5 6; 7 8]
        // A × B = [1*5+2*7  1*6+2*8] = [19 22]
        //         [3*5+4*7  3*6+4*8]   [43 50]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let bt = vec![5.0, 7.0, 6.0, 8.0]; // B transposed, stored [n × k]
        let c = engine.matmul_transposed_b(&a, 2, 2, &bt, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_large_tiled_correctness() {
        // 32×32 matrix multiplication with tile_size=8.
        let engine = TiledMatmul::with_tile_size(8);
        let n = 32;
        let a: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.01).collect();
        let b: Vec<f64> = (0..n * n).map(|i| ((n * n - i) as f64) * 0.01).collect();

        let tiled = engine.matmul(&a, n, n, &b, n);
        let naive = naive_matmul(&a, n, n, &b, n);

        for (i, (t, e)) in tiled.iter().zip(naive.iter()).enumerate() {
            assert!(
                (t - e).abs() < 1e-8,
                "mismatch at [{}, {}]: tiled={t}, naive={e}",
                i / n,
                i % n
            );
        }
    }

    /// Naive O(n³) matmul for verification.
    fn naive_matmul(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
        let mut c = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }
}
