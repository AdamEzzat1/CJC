//! Quantile codebook for the prefix encoder.
//!
//! A [`QuantileCodebook`] holds, for each input dimension, a sorted slice
//! of `n_bins - 1` boundary values. Encoding `x: f64` to a u8 bin works
//! by binary-searching the boundaries:
//!
//! ```text
//!   bin = boundaries.binary_search(&x).unwrap_or_else(|i| i) as u8
//! ```
//!
//! The codebook is **frozen at install time**: once a graph stores a
//! codebook, subsequent attempts to install a different one error.
//! This freezes the deterministic mapping from inputs to prefix bytes —
//! a critical property for hash-chain stability.
//!
//! ## Determinism
//!
//! * Boundary values are stored as `f64` with their `to_bits()` IEEE-754
//!   pattern preserved in canonical encoding.
//! * Binary search uses plain `<` comparisons; no FMA.
//! * `n_bins` is a power of two in {2, 4, 8, 16, 32, 64, 128, 256}, so the
//!   bin index always fits in a u8 (capped at `n_bins - 1`).

/// Errors returned by codebook operations.
#[derive(Debug, PartialEq)]
pub enum CodebookError {
    /// `n_bins` is not a power of two between 2 and 256 inclusive.
    InvalidNumBins(u16),
    /// Boundaries for at least one dimension were not strictly ascending.
    /// Carries the offending dimension index.
    BoundariesNotSorted { dim: u8 },
    /// The boundaries 2-D shape is `[n_dims, n_bins - 1]` but the supplied
    /// flat data length doesn't match.
    ShapeMismatch {
        expected_n_dims: u8,
        expected_per_dim: u16,
        got_len: usize,
    },
    /// `n_dims` overflowed `u8` (max 255 input dimensions in Phase 0.2).
    TooManyDims(usize),
    /// Encode call's input vector length didn't match the codebook's
    /// `n_dims`.
    InputArityMismatch { expected: u8, got: usize },
}

impl std::fmt::Display for CodebookError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodebookError::InvalidNumBins(n) => write!(
                f,
                "abng codebook: n_bins must be a power of two in [2, 256], got {n}"
            ),
            CodebookError::BoundariesNotSorted { dim } => write!(
                f,
                "abng codebook: boundaries for dim {dim} are not strictly ascending"
            ),
            CodebookError::ShapeMismatch {
                expected_n_dims,
                expected_per_dim,
                got_len,
            } => write!(
                f,
                "abng codebook: expected {expected_n_dims}×{expected_per_dim} = {} boundaries, got {got_len}",
                (*expected_n_dims as usize) * (*expected_per_dim as usize)
            ),
            CodebookError::TooManyDims(n) => write!(
                f,
                "abng codebook: {n} dims exceeds the 255-dim cap"
            ),
            CodebookError::InputArityMismatch { expected, got } => write!(
                f,
                "abng codebook: encode expected {expected}-element input, got {got}"
            ),
        }
    }
}

/// Quantile codebook used by the prefix encoder.
#[derive(Debug, Clone)]
pub struct QuantileCodebook {
    /// `n_dims` quantile boundary slices, each of length `n_bins - 1`.
    pub bins: Vec<Vec<f64>>,
    /// Number of input dimensions. Capped at 255.
    pub n_dims: u8,
    /// Number of bins per dimension. Always a power of two in [2, 256].
    pub n_bins: u16,
    /// SHA-256 of the canonical-byte encoding of this codebook. Frozen at
    /// install time and embedded in the snapshot header so that any
    /// downstream replay catches a codebook substitution.
    pub frozen_hash: [u8; 32],
}

impl QuantileCodebook {
    /// Construct a codebook from per-dim flattened boundaries.
    ///
    /// `flat` must be `n_dims × (n_bins - 1)` `f64` values, laid out
    /// row-major: dim 0's boundaries first, then dim 1's, etc.
    pub fn from_flat(n_dims: usize, n_bins: u16, flat: &[f64]) -> Result<Self, CodebookError> {
        if !is_power_of_two_in_range(n_bins) {
            return Err(CodebookError::InvalidNumBins(n_bins));
        }
        if n_dims > u8::MAX as usize {
            return Err(CodebookError::TooManyDims(n_dims));
        }
        let per_dim = (n_bins - 1) as usize;
        let expected_len = n_dims * per_dim;
        if flat.len() != expected_len {
            return Err(CodebookError::ShapeMismatch {
                expected_n_dims: n_dims as u8,
                expected_per_dim: n_bins - 1,
                got_len: flat.len(),
            });
        }
        let mut bins = Vec::with_capacity(n_dims);
        for d in 0..n_dims {
            let row = &flat[d * per_dim..(d + 1) * per_dim];
            // Validate strictly ascending; equal neighbors are rejected so
            // the binary-search bin assignment is unambiguous.
            for w in row.windows(2) {
                if !(w[0] < w[1]) {
                    return Err(CodebookError::BoundariesNotSorted { dim: d as u8 });
                }
            }
            bins.push(row.to_vec());
        }
        let mut cb = Self {
            bins,
            n_dims: n_dims as u8,
            n_bins,
            frozen_hash: [0u8; 32],
        };
        cb.frozen_hash = cjc_snap::hash::sha256(&cb.canonical_bytes());
        Ok(cb)
    }

    /// Encode an input vector to a prefix of u8 bin indices.
    pub fn encode(&self, x: &[f64]) -> Result<Vec<u8>, CodebookError> {
        if x.len() != self.n_dims as usize {
            return Err(CodebookError::InputArityMismatch {
                expected: self.n_dims,
                got: x.len(),
            });
        }
        let mut out = Vec::with_capacity(self.n_dims as usize);
        let max_bin = (self.n_bins - 1) as usize; // cap
        for (d, &v) in x.iter().enumerate() {
            let row = &self.bins[d];
            // Binary search: find the index of the first boundary > v.
            // partition_point returns that; clamp to max_bin to handle
            // any rounding edge case (it can only equal row.len() == n_bins-1
            // anyway, which is valid).
            let bin = row.partition_point(|&b| b < v).min(max_bin);
            out.push(bin as u8);
        }
        Ok(out)
    }

    /// Canonical big-endian byte encoding for hashing.
    ///
    /// Layout:
    /// ```text
    ///   n_dims  u8           (1)
    ///   n_bins  u16 BE       (2)
    ///   for each dim d in 0..n_dims:
    ///     n_boundaries u16 BE (2)        always n_bins - 1
    ///     for each boundary:
    ///       value.to_bits  u64 BE  (8)
    /// ```
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let per_dim = self.n_bins.saturating_sub(1);
        let mut out =
            Vec::with_capacity(1 + 2 + (self.n_dims as usize) * (2 + (per_dim as usize) * 8));
        out.push(self.n_dims);
        out.extend_from_slice(&self.n_bins.to_be_bytes());
        for row in &self.bins {
            out.extend_from_slice(&(row.len() as u16).to_be_bytes());
            for &b in row {
                out.extend_from_slice(&b.to_bits().to_be_bytes());
            }
        }
        out
    }
}

fn is_power_of_two_in_range(n: u16) -> bool {
    matches!(n, 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_uniform(n_dims: usize, n_bins: u16) -> Vec<f64> {
        // Uniformly spaced boundaries 0.5, 1.5, ..., n_bins-1.5 per dim
        let mut out = Vec::new();
        for _ in 0..n_dims {
            for k in 1..n_bins {
                out.push(k as f64 - 0.5);
            }
        }
        out
    }

    #[test]
    fn rejects_bad_n_bins() {
        let err = QuantileCodebook::from_flat(2, 6, &[]).unwrap_err();
        assert!(matches!(err, CodebookError::InvalidNumBins(6)));
    }

    #[test]
    fn rejects_unsorted_boundaries() {
        // dim 0 boundaries: [3.0, 1.0, 2.0] — not ascending
        let flat = vec![3.0, 1.0, 2.0];
        let err = QuantileCodebook::from_flat(1, 4, &flat).unwrap_err();
        assert!(matches!(err, CodebookError::BoundariesNotSorted { dim: 0 }));
    }

    #[test]
    fn rejects_shape_mismatch() {
        let err = QuantileCodebook::from_flat(2, 4, &[0.0, 1.0]).unwrap_err();
        assert!(matches!(err, CodebookError::ShapeMismatch { .. }));
    }

    #[test]
    fn encodes_simple_uniform_buckets() {
        let flat = flat_uniform(2, 4); // boundaries [0.5, 1.5, 2.5] per dim
        let cb = QuantileCodebook::from_flat(2, 4, &flat).unwrap();
        // Values: 0.0 → bin 0, 1.0 → bin 1, 2.0 → bin 2, 100.0 → bin 3 (max).
        assert_eq!(cb.encode(&[0.0, 0.0]).unwrap(), vec![0, 0]);
        assert_eq!(cb.encode(&[1.0, 2.0]).unwrap(), vec![1, 2]);
        assert_eq!(cb.encode(&[100.0, -100.0]).unwrap(), vec![3, 0]);
    }

    #[test]
    fn boundary_value_lands_in_upper_bin() {
        // Boundaries are exclusive on the lower side: value == boundary
        // belongs to the bin *above* it.
        let flat = vec![1.0, 2.0, 3.0]; // 4 bins, 3 boundaries
        let cb = QuantileCodebook::from_flat(1, 4, &flat).unwrap();
        // partition_point(|b| b < v) — for v=1.0, no boundary is < 1.0, so bin 0
        assert_eq!(cb.encode(&[1.0]).unwrap(), vec![0]);
        // for v=1.5: 1.0 < 1.5, so partition_point is 1 → bin 1
        assert_eq!(cb.encode(&[1.5]).unwrap(), vec![1]);
        // for v=2.0: 1.0 < 2.0, 2.0 ≮ 2.0, so partition_point is 1 → bin 1
        assert_eq!(cb.encode(&[2.0]).unwrap(), vec![1]);
        // for v=3.0: 1.0,2.0 < 3.0; 3.0 ≮ 3.0, so partition_point 2 → bin 2
        assert_eq!(cb.encode(&[3.0]).unwrap(), vec![2]);
    }

    #[test]
    fn arity_mismatch_errs() {
        let cb = QuantileCodebook::from_flat(2, 4, &flat_uniform(2, 4)).unwrap();
        let err = cb.encode(&[0.0]).unwrap_err();
        assert!(matches!(err, CodebookError::InputArityMismatch { expected: 2, got: 1 }));
    }

    #[test]
    fn canonical_bytes_size() {
        let cb = QuantileCodebook::from_flat(3, 16, &flat_uniform(3, 16)).unwrap();
        // 1 (n_dims) + 2 (n_bins) + 3 × (2 (n_boundaries) + 15 × 8 (values))
        //   = 3 + 3 × 122 = 369
        assert_eq!(cb.canonical_bytes().len(), 369);
    }

    #[test]
    fn frozen_hash_deterministic() {
        let a = QuantileCodebook::from_flat(2, 4, &flat_uniform(2, 4)).unwrap();
        let b = QuantileCodebook::from_flat(2, 4, &flat_uniform(2, 4)).unwrap();
        assert_eq!(a.frozen_hash, b.frozen_hash);
    }

    #[test]
    fn frozen_hash_changes_when_boundaries_change() {
        let a = QuantileCodebook::from_flat(1, 4, &[0.0, 1.0, 2.0]).unwrap();
        let b = QuantileCodebook::from_flat(1, 4, &[0.0, 1.0, 3.0]).unwrap();
        assert_ne!(a.frozen_hash, b.frozen_hash);
    }
}
