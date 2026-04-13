use std::fmt;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors produced by the CJC runtime.
///
/// All runtime operations that can fail return `Result<T, RuntimeError>`.
/// This enum covers the four main failure modes: index violations, shape
/// incompatibilities, dimension mismatches, and general invalid operations.
///
/// [`RuntimeError`] implements [`std::error::Error`] and [`Display`] for
/// integration with Rust's standard error handling.
#[derive(Debug, Clone)]
pub enum RuntimeError {
    /// An index exceeded the valid range for a buffer or tensor dimension.
    ///
    /// `index` is the out-of-bounds index that was provided, and `length`
    /// is the size of the dimension or buffer that was indexed into.
    IndexOutOfBounds {
        /// The invalid index that was provided.
        index: usize,
        /// The valid range is `0..length`.
        length: usize,
    },
    /// The total number of elements did not match what the shape requires.
    ///
    /// Raised by [`Tensor::from_vec`] when `data.len() != product(shape)`,
    /// and by [`Tensor::reshape`] when the new shape's element count differs.
    ShapeMismatch {
        /// The element count implied by the target shape.
        expected: usize,
        /// The actual element count provided.
        got: usize,
    },
    /// The number of dimensions (rank) did not match what was expected.
    ///
    /// Raised by operations like [`Tensor::matmul`] (requires 2-D) or
    /// [`Tensor::get`] (index length must match ndim).
    DimensionMismatch {
        /// The required number of dimensions.
        expected: usize,
        /// The actual number of dimensions provided.
        got: usize,
    },
    /// A catch-all for operations that are invalid for the given arguments.
    ///
    /// The contained `String` provides a human-readable description of
    /// what went wrong (e.g., "matmul requires 2-D tensors").
    InvalidOperation(String),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeError::IndexOutOfBounds { index, length } => {
                write!(f, "index {index} out of bounds for length {length}")
            }
            RuntimeError::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected} elements, got {got}")
            }
            RuntimeError::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "dimension mismatch: expected {expected} dimensions, got {got}"
                )
            }
            RuntimeError::InvalidOperation(msg) => {
                write!(f, "invalid operation: {msg}")
            }
        }
    }
}

impl std::error::Error for RuntimeError {}

