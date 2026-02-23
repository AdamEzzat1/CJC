use std::fmt;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors produced by the CJC runtime.
#[derive(Debug, Clone)]
pub enum RuntimeError {
    IndexOutOfBounds {
        index: usize,
        length: usize,
    },
    ShapeMismatch {
        expected: usize,
        got: usize,
    },
    DimensionMismatch {
        expected: usize,
        got: usize,
    },
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

