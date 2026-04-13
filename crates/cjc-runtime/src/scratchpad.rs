//! KV-cache scratchpad -- zero-allocation state persistence for transformer inference.
//!
//! Provides [`Scratchpad`], a pre-allocated linear buffer for appending
//! key/value token vectors without per-token heap allocation. The entire
//! `[max_seq_len, dim]` storage is allocated once at construction; subsequent
//! [`append`](Scratchpad::append) calls copy data into existing storage.
//!
//! # NoGC guarantee
//!
//! After construction, `append` performs no heap allocation -- it writes
//! directly into the pre-allocated [`Buffer`]. The [`as_tensor`](Scratchpad::as_tensor)
//! method returns a zero-copy view via `Rc` clone of the underlying buffer.
//!
//! # Relationship to [`PagedKvCache`](crate::paged_kv::PagedKvCache)
//!
//! `Scratchpad` uses a single contiguous buffer (simpler, better for small
//! sequences). [`PagedKvCache`](crate::paged_kv::PagedKvCache) uses block
//! paging (better for large sequences where contiguous allocation may
//! fragment).

use std::fmt;

use crate::buffer::Buffer;
use crate::error::RuntimeError;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// 2b. KV-Cache Scratchpad (Zero-Allocation State Persistence)
// ---------------------------------------------------------------------------

/// A pre-allocated scratch buffer for KV-cache. Allows appending new
/// key/value vectors without re-allocation, up to a fixed `max_seq_len`.
///
/// Layout: `[max_seq_len, dim]` with a `current_len` cursor.
/// All memory is allocated once at construction; `append` only copies
/// new data into existing storage (zero GC pressure per token).
#[derive(Debug, Clone)]
pub struct Scratchpad {
    /// Underlying tensor of shape `[max_seq_len, dim]`.
    buffer: Buffer<f64>,
    /// Maximum sequence length (pre-allocated).
    max_seq_len: usize,
    /// Hidden dimension per token.
    dim: usize,
    /// Current number of tokens stored.
    current_len: usize,
}

impl Scratchpad {
    /// Create a new scratchpad pre-allocated for `max_seq_len` tokens of
    /// dimension `dim`. Zero-fills all storage upfront.
    pub fn new(max_seq_len: usize, dim: usize) -> Self {
        Scratchpad {
            buffer: Buffer::alloc(max_seq_len * dim, 0.0),
            max_seq_len,
            dim,
            current_len: 0,
        }
    }

    /// Number of tokens currently stored.
    pub fn len(&self) -> usize {
        self.current_len
    }

    /// Whether no tokens are stored.
    pub fn is_empty(&self) -> bool {
        self.current_len == 0
    }

    /// Maximum sequence length this scratchpad can hold.
    pub fn capacity(&self) -> usize {
        self.max_seq_len
    }

    /// Hidden dimension per token.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Append a single token vector `[dim]` to the cache.
    /// Returns an error if the cache is full. **Zero allocation.**
    pub fn append(&mut self, token_vec: &[f64]) -> Result<(), RuntimeError> {
        if token_vec.len() != self.dim {
            return Err(RuntimeError::ShapeMismatch {
                expected: self.dim,
                got: token_vec.len(),
            });
        }
        if self.current_len >= self.max_seq_len {
            return Err(RuntimeError::InvalidOperation(
                format!(
                    "Scratchpad full: {} / {} tokens",
                    self.current_len, self.max_seq_len
                ),
            ));
        }
        let base = self.current_len * self.dim;
        self.buffer.make_unique();
        for (i, &val) in token_vec.iter().enumerate() {
            self.buffer.set(base + i, val)?;
        }
        self.current_len += 1;
        Ok(())
    }

    /// Append a batch of token vectors from a tensor of shape `[n, dim]`.
    /// **Zero allocation** — writes directly into pre-allocated storage.
    pub fn append_tensor(&mut self, t: &Tensor) -> Result<(), RuntimeError> {
        if t.ndim() != 2 || t.shape()[1] != self.dim {
            return Err(RuntimeError::InvalidOperation(
                format!(
                    "append_tensor: expected shape [n, {}], got {:?}",
                    self.dim,
                    t.shape()
                ),
            ));
        }
        let n = t.shape()[0];
        if self.current_len + n > self.max_seq_len {
            return Err(RuntimeError::InvalidOperation(
                format!(
                    "Scratchpad overflow: {} + {} > {} max",
                    self.current_len, n, self.max_seq_len
                ),
            ));
        }
        let data = t.to_vec();
        self.buffer.make_unique();
        let base = self.current_len * self.dim;
        for (i, &val) in data.iter().enumerate() {
            self.buffer.set(base + i, val)?;
        }
        self.current_len += n;
        Ok(())
    }

    /// Get a Tensor view `[current_len, dim]` of the stored data.
    /// Shares the underlying buffer (zero-copy).
    pub fn as_tensor(&self) -> Tensor {
        let shape = vec![self.current_len, self.dim];
        Tensor {
            buffer: self.buffer.clone(), // Rc clone, not data copy
            shape: shape.clone(),
            strides: Tensor::compute_strides(&shape),
            offset: 0,
        }
    }

    /// Reset the cache to empty without deallocating.
    /// The underlying buffer is retained for reuse.
    pub fn clear(&mut self) {
        self.current_len = 0;
    }
}

impl fmt::Display for Scratchpad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Scratchpad(len={}, capacity={}, dim={})",
            self.current_len, self.max_seq_len, self.dim
        )
    }
}

