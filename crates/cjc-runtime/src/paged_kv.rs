//! Block-paged KV-cache -- vLLM-style block paging for transformer inference.
//!
//! Instead of a single contiguous pre-allocated tensor (which may fragment
//! or require reallocation), this module manages KV-cache memory in
//! fixed-size 16-token blocks via a logical-to-physical block table.
//!
//! # Benefits
//!
//! - No single large allocation -- blocks are page-sized (16 tokens each).
//! - Zero reallocation on append -- new blocks are pre-allocated at construction.
//! - Logical-to-physical mapping via block table enables flexible memory reuse.
//! - Each block is independently cache-line friendly.
//!
//! # NoGC guarantee
//!
//! All blocks are pre-allocated at construction time. [`PagedKvCache::append`]
//! performs zero heap allocations -- it only copies token data into existing
//! block storage.

use std::fmt;

use crate::error::RuntimeError;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// 2e. BlockPaged KV-Cache — vLLM-style block paging
// ---------------------------------------------------------------------------

/// Number of tokens stored per block in the paged KV-cache.
const BLOCK_TOKEN_COUNT: usize = 16;

/// A single page/block in the KV-cache.
///
/// Pre-allocated at construction with capacity for [`BLOCK_TOKEN_COUNT`]
/// tokens. Data is stored as a flat `Vec<f64>` of shape
/// `[BLOCK_TOKEN_COUNT, dim]`, with a `used` cursor tracking how many
/// token slots have been written.
#[derive(Debug, Clone)]
pub struct KvBlock {
    /// Data storage: [BLOCK_TOKEN_COUNT, dim]. Pre-allocated and zeroed.
    data: Vec<f64>,
    /// Hidden dimension per token.
    dim: usize,
    /// Number of tokens currently written in this block (0..=BLOCK_TOKEN_COUNT).
    used: usize,
}

impl KvBlock {
    /// Create a new zeroed block for tokens of the given hidden dimension.
    fn new(dim: usize) -> Self {
        KvBlock {
            data: vec![0.0; BLOCK_TOKEN_COUNT * dim],
            dim,
            used: 0,
        }
    }

    /// Return `true` if all token slots in this block have been written.
    fn is_full(&self) -> bool {
        self.used >= BLOCK_TOKEN_COUNT
    }

    /// Return the number of unused token slots remaining in this block.
    #[allow(dead_code)]
    fn remaining(&self) -> usize {
        BLOCK_TOKEN_COUNT - self.used
    }

    /// Write a single token vector into the block. Returns error if full.
    fn write_token(&mut self, token: &[f64]) -> Result<(), RuntimeError> {
        if token.len() != self.dim {
            return Err(RuntimeError::ShapeMismatch {
                expected: self.dim,
                got: token.len(),
            });
        }
        if self.is_full() {
            return Err(RuntimeError::InvalidOperation(
                "KvBlock is full".to_string(),
            ));
        }
        let base = self.used * self.dim;
        self.data[base..base + self.dim].copy_from_slice(token);
        self.used += 1;
        Ok(())
    }

    /// Read token at position `idx` within this block.
    fn read_token(&self, idx: usize) -> &[f64] {
        let base = idx * self.dim;
        &self.data[base..base + self.dim]
    }
}

/// A vLLM-style block-paged KV-cache. Instead of one contiguous pre-allocated
/// tensor (which may fragment or require realloc), this manages memory in
/// fixed-size 16-token blocks via a `BlockTable`.
///
/// Benefits:
/// - No single large allocation — blocks are page-sized
/// - Zero reallocation on append (new blocks allocated on demand from pool)
/// - Logical-to-physical mapping via block table
/// - Each block is independently cache-line friendly
#[derive(Debug, Clone)]
pub struct PagedKvCache {
    /// All allocated blocks.
    blocks: Vec<KvBlock>,
    /// Block table: maps logical block indices to physical block indices.
    /// `block_table[i]` = index into `blocks` for the i-th logical block.
    block_table: Vec<usize>,
    /// Hidden dimension per token.
    dim: usize,
    /// Maximum total tokens allowed.
    max_tokens: usize,
    /// Total tokens currently stored.
    current_len: usize,
}

impl PagedKvCache {
    /// Create a paged KV-cache for `max_tokens` tokens of dimension `dim`.
    ///
    /// Pre-allocates all blocks upfront to avoid any heap allocation during
    /// the inference loop. The number of blocks = ceil(max_tokens / 16).
    pub fn new(max_tokens: usize, dim: usize) -> Self {
        let num_blocks = (max_tokens + BLOCK_TOKEN_COUNT - 1) / BLOCK_TOKEN_COUNT;
        let mut blocks = Vec::with_capacity(num_blocks);
        let mut block_table = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            blocks.push(KvBlock::new(dim));
            block_table.push(i); // identity mapping initially
        }
        PagedKvCache {
            blocks,
            block_table,
            dim,
            max_tokens,
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

    /// Maximum tokens this cache can hold.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Hidden dimension per token.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of blocks allocated.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Number of blocks currently in use (partially or fully).
    pub fn blocks_in_use(&self) -> usize {
        if self.current_len == 0 { return 0; }
        (self.current_len + BLOCK_TOKEN_COUNT - 1) / BLOCK_TOKEN_COUNT
    }

    /// Append a single token vector. **Zero allocation** — writes into
    /// the next available slot in the current block.
    pub fn append(&mut self, token: &[f64]) -> Result<(), RuntimeError> {
        if token.len() != self.dim {
            return Err(RuntimeError::ShapeMismatch {
                expected: self.dim,
                got: token.len(),
            });
        }
        if self.current_len >= self.max_tokens {
            return Err(RuntimeError::InvalidOperation(
                format!(
                    "PagedKvCache full: {} / {} tokens",
                    self.current_len, self.max_tokens
                ),
            ));
        }
        let logical_block = self.current_len / BLOCK_TOKEN_COUNT;
        let physical_block = self.block_table[logical_block];
        self.blocks[physical_block].write_token(token)?;
        self.current_len += 1;
        Ok(())
    }

    /// Append a batch of tokens from a 2D tensor `[n, dim]`.
    pub fn append_tensor(&mut self, t: &Tensor) -> Result<(), RuntimeError> {
        if t.ndim() != 2 || t.shape()[1] != self.dim {
            return Err(RuntimeError::InvalidOperation(
                format!(
                    "PagedKvCache.append_tensor: expected [n, {}], got {:?}",
                    self.dim, t.shape()
                ),
            ));
        }
        let n = t.shape()[0];
        if self.current_len + n > self.max_tokens {
            return Err(RuntimeError::InvalidOperation(
                format!(
                    "PagedKvCache overflow: {} + {} > {}",
                    self.current_len, n, self.max_tokens
                ),
            ));
        }
        let data = t.to_vec();
        for i in 0..n {
            let start = i * self.dim;
            self.append(&data[start..start + self.dim])?;
        }
        Ok(())
    }

    /// Materialize all stored tokens into a contiguous Tensor `[current_len, dim]`.
    ///
    /// This is a read operation that copies data from blocks into a flat
    /// buffer. The copy is required since blocks are non-contiguous.
    pub fn as_tensor(&self) -> Tensor {
        if self.current_len == 0 {
            return Tensor::from_vec(vec![], &[0, self.dim])
                .unwrap_or_else(|_| Tensor::zeros(&[0]));
        }
        let mut data = Vec::with_capacity(self.current_len * self.dim);
        let mut remaining = self.current_len;
        for &phys_idx in &self.block_table {
            if remaining == 0 { break; }
            let block = &self.blocks[phys_idx];
            let tokens_in_block = remaining.min(block.used);
            for t in 0..tokens_in_block {
                data.extend_from_slice(block.read_token(t));
            }
            remaining -= tokens_in_block;
        }
        Tensor::from_vec(data, &[self.current_len, self.dim])
            .expect("PagedKvCache::as_tensor shape mismatch")
    }

    /// Reset the cache to empty without deallocating blocks.
    /// Block data is retained; only cursors are reset.
    pub fn clear(&mut self) {
        for block in &mut self.blocks {
            block.used = 0;
        }
        self.current_len = 0;
    }

    /// Read a single token at logical position `idx`.
    pub fn get_token(&self, idx: usize) -> Result<Vec<f64>, RuntimeError> {
        if idx >= self.current_len {
            return Err(RuntimeError::IndexOutOfBounds {
                index: idx,
                length: self.current_len,
            });
        }
        let logical_block = idx / BLOCK_TOKEN_COUNT;
        let offset_in_block = idx % BLOCK_TOKEN_COUNT;
        let physical_block = self.block_table[logical_block];
        Ok(self.blocks[physical_block].read_token(offset_in_block).to_vec())
    }
}

impl fmt::Display for PagedKvCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PagedKvCache(len={}, max={}, dim={}, blocks={}/{})",
            self.current_len,
            self.max_tokens,
            self.dim,
            self.blocks_in_use(),
            self.blocks.len()
        )
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, data={:?})", self.shape, self.to_vec())
    }
}

