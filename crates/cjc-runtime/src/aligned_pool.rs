use std::fmt;
use std::rc::Rc;

use crate::error::RuntimeError;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// 2c. AlignedPool — 16-byte aligned allocation for SIMD readiness
// ---------------------------------------------------------------------------

/// A pre-allocated memory pool with 16-byte alignment guarantee.
///
/// Used by `AlignedByteSlice` to ensure that f32/f64 data mapped from raw
/// bytes starts on a SIMD-friendly boundary. When source bytes are already
/// aligned, no copy is needed; when misaligned, a one-time aligned copy is
/// performed into the pool.
#[derive(Debug, Clone)]
pub struct AlignedPool {
    /// Backing storage. The Vec itself is heap-allocated with alignment ≥ 8.
    /// We over-allocate by 15 bytes and track the aligned offset.
    storage: Vec<u8>,
    /// Byte offset into `storage` where the aligned region begins.
    aligned_offset: usize,
    /// Usable capacity (bytes) from the aligned offset.
    capacity: usize,
    /// Number of bytes currently written.
    len: usize,
}

impl AlignedPool {
    /// Create a new pool with capacity for at least `capacity_bytes` of
    /// 16-byte-aligned data. The actual allocation may be slightly larger.
    pub fn new(capacity_bytes: usize) -> Self {
        // Over-allocate by 15 bytes so we can always find a 16-byte boundary.
        let alloc_size = capacity_bytes + 15;
        let storage = vec![0u8; alloc_size];
        let base_ptr = storage.as_ptr() as usize;
        let aligned_offset = (16 - (base_ptr % 16)) % 16;
        AlignedPool {
            storage,
            aligned_offset,
            capacity: capacity_bytes,
            len: 0,
        }
    }

    /// Returns a pointer to the aligned region.
    pub fn as_ptr(&self) -> *const u8 {
        // SAFETY: aligned_offset is always within bounds by construction.
        unsafe { self.storage.as_ptr().add(self.aligned_offset) }
    }

    /// Returns a mutable pointer to the aligned region.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        unsafe { self.storage.as_mut_ptr().add(self.aligned_offset) }
    }

    /// Returns the aligned region as a byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        &self.storage[self.aligned_offset..self.aligned_offset + self.len]
    }

    /// Check if a raw pointer is 16-byte aligned.
    pub fn is_aligned_16(ptr: *const u8) -> bool {
        (ptr as usize) % 16 == 0
    }

    /// Copy `data` into the pool, returning the aligned byte slice.
    /// Returns an error if data exceeds pool capacity.
    pub fn copy_from(&mut self, data: &[u8]) -> Result<(), RuntimeError> {
        if data.len() > self.capacity {
            return Err(RuntimeError::InvalidOperation(
                format!(
                    "AlignedPool: data length {} exceeds capacity {}",
                    data.len(),
                    self.capacity
                ),
            ));
        }
        let dest = &mut self.storage[self.aligned_offset..self.aligned_offset + data.len()];
        dest.copy_from_slice(data);
        self.len = data.len();
        Ok(())
    }

    /// Current number of bytes stored.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Verify that the aligned pointer is indeed 16-byte aligned.
    pub fn check_alignment(&self) -> bool {
        Self::is_aligned_16(self.as_ptr())
    }
}

impl fmt::Display for AlignedPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AlignedPool(len={}, capacity={}, aligned={})",
            self.len, self.capacity, self.check_alignment()
        )
    }
}

/// An alignment-aware byte slice that guarantees 16-byte alignment for
/// tensor weight mapping. If the source bytes are already aligned, it
/// wraps them directly. If misaligned, it copies into an `AlignedPool`.
#[derive(Debug, Clone)]
pub struct AlignedByteSlice {
    /// The pool holds the aligned copy (if a copy was needed).
    pool: Option<AlignedPool>,
    /// Original bytes (kept for reference / fallback).
    original: Rc<Vec<u8>>,
    /// Whether a copy was performed (true = was misaligned).
    was_copied: bool,
}

impl AlignedByteSlice {
    /// Create an aligned byte slice from raw bytes.
    ///
    /// If the data is already 16-byte aligned, no copy is performed.
    /// If misaligned, the data is copied into an aligned pool and a
    /// warning flag is set.
    pub fn from_bytes(data: Rc<Vec<u8>>) -> Self {
        let ptr = data.as_ptr();
        if AlignedPool::is_aligned_16(ptr) {
            AlignedByteSlice {
                pool: None,
                original: data,
                was_copied: false,
            }
        } else {
            let mut pool = AlignedPool::new(data.len());
            // This cannot fail: pool capacity == data.len()
            pool.copy_from(&data).unwrap();
            AlignedByteSlice {
                pool: Some(pool),
                original: data,
                was_copied: true,
            }
        }
    }

    /// Get the aligned bytes. If a copy was needed, returns the pool's
    /// bytes; otherwise returns the original directly.
    pub fn as_bytes(&self) -> &[u8] {
        match &self.pool {
            Some(pool) => pool.as_bytes(),
            None => &self.original,
        }
    }

    /// Whether a copy was required for alignment.
    pub fn was_realigned(&self) -> bool {
        self.was_copied
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.original.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.original.is_empty()
    }

    /// Map these aligned bytes to a Tensor, identical to Tensor::from_bytes
    /// but with alignment guarantee.
    pub fn as_tensor(&self, shape: &[usize], dtype: &str) -> Result<Tensor, RuntimeError> {
        Tensor::from_bytes(self.as_bytes(), shape, dtype)
    }
}

impl fmt::Display for AlignedByteSlice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AlignedByteSlice(len={}, realigned={})",
            self.len(),
            self.was_copied
        )
    }
}

