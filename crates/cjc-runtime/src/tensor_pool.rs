//! Thread-local tensor buffer pool for reusing allocations.
//!
//! In tight loops (e.g., 50-layer NN forward pass), the same tensor sizes are
//! allocated and freed every iteration. This pool caches freed buffers and
//! returns them on the next allocation of the same size, avoiding repeated
//! malloc/free cycles.
//!
//! # Determinism
//!
//! Pool reuse does NOT affect computed values — only memory addresses change.
//! The buffer contents are always overwritten before use. Snap hashes
//! are computed from data, not addresses, so they remain identical.
//!
//! # Usage
//!
//! ```ignore
//! // Get a buffer (may be recycled or freshly allocated)
//! let mut buf = tensor_pool::acquire(1000);
//! // ... fill buf with data ...
//! // Buffer is returned to pool when dropped (via TensorPool::recycle)
//! ```

use std::cell::RefCell;

/// Maximum number of buffers cached per size class.
const MAX_CACHED_PER_SIZE: usize = 4;

/// Maximum total buffers in the pool (to prevent unbounded growth).
const MAX_TOTAL_CACHED: usize = 64;

/// Thread-local tensor buffer pool.
struct TensorPool {
    /// Cached buffers, sorted by capacity for binary search.
    /// Each entry: (capacity, Vec<f64>).
    buffers: Vec<Vec<f64>>,
}

impl TensorPool {
    fn new() -> Self {
        TensorPool {
            buffers: Vec::new(),
        }
    }

    /// Acquire a buffer of at least `size` elements.
    /// Returns a recycled buffer (cleared to zero) if one of matching size exists,
    /// otherwise allocates a new one.
    fn acquire(&mut self, size: usize) -> Vec<f64> {
        // Look for an exact-size match first (most common case in loops).
        if let Some(pos) = self.buffers.iter().position(|b| b.capacity() == size) {
            let mut buf = self.buffers.swap_remove(pos);
            buf.clear();
            buf.resize(size, 0.0);
            return buf;
        }
        // No match — allocate fresh.
        vec![0.0f64; size]
    }

    /// Return a buffer to the pool for future reuse.
    fn recycle(&mut self, buf: Vec<f64>) {
        if self.buffers.len() >= MAX_TOTAL_CACHED {
            return; // Pool is full, just drop the buffer.
        }
        let cap = buf.capacity();
        // Don't cache if too many of the same size already.
        let same_size_count = self.buffers.iter().filter(|b| b.capacity() == cap).count();
        if same_size_count >= MAX_CACHED_PER_SIZE {
            return;
        }
        self.buffers.push(buf);
    }
}

thread_local! {
    static POOL: RefCell<TensorPool> = RefCell::new(TensorPool::new());
}

/// Acquire a zeroed buffer of `size` f64 elements from the thread-local pool.
///
/// If a buffer of the exact capacity is available in the pool, it's reused
/// (avoiding malloc). Otherwise a new buffer is allocated.
pub fn acquire(size: usize) -> Vec<f64> {
    POOL.with(|pool| pool.borrow_mut().acquire(size))
}

/// Return a buffer to the thread-local pool for future reuse.
///
/// The buffer's contents are irrelevant — it will be cleared on next acquire.
/// If the pool is full, the buffer is simply dropped.
pub fn recycle(buf: Vec<f64>) {
    POOL.with(|pool| pool.borrow_mut().recycle(buf));
}

/// Returns the current number of cached buffers in the pool (for diagnostics).
#[allow(dead_code)]
pub fn pool_size() -> usize {
    POOL.with(|pool| pool.borrow().buffers.len())
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acquire_returns_correct_size() {
        let buf = acquire(100);
        assert_eq!(buf.len(), 100);
        assert!(buf.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_recycle_and_reuse() {
        let buf = acquire(256);
        assert_eq!(pool_size(), 0);

        recycle(buf);
        assert_eq!(pool_size(), 1);

        let buf2 = acquire(256);
        assert_eq!(buf2.len(), 256);
        assert_eq!(pool_size(), 0); // Was taken from pool
    }

    #[test]
    fn test_pool_max_per_size() {
        for _ in 0..10 {
            let buf = acquire(64);
            recycle(buf);
        }
        // Should cap at MAX_CACHED_PER_SIZE
        assert!(pool_size() <= MAX_CACHED_PER_SIZE);
    }

    #[test]
    fn test_pool_total_limit() {
        for size in 0..100 {
            let buf = acquire(size + 1);
            recycle(buf);
        }
        assert!(pool_size() <= MAX_TOTAL_CACHED);
    }

    #[test]
    fn test_acquired_buffer_is_zeroed() {
        let mut buf = acquire(10);
        for x in buf.iter_mut() {
            *x = 42.0; // Dirty the buffer
        }
        recycle(buf);

        let buf2 = acquire(10);
        assert!(buf2.iter().all(|&x| x == 0.0), "Recycled buffer must be zeroed");
    }
}
