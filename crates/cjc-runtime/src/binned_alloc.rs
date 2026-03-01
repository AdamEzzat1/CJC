//! Deterministic Binned Free Lists
//!
//! Provides a size-class allocator with 13 bins (16 B – 64 KB).
//! Each bin maintains a LIFO free list for O(1) alloc/free.
//!
//! # Determinism guarantees
//!
//! - **LIFO push/pop**: same free+alloc sequence → same addresses.
//! - **No OS return**: pages are never returned to the OS during normal
//!   execution. Memory grows monotonically, freed memory is recycled.
//! - **Deterministic iteration**: bins are indexed by size class, not
//!   hash-based — iteration order is always the same.
//!
//! # Size classes
//!
//! | Bin | Size    |
//! |-----|---------|
//! |   0 |    16 B |
//! |   1 |    32 B |
//! |   2 |    48 B |
//! |   3 |    64 B |
//! |   4 |   128 B |
//! |   5 |   256 B |
//! |   6 |   512 B |
//! |   7 |  1 KB   |
//! |   8 |  2 KB   |
//! |   9 |  4 KB   |
//! |  10 |  8 KB   |
//! |  11 | 16 KB   |
//! |  12 | 64 KB   |
//!
//! Allocations larger than 64 KB go to a dedicated overflow list.

/// Number of size-class bins.
const NUM_BINS: usize = 13;

/// Size classes in bytes.
const SIZE_CLASSES: [usize; NUM_BINS] = [
    16, 32, 48, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536,
];

/// Find the bin index for a given allocation size.
/// Returns `None` for sizes > 64 KB (overflow).
fn bin_for_size(size: usize) -> Option<usize> {
    for (i, &class_size) in SIZE_CLASSES.iter().enumerate() {
        if size <= class_size {
            return Some(i);
        }
    }
    None
}

/// A single free-list bin. Stores pre-allocated blocks of a fixed size class.
struct Bin {
    /// Size class in bytes.
    size_class: usize,
    /// LIFO free list of block indices into `BinnedAllocator::storage`.
    free_list: Vec<usize>,
    /// Total blocks ever allocated for this bin.
    total_allocated: usize,
}

impl Bin {
    fn new(size_class: usize) -> Self {
        Bin {
            size_class,
            free_list: Vec::new(),
            total_allocated: 0,
        }
    }
}

/// Block metadata in the global storage.
struct Block {
    /// Offset into the backing buffer.
    offset: usize,
    /// Size class of this block.
    size: usize,
    /// Whether this block is currently in use.
    in_use: bool,
}

/// Deterministic binned allocator with LIFO free lists.
///
/// All memory is backed by a single contiguous `Vec<u8>` that grows
/// monotonically. No memory is ever returned to the OS during execution.
pub struct BinnedAllocator {
    /// Size-class bins (indexed by bin number).
    bins: Vec<Bin>,
    /// Global block registry.
    blocks: Vec<Block>,
    /// Backing storage — grows but never shrinks.
    storage: Vec<u8>,
    /// Overflow blocks (> 64 KB).
    overflow_free: Vec<usize>,
    /// Stats: total allocations.
    pub alloc_count: usize,
    /// Stats: total frees.
    pub free_count: usize,
}

impl BinnedAllocator {
    /// Create a new binned allocator.
    pub fn new() -> Self {
        let bins = SIZE_CLASSES.iter().map(|&s| Bin::new(s)).collect();
        BinnedAllocator {
            bins,
            blocks: Vec::new(),
            storage: Vec::new(),
            overflow_free: Vec::new(),
            alloc_count: 0,
            free_count: 0,
        }
    }

    /// Allocate a block of at least `size` bytes. Returns a block index.
    ///
    /// If a free block of the right size class exists, it is reused (LIFO).
    /// Otherwise, a new block is carved from the backing storage.
    pub fn alloc(&mut self, size: usize) -> usize {
        self.alloc_count += 1;

        if let Some(bin_idx) = bin_for_size(size) {
            let bin = &mut self.bins[bin_idx];
            // Try to reuse a free block.
            if let Some(block_idx) = bin.free_list.pop() {
                self.blocks[block_idx].in_use = true;
                return block_idx;
            }
            // Allocate a new block.
            let class_size = bin.size_class;
            bin.total_allocated += 1;
            self.alloc_new_block(class_size)
        } else {
            // Overflow: try reuse, else allocate.
            if let Some(block_idx) = self.overflow_free.pop() {
                if self.blocks[block_idx].size >= size {
                    self.blocks[block_idx].in_use = true;
                    return block_idx;
                }
                // Put it back — wrong size.
                self.overflow_free.push(block_idx);
            }
            self.alloc_new_block(size)
        }
    }

    /// Free a block, returning it to the appropriate bin's free list.
    pub fn free(&mut self, block_idx: usize) {
        if block_idx >= self.blocks.len() {
            return;
        }
        let block = &mut self.blocks[block_idx];
        if !block.in_use {
            return; // Double-free protection.
        }
        block.in_use = false;
        self.free_count += 1;

        let size = block.size;
        if let Some(bin_idx) = bin_for_size(size) {
            self.bins[bin_idx].free_list.push(block_idx);
        } else {
            self.overflow_free.push(block_idx);
        }
    }

    /// Get a slice to the block's memory.
    pub fn get(&self, block_idx: usize) -> Option<&[u8]> {
        self.blocks.get(block_idx).and_then(|b| {
            if b.in_use {
                self.storage.get(b.offset..b.offset + b.size)
            } else {
                None
            }
        })
    }

    /// Get a mutable slice to the block's memory.
    pub fn get_mut(&mut self, block_idx: usize) -> Option<&mut [u8]> {
        if let Some(b) = self.blocks.get(block_idx) {
            if b.in_use {
                let offset = b.offset;
                let size = b.size;
                return self.storage.get_mut(offset..offset + size);
            }
        }
        None
    }

    /// Write bytes into a block.
    pub fn write(&mut self, block_idx: usize, data: &[u8]) -> bool {
        if let Some(slice) = self.get_mut(block_idx) {
            let len = data.len().min(slice.len());
            slice[..len].copy_from_slice(&data[..len]);
            true
        } else {
            false
        }
    }

    /// Number of blocks currently in use.
    pub fn live_count(&self) -> usize {
        self.blocks.iter().filter(|b| b.in_use).count()
    }

    /// Total blocks ever allocated.
    pub fn total_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Total backing storage in bytes (never shrinks).
    pub fn storage_bytes(&self) -> usize {
        self.storage.len()
    }

    /// Number of free blocks across all bins.
    pub fn free_block_count(&self) -> usize {
        self.bins.iter().map(|b| b.free_list.len()).sum::<usize>()
            + self.overflow_free.len()
    }

    // Internal: allocate a new block from the backing storage.
    fn alloc_new_block(&mut self, size: usize) -> usize {
        let offset = self.storage.len();
        // Align to 8 bytes.
        let aligned = (size + 7) & !7;
        self.storage.resize(self.storage.len() + aligned, 0);
        let idx = self.blocks.len();
        self.blocks.push(Block {
            offset,
            size: aligned,
            in_use: true,
        });
        idx
    }
}

impl Default for BinnedAllocator {
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
    fn test_bin_selection() {
        assert_eq!(bin_for_size(1), Some(0));   // 16 B
        assert_eq!(bin_for_size(16), Some(0));  // 16 B
        assert_eq!(bin_for_size(17), Some(1));  // 32 B
        assert_eq!(bin_for_size(48), Some(2));  // 48 B
        assert_eq!(bin_for_size(65), Some(4));  // 128 B
        assert_eq!(bin_for_size(65536), Some(12)); // 64 KB
        assert_eq!(bin_for_size(65537), None);  // overflow
    }

    #[test]
    fn test_alloc_and_read() {
        let mut alloc = BinnedAllocator::new();
        let b = alloc.alloc(8);
        assert!(alloc.write(b, &[1, 2, 3, 4, 5, 6, 7, 8]));
        assert_eq!(&alloc.get(b).unwrap()[..8], &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_free_and_reuse() {
        let mut alloc = BinnedAllocator::new();
        let b1 = alloc.alloc(16);
        alloc.free(b1);

        // Next alloc of same size class should reuse b1's slot.
        let b2 = alloc.alloc(16);
        assert_eq!(b1, b2, "LIFO reuse");
    }

    #[test]
    fn test_double_free_harmless() {
        let mut alloc = BinnedAllocator::new();
        let b = alloc.alloc(16);
        alloc.free(b);
        alloc.free(b); // should not panic
        assert_eq!(alloc.free_count, 1, "second free should be no-op");
    }

    #[test]
    fn test_storage_never_shrinks() {
        let mut alloc = BinnedAllocator::new();
        for _ in 0..100 {
            let b = alloc.alloc(64);
            alloc.free(b);
        }
        let bytes = alloc.storage_bytes();
        // After many alloc/free cycles, storage should have grown.
        assert!(bytes > 0);
        // Storage never shrinks.
        for _ in 0..100 {
            let b = alloc.alloc(64);
            alloc.free(b);
        }
        assert_eq!(alloc.storage_bytes(), bytes, "storage must not shrink");
    }

    #[test]
    fn test_deterministic_alloc_order() {
        let mut a1 = BinnedAllocator::new();
        let mut a2 = BinnedAllocator::new();

        let seq1: Vec<usize> = (0..10).map(|_| a1.alloc(32)).collect();
        let seq2: Vec<usize> = (0..10).map(|_| a2.alloc(32)).collect();

        assert_eq!(seq1, seq2, "same alloc sequence → same block indices");
    }

    #[test]
    fn test_lifo_free_order() {
        let mut alloc = BinnedAllocator::new();
        let b1 = alloc.alloc(32);
        let b2 = alloc.alloc(32);
        let b3 = alloc.alloc(32);

        // Free in order: b1, b2, b3
        alloc.free(b1);
        alloc.free(b2);
        alloc.free(b3);

        // LIFO: next alloc should return b3, then b2, then b1.
        assert_eq!(alloc.alloc(32), b3);
        assert_eq!(alloc.alloc(32), b2);
        assert_eq!(alloc.alloc(32), b1);
    }

    #[test]
    fn test_overflow_alloc() {
        let mut alloc = BinnedAllocator::new();
        let b = alloc.alloc(100_000); // > 64 KB
        assert!(alloc.write(b, &[0xAB; 100]));
        assert_eq!(alloc.get(b).unwrap()[0], 0xAB);
    }

    #[test]
    fn test_live_count() {
        let mut alloc = BinnedAllocator::new();
        let b1 = alloc.alloc(16);
        let b2 = alloc.alloc(32);
        let _b3 = alloc.alloc(64);
        assert_eq!(alloc.live_count(), 3);

        alloc.free(b1);
        assert_eq!(alloc.live_count(), 2);

        alloc.free(b2);
        assert_eq!(alloc.live_count(), 1);
    }

    #[test]
    fn test_freed_block_not_readable() {
        let mut alloc = BinnedAllocator::new();
        let b = alloc.alloc(16);
        alloc.free(b);
        assert!(alloc.get(b).is_none(), "freed block should not be readable");
    }

    #[test]
    fn test_mixed_size_classes() {
        let mut alloc = BinnedAllocator::new();
        let small = alloc.alloc(8);
        let medium = alloc.alloc(256);
        let large = alloc.alloc(4096);

        alloc.free(small);
        alloc.free(medium);
        alloc.free(large);

        // Each reuses its own size class.
        let s2 = alloc.alloc(8);
        let m2 = alloc.alloc(256);
        let l2 = alloc.alloc(4096);

        assert_eq!(s2, small);
        assert_eq!(m2, medium);
        assert_eq!(l2, large);
    }
}
