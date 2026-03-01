//! Frame Arena — bump allocator for non-escaping values.
//!
//! Each function invocation gets a `FrameArena` that satisfies all
//! `AllocHint::Arena` allocations. The arena is bulk-freed when the
//! function returns (no individual deallocation).
//!
//! # Determinism
//!
//! - Allocation order is sequential (bump pointer, no fragmentation).
//! - No OS memory is returned during normal execution — arenas grow but
//!   never shrink. After bulk-free, memory is retained for reuse.
//! - Same sequence of allocations → same layout, every time.
//!
//! # Design
//!
//! The arena manages a list of 4 KB pages. Each page is a `Vec<u8>` that
//! is never deallocated during program execution. When a page is exhausted,
//! a new page is allocated. On `reset()`, the bump pointer returns to the
//! start of the first page — all subsequent pages remain allocated but
//! unused (retained for future frames).
//!
//! The arena does NOT hand out typed references — it stores opaque byte
//! slices. The caller is responsible for alignment and type safety.
//! For CJC's `Value` type (which is always the same size), alignment is
//! guaranteed by the page alignment.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

/// Default page size: 4 KB (fits ~64 Values at 64 bytes each).
const PAGE_SIZE: usize = 4096;

/// A bump-arena that allocates sequentially within fixed-size pages.
///
/// Bulk-freed via `reset()` at function return. Pages are never returned
/// to the OS during execution — they are retained for reuse.
pub struct FrameArena {
    /// All allocated pages (never freed during program lifetime).
    pages: Vec<Vec<u8>>,
    /// Index of the current page being filled.
    current_page: usize,
    /// Byte offset within the current page.
    cursor: usize,
    /// Total bytes allocated across all arena lifetimes (monotonic counter).
    pub total_allocated: usize,
    /// Number of reset() calls.
    pub reset_count: usize,
}

impl FrameArena {
    /// Create a new arena with one pre-allocated page.
    pub fn new() -> Self {
        FrameArena {
            pages: vec![vec![0u8; PAGE_SIZE]],
            current_page: 0,
            cursor: 0,
            total_allocated: 0,
            reset_count: 0,
        }
    }

    /// Create an arena with a custom page size (for testing).
    pub fn with_page_size(page_size: usize) -> Self {
        let size = if page_size == 0 { PAGE_SIZE } else { page_size };
        FrameArena {
            pages: vec![vec![0u8; size]],
            current_page: 0,
            cursor: 0,
            total_allocated: 0,
            reset_count: 0,
        }
    }

    /// Allocate `size` bytes from the arena. Returns the (page_index, offset)
    /// pair that identifies the allocation.
    ///
    /// If the current page doesn't have enough room, advances to the next
    /// page (allocating a new one if needed).
    pub fn alloc_bytes(&mut self, size: usize) -> (usize, usize) {
        // Align to 8 bytes for safe casting.
        let aligned = (size + 7) & !7;
        let page_size = self.page_size();

        // If the allocation is larger than a page, allocate a dedicated page.
        if aligned > page_size {
            let page_idx = self.pages.len();
            self.pages.push(vec![0u8; aligned]);
            self.total_allocated += aligned;
            return (page_idx, 0);
        }

        // Check if current page has room.
        if self.cursor + aligned > page_size {
            // Move to next page.
            self.current_page += 1;
            self.cursor = 0;
            if self.current_page >= self.pages.len() {
                self.pages.push(vec![0u8; page_size]);
            }
        }

        let offset = self.cursor;
        self.cursor += aligned;
        self.total_allocated += aligned;
        (self.current_page, offset)
    }

    /// Get a reference to the bytes at the given (page, offset) location.
    pub fn get_bytes(&self, page: usize, offset: usize, len: usize) -> Option<&[u8]> {
        self.pages
            .get(page)
            .and_then(|p| p.get(offset..offset + len))
    }

    /// Get a mutable reference to the bytes at the given (page, offset) location.
    pub fn get_bytes_mut(&mut self, page: usize, offset: usize, len: usize) -> Option<&mut [u8]> {
        self.pages
            .get_mut(page)
            .and_then(|p| p.get_mut(offset..offset + len))
    }

    /// Reset the arena for reuse. All previous allocations are invalidated.
    /// Pages are NOT freed — they are retained for the next frame.
    pub fn reset(&mut self) {
        self.current_page = 0;
        self.cursor = 0;
        self.reset_count += 1;
    }

    /// Number of pages currently allocated (including unused retained pages).
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Total capacity in bytes across all pages.
    pub fn capacity(&self) -> usize {
        self.pages.iter().map(|p| p.len()).sum()
    }

    /// Bytes currently in use (from start of page 0 to current cursor).
    pub fn used_bytes(&self) -> usize {
        if self.pages.is_empty() {
            return 0;
        }
        let page_size = self.page_size();
        self.current_page * page_size + self.cursor
    }

    fn page_size(&self) -> usize {
        self.pages.first().map(|p| p.len()).unwrap_or(PAGE_SIZE)
    }
}

impl Default for FrameArena {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Type-erased arena storage for CJC Values
// ---------------------------------------------------------------------------

/// A type-erased arena entry. Stores an `Rc<RefCell<Box<dyn Any>>>` so it
/// can be shared with the Value system while being backed by arena memory.
///
/// In the current implementation, the arena provides the backing store
/// discipline (bulk-free on reset), while Rc provides the sharing semantics
/// needed by CJC's Value type. The key benefit: arena values don't need
/// individual deallocation and can be bulk-freed.
pub struct ArenaEntry {
    /// The stored value. Uses Rc for compatibility with Value::ClassRef etc.
    pub value: Rc<RefCell<Box<dyn Any>>>,
}

/// Arena-backed object store. Combines FrameArena with a typed entry list
/// for objects that need to be accessed by index.
pub struct ArenaStore {
    /// The underlying bump arena (tracks memory pages).
    pub arena: FrameArena,
    /// Type-erased entries stored in this frame.
    entries: Vec<Option<ArenaEntry>>,
    /// Free list for entry reuse after reset.
    free_slots: Vec<usize>,
}

impl ArenaStore {
    pub fn new() -> Self {
        ArenaStore {
            arena: FrameArena::new(),
            entries: Vec::new(),
            free_slots: Vec::new(),
        }
    }

    /// Allocate a new entry in the arena store.
    pub fn alloc<T: Any + 'static>(&mut self, value: T) -> usize {
        let entry = ArenaEntry {
            value: Rc::new(RefCell::new(Box::new(value))),
        };
        if let Some(idx) = self.free_slots.pop() {
            self.entries[idx] = Some(entry);
            idx
        } else {
            let idx = self.entries.len();
            self.entries.push(Some(entry));
            idx
        }
    }

    /// Get a reference to the stored value at the given index.
    pub fn get<T: Any + 'static>(&self, index: usize) -> Option<&T> {
        self.entries.get(index).and_then(|slot| {
            slot.as_ref().and_then(|entry| {
                let borrowed = entry.value.as_ref();
                // Safety: we need to extend the lifetime through the RefCell.
                // This is safe because the ArenaStore owns the Rc and the
                // caller borrows &self.
                let ptr = borrowed.as_ptr();
                unsafe { (*ptr).downcast_ref::<T>() }
            })
        })
    }

    /// Number of live entries.
    pub fn live_count(&self) -> usize {
        self.entries.iter().filter(|e| e.is_some()).count()
    }

    /// Reset the store — mark all entries as free, reset the arena.
    pub fn reset(&mut self) {
        self.free_slots.clear();
        for i in 0..self.entries.len() {
            if self.entries[i].is_some() {
                self.entries[i] = None;
                self.free_slots.push(i);
            }
        }
        // Reverse so LIFO pop gives lowest indices first (deterministic).
        self.free_slots.reverse();
        self.arena.reset();
    }
}

impl Default for ArenaStore {
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
    fn test_arena_basic_alloc() {
        let mut arena = FrameArena::new();
        let (p1, o1) = arena.alloc_bytes(16);
        let (p2, o2) = arena.alloc_bytes(16);

        // Both should be on page 0, sequential.
        assert_eq!(p1, 0);
        assert_eq!(o1, 0);
        assert_eq!(p2, 0);
        assert_eq!(o2, 16); // 16 bytes aligned to 8 → 16
    }

    #[test]
    fn test_arena_page_overflow() {
        let mut arena = FrameArena::with_page_size(32);
        let (p1, _) = arena.alloc_bytes(16);
        let (p2, _) = arena.alloc_bytes(16);
        // 32 bytes fills page 0.
        assert_eq!(p1, 0);
        assert_eq!(p2, 0);

        // Next alloc should go to page 1.
        let (p3, o3) = arena.alloc_bytes(8);
        assert_eq!(p3, 1);
        assert_eq!(o3, 0);
        assert_eq!(arena.page_count(), 2);
    }

    #[test]
    fn test_arena_reset_reuses_pages() {
        let mut arena = FrameArena::with_page_size(32);
        arena.alloc_bytes(16);
        arena.alloc_bytes(16);
        arena.alloc_bytes(16); // page 1
        assert_eq!(arena.page_count(), 2);

        arena.reset();
        assert_eq!(arena.page_count(), 2); // Pages retained
        assert_eq!(arena.used_bytes(), 0);
        assert_eq!(arena.reset_count, 1);

        // New allocs reuse page 0.
        let (p, o) = arena.alloc_bytes(16);
        assert_eq!(p, 0);
        assert_eq!(o, 0);
    }

    #[test]
    fn test_arena_alignment() {
        let mut arena = FrameArena::new();
        let (_, o1) = arena.alloc_bytes(1);  // 1 byte → aligned to 8
        let (_, o2) = arena.alloc_bytes(1);  // next starts at 8
        assert_eq!(o1, 0);
        assert_eq!(o2, 8);
    }

    #[test]
    fn test_arena_oversized_alloc() {
        let mut arena = FrameArena::with_page_size(32);
        // Allocate something bigger than a page.
        let (p, o) = arena.alloc_bytes(64);
        assert_eq!(o, 0);
        // Should be on a dedicated page.
        assert!(arena.pages[p].len() >= 64);
    }

    #[test]
    fn test_arena_get_bytes() {
        let mut arena = FrameArena::new();
        let (p, o) = arena.alloc_bytes(8);
        // Write some data.
        let bytes = arena.get_bytes_mut(p, o, 8).unwrap();
        bytes.copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        // Read it back.
        let read = arena.get_bytes(p, o, 8).unwrap();
        assert_eq!(read, &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_arena_deterministic_layout() {
        // Same allocation sequence → same layout.
        let mut a1 = FrameArena::with_page_size(64);
        let mut a2 = FrameArena::with_page_size(64);

        let r1: Vec<_> = (0..10).map(|_| a1.alloc_bytes(8)).collect();
        let r2: Vec<_> = (0..10).map(|_| a2.alloc_bytes(8)).collect();

        assert_eq!(r1, r2, "deterministic layout");
    }

    // -- ArenaStore tests ---------------------------------------------------

    #[test]
    fn test_store_alloc_and_read() {
        let mut store = ArenaStore::new();
        let idx = store.alloc(42i64);
        assert_eq!(*store.get::<i64>(idx).unwrap(), 42);
    }

    #[test]
    fn test_store_reset_frees_entries() {
        let mut store = ArenaStore::new();
        store.alloc(1i64);
        store.alloc(2i64);
        assert_eq!(store.live_count(), 2);

        store.reset();
        assert_eq!(store.live_count(), 0);
    }

    #[test]
    fn test_store_reset_reuses_slots() {
        let mut store = ArenaStore::new();
        let _a = store.alloc(1i64);
        let _b = store.alloc(2i64);
        store.reset();

        // After reset, new allocs reuse old slots.
        let c = store.alloc(99i64);
        assert!(c < 2, "should reuse a freed slot, got {c}");
        assert_eq!(*store.get::<i64>(c).unwrap(), 99);
    }

    #[test]
    fn test_store_type_mismatch() {
        let mut store = ArenaStore::new();
        let idx = store.alloc(42i64);
        assert!(store.get::<String>(idx).is_none());
    }

    #[test]
    fn test_arena_capacity_grows_not_shrinks() {
        let mut arena = FrameArena::with_page_size(32);
        for _ in 0..10 {
            arena.alloc_bytes(16);
        }
        let cap = arena.capacity();
        arena.reset();
        assert_eq!(arena.capacity(), cap, "capacity must not shrink after reset");
    }
}
