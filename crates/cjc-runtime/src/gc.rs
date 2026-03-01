//! GC compatibility layer — thin wrapper over `ObjectSlab`.
//!
//! The mark-sweep garbage collector has been removed. This module preserves
//! the `GcRef` and `GcHeap` API for backward compatibility, but all allocation
//! is now backed by deterministic reference counting via `ObjectSlab`.
//!
//! # What changed
//!
//! - `GcHeap::alloc()` → delegates to `ObjectSlab::alloc()` (RC-backed)
//! - `GcHeap::collect()` → no-op (RC handles deallocation)
//! - `GcHeap::mark()` / `sweep()` → removed (no mark-sweep semantics)
//! - `GcHeap::get()` → delegates to `ObjectSlab::get()`
//! - `GcHeap::live_count()` → delegates to `ObjectSlab::live_count()`
//!
//! # Determinism
//!
//! The `ObjectSlab` provides deterministic LIFO slot reuse. Same allocation
//! sequence → same slot indices. No stop-the-world pauses.

use std::any::Any;
use std::fmt;

use crate::object_slab::{ObjectSlab, SlabRef};

/// A handle into the GC heap. Lightweight, copyable index.
///
/// Now backed by `SlabRef` (RC-based slab) instead of mark-sweep GC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GcRef {
    pub index: usize,
}

impl GcRef {
    fn from_slab(sr: SlabRef) -> Self {
        GcRef { index: sr.index }
    }

    fn to_slab(self) -> SlabRef {
        SlabRef { index: self.index }
    }
}

/// Backward-compatible GC heap interface backed by `ObjectSlab`.
///
/// All mark-sweep semantics have been removed. Objects are reference-counted
/// and freed deterministically when no references remain.
pub struct GcHeap {
    slab: ObjectSlab,
    /// Maintained for API compat; incremented on `collect()` calls.
    collection_count: u64,
}

impl fmt::Debug for GcHeap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GcHeap")
            .field("live_count", &self.slab.live_count())
            .field("capacity", &self.slab.capacity())
            .field("collection_count", &self.collection_count)
            .finish()
    }
}

impl GcHeap {
    /// Create a new heap. The `_collection_threshold` parameter is accepted
    /// for API compatibility but ignored (no automatic collection in RC mode).
    pub fn new(_collection_threshold: usize) -> Self {
        GcHeap {
            slab: ObjectSlab::new(),
            collection_count: 0,
        }
    }

    /// Allocate a value on the heap, returning a handle.
    pub fn alloc<T: Any + 'static>(&mut self, value: T) -> GcRef {
        GcRef::from_slab(self.slab.alloc(value))
    }

    /// Allocate with "auto-collection" — for API compat.
    /// The `_roots` parameter is ignored (no GC to trigger).
    pub fn alloc_auto<T: Any + 'static>(&mut self, value: T, _roots: &[GcRef]) -> GcRef {
        self.alloc(value)
    }

    /// Read a reference to the value behind `gc_ref`, downcasting to `T`.
    pub fn get<T: Any + 'static>(&self, gc_ref: GcRef) -> Option<&T> {
        self.slab.get::<T>(gc_ref.to_slab())
    }

    /// No-op: mark-sweep has been removed. Objects are reference-counted.
    pub fn collect(&mut self, _roots: &[GcRef]) {
        self.collection_count += 1;
        self.slab.collect_noop();
    }

    /// Number of live objects on the heap.
    pub fn live_count(&self) -> usize {
        self.slab.live_count()
    }

    /// Total capacity (number of slots, including freed ones).
    pub fn capacity(&self) -> usize {
        self.slab.capacity()
    }

    /// Access the free list (for backward compat with tests).
    pub fn free_list(&self) -> &[usize] {
        &self.slab.free_list
    }

    /// Explicitly free a slot.
    pub fn free(&mut self, gc_ref: GcRef) {
        self.slab.free(gc_ref.to_slab());
    }
}

impl Default for GcHeap {
    fn default() -> Self {
        Self::new(1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_and_read_back() {
        let mut heap = GcHeap::new(1024);
        let r = heap.alloc(42i64);
        assert_eq!(heap.get::<i64>(r), Some(&42));
        assert_eq!(heap.live_count(), 1);
    }

    #[test]
    fn collect_is_noop_objects_survive() {
        let mut heap = GcHeap::new(1024);
        let r1 = heap.alloc(10i64);
        let r2 = heap.alloc(20i64);
        // Collect with partial roots — but since it's RC-backed, ALL survive
        heap.collect(&[r1]);
        assert_eq!(heap.live_count(), 2, "RC keeps all objects alive");
        assert_eq!(heap.get::<i64>(r1), Some(&10));
        assert_eq!(heap.get::<i64>(r2), Some(&20));
    }

    #[test]
    fn explicit_free_and_slot_reuse() {
        let mut heap = GcHeap::new(1024);
        let r1 = heap.alloc(1i64);
        let r2 = heap.alloc(2i64);
        let _r3 = heap.alloc(3i64);

        // Explicitly free r2
        heap.free(r2);
        assert_eq!(heap.free_list().len(), 1);

        // New alloc reuses freed slot (LIFO)
        let r4 = heap.alloc(4i64);
        assert_eq!(r4.index, r2.index, "LIFO slot reuse");
        assert_eq!(heap.get::<i64>(r4), Some(&4));
        assert_eq!(heap.get::<i64>(r1), Some(&1));
    }

    #[test]
    fn type_mismatch_returns_none() {
        let mut heap = GcHeap::new(1024);
        let r = heap.alloc(42i64);
        assert_eq!(heap.get::<String>(r), None);
        assert_eq!(heap.get::<i64>(r), Some(&42));
    }

    #[test]
    fn alloc_auto_compat() {
        let mut heap = GcHeap::new(2);
        let r1 = heap.alloc(1i64);
        let _ = heap.alloc(2i64);
        // alloc_auto ignores roots in RC mode
        let r3 = heap.alloc_auto(3i64, &[r1]);
        assert_eq!(heap.get::<i64>(r1), Some(&1));
        assert_eq!(heap.get::<i64>(r3), Some(&3));
        // All objects still alive (RC, not GC)
        assert_eq!(heap.live_count(), 3);
    }

    #[test]
    fn deterministic_slot_order() {
        let mut h1 = GcHeap::new(1024);
        let mut h2 = GcHeap::new(1024);

        let a1 = h1.alloc(10i64);
        let a2 = h1.alloc(20i64);
        h1.free(a1);
        let a3 = h1.alloc(30i64);

        let b1 = h2.alloc(10i64);
        let b2 = h2.alloc(20i64);
        h2.free(b1);
        let b3 = h2.alloc(30i64);

        assert_eq!(a1.index, b1.index);
        assert_eq!(a2.index, b2.index);
        assert_eq!(a3.index, b3.index, "LIFO reuse deterministic");
    }
}
