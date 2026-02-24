use std::any::Any;
use std::fmt;

// ---------------------------------------------------------------------------
// 3. GC for Layer 3 — simple mark-sweep
// ---------------------------------------------------------------------------

/// A handle into the GC heap. Lightweight, copyable index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GcRef {
    pub index: usize,
}

/// A type-erased object managed by the GC.
pub struct GcObject {
    marked: bool,
    value: Box<dyn Any>,
}

impl fmt::Debug for GcObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GcObject")
            .field("marked", &self.marked)
            .field("value", &"<dyn Any>")
            .finish()
    }
}

/// A simple mark-sweep garbage-collected heap.
///
/// Objects are stored in a `Vec` with `Option` slots; freed slots are reused
/// via a free-list. Collection is triggered automatically every
/// `collection_threshold` allocations.
#[derive(Debug)]
pub struct GcHeap {
    objects: Vec<Option<GcObject>>,
    pub free_list: Vec<usize>,
    alloc_count: usize,
    collection_threshold: usize,
}

impl GcHeap {
    /// Create a new GC heap with the given collection threshold.
    pub fn new(collection_threshold: usize) -> Self {
        GcHeap {
            objects: Vec::new(),
            free_list: Vec::new(),
            alloc_count: 0,
            collection_threshold,
        }
    }

    /// Allocate a value on the GC heap, returning a handle.
    ///
    /// If the allocation counter has reached the threshold, a collection is
    /// triggered first (the caller must provide roots).
    pub fn alloc<T: Any + 'static>(&mut self, value: T) -> GcRef {
        let obj = GcObject {
            marked: false,
            value: Box::new(value),
        };

        let index = if let Some(idx) = self.free_list.pop() {
            self.objects[idx] = Some(obj);
            idx
        } else {
            self.objects.push(Some(obj));
            self.objects.len() - 1
        };

        self.alloc_count += 1;
        GcRef { index }
    }

    /// Allocate with automatic collection when the threshold is reached.
    pub fn alloc_auto<T: Any + 'static>(&mut self, value: T, roots: &[GcRef]) -> GcRef {
        if self.alloc_count >= self.collection_threshold {
            self.collect(roots);
            self.alloc_count = 0;
        }
        self.alloc(value)
    }

    /// Read a reference to the value behind `gc_ref`, downcasting to `T`.
    /// Returns `None` if the slot is empty or the type does not match.
    pub fn get<T: Any + 'static>(&self, gc_ref: GcRef) -> Option<&T> {
        self.objects
            .get(gc_ref.index)
            .and_then(|slot| slot.as_ref())
            .and_then(|obj| obj.value.downcast_ref::<T>())
    }

    /// Mark a single object as reachable.
    pub fn mark(&mut self, gc_ref: GcRef) {
        if let Some(Some(obj)) = self.objects.get_mut(gc_ref.index) {
            obj.marked = true;
        }
    }

    /// Sweep all unmarked objects, freeing their memory and returning their
    /// slots to the free-list. Resets the mark flag on surviving objects.
    pub fn sweep(&mut self) {
        for i in 0..self.objects.len() {
            let should_free = match &self.objects[i] {
                Some(obj) => !obj.marked,
                None => false,
            };
            if should_free {
                self.objects[i] = None;
                self.free_list.push(i);
            } else if let Some(obj) = &mut self.objects[i] {
                obj.marked = false; // reset for next cycle
            }
        }
    }

    /// Run a full mark-sweep collection. Only objects reachable from `roots`
    /// survive.
    pub fn collect(&mut self, roots: &[GcRef]) {
        // Mark phase
        for &root in roots {
            self.mark(root);
        }
        // Sweep phase
        self.sweep();
    }

    /// Number of live (non-None) objects on the heap.
    pub fn live_count(&self) -> usize {
        self.objects.iter().filter(|s| s.is_some()).count()
    }

    /// Total capacity (number of slots, including freed ones).
    pub fn capacity(&self) -> usize {
        self.objects.len()
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
    fn sweep_unmarked_frees_object() {
        let mut heap = GcHeap::new(1024);
        let r1 = heap.alloc(10i64);
        let _r2 = heap.alloc(20i64);
        // Mark only r1 as reachable
        heap.collect(&[r1]);
        assert_eq!(heap.live_count(), 1);
        assert_eq!(heap.get::<i64>(r1), Some(&10));
    }

    #[test]
    fn free_list_slot_reuse() {
        let mut heap = GcHeap::new(1024);
        let r1 = heap.alloc(1i64);
        let _r2 = heap.alloc(2i64);
        // Collect with only r2 unreachable... actually keep r1
        heap.collect(&[r1]);
        assert_eq!(heap.free_list.len(), 1, "one slot freed");
        // Allocate again — should reuse the freed slot
        let r3 = heap.alloc(3i64);
        assert_eq!(heap.free_list.len(), 0, "free list drained");
        assert_eq!(heap.get::<i64>(r3), Some(&3));
        assert_eq!(heap.capacity(), 2, "no new slots allocated");
    }

    #[test]
    fn collect_all_unreachable() {
        let mut heap = GcHeap::new(1024);
        let _r1 = heap.alloc(1i64);
        let _r2 = heap.alloc(2i64);
        let _r3 = heap.alloc(3i64);
        heap.collect(&[]); // no roots
        assert_eq!(heap.live_count(), 0);
        assert_eq!(heap.free_list.len(), 3);
    }

    #[test]
    fn alloc_auto_triggers_collection() {
        let mut heap = GcHeap::new(2); // threshold = 2
        let r1 = heap.alloc(1i64);
        let _r2 = heap.alloc(2i64);
        // alloc_auto should trigger collection before allocating
        let r3 = heap.alloc_auto(3i64, &[r1]);
        // r1 survived (was a root), _r2 was collected
        assert_eq!(heap.get::<i64>(r1), Some(&1));
        assert_eq!(heap.get::<i64>(r3), Some(&3));
        // _r2 was swept
        assert_eq!(heap.live_count(), 2);
    }

    #[test]
    fn type_mismatch_returns_none() {
        let mut heap = GcHeap::new(1024);
        let r = heap.alloc(42i64);
        // Wrong type should return None
        assert_eq!(heap.get::<String>(r), None);
        // Correct type works
        assert_eq!(heap.get::<i64>(r), Some(&42));
    }

    #[test]
    fn mark_sweep_preserves_mark_reset() {
        let mut heap = GcHeap::new(1024);
        let r1 = heap.alloc(1i64);
        let r2 = heap.alloc(2i64);
        // First collection: keep both
        heap.collect(&[r1, r2]);
        assert_eq!(heap.live_count(), 2);
        // Second collection: keep only r1
        heap.collect(&[r1]);
        assert_eq!(heap.live_count(), 1);
        assert_eq!(heap.get::<i64>(r1), Some(&1));
    }
}

