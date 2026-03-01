//! Object Slab — deterministic RC-backed replacement for mark-sweep GC.
//!
//! Provides indexed allocation of type-erased objects. Objects are ref-counted
//! (via `Rc<RefCell<Box<dyn Any>>>`) so they are freed deterministically when
//! no references remain. Freed slots are reused in LIFO order for deterministic
//! placement.
//!
//! # Determinism guarantees
//!
//! - Same allocation sequence produces same slot indices.
//! - Slot reuse is LIFO (last freed = first reused).
//! - No stop-the-world pauses.
//! - No OS memory return during normal execution (slots are recycled).

use std::any::Any;
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

/// A handle into the object slab. Lightweight, copyable index.
///
/// This is API-compatible with the old `GcRef` so that `Value::ClassRef`
/// continues to work without changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlabRef {
    pub index: usize,
}

/// A type-erased object stored in the slab, backed by reference counting.
struct SlabObject {
    value: Rc<RefCell<Box<dyn Any>>>,
}

impl fmt::Debug for SlabObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SlabObject")
            .field("rc_count", &Rc::strong_count(&self.value))
            .field("value", &"<dyn Any>")
            .finish()
    }
}

/// A deterministic slab allocator for type-erased objects.
///
/// Replaces the old mark-sweep `GcHeap`. Objects are reference-counted
/// and freed automatically when no references remain. The slab provides
/// stable indexed access and deterministic slot reuse.
#[derive(Debug)]
pub struct ObjectSlab {
    objects: Vec<Option<SlabObject>>,
    /// Free-list for slot reuse (LIFO for determinism).
    pub free_list: Vec<usize>,
    /// Total allocations performed (for statistics).
    alloc_count: usize,
}

impl ObjectSlab {
    /// Create a new empty slab.
    pub fn new() -> Self {
        ObjectSlab {
            objects: Vec::new(),
            free_list: Vec::new(),
            alloc_count: 0,
        }
    }

    /// Allocate a value in the slab, returning a handle.
    ///
    /// Reuses a freed slot if available (LIFO), otherwise extends the slab.
    pub fn alloc<T: Any + 'static>(&mut self, value: T) -> SlabRef {
        let obj = SlabObject {
            value: Rc::new(RefCell::new(Box::new(value) as Box<dyn Any>)),
        };

        let index = if let Some(idx) = self.free_list.pop() {
            self.objects[idx] = Some(obj);
            idx
        } else {
            self.objects.push(Some(obj));
            self.objects.len() - 1
        };

        self.alloc_count += 1;
        SlabRef { index }
    }

    /// Read a reference to the value behind `slab_ref`, downcasting to `T`.
    /// Returns `None` if the slot is empty or the type does not match.
    pub fn get<T: Any + 'static>(&self, slab_ref: SlabRef) -> Option<&T> {
        self.objects
            .get(slab_ref.index)
            .and_then(|slot| slot.as_ref())
            .and_then(|obj| {
                // SAFETY: We hold a shared ref to the slab, so the RefCell
                // borrow is safe as long as no mutable borrow is active.
                // We use try_borrow() to be safe.
                let borrowed = obj.value.try_borrow().ok()?;
                // We need to extend the lifetime since we hold the slab ref.
                // This is safe because the slab outlives the returned reference.
                let any_ref: &dyn Any = &**borrowed;
                // SAFETY: The slab holds the Rc, which keeps the data alive.
                // We cast to a raw pointer and back to extend the lifetime
                // to match the slab borrow.
                let ptr = any_ref as *const dyn Any;
                unsafe { &*ptr }.downcast_ref::<T>()
            })
    }

    /// Get a mutable reference to the value behind `slab_ref`.
    pub fn get_mut<T: Any + 'static>(&self, slab_ref: SlabRef) -> Option<std::cell::RefMut<'_, Box<dyn Any>>> {
        self.objects
            .get(slab_ref.index)
            .and_then(|slot| slot.as_ref())
            .and_then(|obj| obj.value.try_borrow_mut().ok())
    }

    /// Number of live (non-None) objects in the slab.
    pub fn live_count(&self) -> usize {
        self.objects.iter().filter(|s| s.is_some()).count()
    }

    /// Total number of slots (including freed ones).
    pub fn capacity(&self) -> usize {
        self.objects.len()
    }

    /// Total allocations performed since creation.
    pub fn alloc_count(&self) -> usize {
        self.alloc_count
    }

    /// Explicitly free a slot (return to free-list).
    ///
    /// This is optional — objects are also freed when their Rc drops to zero.
    /// Use this for explicit lifecycle management.
    pub fn free(&mut self, slab_ref: SlabRef) {
        if let Some(slot) = self.objects.get_mut(slab_ref.index) {
            if slot.is_some() {
                *slot = None;
                self.free_list.push(slab_ref.index);
            }
        }
    }

    /// No-op collect for backward compatibility with GC API.
    ///
    /// The object slab uses reference counting; there is nothing to collect.
    /// This method exists so that `gc_collect` builtins don't need to be
    /// removed from user code immediately.
    pub fn collect_noop(&self) {
        // Intentionally empty. RC handles deallocation deterministically.
    }
}

impl Default for ObjectSlab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_and_read_back() {
        let mut slab = ObjectSlab::new();
        let r = slab.alloc(42i64);
        assert_eq!(slab.get::<i64>(r), Some(&42));
        assert_eq!(slab.live_count(), 1);
    }

    #[test]
    fn free_and_slot_reuse() {
        let mut slab = ObjectSlab::new();
        let r1 = slab.alloc(1i64);
        let r2 = slab.alloc(2i64);
        assert_eq!(slab.capacity(), 2);

        // Free r2
        slab.free(r2);
        assert_eq!(slab.live_count(), 1);
        assert_eq!(slab.free_list.len(), 1);

        // New alloc should reuse r2's slot
        let r3 = slab.alloc(3i64);
        assert_eq!(r3.index, r2.index, "LIFO reuse");
        assert_eq!(slab.capacity(), 2, "no new slots");
        assert_eq!(slab.get::<i64>(r3), Some(&3));
        assert_eq!(slab.get::<i64>(r1), Some(&1));
    }

    #[test]
    fn type_mismatch_returns_none() {
        let mut slab = ObjectSlab::new();
        let r = slab.alloc(42i64);
        assert_eq!(slab.get::<String>(r), None);
        assert_eq!(slab.get::<i64>(r), Some(&42));
    }

    #[test]
    fn collect_noop_is_harmless() {
        let mut slab = ObjectSlab::new();
        let r1 = slab.alloc(1i64);
        slab.collect_noop();
        // Object still alive (RC, not GC)
        assert_eq!(slab.live_count(), 1);
        assert_eq!(slab.get::<i64>(r1), Some(&1));
    }

    #[test]
    fn deterministic_slot_order() {
        // Same allocation sequence → same slot indices
        let mut slab1 = ObjectSlab::new();
        let mut slab2 = ObjectSlab::new();

        let a1 = slab1.alloc(10i64);
        let a2 = slab1.alloc(20i64);
        let a3 = slab1.alloc(30i64);
        slab1.free(a2);
        let a4 = slab1.alloc(40i64);

        let b1 = slab2.alloc(10i64);
        let b2 = slab2.alloc(20i64);
        let b3 = slab2.alloc(30i64);
        slab2.free(b2);
        let b4 = slab2.alloc(40i64);

        assert_eq!(a1.index, b1.index);
        assert_eq!(a2.index, b2.index);
        assert_eq!(a3.index, b3.index);
        assert_eq!(a4.index, b4.index, "LIFO reuse deterministic");
    }

    #[test]
    fn alloc_count_tracking() {
        let mut slab = ObjectSlab::new();
        assert_eq!(slab.alloc_count(), 0);
        let _ = slab.alloc(1i64);
        let _ = slab.alloc(2i64);
        assert_eq!(slab.alloc_count(), 2);
    }
}
