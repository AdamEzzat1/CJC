//! COW (Copy-on-Write) buffer -- the fundamental memory primitive under [`Tensor`].
//!
//! [`Buffer<T>`] provides reference-counted storage with lazy deep-copy
//! semantics. Cloning a buffer increments the refcount (O(1)); mutation
//! via [`Buffer::set`] or [`Buffer::make_unique`] triggers a deep copy
//! only when the buffer is shared (`refcount > 1`).
//!
//! # Determinism
//!
//! Buffer operations are fully deterministic. No randomized hashing,
//! no platform-dependent allocation strategies. The COW mechanism
//! ensures that mutation of shared data always produces a fresh,
//! independent copy.
//!
//! [`Tensor`]: crate::tensor::Tensor

use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

use crate::error::RuntimeError;

// ---------------------------------------------------------------------------
// 1. Buffer<T> -- Deterministic memory allocation with COW semantics
// ---------------------------------------------------------------------------

/// A reference-counted buffer with copy-on-write semantics.
///
/// Internally backed by `Rc<RefCell<Vec<T>>>`. When multiple `Buffer`s share
/// the same underlying storage, a mutation via [`Buffer::set`] or
/// [`Buffer::make_unique`] triggers a deep copy so that other holders are
/// unaffected.
#[derive(Debug)]
pub struct Buffer<T: Clone> {
    inner: Rc<RefCell<Vec<T>>>,
}

impl<T: Clone> Buffer<T> {
    /// Allocate a buffer of `len` elements, each initialized to `default`.
    pub fn alloc(len: usize, default: T) -> Self {
        Buffer {
            inner: Rc::new(RefCell::new(vec![default; len])),
        }
    }

    /// Create a buffer from an existing `Vec<T>`.
    pub fn from_vec(data: Vec<T>) -> Self {
        Buffer {
            inner: Rc::new(RefCell::new(data)),
        }
    }

    /// Number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.inner.borrow().len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.borrow().is_empty()
    }

    /// Read the value at `idx`. Returns `None` if out of bounds.
    pub fn get(&self, idx: usize) -> Option<T> {
        self.inner.borrow().get(idx).cloned()
    }

    /// Write `val` at `idx`. If the buffer is shared (refcount > 1), a deep
    /// copy is performed first (COW). Returns `Err` if `idx` is out of bounds.
    pub fn set(&mut self, idx: usize, val: T) -> Result<(), RuntimeError> {
        self.make_unique();
        let mut vec = self.inner.borrow_mut();
        if idx >= vec.len() {
            return Err(RuntimeError::IndexOutOfBounds {
                index: idx,
                length: vec.len(),
            });
        }
        vec[idx] = val;
        Ok(())
    }

    /// Return a snapshot of the data as a `Vec<T>`.
    pub fn as_slice(&self) -> Vec<T> {
        self.inner.borrow().clone()
    }

    /// Borrow the underlying Vec without cloning.
    /// The returned `Ref` guard keeps the borrow alive.
    pub fn borrow_data(&self) -> Ref<Vec<T>> {
        self.inner.borrow()
    }

    /// Mutably borrow the underlying Vec without cloning.
    /// Call `make_unique()` first if the buffer may be shared.
    pub fn borrow_data_mut(&self) -> RefMut<Vec<T>> {
        self.inner.borrow_mut()
    }

    /// Force a deep copy, returning a new `Buffer` that does not share
    /// storage with `self`.
    pub fn clone_buffer(&self) -> Buffer<T> {
        Buffer {
            inner: Rc::new(RefCell::new(self.inner.borrow().clone())),
        }
    }

    /// Ensure this `Buffer` has exclusive ownership of the underlying data.
    /// If the refcount is > 1, the data is deep-copied and this `Buffer` is
    /// re-pointed to the fresh copy.
    pub fn make_unique(&mut self) {
        if Rc::strong_count(&self.inner) > 1 {
            let data = self.inner.borrow().clone();
            self.inner = Rc::new(RefCell::new(data));
        }
    }

    /// Number of live references to the underlying storage.
    pub fn refcount(&self) -> usize {
        Rc::strong_count(&self.inner)
    }
}

impl<T: Clone> Clone for Buffer<T> {
    /// Cloning a `Buffer` increments the refcount — it does NOT copy data.
    fn clone(&self) -> Self {
        Buffer {
            inner: Rc::clone(&self.inner),
        }
    }
}

