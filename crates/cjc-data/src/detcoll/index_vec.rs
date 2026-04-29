//! `IndexVec<I, V>` — dense ID → value table.
//!
//! Strongly-typed `Vec` indexed by a newtype ID instead of `usize`. Use
//! for "every entity gets a sequentially-allocated ID" patterns:
//! `SymbolId → SymbolData`, `NodeId → NodeData`, `TypeId → TypeInfo`,
//! `BlockId → BasicBlock`. Lookup is `O(1)` array access; iteration is
//! deterministic insertion order.
//!
//! No hashing, no probing, no allocator overhead beyond `Vec` growth.
//! For dense IDs this is strictly better than `BTreeMap<u32, V>`:
//! constant-time lookup, contiguous memory, zero indirection.

use std::marker::PhantomData;

/// Newtype ID trait. Implement for any newtype wrapping a small
/// non-negative integer (`u32` is the conventional choice).
pub trait Idx: Copy + Eq + std::fmt::Debug {
    fn from_usize(i: usize) -> Self;
    fn index(self) -> usize;
}

/// Convenience macro to declare a newtype ID.
#[macro_export]
macro_rules! det_idx {
    ($name:ident) => {
        #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
        pub struct $name(pub u32);
        impl $crate::detcoll::Idx for $name {
            #[inline]
            fn from_usize(i: usize) -> Self {
                debug_assert!(i <= u32::MAX as usize);
                Self(i as u32)
            }
            #[inline]
            fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

#[derive(Debug, Clone)]
pub struct IndexVec<I: Idx, V> {
    data: Vec<V>,
    _marker: PhantomData<I>,
}

impl<I: Idx, V> Default for IndexVec<I, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Idx, V> IndexVec<I, V> {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            _marker: PhantomData,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            data: Vec::with_capacity(cap),
            _marker: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Append a value, returning the assigned ID. Allocation is
    /// monotonic — IDs are stable for the lifetime of this `IndexVec`.
    pub fn push(&mut self, value: V) -> I {
        let i = self.data.len();
        self.data.push(value);
        I::from_usize(i)
    }

    /// `O(1)` lookup. Returns `None` for out-of-range IDs.
    pub fn get(&self, id: I) -> Option<&V> {
        self.data.get(id.index())
    }

    /// `O(1)` mutable lookup. Returns `None` for out-of-range IDs.
    pub fn get_mut(&mut self, id: I) -> Option<&mut V> {
        self.data.get_mut(id.index())
    }

    /// Iterate `(id, &value)` pairs in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (I, &V)> + '_ {
        self.data
            .iter()
            .enumerate()
            .map(|(i, v)| (I::from_usize(i), v))
    }

    /// Iterate `&value` only, in insertion order.
    pub fn values(&self) -> impl Iterator<Item = &V> + '_ {
        self.data.iter()
    }

    /// Borrow as a slice, for code that wants positional access.
    pub fn as_slice(&self) -> &[V] {
        &self.data
    }
}

impl<I: Idx, V> std::ops::Index<I> for IndexVec<I, V> {
    type Output = V;
    fn index(&self, id: I) -> &V {
        &self.data[id.index()]
    }
}

impl<I: Idx, V> std::ops::IndexMut<I> for IndexVec<I, V> {
    fn index_mut(&mut self, id: I) -> &mut V {
        &mut self.data[id.index()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::det_idx!(NodeId);

    #[test]
    fn push_and_lookup_roundtrip() {
        let mut iv: IndexVec<NodeId, &'static str> = IndexVec::new();
        let a = iv.push("alpha");
        let b = iv.push("beta");
        let c = iv.push("gamma");
        assert_eq!(a, NodeId(0));
        assert_eq!(b, NodeId(1));
        assert_eq!(c, NodeId(2));
        assert_eq!(iv[a], "alpha");
        assert_eq!(iv[b], "beta");
        assert_eq!(iv[c], "gamma");
    }

    #[test]
    fn out_of_range_get_returns_none() {
        let iv: IndexVec<NodeId, i32> = IndexVec::new();
        assert!(iv.get(NodeId(0)).is_none());
    }

    #[test]
    fn iter_yields_insertion_order() {
        let mut iv: IndexVec<NodeId, i32> = IndexVec::new();
        iv.push(10);
        iv.push(20);
        iv.push(30);
        let collected: Vec<i32> = iv.values().copied().collect();
        assert_eq!(collected, vec![10, 20, 30]);
    }

    #[test]
    fn deterministic_under_repeat() {
        let mut a: IndexVec<NodeId, i32> = IndexVec::new();
        let mut b: IndexVec<NodeId, i32> = IndexVec::new();
        for v in [3, 1, 4, 1, 5, 9, 2, 6] {
            a.push(v);
            b.push(v);
        }
        let av: Vec<i32> = a.values().copied().collect();
        let bv: Vec<i32> = b.values().copied().collect();
        assert_eq!(av, bv);
    }
}
