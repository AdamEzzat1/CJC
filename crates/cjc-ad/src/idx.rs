//! Phase 2 — typed-ID newtypes for autodiff metadata.
//!
//! Three small `repr(transparent)` wrappers over `u32` that stop us from
//! accidentally mixing up the three logical kinds of integer index that
//! pervade the AD layer:
//!
//!   - [`NodeIdx`]: an index into the `GradGraph`'s parallel arrays
//!     (`ops`, `tensors`, `param_grads`). Every node in the autodiff
//!     tape — input, parameter, op, fused layer — lives at one
//!     `NodeIdx`.
//!   - [`ParamIdx`]: an index into a per-parameter optimizer-state
//!     buffer (Adam's `m`/`v`, SGD's velocity). Distinct from `NodeIdx`
//!     because the optimizer enumerates trainable parameters in its own
//!     order, which need not match graph-node insertion order.
//!   - [`LayerIdx`]: an index into the architecture description (e.g.,
//!     `pinn::MlpArch::layers`). Distinct from `NodeIdx` because each
//!     layer *contains* node indices for its weight, bias, and output.
//!
//! Why all three? The previous `usize`-everywhere API made it easy to
//! pass a parameter index into a method expecting a node index — both
//! are `usize`, both are valid for many ops, but the value flows mean
//! something completely different. Typed IDs catch this at compile
//! time. See `docs/ml_training/PHASE_2_AUDIT.md` for the survey of
//! sites that motivated this PR.
//!
//! ### Storage choice
//!
//! Phase 2 keeps `GradGraph`'s internal storage as `Vec<T>` (with
//! `tensors[idx.0 as usize]` access) rather than introducing
//! `IndexVec<NodeIdx, T>`. The audit's "use IndexVec" recommendation
//! was an implementation detail; the actual goal is typed IDs at the
//! API boundary, which `Vec<T>` + a newtype delivers with zero
//! cross-crate plumbing. A future PR can lift `IndexVec` from
//! `cjc-data` to `cjc-runtime` and migrate the storage if the
//! ergonomic case is made.
//!
//! ### Repr
//!
//! All three are `repr(transparent)` over `u32`. That guarantees the
//! ABI is identical to `u32`, so any future C / FFI / bytecode users
//! see the same bit pattern.

/// Index into the GradGraph's parallel arrays. Returned by every node-
/// constructing method (`input`, `parameter`, `add`, `mul`, …) and
/// consumed by every node-referencing method.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct NodeIdx(pub u32);

impl NodeIdx {
    #[inline]
    pub fn from_usize(i: usize) -> Self {
        debug_assert!(i <= u32::MAX as usize, "NodeIdx overflow");
        Self(i as u32)
    }
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<NodeIdx> for usize {
    #[inline]
    fn from(idx: NodeIdx) -> Self {
        idx.0 as usize
    }
}

impl From<NodeIdx> for i64 {
    #[inline]
    fn from(idx: NodeIdx) -> Self {
        idx.0 as i64
    }
}

impl std::fmt::Display for NodeIdx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Index into a per-parameter optimizer-state buffer. The optimizer
/// enumerates trainable parameters in registration order; this index
/// is *not* a `NodeIdx` even though both number from zero.
///
/// Used by `cjc_runtime::ml::AdamState::{m, v}` and the SGD velocity
/// buffer.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ParamIdx(pub u32);

impl ParamIdx {
    #[inline]
    pub fn from_usize(i: usize) -> Self {
        debug_assert!(i <= u32::MAX as usize, "ParamIdx overflow");
        Self(i as u32)
    }
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<ParamIdx> for usize {
    #[inline]
    fn from(idx: ParamIdx) -> Self {
        idx.0 as usize
    }
}

impl std::fmt::Display for ParamIdx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Index into a layered architecture description. Each layer typically
/// holds three `NodeIdx` values (weight, bias, output) — they live at
/// different conceptual levels.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct LayerIdx(pub u32);

impl LayerIdx {
    #[inline]
    pub fn from_usize(i: usize) -> Self {
        debug_assert!(i <= u32::MAX as usize, "LayerIdx overflow");
        Self(i as u32)
    }
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<LayerIdx> for usize {
    #[inline]
    fn from(idx: LayerIdx) -> Self {
        idx.0 as usize
    }
}

impl std::fmt::Display for LayerIdx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newtype_roundtrip() {
        let n = NodeIdx::from_usize(42);
        assert_eq!(n.index(), 42);
        assert_eq!(usize::from(n), 42);
        assert_eq!(i64::from(n), 42);
    }

    #[test]
    fn distinct_types_dont_mix() {
        // This is a compile-time test; we just sanity-check the Debug
        // and equality at runtime.
        let n = NodeIdx(7);
        let p = ParamIdx(7);
        let l = LayerIdx(7);
        assert_eq!(n.0, p.0);
        assert_eq!(p.0, l.0);
        // n == p would be a type error if uncommented:
        // let _ = n == p;
        let _ = format!("{n} {p} {l}");
    }

    #[test]
    fn repr_transparent_means_size_eq_u32() {
        assert_eq!(std::mem::size_of::<NodeIdx>(), std::mem::size_of::<u32>());
        assert_eq!(std::mem::size_of::<ParamIdx>(), std::mem::size_of::<u32>());
        assert_eq!(std::mem::size_of::<LayerIdx>(), std::mem::size_of::<u32>());
    }
}
