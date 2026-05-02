//! Phase 2b — typed-ID newtype(s) for ML metadata in `cjc-runtime`.
//!
//! Currently provides only [`ParamIdx`], used by `crate::ml::AdamState`
//! and `crate::ml::SgdState` to type optimizer-state buffers.
//!
//! ## Why a separate file from `cjc-ad/src/idx.rs`?
//!
//! Phase 2a (PR #8) introduced `NodeIdx`, `ParamIdx`, `LayerIdx` in
//! `cjc-ad/src/idx.rs`. Architecturally those newtypes belong in
//! `cjc-runtime` (the foundation crate, upstream of `cjc-ad`), but
//! moving them at this stage would conflict with Phase 2a's open PR
//! diff. Instead, this Phase 2b PR defines `ParamIdx` *locally* in
//! `cjc-runtime` for the optimizer-state migration; a future
//! "Phase 2 cleanup" PR will consolidate the typed-ID newtypes in
//! one canonical home.
//!
//! Until then, `cjc_runtime::idx::ParamIdx` and
//! `cjc_ad::idx::ParamIdx` are distinct types with identical shape.
//! They are not used in the same code today, so the duplication is
//! harmless.
//!
//! ## Repr
//!
//! `repr(transparent)` over `u32`: ABI-identical to `u32`, zero
//! runtime overhead, FFI-stable bit pattern.

/// Index into a per-parameter optimizer-state buffer.
///
/// The optimizer enumerates trainable parameters in registration order.
/// `ParamIdx(0)` always refers to the first registered parameter,
/// `ParamIdx(1)` to the second, etc. This is *not* a graph-node index
/// (those are `cjc_ad::NodeIdx`); the two integer spaces are unrelated
/// even though both number from zero.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_usize_roundtrip() {
        for i in [0usize, 1, 7, 64, 1024] {
            let p = ParamIdx::from_usize(i);
            assert_eq!(p.index(), i);
            assert_eq!(usize::from(p), i);
        }
    }

    #[test]
    fn ord_matches_inner() {
        assert!(ParamIdx(0) < ParamIdx(1));
        assert!(ParamIdx(7) < ParamIdx(8));
    }

    #[test]
    fn repr_transparent_means_size_eq_u32() {
        assert_eq!(std::mem::size_of::<ParamIdx>(), std::mem::size_of::<u32>());
    }

    #[test]
    fn display_format() {
        assert_eq!(format!("{}", ParamIdx(42)), "42");
    }
}
