//! Per-function reduction-kind histogram.
//!
//! Reductions are the *single most important determinism signal* in CJC-Lang.
//! `cjc-mir::reduction` already classifies every accumulation and builtin
//! reduction call by [`ReductionKind`]; CANA simply counts them per function
//! so the legality gate can answer the question:
//!
//! > *"Would reordering or parallelizing the passes inside this function
//! > violate any reduction's stated semantics?"*
//!
//! The histogram is the **only** floating-point-adjacent feature CANA exposes
//! in Phase 1 (and even here it's just integer counts of strict reductions).
//! Phase 2+ uses the counts as inputs to the legality gate and (later) to a
//! "should I attempt parallel lowering of this function?" cost-model query.

use cjc_mir::reduction::{ReductionInfo, ReductionKind, ReductionReport};

use crate::hash::CanaHasher;

/// Histogram of `ReductionKind` variants found in one function.
///
/// All counts default to zero. The order of fields here is canonical for
/// hashing — never reorder existing fields, only append new ones (with a new
/// `ReductionKind` variant).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReductionAxes {
    /// `StrictFold` — sequential left fold, order-dependent.
    pub strict_fold: u32,
    /// `KahanFold` — compensated, order-dependent, numerically stable.
    pub kahan_fold: u32,
    /// `BinnedFold` — reorderable within bin capacity, parallelizable.
    pub binned_fold: u32,
    /// `FixedTree` — fixed-shape reduction tree, parallel along branches.
    pub fixed_tree: u32,
    /// `BuiltinReduction` — `sum`/`mean`/`dot` calls; runtime decides.
    pub builtin_reduction: u32,
    /// `Unknown` — conservative; no reorder allowed.
    pub unknown: u32,
}

impl ReductionAxes {
    /// Build a histogram for one function by filtering a `ReductionReport`.
    pub fn from_report_for_fn(report: &ReductionReport, fn_name: &str) -> Self {
        let mut axes = Self::default();
        for ri in &report.reductions {
            if ri.function_name == fn_name {
                axes.bump(ri);
            }
        }
        axes
    }

    fn bump(&mut self, ri: &ReductionInfo) {
        match ri.kind {
            ReductionKind::StrictFold => self.strict_fold = self.strict_fold.saturating_add(1),
            ReductionKind::KahanFold => self.kahan_fold = self.kahan_fold.saturating_add(1),
            ReductionKind::BinnedFold => self.binned_fold = self.binned_fold.saturating_add(1),
            ReductionKind::FixedTree => self.fixed_tree = self.fixed_tree.saturating_add(1),
            ReductionKind::BuiltinReduction => {
                self.builtin_reduction = self.builtin_reduction.saturating_add(1);
            }
            ReductionKind::Unknown => self.unknown = self.unknown.saturating_add(1),
        }
    }

    /// Total reductions across all kinds.
    pub fn total(&self) -> u32 {
        self.strict_fold
            .saturating_add(self.kahan_fold)
            .saturating_add(self.binned_fold)
            .saturating_add(self.fixed_tree)
            .saturating_add(self.builtin_reduction)
            .saturating_add(self.unknown)
    }

    /// Count of reductions whose semantics forbid reordering. The legality
    /// gate uses this to decide whether a reorder-class pass is safe to
    /// recommend on this function.
    pub fn strict_count(&self) -> u32 {
        // Mirrors `ReductionKind::is_strict()`: StrictFold + KahanFold + Unknown.
        self.strict_fold
            .saturating_add(self.kahan_fold)
            .saturating_add(self.unknown)
    }

    /// Count of reductions safe to reorder (`BinnedFold` only at present).
    pub fn reorderable_count(&self) -> u32 {
        self.binned_fold
    }

    /// `true` if this function contains *any* reduction the legality gate
    /// would refuse to let the optimizer reorder.
    pub fn has_strict_reduction(&self) -> bool {
        self.strict_count() > 0
    }

    pub(crate) fn feed(&self, hasher: &mut CanaHasher) {
        hasher.write_tag(TAG_REDUCTION_AXES);
        hasher.write_u32(self.strict_fold);
        hasher.write_u32(self.kahan_fold);
        hasher.write_u32(self.binned_fold);
        hasher.write_u32(self.fixed_tree);
        hasher.write_u32(self.builtin_reduction);
        hasher.write_u32(self.unknown);
    }
}

const TAG_REDUCTION_AXES: u8 = 0xB0;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_mir::loop_analysis::LoopId;
    use cjc_mir::reduction::{
        AccumulatorSemantics, ReductionId, ReductionInfo, ReductionKind, ReductionOp,
        ReductionReport,
    };

    fn ri(fn_name: &str, kind: ReductionKind, id: u32) -> ReductionInfo {
        ReductionInfo {
            id: ReductionId(id),
            accumulator_var: "acc".to_string(),
            op: ReductionOp::Add,
            kind,
            loop_id: Some(LoopId(0)),
            function_name: fn_name.to_string(),
            builtin_name: None,
            reassociation_forbidden: matches!(
                kind,
                ReductionKind::StrictFold | ReductionKind::KahanFold | ReductionKind::Unknown
            ),
            strict_order_required: matches!(
                kind,
                ReductionKind::StrictFold | ReductionKind::KahanFold
            ),
            accumulator_semantics: match kind {
                ReductionKind::KahanFold => AccumulatorSemantics::Kahan,
                ReductionKind::BinnedFold => AccumulatorSemantics::Binned,
                _ => AccumulatorSemantics::Plain,
            },
        }
    }

    #[test]
    fn empty_report_yields_zero_axes() {
        let report = ReductionReport { reductions: vec![] };
        let axes = ReductionAxes::from_report_for_fn(&report, "f");
        assert_eq!(axes.total(), 0);
        assert_eq!(axes.strict_count(), 0);
        assert!(!axes.has_strict_reduction());
    }

    #[test]
    fn filter_by_function_name() {
        let report = ReductionReport {
            reductions: vec![
                ri("f", ReductionKind::StrictFold, 0),
                ri("g", ReductionKind::StrictFold, 1),
                ri("f", ReductionKind::BinnedFold, 2),
            ],
        };
        let f_axes = ReductionAxes::from_report_for_fn(&report, "f");
        assert_eq!(f_axes.strict_fold, 1);
        assert_eq!(f_axes.binned_fold, 1);
        assert_eq!(f_axes.total(), 2);

        let g_axes = ReductionAxes::from_report_for_fn(&report, "g");
        assert_eq!(g_axes.strict_fold, 1);
        assert_eq!(g_axes.total(), 1);
    }

    #[test]
    fn strict_count_matches_kind_is_strict() {
        let report = ReductionReport {
            reductions: vec![
                ri("f", ReductionKind::StrictFold, 0),
                ri("f", ReductionKind::KahanFold, 1),
                ri("f", ReductionKind::Unknown, 2),
                ri("f", ReductionKind::BinnedFold, 3),
                ri("f", ReductionKind::FixedTree, 4),
                ri("f", ReductionKind::BuiltinReduction, 5),
            ],
        };
        let axes = ReductionAxes::from_report_for_fn(&report, "f");
        // 3 strict (StrictFold + KahanFold + Unknown), 1 reorderable (Binned),
        // 1 fixed-tree, 1 builtin — total 6.
        assert_eq!(axes.strict_count(), 3);
        assert_eq!(axes.reorderable_count(), 1);
        assert_eq!(axes.total(), 6);
        assert!(axes.has_strict_reduction());
    }

    #[test]
    fn axes_are_clone_copy() {
        // Compile-time check.
        fn _f<T: Copy + Clone>(_: T) {}
        _f(ReductionAxes::default());
    }
}
