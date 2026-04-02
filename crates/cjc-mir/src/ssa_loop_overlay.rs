//! SSA Loop/Reduction Overlay — Connects SSA definitions to loop structure
//!
//! This module builds an overlay that maps SSA variable definitions to the
//! loops they belong to, identifies loop-carried dependencies (variables
//! defined via phi nodes at loop headers), and cross-references reductions
//! with their SSA accumulator versions.
//!
//! ## Design decisions
//!
//! - **Additive overlay** — does not modify SSA, CFG, loop tree, or reductions
//! - **Vec + ID indexing** — uses existing `LoopId`, `BlockId`, `ReductionId`
//! - **Deterministic** — all iteration uses sorted structures (BTreeMap/BTreeSet)
//! - **Derived lazily** — computed on demand from SSA + loop tree + reductions
//!
//! ## Usage
//!
//! ```rust,ignore
//! use cjc_mir::ssa_loop_overlay::SsaLoopOverlay;
//!
//! let overlay = SsaLoopOverlay::compute(&ssa, &loop_tree, &reduction_report);
//! let carried = overlay.loop_carried_vars(LoopId(0));
//! ```

use std::collections::{BTreeMap, BTreeSet};

use crate::loop_analysis::{LoopId, LoopTree};
use crate::reduction::{ReductionId, ReductionReport};
use crate::ssa::{SsaForm, SsaVar};
use crate::BlockId;

// ---------------------------------------------------------------------------
// Overlay
// ---------------------------------------------------------------------------

/// SSA-level loop information overlay.
///
/// Connects three analyses — SSA form, loop tree, reduction report — into
/// a unified view that answers questions like:
///
/// - Which SSA variables are defined inside loop L?
/// - Which variables are loop-carried (phi at header with back-edge source)?
/// - Which reductions correspond to which loop-carried variables?
#[derive(Debug, Clone)]
pub struct SsaLoopOverlay {
    /// For each loop, the set of SSA variables defined within that loop's body.
    /// Indexed by `LoopId` → set of `(var_name, version)`.
    pub defs_by_loop: BTreeMap<u32, BTreeSet<(String, u32)>>,

    /// Loop-carried variables: SSA variables that have a phi node at a loop
    /// header with at least one operand from a back-edge source.
    /// Maps `LoopId` → vec of `(var_name, phi_target_version)`.
    pub loop_carried: BTreeMap<u32, Vec<(String, u32)>>,

    /// Cross-reference: for each reduction that was detected inside a loop,
    /// maps `ReductionId` → the SSA variable name that serves as accumulator.
    /// This allows connecting reduction analysis to SSA def-use chains.
    pub reduction_to_ssa_var: BTreeMap<u32, String>,
}

impl SsaLoopOverlay {
    /// Compute the overlay from SSA form, loop tree, and reduction report.
    ///
    /// ## Determinism
    ///
    /// All collections use BTreeMap/BTreeSet.  Iteration order is deterministic.
    pub fn compute(
        ssa: &SsaForm,
        loop_tree: &LoopTree,
        reduction_report: &ReductionReport,
    ) -> Self {
        let mut defs_by_loop: BTreeMap<u32, BTreeSet<(String, u32)>> = BTreeMap::new();
        let mut loop_carried: BTreeMap<u32, Vec<(String, u32)>> = BTreeMap::new();
        let reduction_to_ssa_var: BTreeMap<u32, String> = BTreeMap::new();

        // Initialize entries for each loop.
        for info in &loop_tree.loops {
            defs_by_loop.insert(info.id.0, BTreeSet::new());
            loop_carried.insert(info.id.0, Vec::new());
        }

        // Step 1: Map SSA definitions to loops.
        // Each entry in def_versions is (block_id, var_name, stmt_index) → version.
        for ((block_idx, var_name, _stmt_idx), version) in &ssa.def_versions {
            let block = BlockId(*block_idx);
            if let Some(loop_id) = loop_tree.loop_for_block(block) {
                if let Some(set) = defs_by_loop.get_mut(&loop_id.0) {
                    set.insert((var_name.clone(), *version));
                }
            }
        }

        // Step 2: Identify loop-carried variables via phi nodes at loop headers.
        for info in &loop_tree.loops {
            let header_idx = info.header.0 as usize;
            if header_idx >= ssa.phis.len() {
                continue;
            }
            let phis = &ssa.phis[header_idx];
            for phi in phis {
                // A phi is loop-carried if at least one source comes from a
                // back-edge (i.e., from a block inside the loop body).
                let has_back_edge_source = phi.sources.iter().any(|(pred_block, _)| {
                    info.back_edge_sources.binary_search(pred_block).is_ok()
                });
                if has_back_edge_source {
                    if let Some(carried) = loop_carried.get_mut(&info.id.0) {
                        carried.push((phi.target.name.clone(), phi.target.version));
                    }
                }
            }
            // Sort for determinism.
            if let Some(carried) = loop_carried.get_mut(&info.id.0) {
                carried.sort();
                carried.dedup();
            }
        }

        // Step 3: Cross-reference reductions with SSA accumulator names.
        // We match by accumulator_var name from the reduction report.
        let mut red_map = reduction_to_ssa_var;
        for r in &reduction_report.reductions {
            if !r.accumulator_var.is_empty() {
                red_map.insert(r.id.0, r.accumulator_var.clone());
            }
        }

        SsaLoopOverlay {
            defs_by_loop,
            loop_carried,
            reduction_to_ssa_var: red_map,
        }
    }

    /// Get all SSA variable definitions inside a loop.
    pub fn defs_in_loop(&self, loop_id: LoopId) -> Vec<(String, u32)> {
        self.defs_by_loop
            .get(&loop_id.0)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get loop-carried variables for a loop.
    pub fn loop_carried_vars(&self, loop_id: LoopId) -> &[(String, u32)] {
        self.loop_carried
            .get(&loop_id.0)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get the SSA accumulator variable name for a reduction, if known.
    pub fn ssa_var_for_reduction(&self, reduction_id: ReductionId) -> Option<&str> {
        self.reduction_to_ssa_var
            .get(&reduction_id.0)
            .map(|s| s.as_str())
    }

    /// Returns true if a variable is loop-carried in the given loop.
    pub fn is_loop_carried(&self, loop_id: LoopId, var_name: &str) -> bool {
        self.loop_carried
            .get(&loop_id.0)
            .map(|v| v.iter().any(|(name, _)| name == var_name))
            .unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduction::{
        AccumulatorSemantics, ReductionId, ReductionInfo, ReductionKind, ReductionOp,
        ReductionReport,
    };
    use crate::ssa::SsaForm;
    use crate::loop_analysis::{LoopId, LoopInfo, LoopTree, SchedulePlan};

    /// Create an empty SSA form for testing.
    fn empty_ssa(num_blocks: usize) -> SsaForm {
        SsaForm {
            phis: vec![Vec::new(); num_blocks],
            num_blocks,
            entry: BlockId(0),
            version_counts: BTreeMap::new(),
            def_versions: BTreeMap::new(),
            params: Vec::new(),
        }
    }

    /// Create an empty loop tree for testing.
    fn empty_loop_tree(num_blocks: usize) -> LoopTree {
        LoopTree {
            loops: Vec::new(),
            block_to_loop: vec![None; num_blocks],
            num_blocks,
        }
    }

    #[test]
    fn test_empty_overlay() {
        let ssa = empty_ssa(4);
        let loop_tree = empty_loop_tree(4);
        let report = ReductionReport {
            reductions: Vec::new(),
        };
        let overlay = SsaLoopOverlay::compute(&ssa, &loop_tree, &report);
        assert!(overlay.defs_by_loop.is_empty());
        assert!(overlay.loop_carried.is_empty());
        assert!(overlay.reduction_to_ssa_var.is_empty());
    }

    #[test]
    fn test_defs_in_loop() {
        let mut ssa = empty_ssa(4);
        // Simulate: block 1 defines "x" version 1, block 2 defines "y" version 0
        ssa.def_versions.insert((1, "x".to_string(), 0), 1);
        ssa.def_versions.insert((2, "y".to_string(), 0), 0);

        // Loop with body blocks [1, 2]
        let mut loop_tree = empty_loop_tree(4);
        loop_tree.loops.push(LoopInfo {
            id: LoopId(0),
            header: BlockId(1),
            body_blocks: vec![BlockId(1), BlockId(2)],
            back_edge_sources: vec![BlockId(2)],
            exit_blocks: vec![BlockId(3)],
            preheader: Some(BlockId(0)),
            parent: None,
            children: Vec::new(),
            depth: 0,
            is_countable: false,
            trip_count_hint: None,
            num_exits: 1,
            schedule: SchedulePlan::default(),
        });
        loop_tree.block_to_loop = vec![None, Some(LoopId(0)), Some(LoopId(0)), None];

        let report = ReductionReport {
            reductions: Vec::new(),
        };
        let overlay = SsaLoopOverlay::compute(&ssa, &loop_tree, &report);

        let defs = overlay.defs_in_loop(LoopId(0));
        assert_eq!(defs.len(), 2);
        assert!(defs.contains(&("x".to_string(), 1)));
        assert!(defs.contains(&("y".to_string(), 0)));
    }

    #[test]
    fn test_loop_carried_detection() {
        use crate::ssa::PhiNode;

        let mut ssa = empty_ssa(4);
        // Phi at block 1 (loop header): acc_1 = phi(acc_0 from block 0, acc_2 from block 2)
        ssa.phis[1].push(PhiNode {
            target: SsaVar {
                name: "acc".to_string(),
                version: 1,
            },
            sources: vec![
                (
                    BlockId(0),
                    SsaVar {
                        name: "acc".to_string(),
                        version: 0,
                    },
                ),
                (
                    BlockId(2),
                    SsaVar {
                        name: "acc".to_string(),
                        version: 2,
                    },
                ),
            ],
        });

        let mut loop_tree = empty_loop_tree(4);
        loop_tree.loops.push(LoopInfo {
            id: LoopId(0),
            header: BlockId(1),
            body_blocks: vec![BlockId(1), BlockId(2)],
            back_edge_sources: vec![BlockId(2)],
            exit_blocks: vec![BlockId(3)],
            preheader: Some(BlockId(0)),
            parent: None,
            children: Vec::new(),
            depth: 0,
            is_countable: false,
            trip_count_hint: None,
            num_exits: 1,
            schedule: SchedulePlan::default(),
        });
        loop_tree.block_to_loop = vec![None, Some(LoopId(0)), Some(LoopId(0)), None];

        let report = ReductionReport {
            reductions: Vec::new(),
        };
        let overlay = SsaLoopOverlay::compute(&ssa, &loop_tree, &report);

        assert!(overlay.is_loop_carried(LoopId(0), "acc"));
        assert!(!overlay.is_loop_carried(LoopId(0), "other"));

        let carried = overlay.loop_carried_vars(LoopId(0));
        assert_eq!(carried.len(), 1);
        assert_eq!(carried[0].0, "acc");
    }

    #[test]
    fn test_reduction_cross_reference() {
        let ssa = empty_ssa(4);
        let loop_tree = empty_loop_tree(4);
        let report = ReductionReport {
            reductions: vec![ReductionInfo {
                id: ReductionId(0),
                accumulator_var: "total".to_string(),
                op: ReductionOp::Add,
                kind: ReductionKind::StrictFold,
                loop_id: Some(LoopId(0)),
                function_name: "test".to_string(),
                builtin_name: None,
                reassociation_forbidden: true,
                strict_order_required: true,
                accumulator_semantics: AccumulatorSemantics::Plain,
            }],
        };
        let overlay = SsaLoopOverlay::compute(&ssa, &loop_tree, &report);

        assert_eq!(
            overlay.ssa_var_for_reduction(ReductionId(0)),
            Some("total")
        );
        assert_eq!(overlay.ssa_var_for_reduction(ReductionId(1)), None);
    }
}
