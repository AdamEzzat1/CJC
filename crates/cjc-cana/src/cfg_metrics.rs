//! Per-function CFG + loop metrics.
//!
//! Wraps the existing `cjc-mir::cfg`, `cjc-mir::dominators`, and
//! `cjc-mir::loop_analysis` surface. The featurizer doesn't reinvent any
//! analysis — it counts what's already there.
//!
//! ## What we expose
//!
//! For each function, a small integer-valued struct:
//!
//! | Field                   | Source                                      |
//! |-------------------------|---------------------------------------------|
//! | `block_count`           | `cfg.basic_blocks.len()`                    |
//! | `edge_count`            | Σ `terminator.successors().len()`           |
//! | `branch_count`          | # `Terminator::Branch`                      |
//! | `return_count`          | # `Terminator::Return`                      |
//! | `unreachable_count`     | # `Terminator::Unreachable`                 |
//! | `goto_count`            | # `Terminator::Goto`                        |
//! | `max_branch_factor`     | max `successors().len()` (0, 1, or 2)       |
//! | `loop_count`            | `loop_tree.len()`                           |
//! | `max_loop_depth`        | max `LoopInfo::depth + 1` (0 if no loops)   |
//! | `back_edge_count`       | Σ `loop_info.back_edge_sources.len()`       |
//! | `countable_loop_count`  | # loops with `is_countable == true`         |
//! | `cfg_hash`              | FNV-1a over canonical CFG shape             |
//!
//! ## What we do NOT expose
//!
//! - Anything float-valued. Phase 1 stays integer-only so the feature struct
//!   is trivially hashable and bit-identical across platforms.
//! - Any expression *contents* — only structural shape. Two CFGs whose
//!   constant-folded equivalents are identical will hash identically, which
//!   is what we want for caching cost-model predictions.

use cjc_mir::cfg::{MirCfg, Terminator};
use cjc_mir::dominators::DominatorTree;
use cjc_mir::loop_analysis::{compute_loop_tree, LoopTree};
use cjc_mir::MirFunction;

use crate::hash::{CanaHasher, CfgHash};

/// Integer-valued shape metrics for a single MIR function's CFG + loop tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CfgMetrics {
    /// Number of basic blocks (always ≥ 1).
    pub block_count: u32,
    /// Total number of CFG edges (sum of `successors().len()` over all blocks).
    pub edge_count: u32,
    /// Number of conditional-branch terminators.
    pub branch_count: u32,
    /// Number of return terminators.
    pub return_count: u32,
    /// Number of unreachable terminators (dead code marker).
    pub unreachable_count: u32,
    /// Number of unconditional-goto terminators.
    pub goto_count: u32,
    /// Maximum out-degree of any block (0, 1, or 2 in current MIR).
    pub max_branch_factor: u32,
    /// Number of natural loops detected.
    pub loop_count: u32,
    /// Maximum loop nesting depth (0 if no loops; otherwise `max(depth) + 1`).
    pub max_loop_depth: u32,
    /// Total number of back-edges across all loops.
    pub back_edge_count: u32,
    /// Number of loops with a statically known trip count.
    pub countable_loop_count: u32,
    /// FNV-1a hash of the canonical CFG shape (terminator kinds + edges).
    pub cfg_hash: CfgHash,
}

impl CfgMetrics {
    /// Extract metrics from a fully built CFG + loop tree.
    pub fn from_cfg(cfg: &MirCfg, loops: &LoopTree) -> Self {
        let mut edge_count = 0u32;
        let mut branch_count = 0u32;
        let mut return_count = 0u32;
        let mut unreachable_count = 0u32;
        let mut goto_count = 0u32;
        let mut max_branch_factor = 0u32;

        let mut hasher = CanaHasher::new();
        // Tag the type so a CFG hash never collides with a raw FeatureHash.
        hasher.write_tag(0xC0); // 'C' for CFG
        hasher.write_u32(cfg.basic_blocks.len() as u32);

        for bb in &cfg.basic_blocks {
            let succs = bb.terminator.successors();
            let n_succ = succs.len() as u32;
            edge_count = edge_count.saturating_add(n_succ);
            if n_succ > max_branch_factor {
                max_branch_factor = n_succ;
            }

            match &bb.terminator {
                Terminator::Goto(_) => {
                    goto_count = goto_count.saturating_add(1);
                    hasher.write_tag(0x01);
                }
                Terminator::Branch { .. } => {
                    branch_count = branch_count.saturating_add(1);
                    hasher.write_tag(0x02);
                }
                Terminator::Return(_) => {
                    return_count = return_count.saturating_add(1);
                    hasher.write_tag(0x03);
                }
                Terminator::Unreachable => {
                    unreachable_count = unreachable_count.saturating_add(1);
                    hasher.write_tag(0x04);
                }
            }

            // Feed successor IDs in their natural order — they're already
            // deterministic in cjc-mir because successors() returns a fixed
            // order per terminator variant.
            for s in &succs {
                hasher.write_u32(s.0);
            }
        }

        let loop_count = loops.len() as u32;
        let mut max_loop_depth = 0u32;
        let mut back_edge_count = 0u32;
        let mut countable_loop_count = 0u32;
        for li in &loops.loops {
            // depth is 0-indexed in LoopInfo (0 = outermost).
            let d = li.depth.saturating_add(1);
            if d > max_loop_depth {
                max_loop_depth = d;
            }
            back_edge_count =
                back_edge_count.saturating_add(li.back_edge_sources.len() as u32);
            if li.is_countable {
                countable_loop_count = countable_loop_count.saturating_add(1);
            }
        }

        let cfg_hash = CfgHash(hasher.finish());

        Self {
            block_count: cfg.basic_blocks.len() as u32,
            edge_count,
            branch_count,
            return_count,
            unreachable_count,
            goto_count,
            max_branch_factor,
            loop_count,
            max_loop_depth,
            back_edge_count,
            countable_loop_count,
            cfg_hash,
        }
    }

    /// Convenience: build the CFG + dominator tree + loop tree from a function
    /// and extract metrics in one call.
    ///
    /// Uses an internal `MirFunction` clone (we need `build_cfg` which takes
    /// `&mut self`, but we want a `&MirFunction` interface here so callers
    /// don't have to mutate the program).
    pub fn from_function(func: &MirFunction) -> Self {
        let cfg = cjc_mir::cfg::CfgBuilder::build(&func.body);
        let dom = DominatorTree::compute(&cfg);
        let loops = compute_loop_tree(&cfg, &dom);
        Self::from_cfg(&cfg, &loops)
    }

    /// Feed this metrics block into a hasher (used by `FeatureHash`).
    pub(crate) fn feed(&self, hasher: &mut CanaHasher) {
        hasher.write_tag(0xC1);
        hasher.write_u32(self.block_count);
        hasher.write_u32(self.edge_count);
        hasher.write_u32(self.branch_count);
        hasher.write_u32(self.return_count);
        hasher.write_u32(self.unreachable_count);
        hasher.write_u32(self.goto_count);
        hasher.write_u32(self.max_branch_factor);
        hasher.write_u32(self.loop_count);
        hasher.write_u32(self.max_loop_depth);
        hasher.write_u32(self.back_edge_count);
        hasher.write_u32(self.countable_loop_count);
        hasher.write_u64(self.cfg_hash.0);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_mir::{MirBody, MirExpr, MirExprKind, MirFnId, MirFunction, MirStmt};

    fn empty_main() -> MirFunction {
        MirFunction {
            id: MirFnId(0),
            name: "__main".to_string(),
            type_params: vec![],
            params: vec![],
            return_type: None,
            body: MirBody {
                stmts: vec![],
                result: None,
            },
            is_nogc: false,
            cfg_body: None,
            decorators: vec![],
            vis: cjc_ast::Visibility::Public,
            local_count: 0,
        }
    }

    fn int_lit(v: i64) -> MirExpr {
        MirExpr {
            kind: MirExprKind::IntLit(v),
        }
    }

    #[test]
    fn empty_function_has_one_block_one_return() {
        let func = empty_main();
        let m = CfgMetrics::from_function(&func);
        // A trivial body still produces one block with a Return.
        assert_eq!(m.block_count, 1);
        assert_eq!(m.return_count, 1);
        assert_eq!(m.branch_count, 0);
        assert_eq!(m.loop_count, 0);
        assert_eq!(m.max_loop_depth, 0);
        assert_eq!(m.back_edge_count, 0);
        // Edge count is zero because Return has no successors.
        assert_eq!(m.edge_count, 0);
    }

    #[test]
    fn if_statement_creates_branch() {
        let mut func = empty_main();
        func.body.stmts.push(MirStmt::If {
            cond: int_lit(1),
            then_body: MirBody {
                stmts: vec![],
                result: None,
            },
            else_body: Some(MirBody {
                stmts: vec![],
                result: None,
            }),
        });
        let m = CfgMetrics::from_function(&func);
        // We expect at least one Branch terminator.
        assert!(m.branch_count >= 1, "got branch_count={}", m.branch_count);
        // And max_branch_factor == 2 (Branch has two successors).
        assert_eq!(m.max_branch_factor, 2);
    }

    #[test]
    fn while_statement_creates_loop_and_back_edge() {
        let mut func = empty_main();
        func.body.stmts.push(MirStmt::While {
            cond: int_lit(1),
            body: MirBody {
                stmts: vec![MirStmt::Expr(int_lit(2))],
                result: None,
            },
        });
        let m = CfgMetrics::from_function(&func);
        assert_eq!(m.loop_count, 1, "expected exactly one loop");
        assert!(m.back_edge_count >= 1);
        assert!(m.max_loop_depth >= 1);
    }

    #[test]
    fn cfg_hash_is_deterministic_across_calls() {
        let mut func = empty_main();
        func.body.stmts.push(MirStmt::Expr(int_lit(42)));
        let m1 = CfgMetrics::from_function(&func);
        let m2 = CfgMetrics::from_function(&func);
        assert_eq!(m1.cfg_hash, m2.cfg_hash);
        // Same metrics across both extractions.
        assert_eq!(m1, m2);
    }

    #[test]
    fn cfg_hash_distinguishes_branch_from_goto() {
        let mut with_if = empty_main();
        with_if.body.stmts.push(MirStmt::If {
            cond: int_lit(1),
            then_body: MirBody {
                stmts: vec![],
                result: None,
            },
            else_body: None,
        });

        let plain = empty_main();
        let m_if = CfgMetrics::from_function(&with_if);
        let m_plain = CfgMetrics::from_function(&plain);
        assert_ne!(m_if.cfg_hash, m_plain.cfg_hash);
    }
}
