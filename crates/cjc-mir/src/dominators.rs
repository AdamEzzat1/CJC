//! Dominator tree computation and dominance frontiers.
//!
//! Uses the iterative dominator algorithm (Cooper, Harvey & Kennedy, 2001)
//! which is simple, efficient for reducible CFGs, and zero-dependency.
//!
//! All algorithms produce deterministic results for a given CFG.

use crate::cfg::MirCfg;
use crate::BlockId;

/// Dominator tree: maps each block to its immediate dominator.
///
/// `idom[i]` is the immediate dominator of block `i`.
/// The entry block dominates itself: `idom[entry] == entry`.
#[derive(Debug, Clone)]
pub struct DominatorTree {
    /// Immediate dominator for each block (indexed by block ID).
    pub idom: Vec<BlockId>,
    /// Number of blocks.
    pub num_blocks: usize,
    /// Entry block ID.
    pub entry: BlockId,
}

impl DominatorTree {
    /// Compute the dominator tree for a CFG using the iterative algorithm.
    ///
    /// Uses the Cooper-Harvey-Kennedy (2001) algorithm which converges in
    /// linear time for reducible CFGs (the only kind produced by CJC's
    /// structured control flow).
    ///
    /// # Arguments
    ///
    /// * `cfg` - The control-flow graph to analyze.
    ///
    /// # Returns
    ///
    /// A [`DominatorTree`] where `idom[i]` is the immediate dominator of block `i`.
    pub fn compute(cfg: &MirCfg) -> Self {
        let n = cfg.basic_blocks.len();
        let entry = cfg.entry;
        let preds = cfg.predecessors();

        // Compute reverse postorder (RPO)
        let rpo = reverse_postorder(cfg);
        let mut rpo_number = vec![usize::MAX; n]; // block_id -> position in RPO
        for (pos, &block_id) in rpo.iter().enumerate() {
            rpo_number[block_id.0 as usize] = pos;
        }

        // Initialize idom: all undefined except entry
        const UNDEF: u32 = u32::MAX;
        let mut idom = vec![UNDEF; n];
        idom[entry.0 as usize] = entry.0;

        let mut changed = true;
        while changed {
            changed = false;
            for &block in &rpo {
                if block == entry {
                    continue;
                }
                let b = block.0 as usize;

                // Find the first processed predecessor
                let mut new_idom = UNDEF;
                for &pred in &preds[b] {
                    let p = pred.0 as usize;
                    if idom[p] != UNDEF {
                        new_idom = pred.0;
                        break;
                    }
                }

                if new_idom == UNDEF {
                    continue;
                }

                // Intersect with remaining predecessors
                for &pred in &preds[b] {
                    let p = pred.0 as usize;
                    if idom[p] != UNDEF {
                        new_idom = intersect(&idom, &rpo_number, new_idom, pred.0);
                    }
                }

                if idom[b] != new_idom {
                    idom[b] = new_idom;
                    changed = true;
                }
            }
        }

        DominatorTree {
            idom: idom.iter().map(|&id| BlockId(id)).collect(),
            num_blocks: n,
            entry,
        }
    }

    /// Return true if block `a` dominates block `b`.
    ///
    /// A block dominates itself. The check walks up the immediate dominator
    /// chain from `b` toward the entry block, returning true if `a` is
    /// encountered.
    pub fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        let mut current = b;
        loop {
            if current == a {
                return true;
            }
            let idom = self.idom[current.0 as usize];
            if idom == current {
                // Reached the entry's self-loop
                return current == a;
            }
            current = idom;
        }
    }

    /// Compute the dominance frontier for each block.
    ///
    /// DF(b) = { y | ∃ pred of y where b dominates pred but b does NOT
    ///           strictly dominate y }
    pub fn dominance_frontiers(&self, cfg: &MirCfg) -> Vec<Vec<BlockId>> {
        let n = self.num_blocks;
        let preds = cfg.predecessors();
        let mut df: Vec<Vec<BlockId>> = vec![Vec::new(); n];

        for b in 0..n {
            // Skip unreachable blocks (idom not set).
            if self.idom[b].0 as usize >= n {
                continue;
            }
            if preds[b].len() >= 2 {
                for &pred in &preds[b] {
                    let mut runner = pred;
                    // Guard: stop if runner is unreachable (idom out of range).
                    while runner != self.idom[b as usize]
                        && (runner.0 as usize) < n
                    {
                        let block_b = BlockId(b as u32);
                        if !df[runner.0 as usize].contains(&block_b) {
                            df[runner.0 as usize].push(block_b);
                        }
                        runner = self.idom[runner.0 as usize];
                    }
                }
            }
        }

        // Sort for determinism
        for frontier in &mut df {
            frontier.sort_by_key(|b| b.0);
        }

        df
    }

    /// Get the immediate children of a node in the dominator tree.
    ///
    /// Returns all blocks whose immediate dominator is `block`.
    pub fn children(&self, block: BlockId) -> Vec<BlockId> {
        let mut result = Vec::new();
        for i in 0..self.num_blocks {
            let id = BlockId(i as u32);
            if id != block && self.idom[i] == block {
                result.push(id);
            }
        }
        result
    }
}

/// Intersect two dominators in the iterative algorithm.
fn intersect(idom: &[u32], rpo_number: &[usize], mut a: u32, mut b: u32) -> u32 {
    while a != b {
        while rpo_number[a as usize] > rpo_number[b as usize] {
            a = idom[a as usize];
        }
        while rpo_number[b as usize] > rpo_number[a as usize] {
            b = idom[b as usize];
        }
    }
    a
}

/// Compute reverse postorder traversal of the CFG.
fn reverse_postorder(cfg: &MirCfg) -> Vec<BlockId> {
    let n = cfg.basic_blocks.len();
    let mut visited = vec![false; n];
    let mut postorder = Vec::with_capacity(n);

    fn dfs(
        cfg: &MirCfg,
        block: BlockId,
        visited: &mut Vec<bool>,
        postorder: &mut Vec<BlockId>,
    ) {
        let b = block.0 as usize;
        if visited[b] {
            return;
        }
        visited[b] = true;
        for succ in cfg.successors(block) {
            dfs(cfg, succ, visited, postorder);
        }
        postorder.push(block);
    }

    dfs(cfg, cfg.entry, &mut visited, &mut postorder);
    postorder.reverse();
    postorder
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{BasicBlock, CfgBuilder, MirCfg, Terminator};
    use crate::{MirBody, MirExpr, MirExprKind, MirStmt};

    fn int_expr(v: i64) -> MirExpr {
        MirExpr { kind: MirExprKind::IntLit(v) }
    }

    fn bool_expr(b: bool) -> MirExpr {
        MirExpr { kind: MirExprKind::BoolLit(b) }
    }

    #[test]
    fn test_domtree_single_block() {
        let cfg = MirCfg {
            basic_blocks: vec![BasicBlock {
                id: BlockId(0),
                statements: vec![],
                terminator: Terminator::Return(None),
            }],
            entry: BlockId(0),
        };
        let domtree = DominatorTree::compute(&cfg);
        assert_eq!(domtree.idom[0], BlockId(0)); // entry dominates itself
    }

    #[test]
    fn test_domtree_linear_chain() {
        // 0 -> 1 -> 2 -> return
        let cfg = MirCfg {
            basic_blocks: vec![
                BasicBlock { id: BlockId(0), statements: vec![], terminator: Terminator::Goto(BlockId(1)) },
                BasicBlock { id: BlockId(1), statements: vec![], terminator: Terminator::Goto(BlockId(2)) },
                BasicBlock { id: BlockId(2), statements: vec![], terminator: Terminator::Return(None) },
            ],
            entry: BlockId(0),
        };
        let domtree = DominatorTree::compute(&cfg);
        assert_eq!(domtree.idom[0], BlockId(0));
        assert_eq!(domtree.idom[1], BlockId(0));
        assert_eq!(domtree.idom[2], BlockId(1));
        assert!(domtree.dominates(BlockId(0), BlockId(2)));
    }

    #[test]
    fn test_domtree_diamond() {
        // 0 -> (1, 2) -> 3
        let cfg = MirCfg {
            basic_blocks: vec![
                BasicBlock { id: BlockId(0), statements: vec![], terminator: Terminator::Branch {
                    cond: bool_expr(true), then_block: BlockId(1), else_block: BlockId(2),
                }},
                BasicBlock { id: BlockId(1), statements: vec![], terminator: Terminator::Goto(BlockId(3)) },
                BasicBlock { id: BlockId(2), statements: vec![], terminator: Terminator::Goto(BlockId(3)) },
                BasicBlock { id: BlockId(3), statements: vec![], terminator: Terminator::Return(None) },
            ],
            entry: BlockId(0),
        };
        let domtree = DominatorTree::compute(&cfg);
        assert_eq!(domtree.idom[1], BlockId(0));
        assert_eq!(domtree.idom[2], BlockId(0));
        assert_eq!(domtree.idom[3], BlockId(0)); // merge dominated by entry
    }

    #[test]
    fn test_dominance_frontier_diamond() {
        let cfg = MirCfg {
            basic_blocks: vec![
                BasicBlock { id: BlockId(0), statements: vec![], terminator: Terminator::Branch {
                    cond: bool_expr(true), then_block: BlockId(1), else_block: BlockId(2),
                }},
                BasicBlock { id: BlockId(1), statements: vec![], terminator: Terminator::Goto(BlockId(3)) },
                BasicBlock { id: BlockId(2), statements: vec![], terminator: Terminator::Goto(BlockId(3)) },
                BasicBlock { id: BlockId(3), statements: vec![], terminator: Terminator::Return(None) },
            ],
            entry: BlockId(0),
        };
        let domtree = DominatorTree::compute(&cfg);
        let df = domtree.dominance_frontiers(&cfg);

        // DF(0) = {} (entry dominates everything)
        assert!(df[0].is_empty(), "DF(entry) should be empty");
        // DF(1) = {3} (block 3 is the merge point)
        assert_eq!(df[1], vec![BlockId(3)]);
        // DF(2) = {3}
        assert_eq!(df[2], vec![BlockId(3)]);
        // DF(3) = {}
        assert!(df[3].is_empty());
    }

    #[test]
    fn test_domtree_from_mir_body() {
        let body = MirBody {
            stmts: vec![
                MirStmt::If {
                    cond: bool_expr(true),
                    then_body: MirBody { stmts: vec![], result: Some(Box::new(int_expr(1))) },
                    else_body: Some(MirBody { stmts: vec![], result: Some(Box::new(int_expr(2))) }),
                },
            ],
            result: None,
        };
        let cfg = CfgBuilder::build(&body);
        let domtree = DominatorTree::compute(&cfg);

        // Entry should dominate all blocks
        for i in 0..cfg.basic_blocks.len() {
            assert!(
                domtree.dominates(cfg.entry, BlockId(i as u32)),
                "entry should dominate block {}", i
            );
        }
    }

    #[test]
    fn test_reverse_postorder() {
        let cfg = MirCfg {
            basic_blocks: vec![
                BasicBlock { id: BlockId(0), statements: vec![], terminator: Terminator::Goto(BlockId(1)) },
                BasicBlock { id: BlockId(1), statements: vec![], terminator: Terminator::Return(None) },
            ],
            entry: BlockId(0),
        };
        let rpo = reverse_postorder(&cfg);
        assert_eq!(rpo, vec![BlockId(0), BlockId(1)]);
    }
}
