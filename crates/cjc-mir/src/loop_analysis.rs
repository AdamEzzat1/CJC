//! Loop Analysis — Structured Loop Tree from CFG
//!
//! Provides a proper loop tree (loop nesting forest) built from the CFG using
//! dominance information.  This replaces the heuristic `is_loop_header()` with
//! a full loop structure that captures:
//!
//! - Loop identity (`LoopId`, dense Vec index)
//! - Header block, body blocks, exit blocks, back-edge sources
//! - Nesting depth and parent/child relationships
//! - Preheader block (if it exists)
//!
//! ## Design decisions
//!
//! - **Vec + ID indexing** (not BTreeMap) for all loop structures
//! - **Additive overlay** — does not modify the CFG or MIR
//! - **Deterministic** — DFS traversal order, sorted block sets
//! - **Derived lazily** — computed on demand from CFG + dominator tree
//!
//! ## Algorithm
//!
//! 1. Identify natural loops via back-edges (target dominates source)
//! 2. Compute loop body via reverse walk from back-edge source to header
//! 3. Build nesting tree by checking header containment
//! 4. Compute exit blocks and preheaders
//!
//! ## Determinism guarantee
//!
//! Loop IDs are assigned in header block-ID order (ascending).  For a given
//! CFG, the resulting `LoopTree` is always identical.

use crate::cfg::MirCfg;
use crate::dominators::DominatorTree;
use crate::BlockId;

// ---------------------------------------------------------------------------
// IDs
// ---------------------------------------------------------------------------

/// Dense loop identifier.  Index into `LoopTree::loops`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LoopId(pub u32);

// ---------------------------------------------------------------------------
// Loop descriptor
// ---------------------------------------------------------------------------

/// Information about a single natural loop.
#[derive(Debug, Clone)]
pub struct LoopInfo {
    /// Unique ID (= index in `LoopTree::loops`).
    pub id: LoopId,
    /// The loop header block — dominates all blocks in the loop body.
    pub header: BlockId,
    /// All blocks that belong to this loop (including the header).
    /// Sorted by `BlockId` for deterministic iteration.
    pub body_blocks: Vec<BlockId>,
    /// Back-edge source blocks (blocks that jump back to the header).
    /// Sorted by `BlockId`.
    pub back_edge_sources: Vec<BlockId>,
    /// Exit blocks — blocks outside the loop that are successors of loop body
    /// blocks.  Sorted by `BlockId`.
    pub exit_blocks: Vec<BlockId>,
    /// Preheader block, if one exists: the unique predecessor of the header
    /// that is NOT a back-edge source.  `None` if the header has multiple
    /// non-back-edge predecessors or none.
    pub preheader: Option<BlockId>,
    /// Parent loop (the immediately enclosing loop), if any.
    pub parent: Option<LoopId>,
    /// Direct child loops (immediately nested).  Sorted by `LoopId`.
    pub children: Vec<LoopId>,
    /// Nesting depth: 0 = outermost loop.
    pub depth: u32,
}

// ---------------------------------------------------------------------------
// Loop tree
// ---------------------------------------------------------------------------

/// A loop nesting forest for a single function's CFG.
///
/// `loops` is indexed by `LoopId`.  The loops are sorted by header block ID
/// (ascending), which means outer loops appear before inner loops when they
/// share the same header — but typically each header defines exactly one loop.
#[derive(Debug, Clone)]
pub struct LoopTree {
    /// All loops, indexed by `LoopId(i)` → `loops[i]`.
    pub loops: Vec<LoopInfo>,
    /// Mapping from `BlockId` → innermost `LoopId` containing that block.
    /// Indexed by block index.  `None` if the block is not inside any loop.
    pub block_to_loop: Vec<Option<LoopId>>,
    /// Number of blocks in the CFG (for bounds checking).
    pub num_blocks: usize,
}

impl LoopTree {
    /// Returns the number of loops.
    pub fn len(&self) -> usize {
        self.loops.len()
    }

    /// Returns true if there are no loops.
    pub fn is_empty(&self) -> bool {
        self.loops.is_empty()
    }

    /// Get loop info by ID.
    pub fn get(&self, id: LoopId) -> &LoopInfo {
        &self.loops[id.0 as usize]
    }

    /// Get the innermost loop containing a block, if any.
    pub fn loop_for_block(&self, block: BlockId) -> Option<LoopId> {
        self.block_to_loop.get(block.0 as usize).copied().flatten()
    }

    /// Returns true if `block` is inside loop `loop_id` (at any nesting depth).
    pub fn is_block_in_loop(&self, block: BlockId, loop_id: LoopId) -> bool {
        let mut current = self.loop_for_block(block);
        while let Some(lid) = current {
            if lid == loop_id {
                return true;
            }
            current = self.loops[lid.0 as usize].parent;
        }
        false
    }

    /// Returns true if `inner` is nested inside `outer` (directly or transitively).
    pub fn is_nested_in(&self, inner: LoopId, outer: LoopId) -> bool {
        let mut current = Some(inner);
        while let Some(lid) = current {
            if lid == outer {
                return true;
            }
            current = self.loops[lid.0 as usize].parent;
        }
        false
    }

    /// Return all top-level (outermost) loops.
    pub fn root_loops(&self) -> Vec<LoopId> {
        self.loops
            .iter()
            .filter(|l| l.parent.is_none())
            .map(|l| l.id)
            .collect()
    }

    /// Return the maximum nesting depth across all loops.
    /// Returns 0 if there are no loops.
    pub fn max_depth(&self) -> u32 {
        self.loops.iter().map(|l| l.depth).max().unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

/// Build a `LoopTree` from a CFG and its dominator tree.
///
/// ## Algorithm
///
/// 1. **Find back-edges:** An edge `src → dst` is a back-edge if `dst`
///    dominates `src`.  Each unique `dst` (header) defines one natural loop.
///
/// 2. **Compute loop body:** Starting from each back-edge source, walk
///    predecessors until reaching the header.  All visited blocks are in the
///    loop body.
///
/// 3. **Build nesting:** If loop A's header is contained in loop B's body,
///    then A is nested inside B.  Assign parent/child and depth.
///
/// 4. **Compute exits and preheader:** For each loop, exit blocks are
///    successors of body blocks that are not in the body.  The preheader is
///    the unique non-back-edge predecessor of the header.
///
/// ## Determinism
///
/// - Back-edges collected by iterating blocks in ID order
/// - Headers sorted by BlockId → LoopId assignment is deterministic
/// - Body computed via deterministic BFS (sorted worklist)
pub fn compute_loop_tree(cfg: &MirCfg, domtree: &DominatorTree) -> LoopTree {
    let num_blocks = cfg.basic_blocks.len();
    let preds = cfg.predecessors();

    // Step 1: Find back-edges and group by header.
    // back_edges[header_block_index] = vec of source block IDs
    let mut back_edges: Vec<Vec<BlockId>> = vec![Vec::new(); num_blocks];
    for bb in &cfg.basic_blocks {
        for succ in bb.terminator.successors() {
            if domtree.dominates(succ, bb.id) {
                back_edges[succ.0 as usize].push(bb.id);
            }
        }
    }

    // Step 2: For each header with back-edges, compute the natural loop body.
    // Collect headers in block-ID order for deterministic LoopId assignment.
    let mut loop_infos: Vec<LoopInfo> = Vec::new();
    let mut next_loop_id: u32 = 0;

    for header_idx in 0..num_blocks {
        let sources = &back_edges[header_idx];
        if sources.is_empty() {
            continue;
        }
        let header = BlockId(header_idx as u32);
        let mut sorted_sources = sources.clone();
        sorted_sources.sort();

        // Reverse walk from back-edge sources to header to find body.
        let body = compute_loop_body(header, &sorted_sources, &preds, num_blocks);

        let lid = LoopId(next_loop_id);
        next_loop_id += 1;

        loop_infos.push(LoopInfo {
            id: lid,
            header,
            body_blocks: body,
            back_edge_sources: sorted_sources,
            exit_blocks: Vec::new(),   // filled in step 4
            preheader: None,           // filled in step 4
            parent: None,              // filled in step 3
            children: Vec::new(),      // filled in step 3
            depth: 0,                  // filled in step 3
        });
    }

    // Step 3: Build nesting relationships.
    // A loop A is nested in loop B if A's header is in B's body AND A != B.
    // For each loop, find the *innermost* enclosing loop as parent.
    let n_loops = loop_infos.len();
    for i in 0..n_loops {
        let header_i = loop_infos[i].header;
        let mut best_parent: Option<LoopId> = None;
        let mut best_parent_size = usize::MAX;

        for j in 0..n_loops {
            if i == j {
                continue;
            }
            // Is header_i in loop j's body?
            if loop_infos[j]
                .body_blocks
                .binary_search(&header_i)
                .is_ok()
            {
                // loop j contains loop i.  Pick the smallest (innermost).
                let size = loop_infos[j].body_blocks.len();
                if size < best_parent_size {
                    best_parent = Some(loop_infos[j].id);
                    best_parent_size = size;
                }
            }
        }

        loop_infos[i].parent = best_parent;
    }

    // Fill children.
    for i in 0..n_loops {
        let parent = loop_infos[i].parent;
        if let Some(pid) = parent {
            // We'll collect children after this pass.
            let _ = pid; // used below
        }
    }
    // Second pass to collect children (can't borrow mutably in the loop above).
    let parents: Vec<Option<LoopId>> = loop_infos.iter().map(|l| l.parent).collect();
    for (i, parent) in parents.iter().enumerate() {
        if let Some(pid) = parent {
            loop_infos[pid.0 as usize].children.push(LoopId(i as u32));
        }
    }
    // Sort children for determinism.
    for info in &mut loop_infos {
        info.children.sort();
    }

    // Compute depths via parent chain.
    for i in 0..n_loops {
        let mut depth = 0u32;
        let mut cur = loop_infos[i].parent;
        while let Some(pid) = cur {
            depth += 1;
            cur = loop_infos[pid.0 as usize].parent;
        }
        loop_infos[i].depth = depth;
    }

    // Step 4: Compute exit blocks and preheaders.
    for i in 0..n_loops {
        // Exit blocks: successors of body blocks that are NOT in the body.
        let body = &loop_infos[i].body_blocks;
        let mut exits: Vec<BlockId> = Vec::new();
        for &block in body {
            for succ in cfg.block(block).terminator.successors() {
                if body.binary_search(&succ).is_err() && !exits.contains(&succ) {
                    exits.push(succ);
                }
            }
        }
        exits.sort();
        loop_infos[i].exit_blocks = exits;

        // Preheader: the unique predecessor of the header that is NOT a
        // back-edge source.
        let header = loop_infos[i].header;
        let back_srcs = &loop_infos[i].back_edge_sources;
        let non_back_preds: Vec<BlockId> = preds[header.0 as usize]
            .iter()
            .filter(|p| back_srcs.binary_search(p).is_err())
            .copied()
            .collect();
        loop_infos[i].preheader = if non_back_preds.len() == 1 {
            Some(non_back_preds[0])
        } else {
            None
        };
    }

    // Build block-to-loop mapping (innermost loop).
    let mut block_to_loop: Vec<Option<LoopId>> = vec![None; num_blocks];
    // Process loops from outermost to innermost (by body size descending)
    // so that inner loops overwrite outer.
    let mut by_size: Vec<usize> = (0..n_loops).collect();
    by_size.sort_by(|&a, &b| {
        loop_infos[b]
            .body_blocks
            .len()
            .cmp(&loop_infos[a].body_blocks.len())
    });
    for idx in by_size {
        let lid = loop_infos[idx].id;
        for &block in &loop_infos[idx].body_blocks {
            block_to_loop[block.0 as usize] = Some(lid);
        }
    }

    LoopTree {
        loops: loop_infos,
        block_to_loop,
        num_blocks,
    }
}

/// Compute the set of blocks in a natural loop defined by a header and
/// back-edge sources.  Uses reverse predecessor walk (Appel's algorithm).
///
/// Returns sorted `Vec<BlockId>`.
fn compute_loop_body(
    header: BlockId,
    back_edge_sources: &[BlockId],
    preds: &[Vec<BlockId>],
    _num_blocks: usize,
) -> Vec<BlockId> {
    let mut in_loop = vec![false; preds.len()];
    in_loop[header.0 as usize] = true;

    let mut worklist: Vec<BlockId> = Vec::new();
    for &src in back_edge_sources {
        if !in_loop[src.0 as usize] {
            in_loop[src.0 as usize] = true;
            worklist.push(src);
        }
    }

    // Reverse walk: for each block in worklist, add its predecessors.
    while let Some(block) = worklist.pop() {
        for &pred in &preds[block.0 as usize] {
            if !in_loop[pred.0 as usize] {
                in_loop[pred.0 as usize] = true;
                worklist.push(pred);
            }
        }
    }

    let mut body: Vec<BlockId> = in_loop
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| if b { Some(BlockId(i as u32)) } else { None })
        .collect();
    body.sort();
    body
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{BasicBlock, CfgBuilder, MirCfg, Terminator};
    use crate::dominators::DominatorTree;
    use crate::{MirBody, MirExpr, MirExprKind, MirStmt};

    fn int_expr(v: i64) -> MirExpr {
        MirExpr {
            kind: MirExprKind::IntLit(v),
        }
    }

    fn bool_expr(b: bool) -> MirExpr {
        MirExpr {
            kind: MirExprKind::BoolLit(b),
        }
    }

    // -----------------------------------------------------------------------
    // Helper: build CFG + domtree + loop tree from MirBody
    // -----------------------------------------------------------------------
    fn loop_tree_from_body(body: &MirBody) -> (MirCfg, DominatorTree, LoopTree) {
        let cfg = CfgBuilder::build(body);
        let domtree = DominatorTree::compute(&cfg);
        let loops = compute_loop_tree(&cfg, &domtree);
        (cfg, domtree, loops)
    }

    // -----------------------------------------------------------------------
    // Test 1: No loops → empty LoopTree
    // -----------------------------------------------------------------------
    #[test]
    fn test_no_loops_empty_tree() {
        let body = MirBody {
            stmts: vec![MirStmt::Let {
                name: "x".into(),
                mutable: false,
                init: int_expr(42),
                alloc_hint: None,
            }],
            result: Some(Box::new(int_expr(42))),
        };
        let (_, _, loops) = loop_tree_from_body(&body);
        assert!(loops.is_empty());
        assert_eq!(loops.len(), 0);
        assert_eq!(loops.max_depth(), 0);
    }

    // -----------------------------------------------------------------------
    // Test 2: Single while loop → one loop with header, body, exit
    // -----------------------------------------------------------------------
    #[test]
    fn test_single_while_loop() {
        let body = MirBody {
            stmts: vec![MirStmt::While {
                cond: bool_expr(true),
                body: MirBody {
                    stmts: vec![MirStmt::Expr(int_expr(1))],
                    result: None,
                },
            }],
            result: None,
        };
        let (cfg, _, loops) = loop_tree_from_body(&body);

        assert_eq!(loops.len(), 1, "should detect exactly 1 loop");

        let loop0 = loops.get(LoopId(0));
        assert_eq!(loop0.depth, 0, "outermost loop has depth 0");
        assert!(loop0.parent.is_none(), "no parent loop");
        assert!(loop0.children.is_empty(), "no child loops");
        assert!(
            !loop0.body_blocks.is_empty(),
            "body should have at least header"
        );
        assert!(
            loop0.body_blocks.contains(&loop0.header),
            "header is part of body"
        );
        assert!(
            !loop0.back_edge_sources.is_empty(),
            "should have at least one back-edge source"
        );
        assert!(
            !loop0.exit_blocks.is_empty(),
            "should have at least one exit block"
        );

        // Header should be a loop header per the heuristic too.
        assert!(cfg.is_loop_header(loop0.header));
    }

    // -----------------------------------------------------------------------
    // Test 3: Nested loops → two loops with parent/child
    // -----------------------------------------------------------------------
    #[test]
    fn test_nested_while_loops() {
        let body = MirBody {
            stmts: vec![MirStmt::While {
                cond: bool_expr(true),
                body: MirBody {
                    stmts: vec![MirStmt::While {
                        cond: bool_expr(true),
                        body: MirBody {
                            stmts: vec![MirStmt::Expr(int_expr(1))],
                            result: None,
                        },
                    }],
                    result: None,
                },
            }],
            result: None,
        };
        let (_, _, loops) = loop_tree_from_body(&body);

        assert_eq!(loops.len(), 2, "should detect 2 loops (outer + inner)");

        // Find outer and inner by depth.
        let outer = loops.loops.iter().find(|l| l.depth == 0).unwrap();
        let inner = loops.loops.iter().find(|l| l.depth == 1).unwrap();

        assert!(outer.parent.is_none());
        assert_eq!(inner.parent, Some(outer.id));
        assert!(outer.children.contains(&inner.id));
        assert!(inner.children.is_empty());

        // Inner header should be inside outer body.
        assert!(outer.body_blocks.contains(&inner.header));

        // Nesting check.
        assert!(loops.is_nested_in(inner.id, outer.id));
        assert!(!loops.is_nested_in(outer.id, inner.id));

        assert_eq!(loops.max_depth(), 1);
        assert_eq!(loops.root_loops().len(), 1);
    }

    // -----------------------------------------------------------------------
    // Test 4: Sequential loops → two independent loops
    // -----------------------------------------------------------------------
    #[test]
    fn test_sequential_loops() {
        let body = MirBody {
            stmts: vec![
                MirStmt::While {
                    cond: bool_expr(true),
                    body: MirBody {
                        stmts: vec![MirStmt::Expr(int_expr(1))],
                        result: None,
                    },
                },
                MirStmt::While {
                    cond: bool_expr(true),
                    body: MirBody {
                        stmts: vec![MirStmt::Expr(int_expr(2))],
                        result: None,
                    },
                },
            ],
            result: None,
        };
        let (_, _, loops) = loop_tree_from_body(&body);

        assert_eq!(loops.len(), 2, "should detect 2 independent loops");

        // Both should be depth 0 (no nesting).
        assert!(loops.loops.iter().all(|l| l.depth == 0));
        assert!(loops.loops.iter().all(|l| l.parent.is_none()));
        assert_eq!(loops.root_loops().len(), 2);

        // They should not be nested in each other.
        assert!(!loops.is_nested_in(LoopId(0), LoopId(1)));
        assert!(!loops.is_nested_in(LoopId(1), LoopId(0)));
    }

    // -----------------------------------------------------------------------
    // Test 5: Preheader detection
    // -----------------------------------------------------------------------
    #[test]
    fn test_preheader_detection() {
        // A simple while loop has a preheader (the entry → header goto block).
        let body = MirBody {
            stmts: vec![MirStmt::While {
                cond: bool_expr(true),
                body: MirBody {
                    stmts: vec![],
                    result: None,
                },
            }],
            result: None,
        };
        let (_, _, loops) = loop_tree_from_body(&body);
        assert_eq!(loops.len(), 1);
        let loop0 = loops.get(LoopId(0));
        // The CfgBuilder creates: entry → goto(header) → branch → body → goto(header)
        // So the header has exactly one non-back-edge predecessor: the entry block.
        assert!(
            loop0.preheader.is_some(),
            "simple while loop should have a preheader"
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: block_to_loop mapping correctness
    // -----------------------------------------------------------------------
    #[test]
    fn test_block_to_loop_mapping() {
        let body = MirBody {
            stmts: vec![MirStmt::While {
                cond: bool_expr(true),
                body: MirBody {
                    stmts: vec![MirStmt::Expr(int_expr(1))],
                    result: None,
                },
            }],
            result: None,
        };
        let (cfg, _, loops) = loop_tree_from_body(&body);
        assert_eq!(loops.len(), 1);

        let loop0 = loops.get(LoopId(0));

        // Every body block should map to this loop.
        for &block in &loop0.body_blocks {
            assert_eq!(
                loops.loop_for_block(block),
                Some(LoopId(0)),
                "body block {:?} should map to loop 0",
                block
            );
        }

        // Exit blocks should NOT be in the loop.
        for &block in &loop0.exit_blocks {
            // Exit blocks may or may not be in the loop — if they're not in
            // body_blocks, they shouldn't map to this loop.
            if loop0.body_blocks.binary_search(&block).is_err() {
                assert_ne!(
                    loops.loop_for_block(block),
                    Some(LoopId(0)),
                    "exit block {:?} should not be in loop 0",
                    block
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 7: Determinism — same input produces identical LoopTree
    // -----------------------------------------------------------------------
    #[test]
    fn test_loop_tree_determinism() {
        let body = MirBody {
            stmts: vec![
                MirStmt::While {
                    cond: bool_expr(true),
                    body: MirBody {
                        stmts: vec![MirStmt::While {
                            cond: bool_expr(true),
                            body: MirBody {
                                stmts: vec![MirStmt::Expr(int_expr(1))],
                                result: None,
                            },
                        }],
                        result: None,
                    },
                },
                MirStmt::While {
                    cond: bool_expr(false),
                    body: MirBody {
                        stmts: vec![],
                        result: None,
                    },
                },
            ],
            result: None,
        };

        // Build twice and compare.
        let (_, _, loops1) = loop_tree_from_body(&body);
        let (_, _, loops2) = loop_tree_from_body(&body);

        assert_eq!(loops1.len(), loops2.len());
        for i in 0..loops1.len() {
            let a = &loops1.loops[i];
            let b = &loops2.loops[i];
            assert_eq!(a.id, b.id);
            assert_eq!(a.header, b.header);
            assert_eq!(a.body_blocks, b.body_blocks);
            assert_eq!(a.back_edge_sources, b.back_edge_sources);
            assert_eq!(a.exit_blocks, b.exit_blocks);
            assert_eq!(a.preheader, b.preheader);
            assert_eq!(a.parent, b.parent);
            assert_eq!(a.children, b.children);
            assert_eq!(a.depth, b.depth);
        }
        assert_eq!(loops1.block_to_loop, loops2.block_to_loop);
    }

    // -----------------------------------------------------------------------
    // Test 8: is_block_in_loop checks transitive nesting
    // -----------------------------------------------------------------------
    #[test]
    fn test_is_block_in_loop_transitive() {
        let body = MirBody {
            stmts: vec![MirStmt::While {
                cond: bool_expr(true),
                body: MirBody {
                    stmts: vec![MirStmt::While {
                        cond: bool_expr(true),
                        body: MirBody {
                            stmts: vec![MirStmt::Expr(int_expr(1))],
                            result: None,
                        },
                    }],
                    result: None,
                },
            }],
            result: None,
        };
        let (_, _, loops) = loop_tree_from_body(&body);
        assert_eq!(loops.len(), 2);

        let outer = loops.loops.iter().find(|l| l.depth == 0).unwrap();
        let inner = loops.loops.iter().find(|l| l.depth == 1).unwrap();

        // Inner loop's header should be in the outer loop.
        assert!(loops.is_block_in_loop(inner.header, outer.id));

        // Inner loop's body blocks should be in the outer loop too.
        for &block in &inner.body_blocks {
            assert!(
                loops.is_block_in_loop(block, outer.id),
                "inner block {:?} should be transitively in outer loop",
                block
            );
        }
    }
}
