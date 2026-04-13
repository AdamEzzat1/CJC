//! CJC MIR Control-Flow Graph (CFG)
//!
//! This module provides a **graph-based representation** of MIR function bodies
//! as an alternative to the tree-form `MirBody`.
//!
//! The CFG is built from a `MirBody` using `CfgBuilder::build()`.  The existing
//! tree-form MIR is preserved and unchanged — the CFG is a derived view that
//! enables analyses that require explicit predecessor/successor relationships:
//!
//! - Live variable analysis (requires predecessor sets)
//! - Loop-Invariant Code Motion (LICM, requires back-edge detection)
//! - SSA construction (requires dominator trees)
//! - Register allocation (requires live ranges)
//!
//! ## Structure
//!
//! ```text
//! MirCfg {
//!     basic_blocks: Vec<BasicBlock>,   // indexed by BlockId
//!     entry: BlockId,                  // always BlockId(0)
//! }
//!
//! BasicBlock {
//!     id: BlockId,
//!     statements: Vec<CfgStmt>,        // side-effect-free internal stmts
//!     terminator: Terminator,          // mandatory, exactly one per block
//! }
//!
//! Terminator {
//!     Goto(BlockId)             — unconditional jump
//!     Branch { cond, then, else_ } — conditional branch
//!     Return(Option<MirExpr>)   — function return
//!     Unreachable               — dead / unimplemented path
//! }
//! ```
//!
//! ## Determinism guarantee
//!
//! Block IDs are assigned in the order the blocks are first created during the
//! single depth-first tree walk of `MirBody`.  For a given `MirBody`, the
//! resulting CFG is always identical (same IDs, same block ordering, same
//! instruction ordering).

use crate::{BlockId, MirBody, MirExpr, MirStmt};

// ---------------------------------------------------------------------------
// CFG types
// ---------------------------------------------------------------------------

/// A CJC CFG function body: a list of basic blocks indexed by `BlockId`.
#[derive(Debug, Clone)]
pub struct MirCfg {
    /// All basic blocks.  Block `i` is at `basic_blocks[i]`.
    /// `basic_blocks[0]` is always the entry block.
    pub basic_blocks: Vec<BasicBlock>,
    /// Entry block ID (always `BlockId(0)`).
    pub entry: BlockId,
}

impl MirCfg {
    /// Return a reference to the entry basic block (always `BlockId(0)`).
    pub fn entry_block(&self) -> &BasicBlock {
        &self.basic_blocks[self.entry.0 as usize]
    }

    /// Return a reference to the block with the given ID.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of bounds.
    pub fn block(&self, id: BlockId) -> &BasicBlock {
        &self.basic_blocks[id.0 as usize]
    }

    /// Collect all successor block IDs for the given block by inspecting
    /// its terminator.
    pub fn successors(&self, id: BlockId) -> Vec<BlockId> {
        self.block(id).terminator.successors()
    }

    /// Compute the predecessor list for every block (inverted adjacency list).
    ///
    /// Returns a `Vec` indexed by block index, where each entry is the list
    /// of blocks that have an edge leading to that block. Deterministic
    /// ordering: predecessors appear in the order they are discovered by
    /// iterating blocks in ID order.
    pub fn predecessors(&self) -> Vec<Vec<BlockId>> {
        let n = self.basic_blocks.len();
        let mut preds: Vec<Vec<BlockId>> = vec![Vec::new(); n];
        for bb in &self.basic_blocks {
            for succ in bb.terminator.successors() {
                preds[succ.0 as usize].push(bb.id);
            }
        }
        preds
    }

    /// Returns true if `candidate` is a back-edge target (loop header).
    /// A block is a loop header if any of its predecessors has a higher ID
    /// (simple heuristic that works for reducible CFGs built from structured code).
    pub fn is_loop_header(&self, id: BlockId) -> bool {
        let preds = self.predecessors();
        preds[id.0 as usize].iter().any(|p| p.0 >= id.0)
    }
}

/// A basic block: a sequence of side-effect-free statements followed by
/// exactly one terminator.
#[derive(Debug, Clone)]
pub struct BasicBlock {
    /// Unique ID; equals the index in `MirCfg::basic_blocks`.
    pub id: BlockId,
    /// Internal statements (let bindings, expression statements).
    /// These never transfer control — control flow lives exclusively in
    /// the `terminator`.
    pub statements: Vec<CfgStmt>,
    /// The block's terminator — mandatory, exactly one per block.
    pub terminator: Terminator,
}

/// A statement inside a basic block (no control flow).
#[derive(Debug, Clone)]
pub enum CfgStmt {
    /// `let name = init;`
    Let {
        name: String,
        mutable: bool,
        init: MirExpr,
    },
    /// A standalone expression (call, assign, etc.).
    Expr(MirExpr),
}

/// The terminator of a basic block — exactly one per block.
///
/// The terminator determines which block(s) execute after this one.
#[derive(Debug, Clone)]
pub enum Terminator {
    /// Unconditional jump to another block.
    Goto(BlockId),
    /// Conditional branch: `if cond { goto then } else { goto else_ }`.
    Branch {
        cond: MirExpr,
        then_block: BlockId,
        else_block: BlockId,
    },
    /// `return expr;` — exits the function.
    Return(Option<MirExpr>),
    /// Marks a dead / unreachable code path.
    Unreachable,
}

impl Terminator {
    /// Collect all successor `BlockId`s of this terminator.
    pub fn successors(&self) -> Vec<BlockId> {
        match self {
            Terminator::Goto(id) => vec![*id],
            Terminator::Branch { then_block, else_block, .. } => {
                vec![*then_block, *else_block]
            }
            Terminator::Return(_) | Terminator::Unreachable => vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// CFG Builder
// ---------------------------------------------------------------------------

/// Builds a `MirCfg` from a tree-form `MirBody`.
///
/// The builder performs a single depth-first traversal of the `MirBody`
/// statement list, emitting `BasicBlock`s in a deterministic order.
///
/// ## Block assignment rules
///
/// | Construct       | Blocks created |
/// |-----------------|----------------|
/// | Straight-line   | stays in current block |
/// | `if/else`       | then-block, else-block (or fallthrough), merge-block |
/// | `while`         | header-block, body-block, exit-block |
/// | `return`        | terminates current block with `Terminator::Return` |
/// | `nogc { ... }`  | inlined into current block (no block boundary) |
pub struct CfgBuilder {
    blocks: Vec<BasicBlock>,
    next_block: u32,
}

impl CfgBuilder {
    /// Build a `MirCfg` from a `MirBody`.
    pub fn build(body: &MirBody) -> MirCfg {
        let mut builder = CfgBuilder {
            blocks: Vec::new(),
            next_block: 0,
        };

        let entry = builder.new_block();
        let (current, stmts, result_expr) = builder.lower_body(body, entry);

        // Write the accumulated statements into the current (last) block
        builder.blocks[current.0 as usize].statements = stmts;

        // Terminate the last block
        let terminator = if let Some(expr) = result_expr {
            Terminator::Return(Some(expr))
        } else {
            Terminator::Return(None)
        };
        builder.blocks[current.0 as usize].terminator = terminator;

        MirCfg {
            basic_blocks: builder.blocks,
            entry,
        }
    }

    fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.next_block);
        self.next_block += 1;
        self.blocks.push(BasicBlock {
            id,
            statements: Vec::new(),
            terminator: Terminator::Unreachable, // placeholder, filled in later
        });
        id
    }

    /// Lower a `MirBody` starting from `start` block.
    ///
    /// Returns `(current_block_id, accumulated_stmts, tail_expr)`.
    /// The caller is responsible for sealing the returned block.
    fn lower_body(
        &mut self,
        body: &MirBody,
        start: BlockId,
    ) -> (BlockId, Vec<CfgStmt>, Option<MirExpr>) {
        let mut current = start;
        let mut stmts: Vec<CfgStmt> = Vec::new();

        for stmt in &body.stmts {
            match stmt {
                MirStmt::Let { name, mutable, init, .. } => {
                    stmts.push(CfgStmt::Let {
                        name: name.clone(),
                        mutable: *mutable,
                        init: init.clone(),
                    });
                }
                MirStmt::Expr(expr) => {
                    stmts.push(CfgStmt::Expr(expr.clone()));
                }
                MirStmt::Return(opt_expr) => {
                    // Seal current block with Return, then start a new
                    // (unreachable) block for any dead code after the return.
                    self.blocks[current.0 as usize].statements =
                        std::mem::take(&mut stmts);
                    self.blocks[current.0 as usize].terminator =
                        Terminator::Return(opt_expr.clone());
                    // Remaining statements after `return` are dead; allocate a
                    // fresh block and continue into it (it will never be reached).
                    current = self.new_block();
                }
                MirStmt::If { cond, then_body, else_body } => {
                    // Seal the current block before the branch
                    let then_id = self.new_block();
                    let merge_id = self.new_block();
                    let else_id = if else_body.is_some() {
                        self.new_block()
                    } else {
                        merge_id
                    };

                    self.blocks[current.0 as usize].statements =
                        std::mem::take(&mut stmts);
                    self.blocks[current.0 as usize].terminator =
                        Terminator::Branch {
                            cond: cond.clone(),
                            then_block: then_id,
                            else_block: else_id,
                        };

                    // Lower then-branch
                    let (then_end, then_stmts, then_result) =
                        self.lower_body(then_body, then_id);
                    self.blocks[then_end.0 as usize].statements = then_stmts;
                    // If the then-body ends with a return (already sealed), don't
                    // add another Goto. Otherwise jump to merge.
                    if matches!(
                        self.blocks[then_end.0 as usize].terminator,
                        Terminator::Unreachable
                    ) {
                        // Possibly emit the tail expr as a statement
                        if let Some(expr) = then_result {
                            self.blocks[then_end.0 as usize]
                                .statements
                                .push(CfgStmt::Expr(expr));
                        }
                        self.blocks[then_end.0 as usize].terminator =
                            Terminator::Goto(merge_id);
                    }

                    // Lower else-branch (if present)
                    if let Some(else_b) = else_body {
                        let (else_end, else_stmts, else_result) =
                            self.lower_body(else_b, else_id);
                        self.blocks[else_end.0 as usize].statements = else_stmts;
                        if matches!(
                            self.blocks[else_end.0 as usize].terminator,
                            Terminator::Unreachable
                        ) {
                            if let Some(expr) = else_result {
                                self.blocks[else_end.0 as usize]
                                    .statements
                                    .push(CfgStmt::Expr(expr));
                            }
                            self.blocks[else_end.0 as usize].terminator =
                                Terminator::Goto(merge_id);
                        }
                    }

                    current = merge_id;
                }
                MirStmt::While { cond, body: loop_body } => {
                    // Seal the current block and jump to the loop header
                    let header_id = self.new_block();
                    let body_id = self.new_block();
                    let exit_id = self.new_block();

                    self.blocks[current.0 as usize].statements =
                        std::mem::take(&mut stmts);
                    self.blocks[current.0 as usize].terminator =
                        Terminator::Goto(header_id);

                    // Header: branch on condition
                    self.blocks[header_id.0 as usize].terminator =
                        Terminator::Branch {
                            cond: cond.clone(),
                            then_block: body_id,
                            else_block: exit_id,
                        };

                    // Body
                    let (body_end, body_stmts, _body_result) =
                        self.lower_body(loop_body, body_id);
                    self.blocks[body_end.0 as usize].statements = body_stmts;
                    if matches!(
                        self.blocks[body_end.0 as usize].terminator,
                        Terminator::Unreachable
                    ) {
                        self.blocks[body_end.0 as usize].terminator =
                            Terminator::Goto(header_id); // back-edge
                    }

                    current = exit_id;
                }
                // Break/Continue: no-op in CFG lowering (handled as control flow in executor)
                MirStmt::Break | MirStmt::Continue => {}
                MirStmt::NoGcBlock(inner_body) => {
                    // NoGC blocks are transparent to the CFG — inline them.
                    let (next_current, inner_stmts, inner_result) =
                        self.lower_body(inner_body, current);
                    // If the inner body created new blocks, we need to flush
                    // the current stmts first.
                    if next_current != current {
                        // The lower_body call already sealed intermediate blocks.
                        // Merge remaining stmts from inner into the new current.
                        let mut combined = stmts;
                        combined.extend(inner_stmts);
                        stmts = combined;
                        current = next_current;
                    } else {
                        stmts.extend(inner_stmts);
                    }
                    if let Some(expr) = inner_result {
                        stmts.push(CfgStmt::Expr(expr));
                    }
                }
            }
        }

        let tail = body.result.as_ref().map(|e| *e.clone());
        (current, stmts, tail)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MirExpr, MirExprKind, MirStmt};

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

    fn empty_body() -> MirBody {
        MirBody {
            stmts: vec![],
            result: None,
        }
    }

    #[test]
    fn test_cfg_straight_line_entry_block() {
        // A body with only let-bindings and a result expr.
        // Should produce a single block with Return terminator.
        let body = MirBody {
            stmts: vec![
                MirStmt::Let {
                    name: "x".into(),
                    mutable: false,
                    init: int_expr(42),
                    alloc_hint: None,
                },
            ],
            result: Some(Box::new(int_expr(42))),
        };
        let cfg = CfgBuilder::build(&body);
        // At minimum one block (entry)
        assert!(cfg.basic_blocks.len() >= 1, "should have at least entry block");
        assert_eq!(cfg.entry, BlockId(0));
        // Entry has Return terminator
        match &cfg.entry_block().terminator {
            Terminator::Return(_) => {}
            other => panic!("expected Return, got {:?}", other),
        }
    }

    #[test]
    fn test_cfg_if_creates_branch_terminator() {
        // A body with a single if/else — should produce a Branch terminator.
        let body = MirBody {
            stmts: vec![
                MirStmt::If {
                    cond: bool_expr(true),
                    then_body: MirBody {
                        stmts: vec![],
                        result: Some(Box::new(int_expr(1))),
                    },
                    else_body: Some(MirBody {
                        stmts: vec![],
                        result: Some(Box::new(int_expr(2))),
                    }),
                },
            ],
            result: None,
        };
        let cfg = CfgBuilder::build(&body);
        assert!(cfg.basic_blocks.len() >= 3, "if/else should produce >= 3 blocks");
        match &cfg.entry_block().terminator {
            Terminator::Branch { then_block, else_block, .. } => {
                assert_ne!(then_block, else_block, "then and else blocks must be distinct");
            }
            other => panic!("entry block should have Branch terminator, got {:?}", other),
        }
    }

    #[test]
    fn test_cfg_while_creates_back_edge() {
        // A while loop — header block should be identified as a loop header.
        let body = MirBody {
            stmts: vec![
                MirStmt::While {
                    cond: bool_expr(false),
                    body: empty_body(),
                },
            ],
            result: None,
        };
        let cfg = CfgBuilder::build(&body);
        // Should have entry (pre-loop), header, body, exit, plus the post-loop block
        assert!(cfg.basic_blocks.len() >= 3, "while should produce >= 3 blocks");

        // The header block (block 1 after entry=0) should be a loop header
        // because the body block's Goto points back to it.
        let header = BlockId(1);
        assert!(
            cfg.is_loop_header(header),
            "block {:?} should be detected as a loop header",
            header
        );
    }

    #[test]
    fn test_cfg_return_terminates_block() {
        let body = MirBody {
            stmts: vec![
                MirStmt::Return(Some(int_expr(99))),
            ],
            result: None,
        };
        let cfg = CfgBuilder::build(&body);
        match &cfg.entry_block().terminator {
            Terminator::Return(Some(expr)) => {
                assert!(matches!(expr.kind, MirExprKind::IntLit(99)));
            }
            other => panic!("expected Return(99), got {:?}", other),
        }
    }

    #[test]
    fn test_cfg_predecessors_entry_has_no_preds() {
        let body = MirBody {
            stmts: vec![],
            result: Some(Box::new(int_expr(0))),
        };
        let cfg = CfgBuilder::build(&body);
        let preds = cfg.predecessors();
        assert_eq!(
            preds[cfg.entry.0 as usize].len(),
            0,
            "entry block should have no predecessors"
        );
    }

    #[test]
    fn test_cfg_goto_terminator_successors() {
        // A block with Goto should have exactly 1 successor.
        let term = Terminator::Goto(BlockId(5));
        let succs = term.successors();
        assert_eq!(succs, vec![BlockId(5)]);
    }

    #[test]
    fn test_cfg_return_has_no_successors() {
        let term = Terminator::Return(None);
        assert!(term.successors().is_empty());
    }

    #[test]
    fn test_cfg_branch_has_two_successors() {
        let term = Terminator::Branch {
            cond: bool_expr(true),
            then_block: BlockId(2),
            else_block: BlockId(3),
        };
        let succs = term.successors();
        assert_eq!(succs.len(), 2);
        assert!(succs.contains(&BlockId(2)));
        assert!(succs.contains(&BlockId(3)));
    }
}
