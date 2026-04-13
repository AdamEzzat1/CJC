//! SSA (Static Single Assignment) form construction and verification.
//!
//! Converts a CFG into SSA form by:
//! 1. Computing the dominator tree (from `dominators.rs`)
//! 2. Computing dominance frontiers
//! 3. Inserting phi functions at join points
//! 4. Renaming variables via DFS of the dominator tree
//!
//! The SSA form is stored as an overlay on the original CFG:
//! - Phi nodes are added to each block
//! - Each variable definition gets a unique version number
//!
//! ## Verifier
//!
//! The verifier checks:
//! - Every SSA variable is defined exactly once
//! - Phi nodes have exactly one operand per predecessor
//! - No phi nodes in the entry block (params are pre-defined)
//! - All phi source predecessors are valid

use std::collections::{BTreeMap, BTreeSet};

use crate::cfg::{CfgStmt, MirCfg};
use crate::dominators::DominatorTree;
use crate::BlockId;
use crate::MirExprKind;

// ---------------------------------------------------------------------------
// SSA Types
// ---------------------------------------------------------------------------

/// An SSA variable: a base name with a version (generation) number.
///
/// Version 0 is typically the first definition (e.g., function parameter or
/// first `let` binding). Each subsequent redefinition increments the version.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SsaVar {
    pub name: String,
    pub version: u32,
}

impl std::fmt::Display for SsaVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}_{}", self.name, self.version)
    }
}

/// A phi node at the beginning of a basic block.
///
/// Semantics: `target = phi(source_from_pred0, source_from_pred1, ...)`
///
/// At runtime, the phi selects the value from whichever predecessor was
/// actually taken. In SSA, phi nodes are the only construct that can merge
/// values from different control-flow paths.
#[derive(Debug, Clone)]
pub struct PhiNode {
    /// The variable being defined by this phi.
    pub target: SsaVar,
    /// One source per predecessor block: `(predecessor_block_id, version_of_var)`.
    /// Sorted by predecessor block ID for determinism.
    pub sources: Vec<(BlockId, SsaVar)>,
}

/// SSA form computed from a `MirCfg`.
///
/// This is an *overlay* on the original CFG — the CFG itself is not modified.
/// The SSA form adds phi nodes and tracks variable versioning.
#[derive(Debug, Clone)]
pub struct SsaForm {
    /// Phi nodes for each basic block, indexed by block ID.
    pub phis: Vec<Vec<PhiNode>>,
    /// Number of blocks in the underlying CFG.
    pub num_blocks: usize,
    /// Entry block ID.
    pub entry: BlockId,
    /// Next version counter for each variable (= total versions allocated).
    pub version_counts: BTreeMap<String, u32>,
    /// Maps each definition site `(block_id, var_name, stmt_index)` to its
    /// SSA version. Statement index is within the block's statement list.
    pub def_versions: BTreeMap<(u32, String, usize), u32>,
    /// Parameter names (defined at entry with version 0).
    pub params: Vec<String>,
}

// ---------------------------------------------------------------------------
// Helpers: collect variable definitions from CFG
// ---------------------------------------------------------------------------

/// Collect all variable names that are defined (via `let` or assignment) in the CFG.
///
/// Scan every basic block's statements for `CfgStmt::Let` and
/// `CfgStmt::Expr(Assign { target: Var(name), .. })` patterns and return
/// the deduplicated set of variable names.  The result is a `BTreeSet` for
/// deterministic iteration order.
fn collect_defined_vars(cfg: &MirCfg) -> BTreeSet<String> {
    let mut vars = BTreeSet::new();
    for block in &cfg.basic_blocks {
        for stmt in &block.statements {
            match stmt {
                CfgStmt::Let { name, .. } => {
                    vars.insert(name.clone());
                }
                CfgStmt::Expr(expr) => {
                    if let MirExprKind::Assign { target, .. } = &expr.kind {
                        if let MirExprKind::Var(name) = &target.kind {
                            vars.insert(name.clone());
                        }
                    }
                }
            }
        }
    }
    vars
}

/// For each variable, collect which blocks contain a definition of it.
///
/// Return a map from variable name to the set of `BlockId`s where that
/// variable is defined (via `let` or assignment).  Uses `BTreeMap` and
/// `BTreeSet` for deterministic iteration.
fn collect_def_blocks(
    cfg: &MirCfg,
    variables: &BTreeSet<String>,
) -> BTreeMap<String, BTreeSet<BlockId>> {
    let mut def_blocks: BTreeMap<String, BTreeSet<BlockId>> = BTreeMap::new();
    for var in variables {
        def_blocks.insert(var.clone(), BTreeSet::new());
    }

    for block in &cfg.basic_blocks {
        for stmt in &block.statements {
            match stmt {
                CfgStmt::Let { name, .. } => {
                    if let Some(set) = def_blocks.get_mut(name) {
                        set.insert(block.id);
                    }
                }
                CfgStmt::Expr(expr) => {
                    if let MirExprKind::Assign { target, .. } = &expr.kind {
                        if let MirExprKind::Var(name) = &target.kind {
                            if let Some(set) = def_blocks.get_mut(name) {
                                set.insert(block.id);
                            }
                        }
                    }
                }
            }
        }
    }

    def_blocks
}

// ---------------------------------------------------------------------------
// SSA rename helpers (free functions for borrow-checker friendliness)
// ---------------------------------------------------------------------------

/// Allocate a fresh SSA version number for the given variable.
///
/// Increment the version counter and push the new version onto the
/// variable's rename stack.  Return the newly allocated version.
fn fresh_version(
    name: &str,
    counters: &mut BTreeMap<String, u32>,
    stacks: &mut BTreeMap<String, Vec<u32>>,
) -> u32 {
    let c = counters.get_mut(name).unwrap();
    let ver = *c;
    *c += 1;
    stacks.get_mut(name).unwrap().push(ver);
    ver
}

/// Return the current (top-of-stack) SSA version for a variable, or `None`
/// if the variable has no version on the stack yet.
fn current_version(name: &str, stacks: &BTreeMap<String, Vec<u32>>) -> Option<u32> {
    stacks.get(name).and_then(|s| s.last().copied())
}

/// DFS through the dominator tree, renaming variable definitions and filling
/// in phi sources.
fn rename_dfs(
    block_id: BlockId,
    cfg: &MirCfg,
    domtree: &DominatorTree,
    preds: &[Vec<BlockId>],
    counters: &mut BTreeMap<String, u32>,
    stacks: &mut BTreeMap<String, Vec<u32>>,
    result_phis: &mut [Vec<PhiNode>],
    def_versions: &mut BTreeMap<(u32, String, usize), u32>,
) {
    let b = block_id.0 as usize;

    // Track how many versions we push per variable (to undo on backtrack).
    let mut push_counts: Vec<(String, usize)> = Vec::new();

    // --- Rename phi targets: each phi defines a new version ---
    for phi in &mut result_phis[b] {
        let ver = fresh_version(&phi.target.name, counters, stacks);
        phi.target.version = ver;

        if let Some(entry) = push_counts.iter_mut().find(|(n, _)| *n == phi.target.name) {
            entry.1 += 1;
        } else {
            push_counts.push((phi.target.name.clone(), 1));
        }
    }

    // --- Rename definitions in block statements ---
    let block = &cfg.basic_blocks[b];
    for (idx, stmt) in block.statements.iter().enumerate() {
        match stmt {
            CfgStmt::Let { name, .. } => {
                // The `let` defines a new version of the variable.
                // (Uses in the initializer see the *previous* version,
                // which is correctly the top of the stack before this push.)
                let ver = fresh_version(name, counters, stacks);
                def_versions.insert((block_id.0, name.clone(), idx), ver);

                if let Some(entry) = push_counts.iter_mut().find(|(n, _)| n == name) {
                    entry.1 += 1;
                } else {
                    push_counts.push((name.clone(), 1));
                }
            }
            CfgStmt::Expr(expr) => {
                if let MirExprKind::Assign { target, .. } = &expr.kind {
                    if let MirExprKind::Var(name) = &target.kind {
                        let ver = fresh_version(name, counters, stacks);
                        def_versions.insert((block_id.0, name.clone(), idx), ver);

                        if let Some(entry) = push_counts.iter_mut().find(|(n, _)| n == name) {
                            entry.1 += 1;
                        } else {
                            push_counts.push((name.clone(), 1));
                        }
                    }
                }
            }
        }
    }

    // --- Fill in phi sources for successor blocks ---
    for succ_id in cfg.successors(block_id) {
        let s = succ_id.0 as usize;
        // Find which predecessor index we are for this successor.
        let pred_idx = preds[s].iter().position(|p| *p == block_id);
        if let Some(j) = pred_idx {
            for phi in &mut result_phis[s] {
                let var_name = &phi.target.name;
                if let Some(ver) = current_version(var_name, stacks) {
                    if j < phi.sources.len() {
                        phi.sources[j] = (
                            block_id,
                            SsaVar {
                                name: var_name.clone(),
                                version: ver,
                            },
                        );
                    }
                }
            }
        }
    }

    // --- Recurse into dominator tree children ---
    let children = domtree.children(block_id);
    for child in children {
        rename_dfs(
            child,
            cfg,
            domtree,
            preds,
            counters,
            stacks,
            result_phis,
            def_versions,
        );
    }

    // --- Pop versions (backtrack) ---
    for (name, count) in &push_counts {
        let stack = stacks.get_mut(name).unwrap();
        for _ in 0..*count {
            stack.pop();
        }
    }
}

// ---------------------------------------------------------------------------
// SSA Construction
// ---------------------------------------------------------------------------

impl SsaForm {
    /// Construct SSA form from a CFG.
    ///
    /// `params` are the function parameter names, which are considered
    /// defined at the entry block with version 0.
    ///
    /// Uses the standard Cytron et al. algorithm:
    /// 1. Compute dominator tree and dominance frontiers
    /// 2. Insert phi functions at dominance frontiers
    /// 3. Rename variables via DFS of dominator tree
    pub fn construct(cfg: &MirCfg, params: &[String]) -> Self {
        let n = cfg.basic_blocks.len();
        if n == 0 {
            return SsaForm {
                phis: vec![],
                num_blocks: 0,
                entry: cfg.entry,
                version_counts: BTreeMap::new(),
                def_versions: BTreeMap::new(),
                params: params.to_vec(),
            };
        }

        let domtree = DominatorTree::compute(cfg);
        let df = domtree.dominance_frontiers(cfg);
        let preds = cfg.predecessors();

        // Collect all variable names and their defining blocks.
        let mut variables = collect_defined_vars(cfg);
        for p in params {
            variables.insert(p.clone());
        }
        let def_blocks = collect_def_blocks(cfg, &variables);

        // ── Phase 1: Phi placement ───────────────────────────────────
        // For each variable, compute which blocks need a phi node.
        let mut phi_vars: Vec<BTreeSet<String>> = vec![BTreeSet::new(); n];

        for (var, defs) in &def_blocks {
            let mut worklist: Vec<BlockId> = defs.iter().copied().collect();
            // Parameters are defined at entry.
            if params.contains(var) {
                worklist.push(cfg.entry);
            }
            let mut has_phi: BTreeSet<u32> = BTreeSet::new();
            let mut ever_on_worklist: BTreeSet<u32> =
                worklist.iter().map(|b| b.0).collect();

            while let Some(block) = worklist.pop() {
                for &frontier_block in &df[block.0 as usize] {
                    if has_phi.insert(frontier_block.0) {
                        phi_vars[frontier_block.0 as usize].insert(var.clone());
                        // A phi is also a definition, so add to worklist.
                        if ever_on_worklist.insert(frontier_block.0) {
                            worklist.push(frontier_block);
                        }
                    }
                }
            }
        }

        // ── Phase 2: Create phi node structures ──────────────────────
        let mut result_phis: Vec<Vec<PhiNode>> = Vec::with_capacity(n);
        for b in 0..n {
            let mut block_phis = Vec::new();
            // Sort variable names for determinism.
            let sorted_vars: Vec<String> = phi_vars[b].iter().cloned().collect();
            for var in sorted_vars {
                let pred_count = preds[b].len();
                block_phis.push(PhiNode {
                    // Version will be set during rename.
                    target: SsaVar {
                        name: var.clone(),
                        version: 0,
                    },
                    // Placeholder sources — filled during rename.
                    sources: vec![
                        (
                            BlockId(u32::MAX),
                            SsaVar {
                                name: var,
                                version: 0,
                            }
                        );
                        pred_count
                    ],
                });
            }
            result_phis.push(block_phis);
        }

        // ── Phase 3: Rename via DFS of dominator tree ────────────────
        let mut counters: BTreeMap<String, u32> = BTreeMap::new();
        let mut stacks: BTreeMap<String, Vec<u32>> = BTreeMap::new();
        for var in &variables {
            counters.insert(var.clone(), 0);
            stacks.insert(var.clone(), Vec::new());
        }

        // Parameters get version 0 at entry.
        for p in params {
            let _ = fresh_version(p, &mut counters, &mut stacks);
        }

        let mut def_versions: BTreeMap<(u32, String, usize), u32> = BTreeMap::new();

        rename_dfs(
            cfg.entry,
            cfg,
            &domtree,
            &preds,
            &mut counters,
            &mut stacks,
            &mut result_phis,
            &mut def_versions,
        );

        // Sort phi sources by predecessor block ID for determinism.
        for block_phis in &mut result_phis {
            for phi in block_phis.iter_mut() {
                phi.sources.sort_by_key(|(bid, _)| bid.0);
            }
        }

        SsaForm {
            phis: result_phis,
            num_blocks: n,
            entry: cfg.entry,
            version_counts: counters,
            def_versions,
            params: params.to_vec(),
        }
    }

    /// Total number of phi nodes across all blocks.
    pub fn phi_count(&self) -> usize {
        self.phis.iter().map(|p| p.len()).sum()
    }

    /// Get phi nodes for a specific block.
    pub fn block_phis(&self, block: BlockId) -> &[PhiNode] {
        &self.phis[block.0 as usize]
    }

    /// Get the SSA version assigned to a definition at a specific site.
    pub fn def_version(&self, block: BlockId, var: &str, stmt_idx: usize) -> Option<u32> {
        self.def_versions.get(&(block.0, var.to_string(), stmt_idx)).copied()
    }

    /// Total number of SSA versions created (across all variables).
    pub fn total_versions(&self) -> u32 {
        self.version_counts.values().sum()
    }
}

// ---------------------------------------------------------------------------
// SSA Verifier
// ---------------------------------------------------------------------------

/// An error found during SSA verification.
///
/// Each variant describes a specific invariant violation detected by
/// [`verify_ssa`].
#[derive(Debug, Clone)]
pub enum SsaError {
    /// A variable is defined more than once (violates single-assignment).
    DuplicateDefinition {
        var: SsaVar,
        block1: BlockId,
        block2: BlockId,
    },
    /// A phi node has wrong number of sources (should equal predecessor count).
    PhiSourceCount {
        block: BlockId,
        var: SsaVar,
        expected: usize,
        got: usize,
    },
    /// A phi source references a predecessor that doesn't exist.
    PhiInvalidPredecessor {
        block: BlockId,
        var: SsaVar,
        pred: BlockId,
    },
    /// Entry block has phi nodes (params are pre-defined, not phi'd).
    EntryHasPhi {
        var: SsaVar,
    },
}

impl std::fmt::Display for SsaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SsaError::DuplicateDefinition { var, block1, block2 } => write!(
                f,
                "SSA: {} defined in both block {} and block {}",
                var, block1.0, block2.0
            ),
            SsaError::PhiSourceCount {
                block,
                var,
                expected,
                got,
            } => write!(
                f,
                "SSA: phi for {} in block {} has {} sources, expected {}",
                var, block.0, got, expected
            ),
            SsaError::PhiInvalidPredecessor { block, var, pred } => write!(
                f,
                "SSA: phi for {} in block {} references non-predecessor block {}",
                var, block.0, pred.0
            ),
            SsaError::EntryHasPhi { var } => {
                write!(f, "SSA: entry block has phi for {}", var)
            }
        }
    }
}

/// Verify SSA form properties.
///
/// Checks:
/// 1. No phi nodes in the entry block
/// 2. Each phi has exactly one source per predecessor
/// 3. Each phi source references a valid predecessor
/// 4. No duplicate SSA variable definitions
pub fn verify_ssa(ssa: &SsaForm, cfg: &MirCfg) -> Result<(), Vec<SsaError>> {
    let mut errors = Vec::new();
    let preds = cfg.predecessors();

    // Check 1: No phi nodes in entry block.
    for phi in &ssa.phis[ssa.entry.0 as usize] {
        errors.push(SsaError::EntryHasPhi {
            var: phi.target.clone(),
        });
    }

    // Check 2 & 3: Phi source count and validity.
    for (b, block_phis) in ssa.phis.iter().enumerate() {
        let pred_count = preds[b].len();
        for phi in block_phis {
            if phi.sources.len() != pred_count {
                errors.push(SsaError::PhiSourceCount {
                    block: BlockId(b as u32),
                    var: phi.target.clone(),
                    expected: pred_count,
                    got: phi.sources.len(),
                });
            }
            for (src_block, _) in &phi.sources {
                if !preds[b].contains(src_block) {
                    errors.push(SsaError::PhiInvalidPredecessor {
                        block: BlockId(b as u32),
                        var: phi.target.clone(),
                        pred: *src_block,
                    });
                }
            }
        }
    }

    // Check 4: No duplicate definitions.
    let mut def_locations: BTreeMap<SsaVar, BlockId> = BTreeMap::new();

    // Phi definitions.
    for (b, block_phis) in ssa.phis.iter().enumerate() {
        let block_id = BlockId(b as u32);
        for phi in block_phis {
            if let Some(&prev_block) = def_locations.get(&phi.target) {
                errors.push(SsaError::DuplicateDefinition {
                    var: phi.target.clone(),
                    block1: prev_block,
                    block2: block_id,
                });
            } else {
                def_locations.insert(phi.target.clone(), block_id);
            }
        }
    }

    // Statement definitions.
    for ((block_num, name, _idx), ver) in &ssa.def_versions {
        let var = SsaVar {
            name: name.clone(),
            version: *ver,
        };
        let block_id = BlockId(*block_num);
        if let Some(&prev_block) = def_locations.get(&var) {
            errors.push(SsaError::DuplicateDefinition {
                var,
                block1: prev_block,
                block2: block_id,
            });
        } else {
            def_locations.insert(var, block_id);
        }
    }

    // Parameter definitions (version 0 at entry).
    for p in &ssa.params {
        let var = SsaVar {
            name: p.clone(),
            version: 0,
        };
        // Params don't conflict with themselves; just register if not present.
        def_locations.entry(var).or_insert(ssa.entry);
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{BasicBlock, CfgBuilder, Terminator};
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

    fn var_expr(name: &str) -> MirExpr {
        MirExpr {
            kind: MirExprKind::Var(name.to_string()),
        }
    }

    fn assign_expr(name: &str, value: MirExpr) -> MirExpr {
        MirExpr {
            kind: MirExprKind::Assign {
                target: Box::new(var_expr(name)),
                value: Box::new(value),
            },
        }
    }

    // ── Single block: no phis needed ─────────────────────────────────

    #[test]
    fn test_ssa_single_block_no_phis() {
        let cfg = MirCfg {
            basic_blocks: vec![BasicBlock {
                id: BlockId(0),
                statements: vec![CfgStmt::Let {
                    name: "x".into(),
                    mutable: false,
                    init: int_expr(42),
                }],
                terminator: Terminator::Return(None),
            }],
            entry: BlockId(0),
        };
        let ssa = SsaForm::construct(&cfg, &[]);
        assert_eq!(ssa.phi_count(), 0, "single block should have no phis");
        assert!(verify_ssa(&ssa, &cfg).is_ok());
    }

    // ── Linear chain: no phis ────────────────────────────────────────

    #[test]
    fn test_ssa_linear_chain_no_phis() {
        let cfg = MirCfg {
            basic_blocks: vec![
                BasicBlock {
                    id: BlockId(0),
                    statements: vec![CfgStmt::Let {
                        name: "x".into(),
                        mutable: true,
                        init: int_expr(1),
                    }],
                    terminator: Terminator::Goto(BlockId(1)),
                },
                BasicBlock {
                    id: BlockId(1),
                    statements: vec![CfgStmt::Expr(assign_expr("x", int_expr(2)))],
                    terminator: Terminator::Return(None),
                },
            ],
            entry: BlockId(0),
        };
        let ssa = SsaForm::construct(&cfg, &[]);
        // No join points, so no phis needed.
        assert_eq!(ssa.phi_count(), 0);
        assert!(verify_ssa(&ssa, &cfg).is_ok());
    }

    // ── Diamond: phi at merge point ──────────────────────────────────

    #[test]
    fn test_ssa_diamond_has_phi_at_merge() {
        // Block 0: define x, branch to 1 or 2
        // Block 1: x = 10, goto 3
        // Block 2: x = 20, goto 3
        // Block 3: merge — should have phi for x
        let cfg = MirCfg {
            basic_blocks: vec![
                BasicBlock {
                    id: BlockId(0),
                    statements: vec![CfgStmt::Let {
                        name: "x".into(),
                        mutable: true,
                        init: int_expr(0),
                    }],
                    terminator: Terminator::Branch {
                        cond: bool_expr(true),
                        then_block: BlockId(1),
                        else_block: BlockId(2),
                    },
                },
                BasicBlock {
                    id: BlockId(1),
                    statements: vec![CfgStmt::Expr(assign_expr("x", int_expr(10)))],
                    terminator: Terminator::Goto(BlockId(3)),
                },
                BasicBlock {
                    id: BlockId(2),
                    statements: vec![CfgStmt::Expr(assign_expr("x", int_expr(20)))],
                    terminator: Terminator::Goto(BlockId(3)),
                },
                BasicBlock {
                    id: BlockId(3),
                    statements: vec![],
                    terminator: Terminator::Return(None),
                },
            ],
            entry: BlockId(0),
        };
        let ssa = SsaForm::construct(&cfg, &[]);

        // Block 3 should have a phi for x.
        let merge_phis = ssa.block_phis(BlockId(3));
        assert_eq!(merge_phis.len(), 1, "merge block should have 1 phi");
        assert_eq!(merge_phis[0].target.name, "x");
        assert_eq!(merge_phis[0].sources.len(), 2, "phi should have 2 sources");

        // Verify SSA form.
        assert!(verify_ssa(&ssa, &cfg).is_ok());
    }

    // ── While loop: phi at header ────────────────────────────────────

    #[test]
    fn test_ssa_while_loop_phi_at_header() {
        // Block 0: let i = 0; goto 1
        // Block 1 (header): branch(i < 5) -> 2 or 3
        // Block 2 (body): i = i + 1; goto 1  (back-edge)
        // Block 3 (exit): return
        let cfg = MirCfg {
            basic_blocks: vec![
                BasicBlock {
                    id: BlockId(0),
                    statements: vec![CfgStmt::Let {
                        name: "i".into(),
                        mutable: true,
                        init: int_expr(0),
                    }],
                    terminator: Terminator::Goto(BlockId(1)),
                },
                BasicBlock {
                    id: BlockId(1),
                    statements: vec![],
                    terminator: Terminator::Branch {
                        cond: bool_expr(true), // placeholder
                        then_block: BlockId(2),
                        else_block: BlockId(3),
                    },
                },
                BasicBlock {
                    id: BlockId(2),
                    statements: vec![CfgStmt::Expr(assign_expr("i", int_expr(1)))],
                    terminator: Terminator::Goto(BlockId(1)), // back-edge
                },
                BasicBlock {
                    id: BlockId(3),
                    statements: vec![],
                    terminator: Terminator::Return(None),
                },
            ],
            entry: BlockId(0),
        };
        let ssa = SsaForm::construct(&cfg, &[]);

        // Block 1 (loop header) should have a phi for i.
        let header_phis = ssa.block_phis(BlockId(1));
        assert_eq!(header_phis.len(), 1, "loop header should have 1 phi");
        assert_eq!(header_phis[0].target.name, "i");
        // Two predecessors: block 0 (initial) and block 2 (back-edge).
        assert_eq!(header_phis[0].sources.len(), 2);

        assert!(verify_ssa(&ssa, &cfg).is_ok());
    }

    // ── Parameters get version 0 ─────────────────────────────────────

    #[test]
    fn test_ssa_params_version_zero() {
        let cfg = MirCfg {
            basic_blocks: vec![BasicBlock {
                id: BlockId(0),
                statements: vec![],
                terminator: Terminator::Return(None),
            }],
            entry: BlockId(0),
        };
        let params = vec!["a".to_string(), "b".to_string()];
        let ssa = SsaForm::construct(&cfg, &params);

        // a and b should each have version count >= 1.
        assert!(ssa.version_counts["a"] >= 1);
        assert!(ssa.version_counts["b"] >= 1);
        assert!(verify_ssa(&ssa, &cfg).is_ok());
    }

    // ── Multiple variables, diamond ──────────────────────────────────

    #[test]
    fn test_ssa_multiple_vars_diamond() {
        // Two variables defined in both branches.
        let cfg = MirCfg {
            basic_blocks: vec![
                BasicBlock {
                    id: BlockId(0),
                    statements: vec![
                        CfgStmt::Let {
                            name: "x".into(),
                            mutable: true,
                            init: int_expr(0),
                        },
                        CfgStmt::Let {
                            name: "y".into(),
                            mutable: true,
                            init: int_expr(0),
                        },
                    ],
                    terminator: Terminator::Branch {
                        cond: bool_expr(true),
                        then_block: BlockId(1),
                        else_block: BlockId(2),
                    },
                },
                BasicBlock {
                    id: BlockId(1),
                    statements: vec![
                        CfgStmt::Expr(assign_expr("x", int_expr(10))),
                        CfgStmt::Expr(assign_expr("y", int_expr(100))),
                    ],
                    terminator: Terminator::Goto(BlockId(3)),
                },
                BasicBlock {
                    id: BlockId(2),
                    statements: vec![
                        CfgStmt::Expr(assign_expr("x", int_expr(20))),
                        CfgStmt::Expr(assign_expr("y", int_expr(200))),
                    ],
                    terminator: Terminator::Goto(BlockId(3)),
                },
                BasicBlock {
                    id: BlockId(3),
                    statements: vec![],
                    terminator: Terminator::Return(None),
                },
            ],
            entry: BlockId(0),
        };
        let ssa = SsaForm::construct(&cfg, &[]);

        // Block 3 should have phis for both x and y.
        let merge_phis = ssa.block_phis(BlockId(3));
        assert_eq!(merge_phis.len(), 2, "merge should have 2 phis");

        let phi_names: Vec<&str> = merge_phis.iter().map(|p| p.target.name.as_str()).collect();
        assert!(phi_names.contains(&"x"));
        assert!(phi_names.contains(&"y"));

        assert!(verify_ssa(&ssa, &cfg).is_ok());
    }

    // ── From MirBody (via CfgBuilder) ────────────────────────────────

    #[test]
    fn test_ssa_from_mir_body_if_else() {
        let body = MirBody {
            stmts: vec![
                MirStmt::Let {
                    name: "x".into(),
                    mutable: true,
                    init: int_expr(0),
                    alloc_hint: None,
                },
                MirStmt::If {
                    cond: bool_expr(true),
                    then_body: MirBody {
                        stmts: vec![MirStmt::Expr(assign_expr("x", int_expr(1)))],
                        result: None,
                    },
                    else_body: Some(MirBody {
                        stmts: vec![MirStmt::Expr(assign_expr("x", int_expr(2)))],
                        result: None,
                    }),
                },
            ],
            result: Some(Box::new(var_expr("x"))),
        };
        let cfg = CfgBuilder::build(&body);
        let ssa = SsaForm::construct(&cfg, &[]);

        // There should be a phi for x at the merge block.
        assert!(ssa.phi_count() >= 1, "should have at least 1 phi");
        assert!(verify_ssa(&ssa, &cfg).is_ok());
    }

    #[test]
    fn test_ssa_from_mir_body_while() {
        let body = MirBody {
            stmts: vec![
                MirStmt::Let {
                    name: "i".into(),
                    mutable: true,
                    init: int_expr(0),
                    alloc_hint: None,
                },
                MirStmt::While {
                    cond: bool_expr(true),
                    body: MirBody {
                        stmts: vec![MirStmt::Expr(assign_expr("i", int_expr(1)))],
                        result: None,
                    },
                },
            ],
            result: None,
        };
        let cfg = CfgBuilder::build(&body);
        let ssa = SsaForm::construct(&cfg, &[]);

        assert!(ssa.phi_count() >= 1, "while loop should produce at least 1 phi");
        assert!(verify_ssa(&ssa, &cfg).is_ok());
    }

    // ── Empty CFG ────────────────────────────────────────────────────

    #[test]
    fn test_ssa_empty_cfg() {
        let cfg = MirCfg {
            basic_blocks: vec![BasicBlock {
                id: BlockId(0),
                statements: vec![],
                terminator: Terminator::Return(None),
            }],
            entry: BlockId(0),
        };
        let ssa = SsaForm::construct(&cfg, &[]);
        assert_eq!(ssa.phi_count(), 0);
        assert_eq!(ssa.total_versions(), 0);
        assert!(verify_ssa(&ssa, &cfg).is_ok());
    }

    // ── Versioning correctness ───────────────────────────────────────

    #[test]
    fn test_ssa_version_numbering() {
        // x = 0 in block 0, x = 1 in block 1 (linear).
        let cfg = MirCfg {
            basic_blocks: vec![
                BasicBlock {
                    id: BlockId(0),
                    statements: vec![CfgStmt::Let {
                        name: "x".into(),
                        mutable: true,
                        init: int_expr(0),
                    }],
                    terminator: Terminator::Goto(BlockId(1)),
                },
                BasicBlock {
                    id: BlockId(1),
                    statements: vec![CfgStmt::Expr(assign_expr("x", int_expr(1)))],
                    terminator: Terminator::Return(None),
                },
            ],
            entry: BlockId(0),
        };
        let ssa = SsaForm::construct(&cfg, &[]);

        // x should have 2 versions: x_0 (block 0, let) and x_1 (block 1, assign).
        assert_eq!(ssa.version_counts["x"], 2);
        assert_eq!(ssa.def_version(BlockId(0), "x", 0), Some(0));
        assert_eq!(ssa.def_version(BlockId(1), "x", 0), Some(1));
        assert!(verify_ssa(&ssa, &cfg).is_ok());
    }

    // ── Phi sources match predecessors ───────────────────────────────

    #[test]
    fn test_ssa_phi_sources_match_predecessors() {
        let cfg = MirCfg {
            basic_blocks: vec![
                BasicBlock {
                    id: BlockId(0),
                    statements: vec![CfgStmt::Let {
                        name: "x".into(),
                        mutable: true,
                        init: int_expr(0),
                    }],
                    terminator: Terminator::Branch {
                        cond: bool_expr(true),
                        then_block: BlockId(1),
                        else_block: BlockId(2),
                    },
                },
                BasicBlock {
                    id: BlockId(1),
                    statements: vec![CfgStmt::Expr(assign_expr("x", int_expr(10)))],
                    terminator: Terminator::Goto(BlockId(3)),
                },
                BasicBlock {
                    id: BlockId(2),
                    statements: vec![CfgStmt::Expr(assign_expr("x", int_expr(20)))],
                    terminator: Terminator::Goto(BlockId(3)),
                },
                BasicBlock {
                    id: BlockId(3),
                    statements: vec![],
                    terminator: Terminator::Return(None),
                },
            ],
            entry: BlockId(0),
        };
        let ssa = SsaForm::construct(&cfg, &[]);
        let phi = &ssa.block_phis(BlockId(3))[0];

        // Sources should come from blocks 1 and 2 (the predecessors of block 3).
        let src_blocks: Vec<u32> = phi.sources.iter().map(|(bid, _)| bid.0).collect();
        assert!(src_blocks.contains(&1));
        assert!(src_blocks.contains(&2));

        // The versions should be different (one from each branch).
        let src_versions: Vec<u32> = phi.sources.iter().map(|(_, v)| v.version).collect();
        assert_ne!(src_versions[0], src_versions[1]);
    }

    // ── Determinism ──────────────────────────────────────────────────

    #[test]
    fn test_ssa_deterministic() {
        let body = MirBody {
            stmts: vec![
                MirStmt::Let {
                    name: "a".into(),
                    mutable: true,
                    init: int_expr(0),
                    alloc_hint: None,
                },
                MirStmt::If {
                    cond: bool_expr(true),
                    then_body: MirBody {
                        stmts: vec![MirStmt::Expr(assign_expr("a", int_expr(1)))],
                        result: None,
                    },
                    else_body: Some(MirBody {
                        stmts: vec![MirStmt::Expr(assign_expr("a", int_expr(2)))],
                        result: None,
                    }),
                },
            ],
            result: None,
        };
        let cfg = CfgBuilder::build(&body);

        let ssa1 = SsaForm::construct(&cfg, &[]);
        let ssa2 = SsaForm::construct(&cfg, &[]);

        assert_eq!(ssa1.phi_count(), ssa2.phi_count());
        assert_eq!(ssa1.version_counts, ssa2.version_counts);
        assert_eq!(ssa1.def_versions, ssa2.def_versions);

        for (b, (p1, p2)) in ssa1.phis.iter().zip(ssa2.phis.iter()).enumerate() {
            assert_eq!(p1.len(), p2.len(), "phi count mismatch at block {}", b);
            for (phi1, phi2) in p1.iter().zip(p2.iter()) {
                assert_eq!(phi1.target, phi2.target);
                assert_eq!(phi1.sources.len(), phi2.sources.len());
            }
        }
    }
}
