# ADR-0012: True CFG with Phi Nodes and Use-Def Chains

**Status:** Proposed
**Date:** 2025-01-01
**Deciders:** Compiler Engineer, Systems Architect
**Supersedes:** Extends ADR-0001 (does not supersede it)

## Context

ADR-0001 established tree-form MIR with a **derived** CFG in `cjc-mir/src/cfg.rs`. The current CFG has:
- `BasicBlock { id, statements: Vec<CfgStmt>, terminator: Terminator }`
- No phi nodes
- No use-def chains
- No dominator tree

The optimizer (Stage 2.4) operates on the tree-form MIR directly using string-based liveness sets (`HashSet<String>`). This approach has correctness and performance limitations:

1. **DCE accuracy**: String-based liveness misses cases where the same name is used in different scopes (shadowing). Use-def chains are precise.
2. **CSE impossibility**: Common sub-expression elimination requires knowing that two expressions compute the same value — only possible with SSA naming.
3. **Inlining prerequisites**: Inlining requires renaming variables to avoid capture; SSA form makes this mechanical.

**Scope of this ADR:** Add phi nodes, use-def chains, and a dominator tree to the *existing derived CFG*. The tree-form MIR and the interpreter are **not changed**.

## Decision

Extend `cjc-mir/src/cfg.rs` with:

**1. Phi nodes at basic block entry:**
```rust
pub struct PhiNode {
    pub result: TempId,
    pub incoming: Vec<(BlockId, TempId)>,
}

pub struct BasicBlock {
    pub id: BlockId,
    pub phis: Vec<PhiNode>,          // NEW
    pub statements: Vec<CfgStmt>,
    pub terminator: Terminator,
}
```

**2. Use-def chains:**
```rust
pub struct UseDefChain {
    pub defs: HashMap<TempId, (BlockId, usize)>,
    pub uses: HashMap<TempId, Vec<(BlockId, usize)>>,
}

impl MirCfg {
    pub fn build_use_def(&self) -> UseDefChain;
}
```

**3. Dominator tree (Cooper et al. iterative algorithm):**
```rust
pub struct DomTree {
    pub idom: Vec<Option<BlockId>>,
    pub frontier: Vec<Vec<BlockId>>,
}

impl MirCfg {
    pub fn build_dom_tree(&self) -> DomTree;
}
```

**4. Minimal SSA phi insertion (for optimizer use):**
```rust
impl CfgBuilder {
    pub fn insert_phis(cfg: &mut MirCfg, dom: &DomTree);
}
```

**5. Update optimizer DCE to use use-def chains** instead of `HashSet<String>`.

## Rationale

- **Precision**: Use-def chains identify exactly which computations are dead — no false negatives or false positives from name-based string matching.
- **Future-proofing**: SSA form is the prerequisite for CSE, GVN (global value numbering), and inlining. Building the infrastructure now makes those optimizations straightforward to add.
- **Non-breaking**: The tree-form MIR and executor are unchanged. The CFG remains a derived analysis view.

## Consequences

**Positive:**
- DCE produces provably correct dead code elimination results.
- The optimizer can be extended with CSE and inlining without architectural changes.
- The dominator tree enables future loop analysis (loop invariant code motion).

**Known limitations:**
- `insert_phis` requires the `TempId` naming scheme to be consistent with the phi insertion algorithm. The current `TempId(u32)` opaque ID system supports this.
- Building the dom tree adds O(n log n) overhead to the optimization pass (acceptable for functions with < 10,000 basic blocks).
- Full SSA construction (renaming all variables) is NOT done in this ADR — only phi insertion. Full SSA is a future Phase 4 item.

## Implementation Notes

- Crates affected: `cjc-mir`, `cjc-mir-exec` (optimizer update)
- Files: `crates/cjc-mir/src/cfg.rs` (phi nodes, use-def, dom tree), `crates/cjc-mir/src/optimize.rs` (update DCE)
- New tests: `tests/hardening_tests/test_h4_mir_cfg.rs` (extend existing), `tests/audit_tests/test_audit_cfg_ssa.rs` (new)
- Regression gate: `cargo test --workspace` passes; `cargo test milestone_2_4 -- optimizer` passes
- See also: ADR-0001 (tree-form foundation), ADR-0003 (executor API stability)
