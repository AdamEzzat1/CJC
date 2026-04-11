---
title: SSA Form
tags: [compiler, ir]
status: Implemented
---

# SSA Form

Static Single Assignment: the canonical form of [[MIR]] for the optimizer.

**Source**: `crates/cjc-mir/src/ssa.rs`, `ssa_loop_overlay.rs`, `ssa_optimize.rs`.

## Properties

- Every variable has exactly one definition.
- Join points have **phi nodes** that select among values coming from different predecessor blocks.
- Dominator information from [[Dominator Tree]] drives phi placement.

## Why SSA

- Enables **SCCP** (Sparse Conditional Constant Propagation), a powerful constant propagation that respects branches.
- Simplifies **SSA-DCE** — a definition is dead iff it has no uses.
- Makes data-flow analyses tractable (the analysis state is "which definition reaches here", not "which variable version").

## Pass sequence

1. Build CFG ([[CFG]]).
2. Compute dominators ([[Dominator Tree]]).
3. Insert phi nodes at the dominance frontier of each definition.
4. Rename variables so each has a unique SSA name.
5. Apply SSA-aware passes in `ssa_optimize.rs`.

## Phi node discipline

Phi nodes exist only at non-entry join points. They have exactly one operand per predecessor, preserved in insertion order of predecessors (which is stabilized by [[CFG]] construction using `BTreeMap`).

## Related

- [[MIR]]
- [[CFG]]
- [[Dominator Tree]]
- [[Loop Analysis]]
- [[MIR Optimizer]]
