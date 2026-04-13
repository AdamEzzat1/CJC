---
title: Dominator Tree
tags: [compiler, analysis]
status: Implemented
---

# Dominator Tree

Source: `crates/cjc-mir/src/dominators.rs`.

## Summary

A classical dominator tree computation for the [[CFG]]. Block A dominates block B if every path from entry to B passes through A. The dominator tree captures this relationship and is the foundation for [[SSA Form]] phi placement (dominance frontier) and several optimizer passes.

## Algorithm

Likely Lengauer-Tarjan or Cooper-Harvey-Kennedy (iterative), but check the source for the exact variant. Both are standard and deterministic on a fixed CFG.

## Used by

- [[SSA Form]] — phi placement uses dominance frontier
- [[MIR Optimizer]] — several optimizations exploit dominance (e.g., LICM safety)
- [[Loop Analysis]] — backedge identification

## Related

- [[CFG]]
- [[SSA Form]]
- [[Loop Analysis]]
- [[MIR Optimizer]]
