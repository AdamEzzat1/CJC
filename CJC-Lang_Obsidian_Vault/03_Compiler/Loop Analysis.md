---
title: Loop Analysis
tags: [compiler, analysis]
status: Implemented
---

# Loop Analysis

Source: `crates/cjc-mir/src/loop_analysis.rs` and `ssa_loop_overlay.rs`.

## Summary

Identifies loops in the [[CFG]] using the [[Dominator Tree]] — backedges (edges from B to A where A dominates B) mark natural loops. Builds a loop tree capturing loop nesting.

## Used by

- **LICM** in the [[MIR Optimizer]] — invariants must be hoisted to the loop preheader.
- **SSA loop overlay** — tracks loop-level metadata on SSA variables.
- **Reduction contract** — loop-carried reductions are tagged so the optimizer refuses to reorder them (see [[Float Reassociation Policy]]).

## Related

- [[CFG]]
- [[Dominator Tree]]
- [[SSA Form]]
- [[MIR Optimizer]]
- [[Float Reassociation Policy]]
