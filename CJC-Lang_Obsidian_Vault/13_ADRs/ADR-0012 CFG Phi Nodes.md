---
title: ADR-0012 CFG Phi Nodes
tags: [adr, proposed, mir, optimizer]
status: Proposed
date: 2025-01-01
source: docs/adr/ADR-0012-true-cfg-phi-nodes.md
---

# ADR-0012 — True CFG with Phi Nodes and Use-Def Chains

**Status:** Proposed · **Date:** 2025-01-01 · **Extends:** [[ADR-0001 Tree-form MIR]] (does not supersede)

## The decision (proposed)

Add **SSA machinery as a view on top of the derived CFG** in `cjc-mir/src/cfg.rs`:

1. **Phi nodes** at basic block entry: `PhiNode { result: TempId, incoming: Vec<(BlockId, TempId)> }`
2. **Use-def chains**: `UseDefChain { defs, uses }` — exact "which temp is defined/used where"
3. **Dominator tree** via Cooper et al. iterative algorithm: `DomTree { idom, frontier }`
4. **Minimal phi insertion** for optimizer use (not full SSA renaming — that's Phase 4)
5. **DCE rewrite** to use use-def chains instead of today's `HashSet<String>`

**Tree-form MIR and both executors remain unchanged.** This is a pure analysis extension.

## Why this matters

- **Accurate DCE.** Today's string-based liveness gets fooled by shadow variables. Use-def chains are name-agnostic and precise.
- **Future optimizations.** CSE, GVN, and inlining all require SSA-level information. Building the infrastructure now makes those passes straightforward later.
- **Non-breaking.** Because the CFG is a *derived* view (the [[ADR-0001 Tree-form MIR]] contract), extending it with phi nodes does not touch the executor or the parity gates.

## What this constrains

- `TempId(u32)` must stay opaque — the phi-insertion algorithm renumbers freely.
- Dom-tree construction adds O(n log n) per optimization pass; acceptable for functions with < 10k basic blocks.
- Full SSA *renaming* (α-renaming every variable) is **explicitly out of scope** for this ADR.

## Related

- [[ADR-0001 Tree-form MIR]] — the ADR this extends
- [[CFG]], [[SSA Form]], [[Dominator Tree]]
- [[MIR Optimizer]]
- [[ADR Index]]
