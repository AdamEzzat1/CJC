---
title: ADR-0001 Tree-form MIR
tags: [adr, accepted, mir, compiler]
status: Accepted
date: 2024-01-15
source: docs/adr/ADR-0001-tree-form-mir-separate-cfg.md
---

# ADR-0001 — Tree-form MIR with Separate Derived CFG

**Status:** Accepted · **Date:** 2024-01-15

## The decision

MIR is stored as **nested tree-form expressions** (like a slightly-desugared AST), *not* as a flat instruction list with basic blocks. A CFG, when needed, is built as a **derived view** on top of the tree — it is analysis, not ground truth.

## Why this matters

- **Simpler execution.** The MIR executor walks the tree directly; no block scheduling, no register allocation.
- **Cheap parity.** Because the tree shape stays close to the AST, [[Parity Gates]] between [[cjc-eval]] and [[cjc-mir-exec]] are mechanical to maintain.
- **Layered complexity.** SSA, phi nodes, dominators, and loop analysis can all be layered *on* the derived CFG view ([[ADR-0012 CFG Phi Nodes]], [[ADR-0014 MIR Analysis Infrastructure]]) without disturbing the executor.

## Alternatives rejected

- **LLVM-style flat instruction list.** Rejected — would require a separate type-aware VM and break the "one language, two executors with identical output" story.
- **Inline CFG in MIR.** Rejected — forces every consumer to know about blocks even when they only want to walk the tree.

## What this constrains

- Any optimizer pass that needs SSA must build it *from* the CFG view and write results back compatibly ([[MIR Optimizer]]).
- The executor must never assume CFG form; all scheduling is implicit in tree order.

## Related

- [[MIR]], [[CFG]], [[SSA Form]]
- [[ADR-0012 CFG Phi Nodes]] — extends this ADR
- [[ADR-0014 MIR Analysis Infrastructure]] — uses the derived-view pattern
- [[ADR Index]]
