---
title: CFG
tags: [compiler, ir]
status: Implemented
---

# CFG

Control-Flow Graph. Source: `crates/cjc-mir/src/cfg.rs`.

## Summary

MIR organizes code into basic blocks. Each block ends with a terminator — `Br`, `CondBr`, `Return`, etc. — that points to successor blocks by `BlockId`. The CFG is the adjacency structure on blocks.

## Determinism

Blocks are enumerated in deterministic order. Predecessor and successor lists are held in structures that preserve insertion order (or are explicitly sorted by `BlockId`) so that every downstream pass — [[SSA Form]] phi placement, [[MIR Optimizer]] DCE, [[Dominator Tree]] computation — sees the same structure across runs.

## Used by

- [[Dominator Tree]] computation
- [[Loop Analysis]]
- [[SSA Form]] construction
- [[MIR Optimizer]] passes (CFG cleanup, LICM)
- [[cjc-mir-exec]] — walks the CFG to execute

## Related

- [[MIR]]
- [[SSA Form]]
- [[Dominator Tree]]
- [[Loop Analysis]]
