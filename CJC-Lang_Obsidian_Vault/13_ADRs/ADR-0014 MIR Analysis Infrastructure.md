---
title: ADR-0014 MIR Analysis Infrastructure
tags: [adr, accepted, mir, analysis, optimizer]
status: Accepted
date: 2026-03-23
source: docs/adr/ADR-0014-mir-analysis-infrastructure.md
---

# ADR-0014 — MIR Analysis Infrastructure: Loop Tree, Reductions, Verifier

**Status:** Accepted (implemented 2026-03-23)

## The decision

Add three new analysis modules to `cjc-mir` as **additive overlays** on the existing MIR/CFG/SSA stack — no existing code changes, no breaking changes:

| Module | File | LOC | What it does |
|---|---|---|---|
| Loop analysis | `loop_analysis.rs` | ~420 | Builds `LoopTree` from CFG + dom tree (back-edge detection, Appel's reverse-predecessor walk, nesting, exits, preheaders) |
| Reduction analysis | `reduction.rs` | ~480 | Detects `acc = acc ⊕ expr` patterns and builtin reductions; classifies them as StrictFold / Kahan / Binned / FixedTree / BuiltinReduction / Unknown |
| Legality verifier | `verify.rs` | ~380 | Checks CFG integrity, loop well-formedness, reduction contracts (StrictFold can't be marked reorderable, etc.), nesting depth < 256 |

**Data structure policy:** everything is `Vec<T>` indexed by a typed newtype ID (`LoopId(u32)`, `ReductionId(u32)`, `block_to_loop: Vec<Option<LoopId>>`). No BTreeMap in the analysis layer. Sorted Vecs for deterministic iteration.

## Why this matters

- **LICM and loop-aware optimization become trivial.** The optimizer no longer needs ad-hoc `is_loop_header()` heuristics; it has a proper loop tree with bodies, headers, exits, and preheaders.
- **The determinism contract becomes explicit in the IR.** A reduction is now *tagged* with whether it can be reordered or parallelized. The verifier rejects inconsistent tagging (e.g., `StrictFold` with `reorderable = true`).
- **Structural corruption fails loudly.** The verifier catches out-of-bounds block references, nesting cycles, and other invariant violations before execution.

## What explicitly did NOT change

- Tree-form MIR ([[ADR-0001 Tree-form MIR]]) — unchanged
- Derived CFG — unchanged
- Cytron minimal SSA overlay — unchanged
- All 6 tree-form optimizer passes — unchanged
- All 6 SSA optimizer passes — unchanged
- NoGC verifier — unchanged
- Escape analysis — unchanged
- All existing tests — unchanged, still passing

This is a textbook application of the derived-view pattern established by [[ADR-0001 Tree-form MIR]].

## Deferred

- **Schedule metadata** (no parallel executor yet — premature)
- **Tiling / vectorization hints** (payoff too small at current scale)
- **Memory SSA / alias analysis** (heavy, not yet needed)
- **Pre/post optimization diffing** (would let the verifier catch optimizer-introduced reduction reordering, but requires two MIR snapshots)

## Tests added

47 new tests total: 8 unit + 7–8 integration per module. All include determinism checks (run twice, compare bytes).

## Related

- [[ADR-0001 Tree-form MIR]] — the derived-view pattern this ADR builds on
- [[MIR Optimizer]]
- [[Loop Analysis]]
- [[ADR Index]]
