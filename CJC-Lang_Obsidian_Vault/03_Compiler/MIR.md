---
title: MIR
tags: [compiler, ir]
status: Implemented
---

# MIR

**Crate**: `cjc-mir` — `crates/cjc-mir/src/` (~12,476 LOC total).

## Summary

Mid-level Intermediate Representation: a control-flow graph of basic blocks containing register-style expressions. MIR is the canonical input to the [[MIR Optimizer]] and to [[cjc-mir-exec]].

## Lowering

`HirToMir` takes [[HIR]] and produces a `MirProgram`. Key moves:
- Flatten the tree into blocks.
- Give every intermediate a register identity.
- Build control flow as explicit `BlockId` edges.
- Place `AllocHint` on allocations (`Stack`, `Arena`, `Rc`) — see [[Escape Analysis]].

## Key types

- `MirProgram { functions: Vec<MirFunction> }`
- `MirFunction { id: MirFnId, body: MirBody, local_count: u32, ... }`
- `MirBody` — blocks, edges, entry/exit
- `MirExpr` / `MirExprKind` (notable variants: `Var(String)` unresolved,
  `VarLocal { name, slot }` slot-resolved — see [[Tier-0 Interpreter Perf]])
- `MirStmt`
- `BlockId`, `MirFnId`
- `AllocHint` enum: `Stack`, `Arena`, `Rc`

## Slot resolution (T0-b Stages 2+3+4, shipped 2026-05-20)

`HirToMir` performs a single-pass slot-resolution walk over each
function body — regular fns, lambda-lifted closure bodies, and match
arm bodies all participate. It emits `MirExprKind::VarLocal {
name, slot }` for references to function-local bindings (params +
`let` + match-pattern bindings). References to top-level functions
and unresolved names stay as `MirExprKind::Var(name)` (the
executor's scope-chain fallback). `MirFunction.local_count` records
how many slots the function needs. `MirStmt::Let` and
`MirPattern::Binding` both carry an optional `slot: Option<u32>`
populated by the same pass.

The only path still on the name fallback after Stage 4 is the
synthetic `__main` function (top-level statements). The executor
(`cjc-mir-exec`) reads `VarLocal` through a flat `frame: Vec<Value>`
indexed by `frame_stack.last() + slot` — a single `Vec` access vs
the old `BTreeMap`-keyed scope walk. The double-bookkeeping with
`self.define(name, val)` is kept as a safety net until Stage 5
retires `Var(String)`. **First measurable win**: Stage 4 cut
chess_rl_v2 wall-clock from 802s → 680s (15% speedup) by routing
match expressions inside the per-move scoring loop through the
frame fast-path. See [[ADR-0024 Tier-0 Slot Resolution]] and
[[Tier-0 Interpreter Perf]].

## Passes

The `cjc-mir` crate contains multiple passes, each in its own file:

| File | Pass |
|---|---|
| `cfg.rs` | Control-flow graph construction |
| `dominators.rs` | Dominator tree (Lengauer–Tarjan or similar) |
| `loop_analysis.rs` | Loop tree construction and nesting |
| `ssa.rs` | [[SSA Form]] transformation (phi insertion) |
| `ssa_loop_overlay.rs` | Loop metadata overlay for SSA |
| `optimize.rs` | Classic passes — CF, SR, DCE, CSE, LICM |
| `ssa_optimize.rs` | SSA-aware SCCP + SSA-DCE + CFG cleanup |
| `monomorph.rs` | Monomorphization of generic functions |
| `nogc_verify.rs` | [[NoGC Verifier]] (call-graph fixpoint) |
| `escape.rs` | [[Escape Analysis]] (Stack/Arena/Rc classification) |
| `reduction.rs` | Reduction (fold) annotations and contract |
| `verify.rs` | Legality verifier (CFG structure, SSA soundness, reduction contract) |
| `inspect.rs` | Pretty-print / debug dump for CLI `emit` |

## Invariants

- Every variable defined exactly once (SSA).
- Phi nodes only at non-entry join points, one operand per predecessor.
- Blocks reachable from entry — dead blocks are removed by `ssa_optimize`.
- Reduction annotations are preserved through optimization (no reassociation).
- CFG is constructed with deterministic block ordering (BTreeMap insertion).

## Related

- [[HIR]]
- [[MIR Optimizer]]
- [[SSA Form]]
- [[CFG]]
- [[Dominator Tree]]
- [[Loop Analysis]]
- [[NoGC Verifier]]
- [[Escape Analysis]]
- [[cjc-mir-exec]]
- [[Float Reassociation Policy]]
- [[Tier-0 Interpreter Perf]] — slot-resolution work on top of MIR
- [[ADR-0024 Tier-0 Slot Resolution]]
