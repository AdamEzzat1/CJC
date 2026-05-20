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

## Slot resolution (T0-b Stage 2, shipped 2026-05-20)

`HirToMir` now performs a single-pass slot-resolution walk over each
function body and emits `MirExprKind::VarLocal { name, slot }` for
references to function-local bindings (params + `let`). References to
top-level functions, captured variables, and pattern-bound names stay
as `MirExprKind::Var(name)` (the executor's scope-chain fallback).
`MirFunction.local_count` records how many slots the function needs.

The synthetic `__main` function and lambda-lifted closures stay at
`local_count = 0` in this stage; they will be lifted in Stage 4. The
executor still does name-based lookup for both variants — the actual
frame-based fast path is Stage 3. See [[ADR-0024 Tier-0 Slot Resolution]]
and [[Tier-0 Interpreter Perf]].

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
