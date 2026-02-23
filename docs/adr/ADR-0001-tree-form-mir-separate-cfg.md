# ADR-0001: Tree-Form MIR + Separate CFG Analysis View

**Status:** Accepted
**Date:** 2024-01-15
**Deciders:** Systems Architect, Compiler Engineer
**Supersedes:** none

## Context

During HIR→MIR lowering design, two competing representations were considered:
1. **CFG-first MIR**: Immediately lower HIR into a control-flow graph with basic blocks and explicit edges. This is the approach taken by LLVM IR, MLIR, and Cranelift.
2. **Tree-form MIR**: Keep MIR as a structured tree (nested `MirBody`, `MirStmt`, `MirExpr`) with a separate derived CFG analysis pass.

The primary tension: CFG-first enables classical optimization passes (SSA, DCE, CSE) but requires dominator tree construction and phi-node insertion before any analysis can run. Tree-form is simpler to lower to, simpler to execute in the reference interpreter, and can have the CFG derived on demand.

Given that CJC's MIR executor (`cjc-mir-exec`) is a **reference interpreter** (not a native code generator), the overhead of SSA construction on every execution would be significant.

## Decision

Adopt **tree-form MIR** as the canonical representation in `cjc-mir/src/lib.rs`. Maintain a **separate derived CFG** in `cjc-mir/src/cfg.rs` built on demand by `CfgBuilder::build(&func.body)`.

The CFG is used exclusively by analysis passes (NoGC verifier, optimizer, shape inference). The executor (`cjc-mir-exec`) operates directly on the tree-form MIR.

## Rationale

- **Simpler execution path**: `MirExecutor::exec_body` traverses a tree — no basic block dispatch loop, no phi-node resolution at runtime.
- **Incremental adoption**: SSA and phi nodes can be layered on top of the CFG analysis view (see ADR-0012) without changing the tree-form or the executor.
- **Compatibility**: The parity gate (G-1/G-2) requires MIR-exec to produce identical results to AST-eval. Tree-form MIR is structurally closer to the AST, making behavioral parity easier to verify.

## Consequences

**Positive:**
- `cjc-mir-exec` interprets MIR without CFG overhead on every call.
- Optimizer operates on derived CFG, leaving the canonical MIR tree immutable.
- Adding new optimization passes does not require changes to the executor.

**Known limitations:**
- Classical register allocation and instruction selection require true SSA form; this decision defers those to a native backend phase.
- The derived CFG must be rebuilt each time the MIR changes, adding overhead to analysis-heavy workloads.

## Implementation Notes

- Crates affected: `cjc-mir`, `cjc-mir-exec`
- Files: `crates/cjc-mir/src/lib.rs` (tree form), `crates/cjc-mir/src/cfg.rs` (derived CFG)
- Regression gate: `cargo test --workspace` must pass with 0 failures
