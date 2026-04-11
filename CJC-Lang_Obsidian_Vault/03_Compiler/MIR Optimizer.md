---
title: MIR Optimizer
tags: [compiler, optimization]
status: Implemented
---

# MIR Optimizer

**Source**: `crates/cjc-mir/src/optimize.rs` and `crates/cjc-mir/src/ssa_optimize.rs`.

## Passes implemented

### Classical pipeline (`optimize.rs`)

1. **Constant Folding (CF)** — fold compile-time computable expressions.
2. **Strength Reduction (SR)** — `x * 2 → x << 1` where semantically safe.
3. **Dead Code Elimination (DCE)** — remove statements whose results are unused.
4. **Common Subexpression Elimination (CSE)** — reuse results of identical expressions.
5. **Loop-Invariant Code Motion (LICM)** — hoist invariants out of loops.
6. **Second-round CF** — catches what earlier passes exposed.

### SSA-aware pipeline (`ssa_optimize.rs`)

1. **Constant Folding on CFG**
2. **Sparse Conditional Constant Propagation (SCCP)** — powerful constant propagation that respects branches.
3. **Strength Reduction**
4. **SSA-based DCE**
5. **CFG cleanup** — remove unreachable blocks, merge linear chains.

## Enabled via CLI

```
cjcl run --mir-opt program.cjcl
```

Without `--mir-opt`, [[cjc-mir-exec]] still works — it just runs unoptimized MIR.

## Determinism constraints on the optimizer

The optimizer is subject to hard rules from [[Float Reassociation Policy]]:

- **No reassociation** of floating-point reductions that carry a reduction annotation (checked by `verify.rs`).
- **No introduction of FMA** — even if target hardware supports it, FMA changes rounding.
- **No non-deterministic code motion** — e.g., hoisting an allocation that would change arena/Rc decisions.

The CLAUDE.md prompt also reinforces: "Parallel operations must produce identical results regardless of thread count."

## Output

After optimization, [[MIR]] is re-verified by `verify.rs` before being handed to [[cjc-mir-exec]]. If any invariant is broken, compilation aborts with a diagnostic.

## Related

- [[MIR]]
- [[SSA Form]]
- [[Float Reassociation Policy]]
- [[Determinism Contract]]
- [[Parity Gates]]
- [[CLI Surfaces]]
