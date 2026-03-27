# ADR-0014: MIR Analysis Infrastructure — Loop Tree, Reduction Analysis, Legality Verifier

## Status

Accepted — Implemented 2026-03-23

## Context

CJC's MIR/CFG/SSA stack was functional but had three gaps:

1. **Loop detection was ad-hoc** — `is_loop_header()` used a simple heuristic (predecessor ID >= block ID). No loop bodies, nesting, exits, or preheaders were tracked.

2. **Reductions were invisible** — Accumulation patterns (`acc = acc + x`) and builtin reduction calls (`sum()`, `mean()`) had no MIR-level representation. The optimizer and verifier could not reason about them.

3. **Legality checking was limited to @nogc** — No structural checks for CFG integrity, loop well-formedness, reduction contract preservation, or nesting depth bounds.

## Decision

Add three new analysis modules as **additive overlays** on the existing MIR/CFG/SSA system:

### What was added

| Module | File | Lines | Purpose |
|--------|------|-------|---------|
| Loop Analysis | `loop_analysis.rs` | ~420 | LoopTree from CFG + dominator tree |
| Reduction Analysis | `reduction.rs` | ~480 | Detect and classify accumulation patterns |
| Legality Verifier | `verify.rs` | ~380 | CFG structure, loop integrity, reduction contracts, nesting bounds |

### What was NOT changed

- Tree-form MIR (canonical execution representation) — **unchanged**
- CFG (derived analysis structure) — **unchanged**
- Classical Cytron minimal SSA (overlay) — **unchanged**
- All 6 tree-form optimizer passes — **unchanged**
- All 6 SSA optimizer passes — **unchanged**
- NoGC verifier — **unchanged**
- Escape analysis — **unchanged**
- All existing tests — **unchanged and still passing**

### Data structure policy

All new structures use **Vec + ID indexing** per the data structure policy:

- `LoopId(u32)` indexes into `Vec<LoopInfo>`
- `ReductionId(u32)` indexes into `Vec<ReductionInfo>`
- `block_to_loop: Vec<Option<LoopId>>` indexed by block index
- No new BTreeMap usage in the analysis layer
- Sorted Vecs for deterministic iteration (body_blocks, exit_blocks, etc.)

## Architecture

### Loop Analysis

```
CFG + DominatorTree → compute_loop_tree() → LoopTree
                                               ├── loops: Vec<LoopInfo>     (indexed by LoopId)
                                               ├── block_to_loop: Vec<Option<LoopId>>
                                               └── queries: loop_for_block(), is_nested_in(), etc.
```

Algorithm:
1. Find back-edges (target dominates source)
2. Compute loop body via reverse predecessor walk (Appel's algorithm)
3. Build nesting by header containment (smallest enclosing body = parent)
4. Compute exit blocks and preheaders

Determinism: headers sorted by BlockId → LoopId assignment is deterministic.

### Reduction Analysis

```
MirProgram [+ LoopTree] → detect_reductions() → ReductionReport
                                                    └── reductions: Vec<ReductionInfo>
```

Two detection passes:
1. **Loop accumulation patterns**: `acc = acc ⊕ expr` inside while loops
2. **Builtin reduction calls**: `sum()`, `mean()`, `dot()`, etc.

Classification:

| Kind | Reorder? | Parallel? | Use case |
|------|----------|-----------|----------|
| StrictFold | NO | NO | Sequential accumulation |
| KahanFold | NO | NO | Compensated summation |
| BinnedFold | YES | YES | Order-invariant accumulation |
| FixedTree | NO | YES | Fixed-shape reduction tree |
| BuiltinReduction | Depends | Depends | Runtime-selected |
| Unknown | NO | NO | Conservative default |

### Legality Verifier

```
MirProgram → verify_mir_legality() → LegalityReport
                                        ├── errors: Vec<LegalityError>
                                        ├── checks_passed: u32
                                        └── checks_total: u32
```

Four checks:
1. **CFG structure** — entry is BlockId(0), successor references in bounds, block IDs match indices
2. **Loop integrity** — header in body, back-edge sources in body, parent/child consistency, no nesting cycles, sorted body_blocks
3. **Reduction contracts** — StrictFold not marked reorderable, Unknown not marked parallelizable
4. **Structural bounds** — nesting depth < 256

## Consequences

### Positive

- Loop analysis enables better LICM (no redundant modified-var collection)
- Reduction classification makes the determinism contract explicit and auditable
- Verifier catches structural corruption early
- All additive — zero risk of breaking existing code
- Vec+ID indexing is cache-friendly and allocation-efficient

### Negative

- Slight increase in cjc-mir crate size (~1,280 lines)
- Loop tree computation adds O(V+E) cost per function (only when requested)

### Deferred

- **Schedule metadata** — No parallel executor yet; premature to add
- **Tiling/vectorization hints** — Insufficient payoff at current scale
- **Memory SSA / alias analysis** — Heavy infrastructure, not needed yet
- **Pre/post optimization comparison** — Checking that optimizer didn't reorder reductions would require diffing two MIR snapshots; deferred to future work

## Tests

| Category | Count | Location |
|----------|-------|----------|
| Loop analysis unit tests | 8 | `cjc-mir/src/loop_analysis.rs` |
| Reduction analysis unit tests | 8 | `cjc-mir/src/reduction.rs` |
| Legality verifier unit tests | 8 | `cjc-mir/src/verify.rs` |
| Loop analysis integration tests | 7 | `tests/mir/test_loop_analysis.rs` |
| Reduction analysis integration tests | 8 | `tests/mir/test_reduction_analysis.rs` |
| Legality verifier integration tests | 8 | `tests/mir/test_verifier.rs` |
| **Total new tests** | **47** | |
| **Existing tests (unchanged)** | **1,487+ lib + 561+ integration** | All passing |

All tests include determinism checks (run twice, compare results).
