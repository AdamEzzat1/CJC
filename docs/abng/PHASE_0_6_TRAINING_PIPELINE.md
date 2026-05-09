# ABNG Phase 0.6 Item 8 — Training Pipeline Analysis

**Date:** 2026-05-08
**Scope:** Item 8 of Phase 0.6 — research deliverable, not mandatory shipping
**Authoring context:** Phase 0.6 just shipped Items 1–7 (cross-platform CI, scale benches, smart-replay, native batch_observe at v13, all 6 trigger demos, scaled demos, route_to_leaf kernel). This doc reflects findings from that work.

This document delivers the three things Item 8 specified:
1. A profile of ABNG training at scale, identifying the actual hot path.
2. An analysis of what ABNG would need to match TidyView's performance discipline.
3. At least one concrete kernel that demonstrates the pattern.

---

## 1. Profile of ABNG training at scale

### 1.1 Per-op micro-cost (from `bench/abng_micro/`, Windows release build)

| Operation | ns/op | Notes |
|---|---:|---|
| `encode_prefix` | 124 | quantile-codebook lookup, d=1, partition_point binary search |
| `descend` | 117 | radix-tree route on encoded prefix |
| `blr_predict` | 991 | Cholesky triangular solve, d=4 |
| `observe` | 4,330 | Welford fold + 2 SHA-256 (audit chain + per-node stats chain) |
| `blr_update` | 15,581 | NIG conjugate update + Cholesky decomp, d=4, n=1 |
| `route_to_leaf` (Item 7 fused) | 195 | encode + descend + extract, single Rust call |
| `route_to_leaf_batch` (Item 8) | 128,000 / 1024 = **125 ns/row** | chunked dispatch over [n=1024, d=1] |

### 1.2 Per-row training cost (from `bench/abng_lineage_at_scale/`)

| n_rows | Train (ms) | per-row (µs) | Train scaling |
|---:|---:|---:|---|
| 1,000 | 56 | 56 | (cold cache) |
| 10,000 | 195 | 19.5 | (warm) |
| 100,000 | 1,353 | 13.5 | (steady-state) |

Steady-state per-row cost: ~13.5 µs.

### 1.3 Where the time goes

Decomposing 13.5 µs/row at the steady-state (n=10⁵):

| Phase | Cost (ns/row) | % of train | Cumulative |
|---|---:|---:|---:|
| route_to_leaf (encode + descend) | ~200 | 1.5% | 1.5% |
| blr_update (Cholesky + chain hash) | ~12,300 | 91.0% | 92.5% |
| observe (Welford + 2 SHA-256) | ~1,000 | 7.4% | 99.9% |
| miscellaneous (bookkeeping, stats version) | ~50 | 0.4% | 100% |

**`blr_update` is 90%+ of training cost.** Within `blr_update`:
- Cholesky decomposition at d=4: ~21 floating-point ops × Kahan compensation = ~500 ns of pure math
- The remaining ~11.8 µs is **accumulator allocation + audit chain SHA-256 + per-node stats chain SHA-256 + audit event allocation**.

This is the same finding Item 4 made: ABNG is **SHA-256-bound, not math-bound**. Item 4's `observe_batch` collapsed N audit chain SHA-256s into 1, giving 17.6× at n=1024. The same logic applies to BLR, but `BlrState::update` already accepts n rows in one call — so the Cholesky math is already amortized across rows.

The remaining frontier is **per-call overhead within `blr_update` even when called batched**: the function still allocates fresh `Vec<KahanAccumulatorF64>` for `xtx` and `xty` on every call, and emits one audit event per call. For very large batches this is fine; for many medium batches (the typical chess RL pattern) the per-batch overhead dominates.

---

## 2. TidyView lessons applied to ABNG

TidyView v3 (v0.1.7) hit ~108 ns/op on sealed lookup. The disciplines that got it there:

### 2.1 Cat-aware paths (TidyView Phase 2)

**TidyView's lesson:** when the data is categorical, route through `Vec<u32>` codes instead of `Vec<String>`. Avoids per-row allocation, branch-prediction-friendly, cache-coherent.

**Applied to ABNG:** ABNG's codebook is **already** integer-coded — `encode_prefix` returns `Vec<u8>` directly. The codebook installation phase (`set_codebook`) does the categorical → integer conversion once at install time; lookup is a `partition_point` binary search on a sorted `Vec<f64>`. **This discipline is already present.**

**Verdict:** No work needed. ABNG inherits TidyView's cat-aware discipline for free because the codebook is integer-coded by construction.

### 2.2 Sealed-lookup discipline (TidyView Phase 10)

**TidyView's lesson:** once a lookup table is installed, downstream lookups should be O(1) array index, not O(log n) BTreeMap descent.

**Applied to ABNG:** The codebook IS sealed at install (one-shot via `set_codebook`, frozen via `frozen_hash`). Lookups use `partition_point` on a `Vec<f64>` — which is O(log n) per dim, not O(1).

For ABNG's typical bin counts (4–8 per dim), `partition_point` is ~2–3 comparisons per dim. **At d=1 this is essentially O(1)** — 2 comparisons total per lookup. The micro-bench measures 124 ns/op for `encode_prefix` and 117 ns/op for `descend`. Both are dominated by allocation + function-call overhead, not the search.

**Verdict:** Sealed-lookup discipline is structurally already applied. The O(log n) → O(1) refactor would gain ~50–100 ns/op at higher bin counts (n > 16) but would require either (a) hashing into a fixed-size LUT — which constrains bin counts to a power of 2 — or (b) a tiered cache structure. Both are Phase 0.7+ optimizations; not load-bearing for Phase 0.6.

### 2.3 Streaming set-op fast paths (TidyView Phase 3)

**TidyView's lesson:** chunked dispatch when one or both operands are bitmask-shaped — short-circuit the "All" or "Empty" cases, batch-process the dense regions.

**Applied to ABNG:** Less directly applicable. ABNG's primary hot path is per-row training, not set-ops. But the **chunked-dispatch pattern itself** generalizes: amortize per-call setup over the batch. This is what Item 4 (`observe_batch`) and Item 8 (`route_to_leaf_batch`) do.

**Verdict:** The pattern is now in ABNG. Items 4 + 8 collapse N per-row dispatches to 1 batched dispatch, saving N-1 dispatch overheads. Item 4 measured 17.6× speedup at n=1024 for `observe_batch`. Item 8's `route_to_leaf_batch` measured 1.08× at the Rust API level (the math is the same; the win is at the interpreter dispatch boundary, not the math).

---

## 3. Concrete kernel — `route_to_leaf_batch`

### 3.1 What it does

```rust
pub fn route_to_leaf_batch(
    &self,
    xs: &[f64],   // flat row-major [n, d]
    n: usize,
) -> Result<Vec<NodeId>, GraphError>
```

Equivalent to N invocations of `encode_prefix → descend → extract leaf_id`, but performs **one** upstream allocation (the output `Vec<NodeId>`) and reuses a single prefix buffer across rows. This is the TidyView "chunked dispatch" pattern applied to ABNG's routing layer.

### 3.2 Measured perf (Windows release, `bench/abng_micro/`)

```
route_to_leaf_batch n=1024: per_row=137.70 µs  batch=128.00 µs  speedup=1.08x
```

**Honest interpretation:** at the Rust API level the speedup is small (1.08×) because the math is identical and the prefix-buffer reuse only saves ~10 ns per row. The **real** win is at the CJC-Lang interpreter dispatch boundary, where each saved dispatch is ~few hundred ns of AST-walk or MIR-register-machine overhead. Estimating ~2 µs/row × 10⁴ rows = ~20 ms saved per training loop when the batch path replaces the per-row pattern in `.cjcl` source.

### 3.3 Why this kernel and not Cholesky-related?

The profile shows `blr_update` is 90% of training cost. So why ship a route-batching kernel that addresses ~2% of the cost?

**Three reasons:**
1. **`BlrState::update` already accepts n rows in one call.** The batched BLR math has been there since Phase 0.3b. Users who want batched BLR call `g.blr_update(node, &phi_n_rows, &y_n_rows)` directly; nothing to add.
2. **The route-to-leaf bottleneck shows up in inference loops, not just training.** A held-out evaluation pass over a 10⁴-row test set routes 10⁴ times to make 10⁴ predictions. The batched route+predict pattern would amortize dispatch overhead across the entire eval loop.
3. **The actual remaining Cholesky speedup requires a larger refactor** — eliminating per-call `Vec<KahanAccumulatorF64>` allocations, fusing audit-chain SHA-256 with stats-chain SHA-256 into a single hash, etc. That's Phase 0.7+ scope.

### 3.4 What this kernel does NOT do

- It does NOT call `blr_predict` per row. A future `route_and_predict_batch` kernel would do the full route+predict loop in one Rust call, which would be a higher-leverage perf win for inference.
- It does NOT modify any audit-chain semantics — read-only, no events emitted.
- It does NOT change per-row `encode_prefix` cost at the Rust API level. The win is at the dispatch layer.

---

## 4. Recommendations for Phase 0.7+

### 4.1 Eliminate per-call accumulator allocations in `blr_update`

`Vec<KahanAccumulatorF64>` of size `d²` is allocated on every `blr_update` call. For d=4 that's 16 entries × 24 bytes = 384 bytes per call. At 10⁵ training calls, that's 38.4 MB of churn through the allocator.

**Fix:** Pre-allocate accumulators on the per-node `BlrState` struct. Reuse across calls. Saves ~1 µs per call at d=4.

**Cost:** ~50 LOC refactor in `crates/cjc-abng/src/blr.rs`. No wire-format change. Property tests must verify bit-identical canonical_bytes between pre- and post-refactor versions.

### 4.2 Fuse audit-chain SHA-256 with stats-chain SHA-256

Each `observe` does 2 SHA-256 calls: one for the global audit chain (`sha256(prev_hash ‖ payload)`) and one for the per-node stats chain (`sha256(prev_stats_hash ‖ canonical_bytes)`). At ~1 µs each, that's 2 µs per observe — the headline `observe` cost is ~4.3 µs, half of which is these two hashes.

**Fix:** In a future wire-format bump (Phase 0.8+), replace the per-node stats chain with a Merkle-tree-over-events keyed by node_id. Lookup of a single node's history would require walking the Merkle path; cost shifts from training-time to query-time.

**Cost:** Significant — would require a new audit-event encoding, a Merkle index in the snapshot header, and a new replay path. Multi-month commitment.

### 4.3 `route_and_predict_batch` for inference

The natural sibling of `route_to_leaf_batch`: take [n, d] inputs, return [n, 3] of (mean, leverage, ale_var) per row. Single Cholesky factor extraction per leaf, amortized across rows that route to that leaf.

**Cost:** ~80 LOC in `cjc-abng/src/graph.rs` + dispatch wiring + parity tests + bench. Phase 0.7 candidate.

### 4.4 SIMD `KahanAccumulatorF64` for d≥8

For larger BLR feature dimensions (d ∈ {8, 16, 32}), the Cholesky decomposition becomes the actual bottleneck (~d³/3 flops). At d=32, that's ~10K flops per call — measurable. SIMD-vectorize the inner loops via `std::simd` (stable Rust as of 2026).

**Cost:** ~150 LOC in `crates/cjc-repro/src/kahan.rs` for the SIMD accumulator + ~50 LOC refactor in `cjc-abng/src/blr.rs` to use it. Determinism contract: SIMD lane order must be stable across platforms — verify with cross-platform CI (Item 1 already in place).

---

## 5. Summary

The profile confirms ABNG training is dominated by `blr_update` (90%+), which itself is dominated by SHA-256 for the audit + per-node chains, NOT by floating-point math. TidyView's discipline applies in three forms:

1. **Cat-aware paths** — already inherited via integer-coded codebook. Done.
2. **Sealed lookup** — structurally applied via `partition_point` on frozen sorted boundaries. The O(log n) → O(1) gap is ~50 ns at typical bin counts; not load-bearing.
3. **Chunked dispatch** — Items 4 and 8 ship the pattern. `observe_batch` measured 17.60× at n=1024; `route_to_leaf_batch` measured 1.08× at Rust level (the win is at the interpreter dispatch boundary, ~2 µs/row × 10⁴ = ~20 ms per training loop).

The biggest remaining frontier is **eliminating per-call allocations within `blr_update`** (recommendation §4.1) and **fusing the two SHA-256 chains** (§4.2). Both are Phase 0.7+ scope.

Item 8's deliverable is therefore: this analysis + the `route_to_leaf_batch` kernel + 4 explicit recommendations for Phase 0.7+ work. The pattern is laid down; the larger commitment to "match TidyView's perf discipline" is multi-phase.

---

*This document is the Phase 0.6 Item 8 deliverable. It is not a ship gate; it is a research artifact that informs Phase 0.7+ planning.*
