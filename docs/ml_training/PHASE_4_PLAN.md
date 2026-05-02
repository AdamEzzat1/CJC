# Phase 4 Plan — Fused Deterministic Kernels

**Status:** Planning doc. No implementation yet.
**Author:** drafted 2026-05-02 to scope Phase 4 across multiple future sessions.

---

## Why this is a planning doc, not a single PR

Phase 4 is the **largest remaining bucket** in the deterministic ML training stack brief — and the only one that genuinely requires multiple weeks of focused engineering across distinct workstreams. Unlike Phases 1, 2a/2b-ml, 3a/3b/3d/3e Tier 1/Tier 2 (each a single-PR scope), Phase 4 has **independent sub-workstreams** with their own design risk, benchmark requirements, and regression surface.

The brief's Phase 4 list:

> - fused linear + bias + activation backward
> - fused softmax + cross entropy
> - layernorm backward
> - attention backward
> - deterministic tiled/parallel matmul policy
> - fused elementwise ops such as `a * b + c` where bit behavior is specified
> - axis-aware reductions (`cross_entropy(logits[B,C], targets[B,C])`, `layer_norm(x, axis=-1)`)
> - fast-path classification (Green / Yellow / Red policy)
> - benchmark requirements ("no claims of speedup are made without benchmark evidence")

Splitting this into a single PR would either (a) exceed safe single-session scope by 5-10×, or (b) require sacrificing the benchmark gate the brief explicitly mandates. Neither is acceptable.

---

## Recommended PR breakdown

Six focused PRs, each ~2-3 days of work, each with its own benchmark gate:

### PR-A — Axis-aware `cross_entropy` and `layer_norm` (highest leverage)

**Why first.** The Phase 3 demo (PR #6) explicitly identified this as the gap: `grad_graph_cross_entropy` and `grad_graph_layer_norm` currently treat their input as a single flat distribution. Batched `[B, C]` losses must be expressed as Python-style per-sample loops (using `gather`). Fixing this **unlocks every demo and example to run without per-sample loops**.

Concrete deliverables:

- New ops: `GradOp::CrossEntropyAxis { logits, targets, axis }`, `GradOp::LayerNormAxis { input, axis }`. (Or new variants of existing ops with `axis: Option<usize>`.)
- Public methods: `GradGraph::cross_entropy_axis(logits, targets, axis)`, `GradGraph::layer_norm_axis(a, axis)`.
- Dispatch arms: `grad_graph_cross_entropy_axis`, `grad_graph_layer_norm_axis`.
- Backward arms (most complex part — per-axis reductions need careful tile/accumulate pattern).
- Tests: parity with the existing flat versions when `axis` covers all elements; analytic correctness on per-row reductions; cross-check vs hand-written batched expression.
- Benchmark: vs existing per-sample `gather` loop pattern from PR #6's MLP classifier demo. Speedup expected; *measure don't claim*.

Estimated: **2-3 days, ~1500 LOC**.

### PR-B — Fused softmax + cross_entropy backward

**Why.** Numerically stable backward for the standard classification loss. The fused gradient formula is `softmax(logits) - targets` — much simpler than the chain `d(loss)/d(softmax) * d(softmax)/d(logits)`. Improves numerical stability and halves graph node count for transformer training.

- Forward kept as the existing `softmax` + `cross_entropy` chain (or a new fused `softmax_cross_entropy(logits, targets)` op).
- Backward bypasses softmax's full Jacobian — single subtraction.
- Tests: bit-equal output with the existing chain on representative inputs; faster gradient by node-count metric.
- Benchmark: vs unfused chain on transformer-shaped batches.

Estimated: **2 days, ~600 LOC**.

### PR-C — LayerNorm backward + fused linear+bias+activation backward

**Why.** Two of the most common per-layer operations in transformers. LayerNorm's backward formula is non-trivial (involves mean/variance gradients) — currently relies on the cjc-ad's existing `LayerNorm` backward which already handles this; this PR's job is making it **axis-aware** (depends on PR-A) and ensuring fused MLP-layer backward stays optimal after Phase 2b's typed-ID migration lands.

- Verify LayerNorm's existing backward arm is bit-correct after axis-aware refactor.
- Fused linear+bias+activation backward exists today via `MlpLayer` — audit whether the gradient path can avoid intermediate tensor allocations (the v2.5 fused op already does this, but verify).
- Add `grad_of` arms for `LayerNorm` (Tier 3 expansion) and `MlpLayer` once the axis-aware version stabilizes.

Estimated: **2-3 days, ~1000 LOC**.

### PR-D — Attention backward + fused attention forward

**Why.** Transformer training's hot path. The attention backward formula has known efficient fused forms (FlashAttention-style); shipping one as a `GradOp::FusedAttention { q, k, v, mask }` halves transformer-training cost vs the unfused chain.

- New op: `FusedAttention { q, k, v, mask, scale }` with own forward + backward.
- Tests: bit-equal output with hand-composed `softmax(q@k^T / √d) @ v` chain.
- Benchmark: forward + backward latency vs unfused chain on standard attention shapes.

Estimated: **3-4 days, ~1200 LOC** (fused backward is mathematically dense).

### PR-E — Deterministic tiled / parallel matmul policy

**Why.** Matmul is the brief's flagship determinism concern. The current `Tensor::matmul_unchecked` is single-threaded sequential. A tiled + multi-threaded version must produce **bit-identical** results regardless of thread count — which means strict tile-major iteration, no work-stealing, no FMA, fixed accumulation order per tile.

- Design a tile shape (e.g., 64×64 or cache-line-aligned) and a deterministic dispatcher that always emits the same per-tile work order regardless of thread availability.
- Pin SIMD usage to non-FMA paths (existing convention).
- Tests: bit-equal output across `RAYON_NUM_THREADS=1, 2, 4, 8, 16`.
- Benchmark: speedup vs single-threaded baseline on various shapes; ensure scaling is sub-linear (deterministic constraint costs throughput).

Estimated: **3-4 days, ~1500 LOC + benchmark suite**.

### PR-F — Fused elementwise ops + fast-path classification

**Why.** Wraps up Phase 4 with a small set of fused micro-ops that the brief calls out: `a * b + c`, `a * (1 + b)`, etc. Plus formalizes the **Green / Yellow / Red fast-path policy**: which ops are bit-deterministic (Green), which need explicit verification (Yellow), which are forbidden (Red). This becomes ADR-0024.

- New ops: `GradOp::Fma { a, b, c }` (a*b + c, *without* hardware FMA — sequential mul-then-add to keep bit-determinism); `GradOp::AffineActivation { a, scale, shift, activation }`.
- ADR-0024 documents:
  - **Green**: elementwise SIMD, fixed-index parallel writes, IndexVec, SealedU64Map, typed tensors.
  - **Yellow**: tiled matmul, parallel RNG, byte-key DHarht, non-cryptographic hashes.
  - **Red unless proved**: hardware FMA, reassociated reductions, unseeded thread scheduling, platform libm differences.

Estimated: **2 days, ~500 LOC + ADR-0024**.

---

## Total estimate

**14-18 days of focused engineering** across 6 PRs. Realistic calendar: 3-4 weeks if PRs land sequentially with review gaps; 2 weeks if parallel review possible.

## Critical-path observations

1. **PR-A unblocks demos.** Should land first; everything else can follow in any order.
2. **PR-D depends on PR-B** (fused softmax+CE backward). Attention's softmax backward IS the fused softmax+CE in matrix form.
3. **PR-E (tiled matmul) is independent** of all other Phase 4 work. Could land in parallel.
4. **PR-F (fused elementwise + ADR)** is the natural wrap-up.

## Benchmark requirement

Per the brief: *"No claims of speedup are made without benchmark evidence."*

Each PR-A through PR-F must include a benchmark comparison (before vs after for the new path) with bit-identical output verified. The benchmark suite location is TBD — likely a new `bench/phase4/` directory.

## Out-of-scope for Phase 4

- Phase 5 (tensor pooling integration) — separate workstream.
- Phase 6 (training manifest + repro-ml) — separate workstream.
- Backward through Reshape/CatOp/GatherOp (deferred to Phase 3e Tier 4).

---

## Recommended next session

Pick **PR-A** as the immediate next implementation slice when Phase 4 work begins. It's the highest-leverage piece (unblocks all batched-loss demos) and is the most-isolated single PR (touches `cjc-ad/src/lib.rs` only, doesn't depend on Phase 4's other workstreams).

Phase 4 PRs are ideally sequenced: A → B → C → D, with E in parallel from the start, F at the end.
