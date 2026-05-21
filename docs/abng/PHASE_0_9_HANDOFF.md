# CJC-Lang ABNG Phase 0.9 — D-HARHT Integration + Baseline Validation

**Date stamped:** 2026-05-15
**Branch at handoff:** `claude/abng-v14-wire-format` @ `e795cd0` (on `origin`)
**Phase 0.8 status:** complete (all 11 items shipped, demos + visualizations committed, branch on origin)
**Reference doc for D-HARHT:** `C:\Users\adame\Downloads\D-HARHT-Blueprint-and-Code.md` (1700 LOC reference impl + benchmark snapshot)

This handoff scopes Phase 0.9, which is a **two-track phase**:

1. **Track P (Project baseline) — RUN THIS FIRST.** Pick a known-solution problem (medical or physics), run ABNG against it, capture speed + accuracy + determinism contract over 3–5 trials. This becomes the apples-to-apples reference for every later improvement.
2. **Track Q (D-HARHT memory-profile integration).** Use the *memory* lookup profile from D-HARHT (~7.9 MB vs ~14.2 MB at the speed profile, near-HashMap lookup speed) to speed up ABNG's training-time routing **without** breaking determinism, auditability, or the v14 wire format.

Plus four supporting tracks (R/S/T/U) that can land in any order after Track P establishes the baseline.

---

## STACKED ROLE GROUP (CLAUDE.md convention)

You are six roles in one team:

1. **Lead Language Architect** — owns the cjcl-facing semantic surface. Phase 0.9 concern: does the D-HARHT integration require new cjcl builtins (`abng_route_cache_*`, `abng_seal_index`)? My pick: no — D-HARHT stays Rust-internal, opaque to cjcl source. The benefit shows up as faster `abng_descend` / `abng_train_step` calls.
2. **Compiler Pipeline Engineer** — owns Lexer → Parser → AST → HIR → MIR → Exec. Phase 0.9 concern: where does D-HARHT plug in? Answer: behind the `descend` Rust API. Both executors call into the same `cjc-runtime::dispatch_abng`; neither sees the change.
3. **Runtime Systems Engineer** — biggest stakeholder for Phase 0.9. Owns the actual D-HARHT integration: where the new data structures live, when they seal, how they interact with `AdaptiveChildren` / per-node storage.
4. **Numerical Computing Engineer** — secondary stakeholder. D-HARHT doesn't directly touch BLR / Welford math, but Track R (other perf wins) might. Concern: any new SIMD path must preserve the post-D2b/D3 bit-deterministic Kahan contract.
5. **Determinism & Reproducibility Auditor** — gate-keeper. Phase 0.9 cannot ship any change that:
   - Shifts the v14 canary set (28 SHA-256 hashes, 9 already re-locked through Phase 0.8)
   - Introduces platform-dependent code paths (no `#[cfg(target_arch)]` for math)
   - Changes `canonical_bytes` for any persisted state
   - Hides allocations behind safe-looking APIs
6. **QA Automation Engineer** — owns the Track P baseline measurement protocol + the integration-test surface that holds every later commit accountable to it.

Each role's concerns are non-overlapping. Phase 0.9 ships when **all six sign off**.

---

## PRIME DIRECTIVES (carried forward from Phase 0.8c)

Unchanged from `PHASE_0_8c_V14_HANDOFF.md`:

1. **No new `Value` enum variants.**
2. **Both executors must agree** — every Rust-API change is reachable from `cjc-eval` AND `cjc-mir-exec` with byte-identical printed output.
3. **No FMA in math kernels.** SIMD paths use plain f64 arithmetic (the D2a / D2b lesson).
4. **No `HashMap` / `HashSet` with random iteration.** `BTreeMap` / `BTreeSet` only.
5. **Seed-based RNG only** — splitmix64 throughout.
6. **Cross-platform byte equality** — every f64 output must be bit-identical on x86_64 ↔ aarch64 ↔ arm64 for the same input.

**New Phase 0.9 directives:**

7. **D-HARHT integration is opt-in.** Phase 0.9 must not change behavior for callers that don't activate the new code path. The baseline measurements in Track P stay valid after every Track Q commit.
8. **The v14 wire format stays at `ABNG\x0E`.** No bump in this phase. Any D-HARHT-related on-disk addition lives in a new trailer block (allocate trailer tag `0x02`, following the `MERKLE_TRAILER_TAG_V1 = 0x01` precedent).
9. **Canary preservation is mandatory.** The 28 SHA-256 canaries (with 9 re-locked through Phase 0.8) must stay locked through Phase 0.9. Any candidate refactor that would shift them is rejected and re-designed.

---

## TRACK P — Baseline validation (RUN FIRST)

### Problem selection: candidate shortlist

The user asked for "medical or physics, known answer." Three concrete candidates, ranked by fit to ABNG's strengths (per-leaf belief state + uncertainty + explainability via the node graph):

| Candidate | Type | Why it fits ABNG | Why it might not |
|---|---|---|---|
| **Wisconsin Breast Cancer (UCI)** | Medical, binary classification | 569 samples × 30 features → ABNG routes on features, trains BLR + Welford per leaf, reports per-node uncertainty. Published ceiling ~98% accuracy. Standard benchmark; reviewers can sanity-check the number. | Small dataset; ABNG's leaf-specialization may overfit. Easy to game by picking favorable splits. |
| **Pima Indians Diabetes (UCI)** | Medical, binary classification | 768 × 8; even smaller feature space → cleaner per-leaf assignment. Published ceiling ~78% (noisier dataset, real-world demographic data). | Smaller per-leaf sample size if we route to many leaves. |
| **Lotka-Volterra (predator–prey)** | Physics, 2-D dynamical system | Known limit cycle → ABNG should converge to bounded prediction error; per-leaf BLR uncertainty should monotonically decrease over training. Aligns with the existing physics-ml/PINN demo infrastructure. | No discrete "right answer"; harder to score with a single accuracy number. Would need a custom L₂-error metric. |

**Recommendation: Wisconsin Breast Cancer.** Easiest to score (binary classification, accuracy is a single number), strongest published reference set, smallest dataset (fast iteration during Phase 0.9), naturally exercises ABNG's per-leaf specialization. The medical framing also gives a credible "explainability" narrative for the per-node reports (drift, calibration, confidence).

**Alternate: Pima diabetes** if the user wants something noisier / closer to real-world tabular ML.

### Measurement protocol (5 runs)

For each seed `s ∈ {1, 2, 3, 4, 5}`, run a complete training + evaluation sweep:

```
1. Load Wisconsin BC dataset (or chosen alternative)
2. Stratified 80/20 train/test split (sklearn-style; deterministic by seed s)
3. Build ABNG graph:
     - Codebook: per-feature quantile bins (e.g. 4 bins per feature)
     - Leaf head: small MLP (1-hidden layer, tanh)
     - BLR prior: precision=2.0, a=1.0, b=0.5
4. Train: for each train sample, abng_train_step(x, phi, y)
     - phi = MLP(x) or just x for simplicity in v1
     - y = label (0 / 1)
5. Evaluate: for each test sample
     - abng_route_to_leaf(x) → leaf id
     - abng_predict(leaf, phi) → (mean, epistemic_leverage, aleatoric_var)
     - Threshold at 0.5 for binary decision
6. Record:
     - Wall-clock total training time
     - Wall-clock per-row training time (mean + median)
     - Memory: graph.approx_memory_bytes()
     - Accuracy on test set
     - chain_head (32-byte hex)
     - merkle_root (32-byte hex)
     - Audit log size (event count + total bytes)
7. Per-leaf explainability report:
     - For each leaf the test set reaches: leaf id, n_train_samples_routed_here, mean BLR prediction, epistemic leverage, calibration ECE, drift score if a baseline is frozen
```

### Determinism gate

Run 1 with seed `s=1`. Run 2 with seed `s=1`. **chain_head and merkle_root must be byte-identical.** If they're not, the test is bug-flagged before any optimization touches the code.

Run 1 with seed `s=1`. Run 2 with seed `s=2`. **chain_head and merkle_root MUST differ.** Accuracy may differ by a few decimal points across seeds — but should stay within a tight band (defined empirically by the 5-run spread).

### Accuracy gate

Wisconsin BC published ceiling is ~98% with calibrated models, ~95% with simple methods. ABNG should hit **≥ 90%** on test split as a Phase 0.9 baseline. If it can't hit 90%, the model has a structural bug, and that's a precondition to anything in Track Q being meaningful.

### Output: `bench_results/phase_0_9_baseline/`

```
bench_results/phase_0_9_baseline/
  wisconsin_bc_5runs.csv          # one row per (seed, metric)
  wisconsin_bc_summary.md         # human-readable: mean/median/min/max per metric
  wisconsin_bc_per_leaf_seed1.csv # explainability report for seed 1
  wisconsin_bc_chain_heads.txt    # 5 hex chain heads + 5 hex Merkle roots
  wisconsin_bc_accuracy.svg       # box plot of accuracy over 5 seeds
  wisconsin_bc_runtime.svg        # bar chart of wall-clock per seed
```

Every `bench_results/phase_0_9_baseline/*.svg` is byte-stable (matches the Phase 0.8 demo SVG-determinism gate).

### Test scaffolding

```
tests/abng/baseline_wisconsin_bc.rs
  fn baseline_wisconsin_bc_determinism_gate()
    # 5 runs at seed 1; assert all chain_heads byte-equal
  fn baseline_wisconsin_bc_seed_sensitivity()
    # 5 runs at seeds 1..5; assert all chain_heads distinct
  fn baseline_wisconsin_bc_accuracy_floor()
    # any seed; assert accuracy >= 0.90
  fn baseline_wisconsin_bc_per_leaf_explainability()
    # exercise the per-leaf report shape
```

---

## TRACK Q — D-HARHT memory-profile integration

### Reference architecture (from the D-HARHT doc)

D-HARHT Memory profile combines:

```
deterministic splitmix64-style scattering
fixed power-of-two shard jump table (256 shards)
sealed sparse 16-bit front directory
  ├─ singleton entries point directly to shard-local leaf IDs
  ├─ MicroBucket4 for tiny collision groups
  ├─ MicroBucket8 for small collision groups
  └─ MicroBucket16 for medium collision groups
per-shard typed slab allocator
  ├─ u32 NodeId child references
  ├─ u32 tagged node entries
  └─ deterministic monotonic allocation
adaptive radix fallback (Node4/Node16/Node32/Node48/Node256)
full key equality check on every successful lookup
```

Benchmark from the doc (100K keys, 2M lookups, 256 shards):

| Engine | ns/op | Memory bytes |
|---|---:|---:|
| D-HARHT speed | 32.214 | 14,219,312 |
| D-HARHT balanced | 37.630 | 14,268,464 |
| **D-HARHT memory** | **37.220** | **7,948,112** |
| HashMap | 37.470 | — |
| Adaptive radix | 45.082 | 5,949,808 |
| Standard radix | 310.042 | 1,098,907,680 |

**Memory profile is the right pick for ABNG**: near-HashMap lookup speed, ~44% smaller than speed profile, structurally similar to ABNG's existing adaptive radix layout but with SSE2 SIMD on Node16 lookup that ABNG currently lacks.

### Three integration options (in order of risk + reward)

#### Option Q1 — Route memoization side cache (LOW RISK, ADDITIVE)

Add a D-HARHT-backed `RouteCache` that memoizes `(prefix_hash → leaf_id)` lookups during training. The `descend` Rust API checks the cache first; on miss, falls through to the existing radix walk + populates the cache.

**Pros:**
- Zero changes to the existing radix tree or `canonical_bytes`.
- Zero canary impact (the cache is in-memory only, never serialized).
- Easy to A/B benchmark (toggle the cache on / off via a graph field).
- Highest wins on workloads with route-key recurrence (PINN collocation, RL state revisits, chess-RL move repetition).

**Cons:**
- Doesn't speed up first-visit lookups (which is the bulk of training).
- Cache invalidation on `add_node` / `Split` / `Merge` (clear the cache when topology changes).

**Where it plugs in:**

```rust
pub struct AdaptiveBeliefGraph {
    // ... existing fields ...
    route_cache: Option<DHarht<NodeId>>,
}

impl AdaptiveBeliefGraph {
    pub fn enable_route_cache(&mut self) { /* allocate D-HARHT */ }
    pub fn descend(&self, prefix: &[u8]) -> RouteEvidence {
        if let Some(cache) = &self.route_cache {
            let prefix_hash = hash_prefix(prefix);
            if let Some(&leaf_id) = cache.get(prefix_hash) {
                return RouteEvidence { leaf_id, ... };
            }
        }
        // existing walk
    }
}
```

**Determinism note:** the cache cannot affect routing outcome — it must produce the same `leaf_id` the walk would have. Validated by a "cache vs walk parity" property test.

#### Option Q2 — Replace `AdaptiveChildren` enum (MEDIUM RISK, STRUCTURAL)

Swap ABNG's `AdaptiveChildren { None, Node4, Node16, Node48, Node256 }` for D-HARHT's `Node4/Node16/Node32/Node48/Node256` slab-allocated variants. The new Node32 fills a gap; the SSE2 Node16 lookup beats ABNG's current linear scan.

**Pros:**
- Speeds up every `descend` walk, not just cached ones.
- The new Node32 layer reduces memory at intermediate node counts (16 < n ≤ 32 currently jumps to Node48's 256-byte child_index).
- Per-shard typed slabs improve cache locality (the D2a/B4 lessons applied to topology).

**Cons:**
- Touches `canonical_bytes` if the on-disk encoding of `AdaptiveChildren` changes shape (the A4 sparse encoding is per-variant — adding Node32 means a new tag + decode arm).
- Likely requires another canary re-lock cycle (small set: only the demos that hit n_children ∈ [17, 32] would shift).
- Larger code change (~600 LOC) and a longer review surface.

**Wire-format impact:** allocate a new variant `Node32` in the A4 sparse encoding. v14 readers gain a new decode arm; v13 readers see a magic mismatch and reject (no compatibility burden).

#### Option Q3 — Full D-HARHT-backed leaf-key index (HIGH RISK, REARCHITECTURE)

Move the leaf-id ↔ leaf-state mapping entirely behind D-HARHT, so per-leaf state lookups (`g.nodes[leaf].blr.update(...)`) become D-HARHT lookups instead of `Vec`-indexed array accesses.

**Pros:**
- Sealed front-directory hop on every leaf access.
- Naturally supports very large leaf counts (millions) with bounded memory.

**Cons:**
- Major rework of `Node` storage. Touches every accessor in `cjc-abng`.
- Likely shifts `canonical_bytes` across the per-node section. Large canary cost.
- Marginal win for current demo workloads (small leaf counts).

**Verdict: NOT recommended for Phase 0.9.** Mention in handoff for completeness; defer to Phase 0.10+ if a workload emerges that actually needs millions of leaves.

### My Phase 0.9 picks for Track Q

1. **First commit:** Option Q1 (route cache) — small, additive, immediately benchmarkable against the Track P baseline.
2. **Second commit (if Q1's measured win justifies more invasive work):** Option Q2 (Node32 layer + SSE2 Node16 lookup). Lock the canary impact (probably 1–2 demos) and re-lock cleanly.
3. **Q3 deferred entirely.** Document the design space; defer the implementation until a Phase 0.10 workload demands it.

---

## TRACK R — Other speed/memory wins (brainstorm)

Ranked by leverage × likely-canary-safety:

| Item | Speed/Memory | Canary impact | Effort |
|---|---|---|---|
| **R1** Merge C3 (per-thread arena observability) from `claude/elastic-kirch-db47b2` | Modest speed (better concurrent training) | None | Low — cherry-pick |
| **R2** Merge D1 (Cholesky factor caching with rank-1 update) from `claude/elastic-kirch-db47b2` | 2–10× on `predict()` at d≥8 | Possible — interacts with D2b's `BlrState::update` | Medium — needs conflict resolution + canary re-check |
| **R3** Batched `train_step_batch(xs, phis, ys)` — single-call N-row trainer | Amortizes routing + audit-emission overhead across rows | None (call sequence stays equivalent) | Medium — new builtin |
| **R4** PGO + LTO release-build flags in `Cargo.toml` | 10–20% global speedup | None | Trivial |
| **R5** Snapshot streaming decompression (lazy-load audit events) | Memory at large-archive replay time | None | Low (extends B1 mmap) |
| **R6** Per-leaf provenance stamp interning | Memory (dedup shared stamps) | Possible — depends on encoding | Medium |
| **R7** Audit-event seq → event D-HARHT index | O(1) audit-event lookup vs O(log N) BTreeMap | None (in-memory index, not serialized) | Low (mirrors Q1) |
| **R8** Compile-time codebook lookup (const-fold quantile bins) | Marginal speed | None | Trivial — already deterministic |

**Highest-leverage subset for Phase 0.9:** R1 (cherry-pick C3), R2 (cherry-pick D1, requires conflict-resolution work), R4 (PGO+LTO), R3 (batched train_step). Together these probably stack to 1.5–2× macro speedup on training-heavy workloads without any canary risk beyond R2.

---

## TRACK S — Operational cleanup

| Item | Status |
|---|---|
| **S1** Cherry-pick C3 from `claude/elastic-kirch-db47b2` onto v14 branch | pending |
| **S2** Cherry-pick D1 from `claude/elastic-kirch-db47b2` onto v14 branch (resolve D2b conflicts) | pending |
| **S3** Re-run PINN macro bench at HEAD post-D2b/D3 | pending |
| **S4** Update `PHASE_0_8_HANDOFF.md` macro bench section with re-run numbers | pending |
| **S5** Open a PR for `claude/abng-v14-wire-format` on GitHub (currently just an origin branch, not a PR) | optional |

---

## TRACK T — CJC-Lang surface exposure (optional, may slip to Phase 0.10)

Per `V14_MIGRATION.md` "Going forward":
> No CJC-Lang-surface builtin was added for `merkle_root` / `merkle_tree` / `verify_chain_par` / `matvec_plus_xty_kahan`; those are Rust-only APIs for now.

Adding cjcl-side builtins:

* `abng_merkle_root(graph) -> bytes` — 32-byte root for the audit chain.
* `abng_merkle_proof(graph, i) -> [bytes]` — inclusion proof for event `i`.
* `abng_merkle_verify_proof(leaf_hash, i, n_leaves, proof, root) -> bool` — pure-function verifier.
* `abng_verify_chain_par(graph, n_threads) -> bool` — parallel verify.

Each follows the canonical wiring pattern (`cjc-runtime/src/builtins.rs` + `cjc-eval` + `cjc-mir-exec`) plus AST↔MIR parity tests.

**Verdict:** ship in Phase 0.9 IFF the baseline-project demo (Track P) needs them in cjcl source. If it doesn't, defer.

---

## TRACK V — Experimental data structures to replace parts of training

Track V is a **research / experiment track**, not an implementation directive. The point is to surface candidates that could replace components of the training pipeline with new data structures, each candidate gated on the Phase 0.9 constraints (determinism, auditability, no canonical_bytes drift, both executors agree, no canary shift).

Each candidate should pass a feasibility study before any code is written:

1. Specify the existing component being replaced (which Rust type, which use site).
2. Specify the new data structure (rough shape + access pattern + memory footprint).
3. Identify the determinism gates that must hold.
4. Bench-quantify the upside on the Track P baseline.
5. Identify the canary-shift risk (Path A / Path B / zero).

### V1 — BLR posterior representation

| Candidate | What replaces | Why it might help | Risk |
|---|---|---|---|
| **Packed lower-triangular L (in-memory)** | The full d×d `precision: Tensor` in `BlrState` | Halves BLR memory (d=16: 2048 B → 1024 B); cache-friendlier | Path B if `canonical_bytes` stays full-matrix; trivial |
| **Banded / sparse precision matrix** | The same | Memory win when features are approximately independent; sparse Cholesky cheaper | Determinism: zero-tolerance check on the "approximately independent" predicate; risk: features that should-be sparse but aren't trigger silent degradation |
| **Diagonal-plus-low-rank parametrization** | `precision = D + UU^T` | O(d·k + k³) Cholesky vs O(d³); win at d ≥ 16 with low effective rank | Algorithm change → BLR posterior bits differ → full canary re-lock |
| **Cached Cholesky factor (D1 from another branch)** | Recomputing L on every `predict()` | 2–10× on `predict()` at d ≥ 8 | Already shipped on `claude/elastic-kirch-db47b2`; merge work (Track S) |
| **Rank-1 Cholesky updates (Givens rotations)** | Full re-factorization on `update()` | O(d²) per row vs O(d³/3) | Algorithm change → canary risk; D1 partially addresses this |

### V2 — Welford / NodeStats representation

| Candidate | What replaces | Why it might help | Risk |
|---|---|---|---|
| **Columnar `NodeStats` storage** | `Vec<NodeStats>` (one struct per node) → parallel `Vec<u64>` + `Vec<f64>` + `Vec<f64>` | Cache locality for batch operations; SIMD-friendly mean/M2 updates | Path B if `canonical_bytes` per-node stays unchanged; non-trivial refactor |
| **Compressed Welford for low-precision tracking** | `(n: u64, mean: f64, m2: f64)` → `(n: u32, mean: f32, m2: f32)` with promotion on overflow | Halves Welford memory at the leaf | Determinism: hard. Bit-exact promotion logic must be platform-independent. **Probably not worth it.** |
| **Approximate quantile sketch alongside Welford** | (would extend `NodeStats`) | Per-leaf median + quantile bounds for richer drift/calibration | Determinism: KLL or t-digest are deterministic if seed-fixed and merge-order is fixed |

### V3 — Audit chain index variants

| Candidate | What replaces | Why it might help | Risk |
|---|---|---|---|
| **D-HARHT-backed `seq → AuditEvent` index** | Currently O(N) walk for `audit.iter()` filter operations | O(1) random access to specific events | Zero canary (in-memory only); same shape as Q1 |
| **Skip-list audit log** | Linear `Vec<AuditEvent>` | O(log N) random access without building a Merkle tree | Skip-list with deterministic level assignment (seed-fixed promotion test); no canary risk |
| **Bloom filter for "has this leaf been observed recently?"** | Linear scan of `audit.iter().filter(node_id)` | Quick negative-answer for drift checks | Bloom filter parameters must be seed-fixed; no canary risk |

### V4 — Trie / routing structure alternatives (beyond D-HARHT)

| Candidate | What replaces | Why it might help | Risk |
|---|---|---|---|
| **HAT-trie (hash-array trie)** | `AdaptiveChildren` enum | Hybrid hash-bucket + radix; lower memory at sparse prefix populations | Path A if implemented at the `canonical_bytes` level; Path B if at the in-memory level only |
| **DAWG / DAFSA (compressed deterministic automaton)** | Same | Merges suffix-shared sub-trees; massive memory win for prefix-heavy workloads | Insertion is hard; works best for read-mostly workloads. Probably not a fit for ABNG's online training. |
| **Patricia-compressed radix** | Same | Collapses single-child paths (which Phase 0.8 doesn't yet do) | ART convention; standard implementation; possible canary impact |

### V5 — Per-leaf state co-location

| Candidate | What replaces | Why it might help | Risk |
|---|---|---|---|
| **AoS → SoA for per-leaf BLR fields** | `Vec<Option<BlrState>>` → parallel `Vec<f64>` columns for mean, precision, a, b | Cache-friendly when iterating "all leaves' BLR posterior" | Path B if disk encoding stays per-leaf; minor refactor |
| **Bit-packed leaf-presence indicator** | `Vec<Option<BlrState>>` | Half a byte per leaf for "has BLR" check | Trivial |
| **Versioned leaf state via copy-on-write** | Direct mutation in place | Snapshot a leaf's BLR cheaply for "compare against historical" queries | New API; no canary risk |

### V6 — Routing key encoding alternatives

| Candidate | What replaces | Why it might help | Risk |
|---|---|---|---|
| **Streaming quantile sketch for codebook** | `set_codebook(boundaries)` with pre-computed quantiles | Online refinement of bin boundaries as more data arrives | Determinism: t-digest is deterministic with seed; codebook re-fitting is structural |
| **Lookup-table accelerated `encode_prefix`** | Per-feature binary search | O(1) bin lookup at the cost of a 256-entry table per feature | Tiny memory; no determinism risk |
| **MinHash-style prefix sketch** | Full prefix bytes | Compress long prefixes into a fixed-length signature for fast routing | Probabilistic; not a fit for ABNG's deterministic contract |

### V-level success criteria

Each Track V candidate must produce, before any merge:

1. **A design note** in `docs/abng/PHASE_0_9_V_<candidate>.md` covering the five feasibility questions above.
2. **A microbench** showing the candidate's measured upside on a representative ABNG operation.
3. **A property test** asserting the candidate produces byte-identical output to the existing component (when both are bit-compatible) or a tolerance-bounded result (when arithmetic differs).
4. **A canary-impact prediction** with at least one demo trial-run to validate the prediction.

Without these four artifacts, no V-track candidate ships.

---

## TRACK W — Numerical kernel research (advanced Kahan, deterministic BLAS, alternatives)

Track W is a **research-only track** for now: identify candidate numerical kernel improvements, document their applicability to ABNG, and surface implementation costs. Implementation itself is gated on the Track P baseline showing the relevant inner loop as a bottleneck.

### W1 — Advanced compensated-summation variants

| Algorithm | Error growth | Cost vs Kahan | Determinism? | Notes |
|---|---|---|---|---|
| **Plain summation** | O(n·ε) | 1× | Yes (with fixed order) | Current fallback |
| **Kahan summation** | O(ε) | 4× | Yes | Currently used everywhere in cjc-abng |
| **Neumaier (Kahan-Babuška)** | O(ε) | 5× | Yes | Handles the case `|new| > |sum|` correctly, which Kahan loses; **likely worth adopting** for the BLR rhs `+ xty[a]` step where xty can dwarf running sum |
| **Klein (second-order Kahan)** | O(log² n · ε²) | 8× | Yes | Tracks compensation-of-compensation. Probably overkill for d ≤ 32; worth investigating at d ≥ 64. |
| **Pairwise summation** | O(log n · ε) | 1× plus log n overhead | Yes (with fixed tree structure) | Requires N inputs up-front; not streaming. Could replace `xty[a].finalize()` if all rows are pre-batched. |
| **Binned summation** (current `cjc-repro::BinnedAccumulatorF64`) | O(ε) regardless of magnitude spread | 6–8× | Yes | Already in use for binned reductions; could be promoted into BLR. |

**Recommendation:** investigate Neumaier as a Phase 0.9 W1 micro-experiment. The BLR rhs computation in `matvec_plus_xty_kahan` adds the matvec sum to `xty[a]`, which can be larger than the running matvec sum at small d — exactly Kahan's known failure mode. Neumaier is one extra `if .abs() > sum.abs()` branch per add.

### W2 — Deterministic BLAS / GEMM kernels

| Approach | Determinism | Effort | Notes |
|---|---|---|---|
| **ReproBLAS** (Demmel et al.) | Yes — uses binned summation | High (external dep, FFI) | Likely overkill; would require new dependency. Research mature. |
| **Hand-rolled deterministic GEMM at d ≤ 16** | Yes — explicit accumulation order | Low (~200 LOC) | Best ROI for ABNG; current `matmul` in `cjc-runtime` is the natural integration point |
| **CADNA** (stochastic-arithmetic audit) | Audit tool, not BLAS | Medium | Useful for *verifying* numerical reproducibility, not for computing it; tangential to ABNG |
| **OpenBLAS / MKL with `OMP_NUM_THREADS=1`** | Deterministic single-threaded; reordering-sensitive multi-threaded | None (already excluded; cjc-runtime is pure Rust) | Not applicable: CJC-Lang is zero-external-deps |

**Recommendation:** for ABNG's d ≤ 16 dominant case, hand-rolled deterministic GEMM with explicit `(i, j, k)` order pinning is the right play. Probably aligns with extending the D3 fused matvec to a full mat-mat product when D2b's SIMD path generalizes.

### W3 — SIMD-without-nightly improvements

Current strategy uses `[f64; N]` plain arrays for auto-vectorization. Honest assessment: it works (D2a was measured at 3–4× on isolated primitives), but is compiler-dependent.

| Approach | Determinism | Notes |
|---|---|---|
| **Plain `[f64; N]` auto-vec** | Yes | Current strategy. Win depends on compiler. |
| **`std::arch::*` intrinsics with feature detection** | Yes (if feature detection is build-time) | D-HARHT uses this for `find_key16_sse2`. Tractable for hotspots. |
| **`std::simd` (portable_simd)** | Yes | Still nightly as of 2026-05. **Skip.** |
| **`packed_simd` crate** | Yes | Unmaintained. **Skip.** |
| **`wide` crate** | Yes (no FMA when configured) | Stable, zero unsafe. Possible drop-in for portable SIMD; adds a dependency. |

**Recommendation:** stay with the current `[f64; N]` auto-vec strategy unless a specific hotspot wants explicit intrinsics (then follow D-HARHT's SSE2 pattern). Defer `std::simd` until it stabilizes.

### W4 — Compilation flags

| Flag | Speed impact | Notes |
|---|---:|---|
| `[profile.release] lto = true` | 5–15% | Currently set? Check `Cargo.toml`. |
| `[profile.release] codegen-units = 1` | 5–10% (paired with LTO) | Single-unit codegen enables more inlining |
| `[profile.release] opt-level = 3` | (default already) | No change |
| `RUSTFLAGS="-C target-cpu=native"` | 5–25% on hot loops | NOT deterministic across machines unless pinned; only use in benches, not release builds |
| **PGO (profile-guided optimization)** | 10–30% | Two-pass build; requires a representative profiling workload. Track P baseline is the natural PGO profile generator. |
| **BOLT (post-link binary optimization)** | 5–15% | Newer; depends on Linux. Probably skip until cross-platform CI is ready. |

**Recommendation:** verify LTO is on; add `codegen-units = 1` if not; investigate PGO using Track P as the profile workload (high leverage since Track P is the canonical end-to-end run).

### W5 — Allocator alternatives

| Allocator | Speed | Cross-platform | Notes |
|---|---|---|---|
| **System** (default) | Baseline | Yes | Current |
| **MiMalloc** | 5–15% | Yes | Microsoft-maintained; small footprint; deterministic allocation order with seeded RNG |
| **jemalloc** | 5–10% | Yes (Unix-leaning) | Mature; Rust-friendly via `tikv-jemallocator` crate |
| **Per-arena allocators** (already C3 on another branch) | 5–10% in concurrent training | Yes | In-tree; no external dep; Phase 0.9 Track S1 |

**Recommendation:** C3 (Track S1) is the lowest-risk win; defer process-wide allocator swap until the C3 lessons are absorbed.

### W6 — Algorithmic / structural ideas (deferred)

- **Mixed-precision training** (f32 forward + f64 reductions): breaks bit-determinism unless carefully gated. **Defer past Phase 0.9.**
- **Quantization-aware leaf MLP weights**: int8 weights with scale factor. Cross-platform determinism is fragile. **Defer.**
- **Coarse-grained checkpointing**: persist every K rows; replay reconstructs the gap. B2 streaming partially addresses this; can extend further. **Track R5 candidate.**
- **Lazy Cholesky** (only factor when `predict()` is called): defer the O(d³) work until needed. **Should be folded into D1's caching, not a separate item.**

### W-level success criteria

For each Track W candidate to advance from "research" to "implementation":

1. **A bench harness** showing the candidate's measured speedup or memory reduction on a *Track P baseline scenario*, not a synthetic benchmark.
2. **A determinism proof**: either bit-equality to the prior path (preferred) or a tolerance bound + audit.
3. **A canary impact prediction** with confirming demo trial.
4. **Plus the V-level criteria** if the candidate also involves a data-structure change.

---

## TRACK U — Visualizations for the baseline + post-Q1 numbers

After Track P and Q1 land, generate two additional SVGs to extend the demo set:

1. `bench_results/phase_0_9_demos/q1_route_cache_speedup.svg` — bar chart of wall-clock per-row training time at cache off vs cache on, on the Wisconsin BC baseline.
2. `bench_results/phase_0_9_demos/q1_explainability_per_leaf.svg` — per-leaf prediction + uncertainty + drift score for a few sample test cases (radar chart or stacked bar).

Determinism gate: re-render twice, assert byte-equal. Same protocol as the Phase 0.8 demos.

---

## SEQUENCING

```
Phase 0.9 recommended order:

  1. Track P (baseline) ← BLOCKING. All later tracks reference this.
     └── tests/abng/baseline_wisconsin_bc.rs + bench_results/phase_0_9_baseline/
     └── COMMIT 1: baseline harness + 5-run results

  2. Track S1+S2 (cherry-pick C3+D1)
     └── COMMIT 2 (or 2 + 3): merged perf items from elastic-kirch-db47b2
     └── re-run baseline; numbers update in bench_results/

  3. Track R4 (PGO + LTO)
     └── COMMIT 4: Cargo.toml release profile bump
     └── re-run baseline; numbers update

  4. Track W1 (Neumaier summation experiment) — RESEARCH SPIKE
     └── microbench against Kahan in the BLR rhs `+ xty` add
     └── if winning: COMMIT 5 (Track V/W bridge: replaces Kahan in matvec_plus_xty_kahan)
     └── if not: design note only

  5. Track Q1 (route memoization) — first D-HARHT integration
     └── COMMIT 6: cjc-abng::route_cache module (Option Q1 from Track Q)
     └── COMMIT 7: integration tests + property tests
     └── re-run baseline; SVG diff

  6. (Optional, gated on Q1 winning) Track Q2 (Node32 + SSE2 Node16)
     └── COMMIT 8-10: layout change + canary re-lock
     └── re-run baseline; SVG diff

  7. (Optional) Track V experiments — RESEARCH SPIKES
     └── V1 packed in-memory Cholesky, V2 columnar NodeStats, V3 audit index, etc.
     └── Each spike: design note + microbench + property test BEFORE any merge
     └── Implementation gated on baseline measuring the relevant inner loop as a bottleneck

  8. Track U (Phase 0.9 demo SVGs)
     └── COMMIT 11+: post-baseline visualizations (cache speedup, per-leaf explainability)

  9. Update docs/abng/CAPABILITIES.md with Phase 0.9 entries
     └── COMMIT N: docs update
```

Total estimated commits: 8–15 depending on V/W spike inclusion. Total estimated LOC: 2000–5000.

**Strict ordering constraints:**

* Track P MUST land before any other track's first commit. Every later commit re-runs the baseline.
* Track V and W spikes are gated on Track P showing the targeted inner loop as a measured bottleneck. **No speculative numerical refactors.**
* Track Q2 is gated on Q1's measured win. If Q1 doesn't win on the Track P baseline, Q2 doesn't ship.

---

## OPEN QUESTIONS (FOR THE NEXT SESSION TO RESOLVE)

1. **Which baseline project?** Wisconsin BC (my pick), Pima diabetes, or Lotka-Volterra?
2. **Dataset source.** Bundle the dataset as a `tests/data/` CSV, or fetch deterministically from a URL? My pick: bundle (no network at test time, fully deterministic).
3. **Per-feature codebook bin count.** 4 bins per feature gives 4^30 ≈ 10^18 possible routes for Wisconsin BC's 30 features — way too sparse. Realistically we'd need to either reduce feature count (PCA / SelectKBest preprocessing) or use very few bins per feature. My pick: pick top-10 features by univariate ANOVA F-score, 4 bins each = 4^10 ≈ 1M routes. Document the preprocessing as part of the baseline.
4. **MLP architecture for `phi`.** Skip the MLP entirely in v1 and use raw features as `phi`? Or wire a small 10→8→8→4 MLP? My pick: raw features (simpler baseline; MLP is a future Phase 0.10 extension).
5. **Q1 cache invalidation policy.** Clear on every `add_node`? Clear on every `Grow/Split/Merge/Prune`? Clear lazily (per-node version stamp)? My pick: clear on topology change events (simple, conservative).
6. **Should the Phase 0.9 work happen on `claude/abng-v14-wire-format` or a fresh branch?** Argument for same branch: continuity, smaller PR. Argument for fresh branch: clean separation between v14 wire-format work and Phase 0.9 perf/baseline work. My pick: **fresh branch `claude/abng-phase-0-9`**, with `claude/abng-v14-wire-format` as the merge-base.

---

## RECOMMENDED NEXT-SESSION PROMPT

Paste this into a fresh Claude Code session checked out to the chosen Phase 0.9 branch:

```
# CJC-Lang ABNG Phase 0.9 — D-HARHT Integration + Baseline Validation

## ROLE

You are a stacked systems team picking up Phase 0.9 of CJC-Lang.
The previous session shipped Phase 0.8 (all 11 items) on
`claude/abng-v14-wire-format` and pushed it to origin. Phase 0.9
is two-track: (P) baseline a known-solution project to establish
speed + accuracy + determinism reference; (Q) integrate the
D-HARHT memory profile to speed up training without breaking
determinism, auditability, or the v14 wire format.

Read `docs/abng/PHASE_0_9_HANDOFF.md` end to end before starting.
The D-HARHT reference impl + benchmark snapshot lives at
`C:\Users\adame\Downloads\D-HARHT-Blueprint-and-Code.md` (1700
LOC). The Phase 0.8 close-out is at
`docs/abng/PHASE_0_8_HANDOFF.md` and `V14_MIGRATION.md`.

## PRIME DIRECTIVES

Carry forward Phase 0.8c directives (no Value variants, both
executors must agree, no FMA, BTreeMap not HashMap, seed-based
RNG, cross-platform byte equality), plus:

* D-HARHT integration must be opt-in — baseline measurements
  stay valid for callers that don't activate it.
* v14 wire format stays at `ABNG\x0E`. No bump.
* 28 SHA-256 canaries (9 already re-locked) stay locked.

## SCOPE — START WITH TRACK P

1. Build the baseline harness (Wisconsin Breast Cancer
   recommended; alternates Pima diabetes, Lotka-Volterra).
2. Run 5 trials at seed=1 to validate determinism (all 5
   chain_heads byte-equal).
3. Run 5 trials at seeds 1..5 to capture seed-spread variance.
4. Emit `bench_results/phase_0_9_baseline/` with CSV + summary
   + SVGs.
5. Open question: confirm with the user which baseline project
   to use BEFORE writing the harness.

## DO NOT

* Do NOT start Track Q (D-HARHT integration) until Track P's
  baseline is locked in.
* Do NOT push commits to origin without explicit user approval.
* Do NOT touch the v14 wire format.
* Do NOT skip the 5-run determinism gate.

## START

1. Read `docs/abng/PHASE_0_9_HANDOFF.md` end to end.
2. Confirm baseline project choice with the user (Wisconsin BC
   is my recommendation; explain the alternates).
3. Ask any open questions surfaced in the handoff (preprocessing,
   bin counts, MLP architecture, branch strategy) before writing
   code.
4. Propose the test scaffolding before implementing it. Wait
   for sign-off on the scaffolding shape before filling in the
   harness body.
```

---

## DELIVERY CHECKLIST FOR PHASE 0.9

**Status (post-close-out, 2026-05-16):** 3 of 8 required items
✅ shipped + 2 of 8 partially shipped via the docs commit (E)
that's landing alongside this annotation + 3 of 8 ❌ deliberately
deferred to Phase 0.10. See [`PHASE_0_9_STATUS.md`](PHASE_0_9_STATUS.md)
for the rationale and Phase 0.10+ candidate list.

**Required:**

- [x] ✅ `tests/abng/baseline_wisconsin_bc.rs` green at 5-run determinism + accuracy floor. — **31 tests, all green; 4 distinct determinism gates (synthetic chain head, real-BC chain head, calibration bits, route-util stats).**
- [x] ✅ `bench_results/phase_0_9_baseline/` populated with CSV + summary + SVGs. — **9 artifacts shipped, byte-stable across re-renders (the 8 deterministic ones).**
- [ ] ⏭ C3 + D1 cherry-picked from `claude/elastic-kirch-db47b2` (Track S1+S2) **or** explicit decision-note explaining deferral. — **Deferred to Phase 0.10.** Reason: user-driven scope expansion (real-data + ensemble + calibration + interpretability) absorbed the available session budget. C3/D1 perf wins benefit from a dedicated Phase 0.10 perf-track with a baseline benchmark to measure against.
- [ ] ⏭ Track Q1 (route cache) implemented + measured against baseline. **At least one D-HARHT integration shipped.** — **Deferred to Phase 0.10.** Same reason; D-HARHT integration is its own substantial body of work and warrants a dedicated phase.
- [ ] ⏭ PGO + LTO Cargo profile (R4) verified or shipped. — **Deferred to Phase 0.10.** Lowest-cost item but requires dedicated bench measurement to validate impact; included in the deferred perf-track.
- [x] ✅ Phase 0.9 demo SVGs (Track U) committed and byte-stable. — **4 SVGs shipped + 8 PNGs (LinkedIn 1200×675 + Instagram 1080×1080) + LinkedIn/Instagram post text for sharing.**
- [x] ✅ `docs/abng/CAPABILITIES.md` updated with Phase 0.9 entries. — **Shipped in Commit E; 8 Phase 0.9 capabilities documented (P-Ensemble, P-Calib, P-Routes, P-PerLeaf, P-Data, P-Harness, P-Strat, P-Bundle).**
- [x] ✅ Phase 0.9 handoff doc (this file) updated with final numbers + lessons learned. — **This annotation block is the "updated with final numbers"; [`PHASE_0_9_STATUS.md`](PHASE_0_9_STATUS.md) is the "updated with lessons learned" (§10).**

**Optional (gated on Track P baseline showing the relevant bottleneck):**

- [ ] ⏭ Track Q2 (Node32 + SSE2 Node16) implemented + canary re-lock cycle completed. — **Deferred.** Per the original handoff, Q2 is gated on Q1's measured win; since Q1 didn't ship, Q2 is N/A.
- [ ] ⏭ At least one Track V experiment shipped — **Deferred to Phase 0.10+.** Research-only by original handoff.
- [ ] ⏭ At least one Track W experiment shipped — **Deferred to Phase 0.10+.** Research-only by original handoff.
- [ ] ⏭ cjcl-surface builtins for v14 Rust-only APIs (Track T). — **Deferred to Phase 0.10.** No current Phase 0.9 consumer needed `abng_merkle_root` / `abng_verify_chain_par` from cjcl source; will land when a cjcl demo wants them.
- [ ] ⏭ Open PR for `claude/abng-v14-wire-format` on GitHub (Track S5). — **Deferred.** Branch is local-only per "no origin pushes without explicit approval" directive.

**Research-only (may produce design notes without code):**

- [ ] ⏭ Track V design notes for any structure candidate that didn't ship.
- [ ] ⏭ Track W design notes for any kernel research that didn't ship.

---

## Phase 0.9 final empirical numbers

### Real Wisconsin Breast Cancer (UCI WDBC, 569 × 30)

15-seed sweep on stratified 80/20 train/test splits, leaf+root ensemble:

| Metric | Value | Floor | Margin |
|---|---:|---:|---:|
| **15-seed mean accuracy** | **0.9519** | 0.95 | +0.0019 ✅ |
| Min seed accuracy | 0.9217 | 0.85 | +0.0717 ✅ |
| Max seed accuracy | 0.9913 | – | – |
| Single-seed (seed=1) accuracy | 0.9391 | 0.90 | +0.0391 ✅ |
| Brier score (15-seed mean) | 0.0753 | 0.18 | well under ✅ |
| NLL (15-seed mean) | 0.3603 | 0.55 | well under ✅ |
| ECE 10-bin (15-seed mean) | 0.1268 | 0.25 | well under ✅ |

### Synthetic (+1.8σ on 10 of 30 features, seed=1)

| Metric | Value | Floor |
|---|---:|---:|
| Accuracy | 0.9739 | 0.95 ✅ |
| Brier score | 0.0163 | 0.05 ✅ |
| NLL | 0.0859 | 0.30 ✅ |
| ECE 10-bin | 0.0760 | 0.10 ✅ |

### Route utilization (real BC, seed=1, 2⁴ binary tree)

* 16 total leaves, **12 populated, 4 dead**
* Per-populated-leaf samples: min=1, max=**227 (50% of train)**, mean=37.83, std=66.99

### Audit chain

* 1,019 audit events per training trial (real BC, seed=1)
* 2 events per training row (TrainStep + BlrUpdated for root ensemble)
* All 5-run determinism gates pass byte-equal

---

## Phase 0.9 lessons learned

1. **Measurement-driven design discoveries beat scope-fidelity.** The leaf+root ensemble (the architectural insight worth a blog post) emerged from the user's mid-phase "push real BC to 0.95+" prompt, not from the original handoff. Phase 0.9's most valuable output wasn't anticipated by Phase 0.9's plan.

2. **Calibration is what makes accuracy trustworthy.** Adding Brier/NLL/ECE alongside accuracy made the baseline significantly more credible. Future ABNG baselines should default to reporting all three.

3. **Per-route interpretability is qualitatively different from per-prediction interpretability.** The "4 dead routes / one leaf holds 50% of train data" finding is the most insightful visualization the project has produced. Future Phase 0.10+ work should add organic Grow/Split + cross-seed topology similarity to make this richer.

4. **Bundled real-data baselines outperform synthetic-only baselines for blog narratives.** Real Wisconsin BC + the SHA-256 provenance pin gives the project a defensible "this works on real data" story that synthetic alone never could.

5. **Architecture insights are worth documenting as soon as they're discovered.** [`PHASE_0_9_ARCHITECTURE_INSIGHTS.md`](PHASE_0_9_ARCHITECTURE_INSIGHTS.md) was written while the insights were fresh and is now the single source of truth for the route/predictor separation + leaf+root ensemble. Without it, those insights would live only in commit messages.

6. **The audit chain handles architectural changes cleanly.** Adding the leaf+root ensemble doubled the per-row event count (TrainStep + BlrUpdated). Zero canary breakage. The determinism contract scales with feature additions; the wire-format scaffolding from Phase 0.8c was the right foundation.

---

*This handoff sits alongside `PHASE_0_8_HANDOFF.md` (the parent phase), `PHASE_0_8c_V14_HANDOFF.md` (the v14 wire-format follow-up), `PHASE_0_9_STATUS.md` (this phase's close-out status), `PHASE_0_9_ARCHITECTURE_INSIGHTS.md` (the two load-bearing design decisions), and `PHASE_0_9_BLOG_VERIFICATION.md` (zero-regression verification of every blog-mentioned feature). Future Phase 0.10 work should start a fresh handoff doc.*
