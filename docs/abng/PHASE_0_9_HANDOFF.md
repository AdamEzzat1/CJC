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

## TRACK U — Visualizations for the baseline + post-Q1 numbers

After Track P and Q1 land, generate two additional SVGs to extend the demo set:

1. `bench_results/phase_0_9_demos/q1_route_cache_speedup.svg` — bar chart of wall-clock per-row training time at cache off vs cache on, on the Wisconsin BC baseline.
2. `bench_results/phase_0_9_demos/q1_explainability_per_leaf.svg` — per-leaf prediction + uncertainty + drift score for a few sample test cases (radar chart or stacked bar).

Determinism gate: re-render twice, assert byte-equal. Same protocol as the Phase 0.8 demos.

---

## SEQUENCING

```
Phase 0.9 recommended order:

  1. Track P (baseline)
     └── tests/abng/baseline_wisconsin_bc.rs + bench_results/phase_0_9_baseline/
     └── COMMIT 1: baseline harness + 5-run results

  2. Track S1+S2 (cherry-pick C3+D1)
     └── COMMIT 2 (or 2 + 3): merged perf items from elastic-kirch-db47b2
     └── re-run baseline; numbers update in bench_results/

  3. Track R4 (PGO + LTO)
     └── COMMIT 4: Cargo.toml release profile bump
     └── re-run baseline; numbers update

  4. Track Q1 (route memoization)
     └── COMMIT 5: cjc-abng::route_cache module
     └── COMMIT 6: integration tests + property tests
     └── re-run baseline; SVG diff

  5. (Optional, gated on Q1 winning) Track Q2 (Node32 + SSE2 Node16)
     └── COMMIT 7-9: layout change + canary re-lock
     └── re-run baseline; SVG diff

  6. Track U (Phase 0.9 demo SVGs)
     └── COMMIT 10: post-baseline visualizations

  7. Update docs/abng/CAPABILITIES.md with Phase 0.9 entries
     └── COMMIT 11: docs update
```

Total estimated commits: 6–11 depending on Q2 inclusion. Total estimated LOC: 1500–3000.

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

When all of these are checked, Phase 0.9 is done:

- [ ] `tests/abng/baseline_wisconsin_bc.rs` (or chosen alternate) green at 5-run determinism + accuracy floor.
- [ ] `bench_results/phase_0_9_baseline/` populated with CSV + summary + SVGs.
- [ ] C3 + D1 cherry-picked from `claude/elastic-kirch-db47b2` (Track S1+S2).
- [ ] Track Q1 (route cache) implemented + measured against baseline.
- [ ] (Optional) Track Q2 (Node32 layer) implemented + canary re-lock cycle completed.
- [ ] PGO + LTO Cargo profile (R4) shipped.
- [ ] Phase 0.9 demo SVGs (Track U) committed and byte-stable.
- [ ] `docs/abng/CAPABILITIES.md` updated with Phase 0.9 entries.
- [ ] Phase 0.9 handoff doc (this file) updated with final numbers + lessons learned.

---

*This handoff sits alongside `PHASE_0_8_HANDOFF.md` (the parent phase) and `PHASE_0_8c_V14_HANDOFF.md` (the v14 wire-format follow-up). Future Phase 0.10 work should start a fresh handoff doc.*
