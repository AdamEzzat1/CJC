# Phase 0.9 — Updated Handoff Note (Session Status)

**Date:** 2026-05-16
**Branch:** `claude/abng-phase-0-9` (16 commits past the v14 close-out at `e795cd0`)
**Sibling docs:**
* [`PHASE_0_9_HANDOFF.md`](PHASE_0_9_HANDOFF.md) — original scope (kept unchanged as history)
* [`PHASE_0_9_ARCHITECTURE_INSIGHTS.md`](PHASE_0_9_ARCHITECTURE_INSIGHTS.md) — the two load-bearing design decisions
* [`PHASE_0_9_BLOG_VERIFICATION.md`](PHASE_0_9_BLOG_VERIFICATION.md) — every feature from the four user blog posts re-verified against the current codebase

This note is the **current** picture of Phase 0.9 after a long
iterative session. It covers:

1. Executive summary (what shipped this session)
2. **Full ABNG architecture** as of Phase 0.9
3. What's still open from the original handoff
4. Deferred items + Phase 0.10+ candidates
5. Three honest paths to close Phase 0.9
6. File / commit / artifact inventory
7. How to resume in a new session

---

## 1. Executive summary

Phase 0.9 was originally a two-track phase: **Track P** (baseline
on a known-solution problem) and **Track Q** (D-HARHT memory-
profile integration to speed up training). The session executed
Track P thoroughly and *expanded scope substantially* in response
to user-driven requests, while leaving Track Q and the other
perf-tracks (R, S, T) untouched.

**Headline accomplishment this session:** ABNG's first real-data
baseline (Wisconsin Breast Cancer) lands at **0.9519 mean
accuracy across 15 seeds** — matching the published linear-
classifier ceiling for the dataset — with full deterministic
audit + calibration + interpretability + sharing-ready artifacts.

**Headline architectural insight that emerged from measurement:**
*Every adaptive local model must have access to a calibrated
global fallback unless explicitly disabled.* The leaf+root
ensemble pattern is the implementation; the principle generalizes
beyond Phase 0.9's specific tree.

---

## 2. What this session accomplished

### 2.1 — Track P (baseline) — fully shipped

| Capability | Commits | Tests added |
|---|---|---|
| Scaffolding + 5-run determinism gate (existed at `a0a8266`, inherited) | n/a | +5 |
| F-score top-K routing feature selection | `a0209fc` | +3 |
| Pre-allocated routing tree (4³ → 4² → 2⁴ iterations) | `91ac223` | +2 |
| Accuracy evaluation + per-leaf explainability | `988196d` | +3 |
| Tune baseline to clear synthetic 0.95 floor | `2e68d25` | (config) |
| Bundled UCI Wisconsin BC + SHA-256 pin (B1) | `b64eae3`, `81bf85f` | +1 |
| Real-data loader with hash-then-parse (B2) | `2f7d1d0` | +3 |
| Real-data accuracy gates + standardize helper (B3) | `03269a2` | +4 |
| Architecture insights doc | `8c677f3` | (docs only) |
| **Leaf+root ensemble** — accuracy push to 0.95+ | `cc302d0` | (config) |
| Calibration metrics (Brier, NLL, ECE) | `e223fab` | +4 |
| Route utilization + per-route calibration | `2145866` | +4 |
| Artifact producer (CSV + summary.md + 4 SVGs) | `439a668`, `41e7422` | +1 |
| SVG fixes (route IDs + floor label) | `98dce94` | (visual) |
| Blog feature verification report | `c1b8ebb` | (docs only) |

**Total: 16 Phase 0.9 commits, +30 baseline tests (8 → 31), all green.**

### 2.2 — Measurement results

| Metric | Real Wisconsin BC | Synthetic (+1.8σ) |
|---|---:|---:|
| 15-seed mean accuracy | **0.9519** | – |
| Single-seed accuracy (seed=1) | 0.9391 | 0.9739 |
| min / max across 15 seeds | 0.9217 / 0.9913 | – |
| Brier score (15-seed mean) | 0.0753 | 0.0163 |
| NLL (15-seed mean) | 0.3603 | 0.0859 |
| ECE 10-bin (15-seed mean) | 0.1268 | 0.0760 |
| Total leaves / populated / dead | 16 / 12 / 4 | 16 / 16 / 0 |
| Max samples per leaf | 227 (50% of train) | 180 |
| Audit events per trial | 1,019 | – |

### 2.3 — Out-of-scope additions (user-driven mid-phase)

* Real Wisconsin BC integration (3 commits) — wasn't in the
  original Phase 0.9; added on user request as "Path B."
* Leaf+root ensemble — wasn't in the original; emerged from the
  user's "push real BC to 0.95+" request.
* Calibration metric layer (Brier/NLL/ECE) — wasn't in the
  original; added on user request as "ABNG's real value
  proposition is accuracy + trustworthy uncertainty."
* Route utilization stats + per-route calibration — wasn't in the
  original; added on user request as the interpretability layer.
* 15-seed sweep (vs handoff's 5) — wasn't in the original;
  added on user request.
* LinkedIn + Instagram PNGs + post text — wasn't in the
  original; added for blog/social sharing.
* Blog feature verification loop — wasn't in the original;
  added on user request to confirm no regressions.

---

## 3. Full ABNG architecture (as of Phase 0.9 end)

ABNG is now a **Bayesian-inspired structurally-adaptive belief
graph** with **eight architectural layers**, each documented in
the codebase and verified by the test suite.

### 3.1 — Routing layer (Phase 0.2)

* **`QuantileCodebook`** — per-feature quantile boundaries,
  one byte per dimension, fixed-length route key. 2/4/8/16/32/64/128/256 bins.
* **`AdaptiveChildren`** — ART-style children layout with
  auto-promotion between `Node4`/`Node16`/`Node48`/`Node256`/`Dense`.
* **`descend`** — longest-prefix walk through the trie; returns
  a `RouteEvidence` with `matched_prefix`, `leaf_id`, full `path`.
* **`descend_traced`** (Phase 0.4 Track A) — additionally emits a
  `Routed` audit event (kind `0x1B`) recording the routing path.
* **`route_to_leaf_batch`** + **`route_to_leaf_batch_par`** (Phase 0.8 C1) —
  batched + parallel routing for high-throughput training.
* **`encode_into`** — buffer-reuse codebook encoding (Phase 0.7 Item B,
  1.67× speedup over fresh-`Vec` allocation).

### 3.2 — Per-node MLP head (Phase 0.3a)

* **`LeafHead`** — feed-forward MLP with Xavier-uniform weights
  init'd deterministically from `mix(graph.seed, node_id, layer_idx, kind_bit)`.
* **`leaf_forward`** — wires the leaf MLP up to the penultimate
  layer into the ambient `cjc-ad::GradGraph`.
* **`leaf_set_param`** + **`leaf_set_params_batch`** — explicit
  parameter setters with audit witnesses.

### 3.3 — Per-node BLR (Phase 0.3b)

* **`BlrState`** — NIG-conjugate Bayesian Linear Regression.
  Closed-form posterior updates, no variational machinery.
* **`blr_update`** — single-row update (NIG-conjugate math via
  Cholesky + hand-rolled triangular solve).
* **`blr_predict`** — returns `(mean, epistemic_leverage,
  aleatoric_var)`. Student-t predictive distribution at df = 2a.
* **`blr_predict_with_fallback`** (Phase 0.4 C-2.3.8) — walks up
  parent chain when the target leaf has `n_seen = 0`.
* **`reset_blr`** (Phase 0.4 C-2.3.5) — clears posterior + refreshes
  `feature_version_hash` after MLP weight changes.
* **`BlrState::combine`** — NIG-aware merge math (precisions and
  precision-weighted means).
* **Phase 0.8c additions:**
  * **`train_step`** (A2) — fuses `blr_update` + `observe` into a
    single audit event (`TrainStep` 0x1E).
  * **Packed lower-triangular precision** (A1) — snapshot size
    reduction.
  * **SIMD-friendly Kahan refactor** (D2a/D2b) — auto-vectorized
    f64 accumulators without FMA.
  * **Fused matvec** (D3) — single-pass `Λ_old · μ_old` evaluation.

### 3.4 — Phase 0.9 leaf+root ensemble (NEW architectural insight)

* **Invariant:** *Every adaptive local model must have access to
  a calibrated global fallback unless explicitly disabled.*
* **Mechanism:** train the root BLR with every sample in
  parallel with the leaves; at predict time, average leaf and
  root posteriors.
* **Effect:** real Wisconsin BC accuracy 0.944 → 0.9519. Real-
  world linear data: root wins. Locally-complex data: leaf wins.
  Ensemble keeps both regimes' strengths.
* **Status:** wired in the Phase 0.9 baseline harness (test-
  level). **Not yet promoted to a first-class `FallbackMode`
  enum on the graph** — that's the primary Phase 0.10+ follow-up.
* **Cost:** doubles per-row audit footprint (TrainStep +
  BlrUpdated per row).

### 3.5 — Uncertainty + OOD machinery (Phase 0.3c)

* **`DensityTracker`** — running diagonal-Gaussian estimate per
  leaf. Mahalanobis distance scoring.
* **`CalibrationBins`** — 15-bin reliability diagram per leaf.
  ECE aggregation with Kahan summation.
* **`DriftBaseline`** — frozen reference density; L2 z-shift
  detection.
* **Composite OOD score** — `max(density_score,
  prefix_distance, epistemic_z)`.
* **Three-tier OOD** — LOW/MID/HIGH abstain bands.
* **Phase 0.9 addition: `CalibrationReport`** — Brier, NLL,
  ECE at the trial level.
* **Phase 0.9 addition: per-leaf calibration** — `PerLeafReport`
  extended with `test_accuracy`, `test_mean_predicted`,
  `test_brier`, `n_test_samples`.

### 3.6 — Structural adaptation engine (Phases 0.3d / 0.4)

* **Six actions** — Grow, Split, Merge, Prune, Compress, Freeze.
* **`decide_step`** — one-pass deterministic policy engine.
* **Fall-through order** — Compress → Merge → Split → Prune →
  Grow → Freeze.
* **At-most-one-action-per-node-per-call invariant.**
* **Snapshot-at-call-entry invariant** — mutations made during
  a `decide_step` call cannot cascade into further triggers in
  the same call.
* **`DecisionPolicy`** — 14 threshold knobs (Phase 0.4-ext v11
  added `ece_stability_max_delta` and `sigma_stability_ratio`).
* **Gates:**
  * Grow — `samples_seen ≥ grow_min` AND route-key entropy.
  * Split — bootstrap held-out ΔNLL gain ≥ threshold.
  * Merge — Hamming similarity AND KL-divergence < threshold.
  * Prune — low `samples_seen` AND signature instability.
  * Compress — sub-tree signature equivalence.
  * Freeze — samples_seen + calibration stable + uncertainty stable.
* **Auto-unfreeze** — drift score crosses `drift_unfreeze`
  threshold (Phase 0.4 B-2.2.7).
* **Phase 0.9 addition: `RouteUtilization`** — graph-level
  visibility into the distribution.

### 3.7 — Maturity, signature, stability (Phase 0.4 B-2.2)

* **`NodeSignature`** — 32-byte Welford-smoothed summary across
  4 channels per node.
* **`Maturity`** — flags (samples_seen, calibration_stable,
  uncertainty_stable, trust_level).
* **3-window ECE/σ stability ring buffers per node.**
* **`unfreeze_count` field** (v11) — observability for the
  `Unfreeze` event.

### 3.8 — Audit chain + determinism (Phases 0.1 through 0.8c)

* **31 audit kinds** (tags `0x00..0x1E`). Phase 0.8c added
  `TrainStep` (0x1E) as the fused per-row event.
* **SHA-256 hash-chained log** — `new_hash =
  sha256(prev_hash ‖ canonical_bytes(payload))`.
* **Genesis hash** — `sha256(b"ABNG-GENESIS-v1")`.
* **Per-node stats chain** — separate from the global chain.
* **Per-node `feature_version_hash`** (Phase 0.4 C-2.3.5).
* **Per-node `provenance_stamp_hash`** (Phase 0.5 Item 1).
* **Three-signal spoof detection** — chain head + per-leaf BLR
  state hash + dataset provenance commitment.
* **Determinism primitives:**
  * Custom FIPS 180-4 SHA-256 (zero external deps, in `cjc-snap`).
  * `cjc_repro::Rng` — SplitMix64 seeded from
    `(graph.seed, node_id, layer_idx, kind_bit)`.
  * `KahanAccumulatorF64` + `pairwise_sum_f64`.
  * `BTreeMap` / `BTreeSet` only (no `HashMap`).
  * `f64::to_bits().to_be_bytes()` canonical bytes.
  * No FMA in belief-touching kernels.
* **Phase 0.8c additions:**
  * **Merkle-indexed audit chain** (A3) — O(log N) inclusion
    proofs via `merkle_root` + `merkle_tree`.
  * **Parallel chain verify** (C2) — `verify_chain_par(n_threads)`.
  * **mmap snapshot replay** (B1) — `replay_mmap`.
  * **Streaming snapshot encode** (B2) — `serialize_into`.
  * **ZSTD-wrapped snapshots** (B3) — `serialize_compressed`.
  * **Columnar `AuditLog`** (B4) — SoA storage.
  * **v14 wire format** — `b"ABNG\x0E"` (was v13 `\x0D`).
    v13 reader retained for backward compatibility.

### 3.9 — CLI + inspection (Phase 0.4 Track A)

* **`cjcl abng inspect`** — walk audit log, dump per-node state.
* **`cjcl abng replay`** — reconstruct graph from snapshot.
* **`cjcl abng diff`** — compare two snapshots.
* **`cjcl abng explain`** — explain a prediction-snap blob.
* **`cjcl abng train`** — train a graph end-to-end from CLI.
* All five subcommands support `--json` output.

### 3.10 — The route/predictor separation (architectural insight, documented Phase 0.9)

* **Signature:** `train_step(x, phi, y)` where `x` is **routing
  representation** (compact, low-dim) and `phi` is **predictive
  representation** (rich, high-dim).
* **Why it matters:** the two roles want fundamentally different
  things from their input — routing wants compact + bin-friendly,
  prediction wants rich + standardization-friendly. Splitting them
  lets each role be tuned independently.
* **Audit-chain payoff:** the chain can witness routing decisions
  separately from predictive updates.

---

## 4. What's still open from the original handoff

The original `PHASE_0_9_HANDOFF.md` listed 8 "Required" items.
Status with respect to the original scope:

| # | Required item | Status |
|---|---|---|
| 1 | Baseline tests green at determinism + accuracy floor | ✅ shipped |
| 2 | `bench_results/phase_0_9_baseline/` populated | ✅ shipped |
| 3 | **Track S1+S2** — C3 + D1 cherry-picks from `claude/elastic-kirch-db47b2` | ❌ not touched |
| 4 | **Track Q1** — D-HARHT route cache integration | ❌ not touched |
| 5 | **Track R4** — PGO + LTO Cargo profile | ❌ not touched |
| 6 | Phase 0.9 demo SVGs (Track U) | ✅ shipped |
| 7 | **`docs/abng/CAPABILITIES.md`** updated with Phase 0.9 entries | ❌ pending (Commit E) |
| 8 | Phase 0.9 handoff doc updated with final numbers + lessons learned | 🟡 this document (partial); needs Commit E for the annotations on the original |

**5 of 8 required items still open** if we hold strictly to the original scope.

### Items 3, 4, 5 — the perf/D-HARHT tracks (not addressed)

The user's mid-phase scope expansion (real BC integration,
leaf+root ensemble, calibration layer, route utilization,
sharing artifacts, verification) was substantially higher-value
than the original perf-tracks plan. The user-driven additions
**were not envisioned in the original handoff**.

Tracks Q1, R4, and S1+S2 remain unaddressed. They are still
valid Phase 0.10+ work but were deprioritized this session in
favor of the user-driven scope.

### Items 7, 8 — docs (the "Commit E" gap)

Two specific docs need updating:

1. **`docs/abng/CAPABILITIES.md`** — currently lives on
   `claude/abng-v14-wire-format` at commit `e795cd0`. Needs to
   be brought onto this branch (or recreated) and extended with
   Phase 0.9 entries:
   * Leaf+root ensemble (new architectural primitive)
   * Calibration metrics (Brier/NLL/ECE)
   * Route utilization stats
   * Per-leaf calibration extension
   * Wisconsin BC bundle (SHA-256-pinned)
   * Baseline harness (`baseline_wisconsin_bc.rs`, 31 tests)

2. **`docs/abng/PHASE_0_9_HANDOFF.md`** — needs ✅/❌/⏭ annotations
   on the delivery checklist + final empirical numbers + lessons
   learned section.

**Effort: ~1 commit, ~30-60 min.**

---

## 5. Deferred items (Phase 0.10+ candidates)

### 5.1 — Direct extensions of Phase 0.9 work

* **Promote leaf+root ensemble to a first-class `FallbackMode`
  enum on `AdaptiveBeliefGraph`** with three modes: `Disabled`
  (default for back-compat), `RootEnsemble` (Phase 0.9 pattern),
  `AncestorChain` (full ancestor traversal). Test harness inlining
  in Phase 0.9 was a *proof of concept*; promoting it to graph-
  level would let any caller use the pattern without re-implementing.
* **`predict_with_fallback_chain(leaf_id, phi)` API** — returns
  posteriors from every ancestor on the descent path. Lets callers
  implement custom ensemble weighting (confidence-weighted,
  sample-weighted, KL-divergence-weighted).
* **Per-route ECE** — per-leaf 10-bin ECE (currently only Brier
  is computed per-leaf because typical n_test_per_leaf is too small
  for 10-bin ECE to be meaningful; could use 3-bin or quartile
  ECE per-leaf).
* **Cross-seed topology similarity** — meaningful only when the
  tree is grown organically (Phase 0.9 used pre-allocated tree
  where topology is identical across seeds by construction).
  Becomes valuable once organic Grow/Split triggers are exercised
  in a Track P-like baseline.
* **Empirical-quantile codebook boundaries** — Phase 0.9 used
  hard-coded z-score boundaries. Empirical quantiles computed from
  the train split would adapt better to non-Gaussian real-world
  features.
* **Platt scaling / isotonic regression for calibration improvement** —
  the current 0.13 ECE on real BC could likely be halved with
  post-hoc calibration.

### 5.2 — Original handoff items not addressed

* **Track Q1** — D-HARHT memory-profile integration for routing
  speedup (route memoization side cache).
* **Track Q2** — D-HARHT Node32 + SSE2 Node16 lookup (bigger
  layout refactor; canary re-lock cycle needed).
* **Track Q3** — Full D-HARHT-backed leaf-key index. Deferred
  in the original handoff to Phase 0.10+; still deferred.
* **Track R** — Other speed/memory wins (R1 C3 merge, R2 D1
  merge, R3 batched train_step, R4 PGO+LTO, R5 streaming
  decompression, R6-R8 misc).
* **Track S** — Cherry-picks from `claude/elastic-kirch-db47b2`
  (C3 per-thread arena, D1 Cholesky factor caching).
* **Track T** — cjcl-surface builtins for v14 Rust-only APIs
  (`abng_merkle_root`, `abng_merkle_proof`,
  `abng_merkle_verify_proof`, `abng_verify_chain_par`).
* **Track V** — Research data-structure replacements (BLR
  representation alternatives, columnar NodeStats, audit-chain
  index variants, HAT-trie alternatives, per-leaf state
  co-location, codebook encoding alternatives).
* **Track W** — Numerical kernel research (Neumaier summation,
  deterministic GEMM, SIMD intrinsics, PGO/BOLT, allocator
  alternatives, mixed-precision training).

### 5.3 — Operational items

* **Open PR for `claude/abng-v14-wire-format` on GitHub** —
  still origin-only.
* **Push `claude/abng-phase-0-9` to origin** — not yet done
  (per directive: no origin pushes without explicit approval).
* **Re-run PINN macro bench at HEAD post-D2b/D3** — original
  handoff item S3.
* **Update `PHASE_0_8_HANDOFF.md` macro bench section with re-run
  numbers** — original handoff item S4.

---

## 6. Three honest paths to close Phase 0.9

### Path A — Ship now (Commit E only)

Treat Phase 0.9 as **done after Commit E**. Update the docs
(`CAPABILITIES.md` + handoff annotations), write a deferral note
explaining why Q1/R/S/T didn't land, push the branch (with user
approval), open a PR, and move to Phase 0.10.

**Effort:** ~1 hour. Ships Phase 0.9 today.

**Phase 0.10 then focuses on:** promoting the leaf+root ensemble
to a graph-level `FallbackMode` enum + tackling Track Q1
(D-HARHT route cache) with a dedicated perf-benchmark harness.

### Path B — Tackle Track S1 (cherry-pick C3), then ship

C3 (per-thread arena observability for concurrent training) is
the lowest-risk perf item — pure observability, no canary risk.
~2-3 commits. Then Commit E.

**Effort:** ~3-5 hours.

### Path C — Full original scope

Cherry-pick C3 + D1 (Track S), implement Q1 route cache (Track Q),
add PGO+LTO (Track R4), wire cjcl-surface builtins (Track T),
then Commit E.

**Effort:** 1-2 full days.

### Recommendation

**Path A**, because the user-driven scope expansion produced a
Phase 0.9 that's much more *shippable as a narrative* than the
original perf-tracks plan. Real-data integration + accuracy push
+ calibration + interpretability + sharing artifacts are
storytellable; PGO/LTO + cherry-picks are not. The perf items
benefit from being their own Phase 0.10 work (with a dedicated
benchmark harness), not bolted onto a baseline phase.

---

## 7. File / commit / artifact inventory

### 7.1 — Commits on `claude/abng-phase-0-9` (16 total since `a0a8266`)

```
c1b8ebb  docs: PHASE_0_9_BLOG_VERIFICATION.md (zero regressions)
98dce94  Track P: SVG fixes (correct leaf IDs + floor label)
41e7422  Track P: .gitattributes -- force LF for bench_results/
439a668  Track P: artifact producer (CSV + summary.md + 4 SVGs)
2145866  Track P: route utilization + per-route calibration
e223fab  Track P: calibration metrics (Brier, NLL, ECE)
8c677f3  docs: architecture insights (route/predictor + root-ensemble)
cc302d0  Track P: push real BC accuracy to 0.95+ via leaf+root ensemble
03269a2  Track P B3: real-data accuracy gates + standardize helper
2f7d1d0  Track P B2: real-data loader load_real_dataset() + 3 tests
81bf85f  Track P B1: .gitattributes -- treat tests/data/* as binary
b64eae3  Track P B1: bundle UCI wdbc.data with SHA-256 pin
2e68d25  Track P: tune baseline config to clear 0.95 accuracy floor
988196d  Track P: accuracy evaluation + per-leaf explainability
91ac223  Track P: pre-allocate 4^3 routing tree (85 nodes)
a0209fc  Track P: F-score top-K routing feature selection
[a0a8266] Pre-existing scaffolding (inherited from prior session)
```

Plus 2 inherited docs commits (`5dca9e1`, `e607ef0`) from
`claude/abng-v14-wire-format` that established the original
handoff doc.

### 7.2 — Files added / modified

**Code:**
* `tests/abng/baseline_wisconsin_bc.rs` — 3,395 LOC (started at 397)
* `tests/data/wisconsin_bc.csv` — bundled UCI dataset (124,103 bytes)
* `.gitattributes` — text/binary EOL rules for repo-tracked artifacts

**Repo-tracked artifacts (`bench_results/phase_0_9_baseline/`):**
* `wisconsin_bc_summary.md`
* `wisconsin_bc_real_15runs.csv`
* `wisconsin_bc_synthetic_5runs.csv`
* `wisconsin_bc_per_leaf_seed1.csv`
* `wisconsin_bc_chain_heads.txt`
* `wisconsin_bc_accuracy.svg`
* `wisconsin_bc_route_utilization.svg`
* `wisconsin_bc_per_leaf_calibration.svg`
* `wisconsin_bc_runtime.svg`

**Docs:**
* `docs/abng/PHASE_0_9_ARCHITECTURE_INSIGHTS.md` (251 LOC, new)
* `docs/abng/PHASE_0_9_BLOG_VERIFICATION.md` (349 LOC, new)
* `docs/abng/PHASE_0_9_STATUS.md` (this doc, new)

**User-local sharing folder (`C:\Users\adame\Downloads\phase_0_9_baseline\`):**
* All 9 `bench_results/` artifacts (mirrored)
* `social/linkedin_*.png` (4 PNGs, 1200×675)
* `social/instagram_*.png` (4 PNGs, 1080×1080 centered)
* `social/linkedin_post.md` (long + short post versions)
* `social/instagram_post.md` (caption + carousel + story tips)

### 7.3 — Test counts (final)

* `cargo test --test abng --release baseline_` → **31 passing,
  2 #[ignore]'d** (the artifact producer + the threshold-sweep
  diagnostic)
* `cargo test --test test_abng_pinn_uncertainty --release` → 13/13
* `cargo test --test test_abng_lineage_attestation --release` → 16/16
* 5 trigger tests (grow/split/prune/compress/freeze) → 39/39
* All blog-mentioned features verified present (zero regressions,
  see `PHASE_0_9_BLOG_VERIFICATION.md`)

### 7.4 — Headline numbers (re-stated for the handoff)

**Real Wisconsin BC (15-seed sweep on the UCI WDBC dataset
569×30, 357 benign / 212 malignant):**

* **Accuracy mean = 0.9519** (min 0.9217, max 0.9913)
* Brier mean = 0.0753, NLL mean = 0.3603, ECE mean = 0.1268
* Wire format v14; audit chain 1,019 events per trial; 5-run
  determinism gate byte-equal across all repetitions.

**Synthetic (+1.8σ on 10 of 30 features):**

* Accuracy seed=1 = 0.9739 (Bayes-optimal LDA ceiling ≈ 0.998)
* Brier 0.0163, NLL 0.0859, ECE 0.0760

**Route utilization (real BC seed=1):**

* 16 total leaves, 12 populated, **4 dead**
* Per-populated-leaf samples: min=1, max=227 (50% of train), mean=37.83, std=66.99

---

## 8. How to resume in a new session

Whatever path is chosen (A / B / C), the next session needs:

### 8.1 — Read first

1. **This document** (`PHASE_0_9_STATUS.md`) for current state.
2. **[`PHASE_0_9_ARCHITECTURE_INSIGHTS.md`](PHASE_0_9_ARCHITECTURE_INSIGHTS.md)** for the
   load-bearing architectural decisions.
3. **[`PHASE_0_9_BLOG_VERIFICATION.md`](PHASE_0_9_BLOG_VERIFICATION.md)** for the
   complete feature inventory verified against the codebase.
4. **[`PHASE_0_9_HANDOFF.md`](PHASE_0_9_HANDOFF.md)** for the
   original scope (still the canonical reference for tracks
   Q/R/S/T that didn't land).
5. **[`ABNG_CURRENT_ARCHITECTURE.md`](ABNG_CURRENT_ARCHITECTURE.md)** —
   the source-of-truth architecture doc (snapshot at Phase 0.4-ext;
   pre-dates Phase 0.8 and 0.9 but covers everything earlier).
6. **`docs/abng/V14_MIGRATION.md`** (on `claude/abng-v14-wire-format`)
   — Phase 0.8c v14 wire-format migration guide.

### 8.2 — Run first

```
cargo test --test abng --release baseline_
```

Should print `31 passed; 0 failed; 0 ignored; 2 ignored (the
producers); finished in ~10s`.

If anything fails, the determinism contract has broken — STOP
and investigate before any new work.

### 8.3 — Pick a path

* **If Path A (Commit E only):** open `docs/abng/CAPABILITIES.md`
  on `claude/abng-v14-wire-format` (commit `e795cd0`), copy it
  onto this branch, extend with Phase 0.9 entries. Then update
  `docs/abng/PHASE_0_9_HANDOFF.md` with ✅/❌/⏭ markers and final
  numbers. One commit.
* **If Path B (S1 cherry-pick first):** `git log
  claude/elastic-kirch-db47b2 -- crates/cjc-abng/` to find the
  C3 commits; cherry-pick onto this branch; resolve conflicts;
  run tests; then Commit E.
* **If Path C:** read the original handoff's Q1 design notes,
  implement the route memoization side cache, measure against
  baseline, then cherry-pick + PGO + cjcl-surface + Commit E.

### 8.4 — Operational

* **Branch:** `claude/abng-phase-0-9` (16 commits on top of
  `a0a8266`, which itself sits 4 commits past
  `origin/claude/abng-v14-wire-format`).
* **NOT pushed to origin** (per directive — no origin pushes
  without explicit user approval). The branch is local-only.
* **No PR opened.**

---

## 9. Open questions for the next session

1. **Path A, B, or C?** (See §6.)
2. **CAPABILITIES.md** — bring across from `claude/abng-v14-wire-format`
   via cherry-pick, or recreate fresh on this branch?
3. **Origin push** — push `claude/abng-phase-0-9` to origin
   (when ready)? Open a PR?
4. **Phase 0.10 framing** — single phase that takes everything
   Phase 0.9 deferred, or split into two (Phase 0.10 = ensemble
   promotion + Q1 D-HARHT; Phase 0.11 = R/S/T perf-tracks)?
5. **Blog post** — should I draft the Phase 0.9 blog post (the
   one the user mentioned writing) as part of Commit E, or leave
   that to the user with my notes as the source material?

---

## 10. Lessons learned this session

1. **Measurement-driven design discoveries beat scope-fidelity.**
   The leaf+root ensemble wasn't in the original scope. It
   emerged from the user's "push real BC to 0.95+" prompt. It
   turned out to be a load-bearing architectural insight worth
   promoting to invariant. The Phase 0.9 plan was too rigid to
   have predicted it.

2. **Calibration is what makes accuracy trustworthy.** Adding
   Brier/NLL/ECE alongside accuracy made the baseline
   significantly more credible — without those, "0.9519
   accuracy" is just a number. With them, it's a *quality
   claim with quantified uncertainty*. Future ABNG baselines
   should default to all three.

3. **Per-route interpretability is qualitatively different from
   per-prediction interpretability.** The "4 dead routes / one
   leaf holds 50% of train data" finding is the most insightful
   visualization the project has produced so far. Future Phase
   0.10+ work should add organic Grow/Split + cross-seed
   topology similarity to make this richer.

4. **Bundled-data baselines outperform synthetic-only baselines
   for blog narratives.** Real Wisconsin BC + the SHA-256
   provenance pin gives the project a defensible "this works on
   real data" story that synthetic alone never could.

5. **Architecture insights are worth documenting as soon as
   they're discovered.** `PHASE_0_9_ARCHITECTURE_INSIGHTS.md`
   was written *while the insights were fresh* and is now the
   single source of truth for the route/predictor separation +
   the leaf+root ensemble. Without it, those insights would
   live only in commit messages.

6. **The audit chain handles architectural changes cleanly.**
   Adding the leaf+root ensemble doubled the per-row event
   count (TrainStep + BlrUpdated). Zero canary breakage. The
   determinism contract scales with feature additions; the
   wire-format scaffolding from Phase 0.8c was the right
   foundation.

---

*Document maintained by the Phase 0.9 close-out session.
Next-session readers: start with §8 ("How to resume").*
