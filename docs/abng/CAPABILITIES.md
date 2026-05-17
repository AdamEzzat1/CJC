# ABNG Phase 0.8 + 0.9 — Capabilities Reference

**Last updated:** 2026-05-16 (post-Phase-0.9 close-out).
**Wire format:** v14 (`ABNG\x0E`) — unchanged through Phase 0.9.
**Phase 0.8 status:** all 11 items shipped on `claude/abng-v14-wire-format`.
**Phase 0.9 status:** Track P shipped + measurement layer + real-data
integration on `claude/abng-phase-0-9` (17 commits past `a0a8266`).

This doc is the single-page reference for ABNG capabilities
introduced in Phase 0.8 and 0.9 — what each one enables, which
API surfaces it lives behind, what the demo or test harness
proves about it, and the concrete numbers it produced.

For wire-format details (magic bytes, audit-event encoding,
backward compatibility), see [`V14_MIGRATION.md`](V14_MIGRATION.md).
For the design history and item-by-item rationale, see
[`PHASE_0_8_HANDOFF.md`](PHASE_0_8_HANDOFF.md),
[`PHASE_0_8c_V14_HANDOFF.md`](PHASE_0_8c_V14_HANDOFF.md), and
[`PHASE_0_9_STATUS.md`](PHASE_0_9_STATUS.md) (Phase 0.9 close-out).
For the load-bearing Phase 0.9 architectural insights, see
[`PHASE_0_9_ARCHITECTURE_INSIGHTS.md`](PHASE_0_9_ARCHITECTURE_INSIGHTS.md).

---

## Headline capabilities (each with a demo + visualization)

### A3 — External Merkle inclusion proofs

**What it enables.** A regulator/auditor with `(merkle_root, leaf_hash, index, n_leaves, proof)` can verify that a specific training event occurred at a specific position in the audit chain, **in `O(log N)` SHA-256 ops, without downloading the full audit log.** Before A3, the only verification path was a full sequential walk over every audit event.

**API surface (Rust only on this branch).**

```rust
let root: [u8; 32] = g.merkle_root();
let tree: MerkleTree = g.merkle_tree();
let proof: Vec<[u8; 32]> = tree.proof(i);
let ok: bool = MerkleTree::verify_proof(leaf, i, n_leaves, &proof, root);
```

Plus the snapshot trailer block (33 bytes per snapshot: `0x01` tag + 32-byte root) that any v14-aware reader cross-checks against the recomputed root.

**Demo.** [`tests/abng/demo_a3_merkle_proof.rs`](../../tests/abng/demo_a3_merkle_proof.rs)
- Workflow: issuer trains 16-event graph → publishes root + proof → auditor verifies → tamper-detection contrast.
- SVG: [`bench_results/phase_0_8_demos/a3_merkle_proof_tree.svg`](../../bench_results/phase_0_8_demos/a3_merkle_proof_tree.svg) — Merkle tree with the highlighted leaf, the proof path (red), and the witness siblings (orange).
- Numbers (N=16): proof = 4 hashes (128 bytes); verification cost = 4 SHA-256 ops vs 16 for a full-log walk → **4× cheaper at N=16, scaling as N/log₂(N).**

---

### A2 — Fused per-row training events

**What it enables.** Replacing the pre-A2 3-call sequence (`blr_update + observe`) with a single `train_step` call collapses the per-row audit footprint from 2 chain events (`BlrUpdated` + `BeliefUpdate`) into 1 (`TrainStep`, tag `0x1E`). The same Welford state + BLR posterior result; half the chain events.

**API surface (cjcl + Rust).**

```rust
g.train_step(&x, &phi, y)?;
```

```cjcl
abng_train_step(g, x, phi, y);
```

**Demo.** [`tests/abng/demo_a2_fused_training.rs`](../../tests/abng/demo_a2_fused_training.rs)
- Workflow: side-by-side 100-row training run, pre-A2 vs v14 A2.
- SVG: [`bench_results/phase_0_8_demos/a2_fused_training_compactness.svg`](../../bench_results/phase_0_8_demos/a2_fused_training_compactness.svg) — grouped bar chart of audit events + payload bytes.
- Numbers (N=100 rows):
  - Audit events: 219 → 119 (**−100, exactly one per row**)
  - Payload bytes: 17,797 → 11,697 (**−6,100 B, 34.3% reduction**)
  - Welford + BLR state: byte-identical on all 4 leaves ✓
  - Chain heads diverge (by design — A2's whole point)

---

### C2 — Parallel chain verification

**What it enables.** `verify_chain_par(n_threads)` splits the audit log into `n_threads` chunks and verifies them concurrently via `std::thread::scope`. At chain sizes ≥ 10,000 events (the threshold gate), the per-thread chunk verification + cross-chunk linkage check delivers measurable speedup.

**API surface (Rust only).**

```rust
g.verify_chain_par(n_threads)?;
```

**Demo.** [`tests/abng/demo_c2_parallel_verify.rs`](../../tests/abng/demo_c2_parallel_verify.rs)
- Workflow: build a 15,000-event chain → wall-clock `verify_chain_par(k)` at k ∈ {1, 2, 4, 8} → tamper test at every k.
- SVG: [`bench_results/phase_0_8_demos/c2_parallel_verify_scalability.svg`](../../bench_results/phase_0_8_demos/c2_parallel_verify_scalability.svg) — bar chart of speedup vs sequential reference.
- Numbers (N=15,000, on the development laptop):
  - Sequential reference: 22.6 ms (median of 5)
  - k=2: 2.18× speedup
  - k=4: **3.82× speedup** (likely the peak — matches 4 physical cores)
  - k=8: 3.76× (hyperthreading not helping)
- Tamper detection: caught at every k ✓
- Below the threshold (e.g. 500 events), `verify_chain_par` transparently falls through to sequential — same outcome as `verify_chain`.

---

### B3 — ZSTD-wrapped snapshots

**What it enables.** `serialize_compressed` produces a self-identifying snapshot blob (magic `ABNGZ\x01`) wrapped in a single zstd frame. Any tool that consumes ZSTD frames (S3 lifecycle, network proxies, archival systems) can store/retrieve ABNG snapshots transparently. `replay` auto-detects the wrapping and dispatches to the zstd decoder.

**API surface (Rust only, gated on `feature = "compression"`).**

```rust
let blob: Vec<u8> = serialize_compressed(&g, /* level */ 3);
let g2 = replay(&blob)?; // auto-detects the wrapping
```

**Demo.** [`tests/abng/demo_b3_zstd_snapshot.rs`](../../tests/abng/demo_b3_zstd_snapshot.rs)
- Workflow: serialize at multiple chain sizes, compressed + uncompressed → assert round-trip yields the same chain head and Merkle root.
- SVG: [`bench_results/phase_0_8_demos/b3_zstd_snapshot.svg`](../../bench_results/phase_0_8_demos/b3_zstd_snapshot.svg) — grouped bar chart, uncompressed vs compressed at multiple N.
- Numbers (level 3):
  - N=100: 14,127 B → 7,614 B (**1.86×**)
  - N=1,000: 137,427 B → 74,920 B (**1.83×**)
  - N=5,000: 685,427 B → 376,345 B (**1.82×**)
- Compression ratio stable at ~1.82× as the audit log scales — the audit log dominates snapshot bytes, so ratio holds.

---

## Infrastructure capabilities (no standalone demo)

### B1 — mmap snapshot replay

**What it enables.** `replay_mmap` / `replay_mmap_with_outcome` open a snapshot file via `mmap` instead of `fs::read`. For multi-GB historical archives, the OS handles paging on demand — no application-side RAM allocation for the whole file.

**API surface.**

```rust
let g = replay_mmap(&path)?;
let (g, outcome) = replay_mmap_with_outcome(&path)?;
```

**Why no standalone demo.** No realistic workload in the current CJC-Lang ecosystem produces multi-GB snapshots. The capability is operational ("when you eventually have huge archives, this will Just Work"), not workflow. Coverage lives in [`tests/abng/serialize_mmap.rs`](../../tests/abng/serialize_mmap.rs).

### B4 — Columnar `AuditLog`

**What it enables.** The audit log is stored as Struct-of-Arrays (`Vec<u64>` for seqs, `Vec<NodeId>` for node_ids, `Vec<[u8; 32]>` for new_hashes / previous_hashes, etc.) instead of `Vec<AuditEvent>` (Array-of-Structs). Critical primitive: `audit.new_hashes() -> &[[u8; 32]]` is a zero-copy slice over the chain witness column.

**Why no standalone demo.** B4 is the structural prerequisite for A3 (Merkle build needs a `&[[u8; 32]]` leaf list) and C2 (parallel chunks need direct column access). Calling B4 a "capability" the headline list does is generous; it's invisible infrastructure that enables the others. Its demo IS A3 + C2.

---

## Performance refactors (no canary-visible API change, no standalone demo)

### D2b — SIMD-friendly Kahan in `BlrState::update`

Replaces three scalar `KahanAccumulatorF64` reduction sites inside `BlrState::update` (`xtx`, `xty`, `yty`) with `KahanAccumulatorF64x4`, processing 4 rows per chunk via `add_lanes`. Bit-identical at `n ≤ 4`, bit-different at `n ≥ 5` (3 canaries re-locked). Win: ~3× on Cholesky-friendly inner loops at d ≥ 8. At d=4 (the default for demos) the absolute speedup is marginal; D2b's value is structural readiness for PINN/tabular workloads at higher feature widths.

### D3 — Fused matvec kernel for `Λ_old · μ_old`

Extracts the rhs computation into a named free helper, `matvec_plus_xty_kahan`, with a scalar path at `d ≤ 4` (bit-identical to pre-D3) and an F64x4 fused path at `d ≥ 8 && d % 4 == 0`. Zero canary impact at d=4 (every currently locked canary takes the scalar path).

---

## Gates this branch holds

| Gate | Count |
|---|---:|
| `cargo test -p cjc-abng --lib --release` | 320 |
| `cargo test --test abng --release` | 542 |
| `cargo test --test abng --release --features compression` | 549 |
| SHA-256 canaries (28 total) | 9 re-locked (6 A2, 3 D2b), 19 unchanged |
| Canary binaries (25 files) | all green |

---

## Honest framing

The headline capabilities split into "ship-ready for known consumers" and "infrastructure for hypothetical consumers":

| Capability | Has a current consumer? |
|---|---|
| **A2** fused training | Yes — the 6 migrated training demos use it today. |
| **A3** Merkle proofs | Not yet — no CJC-Lang user is producing inclusion proofs in production. The capability is structural readiness for that workflow. |
| **C2** parallel verify | Marginal — most current chains are below the 10K-event threshold. Audit-heavy workloads will benefit. |
| **B3** zstd snapshots | Yes — anyone archiving snapshots to cloud storage gains the 1.82× shrink immediately. |
| **B1** mmap replay | Not yet — no current workload produces multi-GB snapshots. |
| **B4** columnar audit | Indirectly — every A3 + C2 user benefits without knowing it. |

The Phase 0.8 wins concentrate in a "future-ready" zone: the infrastructure for tamper-evident external attestation, scalable verification, and snapshot interop with the broader ecosystem is now in place. The next phase's job is bringing consumers to that infrastructure.

---

# Phase 0.9 — Capabilities Added

Phase 0.9 was originally a baseline-validation phase. The
in-flight scope expanded to include real-data integration, a
calibration measurement layer, interpretability/route-utilization
stats, an architectural insight (leaf+root ensemble), and a
sharing-ready artifact bundle. Headline result: ABNG hits
**0.9519 mean accuracy** across 15 seeds on UCI Wisconsin
Breast Cancer — matching the published linear-classifier
ceiling — with full deterministic audit, calibration metrics,
and per-route interpretability.

## Headline capability (the architectural insight)

### P-Ensemble — Leaf+root Bayesian fallback

**What it enables.** The graph trains the root BLR in parallel
with the per-leaf BLRs, then averages leaf and root posterior
means at evaluate time. The root BLR sees every training row and
approximates the **global linear classifier** for the dataset.
The per-leaf BLRs add local specialization for regions where the
boundary isn't globally linear. Ensembling the two recovers
both behaviors.

**Why it's load-bearing.** On real Wisconsin BC, per-leaf BLR
alone hit a soft ceiling at 0.944 (one point below the
published LR ceiling of ~0.95). The leaf+root ensemble pushed
mean accuracy to **0.9519**, closing the gap entirely. The
insight generalizes:

> **Invariant:** *Every adaptive local model must have access
> to a calibrated global fallback unless explicitly disabled.*

See [`PHASE_0_9_ARCHITECTURE_INSIGHTS.md`](PHASE_0_9_ARCHITECTURE_INSIGHTS.md)
for the full design rationale.

**API surface (test-harness level for Phase 0.9).** Implemented in
the baseline harness via two paired calls per training row:

```rust
g.train_step(routing, phi, y)?;       // updates leaf BLR
g.blr_update(0, phi, &[y])?;           // ALSO updates root BLR

// At evaluate time:
let leaf_mean = g.blr_predict_with_fallback(leaf_id, phi)?.0;
let root_mean = g.blr_predict_with_fallback(0, phi)?.0;
let ensemble = 0.5 * (leaf_mean + root_mean);
```

**Promotion path.** Phase 0.10+ will add a graph-level
`FallbackMode` enum (`Disabled` / `RootEnsemble` /
`AncestorChain`) so any caller can opt into the pattern without
re-implementing the dual-train + ensemble logic.

**Cost.** Doubles the per-row audit event count (1 `TrainStep`
+ 1 `BlrUpdated`). On real BC at 454 train rows, total audit
events = 1,019 per trial.

**Demo.** [`tests/abng/baseline_wisconsin_bc.rs`](../../tests/abng/baseline_wisconsin_bc.rs)
* Workflow: train on 80% of UCI WDBC, evaluate on 20%, 15 seeds.
* Numbers (15-seed sweep on real Wisconsin BC):
  * Mean accuracy: **0.9519** (min 0.9217, max 0.9913) — matches LR ceiling.
  * Without ensemble (pure per-leaf): 0.944. Ensemble delta: **+0.008**.
  * On synthetic +1.8σ: 0.9739 (Bayes-optimal LDA ceiling ≈ 0.998).

---

## Measurement-layer capabilities

### P-Calib — Calibration metric layer (Brier / NLL / ECE)

**What it enables.** Every trial now reports trustworthiness
metrics alongside accuracy. `CalibrationReport` is appended to
`TrialResult`. Three metrics chosen because each catches a
different miscalibration mode:

* **Brier** = `mean((p - y)²)` — general prediction-quality MSE.
* **NLL** = `-mean(y log p + (1-y) log(1-p))` — sensitive to
  confidently-wrong predictions.
* **ECE (10-bin)** = Expected Calibration Error — catches systematic
  per-confidence-bucket miscalibration that Brier and NLL miss.

**API surface (test-harness level for Phase 0.9).**

```rust
pub(crate) struct CalibrationReport {
    pub brier_score: f64,
    pub nll: f64,
    pub ece_10_bins: f64,
    pub n_test: usize,
}

pub(crate) fn evaluate_calibration(
    g: &AdaptiveBeliefGraph,
    dataset: &Dataset,
    test_idx: &[usize],
    routing_features: &[usize],
    phi_features: &[usize],
) -> CalibrationReport;
```

Used by `run_trial` (always computed) + the artifact producer
(rendered in `wisconsin_bc_summary.md`).

**Numbers (15-seed mean on real WDBC):**

| Metric | Real BC | Synthetic +1.8σ | Notes |
|---|---:|---:|---|
| Brier | 0.0753 | 0.0163 | < 1/3 of "always-class-prior" baseline 0.234 |
| NLL | 0.3603 | 0.0859 | finite, bounded |
| ECE (10 bins) | 0.1268 | 0.0760 | tightenable via Platt scaling (Phase 0.10+) |

**Determinism gate.** `baseline_calibration_is_deterministic_across_5_runs`
asserts byte-equal Brier/NLL/ECE bits via `f64::to_bits()` across
5 same-seed reruns.

---

### P-Routes — Route utilization stats

**What it enables.** Graph-level visibility into how the data
actually populates the routing tree. Distinguishes routes that
exist (codebook can produce them) from routes that are visited
(training data lands there).

**API surface (test-harness level for Phase 0.9).**

```rust
pub(crate) struct RouteUtilization {
    pub total_leaves: usize,
    pub populated_leaves: usize,
    pub dead_leaves: usize,
    pub min_samples_per_populated_leaf: u64,
    pub max_samples_per_leaf: u64,
    pub mean_samples_per_populated_leaf: f64,
    pub std_samples_per_populated_leaf: f64,
}

pub(crate) fn compute_route_utilization(
    g: &AdaptiveBeliefGraph,
    dataset: &Dataset,
    train_idx: &[usize],
    routing_features: &[usize],
) -> RouteUtilization;
```

**Numbers (real WDBC, seed=1, 2⁴ binary tree):**

* 16 total leaves, **12 populated, 4 dead**
* Per-populated-leaf samples: min=1, max=**227 (50% of train)**, mean=37.83, std=66.99
* The dead-routes finding is a real interpretability signal —
  some combinations of top-4 F-score features simply don't occur
  in real BC data, even though the codebook can produce them.

**Demo.** SVG visualization at
`bench_results/phase_0_9_baseline/wisconsin_bc_route_utilization.svg`
(also rendered as PNG for LinkedIn/Instagram sharing).

---

### P-PerLeaf — Per-leaf test-set calibration

**What it enables.** Every populated leaf reports its own
test-set accuracy + mean predicted probability + Brier score,
extending the existing `PerLeafReport` shape with test-set
fields. The user-identified "Route 18: accuracy=96%,
confidence=95%, ECE=0.01" pattern.

**API surface (test-harness level).** `PerLeafReport` extended
with `n_test_samples`, `test_accuracy`, `test_mean_predicted`,
`test_brier`. Computed by `collect_per_leaf_reports(g, dataset,
train_idx, test_idx, routing_features, phi_features)`.

**Demo.** SVG scatter at
`bench_results/phase_0_9_baseline/wisconsin_bc_per_leaf_calibration.svg`
plots each populated leaf as a circle (x = mean predicted
probability, y = empirical accuracy, area ∝ n_test), with a
diagonal "perfect calibration" line.

---

## Infrastructure capabilities (Phase 0.9, no standalone demo)

### P-Data — Bundled UCI Wisconsin BC dataset

**What it enables.** Real-data baseline that works offline + is
tamper-detectable. The UCI `wdbc.data` byte-stream is bundled at
`tests/data/wisconsin_bc.csv` (124,103 bytes) with a pinned
SHA-256 constant in the test source. The loader does
hash-then-parse: soft-fail to synthetic on missing/tampered
file, hard-panic on post-hash parse failure.

**API surface.**

```rust
pub(crate) fn load_real_dataset() -> Option<Dataset>;
pub(crate) fn standardize_in_place(dataset: &mut Dataset);
```

Pinned constants:

```rust
const REAL_DATASET_REL_PATH: &str = "tests/data/wisconsin_bc.csv";
const REAL_DATASET_SHA256_HEX: &str =
    "D606AF411F3E5BE8A317A5A8B652B425AAF0FF38CA683D5327FFFF94C3695F4A";
```

`.gitattributes` marks `tests/data/*` as `binary` to prevent
cross-platform CRLF conversion from breaking the pinned hash.

---

### P-Harness — Phase 0.9 baseline test harness

**What it enables.** Self-contained baseline rig (3,395 LOC in
`tests/abng/baseline_wisconsin_bc.rs`) covering all Track P
contracts: 5-run + 15-run determinism gates, F-score top-K
selection, pre-allocated routing tree, leaf+root ensemble
training, full evaluation surface, artifact production.

**Test count.** **31 baseline tests** all green, plus 2
`#[ignore]`'d (the artifact producer + the threshold-sweep
diagnostic). Includes 4 distinct determinism gates:

1. Synthetic 5-run gate (chain head + Merkle root + audit count + accuracy bits).
2. Real-BC 5-run gate (same fields).
3. Calibration metric 5-run gate (Brier + NLL + ECE bits via `to_bits()`).
4. Route utilization 5-run gate (min/max/mean/std stat bits).

---

### P-Strat — Stratified train/test split

**What it enables.** `train_test_split` preserves the dataset's
class proportion (357/212 = 0.627/0.373 for WDBC) exactly in
both train and test, by Fisher-Yates-shuffling within each
class before taking the 80/20 cut. Without stratification,
uniform random splits can produce test-set class-1 counts
varying ±10 across seeds (a ~5σ hypergeometric draw), adding
~0.05 of accuracy noise per seed.

**Effect.** Empirical 15-seed accuracy variance dropped after
stratification was added — the worst seeds improved more than
the best seeds, tightening the distribution around the mean.

---

## Sharing-ready artifact bundle

### P-Bundle — `bench_results/phase_0_9_baseline/`

**What it enables.** Reproducible end-to-end artifact bundle
generated by a single `#[ignore]`'d test:

```bash
cargo test --test abng --release -- --ignored \
  baseline_wisconsin_bc_produce_artifacts
```

Writes to **two locations**:

1. `bench_results/phase_0_9_baseline/` — repo-tracked, byte-stable,
   committed alongside the source for CI regression detection.
2. `~/Downloads/phase_0_9_baseline/` — personal share copy for
   LinkedIn / Instagram / blog posts.

**Files (9 total per bundle):**

| File | Bytes | Description |
|---|---:|---|
| `wisconsin_bc_summary.md` | ~2.7 KB | Human-readable headline + config + metrics |
| `wisconsin_bc_real_15runs.csv` | ~3 KB | One row per real-BC seed |
| `wisconsin_bc_synthetic_5runs.csv` | ~1 KB | One row per synthetic seed |
| `wisconsin_bc_per_leaf_seed1.csv` | ~1 KB | Per-leaf reports (real BC seed=1) |
| `wisconsin_bc_chain_heads.txt` | ~4 KB | 20 chain heads + 20 Merkle roots |
| `wisconsin_bc_accuracy.svg` | ~4 KB | Box plot + 0.95 floor line |
| `wisconsin_bc_route_utilization.svg` | ~5 KB | Per-leaf bars, dead routes in red |
| `wisconsin_bc_per_leaf_calibration.svg` | ~5 KB | Scatter + diagonal |
| `wisconsin_bc_runtime.svg` | ~6 KB | Per-seed wall-clock (NOT byte-stable) |

Plus PNG renders for LinkedIn (1200×675) and Instagram (1080×1080)
in `~/Downloads/phase_0_9_baseline/social/`.

`.gitattributes` forces LF line endings on `bench_results/**/*.{md,csv,txt,svg}`
to keep the committed bytes consistent across platforms.

---

## Phase 0.9 gates this branch holds

| Gate | Count |
|---|---:|
| `cargo test --test abng --release baseline_` | 31 (+ 2 #[ignore]'d) |
| Synthetic 5-run determinism (chain head + Merkle + accuracy bits) | byte-equal ✓ |
| Real-BC 5-run determinism (same fields) | byte-equal ✓ |
| Synthetic accuracy floor (≥ 0.95) | 0.9739 ✓ |
| Real-BC single-seed floor (≥ 0.90) | 0.9391 ✓ |
| Real-BC 15-seed mean floor (≥ 0.95) | 0.9519 ✓ |
| Real-BC per-seed floor (≥ 0.85) | min 0.9217 ✓ |
| SHA-256 canaries (28 from Phase 0.8c, 9 re-locked) | unchanged ✓ |
| All blog-mentioned features still present | zero regressions ✓ ([report](PHASE_0_9_BLOG_VERIFICATION.md)) |

---

## Honest framing (Phase 0.9 edition)

| Capability | Has a current consumer? |
|---|---|
| **P-Ensemble** leaf+root | Yes — the baseline harness uses it today. Phase 0.10+ promotion to graph-level `FallbackMode` enum will broaden consumer surface. |
| **P-Calib** calibration metrics | Yes — every `run_trial` call now reports them; blog / sharing material relies on the numbers. |
| **P-Routes** route utilization | Yes — surfaces the "4 dead routes / one leaf holds 50% of train data" interpretability story directly. |
| **P-PerLeaf** per-leaf calibration | Yes — per-route interpretability foundation; could grow into per-route abstain decisions in Phase 0.10+. |
| **P-Data** bundled WDBC | Yes — the headline 0.9519 number depends on it. Pinning via SHA-256 means future Phase work can swap to other UCI datasets with the same hash-then-parse pattern. |
| **P-Harness** baseline harness | Yes — every Phase 0.10+ perf/architecture change should re-run this harness as the comparison floor. |
| **P-Strat** stratified split | Yes — tightens seed variance, makes 15-seed mean a more meaningful contract. |
| **P-Bundle** artifact bundle | Yes — feeds the blog / LinkedIn / Instagram sharing pipeline directly. |

The Phase 0.9 wins concentrate in **architectural insight +
measurement layer + real-data validation** — they make ABNG's
"accuracy + calibration + interpretability + determinism in one
architecture" pitch defensible with numbers, not just claims.
The next phase's job is to:

1. Promote the leaf+root ensemble to a first-class graph feature.
2. Tackle the Phase 0.8/0.9 perf-tracks (Q1 D-HARHT, R4 PGO+LTO,
   S1+S2 cherry-picks) that were deferred in favor of the
   user-driven scope expansion.
3. Add organic Grow/Split to a Track P-like baseline so
   cross-seed topology similarity becomes a measurable
   interpretability metric.

See [`PHASE_0_9_STATUS.md`](PHASE_0_9_STATUS.md) §5 ("Deferred
items") for the full Phase 0.10+ candidate list.

---

# Phase 0.9.5 — Research Phase R0: result-path performance

Research Phase R0 profiled ABNG's result path on the real Diabetes-130
categorical workload (predictive `phi` width d=247) and cut its
dominant cost. Full detail + the speedup design: [`PHASE_0_9_5_R0_PROFILE.md`](PHASE_0_9_5_R0_PROFILE.md).

## R0 — Result-path profiler (`bench/abng_result_profile`)

Segment-by-segment profiler of `transform → encode_prefix → descend →
train_step → blr_update → predict` on the real Diabetes-130 graph.
Established empirically that `state_hash` (SHA-256 over the d×d BLR
precision matrix) was **67 %** of the 6.88 ms/row result path and
**91 %** of *that* is irreducible SHA-256 compression — and that the
handoff's "1-2 hours for a 20K-row run" was machine contention, not
the algorithm (true cost ~110 s CPU for 16 000 rows).

```bash
cargo run -p abng-result-profile --release
```

## R0-2 — Streaming `state_hash` (Tier 1, byte-identical)

`BlrState::state_hash` streams the canonical bytes into SHA-256 in
4 KiB chunks (`hash_f64_slice_be`) instead of materialising a ~477 KB
`Vec`. Streaming SHA-256 == one-shot, so the digest is **byte-
identical** — the 28 SHA-256 canaries verified unchanged. Drains
~1 MB/row of allocator churn.

## R0-3 — Periodic BLR audit checkpoints (Tier 2, Option C)

**What it enables.** The per-row training witness hashes the full d×d
precision matrix only every `BLR_CHECKPOINT_INTERVAL` (= 64) updates;
intermediate rows carry an all-zero sentinel and stay fully chain-bound
via the outer `new_hash`. `AdaptiveBeliefGraph::checkpoint_blr()`
flushes the final state of mid-interval nodes — **call it once before
`serialize`** (the flush-before-serialize contract; a missed flush is
a loud `BlrStateHashMismatch` at replay, never silent corruption).

* **~2.9× faster result path** (measured): per-row 6.88 → 2.38 ms;
  16 000 training rows ~110 s → ~38 s CPU. Per-row hashing drops from
  O(d²) every row to amortized O(d²/64).
* **No wire-format bump** — `TrainStep` / `BlrUpdated` keep their exact
  byte shapes; v14 stays v14, decoders untouched.
* **Zero canaries re-locked** — the `decide_step` canaries and the
  Wisconsin BC baseline never train through `train_step` / n=1
  `blr_update`, so their chain heads are byte-identical. (`combine`'s
  `BlrUpdated` keeps the full witness, so the Merge-firing canary is
  unaffected.)
* **Auditability:** a tampered mid-interval state is still *detected*;
  *localization* coarsens from the exact row to a 64-row window. Signed
  off by the Lead Architect as an audit-model re-expression, not a
  reduction.

`cargo test --test abng` 624/0 (1 known wall-clock flake passes in
isolation); 9 new tests pin Option C.
