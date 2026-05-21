# CJC-Lang ABNG Phase 0.9.5 — Categorical Medical Scaling + Performance Research (Handoff v2)

**Date stamped:** 2026-05-17
**Branch:** `claude/abng-phase-0-9-5`
**Worktree:** `C:\Users\adame\CJC\.claude\worktrees\silly-brahmagupta-107016`
**Supersedes:** [`PHASE_0_9_5_HANDOFF.md`](PHASE_0_9_5_HANDOFF.md) (v1) — the v1 design
and 9-commit plan are carried forward verbatim below (§1-§10); this v2
**prepends a new first priority — Research Phase R0 — and records the
current build status.**

This is a design+handoff doc. The phase spec (the "ABNG PHASE 0.9.5 —
Medical Scaling + Deterministic Categorical Support" prompt) remains the
authority on the role group, the metric list, and the verification loop.

---

# RESEARCH PHASE R0 — ABNG Performance & Memory Efficiency

**This is the immediate priority.** Before the Phase 0.9.5 commit
sequence continues (COMMIT 5 onward, §7), ABNG's result path must be
made dramatically faster and more memory-efficient. Categorical scaling
is gated on training throughput.

## R0.1 — Goal

Massively reduce the wall-clock and memory cost of ABNG's **result
path** — training (`train_step`) and inference (`predict`) — so that
heterogeneous-categorical workloads (Phase 0.9.5's Diabetes-130 / BRFSS,
predictive `phi` width up to ~270+) train in **minutes, not hours**.
Every existing ABNG guarantee is held invariant.

## R0.2 — Hard constraints (non-negotiable)

* **Determinism.** Same seed + same data + same build → byte-identical
  outputs (predictions, chain head, Merkle root, every snapshot hash).
  No FMA. No nondeterministic parallel reductions. No platform-specific
  intrinsics that change results.
* **Numerical accuracy.** No precision regression. Kahan / Binned
  accumulation wherever floating-point reductions matter.
* **Auditability.** The audit event chain, the per-node stats chain,
  and the Merkle roots stay intact and tamper-evident. Replay
  reconstructs byte-identical state.
* **No exotic hardware.** Pure CPU. No GPU, no non-bit-portable SIMD
  intrinsics, no hardware-specific code paths.
* **All current ABNG features preserved** — the no-GC core boundary,
  dual-executor agreement for any cjcl surface, the 28 SHA-256
  canaries, the Wisconsin BC baseline (0.9519), snapshot v14, the
  decision triggers, route caching, calibration, etc.
* **Backward compatibility preferred.** A snapshot wire-format bump is
  permitted **only with explicit sign-off** (it forces a canary
  re-lock).

## R0.3 — Why this phase exists

Phase 0.9.5's COMMIT 5 (the Diabetes-130 harness) demonstrated that
ABNG training does not scale to wide categorical `phi`: a deterministic
20,000-row sub-sample took **~1-2 hours of wall-clock**. The phase's
categorical-scaling goal cannot be met until per-row training cost
drops by an order of magnitude or more.

## R0.4 — Findings already in hand (a head start)

A first investigation this session produced concrete results:

1. **FIXED — the O(d³) Cholesky.** `BlrState::update` recomputed a full
   Cholesky from scratch on **every rank-1 `train_step`**. Commit
   `08a4a6b` ("abng(0.9.5) perf: O(d²) rank-1 Cholesky update for the
   n=1 BLR hot path") replaced it with a Givens-rotation rank-1 update
   (`chol_rank1_update`, `O(d²)`) and dropped the per-call ~6 MB
   Kahan-accumulator scratch on the n=1 path. Determinism, accuracy,
   and auditability preserved; no wire-format change; the incremental
   factor lives in a private, non-serialized `chol_factor` field, so
   `predict` still uses `cholesky(precision)` and live vs. replayed
   graphs predict bit-identically. No canary re-lock was needed — only
   two cache-*behavior* tests were updated. `cjc-abng` lib 383/383;
   `abng` target 612/0; Wisconsin BC determinism + accuracy floors
   hold.

2. **SUSPECTED #1 remaining hotspot — the per-row audit `state_hash`.**
   `train_step` (`graph.rs:1628`) computes `blr.state_hash()` on every
   row — a SHA-256 over `BlrState::canonical_bytes`, which serializes
   the **full d×d precision matrix** (~585 KB at d=270). The root+leaf
   ensemble pattern does this **twice per row** → on the order of
   ~18 GB of hashing for a 16,000-row train split. It is `O(d²)` per
   row and load-bearing for the audit chain — the hard one.

3. **Memory churn.** The audit chain clones the d×d precision into a
   fresh `canonical_bytes` `Vec` per event; `transform` allocates fresh
   `x` / `phi` `Vec`s per row; the harness materialises a `train_rows`
   clone. None of this causes memory *pressure* (peak ~700 MB), but it
   is real allocate / copy / free CPU cost scaling with d² and row
   count.

4. **The 87-minute stall on the last 20K run was machine CPU
   contention** — the box was oversubscribed; the test averaged
   ~0.8 cores over 87 minutes. It was **not** memory pressure and not
   the algorithm. *Memory efficiency is a genuine secondary speed lever*
   (allocation churn wastes cycles), *but it was not the acute cause of
   that particular stall.* Both levers — algorithmic cost and memory
   churn — are in scope for this phase.

## R0.5 — Research agenda

* **Profile the full result path with real measurements** (not only
  code reading): `load` → `transform` → `train_step` { `encode_prefix`,
  `descend`, `BlrState::update`, `state_hash`, `observe`,
  `append_event` } → `predict`. Rank the hotspots empirically.
* **The audit `state_hash`** (the prime suspect) — evaluate
  determinism-and-auditability-preserving options, e.g.:
  * *Commit-to-inputs-and-replay* — the chain commits to each step's
    `(φ, y)` (hashing a d-vector is `O(d)`) instead of the post-update
    d×d state; replay re-derives state. A standard event-sourcing audit
    model — but an audit-format change.
  * *Incremental / structured state digests* updatable in `O(d)` per
    rank-1 update.
  * *Periodic full-state snapshotting* (every k steps) rather than
    per-step.
  * For each: format change? canary re-lock? wire-format bump?
    Document the tradeoffs — the team decides.
* **Memory** — buffer reuse across the BLR / audit / transform paths
  (ABNG's `encode_into` buffer-reuse convention is the precedent);
  eliminate redundant d×d clones; mutate in place where the borrow
  model allows.
* **Deterministic parallelism** — evaluate carefully. Training is a
  sequential hash chain (events must be ordered), so naive parallelism
  breaks the chain; there may be deterministic batch formulations with
  a fixed reduction order. High risk — gated by the Determinism
  Auditor.
* **`predict`** — the first-predict-per-node `O(d³)` Cholesky
  (post-speedup, lazy) is fine if amortised over an eval pass; revisit
  if not.

## R0.6 — Process

Stack role group:

* **Runtime Systems Engineer** + **Numerical Computing Engineer** own
  the hot path.
* **Determinism & Reproducibility Auditor** gates every change — each
  change is either provably bit-identical or a deliberate, signed-off
  re-lock.
* **QA Automation Engineer** owns verification: the 28 SHA-256
  canaries, AST↔MIR parity, the determinism double-run, the Wisconsin
  BC baseline.
* **Lead Language Architect** rules on any change that touches the
  audit model or the wire format.

Deliverable: a design doc / ADR ranking the hotspots and proposing the
changes, then commit-by-commit implementation, each one re-running the
full determinism gate (`cargo test --test abng`, the canaries, the
Wisconsin baseline).

## R0.7 — Relation to the 9-commit plan

Research Phase R0 takes priority. COMMIT 5 (the Diabetes-130 harness)
is written but **not finalised** — see "Current build status" at the
end of this doc. Once R0 produces a faster training path, COMMIT 5's
always-run test footprint is settled and COMMITs 6-9 (BRFSS harness,
seed sweeps, artifacts, close-out) proceed against the faster baseline.

---

# CARRIED FORWARD FROM v1 — Phase 0.9.5 Design + 9-Commit Plan

*The following §0-§10 are the Phase 0.9.5 categorical-scaling design,
unchanged from `PHASE_0_9_5_HANDOFF.md`.*

## 0. What this phase is / isn't

Phase 0.9.5 is a **scaling + categorical-support** phase. It answers
one question: *can ABNG scale from tiny numeric medical data into
larger heterogeneous categorical medical datasets while preserving
determinism, auditability, calibration visibility, and route
interpretability?*

It is **NOT** a cross-model benchmark. No XGBoost/RF comparison, no
speed/memory leaderboards, no superiority claims. Known baselines are
recorded only as plausibility sanity-checks. Cross-model comparison is
deferred to **Phase 1.0**.

## 1. Datasets — ROLE 1 complete

Both datasets are acquired under `tests/data/` (untracked, ~35 MB,
`.gitignore`'d — reproduction is via the URLs + the `raw_dataset_hash`).

### Dataset A — UCI Diabetes 130-US Hospitals

* **101,766 rows × 50 columns.** `tests/data/diabetes_130/diabetic_data.csv`
  + `IDS_mapping.csv`. Source: UCI dataset 296.
* **Target:** `readmitted` ∈ {`NO`, `<30`, `>30`}. Phase 0.9.5
  binarises to the published **30-day readmission** task: `<30` =
  positive (≈ 11%), `{>30, NO}` = negative. Class-imbalanced →
  balanced accuracy + AUC are the headline metrics, not raw accuracy.
* **Categorical regime — high-cardinality nominal.** `diag_1/2/3` are
  ICD-9 codes with 700+ distinct values each; `medical_specialty` ≈ 70;
  `payer_code` ≈ 18; 23 medication columns (`No/Steady/Up/Down`); plus
  `race`, `gender`, `age` (10 brackets), `weight`, `max_glu_serum`,
  `A1Cresult`, `change`, `diabetesMed`. Missing is the literal `?`.
  Drop `encounter_id`, `patient_nbr` (identifiers — leakage risk).
* **Numeric:** `time_in_hospital`, `num_lab_procedures`,
  `num_procedures`, `num_medications`,
  `number_outpatient/emergency/inpatient`, `number_diagnoses`.

### Dataset B — UCI CDC Diabetes Health Indicators (BRFSS 2015)

* **253,680 rows × 23 columns.** `tests/data/cdc_brfss.csv` (13.5 MB).
  Source: UCI dataset 891.
* **Target:** `Diabetes_binary` ∈ {0, 1} (≈ 14% positive). Drop `ID`.
* **Categorical regime — low-cardinality binary/ordinal.** 14 binary
  0/1 columns; 4 ordinal brackets (`GenHlth` 1-5, `Age` 1-13,
  `Education` 1-6, `Income` 1-8). No explicit missing sentinel.
* **Numeric:** `BMI`, `MentHlth` (0-30), `PhysHlth` (0-30).

### Why this pair — and the honest note on the ~20K target

No clean, freely-downloadable ~20K categorical medical dataset with
published baselines exists. Per the spec's "closest defensible
alternative" clause: the **~20K rung** is a **deterministic stratified
20,000-row sub-sample of Dataset A** (seed-fixed, class-ratio-
preserving). The two datasets are deliberately **complementary
categorical regimes** — A high-cardinality nominal, B low-cardinality
binary/ordinal.

**Scaling rungs:** 569 (Phase 0.9 Wisconsin BC, numeric) → 20K (A
sub-sample) → 101,766 (A full) → 253,680 (B).

## 2. Prime directives

Carried forward from Phase 0.9 / 0.10:

1. **Determinism** — same seed + same data + same build → byte-identical
   outputs.
2. **No `HashMap` / `HashSet` with random iteration** — `BTreeMap` /
   `BTreeSet` / sorted `Vec` only.
3. **No FMA in math kernels**; seed-based splitmix64 RNG only.
4. **Audit chain + Merkle roots preserved.**
5. **Both executors agree** for any cjcl-surface change (the
   categorical subsystem is Rust-internal for 0.9.5 — no cjcl
   builtins).

New for Phase 0.9.5:

6. **Categorical transforms are leakage-free.** The category
   vocabulary, rare-fold decisions, and numerical standardization
   statistics are computed from the **train split only**.
7. **No route explosion.** Routing never consumes raw one-hot columns
   or raw high-cardinality codes. The route key is a small, capped set
   of features.
8. **Phase 0.9 Wisconsin BC must stay green** — `baseline_wisconsin_bc.rs`
   and the 28 canaries must not regress.

## 3. The deterministic categorical subsystem — the core design

### 3.1 — Module placement

Library module **`crates/cjc-abng/src/categorical.rs`** — library code,
not test-harness code, because the schema/vocab hashes are part of the
determinism contract.

### 3.2 — `CategoryDictionary` — deterministic per-feature vocabulary

* Reserved codes first: `CODE_MISSING = 0`, `CODE_UNKNOWN = 1`,
  `CODE_RARE = 2`. Real categories occupy `FIRST_REAL_CODE = 3 ..`,
  ordered by **(−train_frequency, category_string)** — most-frequent
  first, ties lexicographic.
* Built from the **train split only** and then **frozen** (the
  builder-consumes-into-dictionary transition is the freeze).

### 3.3 — Missing / Unknown / Rare

* **Missing** — an empty / `?` cell maps to `CODE_MISSING`; a
  first-class category, not dropped.
* **Unknown** — a category absent from the frozen dictionary maps to
  `CODE_UNKNOWN`; never panics.
* **Rare** — see 3.4.

### 3.4 — Rare-category folding policy

A real category folds into `CODE_RARE` if it fails **either** floor:
absolute count `< min_count` (default 30), **or** train frequency
`< min_frac` (default 0.001). The policy is stored in the schema
snapshot so a replay reproduces the fold set exactly.

### 3.5 — Encoding modes

* **Ordinal (category-code) encoding** — for **routing `x`**.
  High-cardinality features are *binned* by frequency rank into
  ≤ `route_bins` buckets before they may enter `x`.
* **One-hot encoding** — for **prediction `phi`**. A feature with `K`
  post-fold categories expands to `K` indicator columns; a feature
  still wide after folding one-hots only the top-`M` (`max_real`) +
  the RARE/MISSING/UNKNOWN columns, so `phi` width stays bounded.
  (Effect coding is a Phase 1.0 alternative.)

### 3.6 — `SchemaSnapshot` — the determinism hash bundle

Every run stores, via `cjc_snap::hash::sha256`: `raw_dataset_hash`,
`schema_hash`, `categorical_vocab_hash`, `numeric_standardization_hash`,
`feature_transform_version`, `split_seed`, `row_count`,
`target_definition`. `snapshot_hash()` over all of them is the single
value a replay checks.

### 3.7 — Determinism contracts

* **Row-order invariance** — shuffling input rows yields an identical
  `CategoryDictionary`, `schema_hash`, `categorical_vocab_hash`.
* **Train-only** — vocabulary + standardization stats are a pure
  function of the train split.
* **Unknown safety** — transforming a never-seen category never panics;
  it yields `CODE_UNKNOWN`.
* **Rare determinism** — the fold set is a pure function of train
  counts + `RarePolicy`.
* **Double-run** — same seed + same CSV → identical every hash above.

## 4. Feature role split (route vs predict)

Phase 0.9's `train_step(x, phi, y)` seam is the integration point.

* **`x` — routing.** Top-`K` routing features (`K` ≈ 4) by mutual
  information with the target, train split only. Each routing feature
  is capped at `route_bins` (≈ 4-8). **Hard guard:** `diag_1/2/3` and
  the 23 medication columns are **never** routing features — `phi`
  only.
* **`phi` — prediction.** One-hot categoricals (top-`M` cap on wide
  features) + standardized numerics. `phi` width is asserted under a
  ceiling so a schema change cannot silently explode the BLR dimension.

## 5. cjc-abng integration

The categorical subsystem is a preprocessing layer; the ABNG core
barely changes. **Added (library):** the `categorical` module; a
`CategoricalTransform` holding the frozen dictionaries +
standardization stats + `SchemaSnapshot`, producing `(x, phi)` rows.
**Unchanged:** root + leaf BLR ensemble, `descend` routing, the audit
chain, Merkle roots, calibration, route caching. No new `Value`
variants; no cjcl builtins; v14 wire format untouched.

## 6. Metrics, verification, artifacts

* **Metrics:** accuracy, AUC/ROC, F1, balanced accuracy (lead metric
  on the imbalanced targets); Brier, NLL (ε-clipped), ECE; route/leaf
  stats; audit counts + hashes; ABNG-*internal* runtime/memory only.
* **Verification loop:** build + Phase 0.9 regression → unit →
  proptest → bolero fuzz → determinism double-run → seed sweep
  (15 A / ≥ 5 B) → ablation → artifact check → skeptical review.
* **Artifacts:** `bench_results/phase_0_9_5_medical_scaling/` —
  `summary.md`, `dataset_a_results.csv`, `dataset_b_results.csv`,
  per-leaf CSVs, `categorical_vocab_report.csv`, `chain_heads.txt`,
  `known_solution_context.md`, SVGs, `README.md`.

## 7. Implementation sequencing — the 9-commit plan

```
COMMIT 1  categorical.rs — CategoryDictionary + reserved codes +
          RarePolicy + freeze; unit tests.
COMMIT 2  Encoding modes (ordinal + one-hot) + SchemaSnapshot hash
          bundle; unit tests.
COMMIT 3  proptest suite (row-order invariance, train-only, unknown
          safety, rare determinism) + bolero fuzz (malformed CSV,
          huge / Unicode labels, mixed columns, missing target).
COMMIT 4  CategoricalTransform (dictionaries + standardization +
          snapshot -> (x, phi)); MI-based top-K routing-feature
          selection with the route-explosion cap.
COMMIT 5  Dataset-A harness (Diabetes-130): load, binarise target,
          stratified split, train through the root+leaf ensemble,
          metrics, per-leaf report, ablation. + the 20K sub-sample run.
COMMIT 6  Dataset-B harness (BRFSS): same shape, low-cardinality regime.
COMMIT 7  Seed sweeps (15 A / >=5 B) + determinism double-run gate.
COMMIT 8  Artifact producer — CSVs, summary.md, vocab report,
          chain_heads.txt, SVGs, README.
COMMIT 9  known_solution_context.md + the close-out report; CAPABILITIES
          / handoff doc updates.
```

Phase 0.9 regression (`baseline_wisconsin_bc.rs` + canaries) re-runs at
every commit.

## 8. Stack role group — sign-off

ROLE 1 (Dataset Curator) ✅. ROLE 2 (Categorical Systems) owns
COMMITs 1-4. ROLE 3 (ABNG Architecture) owns §5 + COMMITs 5-6. ROLE 4
(Calibration) owns §6 metrics. ROLE 5 (Determinism) owns COMMIT 3 +
7's double-run. ROLE 6 (Benchmark) owns COMMITs 5-8. ROLE 7
(Performance) owns the internal scaling report **and now Research
Phase R0**. ROLE 8 (Skeptical Reviewer) owns the final review.

## 9. Open questions — resolved

* **`phi` encoding** — one-hot for 0.9.5 (effect coding is Phase 1.0).
* **Schema-snapshot audit event** — kept harness-side in
  `chain_heads.txt`, no v14 wire-format touch.
* **20K sub-sample** — a sub-run inside the Dataset-A harness.
* **Defaults** — `K = 4`, `route_bins = 4`, `min_count = 30`,
  `min_frac = 0.001`, `max_real = 32`; tune against the determinism
  gate. (COMMIT 5's harness currently uses `K = 3` to keep the
  pre-allocated routing tree's per-leaf BLR memory modest.)
* **Branch** — `claude/abng-phase-0-9-5`.

## 10. Verdict on the spec's targets

* **~20K Dataset A** — a deterministic 20K stratified sub-sample of
  Diabetes-130, documented honestly.
* **200K+ Dataset B** — hit: BRFSS at 253,680 rows.
* **Categorical support** — the load-bearing new subsystem (§3-4).
* **Determinism / audit / calibration / route interpretability** — all
  preserved by construction.
* **Cross-model comparison** — out of scope; Phase 1.0.

---

# CURRENT BUILD STATUS (2026-05-17)

## Commits on `claude/abng-phase-0-9-5`

```
08a4a6b  abng(0.9.5) perf: O(d^2) rank-1 Cholesky update for the n=1 BLR hot path
c545a28  abng(0.9.5) docs: status note -- COMMIT 3 + 4 complete
7029e2f  abng(0.9.5) COMMIT 4: CategoricalTransform -- raw rows -> (x, phi, y)
8dac3bd  abng(0.9.5) COMMIT 3: proptest properties + bolero fuzz targets
cdcdc58  abng(0.9.5) COMMIT 2: encoding modes + schema snapshot
69dd2ba  abng(0.9.5) COMMIT 1: deterministic CategoryDictionary
15d2c4b  abng(0.9.5) design: categorical medical scaling handoff
c2339e2  abng(0.10) R4 ...   <- branch base
```

## Done

* **COMMIT 1-4** — the deterministic categorical subsystem in
  `crates/cjc-abng/src/categorical.rs`: `CategoryDictionary`,
  `OneHotEncoder`, `route_bucket`, `SchemaSnapshot`, `Standardizer`,
  `CategoricalTransform`. 58 unit tests + 11 proptest properties
  (`tests/abng/categorical_proptest.rs`) + 4 bolero fuzz targets
  (`tests/abng/categorical_fuzz.rs`).
* **Perf commit (`08a4a6b`)** — the O(d²) rank-1 Cholesky update (see
  R0.4 finding 1).
* Test state: `cjc-abng` lib **383/383**; `abng` target **612/0**
  (2 ignored, plus the Diabetes-130 tests). The 28 canaries + the
  Wisconsin BC baseline + AST↔MIR parity + replay all green.

## COMMIT 5 — IN PROGRESS, UNCOMMITTED

* `tests/abng/dataset_a_diabetes130.rs` (~560 lines) — the Diabetes-130
  benchmark harness — is **written and in the working tree but
  uncommitted**, along with an uncommitted `mod dataset_a_diabetes130;`
  line in `tests/abng/mod.rs`.
* The harness covers: CSV reader, the 50-column `diabetes_schema()`,
  binarise / stratified split / stratified sub-sample, graph build
  (codebook + leaf head + tree sized from `CategoricalTransform`),
  the root+leaf training loop, metrics (accuracy, balanced accuracy,
  rank-sum AUC, F1, Brier, NLL, ECE), per-leaf report, root/leaf/
  ensemble ablation. 9 tests (8 + an `#[ignore]`d full-101K run).
* The `select_subsample` off-by-one (it produced 19,999 not 20,000) is
  **fixed** in the working tree.
* **Not finalised:** the always-run test currently trains the full
  20,000-row sub-sample, which is too heavy for `cargo test` (~1-2 hr,
  dominated by machine contention even after the BLR speedup). It must
  be **re-scoped to a small (~2,000-row) stratified sub-sample**, with
  the 20K and full-101K runs `#[ignore]`d. The 20K run has **never
  produced metrics** — the first attempt died on the off-by-one, the
  second was killed during Research Phase R0. So Diabetes-130 has **no
  accuracy/AUC numbers yet.**
* After R0 delivers a faster training path, finalise COMMIT 5 (small
  always-run sub-sample), then proceed to COMMIT 6-9.

## Build / test

* `cargo test -p cjc-abng --lib categorical` — categorical unit tests.
* `cargo test -p cjc-abng --lib` — full lib suite (383).
* `cargo test --test abng -- --skip diabetes130` — abng integration
  suite minus the heavy Diabetes-130 tests (~50 s).
* Full `cargo test --test abng` — a *release* build is ~14 min (LTO).
* **Known flake:** `per_thread_arena_tests::concurrent_training_scales_better_than_serial`
  is a wall-clock scaling test that fails intermittently under the full
  parallel harness; it passes in isolation. Re-run it alone to confirm
  rather than treating a full-suite failure there as a regression.

## Loose ends (not blockers)

* Untracked carry-overs: `docs/abng/PHASE_0_10_HANDOFF.md` +
  `docs/abng/PHASE_1_0_FALLBACK_DEFAULT_DRAFT.md`.
* `bench_results/phase_0_8_demos/*.svg` show as modified — test-run
  noise; `git checkout --` discards.
* `claude/abng-phase-0-9-5` is local-only — not pushed, no PRs.

---

*This v2 handoff supersedes `PHASE_0_9_5_HANDOFF.md` and the
`PHASE_0_9_5_STATUS.md` resume note. Start the next session here:
Research Phase R0 (performance + memory) is the priority; COMMIT 5 is
written-but-unfinalised; the rest of the 9-commit plan follows R0.*
