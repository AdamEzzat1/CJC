# CJC-Lang ABNG Phase 0.9.5 — Categorical Medical Scaling (Design + Handoff)

**Date stamped:** 2026-05-16
**Branch:** `claude/abng-phase-0-9` (HEAD `c2339e2`, after the Phase 0.10 Q1/R4 commits)
**Status:** ROLE 1 (Dataset Curator) complete — both datasets acquired and inspected.
This doc **locks the categorical-subsystem design** against the two real
schemas *before* any determinism-critical code is written, then hands off
the implementation sequence.

This is a design+handoff doc, not the phase spec. The phase spec (the
"ABNG PHASE 0.9.5 — Medical Scaling + Deterministic Categorical Support"
prompt) is the authority on the role group, the metric list, the
artifact list, and the verification loop. This doc adds what the spec
left open: **the concrete categorical subsystem**, grounded in the
actual Diabetes-130 and BRFSS column schemas.

---

## 0. What this phase is / isn't

Phase 0.9.5 is a **scaling + categorical-support** phase. It answers one
question: *can ABNG scale from tiny numeric medical data into larger
heterogeneous categorical medical datasets while preserving determinism,
auditability, calibration visibility, and route interpretability?*

It is **NOT** a cross-model benchmark. No XGBoost/RF comparison, no
speed/memory leaderboards, no superiority claims. Known baselines are
recorded only as plausibility sanity-checks. Cross-model comparison is
deferred to **Phase 1.0**.

---

## 1. Datasets — ROLE 1 complete

Both datasets are acquired under `tests/data/` and inspected.

### Dataset A — UCI Diabetes 130-US Hospitals

* **101,766 rows × 50 columns.** `tests/data/diabetes_130/diabetic_data.csv`
  + `IDS_mapping.csv`. Source: UCI dataset 296.
* **Target:** `readmitted` ∈ {`NO`, `<30`, `>30`}. Phase 0.9.5 binarises
  to the published **30-day readmission** task: `<30` = positive (≈ 11%),
  `{>30, NO}` = negative. Class-imbalanced → balanced accuracy + AUC are
  the headline metrics, not raw accuracy.
* **Categorical regime — high-cardinality nominal.** `diag_1/2/3` are
  ICD-9 codes with 700+ distinct values each; `medical_specialty` ≈ 70;
  `payer_code` ≈ 18; 23 medication columns (`No/Steady/Up/Down`); plus
  `race`, `gender`, `age` (10 brackets), `weight`, `max_glu_serum`,
  `A1Cresult`, `change`, `diabetesMed`. Missing is encoded as the literal
  `?`. Drop `encounter_id`, `patient_nbr` (identifiers — leakage risk).
* **Numeric:** `time_in_hospital`, `num_lab_procedures`, `num_procedures`,
  `num_medications`, `number_outpatient/emergency/inpatient`,
  `number_diagnoses`.

### Dataset B — UCI CDC Diabetes Health Indicators (BRFSS 2015)

* **253,680 rows × 23 columns.** `tests/data/cdc_brfss.csv` (13.5 MB).
  Source: UCI dataset 891.
* **Target:** `Diabetes_binary` ∈ {0, 1} (≈ 14% positive — imbalanced).
  Drop `ID`.
* **Categorical regime — low-cardinality binary/ordinal.** 14 binary
  0/1 columns (`HighBP`, `HighChol`, `CholCheck`, `Smoker`, `Stroke`,
  `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`,
  `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `DiffWalk`,
  `Sex`); 4 ordinal brackets (`GenHlth` 1-5, `Age` 1-13, `Education`
  1-6, `Income` 1-8). No explicit missing sentinel (BRFSS is cleaned).
* **Numeric:** `BMI`, `MentHlth` (0-30), `PhysHlth` (0-30).

### Why this pair — and the honest note on the ~20K target

The spec asked for a ~20K Dataset A and a 200K+ Dataset B. **No clean,
freely-downloadable ~20K categorical medical dataset with published
baselines exists** — the field jumps from tiny (Heart Failure, 299 rows)
to large (BRFSS, 253K). SEER requires a signed NCI data-use agreement;
Kaggle and MIMIC need credentials. Per the spec's explicit "closest
defensible alternative" clause:

* The **~20K rung** is covered by a **deterministic stratified 20,000-row
  sub-sample of Dataset A** (seed-fixed, class-ratio-preserving), run as
  an additional scale point inside the Dataset-A harness.
* The two datasets are deliberately **complementary categorical regimes**
  — A is high-cardinality nominal, B is low-cardinality binary/ordinal —
  so the categorical subsystem is validated across its whole design
  space, not on one regime twice.

**Scaling rungs:** 569 (Phase 0.9 Wisconsin BC, numeric) → 20K (A
sub-sample) → 101,766 (A full) → 253,680 (B). Four points, two real
heterogeneous categorical datasets.

---

## 2. Prime directives

Carried forward from Phase 0.9 / 0.10:

1. **Determinism** — same seed + same data + same build → byte-identical
   outputs (predictions, chain head, Merkle root, all snapshot hashes).
2. **No `HashMap` / `HashSet` with random iteration** — `BTreeMap` /
   `BTreeSet` / sorted `Vec` only.
3. **No FMA in math kernels**; seed-based splitmix64 RNG only.
4. **Audit chain + Merkle roots preserved.**
5. **Both executors agree** for any cjcl-surface change (the categorical
   subsystem is Rust-internal for 0.9.5 — no cjcl builtins planned).

New for Phase 0.9.5:

6. **Categorical transforms are leakage-free.** The category vocabulary,
   rare-fold decisions, and numerical standardization statistics are
   computed from the **train split only** — never the test split, never
   the full dataset. This is the #1 skeptical-review attack surface
   (ROLE 8); the property tests must pin it.
7. **No route explosion.** Routing never consumes raw one-hot columns or
   raw high-cardinality codes. The route key is built from a small,
   capped set of features (§4). A categorical dataset must not produce a
   combinatorially large tree.
8. **Phase 0.9 Wisconsin BC must stay green.** Phase 0.9.5 is additive;
   the existing `baseline_wisconsin_bc.rs` 31-test suite and the 28
   canaries must not regress.

---

## 3. The deterministic categorical subsystem — the core design

### 3.1 — Module placement

New library module: **`crates/cjc-abng/src/categorical.rs`** (or a
`categorical/` submodule if it grows past ~600 LOC). Library code, not
test-harness code — because the schema snapshot hashes and vocab hashes
are part of the determinism contract and must be unit/property/fuzz
tested independently of any dataset.

The subsystem is fundamentally a **preprocessing layer**: it transforms
raw heterogeneous rows into the numeric `x` (routing) and `phi`
(prediction) vectors that ABNG's existing `train_step(x, phi, y)`
already consumes. This keeps the integration light — see §5.

### 3.2 — `CategoryDictionary` — deterministic per-feature vocabulary

```rust
pub struct CategoryDictionary {
    /// Sorted (category string -> code). BTreeMap, never HashMap, so
    /// iteration and the vocab hash are deterministic.
    codes: BTreeMap<String, u32>,
    /// Reverse: code -> category, for reports.
    labels: Vec<String>,
    /// Per-category train-split frequency (for rare-fold + reports).
    counts: Vec<u64>,
    /// Fold/missing/unknown reserved codes (see 3.3).
    n_real: u32,
}
```

**Code assignment — deterministic.** Reserved codes come *first* so they
are fixed regardless of vocabulary size:

* `CODE_MISSING = 0` — the source cell was empty / `?`.
* `CODE_UNKNOWN = 1` — a category not present in the train split (only
  reachable at test/inference time).
* `CODE_RARE = 2` — a real category folded by the rare policy (§3.4).
* Real categories occupy codes `3 .. 3 + n_real`, ordered by
  **(−train_frequency, category_string)** — most-frequent first, ties
  broken lexicographically. Frequency-then-lexical ordering is fully
  deterministic and gives common categories low codes.

The dictionary is **built from the train split only** (directive 6) and
then **frozen** — a `freeze()` call snapshots it; subsequent mutation is
an error, mirroring `QuantileCodebook`'s frozen-at-install contract.

### 3.3 — Missing / Unknown / Rare

* **`MissingCategory`** — an empty or `?` source cell maps to
  `CODE_MISSING`. Missing is a *first-class category*, not dropped — its
  presence is often predictive in clinical data.
* **`UnknownCategory`** — at transform time, a category string absent
  from the frozen dictionary maps to `CODE_UNKNOWN`. Never panics
  (fuzz-tested). Unknown is distinct from missing.
* **`RareCategory`** — see 3.4.

### 3.4 — Rare-category folding policy

A real category whose train-split count is below a threshold is folded
into `CODE_RARE`. The policy is **stored in the schema snapshot** so a
replay reproduces it exactly:

```rust
pub struct RarePolicy {
    /// Absolute count floor. A category with train count < this folds.
    min_count: u64,        // default 30
    /// Fractional floor. A category with train freq < this folds.
    min_frac: f64,         // default 0.001  (0.1% of train rows)
}
```

A category folds if it fails **either** floor. For Diabetes-130's
`diag_1` (700+ ICD-9 codes, a long tail appearing once or twice),
rare-folding is essential — without it the vocabulary and the one-hot
`phi` width explode. The folded categories and the policy parameters are
written to `categorical_vocab_report.csv` and the snapshot.

### 3.5 — Encoding modes

Two modes, matching the route/predictor split:

* **Ordinal (category-code) encoding** — for **routing `x`**. The
  feature's `u32` code (post-fold). High-cardinality features are
  *binned* by frequency rank into ≤ `route_bins` buckets before they may
  enter `x` (§4) — never the raw code.
* **One-hot / effect-coded encoding** — for **prediction `phi`**. A
  feature with `K` post-fold categories expands to `K` indicator columns
  (one-hot) or `K-1` (effect-coded, baseline = most-frequent). For a
  feature still wide after rare-folding (e.g. `diag_1`), `phi` one-hots
  only the **top-M** categories + the `RARE`/`MISSING`/`UNKNOWN` columns,
  `M` a config knob — so `phi` width stays bounded.

### 3.6 — `SchemaSnapshot` — the determinism hash bundle

Every run stores, via `cjc_snap::hash::sha256`:

```rust
pub struct SchemaSnapshot {
    raw_dataset_hash: [u8; 32],        // SHA-256 of the source CSV bytes
    schema_hash: [u8; 32],             // canonical column-name + role list
    categorical_vocab_hash: [u8; 32],  // canonical CategoryDictionary set
    numeric_standardization_hash: [u8; 32], // per-column (mean, std)
    feature_transform_version: u32,    // bump on any transform change
    split_seed: u64,
    row_count: u64,
    target_definition: String,         // e.g. "readmitted == '<30'"
}
```

### 3.7 — Determinism contracts (the property/fuzz test targets)

* **Row-order invariance** — shuffling input rows yields an identical
  `CategoryDictionary`, identical `schema_hash`, identical
  `categorical_vocab_hash`. (Vocab is frequency+lexical sorted; counts
  are order-free.)
* **Train-only** — the vocabulary and standardization stats are a pure
  function of the train split; the test split cannot influence them.
* **Unknown safety** — transforming a row with a never-seen category
  never panics; it yields `CODE_UNKNOWN`.
* **Rare determinism** — the fold set is a pure function of train counts
  + `RarePolicy`.
* **Double-run** — same seed + same CSV → identical every hash above.

---

## 4. Feature role split for categorical data (route vs predict)

Phase 0.9's `train_step(x, phi, y)` seam is the integration point.

### `x` — routing representation (compact, route-explosion-guarded)

* Pick the **top-K routing features** by mutual information with the
  target, computed on the **train split only** (`K` ≈ 4, as in Phase
  0.9). MI is deterministic given the train split.
* Each routing feature's effective (post-fold) cardinality is **capped
  at `route_bins`** (≈ 4-8): a low-cardinality feature uses its code
  directly; a high-cardinality one is binned by frequency rank. The
  route key is then `K` small bytes — the existing `QuantileCodebook` /
  `descend` machinery routes it unchanged.
* **Hard guard:** `diag_1/2/3` and the 23 medication columns are
  **never** routing features for Dataset A — they go to `phi` only.
  Expected Dataset-A routing features: `age` (binned), `A1Cresult`,
  `admission_type_id` (binned), `number_inpatient` (binned numeric).
  Dataset B: `GenHlth`, `HighBP`, `Age` (binned), `BMI` (binned numeric).

### `phi` — predictive representation (rich, bounded)

* Categorical features: one-hot / effect-coded per §3.5, with the top-M
  cap on wide features.
* Numeric features: standardized (z-score, train-split mean/std).
* `phi` width is reported in the artifacts and asserted under a ceiling
  so a schema change can't silently explode the BLR dimension.

This split is the spec's "avoid route explosion" guidance made concrete:
routing sees ≤ `K · route_bins` distinct prefixes; prediction sees the
full (bounded) encoded vector.

---

## 5. cjc-abng integration

Because the categorical subsystem is a preprocessing layer, the ABNG
core barely changes — this is deliberate (Phase 0.9 architecture
preserved):

* **Unchanged:** root + leaf BLR ensemble, `descend` routing, the audit
  chain, Merkle roots, calibration machinery, route-utilization
  reporting, deterministic splits, per-leaf stats. The Phase 0.10 Q1
  route cache and the cherry-picked C3/D1 also stay untouched.
* **Added (library):** the `categorical` module (§3); a
  `CategoricalTransform` that holds the frozen dictionaries +
  standardization stats + `SchemaSnapshot` and produces `(x, phi)` rows.
* **Added (diagnostics):** categorical route summaries, per-feature
  category-frequency / rare / unknown counts (→ `categorical_vocab_report.csv`),
  and the schema/vocab hashes folded into the harness's `chain_heads.txt`
  output. Whether a dedicated audit-event kind is added for the schema
  snapshot is an open question (§9) — default: no, the harness reports
  it alongside the chain head, no wire-format change.

No new `Value` variants; no cjcl-surface builtins; v14 wire format
untouched.

---

## 6. Metrics, verification, artifacts

These follow the phase spec verbatim — not re-derived here. Key points:

* **Metrics:** accuracy, AUC/ROC, F1, balanced accuracy (both targets
  are imbalanced, so balanced accuracy + AUC lead); Brier, NLL (ε-clipped),
  ECE, reliability-diagram data; the ABNG-specific route/leaf stats;
  audit counts + hashes; ABNG-*internal* runtime/memory only (no
  cross-model comparison — Phase 1.0).
* **Verification loop:** build + Phase 0.9 regression → unit → proptest
  → bolero fuzz → determinism double-run → seed sweep (15 for A, ≥ 5 for
  B) → ablation (root-only / leaf-only / root+leaf) → artifact check →
  skeptical review.
* **Artifacts:** `bench_results/phase_0_9_5_medical_scaling/` —
  `summary.md`, `dataset_a_results.csv`, `dataset_b_results.csv`,
  `per_leaf_dataset_a.csv`, `per_leaf_dataset_b.csv`,
  `categorical_vocab_report.csv`, `chain_heads.txt`,
  `known_solution_context.md`, the SVG set, `README.md`.

---

## 7. Implementation sequencing

```
COMMIT 1  categorical.rs — CategoryDictionary + reserved codes +
          RarePolicy + freeze; unit tests (deterministic order,
          missing/unknown/rare, freeze contract).
COMMIT 2  Encoding modes (ordinal + one-hot/effect) + SchemaSnapshot
          hash bundle; unit tests.
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
every commit. Estimated 9-12 commits, 4,000-7,000 LOC — genuinely
multi-day, checkpoint-driven.

---

## 8. Stack role group — sign-off

The phase spec defines eight roles. Sign-off map: ROLE 1 (Dataset
Curator) ✅ done (§1). ROLE 2 (Categorical Systems) owns COMMITs 1-4.
ROLE 3 (ABNG Architecture) owns §5 + COMMITs 5-6. ROLE 4 (Calibration)
owns §6 metrics. ROLE 5 (Determinism) owns COMMIT 3 + 7's double-run.
ROLE 6 (Benchmark) owns COMMITs 5-8. ROLE 7 (Performance) owns the
internal scaling report. ROLE 8 (Skeptical Reviewer) owns the final
review — primary attack surface: train/test leakage via the vocabulary
(directive 6), route explosion (directive 7), threshold overfitting,
dead-route over-interpretation.

---

## 9. Open questions

1. **Schema-snapshot audit event** — fold the schema/vocab hashes into a
   new audit-event kind (wire-format touch, canary re-lock), or keep
   them harness-side in `chain_heads.txt` (zero wire impact)?
   Recommendation: harness-side for 0.9.5.
2. **Effect vs one-hot for `phi`** — effect-coding drops one column per
   feature (baseline = most-frequent). Recommendation: one-hot for 0.9.5
   (simpler; the BLR's prior handles the redundancy), effect-coding a
   Phase 1.0 refinement.
3. **20K sub-sample** — a sub-run inside the Dataset-A harness, or its
   own dataset-A-small harness? Recommendation: a sub-run (one harness,
   parameterised by row budget).
4. **`route_bins` / `K` / `RarePolicy` defaults** — start with `K = 4`,
   `route_bins = 4`, `min_count = 30`, `min_frac = 0.001`, `top_M = 32`;
   tune against the determinism gate, not accuracy.
5. **Branch** — continue on `claude/abng-phase-0-9`, or a fresh
   `claude/abng-phase-0-9-5`? Recommendation: fresh branch.

---

## 10. Verdict on the spec's targets

* **~20K Dataset A** — not separately available as a clean free dataset;
  covered by a deterministic 20K stratified sub-sample of Diabetes-130,
  documented honestly (no silent downgrade — the spec's hard rule).
* **200K+ Dataset B** — hit: BRFSS at 253,680 rows.
* **Categorical support** — designed in §3-4; the load-bearing new
  subsystem.
* **Determinism / audit / calibration / route interpretability** — all
  preserved by construction; §2 directives + §3.7 contracts.
* **Cross-model comparison** — out of scope; Phase 1.0.

*Next: with this design reviewed, execute COMMIT 1. This doc sits
alongside `PHASE_0_9_HANDOFF.md`, `PHASE_0_10_HANDOFF.md`, and the
phase spec.*
