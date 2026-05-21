# ABNG Phase 0.9.5 — Session Resume / Status

**Date:** 2026-05-17
**Branch:** `claude/abng-phase-0-9-5`
**Worktree:** `C:\Users\adame\CJC\.claude\worktrees\silly-brahmagupta-107016`
**Design authority:** [`PHASE_0_9_5_HANDOFF.md`](PHASE_0_9_5_HANDOFF.md) — the
locked design + 9-commit plan. *This* note is the current-progress
bridge against it, written for a fresh-session handoff.

---

## Status: COMMIT 4 of 9 complete

Phase 0.9.5 = categorical medical scaling. The deterministic
categorical subsystem is being built commit-by-commit per the design
doc's §7 sequence. COMMIT 1 through COMMIT 4 are done, tested, committed.

### Commits on `claude/abng-phase-0-9-5`

```
7029e2f  abng(0.9.5) COMMIT 4: CategoricalTransform -- raw rows -> (x, phi, y)
8dac3bd  abng(0.9.5) COMMIT 3: proptest properties + bolero fuzz targets
cdcdc58  abng(0.9.5) COMMIT 2: encoding modes + schema snapshot
69dd2ba  abng(0.9.5) COMMIT 1: deterministic CategoryDictionary
15d2c4b  abng(0.9.5) design: categorical medical scaling handoff
c2339e2  abng(0.10) R4 ...   <- branch base (claude/abng-phase-0-9)
```

### What's built

`crates/cjc-abng/src/categorical.rs` — **58 unit tests, all green**
(`cjc-abng` lib suite 378/378, zero regressions). Property + fuzz
coverage lives in `tests/abng/categorical_proptest.rs` (11 proptest
properties) and `tests/abng/categorical_fuzz.rs` (4 bolero targets).

* **COMMIT 1** — `CategoryDictionary` + `CategoryDictionaryBuilder`;
  reserved codes `MISSING=0 / UNKNOWN=1 / RARE=2`; real categories
  coded by `(-count, label)`; `RarePolicy` (folds by `min_count` 30 /
  `min_frac` 0.001, either floor); `canonical_bytes` / `vocab_hash`.
  Built train-split-only; immutable-by-construction freeze.
* **COMMIT 2** — `OneHotEncoder` (the `phi` encoding + `max_real`
  width cap); `route_bucket` (the `x` encoding + route-explosion
  clamp); `SchemaSnapshot` (the provenance hash bundle) +
  `hash_vocabularies` + `FEATURE_TRANSFORM_VERSION`.
* **COMMIT 3** — proptest + bolero coverage of COMMIT 1-2. Two new
  files under `tests/abng/`: `categorical_proptest.rs` (9 properties
  — the §3.7 contracts: row-order invariance, train-only, unknown
  safety, rare-fold determinism; plus the §3.4 fold rule and the
  encoder / route / snapshot surface) and `categorical_fuzz.rs` (4
  bolero targets — malformed CSV, huge/Unicode labels, mixed columns,
  missing target). Wired into `tests/abng/mod.rs`.
* **COMMIT 4** — `CategoricalTransform`, the capstone preprocessing
  layer. `ColumnRole` / `Schema` (per-column roles; the `*PhiOnly`
  variants are the §4 route-explosion hard guard); `Standardizer`
  (train-only z-score, sorted-then-Kahan so it is row-order
  invariant); `mutual_information`; `TransformConfig` /
  `TransformError`. `fit()` builds per-column dictionaries +
  standardizers, selects the top-K routing features by MI on their
  routing-bucket representation, and assembles the `SchemaSnapshot`.
  `transform()` emits `(x, phi, y)`. +31 tests (29 unit + 2 proptest
  properties — fit row-order invariance + transform output
  well-formedness).

---

## Next: COMMIT 5 onward

Per [`PHASE_0_9_5_HANDOFF.md`](PHASE_0_9_5_HANDOFF.md) §7 (COMMIT 1-4
done):

* **COMMIT 5-6** — benchmark harnesses for Dataset A (Diabetes-130)
  and B (BRFSS), preserving the Phase 0.9 root+leaf ensemble / audit
  chain / calibration architecture. COMMIT 5 also adds the CSV reader
  that feeds `CategoricalTransform` (`fit` takes `&[Vec<String>]`,
  `transform` takes `&[String]`) and wires the codebook from
  `route_bins()` / `n_routing_features()` and the leaf head from
  `phi_width()`.
* **COMMIT 7** — seed sweeps (15 A / ≥5 B) + determinism double-run.
* **COMMIT 8** — artifacts: `bench_results/phase_0_9_5_medical_scaling/`
  (CSVs, SVGs, `summary.md`, vocab report, `chain_heads.txt`, README).
* **COMMIT 9** — verification loop + close-out report +
  `known_solution_context.md`.

---

## Datasets

Both acquired, in `tests/data/`, **untracked** (35 MB total —
`.gitignore`'d, per the design doc's "raw data is not committed"
decision; reproduction is via the URLs + the `raw_dataset_hash`):

* `tests/data/diabetes_130/diabetic_data.csv` — Dataset A, 101,766
  rows × 50 cols (high-cardinality nominal categoricals).
* `tests/data/cdc_brfss.csv` — Dataset B, 253,680 rows × 23 cols
  (low-cardinality binary/ordinal).

Re-fetch if the worktree copies are gone (run in `tests/data/`):

```
curl -sSL -o diabetes_130.zip \
  "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
unzip diabetes_130.zip -d diabetes_130
curl -sSL -o cdc_brfss.csv \
  "https://archive.ics.uci.edu/static/public/891/data.csv"
```

---

## Build / test

* `cargo test -p cjc-abng --lib categorical` — the categorical
  subsystem's unit tests (58, dev profile — fast, seconds).
* `cargo test -p cjc-abng --lib` — full `cjc-abng` lib suite (378).
* `cargo test --test abng categorical` — the proptest + bolero
  property suites (15: 11 proptest + 4 bolero).
* Phase 0.9 regression: `cargo test --test abng` — note that
  Phase 0.10's R4 set `lto = true`, so a *release* build of the
  workspace is ~14 min. The determinism contract (28 SHA-256 canaries,
  Wisconsin BC baseline 0.9519) must stay green. **Known flake:** the
  `per_thread_arena_tests::concurrent_training_scales_better_than_serial`
  wall-clock scaling test fails intermittently under the full parallel
  harness (its spawned threads oversubscribe the test runner's pool);
  it passes reliably in isolation. Re-run it alone to confirm rather
  than treating a full-suite failure there as a regression.

## §9 decisions — resolved / defaults

* `phi` encoding: **one-hot** (user-confirmed; effect coding is a
  Phase 1.0 experiment — see the Phase 1.0 draft doc).
* Branch: **`claude/abng-phase-0-9-5`** — confirmed.
* Schema-snapshot audit event: keep it **harness-side** in
  `chain_heads.txt`, no v14 wire-format touch (recommended default).
* COMMIT 4 starting defaults: `K = 4` routing features,
  `route_bins = 4`, `RarePolicy` 30 / 0.001, `max_real = 32` for wide
  one-hot. Tune against the determinism gate, not accuracy.

---

## Loose ends (not Phase 0.9.5 blockers)

* Untracked carry-overs in the worktree: `docs/abng/PHASE_0_10_HANDOFF.md`
  + `docs/abng/PHASE_1_0_FALLBACK_DEFAULT_DRAFT.md` (Phase 0.10
  planning docs — commit on this branch or move them);
  `bench_results/phase_0_8_demos/*.svg` show as modified (test-run
  noise — `git checkout -- bench_results/phase_0_8_demos/` discards).
* `claude/abng-phase-0-9` (Phase 0.9 close-out + Phase 0.10 Tracks
  S/Q1/R4) and `claude/abng-phase-0-9-5` are **local-only** — not
  pushed to origin, no PRs.
* A `cjc-data` D-HARHT consolidation task was spawned earlier (three
  D-HARHT variants coexist) — a separate, independent work item.

---

## Recommended next-session prompt

```
Continue ABNG Phase 0.9.5 (categorical medical scaling).

Branch: claude/abng-phase-0-9-5, worktree
C:\Users\adame\CJC\.claude\worktrees\silly-brahmagupta-107016.
COMMIT 1-4 are done; pick up at COMMIT 5.

Read first:
  docs/abng/PHASE_0_9_5_STATUS.md   -- this resume note
  docs/abng/PHASE_0_9_5_HANDOFF.md  -- the locked design + 9-commit plan

The categorical subsystem is complete in
crates/cjc-abng/src/categorical.rs (CategoryDictionary, OneHotEncoder,
route_bucket, SchemaSnapshot, Standardizer, CategoricalTransform; 58
unit tests + 11 proptest + 4 bolero, cjc-abng lib 378/378). Next is
COMMIT 5 -- the Dataset-A (Diabetes-130) benchmark harness: CSV reader,
stratified split, fit CategoricalTransform, train through the root+leaf
ensemble, metrics, per-leaf report, ablation, + the 20K sub-sample run.

Build/test: cargo test -p cjc-abng --lib categorical
Datasets: in tests/data/ (untracked); re-fetch URLs in the status note.

Work commit-by-commit per the design doc's §7; verify tests green
before each commit; preserve the determinism contract throughout.
```

---

*Resume note maintained for the Phase 0.9.5 build. Sits alongside
`PHASE_0_9_5_HANDOFF.md` (the design + plan). When Phase 0.9.5 closes
out, fold the final numbers into a close-out report.*
