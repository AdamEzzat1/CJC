# ABNG Phase 0.10 — Diabetes-130 Accuracy Push Handoff

**Date stamped:** 2026-05-28
**Branch suggestion:** continue on `claude/relaxed-hoover-44ac41` or branch off `master` at `ee8e80a` (Locke v0.7 part 1 + per-leaf experiment + diabetes-130 fixture all landed).
**Prior phase:** Phase 0.9.5 result-path performance + categorical subsystem (last handoff: `docs/abng/PHASE_0_9_5_HANDOFF_V3.md`).

---

## 0. The ask

> **Improve ABNG's accuracy on UCI Diabetes-130 as much as possible while keeping every other metric high.** Use Locke (v0.7 part 1) wherever it fits — pre-training data hygiene, per-leaf regularization, drift monitoring, feature pruning. Use the same dataset as the published blog post (`adamezzat1.github.io/blog/posts/abng-diabetes-readmission/`).

The blog post is the public benchmark we are trying to beat. The handoff brief is "beat it without losing the things that made the post defensible" — determinism, audit-chain integrity, leakage-free evaluation protocol, calibration honesty.

---

## 1. Baseline numbers (the bar to beat)

From the published blog (`abng-diabetes-readmission/index.qmd`, §4.2, seed 42, 20K stratified sub-sample, n_test = 3,002):

| Metric | Raw | **Calibrated (Platt)** | Base-rate baseline |
|---|---|---|---|
| AUC | 0.611 | **0.611** | 0.500 |
| Brier | 0.100 | **0.098** | 0.099 |
| Log-loss (NLL) | 0.391 | **0.344** | 0.351 |
| ECE | 0.031 | **0.010** | 0 |
| Balanced accuracy | — | **~0.58** | 0.50 |
| F1 | — | **~0.24** | 0 |
| Platt (a, b) | — | **(4.2946, −2.6183)** | — |

Tuning sweep (blog §5): **`K_ROUTING = 2` wins** (16 leaves, ~875 rows each) vs the harness default `K_ROUTING = 3` (64 leaves, ~219 rows each). Stronger BLR prior also helps monotonically.

The published strong-model band for this task is **AUC 0.64–0.68**. The honest stretch goal is to get into the low end of that band (≥ 0.64) without sacrificing the calibrated Brier / NLL / ECE numbers. Any AUC improvement that breaks calibration is *not* an improvement under this brief.

---

## 2. What is already in place

The infrastructure is mostly there. The harness lives at `tests/abng/dataset_a_diabetes130.rs` (872 lines, 11 tests, 4 of them `#[ignore]`'d on dataset presence). Test functions:

| Test | Status | Purpose |
|---|---|---|
| `diabetes130_dataset_shape` | always-run | CSV row/col count assertions |
| `diabetes130_schema_matches_csv_header` | always-run | 50-column schema sanity |
| `diabetes130_target_is_imbalanced_minority` | always-run | confirms ~11% positive rate |
| `diabetes130_stratified_split_preserves_class_ratio` | always-run | split sanity |
| `diabetes130_subsample_split_is_deterministic` | always-run | RNG determinism |
| `diabetes130_transform_phi_width_under_ceiling` | always-run | feature-explosion guard |
| `diabetes130_subsample_trial` | `#[ignore]` | the 20K trial — full forward path |
| `diabetes130_trial_is_deterministic` | `#[ignore]` | the 6K determinism double-run |
| `diabetes130_full_run` | `#[ignore]` | the 101K full-dataset run |

Constants in the harness file (the dials to tune):

```rust
const ROUTE_BINS: u8 = 4;
const K_ROUTING: usize = 3;            // BLOG SAYS K=2 IS BETTER — leave or change?
const MAX_REAL: u32 = 8;
const PHI_CEILING: usize = 512;
const SUBSAMPLE_ROWS: usize = 20_000;
const DETERMINISM_PROBE_ROWS: usize = 6_000;
const BLR_PRIOR_PRECISION: f64 = 0.1;  // BLOG SAYS STRONGER IS BETTER
const BLR_PRIOR_A: f64 = 1.0;
const BLR_PRIOR_B: f64 = 0.5;
const ECE_N_BINS: usize = 10;
const DECISION_THRESHOLD: f64 = 0.5;   // BLOG TUNED THIS ON VALIDATION
const TRIAL_SEED: u64 = 42;
```

**Note that `K_ROUTING = 3` in the source contradicts the blog's "K = 2 won decisively" finding.** Either the blog used a swept K and never landed the winning value back in the source, or the source has since drifted. Step 1 below resolves this.

Locke integration (Phase 0.10 starting point):

| Asset | Path | What it gives you |
|---|---|---|
| Per-leaf belief experiment (synthetic fixture) | `tests/abng/per_leaf_belief.rs` | 4 tests, all passing. Confirms the pipeline works. |
| Per-leaf belief experiment (real dataset) | `tests/abng/per_leaf_belief_diabetes130.rs` | 3 `#[ignore]` tests; runs with `cargo test --test abng per_leaf_belief_diabetes130 -- --ignored`. Emits `target/diabetes_per_leaf_belief.csv`. |
| Locke v0.7 part 1 algebra | `crates/cjc-locke/src/algebra.rs` | `compose`, `compose_many_arithmetic`, `compose_weighted` for cross-leaf belief aggregation. |
| Locke validate / belief / drift / leakage / shape / PII / categorical | `crates/cjc-locke/src/*.rs` | 268 lib tests passing. All `cjcl locke <subcommand>` CLI subcommands ship. |
| Multi-class target-leakage AUC (E9063) | `crates/cjc-locke/src/leakage.rs` | **Will fire on diabetes-130's 3-class `readmitted` target.** Use this — see §4.B below. |
| Design memo | `docs/locke/ABNG_PER_LEAF_BELIEF.md` | Directional hypothesis: dirty-leaf belief < clean-leaf belief. Held on synthetic. Unverified on real data. |

---

## 3. The accuracy levers — ranked

Each lever has a confidence rating (how likely it is to help) and a determinism risk (whether it threatens the canaries / audit chain).

| # | Lever | Confidence | Determinism risk | Effort |
|---|---|---|---|---|
| 1 | Land `K_ROUTING = 2` + stronger BLR prior (already-swept config) | High | None (config knobs only) | Low |
| 2 | Use Locke to drop target-leakage features pre-training | High | None (pre-training) | Low |
| 3 | MLP leaf heads in place of linear BLR | High (lifts ranking ceiling) | Medium (new audit kinds?) | High |
| 4 | Full 101,766-row run at tuned config | Medium-High | None | Low |
| 5 | Locke-belief-weighted per-leaf BLR prior | Medium (novel) | Low (per-leaf prior already supported?) | Medium |
| 6 | Multi-seed variance for honest mean ± spread | (not an accuracy lever) | None | Medium |
| 7 | Better calibration (isotonic instead of Platt) | Low (Platt already at ECE 0.010) | None | Low |

The blog's §9 explicitly named #3, #4, #6 as pending. The brief here adds #2 and #5 because Locke didn't exist when the blog was published.

---

## 4. Concrete experiment plan (priority order)

### A — Land the tuned config (1 hour, low risk)

Resolve the source-vs-blog drift on `K_ROUTING` and `BLR_PRIOR_PRECISION`.

1. Verify the blog's claim by running the validation sweep yourself once: `K ∈ {2, 3, 4}` × `prior ∈ {0.05, 0.1, 0.5}` on the **validation** split of the 20K sub-sample. Should reproduce the blog's "K = 2 + stronger prior" win.
2. Update the constants:
   ```rust
   const K_ROUTING: usize = 2;
   const BLR_PRIOR_PRECISION: f64 = 0.5;  // tune by sweep
   ```
3. The 28 SHA-256 canaries should remain unmoved — these are config knobs ABNG already exposes, not new mechanisms. Confirm with `cargo test --test abng --release` (no `--ignored`).
4. Re-run `diabetes130_subsample_trial -- --ignored` and record raw + calibrated metrics.

**Expected outcome:** matches the blog's AUC 0.611 / calibrated metrics. If it doesn't, the source has drifted further than just the two constants — investigate the categorical transform.

### B — Locke pre-training feature audit (2 hours, high leverage)

This is the new lever. Before any model training, run Locke on the training split and use the findings to prune / transform features.

```bash
cargo build --release -p cjc-cli
./target/release/cjcl.exe locke validate \
    tests/data/diabetes_130/diabetic_data.csv \
    --target readmitted \
    --primary-key patient_nbr \
    --save-json bench_results/diabetes_locke_baseline.json \
    --html bench_results/diabetes_locke_baseline.html
```

Expected findings on diabetes-130 (predicted from the data's known properties):

| Code | Severity | Likely column(s) | Action for training |
|---|---|---|---|
| **E9063** (multi-class leakage, v0.6.3) | Error/Warning | `discharge_disposition_id` | **Drop or filter** codes 11/13/14/19/20/21 (death/hospice) — these are leakage by construction on a `readmitted` target. |
| **E9007** sentinel | Info | `race`, `payer_code`, `medical_specialty`, `weight` | Already handled by Locke's `?` sentinel default; confirm transform's `missing_markers` includes both `?` and empty. |
| **E9072** ID-like | Notice | `encounter_id`, `patient_nbr` | Should already be `Ignore` in the schema — verify. |
| **E9023** label-encoding risk (v0.6 batch 2) | Notice | `admission_type_id`, `discharge_disposition_id`, `admission_source_id` | These are nominal codes 1..N; the schema marks them `Categorical` — confirm one-hot is applied, NOT ordinal. |
| **E9015** high-cardinality | Notice | `diag_1`, `diag_2`, `diag_3` (700+ ICD-9 codes each) | Already `CategoricalPhiOnly` so they don't route — confirm. Consider hashing-trick or 3-digit ICD-9 prefix grouping. |
| **E9018**/**E9019** drift between train/test | varies | numeric + categorical | Run `cjcl locke drift train.csv test.csv` after the stratified split — should show low drift since stratified, but lock it in. |
| **E9070** conditional missingness | Notice | `weight` ↔ `payer_code`? | Possible — if so, joint imputation matters. |

**Concrete next actions from this:**

1. Whitelist the `discharge_disposition_id` codes that are *not* discharge-to-death / hospice. Add an explicit filter step in the harness before transform.fit().
2. Verify `weight` is being dropped or imputed (97% missing — it's noise).
3. Verify the medication columns (23 of them) are marked `CategoricalPhiOnly` so they reach phi but don't blow up routing.
4. Add an `E9063 == 0` assertion to the always-run tests so future schema changes that re-introduce leakage fail loudly.

**The deliverable:** a pruned-feature variant of the harness that demonstrates a non-trivial AUC bump (predicted: ~0.62–0.63) purely from removing leakage. **This is the single best Locke ROI for the accuracy push.**

### C — MLP leaf heads (1–2 sessions, biggest single accuracy lever)

The blog explicitly identifies the linear BLR as the *ranking ceiling*: "calibration is a monotonic transform, so it cannot change the AUC". To break 0.611 you need a nonlinear head.

ABNG already has a `pinn` module for MLP infrastructure. The clean path:

1. Add `LeafHeadKind::Mlp { hidden: usize }` (alongside the existing `LeafHeadKind::Blr`).
2. Per-leaf MLP: 2-layer ReLU, hidden width 16 or 32 (tunable). Train on the leaf's rows with Adam (the `adam_step` builtin already in `cjc-runtime`).
3. **The audit chain needs a new event kind** for "MlpLeafUpdate" because the BLR closed-form update is replaced by N gradient steps. This is the determinism risk: every new audit kind needs a SHA-256 canary lock. Plan for 1-2 new canaries.
4. Train the same way, calibrate with Platt the same way, evaluate the same way.

**Predicted outcome:** AUC 0.62–0.64 (lifting the ceiling); Brier / NLL / ECE should stay competitive after re-calibration. If AUC moves to 0.64+ but calibration degrades, isotonic regression instead of Platt is the next move.

### D — Locke-belief-weighted per-leaf prior (novel, exploratory)

Run `tests/abng/per_leaf_belief_diabetes130.rs` to get per-leaf BeliefScores. Then for each leaf:

- High belief (clean data) → lighter BLR prior (let the leaf learn).
- Low belief (dirty data) → stronger BLR prior (regularize the leaf hard).

The mapping is e.g. `prior_precision_leaf = BLR_PRIOR_PRECISION * (2.0 − overall_belief)`. So a leaf with overall belief 1.0 gets prior 0.5; a leaf with belief 0.5 gets prior 0.75.

This is the directional hypothesis from `docs/locke/ABNG_PER_LEAF_BELIEF.md` operationalized. Run the synthetic fixture in `tests/abng/per_leaf_belief.rs` first to confirm the per-leaf belief differentiates dirty from clean — it does on synthetic data. Then port the same logic into `build_graph` so the prior is per-leaf at graph-construction time.

**Predicted outcome:** small but real AUC bump (~0.005–0.015), bigger calibration improvement (per-leaf regularization makes per-leaf probabilities better-calibrated).

This is also the experiment that converts the design memo's hypothesis from "supported on synthetic fixture" to "validated on real medical data" — which is publishable independent of the accuracy result.

### E — Full 101,766-row run at tuned config (1 hour, deferred until A/B/C have landed)

Once A + B (and ideally C) are in place, re-run `diabetes130_full_run -- --ignored` for the headline number. The blog explicitly named this as future work. Expect AUC 0.62 or higher if the tuned config + leakage prune both stack.

### F — Multi-seed variance (1 hour, after E)

Run the full pipeline at 10 seeds. Report mean ± spread. This is the variance honesty the blog called out as pending.

---

## 5. Constraints / non-goals

Things to **not** break in pursuit of accuracy:

1. **Determinism contract.** The 28 SHA-256 canaries are the load-bearing claim. Any change to a core update rule (the BLR conjugate update, the descend walk, the codebook encode) needs new canary locks. Config knobs (K_ROUTING, BLR_PRIOR_PRECISION, DECISION_THRESHOLD) are safe to change.
2. **Audit-chain integrity.** Every training step appends to the SHA-256 chain; the chain verifies end-to-end. New `AuditKind` variants are allowed (e.g. `MlpLeafUpdate`) but must be canary-locked.
3. **Leakage-free evaluation protocol.** The 70/15/20 train/val/test split with all tuning on validation, single test touch — this is non-negotiable. The blog's defensibility rests on it.
4. **Calibration metrics must stay below baseline.** Brier ≤ 0.099, NLL ≤ 0.351, ECE ≤ 0.020 after calibration. An AUC jump that breaks these is not an improvement.
5. **Reproducibility.** Two runs must produce bit-identical metrics, as the blog §7 demonstrates. `cjcl locke verify --runs 2` on the saved Locke audit-bundle should also stay green.

Non-goals (explicit):
- Don't reach for non-architecture rewrites without a measured reason. The blog's lesson — "config + post-processing beat surgery" — applies.
- Don't use the test set for tuning. Ever. The blog made the point explicitly.
- Don't headline raw accuracy (89% by predicting "no readmission" for everyone). Headline AUC, balanced accuracy, calibrated Brier/NLL/ECE.

---

## 6. Where things live

| Asset | Path |
|---|---|
| Diabetes-130 harness | `tests/abng/dataset_a_diabetes130.rs` (872 LOC) |
| Locke per-leaf experiment (synthetic, passing) | `tests/abng/per_leaf_belief.rs` |
| Locke per-leaf experiment (real, ignored) | `tests/abng/per_leaf_belief_diabetes130.rs` |
| Locke algebra (v0.7 part 1) | `crates/cjc-locke/src/algebra.rs` |
| Multi-class leakage detector | `crates/cjc-locke/src/leakage.rs` |
| Locke validate CLI | `crates/cjc-cli/src/commands/locke.rs` |
| Phase 0.9.5 prior handoff | `docs/abng/PHASE_0_9_5_HANDOFF_V3.md` |
| Design memo for per-leaf belief | `docs/locke/ABNG_PER_LEAF_BELIEF.md` |
| Published blog post | `C:/Users/adame/AdamEzzat1.github.io/blog/posts/abng-diabetes-readmission/index.qmd` |
| Dataset (untracked, gitignore'd) | `tests/data/diabetes_130/diabetic_data.csv` (+ `IDS_mapping.csv`) |
| Recent commits | `master`/`claude/relaxed-hoover-44ac41`: `ee8e80a` (diabetes per-leaf), `ce59000` (algebra), `67c5128` (Locke v0.6.3), `1374eda` (per-leaf synthetic), `69fe6cb` (Locke v0.6 batch 2), `e0a3208` (Locke v0.6 batch 1), `2953105` (Locke v0.6 base), `b1f918b` (Locke v0.1.9 release tag) |

---

## 7. Pre-flight checklist (run before changing anything)

```bash
# 1. Confirm the workspace builds clean.
cargo build --workspace --release

# 2. Confirm the always-run ABNG suite is green.
cargo test --test abng --release        # → expect 629/629 passing

# 3. Confirm Locke is green.
cargo test -p cjc-locke --release       # → expect 268 lib tests passing
cargo test --test locke --release       # → expect 176 integration tests passing

# 4. Confirm the dataset is present (otherwise the gated tests can't run).
ls tests/data/diabetes_130/diabetic_data.csv

# 5. Confirm the diabetes harness still compiles and skips cleanly.
cargo test --test abng --release diabetes130 -- --include-ignored --skip diabetes130_full_run --skip diabetes130_subsample_trial --skip diabetes130_trial_is_deterministic 2>&1 | tail -5

# 6. Run the synthetic per-leaf experiment to confirm the Locke<>ABNG plumbing is alive.
cargo test --test abng per_leaf_belief --release        # → 4 passing

# 7. Run the diabetes-130 per-leaf experiment for the baseline per-leaf belief CSV.
cargo test --test abng per_leaf_belief_diabetes130 --release -- --ignored --nocapture
cat target/diabetes_per_leaf_belief.csv        # → record the baseline
```

Save the output of step 7 — it's the "before" snapshot to compare against after the Locke-driven feature audit in §4.B.

---

## 8. Recording results

Use the same defensibility standard as the blog. For each experiment variant, record:

- Raw + calibrated metrics: AUC, balanced accuracy, F1, Brier, NLL, ECE.
- Platt (a, b) parameters.
- ABNG chain head + Merkle root hex (proof of determinism).
- Wall-clock + n_train + n_test.
- Per-leaf populated / dead leaf count.
- Per-leaf Locke belief vector (CSV).
- Which Locke findings drove which schema decisions.

Emit to `bench_results/diabetes_phase_0_10_<variant>/`. The blog post can then be updated with the new headline if the result is publishable.

---

## 9. Open questions for the next session

- Does `K = 2` reproducibly beat `K = 3` on this machine? The blog says yes; the source still defaults to 3. Reproduce before changing.
- How invasive is the MLP-leaf-head path? Does ABNG's existing `pinn` module suffice, or does this need a new module?
- For per-leaf belief-weighted regularization, does ABNG expose a per-leaf-set-prior API, or does the prior have to be set graph-wide at codebook freeze time?
- For the leakage filter on `discharge_disposition_id`, is dropping rows OK (loses ~5% of data) or should they be moved to a "censored" class?
- Does isotonic calibration on raw predictions beat Platt? Only relevant if MLP heads push the calibration story off the current sweet spot.

---

## 10. Recommended pickup order

1. **Pre-flight checklist (§7)** — confirms the baseline.
2. **§4.A — Land the tuned config.** Verify K=2 wins, update constants. Reproduce the blog's 0.611 AUC.
3. **§4.B — Locke pre-training audit.** Run `cjcl locke validate` on the CSV. Read the JSON. Drop leakage codes. **This is the highest expected ROI for accuracy.**
4. **§4.E — Full 101,766-row run at tuned + Locke-pruned config.** Get the headline number for the pruned-features variant.
5. **§4.C — MLP leaf heads.** The ranking-ceiling lever. New audit kind + new canary lock; this is the multi-session work.
6. **§4.D — Locke-belief-weighted prior.** Run the diabetes-130 per-leaf belief, port into `build_graph`, re-trial.
7. **§4.F — Multi-seed variance.** For the final headline.

If only **two** items fit in the next session, do **§4.A + §4.B**. They are the cheapest, highest-confidence, lowest-risk steps and demonstrate the Locke × ABNG integration concretely. The blog's headline can be updated after §4.E.

---

## 11. Honest stretch goal

- **Floor:** reproduce the blog's 0.611 AUC + calibrated metrics. Confirms the harness still works.
- **Target:** AUC 0.62 + Brier ≤ 0.098 + NLL ≤ 0.344 + ECE ≤ 0.010 from Locke-driven feature pruning alone (§4.B).
- **Stretch:** AUC ≥ 0.64 from MLP leaf heads (§4.C) — into the "strong published model" band.
- **Bonus:** the per-leaf belief × abstain correlation from §4.D becomes the second blog post.

Any of these is a publishable result. The floor is defensible reproduction; the target is a real Locke contribution; the stretch is an honest architectural advance.

Good luck. The infrastructure is in place; the levers are identified; the data is the same as the blog. The accuracy push is teed up.
