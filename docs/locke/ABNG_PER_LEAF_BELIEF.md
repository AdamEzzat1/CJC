# Per-leaf Locke `BeliefScore` × ABNG `ood_score`: a research sketch

**Status:** design memo. Nothing in `tests/abng/` is changed by this document. It exists to anchor the next experiment.

## The question

ABNG (Phase 0.9.5) partitions input space via a learned routing tree and assigns each example to a single leaf. Each leaf carries:

- A Bayesian linear regression posterior (`BlrState`)
- A composite `ood_score = max(density, prefix_distance, epi_z)` that controls auto-abstain at inference time

Locke (v0.6+) produces an 8-dimensional `BeliefScore` per dataset slice, with each axis on `[0, 1]`. The axes that matter most for medical data — `missingness`, `constraint`, `schema`, `leakage` — are computed from typed evidence and are reproducible byte-for-byte across runs.

**Hypothesis:** the `BeliefScore` of the *training subset routed to leaf L* is a useful predictor of how often leaf L should abstain at inference time. Concretely:

> Leaves trained on data slices with low `missingness_score` or low `constraint_score` should produce higher `ood_score` more often than leaves trained on clean slices, *holding routing-codebook capacity constant*.

If true: Locke's per-leaf belief becomes a **data-side explanation** for ABNG's model-side abstain decisions. If false: data quality is decoupled from epistemic uncertainty at the leaf level, and the abstain decision is driven entirely by routing geometry. Either result is publishable.

## Why this fits the diabetes-130 workload

The dataset has known properties that vary by leaf:

| Property | What changes per leaf |
|---|---|
| `missingness_score` | `weight` is ~97% missing globally but probably much lower for the inpatient-only sub-cohorts. Leaves routed on `admission_type_id` or `discharge_disposition_id` will see very different missingness profiles. |
| `constraint_score` | The `?` sentinel in `race` / `payer_code` is concentrated in some leaves (uninsured rows) and absent in others. |
| `leakage_score` | `discharge_disposition_id ∈ {expired, hospice}` is target-leakage by construction — leaves with high incidence of those codes should have AUC against `readmitted` near 1.0 on the training side, then high abstain at inference (because few test rows have these codes). |
| `duplication_score` | `patient_nbr` recurrences cluster: certain frequent re-admitters concentrate into the same leaves. |

So we expect the per-leaf belief vector to **vary meaningfully across leaves** — which is necessary for the hypothesis to be testable at all.

## Concrete experiment

### Step 1 — capture routing assignments

In `tests/abng/dataset_a_diabetes130.rs`, after the deterministic 20K subsample is built and the routing tree is frozen, capture the per-row leaf-id assignment:

```rust
// Sketch — pseudocode, not yet implemented.
let leaf_ids: Vec<u64> = train_rows
    .iter()
    .map(|row| graph.route_to_leaf(row).leaf_id())
    .collect();
```

This is a `Vec<u64>` of length `n_train`. Already deterministic by ABNG's contract.

### Step 2 — bucket training rows by leaf

Group `train_rows` by `leaf_ids`:

```rust
use std::collections::BTreeMap;
let mut by_leaf: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
for (row_idx, &lid) in leaf_ids.iter().enumerate() {
    by_leaf.entry(lid).or_default().push(row_idx);
}
```

`BTreeMap` keeps the iteration order deterministic. Skip leaves below a minimum size (e.g. < 50 rows) — belief scores on tiny slices are unreliable (the `sample_score` axis already encodes that, but the noise on `missingness_score` is also large below ~50).

### Step 3 — Locke-validate each leaf's slice

For each leaf, build a `DataFrame` from the rows routed to that leaf and run Locke:

```rust
use cjc_locke::{validate, ValidateOptions, ValidationConfig, belief_report_from_locke};

let mut per_leaf_belief: BTreeMap<u64, BeliefScore> = BTreeMap::new();
for (leaf_id, row_indices) in &by_leaf {
    let slice_df = build_slice(&train_df, row_indices);
    let opts = ValidateOptions {
        dataset_label: format!("leaf_{}", leaf_id),
        config: ValidationConfig::default(),
        target_column: Some("readmitted".into()),
        primary_key: Some("patient_nbr".into()),
        ..Default::default()
    };
    let report = validate(&slice_df, &opts);
    let belief = belief_report_from_locke(&report);
    per_leaf_belief.insert(*leaf_id, belief.score);
}
```

This produces an 8-dimensional belief vector per leaf, each component reproducible byte-for-byte.

### Step 4 — capture `ood_score` distribution per leaf at inference time

Run the held-out 4K test rows through ABNG. For each test row, record:

- Which leaf it was routed to
- The leaf's `ood_score` at that row
- Whether the leaf abstained (`ood_score > threshold`)

Aggregate per leaf:

```rust
struct LeafInferenceStats {
    leaf_id: u64,
    n_test_rows: usize,
    abstain_rate: f64,
    median_ood: f64,
}
```

### Step 5 — the correlation table

Produce a `BTreeMap<u64, (BeliefScore, LeafInferenceStats)>` and compute, across leaves:

- Pearson r between each `BeliefScore` axis and `abstain_rate`.
- Spearman ρ as a non-parametric backup.
- A small CSV emit (`per_leaf_belief_vs_ood.csv`) for plotting.

The hypothesis predicts:

- `r(missingness_score, abstain_rate) < 0` (low missingness → low abstain)
- `r(constraint_score, abstain_rate) < 0`
- `r(leakage_score, abstain_rate) ≈ 0` or weak (leakage hurts generalisation but ABNG's `ood_score` is about routing, not feature quality)
- `r(sample_score, abstain_rate) < 0` (small leaves → noisy posterior → more abstain)

### Step 6 — pre-registration

Before running, write down the predicted signs in a `predictions.json` and only then run the experiment. ABNG's tests already enforce determinism, so the correlation values are themselves reproducible — there's no "ran it 100 times and picked the run that worked" failure mode.

## What this gives you

If even one correlation is `|r| > 0.5` with the predicted sign:

- The `BeliefScore` axes are *prospectively useful* for predicting model behavior on a leaf basis.
- ABNG could surface "this leaf's belief is suspect" as a *data-side* explanation of its abstain decision, alongside the existing `ood_score` text.
- The auto-abstain threshold could be made per-leaf, gated by belief, rather than global.

If no axis correlates:

- The hypothesis is wrong on this workload, and that's a useful finding too — it would mean ABNG's epistemic uncertainty is independent of dataset-quality issues at the leaf level, which would be evidence that the routing structure dominates.

Either way, the experiment is determinism-respecting, byte-reproducible, and falsifiable. It costs roughly:

- ~340 ms × n_leaves for the Locke validation (Locke is 17 µs/row; even with ~50 leaves of average 400 rows each, it's well under a second total).
- Negligible extra ABNG inference cost (just emitting `ood_score` per test row, which the existing code already computes internally).

## What needs to land first

Nothing blocks the experiment in code. The pieces required:

| Piece | Status |
|---|---|
| `validate(df) -> LockeReport` | Shipped (v0.5). |
| `belief_report_from_locke(report) -> BeliefReport` | Shipped (v0.5). |
| 8-axis `BeliefScore` with `schema/missingness/drift/leakage/lineage/sample/duplication/constraint` | Shipped. |
| E9060/E9061 target-leakage AUC | Shipped (v0.5). |
| E9007 sentinel detection — needed for the `?` literals in race/payer_code | Shipped (v0.4). |
| E9010 rare categories — needed for long-tail medical-specialty cohorts | **New in v0.6 (this commit).** |
| E9072 ID-like cardinality | Shipped (v0.5). |
| Per-row leaf-id capture from ABNG | **Not yet exposed.** `route_to_leaf` exists internally but doesn't return a public leaf id. Needs a small API addition in `cjc-abng::graph`. |
| `build_slice(train_df, row_indices)` helper | **Not yet implemented.** A 20-line `cjc-data` helper. |

So the only code work blocking the experiment is:

1. Expose `leaf_id` from ABNG's `route_to_leaf`.
2. Add a `DataFrame::take_rows(&[usize]) -> DataFrame` to `cjc-data` (probably already exists via `lazy::filter_indices` — would need to check).

Both are tractable and unrelated to Locke's roadmap.

## Out of scope for the sketch

- Whether the belief-axis weights should be re-tuned per leaf. Default `BeliefWeights` is uniform; per-leaf weights are a follow-on question.
- Whether the experiment generalises beyond diabetes-130. The five seeded properties of the dataset (NaN concentration, sentinel `?`, target-leakage in `discharge_disposition_id`, ID-like cardinality on `encounter_id` / `patient_nbr`, duplicate `patient_nbr`) are common to medical datasets but not universal.
- The causal direction. Belief-quality and abstain are *correlated*; whether one *causes* the other is a separate question requiring an intervention experiment.

## Open questions for the next session

- Is there an existing `DataFrame::take_rows` or do we add one? (Affects scope of "Step 2" above.)
- What is the right Pearson-r threshold to call the hypothesis "supported"? Suggest `|r| ≥ 0.4` with N ≥ 30 leaves as a minimum.
- Should the per-leaf experiment be a `#[test] #[ignore]` integration test (run with `cargo test -- --ignored`) or a separate benchmark crate?

These are decisions for when we move from sketch to implementation.
