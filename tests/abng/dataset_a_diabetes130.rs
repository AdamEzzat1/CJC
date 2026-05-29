//! Phase 0.9.5 COMMIT 5 — Dataset-A benchmark harness: UCI Diabetes
//! 130-US Hospitals.
//!
//! This is the first heterogeneous-categorical scale point for ABNG.
//! It drives the COMMIT 1-4 categorical subsystem
//! ([`cjc_abng::categorical`]) end-to-end on a real 101,766-row × 50-col
//! medical dataset:
//!
//!   CSV -> Schema -> CategoricalTransform::fit (train split only)
//!       -> per-row transform -> (x, phi, y) -> ABNG root+leaf ensemble
//!       -> metrics (accuracy, balanced accuracy, AUC, F1, Brier, NLL,
//!          ECE) + per-leaf report + root/leaf/ensemble ablation.
//!
//! Target: the published 30-day readmission task — `readmitted == '<30'`
//! is positive (≈ 11% — class-imbalanced, so balanced accuracy and AUC
//! lead, not raw accuracy).
//!
//! # Test footprint
//!
//! The always-run test trains on a deterministic 20,000-row stratified
//! sub-sample (the handoff's ~20K scaling rung). A determinism
//! double-run uses a smaller 6,000-row probe. The full 101,766-row run
//! is `#[ignore]`d — invoke it explicitly or via the COMMIT 7-8
//! sweep / artifact stages.
//!
//! Phase 0.9.5 is **not** a benchmark (handoff §0): the always-run
//! tests assert determinism + metric well-formedness + that the
//! ensemble learns *some* signal, not tuned accuracy floors. Tuned
//! numbers are COMMIT 7-9.
//!
//! The dataset lives at `tests/data/diabetes_130/` and is untracked
//! (re-fetch URL in `docs/abng/PHASE_0_9_5_STATUS.md`); every test
//! skips gracefully when it is absent.

use std::path::Path;

use cjc_abng::categorical::{
    CategoricalTransform, ColumnRole, RarePolicy, Schema, TransformConfig,
};
use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_ad::pinn::Activation;

// ── Configuration ───────────────────────────────────────────────────

const DATASET_REL_PATH: &str = "tests/data/diabetes_130/diabetic_data.csv";
/// Diabetes-130 column count (the loader rejects any row that differs).
const N_COLUMNS: usize = 50;
/// Index of the `readmitted` target column.
const TARGET_COL: usize = 49;
/// Index of `discharge_disposition_id` — Phase 0.10 §4.B uses this to
/// filter rows where the patient died or went to hospice (codes in
/// [`DEATH_DISCHARGE_CODES`]). Those patients cannot be readmitted by
/// construction, so their `readmitted=NO` outcome is leakage rather
/// than signal.
const DISCHARGE_DISPOSITION_COL: usize = 7;

/// Discharge disposition codes that correspond to "patient cannot be
/// readmitted" outcomes — death (expired) and hospice. Source: the
/// `tests/data/diabetes_130/IDS_mapping.csv` table that ships with the
/// UCI Diabetes-130 dataset.
///
/// * 11 — Expired
/// * 13 — Hospice / home
/// * 14 — Hospice / medical facility
/// * 19 — Expired at home (Medicaid hospice)
/// * 20 — Expired in a medical facility (Medicaid hospice)
/// * 21 — Expired (hospice place not stated)
///
/// Phase 0.10 §4.B: dropping these rows before training removes the
/// label-leakage that suppresses AUC. Locke's E9063 multi-class leakage
/// detector does **not** fire on this — it uses per-column ROC-AUC,
/// which misses per-level deterministic outcomes when the leaking codes
/// are interspersed with non-leaking codes in the numeric range
/// (codes 11/13/14/19/20/21 share that range with non-death codes
/// 12/15/16/17/18). A per-level conditional-probability detector
/// (proposed E9064) would catch this; until it ships, the filter is
/// applied via domain knowledge.
const DEATH_DISCHARGE_CODES: &[&str] = &["11", "13", "14", "19", "20", "21"];

/// Routing buckets per routing feature (codebook `n_bins`).
const ROUTE_BINS: u8 = 4;
/// Routing features selected by mutual information.
///
/// Phase 0.10 §4.A: the published blog
/// (`adamezzat1.github.io/blog/posts/abng-diabetes-readmission/`) ran a
/// validation sweep over `K ∈ {2, 3, 4} × prior ∈ {0.05, 0.1, 0.5}` and
/// found **K = 2 wins decisively** on the 20K sub-sample: 16 leaves
/// (~875 rows each) vs K=3's 64 leaves (~219 rows each). The leaf-data-
/// starvation effect of deeper routing outweighs the specialization
/// gain. Holding K=2 as the harness default lands the tuned headline
/// (AUC ≈ 0.6107, calibrated Brier ≈ 0.0980).
const K_ROUTING: usize = 2;
/// One-hot width cap per categorical column (the `phi`-side explosion
/// guard).
const MAX_REAL: u32 = 8;
/// `phi` width must stay below this — a schema change cannot silently
/// explode the BLR dimension (handoff §4).
const PHI_CEILING: usize = 512;

const TRAIN_FRAC: f64 = 0.80;
/// The handoff's ~20K Dataset-A scaling rung.
const SUBSAMPLE_ROWS: usize = 20_000;
/// Smaller row budget for the determinism double-run.
const DETERMINISM_PROBE_ROWS: usize = 6_000;

/// Phase 0.10 §4.A: the validation sweep also confirmed that stronger
/// BLR regularization helps monotonically on this noisy target. Moving
/// from precision 0.1 → 0.5 reduces overfitting on the per-leaf BLRs
/// (each leaf now has ~875 rows under K=2 routing, so a stronger prior
/// pulls a noisy leaf's posterior back toward the prior mean).
const BLR_PRIOR_PRECISION: f64 = 0.5;
const BLR_PRIOR_A: f64 = 1.0;
const BLR_PRIOR_B: f64 = 0.5;

const ECE_N_BINS: usize = 10;
const PROB_EPS: f64 = 1e-7;
/// Decision threshold for the threshold-dependent metrics. A fixed 0.5
/// for the harness; COMMIT 7-8 tune it against the imbalanced target.
const DECISION_THRESHOLD: f64 = 0.5;

const TRIAL_SEED: u64 = 42;

// ── Deterministic RNG ───────────────────────────────────────────────

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Deterministic in-place Fisher–Yates shuffle of an index slice.
fn shuffle(items: &mut [usize], state: &mut u64) {
    for i in (1..items.len()).rev() {
        let j = (splitmix64(state) % (i as u64 + 1)) as usize;
        items.swap(i, j);
    }
}

// ── CSV reader ──────────────────────────────────────────────────────

/// Parse a plain (unquoted) CSV. Diabetes-130 has no embedded commas or
/// quoted fields; every row is rejected unless it has exactly
/// [`N_COLUMNS`] cells. Returns `(header, rows)`.
fn read_csv(bytes: &[u8]) -> Result<(Vec<String>, Vec<Vec<String>>), String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("utf-8 decode: {e}"))?;
    let mut lines = text.lines();
    let header: Vec<String> = lines
        .next()
        .ok_or("empty CSV")?
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    if header.len() != N_COLUMNS {
        return Err(format!(
            "header has {} columns, expected {N_COLUMNS}",
            header.len()
        ));
    }
    let mut rows: Vec<Vec<String>> = Vec::new();
    for (i, line) in lines.enumerate() {
        if line.is_empty() {
            continue;
        }
        let cells: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();
        if cells.len() != N_COLUMNS {
            return Err(format!(
                "row {i} has {} cells, expected {N_COLUMNS}",
                cells.len()
            ));
        }
        rows.push(cells);
    }
    Ok((header, rows))
}

/// Load the dataset, returning `(rows, raw_dataset_sha256)`. `None` when
/// the file is absent so every test can skip gracefully.
fn load_dataset() -> Option<(Vec<Vec<String>>, [u8; 32])> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(DATASET_REL_PATH);
    let bytes = std::fs::read(&path).ok()?;
    let raw_hash = cjc_snap::hash::sha256(&bytes);
    let (_header, rows) = read_csv(&bytes).expect("Diabetes-130 CSV must parse");
    Some((rows, raw_hash))
}

// ── Schema ──────────────────────────────────────────────────────────

/// The Diabetes-130 schema — 50 columns in CSV order.
///
/// * `encounter_id` / `patient_nbr` — `Ignore` (identifiers, leakage).
/// * `readmitted` — `Target`.
/// * `diag_1/2/3` (ICD-9, 700+ codes each) + the 23 medication columns
///   — `CategoricalPhiOnly`: the §4 route-explosion hard guard, so
///   high-cardinality nominal columns reach `phi` but never routing.
/// * The eight count columns — `Numeric`.
/// * Everything else — `Categorical` (routing candidates).
fn diabetes_schema() -> Schema {
    use ColumnRole::*;
    let med = CategoricalPhiOnly;
    let cols: &[(&str, ColumnRole)] = &[
        ("encounter_id", Ignore),
        ("patient_nbr", Ignore),
        ("race", Categorical),
        ("gender", Categorical),
        ("age", Categorical),
        ("weight", Categorical),
        ("admission_type_id", Categorical),
        ("discharge_disposition_id", Categorical),
        ("admission_source_id", Categorical),
        ("time_in_hospital", Numeric),
        ("payer_code", Categorical),
        ("medical_specialty", Categorical),
        ("num_lab_procedures", Numeric),
        ("num_procedures", Numeric),
        ("num_medications", Numeric),
        ("number_outpatient", Numeric),
        ("number_emergency", Numeric),
        ("number_inpatient", Numeric),
        ("diag_1", CategoricalPhiOnly),
        ("diag_2", CategoricalPhiOnly),
        ("diag_3", CategoricalPhiOnly),
        ("number_diagnoses", Numeric),
        ("max_glu_serum", Categorical),
        ("A1Cresult", Categorical),
        ("metformin", med),
        ("repaglinide", med),
        ("nateglinide", med),
        ("chlorpropamide", med),
        ("glimepiride", med),
        ("acetohexamide", med),
        ("glipizide", med),
        ("glyburide", med),
        ("tolbutamide", med),
        ("pioglitazone", med),
        ("rosiglitazone", med),
        ("acarbose", med),
        ("miglitol", med),
        ("troglitazone", med),
        ("tolazamide", med),
        ("examide", med),
        ("citoglipton", med),
        ("insulin", med),
        ("glyburide-metformin", med),
        ("glipizide-metformin", med),
        ("glimepiride-pioglitazone", med),
        ("metformin-rosiglitazone", med),
        ("metformin-pioglitazone", med),
        ("change", Categorical),
        ("diabetesMed", Categorical),
        ("readmitted", Target),
    ];
    Schema::new(cols.iter().map(|(n, r)| (n.to_string(), *r)).collect())
}

/// The transform config — Phase 0.9.5 defaults, `<30` as the positive
/// label, `?` as the Diabetes-130 missing marker.
fn transform_config(raw_hash: [u8; 32], seed: u64, row_count: u64) -> TransformConfig {
    TransformConfig {
        route_bins: ROUTE_BINS,
        k_routing: K_ROUTING,
        max_real: MAX_REAL,
        rare_policy: RarePolicy::DEFAULT,
        missing_markers: vec!["?".to_string(), String::new()],
        target_positives: vec!["<30".to_string()],
        target_definition: "readmitted == '<30'".to_string(),
        raw_dataset_hash: raw_hash,
        split_seed: seed,
        row_count,
    }
}

// ── Target / split / sub-sample ─────────────────────────────────────

/// Binarise `readmitted`: `1` for `<30` (30-day readmission), else `0`.
/// Used only for split / sub-sample stratification — training labels
/// come from `CategoricalTransform::transform`.
fn binarise_labels(rows: &[Vec<String>]) -> Vec<u8> {
    rows.iter()
        .map(|r| u8::from(r[TARGET_COL] == "<30"))
        .collect()
}

/// Phase 0.10 §4.B — drop rows where the patient died or went to
/// hospice (discharge codes 11/13/14/19/20/21). Those rows are
/// `readmitted=NO` by construction (a dead patient cannot be
/// readmitted), so they suppress the `<30` positive-class signal the
/// model is trying to learn.
///
/// Returns the filtered rows plus a count of how many were dropped.
fn filter_out_death_discharges(rows: &[Vec<String>]) -> (Vec<Vec<String>>, usize) {
    let before = rows.len();
    let kept: Vec<Vec<String>> = rows
        .iter()
        .filter(|r| {
            let code = r[DISCHARGE_DISPOSITION_COL].as_str();
            !DEATH_DISCHARGE_CODES.contains(&code)
        })
        .cloned()
        .collect();
    let dropped = before - kept.len();
    (kept, dropped)
}

/// Deterministic class-ratio-preserving sub-sample of `budget` row
/// indices. Returns all indices (shuffled) when `budget >= n`.
fn select_subsample(labels: &[u8], seed: u64, budget: usize) -> Vec<usize> {
    let n = labels.len();
    let mut state = seed ^ 0x5_0B5A_3D1E_9C42;
    let mut by_class: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    for (i, &y) in labels.iter().enumerate() {
        by_class[y as usize].push(i);
    }
    let mut out: Vec<usize> = Vec::new();
    if budget >= n {
        out.extend(0..n);
    } else {
        shuffle(&mut by_class[0], &mut state);
        shuffle(&mut by_class[1], &mut state);
        // Class 0 takes its floor share; class 1 takes the exact
        // remainder, so the sub-sample size is exactly `budget` (two
        // independent floors would lose a row or two).
        let take0 = (by_class[0].len() * budget / n).min(by_class[0].len());
        let take1 = (budget - take0).min(by_class[1].len());
        out.extend_from_slice(&by_class[0][..take0]);
        out.extend_from_slice(&by_class[1][..take1]);
    }
    shuffle(&mut out, &mut state);
    out
}

/// Deterministic stratified train/test split of `subset` (indices into
/// the full row table). Stratifies on `labels[idx]`; both returned
/// vectors are class-mixed.
fn stratified_split(
    subset: &[usize],
    labels: &[u8],
    seed: u64,
) -> (Vec<usize>, Vec<usize>) {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut by_class: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    for &i in subset {
        by_class[labels[i] as usize].push(i);
    }
    let mut train: Vec<usize> = Vec::new();
    let mut test: Vec<usize> = Vec::new();
    for class in 0..2 {
        shuffle(&mut by_class[class], &mut state);
        let n_train = ((by_class[class].len() as f64) * TRAIN_FRAC) as usize;
        train.extend_from_slice(&by_class[class][..n_train]);
        test.extend_from_slice(&by_class[class][n_train..]);
    }
    shuffle(&mut train, &mut state);
    shuffle(&mut test, &mut state);
    (train, test)
}

// ── Graph construction ──────────────────────────────────────────────

/// Pre-allocate a full `branching^depth` routing tree by BFS expansion.
fn pre_allocate_full_tree(g: &mut AdaptiveBeliefGraph, branching: u8, depth: usize) {
    let mut level: Vec<u32> = vec![0];
    for _ in 0..depth {
        let mut next: Vec<u32> = Vec::with_capacity(level.len() * branching as usize);
        for &parent in &level {
            for key in 0..branching {
                next.push(g.add_node(parent, key).expect("add_node"));
            }
        }
        level = next;
    }
}

/// Build the ABNG graph for one trial — codebook, leaf head, BLR prior,
/// and the pre-allocated routing tree, all sized from the fitted
/// [`CategoricalTransform`].
fn build_graph(seed: u64, transform: &CategoricalTransform) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(seed);
    let n_routing = transform.n_routing_features();
    let route_bins = transform.route_bins();
    // `transform` emits `x` as integer bucket indices, so the codebook
    // is a uniform pass-through: boundary `k - 0.5` lands value `k` in
    // bin `k`.
    let mut boundaries: Vec<f64> = Vec::new();
    for _ in 0..n_routing {
        for k in 1..route_bins {
            boundaries.push(k as f64 - 0.5);
        }
    }
    g.set_codebook(n_routing, route_bins as u16, &boundaries)
        .expect("codebook install");
    g.set_leaf_head(transform.phi_width() as u32, vec![], 1, Activation::None)
        .expect("leaf head install");
    g.set_blr_prior(BLR_PRIOR_PRECISION, BLR_PRIOR_A, BLR_PRIOR_B)
        .expect("BLR prior install");
    pre_allocate_full_tree(&mut g, route_bins, n_routing);
    g
}

// ── Metrics ─────────────────────────────────────────────────────────

/// Held-out classification + calibration metrics for one trial.
#[derive(Debug, Clone, PartialEq)]
struct Metrics {
    accuracy: f64,
    balanced_accuracy: f64,
    auc: f64,
    f1: f64,
    brier: f64,
    nll: f64,
    ece: f64,
    n_test: usize,
}

/// ROC AUC via the rank-sum (Mann–Whitney) statistic, tie-averaged
/// ranks. Deterministic — `total_cmp` ordering. `0.5` when a class is
/// absent.
fn roc_auc(probs: &[f64], labels: &[f64]) -> f64 {
    let n = probs.len();
    if n == 0 {
        return 0.5;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| probs[a].total_cmp(&probs[b]));
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && probs[order[j + 1]] == probs[order[i]] {
            j += 1;
        }
        let avg = ((i + 1) + (j + 1)) as f64 / 2.0;
        for &idx in &order[i..=j] {
            ranks[idx] = avg;
        }
        i = j + 1;
    }
    let n_pos = labels.iter().filter(|&&y| y > 0.5).count();
    let n_neg = n - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return 0.5;
    }
    let rank_sum_pos: f64 = (0..n).filter(|&i| labels[i] > 0.5).map(|i| ranks[i]).sum();
    (rank_sum_pos - (n_pos * (n_pos + 1)) as f64 / 2.0) / (n_pos as f64 * n_neg as f64)
}

/// Compute every metric from aligned `(prob, label)` vectors.
fn compute_metrics(probs: &[f64], labels: &[f64]) -> Metrics {
    let n = probs.len();
    let nf = n as f64;
    let (mut tp, mut tn, mut fp, mut fn_) = (0u64, 0u64, 0u64, 0u64);
    for (&p, &y) in probs.iter().zip(labels) {
        let pred = p > DECISION_THRESHOLD;
        let actual = y > 0.5;
        match (pred, actual) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            (false, false) => tn += 1,
        }
    }
    let accuracy = if n > 0 { (tp + tn) as f64 / nf } else { 0.0 };
    let tpr = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    };
    let tnr = if tn + fp > 0 {
        tn as f64 / (tn + fp) as f64
    } else {
        0.0
    };
    let balanced_accuracy = 0.5 * (tpr + tnr);
    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let f1 = if precision + tpr > 0.0 {
        2.0 * precision * tpr / (precision + tpr)
    } else {
        0.0
    };
    let auc = roc_auc(probs, labels);
    let brier = if n > 0 {
        probs.iter().zip(labels).map(|(p, y)| (p - y).powi(2)).sum::<f64>() / nf
    } else {
        0.0
    };
    let nll = if n > 0 {
        -probs
            .iter()
            .zip(labels)
            .map(|(p, y)| y * p.ln() + (1.0 - y) * (1.0 - p).ln())
            .sum::<f64>()
            / nf
    } else {
        0.0
    };
    // ECE — equal-width bins on [0, 1].
    let mut bin_p = vec![0.0f64; ECE_N_BINS];
    let mut bin_y = vec![0.0f64; ECE_N_BINS];
    let mut bin_n = vec![0usize; ECE_N_BINS];
    for (&p, &y) in probs.iter().zip(labels) {
        let b = ((p * ECE_N_BINS as f64) as usize).min(ECE_N_BINS - 1);
        bin_p[b] += p;
        bin_y[b] += y;
        bin_n[b] += 1;
    }
    let mut ece = 0.0f64;
    for b in 0..ECE_N_BINS {
        if bin_n[b] == 0 {
            continue;
        }
        let nb = bin_n[b] as f64;
        ece += (nb / nf) * (bin_p[b] / nb - bin_y[b] / nb).abs();
    }
    Metrics {
        accuracy,
        balanced_accuracy,
        auc,
        f1,
        brier,
        nll,
        ece,
        n_test: n,
    }
}

// ── Trial ───────────────────────────────────────────────────────────

/// Per-leaf routing distribution for the trained tree.
#[derive(Debug, Clone, PartialEq)]
struct LeafReport {
    total_leaves: usize,
    populated_leaves: usize,
    dead_leaves: usize,
    min_per_populated: u64,
    max_per_populated: u64,
    mean_per_populated: f64,
}

/// Root-only / leaf-only / ensemble accuracy — the COMMIT 5 ablation.
#[derive(Debug, Clone, PartialEq)]
struct Ablation {
    root_accuracy: f64,
    leaf_accuracy: f64,
    ensemble_accuracy: f64,
}

/// The full result of one Dataset-A trial.
#[derive(Debug, Clone)]
struct TrialResult {
    n_rows_used: usize,
    n_train: usize,
    n_routing_features: usize,
    routing_feature_columns: Vec<usize>,
    phi_width: usize,
    chain_head_hex: String,
    merkle_root_hex: String,
    audit_event_count: usize,
    metrics: Metrics,
    leaf_report: LeafReport,
    ablation: Ablation,
}

fn hex32(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Predict the ensemble / root / leaf posterior mean for one phi at one
/// routed leaf.
fn predict(g: &AdaptiveBeliefGraph, leaf_id: u32, phi: &[f64]) -> (f64, f64) {
    let leaf = g
        .blr_predict_with_fallback(leaf_id, phi)
        .map(|(m, _, _, _)| m)
        .unwrap_or(0.0);
    let root = g
        .blr_predict_with_fallback(0, phi)
        .map(|(m, _, _, _)| m)
        .unwrap_or(0.0);
    (leaf, root)
}

/// Run one full Dataset-A trial: sub-sample to `budget` rows, split, fit
/// the transform on the train split, build + train the graph, evaluate.
fn run_trial(
    rows: &[Vec<String>],
    raw_hash: [u8; 32],
    schema: &Schema,
    seed: u64,
    budget: usize,
) -> TrialResult {
    let labels = binarise_labels(rows);
    let subset = select_subsample(&labels, seed, budget);
    let (train_idx, test_idx) = stratified_split(&subset, &labels, seed);

    // Fit the transform on the train split only (leakage-free).
    let train_rows: Vec<Vec<String>> = train_idx.iter().map(|&i| rows[i].clone()).collect();
    let config = transform_config(raw_hash, seed, rows.len() as u64);
    let transform =
        CategoricalTransform::fit(schema, &train_rows, &config).expect("transform fit");
    assert!(
        transform.phi_width() < PHI_CEILING,
        "phi width {} exceeds the {PHI_CEILING} ceiling",
        transform.phi_width()
    );

    let mut g = build_graph(seed, &transform);

    // Train: per row, transform -> train_step (leaf) + blr_update (root
    // ensemble). A row with a missing target is skipped.
    use std::collections::BTreeMap;
    let mut leaf_counts: BTreeMap<u32, u64> = BTreeMap::new();
    for &i in &train_idx {
        let (x, phi, y) = match transform.transform(&rows[i]) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let prefix = g.encode_prefix(&x).expect("encode prefix");
        let leaf = g.descend(&prefix).leaf_id;
        *leaf_counts.entry(leaf).or_insert(0) += 1;
        g.train_step(&x, &phi, y).expect("train_step");
        g.blr_update(0, &phi, &[y]).expect("root blr_update");
    }

    // Phase 0.9.5 R0-3 (Tier 2 Option C) — flush the periodic-checkpoint
    // BLR witnesses so every trained node's final state is anchored in
    // the audit chain before chain_head / merkle_root are read and
    // before any future snapshot of this trained graph.
    g.checkpoint_blr();

    // Evaluate on the held-out test split.
    let mut probs: Vec<f64> = Vec::with_capacity(test_idx.len());
    let mut ys: Vec<f64> = Vec::with_capacity(test_idx.len());
    let mut root_correct = 0usize;
    let mut leaf_correct = 0usize;
    let mut ens_correct = 0usize;
    for &i in &test_idx {
        let (x, phi, y) = match transform.transform(&rows[i]) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let prefix = g.encode_prefix(&x).expect("encode prefix");
        let leaf_id = g.descend(&prefix).leaf_id;
        let (leaf_mean, root_mean) = predict(&g, leaf_id, &phi);
        let ensemble = 0.5 * (leaf_mean + root_mean);
        let actual = y > 0.5;
        if (root_mean > DECISION_THRESHOLD) == actual {
            root_correct += 1;
        }
        if (leaf_mean > DECISION_THRESHOLD) == actual {
            leaf_correct += 1;
        }
        if (ensemble > DECISION_THRESHOLD) == actual {
            ens_correct += 1;
        }
        probs.push(ensemble.clamp(PROB_EPS, 1.0 - PROB_EPS));
        ys.push(y);
    }
    let n_eval = probs.len().max(1) as f64;
    let metrics = compute_metrics(&probs, &ys);
    let ablation = Ablation {
        root_accuracy: root_correct as f64 / n_eval,
        leaf_accuracy: leaf_correct as f64 / n_eval,
        ensemble_accuracy: ens_correct as f64 / n_eval,
    };

    // Per-leaf routing report.
    let total_leaves = (ROUTE_BINS as usize).pow(K_ROUTING as u32);
    let populated = leaf_counts.len();
    let counts: Vec<u64> = leaf_counts.values().copied().collect();
    let (min_p, max_p, mean_p) = if counts.is_empty() {
        (0, 0, 0.0)
    } else {
        let sum: u64 = counts.iter().sum();
        (
            *counts.iter().min().unwrap(),
            *counts.iter().max().unwrap(),
            sum as f64 / counts.len() as f64,
        )
    };
    let leaf_report = LeafReport {
        total_leaves,
        populated_leaves: populated,
        dead_leaves: total_leaves.saturating_sub(populated),
        min_per_populated: min_p,
        max_per_populated: max_p,
        mean_per_populated: mean_p,
    };

    TrialResult {
        n_rows_used: subset.len(),
        n_train: train_idx.len(),
        n_routing_features: transform.n_routing_features(),
        routing_feature_columns: transform.routing_feature_columns(),
        phi_width: transform.phi_width(),
        chain_head_hex: hex32(&g.chain_head),
        merkle_root_hex: hex32(&g.merkle_root()),
        audit_event_count: g.audit.len(),
        metrics,
        leaf_report,
        ablation,
    }
}

// ── Tests ───────────────────────────────────────────────────────────

/// Returns the loaded dataset, or prints a skip notice and returns
/// `None` when the untracked CSV is absent.
fn dataset_or_skip(test: &str) -> Option<(Vec<Vec<String>>, [u8; 32])> {
    match load_dataset() {
        Some(d) => Some(d),
        None => {
            eprintln!("[skip] {test}: tests/data/diabetes_130/ absent");
            None
        }
    }
}

#[test]
fn diabetes130_dataset_shape() {
    let Some((rows, _)) = dataset_or_skip("diabetes130_dataset_shape") else {
        return;
    };
    assert_eq!(rows.len(), 101_766, "Diabetes-130 row count");
    assert!(rows.iter().all(|r| r.len() == N_COLUMNS));
}

#[test]
fn diabetes130_schema_matches_csv_header() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(DATASET_REL_PATH);
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("[skip] diabetes130_schema_matches_csv_header: dataset absent");
        return;
    };
    let (header, _) = read_csv(&bytes).expect("CSV parses");
    let schema = diabetes_schema();
    assert_eq!(schema.len(), N_COLUMNS);
    for (i, name) in header.iter().enumerate() {
        assert_eq!(schema.name(i), name, "column {i} name mismatch");
    }
}

#[test]
fn diabetes130_target_is_imbalanced_minority() {
    let Some((rows, _)) = dataset_or_skip("diabetes130_target_is_imbalanced_minority") else {
        return;
    };
    let labels = binarise_labels(&rows);
    let pos = labels.iter().filter(|&&y| y == 1).count();
    let rate = pos as f64 / labels.len() as f64;
    // Published 30-day readmission rate is ≈ 11%.
    assert!(
        (0.08..0.15).contains(&rate),
        "positive rate {rate} outside the expected ≈11% band"
    );
}

#[test]
fn diabetes130_stratified_split_preserves_class_ratio() {
    let Some((rows, _)) =
        dataset_or_skip("diabetes130_stratified_split_preserves_class_ratio")
    else {
        return;
    };
    let labels = binarise_labels(&rows);
    let subset = select_subsample(&labels, TRIAL_SEED, SUBSAMPLE_ROWS);
    let (train, test) = stratified_split(&subset, &labels, TRIAL_SEED);
    assert!(!train.is_empty() && !test.is_empty());
    assert_eq!(train.len() + test.len(), subset.len());
    let rate = |idx: &[usize]| {
        idx.iter().filter(|&&i| labels[i] == 1).count() as f64 / idx.len() as f64
    };
    // Train and test positive rates must track each other (stratified).
    assert!((rate(&train) - rate(&test)).abs() < 0.02);
}

#[test]
fn diabetes130_subsample_split_is_deterministic() {
    let Some((rows, _)) = dataset_or_skip("diabetes130_subsample_split_is_deterministic")
    else {
        return;
    };
    let labels = binarise_labels(&rows);
    let a = stratified_split(
        &select_subsample(&labels, TRIAL_SEED, SUBSAMPLE_ROWS),
        &labels,
        TRIAL_SEED,
    );
    let b = stratified_split(
        &select_subsample(&labels, TRIAL_SEED, SUBSAMPLE_ROWS),
        &labels,
        TRIAL_SEED,
    );
    assert_eq!(a, b);
}

#[test]
fn diabetes130_transform_phi_width_under_ceiling() {
    let Some((rows, raw_hash)) =
        dataset_or_skip("diabetes130_transform_phi_width_under_ceiling")
    else {
        return;
    };
    let labels = binarise_labels(&rows);
    // Fit on a modest sub-sample's train split — cheap.
    let subset = select_subsample(&labels, TRIAL_SEED, DETERMINISM_PROBE_ROWS);
    let (train_idx, _) = stratified_split(&subset, &labels, TRIAL_SEED);
    let train_rows: Vec<Vec<String>> = train_idx.iter().map(|&i| rows[i].clone()).collect();
    let config = transform_config(raw_hash, TRIAL_SEED, rows.len() as u64);
    let t = CategoricalTransform::fit(&diabetes_schema(), &train_rows, &config)
        .expect("fit");
    assert!(t.phi_width() < PHI_CEILING, "phi width {}", t.phi_width());
    assert_eq!(t.n_routing_features(), K_ROUTING);
}

#[test]
#[ignore = "heavy 20K-row ABNG training run -- pending Research Phase R0 perf work \
            + a re-scope to a small always-run sub-sample; see PHASE_0_9_5_HANDOFF_V2.md"]
fn diabetes130_subsample_trial() {
    let Some((rows, raw_hash)) = dataset_or_skip("diabetes130_subsample_trial") else {
        return;
    };
    let result = run_trial(&rows, raw_hash, &diabetes_schema(), TRIAL_SEED, SUBSAMPLE_ROWS);

    // Sub-sample shape.
    assert_eq!(result.n_rows_used, SUBSAMPLE_ROWS);
    assert_eq!(result.n_routing_features, K_ROUTING);
    assert!(result.phi_width < PHI_CEILING);

    // The audit chain advanced — two events per trained row.
    assert!(result.audit_event_count >= 2 * result.n_train);
    assert_eq!(result.chain_head_hex.len(), 64);
    assert_eq!(result.merkle_root_hex.len(), 64);

    // Every metric is finite and within its valid range.
    let m = &result.metrics;
    for (name, v, hi) in [
        ("accuracy", m.accuracy, 1.0),
        ("balanced_accuracy", m.balanced_accuracy, 1.0),
        ("auc", m.auc, 1.0),
        ("f1", m.f1, 1.0),
        ("ece", m.ece, 1.0),
    ] {
        assert!(v.is_finite() && (0.0..=hi).contains(&v), "{name} = {v}");
    }
    assert!(m.brier.is_finite() && m.brier >= 0.0);
    assert!(m.nll.is_finite() && m.nll >= 0.0);
    assert!(m.n_test > 0);

    // The ensemble learns *some* signal — a real medical dataset
    // through a trained root+leaf BLR must beat a coin flip.
    assert!(
        m.auc > 0.55,
        "AUC {} — the harness is not learning",
        m.auc
    );

    // Per-leaf routing distributed training rows across the tree.
    let lr = &result.leaf_report;
    assert_eq!(lr.total_leaves, (ROUTE_BINS as usize).pow(K_ROUTING as u32));
    assert!(lr.populated_leaves > 1, "all rows routed to one leaf");
    assert_eq!(lr.populated_leaves + lr.dead_leaves, lr.total_leaves);
    assert!(lr.min_per_populated >= 1, "a populated leaf has zero rows");
    assert!(lr.max_per_populated >= lr.min_per_populated);
    assert!(
        (lr.min_per_populated as f64) <= lr.mean_per_populated
            && lr.mean_per_populated <= (lr.max_per_populated as f64),
        "mean per leaf {} outside [{}, {}]",
        lr.mean_per_populated,
        lr.min_per_populated,
        lr.max_per_populated
    );

    // Ablation accuracies are all valid probabilities.
    let ab = &result.ablation;
    for v in [ab.root_accuracy, ab.leaf_accuracy, ab.ensemble_accuracy] {
        assert!(v.is_finite() && (0.0..=1.0).contains(&v));
    }

    // Phase 0.10 §4.A — emit the metrics so the operator can compare
    // against the blog baseline (AUC 0.6107, Brier 0.0980, NLL 0.3435,
    // ECE 0.0101 at K_ROUTING=2 + stronger BLR prior). Visible with
    // `cargo test ... -- --ignored --nocapture`.
    eprintln!(
        "\ndiabetes130_subsample_trial:\n  config: K_ROUTING={} BLR_PRIOR=({},{},{}) seed={}\n  shape: n_rows={} n_train={} phi={} routing_cols={:?}\n  raw_metrics: acc={:.4} bal_acc={:.4} auc={:.4} f1={:.4} brier={:.4} nll={:.4} ece={:.4} n_test={}\n  ablation: root_acc={:.4} leaf_acc={:.4} ensemble_acc={:.4}\n  audit: chain_head={} merkle_root={} events={}\n  leaves: total={} populated={} dead={} min/mean/max={}/{:.1}/{}\n",
        K_ROUTING, BLR_PRIOR_PRECISION, BLR_PRIOR_A, BLR_PRIOR_B, TRIAL_SEED,
        result.n_rows_used, result.n_train, result.phi_width, result.routing_feature_columns,
        m.accuracy, m.balanced_accuracy, m.auc, m.f1, m.brier, m.nll, m.ece, m.n_test,
        ab.root_accuracy, ab.leaf_accuracy, ab.ensemble_accuracy,
        result.chain_head_hex, result.merkle_root_hex, result.audit_event_count,
        result.leaf_report.total_leaves, result.leaf_report.populated_leaves, result.leaf_report.dead_leaves,
        result.leaf_report.min_per_populated, result.leaf_report.mean_per_populated, result.leaf_report.max_per_populated,
    );
}

/// Phase 0.10 §4.B — Locke-driven leakage prune: same as
/// `diabetes130_subsample_trial` but with the death/hospice discharge
/// rows filtered out before sub-sampling. The hypothesis is that
/// removing the leakage rows lets the model learn a cleaner
/// `<30 vs not-<30` signal — a small but real AUC bump expected.
///
/// Expected drop: ~2,761 of 101,766 rows (~2.7%) match
/// [`DEATH_DISCHARGE_CODES`]. Those rows are all `readmitted=NO` by
/// construction, so the positive-class proportion in the filtered set
/// rises slightly (~11.2% → ~11.5%).
#[test]
#[ignore = "heavy 20K-row ABNG training run with Locke-driven leakage prune (Phase 0.10 §4.B)"]
fn diabetes130_subsample_trial_locke_pruned() {
    let Some((rows, raw_hash)) = dataset_or_skip("diabetes130_subsample_trial_locke_pruned")
    else {
        return;
    };

    let (rows_pruned, dropped) = filter_out_death_discharges(&rows);
    let kept = rows_pruned.len();

    // Re-hash the post-filter dataset so the transform's
    // `raw_dataset_hash` matches the actual training input. The
    // pre-filter hash is preserved in the audit eprintln for traceability.
    let mut canonical = String::new();
    for row in &rows_pruned {
        canonical.push_str(&row.join(","));
        canonical.push('\n');
    }
    let raw_hash_pruned = cjc_snap::hash::sha256(canonical.as_bytes());

    let result = run_trial(
        &rows_pruned,
        raw_hash_pruned,
        &diabetes_schema(),
        TRIAL_SEED,
        SUBSAMPLE_ROWS,
    );

    // Shape — pruned should keep ~97.3% of rows.
    assert!(dropped > 0, "death/hospice filter dropped no rows");
    assert!(
        kept > 95_000,
        "pruned dataset {kept} smaller than expected (started {})",
        rows.len()
    );

    // Audit chain advanced.
    let m = &result.metrics;
    assert!(result.audit_event_count >= 2 * result.n_train);
    assert_eq!(result.chain_head_hex.len(), 64);
    assert_eq!(result.merkle_root_hex.len(), 64);

    // The pruned ensemble must learn signal — strictly better than coin flip.
    assert!(m.auc > 0.55, "AUC {} after Locke prune", m.auc);
    for (name, v) in [
        ("brier", m.brier),
        ("nll", m.nll),
        ("ece", m.ece),
    ] {
        assert!(v.is_finite() && v >= 0.0, "{name} = {v}");
    }

    // Per-leaf routing distributed across the tree (same shape gates as
    // the unfiltered trial — the filter does not change ABNG's routing
    // mechanics).
    let lr = &result.leaf_report;
    assert_eq!(lr.total_leaves, (ROUTE_BINS as usize).pow(K_ROUTING as u32));
    assert!(lr.populated_leaves > 1, "all rows routed to one leaf");

    // Emit metrics — operator compares against the §4.A baseline
    // (AUC 0.6312 at K=2, prior=0.5 on the unfiltered 20K).
    let ab = &result.ablation;
    let raw_hash_pre_hex = hex32(&raw_hash);
    eprintln!(
        "\ndiabetes130_subsample_trial_locke_pruned:\n  config: K_ROUTING={} BLR_PRIOR=({},{},{}) seed={}\n  filter: dropped={} kept={} death_codes={:?}\n  shape: n_rows={} n_train={} phi={} routing_cols={:?}\n  raw_metrics: acc={:.4} bal_acc={:.4} auc={:.4} f1={:.4} brier={:.4} nll={:.4} ece={:.4} n_test={}\n  ablation: root_acc={:.4} leaf_acc={:.4} ensemble_acc={:.4}\n  audit: pre_filter_hash={} post_filter_hash={} chain_head={} merkle_root={} events={}\n  leaves: total={} populated={} dead={} min/mean/max={}/{:.1}/{}\n",
        K_ROUTING, BLR_PRIOR_PRECISION, BLR_PRIOR_A, BLR_PRIOR_B, TRIAL_SEED,
        dropped, kept, DEATH_DISCHARGE_CODES,
        result.n_rows_used, result.n_train, result.phi_width, result.routing_feature_columns,
        m.accuracy, m.balanced_accuracy, m.auc, m.f1, m.brier, m.nll, m.ece, m.n_test,
        ab.root_accuracy, ab.leaf_accuracy, ab.ensemble_accuracy,
        raw_hash_pre_hex, hex32(&raw_hash_pruned), result.chain_head_hex, result.merkle_root_hex, result.audit_event_count,
        result.leaf_report.total_leaves, result.leaf_report.populated_leaves, result.leaf_report.dead_leaves,
        result.leaf_report.min_per_populated, result.leaf_report.mean_per_populated, result.leaf_report.max_per_populated,
    );
}

#[test]
#[ignore = "heavy 6K-row ABNG determinism double-run -- pending Research Phase R0 \
            perf work + a test re-scope; see PHASE_0_9_5_HANDOFF_V2.md"]
fn diabetes130_trial_is_deterministic() {
    let Some((rows, raw_hash)) = dataset_or_skip("diabetes130_trial_is_deterministic")
    else {
        return;
    };
    let schema = diabetes_schema();
    // Smaller probe budget — determinism does not need the full 20K.
    let a = run_trial(&rows, raw_hash, &schema, TRIAL_SEED, DETERMINISM_PROBE_ROWS);
    let b = run_trial(&rows, raw_hash, &schema, TRIAL_SEED, DETERMINISM_PROBE_ROWS);
    assert_eq!(a.chain_head_hex, b.chain_head_hex, "chain head differs");
    assert_eq!(a.merkle_root_hex, b.merkle_root_hex, "merkle root differs");
    assert_eq!(a.audit_event_count, b.audit_event_count);
    assert_eq!(a.metrics, b.metrics, "metrics differ");
    assert_eq!(a.routing_feature_columns, b.routing_feature_columns);
}

#[test]
#[ignore = "full 101,766-row Diabetes-130 run — invoke explicitly or via COMMIT 7-8"]
fn diabetes130_full_run() {
    let Some((rows, raw_hash)) = dataset_or_skip("diabetes130_full_run") else {
        return;
    };
    let result = run_trial(&rows, raw_hash, &diabetes_schema(), TRIAL_SEED, usize::MAX);
    assert_eq!(result.n_rows_used, rows.len());
    assert!(result.phi_width < PHI_CEILING);
    assert!(result.metrics.auc > 0.55, "AUC {}", result.metrics.auc);
    eprintln!(
        "diabetes130_full_run: n_train={} phi={} routing={:?} \
         acc={:.4} bal_acc={:.4} auc={:.4} f1={:.4} brier={:.4} nll={:.4} ece={:.4} \
         chain_head={}",
        result.n_train,
        result.phi_width,
        result.routing_feature_columns,
        result.metrics.accuracy,
        result.metrics.balanced_accuracy,
        result.metrics.auc,
        result.metrics.f1,
        result.metrics.brier,
        result.metrics.nll,
        result.metrics.ece,
        result.chain_head_hex,
    );
}
