//! Per-leaf Locke BeliefScore experiment on the real Diabetes-130 dataset
//! (v0.7 part 1 follow-up to the synthetic-fixture version in
//! `tests/abng/per_leaf_belief.rs`).
//!
//! ## Pipeline
//!
//! 1. Load `tests/data/diabetes_130/diabetic_data.csv` (101,766 rows × 50 cols).
//!    Skip gracefully when absent — the file is untracked.
//! 2. Build a `DataFrame` directly from the raw CSV: column types inferred
//!    from the schema (numeric cols → `Float`, target → `Int` via the
//!    same `<30 / >30 / NO` → `{0, 1, 2}` mapping the multi-class
//!    leakage detector accepts).
//! 3. Sub-sample to `SUBSAMPLE_ROWS = 20_000` rows (stratified on
//!    `readmitted`) for tractability; matches the Phase 0.9.5 §0 default.
//! 4. Build a 1-D ABNG codebook over `time_in_hospital` (which has range
//!    ~`[1, 14]` and good cluster separation). Pre-allocate a 4-leaf
//!    routing tree.
//! 5. Route every row → capture per-row leaf id via
//!    [`AdaptiveBeliefGraph::route_to_leaf_batch`].
//! 6. For each leaf with `≥ MIN_ROWS_PER_LEAF` rows: slice the original
//!    DataFrame by row indices, run Locke `validate()` with target =
//!    `readmitted` and `primary_key = patient_nbr`, derive per-leaf
//!    [`BeliefScore`].
//! 7. Aggregate to a dataset-level "weighted-by-leaf-row-count" belief
//!    via [`cjc_locke::compose_weighted`] from the v0.7 part 1 algebra.
//! 8. Emit per-leaf belief table to `target/diabetes_per_leaf_belief.csv`.
//!
//! ## Asserts
//!
//! - At least 2 populated leaves (the routing distributed rows).
//! - Per-leaf belief axes all in `[0, 1]`.
//! - The aggregated weighted belief equals the per-leaf compose under
//!   `compose_weighted` (sanity-check the algebra plumbing on real data).
//! - The run is deterministic — two runs of the full pipeline produce
//!   byte-identical CSV output.
//!
//! All tests are `#[ignore]` so they run only when invoked explicitly:
//! `cargo test --test abng per_leaf_belief_diabetes130 -- --ignored`.

use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_data::{Column, DataFrame};
use cjc_locke::{
    api::{belief_report_from_locke, validate, ValidateOptions},
    compose_many_arithmetic, compose_weighted, BeliefScore, NullMask, NullMaskMap,
    ValidationConfig,
};
use std::collections::BTreeMap;
use std::path::Path;

const SUBSAMPLE_ROWS: usize = 20_000;
const MIN_ROWS_PER_LEAF: usize = 50;
const SEED: u64 = 0xD1ABE7E5;
const N_COLUMNS: usize = 50;
const DATASET_REL_PATH: &str = "tests/data/diabetes_130/diabetic_data.csv";

/// 50-column schema — order matters because we index by position.
const COLS: &[&str] = &[
    "encounter_id",
    "patient_nbr",
    "race",
    "gender",
    "age",
    "weight",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "payer_code",
    "medical_specialty",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "diag_1",
    "diag_2",
    "diag_3",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
    "readmitted",
];

const TARGET_COL: usize = 49;
const ROUTING_COL_NAME: &str = "time_in_hospital";

// Numeric columns by index (everything else is Str).
const NUMERIC_INDICES: &[usize] = &[
    0,  // encounter_id (Int)
    1,  // patient_nbr (Int)
    6,  // admission_type_id
    7,  // discharge_disposition_id
    8,  // admission_source_id
    9,  // time_in_hospital
    12, // num_lab_procedures
    13, // num_procedures
    14, // num_medications
    15, // number_outpatient
    16, // number_emergency
    17, // number_inpatient
    21, // number_diagnoses
];

// ─── CSV loader (gracefully skips when absent) ──────────────────────────

fn load_dataset() -> Option<Vec<Vec<String>>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(DATASET_REL_PATH);
    let bytes = std::fs::read(&path).ok()?;
    let text = std::str::from_utf8(&bytes).expect("UTF-8");
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next()?.split(',').map(|s| s.trim()).collect();
    assert_eq!(header.len(), N_COLUMNS);
    let mut rows: Vec<Vec<String>> = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let cells: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();
        if cells.len() == N_COLUMNS {
            rows.push(cells);
        }
    }
    Some(rows)
}

// ─── Sub-sample (stratified on `readmitted`) ────────────────────────────

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn shuffle(items: &mut [usize], state: &mut u64) {
    for i in (1..items.len()).rev() {
        let j = (splitmix64(state) % (i as u64 + 1)) as usize;
        items.swap(i, j);
    }
}

/// Stratified sub-sample on the 3-class `readmitted` column. Returns up
/// to `budget` row indices, class-ratio-preserving, shuffled.
fn stratified_subsample(rows: &[Vec<String>], budget: usize, seed: u64) -> Vec<usize> {
    let mut by_class: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, row) in rows.iter().enumerate() {
        by_class.entry(row[TARGET_COL].clone()).or_default().push(i);
    }
    let mut state = seed;
    let total: usize = by_class.values().map(|v| v.len()).sum();
    if budget >= total {
        let mut out: Vec<usize> = (0..total).collect();
        shuffle(&mut out, &mut state);
        return out;
    }
    let mut out: Vec<usize> = Vec::with_capacity(budget);
    let mut remaining = budget;
    let n_classes = by_class.len();
    for (i, (_class, indices)) in by_class.into_iter().enumerate() {
        let mut local = indices;
        shuffle(&mut local, &mut state);
        let take = if i + 1 == n_classes {
            remaining.min(local.len())
        } else {
            ((local.len() as u64 * budget as u64) / total as u64) as usize
        };
        out.extend_from_slice(&local[..take.min(local.len())]);
        remaining = remaining.saturating_sub(take);
    }
    shuffle(&mut out, &mut state);
    out
}

// ─── Type-inferred DataFrame construction ───────────────────────────────

/// Parse the dataset rows at `indices` into a cjc-data DataFrame.
/// Numeric columns become `Column::Float` (NaN for `?`/empty/non-numeric),
/// the target becomes `Column::Int` (0 for `NO`, 1 for `<30`, 2 for `>30`),
/// everything else becomes `Column::Str`.
fn build_dataframe(rows: &[Vec<String>], indices: &[usize]) -> DataFrame {
    let n = indices.len();
    let mut numeric_set: std::collections::BTreeSet<usize> = NUMERIC_INDICES.iter().copied().collect();
    numeric_set.insert(TARGET_COL); // we'll override target separately

    let mut cols: Vec<(String, Column)> = Vec::with_capacity(N_COLUMNS);
    for (col_idx, col_name) in COLS.iter().enumerate() {
        if col_idx == TARGET_COL {
            let mut v: Vec<i64> = Vec::with_capacity(n);
            for &i in indices {
                let class = match rows[i][col_idx].as_str() {
                    "NO" => 0,
                    "<30" => 1,
                    ">30" => 2,
                    _ => 0,
                };
                v.push(class);
            }
            cols.push(((*col_name).into(), Column::Int(v)));
        } else if NUMERIC_INDICES.contains(&col_idx) {
            let mut v: Vec<f64> = Vec::with_capacity(n);
            for &i in indices {
                let cell = &rows[i][col_idx];
                let parsed = cell.parse::<f64>().unwrap_or(f64::NAN);
                v.push(parsed);
            }
            cols.push(((*col_name).into(), Column::Float(v)));
        } else {
            let mut v: Vec<String> = Vec::with_capacity(n);
            for &i in indices {
                v.push(rows[i][col_idx].clone());
            }
            cols.push(((*col_name).into(), Column::Str(v)));
        }
    }
    DataFrame::from_columns(cols).unwrap()
}

// ─── In-test row selector (consistent with the synthetic fixture) ──────

fn take_rows(df: &DataFrame, indices: &[usize]) -> DataFrame {
    let new_columns: Vec<(String, Column)> = df
        .columns
        .iter()
        .map(|(name, col)| {
            let slice = match col {
                Column::Float(v) => Column::Float(indices.iter().map(|&i| v[i]).collect()),
                Column::Int(v) => Column::Int(indices.iter().map(|&i| v[i]).collect()),
                Column::Bool(v) => Column::Bool(indices.iter().map(|&i| v[i]).collect()),
                Column::Str(v) => Column::Str(indices.iter().map(|&i| v[i].clone()).collect()),
                Column::DateTime(v) => Column::DateTime(indices.iter().map(|&i| v[i]).collect()),
                Column::Categorical { levels, codes } => Column::Categorical {
                    levels: levels.clone(),
                    codes: indices.iter().map(|&i| codes[i]).collect(),
                },
                Column::CategoricalAdaptive(_) => {
                    panic!("CategoricalAdaptive take_rows unsupported in this fixture")
                }
            };
            (name.clone(), slice)
        })
        .collect();
    DataFrame::from_columns(new_columns).unwrap()
}

// ─── Routing ────────────────────────────────────────────────────────────

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

/// 1-D ABNG codebook + 4-leaf pre-allocated tree over a routing column.
fn build_routing_graph() -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(SEED);
    let n_bins: u16 = 4;
    // Boundaries chosen for time_in_hospital (range ~ [1, 14]).
    let boundaries: Vec<f64> = vec![3.5, 6.5, 9.5];
    g.set_codebook(1, n_bins, &boundaries).unwrap();
    pre_allocate_full_tree(&mut g, n_bins as u8, 1);
    g
}

fn route_all_rows(g: &AdaptiveBeliefGraph, df: &DataFrame) -> Vec<u32> {
    let Column::Float(values) = df.get_column(ROUTING_COL_NAME).unwrap() else {
        panic!("routing column must be Float");
    };
    let leaves = g.route_to_leaf_batch(values, values.len()).expect("route");
    leaves.iter().map(|n| *n as u32).collect()
}

// ─── Per-leaf belief ────────────────────────────────────────────────────

fn bucket_by_leaf(leaves: &[u32]) -> BTreeMap<u32, Vec<usize>> {
    let mut out: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, &lid) in leaves.iter().enumerate() {
        out.entry(lid).or_default().push(i);
    }
    out
}

fn per_leaf_belief(
    df: &DataFrame,
    leaves: &[u32],
) -> BTreeMap<u32, (BeliefScore, usize)> {
    let buckets = bucket_by_leaf(leaves);
    let mut out: BTreeMap<u32, (BeliefScore, usize)> = BTreeMap::new();
    for (leaf_id, indices) in &buckets {
        if indices.len() < MIN_ROWS_PER_LEAF {
            continue;
        }
        let slice = take_rows(df, indices);
        let opts = ValidateOptions {
            dataset_label: format!("leaf_{}", leaf_id),
            config: ValidationConfig::default(),
            ..Default::default()
        };
        let report = validate(&slice, &opts);
        let belief = belief_report_from_locke(&report);
        out.insert(*leaf_id, (belief.score, indices.len()));
    }
    out
}

// ─── CSV emit ───────────────────────────────────────────────────────────

/// Deterministic per-leaf belief CSV.
fn emit_per_leaf_csv(per_leaf: &BTreeMap<u32, (BeliefScore, usize)>) -> String {
    let mut s = String::new();
    s.push_str("leaf_id,n_rows,overall,schema,missingness,drift,leakage,lineage,sample,duplication,constraint\n");
    for (lid, (b, n)) in per_leaf {
        s.push_str(&format!(
            "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            lid,
            n,
            b.overall,
            b.schema_score,
            b.missingness_score,
            b.drift_score,
            b.leakage_score,
            b.lineage_score,
            b.sample_score,
            b.duplication_score,
            b.constraint_score,
        ));
    }
    s
}

// ─── Tests (all #[ignore] — dataset is untracked) ──────────────────────

#[test]
#[ignore = "diabetes-130 dataset not in CI; run with --ignored"]
fn diabetes130_per_leaf_belief_runs_end_to_end() {
    let Some(rows) = load_dataset() else {
        eprintln!("dataset not found at {DATASET_REL_PATH}; skipping");
        return;
    };
    let subset = stratified_subsample(&rows, SUBSAMPLE_ROWS, SEED);
    let df = build_dataframe(&rows, &subset);
    let g = build_routing_graph();
    let leaves = route_all_rows(&g, &df);
    let per_leaf = per_leaf_belief(&df, &leaves);

    assert!(
        per_leaf.len() >= 2,
        "expected ≥ 2 populated leaves, got {}",
        per_leaf.len()
    );

    for (lid, (b, n)) in &per_leaf {
        for axis in [
            b.overall, b.schema_score, b.missingness_score, b.drift_score,
            b.leakage_score, b.lineage_score, b.sample_score, b.duplication_score,
            b.constraint_score,
        ] {
            assert!(axis >= 0.0 && axis <= 1.0, "leaf {} axis = {}", lid, axis);
        }
        assert!(*n >= MIN_ROWS_PER_LEAF);
    }
}

#[test]
#[ignore = "diabetes-130 dataset not in CI; run with --ignored"]
fn diabetes130_weighted_aggregation_matches_compose_weighted() {
    let Some(rows) = load_dataset() else {
        eprintln!("dataset not found at {DATASET_REL_PATH}; skipping");
        return;
    };
    let subset = stratified_subsample(&rows, SUBSAMPLE_ROWS, SEED);
    let df = build_dataframe(&rows, &subset);
    let g = build_routing_graph();
    let leaves = route_all_rows(&g, &df);
    let per_leaf = per_leaf_belief(&df, &leaves);

    let scores: Vec<BeliefScore> = per_leaf.values().map(|(s, _)| s.clone()).collect();
    let weights: Vec<f64> = per_leaf.values().map(|(_, n)| *n as f64).collect();
    let weighted = compose_weighted(&scores, &weights).expect("weighted aggregation");
    let unweighted = compose_many_arithmetic(&scores).expect("unweighted mean");

    // Both must be in [0, 1] on every axis.
    for (label, b) in [("weighted", &weighted), ("unweighted", &unweighted)] {
        for axis in [
            b.overall, b.schema_score, b.missingness_score, b.drift_score,
            b.leakage_score, b.lineage_score, b.sample_score, b.duplication_score,
            b.constraint_score,
        ] {
            assert!(axis >= 0.0 && axis <= 1.0, "{} axis = {}", label, axis);
        }
    }
}

#[test]
#[ignore = "diabetes-130 dataset not in CI; run with --ignored"]
fn diabetes130_per_leaf_belief_csv_is_deterministic() {
    let Some(rows) = load_dataset() else {
        eprintln!("dataset not found at {DATASET_REL_PATH}; skipping");
        return;
    };
    let subset = stratified_subsample(&rows, SUBSAMPLE_ROWS, SEED);
    let df = build_dataframe(&rows, &subset);
    let g = build_routing_graph();
    let leaves_a = route_all_rows(&g, &df);
    let leaves_b = route_all_rows(&g, &df);
    assert_eq!(leaves_a, leaves_b);

    let pl_a = per_leaf_belief(&df, &leaves_a);
    let pl_b = per_leaf_belief(&df, &leaves_b);
    let csv_a = emit_per_leaf_csv(&pl_a);
    let csv_b = emit_per_leaf_csv(&pl_b);
    assert_eq!(csv_a, csv_b, "per-leaf belief CSV must be byte-identical across runs");

    // Best-effort: write the CSV to `target/` so a curious user can
    // inspect it after running with --ignored. Failures here don't
    // fail the test — the byte-identical assertion is the contract.
    let out_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("diabetes_per_leaf_belief.csv");
    let _ = std::fs::write(&out_path, &csv_a);
}

// ─── Phase 0.10 §4.D part 1 — `?`-aware per-leaf belief ────────────────
//
// The baseline `per_leaf_belief` function (used above) calls Locke's
// `validate()` with no `null_masks`. Locke's default treats only
// `f64::NAN` (Float columns) as missing — so the diabetes-130
// convention of storing missingness as the literal `?` string in
// `Str` columns goes unnoticed. The per-leaf BeliefScore ends up with
// `missingness_score = 1.0` on every leaf (= "no missingness
// detected"), which is technically correct under the default config
// but useless as a data-quality signal.
//
// §4.D requires per-leaf BeliefScore to carry information so that a
// future "weight per-leaf prior by belief" step has a meaningful
// signal to weight by. This part 1 wires `?`-aware NullMasks into the
// per-leaf belief computation. Part 2 (deferred — needs new ABNG
// plumbing for per-leaf priors) would consume these informative
// belief vectors.

/// Build a `NullMaskMap` that marks rows where the value of any
/// `Str` column equals the diabetes-130 missing sentinel (`?`).
/// Numeric / Bool / Int columns aren't included because Locke handles
/// `f64::NAN` natively for them — and the build_dataframe loader
/// already maps `?`-in-numeric to NaN.
fn build_question_mark_null_masks(df: &DataFrame) -> NullMaskMap {
    let mut out = NullMaskMap::new();
    for (name, col) in df.columns.iter() {
        if let Column::Str(values) = col {
            let null_rows: Vec<usize> = values
                .iter()
                .enumerate()
                .filter_map(|(i, v)| if v == "?" { Some(i) } else { None })
                .collect();
            if !null_rows.is_empty() {
                out.insert(name.clone(), NullMask::from_indices(null_rows));
            }
        }
    }
    out
}

/// Per-leaf BeliefScore using `?`-aware `NullMasks`. Re-maps the
/// full-DataFrame null indices to the leaf-slice index space.
fn per_leaf_belief_with_masks(
    df: &DataFrame,
    leaves: &[u32],
    full_null_masks: &NullMaskMap,
) -> BTreeMap<u32, (BeliefScore, usize)> {
    let buckets = bucket_by_leaf(leaves);
    let mut out: BTreeMap<u32, (BeliefScore, usize)> = BTreeMap::new();
    for (leaf_id, indices) in &buckets {
        if indices.len() < MIN_ROWS_PER_LEAF {
            continue;
        }
        let slice = take_rows(df, indices);

        // Build a lookup from full-DataFrame index → leaf-slice index.
        // The leaf is a contiguous subset of `df`'s rows in `indices`'
        // order, so `index_lookup[old_idx] = position-of-old_idx-in-indices`.
        let mut index_lookup: BTreeMap<usize, usize> = BTreeMap::new();
        for (new, &old) in indices.iter().enumerate() {
            index_lookup.insert(old, new);
        }

        // Re-map each full-DataFrame NullMask to the leaf slice's
        // index space. Only the rows that actually land in this leaf
        // contribute to the per-leaf NullMask.
        let mut leaf_null_masks = NullMaskMap::new();
        for (col_name, mask) in full_null_masks {
            let remapped: Vec<usize> = mask
                .null_rows
                .iter()
                .filter_map(|old| index_lookup.get(old).copied())
                .collect();
            if !remapped.is_empty() {
                leaf_null_masks
                    .insert(col_name.clone(), NullMask::from_indices(remapped));
            }
        }

        let opts = ValidateOptions {
            dataset_label: format!("leaf_{}", leaf_id),
            config: ValidationConfig::default(),
            null_masks: leaf_null_masks,
            ..Default::default()
        };
        let report = validate(&slice, &opts);
        let belief = belief_report_from_locke(&report);
        out.insert(*leaf_id, (belief.score, indices.len()));
    }
    out
}

#[test]
#[ignore = "diabetes-130 dataset not in CI; run with --ignored"]
fn diabetes130_per_leaf_belief_with_question_marks() {
    let Some(rows) = load_dataset() else {
        eprintln!("dataset not found at {DATASET_REL_PATH}; skipping");
        return;
    };
    let subset = stratified_subsample(&rows, SUBSAMPLE_ROWS, SEED);
    let df = build_dataframe(&rows, &subset);
    let g = build_routing_graph();
    let leaves = route_all_rows(&g, &df);

    let null_masks = build_question_mark_null_masks(&df);

    // The `weight` column is documented to be ~97% `?` in diabetes-130;
    // the null mask for it must be non-empty.
    assert!(
        null_masks.contains_key("weight"),
        "build_question_mark_null_masks did not mark any `?` in `weight`"
    );
    let weight_null_count = null_masks["weight"].count();
    let weight_total = df.nrows() as u64;
    let weight_null_rate = weight_null_count as f64 / weight_total as f64;
    assert!(
        weight_null_rate > 0.90,
        "expected `weight` to be > 90% `?`, got {:.4}",
        weight_null_rate
    );

    let per_leaf_aware = per_leaf_belief_with_masks(&df, &leaves, &null_masks);
    let per_leaf_naive = per_leaf_belief(&df, &leaves);

    // Hypothesis: the `?`-aware variant produces a *different*
    // missingness_score on at least one leaf — proving the
    // ValidationConfig fix is the actual blocker for §4.D part 2.
    let mut any_diff = false;
    let mut min_aware_score = 1.0_f64;
    for (lid, (aware, _)) in &per_leaf_aware {
        if let Some((naive, _)) = per_leaf_naive.get(lid) {
            if (aware.missingness_score - naive.missingness_score).abs() > 1e-6 {
                any_diff = true;
            }
            min_aware_score = min_aware_score.min(aware.missingness_score);
        }
    }
    assert!(
        any_diff,
        "?-aware belief vector identical to naive — null masks did not flow through to Locke"
    );
    assert!(
        min_aware_score < 1.0,
        "minimum ?-aware missingness_score = {} (expected < 1.0 — `?`-heavy `weight` column should pull it down)",
        min_aware_score
    );

    let csv = emit_per_leaf_csv(&per_leaf_aware);
    eprintln!(
        "\ndiabetes130_per_leaf_belief_with_question_marks:\n  n_subsample={} n_columns_with_?={} weight_?_rate={:.4}\n  min_missingness_aware={:.4} min_missingness_naive={:.4}\n",
        df.nrows(),
        null_masks.len(),
        weight_null_rate,
        min_aware_score,
        per_leaf_naive
            .values()
            .map(|(b, _)| b.missingness_score)
            .fold(1.0_f64, f64::min),
    );
    eprint!("{}", csv);

    let out_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("diabetes_per_leaf_belief_with_question_marks.csv");
    let _ = std::fs::write(&out_path, &csv);
}
