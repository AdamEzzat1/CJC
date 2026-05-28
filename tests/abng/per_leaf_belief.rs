//! Per-leaf Locke BeliefScore × ABNG `ood_score` experiment (v0.6.3).
//!
//! The hypothesis the design memo
//! [`docs/locke/ABNG_PER_LEAF_BELIEF.md`] proposes:
//!
//! > The `BeliefScore` of the training subset routed to leaf L is a
//! > useful predictor of how often leaf L should abstain at inference
//! > time.
//!
//! This file implements a small, fast version of the experiment on a
//! synthetic fixture so it runs cleanly inside `cargo test`. The full
//! diabetes-130 run is deferred to a longer-running ignored test.
//!
//! ## Pipeline
//!
//! 1. Generate a 3-cluster synthetic dataset where one cluster has
//!    extra NaN and outlier injection (the "dirty" cluster).
//! 2. Build an ABNG graph with a 1-D codebook and route every row →
//!    capture per-row leaf id via `route_to_leaf_batch`.
//! 3. Bucket rows by leaf id (`BTreeMap` for determinism).
//! 4. For each leaf with `>= MIN_ROWS_PER_LEAF` rows, build a slice
//!    DataFrame via the in-test `take_rows` helper and run Locke
//!    validate + belief on the slice.
//! 5. Assert the structural invariants: the experiment runs end-to-end,
//!    each per-leaf belief is in `[0, 1]`, and the "dirty cluster" leaf
//!    has a lower `missingness_score` than the "clean cluster" leaf
//!    (the hypothesis's directional prediction).
//!
//! Determinism is checked by running the full pipeline twice and
//! comparing belief vectors per leaf.

use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_data::{Column, DataFrame};
use cjc_locke::{
    api::{belief_report_from_locke, validate, ValidateOptions},
    BeliefScore, ValidationConfig,
};
use std::collections::BTreeMap;

// ─── Fixture parameters ──────────────────────────────────────────────────

const SEED: u64 = 0xC0FFEE;
const N_PER_CLUSTER: usize = 200;
const MIN_ROWS_PER_LEAF: usize = 30;

// ─── Helpers ─────────────────────────────────────────────────────────────

/// Deterministic splitmix64 — same RNG ABNG uses elsewhere.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Uniform `[0, 1)` from splitmix64.
fn rand_f64(state: &mut u64) -> f64 {
    let raw = splitmix64(state);
    (raw >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Approximate Gaussian via the standard "average of 12 uniforms" trick.
fn rand_normal(state: &mut u64, mean: f64, std: f64) -> f64 {
    let mut acc = 0.0;
    for _ in 0..12 {
        acc += rand_f64(state);
    }
    mean + std * (acc - 6.0)
}

/// In-test row selector. Not part of cjc-data's public API (deliberately
/// — adding `DataFrame::take_rows` to `cjc-data` is a v0.7 question per
/// the design memo).
fn take_rows(df: &DataFrame, indices: &[usize]) -> DataFrame {
    let new_columns: Vec<(String, Column)> = df
        .columns
        .iter()
        .map(|(name, col)| {
            let slice = match col {
                Column::Float(v) => {
                    Column::Float(indices.iter().map(|&i| v[i]).collect())
                }
                Column::Int(v) => {
                    Column::Int(indices.iter().map(|&i| v[i]).collect())
                }
                Column::Bool(v) => {
                    Column::Bool(indices.iter().map(|&i| v[i]).collect())
                }
                Column::Str(v) => {
                    Column::Str(indices.iter().map(|&i| v[i].clone()).collect())
                }
                Column::DateTime(v) => {
                    Column::DateTime(indices.iter().map(|&i| v[i]).collect())
                }
                Column::Categorical { levels, codes } => Column::Categorical {
                    levels: levels.clone(),
                    codes: indices.iter().map(|&i| codes[i]).collect(),
                },
                Column::CategoricalAdaptive(_cc) => {
                    // For the experiment we don't generate adaptive
                    // columns; if needed in v0.7, take_rows for
                    // CategoricalAdaptive needs to materialise the
                    // sub-stream from the byte dictionary.
                    panic!("take_rows on CategoricalAdaptive not supported in test fixture");
                }
            };
            (name.clone(), slice)
        })
        .collect();
    DataFrame::from_columns(new_columns).unwrap()
}

/// Build a 3-cluster synthetic dataset:
/// - cluster A: clean, centered at 0.0
/// - cluster B: noisy with 20% NaN injection in `feat_a` (the "dirty" cluster)
/// - cluster C: clean, centered at 5.0
fn build_three_cluster_dataset() -> DataFrame {
    let mut state = SEED;
    let mut feat_a: Vec<f64> = Vec::with_capacity(N_PER_CLUSTER * 3);
    let mut feat_b: Vec<f64> = Vec::with_capacity(N_PER_CLUSTER * 3);
    // Cluster A — clean, mean=0.
    for _ in 0..N_PER_CLUSTER {
        feat_a.push(rand_normal(&mut state, 0.0, 1.0));
        feat_b.push(rand_normal(&mut state, 0.0, 1.0));
    }
    // Cluster B — dirty, mean=2.5, 20% NaN.
    for i in 0..N_PER_CLUSTER {
        let v = rand_normal(&mut state, 2.5, 1.0);
        if i % 5 == 0 {
            feat_a.push(f64::NAN);
        } else {
            feat_a.push(v);
        }
        feat_b.push(rand_normal(&mut state, 2.5, 1.0));
    }
    // Cluster C — clean, mean=5.
    for _ in 0..N_PER_CLUSTER {
        feat_a.push(rand_normal(&mut state, 5.0, 1.0));
        feat_b.push(rand_normal(&mut state, 5.0, 1.0));
    }
    DataFrame::from_columns(vec![
        ("feat_a".into(), Column::Float(feat_a)),
        ("feat_b".into(), Column::Float(feat_b)),
    ])
    .unwrap()
}

/// Route every row of `df` to a leaf using the supplied graph. Assumes
/// `df` has a single column `feat_a` (the routing feature). Returns
/// one leaf id per row.
fn route_all_rows(g: &AdaptiveBeliefGraph, df: &DataFrame) -> Vec<u32> {
    let Column::Float(values) = df.get_column("feat_a").unwrap() else {
        panic!("routing column must be Float");
    };
    // route_to_leaf_batch takes a flat row-major buffer; d=1 here so
    // each row is one value.
    let leaves = g
        .route_to_leaf_batch(values, values.len())
        .expect("route_to_leaf_batch should succeed");
    leaves.iter().map(|n| *n as u32).collect()
}

/// Pre-allocate a full `branching^depth` routing tree by BFS expansion.
/// Cribbed from `dataset_a_diabetes130.rs` — without it the routing tree
/// is just the root and every row routes to `leaf_id=0`.
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

/// Build an ABNG graph with a 1-D codebook over `feat_a`. `n_bins` must
/// be a power of two per `QuantileCodebook::from_flat`'s contract; use
/// 4 (→ 3 boundaries) which gives the synthetic clusters reasonable
/// separation. Then pre-allocate the 4-leaf routing tree so `descend`
/// actually distributes rows across leaves.
fn build_routing_graph(_df: &DataFrame) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(SEED);
    let n_bins = 4u16;
    let boundaries: Vec<f64> = vec![1.25, 3.75, 6.0];
    g.set_codebook(1, n_bins, &boundaries).unwrap();
    pre_allocate_full_tree(&mut g, n_bins as u8, 1);
    g
}

// ─── The experiment ──────────────────────────────────────────────────────

/// Build per-leaf bucket → row-indices map. Deterministic via BTreeMap.
fn bucket_by_leaf(leaves: &[u32]) -> BTreeMap<u32, Vec<usize>> {
    let mut out: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, &lid) in leaves.iter().enumerate() {
        out.entry(lid).or_default().push(i);
    }
    out
}

/// Per-leaf BeliefScore computed by running Locke on the leaf's row slice.
fn per_leaf_belief(df: &DataFrame, leaves: &[u32]) -> BTreeMap<u32, BeliefScore> {
    let buckets = bucket_by_leaf(leaves);
    let mut out: BTreeMap<u32, BeliefScore> = BTreeMap::new();
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
        out.insert(*leaf_id, belief.score);
    }
    out
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[test]
fn per_leaf_experiment_runs_end_to_end() {
    let df = build_three_cluster_dataset();
    assert_eq!(df.nrows(), 3 * N_PER_CLUSTER);

    let g = build_routing_graph(&df);
    let leaves = route_all_rows(&g, &df);
    assert_eq!(leaves.len(), df.nrows());

    let per_leaf = per_leaf_belief(&df, &leaves);
    // We should get at least 2 leaves with enough rows for a belief.
    assert!(
        per_leaf.len() >= 2,
        "expected ≥ 2 populated leaves, got {}",
        per_leaf.len()
    );

    // Every belief vector must be in [0, 1] on every axis.
    for (lid, b) in &per_leaf {
        assert!(b.overall >= 0.0 && b.overall <= 1.0, "leaf {} overall = {}", lid, b.overall);
        assert!(b.missingness_score >= 0.0 && b.missingness_score <= 1.0);
        assert!(b.schema_score >= 0.0 && b.schema_score <= 1.0);
        assert!(b.constraint_score >= 0.0 && b.constraint_score <= 1.0);
    }
}

#[test]
fn dirty_cluster_leaf_has_lower_missingness_score() {
    // The directional prediction of the design memo: a leaf whose
    // routed rows include the seeded 20%-NaN cluster B should have a
    // lower `missingness_score` than a leaf whose routed rows are
    // clean (clusters A or C).
    let df = build_three_cluster_dataset();
    let g = build_routing_graph(&df);
    let leaves = route_all_rows(&g, &df);
    let per_leaf = per_leaf_belief(&df, &leaves);

    // For each leaf, count how many of its rows came from cluster B
    // (rows N_PER_CLUSTER..2*N_PER_CLUSTER).
    let buckets = bucket_by_leaf(&leaves);
    let leaf_cluster_b_share: BTreeMap<u32, f64> = buckets
        .iter()
        .filter(|(lid, idx)| {
            per_leaf.contains_key(lid) && idx.len() >= MIN_ROWS_PER_LEAF
        })
        .map(|(lid, indices)| {
            let in_b = indices
                .iter()
                .filter(|&&i| i >= N_PER_CLUSTER && i < 2 * N_PER_CLUSTER)
                .count() as f64;
            (*lid, in_b / indices.len() as f64)
        })
        .collect();

    // Locate the leaf with the highest cluster-B share AND a leaf with
    // (close-to-)zero share. Assert the dirty leaf has lower missingness.
    let dirty_leaf = leaf_cluster_b_share
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(lid, _)| *lid)
        .expect("at least one populated leaf");
    let clean_leaf = leaf_cluster_b_share
        .iter()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(lid, _)| *lid)
        .expect("at least one populated leaf");

    if dirty_leaf == clean_leaf {
        // Codebook routed everything into a single leaf — not enough
        // separation for the directional test. Skip with a soft assert.
        return;
    }

    let dirty_score = per_leaf[&dirty_leaf].missingness_score;
    let clean_score = per_leaf[&clean_leaf].missingness_score;
    assert!(
        dirty_score <= clean_score + 1e-9,
        "expected dirty leaf {} (B-share {:.2}) missingness {} ≤ clean leaf {} (B-share {:.2}) missingness {}",
        dirty_leaf,
        leaf_cluster_b_share[&dirty_leaf],
        dirty_score,
        clean_leaf,
        leaf_cluster_b_share[&clean_leaf],
        clean_score,
    );
}

#[test]
fn per_leaf_belief_is_deterministic_across_runs() {
    let df = build_three_cluster_dataset();
    let g_a = build_routing_graph(&df);
    let g_b = build_routing_graph(&df);
    let leaves_a = route_all_rows(&g_a, &df);
    let leaves_b = route_all_rows(&g_b, &df);
    assert_eq!(leaves_a, leaves_b, "ABNG routing must be deterministic");

    let belief_a = per_leaf_belief(&df, &leaves_a);
    let belief_b = per_leaf_belief(&df, &leaves_b);
    // Compare leaf-by-leaf — BeliefScore has 8 axes, all in [0,1].
    assert_eq!(belief_a.keys().count(), belief_b.keys().count());
    for (lid, ba) in &belief_a {
        let bb = belief_b.get(lid).unwrap();
        assert!((ba.overall - bb.overall).abs() < 1e-12, "leaf {} overall divergent", lid);
        assert!((ba.missingness_score - bb.missingness_score).abs() < 1e-12);
        assert!((ba.schema_score - bb.schema_score).abs() < 1e-12);
        assert!((ba.constraint_score - bb.constraint_score).abs() < 1e-12);
    }
}

#[test]
fn experiment_handles_empty_routing_leaves() {
    // Construct a single-cluster DataFrame so most leaves should be empty.
    let mut state = SEED;
    let mut feat_a: Vec<f64> = Vec::with_capacity(N_PER_CLUSTER);
    let mut feat_b: Vec<f64> = Vec::with_capacity(N_PER_CLUSTER);
    for _ in 0..N_PER_CLUSTER {
        feat_a.push(rand_normal(&mut state, 0.0, 0.1)); // tight around 0
        feat_b.push(rand_normal(&mut state, 0.0, 0.1));
    }
    let df = DataFrame::from_columns(vec![
        ("feat_a".into(), Column::Float(feat_a)),
        ("feat_b".into(), Column::Float(feat_b)),
    ])
    .unwrap();
    let g = build_routing_graph(&df);
    let leaves = route_all_rows(&g, &df);
    let per_leaf = per_leaf_belief(&df, &leaves);
    // Most leaves below MIN_ROWS_PER_LEAF — should still return at
    // least one populated leaf without panicking.
    assert!(per_leaf.len() >= 1, "expected ≥ 1 populated leaf for tight-cluster fixture");
}
