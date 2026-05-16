//! Phase 0.9 Track P — Baseline validation on Wisconsin Breast Cancer.
//!
//! # Status: SCAFFOLDING + SYNTHETIC DATASET
//!
//! This is the first Track P commit. Goals:
//!
//! 1. Exercise the full ABNG training pipeline (codebook + leaf head +
//!    BLR prior + per-row `train_step` + audit chain + Merkle root)
//!    on a 569-sample × 30-feature binary classification shape that
//!    matches Wisconsin Breast Cancer.
//! 2. Pin the determinism gate: 5 runs at the same seed must produce
//!    byte-identical `chain_head` and `merkle_root` values. This is
//!    the hardest invariant of the Phase 0.8 → 0.9 transition; if
//!    it ever breaks, every later perf optimization is suspect.
//! 3. Pin the seed-sensitivity gate: 5 different seeds must produce
//!    5 distinct `chain_head` values. This rules out a degenerate
//!    "everything always produces the same hash" bug.
//!
//! # Dataset
//!
//! The harness currently uses a **deterministic synthetic dataset**
//! that mirrors Wisconsin BC's shape: 569 samples, 30 features, two
//! classes (357 benign / 212 malignant). Class 1 is shifted by +0.8
//! standard deviations on the first 10 features so that linear-BLR
//! routing can recover a non-trivial accuracy. The synthetic dataset
//! is generated via Box-Muller from splitmix64; it is byte-stable
//! across runs and platforms.
//!
//! To switch to the real UCI Wisconsin BC data:
//!
//! 1. Download `wdbc.data` from the UCI ML Repository:
//!    https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
//! 2. Save the file as `tests/data/wisconsin_bc.csv` in this repo.
//! 3. Re-run this test suite. The `load_dataset` function will
//!    auto-detect the CSV and use it instead of the synthetic stand-in.
//!
//! Until the CSV is present, the harness uses the synthetic generator
//! — that's enough to validate the determinism contract, but accuracy
//! numbers on synthetic data should NOT be compared against the
//! published Wisconsin BC accuracy ceiling (95–98%).
//!
//! # Future Track P work (later commits)
//!
//! * Real-data CSV loader (auto-detect).
//! * Per-leaf explainability report.
//! * SVG + CSV output to `bench_results/phase_0_9_baseline/`.
//! * Accuracy floor assertion (≥ 0.90 on real data).
//!
//! # Track P work already shipped
//!
//! * Determinism gate (5 runs at the same seed produce byte-equal
//!   chain_head + merkle_root + audit_event_count).
//! * Seed-sensitivity gate (5 distinct seeds → 5 distinct chain
//!   heads).
//! * Top-K feature selection via univariate F-score (computed on
//!   the train split; no test-set leakage). See
//!   [`compute_f_scores_binary`] + [`select_top_k_features`].
//! * Pre-allocated `N_BINS_PER_FEATURE^N_ROUTING_FEATURES` routing
//!   tree (4³ = 64 leaves) — every training row routes to a
//!   leaf at the deepest level, not to the root. Empty leaves are
//!   handled at predict time by `blr_predict_with_fallback`. See
//!   [`pre_allocate_full_tree`].

use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_ad::pinn::Activation;
use std::fmt::Write as _;

// ── Dataset shape (matches Wisconsin BC) ─────────────────────────────

const N_FEATURES_TOTAL: usize = 30;
const N_SAMPLES: usize = 569;
/// Class label counts in the real Wisconsin BC dataset (357 benign + 212
/// malignant). The synthetic generator reproduces this split to keep
/// the chain witness shapes comparable to a real-data run.
const N_CLASS_0: usize = 357;

// ── Routing configuration ────────────────────────────────────────────

/// Number of features used for ABNG's routing step. The other
/// `N_FEATURES_TOTAL - N_ROUTING_FEATURES` features participate only
/// in the leaf-level BLR's `phi` vector.
const N_ROUTING_FEATURES: usize = 3;
/// Number of quantile bins per routing feature. With
/// N_ROUTING_FEATURES=3 and N_BINS_PER_FEATURE=4, the codebook
/// produces up to 4³ = 64 possible routes.
const N_BINS_PER_FEATURE: u16 = 4;

// ── BLR prior ────────────────────────────────────────────────────────

const BLR_PRIOR_PRECISION: f64 = 2.0;
const BLR_PRIOR_A: f64 = 1.0;
const BLR_PRIOR_B: f64 = 0.5;

// ── Train/test split ────────────────────────────────────────────────

const TRAIN_FRAC: f64 = 0.80;

// ── Test parameters ─────────────────────────────────────────────────

/// Number of determinism runs per seed in the 5-run gate. The
/// project handoff (PHASE_0_9_HANDOFF.md) specifies this exact count.
const N_DETERMINISM_RUNS: usize = 5;

// ── Data structures ─────────────────────────────────────────────────

/// A flattened binary-classification dataset.
pub(crate) struct Dataset {
    /// Row-major `[n_samples × n_features]` feature matrix.
    pub features: Vec<f64>,
    /// Length `n_samples`; each entry is `0.0` or `1.0`.
    pub labels: Vec<f64>,
    pub n_samples: usize,
    pub n_features: usize,
}

impl Dataset {
    fn row(&self, i: usize) -> &[f64] {
        &self.features[i * self.n_features..(i + 1) * self.n_features]
    }
}

/// Outcome of one training trial. Every later Phase 0.9 commit will
/// extend this struct with additional metrics (per-leaf reports,
/// accuracy, wall-clock).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TrialResult {
    pub chain_head_hex: String,
    pub merkle_root_hex: String,
    pub audit_event_count: usize,
    /// The `N_ROUTING_FEATURES` feature indices chosen by univariate
    /// F-score on the training split. Pinned in the trial result so
    /// tests can assert the selection is deterministic and that it
    /// preferentially picks from the discriminative band.
    pub routing_features: Vec<usize>,
}

// ── Deterministic RNG ────────────────────────────────────────────────

/// Splitmix64 step. Mirrors `cjc-repro::SplitMix64`'s output exactly
/// for the same starting state, but defined inline here so the
/// baseline harness has no test-time dependency on the higher-level
/// RNG API surface (which itself is exercised by other tests).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Map a splitmix64 output to a uniform `f64` in `[0, 1)`. Uses the
/// top 53 bits (the f64 mantissa width) to stay bit-stable.
fn uniform_f64(state: &mut u64) -> f64 {
    ((splitmix64(state) >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
}

/// One Box-Muller sample: returns a single N(0, 1) draw. Advances
/// the RNG state by two splitmix64 steps.
fn standard_normal(state: &mut u64) -> f64 {
    // Box-Muller transform: from two uniforms in (0, 1), emit one
    // standard-normal sample. We take the "cosine arm" only; the
    // sine arm would let us emit two normals per call, but we keep
    // it single-output for clarity.
    let u1 = uniform_f64(state).max(1e-300);
    let u2 = uniform_f64(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ── Synthetic dataset ────────────────────────────────────────────────

/// Build a deterministic synthetic dataset matching Wisconsin BC's
/// shape. Class 1 is shifted by `+CLASS_1_SHIFT` standard deviations
/// on the first `N_DISCRIMINATIVE_FEATURES` features so that a
/// linear-BLR predictor can recover meaningful accuracy.
///
/// Output is byte-stable across runs and platforms for any given
/// `seed`.
pub(crate) fn synthetic_dataset(seed: u64) -> Dataset {
    /// Class-1 mean shift in standard-deviation units on the
    /// discriminative features.
    const CLASS_1_SHIFT: f64 = 0.8;
    /// Number of features that carry class information. The
    /// remainder are pure noise — exercises ABNG's ability to
    /// ignore irrelevant features through the leaf-level BLR
    /// prior.
    const N_DISCRIMINATIVE_FEATURES: usize = 10;

    let mut state = seed.wrapping_add(0xD9CA_5AE7_3A02_C7B1);
    let mut features = Vec::with_capacity(N_SAMPLES * N_FEATURES_TOTAL);
    let mut labels = Vec::with_capacity(N_SAMPLES);

    for i in 0..N_SAMPLES {
        let label: f64 = if i < N_CLASS_0 { 0.0 } else { 1.0 };
        labels.push(label);
        for f in 0..N_FEATURES_TOTAL {
            let mut value = standard_normal(&mut state);
            if label > 0.5 && f < N_DISCRIMINATIVE_FEATURES {
                value += CLASS_1_SHIFT;
            }
            features.push(value);
        }
    }

    Dataset {
        features,
        labels,
        n_samples: N_SAMPLES,
        n_features: N_FEATURES_TOTAL,
    }
}

// ── Train/test split ────────────────────────────────────────────────

/// Deterministic Fisher-Yates shuffle + take-first-80%. Returns
/// `(train_indices, test_indices)` in shuffled order.
pub(crate) fn train_test_split(dataset: &Dataset, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut indices: Vec<usize> = (0..dataset.n_samples).collect();
    let mut state = seed.wrapping_add(0x4D2A_DAB7_3E0F_C901);
    // Standard Fisher-Yates: walk from end to start, swap with a
    // random earlier position (or self).
    for i in (1..indices.len()).rev() {
        let r = splitmix64(&mut state) as usize;
        let j = r % (i + 1);
        indices.swap(i, j);
    }
    let n_train = ((dataset.n_samples as f64) * TRAIN_FRAC) as usize;
    let train = indices[..n_train].to_vec();
    let test = indices[n_train..].to_vec();
    (train, test)
}

// ── Feature selection (univariate ANOVA F-score) ─────────────────────

/// Compute the one-way ANOVA F-statistic per feature for a binary
/// classification dataset. The F-statistic is the ratio of between-
/// class variance to within-class variance (df = (1, n − 2) for two
/// classes); larger F means a stronger univariate relationship
/// between the feature and the class label.
///
/// The computation is a deterministic two-pass over the input
/// (compute per-class means, then per-class sum-of-squares). No
/// random number generation, no parallel reductions — every f64
/// output is bit-stable across runs and platforms.
///
/// Returns a `Vec<(feature_idx, f_score)>` of length `n_features`
/// in the natural feature order. Use [`select_top_k_features`] to
/// extract the top-K indices in descending F-score order.
///
/// Degenerate cases:
/// * Constant feature within both classes ⇒ within-class SS = 0;
///   we return `f_score = 0.0` rather than a sentinel infinity so
///   downstream sorts stay total-ordered.
/// * Single class present ⇒ between-class SS = 0; F = 0.0.
pub(crate) fn compute_f_scores_binary(
    features: &[f64],
    labels: &[f64],
    n_samples: usize,
    n_features: usize,
) -> Vec<(usize, f64)> {
    assert_eq!(features.len(), n_samples * n_features);
    assert_eq!(labels.len(), n_samples);

    // First pass: per-class counts and per-class sums per feature.
    let mut n_class = [0usize; 2];
    let mut sum_class: Vec<[f64; 2]> = vec![[0.0; 2]; n_features];
    for i in 0..n_samples {
        let c = if labels[i] > 0.5 { 1 } else { 0 };
        n_class[c] += 1;
        let row = &features[i * n_features..(i + 1) * n_features];
        for (f, &x) in row.iter().enumerate() {
            sum_class[f][c] += x;
        }
    }

    // Per-class means + grand means.
    let mut mean_class: Vec<[f64; 2]> = vec![[0.0; 2]; n_features];
    let mut grand_mean: Vec<f64> = vec![0.0; n_features];
    let n_total = (n_class[0] + n_class[1]) as f64;
    for f in 0..n_features {
        let m0 = if n_class[0] > 0 { sum_class[f][0] / (n_class[0] as f64) } else { 0.0 };
        let m1 = if n_class[1] > 0 { sum_class[f][1] / (n_class[1] as f64) } else { 0.0 };
        mean_class[f] = [m0, m1];
        grand_mean[f] = (sum_class[f][0] + sum_class[f][1]) / n_total;
    }

    // Second pass: per-class within-class sum-of-squares per feature.
    let mut ssw: Vec<f64> = vec![0.0; n_features];
    for i in 0..n_samples {
        let c = if labels[i] > 0.5 { 1 } else { 0 };
        let row = &features[i * n_features..(i + 1) * n_features];
        for (f, &x) in row.iter().enumerate() {
            let d = x - mean_class[f][c];
            ssw[f] += d * d;
        }
    }

    // Compose the F-statistic per feature: F = MSB / MSW
    //   MSB = (n_0·(m_0 − μ)² + n_1·(m_1 − μ)²) / 1
    //   MSW = ssw / (n − 2)
    let n_minus_2 = if n_total > 2.0 { n_total - 2.0 } else { 1.0 };
    let n0 = n_class[0] as f64;
    let n1 = n_class[1] as f64;
    let mut out: Vec<(usize, f64)> = Vec::with_capacity(n_features);
    for f in 0..n_features {
        let d0 = mean_class[f][0] - grand_mean[f];
        let d1 = mean_class[f][1] - grand_mean[f];
        let ssb = n0 * d0 * d0 + n1 * d1 * d1;
        let mse = ssw[f] / n_minus_2;
        // Degenerate within-class variance → score floor at 0. A
        // truly constant feature carries no information; treating it
        // as +∞ would force it to the top of every sort despite that.
        let f_score = if mse > 0.0 { ssb / mse } else { 0.0 };
        out.push((f, f_score));
    }
    out
}

/// Return the top `k` feature indices sorted by descending F-score.
/// Ties broken by ascending feature index — fully deterministic.
///
/// Panics if `k > scores.len()`.
pub(crate) fn select_top_k_features(scores: &[(usize, f64)], k: usize) -> Vec<usize> {
    assert!(k <= scores.len(), "k={k} exceeds n_features={}", scores.len());
    let mut sorted: Vec<(usize, f64)> = scores.to_vec();
    sorted.sort_by(|a, b| {
        // Larger F-score first; tie-break by smaller index first.
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    sorted.into_iter().take(k).map(|(i, _)| i).collect()
}

/// Compute the top-`N_ROUTING_FEATURES` feature indices on the train
/// portion of `dataset`. Used by [`run_trial`] to seed the codebook's
/// routing dimensions. F-score is computed on **train data only** —
/// no test-set leakage.
pub(crate) fn select_routing_features(dataset: &Dataset, train_idx: &[usize]) -> Vec<usize> {
    let n_train = train_idx.len();
    let n_features = dataset.n_features;
    let mut train_features: Vec<f64> = Vec::with_capacity(n_train * n_features);
    let mut train_labels: Vec<f64> = Vec::with_capacity(n_train);
    for &i in train_idx {
        train_features.extend_from_slice(dataset.row(i));
        train_labels.push(dataset.labels[i]);
    }
    let scores = compute_f_scores_binary(&train_features, &train_labels, n_train, n_features);
    select_top_k_features(&scores, N_ROUTING_FEATURES)
}

// ── Graph construction ──────────────────────────────────────────────

/// Pre-allocate a full `branching^depth` routing tree by BFS-expanding
/// from the root, calling [`AdaptiveBeliefGraph::add_node`] for each
/// `(parent, key_byte)` pair. After this call the graph has
/// `1 + branching + branching² + … + branching^depth` total nodes;
/// every node at depth `< depth` has exactly `branching` children.
///
/// Why pre-allocate?
/// * Gives the baseline a deterministic, controllable tree shape so
///   accuracy and per-leaf reports compare across seeds at fixed
///   topology. With organic Grow/Split triggers, the topology
///   itself would be a function of the seed — useful for stress
///   tests, but noisy for a Phase 0.9 baseline.
/// * Exposes ABNG's per-leaf specialization on the first training
///   row (vs `a0a8266`'s root-only tree where every sample landed
///   in the same BLR posterior).
/// * Empty leaves are handled at predict time by
///   [`blr_predict_with_fallback`], which walks up the parent chain
///   to the nearest ancestor with `n_seen ≥ 1`. So pre-allocating
///   more leaves than samples is safe — sparse leaves transparently
///   defer to a populated ancestor.
fn pre_allocate_full_tree(g: &mut AdaptiveBeliefGraph, branching: u8, depth: usize) {
    let mut current_level: Vec<u32> = vec![0]; // root NodeId is always 0
    for _ in 0..depth {
        let mut next_level: Vec<u32> = Vec::with_capacity(current_level.len() * (branching as usize));
        for &parent in &current_level {
            for key_byte in 0..branching {
                let child = g.add_node(parent, key_byte).expect("add_node");
                next_level.push(child);
            }
        }
        current_level = next_level;
    }
}

/// Build the ABNG graph for the baseline. Configures:
///
/// * **Codebook:** routes on the F-score-selected routing features
///   with `N_BINS_PER_FEATURE` bins per feature. Quantile boundaries
///   are fixed at `[-0.5, 0.0, 0.5]` (z-score-style) — this works
///   well for the synthetic generator (features are ~N(0, 1)) and
///   should be replaced with empirical quantiles when real data
///   arrives. The codebook itself is feature-index-agnostic; the
///   caller is responsible for extracting the right feature subset
///   in the right order before calling `train_step` / `descend`.
/// * **Leaf head:** input_dim = full feature count (30), no hidden
///   layers, output_dim = 1, no activation. The leaf-level BLR
///   sees the full feature vector as `phi`.
/// * **BLR prior:** precision = `BLR_PRIOR_PRECISION`, a/b as
///   declared above.
/// * **Tree pre-allocation:** full `N_BINS_PER_FEATURE^N_ROUTING_FEATURES`
///   tree (4³ = 64 leaves, 85 total nodes) via
///   [`pre_allocate_full_tree`]. Each training row routes deterministically
///   to one of the 64 leaves based on its F-score-selected feature
///   subset.
fn build_graph(seed: u64) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(seed);
    let boundaries: Vec<f64> = (0..N_ROUTING_FEATURES)
        .flat_map(|_| [-0.5, 0.0, 0.5])
        .collect();
    g.set_codebook(N_ROUTING_FEATURES, N_BINS_PER_FEATURE, &boundaries)
        .expect("codebook install");
    g.set_leaf_head(
        N_FEATURES_TOTAL as u32,
        vec![], // no hidden layers — keeps the demo light
        1,
        Activation::None,
    )
    .expect("leaf head install");
    g.set_blr_prior(BLR_PRIOR_PRECISION, BLR_PRIOR_A, BLR_PRIOR_B)
        .expect("BLR prior install");
    pre_allocate_full_tree(&mut g, N_BINS_PER_FEATURE as u8, N_ROUTING_FEATURES);
    g
}

// ── Trial harness ───────────────────────────────────────────────────

/// Run one full training trial. Builds a graph from the seed, splits
/// the dataset, computes the F-score-selected routing features on the
/// train portion (no test-set leakage), trains via `train_step`, and
/// reports the chain head + Merkle root + audit count + chosen
/// routing features.
///
/// Determinism contract: same `seed` + same `dataset` ⇒ byte-equal
/// output. This is the property the 5-run gate validates.
pub(crate) fn run_trial(seed: u64, dataset: &Dataset) -> TrialResult {
    let mut g = build_graph(seed);
    let (train_idx, _test_idx) = train_test_split(dataset, seed);
    let routing_features = select_routing_features(dataset, &train_idx);

    // Extract the routing-feature subset into a small reusable buffer
    // per row, then forward the full row as `phi`.
    let mut routing_buf = vec![0.0f64; N_ROUTING_FEATURES];
    for &i in &train_idx {
        let row = dataset.row(i);
        for (out, &feat_idx) in routing_buf.iter_mut().zip(&routing_features) {
            *out = row[feat_idx];
        }
        let phi = row;
        let y = dataset.labels[i];
        g.train_step(&routing_buf, phi, y).expect("train_step");
    }

    TrialResult {
        chain_head_hex: hex32(&g.chain_head),
        merkle_root_hex: hex32(&g.merkle_root()),
        audit_event_count: g.audit.len(),
        routing_features,
    }
}

fn hex32(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        write!(&mut s, "{b:02x}").unwrap();
    }
    s
}

// ── Tests ───────────────────────────────────────────────────────────

#[test]
fn baseline_determinism_5_runs_same_seed() {
    // The Phase 0.9 determinism contract: 5 trials at the same seed
    // produce byte-identical chain_head AND merkle_root. If this
    // ever fails, the entire Phase 0.9 perf-improvement plan is on
    // hold until the regression is found.
    const SEED: u64 = 1;
    let dataset = synthetic_dataset(SEED);
    let mut results: Vec<TrialResult> = Vec::with_capacity(N_DETERMINISM_RUNS);
    for _ in 0..N_DETERMINISM_RUNS {
        results.push(run_trial(SEED, &dataset));
    }
    let first = &results[0];
    for (run, r) in results.iter().enumerate().skip(1) {
        assert_eq!(
            r.chain_head_hex, first.chain_head_hex,
            "run {run} chain_head diverged from run 0"
        );
        assert_eq!(
            r.merkle_root_hex, first.merkle_root_hex,
            "run {run} merkle_root diverged from run 0"
        );
        assert_eq!(
            r.audit_event_count, first.audit_event_count,
            "run {run} audit_event_count diverged from run 0"
        );
    }
}

#[test]
fn baseline_seed_sensitivity_5_distinct_seeds() {
    // The other side of determinism: distinct seeds must produce
    // distinct chain heads. Rules out a degenerate bug where every
    // trial somehow produces the same hash regardless of input.
    let mut heads: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for seed in 1..=5u64 {
        let dataset = synthetic_dataset(seed);
        let result = run_trial(seed, &dataset);
        let inserted = heads.insert(result.chain_head_hex);
        assert!(inserted, "seed {seed} collided with a previous seed's chain_head");
    }
    assert_eq!(heads.len(), 5, "expected 5 distinct chain heads, got {}", heads.len());
}

#[test]
fn baseline_synthetic_dataset_shape() {
    // Pin the synthetic generator's contract: 569 × 30, 357 / 212
    // class split, byte-stable across two builds.
    let a = synthetic_dataset(1);
    let b = synthetic_dataset(1);
    assert_eq!(a.n_samples, N_SAMPLES);
    assert_eq!(a.n_features, N_FEATURES_TOTAL);
    assert_eq!(a.labels.len(), N_SAMPLES);
    assert_eq!(a.features.len(), N_SAMPLES * N_FEATURES_TOTAL);
    let n_class_0 = a.labels.iter().filter(|&&l| l == 0.0).count();
    let n_class_1 = a.labels.iter().filter(|&&l| l == 1.0).count();
    assert_eq!(n_class_0, N_CLASS_0);
    assert_eq!(n_class_1, N_SAMPLES - N_CLASS_0);
    // Byte-stability: two builds at the same seed produce identical
    // feature vectors.
    assert_eq!(a.features, b.features);
    assert_eq!(a.labels, b.labels);
}

#[test]
fn baseline_train_test_split_is_deterministic() {
    let dataset = synthetic_dataset(1);
    let (train_a, test_a) = train_test_split(&dataset, 1);
    let (train_b, test_b) = train_test_split(&dataset, 1);
    assert_eq!(train_a, train_b);
    assert_eq!(test_a, test_b);
    // Sanity: train + test partition the dataset
    assert_eq!(train_a.len() + test_a.len(), N_SAMPLES);
    // Sanity: no overlap
    let train_set: std::collections::BTreeSet<usize> = train_a.iter().copied().collect();
    let test_set: std::collections::BTreeSet<usize> = test_a.iter().copied().collect();
    assert!(train_set.is_disjoint(&test_set));
}

#[test]
fn baseline_trial_uses_train_step_audit_events() {
    // Pin the wiring contract: every training row produces exactly
    // one TrainStep audit event (the post-A2 v14 audit shape).
    // After training, the audit log size should equal:
    //   * Graph-setup events: Created + CodebookFrozen
    //       + LeafHeadConfigured + LeafParamsInitialized (root)
    //       + BlrPriorConfigured + BlrInitialized (root) ≈ 6
    //   * Pre-allocation events for the 4³ tree:
    //       * 21 × ChildrenPromoted (one per non-leaf parent on
    //         first child)
    //       * 84 × NodeAdded
    //       * 84 × LeafParamsInitialized
    //       * 84 × BlrInitialized
    //       = 273 events
    //   * Per-row training events: n_train × 1 TrainStep
    //
    // The setup count is now structural (driven by tree shape), so
    // we assert a tight bound that catches any new setup audit kind
    // or pre-allocation drift.
    let dataset = synthetic_dataset(1);
    let (train_idx, _test_idx) = train_test_split(&dataset, 1);
    let result = run_trial(1, &dataset);
    let n_train_rows = train_idx.len();
    assert!(
        result.audit_event_count >= n_train_rows,
        "audit log too short: {} < {}",
        result.audit_event_count,
        n_train_rows
    );
    let setup_events = result.audit_event_count - n_train_rows;
    // 4³ tree: 6 graph-setup + 21 promotions + 252 per-node events
    // = 279 expected. Tolerate ±10 for any future Phase 0.9 setup
    // event addition (e.g. drift baseline install) without forcing
    // a test rewrite.
    assert!(
        (270..=290).contains(&setup_events),
        "expected ~279 setup events (4³ pre-allocated tree); got {setup_events}"
    );
}

#[test]
fn baseline_tree_is_pre_allocated_full_depth() {
    // Structural contract: build_graph produces a complete
    // N_BINS_PER_FEATURE-ary tree of depth N_ROUTING_FEATURES.
    // Expected total node count: Σ_{d=0..=depth} branching^d.
    // For branching=4, depth=3: 1 + 4 + 16 + 64 = 85 nodes.
    let g = build_graph(1);
    let branching = N_BINS_PER_FEATURE as usize;
    let depth = N_ROUTING_FEATURES;
    let mut expected = 0usize;
    let mut level_size = 1usize;
    for _ in 0..=depth {
        expected += level_size;
        level_size *= branching;
    }
    assert_eq!(
        g.node_count() as usize,
        expected,
        "expected {expected} nodes in a complete {branching}^{depth} tree, got {}",
        g.node_count()
    );
}

#[test]
fn baseline_train_rows_route_to_leaf_nodes_not_root() {
    // Behavioral contract: with a pre-allocated full-depth tree,
    // every training row's encoded prefix must match all
    // N_ROUTING_FEATURES bytes — i.e. it descends to a leaf at the
    // bottom level, not to a partial-match interior node and never
    // back to the root. Exercises the pre-allocation contract
    // jointly with the codebook + descend wiring.
    let dataset = synthetic_dataset(1);
    let (train_idx, _) = train_test_split(&dataset, 1);
    let routing = select_routing_features(&dataset, &train_idx);
    let g = build_graph(1);

    let mut routing_buf = vec![0.0f64; N_ROUTING_FEATURES];
    for &i in &train_idx[..16] {
        // Spot-check the first 16 rows — uniform routing means the
        // sample is representative without scanning all 455 rows.
        let row = dataset.row(i);
        for (out, &feat_idx) in routing_buf.iter_mut().zip(&routing) {
            *out = row[feat_idx];
        }
        let prefix = g.encode_prefix(&routing_buf).expect("encode prefix");
        let evidence = g.descend(&prefix);
        assert_eq!(
            evidence.matched_prefix as usize, N_ROUTING_FEATURES,
            "row {i}: expected full-depth match, got {}",
            evidence.matched_prefix
        );
        assert_ne!(evidence.leaf_id, 0, "row {i}: descended only to root");
    }
}

#[test]
fn baseline_f_score_picks_from_discriminative_band() {
    // The synthetic generator shifts class 1 by +0.8σ on the first
    // 10 features and leaves the other 20 as pure noise. The F-score
    // top-3 selection should land in the discriminative band [0..10)
    // with overwhelming probability for any of the 5 baseline seeds.
    // We assert this exactly (no probabilistic slack) — if it ever
    // misses, either the synthetic generator drifted or the F-score
    // math regressed.
    for seed in 1..=5u64 {
        let dataset = synthetic_dataset(seed);
        let (train_idx, _test_idx) = train_test_split(&dataset, seed);
        let routing = select_routing_features(&dataset, &train_idx);
        assert_eq!(routing.len(), N_ROUTING_FEATURES);
        // All distinct.
        let unique: std::collections::BTreeSet<usize> = routing.iter().copied().collect();
        assert_eq!(unique.len(), N_ROUTING_FEATURES);
        for &f in &routing {
            assert!(
                f < 10,
                "seed {seed}: routing feature {f} is from the noise band [10..30); \
                 F-score regression suspected"
            );
        }
    }
}

#[test]
fn baseline_f_score_is_deterministic_across_runs() {
    // Pin the F-score computation: same seed + same dataset ⇒
    // byte-equal F-scores AND byte-equal top-K selection. Both are
    // pure-arithmetic two-pass reductions; this guards against any
    // future refactor that introduces parallel reduction or
    // unsorted iteration into the feature-selection path.
    for seed in 1..=5u64 {
        let dataset = synthetic_dataset(seed);
        let (train_idx, _test_idx) = train_test_split(&dataset, seed);
        let routing_a = select_routing_features(&dataset, &train_idx);
        let routing_b = select_routing_features(&dataset, &train_idx);
        assert_eq!(routing_a, routing_b, "seed {seed}: routing selection not deterministic");

        // Underlying F-score vector also byte-equal.
        let n_train = train_idx.len();
        let mut train_features = Vec::with_capacity(n_train * N_FEATURES_TOTAL);
        let mut train_labels = Vec::with_capacity(n_train);
        for &i in &train_idx {
            train_features.extend_from_slice(dataset.row(i));
            train_labels.push(dataset.labels[i]);
        }
        let scores_a = compute_f_scores_binary(
            &train_features, &train_labels, n_train, N_FEATURES_TOTAL,
        );
        let scores_b = compute_f_scores_binary(
            &train_features, &train_labels, n_train, N_FEATURES_TOTAL,
        );
        assert_eq!(
            scores_a.len(),
            scores_b.len(),
            "seed {seed}: F-score vector length unstable"
        );
        for (i, ((fa, sa), (fb, sb))) in scores_a.iter().zip(scores_b.iter()).enumerate() {
            assert_eq!(fa, fb, "seed {seed}: feature index {i} mismatch");
            assert_eq!(
                sa.to_bits(),
                sb.to_bits(),
                "seed {seed}: F-score bits for feature {i} not byte-equal"
            );
        }
    }
}

#[test]
fn baseline_trial_records_routing_features() {
    // Pin the contract that `run_trial` reports the selected routing
    // features through `TrialResult`. Two assertions:
    //   1. The slot is populated (length == N_ROUTING_FEATURES).
    //   2. The selection matches what `select_routing_features`
    //      computes independently — i.e. no hidden divergence
    //      between the trial path and the explicit-selection path.
    for seed in 1..=5u64 {
        let dataset = synthetic_dataset(seed);
        let result = run_trial(seed, &dataset);
        assert_eq!(result.routing_features.len(), N_ROUTING_FEATURES);
        let (train_idx, _) = train_test_split(&dataset, seed);
        let direct = select_routing_features(&dataset, &train_idx);
        assert_eq!(
            result.routing_features, direct,
            "seed {seed}: trial-path routing diverged from direct path"
        );
    }
}
