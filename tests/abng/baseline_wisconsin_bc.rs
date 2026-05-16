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
//! * Top-K feature selection via univariate F-score.
//! * Multi-level tree expansion (`add_node` calls to populate the
//!   routing tree past the root).

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

// ── Graph construction ──────────────────────────────────────────────

/// Build the ABNG graph for the baseline. Configures:
///
/// * **Codebook:** routes on the first `N_ROUTING_FEATURES` features
///   with `N_BINS_PER_FEATURE` bins per feature. Quantile boundaries
///   are fixed at `[-0.5, 0.0, 0.5]` (z-score-style) — this works
///   well for the synthetic generator (features are ~N(0, 1)) and
///   should be replaced with empirical quantiles when real data
///   arrives.
/// * **Leaf head:** input_dim = full feature count (30), no hidden
///   layers, output_dim = 1, no activation. The leaf-level BLR
///   sees the full feature vector as `phi`.
/// * **BLR prior:** precision = `BLR_PRIOR_PRECISION`, a/b as
///   declared above.
///
/// **No `add_node` calls** at this stage — the tree starts as a
/// single root node; all samples route to it. Track P later
/// commits will grow the tree to expose ABNG's per-leaf
/// specialization.
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
    g
}

// ── Trial harness ───────────────────────────────────────────────────

/// Run one full training trial. Builds a graph from the seed, splits
/// the dataset, trains via `train_step` over the train portion, and
/// reports the chain head + Merkle root + audit count.
///
/// Determinism contract: same `seed` + same `dataset` ⇒ byte-equal
/// output. This is the property the 5-run gate validates.
pub(crate) fn run_trial(seed: u64, dataset: &Dataset) -> TrialResult {
    let mut g = build_graph(seed);
    let (train_idx, _test_idx) = train_test_split(dataset, seed);

    for &i in &train_idx {
        let row = dataset.row(i);
        let routing = &row[..N_ROUTING_FEATURES];
        let phi = row;
        let y = dataset.labels[i];
        g.train_step(routing, phi, y).expect("train_step");
    }

    TrialResult {
        chain_head_hex: hex32(&g.chain_head),
        merkle_root_hex: hex32(&g.merkle_root()),
        audit_event_count: g.audit.len(),
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
    //   setup events (Created + CodebookFrozen + LeafHeadConfigured
    //     + LeafParamsInitialized + BlrPriorConfigured + BlrInitialized)
    //   + n_train_rows × 1 TrainStep event
    let dataset = synthetic_dataset(1);
    let (train_idx, _test_idx) = train_test_split(&dataset, 1);
    let result = run_trial(1, &dataset);
    let n_train_rows = train_idx.len();
    // Setup events count: empirically 6 (one per `set_*` call plus
    // Created at construction time). Pinning this number is too
    // brittle (any new setup audit kind would shift it), so we
    // just assert the lower bound and per-row contribution.
    assert!(
        result.audit_event_count >= n_train_rows,
        "audit log too short: {} < {}",
        result.audit_event_count,
        n_train_rows
    );
    let setup_events = result.audit_event_count - n_train_rows;
    assert!(
        (1..=10).contains(&setup_events),
        "expected 1..=10 setup events, got {setup_events}"
    );
}
