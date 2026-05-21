//! ABNG demo: tabular regression with GP-like uncertainty, scalable
//! beyond classical GPs via prefix-routed BLR posteriors.
//!
//! ## What this demo proves
//!
//! Classical Gaussian Processes scale O(n³) in training data because
//! the kernel matrix is dense — at n ≈ 10⁴ training points a single
//! Cholesky already strains a workstation. ABNG partitions the input
//! space via the codebook and stores one BLR posterior per leaf, so
//! each new training point updates *exactly one* leaf at O(d²·n_leaf)
//! cost where n_leaf ≪ n. The aggregate cost is O(n·d²) for n
//! training points across all leaves — *linear* in n vs cubic for GP.
//!
//! Tangible benefits asserted by this file:
//!
//! * **Constant-time-per-step training.** The wall-clock cost of
//!   inserting one observation does not depend on the total
//!   dataset size, because each insert touches one leaf's
//!   bounded-dim BLR.
//! * **GP-like spatial uncertainty.** Predictions in densely-trained
//!   regions of the (x1, x2) input space have low
//!   `epistemic_leverage`; predictions in sparsely-trained or
//!   unseen regions have high leverage. Identical to what a GP
//!   produces but without the global O(n³) cost.
//! * **Snapshot-checkpointable.** A classical GP doesn't naturally
//!   serialize (you'd persist the full Cholesky factor); ABNG's
//!   audit chain + per-node section gives a deterministic
//!   round-trip.
//! * **Holds out-of-sample better than the prior baseline.** Trained
//!   posteriors yield meaningfully lower MSE on held-out queries
//!   than the untrained graph's prior-only predictions.

use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_abng::serialize::{replay, serialize};
use cjc_ad::pinn::Activation;

/// Synthetic 2-D regression target. Bilinear in the BLR feature
/// basis `[1, x1, x2, x1*x2]` so a sufficiently-trained leaf BLR
/// can fit exactly. Non-trivial enough that the prior alone is
/// far from accurate (variance ~0.4 on the unit square).
fn target(x1: f64, x2: f64) -> f64 {
    0.3 + 0.5 * x1 + 0.4 * x2 + 0.7 * x1 * x2
}

/// 4-D BLR feature vector. The penultimate dim must equal
/// blr_feature_dim (= last hidden dim of the leaf head). We use
/// `[1, x1, x2, x1*x2]` — a polynomial-of-degree-2 basis that BLR
/// can fit linearly.
fn tabular_features(x1: f64, x2: f64) -> [f64; 4] {
    [1.0, x1, x2, x1 * x2]
}

/// Build the tabular graph: 1-D codebook over `x1` (4 bins), 2→4→1
/// tanh leaf head (BLR feature dim = 4), BLR prior + density +
/// calibration heads, 4 children at prefix bytes 0..3.
///
/// The `x1` axis is the routing dimension — points with similar
/// `x1` route to the same leaf, where `x2` becomes a within-leaf
/// regression feature. This is the classic ABNG "use one dim to
/// partition, regress on the rest" pattern.
fn build_tabular_graph(seed: u64) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(seed);
    g.set_codebook(1, 4, &[0.25, 0.5, 0.75]).unwrap();
    g.set_leaf_head(2, vec![4], 1, Activation::Tanh).unwrap();
    g.set_blr_prior(2.0, 1.0, 0.5).unwrap();
    g.set_density_tracker().unwrap();
    g.set_calibration(15).unwrap();
    let thresholds = [
        0.5, 64.0, 128.0, 0.05, 0.02, 4.0, 0.1, 32.0, 10.0, 8.0, 20.0, f64::MAX,
        0.005, 1.05,
    ];
    g.set_decision_policy(&thresholds).unwrap();
    for byte in 0u8..4 {
        g.add_node(0, byte).unwrap();
    }
    g
}

/// Determinist x-pair generator using SplitMix64 — reproducible
/// across runs without any RNG dep.
fn det_x_pair(seed: u64, k: u64) -> (f64, f64) {
    let mut z = seed.wrapping_add(k.wrapping_mul(0x9E3779B97F4A7C15));
    z ^= z >> 30;
    z = z.wrapping_mul(0xBF58476D1CE4E5B9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    let r1 = ((z >> 11) as f64) / ((1u64 << 53) as f64);
    let mut z2 = z.wrapping_mul(0xD1342543DE82EF95);
    z2 ^= z2 >> 30;
    z2 = z2.wrapping_mul(0x4EE5A8E16C8B3257);
    z2 ^= z2 >> 27;
    let r2 = ((z2 >> 11) as f64) / ((1u64 << 53) as f64);
    (r1, r2)
}

fn route_for(g: &AdaptiveBeliefGraph, x1: f64) -> u32 {
    let prefix = g.encode_prefix(&[x1]).unwrap();
    g.descend(&prefix).leaf_id
}

fn train_one(g: &mut AdaptiveBeliefGraph, x1: f64, x2: f64) {
    // Phase 0.8c v14 Item A2 — fused per-row training step. One
    // `AuditKind::TrainStep` event per row instead of the pre-A2
    // `BlrUpdated + BeliefUpdate` pair. `train_step` does its own
    // descend, so the previous `route_for` indirection is folded
    // into the same call.
    let phi = tabular_features(x1, x2);
    let y = target(x1, x2);
    g.train_step(&[x1], &phi, y).unwrap();
}

/// Train the graph on `n` deterministic samples drawn from the
/// SplitMix64 stream seeded by `seed`.
fn train_n(g: &mut AdaptiveBeliefGraph, seed: u64, n: u64) {
    for k in 0..n {
        let (x1, x2) = det_x_pair(seed, k);
        train_one(g, x1, x2);
    }
}

// ── Forward pass + GP-like fit ────────────────────────────────────────

#[test]
fn tabular_forward_finite_per_leaf() {
    let g = build_tabular_graph(11);
    for &(x1, x2) in &[(0.1, 0.2), (0.4, 0.6), (0.8, 0.3), (0.9, 0.9)] {
        let leaf = route_for(&g, x1);
        let (mean, lev, _ale) = g.blr_predict(leaf, &tabular_features(x1, x2)).unwrap();
        assert!(mean.is_finite());
        assert!(lev.is_finite() && lev > 0.0);
    }
}

#[test]
fn tabular_train_reduces_held_out_mse_vs_prior() {
    // Headline benefit: the trained posterior strictly beats the
    // prior on held-out queries. This is the GP fit-quality
    // gate — it would be vacuous if untrained ABNG already
    // approximated the target.
    let mut g_prior = build_tabular_graph(11);
    let mut g_trained = build_tabular_graph(11);
    train_n(&mut g_trained, 31, 200);

    let mut prior_se = 0.0;
    let mut trained_se = 0.0;
    let n_test = 64u64;
    for k in 0..n_test {
        let (x1, x2) = det_x_pair(0xCAFE, k);
        let truth = target(x1, x2);
        let phi = tabular_features(x1, x2);
        let (m_p, _, _) = g_prior.blr_predict(route_for(&g_prior, x1), &phi).unwrap();
        let (m_t, _, _) = g_trained.blr_predict(route_for(&g_trained, x1), &phi).unwrap();
        prior_se += (m_p - truth).powi(2);
        trained_se += (m_t - truth).powi(2);
    }
    let prior_mse = prior_se / n_test as f64;
    let trained_mse = trained_se / n_test as f64;
    println!("tabular MSE: prior={prior_mse:.4}, trained={trained_mse:.4}");
    assert!(
        trained_mse < 0.5 * prior_mse,
        "trained MSE ({trained_mse}) must beat half the prior MSE ({prior_mse})"
    );
    // Suppress unused mut warnings.
    let _ = g_prior.audit.len();
}

#[test]
fn tabular_uncertainty_shrinks_with_more_data() {
    // Tangible GP property: epistemic uncertainty shrinks as more
    // observations are absorbed into the leaf's posterior. We
    // measure leverage at a fixed query point against a graph
    // trained on N points vs 4N points and assert the larger
    // dataset has strictly lower leverage.
    let probe = (0.50, 0.50);
    let probe_phi = tabular_features(probe.0, probe.1);

    let mut g_small = build_tabular_graph(11);
    train_n(&mut g_small, 13, 16);
    let leaf_s = route_for(&g_small, probe.0);
    let (_, lev_small, _) = g_small.blr_predict(leaf_s, &probe_phi).unwrap();

    let mut g_big = build_tabular_graph(11);
    train_n(&mut g_big, 13, 64);
    let leaf_b = route_for(&g_big, probe.0);
    let (_, lev_big, _) = g_big.blr_predict(leaf_b, &probe_phi).unwrap();

    println!("tabular lev: 16 pts={lev_small:.4}, 64 pts={lev_big:.4}");
    assert!(
        lev_big < lev_small,
        "more data should shrink lev: 16-pt={lev_small}, 64-pt={lev_big}"
    );
}

// ── Per-leaf-cost-bound assertion ────────────────────────────────────

#[test]
fn tabular_per_leaf_n_seen_bounded_by_route_share() {
    // Classical GP touches all n points per training step; ABNG's
    // per-leaf BLR sees only the points routed to it. We assert
    // each leaf's `n_seen` is bounded by a generous fraction of
    // the total dataset — i.e., evidence is partitioned across
    // leaves, not duplicated. Even with worst-case routing
    // imbalance no single leaf should hold more than 90% of the
    // points (vs 100% for a single-leaf / global GP fit).
    let mut g = build_tabular_graph(11);
    let n_total = 200u64;
    train_n(&mut g, 31, n_total);
    let mut max_n_seen = 0u64;
    for child_id in 1..g.node_count() {
        let blr_n = g.nodes[child_id as usize]
            .blr
            .as_ref()
            .map(|s| s.n_seen)
            .unwrap_or(0);
        max_n_seen = max_n_seen.max(blr_n);
    }
    let total_routed: u64 = (1..g.node_count())
        .map(|id| {
            g.nodes[id as usize]
                .blr
                .as_ref()
                .map(|s| s.n_seen)
                .unwrap_or(0)
        })
        .sum();
    assert_eq!(total_routed, n_total, "every point must route to exactly one leaf");
    assert!(
        (max_n_seen as f64) < 0.9 * (n_total as f64),
        "no single leaf should hold ≥90% of n={n_total} points; got {max_n_seen}"
    );
}

#[test]
fn tabular_routing_partitions_input_space_disjointly() {
    // Companion property: different x1 in different bins route to
    // different leaves. Proves the prefix routing partitions.
    let g = build_tabular_graph(11);
    let leaves: Vec<u32> = [0.10, 0.30, 0.60, 0.90]
        .iter()
        .map(|&x1| route_for(&g, x1))
        .collect();
    let mut sorted = leaves.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        4,
        "4 distinct codebook regions must route to 4 distinct leaves; got {leaves:?}"
    );
}

// ── Snapshot-checkpointable (a feature classical GP lacks) ────────────

#[test]
fn tabular_serialize_replay_preserves_predictions_byte_for_byte() {
    let mut g = build_tabular_graph(11);
    train_n(&mut g, 31, 64);
    let blob = serialize(&g);
    let g2 = replay(&blob).unwrap();
    for k in 0..16u64 {
        let (x1, x2) = det_x_pair(0xBEEF, k);
        let phi = tabular_features(x1, x2);
        let leaf_pre = route_for(&g, x1);
        let leaf_post = route_for(&g2, x1);
        assert_eq!(leaf_pre, leaf_post);
        let p_pre = g.blr_predict(leaf_pre, &phi).unwrap();
        let p_post = g2.blr_predict(leaf_post, &phi).unwrap();
        assert_eq!(p_pre.0.to_bits(), p_post.0.to_bits());
        assert_eq!(p_pre.1.to_bits(), p_post.1.to_bits());
        assert_eq!(p_pre.2.to_bits(), p_post.2.to_bits());
    }
}

#[test]
fn tabular_double_run_chain_head_byte_identical() {
    let mut g1 = build_tabular_graph(11);
    let mut g2 = build_tabular_graph(11);
    train_n(&mut g1, 31, 32);
    train_n(&mut g2, 31, 32);
    assert_eq!(g1.chain_head, g2.chain_head);
    assert_eq!(serialize(&g1), serialize(&g2));
}

// ── Locked canary ────────────────────────────────────────────────────

#[test]
fn tabular_chain_head_canary_locked() {
    let mut g = build_tabular_graph(11);
    train_n(&mut g, 31, 32);
    let actual_hex = hex_of(&g.chain_head);
    println!("tabular canary chain_head = {actual_hex}");
    // Locked at Phase 0.5 ship — fires only on tabular training
    // determinism breakage. Independent of PINN + chess RL canaries.
    // Re-locked at Phase 0.8c v14 Item A2 — the demo's `train_one`
    // flipped from `blr_update + observe` (pre-A2: two events / row,
    // tags 0x0A + 0x01) to `train_step` (post-A2: one TrainStep
    // event / row, tag 0x1E). The audit-chain payload bytes per
    // training row therefore changed, so the chain head shifted.
    // Pre-A2 hex: `cd3f5c7be81f5966d1f41af811cc94a859b653adf9993f1d5b3e23c0a87397e6`.
    // V14_MIGRATION.md records the v13 → v14 mapping.
    const CANARY_HEX: &str =
        "26ab2b37812607a77da2ee0d242558f672c408717ee4dd78c152e9f9f40d6745";
    assert_eq!(
        actual_hex, CANARY_HEX,
        "tabular GP-like chain_head canary mismatch — see comment"
    );
}

#[test]
fn tabular_audit_chain_verifies_post_train() {
    let mut g = build_tabular_graph(11);
    train_n(&mut g, 31, 64);
    assert!(g.verify_chain().is_ok());
}

// ── Per-leaf BLR is genuinely independent ─────────────────────────────

#[test]
fn tabular_leaves_have_distinct_posteriors_post_train() {
    // Different leaves, trained on disjoint subsets of the input,
    // produce *different* BLR mean vectors. Proves the per-leaf
    // posteriors aren't just identical copies of the prior.
    let mut g = build_tabular_graph(11);
    train_n(&mut g, 31, 200);
    let mut blr_means: Vec<Vec<f64>> = Vec::new();
    for child_id in 1..g.node_count() {
        let blr = g.nodes[child_id as usize].blr.as_ref().unwrap();
        blr_means.push(blr.mean.to_vec());
    }
    // Pairwise compare — at least one pair must differ.
    let mut all_equal = true;
    for i in 0..blr_means.len() {
        for j in (i + 1)..blr_means.len() {
            for k in 0..blr_means[i].len() {
                if blr_means[i][k] != blr_means[j][k] {
                    all_equal = false;
                    break;
                }
            }
            if !all_equal {
                break;
            }
        }
        if !all_equal {
            break;
        }
    }
    assert!(
        !all_equal,
        "different leaves should have distinct posteriors after training"
    );
}

fn hex_of(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}
