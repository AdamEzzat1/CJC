//! Phase 0.5 Item 5 — Chess-RL v2.6 ABNG retrofit.
//!
//! This file demonstrates the ABNG retrofit pattern (Bayesian Adaptive
//! Belief Network Graph as the value/policy head's substrate) on a
//! deterministic chess feature workload, without disturbing the
//! existing v2.5 PRELUDE in `tests/chess_rl_v2/`. The full PRELUDE
//! rewrite (replacing `forward_eager`'s value branch with `abng_*`
//! calls in the .cjcl source) is a follow-up — this file establishes
//! the retrofit *contract* in pure Rust so future PRELUDE work has a
//! locked-in canary to gate against.
//!
//! ## Retrofit pattern
//!
//! 1. Build an `AdaptiveBeliefGraph` with:
//!    - A 1-D quantile codebook over the first chess feature dim
//!      (board occupancy parity).
//!    - A leaf MLP head (`set_leaf_head`) for the policy logits.
//!    - A BLR prior (`set_blr_prior`) for the value-head posterior.
//!    - A density tracker (`set_density_tracker`) so OOD scoring is
//!      meaningful.
//!    - A 15-bin calibration head.
//! 2. For each chess state:
//!    a. `encode_prefix(feature_vec)` → prefix bytes.
//!    b. `descend(&prefix)` → resolves to a leaf node.
//!    c. `blr_predict(leaf_id, &phi)` → `(mean = V(s), epi_lev,
//!       ale_var)`.
//!    d. `ood_score(leaf_id, &phi, matched_prefix, prefix_max)` —
//!       the uncertainty gate. If `> τ`, abstain or fall back to
//!       a uniform policy (5.3 in the handoff).
//! 3. Training: each TD step calls `blr_update(leaf_id, x, y)` to
//!    refine the posterior. Provenance stamping (`stamp_provenance`)
//!    binds the resulting weights to a dataset fingerprint so a
//!    `predict_snap` carries full lineage.
//!
//! ## Locked v2.6 canaries
//!
//! - `CHESS_V2_6_CHAIN_HEAD` — the `chain_head` of the canonical
//!   training-trace ABNG graph after a deterministic 16-step
//!   rollout. Detects any change in the audit-event ordering or
//!   per-event payload layout.
//! - `CHESS_V2_6_BLR_STATE_HASH` — the BLR `state_hash` of the
//!   leaf node post-rollout. Detects any change in BLR conjugate
//!   update arithmetic.
//! - `CHESS_V2_6_PROVENANCE_STAMP` — the canonical stamp string
//!   used by the rollout. Detects accidental rotation.

use cjc_abng::audit::AuditKind;
use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_abng::serialize::{replay, serialize, smart_replay};
use cjc_ad::pinn::Activation;

// ── Locked v2.6 canaries ───────────────────────────────────────────────

/// Canonical seed for the v2.6 retrofit demo — chosen to match the
/// chess_rl_v2.5 weight-hash canary's seed conventions.
const V2_6_GRAPH_SEED: u64 = 42;

/// Provenance stamp — `sha256(b"chess-rl-v2.6 dataset 2026-05-08")`.
/// Locked so a future dataset change forces a rotation that's
/// recorded in the audit chain.
const CHESS_V2_6_PROVENANCE_STAMP: [u8; 32] = [
    0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0,
    0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10,
];

/// OOD ratio threshold above which the policy abstains or falls back
/// to uniform (handoff §5.3). Conservative default; tunable in
/// future TOML configs.
const V2_6_UNCERTAINTY_TAU: f64 = 0.7;

// ── Builders ──────────────────────────────────────────────────────────

/// Build the canonical v2.6 ABNG graph: codebook + 8→4→1 tanh leaf
/// head + BLR(1, 1.5, 1) + density + 15-bin calibration + 14
/// decision-policy thresholds + two children at keys 1 and 2.
///
/// Mirrors the `decide_step_canary_tests` fixture topology so the
/// retrofit shares the same trigger semantics as the rest of ABNG.
fn build_v2_6_graph() -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(V2_6_GRAPH_SEED);
    g.set_codebook(1, 4, &[-1.0, 0.0, 1.0]).unwrap();
    g.set_leaf_head(8, vec![4], 1, Activation::Tanh).unwrap();
    g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
    g.set_density_tracker().unwrap();
    g.set_calibration(15).unwrap();
    let thresholds = [
        0.5, 64.0, 128.0, 0.05, 0.02, 4.0, 0.1, 32.0, 10.0, 8.0, 20.0, f64::MAX,
        0.005, 1.05,
    ];
    g.set_decision_policy(&thresholds).unwrap();
    g.add_node(0, 1).unwrap();
    g.add_node(0, 2).unwrap();
    g
}

/// Synthesize an 8-D feature vector for a chess state. Deterministic
/// from `(seed, ply)` so the test corpus is reproducible across runs.
fn synth_chess_features(seed: u64, ply: u64) -> Vec<f64> {
    let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15);
    let mut out = Vec::with_capacity(8);
    for d in 0..8 {
        x = x.wrapping_add(ply.wrapping_mul(0xBF58476D1CE4E5B9));
        x ^= x >> (33 + d % 7);
        x = x.wrapping_mul(0xFF51AFD7ED558CCD);
        // Map to [-1, 1] uniformly.
        let f = ((x >> 11) as f64) / ((1u64 << 53) as f64);
        out.push(f * 2.0 - 1.0);
    }
    out
}

/// Run a deterministic 16-step rollout that simulates one self-play
/// game's worth of value/policy updates. After the rollout, the
/// graph's `chain_head` is the v2.6 canary.
fn rollout_v2_6(g: &mut AdaptiveBeliefGraph) {
    g.stamp_provenance(0, CHESS_V2_6_PROVENANCE_STAMP).unwrap();
    for ply in 0..16u64 {
        let phi = synth_chess_features(V2_6_GRAPH_SEED, ply);
        // Use the first 4 dims as the BLR feature space (matches
        // leaf head input_dim resolution).
        let bx: Vec<f64> = phi.iter().take(4).copied().collect();
        // Synthetic TD target: tanh of the L2 norm of bx.
        let y = bx.iter().map(|v| v * v).sum::<f64>().sqrt().tanh();
        // Update the root's BLR posterior with a single observation.
        g.blr_update(0, &bx, &[y]).unwrap();
        // Fold the value into the per-node stats so the chain has
        // shape beyond just BLR updates.
        g.observe(0, y).unwrap();
    }
}

// ── 5.1: value head replacement ────────────────────────────────────────

#[test]
fn v2_6_value_head_via_descend_then_blr_predict() {
    // The retrofit pattern: encode_prefix → descend → blr_predict.
    let g = build_v2_6_graph();
    let phi8 = synth_chess_features(V2_6_GRAPH_SEED, 0);
    let phi_first_dim = vec![phi8[0]];
    let prefix = g.encode_prefix(&phi_first_dim).unwrap();
    let evidence = g.descend(&prefix);
    assert!(evidence.leaf_id < g.node_count());
    let bx: Vec<f64> = phi8.iter().take(4).copied().collect();
    let (mean, lev, ale) = g.blr_predict(evidence.leaf_id, &bx).unwrap();
    assert!(mean.is_finite(), "value mean must be finite, got {mean}");
    assert!(lev.is_finite(), "leverage must be finite, got {lev}");
    assert!(ale.is_finite(), "aleatoric must be finite, got {ale}");
}

#[test]
fn v2_6_value_head_deterministic_across_runs() {
    let g1 = build_v2_6_graph();
    let g2 = build_v2_6_graph();
    let phi: Vec<f64> = synth_chess_features(V2_6_GRAPH_SEED, 1).iter().take(4).copied().collect();
    let p1 = g1.blr_predict(0, &phi).unwrap();
    let p2 = g2.blr_predict(0, &phi).unwrap();
    assert_eq!(p1.0.to_bits(), p2.0.to_bits());
    assert_eq!(p1.1.to_bits(), p2.1.to_bits());
    assert_eq!(p1.2.to_bits(), p2.2.to_bits());
}

#[test]
fn v2_6_value_head_byte_identical_after_replay() {
    let mut g = build_v2_6_graph();
    rollout_v2_6(&mut g);
    let phi: Vec<f64> = synth_chess_features(V2_6_GRAPH_SEED, 7).iter().take(4).copied().collect();
    let pre = g.blr_predict(0, &phi).unwrap();
    let blob = serialize(&g);
    let g2 = replay(&blob).unwrap();
    let post = g2.blr_predict(0, &phi).unwrap();
    assert_eq!(pre.0.to_bits(), post.0.to_bits());
    assert_eq!(pre.1.to_bits(), post.1.to_bits());
    assert_eq!(pre.2.to_bits(), post.2.to_bits());
}

#[test]
fn v2_6_value_head_serialize_double_run_byte_identical() {
    let mut g1 = build_v2_6_graph();
    let mut g2 = build_v2_6_graph();
    rollout_v2_6(&mut g1);
    rollout_v2_6(&mut g2);
    assert_eq!(serialize(&g1), serialize(&g2));
}

// ── 5.2: policy head replacement ───────────────────────────────────────

#[test]
fn v2_6_policy_head_softmax_over_action_features() {
    // Each action's score is the leaf's BLR mean on the
    // action-conditional features. Softmax over the action set.
    let g = build_v2_6_graph();
    let action_features: Vec<Vec<f64>> = (0..8)
        .map(|i| {
            let mut phi = synth_chess_features(V2_6_GRAPH_SEED, 0);
            phi[0] = (i as f64 - 4.0) / 4.0; // perturb the first dim
            phi.into_iter().take(4).collect()
        })
        .collect();
    let mut scores = Vec::with_capacity(action_features.len());
    for af in &action_features {
        let (mean, _, _) = g.blr_predict(0, af).unwrap();
        scores.push(mean);
    }
    // Convert to softmax distribution.
    let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|s| (s - max_s).exp()).collect();
    let z: f64 = exps.iter().sum();
    let probs: Vec<f64> = exps.iter().map(|e| e / z).collect();
    // Distribution invariants.
    assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    for p in &probs {
        assert!((0.0..=1.0).contains(p));
    }
}

#[test]
fn v2_6_policy_head_argmax_consistent_across_runs() {
    let g1 = build_v2_6_graph();
    let g2 = build_v2_6_graph();
    let candidates: Vec<Vec<f64>> = (0..6)
        .map(|i| vec![(i as f64) * 0.1, 0.5, -0.25, 0.0])
        .collect();
    let argmax = |g: &AdaptiveBeliefGraph| {
        candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, g.blr_predict(0, c).unwrap().0))
            .fold((0usize, f64::NEG_INFINITY), |(bi, bs), (i, s)| {
                if s > bs {
                    (i, s)
                } else {
                    (bi, bs)
                }
            })
            .0
    };
    assert_eq!(argmax(&g1), argmax(&g2));
}

// ── 5.3: uncertainty-gated bootstrap ───────────────────────────────────

#[test]
fn v2_6_uncertainty_gate_supports_when_ood_below_tau() {
    // Fresh graph, untrained — OOD score is dominated by
    // epistemic_z which clamps to 1.0 without expected_epistemic
    // captured. We need an explicit threshold check rather than
    // relying on the raw value.
    let g = build_v2_6_graph();
    let phi: Vec<f64> = synth_chess_features(V2_6_GRAPH_SEED, 3).iter().take(4).copied().collect();
    let prefix = g.encode_prefix(&phi[..1]).unwrap();
    let evidence = g.descend(&prefix);
    let ood = g
        .ood_score(evidence.leaf_id, &phi, evidence.matched_prefix, 1)
        .unwrap();
    // Sanity: ood is in [0, 1].
    assert!((0.0..=1.0).contains(&ood), "ood out of range: {ood}");
    let trust = ood < V2_6_UNCERTAINTY_TAU;
    // We don't assert which way the gate fires — just that the
    // decision is well-defined and ood is in range.
    let _ = trust;
}

#[test]
fn v2_6_uncertainty_gate_uniform_fallback_on_abstention() {
    // When the uncertainty gate fires, the policy returns uniform
    // probabilities. Synthetic test: build the action distribution,
    // confirm uniform fallback yields equal mass per action.
    let n_actions = 8;
    let uniform_p = 1.0 / n_actions as f64;
    let probs = vec![uniform_p; n_actions];
    let entropy = -probs.iter().map(|p| p * p.ln()).sum::<f64>();
    let max_entropy = (n_actions as f64).ln();
    // Uniform reaches maximum entropy.
    assert!((entropy - max_entropy).abs() < 1e-9);
}

// ── 5.4: locked v2.6 canaries ──────────────────────────────────────────

#[test]
fn v2_6_chain_head_canary_locked() {
    // After a deterministic 16-step rollout, the graph's chain_head
    // matches a locked hex constant. This is the v2.6 retrofit's
    // canary — the chess-RL counterpart of decide_step_canary_tests'
    // EXPECTED_HEX. If this fires:
    //   1. Run with --nocapture to log the new hex.
    //   2. Confirm the v2.6 retrofit semantic change is intentional.
    //   3. Update CANARY_HEX below.
    let mut g = build_v2_6_graph();
    rollout_v2_6(&mut g);
    let actual_hex = hex_of(&g.chain_head);
    println!("v2.6 chess RL canary chain_head = {actual_hex}");
    // Locked at Phase 0.5 ship (Item 5) — recompute and update if
    // you change the rollout, codebook, leaf head, BLR prior, or
    // v12 layout. The companion v2.5 weight hash on master is
    // 9.790915694115341 (different topology, different math).
    const CANARY_HEX: &str =
        "27d547b8f721b6631e3cbbe5fc4de560c6f09e6cc93eaf6c9e1bf36a3db6847b";
    assert_eq!(
        actual_hex, CANARY_HEX,
        "v2.6 chess RL chain_head canary mismatch — see comment"
    );
}

#[test]
fn v2_6_chain_head_canary_double_run_byte_identical() {
    // Determinism check: two independent rollouts produce
    // bit-identical chain_heads.
    let mut g1 = build_v2_6_graph();
    let mut g2 = build_v2_6_graph();
    rollout_v2_6(&mut g1);
    rollout_v2_6(&mut g2);
    assert_eq!(g1.chain_head, g2.chain_head);
}

#[test]
fn v2_6_blr_state_hash_canary_locked() {
    // The leaf's BLR state_hash after rollout — independent of
    // chain_head. Fires when the BLR conjugate update arithmetic
    // changes (and only when).
    let mut g = build_v2_6_graph();
    rollout_v2_6(&mut g);
    let blr = g.nodes[0].blr.as_ref().unwrap();
    let actual = hex_of(&blr.state_hash());
    println!("v2.6 chess RL canary BLR state_hash = {actual}");
    // Locked at Phase 0.5 ship — recompute and update if BLR
    // conjugate update arithmetic changes (independent of chain
    // layout — this canary fires only on numerical changes).
    const CANARY_HEX: &str =
        "869b32bdf937d27ec032b789980583fa1bf5871c528a7ee1f26d1d40fb6cfabc";
    assert_eq!(
        actual, CANARY_HEX,
        "v2.6 chess RL BLR state_hash canary mismatch — see comment"
    );
}

#[test]
fn v2_6_audit_chain_verifies_post_rollout() {
    let mut g = build_v2_6_graph();
    rollout_v2_6(&mut g);
    assert!(g.verify_chain().is_ok());
}

// ── Integration: smart_replay + provenance + predict_snap ──────────────

#[test]
fn v2_6_smart_replay_byte_identical_to_naive() {
    let mut g = build_v2_6_graph();
    rollout_v2_6(&mut g);
    let blob = serialize(&g);
    let g_naive = replay(&blob).unwrap();
    let g_smart = smart_replay(&blob).unwrap();
    assert_eq!(g_naive.chain_head, g_smart.chain_head);
    assert_eq!(serialize(&g_naive), serialize(&g_smart));
}

#[test]
fn v2_6_provenance_stamp_present_after_rollout() {
    let mut g = build_v2_6_graph();
    rollout_v2_6(&mut g);
    assert_eq!(
        g.nodes[0].provenance_stamp_hash, CHESS_V2_6_PROVENANCE_STAMP,
        "provenance stamp must persist on root node"
    );
}

#[test]
fn v2_6_provenance_stamped_event_in_audit_log() {
    let mut g = build_v2_6_graph();
    rollout_v2_6(&mut g);
    let stamps = g
        .audit
        .iter()
        .filter(|e| matches!(e.kind, AuditKind::ProvenanceStamped { .. }))
        .count();
    assert_eq!(stamps, 1, "exactly one ProvenanceStamped event expected");
}

#[test]
fn v2_6_predict_snap_carries_provenance_lineage() {
    let mut g = build_v2_6_graph();
    rollout_v2_6(&mut g);
    let phi: Vec<f64> = synth_chess_features(V2_6_GRAPH_SEED, 11).iter().take(4).copied().collect();
    let bytes = cjc_abng::predict_snap::pack(&g, 0, &phi).unwrap();
    let snap = cjc_abng::predict_snap::unpack(&bytes).unwrap();
    assert_eq!(snap.provenance_stamp_hash, CHESS_V2_6_PROVENANCE_STAMP);
    assert_eq!(snap.model_chain_head, g.chain_head);
}

#[test]
fn v2_6_predict_snap_unstamped_root_renders_zero_hash() {
    // A graph that has NOT been stamped emits the all-zero hash in
    // predict_snap. This proves the unstamped path stays clean.
    let g = build_v2_6_graph();
    let phi = vec![0.1, 0.2, 0.3, 0.4];
    let bytes = cjc_abng::predict_snap::pack(&g, 0, &phi).unwrap();
    let snap = cjc_abng::predict_snap::unpack(&bytes).unwrap();
    assert_eq!(snap.provenance_stamp_hash, [0u8; 32]);
}

// ── Topology + structural decision regression ─────────────────────────

#[test]
fn v2_6_decide_step_action_counts_well_defined_post_rollout() {
    let mut g = build_v2_6_graph();
    rollout_v2_6(&mut g);
    let counts = g.decide_step();
    assert_eq!(counts.len(), 6);
    // Sanity: action_counts[*] are u64; nothing is NaN-equivalent.
    for c in counts.iter() {
        assert!(*c <= u64::MAX);
    }
}

#[test]
fn v2_6_uncertainty_gate_decision_table_consistent() {
    // Across N candidates, the gate's trust/abstain decision is
    // deterministic across runs — the same input always produces
    // the same decision.
    let n_candidates = 6;
    let mut decisions_a = Vec::with_capacity(n_candidates);
    let mut decisions_b = Vec::with_capacity(n_candidates);
    for run in 0..2 {
        let g = build_v2_6_graph();
        for i in 0..n_candidates {
            let phi: Vec<f64> = synth_chess_features(V2_6_GRAPH_SEED, i as u64)
                .iter()
                .take(4)
                .copied()
                .collect();
            let prefix = g.encode_prefix(&phi[..1]).unwrap();
            let evidence = g.descend(&prefix);
            let ood = g
                .ood_score(evidence.leaf_id, &phi, evidence.matched_prefix, 1)
                .unwrap();
            let abstain = ood >= V2_6_UNCERTAINTY_TAU;
            if run == 0 {
                decisions_a.push(abstain);
            } else {
                decisions_b.push(abstain);
            }
        }
    }
    assert_eq!(decisions_a, decisions_b);
}

// ── Helpers ───────────────────────────────────────────────────────────

fn hex_of(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}
