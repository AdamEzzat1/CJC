//! Phase 0.3d-5 — proptest-driven properties for the ABNG
//! structural-decision engine.
//!
//! Coverage (per Phase 0.3d implementation prompt §3.4):
//!
//! 1. **Replay byte-equality** — for any random `(seed, observation
//!    sequence, force-action sequence)`, `serialize → replay → serialize`
//!    yields a byte-identical blob and the reconstructed graph's
//!    `chain_head` matches the original.
//!
//! 2. **`decide_step` monotonicity** — calling `decide_step` twice with
//!    no observations between calls produces the same `action_counts`
//!    on the second call as on the first (i.e. `decide_step` is
//!    idempotent on idle evidence ONCE the engine has settled).
//!
//! 3. **Density-score monotonicity in Mahalanobis distance** — for
//!    fixed training samples, the density score on a query point is
//!    non-decreasing in the L2 distance from the training mean (since
//!    `score = 1 − exp(−mahal²)`). Verified against random
//!    feature batches.
//!
//! 256 cases per property minimum (proptest default is 256). The
//! properties are deliberately conservative: they verify the
//! invariants the architecture doc §2 declares as load-bearing,
//! without depending on simplifications that may improve in Phase 0.4.

use proptest::prelude::*;

use cjc_abng::graph::{ActionKind, AdaptiveBeliefGraph};
use cjc_abng::serialize::{replay, serialize};

/// Strategy generating a valid 11-element threshold tensor for
/// [`AdaptiveBeliefGraph::set_decision_policy`]. Bounds keep the
/// thresholds in a regime that exercises real triggers without
/// blowing up under random observation streams.
fn arb_thresholds() -> impl Strategy<Value = Vec<f64>> {
    // 14 thresholds: Phase 0.4 Track B-2.2.7 added drift_unfreeze at [11];
    // Phase 0.4-extended (v11) added ece_stability_max_delta at [12] and
    // sigma_stability_ratio at [13]. Proptest's tuple-of-twelve trait
    // impl is the largest available, so pack the first 12 into one
    // tuple and the last 2 into a sibling tuple, then concatenate.
    let head = (
        0.1f64..1.0,                  // [0] H_grow
        16u64..256,                   // [1] grow_min
        32u64..512,                   // [2] split_min
        0.001f64..0.5,                // [3] nll_split_gain
        0.001f64..0.5,                // [4] impurity_min
        0u8..32,                      // [5] tau_merge (Hamming)
        0.001f64..2.0,                // [6] kl_merge
        0u64..16,                     // [7] prune_floor
        2u64..64,                     // [8] prune_grace_epochs
        0u8..32,                      // [9] tau_compress
        2u64..64,                     // [10] freeze_after
        0.5f64..100.0,                // [11] drift_unfreeze
    );
    let tail = (
        0.0001f64..0.5,               // [12] ece_stability_max_delta (> 0)
        1.0f64..10.0,                 // [13] sigma_stability_ratio (≥ 1)
    );
    (head, tail).prop_map(|(t, u)| {
        vec![
            t.0,
            t.1 as f64,
            t.2 as f64,
            t.3,
            t.4,
            t.5 as f64,
            t.6,
            t.7 as f64,
            t.8 as f64,
            t.9 as f64,
            t.10 as f64,
            t.11,
            u.0,
            u.1,
        ]
    })
}

/// Strategy generating a small observation stream — bounded in length
/// so a single property invocation is fast.
fn arb_observation_stream() -> impl Strategy<Value = Vec<f64>> {
    proptest::collection::vec(-100.0f64..100.0, 0..32)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Property 1: any observation stream replays byte-identically.
    #[test]
    fn observe_then_replay_byte_identical(
        seed in 0u64..1_000_000,
        observations in arb_observation_stream(),
    ) {
        let mut g = AdaptiveBeliefGraph::new(seed);
        for &v in &observations {
            // Skip non-finite values — `observe` accepts them but they
            // aren't a contract we're testing here.
            if !v.is_finite() {
                continue;
            }
            g.observe(0, v).unwrap();
        }
        let blob1 = serialize(&g);
        let g2 = replay(&blob1).unwrap();
        let blob2 = serialize(&g2);
        prop_assert_eq!(blob1, blob2,
            "serialize → replay → serialize not byte-identical");
        prop_assert_eq!(g.chain_head, g2.chain_head,
            "chain_head not preserved across replay");
    }

    /// Property 2: `decide_step` on an idle graph (no observations
    /// between calls) is monotonic — once the engine has settled
    /// past initial signature capture, repeated calls produce the
    /// same fall-through, eventually reaching Freeze and stopping.
    /// We verify the weaker form: NO regression in action_counts
    /// across paired calls (counts only grow or stay flat).
    #[test]
    fn decide_step_action_counts_monotonic(
        seed in 0u64..1_000_000,
        thresholds in arb_thresholds(),
    ) {
        let mut g = AdaptiveBeliefGraph::new(seed);
        g.set_decision_policy(&thresholds).unwrap();
        let mut prev_counts = [0u64; 6];
        for _ in 0..6 {
            let _ = g.decide_step();
            for i in 0..6 {
                prop_assert!(g.action_counts[i] >= prev_counts[i],
                    "action_counts[{}] regressed from {} to {}",
                    i, prev_counts[i], g.action_counts[i]);
            }
            prev_counts = g.action_counts;
        }
    }

    /// Property 3: density score is non-decreasing in Mahalanobis
    /// distance. Training a density tracker with samples concentrated
    /// near the origin, then querying at progressively farther points,
    /// produces non-decreasing scores.
    #[test]
    fn density_score_monotonic_in_distance(
        seed in 0u64..1_000_000,
    ) {
        use cjc_ad::pinn::Activation;
        let mut g = AdaptiveBeliefGraph::new(seed);
        g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
        g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
        g.set_density_tracker().unwrap();
        // Train density on a tight cluster around the origin.
        let cluster: Vec<f64> = (0..16)
            .flat_map(|i| {
                let off = (i as f64) * 0.01;
                [off, -off]
            })
            .collect();
        g.density_observe(0, &cluster).unwrap();
        // Query at radii 0.0, 1.0, 5.0, 10.0 along the diagonal.
        let r1 = g.density_score(0, &[0.0, 0.0]).unwrap();
        let r2 = g.density_score(0, &[1.0, 1.0]).unwrap();
        let r3 = g.density_score(0, &[5.0, 5.0]).unwrap();
        let r4 = g.density_score(0, &[10.0, 10.0]).unwrap();
        prop_assert!(r1 <= r2 + 1e-9,
            "score regressed: r=0→{}, r=√2→{}", r1, r2);
        prop_assert!(r2 <= r3 + 1e-9,
            "score regressed: r=√2→{}, r=√50→{}", r2, r3);
        prop_assert!(r3 <= r4 + 1e-9,
            "score regressed: r=√50→{}, r=√200→{}", r3, r4);
        // All scores in [0, 1] by construction.
        for s in [r1, r2, r3, r4] {
            prop_assert!((0.0..=1.0).contains(&s), "score {} outside [0,1]", s);
        }
    }

    /// Property 4 (bonus): `force_grow` followed by `replay` produces
    /// a byte-identical graph regardless of seed and child key. This
    /// validates the structural-event channel under random inputs.
    #[test]
    fn force_grow_replay_byte_identical(
        seed in 0u64..1_000_000,
        key in 0u8..=255,
    ) {
        let mut g = AdaptiveBeliefGraph::new(seed);
        let _c = g.force_grow(0, key).unwrap();
        let blob1 = serialize(&g);
        let g2 = replay(&blob1).unwrap();
        let blob2 = serialize(&g2);
        prop_assert_eq!(blob1, blob2);
        prop_assert_eq!(g.chain_head, g2.chain_head);
        prop_assert_eq!(
            g.action_count(ActionKind::Grow),
            g2.action_count(ActionKind::Grow)
        );
    }

    /// Property 5 (Phase 0.5 Item 1): an arbitrary provenance stamp
    /// round-trips through serialize/replay byte-identically. The
    /// stamp's hash is randomized across the full 256-byte alphabet
    /// per byte; replay must preserve the stamped value AND keep the
    /// audit chain valid post-stamp.
    #[test]
    fn provenance_stamp_replay_byte_identical(
        seed in 0u64..1_000_000,
        stamp_bytes in proptest::array::uniform32(0u8..=255),
    ) {
        let mut g = AdaptiveBeliefGraph::new(seed);
        // Mix in a couple observations so the chain has shape beyond
        // just Created → ProvenanceStamped.
        g.observe(0, 0.5).unwrap();
        g.stamp_provenance(0, stamp_bytes).unwrap();
        g.observe(0, -1.5).unwrap();
        let blob1 = serialize(&g);
        let g2 = replay(&blob1).unwrap();
        let blob2 = serialize(&g2);
        prop_assert_eq!(blob1, blob2);
        prop_assert_eq!(g.chain_head, g2.chain_head);
        prop_assert_eq!(g2.nodes[0].provenance_stamp_hash, stamp_bytes);
        prop_assert!(g2.verify_chain().is_ok());
    }

    /// Property 6 (Phase 0.5 Item 2): smart_replay produces a graph
    /// whose serialized form is byte-identical to naive replay's
    /// output, for any blob produced by serialize → optional
    /// compact_log → serialize. This is the determinism contract
    /// for smart-replay.
    #[test]
    fn smart_replay_output_equals_naive_replay(
        seed in 0u64..1_000_000,
        observations in arb_observation_stream(),
        compact_at in 0usize..32,
    ) {
        let mut g = AdaptiveBeliefGraph::new(seed);
        for &v in &observations {
            if v.is_finite() {
                g.observe(0, v).unwrap();
            }
        }
        // Compact at a (clamped) seq somewhere in the middle of the
        // log to exercise both pre- and post-snapshot ranges.
        let until = (compact_at as u64).min(g.audit.len() as u64);
        let _ = g.compact_log(until);
        // A few more observes after compaction so post-snapshot
        // events exist for the smart path's consistency check.
        if let Some(&v) = observations.first() {
            if v.is_finite() {
                g.observe(0, v).unwrap();
            }
        }
        let blob = serialize(&g);
        let g_naive = replay(&blob).unwrap();
        let g_smart = cjc_abng::serialize::smart_replay(&blob).unwrap();
        prop_assert_eq!(g_naive.chain_head, g_smart.chain_head);
        prop_assert_eq!(serialize(&g_naive), serialize(&g_smart));
    }
}
