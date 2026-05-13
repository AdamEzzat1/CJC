//! Phase 0.8c v14 Item A2 — fused `AuditKind::TrainStep` event (tag 0x1E).
//!
//! Replaces the pre-A2 parity test
//! `train_step_chain_head_matches_three_call_sequence` (Phase 0.7),
//! which asserted that `Graph::train_step` produced the EXACT SAME
//! chain head as the 3-call `blr_update + observe` sequence. Under A2,
//! `train_step` deliberately emits a single fused chain event instead,
//! so the chain heads diverge by design — what now matches bit-for-bit
//! is the affected node's post-call `NodeStats.canonical_bytes()` and
//! `BlrState.state_hash()`. These tests pin the new semantics:
//!
//! 1. `single_chain_event_emitted` — one row through `train_step`
//!    appends exactly ONE audit event of kind `TrainStep`.
//! 2. `state_bit_equals_three_call_sequence_modulo_chain_head` — the
//!    Welford stats and BLR state at the leaf are byte-identical to
//!    what the 3-call sequence produces; only the chain head differs.
//! 3. `replay_roundtrip` — a graph with `TrainStep` events serializes
//!    to v14 and replays back to a state byte-equal to the original.
//! 4. `with_numerical_rescue_emits_two_events` — when BLR's `b<ε`
//!    rescue branch fires inside `train_step`, the audit log appends
//!    `TrainStep` followed by `BlrNumericalRescue` (mirrors the pre-A2
//!    pattern of `BlrUpdated + BlrNumericalRescue`).

use cjc_abng::audit::{AuditKind, BLR_RESCUE_B_BELOW_EPSILON};
use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_abng::serialize::{replay, serialize};
use cjc_ad::pinn::Activation;

const SEED: u64 = 42;

/// Build a graph with codebook + leaf head + BLR prior + 4 child
/// nodes, matching `install_full_training_setup` from the dispatch
/// tests but exposed at the Rust API level.
fn install_full_training_setup(seed: u64) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(seed);
    // 1-D codebook with 3 cut points → 4 bins.
    g.set_codebook(1, 4, &[0.25, 0.5, 0.75]).unwrap();
    // input_dim=1, hidden=[4], output_dim=1, activation=tanh.
    g.set_leaf_head(1, vec![4], 1, Activation::Tanh).unwrap();
    // BLR prior with precision=2.0, a=1.0, b=0.5 (healthy — no rescue path).
    g.set_blr_prior(2.0, 1.0, 0.5).unwrap();
    // Four children under root so descend can land on a non-root leaf.
    for byte in 0..4u8 {
        g.add_node(0, byte).unwrap();
    }
    g
}

/// Pre-A2-like sibling: identical graph but populated via the 3-call
/// sequence (`route → blr_update → observe`) instead of `train_step`.
/// State bytes match `train_step`; chain head does not.
fn run_three_call_row(g: &mut AdaptiveBeliefGraph, x: f64, phi: &[f64; 4], y: f64) -> u32 {
    let prefix = g.encode_prefix(&[x]).unwrap();
    let leaf = g.descend(&prefix).leaf_id;
    g.blr_update(leaf, phi, &[y]).unwrap();
    g.observe(leaf, y).unwrap();
    leaf
}

#[test]
fn single_chain_event_emitted() {
    // Single row through `train_step` must append exactly ONE audit
    // event of kind TrainStep. Pre-A2 this would have been two
    // (BlrUpdated + BeliefUpdate).
    let mut g = install_full_training_setup(SEED);
    let pre = g.audit.len();
    let leaf = g.train_step(&[0.30], &[1.0, 0.5, 0.25, 0.125], 0.7).unwrap();
    assert_eq!(g.audit.len(), pre + 1, "exactly one fused chain event");
    let ev = g.audit.get(pre).unwrap();
    assert_eq!(ev.node_id, leaf);
    match ev.kind {
        AuditKind::TrainStep { value, state_hash: _ } => {
            assert_eq!(value, 0.7, "TrainStep.value records the observation");
        }
        ref k => panic!("expected TrainStep, got {k:?}"),
    }
    g.verify_chain().expect("chain still valid post-train_step");
}

#[test]
fn state_bit_equals_three_call_sequence_modulo_chain_head() {
    // Two independent graphs, identical seed + setup. One uses
    // `train_step` (1 fused event/row); the other uses the 3-call
    // sequence (2 events/row). After running the same rows on both,
    // the leaf node's `stats.canonical_bytes()` and BLR `state_hash()`
    // MUST byte-equal. Chain heads MUST differ (the whole point of A2).
    let mut g_fused = install_full_training_setup(SEED);
    let mut g_three = install_full_training_setup(SEED);

    let rows: &[(f64, [f64; 4], f64)] = &[
        (0.10, [1.0, 0.5, 0.25, 0.125], 0.7),
        (0.45, [0.3, 0.6, 0.9, 1.2], 1.1),
        (0.80, [0.8, 0.4, 0.2, 0.1], 0.4),
        (0.55, [0.5, 0.5, 0.5, 0.5], 0.5),
    ];
    for &(x, phi, y) in rows {
        let leaf_fused = g_fused.train_step(&[x], &phi, y).unwrap();
        let leaf_three = run_three_call_row(&mut g_three, x, &phi, y);
        assert_eq!(leaf_fused, leaf_three, "same leaf id at x={x}");

        // Welford state bytes — must match exactly.
        let stats_fused = g_fused.nodes[leaf_fused as usize].stats.canonical_bytes();
        let stats_three = g_three.nodes[leaf_three as usize].stats.canonical_bytes();
        assert_eq!(
            stats_fused, stats_three,
            "stats canonical_bytes diverge at x={x}, y={y}"
        );

        // BLR state hash — must match exactly.
        let blr_fused = g_fused.nodes[leaf_fused as usize]
            .blr
            .as_ref()
            .unwrap()
            .state_hash();
        let blr_three = g_three.nodes[leaf_three as usize]
            .blr
            .as_ref()
            .unwrap()
            .state_hash();
        assert_eq!(
            blr_fused, blr_three,
            "BLR state_hash diverges at x={x}, y={y}"
        );
    }

    // Chain heads MUST diverge (1 event/row vs 2/row).
    assert_ne!(
        g_fused.chain_head, g_three.chain_head,
        "chain heads must differ post-A2 — same chain head would mean the fusion didn't happen"
    );

    // Audit-log lengths reflect the 1-vs-2 event ratio: 4 rows = 4
    // events on the fused side, 8 on the 3-call side. (Setup events
    // are common to both, so the delta from setup-end to current is
    // 4 vs 8.)
    let g_setup = install_full_training_setup(SEED);
    let setup_len = g_setup.audit.len();
    assert_eq!(g_fused.audit.len(), setup_len + 4);
    assert_eq!(g_three.audit.len(), setup_len + 8);
}

#[test]
fn replay_roundtrip() {
    // Train a graph through `train_step`, serialize to v14, replay,
    // and assert the replay produces a byte-identical chain head.
    // This exercises the new tag 0x1E in both encode (write_payload)
    // and decode (apply_event + end-of-replay BLR verifier).
    let mut g = install_full_training_setup(SEED);
    let rows: &[(f64, [f64; 4], f64)] = &[
        (0.10, [1.0, 0.5, 0.25, 0.125], 0.7),
        (0.45, [0.3, 0.6, 0.9, 1.2], 1.1),
        (0.80, [0.8, 0.4, 0.2, 0.1], 0.4),
    ];
    for &(x, phi, y) in rows {
        g.train_step(&[x], &phi, y).unwrap();
    }
    let blob = serialize(&g);
    let g2 = replay(&blob).expect("v14 replay must succeed for TrainStep events");
    assert_eq!(g.chain_head, g2.chain_head, "chain_head must roundtrip");
    // Spot-check that the audit log on the replay side contains the
    // TrainStep events (counts + last-event-kind).
    assert_eq!(g.audit.len(), g2.audit.len());
    let last = g2.audit.last().unwrap();
    assert!(
        matches!(last.kind, AuditKind::TrainStep { .. }),
        "last event after replay should be TrainStep, got {:?}",
        last.kind
    );
}

#[test]
fn with_numerical_rescue_emits_two_events() {
    // BLR `b<ε` rescue branch construction: prior `b = ε/2` and an
    // observation that produces SSR = 0 (y = 0 with prior μ = 0) →
    // post-update `b` would fall below ε, the BLR layer clamps and
    // surfaces `Some(b_pre_clamp)`. Pre-A2 this fired
    // `BlrUpdated + BlrNumericalRescue` (2 events from blr_update,
    // plus a `BeliefUpdate` from observe = 3 events total). Post-A2,
    // `train_step` fires `TrainStep + BlrNumericalRescue` (2 events
    // total — still saves one event vs pre-A2 even in the rescue case).
    let mut g = AdaptiveBeliefGraph::new(SEED);
    g.set_codebook(1, 2, &[0.5]).unwrap();
    // input_dim=1, hidden=[], output_dim=1 — minimal head to keep BLR.d=2.
    g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
    // Degenerate prior: b = ε/2, so SSR-zero update always clamps.
    g.set_blr_prior(1.0, 1.5, f64::EPSILON / 2.0).unwrap();
    g.add_node(0, 0).unwrap();
    g.add_node(0, 1).unwrap();

    let pre = g.audit.len();
    // x routes to a leaf (any value works); phi has length BLR.d = 2;
    // y = 0.0 forces SSR = 0 and triggers the b<ε clamp.
    let leaf = g.train_step(&[0.10], &[1.0, 0.5], 0.0).unwrap();
    assert_eq!(g.audit.len(), pre + 2, "TrainStep + BlrNumericalRescue");

    let ts = g.audit.get(pre).unwrap();
    assert_eq!(ts.node_id, leaf);
    assert!(matches!(ts.kind, AuditKind::TrainStep { value, .. } if value == 0.0));

    let rescue = g.audit.get(pre + 1).unwrap();
    assert_eq!(rescue.node_id, leaf);
    let (reason, b_pre_clamp_bits) = match rescue.kind {
        AuditKind::BlrNumericalRescue { reason, b_pre_clamp_bits } => {
            (reason, b_pre_clamp_bits)
        }
        ref k => panic!("expected BlrNumericalRescue, got {k:?}"),
    };
    assert_eq!(reason, BLR_RESCUE_B_BELOW_EPSILON);
    // pre-clamp value is exactly b_old (= ε/2) since SSR is zero.
    assert_eq!(f64::from_bits(b_pre_clamp_bits), f64::EPSILON / 2.0);

    g.verify_chain().expect("chain valid after TrainStep + rescue");
}
