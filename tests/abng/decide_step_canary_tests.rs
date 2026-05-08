//! Phase 0.4 Track C-2.3.12 — `decide_step` chain-head canary.
//!
//! These tests pin the *exact* `chain_head` produced by a fixed
//! `decide_step` execution sequence. They are narrower than the
//! `prop_tests/abng_decision_props.rs` properties (which cover
//! behavioral determinism) — these check that the *byte-level* output
//! of the canonical canary scenario remains identical to what was
//! locked in at the time of writing.
//!
//! When this canary fires:
//! - **Did you intend to change `decide_step`'s deterministic output?**
//!   Audit-event payload format, signature canonicalization, Welford
//!   state, ring-buffer state, and decision-policy threshold layout
//!   all influence the chain head.
//! - If yes, recompute the expected hex below and update.
//! - If no, the change broke determinism somewhere — investigate
//!   before merging.

use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_ad::pinn::Activation;

fn hex(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Deterministic canary scenario. Identical inputs → identical outputs
/// across runs and platforms. Every operation here contributes to the
/// final chain hash.
fn canary_graph() -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(42);

    // Codebook + leaf head + BLR prior — all one-shot, must precede any
    // add_node. 1-D codebook with 4 bins keeps the prefix encoding
    // small but exercised.
    g.set_codebook(1, 4, &[-1.0, 0.0, 1.0]).unwrap();
    g.set_leaf_head(1, vec![2], 1, Activation::Tanh).unwrap();
    g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
    g.set_density_tracker().unwrap();
    g.set_calibration(15u8).unwrap();

    // Decision policy — 12 thresholds (post-B-2.2.7). drift_unfreeze
    // disabled so the canary doesn't require a frozen baseline.
    let thresholds = [
        0.5, 64.0, 128.0, 0.05, 0.02,
        4.0, 0.1, 32.0, 10.0, 8.0,
        20.0, f64::MAX,
    ];
    g.set_decision_policy(&thresholds).unwrap();

    // Two structurally non-trivial nodes.
    let _ = g.add_node(0, 1).unwrap();
    let _ = g.add_node(0, 2).unwrap();

    // Observe a small fixed sequence on root.
    g.observe(0, 0.10).unwrap();
    g.observe(0, 0.25).unwrap();
    g.observe(0, 0.40).unwrap();

    // Run decide_step three times to advance signature stability and
    // potentially fire structural triggers.
    for _ in 0..3 {
        let _ = g.decide_step();
    }

    g
}

#[test]
fn canary_chain_head_double_run_deterministic() {
    // Pure behavioral determinism: two independent constructions must
    // produce bit-identical chain heads.
    let g1 = canary_graph();
    let g2 = canary_graph();
    assert_eq!(
        g1.chain_head, g2.chain_head,
        "decide_step canary chain_head must be deterministic across runs"
    );
}

#[test]
fn canary_chain_head_byte_layout_locked() {
    // Locked-in chain head. If this assertion fires:
    //   1. Run `cargo test --test abng decide_step_canary -- --nocapture`
    //      to capture the new hex.
    //   2. Confirm the change is intentional (audit payload format,
    //      Welford order, signature layout, etc.).
    //   3. Update the EXPECTED_HEX constant below.
    const EXPECTED_HEX: &str =
        "3acd67fe2e2a657367603fd9f8b452386a6cc051157dadfe790035671260af18";

    let g = canary_graph();
    let actual_hex = hex(&g.chain_head);

    // Always print so a failure is self-diagnosing under --nocapture.
    println!("decide_step canary chain_head = {actual_hex}");

    assert_eq!(
        actual_hex, EXPECTED_HEX,
        "decide_step canary chain_head must match the locked-in value. \
         If this change is intentional, update EXPECTED_HEX in \
         tests/abng/decide_step_canary_tests.rs."
    );
}

#[test]
fn canary_chain_head_audit_chain_verifies() {
    // Sanity gate: the canary's chain must verify cleanly.
    let g = canary_graph();
    assert!(g.verify_chain().is_ok());
}

#[test]
fn canary_chain_head_serialize_round_trip() {
    // Serialize + replay must recover the exact same chain head.
    let g = canary_graph();
    let blob = cjc_abng::serialize::serialize(&g);
    let g2 = cjc_abng::serialize::replay(&blob).unwrap();
    assert_eq!(g.chain_head, g2.chain_head);
    assert_eq!(g.audit_len(), g2.audit_len());
}

#[test]
fn canary_decide_step_action_count_locked() {
    // Locked-in action_counts. Indexed by ActionKind:
    //   [Grow, Split, Merge, Prune, Compress, Freeze]
    // The canary fires exactly one Merge across its 3 decide_step
    // calls — siblings 1 and 2 share the root's signature within
    // tau_merge, posterior KL is 0 (both fresh BLR posteriors), so
    // try_merge succeeds on the first decide_step pass. Lock both
    // the shape and the specific count to catch:
    //   - Structural changes (7th element added → length assertion)
    //   - Trigger semantics changes (different Merge count → value
    //     assertion)
    let g = canary_graph();
    let counts = g.action_counts;
    println!("decide_step canary action_counts = {:?}", counts);
    assert_eq!(counts.len(), 6, "action_counts must remain [u64; 6]");
    assert_eq!(
        counts,
        [0, 0, 1, 0, 0, 0],
        "expected exactly one Merge across the canary's 3 decide_step \
         calls; if this changed, the trigger semantics moved — confirm \
         intentional, then update."
    );
}

#[test]
fn canary_decide_step_return_shape_locked() {
    // The decide_step return is [u64; N_ACTION_KINDS]. Lock the shape —
    // if any future change moves Unfreeze into action_counts (forcing
    // [u64; 7]), this fires.
    let mut g = canary_graph();
    let counts = g.decide_step();
    assert_eq!(
        counts.len(),
        6,
        "decide_step must return [u64; 6] indexed by ActionKind"
    );
}
