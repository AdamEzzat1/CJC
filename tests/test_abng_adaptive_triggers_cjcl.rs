//! ABNG demo: adaptive structural triggers, written in CJC-Lang.
//!
//! Capability demonstrated: `abng_decide_step` runs the policy
//! engine that fires structural mutations (Grow/Split/Merge/
//! Prune/Compress/Freeze) based on per-node maturity + signature
//! stability + decision policy thresholds. The graph's structure
//! adapts to the workload — the marquee feature behind ABNG's
//! "*Adaptive*" name.
//!
//! Headline assertion: after running `decide_step` on a small
//! graph with similar-signature siblings, `action_counts[Merge]`
//! is non-zero AND the audit log grew with structural events.

mod abng_demos;

use abng_demos::harness::{extract_value, run_parity, Backend};

const SEED: u64 = 42;

#[test]
fn adaptive_cjcl_eval_mir_byte_equal() {
    let out = run_parity(abng_demos::adaptive_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("merge_count:")));
    assert!(out.iter().any(|l| l.starts_with("any_action_fired:")));
}

#[test]
fn adaptive_cjcl_audit_chain_verifies_post_decide_step() {
    // Critical: structural mutations must keep the audit chain
    // valid. If decide_step's Merge corrupted the chain hashes,
    // verify_chain would return false.
    let out = run_parity(abng_demos::adaptive_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "verify_chain"), "true");
}

#[test]
fn adaptive_cjcl_some_structural_action_fired() {
    // Headline tangible benefit: the graph's structure changed
    // under the influence of decide_step. Some action_count[*]
    // is non-zero post-training.
    let out = run_parity(abng_demos::adaptive_source::SOURCE, SEED);
    assert_eq!(
        extract_value(&out, "any_action_fired"),
        "true",
        "decide_step must fire at least one structural action"
    );
}

#[test]
fn adaptive_cjcl_merge_count_matches_canary_topology() {
    // Specific expectation for the canary topology (root + 2
    // sibling children with shared signature): exactly one Merge
    // fires across the 3 decide_step passes. This matches the
    // chess RL canary's `[0, 0, 1, 0, 0, 0]` action_counts.
    let out = run_parity(abng_demos::adaptive_source::SOURCE, SEED);
    let merge: u64 = extract_value(&out, "merge_count").parse().unwrap();
    assert_eq!(
        merge, 1,
        "canary topology must fire exactly 1 Merge across 3 decide_steps"
    );
}

#[test]
fn adaptive_cjcl_audit_log_grew_with_decide_step() {
    // The audit log MUST grow under decide_step calls — every
    // structural action emits an event into the chain. If the log
    // didn't grow, decide_step's effect was silent and untraceable.
    let out = run_parity(abng_demos::adaptive_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "audit_grew"), "true");
}

#[test]
fn adaptive_cjcl_action_counts_in_correct_buckets() {
    // Sanity: the 6 action types are distinct buckets — no double-
    // counting (each fired action goes into exactly one bucket).
    let out = run_parity(abng_demos::adaptive_source::SOURCE, SEED);
    let total: u64 = extract_value(&out, "total_actions").parse().unwrap();
    let merge: u64 = extract_value(&out, "merge_count").parse().unwrap();
    let grow: u64 = extract_value(&out, "grow_count").parse().unwrap();
    let split: u64 = extract_value(&out, "split_count").parse().unwrap();
    let prune: u64 = extract_value(&out, "prune_count").parse().unwrap();
    let compress: u64 = extract_value(&out, "compress_count").parse().unwrap();
    let freeze: u64 = extract_value(&out, "freeze_count").parse().unwrap();
    assert_eq!(total, grow + split + merge + prune + compress + freeze);
}

#[test]
fn adaptive_cjcl_chain_head_canary_locked() {
    let out = run_parity(abng_demos::adaptive_source::SOURCE, SEED);
    let chain = extract_value(&out, "chain_head");
    println!("adaptive cjcl canary chain_head = {chain}");
    // This canary deliberately mirrors the Rust-side
    // decide_step_canary_tests.rs scenario, so the chain_head
    // matches the locked v12 canary EXACTLY. Bit-equality across
    // the Rust path AND the CJC-Lang AST path AND the CJC-Lang
    // MIR path is the strongest cross-pipeline determinism gate
    // the project has.
    const CANARY_HEX: &str =
        "d064fb08c546be1b9850bfa91f87f4aed95682aa4fb7f4533cf1ac4da0d87807";
    assert_eq!(
        chain, CANARY_HEX,
        "adaptive cjcl chain_head must match the Rust-side decide_step canary EXACTLY"
    );
}

#[test]
fn adaptive_cjcl_smoke_runs_via_eval() {
    let out = abng_demos::harness::run(Backend::Eval, abng_demos::adaptive_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("merge_count:")));
}

#[test]
fn adaptive_cjcl_smoke_runs_via_mir() {
    let out = abng_demos::harness::run(Backend::Mir, abng_demos::adaptive_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("merge_count:")));
}
