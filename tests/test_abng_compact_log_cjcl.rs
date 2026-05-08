//! ABNG demo: log compaction with StatsSnapshot markers,
//! written in CJC-Lang.
//!
//! Capability demonstrated: `abng_compact_log` emits one
//! `StatsSnapshot` audit event per touched node — these markers
//! are the foundation for the smart-replay fast-forward
//! optimization (the cycle-saving layer ships in a follow-up).
//! Today this demo proves: compact_log is deterministic,
//! grows the audit log by exactly emitted-many events, and
//! preserves chain integrity.

mod abng_demos;

use abng_demos::harness::{extract_value, run_parity, Backend};

const SEED: u64 = 42;

#[test]
fn compact_cjcl_eval_mir_byte_equal() {
    let out = run_parity(abng_demos::compact_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("emitted:")));
}

#[test]
fn compact_cjcl_audit_chain_verifies_post_compact() {
    // Critical: emitting StatsSnapshot events must keep the
    // audit chain valid.
    let out = run_parity(abng_demos::compact_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "verify_pre"), "true");
    assert_eq!(extract_value(&out, "verify_post"), "true");
}

#[test]
fn compact_cjcl_audit_grew_by_exactly_emitted_count() {
    // Headline: post-compact audit length = pre-compact + emitted.
    // Every snapshot marker is recorded as a chain event; nothing
    // is lost.
    let out = run_parity(abng_demos::compact_source::SOURCE, SEED);
    assert_eq!(
        extract_value(&out, "audit_grew_by_emitted"),
        "true",
        "audit log must grow by exactly the number of emitted snapshots"
    );
}

#[test]
fn compact_cjcl_exactly_three_emitted_for_three_touched_nodes() {
    // Specific topology: nodes 0, 1, 2 each receive observations,
    // so compact_log emits exactly 3 StatsSnapshot events.
    let out = run_parity(abng_demos::compact_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "exactly_three_emitted"), "true");
}

#[test]
fn compact_cjcl_chain_head_advances_after_compact() {
    let out = run_parity(abng_demos::compact_source::SOURCE, SEED);
    assert_eq!(
        extract_value(&out, "chain_advanced"),
        "true",
        "chain_head must advance after StatsSnapshot events"
    );
}

#[test]
fn compact_cjcl_smoke_runs_via_eval() {
    let out = abng_demos::harness::run(Backend::Eval, abng_demos::compact_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("emitted:")));
}

#[test]
fn compact_cjcl_smoke_runs_via_mir() {
    let out = abng_demos::harness::run(Backend::Mir, abng_demos::compact_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("emitted:")));
}
