//! Deterministic-replay tests.
//!
//! Asserts that running the same demo twice with the same seed produces
//! identical printed output. This is the contract the brief calls
//! "deterministic replay seed" — and it's the whole reason the SSM cell uses
//! SplitMix64 with no thread-time / process-id leakage.

use cjc_eval::Interpreter;

const DEMO_SRC: &str = include_str!("../../demos/state_space_chess/demo.cjcl");

fn run_demo_eval(seed: u64) -> Vec<String> {
    let (program, diags) = cjc_parser::parse_source(DEMO_SRC);
    assert!(!diags.has_errors());
    let mut interp = Interpreter::new(seed);
    interp.exec(&program).unwrap();
    interp.output
}

fn run_demo_mir(seed: u64) -> Vec<String> {
    let (program, diags) = cjc_parser::parse_source(DEMO_SRC);
    assert!(!diags.has_errors());
    let (_v, exec) = cjc_mir_exec::run_program_with_executor(&program, seed).unwrap();
    exec.output
}

#[test]
fn eval_double_run_is_byte_identical() {
    let a = run_demo_eval(42);
    let b = run_demo_eval(42);
    assert_eq!(a, b, "eval double-run with same seed must match byte-for-byte");
}

#[test]
fn mir_double_run_is_byte_identical() {
    let a = run_demo_mir(42);
    let b = run_demo_mir(42);
    assert_eq!(a, b, "MIR double-run with same seed must match byte-for-byte");
}

#[test]
fn different_seeds_diverge() {
    // Smoke-level: not all output must differ, but at least the first
    // ply's policy probability should change. (We compare the full output
    // and assert it is *not* equal — flake-proof because the demo runs
    // ≥30 plies and the action samples + SSM trajectories diverge by ply 1.)
    let a = run_demo_eval(1);
    let b = run_demo_eval(2);
    assert_ne!(a, b, "different seeds must produce different traces");
}

#[test]
fn snapshot_restore_section_present_under_both_backends() {
    // The demo prints `--- episode 3 (after restore) ---` only after a
    // successful `state_space_restore`. Failure mode: that op silently
    // becomes a no-op or panics. Confirming the section header runs is a
    // cheap structural check that the snapshot/restore path executes.
    for out in [run_demo_eval(42), run_demo_mir(42)] {
        assert!(
            out.iter().any(|l| l.contains("(after restore)")),
            "missing post-restore section marker"
        );
    }
}
