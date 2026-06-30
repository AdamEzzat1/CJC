//! Demo smoke test — runs `demos/state_space_chess/demo.cjcl` end-to-end via
//! both backends and asserts it (a) parses, (b) produces non-empty output,
//! (c) prints the expected section markers.
//!
//! Demo source is `include_str!`'d so the test always runs against the
//! checked-in version, not whatever is in target/.

use cjc_eval::Interpreter;

const DEMO_SRC: &str = include_str!("../../demos/state_space_chess/demo.cjcl");

fn run_demo_eval(seed: u64) -> Vec<String> {
    let (program, diags) = cjc_parser::parse_source(DEMO_SRC);
    assert!(
        !diags.has_errors(),
        "demo source failed to parse:\n{:#?}",
        diags.diagnostics
    );
    let mut interp = Interpreter::new(seed);
    interp
        .exec(&program)
        .unwrap_or_else(|e| panic!("eval demo failed: {e:?}"));
    interp.output
}

fn run_demo_mir(seed: u64) -> Vec<String> {
    let (program, diags) = cjc_parser::parse_source(DEMO_SRC);
    assert!(!diags.has_errors());
    let (_v, exec) = cjc_mir_exec::run_program_with_executor(&program, seed)
        .unwrap_or_else(|e| panic!("MIR demo failed: {e:?}"));
    exec.output
}

#[test]
fn demo_runs_under_eval_seed_42() {
    let out = run_demo_eval(42);
    // Section markers from the demo
    assert!(
        out.iter().any(|l| l.contains("=== state_space_chess demo ===")),
        "missing demo header in output"
    );
    assert!(
        out.iter().any(|l| l.contains("episode_1_result")),
        "missing episode_1_result line"
    );
    assert!(
        out.iter().any(|l| l.contains("episode_3_result")),
        "missing episode_3_result line (snapshot/restore path)"
    );
    assert!(
        out.iter().any(|l| l.contains("=== done ===")),
        "missing trailing `done` marker"
    );
}

#[test]
fn demo_runs_under_mir_seed_42() {
    let out = run_demo_mir(42);
    assert!(out.iter().any(|l| l.contains("=== state_space_chess demo ===")));
    assert!(out.iter().any(|l| l.contains("=== done ===")));
}

#[test]
fn demo_prints_hidden_state_norm() {
    // Verbose episode 1 must print at least one `h_norm_sq` line.
    let out = run_demo_eval(42);
    let any_h_line = out.iter().any(|l| l.contains("h_norm_sq"));
    assert!(any_h_line, "expected at least one hidden-state-norm log line");
}

#[test]
fn demo_runs_under_two_seeds() {
    // Two independent seeds — both must complete without error.
    let out_a = run_demo_eval(7);
    let out_b = run_demo_eval(99);
    assert!(out_a.iter().any(|l| l.contains("=== done ===")));
    assert!(out_b.iter().any(|l| l.contains("=== done ===")));
}
