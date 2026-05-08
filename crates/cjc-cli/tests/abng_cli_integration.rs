//! `cjcl abng …` CLI integration tests (Phase 0.4 Track A).
//!
//! Each test:
//! 1. Programmatically constructs an `AdaptiveBeliefGraph`,
//! 2. Serialises it via `cjc_abng::serialize::serialize`,
//! 3. Writes the bytes to a unique-per-test path under `std::env::temp_dir()`,
//! 4. Invokes the `cjc` binary with `cjcl abng <subcommand> <path>`,
//! 5. Asserts on the exit code and stdout/stderr.
//!
//! No tempfile dep — the helpers below namespace by test name + a
//! per-process counter to avoid races inside a single `cargo test` run.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_abng::serialize;
use cjc_ad::pinn::Activation;

fn run_cjc(args: &[&str]) -> (String, String, i32) {
    let output = Command::new(env!("CARGO_BIN_EXE_cjc"))
        .args(args)
        .output()
        .expect("failed to execute cjc binary");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let code = output.status.code().unwrap_or(-1);
    (stdout, stderr, code)
}

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn unique_temp_path(stem: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let mut p = std::env::temp_dir();
    p.push(format!("cjc-abng-cli-test-{}-{}-{}.snap", stem, std::process::id(), n));
    p
}

fn build_basic_graph(seed: u64) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(seed);
    g.set_codebook(1, 4, &[-1.0, 0.0, 1.0]).unwrap();
    g.set_leaf_head(1, vec![2], 1, Activation::Tanh).unwrap();
    g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
    let _ = g.add_node(0, 1).unwrap();
    g.observe(0, 0.5).unwrap();
    g.observe(0, 0.25).unwrap();
    g
}

fn write_snapshot(g: &AdaptiveBeliefGraph, path: &std::path::Path) {
    let bytes = serialize::serialize(g);
    fs::write(path, bytes).expect("write snapshot");
}

// ── inspect ────────────────────────────────────────────────────────────

#[test]
fn abng_help_prints_subcommands() {
    let (stdout, _stderr, code) = run_cjc(&["abng", "--help"]);
    assert_eq!(code, 0, "abng --help should exit 0");
    for sub in ["inspect", "replay", "diff", "explain", "train"] {
        assert!(
            stdout.contains(sub),
            "expected `{sub}` listed in help, got: {stdout}"
        );
    }
}

#[test]
fn abng_no_subcommand_errors() {
    let (_stdout, stderr, code) = run_cjc(&["abng"]);
    assert_ne!(code, 0);
    assert!(
        stderr.contains("requires a subcommand"),
        "expected 'requires a subcommand' message, got: {stderr}"
    );
}

#[test]
fn abng_unknown_subcommand_errors() {
    let (_stdout, stderr, code) = run_cjc(&["abng", "frobnicate"]);
    assert_ne!(code, 0);
    assert!(
        stderr.contains("unknown abng subcommand"),
        "expected 'unknown abng subcommand', got: {stderr}"
    );
}

#[test]
fn abng_inspect_basic_snapshot_text_summary() {
    let g = build_basic_graph(42);
    let path = unique_temp_path("inspect-basic");
    write_snapshot(&g, &path);

    let (stdout, _stderr, code) = run_cjc(&[
        "abng",
        "inspect",
        path.to_str().unwrap(),
    ]);
    assert_eq!(code, 0, "inspect should exit 0 on a valid snapshot");
    // Header summary lines must be present.
    assert!(stdout.contains("ABNG snapshot:"));
    assert!(stdout.contains("magic:"));
    assert!(stdout.contains("seed:"));
    assert!(stdout.contains("nodes:"));
    assert!(stdout.contains("audit events:"));
    assert!(stdout.contains("chain head:"));
    assert!(stdout.contains("action_counts:"));
    // Seed must echo the value baked into the graph.
    assert!(
        stdout.contains("seed:         42"),
        "expected seed: 42, got: {stdout}"
    );
    // Node count must be 2 (root + one added child).
    assert!(
        stdout.contains("nodes:        2"),
        "expected nodes: 2, got: {stdout}"
    );
    fs::remove_file(&path).ok();
}

#[test]
fn abng_inspect_json_summary_is_parseable() {
    let g = build_basic_graph(7);
    let path = unique_temp_path("inspect-json");
    write_snapshot(&g, &path);

    let (stdout, _stderr, code) = run_cjc(&[
        "abng",
        "inspect",
        path.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(code, 0);
    // Cheap JSON sanity (no parser dep): expect specific keys.
    for k in [
        "\"path\"",
        "\"file_size\"",
        "\"file_sha256\"",
        "\"magic_version\": 10",
        "\"seed\": 7",
        "\"n_nodes\": 2",
        "\"n_events\":",
        "\"chain_head\":",
        "\"action_counts\":",
    ] {
        assert!(stdout.contains(k), "expected `{k}` in JSON output, got: {stdout}");
    }
    fs::remove_file(&path).ok();
}

#[test]
fn abng_inspect_audit_flag_prints_histogram() {
    let g = build_basic_graph(11);
    let path = unique_temp_path("inspect-audit");
    write_snapshot(&g, &path);

    let (stdout, _stderr, code) = run_cjc(&[
        "abng",
        "inspect",
        path.to_str().unwrap(),
        "--audit",
    ]);
    assert_eq!(code, 0);
    assert!(stdout.contains("audit-event histogram"));
    // The basic graph emits at minimum Created + CodebookFrozen +
    // LeafHeadConfigured + BlrPriorConfigured + LeafParamsInitialized
    // (root, BLR install) + BlrInitialized + NodeAdded + LeafParamsInitialized
    // (child) + BlrInitialized (child) + 2 BeliefUpdate.
    assert!(stdout.contains("Created"));
    assert!(stdout.contains("CodebookFrozen"));
    assert!(stdout.contains("BeliefUpdate"));
    fs::remove_file(&path).ok();
}

#[test]
fn abng_inspect_tree_flag_prints_topology() {
    let g = build_basic_graph(13);
    let path = unique_temp_path("inspect-tree");
    write_snapshot(&g, &path);

    let (stdout, _stderr, code) = run_cjc(&[
        "abng",
        "inspect",
        path.to_str().unwrap(),
        "--tree",
    ]);
    assert_eq!(code, 0);
    assert!(stdout.contains("arena topology"));
    assert!(stdout.contains("(root)"));
    fs::remove_file(&path).ok();
}

#[test]
fn abng_inspect_node_flag_prints_per_node_state() {
    let g = build_basic_graph(15);
    let path = unique_temp_path("inspect-node");
    write_snapshot(&g, &path);

    let (stdout, _stderr, code) = run_cjc(&[
        "abng",
        "inspect",
        path.to_str().unwrap(),
        "--node",
        "0",
    ]);
    assert_eq!(code, 0);
    assert!(stdout.contains("node 0:"));
    assert!(stdout.contains("samples_seen:"));
    assert!(stdout.contains("expected_epistemic:"));
    fs::remove_file(&path).ok();
}

#[test]
fn abng_inspect_node_out_of_range_errors() {
    let g = build_basic_graph(17);
    let path = unique_temp_path("inspect-node-oor");
    write_snapshot(&g, &path);

    let (_stdout, stderr, code) = run_cjc(&[
        "abng",
        "inspect",
        path.to_str().unwrap(),
        "--node",
        "99",
    ]);
    assert_ne!(code, 0);
    assert!(
        stderr.contains("out of range"),
        "expected 'out of range' in stderr, got: {stderr}"
    );
    fs::remove_file(&path).ok();
}

#[test]
fn abng_inspect_missing_path_errors() {
    let (_stdout, stderr, code) = run_cjc(&["abng", "inspect"]);
    assert_ne!(code, 0);
    assert!(
        stderr.contains("requires a snapshot path"),
        "expected 'requires a snapshot path', got: {stderr}"
    );
}

#[test]
fn abng_inspect_nonexistent_file_errors() {
    let (_stdout, stderr, code) = run_cjc(&[
        "abng",
        "inspect",
        "definitely-does-not-exist.snap",
    ]);
    assert_ne!(code, 0);
    assert!(
        stderr.contains("not found"),
        "expected 'not found', got: {stderr}"
    );
}

// ── replay ─────────────────────────────────────────────────────────────

#[test]
fn abng_replay_succeeds_on_valid_snapshot() {
    let g = build_basic_graph(42);
    let path = unique_temp_path("replay-valid");
    write_snapshot(&g, &path);

    let (stdout, _stderr, code) = run_cjc(&[
        "abng",
        "replay",
        path.to_str().unwrap(),
    ]);
    assert_eq!(code, 0);
    assert!(stdout.contains("replay ok"));
    assert!(stdout.contains("chain head:"));
    fs::remove_file(&path).ok();
}

#[test]
fn abng_replay_verify_flag_reports_chain_status() {
    let g = build_basic_graph(42);
    let path = unique_temp_path("replay-verify");
    write_snapshot(&g, &path);

    let (stdout, _stderr, code) = run_cjc(&[
        "abng",
        "replay",
        path.to_str().unwrap(),
        "--verify",
    ]);
    assert_eq!(code, 0);
    assert!(stdout.contains("chain verify:"));
    assert!(stdout.contains("OK"));
    fs::remove_file(&path).ok();
}

#[test]
fn abng_replay_corrupt_snapshot_errors() {
    let g = build_basic_graph(42);
    let path = unique_temp_path("replay-corrupt");
    let mut bytes = serialize::serialize(&g);
    // Flip a byte in the audit-event payload region to break the chain.
    if bytes.len() > 100 {
        bytes[100] = bytes[100].wrapping_add(1);
    }
    fs::write(&path, &bytes).unwrap();

    let (_stdout, stderr, code) = run_cjc(&[
        "abng",
        "replay",
        path.to_str().unwrap(),
    ]);
    assert_ne!(code, 0);
    assert!(
        stderr.contains("replay failed"),
        "expected 'replay failed' in stderr, got: {stderr}"
    );
    fs::remove_file(&path).ok();
}

// ── diff ───────────────────────────────────────────────────────────────

#[test]
fn abng_diff_identical_snapshots_match() {
    let g = build_basic_graph(42);
    let a = unique_temp_path("diff-a");
    let b = unique_temp_path("diff-b");
    write_snapshot(&g, &a);
    write_snapshot(&g, &b);

    let (stdout, _stderr, code) = run_cjc(&[
        "abng",
        "diff",
        a.to_str().unwrap(),
        b.to_str().unwrap(),
    ]);
    assert_eq!(code, 0);
    assert!(stdout.contains("chain_head equal:    yes"));
    assert!(stdout.contains("per-node stats_chain_head: identical"));
    fs::remove_file(&a).ok();
    fs::remove_file(&b).ok();
}

#[test]
fn abng_diff_different_seeds_differ() {
    let g1 = build_basic_graph(42);
    let g2 = build_basic_graph(43);
    let a = unique_temp_path("diff-seed-a");
    let b = unique_temp_path("diff-seed-b");
    write_snapshot(&g1, &a);
    write_snapshot(&g2, &b);

    let (stdout, _stderr, code) = run_cjc(&[
        "abng",
        "diff",
        a.to_str().unwrap(),
        b.to_str().unwrap(),
    ]);
    // Different chain heads → exit 1.
    assert_ne!(code, 0);
    assert!(stdout.contains("chain_head equal:    NO"));
    fs::remove_file(&a).ok();
    fs::remove_file(&b).ok();
}

#[test]
fn abng_diff_arity_check() {
    let (_stdout, stderr, code) = run_cjc(&[
        "abng", "diff", "only-one-path.snap",
    ]);
    assert_ne!(code, 0);
    assert!(stderr.contains("requires exactly two snapshot paths"));
}

// ── stub subcommands (explain, train) ──────────────────────────────────

#[test]
fn abng_explain_stub_explains_unimplemented() {
    let (_stdout, stderr, code) = run_cjc(&["abng", "explain", "anything.snap"]);
    assert_ne!(code, 0);
    assert!(
        stderr.contains("not yet shipped"),
        "expected 'not yet shipped' for explain stub, got: {stderr}"
    );
}

#[test]
fn abng_train_stub_explains_unimplemented() {
    let (_stdout, stderr, code) = run_cjc(&["abng", "train"]);
    assert_ne!(code, 0);
    assert!(
        stderr.contains("not yet shipped"),
        "expected 'not yet shipped' for train stub, got: {stderr}"
    );
}
