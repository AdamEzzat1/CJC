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

// ── explain ────────────────────────────────────────────────────────────

fn build_graph_with_blr_and_observations(seed: u64, n_obs: usize) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(seed);
    g.set_codebook(1, 4, &[-1.0, 0.0, 1.0]).unwrap();
    g.set_leaf_head(2, vec![2], 1, Activation::Tanh).unwrap();
    g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
    // Train BLR with `n_obs` observations.
    let mut features = Vec::with_capacity(n_obs * 2);
    let mut ys = Vec::with_capacity(n_obs);
    for i in 0..n_obs {
        let x1 = (i as f64) / (n_obs as f64);
        let x2 = 1.0 - x1;
        features.push(x1);
        features.push(x2);
        ys.push(2.0 * x1 + 3.0 * x2);
    }
    g.blr_update(0, &features, &ys).unwrap();
    g
}

fn write_prediction_snap(
    g: &AdaptiveBeliefGraph,
    node_id: u32,
    phi: &[f64],
    path: &std::path::Path,
) {
    let blob = cjc_abng::predict_snap::pack(g, node_id, phi).unwrap();
    fs::write(path, blob).unwrap();
}

#[test]
fn abng_explain_text_summary_uncalibrated() {
    // Graph with 50 observations but expected_epistemic uncaptured.
    let g = build_graph_with_blr_and_observations(42, 50);
    let pred_path = unique_temp_path("explain-uncal");
    write_prediction_snap(&g, 0, &[1.0, 0.0], &pred_path);

    let (stdout, _stderr, code) = run_cjc(&[
        "abng",
        "explain",
        pred_path.to_str().unwrap(),
    ]);
    assert_eq!(code, 0);
    assert!(stdout.contains("ABNG prediction snapshot"));
    assert!(stdout.contains("ABNG-PRED"));
    assert!(stdout.contains("UNCALIBRATED"));
    fs::remove_file(&pred_path).ok();
}

#[test]
fn abng_explain_text_summary_low_evidence() {
    // Graph with only 2 observations and expected_epistemic captured.
    let mut g = build_graph_with_blr_and_observations(43, 2);
    let _ = g.force_recapture_expected_epistemic(0).unwrap();
    let pred_path = unique_temp_path("explain-lowev");
    write_prediction_snap(&g, 0, &[1.0, 0.0], &pred_path);

    let (stdout, _stderr, code) = run_cjc(&[
        "abng",
        "explain",
        pred_path.to_str().unwrap(),
    ]);
    assert_eq!(code, 0);
    assert!(stdout.contains("LOW EVIDENCE"));
    fs::remove_file(&pred_path).ok();
}

#[test]
fn abng_explain_text_summary_supported() {
    // Plenty of evidence + expected_epistemic captured. Predict at a
    // point inside the trained distribution (similar to training
    // points) so the OOD ratio stays below 1.0.
    let mut g = build_graph_with_blr_and_observations(44, 100);
    let _ = g.force_recapture_expected_epistemic(0).unwrap();
    let pred_path = unique_temp_path("explain-supported");
    // Use a phi very close to the BLR posterior mean so leverage is
    // small relative to expected_epistemic at the mean.
    let mean = g.nodes[0].blr.as_ref().unwrap().mean.to_vec();
    write_prediction_snap(&g, 0, &mean, &pred_path);

    let (stdout, _stderr, code) = run_cjc(&[
        "abng",
        "explain",
        pred_path.to_str().unwrap(),
    ]);
    assert_eq!(code, 0);
    // Either Supported (likely) or OodSaturated (border case). Both
    // are valid outputs; we just confirm explain completes cleanly
    // and reports one of the categorical outcomes.
    assert!(
        stdout.contains("SUPPORTED") || stdout.contains("OOD SATURATED"),
        "expected SUPPORTED or OOD SATURATED, got: {stdout}"
    );
    fs::remove_file(&pred_path).ok();
}

#[test]
fn abng_explain_json_emits_required_keys() {
    let g = build_graph_with_blr_and_observations(45, 50);
    let pred_path = unique_temp_path("explain-json");
    write_prediction_snap(&g, 0, &[1.0, 0.0], &pred_path);

    let (stdout, _stderr, code) = run_cjc(&[
        "abng",
        "explain",
        pred_path.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(code, 0);
    for k in [
        "\"path\"",
        "\"chain_head\"",
        "\"node_id\"",
        "\"codebook_hash\"",
        "\"leaf_head_hash\"",
        "\"blr_state_hash\"",
        "\"blr_n_seen\"",
        "\"phi_dim\"",
        "\"prediction\"",
        "\"abstain\"",
    ] {
        assert!(stdout.contains(k), "expected `{k}` in JSON, got: {stdout}");
    }
    fs::remove_file(&pred_path).ok();
}

#[test]
fn abng_explain_with_matching_model_passes_lineage_check() {
    let g = build_graph_with_blr_and_observations(46, 50);
    let pred_path = unique_temp_path("explain-model-match-pred");
    let model_path = unique_temp_path("explain-model-match-model");
    write_prediction_snap(&g, 0, &[1.0, 0.0], &pred_path);
    write_snapshot(&g, &model_path);

    let (stdout, _stderr, code) = run_cjc(&[
        "abng",
        "explain",
        pred_path.to_str().unwrap(),
        "--model",
        model_path.to_str().unwrap(),
    ]);
    assert_eq!(code, 0);
    assert!(stdout.contains("model lineage check"));
    assert!(stdout.contains("chain_head match:    yes"));
    assert!(stdout.contains("codebook hash match: yes"));
    assert!(stdout.contains("leaf head match:     yes"));
    fs::remove_file(&pred_path).ok();
    fs::remove_file(&model_path).ok();
}

#[test]
fn abng_explain_with_mismatched_model_fails_lineage_check() {
    // pred from one graph, model from a DIFFERENT seed → chain_head
    // mismatch, exit 1.
    let g_pred = build_graph_with_blr_and_observations(47, 50);
    let g_model = build_graph_with_blr_and_observations(48, 50);
    let pred_path = unique_temp_path("explain-model-mismatch-pred");
    let model_path = unique_temp_path("explain-model-mismatch-model");
    write_prediction_snap(&g_pred, 0, &[1.0, 0.0], &pred_path);
    write_snapshot(&g_model, &model_path);

    let (stdout, _stderr, code) = run_cjc(&[
        "abng",
        "explain",
        pred_path.to_str().unwrap(),
        "--model",
        model_path.to_str().unwrap(),
    ]);
    assert_ne!(code, 0, "mismatched lineage should exit 1");
    assert!(stdout.contains("chain_head match:    NO"));
    assert!(stdout.contains("WARNING"));
    fs::remove_file(&pred_path).ok();
    fs::remove_file(&model_path).ok();
}

#[test]
fn abng_explain_missing_path_errors() {
    let (_stdout, stderr, code) = run_cjc(&["abng", "explain"]);
    assert_ne!(code, 0);
    assert!(
        stderr.contains("requires a prediction-snap path"),
        "got: {stderr}"
    );
}

#[test]
fn abng_explain_bad_magic_errors() {
    // Write a file with the wrong magic and confirm explain rejects.
    let bad_path = unique_temp_path("explain-bad-magic");
    fs::write(&bad_path, vec![0u8; 200]).unwrap();
    let (_stdout, stderr, code) = run_cjc(&[
        "abng",
        "explain",
        bad_path.to_str().unwrap(),
    ]);
    assert_ne!(code, 0);
    assert!(
        stderr.contains("decode failed") || stderr.contains("BadMagic"),
        "got: {stderr}"
    );
    fs::remove_file(&bad_path).ok();
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
