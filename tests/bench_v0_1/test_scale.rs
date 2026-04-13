/// Scale benchmarks: larger-scale determinism tests.
///
/// - bench_matmul_sizes: matmul at 8, 32, 64, 128, 256, 512 (crosses path thresholds)
/// - bench_nn_wide: 10-layer, width-256, batch-16 NN forward pass
/// - bench_broadcast_chain: deep chain of broadcast ops on 500×500 tensors
use super::helpers;

// ── Matmul sizes ───────────────────────────────────────────────────────────

#[test]
fn matmul_sizes_deterministic() {
    let src = helpers::load_cjc("bench_matmul_sizes.cjcl");
    let out1 = helpers::run_eval(&src, 42);
    let out2 = helpers::run_eval(&src, 42);
    helpers::assert_deterministic(&out1, &out2);
}

#[test]
fn matmul_sizes_eval_mir_parity() {
    let src = helpers::load_cjc("bench_matmul_sizes.cjcl");
    let eval_out = helpers::run_eval(&src, 42);
    let mir_out = helpers::run_mir(&src, 42);
    helpers::assert_parity(&eval_out, &mir_out);
}

#[test]
fn matmul_sizes_has_6_stages() {
    let src = helpers::load_cjc("bench_matmul_sizes.cjcl");
    let out = helpers::run_eval(&src, 42);
    let stages = helpers::parse_stage_lines(&out);
    assert_eq!(stages.len(), 6, "Expected 6 stage hashes (one per matmul size)");
}

// ── NN Wide ────────────────────────────────────────────────────────────────

#[test]
fn nn_wide_deterministic() {
    let src = helpers::load_cjc("bench_nn_wide.cjcl");
    let out1 = helpers::run_eval(&src, 42);
    let out2 = helpers::run_eval(&src, 42);
    helpers::assert_deterministic(&out1, &out2);
}

#[test]
fn nn_wide_eval_mir_parity() {
    let src = helpers::load_cjc("bench_nn_wide.cjcl");
    let eval_out = helpers::run_eval(&src, 42);
    let mir_out = helpers::run_mir(&src, 42);
    helpers::assert_parity(&eval_out, &mir_out);
}

// ── Broadcast chain ────────────────────────────────────────────────────────

#[test]
fn broadcast_chain_deterministic() {
    let src = helpers::load_cjc("bench_broadcast_chain.cjcl");
    let out1 = helpers::run_eval(&src, 42);
    let out2 = helpers::run_eval(&src, 42);
    helpers::assert_deterministic(&out1, &out2);
}

#[test]
fn broadcast_chain_eval_mir_parity() {
    let src = helpers::load_cjc("bench_broadcast_chain.cjcl");
    let eval_out = helpers::run_eval(&src, 42);
    let mir_out = helpers::run_mir(&src, 42);
    helpers::assert_parity(&eval_out, &mir_out);
}
