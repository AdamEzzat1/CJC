/// MIR optimizer parity tests: verify that --mir-opt produces identical output.
///
/// The MIR optimizer applies CF, SR, DCE, CSE, LICM passes. These MUST NOT
/// change any computed values. These tests verify that for every benchmark.
use super::helpers;

#[test]
fn pipeline_mir_opt_parity() {
    let src = helpers::load_cjc("bench_pipeline.cjc");
    let mir_out = helpers::run_mir(&src, 42);
    let mir_opt_out = helpers::run_mir_optimized(&src, 42);
    helpers::assert_parity(&mir_out, &mir_opt_out);
}

#[test]
fn nn_deep_mir_opt_parity() {
    let src = helpers::load_cjc("bench_nn_deep.cjc");
    let mir_out = helpers::run_mir(&src, 42);
    let mir_opt_out = helpers::run_mir_optimized(&src, 42);
    helpers::assert_parity(&mir_out, &mir_opt_out);
}

#[test]
fn seed_stress_mir_opt_parity() {
    let src = helpers::load_cjc("bench_seed_stress.cjc");
    let mir_out = helpers::run_mir(&src, 42);
    let mir_opt_out = helpers::run_mir_optimized(&src, 42);
    helpers::assert_parity(&mir_out, &mir_opt_out);
}

#[test]
fn primitives_mir_opt_parity() {
    let src = helpers::load_cjc("bench_primitive_coverage.cjc");
    let mir_out = helpers::run_mir(&src, 42);
    let mir_opt_out = helpers::run_mir_optimized(&src, 42);
    helpers::assert_parity(&mir_out, &mir_opt_out);
}

#[test]
fn reorder_mir_opt_parity() {
    let src = helpers::load_cjc("bench_reorder_det.cjc");
    let mir_out = helpers::run_mir(&src, 42);
    let mir_opt_out = helpers::run_mir_optimized(&src, 42);
    helpers::assert_parity(&mir_out, &mir_opt_out);
}

#[test]
fn broadcast_chain_mir_opt_parity() {
    let src = helpers::load_cjc("bench_broadcast_chain.cjc");
    let mir_out = helpers::run_mir(&src, 42);
    let mir_opt_out = helpers::run_mir_optimized(&src, 42);
    helpers::assert_parity(&mir_out, &mir_opt_out);
}
