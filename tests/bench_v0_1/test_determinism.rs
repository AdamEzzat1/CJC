/// Core determinism tests: run-twice-compare + eval/MIR parity.
/// Every benchmark is run twice with seed 42 and hashes are compared.
use super::helpers;

// =========================================================================
// Run-twice determinism (same seed, same engine, same hashes)
// =========================================================================

#[test]
fn pipeline_deterministic() {
    let src = helpers::load_cjc("bench_pipeline.cjcl");
    let out1 = helpers::run_eval(&src, 42);
    let out2 = helpers::run_eval(&src, 42);
    helpers::assert_deterministic(&out1, &out2);
}

#[test]
fn nn_deep_deterministic() {
    let src = helpers::load_cjc("bench_nn_deep.cjcl");
    let out1 = helpers::run_eval(&src, 42);
    let out2 = helpers::run_eval(&src, 42);
    helpers::assert_deterministic(&out1, &out2);
}

#[test]
fn seed_stress_deterministic() {
    let src = helpers::load_cjc("bench_seed_stress.cjcl");
    let out1 = helpers::run_eval(&src, 42);
    let out2 = helpers::run_eval(&src, 42);
    helpers::assert_deterministic(&out1, &out2);
}

#[test]
fn reorder_deterministic() {
    let src = helpers::load_cjc("bench_reorder_det.cjcl");
    let out1 = helpers::run_eval(&src, 42);
    let out2 = helpers::run_eval(&src, 42);
    helpers::assert_deterministic(&out1, &out2);
}

#[test]
fn primitives_deterministic() {
    let src = helpers::load_cjc("bench_primitive_coverage.cjcl");
    let out1 = helpers::run_eval(&src, 42);
    let out2 = helpers::run_eval(&src, 42);
    helpers::assert_deterministic(&out1, &out2);
}

#[test]
fn gc_boundary_deterministic() {
    let src = helpers::load_cjc("bench_gc_boundary.cjcl");
    let out1 = helpers::run_eval(&src, 42);
    let out2 = helpers::run_eval(&src, 42);
    helpers::assert_deterministic(&out1, &out2);
}

// =========================================================================
// Eval vs MIR parity (both engines produce identical hashes)
// =========================================================================

#[test]
fn pipeline_parity_eval_mir() {
    let src = helpers::load_cjc("bench_pipeline.cjcl");
    let eval_out = helpers::run_eval(&src, 42);
    let mir_out = helpers::run_mir(&src, 42);
    helpers::assert_parity(&eval_out, &mir_out);
}

#[test]
fn nn_deep_parity_eval_mir() {
    let src = helpers::load_cjc("bench_nn_deep.cjcl");
    let eval_out = helpers::run_eval(&src, 42);
    let mir_out = helpers::run_mir(&src, 42);
    helpers::assert_parity(&eval_out, &mir_out);
}

#[test]
fn seed_stress_parity() {
    let src = helpers::load_cjc("bench_seed_stress.cjcl");
    let eval_out = helpers::run_eval(&src, 42);
    let mir_out = helpers::run_mir(&src, 42);
    helpers::assert_parity(&eval_out, &mir_out);
}

#[test]
fn reorder_parity() {
    let src = helpers::load_cjc("bench_reorder_det.cjcl");
    let eval_out = helpers::run_eval(&src, 42);
    let mir_out = helpers::run_mir(&src, 42);
    helpers::assert_parity(&eval_out, &mir_out);
}

#[test]
fn primitives_parity() {
    let src = helpers::load_cjc("bench_primitive_coverage.cjcl");
    let eval_out = helpers::run_eval(&src, 42);
    let mir_out = helpers::run_mir(&src, 42);
    helpers::assert_parity(&eval_out, &mir_out);
}

#[test]
fn gc_boundary_parity() {
    let src = helpers::load_cjc("bench_gc_boundary.cjcl");
    let eval_out = helpers::run_eval(&src, 42);
    let mir_out = helpers::run_mir(&src, 42);
    helpers::assert_parity(&eval_out, &mir_out);
}
