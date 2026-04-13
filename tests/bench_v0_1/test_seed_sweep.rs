/// Seed sweep tests: multiple seeds, each internally deterministic,
/// different seeds produce different results.
use super::helpers;

#[test]
fn seed_stress_sweep_internally_deterministic() {
    let src = helpers::load_cjc("bench_seed_stress.cjcl");
    let seeds: &[u64] = &[0, 1, 42, 99, 12345, 999999];

    for &seed in seeds {
        let out1 = helpers::run_eval(&src, seed);
        let out2 = helpers::run_eval(&src, seed);
        helpers::assert_deterministic(&out1, &out2);
    }
}

#[test]
fn seed_stress_cross_seed_differ() {
    let src = helpers::load_cjc("bench_seed_stress.cjcl");
    let out_42 = helpers::run_eval(&src, 42);
    let out_99 = helpers::run_eval(&src, 99);
    helpers::assert_seeds_differ(&out_42, &out_99);
}

#[test]
fn pipeline_seed_42_vs_99() {
    let src = helpers::load_cjc("bench_pipeline.cjcl");
    let out_42 = helpers::run_eval(&src, 42);
    let out_99 = helpers::run_eval(&src, 99);
    helpers::assert_seeds_differ(&out_42, &out_99);
}

#[test]
fn nn_deep_seed_42_vs_99() {
    let src = helpers::load_cjc("bench_nn_deep.cjcl");
    let out_42 = helpers::run_eval(&src, 42);
    let out_99 = helpers::run_eval(&src, 99);
    helpers::assert_seeds_differ(&out_42, &out_99);
}
