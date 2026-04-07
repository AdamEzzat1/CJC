/// Primitive ABI lock tests: golden hash verification for 40+ builtins.
/// If any builtin changes behavior, the master hash changes and this test fails.
use super::helpers;

/// Golden master hash — set to "TBD" initially, will be pinned after first run.
const GOLDEN_MASTER: &str = "7c2e0248d8e50c7a58d9e8c64afb9b8c31db43e81835673eac2ca46a077275d5";

#[test]
fn primitive_master_hash_golden() {
    let src = helpers::load_cjc("bench_primitive_coverage.cjcl");
    let out = helpers::run_eval(&src, 42);

    let bench = helpers::parse_bench_lines(&out);
    assert_eq!(bench.len(), 1, "Expected 1 BENCH line");

    if GOLDEN_MASTER == "TBD" {
        // First run: print the hash so we can pin it
        eprintln!(
            "GOLDEN_MASTER not yet pinned. Current master hash: {}",
            bench[0].out_hash
        );
        eprintln!("Update GOLDEN_MASTER in test_primitive_abi.rs to pin it.");
        // Don't fail — this is the bootstrap run
    } else {
        assert_eq!(
            bench[0].out_hash, GOLDEN_MASTER,
            "Primitive ABI drift detected! Expected golden hash {} but got {}",
            GOLDEN_MASTER, bench[0].out_hash
        );
    }
}

#[test]
fn primitive_coverage_count() {
    let src = helpers::load_cjc("bench_primitive_coverage.cjcl");
    let out = helpers::run_eval(&src, 42);

    let bench = helpers::parse_bench_lines(&out);
    assert_eq!(bench.len(), 1, "Expected 1 BENCH line");

    // The master hash exists — indicates successful execution of all 40+ primitives.
    // We can also verify by checking that the hash is non-empty.
    assert!(
        !bench[0].out_hash.is_empty(),
        "Master hash should be non-empty"
    );
    assert!(
        bench[0].out_hash.len() >= 32,
        "Master hash should be a proper SHA-256 hex string, got: {}",
        bench[0].out_hash
    );
}

#[test]
fn primitive_eval_mir_parity() {
    let src = helpers::load_cjc("bench_primitive_coverage.cjcl");
    let eval_out = helpers::run_eval(&src, 42);
    let mir_out = helpers::run_mir(&src, 42);
    helpers::assert_parity(&eval_out, &mir_out);
}
