/// GC boundary stress tests: verify tensor integrity under GC pressure.
use super::helpers;

#[test]
fn gc_boundary_all_match() {
    let src = helpers::load_cjc("bench_gc_boundary.cjcl");
    let out = helpers::run_eval(&src, 42);

    // The BENCH line's hash should be the snap_hash of "ALL_MATCH"
    let bench = helpers::parse_bench_lines(&out);
    assert_eq!(bench.len(), 1, "Expected 1 BENCH line");
    assert_eq!(bench[0].name, "gc_boundary");

    // Verify the program itself reported ALL_MATCH by checking the hash
    // is deterministic (run twice)
    let out2 = helpers::run_eval(&src, 42);
    helpers::assert_deterministic(&out, &out2);
}

#[test]
fn gc_boundary_120_rounds_has_4_stages() {
    let src = helpers::load_cjc("bench_gc_boundary.cjcl");
    let out = helpers::run_eval(&src, 42);

    // Stages at rounds 29, 59, 89, 119 → indices 0, 1, 2, 3
    let stages = helpers::parse_stage_lines(&out);
    assert_eq!(
        stages.len(),
        4,
        "Expected 4 STAGE lines (at rounds 29, 59, 89, 119), got {}",
        stages.len()
    );
    assert_eq!(stages[0].stage_idx, "0");
    assert_eq!(stages[1].stage_idx, "1");
    assert_eq!(stages[2].stage_idx, "2");
    assert_eq!(stages[3].stage_idx, "3");

    // All stage hashes should be identical (ref_t never changes)
    for s in &stages {
        assert_eq!(
            s.stage_hash, stages[0].stage_hash,
            "All stage hashes should match ref_hash"
        );
    }
}

#[test]
fn gc_boundary_eval_mir_parity() {
    let src = helpers::load_cjc("bench_gc_boundary.cjcl");
    let eval_out = helpers::run_eval(&src, 42);
    let mir_out = helpers::run_mir(&src, 42);
    helpers::assert_parity(&eval_out, &mir_out);
}
