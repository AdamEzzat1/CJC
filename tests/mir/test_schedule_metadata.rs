//! Integration tests for schedule metadata and enriched loop/reduction info.

use cjc_mir::cfg::CfgBuilder;
use cjc_mir::dominators::DominatorTree;
use cjc_mir::loop_analysis::{compute_loop_tree, LoopId, SchedulePlan};
use cjc_mir::reduction::{detect_reductions, AccumulatorSemantics, ReductionKind};
use cjc_mir::verify::verify_mir_legality;

/// Parse CJC source → MIR, build CFG for first user function, return loop tree.
fn loop_tree_from_source(src: &str) -> cjc_mir::loop_analysis::LoopTree {
    let (program, diags) = cjc_parser::parse_source(src);
    if diags.has_errors() {
        panic!("CJC parse failed for:\n{}\n", src);
    }
    let mir = cjc_mir_exec::lower_to_mir(&program);
    let func = mir
        .functions
        .iter()
        .find(|f| f.name != "__main")
        .unwrap_or(&mir.functions[0]);
    let cfg = CfgBuilder::build(&func.body);
    let domtree = DominatorTree::compute(&cfg);
    compute_loop_tree(&cfg, &domtree)
}

fn reductions_from_source(src: &str) -> cjc_mir::reduction::ReductionReport {
    let (program, diags) = cjc_parser::parse_source(src);
    if diags.has_errors() {
        panic!("CJC parse failed for:\n{}\n", src);
    }
    let mir = cjc_mir_exec::lower_to_mir(&program);
    detect_reductions(&mir, &[])
}

// ---------------------------------------------------------------------------
// Schedule metadata defaults
// ---------------------------------------------------------------------------

#[test]
fn test_all_loops_default_sequential_strict() {
    let src = r#"
fn nested(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let mut j: i64 = 0;
        while j < n {
            total = total + 1;
            j = j + 1;
        }
        i = i + 1;
    }
    return total;
}
print(nested(5));
"#;
    let tree = loop_tree_from_source(src);
    assert!(tree.len() >= 2, "should have at least 2 loops");
    for info in &tree.loops {
        assert_eq!(
            info.schedule,
            SchedulePlan::SequentialStrict,
            "all loops should default to SequentialStrict"
        );
    }
}

// ---------------------------------------------------------------------------
// Enriched loop metadata: num_exits
// ---------------------------------------------------------------------------

#[test]
fn test_num_exits_populated() {
    let src = r#"
fn count(n: i64) -> i64 {
    let mut i: i64 = 0;
    while i < n {
        i = i + 1;
    }
    return i;
}
print(count(10));
"#;
    let tree = loop_tree_from_source(src);
    assert_eq!(tree.len(), 1);
    let loop0 = tree.get(LoopId(0));
    assert!(
        loop0.num_exits > 0,
        "simple while loop should have at least 1 exit"
    );
    assert_eq!(
        loop0.num_exits,
        loop0.exit_blocks.len() as u32,
        "num_exits must equal exit_blocks.len()"
    );
}

// ---------------------------------------------------------------------------
// Enriched reduction metadata
// ---------------------------------------------------------------------------

#[test]
fn test_strict_fold_has_correct_metadata() {
    let src = r#"
fn sum_loop(n: i64) -> i64 {
    let mut acc: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        acc = acc + i;
        i = i + 1;
    }
    return acc;
}
print(sum_loop(10));
"#;
    let report = reductions_from_source(src);
    let fold = report
        .reductions
        .iter()
        .find(|r| r.accumulator_var == "acc" && r.kind == ReductionKind::StrictFold)
        .expect("should detect StrictFold on acc");

    assert!(fold.reassociation_forbidden, "StrictFold must forbid reassociation");
    assert!(fold.strict_order_required, "StrictFold must require strict order");
    assert_eq!(
        fold.accumulator_semantics,
        AccumulatorSemantics::Plain,
        "loop accumulation defaults to Plain semantics"
    );
}

#[test]
fn test_builtin_kahan_sum_metadata() {
    let src = r#"
fn compute(arr: Any) -> f64 {
    let result: f64 = kahan_sum(arr);
    return result;
}
print(compute(0.0));
"#;
    let report = reductions_from_source(src);
    let kahan = report
        .reductions
        .iter()
        .find(|r| r.builtin_name.as_deref() == Some("kahan_sum"))
        .expect("should detect kahan_sum builtin");

    assert!(kahan.reassociation_forbidden, "kahan_sum forbids reassociation");
    assert!(kahan.strict_order_required, "kahan_sum requires strict order");
    assert_eq!(
        kahan.accumulator_semantics,
        AccumulatorSemantics::Kahan,
        "kahan_sum has Kahan semantics"
    );
}

#[test]
fn test_builtin_binned_sum_metadata() {
    let src = r#"
fn compute(arr: Any) -> f64 {
    let result: f64 = binned_sum(arr);
    return result;
}
print(compute(0.0));
"#;
    let report = reductions_from_source(src);
    let binned = report
        .reductions
        .iter()
        .find(|r| r.builtin_name.as_deref() == Some("binned_sum"))
        .expect("should detect binned_sum builtin");

    assert!(
        !binned.reassociation_forbidden,
        "binned_sum allows reassociation within bins"
    );
    assert!(!binned.strict_order_required, "binned_sum is order-independent");
    assert_eq!(
        binned.accumulator_semantics,
        AccumulatorSemantics::Binned,
        "binned_sum has Binned semantics"
    );
}

// ---------------------------------------------------------------------------
// Verifier: schedule + metadata checks pass on valid programs
// ---------------------------------------------------------------------------

#[test]
fn test_verifier_passes_with_new_checks() {
    let src = r#"
fn compute(n: i64) -> i64 {
    let mut acc: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        acc = acc + i;
        i = i + 1;
    }
    return acc;
}
print(compute(100));
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut mir = cjc_mir_exec::lower_to_mir(&program);
    mir.build_all_cfgs();
    let report = verify_mir_legality(&mir);
    assert!(
        report.is_ok(),
        "valid program with loops should pass all checks: {:?}",
        report.errors()
    );
    // With CFGs built, we should have: cfg_structure, loop_integrity,
    // structural_bounds, reduction_contract, metadata_consistency, schedule_metadata.
    assert!(
        report.checks_total >= 5,
        "should run at least 5 checks (got {})",
        report.checks_total
    );
}

// ---------------------------------------------------------------------------
// Determinism: schedule metadata is identical across runs
// ---------------------------------------------------------------------------

#[test]
fn test_schedule_metadata_determinism() {
    let src = r#"
fn double_loop(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let mut j: i64 = 0;
        while j < n {
            total = total + 1;
            j = j + 1;
        }
        i = i + 1;
    }
    return total;
}
print(double_loop(3));
"#;
    let tree1 = loop_tree_from_source(src);
    let tree2 = loop_tree_from_source(src);

    assert_eq!(tree1.len(), tree2.len());
    for i in 0..tree1.len() {
        let a = &tree1.loops[i];
        let b = &tree2.loops[i];
        assert_eq!(a.schedule, b.schedule, "schedule must be deterministic");
        assert_eq!(a.num_exits, b.num_exits);
        assert_eq!(a.is_countable, b.is_countable);
        assert_eq!(a.trip_count_hint, b.trip_count_hint);
    }
}
