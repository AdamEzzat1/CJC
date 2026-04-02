//! Integration tests for inspect/diagnostics module — end-to-end from CJC source.

use cjc_mir::cfg::CfgBuilder;
use cjc_mir::dominators::DominatorTree;
use cjc_mir::inspect;
use cjc_mir::loop_analysis::compute_loop_tree;
use cjc_mir::reduction::detect_reductions;
use cjc_mir::verify::verify_mir_legality;

fn inspect_from_source(src: &str) -> (String, String, String, String) {
    let (program, diags) = cjc_parser::parse_source(src);
    if diags.has_errors() {
        panic!("CJC parse failed for:\n{}\n", src);
    }
    let mut mir = cjc_mir_exec::lower_to_mir(&program);
    mir.build_all_cfgs();

    let func = mir
        .functions
        .iter()
        .find(|f| f.name != "__main")
        .unwrap_or(&mir.functions[0]);
    let cfg = CfgBuilder::build(&func.body);
    let domtree = DominatorTree::compute(&cfg);
    let loop_tree = compute_loop_tree(&cfg, &domtree);
    let reduction_report = detect_reductions(&mir, &[]);
    let legality_report = verify_mir_legality(&mir);

    let lt_dump = inspect::dump_loop_tree(&loop_tree);
    let red_dump = inspect::dump_reduction_report(&reduction_report);
    let leg_dump = inspect::dump_legality_report(&legality_report);
    let sched_dump = inspect::dump_schedule_summary(&loop_tree);

    (lt_dump, red_dump, leg_dump, sched_dump)
}

#[test]
fn test_inspect_simple_loop() {
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
    let (lt, _red, leg, sched) = inspect_from_source(src);

    assert!(lt.contains("LoopTree"), "loop tree dump should have header");
    assert!(lt.contains("Loop L0"), "should show loop L0");
    assert!(lt.contains("sequential_strict"), "default schedule");
    assert!(leg.contains("checks passed"), "legality should report checks");
    assert!(sched.contains("sequential_strict"), "schedule summary");
}

#[test]
fn test_inspect_reduction_loop() {
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
    let (_lt, red, _leg, _sched) = inspect_from_source(src);

    assert!(red.contains("ReductionReport"), "reduction dump should have header");
    assert!(red.contains("StrictFold"), "should show StrictFold");
    assert!(red.contains("reassoc_forbidden=true"), "should show metadata");
}

#[test]
fn test_inspect_determinism() {
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
print(nested(3));
"#;
    let (lt1, red1, leg1, sched1) = inspect_from_source(src);
    let (lt2, red2, leg2, sched2) = inspect_from_source(src);

    assert_eq!(lt1, lt2, "loop tree dump must be deterministic");
    assert_eq!(red1, red2, "reduction dump must be deterministic");
    assert_eq!(leg1, leg2, "legality dump must be deterministic");
    assert_eq!(sched1, sched2, "schedule summary must be deterministic");
}

#[test]
fn test_inspect_builtin_reductions() {
    let src = r#"
fn compute(arr: Any) -> f64 {
    let s: f64 = sum(arr);
    let m: f64 = mean(arr);
    return s + m;
}
print(compute(0.0));
"#;
    let (_lt, red, _leg, _sched) = inspect_from_source(src);

    assert!(red.contains("BuiltinReduction"), "should show BuiltinReduction");
    assert!(red.contains("sum"), "should show sum builtin");
    assert!(red.contains("mean"), "should show mean builtin");
}

#[test]
fn test_inspect_no_loops() {
    let src = "let x: i64 = 42;\nprint(x);\n";
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mir = cjc_mir_exec::lower_to_mir(&program);
    let func = &mir.functions[0];
    let cfg = CfgBuilder::build(&func.body);
    let domtree = DominatorTree::compute(&cfg);
    let loop_tree = compute_loop_tree(&cfg, &domtree);

    let lt_dump = inspect::dump_loop_tree(&loop_tree);
    assert!(lt_dump.contains("0 loops"), "should report 0 loops");

    let sched = inspect::dump_schedule_summary(&loop_tree);
    assert!(sched.contains("no loops"), "should say no loops");
}
