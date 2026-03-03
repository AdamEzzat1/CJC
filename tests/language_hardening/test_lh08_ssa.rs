//! LH08: SSA Form + Verifier tests
//!
//! Verifies:
//! - SSA construction from CFG (dominator tree, phi insertion, variable renaming)
//! - Phi nodes placed correctly at dominance frontiers
//! - SSA verifier passes for well-formed SSA
//! - Versioning correctness for linear, diamond, and loop CFGs
//! - Round-trip: CJC source → parse → MIR → CFG → SSA → verify
//! - Determinism of SSA construction

// ── Helper: parse + lower + build CFG + construct SSA ────────────

fn ssa_from_source(src: &str) -> (cjc_mir::cfg::MirCfg, cjc_mir::ssa::SsaForm, Vec<String>) {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);

    let hir = cjc_hir::AstLowering::new().lower_program(&program);
    let mir = cjc_mir::HirToMir::new().lower_program(&hir);

    // Find __main function (entry point)
    let main_fn = mir.functions.iter().find(|f| f.name == "__main").unwrap();
    let cfg = cjc_mir::cfg::CfgBuilder::build(&main_fn.body);
    let ssa = cjc_mir::ssa::SsaForm::construct(&cfg, &[]);
    (cfg, ssa, vec![])
}

fn ssa_from_fn_source(src: &str, fn_name: &str) -> (cjc_mir::cfg::MirCfg, cjc_mir::ssa::SsaForm, Vec<String>) {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);

    let hir = cjc_hir::AstLowering::new().lower_program(&program);
    let mir = cjc_mir::HirToMir::new().lower_program(&hir);

    let func = mir.functions.iter().find(|f| f.name == fn_name).unwrap();
    let params: Vec<String> = func.params.iter().map(|p| p.name.clone()).collect();
    let cfg = cjc_mir::cfg::CfgBuilder::build(&func.body);
    let ssa = cjc_mir::ssa::SsaForm::construct(&cfg, &params);
    (cfg, ssa, params)
}

fn assert_ssa_valid(src: &str) {
    let (cfg, ssa, _) = ssa_from_source(src);
    let result = cjc_mir::ssa::verify_ssa(&ssa, &cfg);
    assert!(result.is_ok(), "SSA verification failed: {:?}", result.err());
}

fn assert_fn_ssa_valid(src: &str, fn_name: &str) {
    let (cfg, ssa, _) = ssa_from_fn_source(src, fn_name);
    let result = cjc_mir::ssa::verify_ssa(&ssa, &cfg);
    assert!(result.is_ok(), "SSA verification for '{}' failed: {:?}", fn_name, result.err());
}

// ── Basic: simple let bindings ───────────────────────────────────

#[test]
fn test_ssa_simple_let() {
    assert_ssa_valid("let x: i64 = 42;");
}

#[test]
fn test_ssa_multiple_lets() {
    assert_ssa_valid(r#"
let x: i64 = 1;
let y: i64 = 2;
let z: i64 = x + y;
"#);
}

#[test]
fn test_ssa_print() {
    assert_ssa_valid("print(42);");
}

// ── Mutation: mutable variables ──────────────────────────────────

#[test]
fn test_ssa_mutation_linear() {
    assert_ssa_valid(r#"
let mut x: i64 = 1;
x = x + 1;
x = x * 3;
print(x);
"#);
}

// ── If/else: phi at merge point ──────────────────────────────────

#[test]
fn test_ssa_if_else_phi() {
    let (cfg, ssa, _) = ssa_from_source(r#"
let mut x: i64 = 0;
if true {
    x = 1;
} else {
    x = 2;
}
print(x);
"#);
    // Should have at least one phi (for x at the merge point).
    assert!(ssa.phi_count() >= 1, "if/else should produce phi for x");
    assert!(cjc_mir::ssa::verify_ssa(&ssa, &cfg).is_ok());
}

#[test]
fn test_ssa_if_no_else_phi() {
    // With only a then-branch modifying x, there should still be a phi
    // at the merge because x may or may not have been redefined.
    let (cfg, ssa, _) = ssa_from_source(r#"
let mut x: i64 = 0;
if true {
    x = 1;
}
print(x);
"#);
    assert!(ssa.phi_count() >= 1, "if-without-else should produce phi for x");
    assert!(cjc_mir::ssa::verify_ssa(&ssa, &cfg).is_ok());
}

// ── While loops: phi at loop header ──────────────────────────────

#[test]
fn test_ssa_while_loop_phi() {
    let (cfg, ssa, _) = ssa_from_source(r#"
let mut i: i64 = 0;
while i < 5 {
    i = i + 1;
}
print(i);
"#);
    assert!(ssa.phi_count() >= 1, "while loop should produce phi for i");
    assert!(cjc_mir::ssa::verify_ssa(&ssa, &cfg).is_ok());
}

#[test]
fn test_ssa_while_multiple_vars() {
    let (cfg, ssa, _) = ssa_from_source(r#"
let mut i: i64 = 0;
let mut sum: i64 = 0;
while i < 5 {
    sum = sum + i;
    i = i + 1;
}
print(sum);
"#);
    // Both i and sum should have phis at the loop header.
    assert!(ssa.phi_count() >= 2, "while with two mutated vars should produce >= 2 phis");
    assert!(cjc_mir::ssa::verify_ssa(&ssa, &cfg).is_ok());
}

// ── Functions with parameters ────────────────────────────────────

#[test]
fn test_ssa_fn_params() {
    let src = r#"
fn add(a: i64, b: i64) -> i64 {
    a + b
}
"#;
    assert_fn_ssa_valid(src, "add");
}

#[test]
fn test_ssa_fn_with_mutation() {
    let src = r#"
fn countdown(n: i64) -> i64 {
    let mut i: i64 = n;
    while i > 0 {
        i = i - 1;
    }
    i
}
"#;
    assert_fn_ssa_valid(src, "countdown");
}

#[test]
fn test_ssa_fn_with_if() {
    let src = r#"
fn abs(x: i64) -> i64 {
    if x < 0 { 0 - x } else { x }
}
"#;
    assert_fn_ssa_valid(src, "abs");
}

// ── Nested control flow ──────────────────────────────────────────

#[test]
fn test_ssa_nested_if() {
    assert_ssa_valid(r#"
let mut x: i64 = 0;
if true {
    if true {
        x = 1;
    } else {
        x = 2;
    }
} else {
    x = 3;
}
print(x);
"#);
}

#[test]
fn test_ssa_if_inside_while() {
    assert_ssa_valid(r#"
let mut i: i64 = 0;
let mut sum: i64 = 0;
while i < 10 {
    if i > 5 {
        sum = sum + i;
    }
    i = i + 1;
}
print(sum);
"#);
}

// ── Version counting ─────────────────────────────────────────────

#[test]
fn test_ssa_version_counts() {
    let (_, ssa, _) = ssa_from_source(r#"
let mut x: i64 = 0;
x = 1;
x = 2;
"#);
    // x should have 3 versions: initial let + 2 reassignments.
    assert!(
        ssa.version_counts.get("x").copied().unwrap_or(0) >= 3,
        "x should have >= 3 versions, got {:?}",
        ssa.version_counts.get("x")
    );
}

// ── No phis for immutable variables ──────────────────────────────

#[test]
fn test_ssa_no_phi_for_immutable() {
    let (_, ssa, _) = ssa_from_source(r#"
let x: i64 = 42;
if true {
    print(x);
} else {
    print(x);
}
"#);
    // x is never reassigned, so no phi should be needed for it.
    // (There may be other phis from the CFG structure, but x specifically
    // should not have one.)
    let x_phis: usize = ssa.phis.iter()
        .flat_map(|block_phis| block_phis.iter())
        .filter(|phi| phi.target.name == "x")
        .count();
    assert_eq!(x_phis, 0, "immutable x should have no phi nodes");
}

// ── Determinism ──────────────────────────────────────────────────

#[test]
fn test_ssa_deterministic_from_source() {
    let src = r#"
let mut x: i64 = 0;
if true {
    x = 1;
} else {
    x = 2;
}
print(x);
"#;
    let (cfg1, ssa1, _) = ssa_from_source(src);
    let (cfg2, ssa2, _) = ssa_from_source(src);

    assert_eq!(ssa1.phi_count(), ssa2.phi_count());
    assert_eq!(ssa1.version_counts, ssa2.version_counts);
    assert_eq!(ssa1.def_versions, ssa2.def_versions);

    // Verify both pass.
    assert!(cjc_mir::ssa::verify_ssa(&ssa1, &cfg1).is_ok());
    assert!(cjc_mir::ssa::verify_ssa(&ssa2, &cfg2).is_ok());
}

// ── All test programs verify ─────────────────────────────────────

#[test]
fn test_ssa_verifies_for_many_programs() {
    let programs = vec![
        "print(42);",
        "let x: i64 = 10; print(x);",
        "let mut x: i64 = 0; x = 1; print(x);",
        r#"let x: i64 = 1; if true { print(1); } else { print(2); }"#,
        r#"let mut i: i64 = 0; while i < 3 { i = i + 1; } print(i);"#,
        r#"print(1 + 2 * 3);"#,
        r#"let x: f64 = 3.14; print(x);"#,
    ];
    for (i, src) in programs.iter().enumerate() {
        let (cfg, ssa, _) = ssa_from_source(src);
        let result = cjc_mir::ssa::verify_ssa(&ssa, &cfg);
        assert!(result.is_ok(), "program {} failed SSA verification: {:?}", i, result.err());
    }
}

// ── Phi node structure ───────────────────────────────────────────

#[test]
fn test_ssa_phi_has_correct_source_count() {
    let (cfg, ssa, _) = ssa_from_source(r#"
let mut x: i64 = 0;
if true {
    x = 1;
} else {
    x = 2;
}
print(x);
"#);
    // Every phi should have exactly as many sources as its block has predecessors.
    let preds = cfg.predecessors();
    for (b, block_phis) in ssa.phis.iter().enumerate() {
        for phi in block_phis {
            assert_eq!(
                phi.sources.len(),
                preds[b].len(),
                "phi for {} in block {} should have {} sources (one per pred)",
                phi.target, b, preds[b].len()
            );
        }
    }
}

// ── Total version count grows with complexity ────────────────────

#[test]
fn test_ssa_total_versions_scale() {
    let (_, ssa_simple, _) = ssa_from_source("let x: i64 = 1;");
    let (_, ssa_complex, _) = ssa_from_source(r#"
let mut x: i64 = 0;
let mut y: i64 = 0;
if true { x = 1; y = 10; } else { x = 2; y = 20; }
"#);
    assert!(
        ssa_complex.total_versions() > ssa_simple.total_versions(),
        "complex program should have more SSA versions"
    );
}
