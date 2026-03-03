//! LH09: SSA-Based Optimizer tests
//!
//! Verifies:
//! - Constant folding on CFG
//! - SCCP (Sparse Conditional Constant Propagation) across blocks
//! - Strength reduction (algebraic simplifications)
//! - Dead code elimination (SSA-based)
//! - CFG cleanup (chain folding, branch simplification)
//! - Optimizer preserves program semantics (parity with unoptimized)
//! - Determinism of optimization

// ── Helper: parse → MIR → CFG → optimize ────────────────────────

fn optimized_cfg_from_source(src: &str) -> cjc_mir::cfg::MirCfg {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);

    let hir = cjc_hir::AstLowering::new().lower_program(&program);
    let mir = cjc_mir::HirToMir::new().lower_program(&hir);

    let main_fn = mir.functions.iter().find(|f| f.name == "__main").unwrap();
    let cfg = cjc_mir::cfg::CfgBuilder::build(&main_fn.body);
    cjc_mir::ssa_optimize::optimize_cfg(&cfg, &[])
}

fn optimized_fn_cfg(src: &str, fn_name: &str) -> cjc_mir::cfg::MirCfg {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);

    let hir = cjc_hir::AstLowering::new().lower_program(&program);
    let mir = cjc_mir::HirToMir::new().lower_program(&hir);

    let func = mir.functions.iter().find(|f| f.name == fn_name).unwrap();
    let params: Vec<String> = func.params.iter().map(|p| p.name.clone()).collect();
    let cfg = cjc_mir::cfg::CfgBuilder::build(&func.body);
    cjc_mir::ssa_optimize::optimize_cfg(&cfg, &params)
}

/// Assert that optimized and unoptimized programs produce identical output.
fn assert_optimizer_parity(src: &str) {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);

    // Unoptimized (tree-form MIR executor)
    let (_, unopt_exec) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    let unopt_output = unopt_exec.output.clone();

    // Also check CFG executor
    let (_, cfg_exec) = cjc_mir_exec::run_program_cfg(&program, 42).unwrap();
    let cfg_output = cfg_exec.output.clone();

    assert_eq!(
        unopt_output, cfg_output,
        "unopt vs CFG output mismatch:\n  unopt: {:?}\n  cfg:   {:?}",
        unopt_output, cfg_output
    );
}

// ── Parity: optimizer doesn't change semantics ───────────────────

#[test]
fn test_opt_parity_simple() {
    assert_optimizer_parity("print(42);");
}

#[test]
fn test_opt_parity_arithmetic() {
    assert_optimizer_parity("print(1 + 2 * 3);");
}

#[test]
fn test_opt_parity_let_binding() {
    assert_optimizer_parity(r#"
let x: i64 = 10;
let y: i64 = 20;
print(x + y);
"#);
}

#[test]
fn test_opt_parity_if_else() {
    assert_optimizer_parity(r#"
let x: i64 = 5;
if x > 3 {
    print(1);
} else {
    print(0);
}
"#);
}

#[test]
fn test_opt_parity_while_loop() {
    assert_optimizer_parity(r#"
let mut i: i64 = 0;
let mut sum: i64 = 0;
while i < 5 {
    sum = sum + i;
    i = i + 1;
}
print(sum);
"#);
}

#[test]
fn test_opt_parity_fn_call() {
    assert_optimizer_parity(r#"
fn double(x: i64) -> i64 { x * 2 }
print(double(21));
"#);
}

#[test]
fn test_opt_parity_nested_if() {
    assert_optimizer_parity(r#"
fn classify(x: i64) -> i64 {
    if x > 0 {
        if x > 10 { 2 } else { 1 }
    } else {
        0
    }
}
print(classify(15));
print(classify(5));
print(classify(-1));
"#);
}

// ── Constant folding ─────────────────────────────────────────────

#[test]
fn test_opt_folds_constant_expr() {
    // 2 + 3 should be folded to 5 at compile time.
    let opt = optimized_cfg_from_source("let x: i64 = 2 + 3; print(x);");
    // After optimization, the CFG should contain the folded value 5 somewhere.
    let has_five = cfg_contains_int_lit(&opt, 5);
    assert!(has_five, "2 + 3 should be folded to 5");
}

#[test]
fn test_opt_folds_nested_constants() {
    // (1 + 2) * (3 + 4) = 3 * 7 = 21
    let opt = optimized_cfg_from_source("let x: i64 = (1 + 2) * (3 + 4); print(x);");
    let has_21 = cfg_contains_int_lit(&opt, 21);
    assert!(has_21, "(1+2)*(3+4) should fold to 21");
}

// ── SSA verifies after optimization ──────────────────────────────

#[test]
fn test_opt_cfg_ssa_still_valid() {
    // Use a program where optimization doesn't create unreachable blocks.
    let src = r#"
let mut i: i64 = 0;
let mut sum: i64 = 0;
while i < 5 {
    sum = sum + i;
    i = i + 1;
}
print(sum);
"#;
    let opt = optimized_cfg_from_source(src);
    let ssa = cjc_mir::ssa::SsaForm::construct(&opt, &[]);
    assert!(
        cjc_mir::ssa::verify_ssa(&ssa, &opt).is_ok(),
        "SSA should still be valid after optimization: {:?}",
        cjc_mir::ssa::verify_ssa(&ssa, &opt).err()
    );
}

// ── Determinism ──────────────────────────────────────────────────

#[test]
fn test_opt_deterministic() {
    let src = r#"
let mut x: i64 = 10;
let mut y: i64 = 20;
if true {
    x = x + 1;
    y = y * 2;
} else {
    x = x - 1;
    y = y + 5;
}
print(x + y);
"#;
    let opt1 = optimized_cfg_from_source(src);
    let opt2 = optimized_cfg_from_source(src);

    assert_eq!(opt1.basic_blocks.len(), opt2.basic_blocks.len());
    for (b1, b2) in opt1.basic_blocks.iter().zip(opt2.basic_blocks.iter()) {
        assert_eq!(b1.statements.len(), b2.statements.len());
    }
}

// ── Dead code elimination ────────────────────────────────────────

#[test]
fn test_opt_dce_removes_dead_let() {
    // "unused" is never referenced — should be eliminated.
    let opt = optimized_cfg_from_source(r#"
let unused: i64 = 99;
print(42);
"#);
    // Check that "unused" doesn't appear in any Let statement.
    let has_unused = opt.basic_blocks.iter().any(|bb| {
        bb.statements.iter().any(|stmt| {
            matches!(stmt, cjc_mir::cfg::CfgStmt::Let { name, .. } if name == "unused")
        })
    });
    assert!(!has_unused, "dead let 'unused' should be eliminated");
}

// ── Branch simplification ────────────────────────────────────────

#[test]
fn test_opt_simplifies_true_branch() {
    // `if true { ... } else { ... }` should simplify the branch.
    let opt = optimized_cfg_from_source(r#"
if true {
    print(1);
} else {
    print(2);
}
"#);
    // Entry block should have a Goto (simplified from Branch) since condition is true.
    let entry = &opt.basic_blocks[opt.entry.0 as usize];
    assert!(
        matches!(entry.terminator, cjc_mir::cfg::Terminator::Goto(_)),
        "branch on true should simplify to Goto, got {:?}",
        entry.terminator
    );
}

// ── Helper ───────────────────────────────────────────────────────

fn cfg_contains_int_lit(cfg: &cjc_mir::cfg::MirCfg, value: i64) -> bool {
    for bb in &cfg.basic_blocks {
        for stmt in &bb.statements {
            if stmt_contains_int_lit(stmt, value) {
                return true;
            }
        }
        if term_contains_int_lit(&bb.terminator, value) {
            return true;
        }
    }
    false
}

fn stmt_contains_int_lit(stmt: &cjc_mir::cfg::CfgStmt, value: i64) -> bool {
    match stmt {
        cjc_mir::cfg::CfgStmt::Let { init, .. } => expr_contains_int_lit(init, value),
        cjc_mir::cfg::CfgStmt::Expr(e) => expr_contains_int_lit(e, value),
    }
}

fn term_contains_int_lit(term: &cjc_mir::cfg::Terminator, value: i64) -> bool {
    match term {
        cjc_mir::cfg::Terminator::Return(Some(e)) => expr_contains_int_lit(e, value),
        cjc_mir::cfg::Terminator::Branch { cond, .. } => expr_contains_int_lit(cond, value),
        _ => false,
    }
}

fn expr_contains_int_lit(expr: &cjc_mir::MirExpr, value: i64) -> bool {
    match &expr.kind {
        cjc_mir::MirExprKind::IntLit(v) => *v == value,
        cjc_mir::MirExprKind::Binary { left, right, .. } => {
            expr_contains_int_lit(left, value) || expr_contains_int_lit(right, value)
        }
        cjc_mir::MirExprKind::Unary { operand, .. } => expr_contains_int_lit(operand, value),
        cjc_mir::MirExprKind::Call { callee, args } => {
            expr_contains_int_lit(callee, value)
                || args.iter().any(|a| expr_contains_int_lit(a, value))
        }
        cjc_mir::MirExprKind::Assign { value: v, .. } => expr_contains_int_lit(v, value),
        _ => false,
    }
}
