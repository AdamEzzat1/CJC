//! Parity tests for PINN builtins.
//!
//! Validates that PINN training builtins produce identical results in both
//! cjc-eval (AST interpreter) and cjc-mir-exec (MIR executor).

/// Run CJC source through eval, return output lines.
fn run_eval(src: &str, seed: u64) -> Vec<String> {
    let (program, _diag) = cjc_parser::parse_source(src);
    let mut interp = cjc_eval::Interpreter::new(seed);
    let _ = interp.exec(&program);
    interp.output
}

/// Run CJC source through MIR-exec, return output lines.
fn run_mir(src: &str, seed: u64) -> Vec<String> {
    let (program, _diag) = cjc_parser::parse_source(src);
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, seed)
        .unwrap_or_else(|e| panic!("MIR-exec failed: {e}"));
    executor.output
}

/// Assert both executors produce identical output.
fn assert_parity(src: &str) {
    let eval_out = run_eval(src, 42);
    let mir_out = run_mir(src, 42);
    assert_eq!(
        eval_out, mir_out,
        "Parity mismatch!\nEval: {eval_out:?}\nMIR:  {mir_out:?}"
    );
}

// ── PINN Burgers Parity ───────────────────────────────────────

#[test]
fn parity_pinn_burgers() {
    // Small training run: verify both executors return the same l2_error
    assert_parity(
        r#"
        let result: Any = pinn_train_burgers(10, 0.001, 4, 0.003183098861837907, 42);
        print(result.l2_error);
        print(result.n_epochs);
    "#,
    );
}

// ── PINN Poisson Parity ──────────────────────────────────────

#[test]
fn parity_pinn_poisson() {
    assert_parity(
        r#"
        let result: Any = pinn_train_poisson(10, 0.001, 4, 42);
        print(result.l2_error);
        print(result.n_epochs);
    "#,
    );
}

// ── PINN Heat Parity ─────────────────────────────────────────

#[test]
fn parity_pinn_heat() {
    assert_parity(
        r#"
        let result: Any = pinn_train_heat(10, 0.001, 4, 0.01, 42);
        print(result.l2_error);
        print(result.n_epochs);
    "#,
    );
}

// ── PINN Result Struct Access Parity ──────────────────────────

#[test]
fn parity_pinn_burgers_all_fields() {
    assert_parity(
        r#"
        let r: Any = pinn_train_burgers(5, 0.001, 2, 0.003183098861837907, 42);
        print(r.l2_error);
        print(r.max_error);
        print(r.mean_residual);
        print(r.n_epochs);
    "#,
    );
}

#[test]
fn parity_pinn_heat_all_fields() {
    assert_parity(
        r#"
        let r: Any = pinn_train_heat(5, 0.001, 2, 0.01, 42);
        print(r.l2_error);
        print(r.max_error);
        print(r.mean_residual);
        print(r.n_epochs);
    "#,
    );
}

// ── New PDE Solver Parity Tests ──────────────────────────────

#[test]
fn parity_pinn_wave() {
    assert_parity(
        r#"
        let r: Any = pinn_train_wave(5, 0.001, 4, 1.0, 42);
        print(r.l2_error);
        print(r.n_epochs);
    "#,
    );
}

#[test]
fn parity_pinn_helmholtz() {
    assert_parity(
        r#"
        let r: Any = pinn_train_helmholtz(5, 0.001, 4, 1.0, 42);
        print(r.l2_error);
        print(r.n_epochs);
    "#,
    );
}

#[test]
fn parity_pinn_diffreact() {
    assert_parity(
        r#"
        let r: Any = pinn_train_diffreact(5, 0.001, 4, 0.01, 1.0, 42);
        print(r.mean_residual);
        print(r.n_epochs);
    "#,
    );
}

#[test]
fn parity_pinn_allen_cahn() {
    assert_parity(
        r#"
        let r: Any = pinn_train_allen_cahn(5, 0.001, 4, 0.01, 42);
        print(r.mean_residual);
        print(r.n_epochs);
    "#,
    );
}

#[test]
fn parity_pinn_kdv() {
    assert_parity(
        r#"
        let r: Any = pinn_train_kdv(5, 0.001, 4, 42);
        print(r.mean_residual);
        print(r.n_epochs);
    "#,
    );
}

#[test]
fn parity_pinn_schrodinger() {
    assert_parity(
        r#"
        let r: Any = pinn_train_schrodinger(5, 0.001, 4, 42);
        print(r.mean_residual);
        print(r.n_epochs);
    "#,
    );
}

#[test]
fn parity_pinn_navier_stokes() {
    assert_parity(
        r#"
        let r: Any = pinn_train_navier_stokes(3, 0.001, 2, 0.01, 42);
        print(r.mean_residual);
        print(r.n_epochs);
    "#,
    );
}

#[test]
fn parity_pinn_burgers_2d() {
    assert_parity(
        r#"
        let r: Any = pinn_train_burgers_2d(3, 0.001, 2, 0.003183098861837907, 42);
        print(r.mean_residual);
        print(r.n_epochs);
    "#,
    );
}
