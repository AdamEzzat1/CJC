//! Runtime Policy Layer — wiring tests.
//!
//! Each test runs a tiny `.cjcl` snippet through both `cjc-eval` (AST) and
//! `cjc-mir-exec` (MIR) and asserts byte-identical printed output. Because the
//! `runtime_policy_*` builtins are stateless dispatch arms in
//! `cjc_runtime::builtins::dispatch_builtin` (called first by both executors),
//! "wiring" here means: the same builtin is reachable and behaves identically
//! from both pipelines.
//!
//! **Thread-local note:** the policy lives in a `thread_local! RefCell` shared
//! by both backends within a single test thread. `assert_parity` runs Eval then
//! Mir on the same thread, so every snippet begins with `runtime_policy_reset()`
//! to be self-contained — the same discipline the grad_graph tests apply with
//! `grad_graph_new()`.
//!
//! Machine-dependent builtins (`runtime_policy_threads`, `runtime_policy_summary`)
//! are checked for parity only (eval == mir), never an absolute value, because
//! the resolved thread count depends on the host core count.

#![allow(clippy::needless_raw_string_hashes)]

#[derive(Clone, Copy, Debug)]
enum Backend {
    Eval,
    Mir,
}

fn run_result(backend: Backend, body: &str, seed: u64) -> Result<Vec<String>, String> {
    let src = format!("fn main() {{\n{body}\n}}\n");
    let (program, diags) = cjc_parser::parse_source(&src);
    assert!(
        !diags.has_errors(),
        "parse errors:\n{:#?}\nsource:\n{src}",
        diags.diagnostics,
    );
    match backend {
        Backend::Eval => {
            let mut interp = cjc_eval::Interpreter::new(seed);
            match interp.exec(&program) {
                Ok(_) => Ok(interp.output),
                Err(e) => Err(format!("{e:?}")),
            }
        }
        Backend::Mir => match cjc_mir_exec::run_program_with_executor(&program, seed) {
            Ok((_v, exec)) => Ok(exec.output),
            Err(e) => Err(format!("{e:?}")),
        },
    }
}

fn run(backend: Backend, body: &str, seed: u64) -> Vec<String> {
    run_result(backend, body, seed)
        .unwrap_or_else(|e| panic!("{backend:?} failed for snippet:\n{body}\nerror: {e}"))
}

/// Assert AST↔MIR produce byte-identical output for `body`.
fn assert_parity(label: &str, body: &str) {
    let eval_out = run(Backend::Eval, body, 42);
    let mir_out = run(Backend::Mir, body, 42);
    assert_eq!(
        eval_out, mir_out,
        "[{label}] AST↔MIR parity violation\n  eval: {eval_out:?}\n  mir : {mir_out:?}",
    );
}

/// Assert AST↔MIR parity *and* that the joined output contains `needle`. Robust
/// to scalar-formatting differences (we only require the token to appear).
fn assert_contains(label: &str, body: &str, needle: &str) {
    let eval_out = run(Backend::Eval, body, 42);
    let mir_out = run(Backend::Mir, body, 42);
    assert_eq!(
        eval_out, mir_out,
        "[{label}] AST↔MIR parity violation\n  eval: {eval_out:?}\n  mir : {mir_out:?}",
    );
    let joined = eval_out.join("\n");
    assert!(
        joined.contains(needle),
        "[{label}] expected output to contain `{needle}`, got:\n{joined}",
    );
}

/// Assert that both backends *reject* `body` with a runtime error.
fn assert_both_error(label: &str, body: &str) {
    let eval = run_result(Backend::Eval, body, 42);
    let mir = run_result(Backend::Mir, body, 42);
    assert!(
        eval.is_err(),
        "[{label}] expected eval to error, got Ok: {eval:?}",
    );
    assert!(
        mir.is_err(),
        "[{label}] expected MIR to error, got Ok: {mir:?}",
    );
}

// ─── Reset + defaults ────────────────────────────────────────────────────

#[test]
fn wiring_reset_returns_zero() {
    assert_contains("reset returns 0", "print(runtime_policy_reset());", "0");
}

#[test]
fn wiring_default_thermal_is_balanced() {
    assert_contains(
        "default thermal balanced",
        r#"
        runtime_policy_reset();
        print(runtime_policy_thermal_mode());
        "#,
        "balanced",
    );
}

#[test]
fn wiring_default_batch_is_128() {
    assert_contains(
        "default batch 128",
        r#"
        runtime_policy_reset();
        print(runtime_policy_batch_size());
        "#,
        "128",
    );
}

#[test]
fn wiring_default_audit_is_full() {
    assert_contains(
        "default audit full",
        r#"
        runtime_policy_reset();
        print(runtime_policy_audit_mode());
        "#,
        "full",
    );
}

#[test]
fn wiring_default_numeric_is_kahan() {
    assert_contains(
        "default numeric kahan",
        r#"
        runtime_policy_reset();
        print(runtime_policy_numeric_mode());
        "#,
        "kahan",
    );
}

// ─── Thermal mode ──────────────────────────────────────────────────────────

#[test]
fn wiring_set_thermal_cool() {
    assert_contains(
        "set thermal cool",
        r#"
        runtime_policy_reset();
        runtime_policy_set_thermal_mode("cool");
        print(runtime_policy_thermal_mode());
        "#,
        "cool",
    );
}

#[test]
fn wiring_set_thermal_returns_canonical() {
    assert_contains(
        "set thermal returns canonical",
        r#"
        runtime_policy_reset();
        print(runtime_policy_set_thermal_mode("max-perf"));
        "#,
        "max-perf",
    );
}

#[test]
fn wiring_set_thermal_alias_normalizes() {
    assert_contains(
        "maxperf alias normalizes",
        r#"
        runtime_policy_reset();
        runtime_policy_set_thermal_mode("maxperf");
        print(runtime_policy_thermal_mode());
        "#,
        "max-perf",
    );
}

#[test]
fn wiring_cool_preset_batch_is_32() {
    assert_contains(
        "cool preset batch 32",
        r#"
        runtime_policy_reset();
        runtime_policy_set_thermal_mode("cool");
        print(runtime_policy_batch_size());
        "#,
        "32",
    );
}

#[test]
fn wiring_cool_preset_audit_is_summary() {
    assert_contains(
        "cool preset audit summary",
        r#"
        runtime_policy_reset();
        runtime_policy_set_thermal_mode("cool");
        print(runtime_policy_audit_mode());
        "#,
        "summary",
    );
}

// ─── Per-field setters + precedence ──────────────────────────────────────

#[test]
fn wiring_set_batch_size() {
    assert_contains(
        "set batch size",
        r#"
        runtime_policy_reset();
        runtime_policy_set_batch_size(64);
        print(runtime_policy_batch_size());
        "#,
        "64",
    );
}

#[test]
fn wiring_set_audit_forensic() {
    assert_contains(
        "set audit forensic",
        r#"
        runtime_policy_reset();
        runtime_policy_set_audit_mode("forensic");
        print(runtime_policy_audit_mode());
        "#,
        "forensic",
    );
}

#[test]
fn wiring_set_numeric_binned() {
    assert_contains(
        "set numeric binned",
        r#"
        runtime_policy_reset();
        runtime_policy_set_numeric_mode("binned");
        print(runtime_policy_numeric_mode());
        "#,
        "binned",
    );
}

#[test]
fn wiring_profile_then_override_precedence() {
    // Profile sets batch to 512; explicit override must win.
    assert_contains(
        "profile then override",
        r#"
        runtime_policy_reset();
        runtime_policy_set_thermal_mode("max-perf");
        runtime_policy_set_batch_size(16);
        print(runtime_policy_batch_size());
        "#,
        "16",
    );
}

// ─── Adaptive (race-to-idle) scheduling ──────────────────────────────────

#[test]
fn wiring_default_adaptive_is_true() {
    assert_contains(
        "default adaptive true",
        r#"
        runtime_policy_reset();
        print(runtime_policy_adaptive());
        "#,
        "true",
    );
}

#[test]
fn wiring_set_adaptive_false() {
    assert_contains(
        "set adaptive false",
        r#"
        runtime_policy_reset();
        runtime_policy_set_adaptive(false);
        print(runtime_policy_adaptive());
        "#,
        "false",
    );
}

#[test]
fn wiring_summary_reports_adaptive() {
    assert_contains(
        "summary has adaptive",
        r#"
        runtime_policy_reset();
        print(runtime_policy_summary());
        "#,
        "adaptive=",
    );
}

#[test]
fn wiring_set_adaptive_bad_type_errors_in_both() {
    assert_both_error(
        "adaptive bad type",
        r#"runtime_policy_set_adaptive("yes");"#,
    );
}

/// The headline determinism guarantee for Phase 2: a parallel-path matmul
/// (256×256 → the rayon path) must produce byte-identical output whether
/// adaptive scheduling runs it at full width (burst) or throttled (fixed),
/// and across both executors. Concurrency changes; the numbers do not.
#[test]
fn adaptive_does_not_change_matmul_results() {
    let body = r#"
        let v: Tensor = Tensor.arange(0.0, 65536.0, 1.0);
        let m: Tensor = v.reshape([256, 256]);
        let p: Tensor = matmul(m, m);
        print(p);
    "#;
    let adaptive_on = run(
        Backend::Eval,
        &format!("runtime_policy_reset();\n{body}"),
        42,
    );
    let adaptive_off = run(
        Backend::Eval,
        &format!("runtime_policy_reset();\nruntime_policy_set_adaptive(false);\n{body}"),
        42,
    );
    assert_eq!(
        adaptive_on, adaptive_off,
        "adaptive on/off must not change matmul output",
    );
    let mir = run(Backend::Mir, &format!("runtime_policy_reset();\n{body}"), 42);
    assert_eq!(adaptive_on, mir, "MIR diverged from eval under adaptive scheduling");
}

// ─── Machine-dependent (parity only) ─────────────────────────────────────

#[test]
fn wiring_threads_parity() {
    assert_parity(
        "threads parity",
        r#"
        runtime_policy_reset();
        print(runtime_policy_threads());
        "#,
    );
}

#[test]
fn wiring_set_threads_parity() {
    assert_parity(
        "set_threads parity",
        r#"
        runtime_policy_reset();
        print(runtime_policy_set_threads(2));
        "#,
    );
}

#[test]
fn wiring_summary_parity() {
    assert_parity(
        "summary parity",
        r#"
        runtime_policy_reset();
        print(runtime_policy_summary());
        "#,
    );
}

// ─── Energy model (deterministic, safe in program logic) ─────────────────

#[test]
fn wiring_energy_estimate_parity() {
    assert_parity(
        "energy_estimate parity",
        "print(energy_estimate(1000000, 2000));",
    );
}

#[test]
fn wiring_energy_estimate_zero_is_zero() {
    assert_contains("energy zero", "print(energy_estimate(0, 0));", "0");
}

#[test]
fn wiring_energy_estimate_accepts_float_args() {
    // FLOP/byte counts often arrive as f64 in numeric CJC-Lang code.
    assert_parity(
        "energy accepts float args",
        "print(energy_estimate(1000000.0, 2000.0));",
    );
}

#[test]
fn wiring_energy_per_flop_parity() {
    assert_parity("energy_per_flop parity", "print(energy_per_flop());");
}

#[test]
fn wiring_energy_per_byte_parity() {
    assert_parity("energy_per_byte parity", "print(energy_per_byte());");
}

// ─── Determinism ─────────────────────────────────────────────────────────

#[test]
fn wiring_energy_is_deterministic_across_runs() {
    let body = "print(energy_estimate(123456, 789));";
    let first = run(Backend::Eval, body, 42);
    for _ in 0..5 {
        assert_eq!(run(Backend::Eval, body, 42), first, "energy estimate drifted");
    }
    // And the MIR pipeline agrees.
    assert_eq!(run(Backend::Mir, body, 42), first, "MIR diverged from eval");
}

// ─── Error parity (invalid inputs rejected the same way) ──────────────────

#[test]
fn wiring_bad_thermal_mode_errors_in_both() {
    assert_both_error(
        "bad thermal mode",
        r#"runtime_policy_set_thermal_mode("blazing");"#,
    );
}

#[test]
fn wiring_bad_audit_mode_errors_in_both() {
    assert_both_error(
        "bad audit mode",
        r#"runtime_policy_set_audit_mode("paranoid");"#,
    );
}

#[test]
fn wiring_bad_numeric_mode_errors_in_both() {
    assert_both_error(
        "bad numeric mode",
        r#"runtime_policy_set_numeric_mode("float80");"#,
    );
}

#[test]
fn wiring_energy_estimate_arity_errors_in_both() {
    assert_both_error("energy arity", "print(energy_estimate(5));");
}
