//! v0.1 Contract Tests: Statistics Surface
//!
//! Locks down: mean() absent as free fn, tensor .mean() works,
//! median/sd/variance/quantile/iqr present, eval/MIR parity.

// ── Helpers ──────────────────────────────────────────────────────

fn parse(src: &str) -> cjc_ast::Program {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    program
}

fn eval_output(src: &str) -> Vec<String> {
    let program = parse(src);
    let mut interp = cjc_eval::Interpreter::new(42);
    let _ = interp.exec(&program).unwrap();
    interp.output.clone()
}

fn mir_output(src: &str) -> Vec<String> {
    let program = parse(src);
    let (_, exec) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    exec.output.clone()
}

// ── Tests ────────────────────────────────────────────────────────

#[test]
fn mean_present_as_free_fn() {
    // mean() is now a free function (Bastion primitive ABI)
    // in addition to being a tensor dot method (t.mean())
    let src = "print(mean([1.0, 2.0, 3.0]));";
    let program = parse(src);
    let (_, exec) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    assert_eq!(exec.output[0], "2");
}

#[test]
fn tensor_dot_mean_works() {
    let src = r#"
let t = Tensor.from_vec([1.0, 2.0, 3.0], [3]);
print(t.mean());
"#;
    let out = eval_output(src);
    assert_eq!(out, vec!["2"]);
}

#[test]
fn median_present() {
    let out = eval_output("print(median([3.0, 1.0, 2.0]));");
    assert_eq!(out, vec!["2"]);
}

#[test]
fn sd_present() {
    let out = eval_output("print(sd([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]));");
    assert!(!out.is_empty(), "sd() should produce output");
    let val: f64 = out[0].parse().expect("sd should return a float");
    assert!(val > 0.0, "sd should be positive");
}

#[test]
fn variance_present() {
    let out = eval_output("print(variance([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]));");
    assert!(!out.is_empty(), "variance() should produce output");
    let val: f64 = out[0].parse().expect("variance should return a float");
    assert!(val > 0.0, "variance should be positive");
}

#[test]
fn quantile_present() {
    let out = eval_output("print(quantile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5));");
    assert_eq!(out, vec!["3"]);
}

#[test]
fn iqr_present() {
    let out = eval_output("print(iqr([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));");
    assert!(!out.is_empty(), "iqr() should produce output");
    let val: f64 = out[0].parse().expect("iqr should return a float");
    assert!(val > 0.0, "iqr should be positive");
}

#[test]
fn stats_parity_eval_mir() {
    let src = r#"
print(median([5.0, 1.0, 3.0]));
print(sd([1.0, 2.0, 3.0]));
"#;
    let eval_out = eval_output(src);
    let mir_out = mir_output(src);
    assert_eq!(eval_out, mir_out, "stats parity: eval vs MIR must match");
}
