/// Shared test helpers for Reinforcement Learning (Language Hardening) test suite.

/// Run a CJC program through MIR-exec and return captured output lines.
pub fn run_mir(src: &str) -> Vec<String> {
    let (program, diag) = cjc_parser::parse_source(src);
    if diag.has_errors() {
        panic!(
            "Parse errors:\n{}",
            diag.render_all(src, "<test>")
        );
    }
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42)
        .unwrap_or_else(|e| panic!("MIR-exec failed: {e}"));
    executor.output
}

/// Run a CJC program through AST-eval and return captured output lines.
pub fn run_eval(src: &str) -> Vec<String> {
    let (program, diag) = cjc_parser::parse_source(src);
    if diag.has_errors() {
        panic!(
            "Parse errors:\n{}",
            diag.render_all(src, "<test>")
        );
    }
    let mut interp = cjc_eval::Interpreter::new(42);
    interp
        .exec(&program)
        .unwrap_or_else(|e| panic!("Eval failed: {e}"));
    interp.output
}

/// Parse output as f64.
pub fn parse_float(s: &str) -> f64 {
    s.trim().parse::<f64>().unwrap_or_else(|e| {
        panic!("cannot parse '{s}' as f64: {e}")
    })
}

/// Assert two floats are approximately equal.
pub fn assert_close(actual: f64, expected: f64, tol: f64) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= tol,
        "expected {expected} +/- {tol}, got {actual} (diff={diff})"
    );
}
