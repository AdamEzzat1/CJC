//! Shared test helpers for the Mathematics Hardening Phase.

/// Run CJC source through MIR-exec and return output lines.
pub fn run_mir(src: &str) -> Vec<String> {
    let (program, _diag) = cjc_parser::parse_source(src);
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42)
        .unwrap_or_else(|e| panic!("MIR-exec failed: {e}"));
    executor.output
}

/// Run CJC source through MIR-exec with a specific seed.
pub fn run_mir_seeded(src: &str, seed: u64) -> Vec<String> {
    let (program, _diag) = cjc_parser::parse_source(src);
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, seed)
        .unwrap_or_else(|e| panic!("MIR-exec failed: {e}"));
    executor.output
}

/// Run CJC source through eval and return output lines.
pub fn run_eval(src: &str) -> Vec<String> {
    let (program, _diag) = cjc_parser::parse_source(src);
    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(&program).unwrap_or_else(|e| panic!("Eval failed: {e}"));
    interp.output.clone()
}

/// Parse a float from output string.
pub fn parse_float(s: &str) -> f64 {
    s.trim().parse::<f64>().unwrap_or_else(|e| panic!("Cannot parse float from '{s}': {e}"))
}

/// Assert that a float output is close to expected value.
pub fn assert_close(actual: f64, expected: f64, tol: f64) {
    let diff = (actual - expected).abs();
    assert!(
        diff < tol,
        "Expected {expected} ± {tol}, got {actual} (diff={diff})"
    );
}

/// Assert that a float output is NaN.
pub fn assert_nan(actual: f64) {
    assert!(actual.is_nan(), "Expected NaN, got {actual}");
}
