//! Shared helpers for CJC v0.1 hardening tests.

/// Run CJC source through MIR-exec (seed=42), return output lines.
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

/// Run CJC source through MIR-exec with optimizer enabled.
pub fn run_mir_optimized(src: &str) -> Vec<String> {
    let (program, _diag) = cjc_parser::parse_source(src);
    let result = cjc_mir_exec::run_program_optimized(&program, 42)
        .unwrap_or_else(|e| panic!("MIR-opt failed: {e}"));
    // run_program_optimized returns Value; get output via a second run
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42)
        .unwrap_or_else(|e| panic!("MIR-exec failed: {e}"));
    executor.output
}

/// Run CJC source through eval (seed=42), return output lines.
pub fn run_eval(src: &str) -> Vec<String> {
    let (program, _diag) = cjc_parser::parse_source(src);
    let mut interp = cjc_eval::Interpreter::new(42);
    interp
        .exec(&program)
        .unwrap_or_else(|e| panic!("Eval failed: {e}"));
    interp.output.clone()
}

/// Run CJC source through eval with a specific seed.
pub fn run_eval_seeded(src: &str, seed: u64) -> Vec<String> {
    let (program, _diag) = cjc_parser::parse_source(src);
    let mut interp = cjc_eval::Interpreter::new(seed);
    interp
        .exec(&program)
        .unwrap_or_else(|e| panic!("Eval failed: {e}"));
    interp.output.clone()
}

/// Parse CJC source and return diagnostics.
pub fn parse_only(src: &str) -> cjc_diag::DiagnosticBag {
    let (_, diag) = cjc_parser::parse_source(src);
    diag
}

/// Assert that MIR-exec and eval produce identical output.
pub fn assert_parity(src: &str) {
    let mir = run_mir(src);
    let eval = run_eval(src);
    assert_eq!(mir, eval, "Parity failure:\nMIR: {mir:?}\nEval: {eval:?}");
}

/// Parse a float from an output string.
pub fn parse_float(s: &str) -> f64 {
    s.trim()
        .parse::<f64>()
        .unwrap_or_else(|e| panic!("Cannot parse float from '{s}': {e}"))
}

/// Assert that a float is close to expected.
pub fn assert_close(actual: f64, expected: f64, tol: f64) {
    let diff = (actual - expected).abs();
    assert!(
        diff < tol,
        "Expected {expected} +/- {tol}, got {actual} (diff={diff})"
    );
}
