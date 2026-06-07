//! Shared executor harness for fused-primitive wiring tests.
//!
//! These helpers were copy-pasted three times before (`test_adam_step.rs`,
//! `tests/fused_matmul_dot/wiring.rs`, `tests/fused_matmul_norm/wiring.rs`)
//! and any change had to be made in lock-step. Extracted here so future
//! fused-primitive suites can include the file via `#[path]` instead of
//! duplicating the bodies.
//!
//! This directory is NOT registered as a `[[test]]` target in Cargo.toml —
//! cargo doesn't auto-discover `tests/<subdir>/mod.rs` files. Each test
//! suite that wants the helpers references this file with a `#[path]`
//! attribute, e.g.
//!
//! ```ignore
//! #[path = "../fused_test_helpers/mod.rs"]
//! mod helpers;
//! use helpers::{run_eval, run_mir, run_parity};
//! ```

/// Parse a CJC-Lang source string, run it through the AST tree-walk
/// interpreter (`cjc_eval`), and return captured stdout lines.
///
/// Panics with a descriptive message on parse errors or runtime errors.
pub fn run_eval(src: &str, seed: u64) -> Vec<String> {
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    let mut interp = cjc_eval::Interpreter::new(seed);
    interp
        .exec(&prog)
        .unwrap_or_else(|e| panic!("eval failed: {e:?}"));
    interp.output
}

/// Parse a CJC-Lang source string, run it through the MIR executor
/// (`cjc_mir_exec`), and return captured stdout lines.
pub fn run_mir(src: &str, seed: u64) -> Vec<String> {
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    let (_val, executor) = cjc_mir_exec::run_program_with_executor(&prog, seed)
        .unwrap_or_else(|e| panic!("mir-exec failed: {e:?}"));
    executor.output
}

/// Run a source string through BOTH executors and assert byte-identical
/// stdout — the load-bearing AST↔MIR parity check used by every
/// fused-primitive wiring test. Returns the shared output (or panics).
pub fn run_parity(src: &str, seed: u64) -> Vec<String> {
    let a = run_eval(src, seed);
    let b = run_mir(src, seed);
    assert_eq!(
        a, b,
        "parity violation between cjc-eval and cjc-mir-exec\neval: {a:?}\nmir: {b:?}"
    );
    a
}
