//! Optimizer identity rewrites must not change program behaviour.
//!
//! Regressions: strength reduction applied `x * 1 => x`, `x + 0 => x`,
//! `x * 2 => x + x`, `true && x => x`, `!!x => x`, … to operands of unknown
//! type (MIR is untyped). That erased runtime type errors (`"a" * 1` became
//! `"a"`), turned errors into different values (`"a" * 2` became `"aa"`), and
//! changed float bits (`-0.0 + 0` is `+0.0`, but the rewrite kept `-0.0`).
//! Rewrites now require the operand's kind to be statically known.

/// Outcome of one run: `Ok(output lines)` or `Err(())` on a runtime error.
type Outcome = Result<Vec<String>, ()>;

fn run_eval(src: &str) -> Outcome {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors in {src:?}");
    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(&program).map(|_| interp.output.clone()).map_err(|_| ())
}

#[derive(Clone, Copy)]
enum Pipeline {
    /// No optimization.
    Plain,
    /// `--mir-opt`: CANA picks a per-function pass plan (it may skip
    /// strength reduction for a given program).
    CanaPlan,
    /// Monomorphize + `optimize_program`: always runs the full default pass
    /// sequence, including strength reduction — so these tests exercise the
    /// rewrites deterministically rather than only when CANA selects them.
    DefaultPasses,
}

fn run_mir(src: &str, p: Pipeline) -> Outcome {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors in {src:?}");
    let r = match p {
        Pipeline::Plain => cjc_mir_exec::run_program_with_executor(&program, 42),
        Pipeline::CanaPlan => cjc_mir_exec::run_program_optimized_with_executor(&program, 42),
        Pipeline::DefaultPasses => cjc_mir_exec::run_program_monomorphized_with_executor(&program, 42),
    };
    r.map(|(_, e)| e.output).map_err(|_| ())
}

/// All paths must agree on success/failure and, on success, output.
fn assert_parity(src: &str) -> Outcome {
    let eval = run_eval(src);
    let mir = run_mir(src, Pipeline::Plain);
    assert_eq!(mir, run_mir(src, Pipeline::CanaPlan), "--mir-opt changed the behaviour of:\n{src}");
    assert_eq!(mir, run_mir(src, Pipeline::DefaultPasses), "default passes changed the behaviour of:\n{src}");
    assert_eq!(eval, mir, "AST-eval vs MIR-exec differ for:\n{src}");
    mir
}

#[test]
fn string_times_one_keeps_type_error() {
    // Whatever `"a" * 1` does at runtime, the optimizer must do the same.
    assert_parity("let s: String = \"a\";\nprint(s * 1);");
    assert_parity("let s: String = \"a\";\nprint(s / 1);");
    assert_parity("let s: String = \"a\";\nprint(s - 0);");
}

#[test]
fn string_times_two_is_not_concatenation() {
    let out = assert_parity("let s: String = \"a\";\nprint(s * 2);");
    assert_ne!(out, Ok(vec!["aa".to_string()]), "`s * 2` was rewritten to `s + s`");
}

#[test]
fn negative_zero_plus_zero_keeps_its_bits() {
    // -0.0 + 0 is +0.0 at runtime; the optimizer must not keep -0.0.
    assert_parity("let z: f64 = -0.0;\nlet w: f64 = z + 0;\nprint(1.0 / w);");
    assert_parity("let z: f64 = -0.0;\nlet w: f64 = 0 + z;\nprint(1.0 / w);");
}

#[test]
fn logic_identities_keep_type_errors() {
    assert_parity("let n: i64 = 5;\nprint(true && n);");
    assert_parity("let n: i64 = 5;\nprint(n || false);");
    assert_parity("let n: i64 = 5;\nprint(!!n);");
    assert_parity("let s: String = \"a\";\nprint(-(-s));");
}

#[test]
fn sound_identities_still_hold() {
    // Known-kind operands: results unchanged by the optimizer.
    let out = assert_parity(
        "let a: i64 = 7;\nprint((a + 1) * 1);\nprint((2.5 * 2.0) - 0);\nprint((3 < 4) && true);",
    );
    let out = out.expect("sound identities must run");
    assert_eq!(out[0], "8");
    assert_eq!(out[2], "true");
}
