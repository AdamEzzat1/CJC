//! MIR-exec tail-call optimisation must only fire in true tail position.
//!
//! Regression (found by `fuzz_generated_eval_mir_parity`): `exec_body` treated
//! ANY body whose result is a direct call as a tail call — including the
//! branches of an `if` expression nested inside a call argument, and loop
//! bodies. The `TailCall` signal then escaped that expression: at top level it
//! surfaced as `Err(TailCall)`, and inside a function the trampoline jumped to
//! the callee mid-expression, silently returning the wrong value.

fn run_eval(src: &str) -> Vec<String> {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors in {src:?}");
    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(&program).expect("eval failed");
    interp.output
}

fn run_mir(src: &str, optimize: bool) -> Vec<String> {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors in {src:?}");
    let r = if optimize {
        cjc_mir_exec::run_program_optimized_with_executor(&program, 42)
    } else {
        cjc_mir_exec::run_program_with_executor(&program, 42)
    };
    r.expect("mir failed").1.output
}

fn run_all(src: &str) -> Vec<String> {
    let eval = run_eval(src);
    let mir = run_mir(src, false);
    assert_eq!(eval, mir, "AST-eval vs MIR-exec differ for:\n{src}");
    assert_eq!(mir, run_mir(src, true), "optimized MIR-exec differs for:\n{src}");
    mir
}

const HELPERS: &str = "
fn inc(a: i64) -> i64 { a + 1 }
fn add(a: i64, b: i64) -> i64 { a + b }
";

#[test]
fn if_expr_call_inside_top_level_argument() {
    // The exact shape the fuzzer found: previously Err(TailCall) at top level.
    let src = format!("{HELPERS}let c: bool = true;\nprint(add(1, if c {{ inc(10) }} else {{ 0 }}));");
    assert_eq!(run_all(&src), vec!["12"]);
}

#[test]
fn if_expr_call_inside_argument_within_function() {
    // Previously the trampoline jumped to `inc` and `outer` returned 11
    // instead of 1 + 11 = 12.
    let src = format!(
        "{HELPERS}fn outer(c: bool) -> i64 {{\n    add(1, if c {{ inc(10) }} else {{ 0 }})\n}}\nprint(outer(true));"
    );
    assert_eq!(run_all(&src), vec!["12"]);
}

#[test]
fn if_expr_call_as_binary_operand_within_function() {
    let src = format!(
        "{HELPERS}fn outer(c: bool) -> i64 {{\n    (if c {{ inc(10) }} else {{ 0 }}) * 2\n}}\nprint(outer(true));"
    );
    assert_eq!(run_all(&src), vec!["22"]);
}

#[test]
fn call_as_loop_body_result_within_function() {
    // A `for` body whose result is a call must run every iteration.
    let src = format!(
        "{HELPERS}fn outer() -> i64 {{\n    let mut n: i64 = 0;\n    for k in 0..3 {{\n        n = inc(n);\n        inc(k)\n    }}\n    n\n}}\nprint(outer());"
    );
    assert_eq!(run_all(&src), vec!["3"]);
}

#[test]
fn genuine_tail_recursion_still_optimized() {
    // Tail calls through if-expression branches in result position must keep
    // trampolining; 200k frames of real recursion would overflow the stack.
    let src = "
fn count(n: i64, acc: i64) -> i64 {
    if n == 0 { acc } else { count(n - 1, acc + 1) }
}
print(count(200000, 0));";
    let (program, _) = cjc_parser::parse_source(src);
    let (_, exec) = cjc_mir_exec::run_program_with_executor(&program, 42).expect("mir failed");
    assert_eq!(exec.output, vec!["200000"]);
}
