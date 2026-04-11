//! Test harness helpers for the chess RL v2 demo.
//!
//! All CJC-Lang source lives in `source.rs`. This module concatenates the
//! pieces into a single program with a `fn main()` of the test's choosing,
//! runs it through the requested executor, and parses the captured output.

use crate::chess_rl_v2::source;

/// Which CJC-Lang executor to run a program through.
#[derive(Clone, Copy, Debug)]
pub enum Backend {
    /// AST tree-walk interpreter (v1).
    Eval,
    /// MIR register-machine executor (v2).
    Mir,
}

/// Build a full CJC-Lang program: library prelude + a caller-supplied `main`.
///
/// `main_body` is the body of `fn main()` — everything between the outer braces.
pub fn build_program(main_body: &str) -> String {
    format!(
        "{prelude}\nfn main() {{\n{body}\n}}\n",
        prelude = source::PRELUDE,
        body = main_body,
    )
}

/// Run a program through the chosen backend with the given RNG seed.
/// Returns the captured `print(...)` output lines.
pub fn run(backend: Backend, main_body: &str, seed: u64) -> Vec<String> {
    let src = build_program(main_body);
    let (program, diags) = cjc_parser::parse_source(&src);
    assert!(
        !diags.has_errors(),
        "parse errors in chess_rl_v2 program: {:#?}",
        diags.diagnostics,
    );
    match backend {
        Backend::Eval => {
            let mut interp = cjc_eval::Interpreter::new(seed);
            interp
                .exec(&program)
                .unwrap_or_else(|e| panic!("eval failed: {e:?}"));
            interp.output
        }
        Backend::Mir => {
            let (_val, executor) = cjc_mir_exec::run_program_with_executor(&program, seed)
                .unwrap_or_else(|e| panic!("MIR-exec failed: {e:?}"));
            executor.output
        }
    }
}

/// Run on both backends and assert the output is byte-identical.
/// Returns the shared output on success.
pub fn run_parity(main_body: &str, seed: u64) -> Vec<String> {
    let eval_out = run(Backend::Eval, main_body, seed);
    let mir_out = run(Backend::Mir, main_body, seed);
    assert_eq!(
        eval_out, mir_out,
        "parity violation between cjc-eval and cjc-mir-exec\neval: {eval_out:?}\nmir: {mir_out:?}"
    );
    eval_out
}

/// Parse the first output line as an i64.
pub fn parse_i64(out: &[String]) -> i64 {
    out[0]
        .trim()
        .parse::<i64>()
        .unwrap_or_else(|_| panic!("expected i64 in first line, got {:?}", out[0]))
}

/// Parse the first output line as an f64.
#[allow(dead_code)]
pub fn parse_f64(out: &[String]) -> f64 {
    out[0]
        .trim()
        .parse::<f64>()
        .unwrap_or_else(|_| panic!("expected f64 in first line, got {:?}", out[0]))
}

/// Split a whitespace-separated line of ints.
pub fn parse_i64_line(line: &str) -> Vec<i64> {
    line.split_whitespace()
        .map(|s| {
            s.parse::<i64>()
                .unwrap_or_else(|_| panic!("bad i64 token: {s:?}"))
        })
        .collect()
}
