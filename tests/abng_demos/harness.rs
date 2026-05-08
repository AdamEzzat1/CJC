//! Shared harness for CJC-Lang ABNG demos.
//!
//! Mirrors `tests/chess_rl_v2/harness.rs` exactly — the same harness
//! pattern is the canonical way to drive a `.cjcl` source const through
//! both executors.

/// Which CJC-Lang executor to run a program through.
#[derive(Clone, Copy, Debug)]
pub enum Backend {
    /// AST tree-walk interpreter (v1).
    Eval,
    /// MIR register-machine executor (v2).
    Mir,
}

/// Run a full CJC-Lang program through the chosen backend with the
/// given RNG seed. Returns the captured `print(...)` output lines.
pub fn run(backend: Backend, src: &str, seed: u64) -> Vec<String> {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(
        !diags.has_errors(),
        "parse errors in ABNG demo program:\n{:#?}",
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
pub fn run_parity(src: &str, seed: u64) -> Vec<String> {
    let eval_out = run(Backend::Eval, src, seed);
    let mir_out = run(Backend::Mir, src, seed);
    assert_eq!(
        eval_out, mir_out,
        "AST↔MIR parity violation in ABNG demo\neval: {eval_out:?}\nmir: {mir_out:?}",
    );
    eval_out
}

/// Find a printed line of the form `key: value` and return the
/// trimmed value as a String.
pub fn extract_value(out: &[String], key: &str) -> String {
    let prefix = format!("{key}:");
    for line in out {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix(prefix.as_str()) {
            return rest.trim().to_string();
        }
    }
    panic!("output did not contain key `{key}:`\nlines: {out:?}");
}
