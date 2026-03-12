//! Shared helpers for chess RL hardening tests.
//!
//! Reuses the CJC source constants from the existing chess_rl_project.

/// Run CJC source through MIR-exec with given seed, return output lines.
pub fn run_mir(src: &str, seed: u64) -> Vec<String> {
    let (program, _diag) = cjc_parser::parse_source(src);
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, seed)
        .unwrap_or_else(|e| panic!("MIR-exec failed: {e}"));
    executor.output
}

/// Parse a float from the first output line.
pub fn parse_float(out: &[String]) -> f64 {
    out[0].trim().parse::<f64>().unwrap_or_else(|_| {
        panic!("cannot parse float from: {:?}", out[0])
    })
}

/// Parse an int from the first output line.
pub fn parse_int(out: &[String]) -> i64 {
    out[0].trim().parse::<i64>().unwrap_or_else(|_| {
        panic!("cannot parse int from: {:?}", out[0])
    })
}

/// Parse space-separated ints from a single output line.
pub fn parse_int_list(line: &str) -> Vec<i64> {
    line.split_whitespace()
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<i64>().unwrap_or_else(|_| panic!("bad int: {s}")))
        .collect()
}

/// The chess environment CJC source.
pub const CHESS_ENV: &str = include_str!("../../tests/chess_rl_project/cjc_source.rs");

// We can't include_str the constants directly, so we duplicate the env/agent/training
// source code references. The test files build their CJC programs by concatenating
// the source strings from the original chess_rl_project module.

/// Build a CJC program with chess env + custom main.
pub fn chess_program(main_body: &str) -> String {
    format!(
        "{}\nfn main() {{\n{}\n}}",
        crate::chess_rl_hardening::helpers::chess_env_source(),
        main_body
    )
}

/// Build a CJC program with chess env + RL agent + custom main.
pub fn chess_agent_program(main_body: &str) -> String {
    format!(
        "{}\n{}\nfn main() {{\n{}\n}}",
        chess_env_source(),
        rl_agent_source(),
        main_body
    )
}

/// Build a CJC program with chess env + RL agent + training + custom main.
pub fn full_program(main_body: &str) -> String {
    format!(
        "{}\n{}\n{}\nfn main() {{\n{}\n}}",
        chess_env_source(),
        rl_agent_source(),
        training_source(),
        main_body
    )
}

/// Chess environment CJC source (piece encoding, board, movegen, legality).
pub fn chess_env_source() -> &'static str {
    // We reference the same constant from cjc_source.rs
    // Rather than duplicating, we use the constants defined there.
    // But since we can't cross-reference test modules easily, we inline here.
    crate::chess_rl_project::cjc_source::CHESS_ENV
}

/// RL agent CJC source (network, forward pass, action selection, REINFORCE).
pub fn rl_agent_source() -> &'static str {
    crate::chess_rl_project::cjc_source::RL_AGENT
}

/// Training loop CJC source.
pub fn training_source() -> &'static str {
    crate::chess_rl_project::cjc_source::TRAINING
}
