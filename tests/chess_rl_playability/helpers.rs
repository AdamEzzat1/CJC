//! Shared helpers for chess RL playability tests.
//!
//! Provides CJC program builders and parsing utilities for testing
//! the engine's interactive-play interface.

/// Run CJC source through MIR-exec with given seed, return output lines.
pub fn run_mir(src: &str, seed: u64) -> Vec<String> {
    let (program, _diag) = cjc_parser::parse_source(src);
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, seed)
        .unwrap_or_else(|e| panic!("MIR-exec failed: {e}"));
    executor.output
}

/// Parse a float from a given output line.
pub fn parse_float_at(out: &[String], idx: usize) -> f64 {
    out[idx].trim().parse::<f64>().unwrap_or_else(|_| {
        panic!("cannot parse float from line {}: {:?}", idx, out[idx])
    })
}

/// Parse an int from a given output line.
pub fn parse_int_at(out: &[String], idx: usize) -> i64 {
    out[idx].trim().parse::<i64>().unwrap_or_else(|_| {
        panic!("cannot parse int from line {}: {:?}", idx, out[idx])
    })
}

/// Parse space-separated ints from a single output line.
pub fn parse_int_list(line: &str) -> Vec<i64> {
    line.split_whitespace()
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<i64>().unwrap_or_else(|_| panic!("bad int: {s}")))
        .collect()
}

/// Chess environment CJC source.
pub fn chess_env_source() -> &'static str {
    crate::chess_rl_project::cjc_source::CHESS_ENV
}

/// RL agent CJC source.
pub fn rl_agent_source() -> &'static str {
    crate::chess_rl_project::cjc_source::RL_AGENT
}

/// Training loop CJC source.
pub fn training_source() -> &'static str {
    crate::chess_rl_project::cjc_source::TRAINING
}

/// Build a CJC program with chess env + custom main.
pub fn chess_program(main_body: &str) -> String {
    format!(
        "{}\nfn main() {{\n{}\n}}",
        chess_env_source(),
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
