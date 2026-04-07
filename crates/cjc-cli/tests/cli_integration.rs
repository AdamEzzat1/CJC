//! CLI integration tests for `cjc` binary.
//!
//! Tests run the built binary via `cargo run` and verify:
//! - Successful execution of valid programs
//! - Error exit codes for invalid input
//! - Stderr output contains diagnostic spans/messages
//! - Deterministic output for the same input

use std::process::Command;

/// Helper: run the cjc binary with the given args, returning (stdout, stderr, exit_code).
fn run_cjc(args: &[&str]) -> (String, String, i32) {
    let output = Command::new(env!("CARGO_BIN_EXE_cjc"))
        .args(args)
        .output()
        .expect("failed to execute cjc binary");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let code = output.status.code().unwrap_or(-1);
    (stdout, stderr, code)
}

#[test]
fn test_run_valid_program() {
    let (stdout, _stderr, code) = run_cjc(&[
        "run",
        "tests/fixtures/hello.cjcl",
    ]);
    assert_eq!(code, 0, "expected exit code 0 for valid program");
    assert!(
        stdout.contains("hello from CJC"),
        "expected output to contain 'hello from CJC', got: {stdout}"
    );
}

#[test]
fn test_error_exit_code_missing_file() {
    let (_stdout, stderr, code) = run_cjc(&["run", "nonexistent_file.cjcl"]);
    assert_ne!(code, 0, "expected non-zero exit code for missing file");
    assert!(
        stderr.contains("error"),
        "expected stderr to contain 'error', got: {stderr}"
    );
}

#[test]
fn test_no_args_shows_usage() {
    let (_stdout, stderr, code) = run_cjc(&[]);
    assert_ne!(code, 0, "expected non-zero exit code for no args");
    assert!(
        stderr.contains("Usage"),
        "expected stderr to contain usage info, got: {stderr}"
    );
}

#[test]
fn test_lex_command() {
    let (stdout, _stderr, code) = run_cjc(&["lex", "tests/fixtures/hello.cjcl"]);
    assert_eq!(code, 0, "lex command should succeed");
    // Should contain token output
    assert!(
        stdout.contains("Fn") || stdout.contains("fn"),
        "expected lex output to contain 'fn' token, got: {stdout}"
    );
}

#[test]
fn test_parse_command() {
    let (stdout, _stderr, code) = run_cjc(&["parse", "tests/fixtures/hello.cjcl"]);
    assert_eq!(code, 0, "parse command should succeed");
    assert!(
        stdout.contains("main"),
        "expected parse output to contain function name 'main', got: {stdout}"
    );
}

#[test]
fn test_check_command_valid() {
    let (stdout, stderr, code) = run_cjc(&["check", "tests/fixtures/hello.cjcl"]);
    assert_eq!(code, 0, "check command should succeed on valid program");
    let combined = format!("{}{}", stdout, stderr);
    assert!(
        combined.contains("OK"),
        "expected check output to contain 'OK', got stdout: {stdout}, stderr: {stderr}"
    );
}

#[test]
fn test_deterministic_output() {
    let (stdout1, _, code1) = run_cjc(&["run", "tests/fixtures/deterministic.cjcl"]);
    let (stdout2, _, code2) = run_cjc(&["run", "tests/fixtures/deterministic.cjcl"]);
    assert_eq!(code1, 0);
    assert_eq!(code2, 0);
    assert_eq!(
        stdout1, stdout2,
        "expected identical output for two runs of the same program"
    );
}

#[test]
fn test_unknown_command_error() {
    let (_stdout, stderr, code) = run_cjc(&["frobnicate", "tests/fixtures/hello.cjcl"]);
    assert_ne!(code, 0, "expected non-zero exit for unknown command");
    assert!(
        stderr.contains("unknown command"),
        "expected 'unknown command' in stderr, got: {stderr}"
    );
}

#[test]
fn test_seed_flag() {
    // Running with explicit seed should succeed
    let (stdout1, _, code) = run_cjc(&[
        "run",
        "tests/fixtures/deterministic.cjcl",
        "--seed",
        "123",
    ]);
    assert_eq!(code, 0);
    // Same seed should produce same output
    let (stdout2, _, _) = run_cjc(&[
        "run",
        "tests/fixtures/deterministic.cjcl",
        "--seed",
        "123",
    ]);
    assert_eq!(stdout1, stdout2, "same seed must produce same output");
}
