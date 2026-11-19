//! Non-ASCII text in string / f-string / raw-string / regex literals must reach
//! the runtime verbatim, identically in both executors.
//!
//! Regression: the lexer pushed literal content one byte per `char` (and the
//! parser did the same for f-string text), so `"é"` became `"Ã©"` — or worse,
//! double-encoded through an f-string — and `len("é")` reported 4, not 2.

fn run_eval(src: &str) -> Vec<String> {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors in {src:?}");
    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(&program).expect("eval failed");
    interp.output
}

fn run_mir(src: &str) -> Vec<String> {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors in {src:?}");
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).expect("mir failed");
    executor.output
}

/// Runs `src` through both executors, asserts parity and determinism, and
/// returns the shared output.
fn run_both(src: &str) -> Vec<String> {
    let eval = run_eval(src);
    let mir = run_mir(src);
    assert_eq!(eval, mir, "AST-eval vs MIR-exec output differs for {src:?}");
    assert_eq!(mir, run_mir(src), "MIR-exec not deterministic for {src:?}");
    mir
}

#[test]
fn string_literal_non_ascii_round_trips() {
    let out = run_both(r#"print("héllo 日本 🦀");"#);
    assert_eq!(out, vec!["héllo 日本 🦀"]);
}

#[test]
fn string_len_counts_utf8_bytes_of_source_text() {
    // 'é' is 2 UTF-8 bytes, '日' is 3, '🦀' is 4.
    let out = run_both(r#"print(len("é")); print(len("日")); print(len("🦀"));"#);
    assert_eq!(out, vec!["2", "3", "4"]);
}

#[test]
fn fstring_non_ascii_literal_and_hole() {
    let out = run_both(
        r#"let x: i64 = 7;
print(f"é={x} ü");
print(f"{"日本"}!");"#,
    );
    assert_eq!(out, vec!["é=7 ü", "日本!"]);
}

#[test]
fn raw_string_non_ascii() {
    let out = run_both(r##"print(r"日本\n"); print(r#"é"q"#);"##);
    assert_eq!(out, vec![r"日本\n", "é\"q"]);
}

#[test]
fn regex_literal_matches_non_ascii() {
    let out = run_both(r#"print("café" ~= /é/); print("cafe" ~= /é/);"#);
    assert_eq!(out, vec!["true", "false"]);
}

#[test]
fn non_ascii_outside_literal_is_a_diagnostic_not_a_panic() {
    let (_, diags) = cjc_parser::parse_source("let x: i64 = 1 ʄ 2;");
    assert!(diags.has_errors());
}
