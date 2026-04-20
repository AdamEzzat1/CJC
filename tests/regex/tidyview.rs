//! TidyView integration tests using new regex engine features.
//!
//! Each test runs CJC-Lang source through both the AST eval (v1) and MIR-exec
//! (v2) pipelines, asserting identical output (parity). Tests exercise:
//!   - str_detect with POSIX class patterns
//!   - str_detect with inline-flag patterns
//!   - str_extract with counted repetition (4-digit year extraction)
//!   - regex_or / regex_seq / regex_explain builtins from CJC-Lang source
//!
//! CJC-Lang syntax rules followed throughout:
//!   - Function params REQUIRE type annotations: `fn f(x: i64)`
//!   - NO semicolons after while/if/for blocks
//!   - `array_push` RETURNS new array
//!   - Use `Any` for dynamic/polymorphic types

use cjc_eval::Interpreter;
use cjc_parser::parse_source;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Parse and execute source via AST eval, return print output lines.
fn eval_output(src: &str) -> Vec<String> {
    let (program, diag) = parse_source(src);
    assert!(
        !diag.has_errors(),
        "Parse errors:\n{}",
        diag.render_all(src, "<tidyview_test>")
    );
    let mut interp = Interpreter::new(42);
    match interp.exec(&program) {
        Ok(_) => {}
        Err(e) => panic!("eval error: {:?}", e),
    }
    interp.output.clone()
}

/// Parse and execute source via MIR-exec, return print output lines.
fn mir_output(src: &str) -> Vec<String> {
    let (program, diag) = parse_source(src);
    assert!(
        !diag.has_errors(),
        "Parse errors:\n{}",
        diag.render_all(src, "<tidyview_test>")
    );
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42)
        .expect("MIR exec failed");
    executor.output.clone()
}

/// Assert AST eval and MIR exec produce identical print output.
fn assert_parity(src: &str) {
    let ast_out = eval_output(src);
    let mir_out = mir_output(src);
    assert_eq!(
        ast_out, mir_out,
        "Parity failure:\n  AST: {:?}\n  MIR: {:?}\nSource:\n{}",
        ast_out, mir_out, src.trim()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// str_detect with POSIX class pattern
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_str_detect_posix_alpha_all_letters() {
    // Detect whether the string contains only alpha characters
    let src = r#"
        let word = "hello";
        print(str_detect(word, "^[[:alpha:]]+$"));
    "#;
    let out = eval_output(src);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn test_str_detect_posix_alpha_all_letters_parity() {
    assert_parity(r#"
        let word = "hello";
        print(str_detect(word, "^[[:alpha:]]+$"));
    "#);
}

#[test]
fn test_str_detect_posix_alpha_rejects_mixed() {
    let src = r#"
        let mixed = "hello123";
        print(str_detect(mixed, "^[[:alpha:]]+$"));
    "#;
    let out = eval_output(src);
    assert_eq!(out, vec!["false"]);
}

#[test]
fn test_str_detect_posix_alpha_rejects_mixed_parity() {
    assert_parity(r#"
        let mixed = "hello123";
        print(str_detect(mixed, "^[[:alpha:]]+$"));
    "#);
}

#[test]
fn test_str_detect_posix_digit_class() {
    let src = r#"
        let code = "12345";
        print(str_detect(code, "^[[:digit:]]+$"));
    "#;
    let out = eval_output(src);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn test_str_detect_posix_digit_class_parity() {
    assert_parity(r#"
        let code = "12345";
        print(str_detect(code, "^[[:digit:]]+$"));
    "#);
}

#[test]
fn test_str_detect_posix_punct_in_sentence() {
    // A sentence ending in "." should be detected
    let src = r#"
        let sentence = "Hello, world!";
        print(str_detect(sentence, "[[:punct:]]"));
    "#;
    let out = eval_output(src);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn test_str_detect_posix_punct_in_sentence_parity() {
    assert_parity(r#"
        let sentence = "Hello, world!";
        print(str_detect(sentence, "[[:punct:]]"));
    "#);
}

// ─────────────────────────────────────────────────────────────────────────────
// str_detect with inline-flag pattern
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_str_detect_inline_flag_case_insensitive() {
    // (?i)hello should match "HELLO"
    let src = r#"
        let s = "HELLO WORLD";
        print(str_detect(s, "(?i)hello"));
    "#;
    let out = eval_output(src);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn test_str_detect_inline_flag_case_insensitive_parity() {
    assert_parity(r#"
        let s = "HELLO WORLD";
        print(str_detect(s, "(?i)hello"));
    "#);
}

#[test]
fn test_str_detect_inline_flag_no_match_without_flag() {
    // Without (?i), uppercase HELLO should not match lowercase "hello" pattern
    let src = r#"
        let s = "HELLO";
        print(str_detect(s, "hello"));
    "#;
    let out = eval_output(src);
    assert_eq!(out, vec!["false"]);
}

#[test]
fn test_str_detect_inline_flag_no_match_without_flag_parity() {
    assert_parity(r#"
        let s = "HELLO";
        print(str_detect(s, "hello"));
    "#);
}

#[test]
fn test_str_detect_inline_flag_scoped() {
    // (?i:hello) — scoped flag, only hello is case-insensitive
    let src = r#"
        let s = "HELLO world";
        print(str_detect(s, "(?i:hello) world"));
    "#;
    let out = eval_output(src);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn test_str_detect_inline_flag_scoped_parity() {
    assert_parity(r#"
        let s = "HELLO world";
        print(str_detect(s, "(?i:hello) world"));
    "#);
}

// ─────────────────────────────────────────────────────────────────────────────
// str_extract with counted repetition (4-digit year)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_str_extract_counted_year() {
    // Extract a 4-digit year from a date string
    let src = r#"
        let date = "Published on 2024-03-15";
        let year = str_extract(date, "[[:digit:]]{4}");
        print(year);
    "#;
    let out = eval_output(src);
    assert_eq!(out, vec!["2024"]);
}

#[test]
fn test_str_extract_counted_year_parity() {
    assert_parity(r#"
        let date = "Published on 2024-03-15";
        let year = str_extract(date, "[[:digit:]]{4}");
        print(year);
    "#);
}

#[test]
fn test_str_extract_counted_no_match_returns_empty() {
    // No 4-digit sequence → empty string
    let src = r#"
        let s = "abc";
        let result = str_extract(s, "[[:digit:]]{4}");
        print(result);
    "#;
    let out = eval_output(src);
    assert_eq!(out, vec![""]);
}

#[test]
fn test_str_extract_counted_no_match_parity() {
    assert_parity(r#"
        let s = "abc";
        let result = str_extract(s, "[[:digit:]]{4}");
        print(result);
    "#);
}

#[test]
fn test_str_extract_counted_repetition_word() {
    // Extract exactly 5 alpha characters
    let src = r#"
        let s = "id: hello world";
        let result = str_extract(s, "[[:alpha:]]{5}");
        print(result);
    "#;
    let out = eval_output(src);
    assert_eq!(out, vec!["hello"]);
}

#[test]
fn test_str_extract_counted_repetition_word_parity() {
    assert_parity(r#"
        let s = "id: hello world";
        let result = str_extract(s, "[[:alpha:]]{5}");
        print(result);
    "#);
}

// ─────────────────────────────────────────────────────────────────────────────
// regex_or builtin from CJC-Lang source
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_regex_or_builtin_basic() {
    // regex_or("cat", "dog") → "(?:cat)|(?:dog)"
    // Then use str_detect with the composed pattern
    let src = r#"
        let pat = regex_or("cat", "dog");
        print(str_detect("I have a cat", pat));
    "#;
    let out = eval_output(src);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn test_regex_or_builtin_basic_parity() {
    assert_parity(r#"
        let pat = regex_or("cat", "dog");
        print(str_detect("I have a dog", pat));
    "#);
}

#[test]
fn test_regex_or_builtin_no_match() {
    let src = r#"
        let pat = regex_or("cat", "dog");
        print(str_detect("I have a bird", pat));
    "#;
    let out = eval_output(src);
    assert_eq!(out, vec!["false"]);
}

#[test]
fn test_regex_or_builtin_no_match_parity() {
    assert_parity(r#"
        let pat = regex_or("cat", "dog");
        print(str_detect("I have a bird", pat));
    "#);
}

#[test]
fn test_regex_or_builtin_three_alternatives() {
    let src = r#"
        let pat = regex_or("red", "green", "blue");
        print(str_detect("my favorite color is green", pat));
    "#;
    let out = eval_output(src);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn test_regex_or_builtin_three_alternatives_parity() {
    assert_parity(r#"
        let pat = regex_or("red", "green", "blue");
        print(str_detect("my favorite color is green", pat));
    "#);
}

// ─────────────────────────────────────────────────────────────────────────────
// regex_seq builtin from CJC-Lang source
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_regex_seq_builtin_basic() {
    // regex_seq("hello", " ", "world") → "(?:hello)(?: )(?:world)"
    let src = r#"
        let pat = regex_seq("hello", " ", "world");
        print(str_detect("hello world", pat));
    "#;
    let out = eval_output(src);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn test_regex_seq_builtin_basic_parity() {
    assert_parity(r#"
        let pat = regex_seq("hello", " ", "world");
        print(str_detect("hello world", pat));
    "#);
}

#[test]
fn test_regex_seq_builtin_no_match_without_space() {
    let src = r#"
        let pat = regex_seq("hello", " ", "world");
        print(str_detect("helloworld", pat));
    "#;
    let out = eval_output(src);
    assert_eq!(out, vec!["false"]);
}

#[test]
fn test_regex_seq_builtin_no_match_parity() {
    assert_parity(r#"
        let pat = regex_seq("hello", " ", "world");
        print(str_detect("helloworld", pat));
    "#);
}

#[test]
fn test_regex_seq_builtin_extract() {
    // Build a date pattern via regex_seq and extract with it
    let src = r#"
        let year_pat = "[[:digit:]]{4}";
        let sep = "-";
        let month_pat = "[[:digit:]]{2}";
        let full_pat = regex_seq(year_pat, sep, month_pat);
        let result = str_extract("date: 2024-07-10 end", full_pat);
        print(result);
    "#;
    let out = eval_output(src);
    assert_eq!(out, vec!["2024-07"]);
}

#[test]
fn test_regex_seq_builtin_extract_parity() {
    assert_parity(r#"
        let year_pat = "[[:digit:]]{4}";
        let sep = "-";
        let month_pat = "[[:digit:]]{2}";
        let full_pat = regex_seq(year_pat, sep, month_pat);
        let result = str_extract("date: 2024-07-10 end", full_pat);
        print(result);
    "#);
}

// ─────────────────────────────────────────────────────────────────────────────
// regex_explain builtin from CJC-Lang source
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_regex_explain_builtin_returns_string() {
    // regex_explain returns a String; we can call str_detect on it to verify
    // it contains "NFA"
    let src = r#"
        let desc = regex_explain("\\d+", "");
        print(str_detect(desc, "NFA"));
    "#;
    let out = eval_output(src);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn test_regex_explain_builtin_returns_string_parity() {
    assert_parity(r#"
        let desc = regex_explain("\\d+", "");
        print(str_detect(desc, "NFA"));
    "#);
}

#[test]
fn test_regex_explain_builtin_with_flags() {
    // regex_explain with two args (pattern + flags)
    let src = r#"
        let desc = regex_explain("[[:alpha:]]+", "i");
        print(str_detect(desc, "NFA"));
    "#;
    let out = eval_output(src);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn test_regex_explain_builtin_with_flags_parity() {
    assert_parity(r#"
        let desc = regex_explain("[[:alpha:]]+", "i");
        print(str_detect(desc, "NFA"));
    "#);
}

#[test]
fn test_regex_explain_builtin_contains_nodes() {
    let src = r#"
        let desc = regex_explain("(?:hello){2,4}", "");
        print(str_detect(desc, "Nodes"));
    "#;
    let out = eval_output(src);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn test_regex_explain_builtin_contains_nodes_parity() {
    assert_parity(r#"
        let desc = regex_explain("(?:hello){2,4}", "");
        print(str_detect(desc, "Nodes"));
    "#);
}
