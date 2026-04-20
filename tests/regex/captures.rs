//! Tests for capture groups (3A), named captures (3B), and capture builtins.
//!
//! Covers:
//! - Basic capture group extraction
//! - Multiple groups
//! - Optional/unmatched groups
//! - Nested groups
//! - Named captures (?P<name>...) and (?<name>...)
//! - Alternation with groups
//! - find_all_captures
//! - Builtins: regex_captures, regex_named_capture, regex_capture_count
//! - Parity tests (AST eval == MIR exec)
//! - Property tests (proptest)
//! - Fuzz target (bolero)

use cjc_regex::{find_captures, find_all_captures, capture_count, Capture};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// 3A: Basic capture groups
// ---------------------------------------------------------------------------

#[test]
fn capture_single_digit_group() {
    let cr = find_captures(r"(\d+)", "", b"abc123def").unwrap();
    assert_eq!(cr.full.start, 3);
    assert_eq!(cr.full.end, 6);
    let g1 = cr.get(1).unwrap();
    assert_eq!(g1.start, 3);
    assert_eq!(g1.end, 6);
    assert_eq!(g1.extract(b"abc123def"), b"123");
}

#[test]
fn capture_two_word_groups() {
    let cr = find_captures(r"(\w+)\s+(\w+)", "", b"hello world").unwrap();
    assert_eq!(cr.full.start, 0);
    assert_eq!(cr.full.end, 11);
    assert_eq!(cr.get(1).unwrap().extract_str(b"hello world"), Some("hello"));
    assert_eq!(cr.get(2).unwrap().extract_str(b"hello world"), Some("world"));
}

#[test]
fn capture_optional_group_present() {
    let cr = find_captures(r"(\d+)?abc", "", b"42abc").unwrap();
    assert_eq!(cr.get(1).unwrap().extract(b"42abc"), b"42");
}

#[test]
fn capture_optional_group_absent() {
    let cr = find_captures(r"(\d+)?abc", "", b"abc").unwrap();
    // Group 1 should be None when the optional group didn't match
    assert!(cr.get(1).is_none());
}

#[test]
fn capture_nested_groups() {
    let hay = b"abcdef";
    let cr = find_captures(r"(a(b)c)", "", hay).unwrap();
    assert_eq!(cr.get(1).unwrap().extract(hay), b"abc");
    assert_eq!(cr.get(2).unwrap().extract(hay), b"b");
}

#[test]
fn capture_alternation_left() {
    let cr = find_captures(r"(cat)|(dog)", "", b"I have a cat").unwrap();
    assert!(cr.get(1).is_some());
    assert_eq!(cr.get(1).unwrap().extract(b"I have a cat"), b"cat");
    assert!(cr.get(2).is_none());
}

#[test]
fn capture_alternation_right() {
    let cr = find_captures(r"(cat)|(dog)", "", b"I have a dog").unwrap();
    assert!(cr.get(1).is_none());
    assert!(cr.get(2).is_some());
    assert_eq!(cr.get(2).unwrap().extract(b"I have a dog"), b"dog");
}

#[test]
fn capture_no_match() {
    assert!(find_captures(r"(\d+)", "", b"no digits").is_none());
}

#[test]
fn capture_group_0_is_full_match() {
    let cr = find_captures(r"(\d+)-(\d+)", "", b"xx12-34yy").unwrap();
    assert_eq!(cr.get(0).unwrap().extract(b"xx12-34yy"), b"12-34");
    assert_eq!(cr.get(1).unwrap().extract(b"xx12-34yy"), b"12");
    assert_eq!(cr.get(2).unwrap().extract(b"xx12-34yy"), b"34");
}

#[test]
fn capture_empty_group() {
    // An empty group () should capture an empty string
    let cr = find_captures(r"a()b", "", b"ab").unwrap();
    let g1 = cr.get(1).unwrap();
    assert_eq!(g1.start, g1.end); // empty capture
}

#[test]
fn capture_repeated_group() {
    // (ab)+ — the group captures the LAST repetition
    let cr = find_captures(r"(ab)+", "", b"ababab").unwrap();
    assert_eq!(cr.full.extract(b"ababab"), b"ababab");
    // Group 1 captures the last iteration of the repetition
    let g1 = cr.get(1).unwrap();
    assert_eq!(g1.extract(b"ababab"), b"ab");
}

#[test]
fn capture_with_anchors() {
    let cr = find_captures(r"^(\w+)$", "", b"hello").unwrap();
    assert_eq!(cr.get(1).unwrap().extract(b"hello"), b"hello");
}

#[test]
fn capture_case_insensitive() {
    let cr = find_captures(r"(\w+)", "i", b"Hello").unwrap();
    assert_eq!(cr.get(1).unwrap().extract(b"Hello"), b"Hello");
}

#[test]
fn capture_within_full_match_bounds() {
    let cr = find_captures(r"(\d+)-(\w+)", "", b"abc123-xyz").unwrap();
    for i in 1..=2 {
        if let Some(g) = cr.get(i) {
            assert!(g.start >= cr.full.start);
            assert!(g.end <= cr.full.end);
        }
    }
}

// ---------------------------------------------------------------------------
// 3A: find_all_captures
// ---------------------------------------------------------------------------

#[test]
fn find_all_captures_basic() {
    let results = find_all_captures(r"(\d+)", "", b"a1b22c333");
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].get(1).unwrap().extract(b"a1b22c333"), b"1");
    assert_eq!(results[1].get(1).unwrap().extract(b"a1b22c333"), b"22");
    assert_eq!(results[2].get(1).unwrap().extract(b"a1b22c333"), b"333");
}

#[test]
fn find_all_captures_multiple_groups() {
    let hay = b"12-34 56-78";
    let results = find_all_captures(r"(\d+)-(\d+)", "", hay);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].get(1).unwrap().extract(hay), b"12");
    assert_eq!(results[0].get(2).unwrap().extract(hay), b"34");
    assert_eq!(results[1].get(1).unwrap().extract(hay), b"56");
    assert_eq!(results[1].get(2).unwrap().extract(hay), b"78");
}

#[test]
fn find_all_captures_no_match() {
    let results = find_all_captures(r"(\d+)", "", b"no digits");
    assert!(results.is_empty());
}

// ---------------------------------------------------------------------------
// 3B: Named captures
// ---------------------------------------------------------------------------

#[test]
fn named_capture_python_syntax() {
    let cr = find_captures(r"(?P<year>\d{4})-(?P<month>\d{2})", "", b"2026-04").unwrap();
    assert_eq!(cr.get_named("year").unwrap().extract(b"2026-04"), b"2026");
    assert_eq!(cr.get_named("month").unwrap().extract(b"2026-04"), b"04");
}

#[test]
fn named_capture_angle_bracket_syntax() {
    let cr = find_captures(r"(?<year>\d{4})-(?<month>\d{2})", "", b"2026-04").unwrap();
    assert_eq!(cr.get_named("year").unwrap().extract(b"2026-04"), b"2026");
    assert_eq!(cr.get_named("month").unwrap().extract(b"2026-04"), b"04");
}

#[test]
fn named_capture_also_indexed() {
    let cr = find_captures(r"(?P<name>\w+)", "", b"hello").unwrap();
    // Named capture should also be accessible by index
    assert_eq!(cr.get(1).unwrap().extract(b"hello"), b"hello");
    assert_eq!(cr.get_named("name").unwrap().extract(b"hello"), b"hello");
}

#[test]
fn named_capture_nonexistent_name() {
    let cr = find_captures(r"(?P<name>\w+)", "", b"hello").unwrap();
    assert!(cr.get_named("nonexistent").is_none());
}

#[test]
fn named_capture_mixed_with_numbered() {
    // Mix named and unnamed groups
    let cr = find_captures(r"(\d+)-(?P<word>\w+)", "", b"42-hello").unwrap();
    assert_eq!(cr.get(1).unwrap().extract(b"42-hello"), b"42");
    assert_eq!(cr.get_named("word").unwrap().extract(b"42-hello"), b"hello");
    assert_eq!(cr.get(2).unwrap().extract(b"42-hello"), b"hello");
}

#[test]
fn named_capture_in_find_all() {
    let results = find_all_captures(r"(?P<num>\d+)", "", b"a1b22");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].get_named("num").unwrap().extract(b"a1b22"), b"1");
    assert_eq!(results[1].get_named("num").unwrap().extract(b"a1b22"), b"22");
}

// ---------------------------------------------------------------------------
// capture_count
// ---------------------------------------------------------------------------

#[test]
fn capture_count_no_groups() {
    assert_eq!(capture_count(r"\d+", ""), 0);
}

#[test]
fn capture_count_one_group() {
    assert_eq!(capture_count(r"(\d+)", ""), 1);
}

#[test]
fn capture_count_multiple_groups() {
    assert_eq!(capture_count(r"(\d+)-(\w+)", ""), 2);
}

#[test]
fn capture_count_non_capturing_not_counted() {
    assert_eq!(capture_count(r"(?:\d+)-(\w+)", ""), 1);
}

#[test]
fn capture_count_named_groups() {
    assert_eq!(capture_count(r"(?P<a>\d+)-(?<b>\w+)", ""), 2);
}

#[test]
fn capture_count_nested() {
    assert_eq!(capture_count(r"(a(b)c)", ""), 2);
}

#[test]
fn capture_count_invalid_pattern() {
    assert_eq!(capture_count(r"(unclosed", ""), 0);
}

// ---------------------------------------------------------------------------
// Builtins: regex_captures, regex_named_capture, regex_capture_count
// ---------------------------------------------------------------------------

fn run_cjc_eval(src: &str) -> Vec<String> {
    let (program, _diag) = cjc_parser::parse_source(src);
    let mut interp = cjc_eval::Interpreter::new(42);
    let _ = interp.exec(&program);
    interp.output
}

fn run_cjc_mir(src: &str) -> Vec<String> {
    let (program, _diag) = cjc_parser::parse_source(src);
    let (_, exec) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    exec.output
}

#[test]
fn builtin_regex_captures_basic() {
    let src = r#"
let result: Any = regex_captures("(\\d+)-(\\w+)", "", "42-hello");
print(result);
"#;
    let out = run_cjc_eval(src);
    let joined = out.join(" ");
    // Result should be an array: ["42-hello", "42", "hello"]
    assert!(joined.contains("42-hello"), "output: {}", joined);
    assert!(joined.contains("hello"), "output: {}", joined);
}

#[test]
fn builtin_regex_captures_no_match() {
    let src = r#"
let result: Any = regex_captures("(\\d+)", "", "no digits");
print(array_len(result));
"#;
    let out = run_cjc_eval(src);
    let joined = out.join(" ");
    assert!(joined.contains("0"), "output: {}", joined);
}

#[test]
fn builtin_regex_named_capture_basic() {
    let src = r#"
let year: Any = regex_named_capture("(?P<year>\\d{4})-(?P<month>\\d{2})", "", "2026-04", "year");
print(year);
"#;
    let out = run_cjc_eval(src);
    let joined = out.join(" ");
    assert!(joined.contains("2026"), "output: {}", joined);
}

#[test]
fn builtin_regex_named_capture_missing() {
    let src = r#"
let x: Any = regex_named_capture("(?P<year>\\d{4})", "", "2026", "month");
print(x);
"#;
    let out = run_cjc_eval(src);
    // Should print empty string — output might be empty or contain ""
    let joined = out.join("");
    assert!(joined.is_empty() || joined.trim().is_empty(), "expected empty, got: {}", joined);
}

#[test]
fn builtin_regex_capture_count_basic() {
    let src = r#"
let n: Any = regex_capture_count("(\\d+)-(\\w+)", "");
print(n);
"#;
    let out = run_cjc_eval(src);
    let joined = out.join(" ");
    assert!(joined.contains("2"), "output: {}", joined);
}

// ---------------------------------------------------------------------------
// Parity tests: AST eval == MIR exec
// ---------------------------------------------------------------------------

#[test]
fn parity_regex_captures() {
    let src = r#"
let result: Any = regex_captures("(\\d+)-(\\w+)", "", "42-hello");
print(result);
"#;
    assert_eq!(run_cjc_eval(src), run_cjc_mir(src), "regex_captures parity failed");
}

#[test]
fn parity_regex_named_capture() {
    let src = r#"
let year: Any = regex_named_capture("(?P<year>\\d{4})", "", "2026", "year");
print(year);
"#;
    assert_eq!(run_cjc_eval(src), run_cjc_mir(src), "regex_named_capture parity failed");
}

#[test]
fn parity_regex_capture_count() {
    let src = r#"
let n: Any = regex_capture_count("(a)(b)(c)", "");
print(n);
"#;
    assert_eq!(run_cjc_eval(src), run_cjc_mir(src), "regex_capture_count parity failed");
}

#[test]
fn parity_regex_captures_no_match() {
    let src = r#"
let result: Any = regex_captures("(xyz)", "", "abc");
print(array_len(result));
"#;
    assert_eq!(run_cjc_eval(src), run_cjc_mir(src), "regex_captures no-match parity failed");
}

// ---------------------------------------------------------------------------
// Determinism tests
// ---------------------------------------------------------------------------

#[test]
fn capture_determinism() {
    for _ in 0..10 {
        let cr = find_captures(r"(\d+)-(\w+)", "", b"42-hello").unwrap();
        assert_eq!(cr.get(1).unwrap(), &Capture { start: 0, end: 2 });
        assert_eq!(cr.get(2).unwrap(), &Capture { start: 3, end: 8 });
    }
}

#[test]
fn named_capture_determinism() {
    for _ in 0..10 {
        let cr = find_captures(r"(?P<a>\d+)", "", b"123").unwrap();
        assert_eq!(cr.get_named("a").unwrap(), &Capture { start: 0, end: 3 });
    }
}

// ---------------------------------------------------------------------------
// Property tests (proptest)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod prop {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn groups_within_full_match(hay in "[a-z0-9]{1,30}") {
            let hay_bytes = hay.as_bytes();
            if let Some(cr) = find_captures(r"(\w+)", "", hay_bytes) {
                for i in 0..cr.groups.len() {
                    if let Some(g) = cr.get(i) {
                        prop_assert!(g.start >= cr.full.start,
                            "group {} start {} < full start {}", i, g.start, cr.full.start);
                        prop_assert!(g.end <= cr.full.end,
                            "group {} end {} > full end {}", i, g.end, cr.full.end);
                    }
                }
            }
        }

        #[test]
        fn capture_count_consistent(pat in r"[\(\)a-z\d\\]+{1,20}") {
            // capture_count should not panic on arbitrary patterns
            let _count = capture_count(&pat, "");
        }

        #[test]
        fn find_captures_never_panics(hay in ".{0,50}") {
            // Calling find_captures with a valid pattern should not panic
            let _ = find_captures(r"(\w+)", "", hay.as_bytes());
        }

        #[test]
        fn group_count_matches_capture_count(pat_body in "[a-z0-9.]+{1,10}") {
            let pat = format!(r"({})", pat_body);
            let count = capture_count(&pat, "");
            // A single capturing group should give count = 1
            prop_assert!(count >= 1,
                "expected at least 1 group for pattern `{}`, got {}", pat, count);
        }
    }
}

// ---------------------------------------------------------------------------
// Fuzz target (bolero)
// ---------------------------------------------------------------------------

#[test]
fn fuzz_capture_no_panic() {
    bolero::check!()
        .with_type::<(String, String)>()
        .for_each(|(pattern, haystack)| {
            // Must never panic, regardless of input
            let _ = find_captures(pattern, "", haystack.as_bytes());
            let _ = capture_count(pattern, "");
        });
}
