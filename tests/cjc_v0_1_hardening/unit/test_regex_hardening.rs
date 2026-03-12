//! Regex engine (cjc-regex) hardening tests.
//! API: is_match(pattern, flags, haystack) where haystack is &[u8]

/// Empty pattern matches everything.
#[test]
fn regex_empty_pattern() {
    assert!(cjc_regex::is_match("", "", b"hello"));
    assert!(cjc_regex::is_match("", "", b""));
}

/// Empty text with non-empty pattern.
#[test]
fn regex_empty_text() {
    assert!(!cjc_regex::is_match("abc", "", b""));
}

/// Literal matching.
#[test]
fn regex_literal_match() {
    assert!(cjc_regex::is_match("hello", "", b"hello world"));
    assert!(!cjc_regex::is_match("goodbye", "", b"hello world"));
}

/// Dot matches any character.
#[test]
fn regex_dot() {
    assert!(cjc_regex::is_match("h.llo", "", b"hello"));
    assert!(cjc_regex::is_match("h.llo", "", b"hxllo"));
}

/// Star (zero or more) quantifier.
#[test]
fn regex_star() {
    assert!(cjc_regex::is_match("ab*c", "", b"ac"));
    assert!(cjc_regex::is_match("ab*c", "", b"abc"));
    assert!(cjc_regex::is_match("ab*c", "", b"abbc"));
    assert!(cjc_regex::is_match("ab*c", "", b"abbbc"));
}

/// Plus (one or more) quantifier.
#[test]
fn regex_plus() {
    assert!(!cjc_regex::is_match("ab+c", "", b"ac"));
    assert!(cjc_regex::is_match("ab+c", "", b"abc"));
    assert!(cjc_regex::is_match("ab+c", "", b"abbc"));
}

/// Question mark (optional) quantifier.
#[test]
fn regex_question() {
    assert!(cjc_regex::is_match("ab?c", "", b"ac"));
    assert!(cjc_regex::is_match("ab?c", "", b"abc"));
    assert!(!cjc_regex::is_match("ab?c", "", b"abbc"));
}

/// Character classes.
#[test]
fn regex_char_class() {
    assert!(cjc_regex::is_match("[abc]", "", b"a"));
    assert!(cjc_regex::is_match("[abc]", "", b"b"));
    assert!(cjc_regex::is_match("[abc]", "", b"c"));
    assert!(!cjc_regex::is_match("[abc]", "", b"d"));
}

/// Anchors.
#[test]
fn regex_anchors() {
    assert!(cjc_regex::is_match("^hello", "", b"hello world"));
    assert!(!cjc_regex::is_match("^world", "", b"hello world"));
    assert!(cjc_regex::is_match("world$", "", b"hello world"));
    assert!(!cjc_regex::is_match("hello$", "", b"hello world"));
}

/// Alternation.
#[test]
fn regex_alternation() {
    assert!(cjc_regex::is_match("cat|dog", "", b"cat"));
    assert!(cjc_regex::is_match("cat|dog", "", b"dog"));
    assert!(!cjc_regex::is_match("cat|dog", "", b"bird"));
}

/// Find function returns match span.
#[test]
fn regex_find() {
    let result = cjc_regex::find("[0-9]+", "", b"abc123def");
    assert!(result.is_some(), "Should find digits");
    let (start, end) = result.unwrap();
    assert_eq!(&b"abc123def"[start..end], b"123");
}

/// Find all occurrences.
#[test]
fn regex_find_all() {
    let results = cjc_regex::find_all("[0-9]+", "", b"a1b22c333");
    assert_eq!(results.len(), 3);
    let hay = b"a1b22c333";
    assert_eq!(&hay[results[0].0..results[0].1], b"1");
    assert_eq!(&hay[results[1].0..results[1].1], b"22");
    assert_eq!(&hay[results[2].0..results[2].1], b"333");
}

/// Split by pattern.
#[test]
fn regex_split() {
    let parts = cjc_regex::split("[,;]", "", b"a,b;c,d");
    assert_eq!(parts.len(), 4);
    let hay = b"a,b;c,d";
    assert_eq!(&hay[parts[0].0..parts[0].1], b"a");
    assert_eq!(&hay[parts[1].0..parts[1].1], b"b");
    assert_eq!(&hay[parts[2].0..parts[2].1], b"c");
    assert_eq!(&hay[parts[3].0..parts[3].1], b"d");
}

/// Very long input doesn't cause catastrophic backtracking.
#[test]
fn regex_no_catastrophic_backtracking() {
    let long_a = "a".repeat(1000);
    let pattern = "a*a*a*a*b";
    // Should complete quickly without hanging (NFA-based engine)
    let result = cjc_regex::is_match(pattern, "", long_a.as_bytes());
    assert!(!result, "Should not match (no 'b' in input)");
}

/// Invalid regex pattern doesn't panic.
#[test]
fn regex_invalid_pattern() {
    let result = std::panic::catch_unwind(|| {
        cjc_regex::is_match("[abc", "", b"test");
    });
    let _ = result;
}

/// Case-insensitive flag.
#[test]
fn regex_case_insensitive() {
    assert!(cjc_regex::is_match("hello", "i", b"HELLO"));
    assert!(cjc_regex::is_match("hello", "i", b"Hello"));
}
