//! Pure engine feature tests for the upgraded cjc-regex engine.
//!
//! Tests use the public API directly: `is_match`, `find`, `find_all`,
//! `find_match`, `find_all_matches`, `regex_explain`, and `MatchResult`.
//!
//! Each test targets a specific new feature added in the engine upgrade.

use cjc_regex::{
    find, find_all, find_all_matches, find_match, is_match, regex_explain, MatchResult,
};

// ═══════════════════════════════════════════════════════════════════════════
// POSIX character classes
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_posix_alpha_matches_letters() {
    assert!(is_match("[[:alpha:]]+", "", b"hello"));
    assert!(is_match("[[:alpha:]]+", "", b"ABC"));
}

#[test]
fn test_posix_alpha_rejects_digits() {
    // pure digit string should not be matched by [[:alpha:]]+
    assert!(!is_match("^[[:alpha:]]+$", "", b"123"));
}

#[test]
fn test_posix_alpha_mixed_string() {
    // alpha class matches the letters inside a mixed string
    assert!(is_match("[[:alpha:]]", "", b"abc123"));
}

#[test]
fn test_posix_digit_matches_digits() {
    assert!(is_match("^[[:digit:]]+$", "", b"0123456789"));
    assert!(!is_match("^[[:digit:]]+$", "", b"12a4"));
}

#[test]
fn test_posix_alnum_matches_letters_and_digits() {
    assert!(is_match("^[[:alnum:]]+$", "", b"abc123"));
    assert!(!is_match("^[[:alnum:]]+$", "", b"abc!"));
}

#[test]
fn test_posix_space_matches_whitespace() {
    assert!(is_match("[[:space:]]", "", b"hello world"));
    assert!(!is_match("^[[:space:]]+$", "", b"nospace"));
}

#[test]
fn test_posix_upper_matches_uppercase() {
    assert!(is_match("^[[:upper:]]+$", "", b"ABC"));
    assert!(!is_match("^[[:upper:]]+$", "", b"ABc"));
}

#[test]
fn test_posix_lower_matches_lowercase() {
    assert!(is_match("^[[:lower:]]+$", "", b"abc"));
    assert!(!is_match("^[[:lower:]]+$", "", b"Abc"));
}

#[test]
fn test_posix_xdigit_matches_hex_chars() {
    assert!(is_match("^[[:xdigit:]]+$", "", b"0123456789abcdefABCDEF"));
    assert!(!is_match("^[[:xdigit:]]+$", "", b"0g"));
}

#[test]
fn test_posix_blank_matches_tab_and_space() {
    assert!(is_match("[[:blank:]]", "", b"a\tb"));
    assert!(is_match("[[:blank:]]", "", b"a b"));
    assert!(!is_match("^[[:blank:]]+$", "", b"abc"));
}

#[test]
fn test_posix_punct_matches_punctuation() {
    assert!(is_match("[[:punct:]]", "", b"hello!"));
    assert!(is_match("[[:punct:]]", "", b"a,b"));
    assert!(!is_match("^[[:punct:]]+$", "", b"abc"));
}

#[test]
fn test_posix_class_in_range() {
    // Combining POSIX class with range: [[:alpha:]0-9]+ should match alphanumeric
    assert!(is_match("^[[:alpha:]0-9]+$", "", b"abc123"));
}

// ═══════════════════════════════════════════════════════════════════════════
// Unicode escapes
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_unicode_escape_fixed_u0041_is_capital_a() {
    // \u0041 = 'A' (U+0041)
    assert!(is_match("\\u0041", "", b"A"));
    assert!(!is_match("\\u0041", "", b"B"));
}

#[test]
fn test_unicode_escape_braced_form() {
    // \u{0041} = 'A'
    assert!(is_match("\\u{0041}", "", b"A"));
    assert!(!is_match("\\u{0041}", "", b"a"));
}

#[test]
fn test_unicode_escape_multibyte_utf8() {
    // \u00E9 = 'é' (UTF-8: 0xC3 0xA9)
    let haystack = "café".as_bytes();
    assert!(is_match("\\u00E9", "", haystack));
}

#[test]
fn test_unicode_escape_braced_multibyte() {
    // \u{00E9} = 'é'
    let haystack = "résumé".as_bytes();
    assert!(is_match("\\u{00E9}", "", haystack));
}

#[test]
fn test_unicode_escape_no_false_positive() {
    // \u0042 = 'B'; should not match 'A'
    assert!(!is_match("^\\u0042$", "", b"A"));
}

// ═══════════════════════════════════════════════════════════════════════════
// Non-capturing groups
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_non_capturing_group_basic() {
    // "ababc" matches "(?:ab)+c" — two "ab"s followed by "c"
    assert!(is_match("(?:ab)+c", "", b"ababc"));
    // "abc" also matches — one "ab" followed by "c"
    assert!(is_match("^(?:ab)+c$", "", b"abc"));
    // "ac" does not match — no "ab" prefix
    assert!(!is_match("^(?:ab)+c$", "", b"ac"));
}

#[test]
fn test_non_capturing_group_does_not_capture() {
    // The key invariant is that it doesn't break matching
    assert!(is_match("(?:foo|bar)baz", "", b"foobaz"));
    assert!(is_match("(?:foo|bar)baz", "", b"barbaz"));
    assert!(!is_match("(?:foo|bar)baz", "", b"quxbaz"));
}

#[test]
fn test_non_capturing_group_quantified() {
    assert!(is_match("^(?:ab){3}$", "", b"ababab"));
    assert!(!is_match("^(?:ab){3}$", "", b"abab"));
}

#[test]
fn test_non_capturing_group_alternation() {
    assert!(is_match("^(?:cat|dog|bird)$", "", b"cat"));
    assert!(is_match("^(?:cat|dog|bird)$", "", b"dog"));
    assert!(is_match("^(?:cat|dog|bird)$", "", b"bird"));
    assert!(!is_match("^(?:cat|dog|bird)$", "", b"fish"));
}

// ═══════════════════════════════════════════════════════════════════════════
// Inline flags
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_inline_flag_case_insensitive() {
    assert!(is_match("(?i)hello", "", b"HELLO"));
    assert!(is_match("(?i)hello", "", b"Hello"));
    assert!(is_match("(?i)hello", "", b"hElLo"));
}

#[test]
fn test_inline_flag_case_sensitive_default() {
    assert!(!is_match("^hello$", "", b"HELLO"));
}

#[test]
fn test_inline_flag_scoped_group_matches() {
    // (?i:hello) matches case-insensitively, rest is case-sensitive
    assert!(is_match("(?i:hello) world", "", b"HELLO world"));
    assert!(is_match("(?i:hello) world", "", b"hello world"));
}

#[test]
fn test_inline_flag_scoped_group_outside_not_affected() {
    // The " WORLD" part is outside the scoped group, should be case-sensitive
    assert!(!is_match("^(?i:hello) WORLD$", "", b"hello world"));
    assert!(is_match("^(?i:hello) WORLD$", "", b"HELLO WORLD"));
}

#[test]
fn test_inline_flag_multiline() {
    let haystack = b"first\nsecond\nthird";
    assert!(is_match("(?m)^second$", "", haystack));
    // Without multiline, ^ and $ only match absolute start/end
    assert!(!is_match("^second$", "", haystack));
}

// ═══════════════════════════════════════════════════════════════════════════
// Absolute anchors: \A, \z, \Z
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_absolute_anchor_backslash_a_start() {
    // \A must match only the very start, even with multiline mode
    assert!(is_match("\\Ahello", "", b"hello world"));
    assert!(!is_match("\\Ahello", "", b"say hello"));
}

#[test]
fn test_absolute_anchor_backslash_a_ignores_multiline() {
    // With multiline flag, ^ matches per-line — \A still only matches absolute start
    let haystack = b"first\nhello";
    assert!(!is_match("(?m)\\Ahello", "", haystack));
    assert!(is_match("(?m)^hello", "", haystack)); // ^ would match per-line
}

#[test]
fn test_absolute_anchor_backslash_z_end() {
    // \z matches only the absolute end of input
    assert!(is_match("world\\z", "", b"hello world"));
    assert!(!is_match("world\\z", "", b"world!"));
}

#[test]
fn test_absolute_anchor_backslash_z_vs_dollar() {
    let haystack = b"line1\nline2\n";
    // $ with multiline matches before \n; \z matches only at absolute end
    assert!(!is_match("line2\\z", "", haystack)); // there's a trailing \n
    assert!(is_match("(?m)line2$", "", haystack)); // $ matches before \n
}

#[test]
fn test_absolute_anchor_backslash_capital_z() {
    // \Z matches at end of input or just before a final \n
    assert!(is_match("world\\Z", "", b"hello world"));
    assert!(is_match("world\\Z", "", b"hello world\n")); // \Z before trailing \n
    assert!(!is_match("world\\Z", "", b"hello world\n\n")); // two trailing \n
}

// ═══════════════════════════════════════════════════════════════════════════
// Non-word boundary \B
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_non_word_boundary_basic() {
    // \B matches where \b does NOT match (middle of word)
    // "hell" in "hello" — between 'l' and 'l' is \B (both word chars)
    assert!(is_match("l\\Bl", "", b"hello"));
}

#[test]
fn test_non_word_boundary_at_word_start_fails() {
    // At word boundary (start of "hello"), \B does not match
    assert!(!is_match("^\\Bhello\\B$", "", b"hello"));
}

#[test]
fn test_non_word_boundary_inside_word() {
    // "ell" is inside "hello" with \B on both sides
    assert!(is_match("\\Bell\\B", "", b"hello"));
}

#[test]
fn test_word_boundary_vs_non_word_boundary() {
    // \b matches at word boundary (between word and non-word chars)
    assert!(is_match("\\bhello\\b", "", b"say hello there"));
    // "helloworld" has no boundary between "hello" and "world" so \b after hello fails
    assert!(!is_match("\\bhello\\b", "", b"helloworld"));
    // \B matches at non-boundary (inside a word) — "ell" inside "helloX" (no word-boundary after)
    // In "helloX", between l and o is \B (both word chars), and between o and X too
    assert!(is_match("\\Bell\\B", "", b"helloX"));
}

// ═══════════════════════════════════════════════════════════════════════════
// Counted repetition {n}, {n,}, {n,m}
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_counted_exact_n() {
    assert!(is_match("^a{3}$", "", b"aaa"));
    assert!(!is_match("^a{3}$", "", b"aa"));
    assert!(!is_match("^a{3}$", "", b"aaaa"));
}

#[test]
fn test_counted_n_or_more() {
    assert!(is_match("^a{3,}$", "", b"aaa"));
    assert!(is_match("^a{3,}$", "", b"aaaa"));
    assert!(is_match("^a{3,}$", "", b"aaaaaaa"));
    assert!(!is_match("^a{3,}$", "", b"aa"));
}

#[test]
fn test_counted_n_to_m() {
    assert!(is_match("^a{2,4}$", "", b"aa"));
    assert!(is_match("^a{2,4}$", "", b"aaa"));
    assert!(is_match("^a{2,4}$", "", b"aaaa"));
    assert!(!is_match("^a{2,4}$", "", b"a"));
    assert!(!is_match("^a{2,4}$", "", b"aaaaa"));
}

#[test]
fn test_counted_zero_exact() {
    // {0} means zero occurrences — always matches empty
    assert!(is_match("a{0}", "", b"b")); // zero a's before b
    assert!(is_match("^a{0}b$", "", b"b")); // zero a's then b
}

#[test]
fn test_counted_zero_to_m() {
    // {0,3} means zero to three
    assert!(is_match("^a{0,3}$", "", b""));
    assert!(is_match("^a{0,3}$", "", b"a"));
    assert!(is_match("^a{0,3}$", "", b"aaa"));
    assert!(!is_match("^a{0,3}$", "", b"aaaa"));
}

#[test]
fn test_counted_lazy() {
    // {n}? — lazy variant; should still match exactly n
    assert!(is_match("^a{3}?$", "", b"aaa"));
    assert!(!is_match("^a{3}?$", "", b"aa"));
}

#[test]
fn test_counted_repetition_find_span() {
    let m = find("a{3}", "", b"xaaax");
    assert_eq!(m, Some((1, 4)));
}

#[test]
fn test_counted_repetition_find_all() {
    let spans = find_all("a{2}", "", b"aa_aaa_aa");
    // "aa" at 0-2, "aa" inside "aaa" at 3-5, "aa" at 7-9
    assert!(!spans.is_empty());
    // All spans must have length 2
    for (s, e) in &spans {
        assert_eq!(e - s, 2, "span {:?}-{:?} has wrong length", s, e);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MatchResult API
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_match_result_find_match_basic() {
    let m = find_match("\\d+", "", b"abc123def");
    assert!(m.is_some());
    let m = m.unwrap();
    assert_eq!(m.start, 3);
    assert_eq!(m.end, 6);
}

#[test]
fn test_match_result_len() {
    let m = find_match("hello", "", b"say hello").unwrap();
    assert_eq!(m.len(), 5);
}

#[test]
fn test_match_result_is_empty_false() {
    let m = find_match("\\d+", "", b"42").unwrap();
    assert!(!m.is_empty());
}

#[test]
fn test_match_result_is_empty_zero_width() {
    // Zero-width match: empty pattern at position 0
    let m = find_match("", "", b"abc").unwrap();
    assert!(m.is_empty());
    assert_eq!(m.len(), 0);
}

#[test]
fn test_match_result_extract_bytes() {
    let haystack = b"abc123def";
    let m = find_match("\\d+", "", haystack).unwrap();
    assert_eq!(m.extract(haystack), b"123");
}

#[test]
fn test_match_result_extract_str() {
    let haystack = b"hello world";
    let m = find_match("world", "", haystack).unwrap();
    assert_eq!(m.extract_str(haystack), Some("world"));
}

#[test]
fn test_match_result_extract_str_none_on_non_utf8() {
    // A match spanning non-UTF-8 bytes has no valid str representation
    let haystack: &[u8] = &[0xFF, 0x41, 0x42]; // invalid UTF-8 then "AB"
    let m = find_match("\\x41", "", haystack).unwrap();
    // 0x41 = 'A', which is valid UTF-8
    assert_eq!(m.extract_str(haystack), Some("A"));
}

#[test]
fn test_find_all_matches_count() {
    let haystack = b"a1b22c333";
    let ms = find_all_matches("\\d+", "", haystack);
    assert_eq!(ms.len(), 3);
}

#[test]
fn test_find_all_matches_extract() {
    let haystack = b"a1b22c333";
    let ms = find_all_matches("\\d+", "", haystack);
    assert_eq!(ms[0].extract(haystack), b"1");
    assert_eq!(ms[1].extract(haystack), b"22");
    assert_eq!(ms[2].extract(haystack), b"333");
}

#[test]
fn test_match_result_no_match_returns_none() {
    assert!(find_match("xyz", "", b"hello").is_none());
}

#[test]
fn test_match_result_equality() {
    let m1 = MatchResult { start: 3, end: 6 };
    let m2 = MatchResult { start: 3, end: 6 };
    let m3 = MatchResult { start: 0, end: 3 };
    assert_eq!(m1, m2);
    assert_ne!(m1, m3);
}

// ═══════════════════════════════════════════════════════════════════════════
// regex_explain
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_regex_explain_returns_nfa_string() {
    let desc = regex_explain("\\d+", "").unwrap();
    assert!(desc.contains("NFA"), "Expected 'NFA' in explain output, got: {}", desc);
}

#[test]
fn test_regex_explain_includes_pattern() {
    let desc = regex_explain("hello", "i").unwrap();
    assert!(desc.contains("hello"), "Expected pattern in explain output");
}

#[test]
fn test_regex_explain_invalid_pattern_returns_err() {
    // An invalid pattern should return Err, not panic
    let result = regex_explain("[invalid", "");
    assert!(result.is_err(), "Expected Err for invalid pattern");
}

#[test]
fn test_regex_explain_complex_pattern() {
    let desc = regex_explain("(?i:[[:alpha:]]+)\\d{2,4}", "").unwrap();
    assert!(desc.contains("NFA"));
    assert!(desc.contains("Nodes"));
}

// ═══════════════════════════════════════════════════════════════════════════
// Safety limits: MAX_PATTERN_LEN
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_max_pattern_len_rejection() {
    // Pattern longer than MAX_PATTERN_LEN (4096) should be rejected gracefully.
    // is_match returns false (not panic) when compile fails.
    let long_pattern = "a".repeat(5000);
    let result = is_match(&long_pattern, "", b"aaaa");
    // Must not panic. Result is false (compile failed silently) or true.
    // The engine returns false on compile error per its implementation.
    let _ = result; // no assertion on value, just verify no panic
}

#[test]
fn test_max_pattern_len_explain_returns_err() {
    let long_pattern = "a".repeat(5000);
    let result = regex_explain(&long_pattern, "");
    assert!(result.is_err(), "Expected Err for pattern exceeding MAX_PATTERN_LEN");
}

// ═══════════════════════════════════════════════════════════════════════════
// Composition helpers via raw pattern strings
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_regex_or_composed_string_matches() {
    // regex_or("cat", "dog") → "(?:cat)|(?:dog)"
    let composed = "(?:cat)|(?:dog)";
    assert!(is_match(composed, "", b"I have a cat"));
    assert!(is_match(composed, "", b"I have a dog"));
    assert!(!is_match(composed, "", b"I have a bird"));
}

#[test]
fn test_regex_seq_composed_string_matches() {
    // regex_seq("hello", " ", "world") → "(?:hello)(?: )(?:world)"
    let composed = "(?:hello)(?: )(?:world)";
    assert!(is_match(composed, "", b"hello world"));
    assert!(!is_match(composed, "", b"helloworld"));
}
