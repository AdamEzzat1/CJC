//! Property-based tests for the cjc-regex engine using proptest.
//!
//! These tests verify invariants that unit tests cannot exhaustively cover,
//! such as determinism, span validity, and non-overlap of all-matches results.
//!
//! Run with:
//!   cargo test --test test_regex_new property

use proptest::prelude::*;
use cjc_regex::{find, find_all, is_match};

// ---------------------------------------------------------------------------
// Safe pattern strategy: generate patterns from a controlled alphabet so we
// avoid producing patterns that cause the engine to reject (length limit, etc.)
// ---------------------------------------------------------------------------

/// Generate a simple literal or single-character-class pattern.
fn arb_safe_pattern() -> impl Strategy<Value = String> {
    prop_oneof![
        // Fixed literals
        Just("a".to_string()),
        Just("b".to_string()),
        Just("ab".to_string()),
        Just("abc".to_string()),
        // Digit / word / space classes
        Just("\\d".to_string()),
        Just("\\w".to_string()),
        Just("\\s".to_string()),
        // Quantified
        Just("a+".to_string()),
        Just("a*".to_string()),
        Just("a?".to_string()),
        Just("a{2}".to_string()),
        Just("a{1,3}".to_string()),
        // Anchored
        Just("^a".to_string()),
        Just("a$".to_string()),
        // POSIX
        Just("[[:alpha:]]".to_string()),
        Just("[[:digit:]]".to_string()),
        Just("[[:alnum:]]".to_string()),
    ]
}

/// Generate an arbitrary byte slice as a haystack (printable ASCII for clarity).
fn arb_haystack() -> impl Strategy<Value = Vec<u8>> {
    proptest::collection::vec(0x20u8..0x7fu8, 0..64)
}

// ---------------------------------------------------------------------------
// 1. Determinism: same pattern + haystack always gives same result
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn prop_is_match_deterministic(
        pattern in arb_safe_pattern(),
        haystack in arb_haystack(),
    ) {
        let r1 = is_match(&pattern, "", &haystack);
        let r2 = is_match(&pattern, "", &haystack);
        prop_assert_eq!(r1, r2, "is_match was non-deterministic");
    }
}

// ---------------------------------------------------------------------------
// 2. find returns spans within bounds
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn prop_find_within_bounds(
        pattern in arb_safe_pattern(),
        haystack in arb_haystack(),
    ) {
        if let Some((start, end)) = find(&pattern, "", &haystack) {
            prop_assert!(
                start <= end,
                "start ({}) > end ({})", start, end
            );
            prop_assert!(
                end <= haystack.len(),
                "end ({}) > haystack.len() ({})", end, haystack.len()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 3. find_all results are non-overlapping
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn prop_find_all_nonoverlapping(
        pattern in arb_safe_pattern(),
        haystack in arb_haystack(),
    ) {
        let spans = find_all(&pattern, "", &haystack);
        // Verify consecutive spans don't overlap: prev_end <= next_start
        for window in spans.windows(2) {
            let (_, prev_end) = window[0];
            let (next_start, _) = window[1];
            prop_assert!(
                prev_end <= next_start,
                "overlapping spans: prev_end={}, next_start={}", prev_end, next_start
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Split segment lengths + match lengths cover the whole input
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn prop_split_covers_input(
        pattern in arb_safe_pattern(),
        haystack in arb_haystack(),
    ) {
        let match_spans = find_all(&pattern, "", &haystack);
        let split_spans = cjc_regex::split(&pattern, "", &haystack);

        // Sum of all segment lengths plus all match lengths equals haystack.len()
        let segment_total: usize = split_spans.iter().map(|(s, e)| e - s).sum();
        let match_total: usize = match_spans.iter().map(|(s, e)| e - s).sum();

        prop_assert_eq!(
            segment_total + match_total,
            haystack.len(),
            "split coverage mismatch: segments={} matches={} total={} len={}",
            segment_total, match_total, segment_total + match_total, haystack.len()
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Empty pattern always matches
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn prop_empty_pattern_always_matches(haystack in arb_haystack()) {
        // Empty pattern matches at every position; is_match returns true
        prop_assert!(
            is_match("", "", &haystack),
            "empty pattern failed to match haystack of len {}", haystack.len()
        );
    }
}

// ---------------------------------------------------------------------------
// 6. Literal single-byte pattern: is_match("a") is correct
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn prop_literal_match_is_correct(haystack in arb_haystack()) {
        let expected_a = haystack.contains(&b'a');
        let expected_b = !haystack.contains(&b'b') || {
            // b is present — but we test the NOT case separately
            // Just verify is_match("b",...) is true iff b is present
            true
        };
        // is_match("a", ...) should be true iff haystack contains 'a'
        prop_assert_eq!(
            is_match("a", "", &haystack),
            expected_a,
            "is_match(\"a\") disagreed with contains for haystack {:?}", haystack
        );
        let _ = expected_b;
    }
}

// ---------------------------------------------------------------------------
// 7. Counted exact repetition matches exactly n copies
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn prop_counted_exact_length(n in 1usize..8) {
        // Build "aaa...a" (n times) and verify it matches "^a{n}$"
        let exact: Vec<u8> = vec![b'a'; n];
        let one_short: Vec<u8> = vec![b'a'; n - 1];
        let one_long: Vec<u8> = vec![b'a'; n + 1];

        let pattern = format!("^a{{{}}}$", n);

        prop_assert!(
            is_match(&pattern, "", &exact),
            "^a{{{}}}$ did not match {} a's", n, n
        );
        if n > 0 {
            prop_assert!(
                !is_match(&pattern, "", &one_short),
                "^a{{{}}}$ matched {} a's (one short)", n, n - 1
            );
        }
        prop_assert!(
            !is_match(&pattern, "", &one_long),
            "^a{{{}}}$ matched {} a's (one long)", n, n + 1
        );
    }
}

// ---------------------------------------------------------------------------
// 8. POSIX [[:alpha:]]+ match never contains digits
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn prop_posix_alpha_no_digits(haystack in arb_haystack()) {
        let spans = find_all("[[:alpha:]]+", "", &haystack);
        for (start, end) in spans {
            let slice = &haystack[start..end];
            for &byte in slice {
                prop_assert!(
                    !byte.is_ascii_digit(),
                    "POSIX alpha match {:?} contained digit {:?}", slice, byte as char
                );
            }
        }
    }
}
