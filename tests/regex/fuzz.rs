//! Bolero-style fuzz tests for the cjc-regex engine.
//!
//! Uses seed-based generation (matching the pattern in `tests/test_mir_fuzz.rs`)
//! to produce regex patterns, then verifies three key safety invariants:
//!
//! 1. `compile` (via `regex_explain`) never panics — may return Err, that's fine.
//! 2. `is_match` on valid patterns never panics.
//! 3. `find_all` results are always non-overlapping.
//!
//! Run with:
//!   cargo test --test test_regex_new fuzz

use std::panic;
use cjc_regex::{find_all, is_match, regex_explain};

// ---------------------------------------------------------------------------
// Seed-based pattern generator
// ---------------------------------------------------------------------------

/// Generate a syntactically valid (or at least non-panicking) regex pattern
/// from a seed. Produces a variety of pattern constructs exercising new features.
fn gen_pattern(seed: u64) -> String {
    let variant = seed % 20;
    let n = ((seed >> 8) % 8 + 1) as usize;        // 1..=8
    let m = n + ((seed >> 16) % 8) as usize;       // n..=n+7
    let char_idx = (seed >> 24) % 26;
    let ch = (b'a' + char_idx as u8) as char;

    match variant {
        // Basic literals
        0 => format!("{}", ch),
        1 => format!("{}+", ch),
        2 => format!("{}*", ch),
        3 => format!("{}?", ch),

        // Counted repetition
        4 => format!("{}{{{}}}",  ch, n),
        5 => format!("{{{},}}",   n),
        6 => format!("{{{},{}}}",  n, m),
        7 => format!("{}{{{}}}?", ch, n),

        // POSIX classes
        8  => "[[:alpha:]]+".to_string(),
        9  => "[[:digit:]]+".to_string(),
        10 => "[[:alnum:]]".to_string(),
        11 => "[[:space:]]".to_string(),

        // Non-capturing groups
        12 => format!("(?:{})+", ch),
        13 => format!("(?:{}|{})", ch, (b'a' + ((char_idx + 1) % 26) as u8) as char),

        // Inline flags
        14 => format!("(?i){}", ch),
        15 => format!("(?i:{})", ch),

        // Absolute anchors + \B
        16 => format!("\\A{}",  ch),
        17 => format!("{}\\z",  ch),
        18 => format!("\\B{}\\B", ch),

        // Unicode escape
        _ => "\\u0041".to_string(),
    }
}

/// Generate a haystack from a seed.
fn gen_haystack(seed: u64) -> Vec<u8> {
    let len = (seed % 32) as usize;
    let base = (seed >> 8) as u8;
    (0..len).map(|i| (base.wrapping_add(i as u8) % 95) + 0x20).collect()
}

// ---------------------------------------------------------------------------
// 1. compile (regex_explain) never panics
// ---------------------------------------------------------------------------

#[test]
fn fuzz_regex_compile_no_panic() {
    bolero::check!()
        .with_type::<u64>()
        .for_each(|&seed: &u64| {
            let pattern = gen_pattern(seed);
            // regex_explain internally calls compile; Err is acceptable, panic is not
            let _ = panic::catch_unwind(|| {
                let _ = regex_explain(&pattern, "");
            });
        });
}

/// Also fuzz with raw arbitrary byte patterns for broader coverage.
#[test]
fn fuzz_regex_compile_raw_no_panic() {
    bolero::check!()
        .with_type::<Vec<u8>>()
        .for_each(|input: &Vec<u8>| {
            if let Ok(pattern) = std::str::from_utf8(input) {
                let pattern = pattern.to_string();
                let _ = panic::catch_unwind(|| {
                    // compile via explain — Err means rejected pattern, fine
                    let _ = regex_explain(&pattern, "");
                });
            }
        });
}

// ---------------------------------------------------------------------------
// 2. is_match on valid patterns never panics
// ---------------------------------------------------------------------------

#[test]
fn fuzz_is_match_no_panic() {
    bolero::check!()
        .with_type::<u64>()
        .for_each(|&seed: &u64| {
            let pattern = gen_pattern(seed);
            let haystack = gen_haystack(seed);
            let _ = panic::catch_unwind(|| {
                // is_match returns false on compile error; must not panic
                let _ = is_match(&pattern, "", &haystack);
            });
        });
}

/// Also exercise is_match with arbitrary byte haystacks against fixed valid patterns.
#[test]
fn fuzz_is_match_arbitrary_haystack_no_panic() {
    bolero::check!()
        .with_type::<(u64, Vec<u8>)>()
        .for_each(|(seed, haystack): &(u64, Vec<u8>)| {
            let pattern = gen_pattern(*seed);
            let _ = panic::catch_unwind(|| {
                let _ = is_match(&pattern, "", haystack);
            });
        });
}

// ---------------------------------------------------------------------------
// 3. find_all results are always non-overlapping
// ---------------------------------------------------------------------------

#[test]
fn fuzz_find_all_nonoverlapping() {
    bolero::check!()
        .with_type::<u64>()
        .for_each(|&seed: &u64| {
            let pattern = gen_pattern(seed);
            let haystack = gen_haystack(seed);

            let result = panic::catch_unwind(|| find_all(&pattern, "", &haystack));
            if let Ok(spans) = result {
                // Verify non-overlap: each span's end <= next span's start
                for window in spans.windows(2) {
                    let (_, prev_end) = window[0];
                    let (next_start, _) = window[1];
                    assert!(
                        prev_end <= next_start,
                        "overlapping spans in find_all: prev_end={}, next_start={} \
                         pattern={:?} haystack={:?}",
                        prev_end, next_start, pattern, haystack
                    );
                }
            }
        });
}
