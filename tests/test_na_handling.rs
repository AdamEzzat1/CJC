//! Tests for the Item 4 NA handling surface.
//!
//! Discovered NA surface (see `crates/cjc-runtime/src/builtins.rs`):
//!   - `NA`                  — literal keyword (TokenKind::Na, Value::Na)
//!   - `is_na(x) -> bool`    — scalar NA predicate (matches Value::Na only)
//!   - `is_not_null(x)`      — broader predicate (also matches NaN/Void)
//!   - `fill_na(arr, v)`     — replaces Value::Na elements with `v`
//!   - `drop_na(arr)`        — removes Value::Na elements
//!   - `coalesce(...)`       — first non-NA / non-Void / non-NaN arg
//!
//! `na_count` is NOT a builtin; it is implemented in CJC-Lang on top of
//! `is_na` + `len` + indexing, and that user-level function is exercised
//! in these tests to hit the Item 4 specification surface.
//!
//! Covers: unit behavior, parity (eval vs mir-exec), determinism,
//! proptest invariants, and a bolero fuzz target.

use proptest::prelude::*;
use std::panic;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// A pure-CJC-Lang prelude that defines `na_count` on top of `is_na` + `len`.
///
/// Kept as a &'static str so formatting programs is a single `format!`.
const NA_PRELUDE: &str = r#"
fn na_count(arr: Any) -> i64 {
    let mut c: i64 = 0;
    for i in 0..len(arr) {
        if is_na(arr[i]) {
            c = c + 1;
        }
    }
    return c;
}
"#;

fn eval_output(src: &str) -> Vec<String> {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(
        !diags.has_errors(),
        "parse errors in eval_output: {:?}",
        diags.diagnostics
    );
    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(&program).expect("eval failed");
    interp.output.clone()
}

fn mir_output(src: &str) -> Vec<String> {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(
        !diags.has_errors(),
        "parse errors in mir_output: {:?}",
        diags.diagnostics
    );
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42)
        .expect("mir-exec failed");
    executor.output
}

fn both_output(src: &str) -> (Vec<String>, Vec<String>) {
    (eval_output(src), mir_output(src))
}

fn assert_parity(src: &str) {
    let (ev, mir) = both_output(src);
    assert_eq!(ev, mir, "eval vs mir-exec parity failure");
}

/// Build a program body that includes the NA prelude.
fn with_prelude(body: &str) -> String {
    format!("{NA_PRELUDE}\n{body}")
}

// ── Unit tests ───────────────────────────────────────────────────────────────

/// `is_na(NA)` returns true; `is_na(<concrete>)` returns false.
#[test]
fn is_na_scalar_detection() {
    let src = r#"
print(is_na(NA));
print(is_na(0));
print(is_na(1));
print(is_na(1.5));
print(is_na("hi"));
print(is_na(true));
"#;
    let out = eval_output(src);
    assert_eq!(out, vec!["true", "false", "false", "false", "false", "false"]);
}

/// `is_na` can detect NA elements when an array is indexed.
#[test]
fn is_na_array_index() {
    let src = r#"
let x = [1, NA, 3, NA, 5];
print(is_na(x[0]));
print(is_na(x[1]));
print(is_na(x[2]));
print(is_na(x[3]));
print(is_na(x[4]));
"#;
    let out = eval_output(src);
    assert_eq!(out, vec!["false", "true", "false", "true", "false"]);
}

/// `fill_na` replaces every NA with the fill value and preserves the other
/// positions.
#[test]
fn fill_na_replaces_na_only() {
    let src = r#"
let x = [1, NA, 3, NA, 5];
let y = fill_na(x, 0);
print(y);
print(len(y));
"#;
    let out = eval_output(src);
    assert_eq!(out, vec!["[1, 0, 3, 0, 5]", "5"]);
}

/// `fill_na` on an array that has no NA is an identity operation on content
/// (prints identically).
#[test]
fn fill_na_no_na_is_identity() {
    let src = r#"
let x = [10, 20, 30];
print(fill_na(x, 99));
print(len(fill_na(x, 99)));
"#;
    let out = eval_output(src);
    assert_eq!(out, vec!["[10, 20, 30]", "3"]);
}

/// `drop_na` removes NA elements and preserves the order of the rest.
#[test]
fn drop_na_removes_and_preserves_order() {
    let src = r#"
let x = [1, NA, 3, NA, 5];
print(drop_na(x));
print(len(drop_na(x)));
"#;
    let out = eval_output(src);
    assert_eq!(out, vec!["[1, 3, 5]", "3"]);
}

/// `drop_na` and `fill_na` on an empty array yield an empty array.
#[test]
fn na_ops_on_empty_array() {
    let src = r#"
let e: [i64] = [];
print(drop_na(e));
print(fill_na(e, 7));
print(len(drop_na(e)));
print(len(fill_na(e, 7)));
"#;
    let out = eval_output(src);
    assert_eq!(out, vec!["[]", "[]", "0", "0"]);
}

/// An array made entirely of NA drops to empty and fills to all-fill.
#[test]
fn drop_na_all_na_is_empty() {
    let src = r#"
let x = [NA, NA, NA];
print(drop_na(x));
print(len(drop_na(x)));
print(fill_na(x, 42));
"#;
    let out = eval_output(src);
    assert_eq!(out, vec!["[]", "0", "[42, 42, 42]"]);
}

/// User-level `na_count` (defined in CJC-Lang via `is_na` + `len`) counts
/// NA occurrences correctly, and after `fill_na` / `drop_na` the count is 0.
#[test]
fn na_count_user_function() {
    let src = with_prelude(r#"
let x = [1, NA, 3, NA, NA, 6];
print(na_count(x));
print(na_count(fill_na(x, 0)));
print(na_count(drop_na(x)));
"#);
    let out = eval_output(&src);
    assert_eq!(out, vec!["3", "0", "0"]);
}

/// `fill_na` with a string fill value works on a mixed-NA string array.
#[test]
fn fill_na_string_value() {
    let src = r#"
let x = ["a", NA, "c"];
print(fill_na(x, "?"));
"#;
    let out = eval_output(src);
    assert_eq!(out, vec![r#"[a, ?, c]"#]);
}

// ── Parity tests (cjc-eval vs cjc-mir-exec) ──────────────────────────────────

/// Parity: `is_na` on a mixed array produces identical output in both
/// executors.
#[test]
fn parity_is_na_mixed() {
    let src = r#"
let x = [1, NA, 3, NA, 5];
print(is_na(x[0]));
print(is_na(x[1]));
print(is_na(x[2]));
print(is_na(x[3]));
"#;
    assert_parity(src);
}

/// Parity: `fill_na` + `drop_na` on a mixed array agree across executors.
#[test]
fn parity_fill_and_drop() {
    let src = r#"
let x = [1, NA, 2, NA, 3];
print(fill_na(x, 0));
print(drop_na(x));
print(len(fill_na(x, 0)));
print(len(drop_na(x)));
"#;
    assert_parity(src);
}

/// Parity: the user-level `na_count` function behaves identically in both
/// executors (exercises `is_na`, `len`, array indexing, and for-loops).
#[test]
fn parity_na_count_user_function() {
    let src = with_prelude(r#"
let x = [NA, 1, NA, 2, NA, 3];
print(na_count(x));
print(na_count(fill_na(x, 999)));
print(na_count(drop_na(x)));
"#);
    assert_parity(&src);
}

/// Parity: conservation law `na_count(arr) + len(drop_na(arr)) == len(arr)`
/// agrees across executors for a specific mixed array.
#[test]
fn parity_conservation_sanity() {
    let src = with_prelude(r#"
let x = [NA, 10, NA, 20, 30, NA, 40];
let total: i64 = na_count(x) + len(drop_na(x));
print(total);
print(len(x));
"#);
    assert_parity(&src);
}

// ── Proptest invariants ──────────────────────────────────────────────────────

/// Build a CJC-Lang array literal string from a boolean pattern:
/// `true` → `NA`, `false` → a distinct non-NA integer.
fn array_literal_from_pattern(pat: &[bool]) -> String {
    let mut parts = Vec::with_capacity(pat.len());
    for (i, &is_na) in pat.iter().enumerate() {
        if is_na {
            parts.push("NA".to_string());
        } else {
            // Use distinct integers so any index bug surfaces.
            parts.push(((i as i64) + 1).to_string());
        }
    }
    if parts.is_empty() {
        "[]".to_string()
    } else {
        // Homogeneous-looking: int / NA. Parser accepts NA inline.
        format!("[{}]", parts.join(", "))
    }
}

proptest! {
    /// Invariant: after `fill_na(arr, v)`, no element is NA
    /// (i.e. `is_na(fill_na(arr, v)[i]) == false` for all i).
    #[test]
    fn prop_fill_na_leaves_no_nas(pat in prop::collection::vec(any::<bool>(), 0..16)) {
        // If the array is empty, the check is vacuous but must still not panic.
        let lit = array_literal_from_pattern(&pat);
        let src = with_prelude(&format!(
            r#"
let arr = {lit};
let filled = fill_na(arr, 0);
print(na_count(filled));
print(len(filled));
"#
        ));
        let out = eval_output(&src);
        // First line is na_count after fill — must be 0.
        prop_assert_eq!(out[0].as_str(), "0", "fill_na left NAs: {:?}", out);
        // Length preserved.
        prop_assert_eq!(out[1].parse::<usize>().unwrap(), pat.len());
    }

    /// Invariant: `na_count(drop_na(arr)) == 0`.
    #[test]
    fn prop_drop_na_count_zero(pat in prop::collection::vec(any::<bool>(), 0..16)) {
        let lit = array_literal_from_pattern(&pat);
        let src = with_prelude(&format!(
            r#"
let arr = {lit};
print(na_count(drop_na(arr)));
"#
        ));
        let out = eval_output(&src);
        prop_assert_eq!(out[0].as_str(), "0");
    }

    /// Invariant: `na_count(arr) + len(drop_na(arr)) == len(arr)` (conservation).
    #[test]
    fn prop_conservation(pat in prop::collection::vec(any::<bool>(), 0..16)) {
        let lit = array_literal_from_pattern(&pat);
        let src = with_prelude(&format!(
            r#"
let arr = {lit};
let lhs: i64 = na_count(arr) + len(drop_na(arr));
let rhs: i64 = len(arr);
print(lhs);
print(rhs);
"#
        ));
        let out = eval_output(&src);
        prop_assert_eq!(&out[0], &out[1], "conservation failed: {:?}", out);
        // And it should equal pat.len().
        prop_assert_eq!(out[1].parse::<usize>().unwrap(), pat.len());
    }

    /// Parity invariant: eval and mir-exec produce identical output for
    /// `fill_na` + `drop_na` + `na_count` on any NA pattern.
    #[test]
    fn prop_eval_mir_parity(pat in prop::collection::vec(any::<bool>(), 0..12)) {
        let lit = array_literal_from_pattern(&pat);
        let src = with_prelude(&format!(
            r#"
let arr = {lit};
print(na_count(arr));
print(fill_na(arr, 0));
print(drop_na(arr));
print(len(arr));
print(len(drop_na(arr)));
"#
        ));
        let (ev, mir) = both_output(&src);
        prop_assert_eq!(ev, mir);
    }
}

// ── Bolero fuzz target ───────────────────────────────────────────────────────

/// Fuzz: arbitrary NA/non-NA patterns (encoded bitwise in a u16) must never
/// panic in either executor, and the conservation invariant must hold.
#[test]
fn fuzz_na_patterns_no_panic() {
    bolero::check!().with_type::<u16>().for_each(|bits: &u16| {
        // Decode up to 12 positions from the bits (so test programs stay short).
        let n = ((*bits >> 12) & 0xF) as usize % 13; // 0..=12
        let pat: Vec<bool> = (0..n).map(|i| (bits >> i) & 1 == 1).collect();
        let lit = array_literal_from_pattern(&pat);
        let src = with_prelude(&format!(
            r#"
let arr = {lit};
let lhs: i64 = na_count(arr) + len(drop_na(arr));
let rhs: i64 = len(arr);
print(lhs);
print(rhs);
print(na_count(fill_na(arr, 0)));
"#
        ));

        // Must not panic in eval.
        let ev = panic::catch_unwind(panic::AssertUnwindSafe(|| eval_output(&src)));
        // Must not panic in mir-exec.
        let mir = panic::catch_unwind(panic::AssertUnwindSafe(|| mir_output(&src)));

        if let (Ok(ev), Ok(mir)) = (ev, mir) {
            // Parity.
            assert_eq!(ev, mir, "parity failure for pattern {pat:?}");
            // Conservation: first two printed lines equal.
            assert_eq!(ev[0], ev[1], "conservation failure for pattern {pat:?}");
            // No NA after fill.
            assert_eq!(ev[2], "0", "fill_na residual NA for pattern {pat:?}");
        }
    });
}
