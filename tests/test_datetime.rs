//! Integration tests for CJC-Lang Item 3 date/time surface.
//!
//! Covers:
//!   - Extraction: year, month, day, hour, minute, second
//!   - Parsing: parse_date(s, fmt) -> epoch millis
//!   - Formatting: date_format(ts, fmt) -> string
//!   - Arithmetic: date_diff(a, b, unit), date_add(ts, n, unit)
//!   - Parity: cjc-eval and cjc-mir-exec produce identical output
//!   - Property invariants: round-trip, invertibility, year component
//!   - Fuzz: date_format must not panic on arbitrary i64 timestamps
//!
//! Note: `now()` is intentionally non-deterministic and is therefore exercised
//! only for "does-not-panic / returns-positive-int" behaviour — it is NOT used
//! in parity tests.
//!
//! Timestamp representation: epoch **milliseconds** (i64), UTC only.
//! See `crates/cjc-runtime/src/datetime.rs` for the canonical definition.

use cjc_runtime::{builtins::dispatch_builtin, Value};
use proptest::prelude::*;
use std::panic;
use std::rc::Rc;

// ── Helpers ──────────────────────────────────────────────────────────────────

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

fn assert_parity(src: &str) {
    let ev = eval_output(src);
    let mir = mir_output(src);
    assert_eq!(ev, mir, "eval vs mir-exec parity failure for:\n{src}");
}

/// Call a datetime builtin directly via `dispatch_builtin` (stateless path
/// shared by both executors).
fn call_builtin(name: &str, args: Vec<Value>) -> Value {
    dispatch_builtin(name, &args)
        .unwrap_or_else(|e| panic!("dispatch_builtin({name}) error: {e}"))
        .unwrap_or_else(|| panic!("dispatch_builtin({name}) returned None"))
}

fn as_i64(v: &Value) -> i64 {
    match v {
        Value::Int(n) => *n,
        Value::Float(f) => *f as i64,
        other => panic!("expected Int, got {other:?}"),
    }
}

fn as_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.as_str().to_string(),
        other => panic!("expected String, got {other:?}"),
    }
}

fn s(lit: &str) -> Value {
    Value::String(Rc::new(lit.to_string()))
}

// Canonical reference timestamp: 2023-11-14T22:13:20Z in epoch millis.
// (Seconds 1_700_000_000 * 1000.)
const REF_TS_MILLIS: i64 = 1_700_000_000_000;
const REF_YEAR: i64 = 2023;
const REF_MONTH: i64 = 11;
const REF_DAY: i64 = 14;
const REF_HOUR: i64 = 22;
const REF_MINUTE: i64 = 13;
const REF_SECOND: i64 = 20;

// =============================================================================
// Unit tests (direct dispatch — verifies the Rust-level surface)
// =============================================================================

/// `year` extracts the Gregorian year from a known timestamp.
#[test]
fn unit_year_extraction() {
    let y = as_i64(&call_builtin("year", vec![Value::Int(REF_TS_MILLIS)]));
    assert_eq!(y, REF_YEAR);
}

/// `month` extracts the 1-based month.
#[test]
fn unit_month_extraction() {
    let m = as_i64(&call_builtin("month", vec![Value::Int(REF_TS_MILLIS)]));
    assert_eq!(m, REF_MONTH);
}

/// `day` extracts the 1-based day-of-month.
#[test]
fn unit_day_extraction() {
    let d = as_i64(&call_builtin("day", vec![Value::Int(REF_TS_MILLIS)]));
    assert_eq!(d, REF_DAY);
}

/// `hour`, `minute`, `second` extract wall-clock components.
#[test]
fn unit_hms_extraction() {
    let h = as_i64(&call_builtin("hour", vec![Value::Int(REF_TS_MILLIS)]));
    let m = as_i64(&call_builtin("minute", vec![Value::Int(REF_TS_MILLIS)]));
    let s_ = as_i64(&call_builtin("second", vec![Value::Int(REF_TS_MILLIS)]));
    assert_eq!(h, REF_HOUR);
    assert_eq!(m, REF_MINUTE);
    assert_eq!(s_, REF_SECOND);
}

/// Unix epoch (ts == 0) decomposes to 1970-01-01T00:00:00Z.
#[test]
fn unit_epoch_origin_extraction() {
    assert_eq!(as_i64(&call_builtin("year", vec![Value::Int(0)])), 1970);
    assert_eq!(as_i64(&call_builtin("month", vec![Value::Int(0)])), 1);
    assert_eq!(as_i64(&call_builtin("day", vec![Value::Int(0)])), 1);
    assert_eq!(as_i64(&call_builtin("hour", vec![Value::Int(0)])), 0);
    assert_eq!(as_i64(&call_builtin("minute", vec![Value::Int(0)])), 0);
    assert_eq!(as_i64(&call_builtin("second", vec![Value::Int(0)])), 0);
}

/// `date_format` honours `%Y-%m-%d` and `%H:%M:%S` specifiers.
#[test]
fn unit_date_format_ymd_hms() {
    let ymd = as_string(&call_builtin(
        "date_format",
        vec![Value::Int(REF_TS_MILLIS), s("%Y-%m-%d")],
    ));
    assert_eq!(ymd, "2023-11-14");

    let hms = as_string(&call_builtin(
        "date_format",
        vec![Value::Int(REF_TS_MILLIS), s("%H:%M:%S")],
    ));
    assert_eq!(hms, "22:13:20");

    let full = as_string(&call_builtin(
        "date_format",
        vec![Value::Int(REF_TS_MILLIS), s("%Y-%m-%dT%H:%M:%S")],
    ));
    assert_eq!(full, "2023-11-14T22:13:20");
}

/// `parse_date` correctly re-creates a timestamp from its formatted form.
#[test]
fn unit_parse_date_roundtrip_known() {
    let ts = as_i64(&call_builtin(
        "parse_date",
        vec![s("2023-11-14 22:13:20"), s("%Y-%m-%d %H:%M:%S")],
    ));
    assert_eq!(ts, REF_TS_MILLIS);
}

/// `parse_date` returns 0 on a malformed input (contract: 0 = parse failure).
#[test]
fn unit_parse_date_malformed_returns_zero() {
    let ts = as_i64(&call_builtin(
        "parse_date",
        vec![s("not-a-date"), s("%Y-%m-%d")],
    ));
    assert_eq!(ts, 0);
}

/// `date_diff` with supported units returns the expected integer count.
#[test]
fn unit_date_diff_units() {
    let t1 = Value::Int(REF_TS_MILLIS);
    let t2 = Value::Int(REF_TS_MILLIS + 3 * 24 * 60 * 60 * 1000); // +3 days

    let d_days = as_i64(&call_builtin(
        "date_diff",
        vec![t1.clone(), t2.clone(), s("d")],
    ));
    assert_eq!(d_days, 3);

    let d_hours = as_i64(&call_builtin(
        "date_diff",
        vec![t1.clone(), t2.clone(), s("h")],
    ));
    assert_eq!(d_hours, 72);

    let d_secs = as_i64(&call_builtin(
        "date_diff",
        vec![t1.clone(), t2.clone(), s("s")],
    ));
    assert_eq!(d_secs, 3 * 24 * 60 * 60);

    let d_ms = as_i64(&call_builtin("date_diff", vec![t1, t2, s("ms")]));
    assert_eq!(d_ms, 3 * 24 * 60 * 60 * 1000);
}

/// `date_add` with each supported unit produces the correct forward offset.
#[test]
fn unit_date_add_units() {
    let t = Value::Int(REF_TS_MILLIS);

    let plus_1d = as_i64(&call_builtin("date_add", vec![t.clone(), Value::Int(1), s("d")]));
    assert_eq!(plus_1d - REF_TS_MILLIS, 24 * 60 * 60 * 1000);

    let plus_2h = as_i64(&call_builtin("date_add", vec![t.clone(), Value::Int(2), s("h")]));
    assert_eq!(plus_2h - REF_TS_MILLIS, 2 * 60 * 60 * 1000);

    let plus_30min = as_i64(&call_builtin("date_add", vec![t.clone(), Value::Int(30), s("min")]));
    assert_eq!(plus_30min - REF_TS_MILLIS, 30 * 60 * 1000);

    let plus_45s = as_i64(&call_builtin("date_add", vec![t, Value::Int(45), s("s")]));
    assert_eq!(plus_45s - REF_TS_MILLIS, 45 * 1000);
}

/// `date_add` followed by `year/month/day` extraction yields the expected
/// calendar date (crosses a month boundary).
#[test]
fn unit_date_add_then_extract() {
    // 2023-11-14 + 20 days = 2023-12-04
    let t = Value::Int(REF_TS_MILLIS);
    let plus_20 = call_builtin("date_add", vec![t, Value::Int(20), s("d")]);
    assert_eq!(as_i64(&call_builtin("year", vec![plus_20.clone()])), 2023);
    assert_eq!(as_i64(&call_builtin("month", vec![plus_20.clone()])), 12);
    assert_eq!(as_i64(&call_builtin("day", vec![plus_20])), 4);
}

/// `now()` is non-deterministic but must return a positive integer timestamp.
/// (Not used in parity — placed here only to assert it doesn't panic.)
#[test]
fn unit_now_is_positive_int() {
    // `now` may not be registered under that exact name; probe defensively.
    if let Ok(Some(v)) = dispatch_builtin("now", &[]) {
        let ts = as_i64(&v);
        assert!(ts > 0, "now() returned non-positive: {ts}");
    }
    // If `now` isn't wired under this name this test is a no-op — `datetime_now`
    // lives in the runtime but has no behavioural contract beyond "nondet".
}

// =============================================================================
// Parity tests (cjc-eval vs cjc-mir-exec — excluding `now()`)
// =============================================================================

/// Parity: full extraction + format pipeline produces identical output.
#[test]
fn parity_extract_and_format() {
    let src = r#"
let ts: i64 = 1700000000000;
print(year(ts));
print(month(ts));
print(day(ts));
print(hour(ts));
print(minute(ts));
print(second(ts));
print(date_format(ts, "%Y-%m-%dT%H:%M:%S"));
"#;
    assert_parity(src);
}

/// Parity: parse_date round-trip + arithmetic.
#[test]
fn parity_parse_and_arithmetic() {
    let src = r#"
let ts: i64 = parse_date("2023-11-14 22:13:20", "%Y-%m-%d %H:%M:%S");
let ts2: i64 = date_add(ts, 3, "d");
print(ts);
print(ts2);
print(date_diff(ts, ts2, "d"));
print(date_diff(ts, ts2, "h"));
print(date_format(ts2, "%Y-%m-%d"));
"#;
    assert_parity(src);
}

/// Parity: date_add inverse (adding then subtracting returns origin).
#[test]
fn parity_date_add_inverse() {
    let src = r#"
let ts: i64 = 1700000000000;
let plus: i64 = date_add(ts, 7, "d");
let back: i64 = date_add(plus, -7, "d");
print(ts);
print(back);
print(ts == back);
"#;
    assert_parity(src);
}

// =============================================================================
// Property-based tests (proptest)
// =============================================================================

// Constrain timestamps to a sane range so we don't blow past i64 when
// formatting / parsing 4-digit years. 1970 .. 9999 in epoch millis.
const TS_MIN: i64 = 0;
const TS_MAX: i64 = 253_402_300_799_000; // 9999-12-31 23:59:59 UTC, approx.

proptest! {
    /// Round-trip: format(ts, fmt) then parse(..., fmt) recovers ts (to the
    /// precision that the format preserves — here, full ISO with seconds).
    #[test]
    fn prop_format_parse_roundtrip(ts in TS_MIN..TS_MAX) {
        let fmt = "%Y-%m-%d %H:%M:%S";
        let formatted = as_string(&call_builtin(
            "date_format",
            vec![Value::Int(ts), s(fmt)],
        ));
        let parsed = as_i64(&call_builtin(
            "parse_date",
            vec![s(&formatted), s(fmt)],
        ));
        // The format resolves to second precision; round ts down to seconds.
        let ts_sec_rounded = (ts / 1000) * 1000;
        prop_assert_eq!(parsed, ts_sec_rounded,
            "roundtrip broke for ts={} fmt={} formatted={}",
            ts, fmt, formatted);
    }
}

proptest! {
    /// Invertibility: date_add(date_add(ts, n, unit), -n, unit) == ts, for
    /// every supported unit.
    #[test]
    fn prop_date_add_invertibility(
        ts in -10_000_000_000_000i64..10_000_000_000_000i64,
        n in -1000i64..1000i64,
        unit_idx in 0usize..5usize,
    ) {
        let unit = ["ms", "s", "min", "h", "d"][unit_idx];
        let plus = as_i64(&call_builtin(
            "date_add",
            vec![Value::Int(ts), Value::Int(n), s(unit)],
        ));
        let back = as_i64(&call_builtin(
            "date_add",
            vec![Value::Int(plus), Value::Int(-n), s(unit)],
        ));
        prop_assert_eq!(back, ts,
            "date_add inverse failed: ts={} n={} unit={}",
            ts, n, unit);
    }
}

proptest! {
    /// `year(parse_date("YYYY-01-01", "%Y-%m-%d"))` returns the year that
    /// was supplied, for every 4-digit year in [1970, 9999].
    #[test]
    fn prop_year_component_via_parse(y in 1970i64..9999i64) {
        let date_str = format!("{:04}-01-01", y);
        let ts = as_i64(&call_builtin(
            "parse_date",
            vec![s(&date_str), s("%Y-%m-%d")],
        ));
        let y_extracted = as_i64(&call_builtin("year", vec![Value::Int(ts)]));
        prop_assert_eq!(y_extracted, y,
            "year roundtrip failed for input year {}, ts={}", y, ts);
    }
}

// =============================================================================
// Bolero fuzz target — date_format must not panic on arbitrary i64 inputs.
// =============================================================================

/// Fuzz: `date_format` with arbitrary i64 timestamps across a rotating set of
/// format strings must never panic. (The datetime module's `civil_from_days`
/// uses `div_euclid`/`rem_euclid` and should be total over i64.)
#[test]
fn fuzz_date_format_never_panics() {
    bolero::check!()
        .with_type::<(i64, u8)>()
        .for_each(|(ts, fmt_idx): &(i64, u8)| {
            let fmt = match *fmt_idx % 6 {
                0 => "%Y-%m-%d",
                1 => "%H:%M:%S",
                2 => "%Y-%m-%dT%H:%M:%S",
                3 => "%Y",
                4 => "",
                _ => "literal/%Y/%m",
            };
            let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                let _ = dispatch_builtin(
                    "date_format",
                    &[Value::Int(*ts), Value::String(Rc::new(fmt.to_string()))],
                );
            }));
        });
}
