//! DateTime support for CJC.
//!
//! Design decisions:
//! - Epoch millis (i64), UTC only — deterministic, no timezone ambiguity
//! - `datetime_now()` is NONDET (uses system clock)
//! - All other operations are pure arithmetic on epoch millis
//! - Leap year handling for year/month/day extraction

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MILLIS_PER_SECOND: i64 = 1_000;
const MILLIS_PER_MINUTE: i64 = 60 * MILLIS_PER_SECOND;
const MILLIS_PER_HOUR: i64 = 60 * MILLIS_PER_MINUTE;
const MILLIS_PER_DAY: i64 = 24 * MILLIS_PER_HOUR;

// Days in each month (non-leap year)
const DAYS_IN_MONTH: [i64; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

// ---------------------------------------------------------------------------
// Core functions
// ---------------------------------------------------------------------------

/// Returns current UTC time as epoch milliseconds.
/// This is NONDET — the only nondeterministic datetime operation.
pub fn datetime_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

/// Create a datetime from epoch milliseconds (identity, but validates type).
pub fn datetime_from_epoch(millis: i64) -> i64 {
    millis
}

/// Create a datetime from year, month, day, hour, minute, second components.
/// All components are 1-based for month/day.
pub fn datetime_from_parts(year: i64, month: i64, day: i64, hour: i64, min: i64, sec: i64) -> i64 {
    let days = days_from_civil(year, month, day);
    days * MILLIS_PER_DAY + hour * MILLIS_PER_HOUR + min * MILLIS_PER_MINUTE + sec * MILLIS_PER_SECOND
}

// ---------------------------------------------------------------------------
// Extraction (pure arithmetic)
// ---------------------------------------------------------------------------

/// Extract the year from epoch millis.
pub fn datetime_year(millis: i64) -> i64 {
    let (y, _, _) = civil_from_days(millis.div_euclid(MILLIS_PER_DAY));
    y
}

/// Extract the month (1-12) from epoch millis.
pub fn datetime_month(millis: i64) -> i64 {
    let (_, m, _) = civil_from_days(millis.div_euclid(MILLIS_PER_DAY));
    m
}

/// Extract the day of month (1-31) from epoch millis.
pub fn datetime_day(millis: i64) -> i64 {
    let (_, _, d) = civil_from_days(millis.div_euclid(MILLIS_PER_DAY));
    d
}

/// Extract the hour (0-23) from epoch millis.
pub fn datetime_hour(millis: i64) -> i64 {
    let day_millis = millis.rem_euclid(MILLIS_PER_DAY);
    day_millis / MILLIS_PER_HOUR
}

/// Extract the minute (0-59) from epoch millis.
pub fn datetime_minute(millis: i64) -> i64 {
    let day_millis = millis.rem_euclid(MILLIS_PER_DAY);
    (day_millis % MILLIS_PER_HOUR) / MILLIS_PER_MINUTE
}

/// Extract the second (0-59) from epoch millis.
pub fn datetime_second(millis: i64) -> i64 {
    let day_millis = millis.rem_euclid(MILLIS_PER_DAY);
    (day_millis % MILLIS_PER_MINUTE) / MILLIS_PER_SECOND
}

// ---------------------------------------------------------------------------
// Arithmetic (pure)
// ---------------------------------------------------------------------------

/// Difference between two datetimes in milliseconds.
pub fn datetime_diff(a: i64, b: i64) -> i64 {
    a - b
}

/// Add milliseconds to a datetime.
pub fn datetime_add_millis(dt: i64, millis: i64) -> i64 {
    dt + millis
}

// ---------------------------------------------------------------------------
// Formatting (pure)
// ---------------------------------------------------------------------------

/// Format a datetime as ISO 8601 UTC string: `YYYY-MM-DDTHH:MM:SSZ`
pub fn datetime_format(millis: i64) -> String {
    let days = millis.div_euclid(MILLIS_PER_DAY);
    let (year, month, day) = civil_from_days(days);
    let day_millis = millis.rem_euclid(MILLIS_PER_DAY);
    let hour = day_millis / MILLIS_PER_HOUR;
    let minute = (day_millis % MILLIS_PER_HOUR) / MILLIS_PER_MINUTE;
    let second = (day_millis % MILLIS_PER_MINUTE) / MILLIS_PER_SECOND;
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hour, minute, second
    )
}

// ---------------------------------------------------------------------------
// Civil date algorithms (adapted from Howard Hinnant's algorithms)
// ---------------------------------------------------------------------------

/// Return `true` if `year` is a leap year under the Gregorian calendar.
fn is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Convert a Gregorian date to days since the Unix epoch (1970-01-01).
///
/// Uses Howard Hinnant's civil date algorithm with a March-based year
/// shift for simplified leap-year handling. Month is 1-based (1 = January),
/// day is 1-based.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    // Shift March to month 1 for easier calculation
    let (y, m) = if month <= 2 {
        (year - 1, month + 9)
    } else {
        (year, month - 3)
    };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400); // year of era [0, 399]
    let doy = (153 * m + 2) / 5 + day - 1; // day of year [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // day of era [0, 146096]
    era * 146097 + doe - 719468 // shift to Unix epoch
}

/// Convert days since the Unix epoch to a `(year, month, day)` triple.
///
/// Inverse of [`days_from_civil`]. Month is 1-based, day is 1-based.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097); // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // year of era [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of year [0, 365]
    let mp = (5 * doy + 2) / 153; // month index [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // day [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Return the number of days in the given month (1-12) of the given year.
///
/// Accounts for leap years in February. Returns `0` for out-of-range months.
pub fn days_in_month(year: i64, month: i64) -> i64 {
    if month == 2 && is_leap_year(year) {
        29
    } else if month >= 1 && month <= 12 {
        DAYS_IN_MONTH[(month - 1) as usize]
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Extended date API: parse_date, date_format, date_diff with units, date_add
// ---------------------------------------------------------------------------

/// Parse a date string according to a format specification.
///
/// Supported format tokens: `%Y` (4-digit year), `%m` (month), `%d` (day),
/// `%H` (hour), `%M` (minute), `%S` (second).
///
/// Returns epoch milliseconds on success, 0 on parse failure.
pub fn parse_date(s: &str, fmt: &str) -> i64 {
    let mut year: i64 = 1970;
    let mut month: i64 = 1;
    let mut day: i64 = 1;
    let mut hour: i64 = 0;
    let mut minute: i64 = 0;
    let mut second: i64 = 0;

    let sbytes = s.as_bytes();
    let fbytes = fmt.as_bytes();
    let mut si = 0usize;
    let mut fi = 0usize;

    while fi < fbytes.len() && si < sbytes.len() {
        if fbytes[fi] == b'%' && fi + 1 < fbytes.len() {
            let spec = fbytes[fi + 1];
            fi += 2;
            match spec {
                b'Y' => {
                    if let Some((val, consumed)) = parse_int_n(sbytes, si, 4) {
                        year = val;
                        si += consumed;
                    } else {
                        return 0;
                    }
                }
                b'm' => {
                    if let Some((val, consumed)) = parse_int_max(sbytes, si, 2) {
                        month = val;
                        si += consumed;
                    } else {
                        return 0;
                    }
                }
                b'd' => {
                    if let Some((val, consumed)) = parse_int_max(sbytes, si, 2) {
                        day = val;
                        si += consumed;
                    } else {
                        return 0;
                    }
                }
                b'H' => {
                    if let Some((val, consumed)) = parse_int_max(sbytes, si, 2) {
                        hour = val;
                        si += consumed;
                    } else {
                        return 0;
                    }
                }
                b'M' => {
                    if let Some((val, consumed)) = parse_int_max(sbytes, si, 2) {
                        minute = val;
                        si += consumed;
                    } else {
                        return 0;
                    }
                }
                b'S' => {
                    if let Some((val, consumed)) = parse_int_max(sbytes, si, 2) {
                        second = val;
                        si += consumed;
                    } else {
                        return 0;
                    }
                }
                _ => {
                    // Unknown specifier — treat as literal %X
                    if si < sbytes.len() && sbytes[si] == spec {
                        si += 1;
                    } else {
                        return 0;
                    }
                }
            }
        } else {
            // Literal character — must match
            if sbytes[si] == fbytes[fi] {
                si += 1;
                fi += 1;
            } else {
                return 0;
            }
        }
    }

    datetime_from_parts(year, month, day, hour, minute, second)
}

/// Format an epoch-millisecond timestamp according to a format specification.
///
/// Supported format tokens: `%Y` (4-digit year), `%m` (2-digit month),
/// `%d` (2-digit day), `%H` (2-digit hour), `%M` (2-digit minute),
/// `%S` (2-digit second).
pub fn date_format_custom(ts: i64, fmt: &str) -> String {
    let days = ts.div_euclid(MILLIS_PER_DAY);
    let (year, month, day) = civil_from_days(days);
    let day_millis = ts.rem_euclid(MILLIS_PER_DAY);
    let hour = day_millis / MILLIS_PER_HOUR;
    let minute = (day_millis % MILLIS_PER_HOUR) / MILLIS_PER_MINUTE;
    let second = (day_millis % MILLIS_PER_MINUTE) / MILLIS_PER_SECOND;

    let fbytes = fmt.as_bytes();
    let mut result = String::new();
    let mut i = 0;
    while i < fbytes.len() {
        if fbytes[i] == b'%' && i + 1 < fbytes.len() {
            match fbytes[i + 1] {
                b'Y' => { result.push_str(&format!("{:04}", year)); i += 2; }
                b'm' => { result.push_str(&format!("{:02}", month)); i += 2; }
                b'd' => { result.push_str(&format!("{:02}", day)); i += 2; }
                b'H' => { result.push_str(&format!("{:02}", hour)); i += 2; }
                b'M' => { result.push_str(&format!("{:02}", minute)); i += 2; }
                b'S' => { result.push_str(&format!("{:02}", second)); i += 2; }
                _ => { result.push('%'); i += 1; }
            }
        } else {
            result.push(fbytes[i] as char);
            i += 1;
        }
    }
    result
}

/// Compute the difference `ts2 - ts1` in the specified unit.
///
/// Units: `"ms"` (milliseconds), `"s"` (seconds), `"min"` (minutes),
/// `"h"` (hours), `"d"` (days).
pub fn date_diff_units(ts1: i64, ts2: i64, unit: &str) -> Result<i64, String> {
    let diff_ms = ts2 - ts1;
    match unit {
        "ms" => Ok(diff_ms),
        "s" => Ok(diff_ms / MILLIS_PER_SECOND),
        "min" => Ok(diff_ms / MILLIS_PER_MINUTE),
        "h" => Ok(diff_ms / MILLIS_PER_HOUR),
        "d" => Ok(diff_ms / MILLIS_PER_DAY),
        _ => Err(format!("date_diff: unknown unit '{}', expected ms|s|min|h|d", unit)),
    }
}

/// Add `amount` of the specified `unit` to a timestamp.
///
/// Units: `"ms"`, `"s"`, `"min"`, `"h"`, `"d"`.
pub fn date_add_units(ts: i64, amount: i64, unit: &str) -> Result<i64, String> {
    let millis = match unit {
        "ms" => amount,
        "s" => amount * MILLIS_PER_SECOND,
        "min" => amount * MILLIS_PER_MINUTE,
        "h" => amount * MILLIS_PER_HOUR,
        "d" => amount * MILLIS_PER_DAY,
        _ => return Err(format!("date_add: unknown unit '{}', expected ms|s|min|h|d", unit)),
    };
    Ok(ts + millis)
}

// Helper: parse exactly `n` digits from `bytes` starting at `pos`.
fn parse_int_n(bytes: &[u8], pos: usize, n: usize) -> Option<(i64, usize)> {
    if pos + n > bytes.len() { return None; }
    let mut val: i64 = 0;
    for i in 0..n {
        let b = bytes[pos + i];
        if !b.is_ascii_digit() { return None; }
        val = val * 10 + (b - b'0') as i64;
    }
    Some((val, n))
}

// Helper: parse up to `max_digits` digits from `bytes` starting at `pos`.
fn parse_int_max(bytes: &[u8], pos: usize, max_digits: usize) -> Option<(i64, usize)> {
    let mut val: i64 = 0;
    let mut count = 0;
    while count < max_digits && pos + count < bytes.len() && bytes[pos + count].is_ascii_digit() {
        val = val * 10 + (bytes[pos + count] - b'0') as i64;
        count += 1;
    }
    if count == 0 { None } else { Some((val, count)) }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epoch_origin() {
        // 1970-01-01 00:00:00 = 0
        let dt = datetime_from_parts(1970, 1, 1, 0, 0, 0);
        assert_eq!(dt, 0);
    }

    #[test]
    fn test_known_date() {
        // 2000-01-01 00:00:00 UTC
        let dt = datetime_from_parts(2000, 1, 1, 0, 0, 0);
        assert_eq!(datetime_year(dt), 2000);
        assert_eq!(datetime_month(dt), 1);
        assert_eq!(datetime_day(dt), 1);
        assert_eq!(datetime_hour(dt), 0);
    }

    #[test]
    fn test_extraction_roundtrip() {
        let dt = datetime_from_parts(2024, 6, 15, 14, 30, 45);
        assert_eq!(datetime_year(dt), 2024);
        assert_eq!(datetime_month(dt), 6);
        assert_eq!(datetime_day(dt), 15);
        assert_eq!(datetime_hour(dt), 14);
        assert_eq!(datetime_minute(dt), 30);
        assert_eq!(datetime_second(dt), 45);
    }

    #[test]
    fn test_leap_year() {
        assert!(is_leap_year(2000));
        assert!(is_leap_year(2024));
        assert!(!is_leap_year(1900));
        assert!(!is_leap_year(2023));
    }

    #[test]
    fn test_days_in_feb_leap() {
        assert_eq!(days_in_month(2024, 2), 29);
        assert_eq!(days_in_month(2023, 2), 28);
    }

    #[test]
    fn test_format_iso8601() {
        let dt = datetime_from_parts(2024, 3, 14, 9, 26, 53);
        let s = datetime_format(dt);
        assert_eq!(s, "2024-03-14T09:26:53Z");
    }

    #[test]
    fn test_diff() {
        let a = datetime_from_parts(2024, 1, 2, 0, 0, 0);
        let b = datetime_from_parts(2024, 1, 1, 0, 0, 0);
        assert_eq!(datetime_diff(a, b), MILLIS_PER_DAY);
    }

    #[test]
    fn test_add_millis() {
        let dt = datetime_from_parts(2024, 1, 1, 0, 0, 0);
        let dt2 = datetime_add_millis(dt, MILLIS_PER_HOUR);
        assert_eq!(datetime_hour(dt2), 1);
    }

    #[test]
    fn test_format_epoch() {
        assert_eq!(datetime_format(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn test_determinism() {
        // Pure operations must produce identical results
        let a = datetime_from_parts(2024, 12, 31, 23, 59, 59);
        let b = datetime_from_parts(2024, 12, 31, 23, 59, 59);
        assert_eq!(a, b);
        assert_eq!(datetime_format(a), datetime_format(b));
    }
}
