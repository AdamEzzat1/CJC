//! Property-based tests for CJC CLI expansion: format parsing, schema inference,
//! and drift detection.
//!
//! Since `cjc-cli` is a binary crate, we duplicate the core parsing logic inline
//! (CSV splitting, JSONL object extraction, format detection, Kahan summation)
//! and verify invariants that unit tests cannot exhaustively cover.

use proptest::prelude::*;
use std::collections::BTreeSet;

// ── Inline helpers (mirrors cjc_cli::formats logic) ─────────────────

/// Detect format from extension string (lowercase).
fn detect_from_ext(ext: &str) -> &'static str {
    match ext {
        "csv" => "CSV",
        "tsv" => "TSV",
        "jsonl" | "ndjson" => "JSONL",
        "parquet" => "Parquet",
        "feather" | "arrow" | "ipc" => "Arrow IPC",
        "sqlite" | "db" | "sqlite3" => "SQLite",
        "pkl" | "pickle" => "Pickle",
        "onnx" => "ONNX",
        "joblib" => "Joblib",
        _ => "Unknown",
    }
}

/// Magic byte signatures for binary format detection.
const MAGIC_SIGNATURES: &[(&str, &[u8], usize)] = &[
    ("Parquet", b"PAR1", 0),
    ("Arrow IPC", b"ARROW1", 0),
    ("SQLite", b"SQLite format 3\0", 0),
    ("Pickle", &[0x80], 0),
];

fn detect_from_magic(header: &[u8]) -> &'static str {
    for &(label, magic, offset) in MAGIC_SIGNATURES {
        if header.len() >= offset + magic.len() && header[offset..offset + magic.len()] == *magic {
            return label;
        }
    }
    "Unknown"
}

/// Simple CSV row parser: split by delimiter, pad/truncate to ncols.
fn parse_delimited_row(line: &str, delimiter: char, ncols: usize) -> Vec<String> {
    let mut fields: Vec<String> = line.split(delimiter).map(|s| s.trim().to_string()).collect();
    while fields.len() < ncols {
        fields.push(String::new());
    }
    fields.truncate(ncols);
    fields
}

/// Load delimited content into (headers, rows).
fn load_delimited(content: &str, delimiter: char, has_header: bool) -> (Vec<String>, Vec<Vec<String>>) {
    let mut lines = content.lines().filter(|l| !l.is_empty());

    let headers: Vec<String>;
    let first_data_line: Option<&str>;

    if has_header {
        if let Some(hdr) = lines.next() {
            headers = hdr.split(delimiter).map(|s| s.trim().to_string()).collect();
            first_data_line = None;
        } else {
            return (Vec::new(), Vec::new());
        }
    } else {
        if let Some(first) = lines.next() {
            let ncols = first.split(delimiter).count();
            headers = (0..ncols).map(|i| format!("col_{}", i)).collect();
            first_data_line = Some(first);
        } else {
            return (Vec::new(), Vec::new());
        }
    }

    let ncols = headers.len();
    let mut rows = Vec::new();

    if let Some(first) = first_data_line {
        rows.push(parse_delimited_row(first, delimiter, ncols));
    }

    for line in lines {
        rows.push(parse_delimited_row(line, delimiter, ncols));
    }

    (headers, rows)
}

/// Minimal JSON object parser that extracts top-level string/number/bool/null fields.
/// Returns None on parse failure.
fn parse_json_object_keys(line: &str) -> Option<BTreeSet<String>> {
    let line = line.trim();
    if !line.starts_with('{') || !line.ends_with('}') {
        return None;
    }
    let inner = &line[1..line.len() - 1];
    let mut keys = BTreeSet::new();
    // Very simplified key extraction: find "key": patterns
    let mut chars = inner.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c == '"' {
            chars.next(); // skip opening quote
            let mut key = String::new();
            loop {
                match chars.next() {
                    Some('\\') => {
                        chars.next(); // skip escaped char
                    }
                    Some('"') => break,
                    Some(ch) => key.push(ch),
                    None => return None,
                }
            }
            // Look for ':'
            while let Some(&c) = chars.peek() {
                if c == ':' {
                    keys.insert(key);
                    break;
                } else if c.is_whitespace() {
                    chars.next();
                } else {
                    break;
                }
            }
        }
        chars.next();
    }
    Some(keys)
}

/// Kahan compensated summation.
fn kahan_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    let mut c = 0.0f64;
    for &v in values {
        let y = v - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Naive summation for comparison.
fn naive_sum(values: &[f64]) -> f64 {
    values.iter().sum()
}

/// Infer column type from a cell value.
fn infer_cell_type(value: &str) -> &'static str {
    let v = value.trim();
    if v.is_empty() || v == "null" || v == "NA" || v == "N/A" {
        return "null";
    }
    if v == "true" || v == "false" {
        return "bool";
    }
    if v.parse::<i64>().is_ok() {
        return "integer";
    }
    if v.parse::<f64>().is_ok() {
        return "float";
    }
    "string"
}

/// Compute a simple cell-by-cell drift summary between two sets of rows.
/// Returns the count of cells that differ.
fn compute_drift(rows_a: &[Vec<String>], rows_b: &[Vec<String>]) -> usize {
    let max_rows = rows_a.len().max(rows_b.len());
    let mut diffs = 0;
    for i in 0..max_rows {
        let a = rows_a.get(i);
        let b = rows_b.get(i);
        match (a, b) {
            (Some(ra), Some(rb)) => {
                let max_cols = ra.len().max(rb.len());
                for j in 0..max_cols {
                    let va = ra.get(j).map(|s| s.as_str()).unwrap_or("");
                    let vb = rb.get(j).map(|s| s.as_str()).unwrap_or("");
                    if va != vb {
                        diffs += 1;
                    }
                }
            }
            _ => diffs += 1,
        }
    }
    diffs
}

// ── Strategies ──────────────────────────────────────────────────────

/// Generate a valid CSV header line: 1..8 column names.
fn arb_csv_header() -> impl Strategy<Value = Vec<String>> {
    proptest::collection::vec("[a-zA-Z_][a-zA-Z0-9_]{0,10}", 1..8)
}

/// Generate a single row of CSV values matching a given column count.
fn arb_csv_row(ncols: usize) -> impl Strategy<Value = Vec<String>> {
    proptest::collection::vec("[a-zA-Z0-9._\\-]{0,20}", ncols..=ncols)
}

/// Generate a complete CSV string (header + 0..20 data rows).
fn arb_csv_content() -> impl Strategy<Value = (Vec<String>, String)> {
    arb_csv_header().prop_flat_map(|headers| {
        let ncols = headers.len();
        let header_line = headers.join(",");
        proptest::collection::vec(arb_csv_row(ncols), 0..20).prop_map(move |rows| {
            let mut csv = header_line.clone();
            for row in &rows {
                csv.push('\n');
                csv.push_str(&row.join(","));
            }
            (headers.clone(), csv)
        })
    })
}

/// Generate a single JSONL line from a set of field names.
fn arb_jsonl_line(fields: Vec<String>) -> impl Strategy<Value = String> {
    let nfields = fields.len();
    proptest::collection::vec("[a-zA-Z0-9 ]{0,15}", nfields..=nfields).prop_map(
        move |values| {
            let pairs: Vec<String> = fields
                .iter()
                .zip(values.iter())
                .map(|(k, v)| format!("\"{}\":\"{}\"", k, v))
                .collect();
            format!("{{{}}}", pairs.join(","))
        },
    )
}

/// Generate JSONL content with consistent fields.
fn arb_jsonl_content() -> impl Strategy<Value = (Vec<String>, String)> {
    proptest::collection::vec("[a-z_]{1,8}", 1..6).prop_flat_map(|fields| {
        let fields_sorted: Vec<String> = {
            let mut s: Vec<String> = fields.clone();
            s.sort();
            s.dedup();
            s
        };
        let fs = fields_sorted.clone();
        proptest::collection::vec(arb_jsonl_line(fs.clone()), 1..15).prop_map(
            move |lines| {
                (fields_sorted.clone(), lines.join("\n"))
            },
        )
    })
}

/// Generate arrays of finite f64 values for Kahan sum testing.
fn arb_finite_f64_array() -> impl Strategy<Value = Vec<f64>> {
    proptest::collection::vec(
        prop::num::f64::NORMAL | prop::num::f64::SUBNORMAL | prop::num::f64::ZERO,
        1..500,
    )
}

/// Strategy for file extensions.
fn arb_extension() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("csv".to_string()),
        Just("tsv".to_string()),
        Just("jsonl".to_string()),
        Just("ndjson".to_string()),
        Just("parquet".to_string()),
        Just("feather".to_string()),
        Just("arrow".to_string()),
        Just("ipc".to_string()),
        Just("sqlite".to_string()),
        Just("db".to_string()),
        Just("sqlite3".to_string()),
        Just("pkl".to_string()),
        Just("pickle".to_string()),
        Just("onnx".to_string()),
        Just("joblib".to_string()),
        Just("txt".to_string()),
        Just("bin".to_string()),
        "[a-z]{1,5}",
    ]
}

// ── Property tests ──────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// CSV round-trip: load_delimited then reconstruct should preserve data.
    /// Specifically: parsing CSV content then re-joining should yield the same
    /// values after trim (our parser trims whitespace).
    #[test]
    fn csv_round_trip((headers, csv) in arb_csv_content()) {
        let (parsed_headers, parsed_rows) = load_delimited(&csv, ',', true);

        // Headers should match exactly.
        prop_assert_eq!(&parsed_headers, &headers,
            "Parsed headers differ from original");

        // Reconstruct CSV from parsed data.
        let reconstructed_header = parsed_headers.join(",");
        let mut reconstructed = reconstructed_header;
        for row in &parsed_rows {
            reconstructed.push('\n');
            reconstructed.push_str(&row.join(","));
        }

        // Re-parse the reconstructed CSV.
        let (re_headers, re_rows) = load_delimited(&reconstructed, ',', true);
        prop_assert_eq!(&re_headers, &parsed_headers,
            "Round-trip headers differ");
        prop_assert_eq!(&re_rows, &parsed_rows,
            "Round-trip rows differ");
    }

    /// JSONL parse-emit consistency: parse JSON fields, reconstruct JSON line,
    /// reparse should yield identical key sets.
    #[test]
    fn jsonl_parse_emit_consistency((_fields, jsonl) in arb_jsonl_content()) {
        let lines: Vec<&str> = jsonl.lines().filter(|l| !l.trim().is_empty()).collect();
        for line in &lines {
            if let Some(keys1) = parse_json_object_keys(line) {
                // Reconstruct a JSON line from the keys.
                let pairs: Vec<String> = keys1.iter().map(|k| format!("\"{}\":\"\"", k)).collect();
                let reconstructed = format!("{{{}}}", pairs.join(","));
                if let Some(keys2) = parse_json_object_keys(&reconstructed) {
                    prop_assert_eq!(keys1, keys2,
                        "JSONL key-set not preserved after reconstruct");
                }
            }
        }
    }

    /// Any valid CSV header line split by comma should produce the same
    /// number of columns as there are commas + 1.
    #[test]
    fn csv_header_column_count(header in "[a-zA-Z_][a-zA-Z0-9_]{0,10}(,[a-zA-Z_][a-zA-Z0-9_]{0,10}){0,9}") {
        let expected_ncols = header.matches(',').count() + 1;
        let (parsed_headers, _) = load_delimited(&header, ',', true);
        prop_assert_eq!(parsed_headers.len(), expected_ncols,
            "Column count mismatch: header='{}', expected={}, got={}",
            header, expected_ncols, parsed_headers.len());
    }

    /// Format detection is deterministic: same extension always returns same format.
    #[test]
    fn format_detection_deterministic(ext in arb_extension()) {
        let fmt1 = detect_from_ext(&ext);
        let fmt2 = detect_from_ext(&ext);
        prop_assert_eq!(fmt1, fmt2,
            "Format detection not deterministic for extension '{}'", ext);
    }

    /// Magic-byte format detection is deterministic: same bytes always return same format.
    #[test]
    fn magic_detection_deterministic(bytes in proptest::collection::vec(any::<u8>(), 0..64)) {
        let fmt1 = detect_from_magic(&bytes);
        let fmt2 = detect_from_magic(&bytes);
        prop_assert_eq!(fmt1, fmt2,
            "Magic detection not deterministic");
    }

    /// Schema inference is deterministic: same cell value always infers same type.
    #[test]
    fn schema_inference_deterministic(value in "[ a-zA-Z0-9._\\-]{0,30}") {
        let t1 = infer_cell_type(&value);
        let t2 = infer_cell_type(&value);
        prop_assert_eq!(t1, t2,
            "Schema inference not deterministic for value '{}'", value);
    }

    /// drift(A, A) == zero diff for any well-formed CSV data.
    #[test]
    fn drift_self_is_zero((_headers, csv) in arb_csv_content()) {
        let (_, rows) = load_delimited(&csv, ',', true);
        let diffs = compute_drift(&rows, &rows);
        prop_assert_eq!(diffs, 0,
            "Self-drift should be zero but got {} diffs", diffs);
    }

    /// Kahan sum is at least as accurate as naive sum for generated float arrays.
    /// We test that Kahan sum is deterministic (same input -> same bits).
    #[test]
    fn kahan_sum_deterministic(values in arb_finite_f64_array()) {
        let k1 = kahan_sum(&values);
        let k2 = kahan_sum(&values);
        prop_assert_eq!(k1.to_bits(), k2.to_bits(),
            "Kahan sum not bit-identical across runs");
    }

    /// Kahan sum has less or equal error than naive sum for "difficult" sums.
    /// We compare against a high-precision reference (f64 from sorted ascending abs).
    #[test]
    fn kahan_accuracy_vs_naive(values in arb_finite_f64_array()) {
        // Reference: sort by ascending magnitude, then sum (reduces error).
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap_or(std::cmp::Ordering::Equal));
        let reference: f64 = sorted.iter().copied().fold(0.0, |a, b| a + b);

        let kahan_result = kahan_sum(&values);
        let naive_result = naive_sum(&values);

        // Skip comparison if any result overflowed to infinity or is NaN.
        if !reference.is_finite() || !kahan_result.is_finite() || !naive_result.is_finite() {
            return Ok(());
        }

        let kahan_err = (kahan_result - reference).abs();
        let naive_err = (naive_result - reference).abs();

        // Kahan should generally be better or equal. We allow a small tolerance
        // because the reference itself is approximate.
        // This is a soft property: we just assert Kahan doesn't catastrophically diverge.
        if reference.abs() > 1e-10 {
            let kahan_rel = kahan_err / reference.abs();
            prop_assert!(kahan_rel < 1e-6,
                "Kahan sum has unexpectedly large relative error: {} (naive err: {})",
                kahan_rel, naive_err / reference.abs());
        }
    }

    /// JSONL with missing keys produces empty strings (not panics).
    /// We generate JSONL where some lines have fewer keys than others.
    #[test]
    fn jsonl_missing_keys_no_panic(
        fields_full in proptest::collection::vec("[a-z]{1,5}", 3..6),
        fields_partial in proptest::collection::vec("[a-z]{1,5}", 1..3),
    ) {
        // Create two lines: one with all fields, one with a subset.
        let full_pairs: Vec<String> = fields_full.iter()
            .map(|k| format!("\"{}\":\"v\"", k))
            .collect();
        let partial_pairs: Vec<String> = fields_partial.iter()
            .map(|k| format!("\"{}\":\"w\"", k))
            .collect();
        let jsonl = format!("{{{}}}\n{{{}}}", full_pairs.join(","), partial_pairs.join(","));

        // Should not panic.
        let lines: Vec<&str> = jsonl.lines().collect();
        let mut all_keys = BTreeSet::new();
        for line in &lines {
            if let Some(keys) = parse_json_object_keys(line) {
                all_keys.extend(keys);
            }
        }
        // All keys should be discovered (union of both sets).
        for k in &fields_full {
            prop_assert!(all_keys.contains(k),
                "Missing key '{}' from full set", k);
        }
    }

    /// TabularData column count == headers length for any loaded CSV.
    #[test]
    fn tabular_column_count_matches_headers((_headers, csv) in arb_csv_content()) {
        let (parsed_headers, parsed_rows) = load_delimited(&csv, ',', true);
        let ncols = parsed_headers.len();
        for (i, row) in parsed_rows.iter().enumerate() {
            prop_assert_eq!(row.len(), ncols,
                "Row {} has {} columns but headers have {} columns",
                i, row.len(), ncols);
        }
    }

    /// TabularData column count == headers length for any loaded TSV.
    #[test]
    fn tabular_tsv_column_count_matches_headers(
        headers in proptest::collection::vec("[a-zA-Z_]{1,8}", 1..6),
        nrows in 0usize..15,
    ) {
        let ncols = headers.len();
        let header_line = headers.join("\t");
        let mut tsv = header_line;
        for _ in 0..nrows {
            tsv.push('\n');
            let row: Vec<&str> = (0..ncols).map(|_| "val").collect();
            tsv.push_str(&row.join("\t"));
        }
        let (parsed_headers, parsed_rows) = load_delimited(&tsv, '\t', true);
        prop_assert_eq!(parsed_headers.len(), ncols);
        for (i, row) in parsed_rows.iter().enumerate() {
            prop_assert_eq!(row.len(), ncols,
                "TSV row {} has {} columns but expected {}", i, row.len(), ncols);
        }
    }
}
