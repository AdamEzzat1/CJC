//! Regression gate: compare a Locke report (read from disk) against
//! `expected_findings.json`.
//!
//! Run order:
//!
//! 1. `cargo run --release -p lendingclub-demo -- --input <csv.gz> --output demos/lendingclub/out/report.json`
//! 2. `cargo test --release -p lendingclub-demo --test expected_findings`
//!
//! The test is *skipped* with a warning print if `out/report.json` does
//! not exist — the CI environment may not have the LC dataset. To make
//! it a hard gate locally, set `LENDINGCLUB_REPORT_PATH=...` in the env
//! and the test will fail-on-absent instead of skipping.
//!
//! The fixture format is intentionally minimal — see expected_findings.json
//! for the full schema. This test parses with a hand-rolled lightweight
//! JSON walker over the LockeReport JSON because we don't pull in serde.

use std::collections::BTreeSet;
use std::path::PathBuf;

/// Where to look for the report. Falls back to the demo's default output
/// path; honors `LENDINGCLUB_REPORT_PATH` if set.
fn report_path() -> PathBuf {
    if let Ok(p) = std::env::var("LENDINGCLUB_REPORT_PATH") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("out")
        .join("report.json")
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("expected_findings.json")
}

/// Parse the LockeReport JSON enough to extract (code, column) tuples.
/// We're not pulling serde into the demo crate just for this — the JSON
/// shape is stable (BTreeMap-ordered) so a simple textual walker works.
///
/// Returns: BTreeSet<(code, Option<column>)> covering every finding.
fn extract_code_column_pairs(json: &str) -> BTreeSet<(String, Option<String>)> {
    let mut out = BTreeSet::new();
    // The findings array contains objects like:
    //   { "code": "E9060", "column": "total_pymnt", ... }
    // We walk linearly, tracking the most recent "code" and "column".
    let mut current_code: Option<String> = None;
    let mut current_column: Option<String> = None;
    let mut depth = 0i32;

    // findings objects sit at depth 2: depth-1 is the outer report object;
    // the `findings` array doesn't change depth; each finding `{` takes
    // us to depth 2. Inner evidence objects at depth 3 are skipped.
    for (i, ch) in json.char_indices() {
        match ch {
            '{' => {
                depth += 1;
                if depth == 2 {
                    current_code = None;
                    current_column = None;
                }
            }
            '}' => {
                if depth == 2 {
                    if let Some(code) = current_code.take() {
                        out.insert((code, current_column.take()));
                    }
                }
                depth -= 1;
            }
            '"' => {
                // Only scan for code/column while inside a finding object
                // (depth 2). Avoids picking up `column_types` keys from
                // the input summary at depth 2 — those look like
                // `"loan_amnt":"Float"`, not `"column":"loan_amnt"`.
                if depth == 2 {
                    if let Some(rest) = json.get(i..) {
                        if rest.starts_with("\"code\"") {
                            if let Some(val) =
                                extract_string_value_after(json, i + "\"code\"".len())
                            {
                                current_code = Some(val);
                            }
                        } else if rest.starts_with("\"column\"") {
                            if let Some(val) =
                                extract_string_value_after(json, i + "\"column\"".len())
                            {
                                current_column = Some(val);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    out
}

fn extract_string_value_after(s: &str, start: usize) -> Option<String> {
    let bytes = s.as_bytes();
    let mut i = start;
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b':' || bytes[i] == b'\t') {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b'"' {
        return None;
    }
    i += 1;
    let mut out = Vec::new();
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'\\' && i + 1 < bytes.len() {
            // tolerate escape sequences; we don't need to decode them
            // perfectly because we only compare against ASCII column
            // names and E-codes.
            out.push(bytes[i + 1]);
            i += 2;
            continue;
        }
        if b == b'"' {
            return String::from_utf8(out).ok();
        }
        out.push(b);
        i += 1;
    }
    None
}

#[derive(Debug)]
struct RequiredFinding {
    code: String,
    min_count: usize,
    fingerprint_columns: Vec<String>,
}

fn parse_fixture(json: &str) -> Vec<RequiredFinding> {
    let mut out = Vec::new();
    let mut i = 0;
    // Find each `"code": "..."` and walk forward to its `min_count` and
    // `fingerprint_columns` array within the same object.
    while let Some(rel) = json[i..].find("\"code\"") {
        let code_start = i + rel;
        let Some(code) = extract_string_value_after(json, code_start + "\"code\"".len()) else {
            break;
        };
        // The object ends at the next top-level `}` — find it by
        // matching braces from after the code.
        let mut depth = 0i32;
        let mut obj_end = json.len();
        for (off, ch) in json[code_start..].char_indices() {
            if ch == '{' {
                depth += 1;
            } else if ch == '}' {
                depth -= 1;
                if depth < 0 {
                    obj_end = code_start + off;
                    break;
                }
            }
        }
        let slice = &json[code_start..obj_end];

        let min_count = slice
            .find("\"min_count\"")
            .and_then(|p| {
                let tail = &slice[p + "\"min_count\"".len()..];
                let tail = tail.trim_start_matches(|c: char| c == ' ' || c == ':' || c == '\t');
                let n_end = tail
                    .find(|c: char| !c.is_ascii_digit())
                    .unwrap_or(tail.len());
                tail[..n_end].parse::<usize>().ok()
            })
            .unwrap_or(1);

        let columns = parse_fingerprint_columns(slice);

        out.push(RequiredFinding {
            code,
            min_count,
            fingerprint_columns: columns,
        });
        i = obj_end + 1;
    }
    out
}

fn parse_fingerprint_columns(obj_slice: &str) -> Vec<String> {
    let key = "\"fingerprint_columns\"";
    let Some(pos) = obj_slice.find(key) else {
        return Vec::new();
    };
    let tail = &obj_slice[pos + key.len()..];
    let Some(arr_start) = tail.find('[') else {
        return Vec::new();
    };
    let Some(arr_end) = tail[arr_start..].find(']') else {
        return Vec::new();
    };
    let arr = &tail[arr_start..arr_start + arr_end];
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(rel) = arr[i..].find('"') {
        let lo = i + rel + 1;
        if let Some(hi_rel) = arr[lo..].find('"') {
            out.push(arr[lo..lo + hi_rel].to_string());
            i = lo + hi_rel + 1;
        } else {
            break;
        }
    }
    out
}

#[test]
fn regression_gate_against_expected_findings() {
    let report_path = report_path();
    let env_set = std::env::var("LENDINGCLUB_REPORT_PATH").is_ok();
    if !report_path.exists() {
        if env_set {
            panic!(
                "LENDINGCLUB_REPORT_PATH was set to {} but the file does not exist",
                report_path.display()
            );
        }
        eprintln!(
            "SKIP: {} not present; run `cargo run --release -p lendingclub-demo -- --input <csv.gz> --output {}` first",
            report_path.display(),
            report_path.display()
        );
        return;
    }

    let report_json = std::fs::read_to_string(&report_path)
        .unwrap_or_else(|e| panic!("read {}: {}", report_path.display(), e));
    let fixture_json = std::fs::read_to_string(fixture_path())
        .unwrap_or_else(|e| panic!("read fixture: {}", e));

    let pairs = extract_code_column_pairs(&report_json);
    let required = parse_fixture(&fixture_json);
    assert!(!required.is_empty(), "fixture parsed zero required findings");

    let mut failures: Vec<String> = Vec::new();

    for req in &required {
        let actual_count = pairs.iter().filter(|(c, _)| c == &req.code).count();
        if actual_count < req.min_count {
            failures.push(format!(
                "{}: required min_count={}, actual={}",
                req.code, req.min_count, actual_count
            ));
            continue;
        }
        for fp_col in &req.fingerprint_columns {
            let hit = pairs
                .iter()
                .any(|(c, col)| c == &req.code && col.as_deref() == Some(fp_col.as_str()));
            if !hit {
                failures.push(format!(
                    "{}: missing fingerprint firing on column `{}`",
                    req.code, fp_col
                ));
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "expected_findings.json regression gate failed ({} issues):\n  - {}",
            failures.len(),
            failures.join("\n  - ")
        );
    }
}

#[test]
fn fixture_itself_is_well_formed() {
    let fixture_json =
        std::fs::read_to_string(fixture_path()).expect("expected_findings.json missing");
    let required = parse_fixture(&fixture_json);
    assert!(
        required.len() >= 3,
        "fixture should declare at least 3 required findings (got {})",
        required.len()
    );
    for r in &required {
        assert!(
            r.code.starts_with('E') && r.code.len() == 5,
            "code `{}` doesn't look like Exxxx",
            r.code
        );
        assert!(r.min_count >= 1, "code {} has min_count < 1", r.code);
    }
}
