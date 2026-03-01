//! Hardening test H9: File I/O, JSON, and DateTime integration tests.
//!
//! Tests the new builtins via the MIR executor pipeline:
//! - File I/O: file_read, file_write, file_exists, file_lines
//! - JSON: json_parse, json_stringify (sorted keys, roundtrip)
//! - DateTime: datetime_from_parts, datetime_year, etc. (pure arithmetic)

use std::fs;
use std::rc::Rc;

/// Helper: parse + MIR-execute a CJC program, return executor output.
fn run_mir(src: &str) -> Vec<String> {
    let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

// ── JSON tests ──────────────────────────────────────────────────────

#[test]
fn h9_json_parse_object() {
    let out = run_mir(r#"
let s = "{\"a\":1,\"b\":\"hello\"}";
let obj = json_parse(s);
print(obj);
"#);
    // JSON output is an object displayed as struct
    assert!(!out.is_empty());
}

#[test]
fn h9_json_roundtrip() {
    let out = run_mir(r#"
let s = "{\"a\":1,\"b\":\"hello\",\"c\":[true,null]}";
let obj = json_parse(s);
let s2 = json_stringify(obj);
print(s2);
"#);
    assert_eq!(out, vec![r#"{"a":1,"b":"hello","c":[true,null]}"#]);
}

#[test]
fn h9_json_sorted_keys() {
    let out = run_mir(r#"
let s = "{\"z\":1,\"a\":2,\"m\":3}";
let obj = json_parse(s);
let s2 = json_stringify(obj);
print(s2);
"#);
    // Keys must be sorted alphabetically
    assert_eq!(out, vec![r#"{"a":2,"m":3,"z":1}"#]);
}

#[test]
fn h9_json_parse_array() {
    let out = run_mir(r#"
let arr = json_parse("[1,2,3]");
let s = json_stringify(arr);
print(s);
"#);
    assert_eq!(out, vec!["[1,2,3]"]);
}

#[test]
fn h9_json_parse_nested() {
    let out = run_mir(r#"
let s = "{\"items\":[{\"id\":1},{\"id\":2}]}";
let obj = json_parse(s);
let s2 = json_stringify(obj);
print(s2);
"#);
    assert_eq!(out, vec![r#"{"items":[{"id":1},{"id":2}]}"#]);
}

// ── DateTime tests ──────────────────────────────────────────────────

#[test]
fn h9_datetime_from_parts_and_extract() {
    let out = run_mir(r#"
let dt = datetime_from_parts(2024, 6, 15, 14, 30, 45);
let y = datetime_year(dt);
let m = datetime_month(dt);
let d = datetime_day(dt);
let h = datetime_hour(dt);
let min = datetime_minute(dt);
let sec = datetime_second(dt);
print(y);
print(m);
print(d);
print(h);
print(min);
print(sec);
"#);
    assert_eq!(out, vec!["2024", "6", "15", "14", "30", "45"]);
}

#[test]
fn h9_datetime_format_iso() {
    let out = run_mir(r#"
let dt = datetime_from_parts(2024, 3, 14, 9, 26, 53);
let s = datetime_format(dt);
print(s);
"#);
    assert_eq!(out, vec!["2024-03-14T09:26:53Z"]);
}

#[test]
fn h9_datetime_diff() {
    let out = run_mir(r#"
let a = datetime_from_parts(2024, 1, 2, 0, 0, 0);
let b = datetime_from_parts(2024, 1, 1, 0, 0, 0);
let diff = datetime_diff(a, b);
print(diff);
"#);
    // Diff should be 86400000 (24 * 60 * 60 * 1000)
    assert_eq!(out, vec!["86400000"]);
}

#[test]
fn h9_datetime_determinism() {
    // Same inputs → identical outputs
    let out1 = run_mir(r#"
let dt = datetime_from_parts(2000, 1, 1, 0, 0, 0);
print(datetime_format(dt));
"#);
    let out2 = run_mir(r#"
let dt = datetime_from_parts(2000, 1, 1, 0, 0, 0);
print(datetime_format(dt));
"#);
    assert_eq!(out1, out2);
    assert_eq!(out1, vec!["2000-01-01T00:00:00Z"]);
}

#[test]
fn h9_datetime_leap_year() {
    let out = run_mir(r#"
let dt = datetime_from_parts(2024, 2, 29, 12, 0, 0);
let m = datetime_month(dt);
let d = datetime_day(dt);
print(m);
print(d);
"#);
    assert_eq!(out, vec!["2", "29"]);
}

// ── File I/O tests ──────────────────────────────────────────────────

#[test]
fn h9_file_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.txt");
    let path_str = path.to_string_lossy().replace('\\', "/");

    let src = format!(
        r#"
file_write("{}", "Hello, CJC!");
let content = file_read("{}");
print(content);
"#,
        path_str, path_str
    );
    let out = run_mir(&src);
    assert_eq!(out, vec!["Hello, CJC!"]);
}

#[test]
fn h9_file_exists() {
    let dir = tempfile::tempdir().unwrap();
    let existing = dir.path().join("exists.txt");
    let nonexistent = dir.path().join("no.txt");
    fs::write(&existing, "data").unwrap();

    let existing_str = existing.to_string_lossy().replace('\\', "/");
    let nonexistent_str = nonexistent.to_string_lossy().replace('\\', "/");

    let src = format!(
        r#"
print(file_exists("{}"));
print(file_exists("{}"));
"#,
        existing_str, nonexistent_str
    );
    let out = run_mir(&src);
    assert_eq!(out, vec!["true", "false"]);
}

#[test]
fn h9_file_lines() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lines.txt");
    fs::write(&path, "line1\nline2\nline3").unwrap();
    let path_str = path.to_string_lossy().replace('\\', "/");

    let src = format!(
        r#"
let lines = file_lines("{}");
print(len(lines));
"#,
        path_str
    );
    let out = run_mir(&src);
    assert_eq!(out, vec!["3"]);
}

// ── JSON + File I/O combined ────────────────────────────────────────

#[test]
fn h9_json_file_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.json");
    let path_str = path.to_string_lossy().replace('\\', "/");

    let src = format!(
        r#"
let data = json_parse("{{\"key\":\"value\",\"num\":42}}");
let s = json_stringify(data);
file_write("{}", s);
let loaded = file_read("{}");
print(loaded);
"#,
        path_str, path_str
    );
    let out = run_mir(&src);
    assert_eq!(out, vec![r#"{"key":"value","num":42}"#]);
}

// ── Effect registry tests ───────────────────────────────────────────

#[test]
fn h9_effect_registry_json_registered() {
    use cjc_types::effect_registry;
    assert!(effect_registry::lookup("json_parse").is_some());
    assert!(effect_registry::lookup("json_stringify").is_some());
    assert!(effect_registry::is_safe_builtin("json_parse"));
}

#[test]
fn h9_effect_registry_datetime_classified() {
    use cjc_types::effect_registry;
    assert!(effect_registry::is_nondeterministic("datetime_now"));
    assert!(!effect_registry::is_nondeterministic("datetime_year"));
    assert!(effect_registry::lookup("datetime_format").is_some());
}

#[test]
fn h9_effect_registry_file_io_classified() {
    use cjc_types::effect_registry;
    use cjc_types::EffectSet;
    let read_effects = effect_registry::lookup("file_read").unwrap();
    assert!(read_effects.has(EffectSet::IO));
    assert!(read_effects.has(EffectSet::ALLOC));
}
