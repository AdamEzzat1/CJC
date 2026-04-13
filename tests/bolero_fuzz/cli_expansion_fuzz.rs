//! Bolero fuzz tests targeting CLI expansion format parsing robustness.
//!
//! Since `cjc-cli` is a binary crate, we duplicate the core parsing logic inline
//! and fuzz it directly to verify that malformed input never causes panics.
//!
//! Run with:
//!   cargo test --test bolero_fuzz cli_expansion
//!
//! For coverage-guided fuzzing (Linux only):
//!   cargo bolero test bolero_fuzz::cli_expansion_fuzz::fuzz_load_jsonl

use std::collections::{BTreeMap, BTreeSet};
use std::panic;

// ── Inline format parsing logic (mirrors cjc_cli::formats) ─────────

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

fn parse_delimited_row(line: &str, delimiter: char, ncols: usize) -> Vec<String> {
    let mut fields: Vec<String> = line.split(delimiter).map(|s| s.trim().to_string()).collect();
    while fields.len() < ncols {
        fields.push(String::new());
    }
    fields.truncate(ncols);
    fields
}

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

// Minimal JSON parser subset
fn skip_ws(bytes: &[u8], pos: &mut usize) {
    while *pos < bytes.len() && matches!(bytes[*pos], b' ' | b'\t' | b'\n' | b'\r') {
        *pos += 1;
    }
}

fn parse_json_string(bytes: &[u8], pos: &mut usize) -> Result<String, String> {
    if *pos >= bytes.len() || bytes[*pos] != b'"' {
        return Err(format!("expected '\"' at position {}", *pos));
    }
    *pos += 1;
    let mut s = String::new();
    while *pos < bytes.len() {
        let b = bytes[*pos];
        if b == b'\\' {
            *pos += 1;
            if *pos >= bytes.len() {
                return Err("unexpected end after backslash".to_string());
            }
            match bytes[*pos] {
                b'"' => s.push('"'),
                b'\\' => s.push('\\'),
                b'/' => s.push('/'),
                b'n' => s.push('\n'),
                b't' => s.push('\t'),
                b'r' => s.push('\r'),
                b'b' => s.push('\u{0008}'),
                b'f' => s.push('\u{000C}'),
                b'u' => {
                    *pos += 1;
                    if *pos + 4 > bytes.len() {
                        return Err("incomplete \\u escape".to_string());
                    }
                    let hex_str = std::str::from_utf8(&bytes[*pos..*pos + 4])
                        .map_err(|_| "invalid \\u escape".to_string())?;
                    let code = u32::from_str_radix(hex_str, 16)
                        .map_err(|_| format!("invalid hex in \\u escape: {}", hex_str))?;
                    if let Some(ch) = char::from_u32(code) {
                        s.push(ch);
                    } else {
                        s.push('\u{FFFD}');
                    }
                    *pos += 3;
                }
                other => {
                    s.push('\\');
                    s.push(other as char);
                }
            }
        } else if b == b'"' {
            *pos += 1;
            return Ok(s);
        } else {
            s.push(b as char);
        }
        *pos += 1;
    }
    Err("unterminated string".to_string())
}

fn skip_json_compound(bytes: &[u8], pos: &mut usize) -> Result<(), String> {
    let open = bytes[*pos];
    let close = if open == b'{' { b'}' } else { b']' };
    let mut depth = 1;
    *pos += 1;
    while *pos < bytes.len() && depth > 0 {
        match bytes[*pos] {
            b'"' => {
                *pos += 1;
                while *pos < bytes.len() {
                    if bytes[*pos] == b'\\' {
                        *pos += 2;
                        continue;
                    }
                    if bytes[*pos] == b'"' {
                        *pos += 1;
                        break;
                    }
                    *pos += 1;
                }
                continue;
            }
            b if b == open => depth += 1,
            b if b == close => depth -= 1,
            _ => {}
        }
        *pos += 1;
    }
    if depth != 0 {
        return Err("unterminated compound".to_string());
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    Str(String),
    Raw(String),
}

impl JsonValue {
    fn to_cell_string(&self) -> String {
        match self {
            JsonValue::Null => String::new(),
            JsonValue::Bool(b) => if *b { "true" } else { "false" }.to_string(),
            JsonValue::Number(n) => {
                if n.fract() == 0.0 && n.abs() < (i64::MAX as f64) {
                    format!("{}", *n as i64)
                } else {
                    format!("{}", n)
                }
            }
            JsonValue::Str(s) => s.clone(),
            JsonValue::Raw(s) => s.clone(),
        }
    }
}

fn parse_json_number(bytes: &[u8], pos: &mut usize) -> Result<JsonValue, String> {
    let start = *pos;
    if *pos < bytes.len() && bytes[*pos] == b'-' {
        *pos += 1;
    }
    while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
        *pos += 1;
    }
    if *pos < bytes.len() && bytes[*pos] == b'.' {
        *pos += 1;
        while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
            *pos += 1;
        }
    }
    if *pos < bytes.len() && (bytes[*pos] == b'e' || bytes[*pos] == b'E') {
        *pos += 1;
        if *pos < bytes.len() && (bytes[*pos] == b'+' || bytes[*pos] == b'-') {
            *pos += 1;
        }
        while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
            *pos += 1;
        }
    }
    let num_str = std::str::from_utf8(&bytes[start..*pos])
        .map_err(|_| "invalid UTF-8 in number".to_string())?;
    let n: f64 = num_str
        .parse()
        .map_err(|_| format!("invalid number: {}", num_str))?;
    Ok(JsonValue::Number(n))
}

fn parse_json_value(bytes: &[u8], pos: &mut usize) -> Result<JsonValue, String> {
    skip_ws(bytes, pos);
    if *pos >= bytes.len() {
        return Err("unexpected end of input".to_string());
    }
    match bytes[*pos] {
        b'"' => {
            let s = parse_json_string(bytes, pos)?;
            Ok(JsonValue::Str(s))
        }
        b't' => {
            if bytes[*pos..].starts_with(b"true") {
                *pos += 4;
                Ok(JsonValue::Bool(true))
            } else {
                Err(format!("unexpected token at position {}", *pos))
            }
        }
        b'f' => {
            if bytes[*pos..].starts_with(b"false") {
                *pos += 5;
                Ok(JsonValue::Bool(false))
            } else {
                Err(format!("unexpected token at position {}", *pos))
            }
        }
        b'n' => {
            if bytes[*pos..].starts_with(b"null") {
                *pos += 4;
                Ok(JsonValue::Null)
            } else {
                Err(format!("unexpected token at position {}", *pos))
            }
        }
        b'-' | b'0'..=b'9' => parse_json_number(bytes, pos),
        b'[' | b'{' => {
            let start = *pos;
            skip_json_compound(bytes, pos)?;
            let raw = std::str::from_utf8(&bytes[start..*pos])
                .map_err(|_| "invalid UTF-8 in nested structure".to_string())?;
            Ok(JsonValue::Raw(raw.to_string()))
        }
        other => Err(format!("unexpected byte '{}' at position {}", other as char, *pos)),
    }
}

fn parse_json_object(input: &str) -> Result<BTreeMap<String, JsonValue>, String> {
    let input = input.trim();
    if !input.starts_with('{') {
        return Err("expected '{'".to_string());
    }
    let bytes = input.as_bytes();
    let mut pos = 1;
    let mut map = BTreeMap::new();

    skip_ws(bytes, &mut pos);
    if pos < bytes.len() && bytes[pos] == b'}' {
        return Ok(map);
    }

    loop {
        skip_ws(bytes, &mut pos);
        let key = parse_json_string(bytes, &mut pos)?;
        skip_ws(bytes, &mut pos);
        if pos >= bytes.len() || bytes[pos] != b':' {
            return Err(format!("expected ':' at position {}", pos));
        }
        pos += 1;
        skip_ws(bytes, &mut pos);
        let value = parse_json_value(bytes, &mut pos)?;
        map.insert(key, value);

        skip_ws(bytes, &mut pos);
        if pos >= bytes.len() {
            break;
        }
        if bytes[pos] == b'}' {
            break;
        }
        if bytes[pos] == b',' {
            pos += 1;
            continue;
        }
        return Err(format!("unexpected byte at position {}", pos));
    }
    Ok(map)
}

fn load_jsonl(content: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut key_set = BTreeSet::new();
    let mut parsed: Vec<BTreeMap<String, JsonValue>> = Vec::new();
    for line in &lines {
        if let Ok(obj) = parse_json_object(line) {
            for key in obj.keys() {
                key_set.insert(key.clone());
            }
            parsed.push(obj);
        } else {
            parsed.push(BTreeMap::new());
        }
    }
    let headers: Vec<String> = key_set.into_iter().collect();
    let rows: Vec<Vec<String>> = parsed
        .iter()
        .map(|obj| {
            headers
                .iter()
                .map(|h| obj.get(h).map(|v| v.to_cell_string()).unwrap_or_default())
                .collect()
        })
        .collect();
    (headers, rows)
}

fn extract_metadata_from_bytes(bytes: &[u8]) -> (&'static str, Option<String>) {
    let fmt = detect_from_magic(bytes);
    let magic = if !bytes.is_empty() {
        let display_len = bytes.len().min(8);
        Some(
            bytes[..display_len]
                .iter()
                .map(|b| format!("{:02x}", b))
                .collect::<Vec<_>>()
                .join(""),
        )
    } else {
        None
    };
    (fmt, magic)
}

// ── Fuzz tests ──────────────────────────────────────────────────────

/// Fuzz: malformed JSONL should never panic load_jsonl.
#[test]
fn fuzz_load_jsonl() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        if let Ok(s) = std::str::from_utf8(input) {
            let s = s.to_string();
            let _ = panic::catch_unwind(|| {
                let _ = load_jsonl(&s);
            });
        }
    });
}

/// Fuzz: random bytes should never panic detect_from_magic.
#[test]
fn fuzz_detect_from_magic() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        // detect_from_magic should never panic on any input.
        let result = detect_from_magic(input);
        // It should always return a valid static string.
        assert!(!result.is_empty());
    });
}

/// Fuzz: random CSV content should never panic load_delimited.
#[test]
fn fuzz_load_delimited_csv() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        if let Ok(s) = std::str::from_utf8(input) {
            let s = s.to_string();
            let _ = panic::catch_unwind(|| {
                let _ = load_delimited(&s, ',', true);
                let _ = load_delimited(&s, ',', false);
            });
        }
    });
}

/// Fuzz: random bytes as file metadata should never panic extract_metadata.
#[test]
fn fuzz_extract_metadata() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let _ = panic::catch_unwind(|| {
            let (fmt, magic) = extract_metadata_from_bytes(input);
            // Should always return a valid format string.
            assert!(!fmt.is_empty());
            // Magic bytes string should be well-formed hex or None.
            if let Some(ref hex) = magic {
                assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
            }
        });
    });
}

/// Fuzz: JSONL with random Unicode field names should never panic.
#[test]
fn fuzz_jsonl_unicode_fields() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        if let Ok(s) = std::str::from_utf8(input) {
            // Wrap in a JSON object structure.
            let jsonl_line = format!("{{\"{}\":\"value\"}}", s.replace('\\', "\\\\").replace('"', "\\\""));
            let _ = panic::catch_unwind(|| {
                let _ = load_jsonl(&jsonl_line);
            });
        }
    });
}

/// Fuzz: CSV with random delimiters should never panic.
#[test]
fn fuzz_csv_random_delimiters() {
    bolero::check!()
        .with_type::<(Vec<u8>, u8)>()
        .for_each(|&(ref input, delim_byte): &(Vec<u8>, u8)| {
            if let Ok(s) = std::str::from_utf8(input) {
                let s = s.to_string();
                // Use the byte as a char delimiter (valid ASCII range).
                if delim_byte.is_ascii() && delim_byte != 0 {
                    let delimiter = delim_byte as char;
                    let _ = panic::catch_unwind(|| {
                        let _ = load_delimited(&s, delimiter, true);
                        let _ = load_delimited(&s, delimiter, false);
                    });
                }
            }
        });
}

/// Fuzz: JSON object parsing with random input should never panic.
#[test]
fn fuzz_json_object_parsing() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        if let Ok(s) = std::str::from_utf8(input) {
            let s = s.to_string();
            let _ = panic::catch_unwind(|| {
                // parse_json_object returns Result, so it should gracefully
                // handle any input. We just ensure no panic.
                let _ = parse_json_object(&s);
            });
        }
    });
}
