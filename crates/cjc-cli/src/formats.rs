//! Shared multi-format tabular data loading for CJC CLI commands.
//!
//! Provides format detection (extension + magic bytes), CSV/TSV/JSONL parsing,
//! safe metadata extraction for binary formats, and a unified `TabularData`
//! abstraction.
//!
//! Design constraints:
//! - Zero external dependencies (std only)
//! - Deterministic output: BTreeMap/BTreeSet everywhere, never HashMap/HashSet
//! - Binary model files (.pkl, .onnx, .joblib) are NEVER deserialized or executed
//! - Parquet/Arrow/SQLite support is metadata-only (no full materialization)

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;

// ── Data format enum ─────────────────────────────────────────────────

/// Recognized file formats for tabular and binary data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DataFormat {
    Csv,
    Tsv,
    Jsonl,
    Parquet,
    ArrowIpc,
    Sqlite,
    Pickle,
    Onnx,
    Joblib,
    Unknown,
}

impl DataFormat {
    /// Human-readable label for display.
    pub fn label(&self) -> &'static str {
        match self {
            DataFormat::Csv => "CSV",
            DataFormat::Tsv => "TSV",
            DataFormat::Jsonl => "JSONL",
            DataFormat::Parquet => "Parquet",
            DataFormat::ArrowIpc => "Arrow IPC",
            DataFormat::Sqlite => "SQLite",
            DataFormat::Pickle => "Pickle",
            DataFormat::Onnx => "ONNX",
            DataFormat::Joblib => "Joblib",
            DataFormat::Unknown => "Unknown",
        }
    }

    /// Whether this format supports full tabular materialization
    /// (i.e. we can parse every row into `TabularData`).
    pub fn is_materializable(&self) -> bool {
        matches!(self, DataFormat::Csv | DataFormat::Tsv | DataFormat::Jsonl)
    }

    /// Whether this format is a binary model file that must never be
    /// deserialized or executed.
    pub fn is_model_file(&self) -> bool {
        matches!(self, DataFormat::Pickle | DataFormat::Onnx | DataFormat::Joblib)
    }
}

// ── Format detection ─────────────────────────────────────────────────

/// Detect format from file extension (case-insensitive).
pub fn detect_from_extension(path: &Path) -> DataFormat {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "csv" => DataFormat::Csv,
        "tsv" => DataFormat::Tsv,
        "jsonl" | "ndjson" => DataFormat::Jsonl,
        "parquet" => DataFormat::Parquet,
        "feather" | "arrow" | "ipc" => DataFormat::ArrowIpc,
        "sqlite" | "db" | "sqlite3" => DataFormat::Sqlite,
        "pkl" | "pickle" => DataFormat::Pickle,
        "onnx" => DataFormat::Onnx,
        "joblib" => DataFormat::Joblib,
        _ => DataFormat::Unknown,
    }
}

/// Magic byte signatures for binary format detection.
/// Each entry: (format, magic_bytes, offset).
const MAGIC_SIGNATURES: &[(DataFormat, &[u8], usize)] = &[
    (DataFormat::Parquet, b"PAR1", 0),
    (DataFormat::ArrowIpc, b"ARROW1", 0),
    (DataFormat::Sqlite, b"SQLite format 3\0", 0),
    // Pickle protocol 2+ starts with \x80
    (DataFormat::Pickle, &[0x80], 0),
];

/// Detect format from magic bytes at the start of a file.
/// Returns `DataFormat::Unknown` if no magic signature matches.
pub fn detect_from_magic(header: &[u8]) -> DataFormat {
    for &(format, magic, offset) in MAGIC_SIGNATURES {
        if header.len() >= offset + magic.len() && header[offset..offset + magic.len()] == *magic {
            return format;
        }
    }
    DataFormat::Unknown
}

/// Detect format using both extension and magic bytes.
/// Extension takes priority; magic bytes are used for confirmation or fallback.
pub fn detect_format(path: &Path) -> DataFormat {
    let ext_fmt = detect_from_extension(path);
    if ext_fmt != DataFormat::Unknown {
        return ext_fmt;
    }
    // Fallback: try magic bytes
    if let Ok(mut f) = File::open(path) {
        let mut buf = [0u8; 32];
        if let Ok(n) = f.read(&mut buf) {
            let magic_fmt = detect_from_magic(&buf[..n]);
            if magic_fmt != DataFormat::Unknown {
                return magic_fmt;
            }
        }
    }
    DataFormat::Unknown
}

// ── Tabular data abstraction ─────────────────────────────────────────

/// Unified tabular data representation.
/// Headers are always present (auto-generated as "col_0", "col_1", ... if absent).
/// Row values are always strings; callers can parse as needed.
#[derive(Debug, Clone)]
pub struct TabularData {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub source_format: DataFormat,
}

impl TabularData {
    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.headers.len()
    }

    /// Number of data rows (excluding header).
    pub fn nrows(&self) -> usize {
        self.rows.len()
    }

    /// Get a column by name. Returns None if not found.
    pub fn column(&self, name: &str) -> Option<Vec<&str>> {
        let idx = self.headers.iter().position(|h| h == name)?;
        Some(
            self.rows
                .iter()
                .map(|row| {
                    if idx < row.len() {
                        row[idx].as_str()
                    } else {
                        ""
                    }
                })
                .collect(),
        )
    }
}

// ── CSV/TSV loading ──────────────────────────────────────────────────

/// Parse delimited text into `TabularData`.
///
/// If `has_header` is true, the first non-empty line is treated as column names.
/// Otherwise, columns are named "col_0", "col_1", etc.
///
/// Fields are split by the delimiter and trimmed of leading/trailing whitespace.
/// This is a simple split-based parser (no RFC 4180 quoted-field handling) that
/// matches the existing CJC CLI parsing pattern used in flow, schema, drift, etc.
pub fn load_delimited(content: &str, delimiter: char, has_header: bool) -> TabularData {
    let format = if delimiter == '\t' {
        DataFormat::Tsv
    } else {
        DataFormat::Csv
    };
    let mut lines = content.lines().filter(|l| !l.is_empty());

    let headers: Vec<String>;
    let first_data_line: Option<&str>;

    if has_header {
        if let Some(hdr) = lines.next() {
            headers = hdr.split(delimiter).map(|s| s.trim().to_string()).collect();
            first_data_line = None;
        } else {
            return TabularData {
                headers: Vec::new(),
                rows: Vec::new(),
                source_format: format,
            };
        }
    } else {
        // Peek at first line to determine column count.
        if let Some(first) = lines.next() {
            let ncols = first.split(delimiter).count();
            headers = (0..ncols).map(|i| format!("col_{}", i)).collect();
            first_data_line = Some(first);
        } else {
            return TabularData {
                headers: Vec::new(),
                rows: Vec::new(),
                source_format: format,
            };
        }
    }

    let ncols = headers.len();
    let mut rows = Vec::new();

    // If we consumed the first data line for column count, include it.
    if let Some(first) = first_data_line {
        rows.push(parse_delimited_row(first, delimiter, ncols));
    }

    for line in lines {
        rows.push(parse_delimited_row(line, delimiter, ncols));
    }

    TabularData {
        headers,
        rows,
        source_format: format,
    }
}

/// Parse a single delimited row, padding or truncating to `ncols`.
fn parse_delimited_row(line: &str, delimiter: char, ncols: usize) -> Vec<String> {
    let mut fields: Vec<String> = line.split(delimiter).map(|s| s.trim().to_string()).collect();
    // Pad with empty strings if row is short.
    while fields.len() < ncols {
        fields.push(String::new());
    }
    // Truncate if row is longer than header.
    fields.truncate(ncols);
    fields
}

/// Streaming delimited reader. Returns an iterator of parsed rows.
///
/// The first call to `next()` on the returned iterator yields the first data row.
/// Headers are extracted separately and returned as the first element of the tuple.
pub fn load_delimited_streaming<R: BufRead>(
    mut reader: R,
    delimiter: char,
    has_header: bool,
) -> (Vec<String>, DelimitedRowIter<R>) {
    let mut first_line = String::new();
    let _ = reader.read_line(&mut first_line);
    let first_line = first_line.trim_end_matches(|c| c == '\n' || c == '\r');

    let (headers, pending) = if has_header {
        let hdrs: Vec<String> = first_line
            .split(delimiter)
            .map(|s| s.trim().to_string())
            .collect();
        (hdrs, None)
    } else {
        let ncols = first_line.split(delimiter).count();
        let hdrs: Vec<String> = (0..ncols).map(|i| format!("col_{}", i)).collect();
        let row = parse_delimited_row(first_line, delimiter, ncols);
        (hdrs, Some(row))
    };

    let ncols = headers.len();
    let iter = DelimitedRowIter {
        reader,
        delimiter,
        ncols,
        buf: String::new(),
        pending,
        done: false,
    };
    (headers, iter)
}

/// Iterator that yields parsed rows from a buffered reader.
pub struct DelimitedRowIter<R: BufRead> {
    reader: R,
    delimiter: char,
    ncols: usize,
    buf: String,
    pending: Option<Vec<String>>,
    done: bool,
}

impl<R: BufRead> Iterator for DelimitedRowIter<R> {
    type Item = Vec<String>;

    fn next(&mut self) -> Option<Vec<String>> {
        // Return pending row first (from no-header mode).
        if let Some(row) = self.pending.take() {
            return Some(row);
        }
        if self.done {
            return None;
        }
        loop {
            self.buf.clear();
            match self.reader.read_line(&mut self.buf) {
                Ok(0) => {
                    self.done = true;
                    return None;
                }
                Ok(_) => {
                    let trimmed = self.buf.trim();
                    if trimmed.is_empty() {
                        continue; // skip blank lines
                    }
                    return Some(parse_delimited_row(trimmed, self.delimiter, self.ncols));
                }
                Err(_) => {
                    self.done = true;
                    return None;
                }
            }
        }
    }
}

// ── Minimal JSON value parser ────────────────────────────────────────

/// Minimal JSON value type for JSONL parsing.
/// Only supports the subset needed for tabular extraction.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    Str(String),
    /// Arrays and nested objects are stored as their raw JSON text.
    Raw(String),
}

impl JsonValue {
    /// Convert to a display string suitable for tabular cells.
    pub fn to_cell_string(&self) -> String {
        match self {
            JsonValue::Null => String::new(),
            JsonValue::Bool(b) => if *b { "true" } else { "false" }.to_string(),
            JsonValue::Number(n) => {
                // Render integers without decimal point.
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

/// Parse a single JSON object from a string. Returns key-value pairs in
/// deterministic (BTreeMap) order.
///
/// This is a minimal parser that handles the subset of JSON needed for
/// line-delimited JSON records. It does NOT attempt full JSON spec compliance;
/// it is sufficient for well-formed JSONL data files.
fn parse_json_object(input: &str) -> Result<BTreeMap<String, JsonValue>, String> {
    let input = input.trim();
    if !input.starts_with('{') {
        return Err("expected JSON object starting with '{'".to_string());
    }

    let bytes = input.as_bytes();
    let mut pos = 1; // skip '{'
    let mut map = BTreeMap::new();

    skip_ws(bytes, &mut pos);
    if pos < bytes.len() && bytes[pos] == b'}' {
        return Ok(map);
    }

    loop {
        skip_ws(bytes, &mut pos);
        // Parse key
        let key = parse_json_string(bytes, &mut pos)?;
        skip_ws(bytes, &mut pos);

        // Expect ':'
        if pos >= bytes.len() || bytes[pos] != b':' {
            return Err(format!("expected ':' at position {}", pos));
        }
        pos += 1;
        skip_ws(bytes, &mut pos);

        // Parse value
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
        return Err(format!("unexpected byte '{}' at position {}", bytes[pos] as char, pos));
    }

    Ok(map)
}

fn skip_ws(bytes: &[u8], pos: &mut usize) {
    while *pos < bytes.len() && matches!(bytes[*pos], b' ' | b'\t' | b'\n' | b'\r') {
        *pos += 1;
    }
}

fn parse_json_string(bytes: &[u8], pos: &mut usize) -> Result<String, String> {
    if *pos >= bytes.len() || bytes[*pos] != b'"' {
        return Err(format!("expected '\"' at position {}", *pos));
    }
    *pos += 1; // skip opening quote
    let mut s = String::new();
    while *pos < bytes.len() {
        let b = bytes[*pos];
        if b == b'\\' {
            *pos += 1;
            if *pos >= bytes.len() {
                return Err("unexpected end of string after backslash".to_string());
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
                    // \uXXXX — parse 4 hex digits
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
                        s.push('\u{FFFD}'); // replacement character
                    }
                    *pos += 3; // loop will advance by 1 more
                }
                other => {
                    s.push('\\');
                    s.push(other as char);
                }
            }
        } else if b == b'"' {
            *pos += 1; // skip closing quote
            return Ok(s);
        } else {
            s.push(b as char);
        }
        *pos += 1;
    }
    Err("unterminated string".to_string())
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
            // Capture raw JSON for nested structures.
            let start = *pos;
            skip_json_compound(bytes, pos)?;
            let raw = std::str::from_utf8(&bytes[start..*pos])
                .map_err(|_| "invalid UTF-8 in nested structure".to_string())?;
            Ok(JsonValue::Raw(raw.to_string()))
        }
        other => Err(format!("unexpected byte '{}' at position {}", other as char, *pos)),
    }
}

fn parse_json_number(bytes: &[u8], pos: &mut usize) -> Result<JsonValue, String> {
    let start = *pos;
    if *pos < bytes.len() && bytes[*pos] == b'-' {
        *pos += 1;
    }
    // Integer part
    while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
        *pos += 1;
    }
    // Fractional part
    if *pos < bytes.len() && bytes[*pos] == b'.' {
        *pos += 1;
        while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
            *pos += 1;
        }
    }
    // Exponent
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

/// Skip over a JSON array or object, tracking nested brackets/braces.
fn skip_json_compound(bytes: &[u8], pos: &mut usize) -> Result<(), String> {
    let open = bytes[*pos];
    let close = if open == b'{' { b'}' } else { b']' };
    let mut depth = 1;
    *pos += 1;
    while *pos < bytes.len() && depth > 0 {
        match bytes[*pos] {
            b'"' => {
                // Skip over string contents to avoid counting brackets inside strings.
                *pos += 1;
                while *pos < bytes.len() {
                    if bytes[*pos] == b'\\' {
                        *pos += 2; // skip escape sequence
                        continue;
                    }
                    if bytes[*pos] == b'"' {
                        *pos += 1;
                        break;
                    }
                    *pos += 1;
                }
                continue; // don't advance again
            }
            b if b == open => depth += 1,
            b if b == close => depth -= 1,
            _ => {}
        }
        *pos += 1;
    }
    if depth != 0 {
        return Err("unterminated compound JSON value".to_string());
    }
    Ok(())
}

// ── JSONL loading ────────────────────────────────────────────────────

/// Maximum number of rows scanned to discover the full set of column headers.
const JSONL_HEADER_SCAN_LIMIT: usize = 100;

/// Load JSONL content into `TabularData`.
///
/// Strategy:
/// 1. Parse up to `JSONL_HEADER_SCAN_LIMIT` rows to discover all unique keys.
/// 2. Key ordering is deterministic via `BTreeSet`.
/// 3. Re-iterate all rows, extracting values in header order.
/// 4. Missing keys produce empty strings.
pub fn load_jsonl(content: &str) -> TabularData {
    let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() {
        return TabularData {
            headers: Vec::new(),
            rows: Vec::new(),
            source_format: DataFormat::Jsonl,
        };
    }

    // Phase 1: discover headers from first N rows.
    let mut key_set = BTreeSet::new();
    let scan_limit = lines.len().min(JSONL_HEADER_SCAN_LIMIT);
    let mut parsed_objects: Vec<BTreeMap<String, JsonValue>> = Vec::with_capacity(lines.len());

    for line in &lines[..scan_limit] {
        if let Ok(obj) = parse_json_object(line) {
            for key in obj.keys() {
                key_set.insert(key.clone());
            }
            parsed_objects.push(obj);
        } else {
            // Skip malformed lines.
            parsed_objects.push(BTreeMap::new());
        }
    }

    // Parse remaining lines (beyond scan limit).
    for line in &lines[scan_limit..] {
        if let Ok(obj) = parse_json_object(line) {
            parsed_objects.push(obj);
        } else {
            parsed_objects.push(BTreeMap::new());
        }
    }

    let headers: Vec<String> = key_set.into_iter().collect();

    // Phase 2: extract rows in header order.
    let rows: Vec<Vec<String>> = parsed_objects
        .iter()
        .map(|obj| {
            headers
                .iter()
                .map(|h| {
                    obj.get(h)
                        .map(|v| v.to_cell_string())
                        .unwrap_or_default()
                })
                .collect()
        })
        .collect();

    TabularData {
        headers,
        rows,
        source_format: DataFormat::Jsonl,
    }
}

/// Streaming JSONL loader. Returns headers (from first N rows) and a row iterator.
///
/// Reads up to `JSONL_HEADER_SCAN_LIMIT` lines to discover headers, then yields
/// those buffered rows followed by remaining lines from the reader.
pub fn load_jsonl_streaming<R: BufRead>(
    reader: R,
) -> (Vec<String>, JsonlRowIter<R>) {
    let mut line_reader = reader;
    let mut key_set = BTreeSet::new();
    let mut buffered: Vec<BTreeMap<String, JsonValue>> = Vec::new();
    let mut buf = String::new();

    // Scan first N lines for headers.
    for _ in 0..JSONL_HEADER_SCAN_LIMIT {
        buf.clear();
        match line_reader.read_line(&mut buf) {
            Ok(0) => break,
            Ok(_) => {
                let trimmed = buf.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if let Ok(obj) = parse_json_object(trimmed) {
                    for key in obj.keys() {
                        key_set.insert(key.clone());
                    }
                    buffered.push(obj);
                } else {
                    buffered.push(BTreeMap::new());
                }
            }
            Err(_) => break,
        }
    }

    let headers: Vec<String> = key_set.into_iter().collect();
    let iter = JsonlRowIter {
        reader: line_reader,
        headers: headers.clone(),
        buffered,
        buf_idx: 0,
        line_buf: String::new(),
        done: false,
    };
    (headers, iter)
}

/// Streaming iterator for JSONL rows.
pub struct JsonlRowIter<R: BufRead> {
    reader: R,
    headers: Vec<String>,
    buffered: Vec<BTreeMap<String, JsonValue>>,
    buf_idx: usize,
    line_buf: String,
    done: bool,
}

impl<R: BufRead> JsonlRowIter<R> {
    fn obj_to_row(&self, obj: &BTreeMap<String, JsonValue>) -> Vec<String> {
        self.headers
            .iter()
            .map(|h| obj.get(h).map(|v| v.to_cell_string()).unwrap_or_default())
            .collect()
    }
}

impl<R: BufRead> Iterator for JsonlRowIter<R> {
    type Item = Vec<String>;

    fn next(&mut self) -> Option<Vec<String>> {
        // Drain buffered rows first.
        if self.buf_idx < self.buffered.len() {
            let row = self.obj_to_row(&self.buffered[self.buf_idx]);
            self.buf_idx += 1;
            return Some(row);
        }
        if self.done {
            return None;
        }
        // Read from underlying reader.
        loop {
            self.line_buf.clear();
            match self.reader.read_line(&mut self.line_buf) {
                Ok(0) => {
                    self.done = true;
                    return None;
                }
                Ok(_) => {
                    let trimmed = self.line_buf.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let obj = parse_json_object(trimmed).unwrap_or_default();
                    return Some(self.obj_to_row(&obj));
                }
                Err(_) => {
                    self.done = true;
                    return None;
                }
            }
        }
    }
}

// ── File metadata extraction ─────────────────────────────────────────

/// Metadata extracted from a data file without full parsing.
#[derive(Debug, Clone)]
pub struct FileMetadata {
    pub format: DataFormat,
    pub size: u64,
    /// Hex representation of the first magic bytes (if any).
    pub magic_bytes: Option<String>,
    /// Key-value metadata pairs. Deterministic ordering via BTreeMap.
    pub header_info: BTreeMap<String, String>,
    /// Whether the file can be safely parsed into TabularData.
    pub is_safe_to_parse: bool,
    /// Limitations or warnings about what we cannot do with this format.
    pub limitations: Vec<String>,
}

/// Extract metadata from a file without full materialization.
///
/// For text formats (CSV/TSV/JSONL), reports size and confirms parseability.
/// For binary formats, reads magic bytes and extracts safe header fields.
/// Model files (.pkl, .onnx, .joblib) are NEVER deserialized.
pub fn extract_metadata(path: &Path) -> FileMetadata {
    let format = detect_format(path);
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    let mut meta = FileMetadata {
        format,
        size,
        magic_bytes: None,
        header_info: BTreeMap::new(),
        is_safe_to_parse: format.is_materializable(),
        limitations: Vec::new(),
    };

    meta.header_info
        .insert("file_size_bytes".to_string(), size.to_string());

    // Read header bytes for binary inspection.
    let header = read_file_header(path, 128);

    if !header.is_empty() {
        let magic_len = match format {
            DataFormat::Parquet => 4,
            DataFormat::ArrowIpc => 6,
            DataFormat::Sqlite => 16,
            DataFormat::Pickle => 1,
            _ => header.len().min(8),
        };
        let display_len = magic_len.min(header.len());
        meta.magic_bytes = Some(bytes_to_hex(&header[..display_len]));
    }

    match format {
        DataFormat::Csv | DataFormat::Tsv => {
            meta.header_info
                .insert("type".to_string(), format.label().to_string());
        }
        DataFormat::Jsonl => {
            meta.header_info
                .insert("type".to_string(), "JSON Lines".to_string());
        }
        DataFormat::Parquet => {
            extract_parquet_metadata(path, &header, &mut meta);
        }
        DataFormat::ArrowIpc => {
            extract_arrow_metadata(&header, &mut meta);
        }
        DataFormat::Sqlite => {
            extract_sqlite_metadata(&header, &mut meta);
        }
        DataFormat::Pickle | DataFormat::Onnx | DataFormat::Joblib => {
            extract_model_metadata(path, &header, &mut meta);
        }
        DataFormat::Unknown => {
            meta.limitations
                .push("Unknown format: cannot determine file structure".to_string());
        }
    }

    // Compute a simple content hash (FNV-1a on first 4KB) for model files.
    if format.is_model_file() {
        let hash_header = read_file_header(path, 4096);
        if !hash_header.is_empty() {
            let hash = fnv1a_hash(&hash_header);
            meta.header_info
                .insert("header_hash_fnv1a".to_string(), format!("{:016x}", hash));
        }
    }

    meta
}

/// Read up to `max_bytes` from the start of a file.
fn read_file_header(path: &Path, max_bytes: usize) -> Vec<u8> {
    let mut buf = vec![0u8; max_bytes];
    if let Ok(mut f) = File::open(path) {
        if let Ok(n) = f.read(&mut buf) {
            buf.truncate(n);
            return buf;
        }
    }
    Vec::new()
}

/// Read up to `max_bytes` from the end of a file.
fn read_file_tail(path: &Path, max_bytes: usize) -> Vec<u8> {
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    if size == 0 {
        return Vec::new();
    }
    let offset = if size > max_bytes as u64 {
        size - max_bytes as u64
    } else {
        0
    };
    if let Ok(mut f) = File::open(path) {
        use std::io::Seek;
        if f.seek(io::SeekFrom::Start(offset)).is_ok() {
            let mut buf = vec![0u8; max_bytes];
            if let Ok(n) = f.read(&mut buf) {
                buf.truncate(n);
                return buf;
            }
        }
    }
    Vec::new()
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect::<Vec<_>>().join(" ")
}

/// FNV-1a 64-bit hash for content fingerprinting.
fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn extract_parquet_metadata(path: &Path, header: &[u8], meta: &mut FileMetadata) {
    meta.is_safe_to_parse = false;

    if header.len() >= 4 && &header[..4] == b"PAR1" {
        meta.header_info
            .insert("magic".to_string(), "PAR1 (valid Parquet)".to_string());
    }

    // Parquet footer: last 8 bytes = [4-byte footer length][PAR1 magic].
    let tail = read_file_tail(path, 8);
    if tail.len() >= 8 {
        let tail_magic = &tail[tail.len() - 4..];
        if tail_magic == b"PAR1" {
            let footer_len_bytes = &tail[tail.len() - 8..tail.len() - 4];
            let footer_len = u32::from_le_bytes([
                footer_len_bytes[0],
                footer_len_bytes[1],
                footer_len_bytes[2],
                footer_len_bytes[3],
            ]);
            meta.header_info
                .insert("footer_offset".to_string(), format!("{}", meta.size as u64 - 8 - footer_len as u64));
            meta.header_info
                .insert("footer_length_bytes".to_string(), footer_len.to_string());

            // Try to read the Thrift-encoded footer for row group count.
            // The footer is a serialized FileMetaData Thrift struct.
            // Row group count is at a known offset in simple files, but we
            // only report what we can safely extract without a Thrift parser.
            meta.header_info
                .insert("footer_encoding".to_string(), "Thrift (compact)".to_string());
        }
    }

    meta.limitations.push(
        "Parquet requires a Thrift decoder for full materialization; metadata-only inspection available"
            .to_string(),
    );
}

fn extract_arrow_metadata(header: &[u8], meta: &mut FileMetadata) {
    meta.is_safe_to_parse = false;

    if header.len() >= 6 && &header[..6] == b"ARROW1" {
        meta.header_info
            .insert("magic".to_string(), "ARROW1 (Arrow IPC)".to_string());
        // Arrow IPC v1 continuation: bytes 6-7 may contain padding or schema.
        if header.len() >= 10 {
            // Bytes 6..10 are a little-endian i32 for the schema message size
            // (or -1 for continuation indicator in IPC v2).
            let indicator = i32::from_le_bytes([
                header[6], header[7], header[8], header[9],
            ]);
            if indicator == -1 {
                meta.header_info
                    .insert("ipc_version".to_string(), "v2 (continuation)".to_string());
            } else {
                meta.header_info
                    .insert("ipc_version".to_string(), "v1".to_string());
                meta.header_info
                    .insert("schema_message_bytes".to_string(), indicator.to_string());
            }
        }
    }

    meta.limitations.push(
        "Arrow IPC requires FlatBuffers decoder for full materialization; metadata-only inspection available"
            .to_string(),
    );
}

fn extract_sqlite_metadata(header: &[u8], meta: &mut FileMetadata) {
    meta.is_safe_to_parse = false;

    if header.len() >= 18 && &header[..16] == b"SQLite format 3\0" {
        meta.header_info
            .insert("magic".to_string(), "SQLite format 3".to_string());
        // Bytes 16-17: page size in bytes (big-endian u16, 1 means 65536).
        let page_size_raw =
            u16::from_be_bytes([header[16], header[17]]);
        let page_size: u32 = if page_size_raw == 1 { 65536 } else { page_size_raw as u32 };
        meta.header_info
            .insert("page_size_bytes".to_string(), page_size.to_string());

        // Bytes 28-31: database size in pages (may be 0 if not set).
        if header.len() >= 32 {
            let db_pages = u32::from_be_bytes([header[28], header[29], header[30], header[31]]);
            if db_pages > 0 {
                meta.header_info
                    .insert("database_pages".to_string(), db_pages.to_string());
                meta.header_info.insert(
                    "estimated_size_bytes".to_string(),
                    (db_pages as u64 * page_size as u64).to_string(),
                );
            }
        }
    }

    meta.limitations.push(
        "SQLite requires a B-tree page parser for full materialization; metadata-only inspection available"
            .to_string(),
    );
}

fn extract_model_metadata(_path: &Path, header: &[u8], meta: &mut FileMetadata) {
    meta.is_safe_to_parse = false;

    match meta.format {
        DataFormat::Pickle => {
            if !header.is_empty() && header[0] == 0x80 {
                let protocol = if header.len() >= 2 { header[1] } else { 0 };
                meta.header_info
                    .insert("pickle_protocol".to_string(), protocol.to_string());
            }
            meta.limitations.push(
                "No deserialization \u{2014} model files are never executed. Only size, hash, and magic byte signature extracted."
                    .to_string(),
            );
        }
        DataFormat::Onnx => {
            // ONNX files are protobuf-encoded. We only report size/hash.
            meta.header_info
                .insert("encoding".to_string(), "Protobuf (ONNX ModelProto)".to_string());
            meta.limitations.push(
                "No deserialization \u{2014} model files are never executed. Only size, hash, and magic byte signature extracted."
                    .to_string(),
            );
        }
        DataFormat::Joblib => {
            // Joblib files may be zlib-compressed pickles or numpy arrays.
            meta.header_info
                .insert("encoding".to_string(), "Joblib (zlib/pickle)".to_string());
            meta.limitations.push(
                "No deserialization \u{2014} model files are never executed. Only size, hash, and magic byte signature extracted."
                    .to_string(),
            );
        }
        _ => {}
    }
}

// ── Format-aware loading ─────────────────────────────────────────────

/// Load tabular data from a file, auto-detecting format from extension.
///
/// For CSV/TSV, uses delimiter parsing. For JSONL, uses the custom JSON parser.
/// For binary formats (Parquet, Arrow, SQLite, Pickle, ONNX, Joblib), returns
/// an error explaining that only metadata inspection is available.
///
/// If `delimiter` is `Some`, it overrides the auto-detected delimiter for
/// CSV/TSV formats.
pub fn load_tabular(
    path: &Path,
    delimiter: Option<char>,
    has_header: bool,
) -> Result<TabularData, String> {
    let format = detect_format(path);

    match format {
        DataFormat::Csv | DataFormat::Tsv => {
            let content = std::fs::read_to_string(path)
                .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
            let delim = delimiter.unwrap_or(if format == DataFormat::Tsv { '\t' } else { ',' });
            Ok(load_delimited(&content, delim, has_header))
        }
        DataFormat::Jsonl => {
            let content = std::fs::read_to_string(path)
                .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
            Ok(load_jsonl(&content))
        }
        DataFormat::Parquet => Err(
            "Parquet files require a Thrift decoder for full materialization. \
             Use `extract_metadata()` for metadata-only inspection."
                .to_string(),
        ),
        DataFormat::ArrowIpc => Err(
            "Arrow IPC files require a FlatBuffers decoder for full materialization. \
             Use `extract_metadata()` for metadata-only inspection."
                .to_string(),
        ),
        DataFormat::Sqlite => Err(
            "SQLite files require a B-tree page parser for full materialization. \
             Use `extract_metadata()` for metadata-only inspection."
                .to_string(),
        ),
        DataFormat::Pickle | DataFormat::Onnx | DataFormat::Joblib => Err(format!(
            "{} files are never deserialized or executed. \
             Use `extract_metadata()` for safe metadata inspection only.",
            format.label()
        )),
        DataFormat::Unknown => {
            // Attempt CSV with provided or default delimiter as fallback.
            let content = std::fs::read_to_string(path)
                .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
            let delim = delimiter.unwrap_or(',');
            Ok(load_delimited(&content, delim, has_header))
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // ── Format detection ─────────────────────────────────────────

    #[test]
    fn detect_csv_extension() {
        assert_eq!(detect_from_extension(Path::new("data.csv")), DataFormat::Csv);
        assert_eq!(detect_from_extension(Path::new("DATA.CSV")), DataFormat::Csv);
    }

    #[test]
    fn detect_tsv_extension() {
        assert_eq!(detect_from_extension(Path::new("data.tsv")), DataFormat::Tsv);
    }

    #[test]
    fn detect_jsonl_extensions() {
        assert_eq!(detect_from_extension(Path::new("log.jsonl")), DataFormat::Jsonl);
        assert_eq!(detect_from_extension(Path::new("log.ndjson")), DataFormat::Jsonl);
    }

    #[test]
    fn detect_parquet_extension() {
        assert_eq!(detect_from_extension(Path::new("table.parquet")), DataFormat::Parquet);
    }

    #[test]
    fn detect_arrow_extensions() {
        assert_eq!(detect_from_extension(Path::new("data.feather")), DataFormat::ArrowIpc);
        assert_eq!(detect_from_extension(Path::new("data.arrow")), DataFormat::ArrowIpc);
        assert_eq!(detect_from_extension(Path::new("data.ipc")), DataFormat::ArrowIpc);
    }

    #[test]
    fn detect_sqlite_extensions() {
        assert_eq!(detect_from_extension(Path::new("app.sqlite")), DataFormat::Sqlite);
        assert_eq!(detect_from_extension(Path::new("app.db")), DataFormat::Sqlite);
        assert_eq!(detect_from_extension(Path::new("app.sqlite3")), DataFormat::Sqlite);
    }

    #[test]
    fn detect_model_extensions() {
        assert_eq!(detect_from_extension(Path::new("model.pkl")), DataFormat::Pickle);
        assert_eq!(detect_from_extension(Path::new("model.pickle")), DataFormat::Pickle);
        assert_eq!(detect_from_extension(Path::new("net.onnx")), DataFormat::Onnx);
        assert_eq!(detect_from_extension(Path::new("pipe.joblib")), DataFormat::Joblib);
    }

    #[test]
    fn detect_unknown_extension() {
        assert_eq!(detect_from_extension(Path::new("readme.txt")), DataFormat::Unknown);
        assert_eq!(detect_from_extension(Path::new("noext")), DataFormat::Unknown);
    }

    #[test]
    fn detect_magic_parquet() {
        assert_eq!(detect_from_magic(b"PAR1rest of file"), DataFormat::Parquet);
    }

    #[test]
    fn detect_magic_arrow() {
        assert_eq!(detect_from_magic(b"ARROW1\x00\x00"), DataFormat::ArrowIpc);
    }

    #[test]
    fn detect_magic_sqlite() {
        assert_eq!(
            detect_from_magic(b"SQLite format 3\0extra bytes here"),
            DataFormat::Sqlite
        );
    }

    #[test]
    fn detect_magic_pickle() {
        assert_eq!(detect_from_magic(&[0x80, 0x05]), DataFormat::Pickle);
    }

    #[test]
    fn detect_magic_no_match() {
        assert_eq!(detect_from_magic(b"hello world"), DataFormat::Unknown);
        assert_eq!(detect_from_magic(&[]), DataFormat::Unknown);
    }

    // ── CSV/TSV loading ──────────────────────────────────────────

    #[test]
    fn load_csv_with_header() {
        let csv = "name,age,city\nAlice,30,NYC\nBob,25,LA\n";
        let td = load_delimited(csv, ',', true);
        assert_eq!(td.headers, vec!["name", "age", "city"]);
        assert_eq!(td.nrows(), 2);
        assert_eq!(td.rows[0], vec!["Alice", "30", "NYC"]);
        assert_eq!(td.rows[1], vec!["Bob", "25", "LA"]);
        assert_eq!(td.source_format, DataFormat::Csv);
    }

    #[test]
    fn load_csv_without_header() {
        let csv = "Alice,30,NYC\nBob,25,LA\n";
        let td = load_delimited(csv, ',', false);
        assert_eq!(td.headers, vec!["col_0", "col_1", "col_2"]);
        assert_eq!(td.nrows(), 2);
        assert_eq!(td.rows[0], vec!["Alice", "30", "NYC"]);
    }

    #[test]
    fn load_tsv() {
        let tsv = "x\ty\n1\t2\n3\t4\n";
        let td = load_delimited(tsv, '\t', true);
        assert_eq!(td.headers, vec!["x", "y"]);
        assert_eq!(td.nrows(), 2);
        assert_eq!(td.source_format, DataFormat::Tsv);
    }

    #[test]
    fn load_csv_ragged_rows() {
        let csv = "a,b,c\n1,2\n4,5,6,7\n";
        let td = load_delimited(csv, ',', true);
        assert_eq!(td.ncols(), 3);
        // Short row is padded.
        assert_eq!(td.rows[0], vec!["1", "2", ""]);
        // Long row is truncated.
        assert_eq!(td.rows[1], vec!["4", "5", "6"]);
    }

    #[test]
    fn load_csv_empty() {
        let td = load_delimited("", ',', true);
        assert_eq!(td.headers.len(), 0);
        assert_eq!(td.nrows(), 0);
    }

    #[test]
    fn load_csv_skips_blank_lines() {
        let csv = "a,b\n\n1,2\n\n3,4\n";
        let td = load_delimited(csv, ',', true);
        assert_eq!(td.nrows(), 2);
    }

    #[test]
    fn column_accessor() {
        let csv = "x,y\n1,2\n3,4\n";
        let td = load_delimited(csv, ',', true);
        assert_eq!(td.column("x"), Some(vec!["1", "3"]));
        assert_eq!(td.column("y"), Some(vec!["2", "4"]));
        assert_eq!(td.column("z"), None);
    }

    // ── Streaming CSV ────────────────────────────────────────────

    #[test]
    fn streaming_csv_with_header() {
        let data = "name,val\nA,1\nB,2\nC,3\n";
        let cursor = io::Cursor::new(data);
        let (headers, iter) = load_delimited_streaming(cursor, ',', true);
        assert_eq!(headers, vec!["name", "val"]);
        let rows: Vec<Vec<String>> = iter.collect();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], vec!["A", "1"]);
    }

    #[test]
    fn streaming_csv_without_header() {
        let data = "A,1\nB,2\n";
        let cursor = io::Cursor::new(data);
        let (headers, iter) = load_delimited_streaming(cursor, ',', false);
        assert_eq!(headers, vec!["col_0", "col_1"]);
        let rows: Vec<Vec<String>> = iter.collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec!["A", "1"]);
    }

    // ── JSON parser ──────────────────────────────────────────────

    #[test]
    fn parse_simple_object() {
        let obj = parse_json_object(r#"{"a": 1, "b": "hello", "c": true, "d": null}"#).unwrap();
        assert_eq!(obj.get("a"), Some(&JsonValue::Number(1.0)));
        assert_eq!(obj.get("b"), Some(&JsonValue::Str("hello".to_string())));
        assert_eq!(obj.get("c"), Some(&JsonValue::Bool(true)));
        assert_eq!(obj.get("d"), Some(&JsonValue::Null));
    }

    #[test]
    fn parse_empty_object() {
        let obj = parse_json_object("{}").unwrap();
        assert!(obj.is_empty());
    }

    #[test]
    fn parse_string_escapes() {
        let obj = parse_json_object(r#"{"msg": "say \"hello\"\nworld"}"#).unwrap();
        assert_eq!(
            obj.get("msg"),
            Some(&JsonValue::Str("say \"hello\"\nworld".to_string()))
        );
    }

    #[test]
    fn parse_unicode_escape() {
        let obj = parse_json_object(r#"{"ch": "\u0041"}"#).unwrap();
        assert_eq!(obj.get("ch"), Some(&JsonValue::Str("A".to_string())));
    }

    #[test]
    fn parse_nested_object_as_raw() {
        let obj =
            parse_json_object(r#"{"name": "x", "meta": {"k": "v"}}"#).unwrap();
        assert_eq!(obj.get("name"), Some(&JsonValue::Str("x".to_string())));
        match obj.get("meta") {
            Some(JsonValue::Raw(s)) => assert_eq!(s, r#"{"k": "v"}"#),
            other => panic!("expected Raw, got {:?}", other),
        }
    }

    #[test]
    fn parse_nested_array_as_raw() {
        let obj = parse_json_object(r#"{"vals": [1, 2, 3]}"#).unwrap();
        match obj.get("vals") {
            Some(JsonValue::Raw(s)) => assert_eq!(s, "[1, 2, 3]"),
            other => panic!("expected Raw, got {:?}", other),
        }
    }

    #[test]
    fn parse_negative_and_float_numbers() {
        let obj = parse_json_object(r#"{"neg": -42, "pi": 3.14, "sci": 1.5e10}"#).unwrap();
        assert_eq!(obj.get("neg"), Some(&JsonValue::Number(-42.0)));
        assert_eq!(obj.get("pi"), Some(&JsonValue::Number(3.14)));
        assert_eq!(obj.get("sci"), Some(&JsonValue::Number(1.5e10)));
    }

    #[test]
    fn parse_error_on_non_object() {
        assert!(parse_json_object("[1,2,3]").is_err());
        assert!(parse_json_object("null").is_err());
        assert!(parse_json_object("").is_err());
    }

    #[test]
    fn json_value_cell_strings() {
        assert_eq!(JsonValue::Null.to_cell_string(), "");
        assert_eq!(JsonValue::Bool(true).to_cell_string(), "true");
        assert_eq!(JsonValue::Bool(false).to_cell_string(), "false");
        assert_eq!(JsonValue::Number(42.0).to_cell_string(), "42");
        assert_eq!(JsonValue::Number(3.14).to_cell_string(), "3.14");
        assert_eq!(JsonValue::Str("hi".to_string()).to_cell_string(), "hi");
        assert_eq!(
            JsonValue::Raw("[1,2]".to_string()).to_cell_string(),
            "[1,2]"
        );
    }

    // ── Key ordering is deterministic ────────────────────────────

    #[test]
    fn json_keys_are_btree_ordered() {
        let obj =
            parse_json_object(r#"{"z": 1, "a": 2, "m": 3}"#).unwrap();
        let keys: Vec<&String> = obj.keys().collect();
        assert_eq!(keys, vec!["a", "m", "z"]);
    }

    // ── JSONL loading ────────────────────────────────────────────

    #[test]
    fn load_jsonl_basic() {
        let jsonl = r#"{"name": "Alice", "age": 30}
{"name": "Bob", "age": 25}
{"name": "Carol", "age": 35}
"#;
        let td = load_jsonl(jsonl);
        assert_eq!(td.headers, vec!["age", "name"]); // BTreeSet order
        assert_eq!(td.nrows(), 3);
        assert_eq!(td.rows[0], vec!["30", "Alice"]);
        assert_eq!(td.rows[1], vec!["25", "Bob"]);
        assert_eq!(td.source_format, DataFormat::Jsonl);
    }

    #[test]
    fn load_jsonl_missing_keys() {
        let jsonl = r#"{"a": 1, "b": 2}
{"a": 3, "c": 4}
"#;
        let td = load_jsonl(jsonl);
        // Headers: a, b, c (BTreeSet order)
        assert_eq!(td.headers, vec!["a", "b", "c"]);
        // Row 0: a=1, b=2, c=(missing)
        assert_eq!(td.rows[0], vec!["1", "2", ""]);
        // Row 1: a=3, b=(missing), c=4
        assert_eq!(td.rows[1], vec!["3", "", "4"]);
    }

    #[test]
    fn load_jsonl_empty() {
        let td = load_jsonl("");
        assert!(td.headers.is_empty());
        assert_eq!(td.nrows(), 0);
    }

    #[test]
    fn load_jsonl_skips_blank_lines() {
        let jsonl = r#"{"x": 1}

{"x": 2}

"#;
        let td = load_jsonl(jsonl);
        assert_eq!(td.nrows(), 2);
    }

    #[test]
    fn load_jsonl_with_nested() {
        let jsonl = r#"{"id": 1, "tags": ["a", "b"], "meta": {"k": "v"}}"#;
        let td = load_jsonl(jsonl);
        assert_eq!(td.headers, vec!["id", "meta", "tags"]);
        assert_eq!(td.rows[0][0], "1");
        assert!(td.rows[0][1].starts_with('{'));
        assert!(td.rows[0][2].starts_with('['));
    }

    // ── Streaming JSONL ──────────────────────────────────────────

    #[test]
    fn streaming_jsonl() {
        let data = r#"{"x": 1, "y": "a"}
{"x": 2, "y": "b"}
{"x": 3, "y": "c"}
"#;
        let cursor = io::Cursor::new(data);
        let (headers, iter) = load_jsonl_streaming(cursor);
        assert_eq!(headers, vec!["x", "y"]);
        let rows: Vec<Vec<String>> = iter.collect();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], vec!["1", "a"]);
    }

    // ── Metadata extraction ──────────────────────────────────────

    #[test]
    fn metadata_unknown_format() {
        let meta = extract_metadata(Path::new("nonexistent.xyz"));
        assert_eq!(meta.format, DataFormat::Unknown);
        assert_eq!(meta.size, 0);
        assert!(!meta.is_safe_to_parse);
    }

    #[test]
    fn metadata_csv_format() {
        // CSV format is detected by extension; file need not exist for
        // the format check, but size will be 0.
        let meta = extract_metadata(Path::new("nonexistent.csv"));
        assert_eq!(meta.format, DataFormat::Csv);
        assert!(meta.is_safe_to_parse);
    }

    // ── Format properties ────────────────────────────────────────

    #[test]
    fn materializable_formats() {
        assert!(DataFormat::Csv.is_materializable());
        assert!(DataFormat::Tsv.is_materializable());
        assert!(DataFormat::Jsonl.is_materializable());
        assert!(!DataFormat::Parquet.is_materializable());
        assert!(!DataFormat::ArrowIpc.is_materializable());
        assert!(!DataFormat::Sqlite.is_materializable());
        assert!(!DataFormat::Pickle.is_materializable());
    }

    #[test]
    fn model_file_formats() {
        assert!(DataFormat::Pickle.is_model_file());
        assert!(DataFormat::Onnx.is_model_file());
        assert!(DataFormat::Joblib.is_model_file());
        assert!(!DataFormat::Csv.is_model_file());
        assert!(!DataFormat::Parquet.is_model_file());
    }

    // ── FNV-1a hash determinism ──────────────────────────────────

    #[test]
    fn fnv1a_deterministic() {
        let data = b"hello world";
        let h1 = fnv1a_hash(data);
        let h2 = fnv1a_hash(data);
        assert_eq!(h1, h2);
        // Different data produces different hash.
        assert_ne!(fnv1a_hash(b"hello world"), fnv1a_hash(b"hello worl!"));
    }

    // ── load_tabular errors for binary formats ───────────────────

    #[test]
    fn load_tabular_rejects_parquet() {
        let result = load_tabular(Path::new("data.parquet"), None, true);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Thrift"));
    }

    #[test]
    fn load_tabular_rejects_arrow() {
        let result = load_tabular(Path::new("data.arrow"), None, true);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("FlatBuffers"));
    }

    #[test]
    fn load_tabular_rejects_sqlite() {
        let result = load_tabular(Path::new("data.sqlite"), None, true);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("B-tree"));
    }

    #[test]
    fn load_tabular_rejects_model_files() {
        for ext in &["pkl", "onnx", "joblib"] {
            let path = format!("model.{}", ext);
            let result = load_tabular(Path::new(&path), None, true);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("never deserialized"));
        }
    }

    // ── Bytes-to-hex helper ──────────────────────────────────────

    #[test]
    fn bytes_to_hex_formatting() {
        assert_eq!(bytes_to_hex(&[0x50, 0x41, 0x52, 0x31]), "50 41 52 31");
        assert_eq!(bytes_to_hex(&[0x00, 0xff]), "00 ff");
        assert_eq!(bytes_to_hex(&[]), "");
    }

    // ── Strings with brackets inside JSON strings ────────────────

    #[test]
    fn json_string_with_brackets() {
        let obj = parse_json_object(r#"{"msg": "array [1,2] and {obj}"}"#).unwrap();
        assert_eq!(
            obj.get("msg"),
            Some(&JsonValue::Str("array [1,2] and {obj}".to_string()))
        );
    }

    // ── DataFormat label and ordering ────────────────────────────

    #[test]
    fn format_labels() {
        assert_eq!(DataFormat::Csv.label(), "CSV");
        assert_eq!(DataFormat::Parquet.label(), "Parquet");
        assert_eq!(DataFormat::Unknown.label(), "Unknown");
    }

    #[test]
    fn format_ordering_is_deterministic() {
        let mut formats = vec![
            DataFormat::Jsonl,
            DataFormat::Csv,
            DataFormat::Parquet,
            DataFormat::Tsv,
        ];
        let mut formats2 = formats.clone();
        formats.sort();
        formats2.sort();
        assert_eq!(formats, formats2);
    }
}
