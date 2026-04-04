//! `cjc patch` — Type-aware data transformation.
//!
//! Streams CSV/TSV/JSONL files and applies column-aware transformations:
//! - NaN replacement (--nan-fill <value>)
//! - Conditional replacement (--replace <col> <from> <to>)
//! - Mean imputation (--impute <col>)
//! - Column type casting (--cast <col> <type>)
//! - Column dropping (--drop <col>)
//! - Column renaming (--rename <old> <new>)
//! - Empty cell filling (--fill-empty <col> <value>)
//!
//! Supports CSV, TSV, and JSONL formats. Never corrupts schema.
//! Streams with O(ncols) memory.

use std::collections::BTreeMap;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process;

use crate::output::{self, OutputMode};

// ── Data format for patch ───────────────────────────────────────────

/// Detected input format for patch operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PatchFormat {
    /// Delimited text (CSV or TSV).
    Delimited,
    /// Line-delimited JSON.
    Jsonl,
}

/// Parsed arguments for `cjc patch`.
pub struct PatchArgs {
    pub file: Option<PathBuf>,
    pub output_file: Option<PathBuf>,
    pub delimiter: char,
    pub has_header: bool,
    pub transforms: Vec<Transform>,
    pub output_mode: OutputMode,
    pub dry_run: bool,
    pub plan: bool,
    pub backup: bool,
    pub in_place: bool,
    pub check: bool,
}

#[derive(Debug, Clone)]
pub enum Transform {
    /// Replace NaN/NA/null with a fixed value in all numeric columns.
    NanFill(String),
    /// Replace NaN/NA in a specific column with the column mean (two-pass).
    Impute(String),
    /// Replace exact value in a named column.
    Replace { column: String, from: String, to: String },
    /// Drop a named column.
    Drop(String),
    /// Rename a column.
    Rename { from: String, to: String },
    /// Fill empty cells in a column with a default.
    FillEmpty { column: String, value: String },
}

impl Transform {
    /// Human-readable description of this transform for --plan / --dry-run.
    fn describe(&self) -> String {
        match self {
            Transform::NanFill(v) => format!("nan-fill: replace NaN/NA/null/None with \"{}\" in all columns", v),
            Transform::Impute(col) => format!("impute: replace missing values in column \"{}\" with column mean", col),
            Transform::Replace { column, from, to } => {
                format!("replace: in column \"{}\", change \"{}\" to \"{}\"", column, from, to)
            }
            Transform::Drop(col) => format!("drop: remove column \"{}\"", col),
            Transform::Rename { from, to } => format!("rename: column \"{}\" -> \"{}\"", from, to),
            Transform::FillEmpty { column, value } => {
                format!("fill-empty: in column \"{}\", fill empty cells with \"{}\"", column, value)
            }
        }
    }
}

impl Default for PatchArgs {
    fn default() -> Self {
        Self {
            file: None,
            output_file: None,
            delimiter: ',',
            has_header: true,
            transforms: Vec::new(),
            output_mode: OutputMode::Color,
            dry_run: false,
            plan: false,
            backup: false,
            in_place: false,
            check: false,
        }
    }
}

pub fn parse_args(args: &[String]) -> PatchArgs {
    let mut pa = PatchArgs::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-d" | "--delimiter" => {
                i += 1;
                if i < args.len() {
                    pa.delimiter = match args[i].as_str() {
                        "tab" | "\\t" => '\t',
                        s if s.len() == 1 => s.chars().next().unwrap(),
                        _ => { eprintln!("error: delimiter must be a single character"); process::exit(1); }
                    };
                }
            }
            "--tsv" => pa.delimiter = '\t',
            "--no-header" => pa.has_header = false,
            "--nan-fill" => {
                i += 1;
                if i < args.len() {
                    pa.transforms.push(Transform::NanFill(args[i].clone()));
                }
            }
            "--impute" => {
                i += 1;
                if i < args.len() {
                    pa.transforms.push(Transform::Impute(args[i].clone()));
                }
            }
            "--replace" => {
                if i + 3 < args.len() {
                    let col = args[i + 1].clone();
                    let from = args[i + 2].clone();
                    let to = args[i + 3].clone();
                    pa.transforms.push(Transform::Replace { column: col, from, to });
                    i += 3;
                } else {
                    eprintln!("error: --replace requires <column> <from> <to>");
                    process::exit(1);
                }
            }
            "--drop" => {
                i += 1;
                if i < args.len() {
                    pa.transforms.push(Transform::Drop(args[i].clone()));
                }
            }
            "--rename" => {
                if i + 2 < args.len() {
                    let from = args[i + 1].clone();
                    let to = args[i + 2].clone();
                    pa.transforms.push(Transform::Rename { from, to });
                    i += 2;
                } else {
                    eprintln!("error: --rename requires <from> <to>");
                    process::exit(1);
                }
            }
            "--fill-empty" => {
                if i + 2 < args.len() {
                    let col = args[i + 1].clone();
                    let val = args[i + 2].clone();
                    pa.transforms.push(Transform::FillEmpty { column: col, value: val });
                    i += 2;
                } else {
                    eprintln!("error: --fill-empty requires <column> <value>");
                    process::exit(1);
                }
            }
            "-o" | "--output" => {
                i += 1;
                if i < args.len() {
                    pa.output_file = Some(PathBuf::from(&args[i]));
                }
            }
            "--plain" => pa.output_mode = OutputMode::Plain,
            "--json" => pa.output_mode = OutputMode::Json,
            "--dry-run" => pa.dry_run = true,
            "--plan" => pa.plan = true,
            "--backup" => pa.backup = true,
            "--in-place" => pa.in_place = true,
            "--check" => pa.check = true,
            other if !other.starts_with('-') => pa.file = Some(PathBuf::from(other)),
            other => {
                eprintln!("error: unknown flag `{}` for `cjc patch`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    pa
}

/// Detect whether the input file is JSONL based on extension.
fn detect_patch_format(path: &Option<PathBuf>) -> PatchFormat {
    match path {
        Some(p) => {
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            match ext {
                "jsonl" | "ndjson" => PatchFormat::Jsonl,
                _ => PatchFormat::Delimited,
            }
        }
        None => PatchFormat::Delimited, // stdin defaults to delimited
    }
}

/// Check whether a cell value is considered missing/NaN.
fn is_missing(val: &str) -> bool {
    let trimmed = val.trim();
    trimmed.is_empty() || trimmed == "NA" || trimmed == "NaN"
        || trimmed == "null" || trimmed == "None"
}

// ── Minimal JSON helpers for JSONL patching ─────────────────────────

/// Parse a JSON object string into ordered key-value pairs.
/// Returns keys in insertion order (preserved via Vec) and values as raw strings.
/// This is intentionally minimal: handles flat JSONL records with string, number,
/// bool, null values. Nested objects/arrays are preserved as raw strings.
fn parse_jsonl_object(input: &str) -> Result<Vec<(String, String)>, String> {
    let input = input.trim();
    if !input.starts_with('{') || !input.ends_with('}') {
        return Err("expected JSON object".to_string());
    }

    let bytes = input.as_bytes();
    let mut pos = 1; // skip '{'
    let mut pairs = Vec::new();

    skip_json_ws(bytes, &mut pos);
    if pos < bytes.len() && bytes[pos] == b'}' {
        return Ok(pairs);
    }

    loop {
        skip_json_ws(bytes, &mut pos);
        if pos >= bytes.len() { break; }

        // Parse key
        let key = parse_json_string(bytes, &mut pos)?;
        skip_json_ws(bytes, &mut pos);
        if pos >= bytes.len() || bytes[pos] != b':' {
            return Err("expected ':'".to_string());
        }
        pos += 1; // skip ':'
        skip_json_ws(bytes, &mut pos);

        // Parse value — capture the raw text
        let val_start = pos;
        skip_json_value(bytes, &mut pos)?;
        let raw_val = std::str::from_utf8(&bytes[val_start..pos])
            .map_err(|_| "invalid UTF-8 in value".to_string())?
            .trim()
            .to_string();

        pairs.push((key, raw_val));

        skip_json_ws(bytes, &mut pos);
        if pos < bytes.len() && bytes[pos] == b',' {
            pos += 1;
        } else {
            break;
        }
    }

    Ok(pairs)
}

/// Extract the unquoted string content of a JSON value.
/// For strings, strips quotes. For others (numbers, bools, null), returns as-is.
fn json_val_to_plain(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
        // Unescape the string
        let inner = &trimmed[1..trimmed.len() - 1];
        let mut out = String::with_capacity(inner.len());
        let mut chars = inner.chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some('/') => out.push('/'),
                    Some('n') => out.push('\n'),
                    Some('r') => out.push('\r'),
                    Some('t') => out.push('\t'),
                    Some(other) => { out.push('\\'); out.push(other); }
                    None => out.push('\\'),
                }
            } else {
                out.push(c);
            }
        }
        out
    } else if trimmed == "null" {
        String::new()
    } else {
        trimmed.to_string()
    }
}

/// Convert a plain string back to a JSON value representation.
/// If the original raw value was a quoted string, re-quote the result.
/// If it was a number/bool/null, emit as-is if still valid, else quote.
fn plain_to_json_val(plain: &str, original_raw: &str) -> String {
    let orig_trimmed = original_raw.trim();
    let was_string = orig_trimmed.starts_with('"');
    let was_null = orig_trimmed == "null";

    if plain.is_empty() && (was_null || was_string) {
        // If originally null and we replaced it, emit as string
        if was_null && !original_raw.trim().is_empty() {
            return json_escape_string(plain);
        }
        return json_escape_string(plain);
    }

    if was_string {
        // Re-quote as string
        json_escape_string(plain)
    } else {
        // Was a non-string type. If the replacement looks like a valid
        // JSON literal (number, bool, null), emit raw; otherwise quote it.
        if plain == "true" || plain == "false" || plain == "null" {
            plain.to_string()
        } else if plain.parse::<f64>().is_ok() {
            plain.to_string()
        } else {
            json_escape_string(plain)
        }
    }
}

/// Escape a string for JSON output.
fn json_escape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Reconstruct a JSON object from ordered key-value pairs (values are raw JSON).
fn rebuild_jsonl_object(pairs: &[(String, String)]) -> String {
    let mut out = String::from("{");
    for (i, (key, val)) in pairs.iter().enumerate() {
        if i > 0 { out.push_str(", "); }
        out.push_str(&json_escape_string(key));
        out.push_str(": ");
        out.push_str(val);
    }
    out.push('}');
    out
}

/// Write a JSONL object directly to a writer, avoiding intermediate String allocation.
fn write_jsonl_object<W: Write>(w: &mut W, pairs: &[(String, String)]) {
    let _ = w.write_all(b"{");
    for (i, (key, val)) in pairs.iter().enumerate() {
        if i > 0 { let _ = w.write_all(b", "); }
        let _ = write!(w, "\"{}\": {}", key.replace('\\', "\\\\").replace('"', "\\\""), val);
    }
    let _ = w.write_all(b"}");
}

fn skip_json_ws(bytes: &[u8], pos: &mut usize) {
    while *pos < bytes.len() && matches!(bytes[*pos], b' ' | b'\t' | b'\n' | b'\r') {
        *pos += 1;
    }
}

fn parse_json_string(bytes: &[u8], pos: &mut usize) -> Result<String, String> {
    if *pos >= bytes.len() || bytes[*pos] != b'"' {
        return Err("expected '\"'".to_string());
    }
    *pos += 1;
    let mut s = String::new();
    while *pos < bytes.len() {
        let b = bytes[*pos];
        if b == b'"' {
            *pos += 1;
            return Ok(s);
        }
        if b == b'\\' {
            *pos += 1;
            if *pos >= bytes.len() { return Err("unexpected end of string escape".to_string()); }
            match bytes[*pos] {
                b'"' => s.push('"'),
                b'\\' => s.push('\\'),
                b'/' => s.push('/'),
                b'n' => s.push('\n'),
                b'r' => s.push('\r'),
                b't' => s.push('\t'),
                b'u' => {
                    // \uXXXX — just push the raw escape for simplicity
                    s.push_str("\\u");
                    *pos += 1;
                    for _ in 0..4 {
                        if *pos < bytes.len() {
                            s.push(bytes[*pos] as char);
                            *pos += 1;
                        }
                    }
                    continue;
                }
                other => { s.push('\\'); s.push(other as char); }
            }
        } else {
            s.push(b as char);
        }
        *pos += 1;
    }
    Err("unterminated string".to_string())
}

/// Skip over a JSON value without fully parsing it. Advances `pos` past the value.
fn skip_json_value(bytes: &[u8], pos: &mut usize) -> Result<(), String> {
    skip_json_ws(bytes, pos);
    if *pos >= bytes.len() { return Err("unexpected end of input".to_string()); }

    match bytes[*pos] {
        b'"' => { let _ = parse_json_string(bytes, pos)?; Ok(()) }
        b'{' => skip_json_braced(bytes, pos, b'{', b'}'),
        b'[' => skip_json_braced(bytes, pos, b'[', b']'),
        b't' | b'f' | b'n' => {
            // true, false, null
            while *pos < bytes.len() && bytes[*pos].is_ascii_alphabetic() {
                *pos += 1;
            }
            Ok(())
        }
        b'-' | b'0'..=b'9' => {
            // number
            while *pos < bytes.len() && matches!(bytes[*pos], b'0'..=b'9' | b'.' | b'-' | b'+' | b'e' | b'E') {
                *pos += 1;
            }
            Ok(())
        }
        other => Err(format!("unexpected byte '{}' in JSON value", other as char)),
    }
}

fn skip_json_braced(bytes: &[u8], pos: &mut usize, open: u8, close: u8) -> Result<(), String> {
    let mut depth = 1;
    *pos += 1; // skip opening brace/bracket
    let mut in_string = false;
    while *pos < bytes.len() && depth > 0 {
        let b = bytes[*pos];
        if in_string {
            if b == b'\\' {
                *pos += 1; // skip escaped char
            } else if b == b'"' {
                in_string = false;
            }
        } else {
            if b == b'"' {
                in_string = true;
            } else if b == open {
                depth += 1;
            } else if b == close {
                depth -= 1;
            }
        }
        *pos += 1;
    }
    if depth != 0 { Err("unmatched braces".to_string()) } else { Ok(()) }
}

// ── Plan / dry-run output ───────────────────────────────────────────

fn print_plan(pa: &PatchArgs, format: PatchFormat) {
    let mode = pa.output_mode;
    let format_label = match format {
        PatchFormat::Delimited => {
            if pa.delimiter == '\t' { "TSV" } else { "CSV" }
        }
        PatchFormat::Jsonl => "JSONL",
    };

    if mode == OutputMode::Json {
        let transforms_json: Vec<String> = pa.transforms.iter().map(|t| {
            format!("\"{}\"", t.describe().replace('"', "\\\""))
        }).collect();
        let source = pa.file.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "<stdin>".to_string());
        println!("{{");
        println!("  \"command\": \"patch\",");
        println!("  \"source\": \"{}\",", source.replace('"', "\\\""));
        println!("  \"format\": \"{}\",", format_label);
        println!("  \"transforms\": [{}]", transforms_json.join(", "));
        println!("}}");
    } else {
        let header = output::colorize(mode, output::BOLD_CYAN, "Patch Plan");
        eprintln!("{}", header);
        let source = pa.file.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "<stdin>".to_string());
        eprintln!("  source: {}", source);
        eprintln!("  format: {}", format_label);
        eprintln!("  transforms ({}):", pa.transforms.len());
        for (idx, t) in pa.transforms.iter().enumerate() {
            let num = output::colorize(mode, output::DIM, &format!("  {}.", idx + 1));
            eprintln!("{} {}", num, t.describe());
        }
        if pa.dry_run {
            eprintln!();
            let note = output::colorize(mode, output::YELLOW, "(dry-run: no output will be written)");
            eprintln!("{}", note);
        }
    }
}

fn print_summary_json(rows_processed: u64, transforms_applied: usize, source: &str, dest: &str) {
    println!("{{");
    println!("  \"rows_processed\": {},", rows_processed);
    println!("  \"transforms_applied\": {},", transforms_applied);
    println!("  \"source\": \"{}\",", source.replace('"', "\\\""));
    println!("  \"destination\": \"{}\"", dest.replace('"', "\\\""));
    println!("}}");
}

// ── Backup helper ───────────────────────────────────────────────────

fn create_backup(path: &PathBuf) -> Result<PathBuf, String> {
    let mut bak = path.clone().into_os_string();
    bak.push(".bak");
    let bak_path = PathBuf::from(bak);
    fs::copy(path, &bak_path).map_err(|e| format!("could not create backup `{}`: {}", bak_path.display(), e))?;
    Ok(bak_path)
}

// ── Validation for --check mode ─────────────────────────────────────

/// Validate that transforms would apply cleanly. Returns true if clean, false if issues.
fn validate_transforms(pa: &PatchArgs, header_names: &[String]) -> bool {
    let mut clean = true;

    for t in &pa.transforms {
        match t {
            Transform::Impute(col) | Transform::Drop(col) => {
                if pa.has_header && !header_names.is_empty() && !header_names.contains(col) {
                    eprintln!("check: column \"{}\" not found in headers", col);
                    clean = false;
                }
            }
            Transform::Replace { column, .. } | Transform::FillEmpty { column, .. } => {
                if pa.has_header && !header_names.is_empty() && !header_names.contains(column) {
                    eprintln!("check: column \"{}\" not found in headers", column);
                    clean = false;
                }
            }
            Transform::Rename { from, .. } => {
                if pa.has_header && !header_names.is_empty() && !header_names.contains(from) {
                    eprintln!("check: column \"{}\" not found in headers for rename", from);
                    clean = false;
                }
            }
            Transform::NanFill(_) => { /* always applicable */ }
        }
    }

    clean
}

// ── Entry point ─────────────────────────────────────────────────────

/// Entry point for `cjc patch`.
pub fn run(args: &[String]) {
    let pa = parse_args(args);

    if pa.transforms.is_empty() {
        eprintln!("error: no transforms specified. Use --nan-fill, --impute, --replace, --drop, etc.");
        process::exit(1);
    }

    // --in-place requires a file argument
    if pa.in_place && pa.file.is_none() {
        eprintln!("error: --in-place requires a file argument (cannot modify stdin)");
        process::exit(1);
    }

    // --in-place and -o are mutually exclusive
    if pa.in_place && pa.output_file.is_some() {
        eprintln!("error: --in-place and --output are mutually exclusive");
        process::exit(1);
    }

    let format = detect_patch_format(&pa.file);

    // --plan: show structured plan and exit
    if pa.plan {
        print_plan(&pa, format);
        return;
    }

    // --dry-run: show plan (without executing) and exit
    if pa.dry_run {
        print_plan(&pa, format);
        return;
    }

    match format {
        PatchFormat::Delimited => run_delimited(&pa),
        PatchFormat::Jsonl => run_jsonl(&pa),
    }
}

// ── Delimited (CSV/TSV) path ────────────────────────────────────────

fn run_delimited(pa: &PatchArgs) {
    // Check if any transform requires two passes (e.g. --impute needs mean first)
    let needs_two_pass = pa.transforms.iter().any(|t| matches!(t, Transform::Impute(_)));

    let reader_fn = || -> Box<dyn BufRead> {
        match &pa.file {
            Some(path) => {
                let f = fs::File::open(path).unwrap_or_else(|e| {
                    eprintln!("error: could not open `{}`: {}", path.display(), e);
                    process::exit(1);
                });
                Box::new(io::BufReader::new(f))
            }
            None => Box::new(io::BufReader::new(io::stdin())),
        }
    };

    // First pass: compute column means for impute (if needed)
    let mut col_means: BTreeMap<String, (f64, f64, u64)> = BTreeMap::new();
    let mut header_names: Vec<String> = Vec::new();

    if needs_two_pass {
        if pa.file.is_none() {
            eprintln!("error: --impute requires a file argument (cannot two-pass stdin)");
            process::exit(1);
        }

        let reader = reader_fn();
        let mut lines = reader.lines();

        if pa.has_header {
            if let Some(Ok(hdr)) = lines.next() {
                header_names = hdr.split(pa.delimiter).map(|s| s.trim().to_string()).collect();
            }
        }

        // Initialize accumulators for impute columns
        for t in &pa.transforms {
            if let Transform::Impute(col) = t {
                col_means.insert(col.clone(), (0.0, 0.0, 0)); // sum, comp, count (Kahan)
            }
        }

        for line_result in lines {
            let line = match line_result { Ok(l) => l, Err(_) => break };
            let fields: Vec<&str> = line.split(pa.delimiter).collect();
            for (col_idx, field) in fields.iter().enumerate() {
                let col_name = header_names.get(col_idx).map(|s| s.as_str()).unwrap_or("");
                if let Some(acc) = col_means.get_mut(col_name) {
                    let trimmed = field.trim();
                    if let Ok(v) = trimmed.parse::<f64>() {
                        if !v.is_nan() {
                            // Kahan summation
                            let y = v - acc.1;
                            let t = acc.0 + y;
                            acc.1 = (t - acc.0) - y;
                            acc.0 = t;
                            acc.2 += 1;
                        }
                    }
                }
            }
        }
    }

    let impute_means: BTreeMap<String, String> = col_means.iter()
        .map(|(name, (sum, _, count))| {
            let mean = if *count > 0 { sum / *count as f64 } else { 0.0 };
            (name.clone(), format!("{}", mean))
        })
        .collect();

    // --check mode: validate and exit
    if pa.check {
        // Read header for validation
        if header_names.is_empty() && pa.has_header {
            let reader = reader_fn();
            let mut lines = reader.lines();
            if let Some(Ok(hdr)) = lines.next() {
                header_names = hdr.split(pa.delimiter).map(|s| s.trim().to_string()).collect();
            }
        }
        let clean = validate_transforms(pa, &header_names);
        if clean {
            process::exit(0);
        } else {
            process::exit(1);
        }
    }

    // --backup: create backup before writing
    if pa.backup {
        if let Some(ref path) = pa.file {
            if pa.in_place || pa.output_file.as_ref() == Some(path) {
                match create_backup(path) {
                    Ok(bak) => eprintln!("backup: {}", bak.display()),
                    Err(e) => { eprintln!("error: {}", e); process::exit(1); }
                }
            }
        }
    }

    // Determine output destination
    let effective_output: Option<PathBuf> = if pa.in_place {
        // Write to a temp file, then rename over the original
        pa.file.clone()
    } else {
        pa.output_file.clone()
    };

    // For --in-place, write to a temp file first
    let (temp_path, actual_output): (Option<PathBuf>, Option<PathBuf>) = if pa.in_place {
        let src = pa.file.as_ref().unwrap();
        let mut tmp = src.clone().into_os_string();
        tmp.push(".cjc_patch_tmp");
        let tmp_path = PathBuf::from(tmp);
        (Some(tmp_path.clone()), Some(tmp_path))
    } else {
        (None, effective_output)
    };

    // Second (or only) pass: apply transforms and write output
    let reader = reader_fn();
    let mut lines = reader.lines();

    let out_writer: Box<dyn Write> = match &actual_output {
        Some(path) => {
            let f = fs::File::create(path).unwrap_or_else(|e| {
                eprintln!("error: could not create `{}`: {}", path.display(), e);
                process::exit(1);
            });
            Box::new(io::BufWriter::new(f))
        }
        None => Box::new(io::BufWriter::new(io::stdout())),
    };
    let mut out = out_writer;

    // Determine which columns to drop
    let drop_cols: Vec<String> = pa.transforms.iter().filter_map(|t| {
        if let Transform::Drop(col) = t { Some(col.clone()) } else { None }
    }).collect();

    // Process header
    if pa.has_header {
        if let Some(Ok(hdr)) = lines.next() {
            if header_names.is_empty() {
                header_names = hdr.split(pa.delimiter).map(|s| s.trim().to_string()).collect();
            }

            // Apply renames
            let mut output_names = header_names.clone();
            for t in &pa.transforms {
                if let Transform::Rename { from, to } = t {
                    for name in &mut output_names {
                        if name == from { *name = to.clone(); }
                    }
                }
            }

            // Drop columns from header
            let keep_indices: Vec<usize> = (0..output_names.len())
                .filter(|i| !drop_cols.contains(&header_names[*i]))
                .collect();

            let header_out: Vec<&str> = keep_indices.iter().map(|&i| output_names[i].as_str()).collect();
            let _ = writeln!(out, "{}", header_out.join(&pa.delimiter.to_string()));
        }
    }

    // Build keep_indices for column dropping
    let keep_indices: Vec<usize> = (0..header_names.len())
        .filter(|i| !drop_cols.contains(&header_names[*i]))
        .collect();

    let mut rows_processed = 0u64;

    for line_result in lines {
        let line = match line_result { Ok(l) => l, Err(_) => break };
        if line.trim().is_empty() { continue; }

        let fields: Vec<String> = line.split(pa.delimiter).map(|s| s.to_string()).collect();
        let mut transformed = fields.clone();

        // Apply transforms per field
        for (col_idx, field) in transformed.iter_mut().enumerate() {
            let col_name = header_names.get(col_idx).map(|s| s.as_str()).unwrap_or("");

            for t in &pa.transforms {
                match t {
                    Transform::NanFill(fill) => {
                        if is_missing(field) {
                            *field = fill.clone();
                        }
                    }
                    Transform::Impute(target_col) if col_name == target_col => {
                        if is_missing(field) {
                            if let Some(mean_val) = impute_means.get(target_col) {
                                *field = mean_val.clone();
                            }
                        }
                    }
                    Transform::Replace { column, from, to } if col_name == column => {
                        if field.trim() == from {
                            *field = to.clone();
                        }
                    }
                    Transform::FillEmpty { column, value } if col_name == column => {
                        if field.trim().is_empty() {
                            *field = value.clone();
                        }
                    }
                    _ => {}
                }
            }
        }

        // Filter to kept columns
        let output_fields: Vec<&str> = keep_indices.iter()
            .map(|&i| transformed.get(i).map(|s| s.as_str()).unwrap_or(""))
            .collect();
        let _ = writeln!(out, "{}", output_fields.join(&pa.delimiter.to_string()));
        rows_processed += 1;
    }

    let _ = out.flush();
    drop(out);

    // --in-place: rename temp file over the original
    if pa.in_place {
        if let (Some(tmp), Some(src)) = (&temp_path, &pa.file) {
            if let Err(e) = fs::rename(tmp, src) {
                // Fallback: copy + remove (rename fails across filesystems)
                if let Err(e2) = fs::copy(tmp, src) {
                    eprintln!("error: could not write in-place `{}`: {} (rename: {})", src.display(), e2, e);
                    process::exit(1);
                }
                let _ = fs::remove_file(tmp);
            }
        }
    }

    // Status output
    let source_label = pa.file.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "<stdin>".to_string());
    let dest_label = if pa.in_place {
        pa.file.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "<stdout>".to_string())
    } else {
        pa.output_file.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "<stdout>".to_string())
    };

    match pa.output_mode {
        OutputMode::Json => {
            print_summary_json(rows_processed, pa.transforms.len(), &source_label, &dest_label);
        }
        _ => {
            eprintln!("patched {} rows", rows_processed);
        }
    }
}

// ── JSONL path ──────────────────────────────────────────────────────

fn run_jsonl(pa: &PatchArgs) {
    let needs_two_pass = pa.transforms.iter().any(|t| matches!(t, Transform::Impute(_)));

    let reader_fn = || -> Box<dyn BufRead> {
        match &pa.file {
            Some(path) => {
                let f = fs::File::open(path).unwrap_or_else(|e| {
                    eprintln!("error: could not open `{}`: {}", path.display(), e);
                    process::exit(1);
                });
                Box::new(io::BufReader::new(f))
            }
            None => Box::new(io::BufReader::new(io::stdin())),
        }
    };

    // Discover field names from first pass (or from scanning)
    let mut all_keys: Vec<String> = Vec::new();
    let mut col_means: BTreeMap<String, (f64, f64, u64)> = BTreeMap::new();

    // First pass: discover keys and compute impute means
    if needs_two_pass {
        if pa.file.is_none() {
            eprintln!("error: --impute requires a file argument (cannot two-pass stdin)");
            process::exit(1);
        }

        for t in &pa.transforms {
            if let Transform::Impute(col) = t {
                col_means.insert(col.clone(), (0.0, 0.0, 0));
            }
        }

        let reader = reader_fn();
        let mut key_set = std::collections::BTreeSet::new();
        for line_result in reader.lines() {
            let line = match line_result { Ok(l) => l, Err(_) => break };
            let trimmed = line.trim();
            if trimmed.is_empty() { continue; }

            if let Ok(pairs) = parse_jsonl_object(trimmed) {
                for (key, raw_val) in &pairs {
                    key_set.insert(key.clone());
                    if let Some(acc) = col_means.get_mut(key) {
                        let plain = json_val_to_plain(raw_val);
                        if let Ok(v) = plain.parse::<f64>() {
                            if !v.is_nan() {
                                // Kahan summation
                                let y = v - acc.1;
                                let t = acc.0 + y;
                                acc.1 = (t - acc.0) - y;
                                acc.0 = t;
                                acc.2 += 1;
                            }
                        }
                    }
                }
            }
        }
        all_keys = key_set.into_iter().collect();
    }

    let impute_means: BTreeMap<String, String> = col_means.iter()
        .map(|(name, (sum, _, count))| {
            let mean = if *count > 0 { sum / *count as f64 } else { 0.0 };
            (name.clone(), format!("{}", mean))
        })
        .collect();

    // --check mode for JSONL
    if pa.check {
        // For JSONL, we just validate that impute columns exist if we did a first pass
        let clean = if !all_keys.is_empty() {
            validate_transforms(pa, &all_keys)
        } else {
            // Quick scan for field names
            if let Some(ref path) = pa.file {
                let reader = reader_fn();
                let mut key_set = std::collections::BTreeSet::new();
                for line_result in reader.lines().take(100) {
                    let line = match line_result { Ok(l) => l, Err(_) => break };
                    if let Ok(pairs) = parse_jsonl_object(line.trim()) {
                        for (k, _) in pairs { key_set.insert(k); }
                    }
                }
                let keys: Vec<String> = key_set.into_iter().collect();
                validate_transforms(pa, &keys)
            } else {
                true // cannot validate stdin
            }
        };
        if clean { process::exit(0); } else { process::exit(1); }
    }

    // --backup
    if pa.backup {
        if let Some(ref path) = pa.file {
            if pa.in_place || pa.output_file.as_ref() == Some(path) {
                match create_backup(path) {
                    Ok(bak) => eprintln!("backup: {}", bak.display()),
                    Err(e) => { eprintln!("error: {}", e); process::exit(1); }
                }
            }
        }
    }

    // Determine output destination
    let (temp_path, actual_output): (Option<PathBuf>, Option<PathBuf>) = if pa.in_place {
        let src = pa.file.as_ref().unwrap();
        let mut tmp = src.clone().into_os_string();
        tmp.push(".cjc_patch_tmp");
        let tmp_path = PathBuf::from(tmp);
        (Some(tmp_path.clone()), Some(tmp_path))
    } else {
        (None, pa.output_file.clone())
    };

    // Second (or only) pass: apply transforms
    let reader = reader_fn();
    let out_writer: Box<dyn Write> = match &actual_output {
        Some(path) => {
            let f = fs::File::create(path).unwrap_or_else(|e| {
                eprintln!("error: could not create `{}`: {}", path.display(), e);
                process::exit(1);
            });
            Box::new(io::BufWriter::new(f))
        }
        None => Box::new(io::BufWriter::new(io::stdout())),
    };
    let mut out = out_writer;

    let drop_cols: Vec<String> = pa.transforms.iter().filter_map(|t| {
        if let Transform::Drop(col) = t { Some(col.clone()) } else { None }
    }).collect();

    let mut rows_processed = 0u64;

    for line_result in reader.lines() {
        let line = match line_result { Ok(l) => l, Err(_) => break };
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }

        let pairs = match parse_jsonl_object(trimmed) {
            Ok(p) => p,
            Err(_) => {
                // Pass through malformed lines unchanged
                let _ = writeln!(out, "{}", trimmed);
                rows_processed += 1;
                continue;
            }
        };

        // Build a mutable list of (key, raw_json_value) pairs
        let mut result_pairs: Vec<(String, String)> = Vec::with_capacity(pairs.len());

        for (key, raw_val) in &pairs {
            // Skip dropped columns
            if drop_cols.contains(key) { continue; }

            let plain = json_val_to_plain(raw_val);
            let mut current_plain = plain.clone();

            // Apply transforms
            for t in &pa.transforms {
                match t {
                    Transform::NanFill(fill) => {
                        if is_missing(&current_plain) {
                            current_plain = fill.clone();
                        }
                    }
                    Transform::Impute(target_col) if key == target_col => {
                        if is_missing(&current_plain) {
                            if let Some(mean_val) = impute_means.get(target_col) {
                                current_plain = mean_val.clone();
                            }
                        }
                    }
                    Transform::Replace { column, from, to } if key == column => {
                        if current_plain.trim() == from {
                            current_plain = to.clone();
                        }
                    }
                    Transform::FillEmpty { column, value } if key == column => {
                        if current_plain.trim().is_empty() {
                            current_plain = value.clone();
                        }
                    }
                    _ => {}
                }
            }

            // Apply renames
            let mut output_key = key.clone();
            for t in &pa.transforms {
                if let Transform::Rename { from, to } = t {
                    if &output_key == from {
                        output_key = to.clone();
                    }
                }
            }

            // Convert back to JSON value
            let output_val = if current_plain == plain {
                // Unchanged — preserve original raw value
                raw_val.clone()
            } else {
                plain_to_json_val(&current_plain, raw_val)
            };

            result_pairs.push((output_key, output_val));
        }

        // Write directly to output buffer instead of building intermediate String
        write_jsonl_object(&mut out, &result_pairs);
        let _ = out.write_all(b"\n");
        rows_processed += 1;
    }

    let _ = out.flush();
    drop(out);

    // --in-place: rename temp file over the original
    if pa.in_place {
        if let (Some(tmp), Some(src)) = (&temp_path, &pa.file) {
            if let Err(e) = fs::rename(tmp, src) {
                if let Err(e2) = fs::copy(tmp, src) {
                    eprintln!("error: could not write in-place `{}`: {} (rename: {})", src.display(), e2, e);
                    process::exit(1);
                }
                let _ = fs::remove_file(tmp);
            }
        }
    }

    // Status output
    let source_label = pa.file.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "<stdin>".to_string());
    let dest_label = if pa.in_place {
        pa.file.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "<stdout>".to_string())
    } else {
        pa.output_file.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "<stdout>".to_string())
    };

    match pa.output_mode {
        OutputMode::Json => {
            print_summary_json(rows_processed, pa.transforms.len(), &source_label, &dest_label);
        }
        _ => {
            eprintln!("patched {} rows", rows_processed);
        }
    }
}

// ── Help ────────────────────────────────────────────────────────────

pub fn print_help() {
    eprintln!("cjc patch -- Type-aware data transformation");
    eprintln!();
    eprintln!("Usage: cjc patch <file> [transforms] [flags]");
    eprintln!("       cat data.csv | cjc patch [transforms] [flags]");
    eprintln!();
    eprintln!("Supported formats: CSV, TSV, JSONL (detected by extension)");
    eprintln!();
    eprintln!("Transforms:");
    eprintln!("  --nan-fill <value>               Replace NaN/NA/null with <value>");
    eprintln!("  --impute <column>                Replace NaN with column mean (requires file)");
    eprintln!("  --replace <col> <from> <to>      Replace exact values in column");
    eprintln!("  --drop <column>                  Remove a column");
    eprintln!("  --rename <old> <new>             Rename a column");
    eprintln!("  --fill-empty <col> <value>       Fill empty cells in column");
    eprintln!();
    eprintln!("Output flags:");
    eprintln!("  -o, --output <file>              Output file (default: stdout)");
    eprintln!("  --in-place                       Modify input file directly (requires file arg)");
    eprintln!("  --backup                         Save a .bak copy before overwriting");
    eprintln!("  --plain                          Plain text status (no colors)");
    eprintln!("  --json                           JSON summary output");
    eprintln!("  --dry-run                        Show what would be applied without writing");
    eprintln!("  --plan                           Show structured transform plan");
    eprintln!("  --check                          Validate transforms apply cleanly (exit 0/1)");
    eprintln!();
    eprintln!("Format flags:");
    eprintln!("  -d, --delimiter <c>              Column delimiter (default: ,)");
    eprintln!("  --tsv                            Use tab delimiter");
    eprintln!("  --no-header                      Input has no header (CSV/TSV only)");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_args_basic() {
        let args: Vec<String> = vec![
            "data.csv".into(), "--nan-fill".into(), "0".into(),
        ];
        let pa = parse_args(&args);
        assert!(pa.file.is_some());
        assert_eq!(pa.transforms.len(), 1);
    }

    #[test]
    fn test_parse_args_output_flags() {
        let args: Vec<String> = vec![
            "data.csv".into(), "--nan-fill".into(), "0".into(),
            "--plain".into(), "--dry-run".into(), "--backup".into(),
        ];
        let pa = parse_args(&args);
        assert_eq!(pa.output_mode, OutputMode::Plain);
        assert!(pa.dry_run);
        assert!(pa.backup);
    }

    #[test]
    fn test_parse_args_json_mode() {
        let args: Vec<String> = vec![
            "data.csv".into(), "--nan-fill".into(), "0".into(), "--json".into(),
        ];
        let pa = parse_args(&args);
        assert_eq!(pa.output_mode, OutputMode::Json);
    }

    #[test]
    fn test_parse_args_in_place() {
        let args: Vec<String> = vec![
            "data.csv".into(), "--nan-fill".into(), "0".into(), "--in-place".into(),
        ];
        let pa = parse_args(&args);
        assert!(pa.in_place);
    }

    #[test]
    fn test_parse_args_check() {
        let args: Vec<String> = vec![
            "data.csv".into(), "--nan-fill".into(), "0".into(), "--check".into(),
        ];
        let pa = parse_args(&args);
        assert!(pa.check);
    }

    #[test]
    fn test_detect_format_csv() {
        assert_eq!(detect_patch_format(&Some(PathBuf::from("data.csv"))), PatchFormat::Delimited);
        assert_eq!(detect_patch_format(&Some(PathBuf::from("data.tsv"))), PatchFormat::Delimited);
    }

    #[test]
    fn test_detect_format_jsonl() {
        assert_eq!(detect_patch_format(&Some(PathBuf::from("data.jsonl"))), PatchFormat::Jsonl);
        assert_eq!(detect_patch_format(&Some(PathBuf::from("data.ndjson"))), PatchFormat::Jsonl);
    }

    #[test]
    fn test_detect_format_stdin() {
        assert_eq!(detect_patch_format(&None), PatchFormat::Delimited);
    }

    #[test]
    fn test_is_missing() {
        assert!(is_missing(""));
        assert!(is_missing("NA"));
        assert!(is_missing("NaN"));
        assert!(is_missing("null"));
        assert!(is_missing("None"));
        assert!(is_missing("  NaN  "));
        assert!(!is_missing("42"));
        assert!(!is_missing("hello"));
    }

    #[test]
    fn test_parse_jsonl_object_simple() {
        let pairs = parse_jsonl_object(r#"{"a": 1, "b": "hello", "c": true}"#).unwrap();
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[0].0, "a");
        assert_eq!(pairs[0].1, "1");
        assert_eq!(pairs[1].0, "b");
        assert_eq!(pairs[1].1, "\"hello\"");
        assert_eq!(pairs[2].0, "c");
        assert_eq!(pairs[2].1, "true");
    }

    #[test]
    fn test_parse_jsonl_object_null() {
        let pairs = parse_jsonl_object(r#"{"x": null}"#).unwrap();
        assert_eq!(pairs[0].1, "null");
    }

    #[test]
    fn test_parse_jsonl_object_empty() {
        let pairs = parse_jsonl_object("{}").unwrap();
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_json_val_to_plain() {
        assert_eq!(json_val_to_plain("\"hello\""), "hello");
        assert_eq!(json_val_to_plain("42"), "42");
        assert_eq!(json_val_to_plain("true"), "true");
        assert_eq!(json_val_to_plain("null"), "");
        assert_eq!(json_val_to_plain("3.14"), "3.14");
    }

    #[test]
    fn test_plain_to_json_val_string() {
        assert_eq!(plain_to_json_val("world", "\"hello\""), "\"world\"");
    }

    #[test]
    fn test_plain_to_json_val_number() {
        assert_eq!(plain_to_json_val("99", "42"), "99");
    }

    #[test]
    fn test_plain_to_json_val_from_null() {
        // Replacing a null with a numeric string
        assert_eq!(plain_to_json_val("3.14", "null"), "3.14");
    }

    #[test]
    fn test_rebuild_jsonl_object() {
        let pairs = vec![
            ("name".to_string(), "\"Alice\"".to_string()),
            ("age".to_string(), "30".to_string()),
        ];
        let result = rebuild_jsonl_object(&pairs);
        assert_eq!(result, r#"{"name": "Alice", "age": 30}"#);
    }

    #[test]
    fn test_json_escape_string() {
        assert_eq!(json_escape_string("hello"), "\"hello\"");
        assert_eq!(json_escape_string("say \"hi\""), r#""say \"hi\"""#);
        assert_eq!(json_escape_string("a\nb"), r#""a\nb""#);
    }

    #[test]
    fn test_transform_describe() {
        let t = Transform::NanFill("0".to_string());
        assert!(t.describe().contains("nan-fill"));

        let t = Transform::Drop("col1".to_string());
        assert!(t.describe().contains("drop"));
        assert!(t.describe().contains("col1"));
    }

    #[test]
    fn test_validate_transforms_clean() {
        let pa = PatchArgs {
            has_header: true,
            transforms: vec![Transform::Drop("age".to_string())],
            ..PatchArgs::default()
        };
        let headers = vec!["name".to_string(), "age".to_string()];
        assert!(validate_transforms(&pa, &headers));
    }

    #[test]
    fn test_validate_transforms_missing_col() {
        let pa = PatchArgs {
            has_header: true,
            transforms: vec![Transform::Drop("nonexistent".to_string())],
            ..PatchArgs::default()
        };
        let headers = vec!["name".to_string(), "age".to_string()];
        assert!(!validate_transforms(&pa, &headers));
    }
}
