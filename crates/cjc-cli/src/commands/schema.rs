//! `cjcl schema` — Schema inference and reporting for datasets.
//!
//! Infers column types, null counts, unique value counts, and basic statistics
//! from CSV/TSV data. Output is always deterministic with stable column ordering.
//!
//! Supports:
//! - CSV/TSV: full schema inference (original code path)
//! - JSONL/NDJSON: schema inference via `crate::formats::load_jsonl`
//! - Parquet/Arrow/SQLite: metadata-only display via `crate::formats::extract_metadata`
//!
//! Second-mode flags:
//! - `--save <schema-file>`: save inferred schema to a JSON file
//! - `--check <schema-file>`: validate data against a saved schema, exit 1 on mismatch
//! - `--diff <schema-file>`: compare current schema vs saved schema, show differences
//! - `--strict`: treat warnings as errors (type mismatches in --check)
//! - `--full`: show type distribution percentages and sample values

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::process;
use crate::output::{self, OutputMode};

// ── Arguments ───────────────────────────────────────────────────────

/// Second-mode operation for schema comparison/validation.
#[derive(Debug, Clone, PartialEq, Eq)]
enum SchemaMode {
    /// Default: just display the inferred schema.
    Infer,
    /// Save inferred schema to a JSON file.
    Save(String),
    /// Check data against a saved schema file; exit 1 on mismatch.
    Check(String),
    /// Diff current schema against a saved schema file.
    Diff(String),
}

pub struct SchemaArgs {
    pub file: String,
    pub delimiter: char,
    pub has_header: bool,
    pub sample_rows: Option<usize>,
    pub output: OutputMode,
    pub show_uniques: bool,
    pub strict: bool,
    pub full: bool,
    mode: SchemaMode,
}

impl Default for SchemaArgs {
    fn default() -> Self {
        Self {
            file: String::new(),
            delimiter: ',',
            has_header: true,
            sample_rows: None,
            output: OutputMode::Color,
            show_uniques: true,
            strict: false,
            full: false,
            mode: SchemaMode::Infer,
        }
    }
}

pub fn parse_args(args: &[String]) -> SchemaArgs {
    let mut sa = SchemaArgs::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-d" | "--delimiter" => {
                i += 1;
                if i < args.len() {
                    sa.delimiter = match args[i].as_str() {
                        "tab" | "\\t" => '\t',
                        s if s.len() == 1 => s.chars().next().unwrap(),
                        _ => ',',
                    };
                }
            }
            "--tsv" => sa.delimiter = '\t',
            "--no-header" => sa.has_header = false,
            "--no-uniques" => sa.show_uniques = false,
            "--sample" => {
                i += 1;
                if i < args.len() { sa.sample_rows = args[i].parse().ok(); }
            }
            "--plain" => sa.output = OutputMode::Plain,
            "--json" => sa.output = OutputMode::Json,
            "--color" => sa.output = OutputMode::Color,
            "--strict" => sa.strict = true,
            "--full" => sa.full = true,
            "--save" => {
                i += 1;
                if i < args.len() {
                    sa.mode = SchemaMode::Save(args[i].clone());
                } else {
                    eprintln!("error: --save requires a file argument");
                    process::exit(1);
                }
            }
            "--check" => {
                i += 1;
                if i < args.len() {
                    sa.mode = SchemaMode::Check(args[i].clone());
                } else {
                    eprintln!("error: --check requires a file argument");
                    process::exit(1);
                }
            }
            "--diff" => {
                i += 1;
                if i < args.len() {
                    sa.mode = SchemaMode::Diff(args[i].clone());
                } else {
                    eprintln!("error: --diff requires a file argument");
                    process::exit(1);
                }
            }
            other if !other.starts_with('-') => sa.file = other.to_string(),
            other => {
                eprintln!("error: unknown flag `{}` for `cjcl schema`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    if sa.file.is_empty() {
        eprintln!("error: `cjcl schema` requires a file argument");
        process::exit(1);
    }
    sa
}

// ── Column schema (shared across CSV/TSV/JSONL) ─────────────────────

struct ColumnSchema {
    name: String,
    inferred_type: &'static str,
    null_count: u64,
    total_count: u64,
    unique_values: std::collections::BTreeSet<String>,
    numeric_count: u64,
    string_count: u64,
    bool_count: u64,
    int_count: u64,
    float_count: u64,
    min: f64,
    max: f64,
}

impl ColumnSchema {
    fn new(name: String) -> Self {
        Self {
            name,
            inferred_type: "unknown",
            null_count: 0,
            total_count: 0,
            unique_values: BTreeSet::new(),
            numeric_count: 0,
            string_count: 0,
            bool_count: 0,
            int_count: 0,
            float_count: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Accumulate a single cell value into this column's statistics.
    fn observe(&mut self, val: &str, track_uniques: bool) {
        self.total_count += 1;

        if val.is_empty() || val == "NA" || val == "NaN" || val == "null" || val == "None" {
            self.null_count += 1;
            return;
        }

        if track_uniques && self.unique_values.len() < 1000 {
            self.unique_values.insert(val.to_string());
        }

        let lower = val.to_ascii_lowercase();
        if lower == "true" || lower == "false" {
            self.bool_count += 1;
        } else if let Ok(v) = val.parse::<i64>() {
            self.int_count += 1;
            self.numeric_count += 1;
            let vf = v as f64;
            if vf < self.min { self.min = vf; }
            if vf > self.max { self.max = vf; }
        } else if let Ok(v) = val.parse::<f64>() {
            if !v.is_nan() {
                self.float_count += 1;
                self.numeric_count += 1;
                if v < self.min { self.min = v; }
                if v > self.max { self.max = v; }
            } else {
                self.null_count += 1;
            }
        } else {
            self.string_count += 1;
        }
    }

    /// Finalize the inferred type based on accumulated counts.
    fn finalize_type(&mut self) {
        let non_null = self.total_count - self.null_count;
        if non_null == 0 {
            self.inferred_type = "null";
        } else if self.bool_count == non_null {
            self.inferred_type = "bool";
        } else if self.int_count == non_null {
            self.inferred_type = "int";
        } else if self.float_count > 0 && self.string_count == 0 && self.bool_count == 0 {
            self.inferred_type = "float";
        } else if self.numeric_count == non_null {
            self.inferred_type = "numeric";
        } else if self.string_count == non_null {
            self.inferred_type = "string";
        } else {
            self.inferred_type = "mixed";
        }
    }

    /// Type distribution as a sorted list of (type_name, percentage) pairs.
    /// Only types with count > 0 are included.
    fn type_distribution(&self) -> Vec<(&'static str, f64)> {
        let non_null = self.total_count - self.null_count;
        if non_null == 0 {
            return vec![("null", 100.0)];
        }
        let mut dist = Vec::new();
        let total = non_null as f64;
        if self.bool_count > 0 {
            dist.push(("bool", self.bool_count as f64 / total * 100.0));
        }
        if self.int_count > 0 {
            dist.push(("int", self.int_count as f64 / total * 100.0));
        }
        if self.float_count > 0 {
            dist.push(("float", self.float_count as f64 / total * 100.0));
        }
        if self.string_count > 0 {
            dist.push(("string", self.string_count as f64 / total * 100.0));
        }
        // Sort by type name for determinism
        dist.sort_by(|a, b| a.0.cmp(b.0));
        dist
    }

    /// Return up to 5 sample unique values, sorted for determinism.
    fn sample_values(&self, limit: usize) -> Vec<String> {
        self.unique_values.iter().take(limit).cloned().collect()
    }
}

// ── Saved schema JSON format ────────────────────────────────────────

/// Minimal JSON representation of a column for save/check/diff.
/// Uses deterministic field ordering (alphabetical keys).
struct SavedColumn {
    name: String,
    inferred_type: String,
    null_count: u64,
    total_count: u64,
    unique_count: usize,
    min: Option<f64>,
    max: Option<f64>,
}

struct SavedSchema {
    file: String,
    rows: usize,
    columns: Vec<SavedColumn>,
}

impl SavedSchema {
    /// Serialize to deterministic JSON (no external dependency).
    fn to_json(&self) -> String {
        let mut out = String::from("{\n");
        out.push_str(&format!("  \"file\": \"{}\",\n", self.file.replace('\\', "/")));
        out.push_str(&format!("  \"rows\": {},\n", self.rows));
        out.push_str("  \"columns\": [\n");
        for (i, col) in self.columns.iter().enumerate() {
            out.push_str("    {\n");
            out.push_str(&format!("      \"max\": {},\n",
                col.max.map(|v| output::format_f64(v, 6)).unwrap_or_else(|| "null".to_string())));
            out.push_str(&format!("      \"min\": {},\n",
                col.min.map(|v| output::format_f64(v, 6)).unwrap_or_else(|| "null".to_string())));
            out.push_str(&format!("      \"name\": \"{}\",\n", col.name));
            out.push_str(&format!("      \"null_count\": {},\n", col.null_count));
            out.push_str(&format!("      \"total_count\": {},\n", col.total_count));
            out.push_str(&format!("      \"type\": \"{}\",\n", col.inferred_type));
            out.push_str(&format!("      \"unique_count\": {}\n", col.unique_count));
            out.push_str("    }");
            if i + 1 < self.columns.len() { out.push(','); }
            out.push('\n');
        }
        out.push_str("  ]\n");
        out.push('}');
        out
    }

    /// Parse from JSON string (minimal hand-rolled parser).
    fn from_json(content: &str) -> Result<Self, String> {
        // Extract top-level fields with simple string scanning.
        let file = extract_json_string(content, "file")
            .ok_or("missing \"file\" field in schema JSON")?;
        let rows = extract_json_number(content, "rows")
            .ok_or("missing \"rows\" field in schema JSON")? as usize;

        // Extract the columns array.
        let cols_start = content.find("\"columns\"")
            .ok_or("missing \"columns\" field in schema JSON")?;
        let arr_start = content[cols_start..].find('[')
            .ok_or("malformed columns array")?;
        let arr_start = cols_start + arr_start;

        // Find matching bracket.
        let arr_content = &content[arr_start..];
        let arr_end = find_matching_bracket(arr_content, '[', ']')
            .ok_or("unclosed columns array")?;
        let arr_inner = &arr_content[1..arr_end];

        // Split into individual objects.
        let mut columns = Vec::new();
        let mut depth = 0;
        let mut obj_start = None;
        for (pos, ch) in arr_inner.char_indices() {
            match ch {
                '{' => {
                    if depth == 0 { obj_start = Some(pos); }
                    depth += 1;
                }
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        if let Some(start) = obj_start {
                            let obj_str = &arr_inner[start..=pos];
                            let col = parse_saved_column(obj_str)?;
                            columns.push(col);
                        }
                        obj_start = None;
                    }
                }
                _ => {}
            }
        }

        Ok(SavedSchema { file, rows, columns })
    }
}

fn parse_saved_column(s: &str) -> Result<SavedColumn, String> {
    let name = extract_json_string(s, "name")
        .ok_or("missing \"name\" in column")?;
    let inferred_type = extract_json_string(s, "type")
        .ok_or("missing \"type\" in column")?;
    let null_count = extract_json_number(s, "null_count")
        .ok_or("missing \"null_count\" in column")? as u64;
    let total_count = extract_json_number(s, "total_count")
        .ok_or("missing \"total_count\" in column")? as u64;
    let unique_count = extract_json_number(s, "unique_count")
        .ok_or("missing \"unique_count\" in column")? as usize;
    let min = extract_json_number_or_null(s, "min");
    let max = extract_json_number_or_null(s, "max");

    Ok(SavedColumn {
        name,
        inferred_type,
        null_count,
        total_count,
        unique_count,
        min,
        max,
    })
}

/// Extract a string value for a given key from a JSON fragment.
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\"", key);
    let idx = json.find(&pattern)?;
    let after_key = &json[idx + pattern.len()..];
    // Skip optional whitespace and colon.
    let after_colon = after_key.trim_start().strip_prefix(':')?;
    let after_ws = after_colon.trim_start();
    if !after_ws.starts_with('"') {
        return None;
    }
    let str_start = 1; // skip opening quote
    let str_content = &after_ws[str_start..];
    // Find closing quote (handle escaped quotes).
    let mut escaped = false;
    let mut end = 0;
    for (i, ch) in str_content.char_indices() {
        if escaped {
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            end = i;
            break;
        }
    }
    Some(str_content[..end].replace("\\\"", "\"").replace("\\\\", "\\"))
}

/// Extract a numeric value for a given key from a JSON fragment.
fn extract_json_number(json: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{}\"", key);
    let idx = json.find(&pattern)?;
    let after_key = &json[idx + pattern.len()..];
    let after_colon = after_key.trim_start().strip_prefix(':')?;
    let after_ws = after_colon.trim_start();
    // Read until comma, whitespace, or closing bracket/brace.
    let end = after_ws.find(|c: char| c == ',' || c == '}' || c == ']' || c == '\n')
        .unwrap_or(after_ws.len());
    let num_str = after_ws[..end].trim();
    if num_str == "null" {
        return None;
    }
    num_str.parse::<f64>().ok()
}

/// Extract a numeric value that may be null.
fn extract_json_number_or_null(json: &str, key: &str) -> Option<f64> {
    extract_json_number(json, key)
}

/// Find the position of the matching closing bracket.
fn find_matching_bracket(s: &str, open: char, close: char) -> Option<usize> {
    let mut depth = 0;
    for (i, ch) in s.char_indices() {
        if ch == open { depth += 1; }
        if ch == close {
            depth -= 1;
            if depth == 0 { return Some(i); }
        }
    }
    None
}

// ── Format detection helper ─────────────────────────────────────────

/// Determine the data format from the file path using crate::formats.
fn detect_file_format(path: &Path) -> crate::formats::DataFormat {
    crate::formats::detect_format(path)
}

// ── Main entry points ───────────────────────────────────────────────

pub fn run(args: &[String]) {
    let sa = parse_args(args);
    let path = Path::new(&sa.file);

    let format = detect_file_format(path);

    match format {
        crate::formats::DataFormat::Csv | crate::formats::DataFormat::Tsv => {
            run_delimited(&sa, path);
        }
        crate::formats::DataFormat::Jsonl => {
            run_jsonl(&sa, path);
        }
        crate::formats::DataFormat::Parquet
        | crate::formats::DataFormat::ArrowIpc
        | crate::formats::DataFormat::Sqlite => {
            run_binary_metadata(&sa, path);
        }
        crate::formats::DataFormat::Pickle
        | crate::formats::DataFormat::Onnx
        | crate::formats::DataFormat::Joblib => {
            eprintln!("error: `{}` is a model file ({}); schema inference is not supported",
                sa.file, format.label());
            process::exit(1);
        }
        crate::formats::DataFormat::Unknown => {
            // Fall back to delimited parsing (legacy behavior for files without extension).
            run_delimited(&sa, path);
        }
    }
}

// ── CSV/TSV code path (preserved exactly from original) ─────────────

fn run_delimited(sa: &SchemaArgs, path: &Path) {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => { eprintln!("error: could not read `{}`: {}", sa.file, e); process::exit(1); }
    };

    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        eprintln!("error: empty file");
        process::exit(1);
    }

    let headers: Vec<String> = if sa.has_header {
        lines[0].split(sa.delimiter).map(|s| s.trim().to_string()).collect()
    } else {
        let ncols = lines[0].split(sa.delimiter).count();
        (0..ncols).map(|i| format!("col_{}", i)).collect()
    };

    let data_start = if sa.has_header { 1 } else { 0 };
    let data_lines = &lines[data_start..];
    let max_rows = sa.sample_rows.unwrap_or(data_lines.len()).min(data_lines.len());

    let mut columns: Vec<ColumnSchema> = headers.iter().map(|h| ColumnSchema::new(h.clone())).collect();

    for line in &data_lines[..max_rows] {
        let fields: Vec<&str> = line.split(sa.delimiter).collect();
        for (ci, col) in columns.iter_mut().enumerate() {
            let val = fields.get(ci).map(|s| s.trim()).unwrap_or("");
            col.observe(val, sa.show_uniques);
        }
    }

    // Infer types
    for col in &mut columns {
        col.finalize_type();
    }

    // Handle second-mode operations.
    handle_mode(sa, &columns, max_rows);

    // Output
    print_schema(sa, &columns, max_rows);
}

// ── JSONL code path ─────────────────────────────────────────────────

fn run_jsonl(sa: &SchemaArgs, path: &Path) {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => { eprintln!("error: could not read `{}`: {}", sa.file, e); process::exit(1); }
    };

    let tabular = crate::formats::load_jsonl(&content);

    if tabular.headers.is_empty() {
        eprintln!("error: no valid JSONL objects found in `{}`", sa.file);
        process::exit(1);
    }

    let max_rows = sa.sample_rows.unwrap_or(tabular.nrows()).min(tabular.nrows());

    let mut columns: Vec<ColumnSchema> = tabular.headers.iter()
        .map(|h| ColumnSchema::new(h.clone()))
        .collect();

    for row in tabular.rows.iter().take(max_rows) {
        for (ci, col) in columns.iter_mut().enumerate() {
            let val = row.get(ci).map(|s| s.as_str()).unwrap_or("");
            col.observe(val, sa.show_uniques);
        }
    }

    for col in &mut columns {
        col.finalize_type();
    }

    handle_mode(sa, &columns, max_rows);
    print_schema(sa, &columns, max_rows);
}

// ── Binary metadata code path ───────────────────────────────────────

fn run_binary_metadata(sa: &SchemaArgs, path: &Path) {
    let meta = crate::formats::extract_metadata(path);

    // Binary formats don't support --save/--check/--diff since they have no column-level schema.
    match &sa.mode {
        SchemaMode::Save(f) => {
            eprintln!("error: --save is not supported for binary format {} (no column-level schema)",
                meta.format.label());
            let _ = f;
            process::exit(1);
        }
        SchemaMode::Check(f) => {
            eprintln!("error: --check is not supported for binary format {} (no column-level schema)",
                meta.format.label());
            let _ = f;
            process::exit(1);
        }
        SchemaMode::Diff(f) => {
            eprintln!("error: --diff is not supported for binary format {} (no column-level schema)",
                meta.format.label());
            let _ = f;
            process::exit(1);
        }
        SchemaMode::Infer => {}
    }

    match sa.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file\": \"{}\",", sa.file.replace('\\', "/"));
            println!("  \"format\": \"{}\",", meta.format.label());
            println!("  \"size\": {},", meta.size);
            if let Some(ref magic) = meta.magic_bytes {
                println!("  \"magic_bytes\": \"{}\",", magic);
            }
            println!("  \"is_safe_to_parse\": {},", meta.is_safe_to_parse);
            println!("  \"metadata\": {{");
            let entries: Vec<_> = meta.header_info.iter().collect();
            for (i, (k, v)) in entries.iter().enumerate() {
                print!("    \"{}\": \"{}\"", k, v);
                if i + 1 < entries.len() { print!(","); }
                println!();
            }
            println!("  }},");
            println!("  \"limitations\": [");
            for (i, lim) in meta.limitations.iter().enumerate() {
                print!("    \"{}\"", lim);
                if i + 1 < meta.limitations.len() { print!(","); }
                println!();
            }
            println!("  ]");
            println!("}}");
        }
        _ => {
            eprintln!("Metadata for `{}` (format: {}):", sa.file.replace('\\', "/"), meta.format.label());
            eprintln!();

            let mut t = crate::table::Table::new(vec!["Property", "Value"]);
            t.add_row_owned(vec!["Format".to_string(), meta.format.label().to_string()]);
            t.add_row_owned(vec!["File size".to_string(), output::format_size(meta.size)]);
            if let Some(ref magic) = meta.magic_bytes {
                t.add_row_owned(vec!["Magic bytes".to_string(), magic.clone()]);
            }
            t.add_row_owned(vec!["Safe to parse".to_string(),
                if meta.is_safe_to_parse { "yes" } else { "no" }.to_string()]);

            for (k, v) in &meta.header_info {
                t.add_row_owned(vec![k.clone(), v.clone()]);
            }
            eprint!("{}", t.render());

            if !meta.limitations.is_empty() {
                eprintln!();
                eprintln!("Limitations:");
                for lim in &meta.limitations {
                    eprintln!("  - {}", lim);
                }
            }
        }
    }
}

// ── Schema output (shared by CSV/TSV and JSONL) ─────────────────────

fn print_schema(sa: &SchemaArgs, columns: &[ColumnSchema], max_rows: usize) {
    match sa.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file\": \"{}\",", sa.file.replace('\\', "/"));
            println!("  \"rows\": {},", max_rows);
            println!("  \"columns\": [");
            for (i, col) in columns.iter().enumerate() {
                print!("    {{\"name\": \"{}\", \"type\": \"{}\", \"nulls\": {}, \"total\": {}",
                    col.name, col.inferred_type, col.null_count, col.total_count);
                if sa.show_uniques {
                    print!(", \"unique\": {}", col.unique_values.len());
                }
                if col.numeric_count > 0 {
                    print!(", \"min\": {}, \"max\": {}",
                        output::format_f64(col.min, 6), output::format_f64(col.max, 6));
                }
                if sa.full {
                    let dist = col.type_distribution();
                    let dist_str: Vec<String> = dist.iter()
                        .map(|(t, pct)| format!("\"{}\":{}", t, output::format_f64(*pct, 1)))
                        .collect();
                    print!(", \"type_distribution\": {{{}}}", dist_str.join(","));
                    let samples = col.sample_values(5);
                    let samples_json: Vec<String> = samples.iter()
                        .map(|s| format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")))
                        .collect();
                    print!(", \"sample_values\": [{}]", samples_json.join(","));
                }
                print!("}}");
                if i + 1 < columns.len() { print!(","); }
                println!();
            }
            println!("  ]");
            println!("}}");
        }
        _ => {
            eprintln!("Schema for `{}` ({} rows sampled):", sa.file.replace('\\', "/"), max_rows);
            eprintln!();

            let mut headers_vec = vec!["Column", "Type", "Nulls", "Total"];
            if sa.show_uniques { headers_vec.push("Unique"); }
            headers_vec.extend_from_slice(&["Min", "Max"]);
            if sa.full {
                headers_vec.push("Distribution");
                headers_vec.push("Samples");
            }

            let mut t = crate::table::Table::new(headers_vec);
            for col in columns {
                let (min, max) = if col.numeric_count > 0 {
                    (output::format_f64(col.min, 4), output::format_f64(col.max, 4))
                } else {
                    ("-".into(), "-".into())
                };
                let mut row = vec![
                    col.name.clone(),
                    col.inferred_type.to_string(),
                    format!("{}", col.null_count),
                    format!("{}", col.total_count),
                ];
                if sa.show_uniques {
                    row.push(format!("{}", col.unique_values.len()));
                }
                row.push(min);
                row.push(max);
                if sa.full {
                    let dist = col.type_distribution();
                    let dist_str: Vec<String> = dist.iter()
                        .map(|(t, pct)| format!("{}:{:.1}%", t, pct))
                        .collect();
                    row.push(dist_str.join(" "));
                    let samples = col.sample_values(5);
                    row.push(samples.join(", "));
                }
                t.add_row_owned(row);
            }
            eprint!("{}", t.render());
        }
    }
}

// ── Second-mode operations (save/check/diff) ────────────────────────

fn handle_mode(sa: &SchemaArgs, columns: &[ColumnSchema], max_rows: usize) {
    match &sa.mode {
        SchemaMode::Infer => {}
        SchemaMode::Save(schema_file) => {
            do_save(sa, columns, max_rows, schema_file);
        }
        SchemaMode::Check(schema_file) => {
            do_check(sa, columns, max_rows, schema_file);
        }
        SchemaMode::Diff(schema_file) => {
            do_diff(sa, columns, schema_file);
        }
    }
}

fn columns_to_saved(sa: &SchemaArgs, columns: &[ColumnSchema], max_rows: usize) -> SavedSchema {
    let saved_cols: Vec<SavedColumn> = columns.iter().map(|col| {
        SavedColumn {
            name: col.name.clone(),
            inferred_type: col.inferred_type.to_string(),
            null_count: col.null_count,
            total_count: col.total_count,
            unique_count: col.unique_values.len(),
            min: if col.numeric_count > 0 { Some(col.min) } else { None },
            max: if col.numeric_count > 0 { Some(col.max) } else { None },
        }
    }).collect();

    SavedSchema {
        file: sa.file.replace('\\', "/"),
        rows: max_rows,
        columns: saved_cols,
    }
}

fn do_save(sa: &SchemaArgs, columns: &[ColumnSchema], max_rows: usize, schema_file: &str) {
    let saved = columns_to_saved(sa, columns, max_rows);
    let json = saved.to_json();
    match fs::write(schema_file, json.as_bytes()) {
        Ok(_) => eprintln!("Schema saved to `{}`", schema_file),
        Err(e) => {
            eprintln!("error: could not write schema to `{}`: {}", schema_file, e);
            process::exit(1);
        }
    }
}

fn do_check(sa: &SchemaArgs, columns: &[ColumnSchema], max_rows: usize, schema_file: &str) {
    let saved_content = match fs::read_to_string(schema_file) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: could not read schema file `{}`: {}", schema_file, e);
            process::exit(1);
        }
    };
    let saved = match SavedSchema::from_json(&saved_content) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: invalid schema file `{}`: {}", schema_file, e);
            process::exit(1);
        }
    };

    let current = columns_to_saved(sa, columns, max_rows);
    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    // Check column count.
    if current.columns.len() != saved.columns.len() {
        errors.push(format!("column count mismatch: current has {}, saved has {}",
            current.columns.len(), saved.columns.len()));
    }

    // Check each column by name.
    let saved_by_name: std::collections::BTreeMap<&str, &SavedColumn> = saved.columns.iter()
        .map(|c| (c.name.as_str(), c))
        .collect();

    for cur_col in &current.columns {
        if let Some(sav_col) = saved_by_name.get(cur_col.name.as_str()) {
            if cur_col.inferred_type != sav_col.inferred_type {
                let msg = format!("column `{}`: type mismatch (current: {}, saved: {})",
                    cur_col.name, cur_col.inferred_type, sav_col.inferred_type);
                if sa.strict {
                    errors.push(msg);
                } else {
                    warnings.push(msg);
                }
            }
        } else {
            errors.push(format!("column `{}`: not found in saved schema", cur_col.name));
        }
    }

    // Check for columns in saved but not in current.
    let current_names: BTreeSet<&str> = current.columns.iter()
        .map(|c| c.name.as_str())
        .collect();
    for sav_col in &saved.columns {
        if !current_names.contains(sav_col.name.as_str()) {
            errors.push(format!("column `{}`: present in saved schema but missing from data",
                sav_col.name));
        }
    }

    // Report results.
    if !warnings.is_empty() {
        for w in &warnings {
            eprintln!("warning: {}", w);
        }
    }
    if !errors.is_empty() {
        for e in &errors {
            eprintln!("error: {}", e);
        }
        eprintln!();
        eprintln!("Schema check FAILED ({} error(s), {} warning(s))", errors.len(), warnings.len());
        process::exit(1);
    }

    eprintln!("Schema check passed ({} warning(s))", warnings.len());
}

fn do_diff(sa: &SchemaArgs, columns: &[ColumnSchema], schema_file: &str) {
    let _ = sa;
    let saved_content = match fs::read_to_string(schema_file) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: could not read schema file `{}`: {}", schema_file, e);
            process::exit(1);
        }
    };
    let saved = match SavedSchema::from_json(&saved_content) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: invalid schema file `{}`: {}", schema_file, e);
            process::exit(1);
        }
    };

    let saved_by_name: std::collections::BTreeMap<&str, &SavedColumn> = saved.columns.iter()
        .map(|c| (c.name.as_str(), c))
        .collect();

    let current_names: BTreeSet<&str> = columns.iter()
        .map(|c| c.name.as_str())
        .collect();
    let saved_names: BTreeSet<&str> = saved.columns.iter()
        .map(|c| c.name.as_str())
        .collect();

    let added: BTreeSet<&&str> = current_names.difference(&saved_names).collect();
    let removed: BTreeSet<&&str> = saved_names.difference(&current_names).collect();
    let common: BTreeSet<&&str> = current_names.intersection(&saved_names).collect();

    let mut diffs: Vec<String> = Vec::new();

    for name in &added {
        diffs.push(format!("+ column `{}` (new)", name));
    }
    for name in &removed {
        diffs.push(format!("- column `{}` (removed)", name));
    }

    for name in &common {
        let cur = columns.iter().find(|c| c.name == ***name).unwrap();
        let sav = saved_by_name[**name];

        let mut col_diffs = Vec::new();
        if cur.inferred_type != sav.inferred_type {
            col_diffs.push(format!("type: {} -> {}", sav.inferred_type, cur.inferred_type));
        }
        if cur.null_count != sav.null_count {
            col_diffs.push(format!("nulls: {} -> {}", sav.null_count, cur.null_count));
        }
        if cur.unique_values.len() != sav.unique_count {
            col_diffs.push(format!("unique: {} -> {}", sav.unique_count, cur.unique_values.len()));
        }

        if !col_diffs.is_empty() {
            diffs.push(format!("~ column `{}`: {}", name, col_diffs.join(", ")));
        }
    }

    if diffs.is_empty() {
        eprintln!("No differences found.");
    } else {
        eprintln!("Schema diff ({} change(s)):", diffs.len());
        eprintln!();
        for d in &diffs {
            eprintln!("  {}", d);
        }
    }
}

// ── Help ────────────────────────────────────────────────────────────

pub fn print_help() {
    eprintln!("cjcl schema — Schema inference and reporting");
    eprintln!();
    eprintln!("Usage: cjcl schema <file> [flags]");
    eprintln!();
    eprintln!("Supported formats:");
    eprintln!("  .csv                  CSV (comma-separated)");
    eprintln!("  .tsv                  TSV (tab-separated)");
    eprintln!("  .jsonl, .ndjson       JSON Lines (newline-delimited JSON)");
    eprintln!("  .parquet              Parquet (metadata only)");
    eprintln!("  .arrow, .feather      Arrow IPC (metadata only)");
    eprintln!("  .sqlite, .db          SQLite (metadata only)");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  -d, --delimiter <c>   Column delimiter (default: ,)");
    eprintln!("  --tsv                 Use tab delimiter");
    eprintln!("  --no-header           Input has no header row");
    eprintln!("  --no-uniques          Skip unique value counting");
    eprintln!("  --sample <N>          Sample only first N rows");
    eprintln!("  --plain               Plain text output");
    eprintln!("  --json                JSON output");
    eprintln!("  --color               Color output (default)");
    eprintln!("  --full                Show type distribution and sample values");
    eprintln!("  --strict              Treat warnings as errors in --check");
    eprintln!();
    eprintln!("Schema operations:");
    eprintln!("  --save <file.json>    Save inferred schema to JSON");
    eprintln!("  --check <file.json>   Validate data against saved schema (exit 1 on mismatch)");
    eprintln!("  --diff <file.json>    Compare current schema against saved schema");
}
