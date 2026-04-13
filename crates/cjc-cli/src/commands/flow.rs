//! `cjcl flow` — Streaming computation engine.
//!
//! Processes CSV/TSV/JSONL data streams with O(ncols) memory using:
//! - Kahan summation for numeric stability
//! - Single-pass streaming aggregates (sum, mean, min, max, count, var, std)
//! - Deterministic output regardless of data size
//!
//! Also supports metadata-only inspection of binary formats (Parquet, Arrow, Feather).
//!
//! Designed for massive datasets that cannot fit in memory.

use std::fs;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process;
use crate::formats::{self, DataFormat};
use crate::output::{self, OutputMode};

/// Parsed arguments for `cjcl flow`.
pub struct FlowArgs {
    pub file: Option<PathBuf>,
    pub delimiter: char,
    pub has_header: bool,
    pub ops: Vec<AggOp>,
    pub columns: Option<Vec<String>>,
    pub output: OutputMode,
    pub precision: usize,
    /// --verify: run the flow twice and confirm identical output (determinism check).
    pub verify: bool,
    /// --sort-by <metric>: sort output columns by a metric.
    pub sort_by: Option<AggOp>,
    /// --top <N>: only show top N columns by the first aggregation metric.
    pub top: Option<usize>,
    /// --out <file>: write output to a file instead of stdout.
    pub out_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggOp {
    Sum,
    Mean,
    Min,
    Max,
    Count,
    Var,
    Std,
}

impl AggOp {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "sum" => Some(AggOp::Sum),
            "mean" | "avg" => Some(AggOp::Mean),
            "min" => Some(AggOp::Min),
            "max" => Some(AggOp::Max),
            "count" => Some(AggOp::Count),
            "var" | "variance" => Some(AggOp::Var),
            "std" | "stddev" => Some(AggOp::Std),
            _ => None,
        }
    }

    fn label(&self) -> &'static str {
        match self {
            AggOp::Sum => "sum (Kahan)",
            AggOp::Mean => "mean",
            AggOp::Min => "min",
            AggOp::Max => "max",
            AggOp::Count => "count",
            AggOp::Var => "var",
            AggOp::Std => "std",
        }
    }
}

impl Default for FlowArgs {
    fn default() -> Self {
        Self {
            file: None,
            delimiter: ',',
            has_header: true,
            ops: vec![AggOp::Sum, AggOp::Mean, AggOp::Min, AggOp::Max, AggOp::Count],
            columns: None,
            output: OutputMode::Color,
            precision: 6,
            verify: false,
            sort_by: None,
            top: None,
            out_file: None,
        }
    }
}

pub fn parse_args(args: &[String]) -> FlowArgs {
    let mut fa = FlowArgs::default();
    let mut custom_ops = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-d" | "--delimiter" => {
                i += 1;
                if i < args.len() {
                    fa.delimiter = match args[i].as_str() {
                        "tab" | "\\t" | "\t" => '\t',
                        s if s.len() == 1 => s.chars().next().unwrap(),
                        _ => {
                            eprintln!("error: delimiter must be a single character or 'tab'");
                            process::exit(1);
                        }
                    };
                }
            }
            "--no-header" => fa.has_header = false,
            "--tsv" => { fa.delimiter = '\t'; }
            "--op" | "--ops" => {
                i += 1;
                if i < args.len() {
                    if !custom_ops {
                        fa.ops.clear();
                        custom_ops = true;
                    }
                    for op_str in args[i].split(',') {
                        match AggOp::from_str(op_str.trim()) {
                            Some(op) => fa.ops.push(op),
                            None => {
                                eprintln!("error: unknown operation `{}`", op_str);
                                eprintln!("  available: sum, mean, min, max, count, var, std");
                                process::exit(1);
                            }
                        }
                    }
                }
            }
            "--columns" | "-c" => {
                i += 1;
                if i < args.len() {
                    fa.columns = Some(args[i].split(',').map(|s| s.trim().to_string()).collect());
                }
            }
            "--precision" => {
                i += 1;
                if i < args.len() {
                    fa.precision = args[i].parse().unwrap_or(6);
                }
            }
            "--plain" => fa.output = OutputMode::Plain,
            "--json" => fa.output = OutputMode::Json,
            "--table" => fa.output = OutputMode::Table,
            "--color" => fa.output = OutputMode::Color,
            "--verify" => fa.verify = true,
            "--sort-by" => {
                i += 1;
                if i < args.len() {
                    match AggOp::from_str(args[i].trim()) {
                        Some(op) => fa.sort_by = Some(op),
                        None => {
                            eprintln!("error: unknown sort metric `{}`", args[i]);
                            eprintln!("  available: sum, mean, min, max, count, var, std");
                            process::exit(1);
                        }
                    }
                }
            }
            "--top" => {
                i += 1;
                if i < args.len() {
                    match args[i].parse::<usize>() {
                        Ok(n) if n > 0 => fa.top = Some(n),
                        _ => {
                            eprintln!("error: --top requires a positive integer");
                            process::exit(1);
                        }
                    }
                }
            }
            "--out" => {
                i += 1;
                if i < args.len() {
                    fa.out_file = Some(PathBuf::from(&args[i]));
                }
            }
            other if !other.starts_with('-') => fa.file = Some(PathBuf::from(other)),
            other => {
                eprintln!("error: unknown flag `{}` for `cjcl flow`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    if fa.ops.is_empty() {
        fa.ops = vec![AggOp::Sum, AggOp::Mean, AggOp::Min, AggOp::Max, AggOp::Count];
    }
    fa
}

/// Streaming accumulator for a single column. O(1) memory per column.
struct ColumnAccum {
    name: String,
    count: u64,
    // Kahan summation state
    sum: f64,
    sum_comp: f64,  // Kahan compensation
    // Welford's online variance
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
    is_numeric: bool,
}

impl ColumnAccum {
    fn new(name: String) -> Self {
        Self {
            name,
            count: 0,
            sum: 0.0,
            sum_comp: 0.0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            is_numeric: true,
        }
    }

    fn update(&mut self, value: &str) {
        let trimmed = value.trim();
        if trimmed.is_empty() || trimmed == "NA" || trimmed == "NaN" || trimmed == "null" {
            return;
        }

        match trimmed.parse::<f64>() {
            Ok(v) if !v.is_nan() => {
                self.count += 1;

                // Kahan summation
                let y = v - self.sum_comp;
                let t = self.sum + y;
                self.sum_comp = (t - self.sum) - y;
                self.sum = t;

                // Welford's online mean/variance
                let delta = v - self.mean;
                self.mean += delta / self.count as f64;
                let delta2 = v - self.mean;
                self.m2 += delta * delta2;

                if v < self.min { self.min = v; }
                if v > self.max { self.max = v; }
            }
            _ => {
                self.is_numeric = false;
                self.count += 1;
            }
        }
    }

    fn get(&self, op: AggOp, precision: usize) -> String {
        if !self.is_numeric && op != AggOp::Count {
            return "-".to_string();
        }
        if self.count == 0 && op != AggOp::Count {
            return "NaN".to_string();
        }
        match op {
            AggOp::Count => format!("{}", self.count),
            AggOp::Sum => output::format_f64(self.sum, precision),
            AggOp::Mean => output::format_f64(self.mean, precision),
            AggOp::Min => output::format_f64(self.min, precision),
            AggOp::Max => output::format_f64(self.max, precision),
            AggOp::Var => {
                let var = if self.count > 1 { self.m2 / (self.count - 1) as f64 } else { 0.0 };
                output::format_f64(var, precision)
            }
            AggOp::Std => {
                let var = if self.count > 1 { self.m2 / (self.count - 1) as f64 } else { 0.0 };
                output::format_f64(var.sqrt(), precision)
            }
        }
    }

    /// Get a raw f64 value for sorting purposes.
    fn get_raw(&self, op: AggOp) -> f64 {
        if !self.is_numeric && op != AggOp::Count {
            return f64::NEG_INFINITY;
        }
        if self.count == 0 && op != AggOp::Count {
            return f64::NAN;
        }
        match op {
            AggOp::Count => self.count as f64,
            AggOp::Sum => self.sum,
            AggOp::Mean => self.mean,
            AggOp::Min => self.min,
            AggOp::Max => self.max,
            AggOp::Var => {
                if self.count > 1 { self.m2 / (self.count - 1) as f64 } else { 0.0 }
            }
            AggOp::Std => {
                let var = if self.count > 1 { self.m2 / (self.count - 1) as f64 } else { 0.0 };
                var.sqrt()
            }
        }
    }
}

/// Entry point for `cjcl flow`.
pub fn run(args: &[String]) {
    let fa = parse_args(args);

    // If --verify is set, we need a file (cannot verify stdin).
    if fa.verify && fa.file.is_none() {
        eprintln!("error: --verify requires a file argument (cannot verify stdin)");
        process::exit(1);
    }

    // Check for binary/metadata-only formats when a file is provided.
    if let Some(ref path) = fa.file {
        let format = formats::detect_format(path);
        match format {
            DataFormat::Parquet | DataFormat::ArrowIpc | DataFormat::Sqlite => {
                run_binary_metadata(path, format, &fa);
                return;
            }
            DataFormat::Pickle | DataFormat::Onnx | DataFormat::Joblib => {
                eprintln!("error: `{}` is a {} model file — cannot aggregate", path.display(), format.label());
                eprintln!("  Model files are never deserialized for safety reasons.");
                eprintln!("  Use `cjcl schema {}` for metadata inspection.", path.display());
                process::exit(1);
            }
            DataFormat::Jsonl => {
                run_jsonl(path, &fa);
                return;
            }
            // CSV, TSV, Unknown: fall through to existing streaming path.
            _ => {}
        }
    }

    // Existing CSV/TSV streaming path (unchanged).
    let result = run_streaming_csv(&fa);

    if fa.verify {
        // Run a second pass and compare.
        let result2 = run_streaming_csv(&fa);
        if result == result2 {
            eprintln!("(verify: PASS — both runs produced identical output)");
        } else {
            eprintln!("(verify: FAIL — outputs differ between runs!)");
            process::exit(1);
        }
    }

    emit_output(&result, &fa);
}

/// Result from a streaming aggregation pass.
#[derive(PartialEq)]
struct FlowResult {
    rows_processed: u64,
    /// (column_name, accum_values_per_op)
    columns: Vec<ColumnResult>,
}

#[derive(PartialEq)]
struct ColumnResult {
    name: String,
    values: Vec<String>,
    /// Raw sort key (f64 value of the sort-by metric, or first op if no sort-by).
    sort_key: f64,
}

/// Run the streaming CSV/TSV aggregation, returning structured results.
fn run_streaming_csv(fa: &FlowArgs) -> FlowResult {
    let reader: Box<dyn BufRead> = match &fa.file {
        Some(path) => {
            let f = fs::File::open(path).unwrap_or_else(|e| {
                eprintln!("error: could not open `{}`: {}", path.display(), e);
                process::exit(1);
            });
            Box::new(io::BufReader::new(f))
        }
        None => Box::new(io::BufReader::new(io::stdin())),
    };

    let mut lines = reader.lines();
    let mut accums: Vec<ColumnAccum> = Vec::new();
    let mut header_names: Vec<String> = Vec::new();
    let mut rows_processed = 0u64;

    // Read header
    if fa.has_header {
        if let Some(Ok(header_line)) = lines.next() {
            header_names = header_line.split(fa.delimiter)
                .map(|s| s.trim().to_string())
                .collect();
        }
    }

    // Process rows
    for line_result in lines {
        let line = match line_result {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() { continue; }

        let fields: Vec<&str> = line.split(fa.delimiter).collect();

        // Initialize accumulators on first data row
        if accums.is_empty() {
            let ncols = fields.len();
            for col_idx in 0..ncols {
                let name = if col_idx < header_names.len() {
                    header_names[col_idx].clone()
                } else {
                    format!("col_{}", col_idx)
                };
                accums.push(ColumnAccum::new(name));
            }
        }

        for (col_idx, field) in fields.iter().enumerate() {
            if col_idx < accums.len() {
                accums[col_idx].update(field);
            }
        }
        rows_processed += 1;
    }

    accums_to_result(accums, rows_processed, fa)
}

/// Run JSONL aggregation using the streaming JSONL loader from the formats module.
fn run_jsonl(path: &PathBuf, fa: &FlowArgs) {
    let result = run_jsonl_pass(path, fa);

    if fa.verify {
        let result2 = run_jsonl_pass(path, fa);
        if result == result2 {
            eprintln!("(verify: PASS — both runs produced identical output)");
        } else {
            eprintln!("(verify: FAIL — outputs differ between runs!)");
            process::exit(1);
        }
    }

    emit_output(&result, fa);
}

fn run_jsonl_pass(path: &PathBuf, fa: &FlowArgs) -> FlowResult {
    let f = fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("error: could not open `{}`: {}", path.display(), e);
        process::exit(1);
    });
    let reader = io::BufReader::new(f);
    let (headers, row_iter) = formats::load_jsonl_streaming(reader);

    if headers.is_empty() {
        eprintln!("warning: no columns discovered in JSONL file");
        return FlowResult { rows_processed: 0, columns: Vec::new() };
    }

    let mut accums: Vec<ColumnAccum> = headers
        .iter()
        .map(|h| ColumnAccum::new(h.clone()))
        .collect();
    let mut rows_processed = 0u64;

    for row in row_iter {
        for (col_idx, field) in row.iter().enumerate() {
            if col_idx < accums.len() {
                accums[col_idx].update(field);
            }
        }
        rows_processed += 1;
    }

    accums_to_result(accums, rows_processed, fa)
}

/// Convert accumulators into a FlowResult, applying column filter, sort, and top-N.
fn accums_to_result(accums: Vec<ColumnAccum>, rows_processed: u64, fa: &FlowArgs) -> FlowResult {
    // Filter to requested columns.
    let active_accums: Vec<&ColumnAccum> = if let Some(ref cols) = fa.columns {
        accums.iter().filter(|a| cols.contains(&a.name)).collect()
    } else {
        accums.iter().collect()
    };

    // Determine sort metric: --sort-by if given, otherwise first op.
    let sort_metric = fa.sort_by.unwrap_or_else(|| fa.ops.first().copied().unwrap_or(AggOp::Sum));

    let mut columns: Vec<ColumnResult> = active_accums
        .iter()
        .map(|acc| {
            let values: Vec<String> = fa.ops.iter().map(|op| acc.get(*op, fa.precision)).collect();
            let sort_key = acc.get_raw(sort_metric);
            ColumnResult {
                name: acc.name.clone(),
                values,
                sort_key,
            }
        })
        .collect();

    // Apply --sort-by: sort descending by the metric (NaN sorts last).
    if fa.sort_by.is_some() {
        columns.sort_by(|a, b| {
            // Deterministic total ordering: NaN < finite, then descending.
            match (a.sort_key.is_nan(), b.sort_key.is_nan()) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater,  // NaN sorts to end
                (false, true) => std::cmp::Ordering::Less,
                (false, false) => b.sort_key.partial_cmp(&a.sort_key).unwrap_or(std::cmp::Ordering::Equal),
            }
        });
    }

    // Apply --top N.
    if let Some(top_n) = fa.top {
        // If not already sorted by --sort-by, sort descending by first op for top-N.
        if fa.sort_by.is_none() {
            columns.sort_by(|a, b| {
                match (a.sort_key.is_nan(), b.sort_key.is_nan()) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (true, false) => std::cmp::Ordering::Greater,
                    (false, true) => std::cmp::Ordering::Less,
                    (false, false) => b.sort_key.partial_cmp(&a.sort_key).unwrap_or(std::cmp::Ordering::Equal),
                }
            });
        }
        columns.truncate(top_n);
    }

    FlowResult { rows_processed, columns }
}

/// Emit results to stdout (or --out file), respecting the output mode.
fn emit_output(result: &FlowResult, fa: &FlowArgs) {
    let output_text = format_result(result, fa);

    if let Some(ref out_path) = fa.out_file {
        let mut f = fs::File::create(out_path).unwrap_or_else(|e| {
            eprintln!("error: could not create output file `{}`: {}", out_path.display(), e);
            process::exit(1);
        });
        f.write_all(output_text.as_bytes()).unwrap_or_else(|e| {
            eprintln!("error: could not write to `{}`: {}", out_path.display(), e);
            process::exit(1);
        });
        eprintln!("({} rows processed, output written to {})", result.rows_processed, out_path.display());
    } else {
        print!("{}", output_text);
        eprintln!("({} rows processed)", result.rows_processed);
    }
}

/// Format the result according to output mode.
fn format_result(result: &FlowResult, fa: &FlowArgs) -> String {
    match fa.output {
        OutputMode::Json => {
            let mut out = String::new();
            out.push_str("{\n");
            out.push_str(&format!("  \"rows_processed\": {},\n", result.rows_processed));
            out.push_str("  \"columns\": [\n");
            for (i, col) in result.columns.iter().enumerate() {
                out.push_str(&format!("    {{\"name\": \"{}\"", col.name));
                for (j, op) in fa.ops.iter().enumerate() {
                    out.push_str(&format!(", \"{}\": {}", op.label(), col.values[j]));
                }
                out.push('}');
                if i + 1 < result.columns.len() { out.push(','); }
                out.push('\n');
            }
            out.push_str("  ]\n");
            out.push_str("}\n");
            out
        }
        _ => {
            let mut headers = vec!["Column"];
            let op_labels: Vec<&str> = fa.ops.iter().map(|o| o.label()).collect();
            headers.extend(op_labels.iter());

            let mut t = crate::table::Table::new(headers);
            for col in &result.columns {
                let mut row = vec![col.name.clone()];
                row.extend(col.values.iter().cloned());
                t.add_row_owned(row);
            }
            t.render()
        }
    }
}

/// Handle binary formats in metadata-only mode.
fn run_binary_metadata(path: &PathBuf, format: DataFormat, fa: &FlowArgs) {
    let meta = formats::extract_metadata(path);

    match fa.output {
        OutputMode::Json => {
            let mut out = String::from("{\n");
            out.push_str(&format!("  \"format\": \"{}\",\n", format.label()));
            out.push_str(&format!("  \"file\": \"{}\",\n", path.display()));
            out.push_str(&format!("  \"size_bytes\": {},\n", meta.size));
            out.push_str("  \"aggregation_supported\": false,\n");

            // Emit header_info as JSON object.
            out.push_str("  \"metadata\": {\n");
            let info_entries: Vec<(&String, &String)> = meta.header_info.iter().collect();
            for (i, (key, val)) in info_entries.iter().enumerate() {
                // Try to emit numeric values without quotes.
                if val.parse::<f64>().is_ok() {
                    out.push_str(&format!("    \"{}\": {}", key, val));
                } else {
                    out.push_str(&format!("    \"{}\": \"{}\"", key, val.replace('\\', "\\\\").replace('"', "\\\"")));
                }
                if i + 1 < info_entries.len() { out.push(','); }
                out.push('\n');
            }
            out.push_str("  },\n");

            // Emit limitations.
            out.push_str("  \"limitations\": [\n");
            for (i, lim) in meta.limitations.iter().enumerate() {
                out.push_str(&format!("    \"{}\"", lim.replace('"', "\\\"")));
                if i + 1 < meta.limitations.len() { out.push(','); }
                out.push('\n');
            }
            out.push_str("  ]\n");
            out.push_str("}\n");

            if let Some(ref out_path) = fa.out_file {
                write_to_file(out_path, &out);
            } else {
                print!("{}", out);
            }
        }
        _ => {
            eprintln!("cjcl flow — {} metadata (aggregation not available)", format.label());
            eprintln!();

            let mut t = crate::table::Table::new(vec!["Property", "Value"]);
            t.add_row_owned(vec!["Format".to_string(), format.label().to_string()]);
            t.add_row_owned(vec!["File".to_string(), path.display().to_string()]);
            t.add_row_owned(vec!["Size".to_string(), output::format_size(meta.size)]);

            if let Some(ref magic) = meta.magic_bytes {
                t.add_row_owned(vec!["Magic bytes".to_string(), magic.clone()]);
            }

            for (key, val) in &meta.header_info {
                t.add_row_owned(vec![key.clone(), val.clone()]);
            }

            if !meta.limitations.is_empty() {
                t.add_row_owned(vec!["---".to_string(), "---".to_string()]);
                for lim in &meta.limitations {
                    t.add_row_owned(vec!["Limitation".to_string(), lim.clone()]);
                }
            }

            let rendered = t.render();
            if let Some(ref out_path) = fa.out_file {
                write_to_file(out_path, &rendered);
            } else {
                print!("{}", rendered);
            }

            eprintln!();
            eprintln!("note: Full streaming aggregation is not available for {} files.", format.label());
            eprintln!("      CJC does not include a {} parser — only metadata is shown.", format.label());
        }
    }
}

/// Write a string to a file, exiting on error.
fn write_to_file(path: &PathBuf, content: &str) {
    let mut f = fs::File::create(path).unwrap_or_else(|e| {
        eprintln!("error: could not create output file `{}`: {}", path.display(), e);
        process::exit(1);
    });
    f.write_all(content.as_bytes()).unwrap_or_else(|e| {
        eprintln!("error: could not write to `{}`: {}", path.display(), e);
        process::exit(1);
    });
    eprintln!("(output written to {})", path.display());
}

pub fn print_help() {
    eprintln!("cjcl flow — Streaming computation engine");
    eprintln!();
    eprintln!("Usage: cjcl flow [file] [flags]");
    eprintln!("       cat data.csv | cjc flow [flags]");
    eprintln!();
    eprintln!("Supported formats:");
    eprintln!("  CSV, TSV            Full streaming aggregation");
    eprintln!("  JSONL, NDJSON       Full streaming aggregation");
    eprintln!("  Parquet, Arrow, Feather  Metadata-only (no full aggregation)");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  -d, --delimiter <c>   Column delimiter (default: ,)");
    eprintln!("  --tsv                 Use tab delimiter");
    eprintln!("  --no-header           Input has no header row");
    eprintln!("  --op <ops>            Operations: sum,mean,min,max,count,var,std");
    eprintln!("  -c, --columns <cols>  Only process named columns (comma-separated)");
    eprintln!("  --precision <N>       Decimal precision (default: 6)");
    eprintln!("  --plain               Plain text output");
    eprintln!("  --json                JSON output");
    eprintln!("  --color               Color output (default)");
    eprintln!();
    eprintln!("Second-mode flags:");
    eprintln!("  --verify              Run twice, confirm identical output (determinism check)");
    eprintln!("  --sort-by <metric>    Sort columns by metric (sum, mean, min, max, count, var, std)");
    eprintln!("  --top <N>             Show only top N columns by first aggregation metric");
    eprintln!("  --out <file>          Write output to a file instead of stdout");
}
