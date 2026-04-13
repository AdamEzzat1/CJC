//! `cjcl drift` — Mathematical and data diff engine.
//!
//! Compares two files or data sources and computes:
//! - Text diff (line-by-line)
//! - Structured data diff (CSV column comparison)
//! - JSONL diff (parsed to tabular, then cell-by-cell like CSV)
//! - Tensor/numeric diff (Frobenius norm, max deviation, element-wise)
//! - Shape mismatch detection
//! - NaN divergence detection
//!
//! All output is deterministic and reproducible.

use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use crate::output::{self, OutputMode};
use crate::formats;

pub struct DriftArgs {
    pub file_a: PathBuf,
    pub file_b: PathBuf,
    pub mode: DiffMode,
    pub tolerance: f64,
    pub precision: usize,
    pub output: OutputMode,
    pub max_diffs: usize,
    pub delimiter: char,
    // Second-mode flags
    pub fail_on_diff: bool,
    pub fail_on_schema_diff: bool,
    pub summary_only: bool,
    pub stats_only: bool,
    pub report: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffMode {
    Auto,
    Text,
    Numeric,
    Csv,
    Jsonl,
}

impl Default for DriftArgs {
    fn default() -> Self {
        Self {
            file_a: PathBuf::new(),
            file_b: PathBuf::new(),
            mode: DiffMode::Auto,
            tolerance: 0.0,
            precision: 6,
            output: OutputMode::Color,
            max_diffs: 50,
            delimiter: ',',
            fail_on_diff: false,
            fail_on_schema_diff: false,
            summary_only: false,
            stats_only: false,
            report: None,
        }
    }
}

pub fn parse_args(args: &[String]) -> DriftArgs {
    let mut da = DriftArgs::default();
    let mut positionals = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--text" => da.mode = DiffMode::Text,
            "--numeric" => da.mode = DiffMode::Numeric,
            "--csv" => da.mode = DiffMode::Csv,
            "--jsonl" => da.mode = DiffMode::Jsonl,
            "--tolerance" | "--tol" | "-e" | "--threshold" => {
                i += 1;
                if i < args.len() {
                    da.tolerance = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("error: --tolerance requires a numeric argument");
                        process::exit(1);
                    });
                }
            }
            "--precision" => {
                i += 1;
                if i < args.len() { da.precision = args[i].parse().unwrap_or(6); }
            }
            "--max-diffs" => {
                i += 1;
                if i < args.len() { da.max_diffs = args[i].parse().unwrap_or(50); }
            }
            "-d" | "--delimiter" => {
                i += 1;
                if i < args.len() {
                    da.delimiter = match args[i].as_str() {
                        "tab" | "\\t" => '\t',
                        s if s.len() == 1 => s.chars().next().unwrap(),
                        _ => ',',
                    };
                }
            }
            "--plain" => da.output = OutputMode::Plain,
            "--json" => da.output = OutputMode::Json,
            "--color" => da.output = OutputMode::Color,
            "--fail-on-diff" => da.fail_on_diff = true,
            "--fail-on-schema-diff" => da.fail_on_schema_diff = true,
            "--summary-only" => da.summary_only = true,
            "--stats-only" => da.stats_only = true,
            "--report" => {
                i += 1;
                if i < args.len() {
                    da.report = Some(PathBuf::from(&args[i]));
                } else {
                    eprintln!("error: --report requires a file path argument");
                    process::exit(1);
                }
            }
            other if !other.starts_with('-') => positionals.push(other.to_string()),
            other => {
                eprintln!("error: unknown flag `{}` for `cjcl drift`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    if positionals.len() < 2 {
        eprintln!("error: `cjcl drift` requires two file arguments");
        process::exit(1);
    }
    da.file_a = PathBuf::from(&positionals[0]);
    da.file_b = PathBuf::from(&positionals[1]);
    da
}

pub fn run(args: &[String]) {
    let da = parse_args(args);

    let content_a = fs::read_to_string(&da.file_a).unwrap_or_else(|e| {
        eprintln!("error: could not read `{}`: {}", da.file_a.display(), e);
        process::exit(1);
    });
    let content_b = fs::read_to_string(&da.file_b).unwrap_or_else(|e| {
        eprintln!("error: could not read `{}`: {}", da.file_b.display(), e);
        process::exit(1);
    });

    let mode = if da.mode == DiffMode::Auto {
        auto_detect(&da.file_a, &content_a)
    } else {
        da.mode
    };

    match mode {
        DiffMode::Text | DiffMode::Auto => diff_text(&da, &content_a, &content_b),
        DiffMode::Numeric => diff_numeric(&da, &content_a, &content_b),
        DiffMode::Csv => diff_csv(&da, &content_a, &content_b),
        DiffMode::Jsonl => diff_jsonl(&da, &content_a, &content_b),
    }
}

fn auto_detect(path: &Path, _content: &str) -> DiffMode {
    match path.extension().and_then(|e| e.to_str()) {
        Some("csv") | Some("tsv") => DiffMode::Csv,
        Some("jsonl") | Some("ndjson") => DiffMode::Jsonl,
        _ => DiffMode::Text,
    }
}

// ── Shared diff result for report/summary/stats modes ───────────────

/// Unified diff result used for --report, --summary-only, --stats-only, and exit code logic.
struct TabularDiffResult {
    mode_label: String,
    rows_a: usize,
    rows_b: usize,
    columns_a: Vec<String>,
    columns_b: Vec<String>,
    cell_diffs: u64,
    nan_diffs: u64,
    max_deviation: f64,
    frobenius_norm: f64,
    numeric_cells: u64,
    mean_deviation: f64,
    diff_details: Vec<(usize, usize, String, String)>,
    schema_match: bool,
}

impl TabularDiffResult {
    fn is_identical(&self) -> bool {
        self.cell_diffs == 0 && self.nan_diffs == 0
    }

    fn to_json(&self, precision: usize) -> String {
        let mut out = String::from("{\n");
        out.push_str(&format!("  \"mode\": \"{}\",\n", self.mode_label));
        out.push_str(&format!("  \"rows_a\": {},\n", self.rows_a));
        out.push_str(&format!("  \"rows_b\": {},\n", self.rows_b));
        out.push_str(&format!("  \"columns_a\": [{}],\n", self.columns_a.iter()
            .map(|c| format!("\"{}\"", json_escape(c))).collect::<Vec<_>>().join(", ")));
        out.push_str(&format!("  \"columns_b\": [{}],\n", self.columns_b.iter()
            .map(|c| format!("\"{}\"", json_escape(c))).collect::<Vec<_>>().join(", ")));
        out.push_str(&format!("  \"schema_match\": {},\n", self.schema_match));
        out.push_str(&format!("  \"cell_differences\": {},\n", self.cell_diffs));
        out.push_str(&format!("  \"nan_differences\": {},\n", self.nan_diffs));
        out.push_str(&format!("  \"max_deviation\": {},\n", output::format_f64(self.max_deviation, precision)));
        out.push_str(&format!("  \"mean_deviation\": {},\n", output::format_f64(self.mean_deviation, precision)));
        out.push_str(&format!("  \"frobenius_norm\": {},\n", output::format_f64(self.frobenius_norm, precision)));
        out.push_str(&format!("  \"numeric_cells_compared\": {},\n", self.numeric_cells));
        out.push_str(&format!("  \"identical\": {},\n", self.is_identical()));

        // Diff details array
        out.push_str("  \"differences\": [\n");
        for (i, (row, col, va, vb)) in self.diff_details.iter().enumerate() {
            out.push_str(&format!("    {{\"row\": {}, \"col\": {}, \"a\": \"{}\", \"b\": \"{}\"}}",
                row, col, json_escape(va), json_escape(vb)));
            if i + 1 < self.diff_details.len() { out.push(','); }
            out.push('\n');
        }
        out.push_str("  ]\n");
        out.push('}');
        out
    }
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
     .replace('"', "\\\"")
     .replace('\n', "\\n")
     .replace('\r', "\\r")
     .replace('\t', "\\t")
}

/// Write report JSON to a file if --report was specified.
fn maybe_write_report(da: &DriftArgs, result: &TabularDiffResult) {
    if let Some(ref path) = da.report {
        let json = result.to_json(da.precision);
        fs::write(path, &json).unwrap_or_else(|e| {
            eprintln!("error: could not write report to `{}`: {}", path.display(), e);
            process::exit(1);
        });
        eprintln!("Report written to {}", path.display());
    }
}

/// Determine exit code based on --fail-on-diff and --fail-on-schema-diff flags.
fn compute_exit_code(da: &DriftArgs, result: &TabularDiffResult) -> Option<i32> {
    if da.fail_on_schema_diff && !result.schema_match {
        return Some(1);
    }
    if da.fail_on_diff && !result.is_identical() {
        return Some(1);
    }
    None
}

/// Render the summary table for tabular diff results.
fn render_summary_table(da: &DriftArgs, result: &TabularDiffResult) {
    let mut t = crate::table::Table::new(vec!["Metric", "Value"]);
    t.add_row_owned(vec!["Rows (A)".into(), format!("{}", result.rows_a)]);
    t.add_row_owned(vec!["Rows (B)".into(), format!("{}", result.rows_b)]);
    t.add_row_owned(vec!["Schema match".into(), format!("{}", result.schema_match)]);
    t.add_row_owned(vec!["Cell differences".into(), format!("{}", result.cell_diffs)]);
    t.add_row_owned(vec!["NaN divergences".into(), format!("{}", result.nan_diffs)]);
    t.add_row_owned(vec!["Max deviation".into(), output::format_f64(result.max_deviation, da.precision)]);
    t.add_row_owned(vec!["Mean deviation".into(), output::format_f64(result.mean_deviation, da.precision)]);
    t.add_row_owned(vec!["Frobenius norm".into(), output::format_f64(result.frobenius_norm, da.precision)]);
    eprint!("{}", t.render());
}

/// Render only statistical metrics.
fn render_stats_table(da: &DriftArgs, result: &TabularDiffResult) {
    let mut t = crate::table::Table::new(vec!["Statistic", "Value"]);
    t.add_row_owned(vec!["Max deviation".into(), output::format_f64(result.max_deviation, da.precision)]);
    t.add_row_owned(vec!["Mean deviation".into(), output::format_f64(result.mean_deviation, da.precision)]);
    t.add_row_owned(vec!["Frobenius norm".into(), output::format_f64(result.frobenius_norm, da.precision)]);
    t.add_row_owned(vec!["Numeric cells compared".into(), format!("{}", result.numeric_cells)]);
    t.add_row_owned(vec!["NaN divergences".into(), format!("{}", result.nan_diffs)]);
    eprint!("{}", t.render());
}

// ── Text diff ───────────────────────────────────────────────────────

fn diff_text(da: &DriftArgs, a: &str, b: &str) {
    let lines_a: Vec<&str> = a.lines().collect();
    let lines_b: Vec<&str> = b.lines().collect();

    let mut diffs = Vec::new();
    let max_lines = lines_a.len().max(lines_b.len());

    for i in 0..max_lines {
        let la = lines_a.get(i).copied();
        let lb = lines_b.get(i).copied();
        if la != lb {
            diffs.push((i + 1, la, lb));
            if diffs.len() >= da.max_diffs { break; }
        }
    }

    if da.output == OutputMode::Json {
        println!("{{");
        println!("  \"mode\": \"text\",");
        println!("  \"lines_a\": {},", lines_a.len());
        println!("  \"lines_b\": {},", lines_b.len());
        println!("  \"differences\": {},", diffs.len());
        println!("  \"identical\": {}", diffs.is_empty());
        println!("}}");
        return;
    }

    if diffs.is_empty() {
        eprintln!("{}", output::colorize(da.output, output::BOLD_GREEN, "identical"));
    } else {
        eprintln!("{} differences found (showing up to {})",
            diffs.len(), da.max_diffs);
        eprintln!();
        for (line, la, lb) in &diffs {
            let a_str = la.unwrap_or("<missing>");
            let b_str = lb.unwrap_or("<missing>");
            eprintln!("  line {}:", line);
            eprintln!("    {} {}", output::colorize(da.output, output::RED, "-"), a_str);
            eprintln!("    {} {}", output::colorize(da.output, output::GREEN, "+"), b_str);
        }
        process::exit(1);
    }
}

// ── Core tabular diff engine (shared by CSV and JSONL) ──────────────

fn diff_tabular(
    da: &DriftArgs,
    mode_label: &str,
    headers_a: &[String],
    headers_b: &[String],
    rows_a: &[Vec<String>],
    rows_b: &[Vec<String>],
) {
    // Check schema match
    let schema_match = headers_a == headers_b;

    // If --fail-on-schema-diff and schemas differ, report and exit immediately
    if da.fail_on_schema_diff && !schema_match {
        if da.output == OutputMode::Json {
            println!("{{");
            println!("  \"mode\": \"{}\",", mode_label);
            println!("  \"schema_match\": false,");
            println!("  \"columns_a\": [{}],", headers_a.iter()
                .map(|c| format!("\"{}\"", json_escape(c))).collect::<Vec<_>>().join(", "));
            println!("  \"columns_b\": [{}]", headers_b.iter()
                .map(|c| format!("\"{}\"", json_escape(c))).collect::<Vec<_>>().join(", "));
            println!("}}");
        } else {
            eprintln!("Schema mismatch:");
            eprintln!("  A columns: {:?}", headers_a);
            eprintln!("  B columns: {:?}", headers_b);
        }
        process::exit(1);
    }

    let mut cell_diffs = 0u64;
    let mut nan_diffs = 0u64;
    let mut max_deviation = 0.0f64;
    let mut sum_sq_dev = 0.0f64;
    let mut numeric_cells = 0u64;
    let mut sum_deviation = 0.0f64;
    let max_rows = rows_a.len().max(rows_b.len());
    let mut diff_details: Vec<(usize, usize, String, String)> = Vec::new();

    for row_idx in 0..max_rows {
        let ra = rows_a.get(row_idx);
        let rb = rows_b.get(row_idx);
        let max_cols = ra.map(|r| r.len()).unwrap_or(0).max(rb.map(|r| r.len()).unwrap_or(0));

        for col_idx in 0..max_cols {
            let va = ra.and_then(|r| r.get(col_idx)).map(|s| s.as_str()).unwrap_or("");
            let vb = rb.and_then(|r| r.get(col_idx)).map(|s| s.as_str()).unwrap_or("");

            if va == vb { continue; }

            // Try numeric comparison
            if let (Ok(fa), Ok(fb)) = (va.parse::<f64>(), vb.parse::<f64>()) {
                if fa.is_nan() != fb.is_nan() {
                    nan_diffs += 1;
                }
                if !fa.is_nan() && !fb.is_nan() {
                    let dev = (fa - fb).abs();
                    if dev > da.tolerance {
                        cell_diffs += 1;
                        if dev > max_deviation { max_deviation = dev; }
                        sum_sq_dev += dev * dev;
                        sum_deviation += dev;
                        numeric_cells += 1;
                    }
                }
            } else {
                cell_diffs += 1;
            }

            if diff_details.len() < da.max_diffs {
                diff_details.push((row_idx + 1, col_idx + 1, va.to_string(), vb.to_string()));
            }
        }
    }

    let frobenius = sum_sq_dev.sqrt();
    let mean_deviation = if numeric_cells > 0 {
        sum_deviation / numeric_cells as f64
    } else {
        0.0
    };

    let result = TabularDiffResult {
        mode_label: mode_label.to_string(),
        rows_a: rows_a.len(),
        rows_b: rows_b.len(),
        columns_a: headers_a.to_vec(),
        columns_b: headers_b.to_vec(),
        cell_diffs,
        nan_diffs,
        max_deviation,
        frobenius_norm: frobenius,
        numeric_cells,
        mean_deviation,
        diff_details,
        schema_match,
    };

    // Write report file if requested
    maybe_write_report(da, &result);

    // JSON output mode
    if da.output == OutputMode::Json {
        println!("{}", result.to_json(da.precision));
        if let Some(exit) = compute_exit_code(da, &result) {
            process::exit(exit);
        }
        if !result.is_identical() {
            process::exit(1);
        }
        return;
    }

    // --stats-only: show only statistical metrics
    if da.stats_only {
        render_stats_table(da, &result);
        if let Some(exit) = compute_exit_code(da, &result) {
            process::exit(exit);
        }
        if !result.is_identical() {
            process::exit(1);
        }
        return;
    }

    // --summary-only: show summary table but skip diff details
    if da.summary_only {
        render_summary_table(da, &result);
        if result.is_identical() {
            eprintln!("{}", output::colorize(da.output, output::BOLD_GREEN, "identical"));
        }
        if let Some(exit) = compute_exit_code(da, &result) {
            process::exit(exit);
        }
        if !result.is_identical() {
            process::exit(1);
        }
        return;
    }

    // Default: full output with summary table + diff details
    render_summary_table(da, &result);

    if result.is_identical() {
        eprintln!("{}", output::colorize(da.output, output::BOLD_GREEN, "identical"));
        if let Some(exit) = compute_exit_code(da, &result) {
            process::exit(exit);
        }
    } else {
        if !result.diff_details.is_empty() {
            eprintln!("\nFirst differences:");
            for (row, col, va, vb) in result.diff_details.iter().take(10) {
                eprintln!("  [{},{}]: {:?} -> {:?}", row, col, va, vb);
            }
        }
        if let Some(exit) = compute_exit_code(da, &result) {
            process::exit(exit);
        }
        process::exit(1);
    }
}

// ── CSV diff ────────────────────────────────────────────────────────

fn diff_csv(da: &DriftArgs, a: &str, b: &str) {
    let parse_csv = |content: &str| -> Vec<Vec<String>> {
        content.lines()
            .map(|line| line.split(da.delimiter).map(|s| s.trim().to_string()).collect())
            .collect()
    };

    let rows_a = parse_csv(a);
    let rows_b = parse_csv(b);

    // Extract headers (first row) if present, otherwise generate column names
    let (headers_a, data_a) = if rows_a.is_empty() {
        (Vec::new(), Vec::new())
    } else {
        (rows_a[0].clone(), rows_a[1..].to_vec())
    };
    let (headers_b, data_b) = if rows_b.is_empty() {
        (Vec::new(), Vec::new())
    } else {
        (rows_b[0].clone(), rows_b[1..].to_vec())
    };

    // Always use structured mode: first row = header, rest = data.
    // This ensures consistent row counts regardless of which flags are used.
    diff_tabular(da, "csv", &headers_a, &headers_b, &data_a, &data_b);
}

// ── JSONL diff ──────────────────────────────────────────────────────

fn diff_jsonl(da: &DriftArgs, a: &str, b: &str) {
    let table_a = formats::load_jsonl(a);
    let table_b = formats::load_jsonl(b);

    diff_tabular(
        da,
        "jsonl",
        &table_a.headers,
        &table_b.headers,
        &table_a.rows,
        &table_b.rows,
    );
}

// ── Numeric diff ────────────────────────────────────────────────────

fn diff_numeric(da: &DriftArgs, a: &str, b: &str) {
    // Parse as flat lists of numbers (one per line or whitespace-separated)
    let parse_nums = |content: &str| -> Vec<f64> {
        content.split_whitespace()
            .filter_map(|s| s.parse::<f64>().ok())
            .collect()
    };

    let nums_a = parse_nums(a);
    let nums_b = parse_nums(b);

    if nums_a.len() != nums_b.len() {
        eprintln!("Shape mismatch: {} values vs {} values", nums_a.len(), nums_b.len());
        if da.output == OutputMode::Json {
            println!("{{\"mode\": \"numeric\", \"shape_mismatch\": true, \"len_a\": {}, \"len_b\": {}}}", nums_a.len(), nums_b.len());
        }
        process::exit(1);
    }

    let mut max_dev = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut diffs = 0u64;
    let mut nan_diffs = 0u64;
    let mut sum_dev = 0.0f64;
    let mut numeric_count = 0u64;

    for (_i, (a, b)) in nums_a.iter().zip(&nums_b).enumerate() {
        if a.is_nan() != b.is_nan() { nan_diffs += 1; continue; }
        if a.is_nan() && b.is_nan() { continue; }
        let dev = (a - b).abs();
        sum_sq += dev * dev;
        if dev > da.tolerance {
            diffs += 1;
            sum_dev += dev;
            numeric_count += 1;
            if dev > max_dev { max_dev = dev; }
        }
    }

    let frobenius = sum_sq.sqrt();
    let mean_dev = if numeric_count > 0 { sum_dev / numeric_count as f64 } else { 0.0 };
    let is_identical = diffs == 0 && nan_diffs == 0;

    // Handle --report for numeric mode
    if let Some(ref path) = da.report {
        let mut out = String::from("{\n");
        out.push_str("  \"mode\": \"numeric\",\n");
        out.push_str(&format!("  \"elements\": {},\n", nums_a.len()));
        out.push_str(&format!("  \"differences\": {},\n", diffs));
        out.push_str(&format!("  \"nan_differences\": {},\n", nan_diffs));
        out.push_str(&format!("  \"max_deviation\": {},\n", output::format_f64(max_dev, da.precision)));
        out.push_str(&format!("  \"mean_deviation\": {},\n", output::format_f64(mean_dev, da.precision)));
        out.push_str(&format!("  \"frobenius_norm\": {},\n", output::format_f64(frobenius, da.precision)));
        out.push_str(&format!("  \"identical\": {}\n", is_identical));
        out.push('}');
        fs::write(path, &out).unwrap_or_else(|e| {
            eprintln!("error: could not write report to `{}`: {}", path.display(), e);
            process::exit(1);
        });
        eprintln!("Report written to {}", path.display());
    }

    if da.output == OutputMode::Json {
        println!("{{");
        println!("  \"mode\": \"numeric\",");
        println!("  \"elements\": {},", nums_a.len());
        println!("  \"differences\": {},", diffs);
        println!("  \"nan_differences\": {},", nan_diffs);
        println!("  \"max_deviation\": {},", output::format_f64(max_dev, da.precision));
        println!("  \"mean_deviation\": {},", output::format_f64(mean_dev, da.precision));
        println!("  \"frobenius_norm\": {},", output::format_f64(frobenius, da.precision));
        println!("  \"identical\": {}", is_identical);
        println!("}}");
        if da.fail_on_diff && !is_identical { process::exit(1); }
        return;
    }

    if da.stats_only {
        let mut t = crate::table::Table::new(vec!["Statistic", "Value"]);
        t.add_row_owned(vec!["Max deviation".into(), output::format_f64(max_dev, da.precision)]);
        t.add_row_owned(vec!["Mean deviation".into(), output::format_f64(mean_dev, da.precision)]);
        t.add_row_owned(vec!["Frobenius norm".into(), output::format_f64(frobenius, da.precision)]);
        t.add_row_owned(vec!["Numeric elements".into(), format!("{}", nums_a.len())]);
        t.add_row_owned(vec!["NaN divergences".into(), format!("{}", nan_diffs)]);
        eprint!("{}", t.render());
        if da.fail_on_diff && !is_identical { process::exit(1); }
        if !is_identical { process::exit(1); }
        return;
    }

    let mut t = crate::table::Table::new(vec!["Metric", "Value"]);
    t.add_row_owned(vec!["Elements".into(), format!("{}", nums_a.len())]);
    t.add_row_owned(vec!["Differences".into(), format!("{}", diffs)]);
    t.add_row_owned(vec!["NaN divergences".into(), format!("{}", nan_diffs)]);
    t.add_row_owned(vec!["Max deviation".into(), output::format_f64(max_dev, da.precision)]);
    t.add_row_owned(vec!["Mean deviation".into(), output::format_f64(mean_dev, da.precision)]);
    t.add_row_owned(vec!["Frobenius norm".into(), output::format_f64(frobenius, da.precision)]);
    eprint!("{}", t.render());

    if is_identical {
        eprintln!("{}", output::colorize(da.output, output::BOLD_GREEN, "identical"));
        if da.fail_on_diff { /* identical, no exit */ }
    } else {
        if da.fail_on_diff { process::exit(1); }
        process::exit(1);
    }
}

// ── Help ────────────────────────────────────────────────────────────

pub fn print_help() {
    eprintln!("cjcl drift -- Mathematical and data diff engine");
    eprintln!();
    eprintln!("Usage: cjcl drift <file_a> <file_b> [flags]");
    eprintln!();
    eprintln!("Modes:");
    eprintln!("  --text                Line-by-line text diff");
    eprintln!("  --numeric             Numeric element-wise diff");
    eprintln!("  --csv                 CSV cell-by-cell diff");
    eprintln!("  --jsonl               JSONL cell-by-cell diff (parsed to tabular)");
    eprintln!("  (auto-detected from extension if not specified)");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  -e, --tolerance <N>   Numeric tolerance (default: 0.0)");
    eprintln!("  --threshold <N>       Alias for --tolerance");
    eprintln!("  --precision <N>       Output decimal precision (default: 6)");
    eprintln!("  --max-diffs <N>       Max differences to show (default: 50)");
    eprintln!("  -d, --delimiter <c>   CSV delimiter (default: ,)");
    eprintln!("  --plain               Plain text output");
    eprintln!("  --json                JSON output");
    eprintln!("  --color               Color output (default)");
    eprintln!();
    eprintln!("CI / gating flags:");
    eprintln!("  --fail-on-diff        Exit code 1 if ANY differences found");
    eprintln!("  --fail-on-schema-diff Exit code 1 if column names differ");
    eprintln!("  --summary-only        Only show summary metrics, skip diff details");
    eprintln!("  --stats-only          Only show statistical metrics");
    eprintln!("  --report <file>       Write full diff report to a JSON file");
}
