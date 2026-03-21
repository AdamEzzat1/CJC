//! `cjc drift` — Mathematical and data diff engine.
//!
//! Compares two files or data sources and computes:
//! - Text diff (line-by-line)
//! - Structured data diff (CSV column comparison)
//! - Tensor/numeric diff (Frobenius norm, max deviation, element-wise)
//! - Shape mismatch detection
//! - NaN divergence detection
//!
//! All output is deterministic and reproducible.

use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use crate::output::{self, OutputMode};

pub struct DriftArgs {
    pub file_a: PathBuf,
    pub file_b: PathBuf,
    pub mode: DiffMode,
    pub tolerance: f64,
    pub precision: usize,
    pub output: OutputMode,
    pub max_diffs: usize,
    pub delimiter: char,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffMode {
    Auto,
    Text,
    Numeric,
    Csv,
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
            "--tolerance" | "--tol" | "-e" => {
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
            other if !other.starts_with('-') => positionals.push(other.to_string()),
            other => {
                eprintln!("error: unknown flag `{}` for `cjc drift`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    if positionals.len() < 2 {
        eprintln!("error: `cjc drift` requires two file arguments");
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
    }
}

fn auto_detect(path: &Path, _content: &str) -> DiffMode {
    match path.extension().and_then(|e| e.to_str()) {
        Some("csv") | Some("tsv") => DiffMode::Csv,
        _ => DiffMode::Text,
    }
}

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

fn diff_csv(da: &DriftArgs, a: &str, b: &str) {
    let parse_csv = |content: &str| -> Vec<Vec<String>> {
        content.lines()
            .map(|line| line.split(da.delimiter).map(|s| s.trim().to_string()).collect())
            .collect()
    };

    let rows_a = parse_csv(a);
    let rows_b = parse_csv(b);

    let mut cell_diffs = 0u64;
    let mut nan_diffs = 0u64;
    let mut max_deviation = 0.0f64;
    let mut sum_sq_dev = 0.0f64; // For Frobenius norm
    let mut numeric_cells = 0u64;
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

    if da.output == OutputMode::Json {
        println!("{{");
        println!("  \"mode\": \"csv\",");
        println!("  \"rows_a\": {},", rows_a.len());
        println!("  \"rows_b\": {},", rows_b.len());
        println!("  \"cell_differences\": {},", cell_diffs);
        println!("  \"nan_differences\": {},", nan_diffs);
        println!("  \"max_deviation\": {},", output::format_f64(max_deviation, da.precision));
        println!("  \"frobenius_norm\": {},", output::format_f64(frobenius, da.precision));
        println!("  \"identical\": {}", cell_diffs == 0 && nan_diffs == 0);
        println!("}}");
        return;
    }

    let mut t = crate::table::Table::new(vec!["Metric", "Value"]);
    t.add_row_owned(vec!["Rows (A)".into(), format!("{}", rows_a.len())]);
    t.add_row_owned(vec!["Rows (B)".into(), format!("{}", rows_b.len())]);
    t.add_row_owned(vec!["Cell differences".into(), format!("{}", cell_diffs)]);
    t.add_row_owned(vec!["NaN divergences".into(), format!("{}", nan_diffs)]);
    t.add_row_owned(vec!["Max deviation".into(), output::format_f64(max_deviation, da.precision)]);
    t.add_row_owned(vec!["Frobenius norm".into(), output::format_f64(frobenius, da.precision)]);
    eprint!("{}", t.render());

    if cell_diffs == 0 && nan_diffs == 0 {
        eprintln!("{}", output::colorize(da.output, output::BOLD_GREEN, "identical"));
    } else {
        if !diff_details.is_empty() {
            eprintln!("\nFirst differences:");
            for (row, col, va, vb) in diff_details.iter().take(10) {
                eprintln!("  [{},{}]: {:?} → {:?}", row, col, va, vb);
            }
        }
        process::exit(1);
    }
}

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

    for (i, (a, b)) in nums_a.iter().zip(&nums_b).enumerate() {
        if a.is_nan() != b.is_nan() { nan_diffs += 1; continue; }
        if a.is_nan() && b.is_nan() { continue; }
        let dev = (a - b).abs();
        sum_sq += dev * dev;
        if dev > da.tolerance {
            diffs += 1;
            if dev > max_dev { max_dev = dev; }
        }
    }

    let frobenius = sum_sq.sqrt();

    if da.output == OutputMode::Json {
        println!("{{");
        println!("  \"mode\": \"numeric\",");
        println!("  \"elements\": {},", nums_a.len());
        println!("  \"differences\": {},", diffs);
        println!("  \"nan_differences\": {},", nan_diffs);
        println!("  \"max_deviation\": {},", output::format_f64(max_dev, da.precision));
        println!("  \"frobenius_norm\": {},", output::format_f64(frobenius, da.precision));
        println!("  \"identical\": {}", diffs == 0 && nan_diffs == 0);
        println!("}}");
        return;
    }

    let mut t = crate::table::Table::new(vec!["Metric", "Value"]);
    t.add_row_owned(vec!["Elements".into(), format!("{}", nums_a.len())]);
    t.add_row_owned(vec!["Differences".into(), format!("{}", diffs)]);
    t.add_row_owned(vec!["NaN divergences".into(), format!("{}", nan_diffs)]);
    t.add_row_owned(vec!["Max deviation".into(), output::format_f64(max_dev, da.precision)]);
    t.add_row_owned(vec!["Frobenius norm".into(), output::format_f64(frobenius, da.precision)]);
    eprint!("{}", t.render());

    if diffs == 0 && nan_diffs == 0 {
        eprintln!("{}", output::colorize(da.output, output::BOLD_GREEN, "identical"));
    } else {
        process::exit(1);
    }
}

pub fn print_help() {
    eprintln!("cjc drift — Mathematical and data diff engine");
    eprintln!();
    eprintln!("Usage: cjc drift <file_a> <file_b> [flags]");
    eprintln!();
    eprintln!("Modes:");
    eprintln!("  --text                Line-by-line text diff");
    eprintln!("  --numeric             Numeric element-wise diff");
    eprintln!("  --csv                 CSV cell-by-cell diff");
    eprintln!("  (auto-detected from extension if not specified)");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  -e, --tolerance <N>   Numeric tolerance (default: 0.0)");
    eprintln!("  --precision <N>       Output decimal precision (default: 6)");
    eprintln!("  --max-diffs <N>       Max differences to show (default: 50)");
    eprintln!("  -d, --delimiter <c>   CSV delimiter (default: ,)");
    eprintln!("  --plain               Plain text output");
    eprintln!("  --json                JSON output");
    eprintln!("  --color               Color output (default)");
}
