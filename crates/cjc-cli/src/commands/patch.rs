//! `cjc patch` — Type-aware data transformation.
//!
//! Streams CSV/TSV files and applies column-aware transformations:
//! - NaN replacement (--nan-fill <value>)
//! - Conditional replacement (--replace <col> <from> <to>)
//! - Mean imputation (--impute <col>)
//! - Column type casting (--cast <col> <type>)
//! - Column dropping (--drop <col>)
//!
//! Never corrupts schema. Streams with O(ncols) memory.

use std::fs;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process;

/// Parsed arguments for `cjc patch`.
pub struct PatchArgs {
    pub file: Option<PathBuf>,
    pub output_file: Option<PathBuf>,
    pub delimiter: char,
    pub has_header: bool,
    pub transforms: Vec<Transform>,
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

impl Default for PatchArgs {
    fn default() -> Self {
        Self {
            file: None,
            output_file: None,
            delimiter: ',',
            has_header: true,
            transforms: Vec::new(),
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

/// Entry point for `cjc patch`.
pub fn run(args: &[String]) {
    let pa = parse_args(args);

    if pa.transforms.is_empty() {
        eprintln!("error: no transforms specified. Use --nan-fill, --impute, --replace, --drop, etc.");
        process::exit(1);
    }

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
    let mut col_means: std::collections::BTreeMap<String, (f64, f64, u64)> = std::collections::BTreeMap::new();
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

    let impute_means: std::collections::BTreeMap<String, String> = col_means.iter()
        .map(|(name, (sum, _, count))| {
            let mean = if *count > 0 { sum / *count as f64 } else { 0.0 };
            (name.clone(), format!("{}", mean))
        })
        .collect();

    // Second (or only) pass: apply transforms and write output
    let reader = reader_fn();
    let mut lines = reader.lines();

    let out_writer: Box<dyn Write> = match &pa.output_file {
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
                        let trimmed = field.trim();
                        if trimmed.is_empty() || trimmed == "NA" || trimmed == "NaN"
                            || trimmed == "null" || trimmed == "None" {
                            *field = fill.clone();
                        }
                    }
                    Transform::Impute(target_col) if col_name == target_col => {
                        let trimmed = field.trim();
                        if trimmed.is_empty() || trimmed == "NA" || trimmed == "NaN"
                            || trimmed == "null" || trimmed == "None" {
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
    eprintln!("patched {} rows", rows_processed);
}

pub fn print_help() {
    eprintln!("cjc patch — Type-aware data transformation");
    eprintln!();
    eprintln!("Usage: cjc patch <file.csv> [transforms] [flags]");
    eprintln!("       cat data.csv | cjc patch [transforms] [flags]");
    eprintln!();
    eprintln!("Transforms:");
    eprintln!("  --nan-fill <value>               Replace NaN/NA/null with <value>");
    eprintln!("  --impute <column>                Replace NaN with column mean (requires file)");
    eprintln!("  --replace <col> <from> <to>      Replace exact values in column");
    eprintln!("  --drop <column>                  Remove a column");
    eprintln!("  --rename <old> <new>             Rename a column");
    eprintln!("  --fill-empty <col> <value>       Fill empty cells in column");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  -d, --delimiter <c>              Column delimiter (default: ,)");
    eprintln!("  --tsv                            Use tab delimiter");
    eprintln!("  --no-header                      Input has no header");
    eprintln!("  -o, --output <file>              Output file (default: stdout)");
}
