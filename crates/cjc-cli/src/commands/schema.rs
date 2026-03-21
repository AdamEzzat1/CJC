//! `cjc schema` — Schema inference and reporting for datasets.
//!
//! Infers column types, null counts, unique value counts, and basic statistics
//! from CSV/TSV data. Output is always deterministic with stable column ordering.

use std::fs;
use std::path::Path;
use std::process;
use crate::output::{self, OutputMode};

pub struct SchemaArgs {
    pub file: String,
    pub delimiter: char,
    pub has_header: bool,
    pub sample_rows: Option<usize>,
    pub output: OutputMode,
    pub show_uniques: bool,
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
            other if !other.starts_with('-') => sa.file = other.to_string(),
            other => {
                eprintln!("error: unknown flag `{}` for `cjc schema`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    if sa.file.is_empty() {
        eprintln!("error: `cjc schema` requires a file argument");
        process::exit(1);
    }
    sa
}

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

pub fn run(args: &[String]) {
    let sa = parse_args(args);
    let path = Path::new(&sa.file);

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

    let mut columns: Vec<ColumnSchema> = headers.iter().map(|h| ColumnSchema {
        name: h.clone(),
        inferred_type: "unknown",
        null_count: 0, total_count: 0,
        unique_values: std::collections::BTreeSet::new(),
        numeric_count: 0, string_count: 0, bool_count: 0,
        int_count: 0, float_count: 0,
        min: f64::INFINITY, max: f64::NEG_INFINITY,
    }).collect();

    for line in &data_lines[..max_rows] {
        let fields: Vec<&str> = line.split(sa.delimiter).collect();
        for (ci, col) in columns.iter_mut().enumerate() {
            let val = fields.get(ci).map(|s| s.trim()).unwrap_or("");
            col.total_count += 1;

            if val.is_empty() || val == "NA" || val == "NaN" || val == "null" || val == "None" {
                col.null_count += 1;
                continue;
            }

            if sa.show_uniques && col.unique_values.len() < 1000 {
                col.unique_values.insert(val.to_string());
            }

            if val == "true" || val == "false" {
                col.bool_count += 1;
            } else if let Ok(v) = val.parse::<i64>() {
                col.int_count += 1;
                col.numeric_count += 1;
                let vf = v as f64;
                if vf < col.min { col.min = vf; }
                if vf > col.max { col.max = vf; }
            } else if let Ok(v) = val.parse::<f64>() {
                if !v.is_nan() {
                    col.float_count += 1;
                    col.numeric_count += 1;
                    if v < col.min { col.min = v; }
                    if v > col.max { col.max = v; }
                } else {
                    col.null_count += 1;
                }
            } else {
                col.string_count += 1;
            }
        }
    }

    // Infer types
    for col in &mut columns {
        let non_null = col.total_count - col.null_count;
        if non_null == 0 {
            col.inferred_type = "null";
        } else if col.bool_count == non_null {
            col.inferred_type = "bool";
        } else if col.int_count == non_null {
            col.inferred_type = "int";
        } else if col.float_count > 0 && col.string_count == 0 && col.bool_count == 0 {
            col.inferred_type = "float";
        } else if col.numeric_count == non_null {
            col.inferred_type = "numeric";
        } else if col.string_count == non_null {
            col.inferred_type = "string";
        } else {
            col.inferred_type = "mixed";
        }
    }

    // Output
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

            let mut t = crate::table::Table::new(headers_vec);
            for col in &columns {
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
                t.add_row_owned(row);
            }
            eprint!("{}", t.render());
        }
    }
}

pub fn print_help() {
    eprintln!("cjc schema — Schema inference and reporting");
    eprintln!();
    eprintln!("Usage: cjc schema <file.csv> [flags]");
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
}
