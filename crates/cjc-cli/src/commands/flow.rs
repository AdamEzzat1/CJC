//! `cjc flow` — Streaming computation engine.
//!
//! Processes CSV/TSV data streams with O(ncols) memory using:
//! - Kahan summation for numeric stability
//! - Single-pass streaming aggregates (sum, mean, min, max, count, var, std)
//! - Deterministic output regardless of data size
//!
//! Designed for massive datasets that cannot fit in memory.

use std::fs;
use std::io::{self, BufRead};
use std::path::PathBuf;
use std::process;
use crate::output::{self, OutputMode};

/// Parsed arguments for `cjc flow`.
pub struct FlowArgs {
    pub file: Option<PathBuf>,
    pub delimiter: char,
    pub has_header: bool,
    pub ops: Vec<AggOp>,
    pub columns: Option<Vec<String>>,
    pub output: OutputMode,
    pub precision: usize,
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
            other if !other.starts_with('-') => fa.file = Some(PathBuf::from(other)),
            other => {
                eprintln!("error: unknown flag `{}` for `cjc flow`", other);
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
}

/// Entry point for `cjc flow`.
pub fn run(args: &[String]) {
    let fa = parse_args(args);

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

    // Filter to requested columns
    let active_accums: Vec<&ColumnAccum> = if let Some(ref cols) = fa.columns {
        accums.iter().filter(|a| cols.contains(&a.name)).collect()
    } else {
        accums.iter().collect()
    };

    // Output
    match fa.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"rows_processed\": {},", rows_processed);
            println!("  \"columns\": [");
            for (i, acc) in active_accums.iter().enumerate() {
                print!("    {{\"name\": \"{}\"", acc.name);
                for op in &fa.ops {
                    print!(", \"{}\": {}", op.label(), acc.get(*op, fa.precision));
                }
                print!("}}");
                if i + 1 < active_accums.len() { print!(","); }
                println!();
            }
            println!("  ]");
            println!("}}");
        }
        _ => {
            let mut headers = vec!["Column"];
            let op_labels: Vec<&str> = fa.ops.iter().map(|o| o.label()).collect();
            headers.extend(op_labels.iter());

            let mut t = crate::table::Table::new(headers);
            for acc in &active_accums {
                let mut row = vec![acc.name.clone()];
                for op in &fa.ops {
                    row.push(acc.get(*op, fa.precision));
                }
                t.add_row_owned(row);
            }
            print!("{}", t.render());
            eprintln!("({} rows processed)", rows_processed);
        }
    }
}

pub fn print_help() {
    eprintln!("cjc flow — Streaming computation engine");
    eprintln!();
    eprintln!("Usage: cjc flow [file.csv] [flags]");
    eprintln!("       cat data.csv | cjc flow [flags]");
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
}
