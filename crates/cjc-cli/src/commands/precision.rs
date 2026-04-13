//! `cjcl precision` — Precision analysis for numeric output.
//!
//! Runs a CJC program, captures numeric output lines, and analyzes
//! the precision characteristics by comparing f64 values against
//! f32-truncated approximations. Reports relative error per line.

use std::fs;
use std::process;
use crate::output::{self, OutputMode};

pub struct PrecisionArgs {
    pub file: String,
    pub seed: u64,
    pub output: OutputMode,
    pub epsilon: f64,
    pub fail_on_instability: bool,
    pub report: Option<String>,
    pub summary_only: bool,
}

impl Default for PrecisionArgs {
    fn default() -> Self {
        Self {
            file: String::new(),
            seed: 42,
            output: OutputMode::Color,
            epsilon: 1.1920929e-7, // f32::EPSILON as f64
            fail_on_instability: false,
            report: None,
            summary_only: false,
        }
    }
}

pub fn parse_args(args: &[String]) -> PrecisionArgs {
    let mut pa = PrecisionArgs::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                if i < args.len() { pa.seed = args[i].parse().unwrap_or(42); }
            }
            "--plain" => pa.output = OutputMode::Plain,
            "--json" => pa.output = OutputMode::Json,
            "--color" => pa.output = OutputMode::Color,
            "--epsilon" => {
                i += 1;
                if i < args.len() {
                    match args[i].parse::<f64>() {
                        Ok(v) if v > 0.0 => pa.epsilon = v,
                        _ => {
                            eprintln!("error: --epsilon requires a positive numeric argument");
                            process::exit(1);
                        }
                    }
                } else {
                    eprintln!("error: --epsilon requires a numeric argument");
                    process::exit(1);
                }
            }
            "--fail-on-instability" => pa.fail_on_instability = true,
            "--report" => {
                i += 1;
                if i < args.len() {
                    pa.report = Some(args[i].clone());
                } else {
                    eprintln!("error: --report requires a file argument");
                    process::exit(1);
                }
            }
            "--summary-only" => pa.summary_only = true,
            other if !other.starts_with('-') => pa.file = other.to_string(),
            other => {
                eprintln!("error: unknown flag `{}` for `cjcl precision`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    if pa.file.is_empty() {
        eprintln!("error: `cjcl precision` requires a .cjcl file argument");
        process::exit(1);
    }
    pa
}

struct PrecisionEntry {
    line_num: usize,
    f64_val: f64,
    f32_val: f64,
    rel_error: f64,
}

fn write_json_report(
    path: &str,
    filename: &str,
    seed: u64,
    epsilon: f64,
    entries: &[PrecisionEntry],
    max_rel_error: f64,
    stable: bool,
) {
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"file\": \"{}\",\n", filename));
    json.push_str(&format!("  \"seed\": {},\n", seed));
    json.push_str(&format!("  \"epsilon\": {},\n", format_sci(epsilon)));
    json.push_str(&format!("  \"numeric_values\": {},\n", entries.len()));
    json.push_str(&format!("  \"max_relative_error\": {},\n", format_sci(max_rel_error)));
    json.push_str(&format!("  \"stable\": {},\n", stable));
    json.push_str("  \"entries\": [\n");
    for (i, e) in entries.iter().enumerate() {
        json.push_str(&format!(
            "    {{\"line\": {}, \"f64\": {}, \"f32\": {}, \"rel_error\": {}}}",
            e.line_num,
            output::format_f64(e.f64_val, 12),
            output::format_f64(e.f32_val, 12),
            format_sci(e.rel_error)));
        if i + 1 < entries.len() { json.push(','); }
        json.push('\n');
    }
    json.push_str("  ]\n");
    json.push_str("}\n");

    if let Err(e) = fs::write(path, &json) {
        eprintln!("error: could not write report to `{}`: {}", path, e);
        process::exit(1);
    }
}

pub fn run(args: &[String]) {
    let pa = parse_args(args);

    let source = match fs::read_to_string(&pa.file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read `{}`: {}", pa.file, e);
            process::exit(1);
        }
    };

    let filename = pa.file.replace('\\', "/");

    let (program, diags) = cjc_parser::parse_source(&source);
    if diags.has_errors() {
        let rendered = diags.render_all_color(&source, &filename, pa.output.use_color());
        eprint!("{}", rendered);
        process::exit(1);
    }

    // Execute
    let mut interpreter = cjc_eval::Interpreter::new(pa.seed);
    if let Err(e) = interpreter.exec(&program) {
        eprintln!("error: execution failed: {}", e);
        process::exit(1);
    }

    // Analyze numeric output lines
    let mut entries: Vec<PrecisionEntry> = Vec::new();

    for (idx, line) in interpreter.output.iter().enumerate() {
        // Try to parse each whitespace-separated token as f64
        for token in line.split_whitespace() {
            if let Ok(f64_val) = token.parse::<f64>() {
                if f64_val.is_nan() || f64_val.is_infinite() {
                    continue;
                }
                let f32_val = f64_val as f32 as f64;
                let rel_error = if f64_val.abs() > 0.0 {
                    ((f64_val - f32_val) / f64_val).abs()
                } else {
                    0.0
                };
                entries.push(PrecisionEntry {
                    line_num: idx + 1,
                    f64_val,
                    f32_val,
                    rel_error,
                });
            }
        }
    }

    let max_rel_error = entries.iter()
        .map(|e| e.rel_error)
        .fold(0.0f64, f64::max);
    let stable = max_rel_error < pa.epsilon;
    let unstable_count = entries.iter().filter(|e| e.rel_error >= pa.epsilon).count();

    // Save report if requested
    if let Some(ref report_path) = pa.report {
        write_json_report(report_path, &filename, pa.seed, pa.epsilon, &entries, max_rel_error, stable);
    }

    match pa.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file\": \"{}\",", filename);
            println!("  \"seed\": {},", pa.seed);
            println!("  \"epsilon\": {},", format_sci(pa.epsilon));
            println!("  \"numeric_values\": {},", entries.len());
            println!("  \"max_relative_error\": {},", format_sci(max_rel_error));
            println!("  \"stable\": {},", stable);
            println!("  \"entries\": [");
            if !pa.summary_only {
                for (i, e) in entries.iter().enumerate() {
                    print!("    {{\"line\": {}, \"f64\": {}, \"f32\": {}, \"rel_error\": {}}}",
                        e.line_num,
                        output::format_f64(e.f64_val, 12),
                        output::format_f64(e.f32_val, 12),
                        format_sci(e.rel_error));
                    if i + 1 < entries.len() { print!(","); }
                    println!();
                }
            }
            println!("  ]");
            println!("}}");
        }
        _ => {
            eprintln!("{} Precision analysis for `{}` (seed={}, epsilon={})",
                output::colorize(pa.output, output::BOLD_CYAN, "[precision]"),
                filename, pa.seed, format_sci(pa.epsilon));
            eprintln!();

            if entries.is_empty() {
                eprintln!("No numeric output found.");
                return;
            }

            if !pa.summary_only {
                let mut t = crate::table::Table::new(vec!["Line", "f64", "f32", "Rel. error"]);
                for e in entries.iter().take(50) {
                    t.add_row_owned(vec![
                        format!("{}", e.line_num),
                        output::format_f64(e.f64_val, 12),
                        output::format_f64(e.f32_val, 12),
                        format_sci(e.rel_error),
                    ]);
                }
                eprint!("{}", t.render());

                if entries.len() > 50 {
                    eprintln!("... ({} more entries)", entries.len() - 50);
                }

                eprintln!();
            }

            // Summary stats
            let mut st = crate::table::Table::new(vec!["Metric", "Value"]);
            st.add_row_owned(vec!["Numeric values".into(), format!("{}", entries.len())]);
            st.add_row_owned(vec!["Max relative error".into(), format_sci(max_rel_error)]);
            st.add_row_owned(vec!["Epsilon threshold".into(), format_sci(pa.epsilon)]);
            st.add_row_owned(vec!["Unstable values".into(), format!("{}", unstable_count)]);
            eprint!("{}", st.render());

            let verdict = if stable {
                output::colorize(pa.output, output::BOLD_GREEN,
                    "STABLE (within epsilon bounds)")
            } else {
                output::colorize(pa.output, output::BOLD_YELLOW,
                    "SENSITIVE (exceeds epsilon)")
            };
            eprintln!("Verdict: {}", verdict);
        }
    }

    // --fail-on-instability: exit 1 if any value exceeds epsilon
    if pa.fail_on_instability && !stable {
        process::exit(1);
    }
}

fn format_sci(v: f64) -> String {
    if v == 0.0 {
        "0.00e+00".to_string()
    } else {
        format!("{:.2e}", v)
    }
}

pub fn print_help() {
    eprintln!("cjcl precision — Precision analysis for numeric output");
    eprintln!();
    eprintln!("Usage: cjcl precision <file.cjcl> [flags]");
    eprintln!();
    eprintln!("Runs the program, captures numeric output, and compares f64");
    eprintln!("values against f32-truncated approximations.");
    eprintln!();
    eprintln!("Reports: relative error per value, max error, stability verdict.");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  --seed <N>              RNG seed (default: 42)");
    eprintln!("  --epsilon <N>           Custom epsilon for comparison (default: f32::EPSILON)");
    eprintln!("  --fail-on-instability   Exit code 1 if any value exceeds epsilon");
    eprintln!("  --report <file>         Save precision report as JSON");
    eprintln!("  --summary-only          Only show overall pass/fail and stats");
    eprintln!("  --plain                 Plain text output");
    eprintln!("  --json                  JSON output");
    eprintln!("  --color                 Color output (default)");
}
