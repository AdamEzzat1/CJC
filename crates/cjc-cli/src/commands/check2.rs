//! `cjcl check` — Dual-mode: type-checking and output validation.
//!
//! Mode 1 (type-check): `cjcl check file.cjcl` — type-checks a CJC source file.
//! Mode 2 (validate):   `cjcl check file --against expected` — validates output.
//!
//! Validation supports exact match, tolerance-based numeric comparison,
//! and schema-level structural matching.

use std::fs;
use std::path::Path;
use std::process;
use crate::output::{self, OutputMode};

pub struct CheckArgs {
    pub file: String,
    pub against: Option<String>,
    pub expect: Option<String>,
    pub tolerance: f64,
    pub schema_only: bool,
    pub output: OutputMode,
}

impl Default for CheckArgs {
    fn default() -> Self {
        Self {
            file: String::new(),
            against: None,
            expect: None,
            tolerance: 0.0,
            schema_only: false,
            output: OutputMode::Color,
        }
    }
}

pub fn parse_args(args: &[String]) -> CheckArgs {
    let mut ca = CheckArgs::default();
    let mut positionals = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--against" | "-a" => {
                i += 1;
                if i < args.len() { ca.against = Some(args[i].clone()); }
            }
            "--expect" | "-e" => {
                i += 1;
                if i < args.len() { ca.expect = Some(args[i].clone()); }
            }
            "--tolerance" | "--tol" => {
                i += 1;
                if i < args.len() {
                    ca.tolerance = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("error: --tolerance requires a numeric argument");
                        process::exit(1);
                    });
                }
            }
            "--schema-only" => ca.schema_only = true,
            "--plain" => ca.output = OutputMode::Plain,
            "--json" => ca.output = OutputMode::Json,
            "--color" => ca.output = OutputMode::Color,
            other if !other.starts_with('-') => positionals.push(other.to_string()),
            other => {
                eprintln!("error: unknown flag `{}` for `cjcl check`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    if positionals.is_empty() {
        eprintln!("error: `cjcl check` requires a file argument");
        process::exit(1);
    }
    ca.file = positionals[0].clone();
    ca
}

pub fn run(args: &[String]) {
    let ca = parse_args(args);

    // Determine mode: if validation flags present, validate; otherwise type-check
    let is_validation = ca.against.is_some() || ca.expect.is_some() || ca.schema_only;

    if is_validation {
        run_validate(&ca);
    } else {
        run_typecheck(&ca);
    }
}

fn run_typecheck(ca: &CheckArgs) {
    let path = Path::new(&ca.file);
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read `{}`: {}", ca.file, e);
            process::exit(1);
        }
    };

    let (program, parse_diags) = cjc_parser::parse_source(&source);

    if parse_diags.has_errors() {
        let rendered = parse_diags.render_all_color(&source, &ca.file, ca.output.use_color());
        eprint!("{}", rendered);
        process::exit(1);
    }

    let mut checker = cjc_types::TypeChecker::new();
    checker.check_program(&program);

    if checker.diagnostics.has_errors() {
        let rendered = checker.diagnostics.render_all_color(&source, &ca.file, ca.output.use_color());
        eprint!("{}", rendered);
        process::exit(1);
    }

    let warnings = checker.diagnostics.diagnostics.len();

    match ca.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file\": \"{}\",", ca.file.replace('\\', "/"));
            println!("  \"mode\": \"typecheck\",");
            println!("  \"errors\": 0,");
            println!("  \"warnings\": {},", warnings);
            println!("  \"status\": \"OK\"");
            println!("}}");
        }
        _ => {
            if warnings > 0 {
                let rendered = checker.diagnostics.render_all_color(&source, &ca.file, ca.output.use_color());
                eprint!("{}", rendered);
            }
            eprintln!("{} — no errors in `{}`",
                output::colorize(ca.output, output::BOLD_GREEN, "OK"),
                ca.file.replace('\\', "/"));
        }
    }
}

fn run_validate(ca: &CheckArgs) {
    let content_a = match fs::read_to_string(&ca.file) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: could not read `{}`: {}", ca.file, e);
            process::exit(1);
        }
    };

    // --expect: compare file content against a literal string
    if let Some(ref expected) = ca.expect {
        let actual = content_a.trim();
        let pass = actual == expected.trim();

        match ca.output {
            OutputMode::Json => {
                println!("{{");
                println!("  \"file\": \"{}\",", ca.file.replace('\\', "/"));
                println!("  \"mode\": \"expect\",");
                println!("  \"match\": {}", pass);
                println!("}}");
            }
            _ => {
                if pass {
                    eprintln!("{} — content matches expected value",
                        output::colorize(ca.output, output::BOLD_GREEN, "PASS"));
                } else {
                    eprintln!("{} — content does not match expected value",
                        output::colorize(ca.output, output::BOLD_RED, "FAIL"));
                    eprintln!("  Expected: {:?}", expected.trim());
                    let preview = if actual.len() > 100 { &actual[..100] } else { actual };
                    eprintln!("  Actual:   {:?}", preview);
                    process::exit(1);
                }
            }
        }
        return;
    }

    // --against: compare two files
    if let Some(ref against_file) = ca.against {
        let content_b = match fs::read_to_string(against_file) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: could not read `{}`: {}", against_file, e);
                process::exit(1);
            }
        };

        if ca.schema_only {
            validate_schema(&ca, &content_a, &content_b, against_file);
        } else if ca.tolerance > 0.0 {
            validate_tolerance(&ca, &content_a, &content_b, against_file);
        } else {
            validate_exact(&ca, &content_a, &content_b, against_file);
        }
    }
}

fn validate_exact(ca: &CheckArgs, a: &str, b: &str, b_name: &str) {
    let pass = a == b;
    let lines_a = a.lines().count();
    let lines_b = b.lines().count();

    match ca.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file_a\": \"{}\",", ca.file.replace('\\', "/"));
            println!("  \"file_b\": \"{}\",", b_name.replace('\\', "/"));
            println!("  \"mode\": \"exact\",");
            println!("  \"lines_a\": {},", lines_a);
            println!("  \"lines_b\": {},", lines_b);
            println!("  \"match\": {}", pass);
            println!("}}");
        }
        _ => {
            if pass {
                eprintln!("{} — files are identical ({} lines)",
                    output::colorize(ca.output, output::BOLD_GREEN, "PASS"), lines_a);
            } else {
                eprintln!("{} — files differ", output::colorize(ca.output, output::BOLD_RED, "FAIL"));
                // Find first difference
                for (i, (la, lb)) in a.lines().zip(b.lines()).enumerate() {
                    if la != lb {
                        eprintln!("  First difference at line {}:", i + 1);
                        eprintln!("    A: {:?}", la);
                        eprintln!("    B: {:?}", lb);
                        break;
                    }
                }
                if lines_a != lines_b {
                    eprintln!("  Line count: {} vs {}", lines_a, lines_b);
                }
                process::exit(1);
            }
        }
    }
}

fn validate_tolerance(ca: &CheckArgs, a: &str, b: &str, b_name: &str) {
    let nums_a: Vec<f64> = a.split_whitespace().filter_map(|s| s.parse().ok()).collect();
    let nums_b: Vec<f64> = b.split_whitespace().filter_map(|s| s.parse().ok()).collect();

    if nums_a.len() != nums_b.len() {
        eprintln!("{} — element count mismatch: {} vs {}",
            output::colorize(ca.output, output::BOLD_RED, "FAIL"), nums_a.len(), nums_b.len());
        process::exit(1);
    }

    let mut max_dev = 0.0f64;
    let mut failures = 0u64;
    for (va, vb) in nums_a.iter().zip(&nums_b) {
        let dev = (va - vb).abs();
        if dev > ca.tolerance { failures += 1; }
        if dev > max_dev { max_dev = dev; }
    }

    let pass = failures == 0;

    match ca.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file_a\": \"{}\",", ca.file.replace('\\', "/"));
            println!("  \"file_b\": \"{}\",", b_name.replace('\\', "/"));
            println!("  \"mode\": \"tolerance\",");
            println!("  \"tolerance\": {},", output::format_f64(ca.tolerance, 10));
            println!("  \"elements\": {},", nums_a.len());
            println!("  \"failures\": {},", failures);
            println!("  \"max_deviation\": {},", output::format_f64(max_dev, 10));
            println!("  \"match\": {}", pass);
            println!("}}");
        }
        _ => {
            if pass {
                eprintln!("{} — {} elements within tolerance {} (max deviation: {})",
                    output::colorize(ca.output, output::BOLD_GREEN, "PASS"),
                    nums_a.len(), output::format_f64(ca.tolerance, 6),
                    output::format_f64(max_dev, 10));
            } else {
                eprintln!("{} — {} of {} elements exceed tolerance {}",
                    output::colorize(ca.output, output::BOLD_RED, "FAIL"),
                    failures, nums_a.len(), output::format_f64(ca.tolerance, 6));
                eprintln!("  Max deviation: {}", output::format_f64(max_dev, 10));
                process::exit(1);
            }
        }
    }
}

fn validate_schema(ca: &CheckArgs, a: &str, b: &str, b_name: &str) {
    // Compare CSV headers (schema structure)
    let header_a: Vec<&str> = a.lines().next().unwrap_or("").split(',').map(|s| s.trim()).collect();
    let header_b: Vec<&str> = b.lines().next().unwrap_or("").split(',').map(|s| s.trim()).collect();

    let pass = header_a == header_b;

    match ca.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file_a\": \"{}\",", ca.file.replace('\\', "/"));
            println!("  \"file_b\": \"{}\",", b_name.replace('\\', "/"));
            println!("  \"mode\": \"schema\",");
            println!("  \"columns_a\": {},", header_a.len());
            println!("  \"columns_b\": {},", header_b.len());
            println!("  \"match\": {}", pass);
            println!("}}");
        }
        _ => {
            if pass {
                eprintln!("{} — schemas match ({} columns)",
                    output::colorize(ca.output, output::BOLD_GREEN, "PASS"), header_a.len());
            } else {
                eprintln!("{} — schemas differ", output::colorize(ca.output, output::BOLD_RED, "FAIL"));
                eprintln!("  A columns: {:?}", header_a);
                eprintln!("  B columns: {:?}", header_b);
                process::exit(1);
            }
        }
    }
}

pub fn print_help() {
    eprintln!("cjcl check — Type-checking and output validation");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  cjcl check <file.cjcl>                        Type-check a CJC source file");
    eprintln!("  cjcl check <file> --against <expected>        Compare files for equality");
    eprintln!("  cjcl check <file> --expect <value>            Check content matches string");
    eprintln!("  cjcl check <file> --against <b> --tol <N>     Numeric tolerance comparison");
    eprintln!("  cjcl check <file> --against <b> --schema-only Compare CSV schemas only");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  -a, --against <file>    File to compare against");
    eprintln!("  -e, --expect <value>    Expected content string");
    eprintln!("  --tolerance, --tol <N>  Numeric tolerance (default: 0.0)");
    eprintln!("  --schema-only           Compare only schema structure");
    eprintln!("  --plain                 Plain text output");
    eprintln!("  --json                  JSON output");
    eprintln!("  --color                 Color output (default)");
}
