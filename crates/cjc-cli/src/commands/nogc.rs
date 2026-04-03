//! `cjc nogc` — NoGC static verification command.
//!
//! Promotes `cjc_mir_exec::verify_nogc` from a library call to a first-class
//! CLI command. Parses a CJC source file, lists all functions with their
//! `#[nogc]` annotation status, runs the NoGC verifier, and reports
//! per-function PASS/FAIL results with violation details.

use std::fs;
use std::path::Path;
use std::process;
use crate::output::{self, OutputMode};

pub struct NogcArgs {
    pub file: String,
    pub output: OutputMode,
    pub verbose: bool,
}

impl Default for NogcArgs {
    fn default() -> Self {
        Self {
            file: String::new(),
            output: OutputMode::Color,
            verbose: false,
        }
    }
}

pub fn parse_args(args: &[String]) -> NogcArgs {
    let mut na = NogcArgs::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-v" | "--verbose" => na.verbose = true,
            "--plain" => na.output = OutputMode::Plain,
            "--json" => na.output = OutputMode::Json,
            "--color" => na.output = OutputMode::Color,
            other if !other.starts_with('-') => na.file = other.to_string(),
            other => {
                eprintln!("error: unknown flag `{}` for `cjc nogc`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    if na.file.is_empty() {
        eprintln!("error: `cjc nogc` requires a .cjc file argument");
        process::exit(1);
    }
    na
}

/// Information about a function found in the program.
struct FnInfo {
    name: String,
    is_nogc: bool,
}

pub fn run(args: &[String]) {
    let na = parse_args(args);
    let path = Path::new(&na.file);

    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read `{}`: {}", na.file, e);
            process::exit(1);
        }
    };

    let filename = na.file.replace('\\', "/");

    let (program, diags) = cjc_parser::parse_source(&source);
    if diags.has_errors() {
        let rendered = diags.render_all_color(&source, &filename, na.output.use_color());
        eprint!("{}", rendered);
        process::exit(1);
    }

    // Collect all functions from the program
    let mut functions: Vec<FnInfo> = Vec::new();
    for decl in &program.declarations {
        if let cjc_ast::DeclKind::Fn(fn_decl) = &decl.kind {
            functions.push(FnInfo {
                name: fn_decl.name.name.clone(),
                is_nogc: fn_decl.is_nogc,
            });
        }
    }

    // Run NoGC verification
    let verify_result = cjc_mir_exec::verify_nogc(&program);

    // Parse violations from error string (format: "nogc violation in `name`: reason")
    let mut violations: Vec<(String, String)> = Vec::new();
    if let Err(ref err_str) = verify_result {
        for line in err_str.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            // Parse: "nogc violation in `funcname`: reason text"
            if let Some(rest) = line.strip_prefix("nogc violation in `") {
                if let Some(backtick_end) = rest.find('`') {
                    let func_name = rest[..backtick_end].to_string();
                    let reason = rest[backtick_end + 1..].trim_start_matches(':').trim().to_string();
                    violations.push((func_name, reason));
                } else {
                    violations.push(("unknown".into(), line.to_string()));
                }
            } else {
                violations.push(("unknown".into(), line.to_string()));
            }
        }
    }

    let all_pass = verify_result.is_ok();
    let nogc_count = functions.iter().filter(|f| f.is_nogc).count();

    match na.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file\": \"{}\",", filename);
            println!("  \"total_functions\": {},", functions.len());
            println!("  \"nogc_annotated\": {},", nogc_count);
            println!("  \"verdict\": \"{}\",", if all_pass { "PASS" } else { "FAIL" });
            println!("  \"functions\": [");
            for (i, f) in functions.iter().enumerate() {
                let has_violation = violations.iter().any(|(vn, _)| vn == &f.name);
                let status = if has_violation { "FAIL" } else { "PASS" };
                let violation_msg = violations.iter()
                    .filter(|(vn, _)| vn == &f.name)
                    .map(|(_, r)| r.replace('\\', "\\\\").replace('"', "\\\""))
                    .collect::<Vec<_>>()
                    .join("; ");
                let comma = if i + 1 < functions.len() { "," } else { "" };
                if violation_msg.is_empty() {
                    println!("    {{\"name\": \"{}\", \"nogc\": {}, \"status\": \"{}\"}}{}",
                        f.name, f.is_nogc, status, comma);
                } else {
                    println!("    {{\"name\": \"{}\", \"nogc\": {}, \"status\": \"{}\", \"violation\": \"{}\"}}{}",
                        f.name, f.is_nogc, status, violation_msg, comma);
                }
            }
            println!("  ],");
            println!("  \"violations\": [");
            for (i, (func, reason)) in violations.iter().enumerate() {
                let comma = if i + 1 < violations.len() { "," } else { "" };
                println!("    {{\"function\": \"{}\", \"reason\": \"{}\"}}{}",
                    func, reason.replace('\\', "\\\\").replace('"', "\\\""), comma);
            }
            println!("  ]");
            println!("}}");
        }
        _ => {
            eprintln!("{} NoGC verification for `{}`",
                output::colorize(na.output, output::BOLD_CYAN, "[nogc]"),
                filename);
            eprintln!();

            // Function listing table
            let mut t = crate::table::Table::new(vec!["Function", "#[nogc]", "Status"]);
            for f in &functions {
                let has_violation = violations.iter().any(|(vn, _)| vn == &f.name);
                let nogc_label = if f.is_nogc {
                    output::colorize(na.output, output::BOLD_CYAN, "yes")
                } else {
                    "no".to_string()
                };
                let status = if has_violation {
                    output::colorize(na.output, output::BOLD_RED, "FAIL")
                } else {
                    output::colorize(na.output, output::BOLD_GREEN, "PASS")
                };
                t.add_row_owned(vec![f.name.clone(), nogc_label, status]);
            }
            eprint!("{}", t.render());

            // Show violation details
            if !violations.is_empty() {
                eprintln!("\nViolations:");
                for (func, reason) in &violations {
                    eprintln!("  {} in `{}`: {}",
                        output::colorize(na.output, output::BOLD_RED, "FAIL"),
                        func,
                        reason);
                }
            }

            // Verbose: show extra info
            if na.verbose {
                eprintln!("\nSummary:");
                eprintln!("  Total functions:   {}", functions.len());
                eprintln!("  #[nogc] annotated: {}", nogc_count);
                eprintln!("  Violations:        {}", violations.len());
            }

            // Verdict
            eprintln!();
            if all_pass {
                eprintln!("Verdict: {} ({} function{}, {} #[nogc])",
                    output::colorize(na.output, output::BOLD_GREEN, "PASS"),
                    functions.len(),
                    if functions.len() != 1 { "s" } else { "" },
                    nogc_count);
            } else {
                eprintln!("Verdict: {} ({} violation{})",
                    output::colorize(na.output, output::BOLD_RED, "FAIL"),
                    violations.len(),
                    if violations.len() != 1 { "s" } else { "" });
                process::exit(1);
            }
        }
    }

    if !all_pass && na.output == OutputMode::Json {
        process::exit(1);
    }
}

pub fn print_help() {
    eprintln!("cjc nogc — NoGC static verification");
    eprintln!();
    eprintln!("Usage: cjc nogc <file.cjc> [flags]");
    eprintln!();
    eprintln!("Verifies that all #[nogc] functions in the program are free of");
    eprintln!("GC allocations. Lists all functions with their annotation status");
    eprintln!("and reports per-function PASS/FAIL results.");
    eprintln!();
    eprintln!("Exit code: 0 if all pass, 1 if any violations found.");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  -v, --verbose   Show detailed summary");
    eprintln!("  --plain         Plain text output");
    eprintln!("  --json          JSON output");
    eprintln!("  --color         Color output (default)");
}
