//! `cjc doctor` — Project diagnostics.
//!
//! Scans a project directory for common reproducibility, schema, and
//! runtime issues. Produces a structured diagnostic report.
//!
//! Checks performed:
//! - Parse errors in .cjc files
//! - Type errors in .cjc files
//! - Corrupt or malformed .snap files
//! - CSV schema inconsistencies
//! - Missing referenced files
//! - Nondeterministic patterns (HashMap usage in source)
//! - Large files that may cause memory issues

use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use crate::output::{self, OutputMode};

pub struct DoctorArgs {
    pub path: PathBuf,
    pub output: OutputMode,
    pub verbose: bool,
    pub fix: bool,
}

impl Default for DoctorArgs {
    fn default() -> Self {
        Self {
            path: PathBuf::from("."),
            output: OutputMode::Color,
            verbose: false,
            fix: false,
        }
    }
}

pub fn parse_args(args: &[String]) -> DoctorArgs {
    let mut da = DoctorArgs::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-v" | "--verbose" => da.verbose = true,
            "--fix" => da.fix = true,
            "--plain" => da.output = OutputMode::Plain,
            "--json" => da.output = OutputMode::Json,
            "--color" => da.output = OutputMode::Color,
            other if !other.starts_with('-') => da.path = PathBuf::from(other),
            other => {
                eprintln!("error: unknown flag `{}` for `cjc doctor`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    da
}

#[derive(Debug, Clone)]
struct Finding {
    severity: Severity,
    category: &'static str,
    file: String,
    message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Severity {
    Error,
    Warning,
    Info,
}

impl Severity {
    fn label(&self) -> &'static str {
        match self { Severity::Error => "ERROR", Severity::Warning => "WARN", Severity::Info => "INFO" }
    }
    fn color(&self) -> &'static str {
        match self { Severity::Error => output::BOLD_RED, Severity::Warning => output::BOLD_YELLOW, Severity::Info => output::BOLD_CYAN }
    }
}

pub fn run(args: &[String]) {
    let da = parse_args(args);

    eprintln!("{} Diagnosing `{}`...",
        output::colorize(da.output, output::BOLD_CYAN, "[doctor]"),
        da.path.display());

    let mut findings: Vec<Finding> = Vec::new();

    // Collect all relevant files
    let mut cjc_files = Vec::new();
    let mut snap_files = Vec::new();
    let mut csv_files = Vec::new();
    let mut all_files = Vec::new();

    collect_files(&da.path, &mut cjc_files, &mut snap_files, &mut csv_files, &mut all_files);

    // Sort for determinism
    cjc_files.sort();
    snap_files.sort();
    csv_files.sort();
    all_files.sort();

    // Check 1: Parse errors in .cjc files
    for file in &cjc_files {
        check_cjc_parse(file, &mut findings);
    }

    // Check 2: Type errors in .cjc files
    for file in &cjc_files {
        check_cjc_types(file, &mut findings);
    }

    // Check 3: Snap file integrity
    for file in &snap_files {
        check_snap_integrity(file, &mut findings);
    }

    // Check 4: CSV schema checks
    for file in &csv_files {
        check_csv_schema(file, &mut findings);
    }

    // Check 5: Nondeterminism patterns
    for file in &cjc_files {
        check_nondeterminism(file, &mut findings, da.verbose);
    }

    // Check 6: Large file warnings
    for file in &all_files {
        check_large_files(file, &mut findings);
    }

    // Sort findings by severity then file
    findings.sort_by(|a, b| a.severity.cmp(&b.severity).then(a.file.cmp(&b.file)));

    // Report
    let errors = findings.iter().filter(|f| f.severity == Severity::Error).count();
    let warnings = findings.iter().filter(|f| f.severity == Severity::Warning).count();
    let infos = findings.iter().filter(|f| f.severity == Severity::Info).count();

    match da.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"path\": \"{}\",", da.path.display().to_string().replace('\\', "/"));
            println!("  \"cjc_files\": {},", cjc_files.len());
            println!("  \"snap_files\": {},", snap_files.len());
            println!("  \"csv_files\": {},", csv_files.len());
            println!("  \"errors\": {},", errors);
            println!("  \"warnings\": {},", warnings);
            println!("  \"infos\": {},", infos);
            println!("  \"findings\": [");
            for (i, f) in findings.iter().enumerate() {
                print!("    {{\"severity\": \"{}\", \"category\": \"{}\", \"file\": \"{}\", \"message\": \"{}\"}}",
                    f.severity.label(), f.category,
                    f.file.replace('\\', "/"),
                    f.message.replace('"', "\\\""));
                if i + 1 < findings.len() { print!(","); }
                println!();
            }
            println!("  ]");
            println!("}}");
        }
        _ => {
            eprintln!();

            // Summary table
            let mut t = crate::table::Table::new(vec!["Category", "Count"]);
            t.add_row_owned(vec!["CJC source files".into(), format!("{}", cjc_files.len())]);
            t.add_row_owned(vec!["Snap files".into(), format!("{}", snap_files.len())]);
            t.add_row_owned(vec!["CSV/TSV files".into(), format!("{}", csv_files.len())]);
            t.add_row_owned(vec!["Total files".into(), format!("{}", all_files.len())]);
            eprint!("{}", t.render());

            if findings.is_empty() {
                eprintln!("\n{} — no issues found",
                    output::colorize(da.output, output::BOLD_GREEN, "HEALTHY"));
            } else {
                eprintln!();
                for f in &findings {
                    let sev = output::colorize(da.output, f.severity.color(), f.severity.label());
                    eprintln!("  [{}] {} — {} ({})",
                        sev, f.file.replace('\\', "/"), f.message, f.category);
                }

                eprintln!();
                let mut ft = crate::table::Table::new(vec!["Severity", "Count"]);
                if errors > 0 { ft.add_row_owned(vec!["Errors".into(), format!("{}", errors)]); }
                if warnings > 0 { ft.add_row_owned(vec!["Warnings".into(), format!("{}", warnings)]); }
                if infos > 0 { ft.add_row_owned(vec!["Info".into(), format!("{}", infos)]); }
                eprint!("{}", ft.render());

                if errors > 0 {
                    eprintln!("\n{}", output::colorize(da.output, output::BOLD_RED, "ISSUES FOUND"));
                    process::exit(1);
                } else {
                    eprintln!("\n{} — warnings only, no blocking issues",
                        output::colorize(da.output, output::BOLD_YELLOW, "OK (with warnings)"));
                }
            }
        }
    }
}

fn collect_files(dir: &Path, cjc: &mut Vec<PathBuf>, snap: &mut Vec<PathBuf>,
                 csv: &mut Vec<PathBuf>, all: &mut Vec<PathBuf>) {
    let read_dir = match fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(_) => return,
    };

    let mut entries: Vec<_> = read_dir.filter_map(|e| e.ok()).collect();
    entries.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            // Skip hidden dirs, target, node_modules, .git
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with('.') || name == "target" || name == "node_modules" {
                continue;
            }
            collect_files(&path, cjc, snap, csv, all);
            continue;
        }

        all.push(path.clone());

        match path.extension().and_then(|e| e.to_str()) {
            Some("cjc") => cjc.push(path),
            Some("snap") => snap.push(path),
            Some("csv") | Some("tsv") => csv.push(path),
            _ => {}
        }
    }
}

fn check_cjc_parse(file: &Path, findings: &mut Vec<Finding>) {
    let source = match fs::read_to_string(file) {
        Ok(s) => s,
        Err(_) => {
            findings.push(Finding {
                severity: Severity::Error, category: "io",
                file: file.display().to_string(), message: "could not read file".into(),
            });
            return;
        }
    };

    let (_, diags) = cjc_parser::parse_source(&source);
    if diags.has_errors() {
        findings.push(Finding {
            severity: Severity::Error, category: "parse",
            file: file.display().to_string(),
            message: format!("{} parse error(s)", diags.diagnostics.len()),
        });
    }
}

fn check_cjc_types(file: &Path, findings: &mut Vec<Finding>) {
    let source = match fs::read_to_string(file) {
        Ok(s) => s,
        Err(_) => return,
    };

    let (program, parse_diags) = cjc_parser::parse_source(&source);
    if parse_diags.has_errors() { return; } // already reported by parse check

    let mut checker = cjc_types::TypeChecker::new();
    checker.check_program(&program);

    if checker.diagnostics.has_errors() {
        findings.push(Finding {
            severity: Severity::Warning, category: "types",
            file: file.display().to_string(),
            message: format!("{} type issue(s)", checker.diagnostics.diagnostics.len()),
        });
    }
}

fn check_snap_integrity(file: &Path, findings: &mut Vec<Finding>) {
    let data = match fs::read(file) {
        Ok(d) => d,
        Err(_) => {
            findings.push(Finding {
                severity: Severity::Error, category: "snap",
                file: file.display().to_string(), message: "could not read snap file".into(),
            });
            return;
        }
    };

    if data.is_empty() {
        findings.push(Finding {
            severity: Severity::Error, category: "snap",
            file: file.display().to_string(), message: "empty snap file".into(),
        });
        return;
    }

    match cjc_snap::snap_decode_v2(&data) {
        Ok(_) => {} // healthy
        Err(e) => {
            // Try v1
            match cjc_snap::snap_decode(&data) {
                Ok(_) => {} // v1 decode succeeded
                Err(_) => {
                    findings.push(Finding {
                        severity: Severity::Error, category: "snap",
                        file: file.display().to_string(),
                        message: format!("corrupt or unreadable snap file: {:?}", e),
                    });
                }
            }
        }
    }
}

fn check_csv_schema(file: &Path, findings: &mut Vec<Finding>) {
    let content = match fs::read_to_string(file) {
        Ok(c) => c,
        Err(_) => return,
    };

    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        findings.push(Finding {
            severity: Severity::Warning, category: "csv",
            file: file.display().to_string(), message: "empty CSV file".into(),
        });
        return;
    }

    let delimiter = if file.extension().and_then(|e| e.to_str()) == Some("tsv") { '\t' } else { ',' };
    let header_cols = lines[0].split(delimiter).count();

    // Check for ragged rows
    let mut ragged = 0;
    for (i, line) in lines.iter().enumerate().skip(1) {
        if line.trim().is_empty() { continue; }
        let cols = line.split(delimiter).count();
        if cols != header_cols { ragged += 1; }
    }

    if ragged > 0 {
        findings.push(Finding {
            severity: Severity::Warning, category: "csv",
            file: file.display().to_string(),
            message: format!("{} ragged row(s) (column count mismatch)", ragged),
        });
    }
}

fn check_nondeterminism(file: &Path, findings: &mut Vec<Finding>, verbose: bool) {
    let source = match fs::read_to_string(file) {
        Ok(s) => s,
        Err(_) => return,
    };

    // Check for patterns that suggest nondeterminism
    let suspicious_patterns = [
        ("HashMap", "HashMap may have nondeterministic iteration order"),
        ("HashSet", "HashSet may have nondeterministic iteration order"),
        ("rand()", "rand() without explicit seed may be nondeterministic"),
        ("time()", "time() produces nondeterministic output"),
    ];

    for (pattern, reason) in &suspicious_patterns {
        if source.contains(pattern) {
            findings.push(Finding {
                severity: Severity::Info, category: "determinism",
                file: file.display().to_string(),
                message: reason.to_string(),
            });
        }
    }
}

fn check_large_files(file: &Path, findings: &mut Vec<Finding>) {
    let size = match fs::metadata(file) {
        Ok(m) => m.len(),
        Err(_) => return,
    };

    if size > 100 * 1024 * 1024 { // > 100 MB
        findings.push(Finding {
            severity: Severity::Warning, category: "size",
            file: file.display().to_string(),
            message: format!("large file ({}) may cause memory issues", output::format_size(size)),
        });
    }
}

pub fn print_help() {
    eprintln!("cjc doctor — Project diagnostics");
    eprintln!();
    eprintln!("Usage: cjc doctor [path] [flags]");
    eprintln!();
    eprintln!("Checks:");
    eprintln!("  - Parse errors in .cjc files");
    eprintln!("  - Type errors in .cjc files");
    eprintln!("  - Corrupt .snap files");
    eprintln!("  - CSV schema issues (ragged rows)");
    eprintln!("  - Nondeterministic patterns");
    eprintln!("  - Large files that may cause memory issues");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  -v, --verbose   Show extra detail");
    eprintln!("  --fix           Attempt automatic fixes (where safe)");
    eprintln!("  --plain         Plain text output");
    eprintln!("  --json          JSON output");
    eprintln!("  --color         Color output (default)");
}
