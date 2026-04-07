//! `cjcl doctor` — Project diagnostics.
//!
//! Scans a project directory for common reproducibility, schema, and
//! runtime issues. Produces a structured diagnostic report.
//!
//! Checks performed:
//! - Parse errors in .cjcl files
//! - Type errors in .cjcl files
//! - Corrupt or malformed .snap files
//! - CSV schema inconsistencies
//! - Missing referenced files
//! - Nondeterministic patterns (HashMap usage in source)
//! - Large files that may cause memory issues
//! - JSONL/NDJSON validation (first 10 lines)
//! - Parquet magic-byte validation
//! - Model file (.pkl, .onnx, .joblib) presence reporting

use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use crate::output::{self, OutputMode};

pub struct DoctorArgs {
    pub path: PathBuf,
    pub output: OutputMode,
    pub verbose: bool,
    pub fix: bool,
    pub strict: bool,
    pub dry_run: bool,
    pub category: Option<String>,
    pub report: Option<PathBuf>,
    pub summary_only: bool,
    pub fail_on: Option<Severity>,
}

impl Default for DoctorArgs {
    fn default() -> Self {
        Self {
            path: PathBuf::from("."),
            output: OutputMode::Color,
            verbose: false,
            fix: false,
            strict: false,
            dry_run: false,
            category: None,
            report: None,
            summary_only: false,
            fail_on: None,
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
            "--strict" => da.strict = true,
            "--dry-run" => da.dry_run = true,
            "--summary-only" => da.summary_only = true,
            "--plain" => da.output = OutputMode::Plain,
            "--json" => da.output = OutputMode::Json,
            "--color" => da.output = OutputMode::Color,
            "--category" => {
                i += 1;
                if i < args.len() {
                    da.category = Some(args[i].clone());
                } else {
                    eprintln!("error: --category requires a value");
                    process::exit(1);
                }
            }
            "--report" => {
                i += 1;
                if i < args.len() {
                    da.report = Some(PathBuf::from(&args[i]));
                } else {
                    eprintln!("error: --report requires a file path");
                    process::exit(1);
                }
            }
            "--fail-on" => {
                i += 1;
                if i < args.len() {
                    match args[i].as_str() {
                        "error" => da.fail_on = Some(Severity::Error),
                        "warning" => da.fail_on = Some(Severity::Warning),
                        other => {
                            eprintln!("error: --fail-on expects `warning` or `error`, got `{}`", other);
                            process::exit(1);
                        }
                    }
                } else {
                    eprintln!("error: --fail-on requires a value (warning or error)");
                    process::exit(1);
                }
            }
            other if !other.starts_with('-') => da.path = PathBuf::from(other),
            other => {
                eprintln!("error: unknown flag `{}` for `cjcl doctor`", other);
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
pub enum Severity {
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

/// Collected project files, sorted deterministically.
struct ProjectFiles {
    cjc_files: Vec<PathBuf>,
    snap_files: Vec<PathBuf>,
    csv_files: Vec<PathBuf>,
    jsonl_files: Vec<PathBuf>,
    parquet_files: Vec<PathBuf>,
    model_files: Vec<PathBuf>,
    all_files: Vec<PathBuf>,
}

pub fn run(args: &[String]) {
    let da = parse_args(args);

    eprintln!("{} Diagnosing `{}`...",
        output::colorize(da.output, output::BOLD_CYAN, "[doctor]"),
        da.path.display());

    let mut findings: Vec<Finding> = Vec::new();

    // Collect all relevant files
    let mut pf = ProjectFiles {
        cjc_files: Vec::new(),
        snap_files: Vec::new(),
        csv_files: Vec::new(),
        jsonl_files: Vec::new(),
        parquet_files: Vec::new(),
        model_files: Vec::new(),
        all_files: Vec::new(),
    };

    collect_files(&da.path, &mut pf);

    // Sort for determinism
    pf.cjc_files.sort();
    pf.snap_files.sort();
    pf.csv_files.sort();
    pf.jsonl_files.sort();
    pf.parquet_files.sort();
    pf.model_files.sort();
    pf.all_files.sort();

    // Check 1: Parse errors in .cjcl files
    for file in &pf.cjc_files {
        check_cjc_parse(file, &mut findings);
    }

    // Check 2: Type errors in .cjcl files
    for file in &pf.cjc_files {
        check_cjc_types(file, &mut findings);
    }

    // Check 3: Snap file integrity
    for file in &pf.snap_files {
        check_snap_integrity(file, &mut findings);
    }

    // Check 4: CSV schema checks
    for file in &pf.csv_files {
        check_csv_schema(file, &mut findings, &da);
    }

    // Check 5: Nondeterminism patterns
    for file in &pf.cjc_files {
        check_nondeterminism(file, &mut findings, da.verbose);
    }

    // Check 6: Large file warnings
    for file in &pf.all_files {
        check_large_files(file, &mut findings);
    }

    // Check 7: JSONL validation
    for file in &pf.jsonl_files {
        check_jsonl(file, &mut findings);
    }

    // Check 8: Parquet magic bytes
    for file in &pf.parquet_files {
        check_parquet(file, &mut findings);
    }

    // Check 9: Model file presence
    for file in &pf.model_files {
        check_model_file(file, &mut findings);
    }

    // --fix: trailing whitespace and trailing newlines for .cjcl files
    if da.fix {
        for file in &pf.cjc_files {
            fix_cjc_whitespace(file, &mut findings, da.dry_run);
        }
    }

    // Filter by category if requested
    if let Some(ref cat) = da.category {
        findings.retain(|f| f.category == cat.as_str());
    }

    // Sort findings by severity then file
    findings.sort_by(|a, b| a.severity.cmp(&b.severity).then(a.file.cmp(&b.file)));

    // Report
    let errors = findings.iter().filter(|f| f.severity == Severity::Error).count();
    let warnings = findings.iter().filter(|f| f.severity == Severity::Warning).count();
    let infos = findings.iter().filter(|f| f.severity == Severity::Info).count();

    // --report: save findings to JSON file
    if let Some(ref report_path) = da.report {
        write_report(report_path, &da, &pf, &findings);
    }

    match da.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"path\": \"{}\",", da.path.display().to_string().replace('\\', "/"));
            println!("  \"cjc_files\": {},", pf.cjc_files.len());
            println!("  \"snap_files\": {},", pf.snap_files.len());
            println!("  \"csv_files\": {},", pf.csv_files.len());
            println!("  \"jsonl_files\": {},", pf.jsonl_files.len());
            println!("  \"parquet_files\": {},", pf.parquet_files.len());
            println!("  \"model_files\": {},", pf.model_files.len());
            println!("  \"errors\": {},", errors);
            println!("  \"warnings\": {},", warnings);
            println!("  \"infos\": {},", infos);
            println!("  \"findings\": [");
            if !da.summary_only {
                for (i, f) in findings.iter().enumerate() {
                    print!("    {{\"severity\": \"{}\", \"category\": \"{}\", \"file\": \"{}\", \"message\": \"{}\"}}",
                        f.severity.label(), f.category,
                        f.file.replace('\\', "/"),
                        f.message.replace('"', "\\\""));
                    if i + 1 < findings.len() { print!(","); }
                    println!();
                }
            }
            println!("  ]");
            println!("}}");
        }
        _ => {
            eprintln!();

            // Summary table
            let mut t = crate::table::Table::new(vec!["Category", "Count"]);
            t.add_row_owned(vec!["CJC source files".into(), format!("{}", pf.cjc_files.len())]);
            t.add_row_owned(vec!["Snap files".into(), format!("{}", pf.snap_files.len())]);
            t.add_row_owned(vec!["CSV/TSV files".into(), format!("{}", pf.csv_files.len())]);
            t.add_row_owned(vec!["JSONL/NDJSON files".into(), format!("{}", pf.jsonl_files.len())]);
            t.add_row_owned(vec!["Parquet files".into(), format!("{}", pf.parquet_files.len())]);
            t.add_row_owned(vec!["Model files".into(), format!("{}", pf.model_files.len())]);
            t.add_row_owned(vec!["Total files".into(), format!("{}", pf.all_files.len())]);
            eprint!("{}", t.render());

            if findings.is_empty() {
                eprintln!("\n{} — no issues found",
                    output::colorize(da.output, output::BOLD_GREEN, "HEALTHY"));
            } else {
                if !da.summary_only {
                    eprintln!();
                    for f in &findings {
                        let sev = output::colorize(da.output, f.severity.color(), f.severity.label());
                        eprintln!("  [{}] {} — {} ({})",
                            sev, f.file.replace('\\', "/"), f.message, f.category);
                    }
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
                    if da.strict {
                        process::exit(1);
                    }
                }
            }
        }
    }

    // --fail-on: exit 1 if findings at specified severity or above
    if let Some(ref level) = da.fail_on {
        let has_matching = findings.iter().any(|f| f.severity <= *level);
        if has_matching {
            process::exit(1);
        }
    }
}

fn collect_files(dir: &Path, pf: &mut ProjectFiles) {
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
            collect_files(&path, pf);
            continue;
        }

        pf.all_files.push(path.clone());

        match path.extension().and_then(|e| e.to_str()) {
            Some("cjcl") => pf.cjc_files.push(path),
            Some("snap") => pf.snap_files.push(path),
            Some("csv") | Some("tsv") => pf.csv_files.push(path),
            Some("jsonl") | Some("ndjson") => pf.jsonl_files.push(path),
            Some("parquet") => pf.parquet_files.push(path),
            Some("pkl") | Some("onnx") | Some("joblib") => pf.model_files.push(path),
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

fn check_csv_schema(file: &Path, findings: &mut Vec<Finding>, da: &DoctorArgs) {
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
    let mut ragged_rows: Vec<(usize, usize)> = Vec::new(); // (line_index, col_count)
    for (i, line) in lines.iter().enumerate().skip(1) {
        if line.trim().is_empty() { continue; }
        let cols = line.split(delimiter).count();
        if cols != header_cols {
            ragged_rows.push((i, cols));
        }
    }

    if ragged_rows.is_empty() {
        return;
    }

    findings.push(Finding {
        severity: Severity::Warning, category: "csv",
        file: file.display().to_string(),
        message: format!("{} ragged row(s) (column count mismatch)", ragged_rows.len()),
    });

    // --fix: pad short rows with empty fields, truncate extra fields
    if da.fix {
        // Only fix if we have a clear header count (header_cols > 0)
        if header_cols == 0 {
            findings.push(Finding {
                severity: Severity::Info, category: "csv",
                file: file.display().to_string(),
                message: "cannot fix ragged rows: ambiguous header column count (0)".into(),
            });
            return;
        }

        let delim_str = if delimiter == '\t' { "\t" } else { "," };
        let mut fixed_lines: Vec<String> = Vec::with_capacity(lines.len());
        let mut fixes_applied = 0usize;

        for (i, line) in lines.iter().enumerate() {
            if i == 0 {
                // Header line kept as-is
                fixed_lines.push(line.to_string());
                continue;
            }
            if line.trim().is_empty() {
                fixed_lines.push(line.to_string());
                continue;
            }

            let fields: Vec<&str> = line.split(delimiter).collect();
            let col_count = fields.len();

            if col_count == header_cols {
                fixed_lines.push(line.to_string());
            } else if col_count < header_cols {
                // Pad with empty fields
                let mut padded = fields.iter().map(|f| f.to_string()).collect::<Vec<_>>();
                for _ in col_count..header_cols {
                    padded.push(String::new());
                }
                fixed_lines.push(padded.join(delim_str));
                fixes_applied += 1;
            } else {
                // Truncate extra fields
                let truncated: Vec<&str> = fields[..header_cols].to_vec();
                fixed_lines.push(truncated.join(delim_str));
                fixes_applied += 1;
            }
        }

        if fixes_applied > 0 {
            let action = if da.dry_run { "would fix" } else { "fixed" };
            findings.push(Finding {
                severity: Severity::Info, category: "csv",
                file: file.display().to_string(),
                message: format!("{} {} ragged row(s) to match header column count ({})",
                    action, fixes_applied, header_cols),
            });

            if !da.dry_run {
                // Reconstruct file content preserving original line ending style
                let has_trailing_newline = content.ends_with('\n');
                let mut output = fixed_lines.join("\n");
                if has_trailing_newline {
                    output.push('\n');
                }
                let _ = fs::write(file, output);
            }
        }
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

// ── New checks: JSONL, Parquet, Model files ─────────────────────────

fn check_jsonl(file: &Path, findings: &mut Vec<Finding>) {
    let content = match fs::read_to_string(file) {
        Ok(c) => c,
        Err(_) => {
            findings.push(Finding {
                severity: Severity::Error, category: "jsonl",
                file: file.display().to_string(), message: "could not read JSONL file".into(),
            });
            return;
        }
    };

    if content.trim().is_empty() {
        findings.push(Finding {
            severity: Severity::Warning, category: "jsonl",
            file: file.display().to_string(), message: "empty JSONL file".into(),
        });
        return;
    }

    // Sample first 10 non-empty lines and validate as JSON objects
    let mut malformed_lines: Vec<usize> = Vec::new();
    let mut checked = 0usize;

    for (i, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        if checked >= 10 { break; }
        checked += 1;

        if !is_valid_json_object(trimmed) {
            malformed_lines.push(i + 1); // 1-indexed
        }
    }

    if !malformed_lines.is_empty() {
        let line_list: Vec<String> = malformed_lines.iter().map(|n| n.to_string()).collect();
        findings.push(Finding {
            severity: Severity::Warning, category: "jsonl",
            file: file.display().to_string(),
            message: format!(
                "{} malformed line(s) in first 10 (lines: {})",
                malformed_lines.len(),
                line_list.join(", ")
            ),
        });
    }
}

/// Minimal JSON object validation without external dependencies.
/// Checks that the line starts with `{` and ends with `}`, and that
/// braces/brackets/quotes are balanced. This is a conservative heuristic.
fn is_valid_json_object(s: &str) -> bool {
    let s = s.trim();
    if !s.starts_with('{') || !s.ends_with('}') {
        return false;
    }

    // Walk through checking balanced braces/brackets/quotes
    let mut depth_brace: i32 = 0;
    let mut depth_bracket: i32 = 0;
    let mut in_string = false;
    let mut escape_next = false;

    for ch in s.chars() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape_next = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string { continue; }

        match ch {
            '{' => depth_brace += 1,
            '}' => {
                depth_brace -= 1;
                if depth_brace < 0 { return false; }
            }
            '[' => depth_bracket += 1,
            ']' => {
                depth_bracket -= 1;
                if depth_bracket < 0 { return false; }
            }
            _ => {}
        }
    }

    depth_brace == 0 && depth_bracket == 0 && !in_string
}

fn check_parquet(file: &Path, findings: &mut Vec<Finding>) {
    let data = match fs::read(file) {
        Ok(d) => d,
        Err(_) => {
            findings.push(Finding {
                severity: Severity::Error, category: "parquet",
                file: file.display().to_string(), message: "could not read Parquet file".into(),
            });
            return;
        }
    };

    if data.len() < 4 {
        findings.push(Finding {
            severity: Severity::Error, category: "parquet",
            file: file.display().to_string(),
            message: "file too small to be a valid Parquet file".into(),
        });
        return;
    }

    // Parquet files start with magic bytes "PAR1"
    if &data[..4] != b"PAR1" {
        findings.push(Finding {
            severity: Severity::Error, category: "parquet",
            file: file.display().to_string(),
            message: format!(
                "invalid Parquet magic bytes (expected PAR1, got {:02x}{:02x}{:02x}{:02x})",
                data[0], data[1], data[2], data[3]
            ),
        });
    }
}

fn check_model_file(file: &Path, findings: &mut Vec<Finding>) {
    let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("unknown");
    let size = fs::metadata(file).map(|m| m.len()).unwrap_or(0);
    findings.push(Finding {
        severity: Severity::Info, category: "model",
        file: file.display().to_string(),
        message: format!(
            ".{} model file detected ({}) — model files are not deserialized",
            ext, output::format_size(size)
        ),
    });
}

// ── --fix helpers ───────────────────────────────────────────────────

fn fix_cjc_whitespace(file: &Path, findings: &mut Vec<Finding>, dry_run: bool) {
    let content = match fs::read_to_string(file) {
        Ok(c) => c,
        Err(_) => return,
    };

    let mut changed = false;
    let mut trailing_ws_count = 0usize;

    // Fix trailing whitespace on each line
    let lines: Vec<&str> = content.split('\n').collect();
    let mut fixed_lines: Vec<String> = Vec::with_capacity(lines.len());

    for line in &lines {
        let trimmed = line.trim_end_matches(|c: char| c == ' ' || c == '\t');
        if trimmed.len() < line.len() {
            trailing_ws_count += 1;
            changed = true;
        }
        fixed_lines.push(trimmed.to_string());
    }

    // Reconstruct content
    let mut new_content = fixed_lines.join("\n");

    // Fix trailing newlines: ensure exactly one trailing newline
    let trimmed_end = new_content.trim_end_matches('\n');
    let trailing_newlines = new_content.len() - trimmed_end.len();
    if !new_content.is_empty() && trailing_newlines != 1 {
        new_content = format!("{}\n", trimmed_end);
        changed = true;
    }

    if trailing_ws_count > 0 {
        let action = if dry_run { "would trim" } else { "trimmed" };
        findings.push(Finding {
            severity: Severity::Info, category: "fix",
            file: file.display().to_string(),
            message: format!("{} trailing whitespace on {} line(s)", action, trailing_ws_count),
        });
    }

    // Check if trailing newline normalization changed anything beyond whitespace trimming
    let original_trimmed = content.trim_end_matches('\n');
    let original_trailing = content.len() - original_trimmed.len();
    if original_trailing != 1 && !content.is_empty() {
        let action = if dry_run { "would normalize" } else { "normalized" };
        findings.push(Finding {
            severity: Severity::Info, category: "fix",
            file: file.display().to_string(),
            message: format!("{} trailing newlines (was {}, now 1)", action, original_trailing),
        });
    }

    if changed && !dry_run {
        let _ = fs::write(file, &new_content);
    }
}

// ── --report: write findings to JSON file ───────────────────────────

fn write_report(path: &Path, da: &DoctorArgs, pf: &ProjectFiles, findings: &[Finding]) {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"path\": \"{}\",\n", da.path.display().to_string().replace('\\', "/")));
    out.push_str(&format!("  \"cjc_files\": {},\n", pf.cjc_files.len()));
    out.push_str(&format!("  \"snap_files\": {},\n", pf.snap_files.len()));
    out.push_str(&format!("  \"csv_files\": {},\n", pf.csv_files.len()));
    out.push_str(&format!("  \"jsonl_files\": {},\n", pf.jsonl_files.len()));
    out.push_str(&format!("  \"parquet_files\": {},\n", pf.parquet_files.len()));
    out.push_str(&format!("  \"model_files\": {},\n", pf.model_files.len()));

    let errors = findings.iter().filter(|f| f.severity == Severity::Error).count();
    let warnings = findings.iter().filter(|f| f.severity == Severity::Warning).count();
    let infos = findings.iter().filter(|f| f.severity == Severity::Info).count();
    out.push_str(&format!("  \"errors\": {},\n", errors));
    out.push_str(&format!("  \"warnings\": {},\n", warnings));
    out.push_str(&format!("  \"infos\": {},\n", infos));

    out.push_str("  \"findings\": [\n");
    for (i, f) in findings.iter().enumerate() {
        out.push_str(&format!(
            "    {{\"severity\": \"{}\", \"category\": \"{}\", \"file\": \"{}\", \"message\": \"{}\"}}",
            f.severity.label(), f.category,
            f.file.replace('\\', "/"),
            f.message.replace('"', "\\\"")
        ));
        if i + 1 < findings.len() { out.push(','); }
        out.push('\n');
    }
    out.push_str("  ]\n");
    out.push_str("}\n");

    if let Err(e) = fs::write(path, &out) {
        eprintln!("error: could not write report to {}: {}", path.display(), e);
    }
}

pub fn print_help() {
    eprintln!("cjcl doctor — Project diagnostics");
    eprintln!();
    eprintln!("Usage: cjcl doctor [path] [flags]");
    eprintln!();
    eprintln!("Checks:");
    eprintln!("  - Parse errors in .cjcl files");
    eprintln!("  - Type errors in .cjcl files");
    eprintln!("  - Corrupt .snap files");
    eprintln!("  - CSV schema issues (ragged rows)");
    eprintln!("  - Nondeterministic patterns");
    eprintln!("  - Large files that may cause memory issues");
    eprintln!("  - JSONL/NDJSON validation (malformed JSON lines)");
    eprintln!("  - Parquet file magic-byte validation");
    eprintln!("  - Model file (.pkl, .onnx, .joblib) detection");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  -v, --verbose          Show extra detail");
    eprintln!("  --fix                  Attempt automatic fixes (where safe)");
    eprintln!("  --dry-run              With --fix, preview fixes without applying");
    eprintln!("  --strict               Exit 1 on warnings");
    eprintln!("  --category <type>      Filter findings by category");
    eprintln!("                         (parse, types, snap, csv, determinism, size,");
    eprintln!("                          jsonl, parquet, model, fix, io)");
    eprintln!("  --report <file>        Save findings to a JSON file");
    eprintln!("  --summary-only         Only show summary counts, not individual findings");
    eprintln!("  --fail-on <level>      Exit 1 if findings at severity or above");
    eprintln!("                         (warning, error)");
    eprintln!("  --plain                Plain text output");
    eprintln!("  --json                 JSON output");
    eprintln!("  --color                Color output (default)");
}
