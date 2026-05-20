//! `cjcl explain` — dual-purpose command.
//!
//! Two modes, dispatched on argument shape:
//!
//! 1. `cjcl explain EXXXX` / `cjcl explain WXXXX` — print the long-form
//!    Elm-style explanation for an error code, embedded at build time
//!    from `crates/cjc-diag/explanations/`. The pedagogy half of
//!    CJC-Lang's two-fold error system (the precision half is the
//!    diagnostic emitted at error time by `cjc-diag`).
//!
//! 2. `cjcl explain <file.cjcl>` — lower a CJC source file to HIR and
//!    present the desugared forms (for→while, match→branches, closure
//!    captures). Original behaviour, preserved.
//!
//! The dispatch is purely heuristic on argument shape:
//! `^[EW]\d{4}$` (e.g., `E1003`) routes to the error-code path; anything
//! else is treated as a filename. The conventions match `rustc --explain`
//! and avoid breaking any existing call site.

use std::fs;
use std::process;
use crate::output::{self, OutputMode};
use cjc_diag::ErrorCode;

pub struct ExplainArgs {
    pub file: String,
    pub output: OutputMode,
    pub verbose: bool,
}

impl Default for ExplainArgs {
    fn default() -> Self {
        Self {
            file: String::new(),
            output: OutputMode::Color,
            verbose: false,
        }
    }
}

pub fn parse_args(args: &[String]) -> ExplainArgs {
    let mut ea = ExplainArgs::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-v" | "--verbose" => ea.verbose = true,
            "--plain" => ea.output = OutputMode::Plain,
            "--json" => ea.output = OutputMode::Json,
            "--color" => ea.output = OutputMode::Color,
            other if !other.starts_with('-') => ea.file = other.to_string(),
            other => {
                eprintln!("error: unknown flag `{}` for `cjcl explain`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    if ea.file.is_empty() {
        eprintln!("error: `cjcl explain` requires a .cjcl file or error code argument");
        eprintln!("hint: `cjcl explain foo.cjcl` shows the lowered HIR");
        eprintln!("hint: `cjcl explain E1003` shows the long-form explanation of an error code");
        process::exit(1);
    }
    ea
}

// ── Error-code pedagogy lookup (the Elm-style half of the two-fold error system) ──

/// Returns `true` if `s` looks like a canonical error-code identifier
/// (`^[EW][0-9]{4}$`), e.g., `"E1003"` or `"W0001"`.
///
/// Used to decide whether `cjcl explain <arg>` should be routed to the
/// pedagogy path (this returns true) or the HIR-lowering path (this
/// returns false). Filenames will never accidentally match this shape
/// because the canonical form requires exactly five ASCII characters
/// with no extension.
fn is_error_code_shape(s: &str) -> bool {
    let bytes = s.as_bytes();
    bytes.len() == 5
        && matches!(bytes[0], b'E' | b'W')
        && bytes[1..].iter().all(|b| b.is_ascii_digit())
}

/// Renders the full pedagogy lookup for an error code as a single string.
///
/// Returns `Err(msg)` if the code is unknown. Pure function — does not
/// touch stdout/stderr — so it is easy to unit-test.
pub fn format_error_lookup(code_str: &str) -> Result<String, String> {
    let code = ErrorCode::from_str(code_str)
        .ok_or_else(|| format!("unknown error code `{}`", code_str))?;
    let mut out = String::new();
    out.push_str(&format!(
        "{} [{}] - {}\n",
        code.code_str(),
        code.category(),
        code.message_template(),
    ));
    // ASCII divider to keep output 7-bit clean and reliably 60 columns wide.
    out.push_str(&"-".repeat(60));
    out.push('\n');
    out.push('\n');
    match code.explanation() {
        Some(md) => out.push_str(md),
        None => {
            out.push_str("No detailed explanation is available for this error code yet.\n");
            out.push_str(&format!(
                "(Contributions welcome at crates/cjc-diag/explanations/{}.md)\n",
                code.code_str()
            ));
        }
    }
    Ok(out)
}

/// CLI front-end for the pedagogy path.
fn run_error_lookup(code_str: &str, out_mode: OutputMode) {
    match out_mode {
        OutputMode::Json => match ErrorCode::from_str(code_str) {
            Some(code) => {
                // Minimal JSON for v0 — metadata only, no raw markdown
                // (markdown would need full string escaping for JSON safety).
                let severity = match code.severity() {
                    cjc_diag::Severity::Error => "error",
                    cjc_diag::Severity::Warning => "warning",
                    cjc_diag::Severity::Hint => "hint",
                };
                println!("{{");
                println!("  \"code\": \"{}\",", code.code_str());
                println!("  \"category\": \"{}\",", code.category());
                println!("  \"severity\": \"{}\",", severity);
                println!("  \"message\": \"{}\",", code.message_template());
                println!("  \"has_explanation\": {}", code.explanation().is_some());
                println!("}}");
            }
            None => {
                eprintln!("error: unknown error code `{}`", code_str);
                process::exit(1);
            }
        },
        _ => match format_error_lookup(code_str) {
            Ok(rendered) => {
                print!("{}", rendered);
            }
            Err(msg) => {
                eprintln!("error: {}", msg);
                eprintln!("hint: error codes follow the format EXXXX or WXXXX (e.g., E1003, W0001)");
                eprintln!("hint: try `cjcl explain` for the full help message");
                process::exit(1);
            }
        },
    }
}

pub fn run(args: &[String]) {
    let ea = parse_args(args);

    // Heuristic dispatch: if the positional arg matches the canonical
    // error-code shape, treat it as a pedagogy lookup. Otherwise, fall
    // through to the original HIR-lowering behaviour.
    if is_error_code_shape(&ea.file) {
        run_error_lookup(&ea.file, ea.output);
        return;
    }

    let source = match fs::read_to_string(&ea.file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read `{}`: {}", ea.file, e);
            process::exit(1);
        }
    };

    let filename = ea.file.replace('\\', "/");

    let (program, diags) = cjc_parser::parse_source(&source);
    if diags.has_errors() {
        let rendered = diags.render_all_color(&source, &filename, ea.output.use_color());
        eprint!("{}", rendered);
        process::exit(1);
    }

    // Lower to HIR
    let mut lowering = cjc_hir::AstLowering::new();
    let hir = lowering.lower_program(&program);

    // Extract functions and structs from HirItem list
    let mut functions: Vec<&cjc_hir::HirFn> = Vec::new();
    let mut structs: Vec<&cjc_hir::HirStructDef> = Vec::new();
    for item in &hir.items {
        match item {
            cjc_hir::HirItem::Fn(f) => functions.push(f),
            cjc_hir::HirItem::Struct(s) => structs.push(s),
            _ => {}
        }
    }

    match ea.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file\": \"{}\",", filename);
            println!("  \"functions\": [");
            for (i, func) in functions.iter().enumerate() {
                println!("    {{");
                println!("      \"name\": \"{}\",", func.name);
                println!("      \"params\": [{}],",
                    func.params.iter()
                        .map(|p| format!("\"{}\"", p.name))
                        .collect::<Vec<_>>().join(", "));
                println!("      \"is_nogc\": {}", func.is_nogc);
                print!("    }}");
                if i + 1 < functions.len() { print!(","); }
                println!();
            }
            println!("  ]");
            println!("}}");
        }
        _ => {
            eprintln!("{} Explaining `{}`...",
                output::colorize(ea.output, output::BOLD_CYAN, "[explain]"),
                filename);
            eprintln!();

            if functions.is_empty() {
                eprintln!("No functions found.");
                return;
            }

            for func in &functions {
                // Function header
                let params_str: Vec<String> = func.params.iter()
                    .map(|p| format!("{}: {}", p.name, p.ty_name))
                    .collect();
                let sig = format!("fn {}({})", func.name, params_str.join(", "));
                eprintln!("{}", output::colorize(ea.output, output::BOLD, &sig));

                // Annotations
                if func.is_nogc {
                    eprintln!("  {} #[nogc]",
                        output::colorize(ea.output, output::CYAN, "annotation:"));
                }
                if !func.decorators.is_empty() {
                    eprintln!("  {} @{}",
                        output::colorize(ea.output, output::CYAN, "decorators:"),
                        func.decorators.join(", @"));
                }

                // Body summary
                let stmt_count = func.body.stmts.len();
                let has_result = func.body.expr.is_some();
                eprintln!("  {} {} statements{}",
                    output::colorize(ea.output, output::CYAN, "body:"),
                    stmt_count,
                    if has_result { " + tail expression" } else { "" });

                // Verbose: print HIR body
                if ea.verbose {
                    eprintln!("  {} {{", output::colorize(ea.output, output::DIM, "lowered"));
                    for stmt in &func.body.stmts {
                        eprintln!("    {:#?}", stmt);
                    }
                    if let Some(ref result) = func.body.expr {
                        eprintln!("    => {:#?}", result);
                    }
                    eprintln!("  }}");
                }

                eprintln!();
            }

            // Struct summary
            if !structs.is_empty() {
                eprintln!("{}", output::colorize(ea.output, output::BOLD, "Structs:"));
                for s in &structs {
                    let fields: Vec<String> = s.fields.iter()
                        .map(|(name, ty)| format!("{}: {}", name, ty))
                        .collect();
                    eprintln!("  struct {} {{ {} }}", s.name, fields.join(", "));
                }
                eprintln!();
            }

            // Summary table
            let mut t = crate::table::Table::new(vec!["Element", "Count"]);
            t.add_row_owned(vec!["Functions".into(), format!("{}", functions.len())]);
            t.add_row_owned(vec!["Structs".into(), format!("{}", structs.len())]);
            let nogc_count = functions.iter().filter(|f| f.is_nogc).count();
            t.add_row_owned(vec!["NoGC functions".into(), format!("{}", nogc_count)]);
            t.add_row_owned(vec!["Total items".into(), format!("{}", hir.items.len())]);
            eprint!("{}", t.render());
        }
    }
}

pub fn print_help() {
    eprintln!("cjcl explain - Two modes, dispatched on argument shape.");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  cjcl explain <file.cjcl> [flags]   Show desugared HIR for a CJC source file.");
    eprintln!("  cjcl explain EXXXX                 Show the long-form explanation of an error code.");
    eprintln!("  cjcl explain WXXXX                 Show the long-form explanation of a warning code.");
    eprintln!();
    eprintln!("Error-code mode (pedagogy layer):");
    eprintln!("  - Prints the long-form Elm-style explanation: what happened,");
    eprintln!("    why CJC-Lang enforces this, how to fix, common pitfalls.");
    eprintln!("  - Codes with no explanation yet show a header + a contribution hint.");
    eprintln!("  - Examples: `cjcl explain E1003`, `cjcl explain W0001`");
    eprintln!();
    eprintln!("File mode (HIR lowering):");
    eprintln!("  - Function signatures after lowering");
    eprintln!("  - Desugared loop/match forms (with --verbose)");
    eprintln!("  - NoGC annotations and decorators");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  -v, --verbose   Show full HIR body for each function (file mode only)");
    eprintln!("  --plain         Plain text output");
    eprintln!("  --json          JSON output (metadata only in error-code mode)");
    eprintln!("  --color         Color output (default)");
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── is_error_code_shape ─────────────────────────────────────────────

    #[test]
    fn test_is_error_code_shape_accepts_canonical_codes() {
        assert!(is_error_code_shape("E0001"));
        assert!(is_error_code_shape("E1003"));
        assert!(is_error_code_shape("E9999"));
        assert!(is_error_code_shape("W0001"));
        assert!(is_error_code_shape("W9999"));
        assert!(is_error_code_shape("E0601")); // snap error
    }

    #[test]
    fn test_is_error_code_shape_rejects_filenames() {
        // .cjcl filenames must never accidentally match.
        assert!(!is_error_code_shape("main.cjcl"));
        assert!(!is_error_code_shape("foo"));
        assert!(!is_error_code_shape("test_program.cjcl"));
        assert!(!is_error_code_shape("E1003.md"));
        assert!(!is_error_code_shape("E1003.cjcl"));
        // Paths.
        assert!(!is_error_code_shape("./E1003"));
        assert!(!is_error_code_shape("dir/E1003"));
    }

    #[test]
    fn test_is_error_code_shape_rejects_malformed() {
        assert!(!is_error_code_shape(""));
        assert!(!is_error_code_shape("E"));
        assert!(!is_error_code_shape("E1"));
        assert!(!is_error_code_shape("E123"));
        assert!(!is_error_code_shape("E12345")); // too long
        // Case-sensitive — canonical form is uppercase prefix.
        assert!(!is_error_code_shape("e1003"));
        assert!(!is_error_code_shape("w0001"));
        // Wrong prefix letter.
        assert!(!is_error_code_shape("X1003"));
        assert!(!is_error_code_shape("01003"));
        // Non-digit in digit positions.
        assert!(!is_error_code_shape("E100A"));
        assert!(!is_error_code_shape("EABCD"));
    }

    // ── format_error_lookup ─────────────────────────────────────────────

    #[test]
    fn test_format_error_lookup_known_documented() {
        // E1003 ships with v0 pedagogy — must return a rich explanation.
        let out = format_error_lookup("E1003").expect("E1003 must be known");
        // Header contains the canonical code, category, and template.
        assert!(out.contains("E1003"));
        assert!(out.contains("parser"));
        assert!(out.contains("missing type annotation"));
        // Pedagogy sections from the markdown file are present.
        assert!(out.contains("## What happened"));
        assert!(out.contains("## Why it matters"));
        assert!(out.contains("## How to fix"));
    }

    #[test]
    fn test_format_error_lookup_known_undocumented() {
        // E3001 (borrow/ownership) exists as a code but has no explanation
        // yet -- must succeed with a header + a contribution hint, not error.
        let out = format_error_lookup("E3001").expect("E3001 must be known");
        assert!(out.contains("E3001"));
        assert!(out.contains("ownership"));
        assert!(out.contains("No detailed explanation"));
        assert!(out.contains("Contributions welcome"));
    }

    #[test]
    fn test_format_error_lookup_unknown_code() {
        // Well-formed but not assigned.
        let err = format_error_lookup("E9999")
            .expect_err("E9999 must be rejected as unknown");
        assert!(err.contains("E9999"));
        assert!(err.contains("unknown"));
    }

    #[test]
    fn test_format_error_lookup_malformed() {
        // Note: is_error_code_shape rejects malformed inputs *before* they
        // reach format_error_lookup at runtime, but format_error_lookup
        // itself must still cleanly reject malformed input — defence in
        // depth.
        assert!(format_error_lookup("not_a_code").is_err());
        assert!(format_error_lookup("").is_err());
        assert!(format_error_lookup("e1003").is_err()); // case-sensitive
    }

    #[test]
    fn test_format_error_lookup_all_warning_codes() {
        // Every defined warning code must render without error.
        for code in cjc_diag::ErrorCode::ALL_CODES {
            if code.code_str().starts_with('W') {
                let out = format_error_lookup(code.code_str())
                    .unwrap_or_else(|_| panic!("{} should be renderable", code));
                assert!(out.contains(code.code_str()));
                assert!(out.contains("warning"));
            }
        }
    }
}
