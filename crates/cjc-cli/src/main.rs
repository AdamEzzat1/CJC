//! CJC CLI — Entry point for the CJC programming language.
//!
//! Zero-dependency argument parsing via manual `std::env::args()` iteration.
//! All CLI logic lives in this file — no external parsing crates.
//!
//! Usage:
//!   cjc run <file.cjc>           Run a CJC program
//!   cjc check <file.cjc>         Type-check without running
//!   cjc parse <file.cjc>         Parse and pretty-print
//!   cjc lex <file.cjc>           Tokenize and print tokens
//!   cjc repl                     Start an interactive REPL
//!
//! Flags:
//!   --reproducible               Enable reproducibility mode
//!   --seed <N>                   Set RNG seed (default: 42)
//!   --time                       Print execution time after running
//!   --mir-opt                    Enable MIR optimizations (CF + DCE)
//!   --mir-mono                   Enable MIR monomorphization
//!   --multi-file                 Enable multi-file module resolution
//!   --color                      Force color output
//!   --no-color                   Disable color output
//!   --diagnostic-format <fmt>    Diagnostic format: rich (default) or short
//!   --help, -h                   Print usage and exit
//!   --version, -V                Print version and exit

mod highlight;
mod line_editor;
mod table;

use std::env;
use std::fs;
use std::path::Path;
use std::process;
use std::time::Instant;

const VERSION: &str = "0.1.0";

// ── Typed CLI configuration ──────────────────────────────────────────

/// The subcommand to execute.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Command {
    Lex,
    Parse,
    Check,
    Run,
    Repl,
}

/// Fully parsed CLI configuration. Produced by `Config::from_args()`.
#[derive(Debug)]
struct Config {
    command: Command,
    filename: Option<String>,
    seed: u64,
    reproducible: bool,
    time: bool,
    mir_opt: bool,
    mir_mono: bool,
    multi_file: bool,
    use_color: bool,
    diag_format: cjc_diag::DiagnosticFormat,
}

/// Known flags for typo suggestions.
const KNOWN_FLAGS: &[&str] = &[
    "--reproducible",
    "--seed",
    "--time",
    "--mir-opt",
    "--mir-mono",
    "--multi-file",
    "--color",
    "--no-color",
    "--diagnostic-format",
    "--help",
    "--version",
    "-h",
    "-V",
];

/// Known commands for typo suggestions.
const KNOWN_COMMANDS: &[&str] = &["lex", "parse", "check", "run", "repl"];

impl Config {
    /// Parse CLI arguments from `std::env::args()`. Exits on error.
    fn from_args() -> Self {
        let args: Vec<String> = env::args().collect();

        // Pre-scan for --help and --version (handled before full parse)
        for arg in &args[1..] {
            match arg.as_str() {
                "--help" | "-h" => {
                    print_usage();
                    process::exit(0);
                }
                "--version" | "-V" => {
                    println!("cjc {}", VERSION);
                    process::exit(0);
                }
                _ => {}
            }
        }

        let mut reproducible = false;
        let mut seed: u64 = 42;
        let mut time = false;
        let mut mir_opt = false;
        let mut mir_mono = false;
        let mut multi_file = false;
        let mut force_color: Option<bool> = None;
        let mut diag_format = cjc_diag::DiagnosticFormat::Rich;
        let mut positional: Vec<String> = Vec::new();

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--reproducible" => reproducible = true,
                "--time" => time = true,
                "--mir-opt" => mir_opt = true,
                "--mir-mono" => mir_mono = true,
                "--multi-file" => multi_file = true,
                "--color" => force_color = Some(true),
                "--no-color" => force_color = Some(false),
                "--diagnostic-format" => {
                    i += 1;
                    if i >= args.len() {
                        cli_error("--diagnostic-format requires an argument (short or rich)");
                    }
                    match args[i].as_str() {
                        "short" => diag_format = cjc_diag::DiagnosticFormat::Short,
                        "rich" => diag_format = cjc_diag::DiagnosticFormat::Rich,
                        other => cli_error(&format!(
                            "unknown diagnostic format `{}` (expected `short` or `rich`)", other
                        )),
                    }
                }
                "--seed" => {
                    i += 1;
                    if i >= args.len() {
                        cli_error("--seed requires a numeric argument");
                    }
                    seed = args[i].parse().unwrap_or_else(|_| {
                        cli_error(&format!("invalid seed value `{}`", args[i]));
                    });
                }
                "--help" | "-h" | "--version" | "-V" => { /* already handled */ }
                other if other.starts_with("--") || other.starts_with('-') && other.len() > 1 => {
                    let suggestion = suggest_flag(other);
                    if let Some(s) = suggestion {
                        eprintln!("error: unknown flag `{}`\n  hint: did you mean `{}`?", other, s);
                    } else {
                        eprintln!("error: unknown flag `{}`", other);
                    }
                    process::exit(1);
                }
                _ => positional.push(args[i].clone()),
            }
            i += 1;
        }

        let use_color = force_color.unwrap_or(true);

        if positional.is_empty() {
            print_usage();
            process::exit(1);
        }

        let command = match positional[0].as_str() {
            "lex" => Command::Lex,
            "parse" => Command::Parse,
            "check" => Command::Check,
            "run" => Command::Run,
            "repl" => Command::Repl,
            other => {
                let suggestion = suggest_command(other);
                if let Some(s) = suggestion {
                    eprintln!("error: unknown command `{}`\n  hint: did you mean `{}`?", other, s);
                } else {
                    eprintln!("error: unknown command `{}`", other);
                }
                print_usage();
                process::exit(1);
            }
        };

        // Commands other than `repl` require a filename
        let filename = if command != Command::Repl {
            if positional.len() < 2 {
                cli_error(&format!("command `{}` requires a filename argument", positional[0]));
            }
            Some(positional[1].clone())
        } else {
            None
        };

        Config {
            command,
            filename,
            seed,
            reproducible,
            time,
            mir_opt,
            mir_mono,
            multi_file,
            use_color,
            diag_format,
        }
    }
}

// ── Error helpers ────────────────────────────────────────────────────

/// Print an error message and exit with code 1.
fn cli_error(msg: &str) -> ! {
    eprintln!("error: {}", msg);
    process::exit(1);
}

/// Suggest the closest known flag using edit distance (Levenshtein).
fn suggest_flag(input: &str) -> Option<&'static str> {
    suggest_closest(input, KNOWN_FLAGS, 3)
}

/// Suggest the closest known command using edit distance.
fn suggest_command(input: &str) -> Option<&'static str> {
    suggest_closest(input, KNOWN_COMMANDS, 3)
}

/// Find the closest match within `max_distance` using Levenshtein distance.
fn suggest_closest(input: &str, candidates: &[&'static str], max_distance: usize) -> Option<&'static str> {
    let mut best: Option<&str> = None;
    let mut best_dist = max_distance + 1;
    for &candidate in candidates {
        let d = levenshtein(input, candidate);
        if d < best_dist {
            best_dist = d;
            best = Some(candidate);
        }
    }
    best
}

/// Simple Levenshtein distance (no allocation beyond two rows).
fn levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();
    if m == 0 { return n; }
    if n == 0 { return m; }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1)
                .min(curr[j - 1] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

// ── Entry point ──────────────────────────────────────────────────────

fn main() {
    let config = Config::from_args();

    match config.command {
        Command::Repl => cmd_repl(config.seed, config.use_color),
        _ => {
            let filename = config.filename.as_deref().unwrap();
            let source = match fs::read_to_string(filename) {
                Ok(s) => s,
                Err(e) => cli_error(&format!("could not read `{}`: {}", filename, e)),
            };
            match config.command {
                Command::Lex => cmd_lex(&source, filename, config.use_color, config.diag_format),
                Command::Parse => cmd_parse(&source, filename, config.use_color, config.diag_format),
                Command::Check => cmd_check(&source, filename, config.use_color, config.diag_format),
                Command::Run => cmd_run(&source, filename, &config),
                Command::Repl => unreachable!(),
            }
        }
    }
}

// ── Usage ────────────────────────────────────────────────────────────

fn print_usage() {
    eprintln!("CJC Programming Language v{}", VERSION);
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  cjc lex <file.cjc>              Tokenize and print tokens");
    eprintln!("  cjc parse <file.cjc>            Parse and pretty-print AST");
    eprintln!("  cjc check <file.cjc>            Type-check without running");
    eprintln!("  cjc run <file.cjc>              Run a CJC program");
    eprintln!("  cjc repl                        Start an interactive REPL");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  --reproducible                  Enable reproducibility mode");
    eprintln!("  --seed <N>                      Set RNG seed (default: 42)");
    eprintln!("  --time                          Print execution time after running");
    eprintln!("  --mir-opt                       Enable MIR optimizations (CF + DCE)");
    eprintln!("  --mir-mono                      Enable MIR monomorphization");
    eprintln!("  --multi-file                    Enable multi-file module resolution");
    eprintln!("  --color                         Force color output");
    eprintln!("  --no-color                      Disable color output");
    eprintln!("  --diagnostic-format <fmt>       Diagnostic format: rich (default) or short");
    eprintln!("  --help, -h                      Print this help message");
    eprintln!("  --version, -V                   Print version information");
}

// ── Command implementations ──────────────────────────────────────────

fn render_diags(diags: &cjc_diag::DiagnosticBag, source: &str, filename: &str, use_color: bool, diag_format: cjc_diag::DiagnosticFormat) {
    eprintln!("{}", diags.render_all_with_options(source, filename, use_color, diag_format));
}

fn cmd_lex(source: &str, filename: &str, use_color: bool, diag_format: cjc_diag::DiagnosticFormat) {
    let lexer = cjc_lexer::Lexer::new(source);
    let (tokens, diags) = lexer.tokenize();

    for tok in &tokens {
        println!(
            "{:>4}..{:<4}  {:>15}  {}",
            tok.span.start,
            tok.span.end,
            format!("{:?}", tok.kind),
            tok.text
        );
    }

    if diags.has_errors() {
        eprintln!();
        render_diags(&diags, source, filename, use_color, diag_format);
        process::exit(1);
    }
}

fn cmd_parse(source: &str, filename: &str, use_color: bool, diag_format: cjc_diag::DiagnosticFormat) {
    let lexer = cjc_lexer::Lexer::new(source);
    let (tokens, lex_diags) = lexer.tokenize();

    if lex_diags.has_errors() {
        render_diags(&lex_diags, source, filename, use_color, diag_format);
        process::exit(1);
    }

    let parser = cjc_parser::Parser::new(tokens);
    let (program, parse_diags) = parser.parse_program();

    if parse_diags.has_errors() {
        render_diags(&parse_diags, source, filename, use_color, diag_format);
        process::exit(1);
    }

    let pretty = cjc_ast::PrettyPrinter::new().print_program(&program);
    println!("{}", pretty);
}

fn cmd_check(source: &str, filename: &str, use_color: bool, diag_format: cjc_diag::DiagnosticFormat) {
    let lexer = cjc_lexer::Lexer::new(source);
    let (tokens, lex_diags) = lexer.tokenize();

    if lex_diags.has_errors() {
        render_diags(&lex_diags, source, filename, use_color, diag_format);
        process::exit(1);
    }

    let parser = cjc_parser::Parser::new(tokens);
    let (program, parse_diags) = parser.parse_program();

    if parse_diags.has_errors() {
        render_diags(&parse_diags, source, filename, use_color, diag_format);
        process::exit(1);
    }

    let mut checker = cjc_types::TypeChecker::new();
    checker.check_program(&program);

    if checker.diagnostics.has_errors() {
        render_diags(&checker.diagnostics, source, filename, use_color, diag_format);
        process::exit(1);
    }

    println!("OK — no errors found in `{}`", filename);
}

fn cmd_run(source: &str, filename: &str, config: &Config) {
    let start = Instant::now();

    if config.multi_file {
        let entry_path = Path::new(filename)
            .canonicalize()
            .unwrap_or_else(|e| cli_error(&format!("could not resolve path `{}`: {}", filename, e)));

        match cjc_module::build_module_graph(&entry_path) {
            Ok(graph) => {
                let violations = cjc_module::check_visibility(&graph);
                if !violations.is_empty() {
                    for v in &violations {
                        eprintln!("error: {}", v);
                    }
                    process::exit(1);
                }
            }
            Err(e) => cli_error(&format!("{}", e)),
        }

        if let Err(e) = cjc_mir_exec::run_program_with_modules(&entry_path, config.seed) {
            eprintln!("{}", e);
            process::exit(1);
        }
    } else {
        let lexer = cjc_lexer::Lexer::new(source);
        let (tokens, lex_diags) = lexer.tokenize();

        if lex_diags.has_errors() {
            render_diags(&lex_diags, source, filename, config.use_color, config.diag_format);
            process::exit(1);
        }

        let parser = cjc_parser::Parser::new(tokens);
        let (program, parse_diags) = parser.parse_program();

        if parse_diags.has_errors() {
            render_diags(&parse_diags, source, filename, config.use_color, config.diag_format);
            process::exit(1);
        }

        if let Err(e) = cjc_mir_exec::verify_nogc(&program) {
            eprintln!("NoGC verification failed:\n{}", e);
            process::exit(1);
        }

        if config.mir_mono {
            if let Err(e) = cjc_mir_exec::run_program_monomorphized(&program, config.seed) {
                eprintln!("{}", e);
                process::exit(1);
            }
        } else if config.mir_opt {
            if let Err(e) = cjc_mir_exec::run_program_optimized(&program, config.seed) {
                eprintln!("{}", e);
                process::exit(1);
            }
        } else {
            let mut interpreter = cjc_eval::Interpreter::new(config.seed);
            if let Err(e) = interpreter.exec(&program) {
                eprintln!("{}", e);
                process::exit(1);
            }
        }
    }

    if config.time {
        let elapsed = start.elapsed();
        eprintln!(
            "[cjc --time] Execution took {:.6} seconds ({} us)",
            elapsed.as_secs_f64(),
            elapsed.as_micros()
        );
    }
}

fn cmd_repl(seed: u64, use_color: bool) {
    let mut interpreter = cjc_eval::Interpreter::new(seed);
    let mut editor = line_editor::LineEditor::new_with_color(use_color);
    let mut line_num = 0u64;

    eprintln!("CJC REPL v{}  (type :help for commands, :quit to exit)", VERSION);

    loop {
        let input = match editor.read_line("cjc> ") {
            line_editor::ReadResult::Line(line) => line,
            line_editor::ReadResult::Eof => break,
        };

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Meta-commands (colon-prefixed)
        if trimmed.starts_with(':') {
            match handle_meta_command(trimmed, &interpreter, use_color, seed) {
                MetaResult::Continue => continue,
                MetaResult::Quit => break,
                MetaResult::Reset => {
                    interpreter = cjc_eval::Interpreter::new(seed);
                    line_num = 0;
                    eprintln!("Environment reset.");
                    continue;
                }
            }
        }

        // Legacy exit commands
        if trimmed == "exit" || trimmed == "quit" {
            break;
        }

        line_num += 1;
        let filename = format!("<repl:{}>", line_num);

        let src = trimmed.to_string();
        let (program, diags) = cjc_parser::parse_source(&src);

        if diags.has_errors() {
            let rendered = diags.render_all_color(&src, &filename, use_color);
            eprint!("{}", rendered);
            continue;
        }

        match interpreter.exec(&program) {
            Ok(val) => {
                for out_line in &interpreter.output {
                    println!("{}", out_line);
                }
                interpreter.output.clear();

                match &val {
                    cjc_runtime::Value::Void => {}
                    other => println!("{}", other),
                }
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }
    }
}

enum MetaResult {
    Continue,
    Quit,
    Reset,
}

fn handle_meta_command(
    cmd: &str,
    interpreter: &cjc_eval::Interpreter,
    use_color: bool,
    seed: u64,
) -> MetaResult {
    let parts: Vec<&str> = cmd.splitn(2, ' ').collect();
    let command = parts[0];
    let arg = parts.get(1).copied().unwrap_or("");

    match command {
        ":help" | ":h" => {
            eprintln!("CJC REPL Commands:");
            eprintln!("  :help, :h          Show this help");
            eprintln!("  :quit, :q          Quit the REPL");
            eprintln!("  :reset             Reset environment (clear all bindings)");
            eprintln!("  :type <expr>       Show the type of an expression");
            eprintln!("  :ast <expr>        Show the AST of an expression");
            eprintln!("  :mir <expr>        Show the MIR of an expression");
            eprintln!("  :env               Show current variable bindings");
            eprintln!("  :seed              Show the current RNG seed");
            MetaResult::Continue
        }
        ":quit" | ":q" => MetaResult::Quit,
        ":reset" => MetaResult::Reset,
        ":type" | ":t" => {
            if arg.is_empty() {
                eprintln!("Usage: :type <expression>");
            } else {
                let (program, diags) = cjc_parser::parse_source(arg);
                if diags.has_errors() {
                    let rendered = diags.render_all_color(arg, "<repl:type>", use_color);
                    eprint!("{}", rendered);
                } else {
                    let mut checker = cjc_types::TypeChecker::new();
                    checker.check_program(&program);
                    if let Some(last) = program.declarations.last() {
                        match &last.kind {
                            cjc_ast::DeclKind::Fn(_) => eprintln!("fn"),
                            _ => eprintln!("(type checking complete, no errors)"),
                        }
                    }
                    if !checker.diagnostics.diagnostics.is_empty() {
                        for d in &checker.diagnostics.diagnostics {
                            eprintln!("[{}] {}", d.code, d.message);
                        }
                    }
                }
            }
            MetaResult::Continue
        }
        ":ast" => {
            if arg.is_empty() {
                eprintln!("Usage: :ast <expression>");
            } else {
                let (program, diags) = cjc_parser::parse_source(arg);
                if diags.has_errors() {
                    let rendered = diags.render_all_color(arg, "<repl:ast>", use_color);
                    eprint!("{}", rendered);
                } else {
                    eprintln!("{:#?}", program);
                }
            }
            MetaResult::Continue
        }
        ":mir" => {
            if arg.is_empty() {
                eprintln!("Usage: :mir <expression>");
            } else {
                let (program, diags) = cjc_parser::parse_source(arg);
                if diags.has_errors() {
                    let rendered = diags.render_all_color(arg, "<repl:mir>", use_color);
                    eprint!("{}", rendered);
                } else {
                    let mir_program = cjc_mir_exec::lower_to_mir(&program);
                    for func in &mir_program.functions {
                        eprintln!("fn {}:", func.name);
                        eprintln!("{:#?}", func.body);
                    }
                }
            }
            MetaResult::Continue
        }
        ":env" => {
            // Variable bindings
            let bindings = interpreter.list_bindings();
            if bindings.is_empty() {
                eprintln!("No variable bindings.");
            } else {
                let mut t = table::Table::new(vec!["Name", "Type", "Value"]);
                for (name, ty, val) in &bindings {
                    t.add_row_owned(vec![name.clone(), ty.clone(), val.clone()]);
                }
                eprint!("{}", t.render());
            }

            // Functions
            let fns = interpreter.list_functions();
            if !fns.is_empty() {
                eprintln!();
                let mut t = table::Table::new(vec!["Function"]);
                for name in &fns {
                    t.add_row(vec![name]);
                }
                eprint!("{}", t.render());
            }

            // Structs
            let structs = interpreter.list_structs();
            if !structs.is_empty() {
                eprintln!();
                let mut t = table::Table::new(vec!["Struct"]);
                for name in &structs {
                    t.add_row(vec![name]);
                }
                eprint!("{}", t.render());
            }

            eprintln!("\nseed = {}", seed);
            MetaResult::Continue
        }
        ":seed" => {
            eprintln!("RNG seed: {}", seed);
            MetaResult::Continue
        }
        _ => {
            eprintln!("Unknown command: {}. Type :help for available commands.", command);
            MetaResult::Continue
        }
    }
}

// ── Unit tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(levenshtein("hello", "hello"), 0);
    }

    #[test]
    fn test_levenshtein_empty() {
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("", ""), 0);
    }

    #[test]
    fn test_levenshtein_substitution() {
        assert_eq!(levenshtein("cat", "car"), 1);
    }

    #[test]
    fn test_levenshtein_insertion_deletion() {
        assert_eq!(levenshtein("abc", "abcd"), 1);
        assert_eq!(levenshtein("abcd", "abc"), 1);
    }

    #[test]
    fn test_suggest_flag_close_match() {
        // --colr -> --color (distance 1)
        assert_eq!(suggest_flag("--colr"), Some("--color"));
        // --mir-op -> --mir-opt (distance 1)
        assert_eq!(suggest_flag("--mir-op"), Some("--mir-opt"));
    }

    #[test]
    fn test_suggest_flag_no_match() {
        // Very far from any known flag
        assert_eq!(suggest_flag("--zzzzzzzzzzz"), None);
    }

    #[test]
    fn test_suggest_command_close_match() {
        assert_eq!(suggest_command("rn"), Some("run"));
        assert_eq!(suggest_command("chck"), Some("check"));
        assert_eq!(suggest_command("lx"), Some("lex"));
    }

    #[test]
    fn test_suggest_command_no_match() {
        assert_eq!(suggest_command("foobarquux"), None);
    }
}
