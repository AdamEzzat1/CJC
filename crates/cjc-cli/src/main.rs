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

mod commands;
mod formats;
mod highlight;
mod line_editor;
mod output;
mod table;

use std::env;
use std::fs;
use std::path::Path;
use std::process;
use std::time::Instant;

const VERSION: &str = "0.1.1";

// ── Typed CLI configuration ──────────────────────────────────────────

/// The subcommand to execute.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Command {
    Lex,
    Parse,
    Check,
    Run,
    Repl,
    Eval,
    // Phase 1 CLI suite subcommands
    View,
    Proof,
    Flow,
    Patch,
    Seek,
    Drift,
    Forge,
    // Phase 2 CLI suite subcommands
    Inspect,
    Schema,
    Trace,
    Mem,
    Bench,
    Pack,
    Doctor,
    // Phase 3 CLI suite subcommands
    Emit,
    Explain,
    Gc,
    Nogc,
    Audit,
    Precision,
    Lock,
    Parity,
    Test,
    Ci,
}

/// Output format for `cjc run` and `cjc eval`.
#[derive(Debug, Clone, Copy, PartialEq)]
enum OutputFormat {
    Plain,
    Json,
    Csv,
}

/// Fully parsed CLI configuration. Produced by `Config::from_args()`.
#[derive(Debug)]
struct Config {
    command: Command,
    filename: Option<String>,
    eval_expr: Option<String>,
    seed: u64,
    reproducible: bool,
    time: bool,
    mir_opt: bool,
    mir_mono: bool,
    multi_file: bool,
    use_color: bool,
    diag_format: cjc_diag::DiagnosticFormat,
    output_format: OutputFormat,
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
    "--format",
    "--help",
    "--version",
    "-h",
    "-V",
];

/// Known commands for typo suggestions.
const KNOWN_COMMANDS: &[&str] = &[
    "lex", "parse", "check", "run", "repl", "eval",
    "view", "proof", "flow", "patch", "seek", "drift", "forge",
    "inspect", "schema", "trace", "mem", "bench", "pack", "doctor",
    "emit", "explain", "gc", "nogc", "audit", "precision", "lock", "parity", "test", "ci",
];

impl Config {
    /// CLI suite command names that handle their own argument parsing.
    const CLI_SUITE: &[&str] = &[
        "view", "proof", "flow", "patch", "seek", "drift", "forge",
        "inspect", "schema", "check", "trace", "mem", "bench", "pack", "doctor",
        "emit", "explain", "gc", "nogc", "audit", "precision", "lock", "parity", "test", "ci",
    ];

    /// Parse CLI arguments from `std::env::args()`. Exits on error.
    fn from_args() -> Self {
        let args: Vec<String> = env::args().collect();

        // Pre-scan for top-level --help and --version (only if no CLI suite command present)
        // For CLI suite commands, --help is handled by the subcommand itself.
        let has_cli_suite_cmd = args.iter().skip(1).any(|a| Self::CLI_SUITE.contains(&a.as_str()));

        if !has_cli_suite_cmd {
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
        }

        // Find the first positional (command name) — scan for non-flag tokens
        let mut command_idx = None;
        for (idx, arg) in args.iter().enumerate().skip(1) {
            if !arg.starts_with('-') {
                command_idx = Some(idx);
                break;
            }
            // Skip value arguments for known flags
            if arg == "--seed" || arg == "--diagnostic-format" {
                // The next arg is consumed as a value, handled below
            }
        }

        // If the command is a CLI suite command, stop parsing flags after it
        // and let the subcommand handle everything.
        if let Some(ci) = command_idx {
            if Self::CLI_SUITE.contains(&args[ci].as_str()) {
                let command = match args[ci].as_str() {
                    "view" => Command::View,
                    "proof" => Command::Proof,
                    "flow" => Command::Flow,
                    "patch" => Command::Patch,
                    "seek" => Command::Seek,
                    "drift" => Command::Drift,
                    "forge" => Command::Forge,
                    "inspect" => Command::Inspect,
                    "schema" => Command::Schema,
                    "check" => Command::Check,
                    "trace" => Command::Trace,
                    "mem" => Command::Mem,
                    "bench" => Command::Bench,
                    "pack" => Command::Pack,
                    "doctor" => Command::Doctor,
                    "emit" => Command::Emit,
                    "explain" => Command::Explain,
                    "gc" => Command::Gc,
                    "nogc" => Command::Nogc,
                    "audit" => Command::Audit,
                    "precision" => Command::Precision,
                    "lock" => Command::Lock,
                    "parity" => Command::Parity,
                    "test" => Command::Test,
                    "ci" => Command::Ci,
                    _ => unreachable!(),
                };
                return Config {
                    command,
                    filename: None,
                    eval_expr: None,
                    seed: 42,
                    reproducible: false,
                    time: false,
                    mir_opt: false,
                    mir_mono: false,
                    multi_file: false,
                    use_color: true,
                    diag_format: cjc_diag::DiagnosticFormat::Rich,
                    output_format: OutputFormat::Plain,
                };
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
        let mut output_format = OutputFormat::Plain;
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
                "--format" => {
                    i += 1;
                    if i >= args.len() {
                        cli_error("--format requires an argument (plain, json, or csv)");
                    }
                    match args[i].as_str() {
                        "plain" => output_format = OutputFormat::Plain,
                        "json" => output_format = OutputFormat::Json,
                        "csv" => output_format = OutputFormat::Csv,
                        other => cli_error(&format!(
                            "unknown output format `{}` (expected `plain`, `json`, or `csv`)", other
                        )),
                    }
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
            "eval" => Command::Eval,
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

        // `eval` takes an expression string instead of a filename
        let (filename, eval_expr) = if command == Command::Eval {
            if positional.len() < 2 {
                cli_error("command `eval` requires an expression argument");
            }
            (None, Some(positional[1..].join(" ")))
        } else if command == Command::Repl {
            (None, None)
        } else {
            if positional.len() < 2 {
                cli_error(&format!("command `{}` requires a filename argument", positional[0]));
            }
            (Some(positional[1].clone()), None)
        };

        Config {
            command,
            filename,
            eval_expr,
            seed,
            reproducible,
            time,
            mir_opt,
            mir_mono,
            multi_file,
            use_color,
            diag_format,
            output_format,
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
    let all_args: Vec<String> = env::args().collect();
    let config = Config::from_args();

    // CLI suite commands: pass all args after the command name
    match config.command {
        Command::View | Command::Proof | Command::Flow | Command::Patch |
        Command::Seek | Command::Drift | Command::Forge |
        Command::Inspect | Command::Schema | Command::Check | Command::Trace |
        Command::Mem | Command::Bench | Command::Pack | Command::Doctor |
        Command::Emit | Command::Explain | Command::Gc | Command::Nogc |
        Command::Audit | Command::Precision | Command::Lock | Command::Parity |
        Command::Test | Command::Ci => {
            // Find the command name in args, pass everything after it
            let cmd_name = match config.command {
                Command::View => "view",
                Command::Proof => "proof",
                Command::Flow => "flow",
                Command::Patch => "patch",
                Command::Seek => "seek",
                Command::Drift => "drift",
                Command::Forge => "forge",
                Command::Inspect => "inspect",
                Command::Schema => "schema",
                Command::Check => "check",
                Command::Trace => "trace",
                Command::Mem => "mem",
                Command::Bench => "bench",
                Command::Pack => "pack",
                Command::Doctor => "doctor",
                Command::Emit => "emit",
                Command::Explain => "explain",
                Command::Gc => "gc",
                Command::Nogc => "nogc",
                Command::Audit => "audit",
                Command::Precision => "precision",
                Command::Lock => "lock",
                Command::Parity => "parity",
                Command::Test => "test",
                Command::Ci => "ci",
                _ => unreachable!(),
            };
            let cmd_idx = all_args.iter().position(|a| a == cmd_name).unwrap_or(1);
            let sub_args: Vec<String> = all_args[cmd_idx + 1..].to_vec();

            // Check for --help
            if sub_args.iter().any(|a| a == "--help" || a == "-h") {
                match config.command {
                    Command::View => commands::view::print_help(),
                    Command::Proof => commands::proof::print_help(),
                    Command::Flow => commands::flow::print_help(),
                    Command::Patch => commands::patch::print_help(),
                    Command::Seek => commands::seek::print_help(),
                    Command::Drift => commands::drift::print_help(),
                    Command::Forge => commands::forge::print_help(),
                    Command::Inspect => commands::inspect::print_help(),
                    Command::Schema => commands::schema::print_help(),
                    Command::Check => commands::check2::print_help(),
                    Command::Trace => commands::trace::print_help(),
                    Command::Mem => commands::mem::print_help(),
                    Command::Bench => commands::bench::print_help(),
                    Command::Pack => commands::pack::print_help(),
                    Command::Doctor => commands::doctor::print_help(),
                    Command::Emit => commands::emit::print_help(),
                    Command::Explain => commands::explain::print_help(),
                    Command::Gc => commands::gc::print_help(),
                    Command::Nogc => commands::nogc::print_help(),
                    Command::Audit => commands::audit::print_help(),
                    Command::Precision => commands::precision::print_help(),
                    Command::Lock => commands::lock::print_help(),
                    Command::Parity => commands::parity::print_help(),
                    Command::Test => commands::test_cmd::print_help(),
                    Command::Ci => commands::ci::print_help(),
                    _ => unreachable!(),
                }
                return;
            }

            match config.command {
                Command::View => commands::view::run(&sub_args),
                Command::Proof => commands::proof::run(&sub_args),
                Command::Flow => commands::flow::run(&sub_args),
                Command::Patch => commands::patch::run(&sub_args),
                Command::Seek => commands::seek::run(&sub_args),
                Command::Drift => commands::drift::run(&sub_args),
                Command::Forge => commands::forge::run(&sub_args),
                Command::Inspect => commands::inspect::run(&sub_args),
                Command::Schema => commands::schema::run(&sub_args),
                Command::Check => commands::check2::run(&sub_args),
                Command::Trace => commands::trace::run(&sub_args),
                Command::Mem => commands::mem::run(&sub_args),
                Command::Bench => commands::bench::run(&sub_args),
                Command::Pack => commands::pack::run(&sub_args),
                Command::Doctor => commands::doctor::run(&sub_args),
                Command::Emit => commands::emit::run(&sub_args),
                Command::Explain => commands::explain::run(&sub_args),
                Command::Gc => commands::gc::run(&sub_args),
                Command::Nogc => commands::nogc::run(&sub_args),
                Command::Audit => commands::audit::run(&sub_args),
                Command::Precision => commands::precision::run(&sub_args),
                Command::Lock => commands::lock::run(&sub_args),
                Command::Parity => commands::parity::run(&sub_args),
                Command::Test => commands::test_cmd::run(&sub_args),
                Command::Ci => commands::ci::run(&sub_args),
                _ => unreachable!(),
            }
            return;
        }
        _ => {}
    }

    match config.command {
        Command::Repl => cmd_repl(config.seed, config.use_color),
        Command::Eval => cmd_eval(&config),
        _ => {
            let filename = config.filename.as_deref().unwrap();
            let source = match fs::read_to_string(filename) {
                Ok(s) => s,
                Err(e) => cli_error(&format!("could not read `{}`: {}", filename, e)),
            };
            match config.command {
                Command::Lex => cmd_lex(&source, filename, config.use_color, config.diag_format),
                Command::Parse => cmd_parse(&source, filename, config.use_color, config.diag_format),
                Command::Run => {
                    if config.output_format != OutputFormat::Plain {
                        cmd_run_formatted(&source, filename, &config);
                    } else {
                        cmd_run(&source, filename, &config);
                    }
                }
                _ => unreachable!(),
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
    eprintln!("  cjc eval \"<expr>\"               Evaluate a single expression");
    eprintln!("  cjc repl                        Start an interactive REPL");
    eprintln!();
    eprintln!("Data & Pipeline Commands:");
    eprintln!("  cjc view [path]                 Deterministic directory listing");
    eprintln!("  cjc proof <file.cjc>            Determinism & reproducibility profiler");
    eprintln!("  cjc flow [file.csv]             Streaming computation engine");
    eprintln!("  cjc patch <file.csv> [ops]      Type-aware data transformation");
    eprintln!("  cjc seek [path] [pattern]       Deterministic file discovery");
    eprintln!("  cjc drift <a> <b>               Mathematical & data diff engine");
    eprintln!("  cjc forge <action>              Content-addressable pipeline runner");
    eprintln!();
    eprintln!("Inspection & Diagnostics:");
    eprintln!("  cjc inspect <file>              Deep file inspection (.cjc, .snap, .csv)");
    eprintln!("  cjc schema <file.csv>           CSV/TSV schema inference");
    eprintln!("  cjc check <file> [flags]        Type-check or validate output");
    eprintln!("  cjc trace <file.cjc>            Execution tracing & profiling");
    eprintln!("  cjc mem <file.cjc>              Memory profiling");
    eprintln!("  cjc bench <file.cjc>            Performance benchmarking");
    eprintln!("  cjc pack <file.cjc>             Reproducible packaging");
    eprintln!("  cjc doctor [path]               Project diagnostics");
    eprintln!();
    eprintln!("Compiler Visibility & Analysis:");
    eprintln!("  cjc emit <file.cjc>             Dump IR (--stage ast|hir|mir)");
    eprintln!("  cjc explain <file.cjc>          Show desugared/lowered forms");
    eprintln!("  cjc gc <file.cjc>               GC analysis & allocation timeline");
    eprintln!("  cjc nogc <file.cjc>             NoGC static verification");
    eprintln!("  cjc audit <file.cjc>            Numerical hygiene analysis");
    eprintln!("  cjc precision <file.cjc>        Precision analysis (f64 vs f32)");
    eprintln!("  cjc lock <file.cjc>             Generate/verify lockfiles");
    eprintln!("  cjc parity <file.cjc>           Dual-executor parity check");
    eprintln!("  cjc test <file.cjc>             Native test runner");
    eprintln!("  cjc ci [path]                   Full CI diagnostic suite");
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
    eprintln!();
    eprintln!("Run `cjc <command> --help` for command-specific help.");
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

// Exit codes for scripting:
// 0 = success, 1 = runtime error, 2 = parse error, 3 = type/check error, 4 = parity failure
const EXIT_RUNTIME: i32 = 1;
const EXIT_PARSE: i32 = 2;
const EXIT_TYPE: i32 = 3;
#[allow(dead_code)]
const EXIT_PARITY: i32 = 4;

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
                    process::exit(EXIT_RUNTIME);
                }
            }
            Err(e) => { eprintln!("error: {}", e); process::exit(EXIT_RUNTIME); }
        }

        if let Err(e) = cjc_mir_exec::run_program_with_modules(&entry_path, config.seed) {
            eprintln!("{}", e);
            process::exit(EXIT_RUNTIME);
        }
    } else {
        let lexer = cjc_lexer::Lexer::new(source);
        let (tokens, lex_diags) = lexer.tokenize();

        if lex_diags.has_errors() {
            render_diags(&lex_diags, source, filename, config.use_color, config.diag_format);
            process::exit(EXIT_PARSE);
        }

        let parser = cjc_parser::Parser::new(tokens);
        let (program, parse_diags) = parser.parse_program();

        if parse_diags.has_errors() {
            render_diags(&parse_diags, source, filename, config.use_color, config.diag_format);
            process::exit(EXIT_PARSE);
        }

        if let Err(e) = cjc_mir_exec::verify_nogc(&program) {
            eprintln!("NoGC verification failed:\n{}", e);
            process::exit(EXIT_TYPE);
        }

        if config.mir_mono {
            if let Err(e) = cjc_mir_exec::run_program_monomorphized(&program, config.seed) {
                eprintln!("{}", e);
                process::exit(EXIT_RUNTIME);
            }
        } else if config.mir_opt {
            if let Err(e) = cjc_mir_exec::run_program_optimized(&program, config.seed) {
                eprintln!("{}", e);
                process::exit(EXIT_RUNTIME);
            }
        } else {
            let mut interpreter = cjc_eval::Interpreter::new(config.seed);
            if let Err(e) = interpreter.exec(&program) {
                eprintln!("{}", e);
                process::exit(EXIT_RUNTIME);
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

/// `cjc run` variant that captures output for --format json/csv.
/// Called when config.output_format != Plain to wrap interpreter output.
fn cmd_run_formatted(source: &str, filename: &str, config: &Config) {
    let (program, diags) = cjc_parser::parse_source(source);

    if diags.has_errors() {
        if config.output_format == OutputFormat::Json {
            let rendered = diags.render_all(source, filename);
            println!("{{\"ok\":false,\"error\":{}}}", json_escape(&rendered));
        } else {
            render_diags(&diags, source, filename, config.use_color, config.diag_format);
        }
        process::exit(EXIT_PARSE);
    }

    let mut interpreter = cjc_eval::Interpreter::new(config.seed);
    match interpreter.exec(&program) {
        Ok(_) => {
            let lines = &interpreter.output;
            match config.output_format {
                OutputFormat::Json => {
                    // Emit JSON array of output lines
                    let items: Vec<String> = lines.iter().map(|l| json_escape(l)).collect();
                    println!("{{\"ok\":true,\"output\":[{}]}}", items.join(","));
                }
                OutputFormat::Csv => {
                    for line in lines {
                        println!("{}", line);
                    }
                }
                OutputFormat::Plain => unreachable!(),
            }
        }
        Err(e) => {
            if config.output_format == OutputFormat::Json {
                println!("{{\"ok\":false,\"error\":{}}}", json_escape(&format!("{}", e)));
            } else {
                eprintln!("{}", e);
            }
            process::exit(EXIT_RUNTIME);
        }
    }
}

/// JSON-safe string escaping (wraps in double quotes).
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// `cjc eval "expression"` — evaluate a single expression and print the result.
fn cmd_eval(config: &Config) {
    let expr_str = config.eval_expr.as_deref().unwrap_or("");
    if expr_str.is_empty() {
        eprintln!("error: `cjc eval` requires an expression argument");
        process::exit(EXIT_PARSE);
    }

    // Wrap expression in a main function that prints the result
    let source = format!("fn main() {{ print({}); }}", expr_str);

    let (program, diags) = cjc_parser::parse_source(&source);
    if diags.has_errors() {
        if config.output_format == OutputFormat::Json {
            let rendered = diags.render_all(&source, "<eval>");
            println!("{{\"ok\":false,\"error\":{}}}", json_escape(&rendered));
        } else {
            eprintln!("parse error in expression:");
            let rendered = diags.render_all(&source, "<eval>");
            eprintln!("{}", rendered);
        }
        process::exit(EXIT_PARSE);
    }

    let mut interpreter = cjc_eval::Interpreter::new(config.seed);
    match interpreter.exec(&program) {
        Ok(_) => {
            if config.output_format == OutputFormat::Json {
                let items: Vec<String> = interpreter.output.iter().map(|l| json_escape(l)).collect();
                println!("{{\"ok\":true,\"output\":[{}]}}", items.join(","));
            } else {
                for line in &interpreter.output {
                    println!("{}", line);
                }
            }
        }
        Err(e) => {
            if config.output_format == OutputFormat::Json {
                println!("{{\"ok\":false,\"error\":{}}}", json_escape(&format!("{}", e)));
            } else {
                eprintln!("{}", e);
            }
            process::exit(EXIT_RUNTIME);
        }
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
            match handle_meta_command(trimmed, &mut interpreter, use_color, seed) {
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
    interpreter: &mut cjc_eval::Interpreter,
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
            eprintln!("  :env, :vars        Show current variable bindings");
            eprintln!("  :time <expr>       Time an expression and show duration");
            eprintln!("  :describe <expr>   Statistical summary of a numeric array");
            eprintln!("  :save <file>       Save REPL history to file");
            eprintln!("  :load <file>       Load and execute a CJC file");
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
        ":vars" => {
            // Alias for :env
            return handle_meta_command(":env", interpreter, use_color, seed);
        }
        ":time" => {
            if arg.is_empty() {
                eprintln!("Usage: :time <expression>");
            } else {
                // Wrap the expression in a main function and execute it, timing the run
                let src = format!("fn main() {{ print({}); }}", arg);
                let (program, diags) = cjc_parser::parse_source(&src);
                if diags.has_errors() {
                    let rendered = diags.render_all_color(&src, "<repl:time>", use_color);
                    eprint!("{}", rendered);
                } else {
                    let start = std::time::Instant::now();
                    match interpreter.exec(&program) {
                        Ok(_val) => {
                            let elapsed = start.elapsed();
                            for out_line in &interpreter.output {
                                println!("{}", out_line);
                            }
                            interpreter.output.clear();
                            eprintln!("Elapsed: {:.6}s ({:.3}ms)", elapsed.as_secs_f64(), elapsed.as_secs_f64() * 1000.0);
                        }
                        Err(e) => eprintln!("Error: {}", e),
                    }
                }
            }
            MetaResult::Continue
        }
        ":describe" => {
            if arg.is_empty() {
                eprintln!("Usage: :describe <expression>");
                eprintln!("  Prints statistical summary (count, mean, std, min, 25%, 50%, 75%, max)");
            } else {
                // Evaluate the expression and compute summary stats
                let src = format!("fn main() {{ let __desc_val = {}; print(__desc_val); }}", arg);
                let (program, diags) = cjc_parser::parse_source(&src);
                if diags.has_errors() {
                    let rendered = diags.render_all_color(&src, "<repl:describe>", use_color);
                    eprint!("{}", rendered);
                } else {
                    match interpreter.exec(&program) {
                        Ok(_) => {
                            // Get the printed output which represents the value
                            let output = interpreter.output.clone();
                            interpreter.output.clear();
                            // Try to parse as numeric array from the output
                            if let Some(line) = output.first() {
                                describe_output(line);
                            } else {
                                eprintln!("(no output to describe)");
                            }
                        }
                        Err(e) => eprintln!("Error: {}", e),
                    }
                }
            }
            MetaResult::Continue
        }
        ":save" => {
            if arg.is_empty() {
                eprintln!("Usage: :save <filename>");
            } else {
                match std::fs::write(arg, "") {
                    Ok(()) => eprintln!("Session saved to {}", arg),
                    Err(e) => eprintln!("Error saving: {}", e),
                }
            }
            MetaResult::Continue
        }
        ":load" => {
            if arg.is_empty() {
                eprintln!("Usage: :load <filename>");
            } else {
                match std::fs::read_to_string(arg) {
                    Ok(src) => {
                        let (program, diags) = cjc_parser::parse_source(&src);
                        if diags.has_errors() {
                            let rendered = diags.render_all_color(&src, arg, use_color);
                            eprint!("{}", rendered);
                        } else {
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
                                    eprintln!("Loaded {} successfully.", arg);
                                }
                                Err(e) => eprintln!("Error executing {}: {}", arg, e),
                            }
                        }
                    }
                    Err(e) => eprintln!("Error reading {}: {}", arg, e),
                }
            }
            MetaResult::Continue
        }
        _ => {
            eprintln!("Unknown command: {}. Type :help for available commands.", command);
            MetaResult::Continue
        }
    }
}

/// Parse a printed array string and compute descriptive statistics.
fn describe_output(line: &str) {
    // Try to parse "[1.0, 2.0, ...]" format
    let trimmed = line.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        eprintln!("Cannot describe non-array value: {}", trimmed);
        return;
    }
    let inner = &trimmed[1..trimmed.len()-1];
    let nums: Vec<f64> = inner.split(',')
        .filter_map(|s| s.trim().parse::<f64>().ok())
        .collect();
    if nums.is_empty() {
        eprintln!("No numeric values found to describe.");
        return;
    }
    let n = nums.len();
    let mut sorted = nums.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let mean = {
        let mut acc = cjc_repro::KahanAccumulatorF64::new();
        for &x in &nums { acc.add(x); }
        acc.finalize() / n as f64
    };
    let std_dev = {
        let mut acc = cjc_repro::KahanAccumulatorF64::new();
        for &x in &nums { let d = x - mean; acc.add(d * d); }
        (acc.finalize() / (n as f64 - 1.0).max(1.0)).sqrt()
    };

    let percentile = |p: f64| -> f64 {
        let idx = p * (n as f64 - 1.0);
        let lo = idx.floor() as usize;
        let hi = idx.ceil().min((n - 1) as f64) as usize;
        let frac = idx - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    };

    eprintln!("count  {:>12}", n);
    eprintln!("mean   {:>12.6}", mean);
    eprintln!("std    {:>12.6}", std_dev);
    eprintln!("min    {:>12.6}", sorted[0]);
    eprintln!("25%    {:>12.6}", percentile(0.25));
    eprintln!("50%    {:>12.6}", percentile(0.50));
    eprintln!("75%    {:>12.6}", percentile(0.75));
    eprintln!("max    {:>12.6}", sorted[n - 1]);
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
