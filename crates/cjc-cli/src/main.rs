//! CJC CLI — Entry point for the CJC programming language.
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
//!   --multi-file                 Enable multi-file module resolution
//!   --color                      Force color output
//!   --no-color                   Disable color output
//!   --help                       Print usage and exit
//!   --version                    Print version and exit

use std::env;
use std::fs;
use std::io;
use std::path::Path;
use std::process;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Pre-scan for --help and --version before anything else
    for arg in &args[1..] {
        match arg.as_str() {
            "--help" | "-h" => {
                print_usage();
                return;
            }
            "--version" | "-V" => {
                println!("cjc 0.1.0");
                return;
            }
            _ => {}
        }
    }

    // Parse global flags (can appear anywhere)
    let mut _reproducible = false;
    let mut seed: u64 = 42;
    let mut time_execution = false;
    let mut mir_opt = false;
    let mut mir_mono = false;
    let mut multi_file = false;
    let mut force_color: Option<bool> = None; // None = auto-detect

    // Collect positional args (command and filename)
    let mut positional: Vec<String> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--reproducible" => _reproducible = true,
            "--time" => time_execution = true,
            "--mir-opt" => mir_opt = true,
            "--mir-mono" => mir_mono = true,
            "--multi-file" => multi_file = true,
            "--color" => force_color = Some(true),
            "--no-color" => force_color = Some(false),
            "--seed" => {
                i += 1;
                if i < args.len() {
                    seed = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("error: --seed requires a numeric argument");
                        process::exit(1);
                    });
                }
            }
            "--help" | "-h" | "--version" | "-V" => {
                // Already handled above
            }
            other if other.starts_with("--") => {
                eprintln!("warning: unknown flag `{}`", other);
            }
            _ => {
                positional.push(args[i].clone());
            }
        }
        i += 1;
    }

    // Resolve color: explicit flag overrides, otherwise default to true
    let use_color = force_color.unwrap_or(true);

    if positional.is_empty() {
        print_usage();
        process::exit(1);
    }

    let command = &positional[0];

    // REPL command does not require a filename
    if command == "repl" {
        cmd_repl(seed, use_color);
        return;
    }

    if positional.len() < 2 {
        eprintln!("error: command `{}` requires a filename argument", command);
        print_usage();
        process::exit(1);
    }

    let filename = &positional[1];

    // Read source file
    let source = match fs::read_to_string(filename) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read `{}`: {}", filename, e);
            process::exit(1);
        }
    };

    match command.as_str() {
        "lex" => cmd_lex(&source, filename, use_color),
        "parse" => cmd_parse(&source, filename, use_color),
        "check" => cmd_check(&source, filename, use_color),
        "run" => cmd_run(&source, filename, seed, time_execution, mir_opt, mir_mono, multi_file, use_color),
        _ => {
            eprintln!("error: unknown command `{}`", command);
            print_usage();
            process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!("CJC Programming Language v0.1.0");
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
    eprintln!("  --help, -h                      Print this help message");
    eprintln!("  --version, -V                   Print version information");
}

fn cmd_lex(source: &str, filename: &str, use_color: bool) {
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
        eprintln!("\n{}", diags.render_all_color(source, filename, use_color));
        process::exit(1);
    }
}

fn cmd_parse(source: &str, filename: &str, use_color: bool) {
    let lexer = cjc_lexer::Lexer::new(source);
    let (tokens, lex_diags) = lexer.tokenize();

    if lex_diags.has_errors() {
        eprintln!("{}", lex_diags.render_all_color(source, filename, use_color));
        process::exit(1);
    }

    let parser = cjc_parser::Parser::new(tokens);
    let (program, parse_diags) = parser.parse_program();

    if parse_diags.has_errors() {
        eprintln!("{}", parse_diags.render_all_color(source, filename, use_color));
        process::exit(1);
    }

    let pretty = cjc_ast::PrettyPrinter::new().print_program(&program);
    println!("{}", pretty);
}

fn cmd_check(source: &str, filename: &str, use_color: bool) {
    let lexer = cjc_lexer::Lexer::new(source);
    let (tokens, lex_diags) = lexer.tokenize();

    if lex_diags.has_errors() {
        eprintln!("{}", lex_diags.render_all_color(source, filename, use_color));
        process::exit(1);
    }

    let parser = cjc_parser::Parser::new(tokens);
    let (program, parse_diags) = parser.parse_program();

    if parse_diags.has_errors() {
        eprintln!("{}", parse_diags.render_all_color(source, filename, use_color));
        process::exit(1);
    }

    let mut checker = cjc_types::TypeChecker::new();
    checker.check_program(&program);

    if checker.diagnostics.has_errors() {
        eprintln!("{}", checker.diagnostics.render_all_color(source, filename, use_color));
        process::exit(1);
    }

    println!("OK — no errors found in `{}`", filename);
}

fn cmd_run(
    source: &str,
    filename: &str,
    seed: u64,
    time_execution: bool,
    mir_opt: bool,
    mir_mono: bool,
    multi_file: bool,
    use_color: bool,
) {
    let start = Instant::now();

    if multi_file {
        // Multi-file module system: resolve imports, merge, execute
        let entry_path = Path::new(filename)
            .canonicalize()
            .unwrap_or_else(|e| {
                eprintln!("error: could not resolve path `{}`: {}", filename, e);
                process::exit(1);
            });

        match cjc_mir_exec::run_program_with_modules(&entry_path, seed) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("{}", e);
                process::exit(1);
            }
        }
    } else {
        // Single-file mode (default)
        let lexer = cjc_lexer::Lexer::new(source);
        let (tokens, lex_diags) = lexer.tokenize();

        if lex_diags.has_errors() {
            eprintln!("{}", lex_diags.render_all_color(source, filename, use_color));
            process::exit(1);
        }

        let parser = cjc_parser::Parser::new(tokens);
        let (program, parse_diags) = parser.parse_program();

        if parse_diags.has_errors() {
            eprintln!("{}", parse_diags.render_all_color(source, filename, use_color));
            process::exit(1);
        }

        // Run NoGC verification unconditionally (only errors on is_nogc fns)
        if let Err(e) = cjc_mir_exec::verify_nogc(&program) {
            eprintln!("NoGC verification failed:\n{}", e);
            process::exit(1);
        }

        if mir_mono {
            // Run with MIR monomorphization + optimization
            match cjc_mir_exec::run_program_monomorphized(&program, seed) {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("{}", e);
                    process::exit(1);
                }
            }
        } else if mir_opt {
            // Run with MIR optimizations enabled
            match cjc_mir_exec::run_program_optimized(&program, seed) {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("{}", e);
                    process::exit(1);
                }
            }
        } else {
            // Run the program via AST interpreter (default)
            let mut interpreter = cjc_eval::Interpreter::new(seed);
            match interpreter.exec(&program) {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("{}", e);
                    process::exit(1);
                }
            }
        }
    }

    let elapsed = start.elapsed();

    if time_execution {
        eprintln!(
            "[cjc --time] Execution took {:.6} seconds ({} us)",
            elapsed.as_secs_f64(),
            elapsed.as_micros()
        );
    }
}

fn cmd_repl(seed: u64, use_color: bool) {
    use std::io::{BufRead, Write};

    let mut interpreter = cjc_eval::Interpreter::new(seed);
    let mut line_num = 0u64;

    eprintln!("CJC REPL v0.1.0 (type 'exit' or Ctrl+C to quit)");

    loop {
        eprint!("> ");
        io::stderr().flush().unwrap();

        let mut line = String::new();
        match io::stdin().lock().read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(_) => break,
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
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
                // Print any output lines captured by the interpreter
                for out_line in &interpreter.output {
                    println!("{}", out_line);
                }
                interpreter.output.clear();

                // Print the result value if it is not Void
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
