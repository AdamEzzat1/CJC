//! `cjc test` — Native test runner for CJC source files.
//!
//! Discovers and runs test functions from a CJC source file.
//! Test functions are those whose name starts with `test_` or
//! that are decorated with `@test`. Execution is deterministic:
//! tests run in alphabetical order with a fresh interpreter per test.

use std::fs;
use std::process;
use std::time::Instant;
use crate::output::{self, OutputMode};

pub struct TestArgs {
    pub file: String,
    pub seed: u64,
    pub output: OutputMode,
    pub verbose: bool,
    pub filter: Option<String>,
}

impl Default for TestArgs {
    fn default() -> Self {
        Self {
            file: String::new(),
            seed: 42,
            output: OutputMode::Color,
            verbose: false,
            filter: None,
        }
    }
}

pub fn parse_args(args: &[String]) -> TestArgs {
    let mut ta = TestArgs::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                if i < args.len() { ta.seed = args[i].parse().unwrap_or(42); }
            }
            "--filter" => {
                i += 1;
                if i < args.len() { ta.filter = Some(args[i].clone()); }
            }
            "-v" | "--verbose" => ta.verbose = true,
            "--plain" => ta.output = OutputMode::Plain,
            "--json" => ta.output = OutputMode::Json,
            "--color" => ta.output = OutputMode::Color,
            other if !other.starts_with('-') => ta.file = other.to_string(),
            other => {
                eprintln!("error: unknown flag `{}` for `cjc test`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    if ta.file.is_empty() {
        eprintln!("error: `cjc test` requires a .cjc file argument");
        process::exit(1);
    }
    ta
}

pub fn run(args: &[String]) {
    let ta = parse_args(args);

    let source = match fs::read_to_string(&ta.file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read `{}`: {}", ta.file, e);
            process::exit(1);
        }
    };

    let filename = ta.file.replace('\\', "/");

    let (program, diags) = cjc_parser::parse_source(&source);
    if diags.has_errors() {
        let rendered = diags.render_all_color(&source, &filename, ta.output.use_color());
        eprint!("{}", rendered);
        process::exit(1);
    }

    // Discover test functions
    let mut test_fns: Vec<String> = Vec::new();
    for decl in &program.declarations {
        if let cjc_ast::DeclKind::Fn(f) = &decl.kind {
            let name = &f.name.name;
            let is_test_name = name.starts_with("test_");
            let is_test_decorator = f.decorators.iter().any(|d| d.name.name == "test");
            if is_test_name || is_test_decorator {
                if let Some(ref filter) = ta.filter {
                    if !name.contains(filter.as_str()) {
                        continue;
                    }
                }
                test_fns.push(name.clone());
            }
        }
    }

    // Sort alphabetically for deterministic ordering
    test_fns.sort();

    if test_fns.is_empty() {
        match ta.output {
            OutputMode::Json => {
                println!("{{\"file\": \"{}\", \"tests\": 0, \"passed\": 0, \"failed\": 0}}", filename);
            }
            _ => {
                eprintln!("No test functions found in `{}`", filename);
                eprintln!("(Test functions start with `test_` or use @test decorator)");
            }
        }
        return;
    }

    let total = test_fns.len();
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut results: Vec<(String, bool, f64)> = Vec::new();

    if ta.output != OutputMode::Json {
        eprintln!("{} Running {} tests (seed={})",
            output::colorize(ta.output, output::BOLD_CYAN, "[test]"),
            total, ta.seed);
        eprintln!();
    }

    for test_name in &test_fns {
        // Build a source that includes all declarations plus a call to the test function
        let test_source = format!("{}\n{}();\n", source, test_name);

        let (test_program, test_diags) = cjc_parser::parse_source(&test_source);
        if test_diags.has_errors() {
            failed += 1;
            let duration = 0.0;
            results.push((test_name.clone(), false, duration));
            if ta.output != OutputMode::Json {
                eprintln!("  {} {} {}",
                    test_name,
                    dots(test_name, 40),
                    output::colorize(ta.output, output::BOLD_RED, "FAIL (parse error)"));
            }
            continue;
        }

        let start = Instant::now();
        let mut interpreter = cjc_eval::Interpreter::new(ta.seed);
        let result = interpreter.exec(&test_program);
        let elapsed = start.elapsed();
        let duration_ms = elapsed.as_secs_f64() * 1000.0;

        match result {
            Ok(_) => {
                passed += 1;
                results.push((test_name.clone(), true, duration_ms));
                if ta.output != OutputMode::Json {
                    eprintln!("  {} {} {} ({:.1}ms)",
                        test_name,
                        dots(test_name, 40),
                        output::colorize(ta.output, output::BOLD_GREEN, "PASS"),
                        duration_ms);
                }
                if ta.verbose && !interpreter.output.is_empty() {
                    for line in &interpreter.output {
                        eprintln!("    | {}", line);
                    }
                }
            }
            Err(e) => {
                failed += 1;
                results.push((test_name.clone(), false, duration_ms));
                if ta.output != OutputMode::Json {
                    eprintln!("  {} {} {} ({:.1}ms)",
                        test_name,
                        dots(test_name, 40),
                        output::colorize(ta.output, output::BOLD_RED, "FAIL"),
                        duration_ms);
                    if ta.verbose {
                        eprintln!("    error: {}", e);
                    }
                }
            }
        }
    }

    match ta.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file\": \"{}\",", filename);
            println!("  \"seed\": {},", ta.seed);
            println!("  \"tests\": {},", total);
            println!("  \"passed\": {},", passed);
            println!("  \"failed\": {},", failed);
            println!("  \"results\": [");
            for (i, (name, pass, dur)) in results.iter().enumerate() {
                print!("    {{\"name\": \"{}\", \"passed\": {}, \"duration_ms\": {:.2}}}",
                    name, pass, dur);
                if i + 1 < results.len() { print!(","); }
                println!();
            }
            println!("  ]");
            println!("}}");
        }
        _ => {
            eprintln!();
            eprintln!("{} passed, {} failed",
                output::colorize(ta.output, output::BOLD_GREEN, &format!("{}", passed)),
                if failed > 0 {
                    output::colorize(ta.output, output::BOLD_RED, &format!("{}", failed))
                } else {
                    format!("{}", failed)
                });

            if failed > 0 {
                process::exit(1);
            }
        }
    }
}

fn dots(name: &str, total_width: usize) -> String {
    let name_len = name.len();
    let dot_count = if total_width > name_len + 2 {
        total_width - name_len - 2
    } else {
        2
    };
    ".".repeat(dot_count)
}

pub fn print_help() {
    eprintln!("cjc test — Native test runner for CJC source files");
    eprintln!();
    eprintln!("Usage: cjc test <file.cjc> [flags]");
    eprintln!();
    eprintln!("Discovers functions starting with `test_` or decorated with @test.");
    eprintln!("Tests run in alphabetical order with a fresh interpreter each.");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  --seed <N>         RNG seed (default: 42)");
    eprintln!("  --filter <pattern> Only run tests matching pattern");
    eprintln!("  -v, --verbose      Show test output and error details");
    eprintln!("  --plain            Plain text output");
    eprintln!("  --json             JSON output");
    eprintln!("  --color            Color output (default)");
}
