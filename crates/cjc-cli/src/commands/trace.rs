//! `cjc trace` — Deterministic execution tracing.
//!
//! Traces execution of a CJC script, reporting function definitions,
//! execution events, output, GC behavior, and timing. All event
//! ordering is deterministic.

use std::fs;
use std::path::Path;
use std::process;
use std::time::Instant;
use crate::output::{self, OutputMode};

pub struct TraceArgs {
    pub file: String,
    pub seed: u64,
    pub output: OutputMode,
    pub show_ast: bool,
    pub show_output: bool,
    pub verbose: bool,
}

impl Default for TraceArgs {
    fn default() -> Self {
        Self {
            file: String::new(),
            seed: 42,
            output: OutputMode::Color,
            show_ast: false,
            show_output: true,
            verbose: false,
        }
    }
}

pub fn parse_args(args: &[String]) -> TraceArgs {
    let mut ta = TraceArgs::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                if i < args.len() { ta.seed = args[i].parse().unwrap_or(42); }
            }
            "--ast" => ta.show_ast = true,
            "--no-output" => ta.show_output = false,
            "-v" | "--verbose" => ta.verbose = true,
            "--plain" => ta.output = OutputMode::Plain,
            "--json" => ta.output = OutputMode::Json,
            "--color" => ta.output = OutputMode::Color,
            other if !other.starts_with('-') => ta.file = other.to_string(),
            other => {
                eprintln!("error: unknown flag `{}` for `cjc trace`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    if ta.file.is_empty() {
        eprintln!("error: `cjc trace` requires a .cjc file argument");
        process::exit(1);
    }
    ta
}

pub fn run(args: &[String]) {
    let ta = parse_args(args);
    let path = Path::new(&ta.file);

    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => { eprintln!("error: could not read `{}`: {}", ta.file, e); process::exit(1); }
    };

    let filename = ta.file.replace('\\', "/");

    // Phase 1: Parse
    let parse_start = Instant::now();
    let (program, parse_diags) = cjc_parser::parse_source(&source);
    let parse_time = parse_start.elapsed();

    if parse_diags.has_errors() {
        let rendered = parse_diags.render_all_color(&source, &filename, ta.output.use_color());
        eprint!("{}", rendered);
        process::exit(1);
    }

    // Phase 2: Static analysis of AST
    let mut fn_names = Vec::new();
    let mut struct_names = Vec::new();
    let mut let_count = 0u64;
    let mut stmt_count = 0u64;
    let mut decorator_count = 0u64;
    let mut effect_annotations = std::collections::BTreeSet::new();

    for decl in &program.declarations {
        match &decl.kind {
            cjc_ast::DeclKind::Fn(f) => {
                fn_names.push(f.name.name.clone());
                decorator_count += f.decorators.len() as u64;
                if f.is_nogc { effect_annotations.insert("nogc".to_string()); }
                if let Some(ref effs) = f.effect_annotation {
                    for e in effs { effect_annotations.insert(e.clone()); }
                }
            }
            cjc_ast::DeclKind::Struct(s) => struct_names.push(s.name.name.clone()),
            cjc_ast::DeclKind::Let(_) => let_count += 1,
            cjc_ast::DeclKind::Stmt(_) => stmt_count += 1,
            _ => {}
        }
    }

    // Phase 3: Execute with timing
    let exec_start = Instant::now();
    let mut interpreter = cjc_eval::Interpreter::new(ta.seed);
    let exec_result = interpreter.exec(&program);
    let exec_time = exec_start.elapsed();

    let exec_ok = exec_result.is_ok();
    let exec_error = exec_result.err().map(|e| format!("{}", e));

    // Phase 4: Report
    match ta.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file\": \"{}\",", filename);
            println!("  \"seed\": {},", ta.seed);
            println!("  \"parse_time_us\": {},", parse_time.as_micros());
            println!("  \"exec_time_us\": {},", exec_time.as_micros());
            println!("  \"exec_ok\": {},", exec_ok);
            if let Some(ref err) = exec_error {
                println!("  \"exec_error\": \"{}\",", err.replace('"', "\\\""));
            }
            println!("  \"functions\": {},", fn_names.len());
            println!("  \"structs\": {},", struct_names.len());
            println!("  \"let_bindings\": {},", let_count);
            println!("  \"statements\": {},", stmt_count);
            println!("  \"decorators\": {},", decorator_count);
            println!("  \"output_lines\": {},", interpreter.output.len());
            println!("  \"gc_collections\": {}", interpreter.gc_collections);
            println!("}}");
        }
        _ => {
            eprintln!("{} Tracing `{}`...", output::colorize(ta.output, output::BOLD_CYAN, "[trace]"), filename);
            eprintln!();

            // Summary table
            let mut t = crate::table::Table::new(vec!["Phase", "Detail"]);
            t.add_row_owned(vec!["Parse time".into(), format!("{:.3} ms", parse_time.as_secs_f64() * 1000.0)]);
            t.add_row_owned(vec!["Exec time".into(), format!("{:.3} ms", exec_time.as_secs_f64() * 1000.0)]);
            t.add_row_owned(vec!["Seed".into(), format!("{}", ta.seed)]);
            t.add_row_owned(vec!["Status".into(), if exec_ok {
                output::colorize(ta.output, output::BOLD_GREEN, "OK")
            } else {
                output::colorize(ta.output, output::BOLD_RED, "ERROR")
            }]);
            if let Some(ref err) = exec_error {
                t.add_row_owned(vec!["Error".into(), err.clone()]);
            }
            eprint!("{}", t.render());

            // AST summary
            eprintln!();
            let mut ast_t = crate::table::Table::new(vec!["AST Element", "Count"]);
            ast_t.add_row_owned(vec!["Functions".into(), format!("{}", fn_names.len())]);
            ast_t.add_row_owned(vec!["Structs".into(), format!("{}", struct_names.len())]);
            ast_t.add_row_owned(vec!["Let bindings".into(), format!("{}", let_count)]);
            ast_t.add_row_owned(vec!["Statements".into(), format!("{}", stmt_count)]);
            ast_t.add_row_owned(vec!["Decorators".into(), format!("{}", decorator_count)]);
            if !effect_annotations.is_empty() {
                let effs: Vec<String> = effect_annotations.into_iter().collect();
                ast_t.add_row_owned(vec!["Effects".into(), effs.join(", ")]);
            }
            eprint!("{}", ast_t.render());

            // Runtime summary
            eprintln!();
            let mut rt_t = crate::table::Table::new(vec!["Runtime", "Value"]);
            rt_t.add_row_owned(vec!["Output lines".into(), format!("{}", interpreter.output.len())]);
            rt_t.add_row_owned(vec!["GC collections".into(), format!("{}", interpreter.gc_collections)]);
            eprint!("{}", rt_t.render());

            // Function list
            if ta.verbose && !fn_names.is_empty() {
                eprintln!("\nFunctions defined:");
                for name in &fn_names {
                    eprintln!("  {}", name);
                }
            }

            // Output
            if ta.show_output && !interpreter.output.is_empty() {
                eprintln!("\nProgram output:");
                for line in &interpreter.output {
                    eprintln!("  {}", line);
                }
            }
        }
    }
}

pub fn print_help() {
    eprintln!("cjc trace — Deterministic execution tracing");
    eprintln!();
    eprintln!("Usage: cjc trace <file.cjc> [flags]");
    eprintln!();
    eprintln!("Reports:");
    eprintln!("  - Parse and execution timing");
    eprintln!("  - AST structure (functions, structs, let bindings)");
    eprintln!("  - Runtime behavior (output lines, GC collections)");
    eprintln!("  - Effect annotations");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  --seed <N>      RNG seed (default: 42)");
    eprintln!("  --ast           Show AST details");
    eprintln!("  --no-output     Suppress program output");
    eprintln!("  -v, --verbose   Show function list and extra details");
    eprintln!("  --plain         Plain text output");
    eprintln!("  --json          JSON output");
    eprintln!("  --color         Color output (default)");
}
