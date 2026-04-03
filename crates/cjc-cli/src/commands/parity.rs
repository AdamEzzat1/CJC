//! `cjc parity` — Dual-executor parity checker.
//!
//! Runs both the AST tree-walk interpreter (eval) and the MIR register-machine
//! executor (mir-exec) and compares their output. Verifies that both backends
//! produce identical results for the same program and seed.

use std::fs;
use std::process;
use crate::output::{self, OutputMode};

pub struct ParityArgs {
    pub file: String,
    pub seed: u64,
    pub output: OutputMode,
    pub verbose: bool,
}

impl Default for ParityArgs {
    fn default() -> Self {
        Self {
            file: String::new(),
            seed: 42,
            output: OutputMode::Color,
            verbose: false,
        }
    }
}

pub fn parse_args(args: &[String]) -> ParityArgs {
    let mut pa = ParityArgs::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                if i < args.len() { pa.seed = args[i].parse().unwrap_or(42); }
            }
            "-v" | "--verbose" => pa.verbose = true,
            "--plain" => pa.output = OutputMode::Plain,
            "--json" => pa.output = OutputMode::Json,
            "--color" => pa.output = OutputMode::Color,
            other if !other.starts_with('-') => pa.file = other.to_string(),
            other => {
                eprintln!("error: unknown flag `{}` for `cjc parity`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    if pa.file.is_empty() {
        eprintln!("error: `cjc parity` requires a .cjc file argument");
        process::exit(1);
    }
    pa
}

pub fn run(args: &[String]) {
    let pa = parse_args(args);

    let source = match fs::read_to_string(&pa.file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read `{}`: {}", pa.file, e);
            process::exit(1);
        }
    };

    let filename = pa.file.replace('\\', "/");

    let (program, diags) = cjc_parser::parse_source(&source);
    if diags.has_errors() {
        let rendered = diags.render_all_color(&source, &filename, pa.output.use_color());
        eprint!("{}", rendered);
        process::exit(1);
    }

    // Run eval (v1)
    let eval_result = {
        let mut interpreter = cjc_eval::Interpreter::new(pa.seed);
        match interpreter.exec(&program) {
            Ok(val) => Ok((format!("{}", val), interpreter.output.clone(), interpreter.gc_collections)),
            Err(e) => Err(format!("{}", e)),
        }
    };

    // Run mir-exec (v2)
    let mir_result = {
        match cjc_mir_exec::run_program_with_executor(&program, pa.seed) {
            Ok((val, exec)) => Ok((format!("{}", val), exec.output.clone(), exec.gc_collections)),
            Err(e) => Err(format!("{}", e)),
        }
    };

    // Compare
    let (eval_ok, eval_output, eval_val, eval_gc) = match &eval_result {
        Ok((val, out, gc)) => (true, out.clone(), val.clone(), *gc),
        Err(e) => (false, vec![], e.clone(), 0),
    };

    let (mir_ok, mir_output, mir_val, mir_gc) = match &mir_result {
        Ok((val, out, gc)) => (true, out.clone(), val.clone(), *gc),
        Err(e) => (false, vec![], e.clone(), 0),
    };

    let stdout_match = eval_output == mir_output;
    let value_match = eval_ok && mir_ok && eval_val == mir_val;
    let both_ok = eval_ok && mir_ok;
    let identical = stdout_match && (value_match || (!eval_ok && !mir_ok));

    match pa.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file\": \"{}\",", filename);
            println!("  \"seed\": {},", pa.seed);
            println!("  \"eval_ok\": {},", eval_ok);
            println!("  \"mir_ok\": {},", mir_ok);
            println!("  \"stdout_match\": {},", stdout_match);
            println!("  \"value_match\": {},", value_match);
            println!("  \"eval_gc\": {},", eval_gc);
            println!("  \"mir_gc\": {},", mir_gc);
            println!("  \"verdict\": \"{}\"", if identical { "IDENTICAL" } else { "DIVERGENT" });
            println!("}}");
        }
        _ => {
            eprintln!("{} eval vs mir-exec (seed={})",
                output::colorize(pa.output, output::BOLD_CYAN, "[parity]"),
                pa.seed);
            eprintln!();

            if both_ok {
                // Show output preview
                if pa.verbose {
                    let max_lines = 10;
                    if !eval_output.is_empty() {
                        eprintln!("  eval output ({} lines):", eval_output.len());
                        for line in eval_output.iter().take(max_lines) {
                            eprintln!("    {}", line);
                        }
                    }
                    if !mir_output.is_empty() {
                        eprintln!("  mir-exec output ({} lines):", mir_output.len());
                        for line in mir_output.iter().take(max_lines) {
                            eprintln!("    {}", line);
                        }
                    }
                    eprintln!();
                }
            } else {
                if !eval_ok {
                    eprintln!("  eval: {} {}",
                        output::colorize(pa.output, output::BOLD_RED, "ERROR"),
                        eval_val);
                }
                if !mir_ok {
                    eprintln!("  mir-exec: {} {}",
                        output::colorize(pa.output, output::BOLD_RED, "ERROR"),
                        mir_val);
                }
                eprintln!();
            }

            // Results table
            let mut t = crate::table::Table::new(vec!["Check", "Status"]);
            t.add_row_owned(vec![
                "stdout match".into(),
                if stdout_match {
                    output::colorize(pa.output, output::BOLD_GREEN, "PASS")
                } else {
                    output::colorize(pa.output, output::BOLD_RED, "FAIL")
                },
            ]);
            t.add_row_owned(vec![
                "value match".into(),
                if value_match {
                    output::colorize(pa.output, output::BOLD_GREEN, "PASS")
                } else if !both_ok {
                    output::colorize(pa.output, output::BOLD_YELLOW, "N/A (error)")
                } else {
                    output::colorize(pa.output, output::BOLD_RED, "FAIL")
                },
            ]);
            t.add_row_owned(vec![
                "GC collections".into(),
                format!("eval={}, mir={}", eval_gc, mir_gc),
            ]);
            eprint!("{}", t.render());

            // Show first difference if stdout doesn't match
            if !stdout_match && both_ok {
                let max_lines = eval_output.len().max(mir_output.len());
                for i in 0..max_lines {
                    let a = eval_output.get(i).map(|s| s.as_str()).unwrap_or("<missing>");
                    let b = mir_output.get(i).map(|s| s.as_str()).unwrap_or("<missing>");
                    if a != b {
                        eprintln!("\nFirst divergence at line {}:", i + 1);
                        eprintln!("  eval:     {}", a);
                        eprintln!("  mir-exec: {}", b);
                        break;
                    }
                }
            }

            let verdict = if identical {
                output::colorize(pa.output, output::BOLD_GREEN, "IDENTICAL")
            } else {
                output::colorize(pa.output, output::BOLD_RED, "DIVERGENT")
            };
            eprintln!("\nVerdict: {}", verdict);

            if !identical {
                process::exit(1);
            }
        }
    }
}

pub fn print_help() {
    eprintln!("cjc parity — Dual-executor parity checker");
    eprintln!();
    eprintln!("Usage: cjc parity <file.cjc> [flags]");
    eprintln!();
    eprintln!("Runs both eval (v1) and mir-exec (v2) and compares output.");
    eprintln!("Verifies that both backends produce identical results.");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  --seed <N>      RNG seed (default: 42)");
    eprintln!("  -v, --verbose   Show output from both executors");
    eprintln!("  --plain         Plain text output");
    eprintln!("  --json          JSON output");
    eprintln!("  --color         Color output (default)");
}
