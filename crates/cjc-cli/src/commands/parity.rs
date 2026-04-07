//! `cjcl parity` — Dual-executor parity checker.
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
    pub explain_mismatch: bool,
    pub save_report: Option<String>,
    pub function: Option<String>,
}

impl Default for ParityArgs {
    fn default() -> Self {
        Self {
            file: String::new(),
            seed: 42,
            output: OutputMode::Color,
            verbose: false,
            explain_mismatch: false,
            save_report: None,
            function: None,
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
            "--explain-mismatch" => pa.explain_mismatch = true,
            "--save-report" => {
                i += 1;
                if i < args.len() {
                    pa.save_report = Some(args[i].clone());
                } else {
                    eprintln!("error: --save-report requires a file argument");
                    process::exit(1);
                }
            }
            "--function" => {
                i += 1;
                if i < args.len() {
                    pa.function = Some(args[i].clone());
                } else {
                    eprintln!("error: --function requires a name argument");
                    process::exit(1);
                }
            }
            other if !other.starts_with('-') => pa.file = other.to_string(),
            other => {
                eprintln!("error: unknown flag `{}` for `cjcl parity`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    if pa.file.is_empty() {
        eprintln!("error: `cjcl parity` requires a .cjcl file argument");
        process::exit(1);
    }
    pa
}

/// Build a wrapper program that calls only the specified function.
/// We parse the full source, find the function, then wrap it with a call.
fn build_function_wrapper(source: &str, func_name: &str) -> Option<String> {
    // Simple approach: keep all declarations but append a call to the function
    // at the end. This preserves any dependencies the function needs.
    // We verify the function exists by looking for `fn <name>(`.
    let pattern = format!("fn {}(", func_name);
    if !source.contains(&pattern) {
        return None;
    }
    // Append a bare call. If the function takes arguments this will fail at
    // parse/eval time, which is acceptable (user should target zero-arg functions).
    Some(format!("{}\n{}();\n", source, func_name))
}

fn write_json_report(
    path: &str,
    filename: &str,
    seed: u64,
    eval_ok: bool,
    mir_ok: bool,
    stdout_match: bool,
    value_match: bool,
    eval_gc: u64,
    mir_gc: u64,
    identical: bool,
    eval_output: &[String],
    mir_output: &[String],
) {
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"file\": \"{}\",\n", filename));
    json.push_str(&format!("  \"seed\": {},\n", seed));
    json.push_str(&format!("  \"eval_ok\": {},\n", eval_ok));
    json.push_str(&format!("  \"mir_ok\": {},\n", mir_ok));
    json.push_str(&format!("  \"stdout_match\": {},\n", stdout_match));
    json.push_str(&format!("  \"value_match\": {},\n", value_match));
    json.push_str(&format!("  \"eval_gc\": {},\n", eval_gc));
    json.push_str(&format!("  \"mir_gc\": {},\n", mir_gc));
    json.push_str(&format!("  \"eval_output_lines\": {},\n", eval_output.len()));
    json.push_str(&format!("  \"mir_output_lines\": {},\n", mir_output.len()));
    json.push_str(&format!("  \"verdict\": \"{}\"\n", if identical { "IDENTICAL" } else { "DIVERGENT" }));
    json.push_str("}\n");

    if let Err(e) = fs::write(path, &json) {
        eprintln!("error: could not write report to `{}`: {}", path, e);
        process::exit(1);
    }
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

    // If --function is specified, build a wrapper source
    let effective_source = if let Some(ref func_name) = pa.function {
        match build_function_wrapper(&source, func_name) {
            Some(wrapped) => wrapped,
            None => {
                eprintln!("error: function `{}` not found in `{}`", func_name, filename);
                process::exit(1);
            }
        }
    } else {
        source.clone()
    };

    let (program, diags) = cjc_parser::parse_source(&effective_source);
    if diags.has_errors() {
        let rendered = diags.render_all_color(&effective_source, &filename, pa.output.use_color());
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

    // Save report if requested
    if let Some(ref report_path) = pa.save_report {
        write_json_report(
            report_path, &filename, pa.seed,
            eval_ok, mir_ok, stdout_match, value_match,
            eval_gc, mir_gc, identical,
            &eval_output, &mir_output,
        );
    }

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
            let target_label = if let Some(ref func_name) = pa.function {
                format!(" (function: {})", func_name)
            } else {
                String::new()
            };
            eprintln!("{} eval vs mir-exec (seed={}){}",
                output::colorize(pa.output, output::BOLD_CYAN, "[parity]"),
                pa.seed, target_label);
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

                        // --explain-mismatch: show context around the divergence
                        if pa.explain_mismatch {
                            eprintln!();
                            eprintln!("Context around divergence:");
                            let context_start = if i >= 2 { i - 2 } else { 0 };
                            let context_end = (i + 3).min(max_lines);
                            for ctx_i in context_start..context_end {
                                let ea = eval_output.get(ctx_i).map(|s| s.as_str()).unwrap_or("<missing>");
                                let eb = mir_output.get(ctx_i).map(|s| s.as_str()).unwrap_or("<missing>");
                                let marker = if ctx_i == i { ">>>" } else { "   " };
                                if ea == eb {
                                    eprintln!("  {} line {}: {}", marker, ctx_i + 1, ea);
                                } else {
                                    eprintln!("  {} line {} eval:     {}", marker, ctx_i + 1, ea);
                                    eprintln!("  {} line {} mir-exec: {}", marker, ctx_i + 1, eb);
                                }
                            }
                        }

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
    eprintln!("cjcl parity — Dual-executor parity checker");
    eprintln!();
    eprintln!("Usage: cjcl parity <file.cjcl> [flags]");
    eprintln!();
    eprintln!("Runs both eval (v1) and mir-exec (v2) and compares output.");
    eprintln!("Verifies that both backends produce identical results.");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  --seed <N>            RNG seed (default: 42)");
    eprintln!("  -v, --verbose         Show output from both executors");
    eprintln!("  --explain-mismatch    Show first divergent line with surrounding context");
    eprintln!("  --save-report <file>  Save parity report as JSON");
    eprintln!("  --function <name>     Only run parity check on a specific function");
    eprintln!("  --plain               Plain text output");
    eprintln!("  --json                JSON output");
    eprintln!("  --color               Color output (default)");
}
