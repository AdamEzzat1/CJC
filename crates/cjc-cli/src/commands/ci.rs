//! `cjc ci` — Comprehensive CI diagnostic suite.
//!
//! Orchestrates multiple diagnostic checks in a single command:
//! doctor (parse/type errors), proof (determinism), parity (eval ≡ mir-exec),
//! test (native tests), and nogc (static verification).
//!
//! Exit code 0 = all pass. Non-zero = failure detected.

use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use crate::output::{self, OutputMode};

pub struct CiArgs {
    pub path: PathBuf,
    pub seed: u64,
    pub output: OutputMode,
    pub strict: bool,
}

impl Default for CiArgs {
    fn default() -> Self {
        Self {
            path: PathBuf::from("."),
            seed: 42,
            output: OutputMode::Color,
            strict: false,
        }
    }
}

pub fn parse_args(args: &[String]) -> CiArgs {
    let mut ca = CiArgs::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                if i < args.len() { ca.seed = args[i].parse().unwrap_or(42); }
            }
            "--strict" => ca.strict = true,
            "--plain" => ca.output = OutputMode::Plain,
            "--json" => ca.output = OutputMode::Json,
            "--color" => ca.output = OutputMode::Color,
            other if !other.starts_with('-') => ca.path = PathBuf::from(other),
            other => {
                eprintln!("error: unknown flag `{}` for `cjc ci`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    ca
}

struct CiResult {
    label: &'static str,
    passed: bool,
    detail: String,
}

pub fn run(args: &[String]) {
    let ca = parse_args(args);

    if ca.output != OutputMode::Json {
        eprintln!("{} Running full diagnostic suite on `{}`...",
            output::colorize(ca.output, output::BOLD_CYAN, "[ci]"),
            ca.path.display());
        eprintln!();
    }

    // Collect .cjc files
    let mut cjc_files: Vec<PathBuf> = Vec::new();
    collect_cjc_files(&ca.path, &mut cjc_files);
    cjc_files.sort();

    if cjc_files.is_empty() {
        if ca.output == OutputMode::Json {
            println!("{{\"error\": \"no .cjc files found\"}}");
        } else {
            eprintln!("No .cjc files found in `{}`", ca.path.display());
        }
        process::exit(1);
    }

    let mut results: Vec<CiResult> = Vec::new();

    // ── Step 1: Doctor (parse + type check) ──
    let mut parse_errors = 0usize;
    let mut type_warnings = 0usize;
    for file in &cjc_files {
        let source = match fs::read_to_string(file) {
            Ok(s) => s,
            Err(_) => { parse_errors += 1; continue; }
        };
        let (program, diags) = cjc_parser::parse_source(&source);
        if diags.has_errors() {
            parse_errors += 1;
            continue;
        }
        let mut checker = cjc_types::TypeChecker::new();
        checker.check_program(&program);
        if checker.diagnostics.has_errors() {
            type_warnings += 1;
        }
    }
    let doctor_pass = parse_errors == 0;
    let doctor_detail = if parse_errors > 0 {
        format!("{} parse errors, {} type warnings", parse_errors, type_warnings)
    } else if type_warnings > 0 {
        format!("OK ({} type warnings)", type_warnings)
    } else {
        "OK".to_string()
    };
    results.push(CiResult { label: "doctor", passed: doctor_pass, detail: doctor_detail });

    // ── Step 2: Proof (determinism across 3 seeds) ──
    let seeds = [ca.seed, ca.seed.wrapping_add(57), ca.seed.wrapping_add(113)];
    let mut proof_failures = 0usize;
    let mut proof_tested = 0usize;
    for file in &cjc_files {
        let source = match fs::read_to_string(file) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let (program, diags) = cjc_parser::parse_source(&source);
        if diags.has_errors() { continue; }

        for &seed in &seeds {
            // Run twice with same seed
            let mut interp1 = cjc_eval::Interpreter::new(seed);
            let mut interp2 = cjc_eval::Interpreter::new(seed);
            let r1 = interp1.exec(&program);
            let r2 = interp2.exec(&program);
            proof_tested += 1;
            if r1.is_ok() && r2.is_ok() {
                if interp1.output != interp2.output {
                    proof_failures += 1;
                }
            }
        }
    }
    let proof_pass = proof_failures == 0;
    let proof_detail = if proof_pass {
        format!("PASS ({} checks, {} seeds)", proof_tested, seeds.len())
    } else {
        format!("FAIL ({} divergences)", proof_failures)
    };
    results.push(CiResult { label: "proof", passed: proof_pass, detail: proof_detail });

    // ── Step 3: Parity (eval ≡ mir-exec) ──
    let mut parity_pass_count = 0usize;
    let mut parity_fail_count = 0usize;
    for file in &cjc_files {
        let source = match fs::read_to_string(file) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let (program, diags) = cjc_parser::parse_source(&source);
        if diags.has_errors() { continue; }

        let mut interpreter = cjc_eval::Interpreter::new(ca.seed);
        let eval_ok = interpreter.exec(&program);
        let eval_output = interpreter.output.clone();

        let mir_result = cjc_mir_exec::run_program_with_executor(&program, ca.seed);
        match (&eval_ok, &mir_result) {
            (Ok(_), Ok((_, exec))) => {
                if eval_output == exec.output {
                    parity_pass_count += 1;
                } else {
                    parity_fail_count += 1;
                }
            }
            _ => {
                // If both fail, that's still parity (both error)
                if eval_ok.is_err() && mir_result.is_err() {
                    parity_pass_count += 1;
                } else {
                    parity_fail_count += 1;
                }
            }
        }
    }
    let parity_pass = parity_fail_count == 0;
    let parity_detail = if parity_pass {
        format!("PASS (eval ≡ mir-exec, {} files)", parity_pass_count)
    } else {
        format!("FAIL ({} divergences)", parity_fail_count)
    };
    results.push(CiResult { label: "parity", passed: parity_pass, detail: parity_detail });

    // ── Step 4: Tests (find test_ functions) ──
    let mut test_total = 0usize;
    let mut test_passed = 0usize;
    for file in &cjc_files {
        let source = match fs::read_to_string(file) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let (program, diags) = cjc_parser::parse_source(&source);
        if diags.has_errors() { continue; }

        for decl in &program.declarations {
            if let cjc_ast::DeclKind::Fn(f) = &decl.kind {
                let name = &f.name.name;
                if name.starts_with("test_") || f.decorators.iter().any(|d| d.name.name == "test") {
                    test_total += 1;
                    let test_source = format!("{}\n{}();\n", source, name);
                    let (test_prog, test_diags) = cjc_parser::parse_source(&test_source);
                    if test_diags.has_errors() { continue; }
                    let mut interp = cjc_eval::Interpreter::new(ca.seed);
                    if interp.exec(&test_prog).is_ok() {
                        test_passed += 1;
                    }
                }
            }
        }
    }
    let tests_pass = test_passed == test_total;
    let test_detail = format!("{}/{} passed", test_passed, test_total);
    results.push(CiResult { label: "tests", passed: tests_pass, detail: test_detail });

    // ── Step 5: NoGC verification ──
    let mut nogc_total = 0usize;
    let mut nogc_verified = 0usize;
    for file in &cjc_files {
        let source = match fs::read_to_string(file) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let (program, diags) = cjc_parser::parse_source(&source);
        if diags.has_errors() { continue; }

        nogc_total += 1;
        if cjc_mir_exec::verify_nogc(&program).is_ok() {
            nogc_verified += 1;
        }
    }
    let nogc_pass = nogc_verified == nogc_total;
    let nogc_detail = format!("{}/{} verified", nogc_verified, nogc_total);
    results.push(CiResult { label: "nogc", passed: nogc_pass, detail: nogc_detail });

    // ── Report ──
    let all_pass = results.iter().all(|r| r.passed);
    let any_failure = results.iter().any(|r| !r.passed);

    match ca.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"path\": \"{}\",", ca.path.display().to_string().replace('\\', "/"));
            println!("  \"cjc_files\": {},", cjc_files.len());
            println!("  \"all_pass\": {},", all_pass);
            println!("  \"results\": {{");
            for (i, r) in results.iter().enumerate() {
                print!("    \"{}\": {{\"passed\": {}, \"detail\": \"{}\"}}",
                    r.label, r.passed, r.detail);
                if i + 1 < results.len() { print!(","); }
                println!();
            }
            println!("  }}");
            println!("}}");
        }
        _ => {
            for r in &results {
                let status = if r.passed {
                    output::colorize(ca.output, output::BOLD_GREEN, "PASS")
                } else {
                    output::colorize(ca.output, output::BOLD_RED, "FAIL")
                };
                eprintln!("  {} {} {}", r.label, dots(r.label, 10), format!("{} ({})", status, r.detail));
            }
            eprintln!();

            if all_pass {
                eprintln!("{}", output::colorize(ca.output, output::BOLD_GREEN, "EXIT 0"));
            } else {
                eprintln!("{}", output::colorize(ca.output, output::BOLD_RED, "EXIT 1"));
            }
        }
    }

    if any_failure || (ca.strict && results.iter().any(|r| r.detail.contains("warning"))) {
        process::exit(1);
    }
}

fn collect_cjc_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let read_dir = match fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(_) => {
            // Might be a single file
            if dir.is_file() && dir.extension().and_then(|e| e.to_str()) == Some("cjc") {
                out.push(dir.to_path_buf());
            }
            return;
        }
    };

    let mut entries: Vec<_> = read_dir.filter_map(|e| e.ok()).collect();
    entries.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with('.') || name == "target" || name == "node_modules" {
                continue;
            }
            collect_cjc_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("cjc") {
            out.push(path);
        }
    }
}

fn dots(label: &str, total_width: usize) -> String {
    let len = label.len();
    let count = if total_width > len + 2 { total_width - len } else { 2 };
    ".".repeat(count)
}

pub fn print_help() {
    eprintln!("cjc ci — Comprehensive CI diagnostic suite");
    eprintln!();
    eprintln!("Usage: cjc ci [path] [flags]");
    eprintln!();
    eprintln!("Runs: doctor + proof + parity + test + nogc");
    eprintln!("Exit code 0 = all pass. Non-zero = failure.");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  --seed <N>      Base RNG seed (default: 42)");
    eprintln!("  --strict        Exit non-zero on warnings");
    eprintln!("  --plain         Plain text output");
    eprintln!("  --json          JSON output");
    eprintln!("  --color         Color output (default)");
}
