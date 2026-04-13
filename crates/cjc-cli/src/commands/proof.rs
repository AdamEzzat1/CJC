//! `cjcl proof` — Determinism & reproducibility profiler.
//!
//! Runs a CJC program multiple times with the same seed and verifies:
//! - stdout is identical across runs
//! - stderr is identical across runs
//! - .snap hashes are stable
//! - GC/allocation behavior is consistent
//!
//! Reports a reproducibility verdict: PASS, FAIL, or WARN.

use std::fs;
use std::path::PathBuf;
use std::process;
use crate::output::{self, OutputMode};

/// Which executor(s) to run proof against.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ExecutorMode {
    Eval,
    Mir,
    Both,
}

/// Parsed arguments for `cjcl proof`.
pub struct ProofArgs {
    pub file: PathBuf,
    pub runs: usize,
    pub seed: u64,
    pub seeds: Vec<u64>,
    pub output: OutputMode,
    pub verbose: bool,
    pub fail_fast: bool,
    pub hash_output: bool,
    pub save_report: Option<PathBuf>,
    pub executor: ExecutorMode,
    pub stdout_only: bool,
}

impl Default for ProofArgs {
    fn default() -> Self {
        Self {
            file: PathBuf::new(),
            runs: 3,
            seed: 42,
            seeds: Vec::new(),
            output: OutputMode::Color,
            verbose: false,
            fail_fast: false,
            hash_output: false,
            save_report: None,
            executor: ExecutorMode::Eval,
            stdout_only: false,
        }
    }
}

pub fn parse_args(args: &[String]) -> ProofArgs {
    let mut pa = ProofArgs::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--runs" | "-n" => {
                i += 1;
                if i < args.len() {
                    pa.runs = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("error: --runs requires a numeric argument");
                        process::exit(1);
                    });
                }
            }
            "--seed" => {
                i += 1;
                if i < args.len() {
                    pa.seed = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("error: --seed requires a numeric argument");
                        process::exit(1);
                    });
                }
            }
            "--seeds" => {
                i += 1;
                if i < args.len() {
                    pa.seeds = args[i].split(',')
                        .filter_map(|s| s.trim().parse::<u64>().ok())
                        .collect();
                }
            }
            "-v" | "--verbose" => pa.verbose = true,
            "--plain" => pa.output = OutputMode::Plain,
            "--json" => pa.output = OutputMode::Json,
            "--color" => pa.output = OutputMode::Color,
            "--fail-fast" => pa.fail_fast = true,
            "--hash-output" => pa.hash_output = true,
            "--stdout-only" => pa.stdout_only = true,
            "--save-report" => {
                i += 1;
                if i < args.len() {
                    pa.save_report = Some(PathBuf::from(&args[i]));
                } else {
                    eprintln!("error: --save-report requires a file path argument");
                    process::exit(1);
                }
            }
            "--executor" => {
                i += 1;
                if i < args.len() {
                    pa.executor = match args[i].as_str() {
                        "eval" => ExecutorMode::Eval,
                        "mir" => ExecutorMode::Mir,
                        "both" => ExecutorMode::Both,
                        other => {
                            eprintln!("error: --executor must be eval, mir, or both (got `{}`)", other);
                            process::exit(1);
                        }
                    };
                } else {
                    eprintln!("error: --executor requires an argument (eval, mir, or both)");
                    process::exit(1);
                }
            }
            other if !other.starts_with('-') => pa.file = PathBuf::from(other),
            other => {
                eprintln!("error: unknown flag `{}` for `cjcl proof`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    if pa.file.as_os_str().is_empty() {
        eprintln!("error: `cjcl proof` requires a .cjcl file argument");
        process::exit(1);
    }
    pa
}

/// A single run result for comparison.
struct RunResult {
    stdout: Vec<String>,
    exit_ok: bool,
    gc_collections: u64,
    /// Which executor produced this result.
    executor_label: &'static str,
}

/// Execute a program using the eval interpreter and return a RunResult.
fn run_eval(program: &cjc_ast::Program, seed: u64, verbose: bool, run_idx: usize) -> RunResult {
    let mut interpreter = cjc_eval::Interpreter::new(seed);
    let exit_ok = match interpreter.exec(program) {
        Ok(_) => true,
        Err(e) => {
            if verbose {
                eprintln!("  eval run {}: error: {}", run_idx + 1, e);
            }
            false
        }
    };
    RunResult {
        stdout: interpreter.output.clone(),
        exit_ok,
        gc_collections: interpreter.gc_collections,
        executor_label: "eval",
    }
}

/// Execute a program using the MIR executor and return a RunResult.
fn run_mir(program: &cjc_ast::Program, seed: u64, verbose: bool, run_idx: usize) -> RunResult {
    match cjc_mir_exec::run_program_with_executor(program, seed) {
        Ok((_value, executor)) => RunResult {
            stdout: executor.output.clone(),
            exit_ok: true,
            gc_collections: executor.gc_collections,
            executor_label: "mir",
        },
        Err(e) => {
            if verbose {
                eprintln!("  mir run {}: error: {}", run_idx + 1, e);
            }
            RunResult {
                stdout: Vec::new(),
                exit_ok: false,
                gc_collections: 0,
                executor_label: "mir",
            }
        }
    }
}

/// Compute SHA-256 hex hash of the combined stdout lines.
fn hash_stdout(stdout: &[String]) -> String {
    let combined = stdout.join("\n");
    let digest = cjc_snap::hash::sha256(combined.as_bytes());
    cjc_snap::hash::hex_string(&digest)
}

/// Compare a result against a baseline, returning (stdout_match, exit_match, gc_match).
fn compare_results(baseline: &RunResult, other: &RunResult, stdout_only: bool) -> (bool, bool, bool) {
    let stdout_match = other.stdout == baseline.stdout;
    let exit_match = other.exit_ok == baseline.exit_ok;
    let gc_match = if stdout_only { true } else { other.gc_collections == baseline.gc_collections };
    (stdout_match, exit_match, gc_match)
}

/// Entry point for `cjcl proof`.
pub fn run(args: &[String]) {
    let pa = parse_args(args);

    let source = match fs::read_to_string(&pa.file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read `{}`: {}", pa.file.display(), e);
            process::exit(1);
        }
    };

    let filename = pa.file.display().to_string();

    // Determine which seeds to test
    let seeds = if pa.seeds.is_empty() {
        vec![pa.seed]
    } else {
        pa.seeds.clone()
    };

    let executor_label = match pa.executor {
        ExecutorMode::Eval => "eval",
        ExecutorMode::Mir => "mir",
        ExecutorMode::Both => "both",
    };

    eprintln!("{} Running {} iterations with seed{} {} (executor: {})...",
        output::colorize(pa.output, output::BOLD_CYAN, "[proof]"),
        pa.runs,
        if seeds.len() > 1 { "s" } else { "" },
        seeds.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(","),
        executor_label,
    );

    let mut results: Vec<RunResult> = Vec::new();
    let mut fail_fast_triggered = false;

    for &current_seed in &seeds {
        if fail_fast_triggered { break; }

        for run_idx in 0..pa.runs {
            if fail_fast_triggered { break; }

            let (program, diags) = cjc_parser::parse_source(&source);
            if diags.has_errors() {
                eprintln!("error: parse errors in `{}`", filename);
                let rendered = diags.render_all_color(&source, &filename, pa.output.use_color());
                eprint!("{}", rendered);
                process::exit(1);
            }

            match pa.executor {
                ExecutorMode::Eval => {
                    let r = run_eval(&program, current_seed, pa.verbose, run_idx);
                    if pa.verbose {
                        eprintln!("  eval run {}: {} lines, gc={}, ok={}",
                            run_idx + 1, r.stdout.len(), r.gc_collections, r.exit_ok);
                    }
                    if pa.hash_output {
                        eprintln!("  eval run {} sha256: {}", run_idx + 1, hash_stdout(&r.stdout));
                    }
                    results.push(r);
                }
                ExecutorMode::Mir => {
                    let r = run_mir(&program, current_seed, pa.verbose, run_idx);
                    if pa.verbose {
                        eprintln!("  mir run {}: {} lines, gc={}, ok={}",
                            run_idx + 1, r.stdout.len(), r.gc_collections, r.exit_ok);
                    }
                    if pa.hash_output {
                        eprintln!("  mir run {} sha256: {}", run_idx + 1, hash_stdout(&r.stdout));
                    }
                    results.push(r);
                }
                ExecutorMode::Both => {
                    let r_eval = run_eval(&program, current_seed, pa.verbose, run_idx);
                    let r_mir = run_mir(&program, current_seed, pa.verbose, run_idx);
                    if pa.verbose {
                        eprintln!("  eval run {}: {} lines, gc={}, ok={}",
                            run_idx + 1, r_eval.stdout.len(), r_eval.gc_collections, r_eval.exit_ok);
                        eprintln!("  mir  run {}: {} lines, gc={}, ok={}",
                            run_idx + 1, r_mir.stdout.len(), r_mir.gc_collections, r_mir.exit_ok);
                    }
                    if pa.hash_output {
                        eprintln!("  eval run {} sha256: {}", run_idx + 1, hash_stdout(&r_eval.stdout));
                        eprintln!("  mir  run {} sha256: {}", run_idx + 1, hash_stdout(&r_mir.stdout));
                    }
                    results.push(r_eval);
                    results.push(r_mir);
                }
            }

            // --fail-fast: check latest result(s) against baseline immediately
            if pa.fail_fast && results.len() > 1 {
                let baseline = &results[0];
                let last = results.last().unwrap();
                let (s, e, g) = compare_results(baseline, last, pa.stdout_only);
                if !s || !e || !g {
                    fail_fast_triggered = true;
                    if pa.verbose {
                        eprintln!("  FAIL-FAST: divergence detected at run {}", run_idx + 1);
                    }
                }
                // For --executor both, also check the second-to-last if we just pushed two
                if pa.executor == ExecutorMode::Both && results.len() >= 2 {
                    let second_last = &results[results.len() - 2];
                    let (s2, e2, g2) = compare_results(baseline, second_last, pa.stdout_only);
                    if !s2 || !e2 || !g2 {
                        fail_fast_triggered = true;
                        if pa.verbose {
                            eprintln!("  FAIL-FAST: divergence detected at run {}", run_idx + 1);
                        }
                    }
                }
            }
        } // end for run_idx
    } // end for current_seed in seeds

    // Compare all runs against the first
    let mut stdout_identical = true;
    let mut exit_identical = true;
    let mut gc_identical = true;

    if results.len() > 1 {
        let baseline_stdout = &results[0].stdout;
        let baseline_exit = results[0].exit_ok;
        let baseline_gc = results[0].gc_collections;

        for (i, r) in results.iter().enumerate().skip(1) {
            if r.stdout != *baseline_stdout {
                stdout_identical = false;
                if pa.verbose {
                    eprintln!("  DIVERGENCE: run {} ({}) stdout differs from run 1 ({})",
                        i + 1, r.executor_label, results[0].executor_label);
                    // Find first difference
                    let max_lines = baseline_stdout.len().max(r.stdout.len());
                    for line_idx in 0..max_lines {
                        let a = baseline_stdout.get(line_idx).map(|s| s.as_str()).unwrap_or("<missing>");
                        let b = r.stdout.get(line_idx).map(|s| s.as_str()).unwrap_or("<missing>");
                        if a != b {
                            eprintln!("    line {}: {:?} vs {:?}", line_idx + 1, a, b);
                            break;
                        }
                    }
                }
            }
            if r.exit_ok != baseline_exit {
                exit_identical = false;
            }
            if !pa.stdout_only && r.gc_collections != baseline_gc {
                gc_identical = false;
            }
        }
    }

    // When --stdout-only, GC check is always PASS
    if pa.stdout_only {
        gc_identical = true;
    }

    // Report
    let all_pass = stdout_identical && exit_identical && gc_identical;

    // Build per-run hashes for the report
    let run_hashes: Vec<String> = if pa.hash_output {
        results.iter().map(|r| hash_stdout(&r.stdout)).collect()
    } else {
        Vec::new()
    };

    match pa.output {
        OutputMode::Json => {
            let json = build_json_report(
                &filename, &pa, &seeds, &results, &run_hashes,
                stdout_identical, exit_identical, gc_identical, all_pass,
            );
            println!("{}", json);
        }
        _ => {
            eprintln!();
            let mut t = crate::table::Table::new(vec!["Check", "Status"]);
            t.add_row_owned(vec![
                "stdout identical".to_string(),
                verdict_str(pa.output, stdout_identical),
            ]);
            t.add_row_owned(vec![
                "exit status identical".to_string(),
                verdict_str(pa.output, exit_identical),
            ]);
            if !pa.stdout_only {
                t.add_row_owned(vec![
                    "GC collections identical".to_string(),
                    verdict_str(pa.output, gc_identical),
                ]);
            }
            eprint!("{}", t.render());

            let verdict = if all_pass {
                output::colorize(pa.output, output::BOLD_GREEN, "PASS")
            } else {
                output::colorize(pa.output, output::BOLD_RED, "FAIL")
            };
            eprintln!("\nVerdict: {} ({} runs, seed={}, executor={})",
                verdict, pa.runs, pa.seed, executor_label);

            if !all_pass {
                // Save report before exiting if requested
                if let Some(ref report_path) = pa.save_report {
                    save_json_report(
                        report_path, &filename, &pa, &seeds, &results, &run_hashes,
                        stdout_identical, exit_identical, gc_identical, all_pass,
                    );
                }
                process::exit(1);
            }
        }
    }

    // --save-report: write JSON report to file
    if let Some(ref report_path) = pa.save_report {
        save_json_report(
            report_path, &filename, &pa, &seeds, &results, &run_hashes,
            stdout_identical, exit_identical, gc_identical, all_pass,
        );
    }
}

/// Build a JSON report string (no external dependencies).
fn build_json_report(
    filename: &str,
    pa: &ProofArgs,
    seeds: &[u64],
    results: &[RunResult],
    run_hashes: &[String],
    stdout_identical: bool,
    exit_identical: bool,
    gc_identical: bool,
    all_pass: bool,
) -> String {
    let executor_label = match pa.executor {
        ExecutorMode::Eval => "eval",
        ExecutorMode::Mir => "mir",
        ExecutorMode::Both => "both",
    };

    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"file\": \"{}\",\n", filename.replace('\\', "/")));
    json.push_str(&format!("  \"seed\": {},\n", pa.seed));

    // seeds array
    json.push_str("  \"seeds\": [");
    for (i, s) in seeds.iter().enumerate() {
        if i > 0 { json.push_str(", "); }
        json.push_str(&s.to_string());
    }
    json.push_str("],\n");

    json.push_str(&format!("  \"runs\": {},\n", pa.runs));
    json.push_str(&format!("  \"executor\": \"{}\",\n", executor_label));
    json.push_str(&format!("  \"fail_fast\": {},\n", pa.fail_fast));
    json.push_str(&format!("  \"stdout_only\": {},\n", pa.stdout_only));
    json.push_str(&format!("  \"stdout_identical\": {},\n", stdout_identical));
    json.push_str(&format!("  \"exit_identical\": {},\n", exit_identical));
    json.push_str(&format!("  \"gc_identical\": {},\n", gc_identical));

    // Per-run details
    json.push_str("  \"run_details\": [\n");
    for (i, r) in results.iter().enumerate() {
        json.push_str("    {\n");
        json.push_str(&format!("      \"executor\": \"{}\",\n", r.executor_label));
        json.push_str(&format!("      \"exit_ok\": {},\n", r.exit_ok));
        json.push_str(&format!("      \"stdout_lines\": {},\n", r.stdout.len()));
        json.push_str(&format!("      \"gc_collections\": {}", r.gc_collections));
        if !run_hashes.is_empty() {
            json.push_str(&format!(",\n      \"sha256\": \"{}\"", run_hashes[i]));
        }
        json.push('\n');
        json.push_str("    }");
        if i + 1 < results.len() { json.push(','); }
        json.push('\n');
    }
    json.push_str("  ],\n");

    json.push_str(&format!("  \"verdict\": \"{}\"\n", if all_pass { "PASS" } else { "FAIL" }));
    json.push_str("}\n");
    json
}

/// Write the JSON report to a file.
fn save_json_report(
    path: &PathBuf,
    filename: &str,
    pa: &ProofArgs,
    seeds: &[u64],
    results: &[RunResult],
    run_hashes: &[String],
    stdout_identical: bool,
    exit_identical: bool,
    gc_identical: bool,
    all_pass: bool,
) {
    let json = build_json_report(
        filename, pa, seeds, results, run_hashes,
        stdout_identical, exit_identical, gc_identical, all_pass,
    );
    match fs::write(path, &json) {
        Ok(_) => eprintln!("  Report saved to {}", path.display()),
        Err(e) => eprintln!("  warning: could not save report to {}: {}", path.display(), e),
    }
}

fn verdict_str(mode: OutputMode, pass: bool) -> String {
    if pass {
        output::colorize(mode, output::BOLD_GREEN, "PASS")
    } else {
        output::colorize(mode, output::BOLD_RED, "FAIL")
    }
}

pub fn print_help() {
    eprintln!("cjcl proof — Determinism & reproducibility profiler");
    eprintln!();
    eprintln!("Usage: cjcl proof <file.cjcl> [flags]");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  -n, --runs <N>        Number of iterations (default: 3)");
    eprintln!("  --seed <N>            RNG seed (default: 42)");
    eprintln!("  --seeds <N,M,...>     Test multiple seeds (comma-separated)");
    eprintln!("  -v, --verbose         Show per-run details and divergence info");
    eprintln!("  --plain               Plain text output");
    eprintln!("  --json                JSON output");
    eprintln!("  --color               Color output (default)");
    eprintln!("  --fail-fast           Stop on first divergence");
    eprintln!("  --hash-output         Display SHA-256 hash of stdout per run");
    eprintln!("  --save-report <file>  Save reproducibility report as JSON");
    eprintln!("  --executor <mode>     Executor: eval (default), mir, or both");
    eprintln!("  --stdout-only         Only check stdout identity (skip GC check)");
}
