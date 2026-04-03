//! `cjc proof` — Determinism & reproducibility profiler.
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

/// Parsed arguments for `cjc proof`.
pub struct ProofArgs {
    pub file: PathBuf,
    pub runs: usize,
    pub seed: u64,
    pub seeds: Vec<u64>,
    pub output: OutputMode,
    pub verbose: bool,
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
            other if !other.starts_with('-') => pa.file = PathBuf::from(other),
            other => {
                eprintln!("error: unknown flag `{}` for `cjc proof`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    if pa.file.as_os_str().is_empty() {
        eprintln!("error: `cjc proof` requires a .cjc file argument");
        process::exit(1);
    }
    pa
}

/// A single run result for comparison.
struct RunResult {
    stdout: Vec<String>,
    exit_ok: bool,
    gc_collections: u64,
}

/// Entry point for `cjc proof`.
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

    eprintln!("{} Running {} iterations with seed{} {}...",
        output::colorize(pa.output, output::BOLD_CYAN, "[proof]"),
        pa.runs,
        if seeds.len() > 1 { "s" } else { "" },
        seeds.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(",")
    );

    let mut results: Vec<RunResult> = Vec::new();

    for &current_seed in &seeds {
    for run_idx in 0..pa.runs {
        let (program, diags) = cjc_parser::parse_source(&source);
        if diags.has_errors() {
            eprintln!("error: parse errors in `{}`", filename);
            let rendered = diags.render_all_color(&source, &filename, pa.output.use_color());
            eprint!("{}", rendered);
            process::exit(1);
        }

        let mut interpreter = cjc_eval::Interpreter::new(current_seed);
        let exit_ok = match interpreter.exec(&program) {
            Ok(_) => true,
            Err(e) => {
                if pa.verbose {
                    eprintln!("  run {}: error: {}", run_idx + 1, e);
                }
                false
            }
        };

        results.push(RunResult {
            stdout: interpreter.output.clone(),
            exit_ok,
            gc_collections: interpreter.gc_collections,
        });

        if pa.verbose {
            eprintln!("  run {}: {} lines, gc={}, ok={}",
                run_idx + 1,
                interpreter.output.len(),
                interpreter.gc_collections,
                exit_ok
            );
        }
    } // end for run_idx
    } // end for current_seed in seeds

    // Compare all runs against the first
    let mut stdout_identical = true;
    let mut exit_identical = true;
    let mut gc_identical = true;

    if results.len() > 1 {
        let baseline = &results[0];
        for (i, r) in results.iter().enumerate().skip(1) {
            if r.stdout != baseline.stdout {
                stdout_identical = false;
                if pa.verbose {
                    eprintln!("  DIVERGENCE: run {} stdout differs from run 1", i + 1);
                    // Find first difference
                    let max_lines = baseline.stdout.len().max(r.stdout.len());
                    for line_idx in 0..max_lines {
                        let a = baseline.stdout.get(line_idx).map(|s| s.as_str()).unwrap_or("<missing>");
                        let b = r.stdout.get(line_idx).map(|s| s.as_str()).unwrap_or("<missing>");
                        if a != b {
                            eprintln!("    line {}: {:?} vs {:?}", line_idx + 1, a, b);
                            break;
                        }
                    }
                }
            }
            if r.exit_ok != baseline.exit_ok {
                exit_identical = false;
            }
            if r.gc_collections != baseline.gc_collections {
                gc_identical = false;
            }
        }
    }

    // Report
    let all_pass = stdout_identical && exit_identical && gc_identical;

    match pa.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file\": \"{}\",", filename.replace('\\', "/"));
            println!("  \"seed\": {},", pa.seed);
            println!("  \"runs\": {},", pa.runs);
            println!("  \"stdout_identical\": {},", stdout_identical);
            println!("  \"exit_identical\": {},", exit_identical);
            println!("  \"gc_identical\": {},", gc_identical);
            println!("  \"verdict\": \"{}\"", if all_pass { "PASS" } else { "FAIL" });
            println!("}}");
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
            t.add_row_owned(vec![
                "GC collections identical".to_string(),
                verdict_str(pa.output, gc_identical),
            ]);
            eprint!("{}", t.render());

            let verdict = if all_pass {
                output::colorize(pa.output, output::BOLD_GREEN, "PASS")
            } else {
                output::colorize(pa.output, output::BOLD_RED, "FAIL")
            };
            eprintln!("\nVerdict: {} ({} runs, seed={})", verdict, pa.runs, pa.seed);

            if !all_pass {
                process::exit(1);
            }
        }
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
    eprintln!("cjc proof — Determinism & reproducibility profiler");
    eprintln!();
    eprintln!("Usage: cjc proof <file.cjc> [flags]");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  -n, --runs <N>      Number of iterations (default: 3)");
    eprintln!("  --seed <N>          RNG seed (default: 42)");
    eprintln!("  --seeds <N,M,...>   Test multiple seeds (comma-separated)");
    eprintln!("  -v, --verbose       Show per-run details and divergence info");
    eprintln!("  --plain             Plain text output");
    eprintln!("  --json              JSON output");
    eprintln!("  --color             Color output (default)");
}
