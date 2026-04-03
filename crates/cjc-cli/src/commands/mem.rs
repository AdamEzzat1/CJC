//! `cjc mem` — Runtime memory visibility.
//!
//! Executes a CJC script and reports memory behavior: GC collections,
//! output size, allocation patterns. Reports are deterministic.
//!
//! Instrumentation limitations: CJC's current runtime tracks GC collection
//! counts but does not expose per-allocation tracking. Peak memory and
//! object counts are estimated from interpreter state.

use std::fs;
use std::path::Path;
use std::process;
use std::time::Instant;
use crate::output::{self, OutputMode};

pub struct MemArgs {
    pub file: String,
    pub seed: u64,
    pub runs: usize,
    pub output: OutputMode,
    pub verbose: bool,
    pub nogc_check: bool,
    pub executor: Executor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Executor {
    Eval,
    Mir,
}

impl Default for MemArgs {
    fn default() -> Self {
        Self {
            file: String::new(),
            seed: 42,
            runs: 1,
            output: OutputMode::Color,
            verbose: false,
            nogc_check: false,
            executor: Executor::Eval,
        }
    }
}

pub fn parse_args(args: &[String]) -> MemArgs {
    let mut ma = MemArgs::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                if i < args.len() { ma.seed = args[i].parse().unwrap_or(42); }
            }
            "--runs" | "-n" => {
                i += 1;
                if i < args.len() { ma.runs = args[i].parse().unwrap_or(1); }
            }
            "--nogc" => ma.nogc_check = true,
            "--mir" => ma.executor = Executor::Mir,
            "--eval" => ma.executor = Executor::Eval,
            "-v" | "--verbose" => ma.verbose = true,
            "--plain" => ma.output = OutputMode::Plain,
            "--json" => ma.output = OutputMode::Json,
            "--color" => ma.output = OutputMode::Color,
            other if !other.starts_with('-') => ma.file = other.to_string(),
            other => {
                eprintln!("error: unknown flag `{}` for `cjc mem`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    if ma.file.is_empty() {
        eprintln!("error: `cjc mem` requires a .cjc file argument");
        process::exit(1);
    }
    ma
}

struct MemRun {
    gc_collections: u64,
    output_lines: usize,
    output_bytes: usize,
    exec_time_us: u128,
    gc_heap_objects: usize,
}

pub fn run(args: &[String]) {
    let ma = parse_args(args);
    let path = Path::new(&ma.file);

    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => { eprintln!("error: could not read `{}`: {}", ma.file, e); process::exit(1); }
    };

    let filename = ma.file.replace('\\', "/");

    let (program, diags) = cjc_parser::parse_source(&source);
    if diags.has_errors() {
        let rendered = diags.render_all_color(&source, &filename, ma.output.use_color());
        eprint!("{}", rendered);
        process::exit(1);
    }

    // NoGC pre-check
    if ma.nogc_check {
        if let Err(e) = cjc_mir_exec::verify_nogc(&program) {
            eprintln!("error: NoGC verification failed — refusing to profile");
            eprintln!("{}", e);
            process::exit(1);
        }
    }

    let mut runs: Vec<MemRun> = Vec::new();

    for run_idx in 0..ma.runs {
        let start = Instant::now();
        let mut interpreter = cjc_eval::Interpreter::new(ma.seed);
        let result = interpreter.exec(&program);
        let elapsed = start.elapsed();

        if let Err(e) = &result {
            if ma.verbose {
                eprintln!("  run {}: error: {}", run_idx + 1, e);
            }
        }

        let output_bytes: usize = interpreter.output.iter().map(|s| s.len()).sum();
        let gc_heap_objects = interpreter.gc_heap.live_count();

        runs.push(MemRun {
            gc_collections: interpreter.gc_collections,
            output_lines: interpreter.output.len(),
            output_bytes,
            exec_time_us: elapsed.as_micros(),
            gc_heap_objects,
        });
    }

    // Aggregate
    let total_gc: u64 = runs.iter().map(|r| r.gc_collections).sum();
    let avg_gc = total_gc as f64 / runs.len() as f64;
    let max_gc = runs.iter().map(|r| r.gc_collections).max().unwrap_or(0);
    let min_gc = runs.iter().map(|r| r.gc_collections).min().unwrap_or(0);
    let gc_stable = runs.iter().all(|r| r.gc_collections == runs[0].gc_collections);

    let avg_time_us = runs.iter().map(|r| r.exec_time_us).sum::<u128>() as f64 / runs.len() as f64;
    let max_heap = runs.iter().map(|r| r.gc_heap_objects).max().unwrap_or(0);

    let output_lines = runs[0].output_lines;
    let output_bytes = runs[0].output_bytes;

    match ma.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file\": \"{}\",", filename);
            println!("  \"seed\": {},", ma.seed);
            println!("  \"runs\": {},", runs.len());
            println!("  \"gc_collections_avg\": {},", output::format_f64(avg_gc, 2));
            println!("  \"gc_collections_min\": {},", min_gc);
            println!("  \"gc_collections_max\": {},", max_gc);
            println!("  \"gc_stable\": {},", gc_stable);
            println!("  \"gc_heap_objects_max\": {},", max_heap);
            println!("  \"output_lines\": {},", output_lines);
            println!("  \"output_bytes\": {},", output_bytes);
            println!("  \"avg_exec_time_us\": {}", avg_time_us as u64);
            println!("}}");
        }
        _ => {
            eprintln!("{} Memory profile for `{}` ({} run{}, seed={})",
                output::colorize(ma.output, output::BOLD_CYAN, "[mem]"),
                filename, runs.len(), if runs.len() > 1 { "s" } else { "" }, ma.seed);
            eprintln!();

            let mut t = crate::table::Table::new(vec!["Metric", "Value"]);
            t.add_row_owned(vec!["GC collections (avg)".into(), output::format_f64(avg_gc, 2)]);
            if runs.len() > 1 {
                t.add_row_owned(vec!["GC collections (min)".into(), format!("{}", min_gc)]);
                t.add_row_owned(vec!["GC collections (max)".into(), format!("{}", max_gc)]);
            }
            t.add_row_owned(vec!["GC stable across runs".into(), format!("{}", gc_stable)]);
            t.add_row_owned(vec!["GC heap objects (max)".into(), format!("{}", max_heap)]);
            t.add_row_owned(vec!["Output lines".into(), format!("{}", output_lines)]);
            t.add_row_owned(vec!["Output bytes".into(), format!("{}", output_bytes)]);
            t.add_row_owned(vec!["Avg exec time".into(), format!("{:.1} us", avg_time_us)]);
            eprint!("{}", t.render());

            if ma.verbose && runs.len() > 1 {
                eprintln!("\nPer-run details:");
                let mut rt = crate::table::Table::new(vec!["Run", "GC", "Heap Objs", "Time (us)"]);
                for (i, r) in runs.iter().enumerate() {
                    rt.add_row_owned(vec![
                        format!("{}", i + 1),
                        format!("{}", r.gc_collections),
                        format!("{}", r.gc_heap_objects),
                        format!("{}", r.exec_time_us),
                    ]);
                }
                eprint!("{}", rt.render());
            }

            // Instrumentation note
            eprintln!();
            eprintln!("{}", output::colorize(ma.output, output::DIM,
                "Note: CJC tracks GC collections and heap object counts. Per-allocation \
                 tracking and peak RSS are not yet instrumented."));
        }
    }
}

pub fn print_help() {
    eprintln!("cjc mem — Runtime memory visibility");
    eprintln!();
    eprintln!("Usage: cjc mem <file.cjc> [flags]");
    eprintln!();
    eprintln!("Reports:");
    eprintln!("  - GC collection counts");
    eprintln!("  - GC heap object counts");
    eprintln!("  - Output size (lines, bytes)");
    eprintln!("  - Execution time");
    eprintln!("  - GC stability across runs");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  --seed <N>      RNG seed (default: 42)");
    eprintln!("  -n, --runs <N>  Number of runs (default: 1)");
    eprintln!("  -v, --verbose   Show per-run details");
    eprintln!("  --plain         Plain text output");
    eprintln!("  --json          JSON output");
    eprintln!("  --color         Color output (default)");
    eprintln!();
    eprintln!("Limitations:");
    eprintln!("  CJC currently tracks GC collections and heap objects.");
    eprintln!("  Per-allocation tracking and peak RSS are not yet instrumented.");
}
