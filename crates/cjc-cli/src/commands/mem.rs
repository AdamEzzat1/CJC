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
    pub peak_only: bool,
    pub timeline: bool,
    pub save_report: Option<String>,
    pub fail_on_gc: bool,
    pub compare: bool,
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
            peak_only: false,
            timeline: false,
            save_report: None,
            fail_on_gc: false,
            compare: false,
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
            "--peak-only" => ma.peak_only = true,
            "--timeline" => ma.timeline = true,
            "--save-report" => {
                i += 1;
                if i < args.len() {
                    ma.save_report = Some(args[i].clone());
                } else {
                    eprintln!("error: --save-report requires a file argument");
                    process::exit(1);
                }
            }
            "--fail-on-gc" => ma.fail_on_gc = true,
            "--compare" => {
                i += 1;
                if i < args.len() {
                    match args[i].as_str() {
                        "eval" | "mir" => ma.compare = true,
                        other => {
                            eprintln!("error: --compare expects `eval` or `mir`, got `{}`", other);
                            process::exit(1);
                        }
                    }
                } else {
                    // No argument means compare both
                    ma.compare = true;
                }
            }
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

fn run_eval(program: &cjc_ast::Program, seed: u64) -> MemRun {
    let start = Instant::now();
    let mut interpreter = cjc_eval::Interpreter::new(seed);
    let _result = interpreter.exec(program);
    let elapsed = start.elapsed();

    let output_bytes: usize = interpreter.output.iter().map(|s| s.len()).sum();
    let gc_heap_objects = interpreter.gc_heap.live_count();

    MemRun {
        gc_collections: interpreter.gc_collections,
        output_lines: interpreter.output.len(),
        output_bytes,
        exec_time_us: elapsed.as_micros(),
        gc_heap_objects,
    }
}

fn run_mir(program: &cjc_ast::Program, seed: u64) -> MemRun {
    let start = Instant::now();
    let result = cjc_mir_exec::run_program_with_executor(program, seed);
    let elapsed = start.elapsed();

    match result {
        Ok((_val, exec)) => {
            let output_bytes: usize = exec.output.iter().map(|s| s.len()).sum();
            MemRun {
                gc_collections: exec.gc_collections,
                output_lines: exec.output.len(),
                output_bytes,
                exec_time_us: elapsed.as_micros(),
                gc_heap_objects: exec.gc_heap.live_count(),
            }
        }
        Err(_) => MemRun {
            gc_collections: 0,
            output_lines: 0,
            output_bytes: 0,
            exec_time_us: elapsed.as_micros(),
            gc_heap_objects: 0,
        },
    }
}

fn write_json_report(path: &str, filename: &str, ma: &MemArgs, runs: &[MemRun]) {
    let total_gc: u64 = runs.iter().map(|r| r.gc_collections).sum();
    let avg_gc = total_gc as f64 / runs.len() as f64;
    let max_gc = runs.iter().map(|r| r.gc_collections).max().unwrap_or(0);
    let min_gc = runs.iter().map(|r| r.gc_collections).min().unwrap_or(0);
    let gc_stable = runs.iter().all(|r| r.gc_collections == runs[0].gc_collections);
    let max_heap = runs.iter().map(|r| r.gc_heap_objects).max().unwrap_or(0);
    let avg_time_us = runs.iter().map(|r| r.exec_time_us).sum::<u128>() as f64 / runs.len() as f64;
    let output_lines = runs[0].output_lines;
    let output_bytes = runs[0].output_bytes;

    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"file\": \"{}\",\n", filename));
    json.push_str(&format!("  \"seed\": {},\n", ma.seed));
    json.push_str(&format!("  \"runs\": {},\n", runs.len()));
    json.push_str(&format!("  \"gc_collections_avg\": {},\n", output::format_f64(avg_gc, 2)));
    json.push_str(&format!("  \"gc_collections_min\": {},\n", min_gc));
    json.push_str(&format!("  \"gc_collections_max\": {},\n", max_gc));
    json.push_str(&format!("  \"gc_stable\": {},\n", gc_stable));
    json.push_str(&format!("  \"gc_heap_objects_max\": {},\n", max_heap));
    json.push_str(&format!("  \"output_lines\": {},\n", output_lines));
    json.push_str(&format!("  \"output_bytes\": {},\n", output_bytes));
    json.push_str(&format!("  \"avg_exec_time_us\": {},\n", avg_time_us as u64));
    json.push_str("  \"per_run\": [\n");
    for (i, r) in runs.iter().enumerate() {
        json.push_str(&format!(
            "    {{\"gc_collections\": {}, \"gc_heap_objects\": {}, \"exec_time_us\": {}}}",
            r.gc_collections, r.gc_heap_objects, r.exec_time_us));
        if i + 1 < runs.len() { json.push(','); }
        json.push('\n');
    }
    json.push_str("  ]\n");
    json.push_str("}\n");

    if let Err(e) = fs::write(path, &json) {
        eprintln!("error: could not write report to `{}`: {}", path, e);
        process::exit(1);
    }
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

    // --compare mode: run both executors and compare
    if ma.compare {
        run_compare(&ma, &program, &filename);
        return;
    }

    let mut runs: Vec<MemRun> = Vec::new();

    for run_idx in 0..ma.runs {
        let mem_run = match ma.executor {
            Executor::Eval => run_eval(&program, ma.seed),
            Executor::Mir => run_mir(&program, ma.seed),
        };

        if ma.verbose && mem_run.gc_collections == 0 && mem_run.output_lines == 0 {
            eprintln!("  run {}: possibly errored (no output)", run_idx + 1);
        }

        runs.push(mem_run);
    }

    // Save report if requested
    if let Some(ref report_path) = ma.save_report {
        write_json_report(report_path, &filename, &ma, &runs);
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
            if !ma.peak_only {
                t.add_row_owned(vec!["Output lines".into(), format!("{}", output_lines)]);
                t.add_row_owned(vec!["Output bytes".into(), format!("{}", output_bytes)]);
            }
            t.add_row_owned(vec!["Avg exec time".into(), format!("{:.1} us", avg_time_us)]);
            eprint!("{}", t.render());

            // --timeline: show GC collections per run
            if ma.timeline && runs.len() > 1 {
                eprintln!("\nGC timeline:");
                let mut tl = crate::table::Table::new(vec!["Run", "GC collections", "Heap objects"]);
                for (i, r) in runs.iter().enumerate() {
                    tl.add_row_owned(vec![
                        format!("{}", i + 1),
                        format!("{}", r.gc_collections),
                        format!("{}", r.gc_heap_objects),
                    ]);
                }
                eprint!("{}", tl.render());
            }

            if ma.verbose && runs.len() > 1 && !ma.peak_only {
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

    // --fail-on-gc: exit 1 if any GC collections occurred
    if ma.fail_on_gc && total_gc > 0 {
        eprintln!("error: --fail-on-gc: {} GC collection(s) detected", total_gc);
        process::exit(1);
    }
}

fn run_compare(ma: &MemArgs, program: &cjc_ast::Program, filename: &str) {
    let eval_run = run_eval(program, ma.seed);
    let mir_run = run_mir(program, ma.seed);

    match ma.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file\": \"{}\",", filename);
            println!("  \"seed\": {},", ma.seed);
            println!("  \"eval\": {{\"gc_collections\": {}, \"gc_heap_objects\": {}, \"exec_time_us\": {}}},",
                eval_run.gc_collections, eval_run.gc_heap_objects, eval_run.exec_time_us);
            println!("  \"mir\": {{\"gc_collections\": {}, \"gc_heap_objects\": {}, \"exec_time_us\": {}}}",
                mir_run.gc_collections, mir_run.gc_heap_objects, mir_run.exec_time_us);
            println!("}}");
        }
        _ => {
            eprintln!("{} GC comparison for `{}` (seed={})",
                output::colorize(ma.output, output::BOLD_CYAN, "[mem]"),
                filename, ma.seed);
            eprintln!();

            let mut t = crate::table::Table::new(vec!["Metric", "eval", "mir-exec"]);
            t.add_row_owned(vec![
                "GC collections".into(),
                format!("{}", eval_run.gc_collections),
                format!("{}", mir_run.gc_collections),
            ]);
            t.add_row_owned(vec![
                "GC heap objects".into(),
                format!("{}", eval_run.gc_heap_objects),
                format!("{}", mir_run.gc_heap_objects),
            ]);
            t.add_row_owned(vec![
                "Output lines".into(),
                format!("{}", eval_run.output_lines),
                format!("{}", mir_run.output_lines),
            ]);
            t.add_row_owned(vec![
                "Exec time (us)".into(),
                format!("{}", eval_run.exec_time_us),
                format!("{}", mir_run.exec_time_us),
            ]);
            eprint!("{}", t.render());
        }
    }

    // --fail-on-gc applies in compare mode too
    if ma.fail_on_gc && (eval_run.gc_collections > 0 || mir_run.gc_collections > 0) {
        let total = eval_run.gc_collections + mir_run.gc_collections;
        eprintln!("error: --fail-on-gc: {} GC collection(s) detected", total);
        process::exit(1);
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
    eprintln!("  --seed <N>          RNG seed (default: 42)");
    eprintln!("  -n, --runs <N>      Number of runs (default: 1)");
    eprintln!("  -v, --verbose       Show per-run details");
    eprintln!("  --nogc              Run NoGC verification before profiling");
    eprintln!("  --mir               Use MIR executor");
    eprintln!("  --eval              Use eval executor (default)");
    eprintln!("  --peak-only         Only report peak memory/GC metrics, skip per-run details");
    eprintln!("  --timeline          Show GC collections per run in a timeline format");
    eprintln!("  --save-report <f>   Save memory report as JSON");
    eprintln!("  --fail-on-gc        Exit code 1 if any GC collections occurred");
    eprintln!("  --compare eval|mir  Compare GC behavior between eval and MIR executors");
    eprintln!("  --plain             Plain text output");
    eprintln!("  --json              JSON output");
    eprintln!("  --color             Color output (default)");
    eprintln!();
    eprintln!("Limitations:");
    eprintln!("  CJC currently tracks GC collections and heap objects.");
    eprintln!("  Per-allocation tracking and peak RSS are not yet instrumented.");
}
