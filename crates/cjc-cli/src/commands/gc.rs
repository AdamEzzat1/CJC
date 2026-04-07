//! `cjcl gc` — Dedicated GC analysis.
//!
//! Executes a CJC script multiple times and reports granular GC behavior:
//! allocation timeline, peak heap objects, GC pause count, COW copies,
//! and GC stability across runs. More detailed than `cjcl mem`.

use std::fs;
use std::path::Path;
use std::process;
use std::time::Instant;
use crate::output::{self, OutputMode};

pub struct GcArgs {
    pub file: String,
    pub seed: u64,
    pub runs: usize,
    pub output: OutputMode,
    pub verbose: bool,
}

impl Default for GcArgs {
    fn default() -> Self {
        Self {
            file: String::new(),
            seed: 42,
            runs: 3,
            output: OutputMode::Color,
            verbose: false,
        }
    }
}

pub fn parse_args(args: &[String]) -> GcArgs {
    let mut ga = GcArgs::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                if i < args.len() {
                    ga.seed = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("error: --seed requires a numeric argument");
                        process::exit(1);
                    });
                }
            }
            "--runs" | "-n" => {
                i += 1;
                if i < args.len() {
                    ga.runs = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("error: --runs requires a numeric argument");
                        process::exit(1);
                    });
                }
            }
            "-v" | "--verbose" => ga.verbose = true,
            "--plain" => ga.output = OutputMode::Plain,
            "--json" => ga.output = OutputMode::Json,
            "--color" => ga.output = OutputMode::Color,
            other if !other.starts_with('-') => ga.file = other.to_string(),
            other => {
                eprintln!("error: unknown flag `{}` for `cjcl gc`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    if ga.file.is_empty() {
        eprintln!("error: `cjcl gc` requires a .cjcl file argument");
        process::exit(1);
    }
    ga
}

/// Data captured from a single execution run.
struct GcRun {
    gc_collections: u64,
    gc_heap_objects: usize,
    output_lines: usize,
    output_bytes: usize,
    exec_time_us: u128,
}

pub fn run(args: &[String]) {
    let ga = parse_args(args);
    let path = Path::new(&ga.file);

    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read `{}`: {}", ga.file, e);
            process::exit(1);
        }
    };

    let filename = ga.file.replace('\\', "/");

    let (program, diags) = cjc_parser::parse_source(&source);
    if diags.has_errors() {
        let rendered = diags.render_all_color(&source, &filename, ga.output.use_color());
        eprint!("{}", rendered);
        process::exit(1);
    }

    let mut runs: Vec<GcRun> = Vec::new();

    if ga.output != OutputMode::Json {
        eprintln!("{} GC analysis for `{}` ({} run{}, seed={})",
            output::colorize(ga.output, output::BOLD_CYAN, "[gc]"),
            filename, ga.runs, if ga.runs > 1 { "s" } else { "" }, ga.seed);
    }

    for run_idx in 0..ga.runs {
        let start = Instant::now();
        let mut interpreter = cjc_eval::Interpreter::new(ga.seed);
        let result = interpreter.exec(&program);
        let elapsed = start.elapsed();

        if let Err(e) = &result {
            if ga.verbose {
                eprintln!("  run {}: error: {}", run_idx + 1, e);
            }
        }

        let output_bytes: usize = interpreter.output.iter().map(|s| s.len()).sum();
        let gc_heap_objects = interpreter.gc_heap.live_count();

        runs.push(GcRun {
            gc_collections: interpreter.gc_collections,
            gc_heap_objects,
            output_lines: interpreter.output.len(),
            output_bytes,
            exec_time_us: elapsed.as_micros(),
        });
    }

    // Aggregate statistics
    let total_gc: u64 = runs.iter().map(|r| r.gc_collections).sum();
    let avg_gc = total_gc as f64 / runs.len() as f64;
    let max_gc = runs.iter().map(|r| r.gc_collections).max().unwrap_or(0);
    let min_gc = runs.iter().map(|r| r.gc_collections).min().unwrap_or(0);
    let gc_stable = runs.iter().all(|r| r.gc_collections == runs[0].gc_collections);

    let peak_heap = runs.iter().map(|r| r.gc_heap_objects).max().unwrap_or(0);
    let avg_heap = runs.iter().map(|r| r.gc_heap_objects).sum::<usize>() as f64 / runs.len() as f64;

    let avg_time_us = runs.iter().map(|r| r.exec_time_us).sum::<u128>() as f64 / runs.len() as f64;

    // COW copies placeholder (not yet tracked in runtime)
    let cow_copies: u64 = 0;

    match ga.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file\": \"{}\",", filename);
            println!("  \"seed\": {},", ga.seed);
            println!("  \"runs\": {},", runs.len());
            println!("  \"gc_collections_avg\": {},", output::format_f64(avg_gc, 2));
            println!("  \"gc_collections_min\": {},", min_gc);
            println!("  \"gc_collections_max\": {},", max_gc);
            println!("  \"gc_pause_count\": {},", total_gc);
            println!("  \"gc_stable\": {},", gc_stable);
            println!("  \"peak_heap_objects\": {},", peak_heap);
            println!("  \"avg_heap_objects\": {},", output::format_f64(avg_heap, 2));
            println!("  \"cow_copies\": {},", cow_copies);
            println!("  \"avg_exec_time_us\": {},", avg_time_us as u64);
            println!("  \"timeline\": [");
            for (i, r) in runs.iter().enumerate() {
                let comma = if i + 1 < runs.len() { "," } else { "" };
                println!("    {{\"run\": {}, \"gc_collections\": {}, \"heap_objects\": {}, \"output_lines\": {}, \"output_bytes\": {}, \"exec_time_us\": {}}}{}",
                    i + 1, r.gc_collections, r.gc_heap_objects, r.output_lines, r.output_bytes, r.exec_time_us, comma);
            }
            println!("  ]");
            println!("}}");
        }
        _ => {
            eprintln!();

            // Summary table
            let mut t = crate::table::Table::new(vec!["Metric", "Value"]);
            t.add_row_owned(vec!["GC pause count (total)".into(), format!("{}", total_gc)]);
            t.add_row_owned(vec!["GC collections (avg)".into(), output::format_f64(avg_gc, 2)]);
            if runs.len() > 1 {
                t.add_row_owned(vec!["GC collections (min)".into(), format!("{}", min_gc)]);
                t.add_row_owned(vec!["GC collections (max)".into(), format!("{}", max_gc)]);
            }
            t.add_row_owned(vec!["GC stable across runs".into(),
                if gc_stable {
                    output::colorize(ga.output, output::BOLD_GREEN, "yes")
                } else {
                    output::colorize(ga.output, output::BOLD_RED, "no")
                }]);
            t.add_row_owned(vec!["Peak heap objects".into(), format!("{}", peak_heap)]);
            t.add_row_owned(vec!["Avg heap objects".into(), output::format_f64(avg_heap, 1)]);
            t.add_row_owned(vec!["COW copies".into(), format!("{} (not yet tracked)", cow_copies)]);
            t.add_row_owned(vec!["Avg exec time".into(), format!("{:.1} us", avg_time_us)]);
            eprint!("{}", t.render());

            // Allocation timeline
            if runs.len() > 1 || ga.verbose {
                eprintln!("\nAllocation timeline:");
                let mut tt = crate::table::Table::new(vec!["Run", "GC", "Heap Objs", "Output Lines", "Output Bytes", "Time (us)"]);
                for (i, r) in runs.iter().enumerate() {
                    tt.add_row_owned(vec![
                        format!("{}", i + 1),
                        format!("{}", r.gc_collections),
                        format!("{}", r.gc_heap_objects),
                        format!("{}", r.output_lines),
                        format!("{}", r.output_bytes),
                        format!("{}", r.exec_time_us),
                    ]);
                }
                eprint!("{}", tt.render());
            }

            // GC stability verdict
            eprintln!();
            if gc_stable {
                eprintln!("{} GC behavior is deterministic across {} runs.",
                    output::colorize(ga.output, output::BOLD_GREEN, "STABLE"),
                    runs.len());
            } else {
                eprintln!("{} GC behavior varies across runs.",
                    output::colorize(ga.output, output::BOLD_YELLOW, "UNSTABLE"),
                    );
            }

            eprintln!();
            eprintln!("{}", output::colorize(ga.output, output::DIM,
                "Note: COW copy tracking is not yet instrumented. \
                 GC pause count equals total GC collections across all runs."));
        }
    }
}

pub fn print_help() {
    eprintln!("cjcl gc — Dedicated GC analysis");
    eprintln!();
    eprintln!("Usage: cjcl gc <file.cjcl> [flags]");
    eprintln!();
    eprintln!("Reports:");
    eprintln!("  - GC pause count (total collections)");
    eprintln!("  - Peak and average heap object counts");
    eprintln!("  - COW buffer copies (placeholder — not yet tracked)");
    eprintln!("  - Allocation timeline (per-run checkpoint data)");
    eprintln!("  - GC stability across runs");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  --seed <N>      RNG seed (default: 42)");
    eprintln!("  -n, --runs <N>  Number of runs (default: 3)");
    eprintln!("  -v, --verbose   Show timeline even for single runs");
    eprintln!("  --plain         Plain text output");
    eprintln!("  --json          JSON output");
    eprintln!("  --color         Color output (default)");
}
