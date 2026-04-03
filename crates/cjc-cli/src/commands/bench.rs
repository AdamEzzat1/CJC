//! `cjc bench` — Performance and stability benchmarking.
//!
//! Complements `cjc proof` (which verifies determinism) by focusing on
//! performance: runtime timing, throughput, and run-to-run stability.
//!
//! Distinction from `proof`:
//!   - `proof` asks: "Does the output stay the same across runs?"
//!   - `bench` asks: "How fast is it, and how stable is the timing?"

use std::fs;
use std::path::Path;
use std::process;
use std::time::Instant;
use crate::output::{self, OutputMode};

pub struct BenchArgs {
    pub file: String,
    pub seed: u64,
    pub runs: usize,
    pub warmup: usize,
    pub output: OutputMode,
    pub verbose: bool,
    pub nogc_check: bool,
    pub use_mir: bool,
}

impl Default for BenchArgs {
    fn default() -> Self {
        Self {
            file: String::new(),
            seed: 42,
            runs: 10,
            warmup: 1,
            output: OutputMode::Color,
            verbose: false,
            nogc_check: false,
            use_mir: false,
        }
    }
}

pub fn parse_args(args: &[String]) -> BenchArgs {
    let mut ba = BenchArgs::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                if i < args.len() { ba.seed = args[i].parse().unwrap_or(42); }
            }
            "--runs" | "-n" => {
                i += 1;
                if i < args.len() { ba.runs = args[i].parse().unwrap_or(10); }
            }
            "--warmup" => {
                i += 1;
                if i < args.len() { ba.warmup = args[i].parse().unwrap_or(1); }
            }
            "--nogc" => ba.nogc_check = true,
            "--mir" => ba.use_mir = true,
            "--eval" => ba.use_mir = false,
            "-v" | "--verbose" => ba.verbose = true,
            "--plain" => ba.output = OutputMode::Plain,
            "--json" => ba.output = OutputMode::Json,
            "--color" => ba.output = OutputMode::Color,
            other if !other.starts_with('-') => ba.file = other.to_string(),
            other => {
                eprintln!("error: unknown flag `{}` for `cjc bench`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    if ba.file.is_empty() {
        eprintln!("error: `cjc bench` requires a .cjc file argument");
        process::exit(1);
    }
    ba
}

pub fn run(args: &[String]) {
    let ba = parse_args(args);
    let path = Path::new(&ba.file);

    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => { eprintln!("error: could not read `{}`: {}", ba.file, e); process::exit(1); }
    };

    let filename = ba.file.replace('\\', "/");

    // Parse once (parse time is separate from execution)
    let parse_start = Instant::now();
    let (program, diags) = cjc_parser::parse_source(&source);
    let parse_time = parse_start.elapsed();

    if diags.has_errors() {
        let rendered = diags.render_all_color(&source, &filename, ba.output.use_color());
        eprint!("{}", rendered);
        process::exit(1);
    }

    // NoGC pre-check
    if ba.nogc_check {
        if let Err(e) = cjc_mir_exec::verify_nogc(&program) {
            eprintln!("error: NoGC verification failed — refusing to benchmark");
            eprintln!("{}", e);
            process::exit(1);
        }
    }

    eprintln!("{} Benchmarking `{}` ({} warmup + {} measured runs, seed={})",
        output::colorize(ba.output, output::BOLD_CYAN, "[bench]"),
        filename, ba.warmup, ba.runs, ba.seed);

    // Warmup
    for _ in 0..ba.warmup {
        let mut interpreter = cjc_eval::Interpreter::new(ba.seed);
        let _ = interpreter.exec(&program);
    }

    // Measured runs
    let mut times_us: Vec<u128> = Vec::with_capacity(ba.runs);
    let mut output_lines = 0usize;

    for run_idx in 0..ba.runs {
        let start = Instant::now();
        let mut interpreter = cjc_eval::Interpreter::new(ba.seed);
        let result = interpreter.exec(&program);
        let elapsed = start.elapsed();

        times_us.push(elapsed.as_micros());

        if run_idx == 0 {
            output_lines = interpreter.output.len();
        }

        if let Err(e) = result {
            eprintln!("error: run {} failed: {}", run_idx + 1, e);
            process::exit(1);
        }
    }

    // Statistics (deterministic: sorted for percentiles)
    let mut sorted_times = times_us.clone();
    sorted_times.sort();

    let n = sorted_times.len() as f64;
    let sum: f64 = sorted_times.iter().map(|&t| t as f64).sum();
    let mean = sum / n;

    let variance = if sorted_times.len() > 1 {
        let sq_diff_sum: f64 = sorted_times.iter().map(|&t| {
            let d = t as f64 - mean;
            d * d
        }).sum();
        sq_diff_sum / (n - 1.0)
    } else {
        0.0
    };
    let stddev = variance.sqrt();
    let cv = if mean > 0.0 { stddev / mean * 100.0 } else { 0.0 };

    let min = sorted_times[0] as f64;
    let max = *sorted_times.last().unwrap() as f64;
    let median = if sorted_times.len() % 2 == 0 {
        (sorted_times[sorted_times.len() / 2 - 1] as f64
         + sorted_times[sorted_times.len() / 2] as f64) / 2.0
    } else {
        sorted_times[sorted_times.len() / 2] as f64
    };

    let p95_idx = ((sorted_times.len() as f64 * 0.95) as usize).min(sorted_times.len() - 1);
    let p95 = sorted_times[p95_idx] as f64;

    // Throughput: runs per second
    let throughput = if mean > 0.0 { 1_000_000.0 / mean } else { 0.0 };

    match ba.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file\": \"{}\",", filename);
            println!("  \"seed\": {},", ba.seed);
            println!("  \"warmup_runs\": {},", ba.warmup);
            println!("  \"measured_runs\": {},", ba.runs);
            println!("  \"parse_time_us\": {},", parse_time.as_micros());
            println!("  \"mean_us\": {},", output::format_f64(mean, 2));
            println!("  \"median_us\": {},", output::format_f64(median, 2));
            println!("  \"min_us\": {},", output::format_f64(min, 2));
            println!("  \"max_us\": {},", output::format_f64(max, 2));
            println!("  \"stddev_us\": {},", output::format_f64(stddev, 2));
            println!("  \"cv_percent\": {},", output::format_f64(cv, 2));
            println!("  \"p95_us\": {},", output::format_f64(p95, 2));
            println!("  \"throughput_per_sec\": {},", output::format_f64(throughput, 2));
            println!("  \"output_lines\": {}", output_lines);
            println!("}}");
        }
        _ => {
            eprintln!();

            let mut t = crate::table::Table::new(vec!["Metric", "Value"]);
            t.add_row_owned(vec!["Parse time".into(), format!("{:.1} us", parse_time.as_micros() as f64)]);
            t.add_row_owned(vec!["Mean".into(), format_time(mean)]);
            t.add_row_owned(vec!["Median".into(), format_time(median)]);
            t.add_row_owned(vec!["Min".into(), format_time(min)]);
            t.add_row_owned(vec!["Max".into(), format_time(max)]);
            t.add_row_owned(vec!["Std dev".into(), format_time(stddev)]);
            t.add_row_owned(vec!["CV".into(), format!("{:.1}%", cv)]);
            t.add_row_owned(vec!["P95".into(), format_time(p95)]);
            t.add_row_owned(vec!["Throughput".into(), format!("{:.1} runs/sec", throughput)]);
            t.add_row_owned(vec!["Output lines".into(), format!("{}", output_lines)]);
            eprint!("{}", t.render());

            // Stability assessment
            let stability = if cv < 5.0 {
                output::colorize(ba.output, output::BOLD_GREEN, "STABLE")
            } else if cv < 20.0 {
                output::colorize(ba.output, output::BOLD_YELLOW, "MODERATE")
            } else {
                output::colorize(ba.output, output::BOLD_RED, "UNSTABLE")
            };
            eprintln!("\nStability: {} (CV = {:.1}%)", stability, cv);

            if ba.verbose {
                eprintln!("\nPer-run times (us):");
                let mut rt = crate::table::Table::new(vec!["Run", "Time (us)"]);
                for (i, t) in times_us.iter().enumerate() {
                    rt.add_row_owned(vec![format!("{}", i + 1), format!("{}", t)]);
                }
                eprint!("{}", rt.render());
            }
        }
    }
}

fn format_time(us: f64) -> String {
    if us < 1000.0 {
        format!("{:.1} us", us)
    } else if us < 1_000_000.0 {
        format!("{:.2} ms", us / 1000.0)
    } else {
        format!("{:.3} s", us / 1_000_000.0)
    }
}

pub fn print_help() {
    eprintln!("cjc bench — Performance and stability benchmarking");
    eprintln!();
    eprintln!("Usage: cjc bench <file.cjc> [flags]");
    eprintln!();
    eprintln!("Reports: mean, median, min, max, stddev, CV, P95, throughput");
    eprintln!();
    eprintln!("Distinction from `cjc proof`:");
    eprintln!("  proof = \"Is the output deterministic?\"");
    eprintln!("  bench = \"How fast and stable is the execution?\"");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  --seed <N>      RNG seed (default: 42)");
    eprintln!("  -n, --runs <N>  Measured runs (default: 10)");
    eprintln!("  --warmup <N>    Warmup runs (default: 1)");
    eprintln!("  -v, --verbose   Show per-run times");
    eprintln!("  --plain         Plain text output");
    eprintln!("  --json          JSON output");
    eprintln!("  --color         Color output (default)");
}
