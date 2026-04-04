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
    // Second-mode flags
    pub baseline: Option<String>,
    pub save_baseline: Option<String>,
    pub fail_if_slower_than: Option<f64>,
    pub compare: Option<CompareMode>,
    pub csv: bool,
    pub markdown: bool,
    pub out_file: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareMode {
    Eval,
    Mir,
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
            baseline: None,
            save_baseline: None,
            fail_if_slower_than: None,
            compare: None,
            csv: false,
            markdown: false,
            out_file: None,
        }
    }
}

/// Benchmark statistics for a single executor run.
#[derive(Clone)]
struct BenchStats {
    label: String,
    parse_time_us: u128,
    times_us: Vec<u128>,
    mean: f64,
    median: f64,
    min: f64,
    max: f64,
    stddev: f64,
    cv: f64,
    p95: f64,
    throughput: f64,
    output_lines: usize,
}

/// Baseline JSON structure — deterministic field order via manual serialization.
struct Baseline {
    file: String,
    seed: u64,
    warmup_runs: usize,
    measured_runs: usize,
    parse_time_us: u128,
    mean_us: f64,
    median_us: f64,
    min_us: f64,
    max_us: f64,
    stddev_us: f64,
    cv_percent: f64,
    p95_us: f64,
    throughput_per_sec: f64,
    output_lines: usize,
}

impl Baseline {
    fn from_stats(stats: &BenchStats, file: &str, seed: u64, warmup: usize, runs: usize) -> Self {
        Self {
            file: file.to_string(),
            seed,
            warmup_runs: warmup,
            measured_runs: runs,
            parse_time_us: stats.parse_time_us,
            mean_us: stats.mean,
            median_us: stats.median,
            min_us: stats.min,
            max_us: stats.max,
            stddev_us: stats.stddev,
            cv_percent: stats.cv,
            p95_us: stats.p95,
            throughput_per_sec: stats.throughput,
            output_lines: stats.output_lines,
        }
    }

    fn to_json(&self) -> String {
        let mut s = String::new();
        s.push_str("{\n");
        s.push_str(&format!("  \"file\": \"{}\",\n", self.file));
        s.push_str(&format!("  \"seed\": {},\n", self.seed));
        s.push_str(&format!("  \"warmup_runs\": {},\n", self.warmup_runs));
        s.push_str(&format!("  \"measured_runs\": {},\n", self.measured_runs));
        s.push_str(&format!("  \"parse_time_us\": {},\n", self.parse_time_us));
        s.push_str(&format!("  \"mean_us\": {},\n", output::format_f64(self.mean_us, 2)));
        s.push_str(&format!("  \"median_us\": {},\n", output::format_f64(self.median_us, 2)));
        s.push_str(&format!("  \"min_us\": {},\n", output::format_f64(self.min_us, 2)));
        s.push_str(&format!("  \"max_us\": {},\n", output::format_f64(self.max_us, 2)));
        s.push_str(&format!("  \"stddev_us\": {},\n", output::format_f64(self.stddev_us, 2)));
        s.push_str(&format!("  \"cv_percent\": {},\n", output::format_f64(self.cv_percent, 2)));
        s.push_str(&format!("  \"p95_us\": {},\n", output::format_f64(self.p95_us, 2)));
        s.push_str(&format!("  \"throughput_per_sec\": {},\n", output::format_f64(self.throughput_per_sec, 2)));
        s.push_str(&format!("  \"output_lines\": {}\n", self.output_lines));
        s.push_str("}");
        s
    }

    fn from_json(json_str: &str) -> Result<Self, String> {
        // Minimal hand-rolled JSON parser — zero external dependencies.
        fn extract_str<'a>(json: &'a str, key: &str) -> Result<&'a str, String> {
            let pattern = format!("\"{}\":", key);
            let pos = json.find(&pattern)
                .ok_or_else(|| format!("missing key `{}`", key))?;
            let rest = &json[pos + pattern.len()..];
            let rest = rest.trim_start();
            if !rest.starts_with('"') {
                return Err(format!("expected string for `{}`", key));
            }
            let rest = &rest[1..];
            let end = rest.find('"').ok_or_else(|| format!("unterminated string for `{}`", key))?;
            Ok(&rest[..end])
        }

        fn extract_num(json: &str, key: &str) -> Result<f64, String> {
            let pattern = format!("\"{}\":", key);
            let pos = json.find(&pattern)
                .ok_or_else(|| format!("missing key `{}`", key))?;
            let rest = &json[pos + pattern.len()..];
            let rest = rest.trim_start();
            let end = rest.find(|c: char| c == ',' || c == '\n' || c == '}' || c == ' ')
                .unwrap_or(rest.len());
            let val_str = rest[..end].trim();
            val_str.parse::<f64>().map_err(|e| format!("invalid number for `{}`: {}", key, e))
        }

        Ok(Baseline {
            file: extract_str(json_str, "file")?.to_string(),
            seed: extract_num(json_str, "seed")? as u64,
            warmup_runs: extract_num(json_str, "warmup_runs")? as usize,
            measured_runs: extract_num(json_str, "measured_runs")? as usize,
            parse_time_us: extract_num(json_str, "parse_time_us")? as u128,
            mean_us: extract_num(json_str, "mean_us")?,
            median_us: extract_num(json_str, "median_us")?,
            min_us: extract_num(json_str, "min_us")?,
            max_us: extract_num(json_str, "max_us")?,
            stddev_us: extract_num(json_str, "stddev_us")?,
            cv_percent: extract_num(json_str, "cv_percent")?,
            p95_us: extract_num(json_str, "p95_us")?,
            throughput_per_sec: extract_num(json_str, "throughput_per_sec")?,
            output_lines: extract_num(json_str, "output_lines")? as usize,
        })
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
            "--baseline" => {
                i += 1;
                if i < args.len() { ba.baseline = Some(args[i].clone()); }
            }
            "--save-baseline" => {
                i += 1;
                if i < args.len() { ba.save_baseline = Some(args[i].clone()); }
            }
            "--fail-if-slower-than" => {
                i += 1;
                if i < args.len() {
                    match args[i].parse::<f64>() {
                        Ok(pct) => ba.fail_if_slower_than = Some(pct),
                        Err(_) => {
                            eprintln!("error: --fail-if-slower-than requires a numeric percentage");
                            process::exit(1);
                        }
                    }
                }
            }
            "--compare" => {
                i += 1;
                if i < args.len() {
                    match args[i].as_str() {
                        "eval" => ba.compare = Some(CompareMode::Eval),
                        "mir" => ba.compare = Some(CompareMode::Mir),
                        other => {
                            eprintln!("error: --compare expects `eval` or `mir`, got `{}`", other);
                            process::exit(1);
                        }
                    }
                }
            }
            "--csv" => ba.csv = true,
            "--markdown" => ba.markdown = true,
            "--out" => {
                i += 1;
                if i < args.len() { ba.out_file = Some(args[i].clone()); }
            }
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
    // --fail-if-slower-than requires --baseline
    if ba.fail_if_slower_than.is_some() && ba.baseline.is_none() {
        eprintln!("error: --fail-if-slower-than requires --baseline <file>");
        process::exit(1);
    }
    ba
}

/// Run a benchmark using the AST-eval executor and collect stats.
fn bench_eval(
    program: &cjc_ast::Program,
    seed: u64,
    warmup: usize,
    runs: usize,
    parse_time_us: u128,
) -> Result<BenchStats, String> {
    // Warmup
    for _ in 0..warmup {
        let mut interpreter = cjc_eval::Interpreter::new(seed);
        let _ = interpreter.exec(program);
    }

    // Measured runs
    let mut times_us: Vec<u128> = Vec::with_capacity(runs);
    let mut output_lines = 0usize;

    for run_idx in 0..runs {
        let start = Instant::now();
        let mut interpreter = cjc_eval::Interpreter::new(seed);
        let result = interpreter.exec(program);
        let elapsed = start.elapsed();

        times_us.push(elapsed.as_micros());

        if run_idx == 0 {
            output_lines = interpreter.output.len();
        }

        if let Err(e) = result {
            return Err(format!("eval run {} failed: {}", run_idx + 1, e));
        }
    }

    Ok(compute_stats("eval".to_string(), parse_time_us, times_us, output_lines))
}

/// Run a benchmark using the MIR executor and collect stats.
fn bench_mir(
    program: &cjc_ast::Program,
    seed: u64,
    warmup: usize,
    runs: usize,
    parse_time_us: u128,
) -> Result<BenchStats, String> {
    // Warmup
    for _ in 0..warmup {
        let _ = cjc_mir_exec::run_program_with_executor(program, seed);
    }

    // Measured runs
    let mut times_us: Vec<u128> = Vec::with_capacity(runs);
    let mut output_lines = 0usize;

    for run_idx in 0..runs {
        let start = Instant::now();
        let result = cjc_mir_exec::run_program_with_executor(program, seed);
        let elapsed = start.elapsed();

        times_us.push(elapsed.as_micros());

        match result {
            Ok((_value, executor)) => {
                if run_idx == 0 {
                    output_lines = executor.output.len();
                }
            }
            Err(e) => {
                return Err(format!("mir run {} failed: {}", run_idx + 1, e));
            }
        }
    }

    Ok(compute_stats("mir".to_string(), parse_time_us, times_us, output_lines))
}

/// Compute statistics from raw timing data.
fn compute_stats(
    label: String,
    parse_time_us: u128,
    times_us: Vec<u128>,
    output_lines: usize,
) -> BenchStats {
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

    let throughput = if mean > 0.0 { 1_000_000.0 / mean } else { 0.0 };

    BenchStats {
        label,
        parse_time_us,
        times_us,
        mean,
        median,
        min,
        max,
        stddev,
        cv,
        p95,
        throughput,
        output_lines,
    }
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

    let parse_time_us = parse_time.as_micros();

    // Load baseline if requested
    let loaded_baseline = if let Some(ref bl_path) = ba.baseline {
        match fs::read_to_string(bl_path) {
            Ok(json_str) => match Baseline::from_json(&json_str) {
                Ok(bl) => Some(bl),
                Err(e) => {
                    eprintln!("error: failed to parse baseline `{}`: {}", bl_path, e);
                    process::exit(1);
                }
            },
            Err(e) => {
                eprintln!("error: could not read baseline `{}`: {}", bl_path, e);
                process::exit(1);
            }
        }
    } else {
        None
    };

    // Determine if we are in compare mode
    if let Some(compare_mode) = ba.compare {
        run_compare(&ba, &program, &filename, parse_time_us, compare_mode, loaded_baseline.as_ref());
        return;
    }

    eprintln!("{} Benchmarking `{}` ({} warmup + {} measured runs, seed={})",
        output::colorize(ba.output, output::BOLD_CYAN, "[bench]"),
        filename, ba.warmup, ba.runs, ba.seed);

    let executor_label = if ba.use_mir { "mir" } else { "eval" };
    eprintln!("  executor: {}", executor_label);

    // Run benchmark with selected executor
    let stats = if ba.use_mir {
        match bench_mir(&program, ba.seed, ba.warmup, ba.runs, parse_time_us) {
            Ok(s) => s,
            Err(e) => { eprintln!("error: {}", e); process::exit(1); }
        }
    } else {
        match bench_eval(&program, ba.seed, ba.warmup, ba.runs, parse_time_us) {
            Ok(s) => s,
            Err(e) => { eprintln!("error: {}", e); process::exit(1); }
        }
    };

    // Save baseline if requested
    if let Some(ref save_path) = ba.save_baseline {
        let bl = Baseline::from_stats(&stats, &filename, ba.seed, ba.warmup, ba.runs);
        let json = bl.to_json();
        if let Err(e) = fs::write(save_path, &json) {
            eprintln!("error: could not write baseline to `{}`: {}", save_path, e);
            process::exit(1);
        }
        eprintln!("  baseline saved to `{}`", save_path);
    }

    // Format output
    let output_text = format_single_output(&ba, &stats, &filename, loaded_baseline.as_ref());

    // Write to file or stdout
    emit_output(&ba, &output_text);

    // Check --fail-if-slower-than
    if let (Some(threshold_pct), Some(ref bl)) = (ba.fail_if_slower_than, &loaded_baseline) {
        if bl.mean_us > 0.0 {
            let slowdown_pct = (stats.mean - bl.mean_us) / bl.mean_us * 100.0;
            if slowdown_pct > threshold_pct {
                eprintln!(
                    "\nerror: mean time is {:.1}% slower than baseline (threshold: {:.1}%)",
                    slowdown_pct, threshold_pct
                );
                process::exit(1);
            }
        }
    }
}

/// Run benchmark in --compare mode: both eval and mir, side-by-side.
fn run_compare(
    ba: &BenchArgs,
    program: &cjc_ast::Program,
    filename: &str,
    parse_time_us: u128,
    compare_mode: CompareMode,
    loaded_baseline: Option<&Baseline>,
) {
    eprintln!("{} Comparing executors on `{}` ({} warmup + {} measured runs, seed={})",
        output::colorize(ba.output, output::BOLD_CYAN, "[bench]"),
        filename, ba.warmup, ba.runs, ba.seed);

    // Always run both executors in compare mode. The flag value indicates
    // which executor is considered the "primary" for baseline saving.
    let eval_stats = match bench_eval(program, ba.seed, ba.warmup, ba.runs, parse_time_us) {
        Ok(s) => s,
        Err(e) => { eprintln!("error: {}", e); process::exit(1); }
    };

    let mir_stats = match bench_mir(program, ba.seed, ba.warmup, ba.runs, parse_time_us) {
        Ok(s) => s,
        Err(e) => { eprintln!("error: {}", e); process::exit(1); }
    };

    let primary_stats = match compare_mode {
        CompareMode::Eval => &eval_stats,
        CompareMode::Mir => &mir_stats,
    };

    // Save baseline from the primary executor if requested
    if let Some(ref save_path) = ba.save_baseline {
        let bl = Baseline::from_stats(primary_stats, filename, ba.seed, ba.warmup, ba.runs);
        let json = bl.to_json();
        if let Err(e) = fs::write(save_path, &json) {
            eprintln!("error: could not write baseline to `{}`: {}", save_path, e);
            process::exit(1);
        }
        eprintln!("  baseline saved to `{}`", save_path);
    }

    let output_text = format_compare_output(ba, &eval_stats, &mir_stats, filename, loaded_baseline);

    emit_output(ba, &output_text);

    // Check --fail-if-slower-than against the primary executor
    if let (Some(threshold_pct), Some(bl)) = (ba.fail_if_slower_than, loaded_baseline) {
        if bl.mean_us > 0.0 {
            let slowdown_pct = (primary_stats.mean - bl.mean_us) / bl.mean_us * 100.0;
            if slowdown_pct > threshold_pct {
                eprintln!(
                    "\nerror: {} mean time is {:.1}% slower than baseline (threshold: {:.1}%)",
                    primary_stats.label, slowdown_pct, threshold_pct
                );
                process::exit(1);
            }
        }
    }
}

/// Format output for a single-executor benchmark run.
fn format_single_output(
    ba: &BenchArgs,
    stats: &BenchStats,
    filename: &str,
    loaded_baseline: Option<&Baseline>,
) -> String {
    if ba.csv {
        return format_csv_single(stats, filename, ba.seed, ba.warmup, ba.runs);
    }
    if ba.markdown {
        return format_markdown_single(stats, filename, loaded_baseline);
    }

    match ba.output {
        OutputMode::Json => format_json_single(stats, filename, ba.seed, ba.warmup, ba.runs, loaded_baseline),
        _ => format_table_single(ba, stats, loaded_baseline),
    }
}

/// Format output for a compare-mode benchmark run.
fn format_compare_output(
    ba: &BenchArgs,
    eval_stats: &BenchStats,
    mir_stats: &BenchStats,
    filename: &str,
    loaded_baseline: Option<&Baseline>,
) -> String {
    if ba.csv {
        return format_csv_compare(eval_stats, mir_stats, filename, ba.seed, ba.warmup, ba.runs);
    }
    if ba.markdown {
        return format_markdown_compare(eval_stats, mir_stats, filename, loaded_baseline);
    }

    match ba.output {
        OutputMode::Json => format_json_compare(eval_stats, mir_stats, filename, ba.seed, ba.warmup, ba.runs, loaded_baseline),
        _ => format_table_compare(ba, eval_stats, mir_stats, loaded_baseline),
    }
}

// ── JSON formatting ─────────────────────────────────────────────────

fn format_json_single(
    stats: &BenchStats,
    filename: &str,
    seed: u64,
    warmup: usize,
    runs: usize,
    loaded_baseline: Option<&Baseline>,
) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str(&format!("  \"file\": \"{}\",\n", filename));
    s.push_str(&format!("  \"seed\": {},\n", seed));
    s.push_str(&format!("  \"warmup_runs\": {},\n", warmup));
    s.push_str(&format!("  \"measured_runs\": {},\n", runs));
    s.push_str(&format!("  \"executor\": \"{}\",\n", stats.label));
    s.push_str(&format!("  \"parse_time_us\": {},\n", stats.parse_time_us));
    s.push_str(&format!("  \"mean_us\": {},\n", output::format_f64(stats.mean, 2)));
    s.push_str(&format!("  \"median_us\": {},\n", output::format_f64(stats.median, 2)));
    s.push_str(&format!("  \"min_us\": {},\n", output::format_f64(stats.min, 2)));
    s.push_str(&format!("  \"max_us\": {},\n", output::format_f64(stats.max, 2)));
    s.push_str(&format!("  \"stddev_us\": {},\n", output::format_f64(stats.stddev, 2)));
    s.push_str(&format!("  \"cv_percent\": {},\n", output::format_f64(stats.cv, 2)));
    s.push_str(&format!("  \"p95_us\": {},\n", output::format_f64(stats.p95, 2)));
    s.push_str(&format!("  \"throughput_per_sec\": {},\n", output::format_f64(stats.throughput, 2)));
    s.push_str(&format!("  \"output_lines\": {}", stats.output_lines));

    if let Some(bl) = loaded_baseline {
        s.push_str(",\n  \"baseline_comparison\": {\n");
        let delta_mean = stats.mean - bl.mean_us;
        let delta_pct = if bl.mean_us > 0.0 { delta_mean / bl.mean_us * 100.0 } else { 0.0 };
        s.push_str(&format!("    \"baseline_mean_us\": {},\n", output::format_f64(bl.mean_us, 2)));
        s.push_str(&format!("    \"delta_mean_us\": {},\n", output::format_f64(delta_mean, 2)));
        s.push_str(&format!("    \"delta_percent\": {}\n", output::format_f64(delta_pct, 2)));
        s.push_str("  }");
    }

    s.push_str("\n}");
    s
}

fn format_json_compare(
    eval_stats: &BenchStats,
    mir_stats: &BenchStats,
    filename: &str,
    seed: u64,
    warmup: usize,
    runs: usize,
    loaded_baseline: Option<&Baseline>,
) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str(&format!("  \"file\": \"{}\",\n", filename));
    s.push_str(&format!("  \"seed\": {},\n", seed));
    s.push_str(&format!("  \"warmup_runs\": {},\n", warmup));
    s.push_str(&format!("  \"measured_runs\": {},\n", runs));
    s.push_str(&format!("  \"parse_time_us\": {},\n", eval_stats.parse_time_us));

    // Eval section
    s.push_str("  \"eval\": {\n");
    s.push_str(&format!("    \"mean_us\": {},\n", output::format_f64(eval_stats.mean, 2)));
    s.push_str(&format!("    \"median_us\": {},\n", output::format_f64(eval_stats.median, 2)));
    s.push_str(&format!("    \"min_us\": {},\n", output::format_f64(eval_stats.min, 2)));
    s.push_str(&format!("    \"max_us\": {},\n", output::format_f64(eval_stats.max, 2)));
    s.push_str(&format!("    \"stddev_us\": {},\n", output::format_f64(eval_stats.stddev, 2)));
    s.push_str(&format!("    \"cv_percent\": {},\n", output::format_f64(eval_stats.cv, 2)));
    s.push_str(&format!("    \"p95_us\": {},\n", output::format_f64(eval_stats.p95, 2)));
    s.push_str(&format!("    \"throughput_per_sec\": {},\n", output::format_f64(eval_stats.throughput, 2)));
    s.push_str(&format!("    \"output_lines\": {}\n", eval_stats.output_lines));
    s.push_str("  },\n");

    // MIR section
    s.push_str("  \"mir\": {\n");
    s.push_str(&format!("    \"mean_us\": {},\n", output::format_f64(mir_stats.mean, 2)));
    s.push_str(&format!("    \"median_us\": {},\n", output::format_f64(mir_stats.median, 2)));
    s.push_str(&format!("    \"min_us\": {},\n", output::format_f64(mir_stats.min, 2)));
    s.push_str(&format!("    \"max_us\": {},\n", output::format_f64(mir_stats.max, 2)));
    s.push_str(&format!("    \"stddev_us\": {},\n", output::format_f64(mir_stats.stddev, 2)));
    s.push_str(&format!("    \"cv_percent\": {},\n", output::format_f64(mir_stats.cv, 2)));
    s.push_str(&format!("    \"p95_us\": {},\n", output::format_f64(mir_stats.p95, 2)));
    s.push_str(&format!("    \"throughput_per_sec\": {},\n", output::format_f64(mir_stats.throughput, 2)));
    s.push_str(&format!("    \"output_lines\": {}\n", mir_stats.output_lines));
    s.push_str("  },\n");

    // Speedup ratio
    let speedup = if mir_stats.mean > 0.0 { eval_stats.mean / mir_stats.mean } else { 0.0 };
    s.push_str(&format!("  \"mir_speedup_vs_eval\": {}", output::format_f64(speedup, 3)));

    if let Some(bl) = loaded_baseline {
        s.push_str(",\n  \"baseline_comparison\": {\n");
        s.push_str(&format!("    \"baseline_mean_us\": {}\n", output::format_f64(bl.mean_us, 2)));
        s.push_str("  }");
    }

    s.push_str("\n}");
    s
}

// ── CSV formatting ──────────────────────────────────────────────────

fn format_csv_single(
    stats: &BenchStats,
    filename: &str,
    seed: u64,
    warmup: usize,
    runs: usize,
) -> String {
    let mut s = String::new();
    s.push_str("file,executor,seed,warmup,runs,parse_us,mean_us,median_us,min_us,max_us,stddev_us,cv_pct,p95_us,throughput,output_lines\n");
    s.push_str(&format!(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
        filename,
        stats.label,
        seed,
        warmup,
        runs,
        stats.parse_time_us,
        output::format_f64(stats.mean, 2),
        output::format_f64(stats.median, 2),
        output::format_f64(stats.min, 2),
        output::format_f64(stats.max, 2),
        output::format_f64(stats.stddev, 2),
        output::format_f64(stats.cv, 2),
        output::format_f64(stats.p95, 2),
        output::format_f64(stats.throughput, 2),
        stats.output_lines,
    ));
    s
}

fn format_csv_compare(
    eval_stats: &BenchStats,
    mir_stats: &BenchStats,
    filename: &str,
    seed: u64,
    warmup: usize,
    runs: usize,
) -> String {
    let mut s = String::new();
    s.push_str("file,executor,seed,warmup,runs,parse_us,mean_us,median_us,min_us,max_us,stddev_us,cv_pct,p95_us,throughput,output_lines\n");
    for stats in &[eval_stats, mir_stats] {
        s.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            filename,
            stats.label,
            seed,
            warmup,
            runs,
            stats.parse_time_us,
            output::format_f64(stats.mean, 2),
            output::format_f64(stats.median, 2),
            output::format_f64(stats.min, 2),
            output::format_f64(stats.max, 2),
            output::format_f64(stats.stddev, 2),
            output::format_f64(stats.cv, 2),
            output::format_f64(stats.p95, 2),
            output::format_f64(stats.throughput, 2),
            stats.output_lines,
        ));
    }
    s
}

// ── Markdown formatting ─────────────────────────────────────────────

fn format_markdown_single(
    stats: &BenchStats,
    filename: &str,
    loaded_baseline: Option<&Baseline>,
) -> String {
    let mut s = String::new();
    s.push_str(&format!("## Benchmark: `{}`\n\n", filename));
    s.push_str(&format!("Executor: **{}**\n\n", stats.label));
    s.push_str("| Metric | Value |\n");
    s.push_str("|--------|-------|\n");
    s.push_str(&format!("| Parse time | {} |\n", format_time(stats.parse_time_us as f64)));
    s.push_str(&format!("| Mean | {} |\n", format_time(stats.mean)));
    s.push_str(&format!("| Median | {} |\n", format_time(stats.median)));
    s.push_str(&format!("| Min | {} |\n", format_time(stats.min)));
    s.push_str(&format!("| Max | {} |\n", format_time(stats.max)));
    s.push_str(&format!("| Std dev | {} |\n", format_time(stats.stddev)));
    s.push_str(&format!("| CV | {:.1}% |\n", stats.cv));
    s.push_str(&format!("| P95 | {} |\n", format_time(stats.p95)));
    s.push_str(&format!("| Throughput | {:.1} runs/sec |\n", stats.throughput));
    s.push_str(&format!("| Output lines | {} |\n", stats.output_lines));

    if let Some(bl) = loaded_baseline {
        let delta_pct = if bl.mean_us > 0.0 {
            (stats.mean - bl.mean_us) / bl.mean_us * 100.0
        } else {
            0.0
        };
        let direction = if delta_pct > 0.0 { "slower" } else { "faster" };
        s.push_str(&format!(
            "\n**Baseline comparison:** {:.1}% {} (baseline mean: {})\n",
            delta_pct.abs(), direction, format_time(bl.mean_us)
        ));
    }

    s
}

fn format_markdown_compare(
    eval_stats: &BenchStats,
    mir_stats: &BenchStats,
    filename: &str,
    loaded_baseline: Option<&Baseline>,
) -> String {
    let mut s = String::new();
    s.push_str(&format!("## Benchmark Comparison: `{}`\n\n", filename));
    s.push_str("| Metric | eval | mir |\n");
    s.push_str("|--------|------|-----|\n");
    s.push_str(&format!("| Parse time | {} | {} |\n",
        format_time(eval_stats.parse_time_us as f64),
        format_time(mir_stats.parse_time_us as f64)));
    s.push_str(&format!("| Mean | {} | {} |\n", format_time(eval_stats.mean), format_time(mir_stats.mean)));
    s.push_str(&format!("| Median | {} | {} |\n", format_time(eval_stats.median), format_time(mir_stats.median)));
    s.push_str(&format!("| Min | {} | {} |\n", format_time(eval_stats.min), format_time(mir_stats.min)));
    s.push_str(&format!("| Max | {} | {} |\n", format_time(eval_stats.max), format_time(mir_stats.max)));
    s.push_str(&format!("| Std dev | {} | {} |\n", format_time(eval_stats.stddev), format_time(mir_stats.stddev)));
    s.push_str(&format!("| CV | {:.1}% | {:.1}% |\n", eval_stats.cv, mir_stats.cv));
    s.push_str(&format!("| P95 | {} | {} |\n", format_time(eval_stats.p95), format_time(mir_stats.p95)));
    s.push_str(&format!("| Throughput | {:.1} runs/sec | {:.1} runs/sec |\n", eval_stats.throughput, mir_stats.throughput));
    s.push_str(&format!("| Output lines | {} | {} |\n", eval_stats.output_lines, mir_stats.output_lines));

    let speedup = if mir_stats.mean > 0.0 { eval_stats.mean / mir_stats.mean } else { 0.0 };
    s.push_str(&format!("\n**MIR speedup vs eval:** {:.2}x\n", speedup));

    if let Some(bl) = loaded_baseline {
        s.push_str(&format!("\n**Baseline mean:** {}\n", format_time(bl.mean_us)));
    }

    s
}

// ── Table formatting (human-readable, to stderr) ────────────────────

fn format_table_single(
    ba: &BenchArgs,
    stats: &BenchStats,
    loaded_baseline: Option<&Baseline>,
) -> String {
    let mut out = String::new();
    out.push('\n');

    let mut t = crate::table::Table::new(vec!["Metric", "Value"]);
    t.add_row_owned(vec!["Parse time".into(), format!("{:.1} us", stats.parse_time_us as f64)]);
    t.add_row_owned(vec!["Mean".into(), format_time(stats.mean)]);
    t.add_row_owned(vec!["Median".into(), format_time(stats.median)]);
    t.add_row_owned(vec!["Min".into(), format_time(stats.min)]);
    t.add_row_owned(vec!["Max".into(), format_time(stats.max)]);
    t.add_row_owned(vec!["Std dev".into(), format_time(stats.stddev)]);
    t.add_row_owned(vec!["CV".into(), format!("{:.1}%", stats.cv)]);
    t.add_row_owned(vec!["P95".into(), format_time(stats.p95)]);
    t.add_row_owned(vec!["Throughput".into(), format!("{:.1} runs/sec", stats.throughput)]);
    t.add_row_owned(vec!["Output lines".into(), format!("{}", stats.output_lines)]);
    out.push_str(&t.render());

    // Stability assessment
    let stability = if stats.cv < 5.0 {
        output::colorize(ba.output, output::BOLD_GREEN, "STABLE")
    } else if stats.cv < 20.0 {
        output::colorize(ba.output, output::BOLD_YELLOW, "MODERATE")
    } else {
        output::colorize(ba.output, output::BOLD_RED, "UNSTABLE")
    };
    out.push_str(&format!("\nStability: {} (CV = {:.1}%)\n", stability, stats.cv));

    // Warn about high CV when --fail-if-slower-than is active
    if stats.cv > 15.0 && ba.fail_if_slower_than.is_some() {
        out.push_str(&format!("\n{}: high variance detected (CV = {:.1}%); results may be unreliable for CI gating. Consider using --compare mir for more stable results or increasing --runs.\n",
            output::colorize(ba.output, output::BOLD_YELLOW, "warning"),
            stats.cv));
    }

    // Baseline comparison
    if let Some(bl) = loaded_baseline {
        let delta_pct = if bl.mean_us > 0.0 {
            (stats.mean - bl.mean_us) / bl.mean_us * 100.0
        } else {
            0.0
        };
        let direction = if delta_pct > 0.0 { "slower" } else { "faster" };
        let color = if delta_pct > 5.0 {
            output::BOLD_RED
        } else if delta_pct < -5.0 {
            output::BOLD_GREEN
        } else {
            output::BOLD_YELLOW
        };
        out.push_str(&format!(
            "\nBaseline: {} ({:.1}% {})\n",
            output::colorize(ba.output, color, &format!("{} -> {}", format_time(bl.mean_us), format_time(stats.mean))),
            delta_pct.abs(),
            direction,
        ));
    }

    // Verbose per-run times
    if ba.verbose {
        out.push_str("\nPer-run times (us):\n");
        let mut rt = crate::table::Table::new(vec!["Run", "Time (us)"]);
        for (i, t) in stats.times_us.iter().enumerate() {
            rt.add_row_owned(vec![format!("{}", i + 1), format!("{}", t)]);
        }
        out.push_str(&rt.render());
    }

    out
}

fn format_table_compare(
    ba: &BenchArgs,
    eval_stats: &BenchStats,
    mir_stats: &BenchStats,
    loaded_baseline: Option<&Baseline>,
) -> String {
    let mut out = String::new();
    out.push('\n');

    let mut t = crate::table::Table::new(vec!["Metric", "eval", "mir"]);
    t.add_row_owned(vec![
        "Parse time".into(),
        format!("{:.1} us", eval_stats.parse_time_us as f64),
        format!("{:.1} us", mir_stats.parse_time_us as f64),
    ]);
    t.add_row_owned(vec!["Mean".into(), format_time(eval_stats.mean), format_time(mir_stats.mean)]);
    t.add_row_owned(vec!["Median".into(), format_time(eval_stats.median), format_time(mir_stats.median)]);
    t.add_row_owned(vec!["Min".into(), format_time(eval_stats.min), format_time(mir_stats.min)]);
    t.add_row_owned(vec!["Max".into(), format_time(eval_stats.max), format_time(mir_stats.max)]);
    t.add_row_owned(vec!["Std dev".into(), format_time(eval_stats.stddev), format_time(mir_stats.stddev)]);
    t.add_row_owned(vec!["CV".into(), format!("{:.1}%", eval_stats.cv), format!("{:.1}%", mir_stats.cv)]);
    t.add_row_owned(vec!["P95".into(), format_time(eval_stats.p95), format_time(mir_stats.p95)]);
    t.add_row_owned(vec![
        "Throughput".into(),
        format!("{:.1} runs/sec", eval_stats.throughput),
        format!("{:.1} runs/sec", mir_stats.throughput),
    ]);
    t.add_row_owned(vec![
        "Output lines".into(),
        format!("{}", eval_stats.output_lines),
        format!("{}", mir_stats.output_lines),
    ]);
    out.push_str(&t.render());

    let speedup = if mir_stats.mean > 0.0 { eval_stats.mean / mir_stats.mean } else { 0.0 };
    let speedup_color = if speedup > 1.0 { output::BOLD_GREEN } else { output::BOLD_RED };
    out.push_str(&format!(
        "\nMIR speedup vs eval: {}\n",
        output::colorize(ba.output, speedup_color, &format!("{:.2}x", speedup)),
    ));

    if let Some(bl) = loaded_baseline {
        out.push_str(&format!("\nBaseline mean: {}\n", format_time(bl.mean_us)));
    }

    // Verbose per-run times for both
    if ba.verbose {
        out.push_str("\nPer-run times (us):\n");
        let max_runs = eval_stats.times_us.len().max(mir_stats.times_us.len());
        let mut rt = crate::table::Table::new(vec!["Run", "eval (us)", "mir (us)"]);
        for i in 0..max_runs {
            let eval_val = eval_stats.times_us.get(i).map(|t| format!("{}", t)).unwrap_or_default();
            let mir_val = mir_stats.times_us.get(i).map(|t| format!("{}", t)).unwrap_or_default();
            rt.add_row_owned(vec![format!("{}", i + 1), eval_val, mir_val]);
        }
        out.push_str(&rt.render());
    }

    out
}

// ── Output emission ─────────────────────────────────────────────────

fn emit_output(ba: &BenchArgs, output_text: &str) {
    if let Some(ref out_path) = ba.out_file {
        if let Err(e) = fs::write(out_path, output_text) {
            eprintln!("error: could not write to `{}`: {}", out_path, e);
            process::exit(1);
        }
        eprintln!("  output written to `{}`", out_path);
    } else if ba.csv || ba.markdown {
        // CSV and Markdown go to stdout for piping
        print!("{}", output_text);
    } else {
        match ba.output {
            OutputMode::Json => {
                println!("{}", output_text);
            }
            _ => {
                // Table output goes to stderr (as the original code did)
                eprint!("{}", output_text);
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
    eprintln!("  --seed <N>           RNG seed (default: 42)");
    eprintln!("  -n, --runs <N>       Measured runs (default: 10)");
    eprintln!("  --warmup <N>         Warmup runs (default: 1)");
    eprintln!("  -v, --verbose        Show per-run times");
    eprintln!("  --plain              Plain text output");
    eprintln!("  --json               JSON output");
    eprintln!("  --color              Color output (default)");
    eprintln!("  --mir                Use MIR executor");
    eprintln!("  --eval               Use eval executor (default)");
    eprintln!("  --nogc               Verify NoGC before benchmarking");
    eprintln!();
    eprintln!("Comparison & Baselines:");
    eprintln!("  --baseline <file>          Load baseline JSON for comparison");
    eprintln!("  --save-baseline <file>     Save results as baseline JSON");
    eprintln!("  --fail-if-slower-than <N>  Exit 1 if mean is >N% slower (requires --baseline)");
    eprintln!("  --compare eval|mir         Run both executors, show side-by-side");
    eprintln!();
    eprintln!("Output Formats:");
    eprintln!("  --csv                Output as CSV row");
    eprintln!("  --markdown           Output as Markdown table");
    eprintln!("  --out <file>         Write output to file");
}
