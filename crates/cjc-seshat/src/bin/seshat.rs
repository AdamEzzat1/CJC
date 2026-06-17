//! `seshat` — the Seshat profiler CLI.
//!
//! Subcommands:
//!   seshat analyze <trace.seshat> [--json] [--svg <out.svg>]
//!   seshat diff <baseline.seshat> <candidate.seshat>
//!   seshat record-demo <out.seshat> [--ms <interval>]   (needs --features collect-live)
//!
//! `analyze`/`diff` are pure and always available. `record-demo` installs the
//! live allocator and records a real in-process Rust workload — the proof that
//! Seshat can capture a real `.seshat`, not just analyze synthetic ones.

// Install the live allocator only when the collector feature is on, so the
// default build pays nothing.
#[cfg(feature = "collect-live")]
#[global_allocator]
static GLOBAL: cjc_seshat::collect::SeshatAlloc = cjc_seshat::collect::SeshatAlloc;

use std::process::ExitCode;

use cjc_seshat::{analyze_trace, diff, render, replay};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("help");
    let rest = &args[args.len().min(1)..];

    let result = match cmd {
        "analyze" => cmd_analyze(rest),
        "diff" => cmd_diff(rest),
        "variance" => cmd_variance(rest),
        "merge" => cmd_merge(rest),
        "record-demo" => cmd_record_demo(rest),
        "help" | "-h" | "--help" => {
            print_usage();
            Ok(())
        }
        other => Err(format!("unknown subcommand `{other}`\n\n{USAGE}")),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("seshat: {msg}");
            ExitCode::FAILURE
        }
    }
}

const USAGE: &str = "\
usage:
  seshat analyze <trace.seshat> [--json] [--svg <out.svg>]
  seshat diff <baseline.seshat> <candidate.seshat>
  seshat variance <run1.seshat> <run2.seshat> [run3.seshat ...]
  seshat merge <host.seshat> <native.seshat> [native2.seshat ...] [--under <name>] [--out <merged.seshat>] [--json] [--svg <out.svg>]
  seshat record-demo <out.seshat> [--ms <interval>] [--unwind]   (needs --features collect-live)";

fn print_usage() {
    println!("{USAGE}");
}

fn read_trace(path: &str) -> Result<cjc_seshat::Trace, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read '{path}': {e}"))?;
    replay(&bytes).map_err(|e| format!("'{path}' is not a valid .seshat trace: {e}"))
}

fn cmd_analyze(args: &[String]) -> Result<(), String> {
    let mut path: Option<&str> = None;
    let mut as_json = false;
    let mut svg_out: Option<&str> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => as_json = true,
            "--svg" => {
                i += 1;
                svg_out = Some(args.get(i).ok_or("--svg needs a path")?.as_str());
            }
            p if !p.starts_with("--") => path = Some(p),
            other => return Err(format!("unknown flag `{other}`")),
        }
        i += 1;
    }
    let path = path.ok_or("analyze needs a <trace.seshat> path")?;
    let trace = read_trace(path)?;
    let report = analyze_trace(&trace);

    if let Some(out) = svg_out {
        let svg = render::flamegraph_svg(&report);
        std::fs::write(out, svg).map_err(|e| format!("cannot write '{out}': {e}"))?;
        eprintln!("wrote flamegraph SVG to {out}");
    }

    if as_json {
        println!("{}", render::json(&report));
    } else {
        print!("{}", render::text(&report));
    }
    Ok(())
}

fn cmd_diff(args: &[String]) -> Result<(), String> {
    let files: Vec<&str> = args.iter().map(|s| s.as_str()).filter(|s| !s.starts_with("--")).collect();
    if files.len() != 2 {
        return Err("diff needs exactly two trace paths: <baseline> <candidate>".to_string());
    }
    let base = read_trace(files[0])?;
    let cand = read_trace(files[1])?;
    let d = diff(&base, &cand);
    println!("═══ Seshat regression diff ══════════════════════════════════");
    println!("  baseline samples : {}", d.baseline_samples);
    println!("  candidate samples: {}", d.candidate_samples);
    println!("  {}", d.summary);
    if !d.movers.is_empty() {
        println!("  top movers (candidate − baseline, pp):");
        for m in d.movers.iter().take(10) {
            let sign = if m.delta_milli >= 0 { "+" } else { "-" };
            println!(
                "      {sign}{}.{:03}  {}",
                m.delta_milli.abs() / 1000,
                m.delta_milli.abs() % 1000,
                m.label
            );
        }
    }
    println!("═════════════════════════════════════════════════════════════");
    Ok(())
}

fn cmd_variance(args: &[String]) -> Result<(), String> {
    let files: Vec<&str> = args.iter().map(|s| s.as_str()).filter(|s| !s.starts_with("--")).collect();
    if files.len() < 2 {
        return Err("variance needs at least two trace paths (runs of the same program)".to_string());
    }
    let traces: Vec<cjc_seshat::Trace> = files.iter().map(|f| read_trace(f)).collect::<Result<_, _>>()?;
    // 5 percentage points (= 5_000 milli-percent) spread to flag a frame.
    let v = cjc_seshat::variance(&traces, 5_000);
    println!("═══ Seshat variance — {} runs ═══════════════════════════════", v.runs);
    println!("  samples per run : {:?}", v.total_samples);
    if v.variant_frames.is_empty() {
        println!("  no frame's share varied by ≥5.000 pp across runs (stable).");
    } else {
        println!("  frames whose share varied ≥5.000 pp (likely nondeterministic cost):");
        for f in v.variant_frames.iter().take(15) {
            println!(
                "      spread {:>2}.{:03} pp  {:<44}  (suspect: {})",
                f.spread_milli / 1000,
                f.spread_milli % 1000,
                f.label,
                f.suspected_cause
            );
        }
    }
    println!("═════════════════════════════════════════════════════════════");
    Ok(())
}

fn cmd_merge(args: &[String]) -> Result<(), String> {
    use cjc_seshat::{merge, serialize, MergeOptions};

    let mut paths: Vec<&str> = Vec::new();
    let mut under: Option<String> = None;
    let mut out: Option<&str> = None;
    let mut as_json = false;
    let mut svg_out: Option<&str> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--under" => {
                i += 1;
                under = Some(args.get(i).ok_or("--under needs a name")?.clone());
            }
            "--out" => {
                i += 1;
                out = Some(args.get(i).ok_or("--out needs a path")?.as_str());
            }
            "--json" => as_json = true,
            "--svg" => {
                i += 1;
                svg_out = Some(args.get(i).ok_or("--svg needs a path")?.as_str());
            }
            p if !p.starts_with("--") => paths.push(p),
            other => return Err(format!("unknown flag `{other}`")),
        }
        i += 1;
    }
    if paths.len() < 2 {
        return Err(
            "merge needs a host trace and at least one native trace: \
             <host.seshat> <native.seshat> [native2.seshat ...]"
                .to_string(),
        );
    }
    // Fold each native into the host. Each native grafts under `--under` if given,
    // else its own declared host token (`collect::mark_host`), else the heuristic.
    let mut merged = read_trace(paths[0])?;
    for np in &paths[1..] {
        let native = read_trace(np)?;
        merged = merge(&merged, &native, &MergeOptions { under: under.clone() });
    }

    if let Some(o) = out {
        std::fs::write(o, serialize(&merged)).map_err(|e| format!("cannot write '{o}': {e}"))?;
        eprintln!("wrote merged trace to {o}");
    }

    let report = analyze_trace(&merged);
    if let Some(s) = svg_out {
        let svg = render::flamegraph_svg(&report);
        std::fs::write(s, svg).map_err(|e| format!("cannot write '{s}': {e}"))?;
        eprintln!("wrote flamegraph SVG to {s}");
    }
    if as_json {
        println!("{}", render::json(&report));
    } else {
        print!("{}", render::text(&report));
    }
    Ok(())
}

// ─── record-demo ─────────────────────────────────────────────────────────────

#[cfg(feature = "collect-live")]
fn cmd_record_demo(args: &[String]) -> Result<(), String> {
    use cjc_seshat::collect::{mark_copy, zone, CaptureConfig, Recorder};
    use cjc_seshat::{serialize, OwnershipDomain};

    let mut path: Option<&str> = None;
    let mut interval_ms: u64 = 1;
    let mut unwind = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--ms" => {
                i += 1;
                interval_ms = args
                    .get(i)
                    .ok_or("--ms needs a value")?
                    .parse()
                    .map_err(|_| "--ms must be an integer")?;
            }
            // Attribute allocations to real Rust functions via native unwinding
            // (instead of just the open zone). Adds a backtrace per allocation.
            "--unwind" => unwind = true,
            p if !p.starts_with("--") => path = Some(p),
            other => return Err(format!("unknown flag `{other}`")),
        }
        i += 1;
    }
    let path = path.ok_or("record-demo needs an <out.seshat> path")?;

    eprintln!(
        "recording a demo Rust workload (interval {interval_ms} ms, unwind={unwind})…"
    );
    let rec = Recorder::start_with_config(CaptureConfig { interval_ms, alloc_stacks: unwind });

    // ── stage: parse — lots of small allocations ──
    {
        let _z = zone("parse");
        let mut rows: Vec<Vec<u64>> = Vec::new();
        for r in 0..3_000u64 {
            let mut row = Vec::with_capacity(16);
            for c in 0..16u64 {
                row.push(r.wrapping_mul(c));
            }
            rows.push(row);
        }
        std::hint::black_box(&rows);
        spin(40_000);
    }

    // ── stage: compute — one big buffer + a simulated Rust→NumPy handoff ──
    {
        let _z = zone("compute");
        let data: Vec<f64> = (0..400_000).map(|x| x as f64).collect();
        let mut acc = 0.0f64;
        for &x in &data {
            acc += x * x;
        }
        std::hint::black_box(acc);
        mark_copy(OwnershipDomain::RustHeap, OwnershipDomain::NumPy, 8 * 400_000);
        spin(60_000);
    }

    // ── stage: serialize — string building ──
    {
        let _z = zone("serialize");
        let mut s = String::new();
        for i in 0..40_000u64 {
            s.push_str(&i.to_string());
            s.push(',');
        }
        std::hint::black_box(&s);
        spin(40_000);
    }

    let trace = rec.finish();
    let bytes = serialize(&trace);
    std::fs::write(path, &bytes).map_err(|e| format!("cannot write '{path}': {e}"))?;
    eprintln!(
        "recorded {} events ({} bytes) → {path}",
        trace.num_events(),
        bytes.len()
    );

    // immediately analyze so the user sees it end-to-end
    let report = analyze_trace(&trace);
    print!("{}", render::text(&report));
    Ok(())
}

#[cfg(not(feature = "collect-live"))]
fn cmd_record_demo(_args: &[String]) -> Result<(), String> {
    Err("record-demo needs live capture; rebuild with:\n    \
         cargo run -p cjc-seshat --features collect-live --bin seshat -- record-demo out.seshat"
        .to_string())
}

/// A tiny busy-wait so the wall-clock sampler gets a few ticks per stage without
/// pulling in timing dependencies. The amount is advisory only.
#[cfg(feature = "collect-live")]
fn spin(iters: u64) {
    let mut x = 0u64;
    for i in 0..iters {
        x = x.wrapping_add(i).wrapping_mul(2654435761);
        std::hint::black_box(x);
    }
}
