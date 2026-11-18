//! Postulate Phase 1 — what telemetry and tracking cost (measured, not modelled).
//!
//! `cargo run --release -p postulate-overhead [-- runs]`
//!
//! For each fixture and executor, times `exec` (parse and lowering done once)
//! in four modes, interleaved round-robin so drift hits every mode alike:
//!
//! - `off`      — no tracking, no telemetry (spans are still stamped: always on)
//! - `track`    — `ad.enabled` (groups, off-tape, trained-graph handle)
//! - `tel10`    — telemetry at the default rate, to a file
//! - `tel1`     — telemetry every step, to a file
//! - `mem10`    — telemetry at the default rate, in memory (collection without I/O)
//!
//! Telemetry modes run with tracking off: telemetry does not need it.
//!
//! Reports medians and overhead vs `off`, plus an A/A noise floor: the
//! difference between the medians of `off`'s even and odd runs.
//!
//! `-- <runs> --check <N>` is the Phase 1 overhead gate: exit 1 unless
//! `tel10` costs at most N% on the gated fixture (`mlp_xor`, the one long
//! enough that its A/A floor is under `MAX_AA_PCT`; the others finish in ~5 ms,
//! where the floor is 5–25% and a fixed ~1 ms file-open dominates). If the
//! gated fixture's own floor exceeds `MAX_AA_PCT`, the gate refuses to judge
//! (exit 2) rather than pass on noise; any failure exits 1.

use std::time::Instant;

use cjc_ad::telemetry::{Telemetry, DEFAULT_EVERY};

const GATED: &str = "mlp_xor";
const MAX_AA_PCT: f64 = 5.0;

const FIXTURES: &[(&str, &str, u64)] = &[
    ("mlp_xor", include_str!("../../examples/01_mlp_xor.cjcl"), 42),
    ("linreg", include_str!("../../tests/postulate/linreg.cjcl"), 5),
    ("tanh_fit", include_str!("../../tests/postulate/tanh_fit.cjcl"), 9),
];

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Off,
    Track,
    Tel10,
    Tel1,
    Mem10,
}
const MODES: [Mode; 5] = [Mode::Off, Mode::Track, Mode::Tel10, Mode::Tel1, Mode::Mem10];

fn telemetry(mode: Mode, seed: u64, path: &std::path::Path) -> Option<Telemetry> {
    let _ = std::fs::remove_file(path);
    match mode {
        Mode::Tel10 => Some(Telemetry::to_file(path, seed, DEFAULT_EVERY)),
        Mode::Tel1 => Some(Telemetry::to_file(path, seed, 1)),
        Mode::Mem10 => Some(Telemetry::in_memory(seed, DEFAULT_EVERY)),
        _ => None,
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let runs: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(41);
    let check: Option<f64> = args
        .iter()
        .position(|a| a == "--check")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());
    let mut verdict = 0;
    let path = std::env::temp_dir().join("postulate_overhead_telemetry.jsonl");
    println!("runs per mode: {runs} (interleaved); times are exec-only, ms; release build");
    println!("{:<9} {:<5} {:>8} {:>8} {:>8} {:>8} {:>8}  {:>7} {:>7} {:>7} {:>7}  {:>6}",
        "fixture", "exec", "off", "track", "tel10", "tel1", "mem10", "track%", "tel10%", "tel1%", "mem10%", "A/A%");
    for &(name, src, seed) in FIXTURES {
        let (program, _) = cjc_parser::parse_source(src);
        let hir = cjc_hir::AstLowering::new().lower_program(&program);
        let mut mir = cjc_mir::HirToMir::new().lower_program(&hir);
        cjc_mir::escape::annotate_program(&mut mir);
        for exec in ["eval", "mir"] {
            let mut t: [Vec<f64>; 5] = Default::default();
            for r in 0..runs + 2 {
                for k in 0..5 {
                    let m = MODES[(k + r) % 5];
                    let tel = telemetry(m, seed, &path);
                    let ms = if exec == "eval" {
                        let mut it = cjc_eval::Interpreter::new(seed);
                        it.echo = false;
                        it.ad.enabled = m == Mode::Track;
                        it.ad.telemetry = tel;
                        let s = Instant::now();
                        it.exec(&program).expect("eval");
                        drop(it.ad.telemetry.take()); // the final flush is part of the cost
                        s.elapsed().as_secs_f64() * 1e3
                    } else {
                        let mut ex = cjc_mir_exec::MirExecutor::new(seed);
                        ex.echo = false;
                        ex.ad.enabled = m == Mode::Track;
                        ex.ad.telemetry = tel;
                        ex.scan_ast_imports(&program);
                        let s = Instant::now();
                        ex.exec(&mir).expect("mir");
                        drop(ex.ad.telemetry.take());
                        s.elapsed().as_secs_f64() * 1e3
                    };
                    if r >= 2 {
                        // first two rounds are warm-up
                        t[MODES.iter().position(|x| *x == m).unwrap()].push(ms);
                    }
                }
            }
            let (mut even, mut odd): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
            for (i, x) in t[0].iter().enumerate() {
                if i % 2 == 0 { even.push(*x) } else { odd.push(*x) }
            }
            let aa = (median(&mut even) - median(&mut odd)).abs() / median(&mut t[0].clone()) * 100.0;
            let med: Vec<f64> = t.iter_mut().map(|v| median(v)).collect();
            let pct = |i: usize| (med[i] / med[0] - 1.0) * 100.0;
            println!("{:<9} {:<5} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3}  {:>+6.1}% {:>+6.1}% {:>+6.1}% {:>+6.1}%  {:>5.1}%",
                name, exec, med[0], med[1], med[2], med[3], med[4], pct(1), pct(2), pct(3), pct(4), aa);
            if let (Some(n), true) = (check, name == GATED) {
                if aa > MAX_AA_PCT {
                    println!("  gate: {name}/{exec} A/A {aa:.1}% > {MAX_AA_PCT}% -- too noisy to judge");
                    if verdict == 0 {
                        verdict = 2;
                    }
                } else if pct(2) > n {
                    println!("  gate: FAIL {name}/{exec} tel10 {:+.1}% > {n}%", pct(2));
                    verdict = 1; // a failure outranks "too noisy"
                } else {
                    println!("  gate: ok {name}/{exec} tel10 {:+.1}% <= {n}%", pct(2));
                }
            }
        }
    }
    let _ = std::fs::remove_file(&path);
    std::process::exit(verdict);
}
