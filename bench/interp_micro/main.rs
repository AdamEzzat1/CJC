//! Tier-0 Interpreter Microbench
//! ==============================
//!
//! Measures per-program execution time across MIR-exec (the primary
//! optimization target) and AST eval (for parity-aware deltas). Each
//! workload is a small `.cjcl` source that exercises one specific hot
//! path in the tree-walking executor.
//!
//! Hot paths measured:
//!
//!   - `arith`   -- tight `Int + Int` arithmetic loop (eval_binary)
//!   - `lookup`  -- many name-based variable accesses (lookup / scope chain)
//!   - `call`    -- builtin-heavy loop (dispatch_call / 4-satellite dispatch)
//!   - `mixed`   -- realistic mix of the above
//!
//! Output:
//!
//!   - stdout: JSONL, one line per (workload, backend) pair, suitable for
//!     CI ingestion or diff against a previous run.
//!   - stderr: human-readable scorecard with min/median ns/run.
//!
//! Invocation:
//!
//!     cargo run -p interp-micro --release > bench_results/interp_micro.jsonl
//!
//! Each workload internally runs a fixed number of iterations (~100k), so
//! a single "run" exercises the hot path many times. The outer loop
//! repeats the run N times and reports min + median (min = best case,
//! median = typical). Lowering cost is included in each run -- amortized
//! across the inner iterations, so deltas remain meaningful.

use std::io::Write;
use std::time::Instant;

use cjc_eval::Interpreter;
use cjc_mir_exec::MirExecutor;

const N_WARMUP: usize = 3;
const N_ITERS: usize = 11;
const SEED: u64 = 42;

/// Source for an arithmetic-heavy hot path: tight loop, Int + Int, no
/// allocations or builtin calls.
const SRC_ARITH: &str = r#"
fn main() -> i64 {
    let mut sum: i64 = 0;
    let mut i: i64 = 0;
    while i < 50000 {
        sum = sum + i;
        i = i + 1;
    }
    sum
}
"#;

/// Source for a lookup-heavy hot path: many named bindings, repeated
/// reads inside a tight loop. Exercises the scope-chain lookup.
const SRC_LOOKUP: &str = r#"
fn main() -> i64 {
    let a: i64 = 1;
    let b: i64 = 2;
    let c: i64 = 3;
    let d: i64 = 4;
    let e: i64 = 5;
    let f: i64 = 6;
    let g: i64 = 7;
    let h: i64 = 8;
    let mut sum: i64 = 0;
    let mut i: i64 = 0;
    while i < 50000 {
        sum = sum + a + b + c + d + e + f + g + h;
        i = i + 1;
    }
    sum
}
"#;

/// Source for a builtin-call-heavy hot path: `abs` called every
/// iteration of a tight loop. Exercises dispatch_call + the 4-satellite
/// dispatch table.
const SRC_CALL: &str = r#"
fn main() -> i64 {
    let mut sum: i64 = 0;
    let mut i: i64 = 0;
    while i < 50000 {
        sum = sum + abs(i - 25000);
        i = i + 1;
    }
    sum
}
"#;

/// Source for a mix of all three: arithmetic, variable reads, calls.
/// Closer to realistic numerical-kernel workload.
const SRC_MIXED: &str = r#"
fn main() -> i64 {
    let scale: i64 = 7;
    let mut sum: i64 = 0;
    let mut i: i64 = 0;
    while i < 30000 {
        let delta: i64 = abs(i - 15000);
        sum = sum + delta * scale;
        i = i + 1;
    }
    sum
}
"#;

/// Lower an AST program through HIR -> MIR (+ escape annotation) once,
/// outside any timing loop. Returns the MIR program ready to execute.
fn lower(program: &cjc_ast::Program) -> cjc_mir::MirProgram {
    let mut ast_lowering = cjc_hir::AstLowering::new();
    let hir = ast_lowering.lower_program(program);
    let mut hir_to_mir = cjc_mir::HirToMir::new();
    let mut mir = hir_to_mir.lower_program(&hir);
    cjc_mir::escape::annotate_program(&mut mir);
    mir
}

/// Run a single iteration of MIR-exec on already-lowered MIR. Builds a
/// fresh executor each call so the inline cache starts cold per
/// iteration -- worst-case for the cache. Returns elapsed ns.
fn time_mir_cold(program: &cjc_ast::Program, mir: &cjc_mir::MirProgram) -> u128 {
    let start = Instant::now();
    let mut executor = MirExecutor::new(SEED);
    executor.scan_ast_imports(program);
    let _ = executor.exec(mir);
    start.elapsed().as_nanos()
}

/// Same as `time_mir_cold` but the executor is built once outside the
/// timing loop -- the cache stays warm across iterations. This is the
/// best-case for the inline cache (and the realistic shape for a
/// long-running session calling the same builtins repeatedly).
fn time_mir_warm(executor: &mut MirExecutor, mir: &cjc_mir::MirProgram) -> u128 {
    let start = Instant::now();
    let _ = executor.exec(mir);
    start.elapsed().as_nanos()
}

/// Run a single iteration via AST eval.
fn time_eval(program: &cjc_ast::Program) -> u128 {
    let start = Instant::now();
    let _ = Interpreter::new(SEED).exec(program);
    start.elapsed().as_nanos()
}

/// Bench many iterations of a closure, return (min_ns, median_ns).
fn bench<F: FnMut() -> u128>(mut run: F) -> (u128, u128) {
    for _ in 0..N_WARMUP {
        let _ = run();
    }
    let mut samples: Vec<u128> = (0..N_ITERS).map(|_| run()).collect();
    samples.sort_unstable();
    (samples[0], samples[N_ITERS / 2])
}

/// Parse + lower once, then bench cold-mir, warm-mir, and eval.
fn run_workload(name: &str, source: &str) {
    let (program, diags) = cjc_parser::parse_source(source);
    if diags.has_errors() {
        eprintln!(
            "{}: parse errors -- skipping\n{}",
            name,
            diags.render_all(source, name)
        );
        return;
    }

    // Lower once outside the timing loops -- amortizes the lowering cost
    // across all iterations, so the bench measures only execution.
    let mir = lower(&program);

    let (mir_cold_min, mir_cold_med) = bench(|| time_mir_cold(&program, &mir));

    // Warm-cache bench: build the executor once and reuse across iters.
    let (mir_warm_min, mir_warm_med) = {
        let mut executor = MirExecutor::new(SEED);
        executor.scan_ast_imports(&program);
        // Discard warmup runs.
        for _ in 0..N_WARMUP {
            let _ = executor.exec(&mir);
        }
        let mut samples: Vec<u128> =
            (0..N_ITERS).map(|_| time_mir_warm(&mut executor, &mir)).collect();
        samples.sort_unstable();
        (samples[0], samples[N_ITERS / 2])
    };

    let (eval_min, eval_med) = bench(|| time_eval(&program));

    // JSONL: one record per (workload, backend, cache-state) triple.
    println!(
        r#"{{"workload":"{name}","backend":"mir_cold","min_ns":{mir_cold_min},"median_ns":{mir_cold_med}}}"#
    );
    println!(
        r#"{{"workload":"{name}","backend":"mir_warm","min_ns":{mir_warm_min},"median_ns":{mir_warm_med}}}"#
    );
    println!(
        r#"{{"workload":"{name}","backend":"eval","min_ns":{eval_min},"median_ns":{eval_med}}}"#
    );

    // Human scorecard line: cold-cache mir / warm-cache mir / eval.
    let cm = (mir_cold_min as f64) / 1.0e6;
    let wm = (mir_warm_min as f64) / 1.0e6;
    let em = (eval_min as f64) / 1.0e6;
    let _cmed = (mir_cold_med as f64) / 1.0e6;
    let _wmed = (mir_warm_med as f64) / 1.0e6;
    eprintln!(
        "  {name:<8}  mir_cold: {cm:>7.2} ms   mir_warm: {wm:>7.2} ms   eval: {em:>7.2} ms"
    );
    let _ = std::io::stderr().flush();
}

fn main() {
    eprintln!("=== Tier-0 Interpreter Microbench (baseline) ===");
    eprintln!(
        "  workload  mir: min/med            eval: min/med            ({N_ITERS} iters, warmup {N_WARMUP})"
    );

    run_workload("arith", SRC_ARITH);
    run_workload("lookup", SRC_LOOKUP);
    run_workload("call", SRC_CALL);
    run_workload("mixed", SRC_MIXED);

    eprintln!("=== done ===");
}
