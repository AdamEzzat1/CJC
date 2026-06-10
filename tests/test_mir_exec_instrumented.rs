//! Option B PRs 1-4 — parity gate for `run_program_instrumented`.
//!
//! Critical contract: an instrumented run of any program must produce
//! byte-identical output to an uninstrumented run of the same program
//! with the same seed. If instrumentation ever breaks this, every
//! downstream consumer of MIR-exec results is silently corrupted.
//!
//! PRs 2-4 wired real instrumentation sites (function entry,
//! loop-iteration boundary, branch resolution, FP-op counting, IO/GC
//! flags), so instrumented runs now return non-empty event vecs. The
//! output-parity assertions are unchanged — they are the load-bearing
//! contract; the event-count assertions flipped from `== 0` (PR 1) to
//! `> 0` per the original test's own forward note.

use cjc_mir_exec::{run_program_instrumented, run_program_with_executor};
use cjc_parser::parse_source;

fn run_both(src: &str, seed: u64) -> (String, String, usize) {
    let (program, diags) = parse_source(src);
    assert!(!diags.has_errors(), "parse failed: {:?}", diags.diagnostics);

    let (_v_uninst, exec_uninst) =
        run_program_with_executor(&program, seed).expect("uninstrumented run failed");
    let (_v_inst, exec_inst, events) =
        run_program_instrumented(&program, seed).expect("instrumented run failed");

    let out_uninst = exec_uninst.output.join("\n");
    let out_inst = exec_inst.output.join("\n");
    (out_uninst, out_inst, events.len())
}

#[test]
fn instrumented_run_byte_identical_to_uninstrumented_on_arithmetic() {
    let src = r#"
        fn compute(n: i64) -> i64 {
            let mut total: i64 = 0;
            let mut i: i64 = 0;
            while i < n {
                total = total + i * i;
                i = i + 1;
            }
            return total;
        }
        print(compute(10));
        print(compute(20));
        print(compute(30));
    "#;
    let (uninst, inst, event_count) = run_both(src, 42);
    assert_eq!(
        uninst, inst,
        "program output must be byte-identical between instrumented and uninstrumented runs",
    );
    // 10+20+30 loop iterations + 3 function entries — well over 60
    // events under PRs 2-4 instrumentation.
    assert!(
        event_count >= 60,
        "loop-heavy program must emit one event per iteration; got {event_count}",
    );
}

#[test]
fn instrumented_run_byte_identical_with_branches() {
    let src = r#"
        fn classify(x: i64) -> i64 {
            if x < 0 {
                return -1;
            } else {
                if x > 100 {
                    return 1;
                } else {
                    return 0;
                }
            }
        }
        print(classify(-5));
        print(classify(50));
        print(classify(200));
    "#;
    let (uninst, inst, event_count) = run_both(src, 7);
    assert_eq!(uninst, inst);
    // 3 calls to classify, each resolving ≥1 branch → ≥6 events.
    assert!(
        event_count >= 6,
        "branchy program must emit function-entry + branch events; got {event_count}",
    );
}

#[test]
fn instrumented_run_byte_identical_with_float_arithmetic() {
    // Float arithmetic is the most sensitive case — any reordering
    // would produce different bit patterns at the print() output.
    let src = r#"
        fn sum_floats(n: i64) -> f64 {
            let mut acc: f64 = 0.0;
            let mut i: i64 = 0;
            while i < n {
                acc = acc + 0.1;
                i = i + 1;
            }
            return acc;
        }
        print(sum_floats(100));
    "#;
    let (uninst, inst, event_count) = run_both(src, 42);
    assert_eq!(
        uninst, inst,
        "float arithmetic must produce byte-identical output \
         (any reordering by the instrumentation path would surface here)",
    );
    // 100 loop iterations → ≥100 events.
    assert!(event_count >= 100, "got {event_count}");
}

#[test]
fn instrumented_run_repeatable_across_invocations() {
    // Two consecutive instrumented runs of the same program on the
    // same seed must produce identical output AND identical event
    // sequences. The latter is the trace module's contract, exercised
    // here as a regression gate against any future drift.
    let src = r#"
        fn fib(n: i64) -> i64 {
            if n < 2 {
                return n;
            } else {
                return fib(n - 1) + fib(n - 2);
            }
        }
        print(fib(10));
    "#;
    let (program, _) = parse_source(src);

    let (_, _, events1) = run_program_instrumented(&program, 42).unwrap();
    let (_, _, events2) = run_program_instrumented(&program, 42).unwrap();
    let (_, _, events3) = run_program_instrumented(&program, 42).unwrap();

    assert_eq!(
        events1, events2,
        "two consecutive instrumented runs must produce identical event vec",
    );
    assert_eq!(events2, events3, "three runs, same story");
}

#[test]
fn uninstrumented_runs_do_not_leak_state_into_subsequent_instrumented() {
    // Defensive: if a normal `run_program_with_executor` accidentally
    // touched the TLS collector, the next instrumented run would see
    // stale events. Verify the uninstrumented path is hermetic.
    let src = r#"
        fn main_fn() -> i64 {
            return 42;
        }
        print(main_fn());
    "#;
    let (program, _) = parse_source(src);

    // Run uninstrumented twice (which should not touch the collector).
    let _ = run_program_with_executor(&program, 1).unwrap();
    let _ = run_program_with_executor(&program, 2).unwrap();

    // Now run instrumented — the event vec must reflect ONLY this
    // run. A fresh instrumented run of the same program gives the
    // reference count; leakage from the uninstrumented runs above
    // would inflate the first count.
    let (_, _, events) = run_program_instrumented(&program, 3).unwrap();
    let (_, _, fresh) = run_program_instrumented(&program, 3).unwrap();
    assert_eq!(
        events.len(),
        fresh.len(),
        "uninstrumented runs must not leak state into the instrumented collector",
    );
    assert!(
        !events.is_empty(),
        "function-call program must emit at least the entry event",
    );
}
