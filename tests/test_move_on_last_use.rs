//! ADR-0048: executor-level move on last use.
//!
//! For `x = f(..., x, ...)` where `f` is an owned-argument builtin
//! (`array_push`, `array_pop`, `array_reverse`) and `x` is passed exactly
//! once as a bare argument, both executors move `x` out of its binding
//! before the call so the builtin's `Rc::make_mut` sees refcount 1.
//!
//! Covers: eval ↔ MIR ↔ MIR-opt parity (aliasing, closures, shadowing,
//! slot-resolved locals, params), error-path restoration, determinism,
//! and an O(n) scaling gate for `array_push`.

use std::time::{Duration, Instant};

// ── Helpers ─────────────────────────────────────────────────────

fn parse(src: &str) -> cjc_ast::Program {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(
        diags.diagnostics.is_empty(),
        "parse errors: {:?}",
        diags.diagnostics
    );
    program
}

fn eval_output(src: &str) -> Vec<String> {
    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(&parse(src)).expect("eval failed");
    interp.output.clone()
}

fn mir_output(src: &str) -> Vec<String> {
    let (_, executor) =
        cjc_mir_exec::run_program_with_executor(&parse(src), 42).expect("mir-exec failed");
    executor.output
}

fn mir_opt_output(src: &str) -> Vec<String> {
    let (_, executor) = cjc_mir_exec::run_program_optimized_with_executor(&parse(src), 42)
        .expect("mir-exec (optimized) failed");
    executor.output
}

/// Runs `src` in all three execution modes, asserts they agree, and
/// returns the shared output.
fn parity(src: &str) -> Vec<String> {
    let eval = eval_output(src);
    let mir = mir_output(src);
    let opt = mir_opt_output(src);
    assert_eq!(eval, mir, "eval vs MIR-exec diverged");
    assert_eq!(mir, opt, "MIR-exec vs MIR-exec --mir-opt diverged");
    eval
}

fn eval_error(src: &str) -> String {
    match cjc_eval::Interpreter::new(42).exec(&parse(src)) {
        Ok(_) => panic!("expected eval error"),
        Err(e) => format!("{e}"),
    }
}

fn mir_error(src: &str) -> String {
    match cjc_mir_exec::run_program_with_executor(&parse(src), 42) {
        Ok(_) => panic!("expected mir-exec error"),
        Err(e) => format!("{e}"),
    }
}

// ── Parity: basic shapes ────────────────────────────────────────

#[test]
fn parity_push_loop_top_level() {
    let out = parity(
        r#"
let mut a = [];
let mut i: i64 = 0;
while i < 5 {
    a = array_push(a, i * 10);
    i = i + 1;
}
print(a);
print(array_len(a));
"#,
    );
    assert_eq!(out, vec!["[0, 10, 20, 30, 40]", "5"]);
}

#[test]
fn parity_push_loop_in_function() {
    let out = parity(
        r#"
fn build(n: i64) -> Any {
    let mut a = [];
    let mut i: i64 = 0;
    while i < n {
        a = array_push(a, i);
        i = i + 1;
    }
    a
}
print(build(4));
"#,
    );
    assert_eq!(out, vec!["[0, 1, 2, 3]"]);
}

#[test]
fn parity_push_in_nested_blocks_slot_resolved() {
    // `a` is a slot-resolved local (ADR-0024) written from inside nested
    // for/if blocks; the move must target the outer frame slot.
    let out = parity(
        r#"
fn build() -> Any {
    let mut a = [];
    for i in 0..6 {
        if i % 2 == 0 {
            a = array_push(a, i);
        }
    }
    a
}
print(build());
"#,
    );
    assert_eq!(out, vec!["[0, 2, 4]"]);
}

#[test]
fn parity_shadowed_local_in_inner_block() {
    // The inner `a` shadows the outer one in a distinct slot; moving the
    // inner binding must leave the outer untouched.
    let out = parity(
        r#"
fn f() -> Any {
    let a = [1];
    let mut total: i64 = 0;
    for i in 0..2 {
        let mut a = [100];
        a = array_push(a, i);
        total = total + array_len(a);
    }
    print(total);
    a
}
print(f());
"#,
    );
    assert_eq!(out, vec!["4", "[1]"]);
}

#[test]
fn parity_pop_and_reverse() {
    let out = parity(
        r#"
let mut a = [1, 2, 3];
a = array_reverse(a);
print(a);
let mut t = [7, 8, 9];
t = array_pop(t);
print(t);
"#,
    );
    assert_eq!(out, vec!["[3, 2, 1]", "(9, [7, 8])"]);
}

#[test]
fn parity_bare_arg_in_second_position() {
    // `x` need not be the first argument: here it is the *value* pushed.
    let out = parity(
        r#"
let mut outer = [];
let mut x = [1, 2];
x = array_push(outer, x);
print(x);
print(outer);
"#,
    );
    assert_eq!(out, vec!["[[1, 2]]", "[]"]);
}

// ── Parity: aliasing and value semantics ────────────────────────

#[test]
fn parity_aliased_binding_is_not_mutated() {
    // `b` shares the Rc with `a`; make_mut must copy so `b` is unchanged.
    let out = parity(
        r#"
let mut a = [1, 2, 3];
let b = a;
a = array_push(a, 4);
a = array_reverse(a);
print(a);
print(b);
"#,
    );
    assert_eq!(out, vec!["[4, 3, 2, 1]", "[1, 2, 3]"]);
}

#[test]
fn parity_alias_inside_array_is_not_mutated() {
    let out = parity(
        r#"
let mut a = [1];
let holder = [a, a];
a = array_push(a, 2);
print(a);
print(holder);
"#,
    );
    assert_eq!(out, vec!["[1, 2]", "[[1], [1]]"]);
}

#[test]
fn parity_caller_value_survives_param_mutation() {
    // Inside `grow`, `arr` is a param. Moving it out of the callee's frame
    // must not affect the caller's binding, which still holds a reference.
    let out = parity(
        r#"
fn grow(arr: Any, k: i64) -> Any {
    let mut arr2 = arr;
    arr2 = array_push(arr2, k);
    arr2
}
let base = [1, 2];
let g = grow(base, 3);
print(g);
print(base);
"#,
    );
    assert_eq!(out, vec!["[1, 2, 3]", "[1, 2]"]);
}

#[test]
fn parity_self_push_twice_bare_is_not_moved() {
    // `a` appears twice as a bare argument, so the move does not apply.
    // Semantics: push a snapshot of the old `a` into itself.
    let out = parity(
        r#"
let mut a = [1];
a = array_push(a, a);
print(a);
"#,
    );
    assert_eq!(out, vec!["[1, [1]]"]);
}

#[test]
fn parity_sibling_arg_reads_x_before_move() {
    // `array_len(a)` is evaluated before `a` is moved, so it observes the
    // intact binding.
    let out = parity(
        r#"
let mut a = [5, 6];
a = array_push(a, array_len(a));
a = array_push(a, array_len(a) * 10);
print(a);
"#,
    );
    assert_eq!(out, vec!["[5, 6, 2, 30]"]);
}

// ── Parity: closures and captures ───────────────────────────────

#[test]
fn parity_closure_capture_is_a_snapshot() {
    // The closure captures `a` by value at creation. Moving and growing
    // the binding afterwards must not change what the closure sees.
    let out = parity(
        r#"
fn main() {
    let mut a = [1, 2];
    let f = |k: i64| array_len(a) + k;
    a = array_push(a, 3);
    a = array_push(a, 4);
    print(f(0));
    print(array_len(a));
}
"#,
    );
    assert_eq!(out, vec!["2", "4"]);
}

#[test]
fn parity_move_inside_closure_body() {
    // Inside the lifted closure body, the captured `base` is an ordinary
    // param; moving it must not leak into the closure's env or the outer
    // binding, and repeated calls must see the same snapshot.
    let out = parity(
        r#"
fn main() {
    let base = [0];
    let add = |k: i64| {
        let mut c = base;
        c = array_push(c, k);
        array_len(c)
    };
    print(add(1));
    print(add(2));
    print(base);
}
"#,
    );
    assert_eq!(out, vec!["2", "2", "[0]"]);
}

// ── Parity: shadowing the builtin ───────────────────────────────

#[test]
fn parity_user_fn_shadowing_array_push_is_called() {
    let out = parity(
        r#"
fn array_push(arr: Any, v: i64) -> Any {
    print("user array_push");
    arr
}
let mut a = [1];
a = array_push(a, 2);
print(a);
"#,
    );
    assert_eq!(out, vec!["user array_push", "[1]"]);
}

// ── Error path: the binding is restored, never a placeholder ────

const POP_EMPTY: &str = r#"
let mut a = [];
a = array_pop(a);
"#;

#[test]
fn error_message_parity() {
    let e = eval_error(POP_EMPTY);
    let m = mir_error(POP_EMPTY);
    assert!(e.contains("array_pop: empty array"), "eval: {e}");
    assert!(m.contains("array_pop: empty array"), "mir: {m}");
}

#[test]
fn error_path_arity_parity() {
    let src = "let mut a = [1];\na = array_reverse(a, 2);\n";
    assert!(eval_error(src).contains("array_reverse requires 1 arg"));
    assert!(mir_error(src).contains("array_reverse requires 1 arg"));
}

#[test]
fn error_path_restores_binding_eval() {
    // Reuse the interpreter after the failure (as the REPL does); the
    // binding must still hold the original array, not the Void placeholder.
    let src = "let mut a = [1, 2, 3];\na = array_push(a);\n";
    let mut interp = cjc_eval::Interpreter::new(42);
    assert!(interp.exec(&parse(src)).is_err());
    let a = interp
        .list_bindings()
        .into_iter()
        .find(|(name, _, _)| name == "a")
        .expect("binding `a` must survive the failed call");
    assert_eq!(a.1, "Array");
    assert_eq!(a.2, "[1, 2, 3]");
    interp.exec(&parse("print(a);")).expect("follow-up exec");
    assert_eq!(interp.output.last().unwrap(), "[1, 2, 3]");
}

#[test]
fn error_path_restores_binding_mir() {
    let src = "let mut a = [1, 2, 3];\na = array_push(a);\n";
    let mut exec = cjc_mir_exec::MirExecutor::new(42);
    assert!(exec.exec(&cjc_mir_exec::lower_to_mir(&parse(src))).is_err());
    exec.exec(&cjc_mir_exec::lower_to_mir(&parse("print(a);")))
        .expect("follow-up exec must see the restored binding");
    assert_eq!(exec.output.last().unwrap(), "[1, 2, 3]");
}

#[test]
fn error_path_empty_pop_restores_binding_both() {
    let mut interp = cjc_eval::Interpreter::new(42);
    assert!(interp.exec(&parse(POP_EMPTY)).is_err());
    interp.exec(&parse("print(array_len(a));")).unwrap();

    let mut exec = cjc_mir_exec::MirExecutor::new(42);
    assert!(exec.exec(&cjc_mir_exec::lower_to_mir(&parse(POP_EMPTY))).is_err());
    exec.exec(&cjc_mir_exec::lower_to_mir(&parse("print(array_len(a));")))
        .unwrap();

    assert_eq!(interp.output, vec!["0"]);
    assert_eq!(exec.output, vec!["0"]);
}

#[test]
fn error_in_sibling_arg_leaves_binding_untouched() {
    // The failing sibling argument is evaluated before the move.
    let src = "let mut a = [1];\na = array_push(a, undefined_name);\n";
    let mut interp = cjc_eval::Interpreter::new(42);
    assert!(interp.exec(&parse(src)).is_err());
    interp.exec(&parse("print(a);")).unwrap();

    let mut exec = cjc_mir_exec::MirExecutor::new(42);
    assert!(exec.exec(&cjc_mir_exec::lower_to_mir(&parse(src))).is_err());
    exec.exec(&cjc_mir_exec::lower_to_mir(&parse("print(a);")))
        .unwrap();

    assert_eq!(interp.output, vec!["[1]"]);
    assert_eq!(exec.output, vec!["[1]"]);
}

// ── Determinism ─────────────────────────────────────────────────

const DETERMINISM_SRC: &str = r#"
fn churn(n: i64) -> Any {
    let mut a = [];
    let mut i: i64 = 0;
    while i < n {
        a = array_push(a, (i * 7919) % 101);
        if i % 5 == 4 {
            a = array_pop(a);
            a = match a { (last, rest) => rest };
        }
        if i % 17 == 16 {
            a = array_reverse(a);
        }
        i = i + 1;
    }
    a
}
let r = churn(500);
print(array_len(r));
print(r);
"#;

#[test]
fn determinism_repeat_runs_identical() {
    let first = parity(DETERMINISM_SRC);
    for _ in 0..3 {
        assert_eq!(eval_output(DETERMINISM_SRC), first);
        assert_eq!(mir_output(DETERMINISM_SRC), first);
        assert_eq!(mir_opt_output(DETERMINISM_SRC), first);
    }
    assert_eq!(first[0], "400");
}

#[test]
fn determinism_matches_non_move_reference() {
    // Same computation written so the move never applies (the result goes
    // to a fresh binding, so `a` still holds a reference). Output must be
    // byte-identical to the move path.
    let reference = r#"
fn churn(n: i64) -> Any {
    let mut a = [];
    let mut i: i64 = 0;
    while i < n {
        let b = array_push(a, (i * 7919) % 101);
        a = b;
        if i % 5 == 4 {
            let t = array_pop(a);
            a = match t { (last, rest) => rest };
        }
        if i % 17 == 16 {
            let c = array_reverse(a);
            a = c;
        }
        i = i + 1;
    }
    a
}
let r = churn(500);
print(array_len(r));
print(r);
"#;
    assert_eq!(parity(reference), parity(DETERMINISM_SRC));
}

// ── Scaling: array_push is O(n), not O(n²) ──────────────────────

fn push_loop_src(n: usize, in_function: bool) -> String {
    let body = format!(
        "let mut a = [];\nlet mut i: i64 = 0;\nwhile i < {n} {{\n    a = array_push(a, i);\n    i = i + 1;\n}}\n"
    );
    if in_function {
        format!("fn build() -> i64 {{\n{body}array_len(a)\n}}\nprint(build());\n")
    } else {
        format!("{body}print(array_len(a));\n")
    }
}

/// CPU time consumed by the *current thread*, in arbitrary monotonic units.
///
/// Wall time is useless as a complexity gate on a loaded machine: a
/// thread that is descheduled mid-run absorbs the whole stall (hundreds
/// of ms under a parallel `cargo` build), and a run that just exceeds a
/// scheduling slice is stalled almost every time, so taking the minimum
/// doesn't help. Thread CPU time counts only the work this thread did.
///
/// - Windows: `QueryThreadCycleTime` (CPU cycles, precise). The
///   alternative `GetThreadTimes` only advances at the 15.6 ms clock tick.
/// - Unix: `clock_gettime(CLOCK_THREAD_CPUTIME_ID)` in nanoseconds.
/// - Elsewhere: wall-clock nanoseconds.
fn thread_cpu_units() -> u64 {
    #[cfg(windows)]
    {
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentThread() -> *mut std::ffi::c_void;
            fn QueryThreadCycleTime(thread: *mut std::ffi::c_void, cycles: *mut u64) -> i32;
        }
        let mut cycles = 0u64;
        // SAFETY: GetCurrentThread returns a pseudo-handle valid for the
        // calling thread; `cycles` is a valid out-pointer.
        let ok = unsafe { QueryThreadCycleTime(GetCurrentThread(), &mut cycles) };
        assert!(ok != 0, "QueryThreadCycleTime failed");
        cycles
    }
    #[cfg(all(unix, any(target_os = "linux", target_os = "macos")))]
    {
        #[repr(C)]
        struct Timespec {
            tv_sec: i64,
            tv_nsec: i64,
        }
        extern "C" {
            fn clock_gettime(clock: i32, ts: *mut Timespec) -> i32;
        }
        #[cfg(target_os = "linux")]
        const CLOCK_THREAD_CPUTIME_ID: i32 = 3;
        #[cfg(target_os = "macos")]
        const CLOCK_THREAD_CPUTIME_ID: i32 = 16;
        let mut ts = Timespec { tv_sec: 0, tv_nsec: 0 };
        // SAFETY: `ts` is a valid out-pointer for a 64-bit timespec.
        let rc = unsafe { clock_gettime(CLOCK_THREAD_CPUTIME_ID, &mut ts) };
        assert_eq!(rc, 0, "clock_gettime(CLOCK_THREAD_CPUTIME_ID) failed");
        ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
    }
    #[cfg(not(any(windows, all(unix, any(target_os = "linux", target_os = "macos")))))]
    {
        use std::sync::OnceLock;
        static START: OnceLock<Instant> = OnceLock::new();
        START.get_or_init(Instant::now).elapsed().as_nanos() as u64
    }
}

/// (thread CPU units, wall time) of one call.
fn measure(run: impl FnOnce()) -> (u64, Duration) {
    let (c, t) = (thread_cpu_units(), Instant::now());
    run();
    (thread_cpu_units() - c, t.elapsed())
}

/// Asserts `t(40k) / t(20k)` is linear (~2×). Before ADR-0048 every push
/// copied the array and the measured ratio was 4–5× (quadratic).
///
/// The gate uses thread CPU time (see [`thread_cpu_units`]); wall time is
/// printed alongside for reference. The two sizes are measured
/// interleaved after a warm-up of the large size, and the minimum over the
/// repetitions is compared.
fn assert_linear(label: &str, run: impl Fn(&cjc_ast::Program) -> Vec<String>, in_function: bool) {
    // Don't overlap the scaling tests: concurrent multi-MB allocation churn
    // competes for cache and memory bandwidth, which inflates cycle counts.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let small = parse(&push_loop_src(20_000, in_function));
    let large = parse(&push_loop_src(40_000, in_function));
    assert_eq!(run(&large), vec!["40000"]); // correctness + warm-up
    assert_eq!(run(&small), vec!["20000"]);
    let (mut c_small, mut c_large) = (u64::MAX, u64::MAX);
    let (mut w_small, mut w_large) = (Duration::MAX, Duration::MAX);
    for _ in 0..7 {
        let (c, w) = measure(|| {
            run(&small);
        });
        (c_small, w_small) = (c_small.min(c), w_small.min(w));
        let (c, w) = measure(|| {
            run(&large);
        });
        (c_large, w_large) = (c_large.min(c), w_large.min(w));
    }
    let ratio = c_large as f64 / c_small as f64;
    let wall_ratio = w_large.as_secs_f64() / w_small.as_secs_f64();
    eprintln!(
        "{label}: cpu ratio={ratio:.2} (20k={c_small} 40k={c_large} units) | wall 20k={w_small:?} 40k={w_large:?} ratio={wall_ratio:.2}"
    );
    assert!(
        ratio < 3.0,
        "{label}: array_push loop scales super-linearly: thread-CPU ratio 40k/20k = {ratio:.2} (expected ~2)"
    );
}

fn run_eval(p: &cjc_ast::Program) -> Vec<String> {
    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(p).unwrap();
    interp.output
}

fn run_mir_opt(p: &cjc_ast::Program) -> Vec<String> {
    cjc_mir_exec::run_program_optimized_with_executor(p, 42).unwrap().1.output
}

#[test]
fn scaling_array_push_linear_mir_opt_top_level() {
    assert_linear("mir-opt/top-level", run_mir_opt, false);
}

#[test]
fn scaling_array_push_linear_mir_opt_in_function() {
    assert_linear("mir-opt/in-function", run_mir_opt, true);
}

#[test]
fn scaling_array_push_linear_eval_top_level() {
    assert_linear("eval/top-level", run_eval, false);
}

#[test]
fn scaling_array_push_linear_eval_in_function() {
    assert_linear("eval/in-function", run_eval, true);
}
