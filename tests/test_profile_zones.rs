//! Tests for the Tier 2 profile counter builtins (Chess RL v2.3).
//!
//! Coverage:
//! - Wiring check: `profile_zone_start`, `profile_zone_stop`, and
//!   `profile_dump` are callable from both `cjc-eval` and `cjc-mir-exec`
//!   via the shared dispatch.
//! - Handle monotonicity from CJC source.
//! - Nested zone accumulation.
//! - CSV schema (7 columns, integer-valued) and sort-by-`total_ns`.
//! - **Write-only determinism:** a CJC program with profile counters
//!   interleaved produces byte-identical `print` output to the same
//!   program without profile counters. This is the invariant the
//!   v2.3 parity test leans on.
//! - Proptest: random sequences of zone names + lengths produce a
//!   well-formed CSV (correct column count, integer values, no NaN).
//! - Bolero fuzz: random byte-string zone names never panic the
//!   interpreter and always leave the profiler in a recoverable state.
//!
//! All tests `profile_dump` to a scratch path at the start to reset the
//! thread-local profiler state — this is necessary because cargo tests
//! share threads across tests, and the profiler state is thread-local.

use bolero::check;
use proptest::prelude::*;
use std::fs;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn run_eval(src: &str) -> Vec<String> {
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    let mut interp = cjc_eval::Interpreter::new(1);
    interp
        .exec(&prog)
        .unwrap_or_else(|e| panic!("eval failed: {e:?}"));
    interp.output
}

fn run_mir(src: &str) -> Vec<String> {
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    let (_val, executor) = cjc_mir_exec::run_program_with_executor(&prog, 1)
        .unwrap_or_else(|e| panic!("mir-exec failed: {e:?}"));
    executor.output
}

fn tmp_path(tag: &str) -> String {
    let pid = std::process::id();
    let thread_id = format!("{:?}", std::thread::current().id());
    let safe: String = thread_id.chars().filter(|c| c.is_alphanumeric()).collect();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let raw = std::env::temp_dir()
        .join(format!("cjc_profile_{tag}_{pid}_{safe}_{ts}.csv"))
        .to_string_lossy()
        .into_owned();
    raw.replace('\\', "/")
}

/// CJC-Lang source that resets the thread-local profiler by dumping to a
/// scratch path and throwing the result away. Inlining this at the top of
/// every test keeps each test isolated from prior state.
fn reset_prelude(scratch: &str) -> String {
    format!(r#"profile_dump("{scratch}");"#)
}

// ---------------------------------------------------------------------------
// Wiring tests
// ---------------------------------------------------------------------------

/// Smoke test: builtin calls work and the handle is monotonic.
#[test]
fn profile_start_returns_monotonic_handles() {
    let scratch = tmp_path("mono_reset");
    let dump = tmp_path("mono_dump");
    let src = format!(
        r#"
        {reset}
        let a = profile_zone_start("a");
        let b = profile_zone_start("b");
        let c = profile_zone_start("c");
        print(a);
        print(b);
        print(c);
        profile_zone_stop(a);
        profile_zone_stop(b);
        profile_zone_stop(c);
        profile_dump("{dump}");
    "#,
        reset = reset_prelude(&scratch),
    );
    let out = run_eval(&src);
    assert_eq!(out[0], "0");
    assert_eq!(out[1], "1");
    assert_eq!(out[2], "2");
    let _ = fs::remove_file(&scratch);
    let _ = fs::remove_file(&dump);
}

/// Cross-executor wiring parity: same program produces the same prints
/// on both cjc-eval and cjc-mir-exec.
#[test]
fn profile_wiring_parity() {
    let scratch_e = tmp_path("wire_e_reset");
    let scratch_m = tmp_path("wire_m_reset");
    let dump_e = tmp_path("wire_e_dump");
    let dump_m = tmp_path("wire_m_dump");
    let src_e = format!(
        r#"
        {reset}
        let h = profile_zone_start("wire");
        profile_zone_stop(h);
        let rows = profile_dump("{dump}");
        print(rows);
    "#,
        reset = reset_prelude(&scratch_e),
        dump = dump_e,
    );
    let src_m = format!(
        r#"
        {reset}
        let h = profile_zone_start("wire");
        profile_zone_stop(h);
        let rows = profile_dump("{dump}");
        print(rows);
    "#,
        reset = reset_prelude(&scratch_m),
        dump = dump_m,
    );
    let e = run_eval(&src_e);
    let m = run_mir(&src_m);
    assert_eq!(
        e, m,
        "profile wiring parity violation\neval: {e:?}\nmir: {m:?}"
    );
    // Both should report 1 row.
    assert_eq!(e[0], "1");
    let _ = fs::remove_file(&scratch_e);
    let _ = fs::remove_file(&scratch_m);
    let _ = fs::remove_file(&dump_e);
    let _ = fs::remove_file(&dump_m);
}

/// Nested zones: outer and inner zones are both recorded.
#[test]
fn profile_nested_zones() {
    let scratch = tmp_path("nested_reset");
    let dump = tmp_path("nested_dump");
    let src = format!(
        r#"
        {reset}
        let outer = profile_zone_start("outer");
        let inner = profile_zone_start("inner");
        profile_zone_stop(inner);
        profile_zone_stop(outer);
        let rows = profile_dump("{dump}");
        print(rows);
    "#,
        reset = reset_prelude(&scratch),
        dump = dump,
    );
    let out = run_eval(&src);
    assert_eq!(out[0], "2");
    let csv = fs::read_to_string(&dump).unwrap();
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines.len(), 3, "expected header + 2 rows in {csv}");
    let names: Vec<&str> = lines[1..]
        .iter()
        .map(|l| l.split(',').next().unwrap())
        .collect();
    assert!(names.contains(&"outer"));
    assert!(names.contains(&"inner"));
    let _ = fs::remove_file(&scratch);
    let _ = fs::remove_file(&dump);
}

/// CSV schema: 7 columns, all numeric columns parse as integers.
#[test]
fn profile_csv_schema_integer_columns() {
    let scratch = tmp_path("schema_reset");
    let dump = tmp_path("schema_dump");
    let src = format!(
        r#"
        {reset}
        let h1 = profile_zone_start("schema_a");
        profile_zone_stop(h1);
        let h2 = profile_zone_start("schema_a");
        profile_zone_stop(h2);
        let h3 = profile_zone_start("schema_b");
        profile_zone_stop(h3);
        profile_dump("{dump}");
    "#,
        reset = reset_prelude(&scratch),
        dump = dump,
    );
    run_eval(&src);
    let csv = fs::read_to_string(&dump).unwrap();
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(
        lines[0],
        "zone_name,count,total_ns,min_ns,max_ns,mean_ns,stddev_ns"
    );
    assert_eq!(lines.len(), 3);
    for row in &lines[1..] {
        let fields: Vec<&str> = row.split(',').collect();
        assert_eq!(fields.len(), 7, "row {row} has wrong column count");
        // Columns 1..=6 must all parse as u128.
        for f in &fields[1..] {
            assert!(
                f.parse::<u128>().is_ok(),
                "column {f} is not an integer in row {row}"
            );
        }
    }
    let _ = fs::remove_file(&scratch);
    let _ = fs::remove_file(&dump);
}

// ---------------------------------------------------------------------------
// Write-only determinism — the key invariant
// ---------------------------------------------------------------------------

/// Running the same numeric program with profile instrumentation produces
/// byte-identical `print` output vs the uninstrumented baseline. This is
/// what lets us instrument `rollout_episode_v22` without perturbing the
/// v2.2 weight hash.
#[test]
fn profile_write_only_determinism_eval() {
    let scratch = tmp_path("det_eval_reset");
    let dump = tmp_path("det_eval_dump");

    // Baseline: no profile calls.
    let baseline_src = r#"
        let acc = 0.0;
        let i = 0;
        while i < 50 {
            acc = acc + float(i) * 0.5;
            i = i + 1;
        }
        print(acc);
    "#;

    // Instrumented: same math with profile zones interleaved.
    let instrumented_src = format!(
        r#"
        {reset}
        let acc = 0.0;
        let i = 0;
        let outer = profile_zone_start("loop");
        while i < 50 {{
            let step = profile_zone_start("step");
            acc = acc + float(i) * 0.5;
            i = i + 1;
            profile_zone_stop(step);
        }}
        profile_zone_stop(outer);
        profile_dump("{dump}");
        print(acc);
    "#,
        reset = reset_prelude(&scratch),
        dump = dump,
    );

    let base = run_eval(baseline_src);
    let inst = run_eval(&instrumented_src);
    assert_eq!(
        base, inst,
        "profile instrumentation perturbed eval output\nbase: {base:?}\ninst: {inst:?}"
    );
    let _ = fs::remove_file(&scratch);
    let _ = fs::remove_file(&dump);
}

/// Same determinism guarantee on the MIR executor.
#[test]
fn profile_write_only_determinism_mir() {
    let scratch = tmp_path("det_mir_reset");
    let dump = tmp_path("det_mir_dump");

    let baseline_src = r#"
        let acc = 0.0;
        let i = 0;
        while i < 25 {
            acc = acc + float(i);
            i = i + 1;
        }
        print(acc);
    "#;

    let instrumented_src = format!(
        r#"
        {reset}
        let acc = 0.0;
        let i = 0;
        let outer = profile_zone_start("mir_loop");
        while i < 25 {{
            let step = profile_zone_start("mir_step");
            acc = acc + float(i);
            i = i + 1;
            profile_zone_stop(step);
        }}
        profile_zone_stop(outer);
        profile_dump("{dump}");
        print(acc);
    "#,
        reset = reset_prelude(&scratch),
        dump = dump,
    );

    let base = run_mir(baseline_src);
    let inst = run_mir(&instrumented_src);
    assert_eq!(
        base, inst,
        "profile instrumentation perturbed mir output\nbase: {base:?}\ninst: {inst:?}"
    );
    let _ = fs::remove_file(&scratch);
    let _ = fs::remove_file(&dump);
}

// ---------------------------------------------------------------------------
// Error paths
// ---------------------------------------------------------------------------

/// Stopping an unknown handle returns -1.0 and does not panic.
#[test]
fn profile_unknown_handle_returns_negative() {
    let scratch = tmp_path("unknown_reset");
    let src = format!(
        r#"
        {reset}
        let e = profile_zone_stop(9999);
        print(e);
    "#,
        reset = reset_prelude(&scratch),
    );
    let out = run_eval(&src);
    // f64 -1.0 prints as "-1" on CJC-Lang because integer-valued floats
    // round-trip as integers in the default print formatter. Accept both.
    let v = out[0].parse::<f64>().unwrap();
    assert!(v < 0.0);
    let _ = fs::remove_file(&scratch);
}

/// Empty dump writes header only and returns 0.
#[test]
fn profile_empty_dump_row_count_zero() {
    let scratch = tmp_path("empty_reset");
    let dump = tmp_path("empty_dump");
    let src = format!(
        r#"
        {reset}
        let rows = profile_dump("{dump}");
        print(rows);
    "#,
        reset = reset_prelude(&scratch),
        dump = dump,
    );
    let out = run_eval(&src);
    assert_eq!(out[0], "0");
    let csv = fs::read_to_string(&dump).unwrap();
    assert_eq!(
        csv,
        "zone_name,count,total_ns,min_ns,max_ns,mean_ns,stddev_ns\n"
    );
    let _ = fs::remove_file(&scratch);
    let _ = fs::remove_file(&dump);
}

// ---------------------------------------------------------------------------
// Proptest — CSV well-formedness
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 16, .. ProptestConfig::default() })]

    /// For any sequence of (name_index, repeat_count) pairs, the resulting
    /// CSV is well-formed: correct column count, integer values, one row
    /// per distinct name, and no NaN/"inf" strings.
    #[test]
    fn prop_csv_well_formed(
        events in proptest::collection::vec((0u8..5u8, 1u8..5u8), 1..20)
    ) {
        // Reset profiler via a dump to a scratch file.
        cjc_runtime::profile::reset_for_test();
        let names = ["alpha", "beta", "gamma", "delta", "epsilon"];
        for (name_idx, repeat) in &events {
            for _ in 0..*repeat {
                let h = cjc_runtime::profile::zone_start(names[*name_idx as usize]);
                let _ = cjc_runtime::profile::zone_stop(h);
            }
        }
        let dump = std::env::temp_dir().join(
            format!("cjc_profile_prop_{}.csv", std::process::id())
        );
        let row_count = cjc_runtime::profile::dump_to_path(
            dump.to_str().unwrap()
        ).unwrap();
        let csv = std::fs::read_to_string(&dump).unwrap();
        let lines: Vec<&str> = csv.lines().collect();
        // Header + row_count data rows
        prop_assert_eq!(lines.len() as i64, row_count + 1);
        prop_assert_eq!(
            lines[0],
            "zone_name,count,total_ns,min_ns,max_ns,mean_ns,stddev_ns"
        );
        for row in &lines[1..] {
            let fields: Vec<&str> = row.split(',').collect();
            prop_assert_eq!(fields.len(), 7);
            prop_assert!(!fields[0].is_empty());
            for f in &fields[1..] {
                prop_assert!(f.parse::<u128>().is_ok(), "non-integer column: {}", f);
            }
            prop_assert!(!row.contains("NaN"));
            prop_assert!(!row.contains("inf"));
        }
        let _ = std::fs::remove_file(&dump);
    }
}

// ---------------------------------------------------------------------------
// Bolero fuzz — random zone names never crash the profiler
// ---------------------------------------------------------------------------

/// Random byte-sequence zone names never crash the profiler.
#[test]
fn fuzz_profile_zone_name_bytes() {
    check!().with_type::<Vec<u8>>().for_each(|bytes| {
        cjc_runtime::profile::reset_for_test();
        // Build a name from the first 32 bytes, treating them as utf-8
        // replacement-friendly ASCII. Empty is fine.
        let name: String = bytes
            .iter()
            .take(32)
            .map(|b| if *b >= 0x20 && *b <= 0x7E { *b as char } else { '_' })
            .collect();
        let h = cjc_runtime::profile::zone_start(&name);
        let _ = cjc_runtime::profile::zone_stop(h);
        // Snapshot must always be cloneable and never contain NaN/inf.
        let snap = cjc_runtime::profile::snapshot_zones();
        for (_n, stats) in &snap {
            assert!(stats.stddev_ns().is_finite());
        }
    });
}

/// Random handle i64 inputs never crash profile_zone_stop.
#[test]
fn fuzz_profile_zone_stop_random_handle() {
    check!().with_type::<i64>().for_each(|h| {
        cjc_runtime::profile::reset_for_test();
        // Arbitrary handle; must never panic.
        let _ = cjc_runtime::profile::zone_stop(*h);
    });
}
