//! AST eval ↔ MIR exec parity tests for the `seshat_*` profiler builtins.
//!
//! Each test runs a tiny `.cjcl` snippet through both backends and asserts
//! byte-identical printed output — the canonical gate that the satellite
//! dispatch (`cjc_seshat::dispatch_seshat`) routes identically from both
//! executors, exactly as `grad_graph_*` / `abng_*` / `locke_*` are held.
//!
//! The per-thread trace sink is reset before every backend invocation so the
//! zone-handle space and event count are identical across eval and MIR.

#![allow(clippy::needless_raw_string_hashes)]

use cjc_seshat::dispatch::reset as reset_sink;

#[derive(Clone, Copy, Debug)]
enum Backend {
    Eval,
    Mir,
}

fn run(backend: Backend, body: &str, seed: u64) -> Vec<String> {
    let src = format!("fn main() {{\n{body}\n}}\n");
    let (program, diags) = cjc_parser::parse_source(&src);
    assert!(
        !diags.has_errors(),
        "parse errors:\n{:#?}\nsource:\n{src}",
        diags.diagnostics,
    );
    reset_sink();
    match backend {
        Backend::Eval => {
            let mut interp = cjc_eval::Interpreter::new(seed);
            interp
                .exec(&program)
                .unwrap_or_else(|e| panic!("eval failed for snippet:\n{src}\nerror: {e:?}"));
            interp.output
        }
        Backend::Mir => {
            let (_v, exec) = cjc_mir_exec::run_program_with_executor(&program, seed)
                .unwrap_or_else(|e| panic!("MIR-exec failed for snippet:\n{src}\nerror: {e:?}"));
            exec.output
        }
    }
}

fn assert_parity(label: &str, body: &str) {
    let eval_out = run(Backend::Eval, body, 42);
    let mir_out = run(Backend::Mir, body, 42);
    assert_eq!(
        eval_out, mir_out,
        "[{label}] AST↔MIR parity violation\n  eval: {eval_out:?}\n  mir : {mir_out:?}",
    );
}

#[test]
fn parity_zone_handles_sequential() {
    assert_parity(
        "seshat_zone_start handle sequence",
        r#"
        seshat_reset();
        let a: i64 = seshat_zone_start("parse");
        let b: i64 = seshat_zone_start("compute");
        print(a);
        print(b);
        seshat_zone_stop(b);
        seshat_zone_stop(a);
        "#,
    );
}

#[test]
fn parity_event_count_after_markers() {
    assert_parity(
        "seshat_event_count after a marker sequence",
        r#"
        seshat_reset();
        let z: i64 = seshat_zone_start("compute");
        seshat_alloc_tag("rust", 4096);
        seshat_mark_copy("rust", "numpy", 4096);
        seshat_mark_boundary("pyo3::call");
        seshat_zone_stop(z);
        print(seshat_event_count());
        "#,
    );
}

#[test]
fn parity_alloc_and_copy_markers() {
    assert_parity(
        "alloc + copy markers return 0 deterministically",
        r#"
        seshat_reset();
        print(seshat_alloc_tag("numpy", 1024));
        print(seshat_mark_copy("numpy", "arrow", 1024));
        print(seshat_mark_boundary("ffi"));
        print(seshat_event_count());
        "#,
    );
}

#[test]
fn parity_reset_clears_sink() {
    assert_parity(
        "seshat_reset returns count to zero",
        r#"
        seshat_reset();
        seshat_alloc_tag("rust", 8);
        seshat_alloc_tag("rust", 8);
        print(seshat_event_count());
        seshat_reset();
        print(seshat_event_count());
        "#,
    );
}
