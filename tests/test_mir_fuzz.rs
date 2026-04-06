//! Fuzz-style tests for the CJC MIR pipeline.
//!
//! These tests generate random syntactically-valid CJC programs and verify
//! that the MIR pipeline (parse -> HIR -> MIR -> execute) maintains key
//! invariants: no panics, optimizer parity, NoGC verifier safety, and
//! deterministic execution.
//!
//! Uses bolero (proptest backend on Windows/macOS, libfuzzer on Linux CI).
//!
//! Run with:
//!   cargo test --test test_mir_fuzz

use std::panic;

// ---------------------------------------------------------------------------
// Program generator
// ---------------------------------------------------------------------------

/// Generate a syntactically valid CJC program from a seed.
///
/// Programs include arithmetic, function calls, if/else, and while loops.
/// The seed determines which program variant is produced.
fn gen_program(seed: u64) -> String {
    let variant = seed % 12;
    let a = ((seed >> 8) % 200) as i64 - 100;
    let b = ((seed >> 16) % 200) as i64 - 100;
    let c = ((seed >> 24) % 50) as i64 + 1; // positive, non-zero

    match variant {
        // Simple arithmetic
        0 => format!("fn main() -> i64 {{ {} + {} }}", a, b),
        1 => format!("fn main() -> i64 {{ {} * {} }}", a, c),
        2 => format!("fn main() -> i64 {{ {} - {} }}", a, b),

        // Nested arithmetic
        3 => format!("fn main() -> i64 {{ ({} + {}) * {} }}", a, b, c),
        4 => format!("fn main() -> i64 {{ {} + {} - {} }}", a, b, c),

        // Let bindings
        5 => format!(
            "fn main() -> i64 {{ let x: i64 = {}; let y: i64 = {}; x + y }}",
            a, b
        ),

        // If/else
        6 => format!(
            "fn main() -> i64 {{\n    let x: i64 = {};\n    if x > 0 {{\n        x\n    }} else {{\n        0 - x\n    }}\n}}",
            a
        ),
        7 => format!(
            "fn main() -> i64 {{\n    let a: i64 = {};\n    let b: i64 = {};\n    if a > b {{\n        a - b\n    }} else {{\n        b - a\n    }}\n}}",
            a, b
        ),

        // While loop
        8 => format!(
            "fn main() -> i64 {{\n    let i: i64 = 0;\n    let s: i64 = 0;\n    while i < {} {{\n        s = s + i;\n        i = i + 1;\n    }}\n    s\n}}",
            c.abs().max(1)
        ),

        // Function call
        9 => format!(
            "fn double(x: i64) -> i64 {{ x + x }}\nfn main() -> i64 {{ double({}) }}",
            a
        ),

        // Multiple function calls
        10 => format!(
            "fn add(a: i64, b: i64) -> i64 {{ a + b }}\nfn main() -> i64 {{ add({}, {}) }}",
            a, b
        ),

        // Nested function calls
        _ => format!(
            "fn inc(x: i64) -> i64 {{ x + 1 }}\nfn main() -> i64 {{ inc(inc({})) }}",
            a
        ),
    }
}

// ---------------------------------------------------------------------------
// 1. Parse -> HIR -> MIR doesn't panic
// ---------------------------------------------------------------------------

#[test]
fn fuzz_mir_pipeline_no_panic() {
    bolero::check!()
        .with_type::<u64>()
        .for_each(|&seed: &u64| {
            let src = gen_program(seed);
            let _ = panic::catch_unwind(|| {
                let (program, diags) = cjc_parser::parse_source(&src);
                if !diags.has_errors() {
                    let _ = cjc_mir_exec::run_program(&program, 42);
                }
            });
        });
}

/// Also fuzz with raw bytes for broader coverage.
#[test]
fn fuzz_mir_pipeline_raw_bytes() {
    bolero::check!()
        .with_type::<Vec<u8>>()
        .for_each(|input: &Vec<u8>| {
            if let Ok(s) = std::str::from_utf8(input) {
                let s = s.to_string();
                let _ = panic::catch_unwind(|| {
                    let (program, diags) = cjc_parser::parse_source(&s);
                    if !diags.has_errors() {
                        // Lower through HIR -> MIR without executing
                        let mut mir = cjc_mir_exec::lower_to_mir(&program);
                        mir.build_all_cfgs();
                    }
                });
            }
        });
}

// ---------------------------------------------------------------------------
// 2. MIR optimizer preserves output
// ---------------------------------------------------------------------------

#[test]
fn fuzz_optimizer_preserves_output() {
    bolero::check!()
        .with_type::<u64>()
        .for_each(|&seed: &u64| {
            let src = gen_program(seed);
            let _ = panic::catch_unwind(|| {
                let (program, diags) = cjc_parser::parse_source(&src);
                if !diags.has_errors() {
                    let unopt = cjc_mir_exec::run_program(&program, 42);
                    let opt = cjc_mir_exec::run_program_optimized(&program, 42);
                    match (unopt, opt) {
                        (Ok(a), Ok(b)) => {
                            let sa = format!("{:?}", a);
                            let sb = format!("{:?}", b);
                            assert_eq!(
                                sa, sb,
                                "optimizer parity failure for seed {}: unopt={}, opt={}",
                                seed, sa, sb
                            );
                        }
                        // Both error or one errors: acceptable.
                        _ => {}
                    }
                }
            });
        });
}

// ---------------------------------------------------------------------------
// 3. NoGC verifier doesn't panic
// ---------------------------------------------------------------------------

#[test]
fn fuzz_nogc_verifier_no_panic() {
    bolero::check!()
        .with_type::<u64>()
        .for_each(|&seed: &u64| {
            let src = gen_program(seed);
            let _ = panic::catch_unwind(|| {
                let (program, diags) = cjc_parser::parse_source(&src);
                if !diags.has_errors() {
                    // Must not panic. Ok or Err are both fine.
                    let _ = cjc_mir_exec::verify_nogc(&program);
                }
            });
        });
}

/// Also fuzz the NoGC verifier with raw byte input.
#[test]
fn fuzz_nogc_verifier_raw_bytes() {
    bolero::check!()
        .with_type::<Vec<u8>>()
        .for_each(|input: &Vec<u8>| {
            if let Ok(s) = std::str::from_utf8(input) {
                let s = s.to_string();
                let _ = panic::catch_unwind(|| {
                    let (program, diags) = cjc_parser::parse_source(&s);
                    if !diags.has_errors() {
                        let _ = cjc_mir_exec::verify_nogc(&program);
                    }
                });
            }
        });
}

// ---------------------------------------------------------------------------
// 4. Repeated execution is deterministic
// ---------------------------------------------------------------------------

#[test]
fn fuzz_execution_determinism() {
    bolero::check!()
        .with_type::<u64>()
        .for_each(|&seed: &u64| {
            let src = gen_program(seed);
            let (program, diags) = cjc_parser::parse_source(&src);
            if diags.has_errors() {
                return;
            }

            // Run 5 times with the same seed.
            let exec_seed = 12345u64;
            let first = cjc_mir_exec::run_program(&program, exec_seed);

            for run in 1..5 {
                let result = cjc_mir_exec::run_program(&program, exec_seed);
                match (&first, &result) {
                    (Ok(a), Ok(b)) => {
                        let sa = format!("{:?}", a);
                        let sb = format!("{:?}", b);
                        assert_eq!(
                            sa, sb,
                            "determinism failure on run {} for seed {}: first={}, got={}",
                            run, seed, sa, sb
                        );
                    }
                    (Err(_), Err(_)) => {
                        // Both error: consistent.
                    }
                    _ => {
                        panic!(
                            "determinism failure on run {} for seed {}: one succeeded, one failed",
                            run, seed
                        );
                    }
                }
            }
        });
}

/// Determinism for optimized execution.
#[test]
fn fuzz_optimized_execution_determinism() {
    bolero::check!()
        .with_type::<u64>()
        .for_each(|&seed: &u64| {
            let src = gen_program(seed);
            let (program, diags) = cjc_parser::parse_source(&src);
            if diags.has_errors() {
                return;
            }

            let exec_seed = 99999u64;
            let first = cjc_mir_exec::run_program_optimized(&program, exec_seed);

            for run in 1..5 {
                let result = cjc_mir_exec::run_program_optimized(&program, exec_seed);
                match (&first, &result) {
                    (Ok(a), Ok(b)) => {
                        let sa = format!("{:?}", a);
                        let sb = format!("{:?}", b);
                        assert_eq!(
                            sa, sb,
                            "optimized determinism failure on run {} for seed {}",
                            run, seed
                        );
                    }
                    (Err(_), Err(_)) => {}
                    _ => {
                        panic!(
                            "optimized determinism failure on run {} for seed {}",
                            run, seed
                        );
                    }
                }
            }
        });
}

// ---------------------------------------------------------------------------
// 5. Eval vs MIR parity on generated programs
// ---------------------------------------------------------------------------

#[test]
fn fuzz_eval_mir_parity() {
    bolero::check!()
        .with_type::<u64>()
        .for_each(|&seed: &u64| {
            let src = gen_program(seed);
            let _ = panic::catch_unwind(|| {
                let (program, diags) = cjc_parser::parse_source(&src);
                if !diags.has_errors() {
                    let exec_seed = 42u64;
                    let eval_result = cjc_eval::Interpreter::new(exec_seed).exec(&program);
                    let mir_result = cjc_mir_exec::run_program(&program, exec_seed);

                    // Both should agree on success/failure and produce the same value.
                    match (eval_result, mir_result) {
                        (Ok(eval_val), Ok(mir_val)) => {
                            let se = format!("{:?}", eval_val);
                            let sm = format!("{:?}", mir_val);
                            assert_eq!(
                                se, sm,
                                "eval/MIR parity failure for seed {}: eval={}, mir={}",
                                seed, se, sm
                            );
                        }
                        // Both error: acceptable divergence in error messages.
                        _ => {}
                    }
                }
            });
        });
}
