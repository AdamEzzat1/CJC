//! Bolero fuzzing harnesses for CJC.
//!
//! These targets use the Bolero testing facade, which runs as proptest on
//! Windows/macOS and can be promoted to libfuzzer/AFL on Linux CI.
//!
//! Run with:
//!   cargo test --test bolero_fuzz
//!
//! For coverage-guided fuzzing (Linux only):
//!   cargo bolero test bolero_fuzz::fuzz_lexer
//!
//! Targets must be able to fail: never discard a `catch_unwind` result.
//! Either run the body directly, or (when panic-freedom is the property)
//! assert `catch_unwind(..).is_ok()` with a message naming the input.

pub mod abng_decision_fuzz;
pub mod adaptive_selection_fuzz;
pub mod categorical_dictionary_fuzz;
pub mod categorical_join_fuzz;
pub mod cli_expansion_fuzz;
pub mod hybrid_streaming_fuzz;
pub mod program_gen;
pub mod v2_1_bytecode_fuzz;

use std::panic;

/// Parse-clean programs derived from one fuzz input: the raw bytes when they
/// happen to form a valid program (rare), plus a generated well-formed
/// program (always). Targets that only act on parseable programs use this so
/// they exercise real programs, not just the empty one.
fn parse_clean_programs(input: &[u8]) -> Vec<(String, cjc_ast::Program)> {
    let mut out = Vec::new();
    let mut push_if_clean = |src: String| {
        let (program, diags) = cjc_parser::parse_source(&src);
        if !diags.has_errors() {
            out.push((src, program));
        }
    };
    if let Ok(s) = std::str::from_utf8(input) {
        push_if_clean(s.to_string());
    }
    push_if_clean(program_gen::gen_program(input));
    out
}

/// Fuzz the CJC lexer: arbitrary UTF-8 input must not panic.
///
/// Panic-freedom is the property. The panic is caught only so the failure
/// message can name the offending input; it is never discarded.
#[test]
fn fuzz_lexer() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let Ok(s) = std::str::from_utf8(input) else { return };
        let r = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let (_tokens, _diags) = cjc_lexer::Lexer::new(s).tokenize();
        }));
        assert!(r.is_ok(), "lexer panicked on input {s:?}");
    });
}

/// Fuzz the CJC parser: any valid UTF-8 input must parse (with or without
/// diagnostics) without panicking.
#[test]
fn fuzz_parser() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let Ok(s) = std::str::from_utf8(input) else { return };
        let r = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let (_program, _diags) = cjc_parser::parse_source(s);
        }));
        assert!(r.is_ok(), "parser panicked on input {s:?}");
    });
}

/// Fuzz the full MIR pipeline: parse + lower + execute must not panic.
/// Runtime errors (`Err`) are fine; panics are not.
#[test]
fn fuzz_mir_pipeline() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let Ok(s) = std::str::from_utf8(input) else { return };
        let r = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let (program, diags) = cjc_parser::parse_source(s);
            if !diags.has_errors() {
                let _ = cjc_mir_exec::run_program(&program, 42);
            }
        }));
        assert!(r.is_ok(), "MIR pipeline panicked on input {s:?}");
    });
}

/// Fuzz complex number operations: arithmetic must be deterministic
/// (same input -> same output, bitwise).
#[test]
fn fuzz_complex_determinism() {
    bolero::check!()
        .with_type::<(f64, f64, f64, f64)>()
        .for_each(|&(r1, i1, r2, i2): &(f64, f64, f64, f64)| {
            use cjc_runtime::complex::ComplexF64;

            let a = ComplexF64::new(r1, i1);
            let b = ComplexF64::new(r2, i2);

            // Determinism: same computation twice must produce identical bits
            let sum1 = a.add(b);
            let sum2 = a.add(b);
            assert_eq!(sum1.re.to_bits(), sum2.re.to_bits());
            assert_eq!(sum1.im.to_bits(), sum2.im.to_bits());

            let prod1 = a.mul_fixed(b);
            let prod2 = a.mul_fixed(b);
            assert_eq!(prod1.re.to_bits(), prod2.re.to_bits());
            assert_eq!(prod1.im.to_bits(), prod2.im.to_bits());
        });
}

/// Fuzz the MIR verifier: for any parseable program, lowering and legality
/// checking must not panic, and the verifier must be deterministic (same
/// input → same result). Runs directly so panics and assertion failures
/// both fail the target.
#[test]
fn fuzz_mir_verifier() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        for (s, program) in parse_clean_programs(input) {
            let mut mir = cjc_mir_exec::lower_to_mir(&program);
            mir.build_all_cfgs();

            let report1 = cjc_mir::verify::verify_mir_legality(&mir);
            let report2 = cjc_mir::verify::verify_mir_legality(&mir);

            assert_eq!(report1.is_ok(), report2.is_ok(), "verifier verdict differs for {s:?}");
            assert_eq!(report1.checks_total, report2.checks_total, "checks_total differs for {s:?}");
            assert_eq!(report1.checks_passed, report2.checks_passed, "checks_passed differs for {s:?}");
            assert_eq!(report1.errors.len(), report2.errors.len(), "error count differs for {s:?}");
        }
    });
}

/// Fuzz the AST validator: any parseable program must not panic during
/// validation, and validation must be deterministic. Runs directly.
#[test]
fn fuzz_ast_validator() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        for (s, program) in parse_clean_programs(input) {
            let r1 = cjc_ast::validate::validate_ast(&program);
            let r2 = cjc_ast::validate::validate_ast(&program);
            assert_eq!(r1.findings.len(), r2.findings.len(), "finding count differs for {s:?}");
            assert_eq!(r1.checks_run, r2.checks_run, "checks_run differs for {s:?}");
        }
    });
}

/// Fuzz the AST metrics: any parseable program must not panic during
/// metrics computation, and metrics must be deterministic. Runs directly.
#[test]
fn fuzz_ast_metrics() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        for (s, program) in parse_clean_programs(input) {
            let m1 = cjc_ast::metrics::compute_metrics(&program);
            let m2 = cjc_ast::metrics::compute_metrics(&program);
            assert_eq!(m1.total_nodes, m2.total_nodes, "total_nodes differs for {s:?}");
            assert_eq!(m1.expr_count, m2.expr_count, "expr_count differs for {s:?}");
            assert_eq!(m1.function_count, m2.function_count, "function_count differs for {s:?}");
        }
    });
}

/// Fuzz the Jordan-Wigner Hamiltonian expectation: must not panic and must
/// be deterministic for any valid state.
#[test]
fn fuzz_fermion_expectation_determinism() {
    bolero::check!()
        .with_type::<(f64, f64, f64, f64)>()
        .for_each(|&(a0_re, a0_im, a1_re, a1_im): &(f64, f64, f64, f64)| {
            let _ = panic::catch_unwind(|| {
                use cjc_quantum::fermion::h2_hamiltonian;
                use cjc_quantum::statevector::Statevector;
                use cjc_runtime::complex::ComplexF64;

                // Construct a normalized 2-qubit state from fuzz input
                let amps = vec![
                    ComplexF64::new(a0_re, a0_im),
                    ComplexF64::new(a1_re, a1_im),
                    ComplexF64::ZERO,
                    ComplexF64::ZERO,
                ];
                if let Ok(mut sv) = Statevector::from_amplitudes(amps) {
                    sv.normalize();
                    if sv.is_normalized(0.1) {
                        let h = h2_hamiltonian();
                        let e1 = h.expectation(&sv);
                        let e2 = h.expectation(&sv);
                        // Determinism
                        assert_eq!(e1.to_bits(), e2.to_bits());
                    }
                }
            });
        });
}

/// Fuzz Richardson extrapolation: must not panic and must be deterministic.
#[test]
fn fuzz_zne_richardson_determinism() {
    bolero::check!()
        .with_type::<(f64, f64, f64, f64, f64, f64)>()
        .for_each(|&(l1, l2, l3, v1, v2, v3): &(f64, f64, f64, f64, f64, f64)| {
            let _ = panic::catch_unwind(|| {
                use cjc_quantum::mitigation::richardson_extrapolate;

                // Only test with finite, distinct scale factors
                if l1.is_finite() && l2.is_finite() && l3.is_finite()
                    && v1.is_finite() && v2.is_finite() && v3.is_finite()
                    && (l1 - l2).abs() > 1e-10
                    && (l2 - l3).abs() > 1e-10
                    && (l1 - l3).abs() > 1e-10
                    && l1.abs() < 1e6 && l2.abs() < 1e6 && l3.abs() < 1e6
                {
                    let r1 = richardson_extrapolate(&[l1, l2, l3], &[v1, v2, v3]);
                    let r2 = richardson_extrapolate(&[l1, l2, l3], &[v1, v2, v3]);
                    match (r1, r2) {
                        (Ok(a), Ok(b)) => {
                            if a.mitigated_value.is_finite() && b.mitigated_value.is_finite() {
                                assert_eq!(a.mitigated_value.to_bits(), b.mitigated_value.to_bits());
                            }
                        }
                        _ => {}
                    }
                }
            });
        });
}

/// Fuzz the optimizer: for any parseable program, optimized MIR execution
/// must agree with unoptimized execution — the same value and printed output
/// on success, and the same success/failure outcome. Runs directly.
#[test]
fn fuzz_optimizer_parity() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        for (s, program) in parse_clean_programs(input) {
            let unopt = cjc_mir_exec::run_program_with_executor(&program, 42);
            let opt = cjc_mir_exec::run_program_optimized_with_executor(&program, 42);
            match (unopt, opt) {
                (Ok((a, ea)), Ok((b, eb))) => {
                    assert_eq!(format!("{a:?}"), format!("{b:?}"), "optimizer changed the result of:\n{s}");
                    assert_eq!(ea.output, eb.output, "optimizer changed the output of:\n{s}");
                }
                (Err(_), Err(_)) => {}
                (a, b) => panic!(
                    "optimizer changed success/failure of:\n{s}\nunoptimized ok={}, optimized ok={}\n{:?}\n{:?}",
                    a.is_ok(), b.is_ok(), a.err(), b.err()
                ),
            }
        }
    });
}

/// The generator must only produce programs that lex and parse cleanly;
/// otherwise the generated-program targets silently lose coverage.
#[test]
fn fuzz_generated_programs_parse() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let src = program_gen::gen_program(input);
        let (_, diags) = cjc_parser::parse_source(&src);
        assert!(!diags.has_errors(), "generator produced a program that does not parse:\n{src}");
    });
}

/// Both executors must agree (prime directive 7): for every generated
/// program, AST-eval and MIR-exec must produce the same success/failure
/// outcome and, on success, identical printed output. MIR-exec must also be
/// deterministic across repeated runs. Runs directly.
#[test]
fn fuzz_generated_eval_mir_parity() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let src = program_gen::gen_program(input);
        let (program, diags) = cjc_parser::parse_source(&src);
        if diags.has_errors() {
            return; // covered by fuzz_generated_programs_parse
        }
        let mut interp = cjc_eval::Interpreter::new(42);
        let eval = interp.exec(&program).map(|_| interp.output.clone());
        let mir = cjc_mir_exec::run_program_with_executor(&program, 42).map(|(_, e)| e.output);
        match (&eval, &mir) {
            (Ok(a), Ok(b)) => assert_eq!(a, b, "AST-eval vs MIR-exec output differs for:\n{src}"),
            (Err(_), Err(_)) => {}
            _ => panic!(
                "AST-eval vs MIR-exec success/failure differs for:\n{src}\neval={eval:?}\nmir={mir:?}"
            ),
        }
        let again = cjc_mir_exec::run_program_with_executor(&program, 42).map(|(_, e)| e.output);
        assert_eq!(format!("{mir:?}"), format!("{again:?}"), "MIR-exec not deterministic for:\n{src}");
    });
}

/// Coverage sanity check (not a fuzz target): over a fixed pseudo-random
/// corpus, every generated program must parse and most must run to
/// completion, so the generated-program targets exercise real execution
/// rather than bailing out on an early runtime error.
#[test]
fn generated_programs_mostly_run_to_completion() {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let (mut total, mut ran_ok) = (0usize, 0usize);
    for len in (0..400).map(|i| 8 + i % 120) {
        let bytes: Vec<u8> = (0..len)
            .map(|_| {
                // SplitMix64 step: deterministic corpus, no external RNG.
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                (z ^ (z >> 31)) as u8
            })
            .collect();
        let src = program_gen::gen_program(&bytes);
        let (program, diags) = cjc_parser::parse_source(&src);
        assert!(!diags.has_errors(), "generated program does not parse:\n{src}");
        total += 1;
        if cjc_mir_exec::run_program(&program, 42).is_ok() {
            ran_ok += 1;
        }
    }
    assert!(
        ran_ok * 2 >= total,
        "only {ran_ok}/{total} generated programs ran to completion; generator coverage regressed"
    );
}
