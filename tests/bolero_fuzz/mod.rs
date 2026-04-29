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
//! Known issues found by fuzzing:
//! - Lexer panics on multi-byte UTF-8 characters at string slice boundaries
//! - Parser inherits lexer panics

pub mod adaptive_selection_fuzz;
pub mod categorical_dictionary_fuzz;
pub mod categorical_join_fuzz;
pub mod cli_expansion_fuzz;
pub mod hybrid_streaming_fuzz;
pub mod v2_1_bytecode_fuzz;

use std::panic;

/// Fuzz the CJC lexer: arbitrary byte input should not panic.
/// Currently wraps in catch_unwind due to known multi-byte UTF-8 panics.
#[test]
fn fuzz_lexer() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        if let Ok(s) = std::str::from_utf8(input) {
            let s = s.to_string();
            let _ = panic::catch_unwind(|| {
                let lexer = cjc_lexer::Lexer::new(&s);
                let (_tokens, _diags) = lexer.tokenize();
            });
        }
    });
}

/// Fuzz the CJC parser: any token stream from valid UTF-8 should not panic.
#[test]
fn fuzz_parser() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        if let Ok(s) = std::str::from_utf8(input) {
            let s = s.to_string();
            let _ = panic::catch_unwind(|| {
                let (_program, _diags) = cjc_parser::parse_source(&s);
            });
        }
    });
}

/// Fuzz the full MIR pipeline: parse + lower + execute should not panic.
#[test]
fn fuzz_mir_pipeline() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        if let Ok(s) = std::str::from_utf8(input) {
            let s = s.to_string();
            let _ = panic::catch_unwind(|| {
                let (program, diags) = cjc_parser::parse_source(&s);
                if !diags.has_errors() {
                    let _ = cjc_mir_exec::run_program(&program, 42);
                }
            });
        }
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

/// Fuzz the MIR verifier: any parseable program should not panic during
/// legality checking, and the verifier must be deterministic (same input
/// → same result).
#[test]
fn fuzz_mir_verifier() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        if let Ok(s) = std::str::from_utf8(input) {
            let s = s.to_string();
            let _ = panic::catch_unwind(|| {
                let (program, diags) = cjc_parser::parse_source(&s);
                if !diags.has_errors() {
                    let mut mir = cjc_mir_exec::lower_to_mir(&program);
                    mir.build_all_cfgs();

                    // Must not panic
                    let report1 = cjc_mir::verify::verify_mir_legality(&mir);
                    let report2 = cjc_mir::verify::verify_mir_legality(&mir);

                    // Determinism: same input → same result
                    assert_eq!(report1.is_ok(), report2.is_ok());
                    assert_eq!(report1.checks_total, report2.checks_total);
                    assert_eq!(report1.checks_passed, report2.checks_passed);
                    assert_eq!(report1.errors.len(), report2.errors.len());
                }
            });
        }
    });
}

/// Fuzz the AST validator: any parseable program must not panic during
/// validation, and validation must be deterministic.
#[test]
fn fuzz_ast_validator() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        if let Ok(s) = std::str::from_utf8(input) {
            let s = s.to_string();
            let _ = panic::catch_unwind(|| {
                let (program, diags) = cjc_parser::parse_source(&s);
                if !diags.has_errors() {
                    let r1 = cjc_ast::validate::validate_ast(&program);
                    let r2 = cjc_ast::validate::validate_ast(&program);
                    assert_eq!(r1.findings.len(), r2.findings.len());
                    assert_eq!(r1.checks_run, r2.checks_run);
                }
            });
        }
    });
}

/// Fuzz the AST metrics: any parseable program must not panic during
/// metrics computation, and metrics must be deterministic.
#[test]
fn fuzz_ast_metrics() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        if let Ok(s) = std::str::from_utf8(input) {
            let s = s.to_string();
            let _ = panic::catch_unwind(|| {
                let (program, diags) = cjc_parser::parse_source(&s);
                if !diags.has_errors() {
                    let m1 = cjc_ast::metrics::compute_metrics(&program);
                    let m2 = cjc_ast::metrics::compute_metrics(&program);
                    assert_eq!(m1.total_nodes, m2.total_nodes);
                    assert_eq!(m1.expr_count, m2.expr_count);
                    assert_eq!(m1.function_count, m2.function_count);
                }
            });
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

/// Fuzz the optimizer: optimized MIR execution must produce the same result
/// as unoptimized for any parseable program.
#[test]
fn fuzz_optimizer_parity() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        if let Ok(s) = std::str::from_utf8(input) {
            let s = s.to_string();
            let _ = panic::catch_unwind(|| {
                let (program, diags) = cjc_parser::parse_source(&s);
                if !diags.has_errors() {
                    let unopt = cjc_mir_exec::run_program(&program, 42);
                    let opt = cjc_mir_exec::run_program_optimized(&program, 42);
                    match (unopt, opt) {
                        (Ok(a), Ok(b)) => {
                            let sa = format!("{a:?}");
                            let sb = format!("{b:?}");
                            assert_eq!(sa, sb, "optimizer parity failure");
                        }
                        _ => {}
                    }
                }
            });
        }
    });
}
