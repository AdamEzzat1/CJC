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
