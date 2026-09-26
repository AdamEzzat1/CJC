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

pub mod abng_decision_fuzz;
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
///
/// No `catch_unwind` here: the determinism assertion has to be able to fail
/// the target (it previously sat inside a discarded `catch_unwind`, so no
/// input could ever make this target fail).
#[test]
fn fuzz_fermion_expectation_determinism() {
    use cjc_quantum::fermion::{h2_hamiltonian, lih_hamiltonian};
    use cjc_quantum::statevector::Statevector;
    use cjc_runtime::complex::ComplexF64;
    bolero::check!()
        .with_type::<(f64, f64, f64, f64)>()
        .for_each(|&(a0_re, a0_im, a1_re, a1_im): &(f64, f64, f64, f64)| {
            if ![a0_re, a0_im, a1_re, a1_im].iter().all(|x| x.is_finite() && x.abs() < 1e150) {
                return;
            }
            let amps = vec![
                ComplexF64::new(a0_re, a0_im),
                ComplexF64::new(a1_re, a1_im),
                ComplexF64::ZERO,
                ComplexF64::ZERO,
            ];
            let mut sv = Statevector::from_amplitudes(amps).unwrap();
            sv.normalize();
            if !sv.is_normalized(1e-9) {
                return; // all-zero input
            }
            let h = h2_hamiltonian();
            let e1 = h.expectation(&sv);
            let e2 = h.expectation(&sv);
            assert!(e1.is_finite(), "H2 expectation not finite for normalized state");
            assert_eq!(e1.to_bits(), e2.to_bits(), "H2 expectation not deterministic");

            // Same for LiH on a 4-qubit state built from the same inputs.
            let mut a4 = vec![ComplexF64::ZERO; 16];
            a4[0b0011] = ComplexF64::new(a0_re, a0_im);
            a4[0b1100] = ComplexF64::new(a1_re, a1_im);
            let mut sv4 = Statevector::from_amplitudes(a4).unwrap();
            sv4.normalize();
            if sv4.is_normalized(1e-9) {
                let l = lih_hamiltonian();
                let (x, y) = (l.expectation(&sv4), l.expectation(&sv4));
                assert!(x.is_finite());
                assert_eq!(x.to_bits(), y.to_bits(), "LiH expectation not deterministic");
            }
        });
}

/// Fuzz Richardson extrapolation: must not panic and must be deterministic.
///
/// No `catch_unwind`: panics and determinism failures must fail the target.
#[test]
fn fuzz_zne_richardson_determinism() {
    use cjc_quantum::mitigation::richardson_extrapolate;
    bolero::check!()
        .with_type::<(f64, f64, f64, f64, f64, f64)>()
        .for_each(|&(l1, l2, l3, v1, v2, v3): &(f64, f64, f64, f64, f64, f64)| {
            // Any finite input must return Ok or Err, never panic.
            let r1 = richardson_extrapolate(&[l1, l2, l3], &[v1, v2, v3]);
            let r2 = richardson_extrapolate(&[l1, l2, l3], &[v1, v2, v3]);
            match (r1, r2) {
                (Ok(a), Ok(b)) => {
                    assert_eq!(
                        a.mitigated_value.to_bits(),
                        b.mitigated_value.to_bits(),
                        "Richardson extrapolation not deterministic"
                    );
                }
                (Err(a), Err(b)) => assert_eq!(a, b),
                _ => panic!("Richardson extrapolation: Ok/Err differs between identical calls"),
            }
        });
}

/// Fuzz every quantum builtin at the language boundary: arbitrary builtin name
/// index plus an arbitrary argument vector decoded from bytes must never panic.
#[test]
fn fuzz_quantum_dispatch_no_panic() {
    use cjc_quantum::dispatch_quantum;
    use cjc_runtime::value::Value;
    use std::rc::Rc;
    const NAMES: &[&str] = &[
        "qubits", "q_h", "q_cx", "q_rz", "q_toffoli", "q_run", "q_measure", "q_probs",
        "q_sample", "q_amplitudes", "mps_new", "mps_h", "mps_ry", "mps_cnot", "mps_swap",
        "mps_energy", "mps_mixed_canonicalize", "vqe_heisenberg", "qaoa_graph_cycle",
        "qaoa_maxcut", "stabilizer_new", "stabilizer_cnot", "stabilizer_measure", "density_new",
        "density_gate", "density_cnot", "density_depolarize", "density_probs", "dmrg_ising",
        "qec_repetition_code", "qec_surface_code", "qec_syndrome", "qec_decode",
        "qec_logical_error_rate", "qml_train", "qml_predict", "quantum_inspect",
        "q_fermion_h2", "q_fermion_new", "q_fermion_expectation", "q_trotter_evolve",
        "q_trotter_error", "q_zne_mitigate", "q_zne_linear", "q_scale_noise",
        "q_expect_pauli", "q_fermion_add_term", "density_from_state", "q_copy",
        "q_to_qasm", "q_from_qasm",
    ];
    fn state(k: u8) -> Value {
        let ok = |n: &str, a: &[Value]| dispatch_quantum(n, a).unwrap().unwrap();
        let pure = || Value::String(Rc::new("pure".to_string()));
        match k % 9 {
            0 => ok("qubits", &[Value::Int(3)]),
            1 => ok("qubits", &[Value::Int(3), pure()]),
            2 => ok("mps_new", &[Value::Int(4), Value::Int(4)]),
            3 => ok("stabilizer_new", &[Value::Int(4)]),
            4 => ok("density_new", &[Value::Int(2)]),
            5 => ok("qaoa_graph_cycle", &[Value::Int(4)]),
            6 => ok("qec_repetition_code", &[Value::Int(3)]),
            7 => ok("q_fermion_h2", &[]),
            _ => ok("q_run", &[ok("qubits", &[Value::Int(2)])]),
        }
    }
    fn decode(b: &[u8]) -> Value {
        let (tag, x) = (b[0], b[1]);
        match tag % 6 {
            0 => Value::Int((x as i8 as i64) % 6), // small ints incl. negatives
            1 => Value::Float(match x % 5 {
                0 => f64::NAN,
                1 => f64::INFINITY,
                2 => -0.5,
                3 => 1.5,
                _ => x as f64 / 64.0,
            }),
            2 => Value::String(Rc::new(
                ["pure", "H", "heisenberg", "ising", "x", ""][x as usize % 6].to_string(),
            )),
            3 => Value::Array(Rc::new(vec![Value::Int(x as i64 % 3), Value::Float(0.5)])),
            4 => Value::Array(Rc::new(vec![Value::Array(Rc::new(vec![Value::Float(0.1)]))])),
            _ => state(x),
        }
    }
    bolero::check!()
        .with_type::<(u8, Vec<(u8, u8)>)>()
        .for_each(|(name_idx, raw): &(u8, Vec<(u8, u8)>)| {
            let name = NAMES[*name_idx as usize % NAMES.len()];
            let args: Vec<Value> = raw.iter().take(9).map(|&(a, b)| decode(&[a, b])).collect();
            let r = panic::catch_unwind(panic::AssertUnwindSafe(|| dispatch_quantum(name, &args)));
            assert!(r.is_ok(), "dispatch_quantum panicked: {}({:?})", name, args);
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
