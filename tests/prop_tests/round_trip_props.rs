//! Property-based round-trip tests for the CJC compiler.
//!
//! These tests verify that:
//! 1. Parsing a valid program and running it produces a deterministic result
//! 2. Running the same program twice with the same seed yields identical output
//! 3. The MIR lowering pipeline is deterministic

use proptest::prelude::*;

/// Strategy generating valid integer-returning programs with arithmetic.
fn arb_arithmetic_program() -> impl Strategy<Value = String> {
    let op = prop_oneof![
        Just("+".to_string()),
        Just("-".to_string()),
        Just("*".to_string()),
    ];
    let val = 1i64..100;

    (val.clone(), op, val).prop_map(|(a, op, b)| {
        format!("fn main() -> i64 {{ {} {} {} }}", a, op, b)
    })
}

/// Strategy generating valid programs with let bindings.
fn arb_let_program() -> impl Strategy<Value = String> {
    let val = 1i64..1000;
    let op = prop_oneof![
        Just("+".to_string()),
        Just("-".to_string()),
        Just("*".to_string()),
    ];

    (val.clone(), val.clone(), op).prop_map(|(a, b, op)| {
        format!(
            "fn main() -> i64 {{\n    let x: i64 = {};\n    let y: i64 = {};\n    x {} y\n}}",
            a, b, op
        )
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Determinism: running the same program twice with the same seed gives identical results.
    #[test]
    fn deterministic_execution(src in arb_arithmetic_program()) {
        let (prog, diags) = cjc_parser::parse_source(&src);
        prop_assume!(!diags.has_errors());

        let r1 = cjc_mir_exec::run_program_with_executor(&prog, 42);
        let r2 = cjc_mir_exec::run_program_with_executor(&prog, 42);

        match (r1, r2) {
            (Ok((v1, e1)), Ok((v2, e2))) => {
                prop_assert_eq!(format!("{:?}", v1), format!("{:?}", v2),
                    "Same program, same seed, different result values");
                prop_assert_eq!(e1.output, e2.output,
                    "Same program, same seed, different stdout");
            }
            (Err(ref e1), Err(ref e2)) => {
                prop_assert_eq!(format!("{}", e1), format!("{}", e2),
                    "Same program, same seed, different errors");
            }
            _ => {
                prop_assert!(false, "One run succeeded, the other failed for same input");
            }
        }
    }

    /// Determinism with let bindings: more complex programs are still deterministic.
    #[test]
    fn deterministic_let_programs(src in arb_let_program()) {
        let (prog, diags) = cjc_parser::parse_source(&src);
        prop_assume!(!diags.has_errors());

        let r1 = cjc_mir_exec::run_program_with_executor(&prog, 42);
        let r2 = cjc_mir_exec::run_program_with_executor(&prog, 42);

        match (r1, r2) {
            (Ok((v1, _)), Ok((v2, _))) => {
                prop_assert_eq!(format!("{:?}", v1), format!("{:?}", v2));
            }
            _ => {} // Both errors is also fine for determinism
        }
    }

    /// Parse round-trip: parsing never loses the ability to re-parse.
    /// (We can't test full AST equality without a printer, but we can verify
    /// that the program's function count is consistent.)
    #[test]
    fn parse_preserves_function_count(src in arb_let_program()) {
        let (prog1, diags1) = cjc_parser::parse_source(&src);
        let (prog2, diags2) = cjc_parser::parse_source(&src);

        prop_assert_eq!(diags1.has_errors(), diags2.has_errors());
        prop_assert_eq!(prog1.declarations.len(), prog2.declarations.len());
    }
}
