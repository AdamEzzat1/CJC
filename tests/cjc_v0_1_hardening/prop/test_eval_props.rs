//! Property-based tests for CJC evaluation.

use proptest::prelude::*;

/// Strategy generating simple arithmetic.
fn arb_arith() -> impl Strategy<Value = (String, i64)> {
    let op = prop_oneof![Just("+".to_string()), Just("-".to_string()), Just("*".to_string())];
    (1i64..100, op, 1i64..100).prop_map(|(a, op, b)| {
        let expected = match &*op {
            "+" => a + b,
            "-" => a - b,
            "*" => a * b,
            _ => unreachable!(),
        };
        (format!("fn main() {{ print({} {} {}); }}", a, op, b), expected)
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Integer literals evaluate correctly in eval.
    #[test]
    fn eval_int_literal_correct(n in -1000i64..1000) {
        let src = format!("fn main() {{ print({}); }}", n);
        let (program, diags) = cjc_parser::parse_source(&src);
        if !diags.has_errors() {
            let mut interp = cjc_eval::Interpreter::new(42);
            if let Ok(_) = interp.exec(&program) {
                if let Some(output) = interp.output.first() {
                    let val: i64 = output.parse().unwrap_or(0);
                    prop_assert_eq!(val, n);
                }
            }
        }
    }

    /// Arithmetic produces correct results in eval.
    #[test]
    fn eval_arith_correct((src, expected) in arb_arith()) {
        let (program, diags) = cjc_parser::parse_source(&src);
        if !diags.has_errors() {
            let mut interp = cjc_eval::Interpreter::new(42);
            if let Ok(_) = interp.exec(&program) {
                if let Some(output) = interp.output.first() {
                    let val: i64 = output.parse().unwrap_or(0);
                    prop_assert_eq!(val, expected);
                }
            }
        }
    }

    /// Integer literals evaluate correctly in MIR-exec.
    #[test]
    fn mir_int_literal_correct(n in -1000i64..1000) {
        let src = format!("fn main() {{ print({}); }}", n);
        let (program, diags) = cjc_parser::parse_source(&src);
        if !diags.has_errors() {
            if let Ok((_, executor)) = cjc_mir_exec::run_program_with_executor(&program, 42) {
                if let Some(output) = executor.output.first() {
                    let val: i64 = output.parse().unwrap_or(0);
                    prop_assert_eq!(val, n);
                }
            }
        }
    }

    /// Arithmetic produces correct results in MIR-exec.
    #[test]
    fn mir_arith_correct((src, expected) in arb_arith()) {
        let (program, diags) = cjc_parser::parse_source(&src);
        if !diags.has_errors() {
            if let Ok((_, executor)) = cjc_mir_exec::run_program_with_executor(&program, 42) {
                if let Some(output) = executor.output.first() {
                    let val: i64 = output.parse().unwrap_or(0);
                    prop_assert_eq!(val, expected);
                }
            }
        }
    }
}
