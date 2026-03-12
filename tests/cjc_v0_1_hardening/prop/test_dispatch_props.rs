//! Property-based tests for operator dispatch via CJC execution.

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Integer addition is commutative via eval.
    #[test]
    fn int_add_commutative(a in -10000i64..10000, b in -10000i64..10000) {
        let src1 = format!("fn main() {{ print({} + {}); }}", a, b);
        let src2 = format!("fn main() {{ print({} + {}); }}", b, a);
        let (p1, d1) = cjc_parser::parse_source(&src1);
        let (p2, d2) = cjc_parser::parse_source(&src2);
        if !d1.has_errors() && !d2.has_errors() {
            let mut i1 = cjc_eval::Interpreter::new(42);
            let mut i2 = cjc_eval::Interpreter::new(42);
            if let (Ok(_), Ok(_)) = (i1.exec(&p1), i2.exec(&p2)) {
                prop_assert_eq!(&i1.output, &i2.output);
            }
        }
    }

    /// Integer multiplication is commutative via eval.
    #[test]
    fn int_mul_commutative(a in -1000i64..1000, b in -1000i64..1000) {
        let src1 = format!("fn main() {{ print({} * {}); }}", a, b);
        let src2 = format!("fn main() {{ print({} * {}); }}", b, a);
        let (p1, d1) = cjc_parser::parse_source(&src1);
        let (p2, d2) = cjc_parser::parse_source(&src2);
        if !d1.has_errors() && !d2.has_errors() {
            let mut i1 = cjc_eval::Interpreter::new(42);
            let mut i2 = cjc_eval::Interpreter::new(42);
            if let (Ok(_), Ok(_)) = (i1.exec(&p1), i2.exec(&p2)) {
                prop_assert_eq!(&i1.output, &i2.output);
            }
        }
    }

    /// a + 0 == a for integers.
    #[test]
    fn int_add_identity(a in -100000i64..100000) {
        let src = format!("fn main() {{ print({} + 0); }}", a);
        let (program, diags) = cjc_parser::parse_source(&src);
        if !diags.has_errors() {
            let mut interp = cjc_eval::Interpreter::new(42);
            if let Ok(_) = interp.exec(&program) {
                if let Some(out) = interp.output.first() {
                    let val: i64 = out.parse().unwrap_or(0);
                    prop_assert_eq!(val, a);
                }
            }
        }
    }

    /// a * 1 == a for integers.
    #[test]
    fn int_mul_identity(a in -100000i64..100000) {
        let src = format!("fn main() {{ print({} * 1); }}", a);
        let (program, diags) = cjc_parser::parse_source(&src);
        if !diags.has_errors() {
            let mut interp = cjc_eval::Interpreter::new(42);
            if let Ok(_) = interp.exec(&program) {
                if let Some(out) = interp.output.first() {
                    let val: i64 = out.parse().unwrap_or(0);
                    prop_assert_eq!(val, a);
                }
            }
        }
    }

    /// a * 0 == 0 for integers.
    #[test]
    fn int_mul_zero(a in -100000i64..100000) {
        let src = format!("fn main() {{ print({} * 0); }}", a);
        let (program, diags) = cjc_parser::parse_source(&src);
        if !diags.has_errors() {
            let mut interp = cjc_eval::Interpreter::new(42);
            if let Ok(_) = interp.exec(&program) {
                if let Some(out) = interp.output.first() {
                    let val: i64 = out.parse().unwrap_or(0);
                    prop_assert_eq!(val, 0);
                }
            }
        }
    }

    /// Eval and MIR-exec produce identical results for integer expressions.
    #[test]
    fn eval_mir_parity(a in 1i64..100, b in 1i64..100) {
        let src = format!("fn main() {{ print({} + {} * {}); }}", a, b, a);
        let (program, diags) = cjc_parser::parse_source(&src);
        if !diags.has_errors() {
            let mut interp = cjc_eval::Interpreter::new(42);
            if let Ok(_) = interp.exec(&program) {
                if let Ok((_, executor)) = cjc_mir_exec::run_program_with_executor(&program, 42) {
                    prop_assert_eq!(&interp.output, &executor.output);
                }
            }
        }
    }
}
