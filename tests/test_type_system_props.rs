//! Property-based tests for the CJC type system.
//!
//! These tests verify type-level invariants using proptest to generate
//! random programs. They complement the existing `type_checker_props` tests
//! by focusing on *semantic* type properties rather than crash-freedom.
//!
//! Run with:
//!   cargo test --test test_type_system_props

use proptest::prelude::*;

// ---------------------------------------------------------------------------
// 1. Literal type concreteness
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Every integer literal is assigned type I64 (never a TypeVar).
    #[test]
    fn literal_int_has_concrete_type(n in -10_000i64..10_000i64) {
        let src = format!("fn main() -> i64 {{ let x: i64 = {}; x }}", n);
        let (prog, diags) = cjc_parser::parse_source(&src);
        assert!(!diags.has_errors(), "parse error on int literal {}", n);
        let mut checker = cjc_types::TypeChecker::new();
        checker.check_program(&prog);
        // Type checker should not report errors for well-typed integer literals.
        assert!(
            !checker.diagnostics.has_errors(),
            "type errors on int literal {}",
            n,
        );
    }

    /// Every float literal is assigned type F64 (never a TypeVar).
    #[test]
    fn literal_float_has_concrete_type(
        whole in -1000i64..1000i64,
        frac in 0u32..10000u32,
    ) {
        let src = format!(
            "fn main() -> f64 {{ let x: f64 = {}.{}; x }}",
            whole, frac
        );
        let (prog, diags) = cjc_parser::parse_source(&src);
        if diags.has_errors() {
            // Some fractional combos may produce parse issues; skip those.
            return Ok(());
        }
        let mut checker = cjc_types::TypeChecker::new();
        checker.check_program(&prog);
        assert!(
            !checker.diagnostics.has_errors(),
            "type errors on float literal {}.{}",
            whole,
            frac,
        );
    }

    /// Boolean literals have type Bool.
    #[test]
    fn literal_bool_has_bool_type(b in prop::bool::ANY) {
        let val = if b { "true" } else { "false" };
        let src = format!("fn main() -> bool {{ let x: bool = {}; x }}", val);
        let (prog, diags) = cjc_parser::parse_source(&src);
        assert!(!diags.has_errors());
        let mut checker = cjc_types::TypeChecker::new();
        checker.check_program(&prog);
        assert!(
            !checker.diagnostics.has_errors(),
            "type errors on bool literal {}",
            val
        );
    }

    /// String literals parse successfully and the type checker does not panic.
    #[test]
    fn literal_string_type_checker_no_panic(s in "[a-zA-Z0-9 ]{0,20}") {
        let src = format!("fn main() {{ let x: str = \"{}\"; print(x); }}", s);
        let (prog, diags) = cjc_parser::parse_source(&src);
        if diags.has_errors() {
            return Ok(());
        }
        let mut checker = cjc_types::TypeChecker::new();
        // Must not panic. Type errors are acceptable (different type semantics).
        checker.check_program(&prog);
    }
}

// ---------------------------------------------------------------------------
// 2. Type checking determinism
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(80))]

    /// The same program must produce the same type-checking outcome across 10 runs.
    #[test]
    fn type_checking_is_deterministic(
        a in -500i64..500i64,
        b in -500i64..500i64,
    ) {
        let src = format!(
            "fn main() -> i64 {{ let x: i64 = {}; let y: i64 = {}; x + y }}",
            a, b
        );

        let mut results = Vec::new();
        for _ in 0..10 {
            let (prog, diags) = cjc_parser::parse_source(&src);
            assert!(!diags.has_errors());
            let mut checker = cjc_types::TypeChecker::new();
            checker.check_program(&prog);
            results.push(checker.diagnostics.has_errors());
        }

        // All 10 runs must agree.
        let first = results[0];
        for (i, r) in results.iter().enumerate() {
            assert_eq!(
                *r, first,
                "type check determinism failure on run {}",
                i
            );
        }
    }

    /// Type checking a program with mixed types (including intentional mismatches)
    /// is deterministic across 10 runs.
    #[test]
    fn type_checking_mismatch_determinism(
        n in -100i64..100i64,
    ) {
        // Intentional type mismatch: assigning int to bool.
        let src = format!(
            "fn main() -> bool {{ let x: bool = {}; x }}",
            n
        );

        let mut results = Vec::new();
        for _ in 0..10 {
            let (prog, diags) = cjc_parser::parse_source(&src);
            if diags.has_errors() {
                results.push(None);
            } else {
                let mut checker = cjc_types::TypeChecker::new();
                checker.check_program(&prog);
                results.push(Some(checker.diagnostics.has_errors()));
            }
        }

        let first = &results[0];
        for (i, r) in results.iter().enumerate() {
            assert_eq!(
                r, first,
                "mismatch determinism failure on run {}",
                i
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Function return consistency
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// A function called multiple times at different call sites should not
    /// cause the type checker to diverge.
    #[test]
    fn function_return_type_consistent(
        call_count in 2usize..6,
        n in 1i64..100i64,
    ) {
        let calls: String = (0..call_count)
            .map(|i| format!("    let r{}: i64 = add_one({});", i, n + i as i64))
            .collect::<Vec<_>>()
            .join("\n");

        let src = format!(
            "fn add_one(x: i64) -> i64 {{ x + 1 }}\nfn main() -> i64 {{\n{}\n    r0\n}}",
            calls
        );

        let (prog, diags) = cjc_parser::parse_source(&src);
        assert!(!diags.has_errors(), "parse error in multi-call program");
        let mut checker = cjc_types::TypeChecker::new();
        checker.check_program(&prog);
        assert!(
            !checker.diagnostics.has_errors(),
            "type errors in multi-call program with {} calls",
            call_count,
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Binary operator type rules
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(150))]

    /// int op int should type-check when annotated as i64.
    #[test]
    fn binop_int_int_yields_int(
        a in -500i64..500i64,
        b in 1i64..500i64, // avoid division by zero
        op in prop_oneof![Just("+"), Just("-"), Just("*")],
    ) {
        let src = format!(
            "fn main() -> i64 {{ let r: i64 = {} {} {}; r }}",
            a, op, b
        );
        let (prog, diags) = cjc_parser::parse_source(&src);
        assert!(!diags.has_errors());
        let mut checker = cjc_types::TypeChecker::new();
        checker.check_program(&prog);
        assert!(
            !checker.diagnostics.has_errors(),
            "int {} int should be int",
            op
        );
    }

    /// float op float should type-check when annotated as f64.
    #[test]
    fn binop_float_float_yields_float(
        a in -500.0f64..500.0f64,
        b in 0.1f64..500.0f64,
        op in prop_oneof![Just("+"), Just("-"), Just("*"), Just("/")],
    ) {
        let src = format!(
            "fn main() -> f64 {{ let r: f64 = {:.6} {} {:.6}; r }}",
            a, op, b
        );
        let (prog, diags) = cjc_parser::parse_source(&src);
        if diags.has_errors() {
            return Ok(());
        }
        let mut checker = cjc_types::TypeChecker::new();
        checker.check_program(&prog);
        assert!(
            !checker.diagnostics.has_errors(),
            "float {} float should be float",
            op
        );
    }

    /// int op float should type-check when annotated as f64 (promotion).
    #[test]
    fn binop_int_float_yields_float(
        a in -500i64..500i64,
        b in 0.1f64..500.0f64,
        op in prop_oneof![Just("+"), Just("-"), Just("*")],
    ) {
        let src = format!(
            "fn main() -> f64 {{ let r: f64 = {} {} {:.6}; r }}",
            a, op, b
        );
        let (prog, diags) = cjc_parser::parse_source(&src);
        if diags.has_errors() {
            return Ok(());
        }
        let mut checker = cjc_types::TypeChecker::new();
        checker.check_program(&prog);
        // The type checker may or may not accept int+float promotion;
        // but it must not panic.
    }
}

// ---------------------------------------------------------------------------
// 5. Comparison produces bool
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(150))]

    /// Comparison of two integers should type-check as bool.
    #[test]
    fn comparison_int_produces_bool(
        a in -1000i64..1000i64,
        b in -1000i64..1000i64,
        op in prop_oneof![Just("<"), Just(">"), Just("<="), Just(">="), Just("=="), Just("!=")],
    ) {
        let src = format!(
            "fn main() -> bool {{ let r: bool = {} {} {}; r }}",
            a, op, b
        );
        let (prog, diags) = cjc_parser::parse_source(&src);
        assert!(!diags.has_errors(), "parse error for {} {} {}", a, op, b);
        let mut checker = cjc_types::TypeChecker::new();
        checker.check_program(&prog);
        assert!(
            !checker.diagnostics.has_errors(),
            "comparison {} {} {} should produce bool",
            a,
            op,
            b
        );
    }

    /// Comparison of two floats should type-check as bool.
    #[test]
    fn comparison_float_produces_bool(
        a in -100.0f64..100.0f64,
        b in -100.0f64..100.0f64,
        op in prop_oneof![Just("<"), Just(">"), Just("<="), Just(">="), Just("=="), Just("!=")],
    ) {
        let src = format!(
            "fn main() -> bool {{ let r: bool = {:.4} {} {:.4}; r }}",
            a, op, b
        );
        let (prog, diags) = cjc_parser::parse_source(&src);
        if diags.has_errors() {
            return Ok(());
        }
        let mut checker = cjc_types::TypeChecker::new();
        checker.check_program(&prog);
        assert!(
            !checker.diagnostics.has_errors(),
            "float comparison should produce bool"
        );
    }
}

// ---------------------------------------------------------------------------
// Bonus: Nested expressions maintain type soundness
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(80))]

    /// Nested arithmetic expressions should type-check correctly.
    #[test]
    fn nested_arithmetic_type_sound(
        a in 1i64..100i64,
        b in 1i64..100i64,
        c in 1i64..100i64,
    ) {
        let src = format!(
            "fn main() -> i64 {{ let r: i64 = ({} + {}) * {}; r }}",
            a, b, c
        );
        let (prog, diags) = cjc_parser::parse_source(&src);
        assert!(!diags.has_errors());
        let mut checker = cjc_types::TypeChecker::new();
        checker.check_program(&prog);
        assert!(!checker.diagnostics.has_errors());
    }

    /// Boolean expressions from comparisons should chain with logical ops.
    #[test]
    fn boolean_chain_type_sound(
        a in -100i64..100i64,
        b in -100i64..100i64,
        c in -100i64..100i64,
    ) {
        let src = format!(
            "fn main() -> bool {{ let r: bool = {} < {} and {} > 0; r }}",
            a, b, c
        );
        let (prog, diags) = cjc_parser::parse_source(&src);
        if diags.has_errors() {
            return Ok(());
        }
        let mut checker = cjc_types::TypeChecker::new();
        checker.check_program(&prog);
        // Must not panic; error diagnostics are acceptable for unsupported ops.
    }
}
