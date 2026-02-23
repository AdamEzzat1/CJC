//! Property-based tests for the CJC type checker.
//!
//! These tests verify that the type checker never panics on any input,
//! returning structured errors rather than crashing.

use proptest::prelude::*;

/// Strategy that generates syntactically valid CJC programs with various type annotations.
fn arb_typed_program() -> impl Strategy<Value = String> {
    let ty = prop_oneof![
        Just("i64".to_string()),
        Just("f64".to_string()),
        Just("bool".to_string()),
        Just("str".to_string()),
    ];

    let expr_for_type = prop_oneof![
        Just("42"),
        Just("3.14"),
        Just("true"),
        Just("\"hello\""),
        // Intentional mismatches to test error handling
        Just("\"wrong\""),
    ];

    let binding = (ty.clone(), expr_for_type).prop_map(|(ann_ty, val)| {
        format!("    let x: {} = {};", ann_ty, val)
    });

    let body = proptest::collection::vec(binding, 0..5).prop_map(|bindings| {
        bindings.join("\n")
    });

    (ty, body).prop_map(|(ret_ty, body_stmts)| {
        format!(
            "fn main() -> {} {{\n{}\n    0\n}}",
            ret_ty, body_stmts
        )
    })
}

/// Strategy that generates programs with function calls (may or may not type-check).
fn arb_fn_call_program() -> impl Strategy<Value = String> {
    let arg_count = 0usize..3;
    let fn_name = prop_oneof![
        Just("add".to_string()),
        Just("negate".to_string()),
        Just("identity".to_string()),
    ];

    (fn_name, arg_count).prop_map(|(name, argc)| {
        let params: Vec<String> = (0..argc)
            .map(|i| format!("p{}: i64", i))
            .collect();
        let args: Vec<String> = (0..argc)
            .map(|i| format!("{}", i + 1))
            .collect();
        let body = if argc > 0 { "p0".to_string() } else { "0".to_string() };
        format!(
            "fn {}({}) -> i64 {{ {} }}\nfn main() -> i64 {{ {}({}) }}",
            name,
            params.join(", "),
            body,
            name,
            args.join(", ")
        )
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// The type checker must never panic, even on type-mismatched programs.
    #[test]
    fn type_checker_never_panics(src in arb_typed_program()) {
        let (prog, diags) = cjc_parser::parse_source(&src);
        if !diags.has_errors() {
            let mut checker = cjc_types::TypeChecker::new();
            // Must not panic — may return errors, which is fine
            checker.check_program(&prog);
        }
    }

    /// The type checker handles function call programs without panicking.
    #[test]
    fn type_checker_fn_calls_no_panic(src in arb_fn_call_program()) {
        let (prog, diags) = cjc_parser::parse_source(&src);
        if !diags.has_errors() {
            let mut checker = cjc_types::TypeChecker::new();
            checker.check_program(&prog);
        }
    }

    /// The full MIR pipeline (parse -> lower -> exec) never panics on well-formed programs.
    #[test]
    fn mir_pipeline_never_panics_on_valid(src in arb_fn_call_program()) {
        let (prog, diags) = cjc_parser::parse_source(&src);
        if !diags.has_errors() {
            // If it parses, running it should not panic (errors are OK).
            let _ = cjc_mir_exec::run_program_with_executor(&prog, 42);
        }
    }
}
