//! Property-based tests for the CJC parser.
//!
//! These tests verify that the parser never panics on arbitrary input,
//! including both well-formed and malformed source strings.

use proptest::prelude::*;

/// Strategy that generates random ASCII strings (potential CJC source).
fn arb_ascii_source() -> impl Strategy<Value = String> {
    proptest::collection::vec(0u8..128, 0..200)
        .prop_map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
}

/// Strategy that generates syntactically plausible (but possibly invalid) CJC snippets.
fn arb_plausible_source() -> impl Strategy<Value = String> {
    let keywords = prop_oneof![
        Just("fn".to_string()),
        Just("let".to_string()),
        Just("let mut".to_string()),
        Just("if".to_string()),
        Just("else".to_string()),
        Just("for".to_string()),
        Just("in".to_string()),
        Just("return".to_string()),
        Just("match".to_string()),
        Just("struct".to_string()),
        Just("enum".to_string()),
        Just("true".to_string()),
        Just("false".to_string()),
        Just("print".to_string()),
    ];
    let idents = prop_oneof![
        Just("x".to_string()),
        Just("y".to_string()),
        Just("foo".to_string()),
        Just("bar".to_string()),
        Just("main".to_string()),
        Just("i64".to_string()),
        Just("f64".to_string()),
        Just("str".to_string()),
        Just("bool".to_string()),
    ];
    let punctuation = prop_oneof![
        Just("(".to_string()),
        Just(")".to_string()),
        Just("{".to_string()),
        Just("}".to_string()),
        Just("[".to_string()),
        Just("]".to_string()),
        Just(":".to_string()),
        Just(";".to_string()),
        Just(",".to_string()),
        Just("->".to_string()),
        Just("=".to_string()),
        Just("+".to_string()),
        Just("-".to_string()),
        Just("*".to_string()),
        Just("..".to_string()),
    ];
    let literals = prop_oneof![
        (0i64..1000).prop_map(|n| n.to_string()),
        Just("\"hello\"".to_string()),
        Just("3.14".to_string()),
    ];

    let token = prop_oneof![keywords, idents, punctuation, literals,];

    proptest::collection::vec(token, 1..30).prop_map(|tokens| tokens.join(" "))
}

/// Strategy that generates valid minimal CJC programs.
fn arb_valid_program() -> impl Strategy<Value = String> {
    let body_expr = prop_oneof![
        (1i64..1000).prop_map(|n| format!("{}", n)),
        Just("true".to_string()),
        Just("false".to_string()),
        Just("3.14".to_string()),
        Just("\"hello\"".to_string()),
    ];
    let ret_type = prop_oneof![
        Just("i64".to_string()),
        Just("f64".to_string()),
        Just("bool".to_string()),
        Just("str".to_string()),
    ];

    (ret_type, body_expr).prop_map(|(ty, body)| {
        format!("fn main() -> {} {{ {} }}", ty, body)
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// The parser must never panic on arbitrary ASCII input.
    #[test]
    fn parser_never_panics_on_random_ascii(src in arb_ascii_source()) {
        // We only care that it doesn't panic — errors are fine.
        let _ = cjc_parser::parse_source(&src);
    }

    /// The parser must never panic on plausible-but-possibly-invalid CJC tokens.
    #[test]
    fn parser_never_panics_on_plausible_tokens(src in arb_plausible_source()) {
        let _ = cjc_parser::parse_source(&src);
    }

    /// Valid minimal programs parse without errors.
    #[test]
    fn valid_programs_parse_successfully(src in arb_valid_program()) {
        let (_, diags) = cjc_parser::parse_source(&src);
        prop_assert!(
            !diags.has_errors(),
            "Valid program should parse without errors: {}\nErrors: {:?}",
            src,
            diags.diagnostics
        );
    }
}
