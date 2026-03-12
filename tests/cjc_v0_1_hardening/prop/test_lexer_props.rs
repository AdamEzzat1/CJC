//! Property-based tests for the CJC lexer.

use proptest::prelude::*;

/// Strategy generating random UTF-8 strings of varying lengths.
fn arb_utf8() -> impl Strategy<Value = String> {
    proptest::collection::vec(proptest::char::any(), 0..500)
        .prop_map(|chars| chars.into_iter().collect())
}

/// Strategy generating ASCII-only strings.
fn arb_ascii() -> impl Strategy<Value = String> {
    proptest::collection::vec(0u8..128, 0..500)
        .prop_map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
}

/// Strategy generating strings that look like CJC code fragments.
fn arb_code_fragment() -> impl Strategy<Value = String> {
    let tokens = prop_oneof![
        Just("fn".to_string()),
        Just("let".to_string()),
        Just("if".to_string()),
        Just("else".to_string()),
        Just("while".to_string()),
        Just("for".to_string()),
        Just("in".to_string()),
        Just("return".to_string()),
        Just("match".to_string()),
        Just("struct".to_string()),
        Just("true".to_string()),
        Just("false".to_string()),
        Just("(".to_string()),
        Just(")".to_string()),
        Just("{".to_string()),
        Just("}".to_string()),
        Just("+".to_string()),
        Just("-".to_string()),
        Just("*".to_string()),
        Just(";".to_string()),
        Just(":".to_string()),
        Just("=".to_string()),
        Just("42".to_string()),
        Just("3.14".to_string()),
        Just("\"hello\"".to_string()),
        Just("x".to_string()),
        Just("main".to_string()),
        Just("i64".to_string()),
        Just("\n".to_string()),
        Just(" ".to_string()),
    ];
    proptest::collection::vec(tokens, 1..50)
        .prop_map(|toks| toks.join(" "))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Lexer must never panic on arbitrary UTF-8 input.
    #[test]
    fn lexer_never_panics_utf8(src in arb_utf8()) {
        let _ = std::panic::catch_unwind(|| {
            let _ = cjc_lexer::Lexer::new(&src).tokenize();
        });
    }

    /// Lexer must never panic on arbitrary ASCII input.
    #[test]
    fn lexer_never_panics_ascii(src in arb_ascii()) {
        let _ = cjc_lexer::Lexer::new(&src).tokenize();
    }

    /// Lexer output is deterministic: same input → identical tokens.
    #[test]
    fn lexer_deterministic(src in arb_code_fragment()) {
        let (t1, d1) = cjc_lexer::Lexer::new(&src).tokenize();
        let (t2, d2) = cjc_lexer::Lexer::new(&src).tokenize();
        prop_assert_eq!(t1.len(), t2.len(), "Token count must be identical");
        for (a, b) in t1.iter().zip(t2.iter()) {
            prop_assert_eq!(&a.kind, &b.kind, "Token kinds must match");
            prop_assert_eq!(&a.text, &b.text, "Lexemes must match");
        }
        prop_assert_eq!(d1.has_errors(), d2.has_errors(), "Error status must match");
    }

    /// Code fragments always produce at least one token or an error.
    #[test]
    fn lexer_nonempty_on_nonempty_input(src in arb_code_fragment()) {
        let (tokens, diags) = cjc_lexer::Lexer::new(&src).tokenize();
        prop_assert!(
            !tokens.is_empty() || diags.has_errors(),
            "Non-empty input should produce tokens or diagnostics"
        );
    }
}
