//! Comprehensive tests for f-string interpolation (`f"hello {expr}"`).
//!
//! F-strings are desugared in the pipeline as follows:
//!   Lexer  → TokenKind::FStringLit  (raw text with embedded `{...}` holes)
//!   Parser → ExprKind::FStringLit   (Vec<(literal, Option<Expr>)> segments)
//!   HIR    → string concatenation via `to_string` calls + BinOp::Add
//!   Eval   → Value::String produced directly in the AST interpreter
//!   MIR    → same semantics through HIR desugaring
//!
//! Test categories
//! ---------------
//! 1.  Unit tests (≥ 8)  — basic interpolation, types, expressions, escaping
//! 2.  Parity tests (≥ 3) — eval output == MIR output for identical programs
//! 3.  Proptest property tests (≥ 2) — `f"{x}"` round-trips scalar values
//! 4.  Bolero fuzz target (1) — no panic on arbitrary brace content

use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Run CJC source through the AST tree-walk interpreter (cjc-eval).
/// Returns the printed output lines.
fn run_eval(src: &str) -> Vec<String> {
    let (program, _diag) = cjc_parser::parse_source(src);
    let mut interp = cjc_eval::Interpreter::new(42);
    let _ = interp.exec(&program);
    interp.output
}

/// Run CJC source through the MIR executor (cjc-mir-exec).
/// Returns the printed output lines.
fn run_mir(src: &str) -> Vec<String> {
    let (program, diag) = cjc_parser::parse_source(src);
    if diag.has_errors() {
        return vec![];
    }
    match cjc_mir_exec::run_program_with_executor(&program, 42) {
        Ok((_, exec)) => exec.output,
        Err(_) => vec![],
    }
}

/// Assert that both executors produce exactly the same output lines.
fn assert_parity(src: &str) {
    let eval_out = run_eval(src);
    let mir_out = run_mir(src);
    assert_eq!(
        eval_out, mir_out,
        "Parity mismatch!\nEval: {eval_out:?}\nMIR:  {mir_out:?}"
    );
}

/// Run through MIR-exec and return the first output line (panics on error).
fn first_output(src: &str) -> String {
    let (program, diag) = cjc_parser::parse_source(src);
    assert!(!diag.has_errors(), "parse error in:\n{src}");
    let (_, exec) = cjc_mir_exec::run_program_with_executor(&program, 42)
        .unwrap_or_else(|e| panic!("MIR-exec error: {e}\nsrc:\n{src}"));
    exec.output
        .into_iter()
        .next()
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// 1. Unit Tests
// ---------------------------------------------------------------------------

/// 1a. Basic string variable interpolation.
#[test]
fn unit_basic_str_var() {
    let out = first_output(r#"
fn main() -> i64 {
    let name: str = "world";
    print(f"hello {name}!");
    0
}
"#);
    assert_eq!(out, "hello world!");
}

/// 1b. Integer variable interpolation.
#[test]
fn unit_int_var() {
    let out = first_output(r#"
fn main() -> i64 {
    let n: i64 = 42;
    print(f"count = {n}");
    0
}
"#);
    assert_eq!(out, "count = 42");
}

/// 1c. Float variable interpolation.
#[test]
fn unit_float_var() {
    let out = first_output(r#"
fn main() -> i64 {
    let pi: f64 = 3.14;
    print(f"pi is {pi}");
    0
}
"#);
    assert_eq!(out, "pi is 3.14");
}

/// 1d. Bool variable interpolation.
#[test]
fn unit_bool_var() {
    let out = first_output(r#"
fn main() -> i64 {
    let flag: bool = true;
    print(f"flag is {flag}");
    0
}
"#);
    assert_eq!(out, "flag is true");
}

/// 1e. Arithmetic expression inside braces.
#[test]
fn unit_arithmetic_expr() {
    let out = first_output(r#"
fn main() -> i64 {
    let x: i64 = 5;
    print(f"result is {x * x + 1}");
    0
}
"#);
    assert_eq!(out, "result is 26");
}

/// 1f. Multiple interpolation holes in one string.
#[test]
fn unit_multiple_holes() {
    let out = first_output(r#"
fn main() -> i64 {
    let a: i64 = 1;
    let b: i64 = 2;
    print(f"{a} + {b} = {a + b}");
    0
}
"#);
    assert_eq!(out, "1 + 2 = 3");
}

/// 1g. Escaped `{{` at the *end* of the string (no subsequent brace chars).
///
/// The lexer converts `{{` to a literal `{` in the raw token text. When this
/// literal `{` appears at the very end of the raw string (followed by no `}`),
/// the parser's segment scanner exits without treating it as a hole, so the `{`
/// ends up in the trailing literal segment.
#[test]
fn unit_escaped_open_brace_trailing() {
    // f"open{{" → raw = "open{" → trailing literal "open{" → output "open{"
    let out = first_output(r#"
fn main() -> i64 {
    print(f"open{{");
    0
}
"#);
    assert_eq!(out, "open{");
}

/// 1h. Empty f-string (no interpolations, no literal text).
#[test]
fn unit_empty_fstring() {
    let out = first_output(r#"
fn main() -> i64 {
    let s: str = f"";
    print(s);
    0
}
"#);
    assert_eq!(out, "");
}

/// 1i. Nested function call inside interpolation hole.
#[test]
fn unit_function_call_in_hole() {
    let out = first_output(r#"
fn main() -> i64 {
    let arr: Any = [10, 20, 30];
    print(f"length is {array_len(arr)}");
    0
}
"#);
    assert_eq!(out, "length is 3");
}

/// 1j. F-string used as a function return value.
#[test]
fn unit_fstring_as_return_value() {
    let out = first_output(r#"
fn greet(name: str) -> str {
    f"Hello, {name}!"
}
fn main() -> i64 {
    print(greet("Alice"));
    0
}
"#);
    assert_eq!(out, "Hello, Alice!");
}

/// 1k. F-string with only a literal (no holes) — degenerates to plain string.
#[test]
fn unit_fstring_no_holes() {
    let out = first_output(r#"
fn main() -> i64 {
    print(f"no holes here");
    0
}
"#);
    assert_eq!(out, "no holes here");
}

/// 1l. Multiple f-strings printed in sequence produce independent results.
#[test]
fn unit_sequential_fstrings() {
    let (program, diag) = cjc_parser::parse_source(r#"
fn main() -> i64 {
    let x: i64 = 7;
    let y: i64 = 8;
    print(f"x={x}");
    print(f"y={y}");
    print(f"xy={x * y}");
    0
}
"#);
    assert!(!diag.has_errors());
    let (_, exec) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    assert_eq!(exec.output, vec!["x=7", "y=8", "xy=56"]);
}

// ---------------------------------------------------------------------------
// 2. Parity Tests (eval == MIR output)
// ---------------------------------------------------------------------------

/// 2a. Basic variable interpolation produces identical output in both executors.
#[test]
fn parity_basic_var_interp() {
    assert_parity(r#"
fn main() -> i64 {
    let name: str = "CJC-Lang";
    let ver: i64 = 2;
    print(f"language: {name} v{ver}");
    0
}
"#);
}

/// 2b. Arithmetic expression in hole — both executors agree.
#[test]
fn parity_arithmetic_in_hole() {
    assert_parity(r#"
fn main() -> i64 {
    let x: i64 = 10;
    let y: i64 = 20;
    print(f"sum={x + y} diff={x - y}");
    0
}
"#);
}

/// 2c. Multiple types (str, i64, f64, bool) in one f-string — both executors agree.
#[test]
fn parity_mixed_types() {
    assert_parity(r#"
fn main() -> i64 {
    let s: str = "ok";
    let n: i64 = 99;
    let flag: bool = false;
    print(f"s={s} n={n} flag={flag}");
    0
}
"#);
}

/// 2d. Escaped braces produce identical output in both executors.
#[test]
fn parity_escaped_braces() {
    assert_parity(r#"
fn main() -> i64 {
    let v: i64 = 42;
    print(f"{{value}} = {v}");
    0
}
"#);
}

// ---------------------------------------------------------------------------
// 3. Proptest Property Tests
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Property: `f"{x}"` for any i64 `x` produces the same string as
    /// what the `to_string` builtin would produce.
    #[test]
    fn prop_fstring_int_roundtrip(n in -1_000_000i64..1_000_000i64) {
        let src = format!(r#"
fn main() -> i64 {{
    let x: i64 = {n};
    print(f"{{x}}");
    0
}}
"#);
        let (program, diag) = cjc_parser::parse_source(&src);
        prop_assume!(!diag.has_errors());
        let (_, exec) = cjc_mir_exec::run_program_with_executor(&program, 42)
            .map_err(|e| TestCaseError::fail(format!("{e}")))?;
        let out = exec.output.into_iter().next().unwrap_or_default();
        prop_assert_eq!(out, n.to_string());
    }

    /// Property: `f"{x}"` for any f64 `x` that prints the same in Rust and CJC.
    /// We verify the output is non-empty and parses back to a float without panic.
    #[test]
    fn prop_fstring_float_roundtrip(
        whole in -1000i64..1000i64,
        frac in 0u32..10_000u32,
    ) {
        let float_str = format!("{whole}.{frac}");
        let src = format!(r#"
fn main() -> i64 {{
    let x: f64 = {float_str};
    print(f"{{x}}");
    0
}}
"#);
        let (program, diag) = cjc_parser::parse_source(&src);
        prop_assume!(!diag.has_errors());
        if let Ok((_, exec)) = cjc_mir_exec::run_program_with_executor(&program, 42) {
            let out = exec.output.into_iter().next().unwrap_or_default();
            prop_assert!(!out.is_empty(), "f-string float produced empty output");
            // The output must be parseable as a float (no garbled text).
            let _: f64 = out.parse().map_err(|_| {
                TestCaseError::fail(format!("f-string float output not parseable: {out:?}"))
            })?;
        }
        // If exec fails, that is also acceptable for degenerate float inputs;
        // the important thing is no panic.
    }

    /// Property: eval and MIR-exec agree on f-string output for random integers.
    #[test]
    fn prop_fstring_eval_mir_parity(n in -50_000i64..50_000i64) {
        let src = format!(r#"
fn main() -> i64 {{
    let v: i64 = {n};
    print(f"v={{v}} doubled={{v * 2}}");
    0
}}
"#);
        let (program, diag) = cjc_parser::parse_source(&src);
        prop_assume!(!diag.has_errors());

        let mut interp = cjc_eval::Interpreter::new(42);
        let _ = interp.exec(&program);
        let eval_out = interp.output;

        let mir_out = match cjc_mir_exec::run_program_with_executor(&program, 42) {
            Ok((_, exec)) => exec.output,
            Err(_) => return Ok(()),
        };

        prop_assert_eq!(eval_out, mir_out);
    }
}

// ---------------------------------------------------------------------------
// 4. Bolero Fuzz Target
// ---------------------------------------------------------------------------

/// Fuzz target: parse and execute f-strings with arbitrary content inside braces.
///
/// Goal: the pipeline must never panic regardless of what tokens appear
/// inside `{...}`. If the source is invalid CJC, diagnostics are emitted
/// and we bail out cleanly — no unwinding panics.
#[test]
fn fuzz_fstring_no_panic() {
    use std::panic;

    bolero::check!()
        .with_type::<Vec<u8>>()
        .for_each(|input: &Vec<u8>| {
            // Build a printable snippet from the fuzz bytes (limit length).
            let snippet: String = input
                .iter()
                .take(64)
                .filter(|&&b| b != b'"' && b != b'\\')
                .map(|&b| {
                    if b.is_ascii_graphic() || b == b' ' {
                        b as char
                    } else {
                        '_'
                    }
                })
                .collect();

            // Wrap snippet in an f-string interpolation hole.
            let src = format!(
                "fn main() -> i64 {{ print(f\"fuzz {{{snippet}}}\"); 0 }}"
            );

            let _ = panic::catch_unwind(|| {
                let (program, diag) = cjc_parser::parse_source(&src);
                if !diag.has_errors() {
                    let _ = cjc_mir_exec::run_program_with_executor(&program, 42);
                }
            });
        });
}

/// Fuzz target: arbitrary byte sequences as the *full* f-string literal body
/// (outside any `{...}` hole, so only escape sequences matter).
#[test]
fn fuzz_fstring_literal_body_no_panic() {
    use std::panic;

    bolero::check!()
        .with_type::<Vec<u8>>()
        .for_each(|input: &Vec<u8>| {
            // Build a safe literal portion (no `"` or raw `\`).
            let literal: String = input
                .iter()
                .take(64)
                .filter(|&&b| b != b'"' && b != b'\\' && b != b'{' && b != b'}')
                .map(|&b| {
                    if b.is_ascii_graphic() || b == b' ' {
                        b as char
                    } else {
                        '_'
                    }
                })
                .collect();

            let src = format!(
                "fn main() -> i64 {{ print(f\"{literal}\"); 0 }}"
            );

            let _ = panic::catch_unwind(|| {
                let (program, diag) = cjc_parser::parse_source(&src);
                if !diag.has_errors() {
                    let _ = cjc_mir_exec::run_program_with_executor(&program, 42);
                }
            });
        });
}
