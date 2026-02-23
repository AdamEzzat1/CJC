// Milestone 2.6 — Eval vs MIR-Exec Parity Tests
//
// Runs the same CJC source programs through both the tree-walk evaluator
// (cjc_eval) and the MIR executor (cjc_mir_exec) and verifies that both
// produce identical results (output strings and return values).
//
// CJC uses bare variant names: `Some(42)`, `None`, `Circle(3.14)`.

use cjc_runtime::Value;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn eval_source(src: &str) -> Vec<String> {
    let lexer = cjc_lexer::Lexer::new(src);
    let (tokens, _) = lexer.tokenize();
    let parser = cjc_parser::Parser::new(tokens);
    let (program, _) = parser.parse_program();
    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(&program).expect("eval failed");
    interp.output
}

fn mir_source(src: &str) -> Vec<String> {
    let lexer = cjc_lexer::Lexer::new(src);
    let (tokens, _) = lexer.tokenize();
    let parser = cjc_parser::Parser::new(tokens);
    let (program, _) = parser.parse_program();
    let (_, executor) = cjc_mir_exec::run_program_optimized_with_executor(&program, 42)
        .expect("mir-exec failed");
    executor.output
}

fn eval_result(src: &str) -> Value {
    let lexer = cjc_lexer::Lexer::new(src);
    let (tokens, _) = lexer.tokenize();
    let parser = cjc_parser::Parser::new(tokens);
    let (program, _) = parser.parse_program();
    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(&program).expect("eval failed")
}

fn mir_result(src: &str) -> Value {
    let lexer = cjc_lexer::Lexer::new(src);
    let (tokens, _) = lexer.tokenize();
    let parser = cjc_parser::Parser::new(tokens);
    let (program, _) = parser.parse_program();
    let (val, _) = cjc_mir_exec::run_program_optimized_with_executor(&program, 42)
        .expect("mir-exec failed");
    val
}

/// Assert that both pipelines produce the same output lines.
fn assert_parity_output(src: &str) {
    let eval_out = eval_source(src);
    let mir_out = mir_source(src);
    assert_eq!(
        eval_out, mir_out,
        "parity mismatch:\n  eval output: {:?}\n  mir  output: {:?}",
        eval_out, mir_out
    );
    // Also verify output is non-empty to catch silent failures
    assert!(!eval_out.is_empty(), "both pipelines produced empty output -- likely a parse error");
}

/// Assert that both pipelines produce the same return value (stringified).
fn assert_parity_result(src: &str) {
    let eval_val = eval_result(src);
    let mir_val = mir_result(src);
    let eval_str = format!("{}", eval_val);
    let mir_str = format!("{}", mir_val);
    assert_eq!(
        eval_str, mir_str,
        "parity mismatch:\n  eval result: {}\n  mir  result: {}",
        eval_str, mir_str
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn parity_some_42_result() {
    let src = r#"
        fn main() {
            Some(42)
        }
    "#;
    assert_parity_result(src);
}

#[test]
fn parity_none_result() {
    let src = r#"
        fn main() {
            None
        }
    "#;
    assert_parity_result(src);
}

#[test]
fn parity_match_option_output() {
    let src = r#"
        fn main() {
            let x = Some(10);
            match x {
                Some(v) => print(v),
                None => print("none"),
            }
        }
    "#;
    assert_parity_output(src);
}

#[test]
fn parity_enum_payload_construction_and_match() {
    let src = r#"
        enum Shape { Circle(f64), Rect(f64, f64) }
        fn area(s: Shape) -> f64 {
            match s {
                Circle(r) => r * r,
                Rect(w, h) => w * h,
            }
        }
        fn main() {
            print(area(Circle(3.0)));
            print(area(Rect(4.0, 5.0)));
        }
    "#;
    assert_parity_output(src);
}

#[test]
fn parity_enum_wildcard_match() {
    let src = r#"
        enum Shape { Circle(f64), Rect(f64, f64) }
        fn main() {
            let s = Rect(1.0, 2.0);
            match s {
                Circle(r) => print("circle"),
                _ => print("not circle"),
            }
        }
    "#;
    assert_parity_output(src);
}
