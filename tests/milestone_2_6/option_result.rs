// Milestone 2.6 — Option / Result Prelude Types and ? Operator
//
// Tests for Option<T> (Some / None), Result<T, E> (Ok / Err),
// pattern matching on these types, and the `?` (try) operator.

use cjc_runtime::Value;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn run_eval(source: &str) -> cjc_eval::EvalResult {
    let lexer = cjc_lexer::Lexer::new(source);
    let (tokens, _) = lexer.tokenize();
    let parser = cjc_parser::Parser::new(tokens);
    let (program, _) = parser.parse_program();
    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(&program)
}

fn run_eval_output(source: &str) -> Vec<String> {
    let lexer = cjc_lexer::Lexer::new(source);
    let (tokens, _) = lexer.tokenize();
    let parser = cjc_parser::Parser::new(tokens);
    let (program, _) = parser.parse_program();
    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(&program).expect("eval failed");
    interp.output
}

// ---------------------------------------------------------------------------
// Option construction
// ---------------------------------------------------------------------------

#[test]
fn opt_some_constructs_option_enum() {
    let src = r#"
        fn main() {
            Some(42)
        }
    "#;
    let result = run_eval(src).expect("eval failed");
    match result {
        Value::Enum { enum_name, variant, fields } => {
            assert_eq!(enum_name, "Option");
            assert_eq!(variant, "Some");
            assert_eq!(fields.len(), 1);
            assert!(matches!(fields[0], Value::Int(42)));
        }
        other => panic!("expected Option::Some(42), got {:?}", other),
    }
}

#[test]
fn opt_none_constructs_option_enum() {
    let src = r#"
        fn main() {
            None
        }
    "#;
    let result = run_eval(src).expect("eval failed");
    match result {
        Value::Enum { enum_name, variant, fields } => {
            assert_eq!(enum_name, "Option");
            assert_eq!(variant, "None");
            assert!(fields.is_empty());
        }
        other => panic!("expected Option::None, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Result construction
// ---------------------------------------------------------------------------

#[test]
fn res_ok_constructs_result_enum() {
    let src = r#"
        fn main() {
            Ok(42)
        }
    "#;
    let result = run_eval(src).expect("eval failed");
    match result {
        Value::Enum { enum_name, variant, fields } => {
            assert_eq!(enum_name, "Result");
            assert_eq!(variant, "Ok");
            assert_eq!(fields.len(), 1);
            assert!(matches!(fields[0], Value::Int(42)));
        }
        other => panic!("expected Result::Ok(42), got {:?}", other),
    }
}

#[test]
fn res_err_constructs_result_enum() {
    let src = r#"
        fn main() {
            Err("fail")
        }
    "#;
    let result = run_eval(src).expect("eval failed");
    match result {
        Value::Enum { enum_name, variant, fields } => {
            assert_eq!(enum_name, "Result");
            assert_eq!(variant, "Err");
            assert_eq!(fields.len(), 1);
            match &fields[0] {
                Value::String(s) => assert_eq!(s.as_str(), "fail"),
                other => panic!("expected String field, got {:?}", other),
            }
        }
        other => panic!("expected Result::Err, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Match on Option
// ---------------------------------------------------------------------------

#[test]
fn opt_match_some_arm() {
    let src = r#"
        fn main() -> i64 {
            let opt = Some(10);
            match opt {
                Some(x) => x + 1,
                None => 0,
            }
        }
    "#;
    let result = run_eval(src).expect("eval failed");
    assert!(matches!(result, Value::Int(11)));
}

#[test]
fn opt_match_none_arm() {
    let src = r#"
        fn main() -> i64 {
            let opt = None;
            match opt {
                Some(x) => x,
                None => -1,
            }
        }
    "#;
    let result = run_eval(src).expect("eval failed");
    assert!(matches!(result, Value::Int(-1)));
}

// ---------------------------------------------------------------------------
// Match on Result
// ---------------------------------------------------------------------------

#[test]
fn res_match_ok_arm() {
    let src = r#"
        fn main() -> i64 {
            let r = Ok(100);
            match r {
                Ok(v) => v,
                Err(e) => -1,
            }
        }
    "#;
    let result = run_eval(src).expect("eval failed");
    assert!(matches!(result, Value::Int(100)));
}

#[test]
fn res_match_err_arm() {
    let src = r#"
        fn main() {
            let r = Err("oops");
            match r {
                Ok(v) => print("ok"),
                Err(e) => print(e),
            }
        }
    "#;
    let output = run_eval_output(src);
    assert_eq!(output.len(), 1);
    assert_eq!(output[0], "oops");
}

// ---------------------------------------------------------------------------
// ? operator
// ---------------------------------------------------------------------------

#[test]
fn try_operator_ok_returns_inner() {
    let src = r#"
        fn try_it() -> Result {
            let val = Ok(42)?;
            Ok(val + 1)
        }
        fn main() -> i64 {
            match try_it() {
                Ok(v) => v,
                Err(e) => -1,
            }
        }
    "#;
    let result = run_eval(src).expect("eval failed");
    assert!(matches!(result, Value::Int(43)));
}

#[test]
fn try_operator_err_propagates() {
    let src = r#"
        fn try_it() -> Result {
            let val = Err("bad")?;
            Ok(val)
        }
        fn main() {
            match try_it() {
                Ok(v) => print("ok"),
                Err(e) => print(e),
            }
        }
    "#;
    let output = run_eval_output(src);
    assert_eq!(output.len(), 1);
    assert_eq!(output[0], "bad");
}
