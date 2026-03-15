//! String manipulation builtin tests.
//! These test the new str_* family of functions added in the hardening pass.

use cjc_runtime::{Value, builtins::dispatch_builtin};
use std::rc::Rc;

fn s(val: &str) -> Value {
    Value::String(Rc::new(val.to_string()))
}

#[test]
fn test_str_upper() {
    let result = dispatch_builtin("str_upper", &[s("hello")]).unwrap().unwrap();
    assert_eq!(result, s("HELLO"));
}

#[test]
fn test_str_lower() {
    let result = dispatch_builtin("str_lower", &[s("HELLO")]).unwrap().unwrap();
    assert_eq!(result, s("hello"));
}

#[test]
fn test_str_trim() {
    let result = dispatch_builtin("str_trim", &[s("  hello  ")]).unwrap().unwrap();
    assert_eq!(result, s("hello"));
}

#[test]
fn test_str_contains_true() {
    let result = dispatch_builtin("str_contains", &[s("hello world"), s("world")]).unwrap().unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_str_contains_false() {
    let result = dispatch_builtin("str_contains", &[s("hello"), s("xyz")]).unwrap().unwrap();
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_str_replace() {
    let result = dispatch_builtin("str_replace", &[s("hello world"), s("world"), s("CJC")]).unwrap().unwrap();
    assert_eq!(result, s("hello CJC"));
}

#[test]
fn test_str_split() {
    let result = dispatch_builtin("str_split", &[s("a,b,c"), s(",")]).unwrap().unwrap();
    match result {
        Value::Array(arr) => {
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[0], s("a"));
            assert_eq!(arr[1], s("b"));
            assert_eq!(arr[2], s("c"));
        }
        _ => panic!("expected array"),
    }
}

#[test]
fn test_str_join() {
    let arr = Value::Array(Rc::new(vec![s("a"), s("b"), s("c")]));
    let result = dispatch_builtin("str_join", &[arr, s("-")]).unwrap().unwrap();
    assert_eq!(result, s("a-b-c"));
}

#[test]
fn test_str_starts_with() {
    let result = dispatch_builtin("str_starts_with", &[s("hello"), s("hel")]).unwrap().unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_str_ends_with() {
    let result = dispatch_builtin("str_ends_with", &[s("hello"), s("llo")]).unwrap().unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_str_repeat() {
    let result = dispatch_builtin("str_repeat", &[s("ab"), Value::Int(3)]).unwrap().unwrap();
    assert_eq!(result, s("ababab"));
}

#[test]
fn test_str_chars() {
    let result = dispatch_builtin("str_chars", &[s("abc")]).unwrap().unwrap();
    match result {
        Value::Array(arr) => {
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[0], s("a"));
            assert_eq!(arr[1], s("b"));
            assert_eq!(arr[2], s("c"));
        }
        _ => panic!("expected array"),
    }
}

#[test]
fn test_str_substr() {
    let result = dispatch_builtin("str_substr", &[s("hello world"), Value::Int(6), Value::Int(5)]).unwrap().unwrap();
    assert_eq!(result, s("world"));
}

#[test]
fn test_str_builtins_in_eval() {
    let src = r#"
let x = str_upper("hello");
x
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program).unwrap();
    match result {
        Value::String(s) => assert_eq!(s.as_str(), "HELLO"),
        _ => panic!("expected string, got {:?}", result),
    }
}

#[test]
fn test_str_builtins_parity() {
    let src = r#"
let x = str_replace("foo bar baz", "bar", "qux");
x
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());

    let mut interp = cjc_eval::Interpreter::new(42);
    let eval_result = interp.exec(&program).unwrap();

    let (mir_result, _) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();

    assert_eq!(format!("{}", eval_result), format!("{}", mir_result),
        "eval and MIR must agree on str_replace");
}
