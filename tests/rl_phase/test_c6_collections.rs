//! Phase C test C6: I/O & Collection Utilities
//!
//! Note: read_line is not tested here since it requires stdin.
//! It is registered and dispatched but we skip interactive testing.

fn run_mir(src: &str) -> Vec<String> {
    let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

#[test]
fn c6_push_basic() {
    let out = run_mir(r#"
let arr = [1, 2, 3];
let arr2 = array_push(arr, 4);
print(array_len(arr2));
"#);
    assert_eq!(out, vec!["4"]);
}

#[test]
fn c6_pop_basic() {
    let out = run_mir(r#"
let arr = [10, 20, 30];
let result = array_pop(arr);
print(result);
"#);
    let s = &out[0];
    // Tuple: (last_element, remaining_array)
    assert!(s.contains("30"), "should contain popped element 30, got {s}");
}

#[test]
fn c6_pop_empty_error() {
    let result = std::panic::catch_unwind(|| {
        run_mir(r#"
let arr = [];
let result = array_pop(arr);
"#);
    });
    // Should panic/error for empty array
    assert!(result.is_err());
}

#[test]
fn c6_contains_found() {
    let out = run_mir(r#"
let arr = [1, 2, 3, 4, 5];
print(array_contains(arr, 3));
"#);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn c6_contains_missing() {
    let out = run_mir(r#"
let arr = [1, 2, 3];
print(array_contains(arr, 99));
"#);
    assert_eq!(out, vec!["false"]);
}

#[test]
fn c6_reverse_basic() {
    let out = run_mir(r#"
let arr = [1, 2, 3];
let rev = array_reverse(arr);
print(rev);
"#);
    assert_eq!(out, vec!["[3, 2, 1]"]);
}

#[test]
fn c6_flatten_nested() {
    let out = run_mir(r#"
let arr = [[1, 2], [3, [4, 5]]];
let flat = array_flatten(arr);
print(flat);
"#);
    assert_eq!(out, vec!["[1, 2, 3, 4, 5]"]);
}

#[test]
fn c6_slice_range() {
    let out = run_mir(r#"
let arr = [10, 20, 30, 40, 50];
let s = array_slice(arr, 1, 4);
print(s);
"#);
    assert_eq!(out, vec!["[20, 30, 40]"]);
}

#[test]
fn c6_array_len() {
    let out = run_mir(r#"
let arr = [1, 2, 3, 4];
print(array_len(arr));
"#);
    assert_eq!(out, vec!["4"]);
}

#[test]
fn c6_determinism_array_ops() {
    let src = r#"
let arr = [5, 3, 1, 4, 2];
let arr2 = array_push(arr, 6);
let rev = array_reverse(arr2);
let s = array_slice(rev, 1, 4);
print(s);
"#;
    let out1 = run_mir(src);
    let out2 = run_mir(src);
    assert_eq!(out1, out2);
}

#[test]
fn c6_push_multiple() {
    let out = run_mir(r#"
let arr = [];
let arr = array_push(arr, 1);
let arr = array_push(arr, 2);
let arr = array_push(arr, 3);
print(array_len(arr));
print(arr);
"#);
    assert_eq!(out[0], "3");
    assert_eq!(out[1], "[1, 2, 3]");
}

#[test]
fn c6_contains_string() {
    let out = run_mir(r#"
let arr = ["hello", "world"];
print(array_contains(arr, "hello"));
print(array_contains(arr, "missing"));
"#);
    assert_eq!(out[0], "true");
    assert_eq!(out[1], "false");
}
