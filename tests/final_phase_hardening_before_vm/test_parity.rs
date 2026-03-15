//! Parity tests: AST-eval vs MIR-exec must agree on all features.

fn run_parity(src: &str) -> (String, String) {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);

    let mut interp = cjc_eval::Interpreter::new(42);
    let eval_result = interp.exec(&program).unwrap();
    let eval_str = format!("{}", eval_result);

    let (mir_result, _) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    let mir_str = format!("{}", mir_result);

    (eval_str, mir_str)
}

fn assert_parity(src: &str) {
    let (eval_str, mir_str) = run_parity(src);
    assert_eq!(eval_str, mir_str, "parity failure:\n  eval: {}\n  mir:  {}", eval_str, mir_str);
}

#[test]
fn test_parity_string_upper() {
    assert_parity(r#"str_upper("hello")"#);
}

#[test]
fn test_parity_string_lower() {
    assert_parity(r#"str_lower("WORLD")"#);
}

#[test]
fn test_parity_string_trim() {
    assert_parity(r#"str_trim("  abc  ")"#);
}

#[test]
fn test_parity_string_contains() {
    assert_parity(r#"str_contains("hello world", "world")"#);
}

#[test]
fn test_parity_string_replace() {
    assert_parity(r#"str_replace("foo bar", "bar", "baz")"#);
}

#[test]
fn test_parity_string_starts_with() {
    assert_parity(r#"str_starts_with("hello", "hel")"#);
}

#[test]
fn test_parity_string_ends_with() {
    assert_parity(r#"str_ends_with("hello", "llo")"#);
}

#[test]
fn test_parity_string_repeat() {
    assert_parity(r#"str_repeat("ab", 3)"#);
}

#[test]
fn test_parity_if_expression() {
    assert_parity(r#"
let x = if true { 42 } else { 0 };
x
"#);
}

#[test]
fn test_parity_variadic_function() {
    assert_parity(r#"
fn total(...nums: f64) -> f64 {
    let s = 0.0;
    let i = 0;
    while i < len(nums) {
        s = s + array_get(nums, i);
        i = i + 1;
    }
    s
}
total(1.0, 2.0, 3.0, 4.0)
"#);
}

#[test]
fn test_parity_default_params() {
    assert_parity(r#"
fn greet(name: str, greeting: str = "Hello") -> str {
    str_join([greeting, name], " ")
}
greet("World")
"#);
}

#[test]
fn test_parity_struct_method() {
    assert_parity(r#"
struct Counter { value: i64 }
impl Counter {
    fn get(self: Counter) -> i64 { self.value }
}
let c = Counter { value: 99 };
c.get()
"#);
}

#[test]
fn test_parity_nested_if_expr() {
    assert_parity(r#"
let x = 5;
let result = if x > 10 {
    "big"
} else {
    if x > 3 { "medium" } else { "small" }
};
result
"#);
}

#[test]
fn test_parity_fstring() {
    assert_parity(r#"
let name = "CJC";
let version = 1;
f"Language: {name}, version: {version}"
"#);
}

#[test]
fn test_parity_deterministic_rng() {
    // Same seed must produce identical results in both executors
    assert_parity(r#"
let x = rand();
let y = rand();
x + y
"#);
}
