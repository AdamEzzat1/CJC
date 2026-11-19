//! Parity tests: AST-eval vs MIR-exec must agree on all features.

/// Runs `src` in both executors and returns each one's printed output.
/// Programs end in `print(...)`: a top-level expression statement evaluates
/// to Void, so comparing `exec` return values would pass vacuously.
fn run_parity(src: &str) -> (Vec<String>, Vec<String>) {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);

    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(&program).unwrap();

    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();

    (interp.output.clone(), executor.output)
}

fn assert_parity(src: &str) {
    let (eval_out, mir_out) = run_parity(src);
    assert!(!eval_out.is_empty(), "program printed nothing");
    assert_eq!(eval_out, mir_out, "parity failure:
  eval: {:?}
  mir:  {:?}", eval_out, mir_out);
}

#[test]
fn test_parity_string_upper() {
    assert_parity(r#"print(str_upper("hello"));"#);
}

#[test]
fn test_parity_string_lower() {
    assert_parity(r#"print(str_lower("WORLD"));"#);
}

#[test]
fn test_parity_string_trim() {
    assert_parity(r#"print(str_trim("  abc  "));"#);
}

#[test]
fn test_parity_string_contains() {
    assert_parity(r#"print(str_contains("hello world", "world"));"#);
}

#[test]
fn test_parity_string_replace() {
    assert_parity(r#"print(str_replace("foo bar", "bar", "baz"));"#);
}

#[test]
fn test_parity_string_starts_with() {
    assert_parity(r#"print(str_starts_with("hello", "hel"));"#);
}

#[test]
fn test_parity_string_ends_with() {
    assert_parity(r#"print(str_ends_with("hello", "llo"));"#);
}

#[test]
fn test_parity_string_repeat() {
    assert_parity(r#"print(str_repeat("ab", 3));"#);
}

#[test]
fn test_parity_if_expression() {
    assert_parity(r#"
let x = if true { 42 } else { 0 };
print(x);
"#);
}

#[test]
fn test_parity_variadic_function() {
    assert_parity(r#"
fn total(...nums: f64) -> f64 {
    let s = 0.0;
    let i = 0;
    while i < len(nums) {
        s = s + nums[i];
        i = i + 1;
    }
    s
}
print(total(1.0, 2.0, 3.0, 4.0));
"#);
}

#[test]
fn test_parity_default_params() {
    assert_parity(r#"
fn greet(name: str, greeting: str = "Hello") -> str {
    str_join([greeting, name], " ")
}
print(greet("World"));
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
print(c.get());
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
print(result);
"#);
}

#[test]
fn test_parity_fstring() {
    assert_parity(r#"
let name = "CJC";
let version = 1;
print(f"Language: {name}, version: {version}");
"#);
}

#[test]
fn test_parity_deterministic_rng() {
    // Same seed must produce identical results in both executors
    assert_parity(r#"
let x = Tensor.randn([3]);
let y = Tensor.randn([3]);
print(x + y);
"#);
}
