//! VM / Runtime parity tests for the byte-first audit.

fn eval_and_mir(src: &str, seed: u64) -> (cjc_runtime::Value, cjc_runtime::Value) {
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}",
        diags.diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>());
    let ve = cjc_eval::Interpreter::new(seed).exec(&prog).expect("eval failed");
    let (vm, _) = cjc_mir_exec::run_program_with_executor(&prog, seed)
        .expect("mir failed");
    (ve, vm)
}

fn assert_parity(ve: &cjc_runtime::Value, vm: &cjc_runtime::Value, ctx: &str) {
    match (ve, vm) {
        (cjc_runtime::Value::Int(a), cjc_runtime::Value::Int(b)) =>
            assert_eq!(a, b, "{ctx}: Int mismatch"),
        (cjc_runtime::Value::Float(a), cjc_runtime::Value::Float(b)) =>
            assert_eq!(a.to_bits(), b.to_bits(), "{ctx}: Float {a} vs {b}"),
        (cjc_runtime::Value::Bool(a), cjc_runtime::Value::Bool(b)) =>
            assert_eq!(a, b, "{ctx}: Bool mismatch"),
        (cjc_runtime::Value::String(a), cjc_runtime::Value::String(b)) =>
            assert_eq!(**a, **b, "{ctx}: String mismatch"),
        (cjc_runtime::Value::Void, cjc_runtime::Value::Void) => {}
        _ => assert_eq!(format!("{ve:?}"), format!("{vm:?}"), "{ctx}: mismatch"),
    }
}

#[test]
fn parity_int_arithmetic() {
    let src = r#"fn main() -> i64 { 3 + 4 * 2 }"#;
    let (ve, vm) = eval_and_mir(src, 42);
    assert_parity(&ve, &vm, "int arith");
}

#[test]
fn parity_float_arithmetic() {
    let src = r#"fn main() -> f64 { 1.5 + 2.3 * 0.7 }"#;
    let (ve, vm) = eval_and_mir(src, 42);
    assert_parity(&ve, &vm, "float arith");
}

#[test]
fn parity_string_ops() {
    let src = r#"fn main() -> Any { str_upper("hello") }"#;
    let (ve, vm) = eval_and_mir(src, 42);
    assert_parity(&ve, &vm, "str_upper");
}

#[test]
fn parity_boolean_logic() {
    let src = r#"fn main() -> bool { true && false || !false }"#;
    let (ve, vm) = eval_and_mir(src, 42);
    assert_parity(&ve, &vm, "bool logic");
}

#[test]
fn parity_comparison() {
    let src = r#"fn main() -> bool { 3 > 2 }"#;
    let (ve, vm) = eval_and_mir(src, 42);
    assert_parity(&ve, &vm, "comparison");
}

#[test]
fn parity_array_ops() {
    let src = r#"fn main() -> i64 { array_len(array_push([1, 2, 3], 4)) }"#;
    let (ve, vm) = eval_and_mir(src, 42);
    assert_parity(&ve, &vm, "array ops");
}

#[test]
fn parity_struct() {
    let src = "struct Point { x: f64, y: f64 }\nfn main() -> f64 {\n  let p = Point { x: 1.5, y: 2.5 };\n  p.x + p.y\n}";
    let (ve, vm) = eval_and_mir(src, 42);
    assert_parity(&ve, &vm, "struct");
}

#[test]
fn parity_function_call() {
    let src = "fn add(a: i64, b: i64) -> i64 { a + b }\nfn main() -> i64 { add(10, 20) }";
    let (ve, vm) = eval_and_mir(src, 42);
    assert_parity(&ve, &vm, "function call");
}

#[test]
fn parity_nested_calls() {
    // Test nested function calls (function composition via nesting)
    let src = "fn double(x: i64) -> i64 { x * 2 }\nfn add_one(x: i64) -> i64 { x + 1 }\nfn main() -> i64 { add_one(double(add_one(double(3)))) }";
    let (ve, vm) = eval_and_mir(src, 42);
    assert_parity(&ve, &vm, "closure");
}

#[test]
fn parity_while_loop() {
    let src = "fn sum_to(n: i64) -> i64 {\n  let mut t = 0;\n  let mut i = 1;\n  while i <= n {\n    t = t + i;\n    i = i + 1;\n  }\n  t\n}\nfn main() -> i64 { sum_to(100) }";
    let (ve, vm) = eval_and_mir(src, 42);
    assert_parity(&ve, &vm, "while loop");
}

#[test]
fn parity_for_loop() {
    let src = "fn sum_range() -> i64 {\n  let mut t = 0;\n  for i in 1..11 {\n    t = t + i;\n  }\n  t\n}\nfn main() -> i64 { sum_range() }";
    let (ve, vm) = eval_and_mir(src, 42);
    assert_parity(&ve, &vm, "for loop");
}

#[test]
fn parity_match() {
    let src = "fn main() -> i64 {\n  let x = 42;\n  match x {\n    1 => 10,\n    42 => 420,\n    _ => 0,\n  }\n}";
    let (ve, vm) = eval_and_mir(src, 42);
    assert_parity(&ve, &vm, "match");
}

#[test]
fn parity_tensor_sum() {
    let src = r#"fn main() -> f64 { Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [4]).sum() }"#;
    let (ve, vm) = eval_and_mir(src, 42);
    assert_parity(&ve, &vm, "tensor sum");
}

#[test]
fn parity_recursive() {
    let src = "fn fact(n: i64) -> i64 {\n  if n <= 1 { return 1; }\n  n * fact(n - 1)\n}\nfn main() -> i64 { fact(10) }";
    let (ve, vm) = eval_and_mir(src, 42);
    assert_parity(&ve, &vm, "recursive");
}
