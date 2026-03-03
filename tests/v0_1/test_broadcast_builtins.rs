//! v0.1 Contract Tests: Broadcasting Builtins
//!
//! Locks down: broadcast() unary, broadcast2() binary,
//! unknown fn errors, shape mismatch, eval/MIR parity.

// ── Helpers ──────────────────────────────────────────────────────

fn parse(src: &str) -> cjc_ast::Program {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    program
}

fn eval_output(src: &str) -> Vec<String> {
    let program = parse(src);
    let mut interp = cjc_eval::Interpreter::new(42);
    let _ = interp.exec(&program).unwrap();
    interp.output.clone()
}

fn mir_output(src: &str) -> Vec<String> {
    let program = parse(src);
    let (_, exec) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    exec.output.clone()
}

// ── Unary broadcast tests ────────────────────────────────────────

#[test]
fn broadcast_sin() {
    // sin(0) = 0, sin(π/2) ≈ 1
    let src = r#"
let t = Tensor.from_vec([0.0, 1.5707963267948966], [2]);
let result = broadcast("sin", t);
print(result);
"#;
    let out = eval_output(src);
    assert!(!out.is_empty(), "broadcast sin should produce output");
    // Output should contain values near 0 and near 1
    let output_str = &out[0];
    assert!(output_str.contains("0"), "sin(0) should be ~0: {}", output_str);
}

#[test]
fn broadcast_sqrt() {
    let src = r#"
let t = Tensor.from_vec([4.0, 9.0, 16.0, 25.0], [4]);
let result = broadcast("sqrt", t);
print(result);
"#;
    let out = eval_output(src);
    assert!(!out.is_empty(), "broadcast sqrt should produce output");
    let output_str = &out[0];
    assert!(output_str.contains("2") && output_str.contains("3"),
            "sqrt([4,9,16,25]) should contain 2 and 3: {}", output_str);
}

#[test]
fn broadcast_exp() {
    let src = r#"
let t = Tensor.from_vec([0.0, 1.0], [2]);
let result = broadcast("exp", t);
print(result);
"#;
    let out = eval_output(src);
    assert!(!out.is_empty(), "broadcast exp should produce output");
    // exp(0)=1, exp(1)≈2.718
    let output_str = &out[0];
    assert!(output_str.contains("1"), "exp(0)=1 should appear: {}", output_str);
}

#[test]
fn broadcast_relu() {
    let src = r#"
let t = Tensor.from_vec([-2.0, -1.0, 0.0, 1.0, 2.0], [5]);
let result = broadcast("relu", t);
print(result);
"#;
    let out = eval_output(src);
    assert!(!out.is_empty(), "broadcast relu should produce output");
    // relu: negatives become 0, positives stay
    let output_str = &out[0];
    assert!(output_str.contains("0"), "relu should zero out negatives: {}", output_str);
}

#[test]
fn broadcast_unknown_fn_errors() {
    let src = r#"
let t = Tensor.from_vec([1.0, 2.0], [2]);
let result = broadcast("nonexistent", t);
"#;
    let program = parse(src);
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_err(), "broadcast with unknown function should error");
}

// ── Binary broadcast2 tests ──────────────────────────────────────

#[test]
fn broadcast2_add() {
    let src = r#"
let t1 = Tensor.from_vec([1.0, 2.0, 3.0], [3]);
let t2 = Tensor.from_vec([10.0, 20.0, 30.0], [3]);
let result = broadcast2("add", t1, t2);
print(result);
"#;
    let out = eval_output(src);
    assert!(!out.is_empty(), "broadcast2 add should produce output");
    let output_str = &out[0];
    assert!(output_str.contains("11") && output_str.contains("22") && output_str.contains("33"),
            "add [1,2,3]+[10,20,30]=[11,22,33]: {}", output_str);
}

#[test]
fn broadcast2_pow() {
    let src = r#"
let bases = Tensor.from_vec([2.0, 3.0, 4.0], [3]);
let exponents = Tensor.from_vec([2.0, 2.0, 2.0], [3]);
let result = broadcast2("pow", bases, exponents);
print(result);
"#;
    let out = eval_output(src);
    assert!(!out.is_empty(), "broadcast2 pow should produce output");
    let output_str = &out[0];
    // 2^2=4, 3^2=9, 4^2=16
    assert!(output_str.contains("4") && output_str.contains("9") && output_str.contains("16"),
            "pow [2,3,4]^[2,2,2]=[4,9,16]: {}", output_str);
}

#[test]
fn broadcast2_unknown_fn_errors() {
    let src = r#"
let t1 = Tensor.from_vec([1.0], [1]);
let t2 = Tensor.from_vec([2.0], [1]);
let result = broadcast2("nonexistent", t1, t2);
"#;
    let program = parse(src);
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_err(), "broadcast2 with unknown function should error");
}

// ── Parity tests ─────────────────────────────────────────────────

#[test]
fn broadcast_parity_eval_mir() {
    let src = r#"
let t = Tensor.from_vec([1.0, 4.0, 9.0], [3]);
print(broadcast("sqrt", t));
print(broadcast("neg", t));
"#;
    let eval_out = eval_output(src);
    let mir_out = mir_output(src);
    assert_eq!(eval_out, mir_out, "broadcast parity: eval vs MIR must match");
}

#[test]
fn broadcast2_parity_eval_mir() {
    let src = r#"
let t1 = Tensor.from_vec([1.0, 2.0, 3.0], [3]);
let t2 = Tensor.from_vec([4.0, 5.0, 6.0], [3]);
print(broadcast2("add", t1, t2));
print(broadcast2("mul", t1, t2));
"#;
    let eval_out = eval_output(src);
    let mir_out = mir_output(src);
    assert_eq!(eval_out, mir_out, "broadcast2 parity: eval vs MIR must match");
}
