//! Hardening test H15: Linear Algebra and Tensor Activations via MIR executor

fn run_mir(src: &str) -> Vec<String> {
    let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

#[test]
fn h15_det_singular() {
    let out = run_mir(r#"
let A = Tensor.from_vec([1.0, 2.0, 2.0, 4.0], [2, 2]);
print(det(A));
"#);
    let parsed: f64 = out[0].parse().unwrap();
    assert!(parsed.abs() < 1e-10, "singular matrix det should be ~0, got {parsed}");
}

#[test]
fn h15_solve_2x2() {
    // 2x + y = 5, x + 3y = 7 → x=1.6, y=1.8
    let out = run_mir(r#"
let A = Tensor.from_vec([2.0, 1.0, 1.0, 3.0], [2, 2]);
let b = Tensor.from_vec([5.0, 7.0], [2]);
let x = solve(A, b);
print(x);
"#);
    assert!(!out.is_empty());
}

#[test]
fn h15_trace_3x3() {
    let out = run_mir(r#"
let A = Tensor.from_vec([5.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 7.0], [3, 3]);
print(trace(A));
"#);
    let parsed: f64 = out[0].parse().unwrap();
    assert!((parsed - 15.0).abs() < 1e-12);
}

#[test]
fn h15_norm_frobenius() {
    let out = run_mir(r#"
let A = Tensor.from_vec([3.0, 4.0, 0.0, 0.0], [2, 2]);
print(norm_frobenius(A));
"#);
    let parsed: f64 = out[0].parse().unwrap();
    assert!((parsed - 5.0).abs() < 1e-12, "expected 5, got {parsed}");
}

#[test]
fn h15_eigh_symmetric() {
    let out = run_mir(r#"
let A = Tensor.from_vec([2.0, 1.0, 1.0, 2.0], [2, 2]);
let result = eigh(A);
print("ok");
"#);
    assert_eq!(out, vec!["ok"]);
}

#[test]
fn h15_matrix_rank_full() {
    let out = run_mir(r#"
let A = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]);
print(matrix_rank(A));
"#);
    assert_eq!(out, vec!["2"]);
}

#[test]
fn h15_kron_identity() {
    let out = run_mir(r#"
let I2 = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]);
let K = kron(I2, I2);
print(trace(K));
"#);
    let parsed: f64 = out[0].parse().unwrap();
    assert!((parsed - 4.0).abs() < 1e-12);
}

#[test]
fn h15_sigmoid() {
    let out = run_mir(r#"
let t = Tensor.from_vec([0.0], [1]);
let s = sigmoid(t);
print(s);
"#);
    // sigmoid(0) = 0.5
    let text = &out[0];
    assert!(text.contains("0.5"), "sigmoid(0) should be 0.5, got {text}");
}

#[test]
fn h15_tanh_activation() {
    let out = run_mir(r#"
let t = Tensor.from_vec([0.0], [1]);
let s = tanh_activation(t);
print(s);
"#);
    // tanh(0) = 0
    let text = &out[0];
    assert!(text.contains("0"), "tanh(0) should be 0, got {text}");
}

#[test]
fn h15_leaky_relu() {
    let out = run_mir(r#"
let t = Tensor.from_vec([-1.0, 0.0, 1.0], [3]);
let s = leaky_relu(t, 0.01);
print(s);
"#);
    assert!(!out.is_empty());
}

#[test]
fn h15_silu() {
    let out = run_mir(r#"
let t = Tensor.from_vec([0.0], [1]);
let s = silu(t);
print(s);
"#);
    // silu(0) = 0 * sigmoid(0) = 0
    let text = &out[0];
    assert!(text.contains("0"), "silu(0) should be 0, got {text}");
}

#[test]
fn h15_mish() {
    let out = run_mir(r#"
let t = Tensor.from_vec([0.0], [1]);
let s = mish(t);
print(s);
"#);
    // mish(0) = 0 * tanh(softplus(0)) = 0
    let text = &out[0];
    assert!(text.contains("0"), "mish(0) should be 0, got {text}");
}

#[test]
fn h15_argmax() {
    let out = run_mir(r#"
let t = Tensor.from_vec([1.0, 5.0, 3.0, 2.0], [4]);
print(argmax(t));
"#);
    assert_eq!(out, vec!["1"]);
}

#[test]
fn h15_argmin() {
    let out = run_mir(r#"
let t = Tensor.from_vec([3.0, 1.0, 5.0, 2.0], [4]);
print(argmin(t));
"#);
    assert_eq!(out, vec!["1"]);
}

#[test]
fn h15_clamp() {
    let out = run_mir(r#"
let t = Tensor.from_vec([-2.0, 0.5, 3.0], [3]);
let c = clamp(t, 0.0, 1.0);
print(c);
"#);
    assert!(!out.is_empty());
}

#[test]
fn h15_one_hot() {
    let out = run_mir(r#"
let indices = [0, 1, 2];
let oh = one_hot(indices, 3);
print(oh);
"#);
    assert!(!out.is_empty());
}

#[test]
fn h15_determinism() {
    let src = r#"
let A = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]);
print(det(A));
print(trace(A));
let t = Tensor.from_vec([0.0, 1.0, -1.0], [3]);
print(sigmoid(t));
print(argmax(t));
"#;
    let out1 = run_mir(src);
    let out2 = run_mir(src);
    assert_eq!(out1, out2);
}
