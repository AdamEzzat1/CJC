//! Phase C test C1: GradGraph Language API
//!
//! Tests that GradGraph can be constructed and used from CJC source code,
//! including forward computation, backward pass, gradient access, and
//! parameter update cycles.

fn run_mir(src: &str) -> Vec<String> {
    let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

/// Parse tensor data from its Display format: "Tensor(shape=[...], data=[v1, v2, ...])"
fn parse_tensor_data(s: &str) -> Vec<f64> {
    let data_start = s.find("data=[").expect("no data= in tensor output") + 6;
    let data_end = s[data_start..].find(']').expect("no closing ] in tensor data") + data_start;
    let data_str = &s[data_start..data_end];
    data_str
        .split(", ")
        .map(|v| v.trim().parse::<f64>().unwrap())
        .collect()
}

fn assert_tensor_close(out: &str, expected: &[f64], tol: f64) {
    let actual = parse_tensor_data(out);
    assert_eq!(
        actual.len(),
        expected.len(),
        "tensor length mismatch: got {:?}, expected {:?}",
        actual,
        expected
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "element {i}: got {a}, expected {e} (diff={})",
            (a - e).abs()
        );
    }
}

#[test]
fn c1_construct_graph() {
    let out = run_mir(r#"
let g = GradGraph.new();
print("ok");
"#);
    assert_eq!(out, vec!["ok"]);
}

#[test]
fn c1_forward_add() {
    let out = run_mir(r#"
let g = GradGraph.new();
let a = g.input(Tensor.from_vec([2.0], [1]));
let b = g.input(Tensor.from_vec([3.0], [1]));
let c = g.add(a, b);
print(g.value(c));
"#);
    assert_eq!(out, vec!["5"]);
}

#[test]
fn c1_forward_matmul() {
    // [1, 2] @ [[1],[1]] = [[3]]
    let out = run_mir(r#"
let g = GradGraph.new();
let x = g.input(Tensor.from_vec([1.0, 2.0], [1, 2]));
let w = g.input(Tensor.from_vec([1.0, 1.0], [2, 1]));
let y = g.matmul(x, w);
print(g.value(y));
"#);
    assert_eq!(out, vec!["3"]);
}

#[test]
fn c1_backward_x_squared() {
    // f(x) = x^2 = x * x, f'(3) = 6
    let out = run_mir(r#"
let g = GradGraph.new();
let x = g.parameter(Tensor.from_vec([3.0], [1]));
let sq = g.mul(x, x);
g.backward(sq);
let grad = g.grad(x);
print(grad);
"#);
    assert_tensor_close(&out[0], &[6.0], 1e-10);
}

#[test]
fn c1_backward_linear() {
    // f(W) = sum(x @ W), grad_W = x^T @ ones
    let out = run_mir(r#"
let g = GradGraph.new();
let W = g.parameter(Tensor.from_vec([1.0, 2.0], [2, 1]));
let x = g.input(Tensor.from_vec([3.0, 4.0], [1, 2]));
let y = g.matmul(x, W);
let loss = g.sum(y);
g.backward(loss);
let grad = g.grad(W);
print(grad);
"#);
    // grad_W = x^T @ ones(1,1) = [[3],[4]]
    assert_tensor_close(&out[0], &[3.0, 4.0], 1e-10);
}

#[test]
fn c1_backward_sigmoid() {
    // sigmoid(0) = 0.5, sigmoid'(0) = 0.25
    let out = run_mir(r#"
let g = GradGraph.new();
let x = g.parameter(Tensor.from_vec([0.0], [1]));
let s = g.sigmoid(x);
g.backward(s);
let grad = g.grad(x);
print(grad);
"#);
    assert_tensor_close(&out[0], &[0.25], 1e-10);
}

#[test]
fn c1_backward_relu() {
    // relu'(3) = 1, relu'(-2) = 0
    let out = run_mir(r#"
let g = GradGraph.new();
let x = g.parameter(Tensor.from_vec([3.0, -2.0], [2]));
let r = g.relu(x);
let loss = g.sum(r);
g.backward(loss);
let grad = g.grad(x);
print(grad);
"#);
    assert_tensor_close(&out[0], &[1.0, 0.0], 1e-10);
}

#[test]
fn c1_backward_chain_rule() {
    // f(x) = sin(x^2), f'(x) = cos(x^2) * 2x
    // at x=1: f'(1) = cos(1) * 2
    let out = run_mir(r#"
let g = GradGraph.new();
let x = g.parameter(Tensor.from_vec([1.0], [1]));
let x2 = g.mul(x, x);
let s = g.sin(x2);
g.backward(s);
let grad = g.grad(x);
print(grad);
"#);
    let expected = 1.0_f64.cos() * 2.0;
    assert_tensor_close(&out[0], &[expected], 1e-10);
}

#[test]
fn c1_gradient_descent_step() {
    // Simple: minimize f(x) = x^2 starting at x=5
    // grad = 2x = 10, x_new = 5 - 0.1*10 = 4
    let out = run_mir(r#"
let g = GradGraph.new();
let x = g.parameter(Tensor.from_vec([5.0], [1]));
let sq = g.mul(x, x);
let loss = g.sum(sq);
g.backward(loss);
let grad = g.grad(x);
let old_t = g.tensor(x);
let old_val = old_t.get([0]);
let grad_val = grad.get([0]);
let new_val = old_val - 0.1 * grad_val;
print(new_val);
"#);
    assert_eq!(out, vec!["4"]);
}

#[test]
fn c1_parameter_update_cycle() {
    // Forward -> backward -> update -> check updated value
    let out = run_mir(r#"
let g = GradGraph.new();
let w = g.parameter(Tensor.from_vec([2.0], [1]));
let x = g.input(Tensor.from_vec([3.0], [1]));
let y = g.mul(w, x);
let loss = g.sum(y);
g.backward(loss);
let grad_w = g.grad(w);
let old_w = g.tensor(w);
let new_w_val = old_w.get([0]) - 0.1 * grad_w.get([0]);
g.set_tensor(w, Tensor.from_vec([new_w_val], [1]));
print(g.value(w));
"#);
    // grad_w = x = 3, new_w = 2 - 0.1*3 = 1.7
    assert_eq!(out, vec!["1.7"]);
}

#[test]
fn c1_multi_param_net() {
    // Two-layer: h = relu(x*W1), out = sum(h*W2)
    let out = run_mir(r#"
let g = GradGraph.new();
let W1 = g.parameter(Tensor.from_vec([1.0, 0.5], [1, 2]));
let W2 = g.parameter(Tensor.from_vec([1.0, 1.0], [2, 1]));
let x = g.input(Tensor.from_vec([2.0], [1, 1]));
let h = g.matmul(x, W1);
let h_r = g.relu(h);
let out_node = g.matmul(h_r, W2);
let loss = g.sum(out_node);
g.backward(loss);
let grad_W1 = g.grad(W1);
let grad_W2 = g.grad(W2);
print(grad_W1);
print(grad_W2);
"#);
    assert_eq!(out.len(), 2);
    // Both gradients should be finite non-zero
    assert!(!out[0].contains("NaN"));
    assert!(!out[1].contains("NaN"));
}

#[test]
fn c1_zero_grad_reset() {
    let out = run_mir(r#"
let g = GradGraph.new();
let x = g.parameter(Tensor.from_vec([3.0], [1]));
let sq = g.mul(x, x);
g.backward(sq);
let grad1 = g.grad(x);
print(grad1);
g.zero_grad();
let grad2 = g.grad(x);
print(grad2);
"#);
    assert_tensor_close(&out[0], &[6.0], 1e-10);
    assert_tensor_close(&out[1], &[0.0], 1e-10);
}

#[test]
fn c1_exp_ln_roundtrip() {
    // exp(ln(x)) = x, gradients should be = 1
    let out = run_mir(r#"
let g = GradGraph.new();
let x = g.parameter(Tensor.from_vec([2.0], [1]));
let lnx = g.ln(x);
let elx = g.exp(lnx);
g.backward(elx);
let grad = g.grad(x);
print(g.value(elx));
print(grad);
"#);
    // exp(ln(2)) = 2
    let val: f64 = out[0].parse().unwrap();
    assert!((val - 2.0).abs() < 1e-10);
    // d/dx exp(ln(x)) = 1
    assert_tensor_close(&out[1], &[1.0], 1e-10);
}

#[test]
fn c1_scalar_mul_grad() {
    // f(x) = 3x, f'(x) = 3
    let out = run_mir(r#"
let g = GradGraph.new();
let x = g.parameter(Tensor.from_vec([2.0], [1]));
let y = g.scalar_mul(x, 3.0);
g.backward(y);
let grad = g.grad(x);
print(grad);
"#);
    assert_tensor_close(&out[0], &[3.0], 1e-10);
}

#[test]
fn c1_determinism() {
    let src = r#"
let g = GradGraph.new();
let W = g.parameter(Tensor.from_vec([0.5, 0.3, 0.7, 0.2], [2, 2]));
let x = g.input(Tensor.from_vec([1.0, 2.0], [1, 2]));
let y = g.matmul(x, W);
let loss = g.sum(y);
g.backward(loss);
let grad = g.grad(W);
print(grad);
"#;
    let out1 = run_mir(src);
    let out2 = run_mir(src);
    assert_eq!(out1, out2);
}

#[test]
fn c1_div_forward_backward() {
    // f(x) = x / 2, f'(x) = 0.5
    let out = run_mir(r#"
let g = GradGraph.new();
let x = g.parameter(Tensor.from_vec([6.0], [1]));
let two = g.input(Tensor.from_vec([2.0], [1]));
let y = g.div(x, two);
g.backward(y);
print(g.value(y));
print(g.grad(x));
"#);
    assert_eq!(out[0], "3");
    assert_tensor_close(&out[1], &[0.5], 1e-10);
}

#[test]
fn c1_neg_forward_backward() {
    // f(x) = -x, f'(x) = -1
    let out = run_mir(r#"
let g = GradGraph.new();
let x = g.parameter(Tensor.from_vec([3.0], [1]));
let y = g.neg(x);
g.backward(y);
print(g.value(y));
print(g.grad(x));
"#);
    assert_eq!(out[0], "-3");
    assert_tensor_close(&out[1], &[-1.0], 1e-10);
}
