//! Phase C test C2: Optimizer & Loss Builtins
//!
//! Tests Adam and SGD optimizer construction and stepping from CJC source.

fn run_mir(src: &str) -> Vec<String> {
    let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

fn parse_tensor_data(s: &str) -> Vec<f64> {
    let data_start = s.find("data=[").expect("no data= in tensor output") + 6;
    let data_end = s[data_start..].find(']').expect("no closing ]") + data_start;
    let data_str = &s[data_start..data_end];
    data_str
        .split(", ")
        .map(|v| v.trim().parse::<f64>().unwrap())
        .collect()
}

#[test]
fn c2_adam_basic_step() {
    let out = run_mir(r#"
let opt = Adam.new(2, 0.001);
let params = Tensor.from_vec([1.0, 2.0], [2]);
let grads = Tensor.from_vec([0.1, 0.2], [2]);
let new_params = opt.step(params, grads);
print(new_params);
"#);
    let data = parse_tensor_data(&out[0]);
    // After one Adam step, params should decrease slightly
    assert!(data[0] < 1.0, "param 0 should decrease, got {}", data[0]);
    assert!(data[1] < 2.0, "param 1 should decrease, got {}", data[1]);
}

#[test]
fn c2_adam_bias_correction() {
    // Verify Adam's bias correction kicks in (first step has correction)
    let out = run_mir(r#"
let opt = Adam.new(1, 0.1);
let params = Tensor.from_vec([5.0], [1]);
let grads = Tensor.from_vec([1.0], [1]);
let p1 = opt.step(params, grads);
print(p1);
let p2 = opt.step(p1, grads);
print(p2);
"#);
    let d1 = parse_tensor_data(&out[0]);
    let d2 = parse_tensor_data(&out[1]);
    // Both steps should decrease the parameter
    assert!(d1[0] < 5.0);
    assert!(d2[0] < d1[0]);
}

#[test]
fn c2_sgd_basic_step() {
    // SGD with no momentum: params -= lr * grads
    let out = run_mir(r#"
let opt = Sgd.new(2, 0.1, 0.0);
let params = Tensor.from_vec([1.0, 2.0], [2]);
let grads = Tensor.from_vec([0.5, 1.0], [2]);
let new_params = opt.step(params, grads);
print(new_params);
"#);
    let data = parse_tensor_data(&out[0]);
    // params = [1 - 0.1*0.5, 2 - 0.1*1.0] = [0.95, 1.9]
    assert!((data[0] - 0.95).abs() < 1e-10, "got {}", data[0]);
    assert!((data[1] - 1.9).abs() < 1e-10, "got {}", data[1]);
}

#[test]
fn c2_sgd_momentum() {
    // SGD with momentum accumulates velocity
    let out = run_mir(r#"
let opt = Sgd.new(1, 0.1, 0.9);
let params = Tensor.from_vec([5.0], [1]);
let grads = Tensor.from_vec([1.0], [1]);
let p1 = opt.step(params, grads);
let p2 = opt.step(p1, grads);
print(p1);
print(p2);
"#);
    let d1 = parse_tensor_data(&out[0]);
    let d2 = parse_tensor_data(&out[1]);
    // Step 1: v=1, p=5-0.1*1=4.9
    assert!((d1[0] - 4.9).abs() < 1e-10, "step1 got {}", d1[0]);
    // Step 2: v=0.9*1+1=1.9, p=4.9-0.1*1.9=4.71
    assert!((d2[0] - 4.71).abs() < 1e-10, "step2 got {}", d2[0]);
}

#[test]
fn c2_sgd_zero_momentum() {
    // Without momentum, SGD is simple: params -= lr * grads
    let out = run_mir(r#"
let opt = Sgd.new(1, 0.5);
let params = Tensor.from_vec([10.0], [1]);
let grads = Tensor.from_vec([2.0], [1]);
let new_params = opt.step(params, grads);
print(new_params);
"#);
    let data = parse_tensor_data(&out[0]);
    // 10 - 0.5*2 = 9
    assert!((data[0] - 9.0).abs() < 1e-10);
}

#[test]
fn c2_optimizer_determinism() {
    let src = r#"
let opt = Adam.new(3, 0.01);
let p = Tensor.from_vec([1.0, 2.0, 3.0], [3]);
let g = Tensor.from_vec([0.1, 0.2, 0.3], [3]);
let p1 = opt.step(p, g);
let p2 = opt.step(p1, g);
print(p2);
"#;
    let out1 = run_mir(src);
    let out2 = run_mir(src);
    assert_eq!(out1, out2);
}

#[test]
fn c2_multi_step_convergence() {
    // Multiple SGD steps should converge toward minimum of f(x) = x^2
    let out = run_mir(r#"
let opt = Sgd.new(1, 0.1, 0.0);
let p = Tensor.from_vec([5.0], [1]);
let i = 0;
while i < 20 {
    let grad = Tensor.from_vec([2.0 * p.get([0])], [1]);
    p = opt.step(p, grad);
    i = i + 1;
};
print(p);
"#);
    let data = parse_tensor_data(&out[0]);
    // After 20 steps of gradient descent on x^2, should be near 0
    assert!(data[0].abs() < 0.2, "should converge near 0, got {}", data[0]);
}

#[test]
fn c2_full_train_loop_with_gradgraph() {
    // Full integration: GradGraph forward/backward + optimizer step
    let out = run_mir(r#"
let g = GradGraph.new();
let w = g.parameter(Tensor.from_vec([3.0], [1]));
let x = g.input(Tensor.from_vec([1.0], [1]));
let opt = Sgd.new(1, 0.1, 0.0);
let i = 0;
while i < 5 {
    let pred = g.mul(w, x);
    let loss = g.mul(pred, pred);
    g.backward(loss);
    let grad = g.grad(w);
    let w_tensor = g.tensor(w);
    let new_w = opt.step(w_tensor, grad);
    g.set_tensor(w, new_w);
    g.zero_grad();
    i = i + 1;
};
print(g.value(w));
"#);
    let val: f64 = out[0].parse().unwrap();
    // After 5 gradient steps, w should decrease from 3.0 toward 0
    assert!(val.abs() < 3.0, "w should decrease from 3, got {val}");
}
