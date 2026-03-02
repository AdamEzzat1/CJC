//! Phase B audit test B4: ML Training Extensions

fn run_mir(src: &str) -> Vec<String> {
    let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

fn parse_float(s: &str) -> f64 {
    s.parse::<f64>().unwrap()
}

#[test]
fn b4_batch_norm_identity() {
    let out = run_mir(r#"
let result = batch_norm([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], 0.0);
print(result);
"#);
    assert_eq!(out, vec!["[1, 2, 3]"]);
}

#[test]
fn b4_dropout_mask_determinism() {
    let out = run_mir(r#"
let m1 = dropout_mask(10, 0.5, 42);
let m2 = dropout_mask(10, 0.5, 42);
print(m1);
print(m2);
"#);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0], out[1]);
}

#[test]
fn b4_lr_step_decay() {
    let out = run_mir(r#"
let lr = lr_step_decay(0.1, 0.5, 0, 10);
print(lr);
"#);
    assert_eq!(out, vec!["0.1"]);
}

#[test]
fn b4_lr_step_decay_after_step() {
    let out = run_mir(r#"
let lr = lr_step_decay(0.1, 0.5, 10, 10);
print(lr);
"#);
    let v = parse_float(&out[0]);
    assert!((v - 0.05).abs() < 1e-10);
}

#[test]
fn b4_lr_cosine_start() {
    let out = run_mir(r#"
let lr = lr_cosine(0.1, 0.001, 0, 100);
print(lr);
"#);
    let v = parse_float(&out[0]);
    assert!((v - 0.1).abs() < 1e-10);
}

#[test]
fn b4_lr_cosine_end() {
    let out = run_mir(r#"
let lr = lr_cosine(0.1, 0.001, 100, 100);
print(lr);
"#);
    let v = parse_float(&out[0]);
    assert!((v - 0.001).abs() < 1e-10);
}

#[test]
fn b4_lr_linear_warmup_zero() {
    let out = run_mir(r#"
let lr = lr_linear_warmup(0.1, 0, 10);
print(lr);
"#);
    assert_eq!(out, vec!["0"]);
}

#[test]
fn b4_lr_linear_warmup_full() {
    let out = run_mir(r#"
let lr = lr_linear_warmup(0.1, 15, 10);
print(lr);
"#);
    assert_eq!(out, vec!["0.1"]);
}

#[test]
fn b4_l1_penalty() {
    let out = run_mir(r#"
let p = l1_penalty([1.0, -2.0, 3.0], 0.1);
print(p);
"#);
    let v = parse_float(&out[0]);
    assert!((v - 0.6).abs() < 1e-10);
}

#[test]
fn b4_l2_penalty() {
    let out = run_mir(r#"
let p = l2_penalty([1.0, -2.0, 3.0], 0.1);
print(p);
"#);
    let v = parse_float(&out[0]);
    assert!((v - 0.7).abs() < 1e-10);
}

#[test]
fn b4_topk_basic() {
    let out = run_mir(r#"
let t = Tensor.from_vec([3.0, 1.0, 4.0, 1.0, 5.0, 9.0], [6]);
let result = topk(t, 3);
print(result);
"#);
    // Should return top 3 values: 9, 5, 4
    assert_eq!(out.len(), 1);
}

#[test]
fn b4_dropout_mask_different_seeds() {
    let out = run_mir(r#"
let m1 = dropout_mask(10, 0.5, 42);
let m2 = dropout_mask(10, 0.5, 99);
print(m1);
print(m2);
"#);
    assert_eq!(out.len(), 2);
    assert_ne!(out[0], out[1]);
}

#[test]
fn b4_batch_norm_shift() {
    let out = run_mir(r#"
let result = batch_norm([0.0], [1.0], [4.0], [2.0], [3.0], 0.0);
print(result);
"#);
    // (0 - 1)/sqrt(4) * 2 + 3 = -0.5 * 2 + 3 = 2.0
    assert_eq!(out, vec!["[2]"]);
}

#[test]
fn b4_cat_axis0() {
    let out = run_mir(r#"
let a = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]);
let b = Tensor.from_vec([5.0, 6.0, 7.0, 8.0], [2, 2]);
let c = cat([a, b], 0);
print(c);
"#);
    assert_eq!(out.len(), 1);
}

#[test]
fn b4_stack_axis0() {
    let out = run_mir(r#"
let a = Tensor.from_vec([1.0, 2.0], [2]);
let b = Tensor.from_vec([3.0, 4.0], [2]);
let c = stack([a, b], 0);
print(c);
"#);
    assert_eq!(out.len(), 1);
}
