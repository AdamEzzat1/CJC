//! Transformer kernel tests — Silicon Transformer milestone
//!
//! Tests cover:
//!   1. Tensor literal syntax [| ... |]
//!   2. Softmax kernel (two-pass stable)
//!   3. LayerNorm kernel (Kahan-based)
//!   4. ReLU / GELU activations
//!   5. Batched MatMul (bmm)
//!   6. Linear layer (matmul + bias)
//!   7. Scaled dot-product attention
//!   8. Pipeline parity (AST eval vs MIR-exec)
//!   9. Transformer forward pass integration
//!  10. Determinism double-run gate

use cjc_eval::Interpreter;
use cjc_lexer::{Lexer, TokenKind};
use cjc_parser::parse_source;
use cjc_runtime::Tensor;

fn eval_output(src: &str) -> Vec<String> {
    let (program, diag) = parse_source(src);
    assert!(!diag.has_errors(), "Parse errors:\n{}", diag.render_all(src, "<test>"));
    let mut interp = Interpreter::new(42);
    match interp.exec(&program) {
        Ok(_) => {}
        Err(e) => panic!("eval error: {:?}", e),
    }
    interp.output.clone()
}

fn mir_output(src: &str) -> Vec<String> {
    let (program, diag) = parse_source(src);
    assert!(!diag.has_errors(), "Parse errors:\n{}", diag.render_all(src, "<test>"));
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42)
        .expect("MIR exec failed");
    executor.output.clone()
}

fn assert_parity(src: &str) {
    let ast_out = eval_output(src);
    let mir_out = mir_output(src);
    assert_eq!(
        ast_out, mir_out,
        "AST/MIR parity failure!\nAST: {:?}\nMIR: {:?}\nSource:\n{}",
        ast_out, mir_out, src
    );
}

// ═══════════════════════════════════════════════════════════════════
// Section 1: Tensor Literal Syntax [| ... |]
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_lex_tensor_lit_brackets() {
    let (tokens, _) = Lexer::new("[| 1.0, 2.0 |]").tokenize();
    assert_eq!(tokens[0].kind, TokenKind::LBracketPipe);
    let pipe_rb = tokens.iter().find(|t| t.kind == TokenKind::PipeRBracket);
    assert!(pipe_rb.is_some(), "expected PipeRBracket token");
}

#[test]
fn test_lex_tensor_lit_2d() {
    let (tokens, _) = Lexer::new("[| 1, 2; 3, 4 |]").tokenize();
    assert_eq!(tokens[0].kind, TokenKind::LBracketPipe);
    let semi = tokens.iter().find(|t| t.kind == TokenKind::Semicolon);
    assert!(semi.is_some(), "expected semicolon in tensor literal");
}

#[test]
fn test_lex_brackets_still_work() {
    let (tokens, _) = Lexer::new("[1, 2]").tokenize();
    assert_eq!(tokens[0].kind, TokenKind::LBracket);
}

#[test]
fn test_lex_pipe_rbracket() {
    let (tokens, _) = Lexer::new("|]").tokenize();
    assert_eq!(tokens[0].kind, TokenKind::PipeRBracket);
}

#[test]
fn test_lex_pipe_still_works() {
    let (tokens, _) = Lexer::new("|x| x").tokenize();
    assert_eq!(tokens[0].kind, TokenKind::Pipe);
}

#[test]
fn test_parse_tensor_lit_1d() {
    let (program, diag) = parse_source("let t = [| 1.0, 2.0, 3.0 |];");
    assert!(!diag.has_errors(), "Parse errors: {}", diag.render_all("let t = [| 1.0, 2.0, 3.0 |];", "<test>"));
    assert!(!program.declarations.is_empty());
}

#[test]
fn test_parse_tensor_lit_2d() {
    let (program, diag) = parse_source("let t = [| 1.0, 2.0; 3.0, 4.0 |];");
    assert!(!diag.has_errors(), "Parse errors: {}", diag.render_all("let t = [| 1.0, 2.0; 3.0, 4.0 |];", "<test>"));
    assert!(!program.declarations.is_empty());
}

#[test]
fn test_eval_tensor_lit_1d() {
    let out = eval_output("let t = [| 1.0, 2.0, 3.0 |]; print(t.sum());");
    assert_eq!(out, vec!["6"]);
}

#[test]
fn test_eval_tensor_lit_2d() {
    let out = eval_output("let t = [| 1.0, 0.0; 0.0, 1.0 |]; print(t.shape());");
    assert_eq!(out, vec!["[2, 2]"]);
}

#[test]
fn test_eval_tensor_lit_2d_sum() {
    let out = eval_output("let t = [| 1.0, 2.0; 3.0, 4.0 |]; print(t.sum());");
    assert_eq!(out, vec!["10"]);
}

#[test]
fn test_eval_tensor_lit_integers() {
    // Integers should coerce to f64
    let out = eval_output("let t = [| 10, 20, 30 |]; print(t.sum());");
    assert_eq!(out, vec!["60"]);
}

#[test]
fn test_eval_tensor_lit_matmul() {
    let out = eval_output(
        "let a = [| 1.0, 2.0; 3.0, 4.0 |]; \
         let b = [| 5.0, 6.0; 7.0, 8.0 |]; \
         let c = a.matmul(b); \
         print(c.get([0, 0])); \
         print(c.get([0, 1])); \
         print(c.get([1, 0])); \
         print(c.get([1, 1]));"
    );
    assert_eq!(out, vec!["19", "22", "43", "50"]);
}

// ═══════════════════════════════════════════════════════════════════
// Section 2: Softmax Kernel
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_softmax_basic() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let s = t.softmax().unwrap();
    let data = s.to_vec();
    assert_eq!(data.len(), 3);
    // softmax values should sum to 1
    let sum: f64 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10, "softmax sum: {}", sum);
    // Values should be in ascending order
    assert!(data[0] < data[1]);
    assert!(data[1] < data[2]);
}

#[test]
fn test_softmax_2d() {
    // 2x3 matrix: softmax applied per row
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]).unwrap();
    let s = t.softmax().unwrap();
    let data = s.to_vec();
    // Each row sums to 1
    let row1_sum: f64 = data[0..3].iter().sum();
    let row2_sum: f64 = data[3..6].iter().sum();
    assert!((row1_sum - 1.0).abs() < 1e-10);
    assert!((row2_sum - 1.0).abs() < 1e-10);
    // Row 2: uniform softmax (all equal input)
    assert!((data[3] - data[4]).abs() < 1e-10);
    assert!((data[4] - data[5]).abs() < 1e-10);
}

#[test]
fn test_softmax_large_values_stable() {
    // Large values should not cause overflow with max-subtraction
    let t = Tensor::from_vec(vec![1000.0, 1001.0, 1002.0], &[3]).unwrap();
    let s = t.softmax().unwrap();
    let data = s.to_vec();
    assert!(data.iter().all(|&v| v.is_finite()), "softmax overflow: {:?}", data);
    let sum: f64 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);
}

#[test]
fn test_softmax_negative_values() {
    let t = Tensor::from_vec(vec![-1.0, -2.0, -3.0], &[3]).unwrap();
    let s = t.softmax().unwrap();
    let data = s.to_vec();
    let sum: f64 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);
    // First element is largest (least negative)
    assert!(data[0] > data[1]);
    assert!(data[1] > data[2]);
}

#[test]
fn test_softmax_pipeline() {
    let out = eval_output(
        "let t = Tensor.from_vec([1.0, 2.0, 3.0], [3]); \
         let s = t.softmax(); \
         print(s.shape());"
    );
    assert_eq!(out, vec!["[3]"]);
}

// ═══════════════════════════════════════════════════════════════════
// Section 3: LayerNorm Kernel
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_layer_norm_basic() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
    let gamma = Tensor::ones(&[5]);
    let beta = Tensor::zeros(&[5]);
    let result = t.layer_norm(&gamma, &beta, 1e-5).unwrap();
    let data = result.to_vec();
    // Mean should be ~0
    let mean: f64 = data.iter().sum::<f64>() / 5.0;
    assert!(mean.abs() < 1e-10, "layer_norm mean: {}", mean);
}

#[test]
fn test_layer_norm_with_gamma_beta() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let gamma = Tensor::from_vec(vec![2.0, 2.0, 2.0], &[3]).unwrap();
    let beta = Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3]).unwrap();
    let result = t.layer_norm(&gamma, &beta, 1e-5).unwrap();
    let data = result.to_vec();
    // With gamma=2, beta=1: result = 2 * normalized + 1
    // Mean of result should be 1.0 (gamma * 0 + beta = 1)
    let mean: f64 = data.iter().sum::<f64>() / 3.0;
    assert!((mean - 1.0).abs() < 1e-6, "layer_norm mean with bias: {}", mean);
}

#[test]
fn test_layer_norm_2d() {
    // 2x3: normalize each row independently
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0], &[2, 3]).unwrap();
    let gamma = Tensor::ones(&[3]);
    let beta = Tensor::zeros(&[3]);
    let result = t.layer_norm(&gamma, &beta, 1e-5).unwrap();
    let data = result.to_vec();
    // Each row should have mean ~0
    let mean1: f64 = data[0..3].iter().sum::<f64>() / 3.0;
    let mean2: f64 = data[3..6].iter().sum::<f64>() / 3.0;
    assert!(mean1.abs() < 1e-10, "row1 mean: {}", mean1);
    assert!(mean2.abs() < 1e-10, "row2 mean: {}", mean2);
}

#[test]
fn test_layer_norm_pipeline() {
    let out = eval_output(
        "let t = Tensor.from_vec([1.0, 2.0, 3.0], [3]); \
         let g = Tensor.ones([3]); \
         let b = Tensor.zeros([3]); \
         let r = t.layer_norm(g, b, 0.00001); \
         print(r.shape());"
    );
    assert_eq!(out, vec!["[3]"]);
}

// ═══════════════════════════════════════════════════════════════════
// Section 4: ReLU / GELU Activations
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_relu_positive() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let r = t.relu();
    assert_eq!(r.to_vec(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_relu_negative() {
    let t = Tensor::from_vec(vec![-1.0, -2.0, -3.0], &[3]).unwrap();
    let r = t.relu();
    assert_eq!(r.to_vec(), vec![0.0, 0.0, 0.0]);
}

#[test]
fn test_relu_mixed() {
    let t = Tensor::from_vec(vec![-2.0, 0.0, 3.0], &[3]).unwrap();
    let r = t.relu();
    assert_eq!(r.to_vec(), vec![0.0, 0.0, 3.0]);
}

#[test]
fn test_gelu_basic() {
    let t = Tensor::from_vec(vec![-1.0, 0.0, 1.0], &[3]).unwrap();
    let g = t.gelu();
    let data = g.to_vec();
    // GELU(0) = 0
    assert!(data[1].abs() < 1e-10);
    // GELU(1) ≈ 0.8413 (approximately)
    assert!((data[2] - 0.8413).abs() < 0.01, "GELU(1) = {}", data[2]);
    // GELU(-1) ≈ -0.1587
    assert!((data[0] - (-0.1587)).abs() < 0.01, "GELU(-1) = {}", data[0]);
}

#[test]
fn test_relu_pipeline() {
    let out = eval_output(
        "let t = Tensor.from_vec([-1.0, 0.0, 2.0], [3]); \
         let r = t.relu(); \
         print(r.sum());"
    );
    assert_eq!(out, vec!["2"]);
}

#[test]
fn test_gelu_pipeline() {
    let out = eval_output(
        "let t = Tensor.from_vec([0.0], [1]); \
         let g = t.gelu(); \
         print(g.get([0]));"
    );
    assert_eq!(out, vec!["0"]);
}

// ═══════════════════════════════════════════════════════════════════
// Section 5: Batched MatMul (bmm)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_bmm_2d_delegates_to_matmul() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
    let c = a.bmm(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    let data = c.to_vec();
    assert_eq!(data, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_bmm_3d() {
    // Batch of 2, each 2x2 * 2x2
    let a = Tensor::from_vec(
        vec![1.0, 0.0, 0.0, 1.0, // identity
             2.0, 0.0, 0.0, 2.0], // 2*identity
        &[2, 2, 2]
    ).unwrap();
    let b = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0,
             5.0, 6.0, 7.0, 8.0],
        &[2, 2, 2]
    ).unwrap();
    let c = a.bmm(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2, 2]);
    let data = c.to_vec();
    // Batch 0: identity * b[0] = b[0]
    assert_eq!(data[0], 1.0);
    assert_eq!(data[1], 2.0);
    assert_eq!(data[2], 3.0);
    assert_eq!(data[3], 4.0);
    // Batch 1: 2*identity * b[1] = 2*b[1]
    assert_eq!(data[4], 10.0);
    assert_eq!(data[5], 12.0);
    assert_eq!(data[6], 14.0);
    assert_eq!(data[7], 16.0);
}

#[test]
fn test_bmm_dimension_mismatch() {
    let a = Tensor::from_vec(vec![1.0; 6], &[2, 3]).unwrap();
    let b = Tensor::from_vec(vec![1.0; 6], &[2, 3]).unwrap();
    assert!(a.bmm(&b).is_err()); // 3 != 2
}

#[test]
fn test_bmm_pipeline() {
    let out = eval_output(
        "let a = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]); \
         let b = Tensor.from_vec([5.0, 6.0, 7.0, 8.0], [2, 2]); \
         let c = a.bmm(b); \
         print(c.get([0, 0])); \
         print(c.get([1, 1]));"
    );
    assert_eq!(out, vec!["5", "8"]);
}

// ═══════════════════════════════════════════════════════════════════
// Section 6: Linear Layer
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_linear_basic() {
    // x = [1, 2], W = [[1, 0], [0, 1]], b = [10, 20]
    // output = x @ W^T + b = [1, 2] + [10, 20] = [11, 22]
    let x = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]).unwrap();
    let w = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![10.0, 20.0], &[2]).unwrap();
    let y = x.linear(&w, &b).unwrap();
    assert_eq!(y.shape(), &[1, 2]);
    let data = y.to_vec();
    assert_eq!(data, vec![11.0, 22.0]);
}

#[test]
fn test_linear_batch() {
    // Batch of 2 inputs, 3->2 projection
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let w = Tensor::from_vec(vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[2, 3]).unwrap();
    let b = Tensor::from_vec(vec![0.0, 0.0], &[2]).unwrap();
    let y = x.linear(&w, &b).unwrap();
    assert_eq!(y.shape(), &[2, 2]);
    let data = y.to_vec();
    // Row 0: [1+2+3, 3] = [6, 3]
    assert_eq!(data[0], 6.0);
    assert_eq!(data[1], 3.0);
    // Row 1: [4+5+6, 6] = [15, 6]
    assert_eq!(data[2], 15.0);
    assert_eq!(data[3], 6.0);
}

#[test]
fn test_linear_pipeline() {
    let out = eval_output(
        "let x = Tensor.from_vec([1.0, 2.0], [1, 2]); \
         let w = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]); \
         let b = Tensor.from_vec([0.5, 0.5], [2]); \
         let y = x.linear(w, b); \
         print(y.get([0, 0])); \
         print(y.get([0, 1]));"
    );
    assert_eq!(out, vec!["1.5", "2.5"]);
}

// ═══════════════════════════════════════════════════════════════════
// Section 7: Scaled Dot-Product Attention
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_attention_basic() {
    // Q=K=V = identity-like => attention should roughly return V
    let q = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
    let k = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
    let v = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
    let out = Tensor::scaled_dot_product_attention(&q, &k, &v).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    // Output should be a weighted combination of V rows
    let data = out.to_vec();
    assert!(data.iter().all(|v| v.is_finite()));
}

#[test]
fn test_attention_dimensions() {
    // Q: [4, 3], K: [5, 3], V: [5, 2] => output: [4, 2]
    let q = Tensor::from_vec(vec![0.0; 12], &[4, 3]).unwrap();
    let k = Tensor::from_vec(vec![0.0; 15], &[5, 3]).unwrap();
    let v = Tensor::from_vec(vec![1.0; 10], &[5, 2]).unwrap();
    let out = Tensor::scaled_dot_product_attention(&q, &k, &v).unwrap();
    assert_eq!(out.shape(), &[4, 2]);
}

#[test]
fn test_attention_pipeline() {
    let out = eval_output(
        "let q = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]); \
         let k = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]); \
         let v = Tensor.from_vec([10.0, 20.0, 30.0, 40.0], [2, 2]); \
         let out = attention(q, k, v); \
         print(out.shape());"
    );
    assert_eq!(out, vec!["[2, 2]"]);
}

// ═══════════════════════════════════════════════════════════════════
// Section 8: Transpose Last Two
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_transpose_last_two_2d() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let tt = t.transpose_last_two().unwrap();
    assert_eq!(tt.shape(), &[3, 2]);
    let data = tt.to_vec();
    assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_transpose_last_two_3d() {
    // [2, 2, 3] -> [2, 3, 2]
    let t = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        &[2, 2, 3]
    ).unwrap();
    let tt = t.transpose_last_two().unwrap();
    assert_eq!(tt.shape(), &[2, 3, 2]);
}

// ═══════════════════════════════════════════════════════════════════
// Section 9: Pipeline Parity (AST eval vs MIR-exec)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_parity_tensor_lit_1d() {
    assert_parity("let t = [| 1.0, 2.0, 3.0 |]; print(t.sum());");
}

#[test]
fn test_parity_tensor_lit_2d() {
    assert_parity("let t = [| 1.0, 2.0; 3.0, 4.0 |]; print(t.sum());");
}

#[test]
fn test_parity_softmax() {
    assert_parity(
        "let t = Tensor.from_vec([1.0, 2.0, 3.0], [3]); \
         let s = t.softmax(); \
         print(s.shape());"
    );
}

#[test]
fn test_parity_relu() {
    assert_parity(
        "let t = Tensor.from_vec([-1.0, 0.0, 2.0], [3]); \
         let r = t.relu(); \
         print(r.sum());"
    );
}

#[test]
fn test_parity_gelu() {
    assert_parity(
        "let t = Tensor.from_vec([0.0], [1]); \
         let g = t.gelu(); \
         print(g.get([0]));"
    );
}

#[test]
fn test_parity_layer_norm() {
    assert_parity(
        "let t = Tensor.from_vec([1.0, 2.0, 3.0], [3]); \
         let g = Tensor.ones([3]); \
         let b = Tensor.zeros([3]); \
         let r = t.layer_norm(g, b, 0.00001); \
         print(r.shape());"
    );
}

#[test]
fn test_parity_bmm() {
    assert_parity(
        "let a = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]); \
         let b = Tensor.from_vec([5.0, 6.0, 7.0, 8.0], [2, 2]); \
         let c = a.bmm(b); \
         print(c.get([0, 0])); \
         print(c.get([1, 1]));"
    );
}

#[test]
fn test_parity_linear() {
    assert_parity(
        "let x = Tensor.from_vec([1.0, 2.0], [1, 2]); \
         let w = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]); \
         let b = Tensor.from_vec([0.5, 0.5], [2]); \
         let y = x.linear(w, b); \
         print(y.get([0, 0])); \
         print(y.get([0, 1]));"
    );
}

#[test]
fn test_parity_attention() {
    assert_parity(
        "let q = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]); \
         let k = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]); \
         let v = Tensor.from_vec([10.0, 20.0, 30.0, 40.0], [2, 2]); \
         let out = attention(q, k, v); \
         print(out.shape());"
    );
}

#[test]
fn test_parity_matmul_not_broken() {
    assert_parity(
        "let a = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]); \
         let b = Tensor.from_vec([5.0, 6.0, 7.0, 8.0], [2, 2]); \
         let c = matmul(a, b); \
         print(c.get([0, 0]));"
    );
}

// ═══════════════════════════════════════════════════════════════════
// Section 10: Transformer Forward Pass (Integration)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_transformer_block_forward() {
    // Mini transformer block: LayerNorm -> Linear -> ReLU -> Linear
    let out = eval_output(
        "let x = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]); \
         let g = Tensor.ones([2]); \
         let b = Tensor.zeros([2]); \
         let normed = x.layer_norm(g, b, 0.00001); \
         let w1 = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]); \
         let b1 = Tensor.from_vec([0.0, 0.0], [2]); \
         let h = normed.linear(w1, b1); \
         let h2 = h.relu(); \
         let w2 = Tensor.from_vec([1.0, 1.0, 1.0, 1.0], [2, 2]); \
         let b2 = Tensor.from_vec([0.0, 0.0], [2]); \
         let out = h2.linear(w2, b2); \
         print(out.shape());"
    );
    assert_eq!(out, vec!["[2, 2]"]);
}

#[test]
fn test_transformer_block_parity() {
    assert_parity(
        "let x = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]); \
         let g = Tensor.ones([2]); \
         let b = Tensor.zeros([2]); \
         let normed = x.layer_norm(g, b, 0.00001); \
         let w1 = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]); \
         let b1 = Tensor.from_vec([0.0, 0.0], [2]); \
         let h = normed.linear(w1, b1); \
         let h2 = h.relu(); \
         print(h2.shape());"
    );
}

#[test]
fn test_attention_forward_pass() {
    let out = eval_output(
        "let x = Tensor.from_vec([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5], [4, 2]); \
         let wq = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]); \
         let wk = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]); \
         let wv = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]); \
         let bq = Tensor.zeros([2]); \
         let bk = Tensor.zeros([2]); \
         let bv = Tensor.zeros([2]); \
         let q = x.linear(wq, bq); \
         let k = x.linear(wk, bk); \
         let v = x.linear(wv, bv); \
         let attn_out = attention(q, k, v); \
         print(attn_out.shape());"
    );
    assert_eq!(out, vec!["[4, 2]"]);
}

// ═══════════════════════════════════════════════════════════════════
// Section 11: Determinism Double-Run Gate
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_determinism_softmax_double_run() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
    let run1 = t.softmax().unwrap().to_vec();
    let run2 = t.softmax().unwrap().to_vec();
    assert_eq!(run1, run2, "softmax not deterministic");
}

#[test]
fn test_determinism_layer_norm_double_run() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let g = Tensor::ones(&[3]);
    let b = Tensor::zeros(&[3]);
    let run1 = t.layer_norm(&g, &b, 1e-5).unwrap().to_vec();
    let run2 = t.layer_norm(&g, &b, 1e-5).unwrap().to_vec();
    assert_eq!(run1, run2, "layer_norm not deterministic");
}

#[test]
fn test_determinism_attention_double_run() {
    let q = Tensor::from_vec(vec![1.0, 0.5, 0.2, 0.8], &[2, 2]).unwrap();
    let k = Tensor::from_vec(vec![0.3, 0.7, 0.1, 0.9], &[2, 2]).unwrap();
    let v = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
    let run1 = Tensor::scaled_dot_product_attention(&q, &k, &v).unwrap().to_vec();
    let run2 = Tensor::scaled_dot_product_attention(&q, &k, &v).unwrap().to_vec();
    assert_eq!(run1, run2, "attention not deterministic");
}

#[test]
fn test_determinism_full_pipeline_double_run() {
    let src = "let x = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]); \
               let g = Tensor.ones([2]); \
               let b = Tensor.zeros([2]); \
               let n = x.layer_norm(g, b, 0.00001); \
               let s = n.softmax(); \
               print(s.get([0, 0])); \
               print(s.get([1, 1]));";
    let run1 = eval_output(src);
    let run2 = eval_output(src);
    assert_eq!(run1, run2, "full pipeline not deterministic");
}
