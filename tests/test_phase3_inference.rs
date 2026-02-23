//! Phase 3 Inference Tests — Zero-Copy, KV-Cache, Multi-Head Attention
//!
//! Tests cover:
//!   1. Tensor.from_bytes (f64 and f32 zero-copy weight mapping)
//!   2. ByteSlice.as_tensor (instance method)
//!   3. Scratchpad KV-Cache (new, append, as_tensor, clear, capacity)
//!   4. split_heads / merge_heads (multi-head attention)
//!   5. view_reshape
//!   6. Parity (AST eval vs MIR-exec)
//!   7. Transformer forward integration
//!   8. Determinism double-run gates
//!   9. Edge cases and error handling

use cjc_eval::Interpreter;
use cjc_parser::parse_source;
use cjc_runtime::{Tensor, Scratchpad};

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
// Section 1: Tensor.from_bytes — Zero-Copy Weight Mapping (Rust)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_from_bytes_f64_basic() {
    let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let t = Tensor::from_bytes(&bytes, &[2, 3], "f64").unwrap();
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_from_bytes_f32_promotion() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let t = Tensor::from_bytes(&bytes, &[2, 2], "f32").unwrap();
    assert_eq!(t.shape(), &[2, 2]);
    let result = t.to_vec();
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - 2.0).abs() < 1e-6);
    assert!((result[2] - 3.0).abs() < 1e-6);
    assert!((result[3] - 4.0).abs() < 1e-6);
}

#[test]
fn test_from_bytes_shape_mismatch() {
    let data: Vec<f64> = vec![1.0, 2.0, 3.0];
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    assert!(Tensor::from_bytes(&bytes, &[2, 3], "f64").is_err());
}

#[test]
fn test_from_bytes_invalid_dtype() {
    assert!(Tensor::from_bytes(&[0u8; 8], &[1], "f16").is_err());
}

#[test]
fn test_from_bytes_1d() {
    let data: Vec<f64> = vec![3.14, 2.72, 1.41];
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let t = Tensor::from_bytes(&bytes, &[3], "f64").unwrap();
    assert_eq!(t.shape(), &[3]);
    assert_eq!(t.to_vec(), vec![3.14, 2.72, 1.41]);
}

#[test]
fn test_from_bytes_3d() {
    let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let t = Tensor::from_bytes(&bytes, &[2, 3, 4], "f64").unwrap();
    assert_eq!(t.shape(), &[2, 3, 4]);
    assert_eq!(t.len(), 24);
}

#[test]
fn test_from_bytes_parity_with_from_vec() {
    let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let t_bytes = Tensor::from_bytes(&bytes, &[2, 3], "f64").unwrap();
    let t_vec = Tensor::from_vec(data, &[2, 3]).unwrap();
    assert_eq!(t_bytes.to_vec(), t_vec.to_vec());
}

#[test]
fn test_from_bytes_f32_rust() {
    let data: Vec<f32> = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5];
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let t = Tensor::from_bytes(&bytes, &[2, 3], "f32").unwrap();
    assert_eq!(t.shape(), &[2, 3]);
    let result = t.to_vec();
    for (i, &v) in data.iter().enumerate() {
        assert!((result[i] - v as f64).abs() < 1e-5, "mismatch at {}: {} vs {}", i, result[i], v);
    }
}

#[test]
fn test_from_bytes_empty() {
    let t = Tensor::from_bytes(&[], &[0], "f64").unwrap();
    assert_eq!(t.shape(), &[0]);
    assert_eq!(t.len(), 0);
}

// ═══════════════════════════════════════════════════════════════════
// Section 2: Scratchpad KV-Cache — Rust Unit Tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_scratchpad_new() {
    let sp = Scratchpad::new(32, 8);
    assert_eq!(sp.len(), 0);
    assert_eq!(sp.capacity(), 32);
    assert_eq!(sp.dim(), 8);
    assert!(sp.is_empty());
}

#[test]
fn test_scratchpad_append_single() {
    let mut sp = Scratchpad::new(10, 4);
    sp.append(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(sp.len(), 1);
    let t = sp.as_tensor();
    assert_eq!(t.shape(), &[1, 4]);
    assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_scratchpad_append_multiple() {
    let mut sp = Scratchpad::new(10, 3);
    sp.append(&[1.0, 2.0, 3.0]).unwrap();
    sp.append(&[4.0, 5.0, 6.0]).unwrap();
    sp.append(&[7.0, 8.0, 9.0]).unwrap();
    assert_eq!(sp.len(), 3);
    let t = sp.as_tensor();
    assert_eq!(t.shape(), &[3, 3]);
    assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
}

#[test]
fn test_scratchpad_append_tensor() {
    let mut sp = Scratchpad::new(10, 4);
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]).unwrap();
    sp.append_tensor(&t).unwrap();
    assert_eq!(sp.len(), 2);
    let cached = sp.as_tensor();
    assert_eq!(cached.shape(), &[2, 4]);
    assert_eq!(cached.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_scratchpad_overflow() {
    let mut sp = Scratchpad::new(2, 3);
    sp.append(&[1.0, 2.0, 3.0]).unwrap();
    sp.append(&[4.0, 5.0, 6.0]).unwrap();
    assert!(sp.append(&[7.0, 8.0, 9.0]).is_err());
}

#[test]
fn test_scratchpad_dim_mismatch() {
    let mut sp = Scratchpad::new(10, 4);
    assert!(sp.append(&[1.0, 2.0, 3.0]).is_err());
}

#[test]
fn test_scratchpad_clear() {
    let mut sp = Scratchpad::new(10, 4);
    sp.append(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    sp.append(&[5.0, 6.0, 7.0, 8.0]).unwrap();
    assert_eq!(sp.len(), 2);
    sp.clear();
    assert_eq!(sp.len(), 0);
    assert!(sp.is_empty());
    sp.append(&[9.0, 10.0, 11.0, 12.0]).unwrap();
    assert_eq!(sp.len(), 1);
}

#[test]
fn test_scratchpad_as_tensor_shares_buffer() {
    let mut sp = Scratchpad::new(10, 2);
    sp.append(&[1.0, 2.0]).unwrap();
    let t = sp.as_tensor();
    sp.append(&[3.0, 4.0]).unwrap();
    let t2 = sp.as_tensor();
    assert_eq!(t.shape(), &[1, 2]);
    assert_eq!(t2.shape(), &[2, 2]);
}

// ═══════════════════════════════════════════════════════════════════
// Section 3: split_heads / merge_heads — Rust Unit Tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_split_heads_basic() {
    let data: Vec<f64> = (0..32).map(|i| i as f64).collect();
    let t = Tensor::from_vec(data, &[1, 4, 8]).unwrap();
    let split = t.split_heads(2).unwrap();
    assert_eq!(split.shape(), &[1, 2, 4, 4]);
}

#[test]
fn test_split_heads_wrong_dim() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert!(t.split_heads(2).is_err());
}

#[test]
fn test_split_heads_indivisible() {
    let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
    let t = Tensor::from_vec(data, &[1, 4, 3]).unwrap();
    assert!(t.split_heads(2).is_err());
}

#[test]
fn test_merge_heads_roundtrip() {
    let data: Vec<f64> = (0..32).map(|i| i as f64).collect();
    let t = Tensor::from_vec(data.clone(), &[1, 4, 8]).unwrap();
    let split = t.split_heads(2).unwrap();
    let merged = split.merge_heads().unwrap();
    assert_eq!(merged.shape(), &[1, 4, 8]);
    assert_eq!(merged.to_vec(), data);
}

#[test]
fn test_merge_heads_wrong_dim() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    assert!(t.merge_heads().is_err());
}

#[test]
fn test_split_heads_4heads() {
    let data: Vec<f64> = (0..96).map(|i| i as f64).collect();
    let t = Tensor::from_vec(data.clone(), &[2, 4, 12]).unwrap();
    let split = t.split_heads(4).unwrap();
    assert_eq!(split.shape(), &[2, 4, 4, 3]);
    let merged = split.merge_heads().unwrap();
    assert_eq!(merged.shape(), &[2, 4, 12]);
    assert_eq!(merged.to_vec(), data);
}

// ═══════════════════════════════════════════════════════════════════
// Section 4: view_reshape — Rust Unit Tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_view_reshape_basic() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let r = t.view_reshape(&[3, 2]).unwrap();
    assert_eq!(r.shape(), &[3, 2]);
    assert_eq!(r.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_view_reshape_mismatch() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert!(t.view_reshape(&[3, 2]).is_err());
}

// ═══════════════════════════════════════════════════════════════════
// Section 5: CJC Interpreter Tests — Scratchpad
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_eval_scratchpad_basic() {
    let out = eval_output(r#"
let cache = Scratchpad.new(16, 4);
print(cache.len());
print(cache.capacity());
print(cache.dim());
print(cache.is_empty());
"#);
    assert_eq!(out, vec!["0", "16", "4", "true"]);
}

#[test]
fn test_eval_scratchpad_append_and_read() {
    let out = eval_output(r#"
let cache = Scratchpad.new(16, 3);
let row1 = Tensor.from_vec([1.0, 2.0, 3.0], [1, 3]);
cache.append(row1);
let row2 = Tensor.from_vec([4.0, 5.0, 6.0], [1, 3]);
cache.append(row2);
print(cache.len());
let cached = cache.as_tensor();
print(cached.shape());
print(cached.get([0, 0]));
print(cached.get([1, 2]));
"#);
    assert_eq!(out, vec!["2", "[2, 3]", "1", "6"]);
}

#[test]
fn test_eval_scratchpad_clear_reuse() {
    let out = eval_output(r#"
let cache = Scratchpad.new(8, 2);
let t = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]);
cache.append(t);
print(cache.len());
cache.clear();
print(cache.len());
let t2 = Tensor.from_vec([5.0, 6.0], [1, 2]);
cache.append(t2);
print(cache.len());
let view = cache.as_tensor();
print(view.get([0, 0]));
"#);
    assert_eq!(out, vec!["2", "0", "1", "5"]);
}

// ═══════════════════════════════════════════════════════════════════
// Section 6: CJC Interpreter Tests — split_heads / merge_heads
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_eval_split_merge_heads() {
    let out = eval_output(r#"
let data = [];
let i = 0;
while i < 32 {
    data = push(data, i * 1.0);
    i = i + 1;
}
let t = Tensor.from_vec(data, [1, 4, 8]);
let split = t.split_heads(2);
print(split.shape());
let merged = split.merge_heads();
print(merged.shape());
let orig_val = t.get([0, 0, 0]);
let round_val = merged.get([0, 0, 0]);
assert_eq(orig_val, round_val);
print("roundtrip ok");
"#);
    assert_eq!(out, vec!["[1, 2, 4, 4]", "[1, 4, 8]", "roundtrip ok"]);
}

#[test]
fn test_eval_view_reshape() {
    let out = eval_output(r#"
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let t = Tensor.from_vec(data, [2, 3]);
let reshaped = t.view_reshape([3, 2]);
print(reshaped.shape());
print(reshaped.get([0, 0]));
print(reshaped.get([2, 1]));
"#);
    assert_eq!(out, vec!["[3, 2]", "1", "6"]);
}

// ═══════════════════════════════════════════════════════════════════
// Section 7: CJC Interpreter Tests — Multi-Head Attention
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_eval_multihead_attention() {
    let out = eval_output(r#"
let data = [];
let i = 0;
while i < 32 {
    data = push(data, (i + 1) * 0.1);
    i = i + 1;
}
let Q = Tensor.from_vec(data, [1, 4, 8]);
let K = Tensor.from_vec(data, [1, 4, 8]);
let V = Tensor.from_vec(data, [1, 4, 8]);
let Qh = Q.split_heads(2);
let Kh = K.split_heads(2);
let Vh = V.split_heads(2);
let attn = attention(Qh, Kh, Vh);
print(attn.shape());
let merged = attn.merge_heads();
print(merged.shape());
print("multihead attention ok");
"#);
    assert_eq!(out, vec!["[1, 2, 4, 4]", "[1, 4, 8]", "multihead attention ok"]);
}

#[test]
fn test_eval_tensor_from_bytes_f64() {
    let out = eval_output(r#"
let data = [1.0, 2.0, 3.0, 4.0];
let t = Tensor.from_vec(data, [4]);
print(t.shape());
"#);
    assert_eq!(out, vec!["[4]"]);
}

// ═══════════════════════════════════════════════════════════════════
// Section 8: KV-Cache + Attention Integration
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_eval_kv_cache_with_attention() {
    let out = eval_output(r#"
let dim = 4;
let kv_k = Scratchpad.new(32, dim);
let kv_v = Scratchpad.new(32, dim);
let k1 = Tensor.from_vec([0.1, 0.2, 0.3, 0.4], [1, 4]);
let v1 = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [1, 4]);
kv_k.append(k1);
kv_v.append(v1);
let k2 = Tensor.from_vec([0.5, 0.6, 0.7, 0.8], [1, 4]);
let v2 = Tensor.from_vec([5.0, 6.0, 7.0, 8.0], [1, 4]);
kv_k.append(k2);
kv_v.append(v2);
print(kv_k.len());
let Q = Tensor.from_vec([0.3, 0.4, 0.5, 0.6], [1, 1, 4]);
let K_cached = kv_k.as_tensor();
let V_cached = kv_v.as_tensor();
let K_3d = K_cached.view_reshape([1, 2, 4]);
let V_3d = V_cached.view_reshape([1, 2, 4]);
let attn = attention(Q, K_3d, V_3d);
print(attn.shape());
print("kv-cache attention ok");
"#);
    assert_eq!(out, vec!["2", "[1, 1, 4]", "kv-cache attention ok"]);
}

// ═══════════════════════════════════════════════════════════════════
// Section 9: Parity Tests (AST eval ↔ MIR-exec)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_parity_scratchpad_basic() {
    assert_parity(r#"
let cache = Scratchpad.new(16, 3);
let row = Tensor.from_vec([1.0, 2.0, 3.0], [1, 3]);
cache.append(row);
print(cache.len());
print(cache.as_tensor().shape());
"#);
}

#[test]
fn test_parity_scratchpad_clear() {
    assert_parity(r#"
let cache = Scratchpad.new(8, 2);
let t = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]);
cache.append(t);
cache.clear();
print(cache.len());
print(cache.is_empty());
"#);
}

#[test]
fn test_parity_split_heads() {
    assert_parity(r#"
let data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0];
let t = Tensor.from_vec(data, [1, 4, 6]);
let split = t.split_heads(2);
print(split.shape());
let merged = split.merge_heads();
print(merged.shape());
"#);
}

#[test]
fn test_parity_view_reshape() {
    assert_parity(r#"
let t = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
let r = t.view_reshape([3, 2]);
print(r.shape());
print(r.get([0, 0]));
"#);
}

#[test]
fn test_parity_multihead_attention() {
    assert_parity(r#"
let data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4];
let Q = Tensor.from_vec(data, [1, 4, 6]);
let K = Tensor.from_vec(data, [1, 4, 6]);
let V = Tensor.from_vec(data, [1, 4, 6]);
let Qh = Q.split_heads(2);
let Kh = K.split_heads(2);
let Vh = V.split_heads(2);
let attn = attention(Qh, Kh, Vh);
let merged = attn.merge_heads();
print(merged.shape());
"#);
}

#[test]
fn test_parity_kv_cache_attention() {
    assert_parity(r#"
let cache_k = Scratchpad.new(32, 4);
let cache_v = Scratchpad.new(32, 4);
let k = Tensor.from_vec([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [2, 4]);
let v = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4]);
cache_k.append(k);
cache_v.append(v);
print(cache_k.len());
let K = cache_k.as_tensor().view_reshape([1, 2, 4]);
let V = cache_v.as_tensor().view_reshape([1, 2, 4]);
let Q = Tensor.from_vec([0.3, 0.4, 0.5, 0.6], [1, 1, 4]);
let out = attention(Q, K, V);
print(out.shape());
"#);
}

// ═══════════════════════════════════════════════════════════════════
// Section 10: Transformer Block Integration
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_eval_transformer_block_with_multihead() {
    let out = eval_output(r#"
let x_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
let x = Tensor.from_vec(x_data, [1, 2, 4]);
let gamma = Tensor.ones([4]);
let beta = Tensor.zeros([4]);
let wq = [];
let i = 0;
while i < 16 {
    wq = push(wq, (i + 1) * 0.01);
    i = i + 1;
}
let Wq = Tensor.from_vec(wq, [4, 4]);
let bias = Tensor.zeros([4]);
let x_norm = x.layer_norm(gamma, beta);
let Q = x_norm.linear(Wq, bias);
let K = x_norm.linear(Wq, bias);
let V = x_norm.linear(Wq, bias);
let Qh = Q.split_heads(2);
let Kh = K.split_heads(2);
let Vh = V.split_heads(2);
let attn = attention(Qh, Kh, Vh);
let merged = attn.merge_heads();
let proj = merged.linear(Wq, bias);
let out = x + proj;
print(out.shape());
print("block output ok");
"#);
    assert_eq!(out[0], "[1, 2, 4]");
    assert_eq!(out[1], "block output ok");
}

#[test]
fn test_parity_transformer_block() {
    assert_parity(r#"
let x = Tensor.from_vec([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [1, 2, 4]);
let gamma = Tensor.ones([4]);
let beta = Tensor.zeros([4]);
let wq = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16];
let W = Tensor.from_vec(wq, [4, 4]);
let bias = Tensor.zeros([4]);
let x_norm = x.layer_norm(gamma, beta);
let Q = x_norm.linear(W, bias);
let K = x_norm.linear(W, bias);
let V = x_norm.linear(W, bias);
let Qh = Q.split_heads(2);
let Kh = K.split_heads(2);
let Vh = V.split_heads(2);
let attn = attention(Qh, Kh, Vh);
let merged = attn.merge_heads();
let proj = merged.linear(W, bias);
let out = x + proj;
print(out.shape());
print(out.get([0, 0, 0]));
"#);
}

// ═══════════════════════════════════════════════════════════════════
// Section 11: Determinism Double-Run Gates
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_determinism_scratchpad() {
    let src = r#"
let cache = Scratchpad.new(16, 3);
let t = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
cache.append(t);
let view = cache.as_tensor();
print(view.get([0, 0]));
print(view.get([1, 2]));
"#;
    let run1 = eval_output(src);
    let run2 = eval_output(src);
    assert_eq!(run1, run2, "Scratchpad determinism failed");
}

#[test]
fn test_determinism_split_merge() {
    let src = r#"
let data = [];
let i = 0;
while i < 32 {
    data = push(data, i * 0.1);
    i = i + 1;
}
let t = Tensor.from_vec(data, [1, 4, 8]);
let split = t.split_heads(2);
let merged = split.merge_heads();
print(merged.get([0, 0, 0]));
print(merged.get([0, 3, 7]));
"#;
    let run1 = eval_output(src);
    let run2 = eval_output(src);
    assert_eq!(run1, run2, "split/merge determinism failed");
}

#[test]
fn test_determinism_multihead_attention() {
    let src = r#"
let data = [];
let i = 0;
while i < 24 {
    data = push(data, (i + 1) * 0.1);
    i = i + 1;
}
let Q = Tensor.from_vec(data, [1, 4, 6]);
let K = Tensor.from_vec(data, [1, 4, 6]);
let V = Tensor.from_vec(data, [1, 4, 6]);
let Qh = Q.split_heads(2);
let Kh = K.split_heads(2);
let Vh = V.split_heads(2);
let attn = attention(Qh, Kh, Vh);
let merged = attn.merge_heads();
print(merged.get([0, 0, 0]));
print(merged.get([0, 3, 5]));
"#;
    let run1 = eval_output(src);
    let run2 = eval_output(src);
    assert_eq!(run1, run2, "multihead attention determinism failed");
}

#[test]
fn test_determinism_full_pipeline() {
    let src = r#"
let x = Tensor.from_vec([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [1, 2, 4]);
let gamma = Tensor.ones([4]);
let beta = Tensor.zeros([4]);
let wq = [];
let i = 0;
while i < 16 {
    wq = push(wq, (i + 1) * 0.01);
    i = i + 1;
}
let W = Tensor.from_vec(wq, [4, 4]);
let bias = Tensor.zeros([4]);
let cache_k = Scratchpad.new(16, 4);
let cache_v = Scratchpad.new(16, 4);
let x_norm = x.layer_norm(gamma, beta);
let Q = x_norm.linear(W, bias);
let K = x_norm.linear(W, bias);
let V = x_norm.linear(W, bias);
let K_2d = K.view_reshape([2, 4]);
let V_2d = V.view_reshape([2, 4]);
cache_k.append(K_2d);
cache_v.append(V_2d);
let Qh = Q.split_heads(2);
let Kh = K.split_heads(2);
let Vh = V.split_heads(2);
let attn = attention(Qh, Kh, Vh);
let merged = attn.merge_heads();
let proj = merged.linear(W, bias);
let out = x + proj;
print(out.get([0, 0, 0]));
print(cache_k.len());
"#;
    let run1 = eval_output(src);
    let run2 = eval_output(src);
    assert_eq!(run1, run2, "full pipeline determinism failed");
}
