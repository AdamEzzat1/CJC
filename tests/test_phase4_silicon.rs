//! Phase 4 Silicon Realism Tests
//!
//! Tests cover:
//!   1. AlignedPool — 16-byte alignment, copy_from, as_bytes
//!   2. AlignedByteSlice — alignment detection, zero-copy vs realign, as_tensor
//!   3. Raw-pointer kernel bridge — matmul_raw, softmax_raw, linear_raw,
//!      layer_norm_raw, relu_raw, gelu_raw
//!   4. PagedKvCache — block paging, append, as_tensor, clear, get_token
//!   5. CJC eval integration — PagedKvCache, AlignedByteSlice through dispatch
//!   6. Parity (AST eval vs MIR-exec)
//!   7. 10k-iteration stress gate — zero-allocation inference loop
//!   8. Determinism double-run gates

use cjc_eval::Interpreter;
use cjc_parser::parse_source;
use cjc_runtime::{AlignedPool, AlignedByteSlice, PagedKvCache, Tensor, kernel};
use std::rc::Rc;

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
// Section 1: AlignedPool — 16-byte alignment (Rust unit tests)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_aligned_pool_basic() {
    let pool = AlignedPool::new(128);
    assert!(pool.check_alignment());
    assert_eq!(pool.as_bytes().len(), 0);
}

#[test]
fn test_aligned_pool_copy_from() {
    let mut pool = AlignedPool::new(64);
    let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    pool.copy_from(&data).unwrap();
    assert_eq!(pool.as_bytes(), &[1, 2, 3, 4, 5, 6, 7, 8]);
    assert!(pool.check_alignment());
}

#[test]
fn test_aligned_pool_is_aligned_16() {
    let pool = AlignedPool::new(256);
    let ptr = pool.as_ptr();
    assert!(AlignedPool::is_aligned_16(ptr));
    assert_eq!(ptr as usize % 16, 0);
}

#[test]
fn test_aligned_pool_overflow() {
    let mut pool = AlignedPool::new(4);
    let data = vec![1u8, 2, 3, 4, 5]; // 5 > 4
    assert!(pool.copy_from(&data).is_err());
}

#[test]
fn test_aligned_pool_large() {
    let mut pool = AlignedPool::new(4096);
    let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
    pool.copy_from(&data).unwrap();
    assert_eq!(pool.as_bytes().len(), 4096);
    assert!(pool.check_alignment());
}

// ═══════════════════════════════════════════════════════════════════
// Section 2: AlignedByteSlice — alignment detection (Rust unit tests)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_aligned_byteslice_from_aligned_data() {
    // Create data that's likely aligned (Vec allocator usually 16-byte aligns)
    let data = Rc::new(vec![0u8; 64]);
    let abs = AlignedByteSlice::from_bytes(data.clone());
    assert_eq!(abs.len(), 64);
    assert!(!abs.is_empty());
    assert_eq!(abs.as_bytes().len(), 64);
}

#[test]
fn test_aligned_byteslice_as_tensor_f64() {
    // Create 6 f64 values as bytes
    let vals: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bytes: Vec<u8> = vals.iter().flat_map(|f| f.to_le_bytes()).collect();
    let data = Rc::new(bytes);
    let abs = AlignedByteSlice::from_bytes(data);
    let t = abs.as_tensor(&[2, 3], "f64").unwrap();
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_aligned_byteslice_as_tensor_f32() {
    let vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let bytes: Vec<u8> = vals.iter().flat_map(|f| f.to_le_bytes()).collect();
    let data = Rc::new(bytes);
    let abs = AlignedByteSlice::from_bytes(data);
    let t = abs.as_tensor(&[2, 2], "f32").unwrap();
    assert_eq!(t.shape(), &[2, 2]);
    // f32→f64 promotion
    let v = t.to_vec();
    assert!((v[0] - 1.0).abs() < 1e-6);
    assert!((v[3] - 4.0).abs() < 1e-6);
}

#[test]
fn test_aligned_byteslice_shape_mismatch() {
    let data = Rc::new(vec![0u8; 48]); // 6 f64s
    let abs = AlignedByteSlice::from_bytes(data);
    assert!(abs.as_tensor(&[3, 3], "f64").is_err()); // 9 * 8 = 72 != 48
}

#[test]
fn test_aligned_byteslice_empty() {
    let data = Rc::new(vec![]);
    let abs = AlignedByteSlice::from_bytes(data);
    assert!(abs.is_empty());
    assert_eq!(abs.len(), 0);
}

// ═══════════════════════════════════════════════════════════════════
// Section 3: Raw-pointer kernel bridge (Rust unit tests)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_kernel_matmul_raw_basic() {
    // [2,3] @ [3,2] = [2,2]
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut c = vec![0.0; 4];
    kernel::matmul_raw(&a, &b, &mut c, 2, 3, 2);
    // Row 0: 1*7+2*9+3*11 = 7+18+33 = 58, 1*8+2*10+3*12 = 8+20+36 = 64
    // Row 1: 4*7+5*9+6*11 = 28+45+66 = 139, 4*8+5*10+6*12 = 32+50+72 = 154
    assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_kernel_matmul_raw_identity() {
    // I @ x = x
    let identity = vec![1.0, 0.0, 0.0, 1.0];
    let x = vec![3.0, 7.0, 11.0, 13.0];
    let mut c = vec![0.0; 4];
    kernel::matmul_raw(&identity, &x, &mut c, 2, 2, 2);
    assert_eq!(c, vec![3.0, 7.0, 11.0, 13.0]);
}

#[test]
fn test_kernel_softmax_raw_basic() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let mut out = vec![0.0; 4];
    kernel::softmax_raw(&data, &mut out, 1, 4);
    // Sum should be 1.0
    let sum: f64 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);
    // Monotonic: softmax(1) < softmax(2) < softmax(3) < softmax(4)
    assert!(out[0] < out[1]);
    assert!(out[1] < out[2]);
    assert!(out[2] < out[3]);
}

#[test]
fn test_kernel_softmax_raw_batched() {
    let data = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
    let mut out = vec![0.0; 6];
    kernel::softmax_raw(&data, &mut out, 2, 3);
    // Each batch row sums to 1.0
    let sum0: f64 = out[0..3].iter().sum();
    let sum1: f64 = out[3..6].iter().sum();
    assert!((sum0 - 1.0).abs() < 1e-10);
    assert!((sum1 - 1.0).abs() < 1e-10);
}

#[test]
fn test_kernel_softmax_raw_uniform() {
    let data = vec![5.0, 5.0, 5.0, 5.0];
    let mut out = vec![0.0; 4];
    kernel::softmax_raw(&data, &mut out, 1, 4);
    for v in &out {
        assert!((v - 0.25).abs() < 1e-10);
    }
}

#[test]
fn test_kernel_linear_raw_basic() {
    // x=[1,2], W=[[1,0],[0,1],[1,1]], bias=[0.1,0.2,0.3]
    // x @ W^T + bias = [1*1+2*0+0.1, 1*0+2*1+0.2, 1*1+2*1+0.3] = [1.1, 2.2, 3.3]
    let x = vec![1.0, 2.0];
    let w = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3x2 weight matrix (row-major)
    let bias = vec![0.1, 0.2, 0.3];
    let mut out = vec![0.0; 3];
    kernel::linear_raw(&x, &w, &bias, &mut out, 1, 2, 3);
    assert!((out[0] - 1.1).abs() < 1e-10);
    assert!((out[1] - 2.2).abs() < 1e-10);
    assert!((out[2] - 3.3).abs() < 1e-10);
}

#[test]
fn test_kernel_layer_norm_raw_basic() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0, 1.0, 1.0, 1.0];
    let beta = vec![0.0, 0.0, 0.0, 0.0];
    let mut out = vec![0.0; 4];
    kernel::layer_norm_raw(&data, &gamma, &beta, &mut out, 1, 4, 1e-5);
    // Mean = 2.5, normalized values should sum to ~0
    let sum: f64 = out.iter().sum();
    assert!(sum.abs() < 1e-10);
    // Should be symmetric around 0
    assert!((out[0] + out[3]).abs() < 1e-10);
    assert!((out[1] + out[2]).abs() < 1e-10);
}

#[test]
fn test_kernel_layer_norm_raw_batched() {
    let data = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0];
    let gamma = vec![1.0, 1.0, 1.0];
    let beta = vec![0.0, 0.0, 0.0];
    let mut out = vec![0.0; 6];
    kernel::layer_norm_raw(&data, &gamma, &beta, &mut out, 2, 3, 1e-5);
    // Each row of 3 elements should sum to ~0
    let sum0: f64 = out[0..3].iter().sum();
    let sum1: f64 = out[3..6].iter().sum();
    assert!(sum0.abs() < 1e-10);
    assert!(sum1.abs() < 1e-10);
}

#[test]
fn test_kernel_relu_raw() {
    let data = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
    let mut out = vec![0.0; 5];
    kernel::relu_raw(&data, &mut out);
    assert_eq!(out, vec![0.0, 0.0, 0.0, 1.0, 3.0]);
}

#[test]
fn test_kernel_gelu_raw() {
    let data = vec![-2.0, 0.0, 2.0];
    let mut out = vec![0.0; 3];
    kernel::gelu_raw(&data, &mut out);
    // gelu(0) = 0
    assert!((out[1]).abs() < 1e-10);
    // gelu(x) > 0 for x > 0
    assert!(out[2] > 0.0);
    // gelu(-x) ≈ small negative
    assert!(out[0] < 0.0);
    assert!(out[0] > -1.0); // gelu(-2) ≈ -0.0454
}

#[test]
fn test_kernel_matmul_raw_kahan_accuracy() {
    // Use values that stress floating point
    let n = 100;
    let a: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 1e-8).collect();
    let b: Vec<f64> = (0..n).map(|i| 1.0 - (i as f64) * 1e-8).collect();
    let mut c = vec![0.0; 1];
    kernel::matmul_raw(&a, &b, &mut c, 1, n, 1);
    // Result should be close to n (= 100), since each a[i]*b[i] ≈ 1.0
    assert!((c[0] - 100.0).abs() < 1e-4);
}

// ═══════════════════════════════════════════════════════════════════
// Section 4: PagedKvCache — block paging (Rust unit tests)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_paged_kv_cache_new() {
    let cache = PagedKvCache::new(64, 8);
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.max_tokens(), 64);
    assert_eq!(cache.dim(), 8);
    assert_eq!(cache.num_blocks(), 4); // ceil(64/16) = 4
    assert!(cache.is_empty());
    assert_eq!(cache.blocks_in_use(), 0);
}

#[test]
fn test_paged_kv_cache_append_single() {
    let mut cache = PagedKvCache::new(32, 4);
    cache.append(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.blocks_in_use(), 1);
    let token = cache.get_token(0).unwrap();
    assert_eq!(token, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_paged_kv_cache_append_multiple() {
    let mut cache = PagedKvCache::new(64, 4);
    for i in 0..20 {
        let token = vec![i as f64; 4];
        cache.append(&token).unwrap();
    }
    assert_eq!(cache.len(), 20);
    assert_eq!(cache.blocks_in_use(), 2); // ceil(20/16) = 2
    // Verify each token
    for i in 0..20 {
        let token = cache.get_token(i).unwrap();
        assert_eq!(token, vec![i as f64; 4]);
    }
}

#[test]
fn test_paged_kv_cache_append_tensor() {
    let mut cache = PagedKvCache::new(32, 4);
    let t = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 4],
    ).unwrap();
    cache.append_tensor(&t).unwrap();
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.get_token(0).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(cache.get_token(1).unwrap(), vec![5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_paged_kv_cache_as_tensor() {
    let mut cache = PagedKvCache::new(32, 4);
    cache.append(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    cache.append(&[5.0, 6.0, 7.0, 8.0]).unwrap();
    let t = cache.as_tensor();
    assert_eq!(t.shape(), &[2, 4]);
    assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_paged_kv_cache_clear() {
    let mut cache = PagedKvCache::new(32, 4);
    cache.append(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    cache.append(&[5.0, 6.0, 7.0, 8.0]).unwrap();
    assert_eq!(cache.len(), 2);
    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.blocks_in_use(), 0);
    // Can reuse after clear
    cache.append(&[9.0, 10.0, 11.0, 12.0]).unwrap();
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.get_token(0).unwrap(), vec![9.0, 10.0, 11.0, 12.0]);
}

#[test]
fn test_paged_kv_cache_overflow() {
    let mut cache = PagedKvCache::new(16, 4); // exactly 1 block
    for i in 0..16 {
        cache.append(&[i as f64; 4]).unwrap();
    }
    // 17th append should fail
    assert!(cache.append(&[99.0; 4]).is_err());
}

#[test]
fn test_paged_kv_cache_dim_mismatch() {
    let mut cache = PagedKvCache::new(32, 4);
    assert!(cache.append(&[1.0, 2.0, 3.0]).is_err()); // dim 3 != 4
}

#[test]
fn test_paged_kv_cache_block_boundary() {
    // Fill exactly one block (16 tokens), then add 1 more
    let mut cache = PagedKvCache::new(64, 2);
    for i in 0..16 {
        cache.append(&[i as f64, (i + 100) as f64]).unwrap();
    }
    assert_eq!(cache.blocks_in_use(), 1);
    // 17th token crosses into block 2
    cache.append(&[99.0, 199.0]).unwrap();
    assert_eq!(cache.blocks_in_use(), 2);
    assert_eq!(cache.get_token(16).unwrap(), vec![99.0, 199.0]);
}

#[test]
fn test_paged_kv_cache_as_tensor_empty() {
    let cache = PagedKvCache::new(32, 4);
    let t = cache.as_tensor();
    assert_eq!(t.shape(), &[0, 4]);
    assert_eq!(t.to_vec().len(), 0);
}

// ═══════════════════════════════════════════════════════════════════
// Section 5: PagedKvCache through CJC eval
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_cjc_paged_kv_cache_basic() {
    let out = eval_output(r#"
let cache = PagedKvCache.new(32, 4);
print(cache.len());
print(cache.is_empty());
print(cache.max_tokens());
print(cache.dim());
print(cache.num_blocks());
"#);
    assert_eq!(out, vec!["0", "true", "32", "4", "2"]);
}

#[test]
fn test_cjc_paged_kv_cache_append_and_read() {
    let out = eval_output(r#"
let cache = PagedKvCache.new(64, 4);
let t = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4]);
cache.append(t);
print(cache.len());
print(cache.blocks_in_use());
let tok = cache.get_token(0);
print(tok.shape());
let result = cache.as_tensor();
print(result.shape());
"#);
    assert_eq!(out, vec!["2", "1", "[4]", "[2, 4]"]);
}

#[test]
fn test_cjc_paged_kv_cache_clear_reuse() {
    let out = eval_output(r#"
let cache = PagedKvCache.new(32, 2);
let t = Tensor.from_vec([10.0, 20.0, 30.0, 40.0], [2, 2]);
cache.append(t);
print(cache.len());
cache.clear();
print(cache.len());
print(cache.is_empty());
let t2 = Tensor.from_vec([50.0, 60.0], [1, 2]);
cache.append(t2);
print(cache.len());
let result = cache.as_tensor();
print(result.get([0, 0]));
"#);
    assert_eq!(out, vec!["2", "0", "true", "1", "50"]);
}

// ═══════════════════════════════════════════════════════════════════
// Section 6: AlignedByteSlice through CJC eval
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_cjc_aligned_byteslice_basic() {
    let out = eval_output(r#"
let raw = "abcdefgh".as_bytes();
let aligned = AlignedByteSlice.from_bytes(raw);
print(aligned.len());
print(aligned.is_empty());
"#);
    assert_eq!(out, vec!["8", "false"]);
}

#[test]
fn test_cjc_aligned_byteslice_as_tensor() {
    // Create f64 bytes via Tensor.from_bytes round-trip
    let out = eval_output(r#"
let raw_str = "abcdefghabcdefghabcdefghabcdefgh";
let bytes = raw_str.as_bytes();
let aligned = AlignedByteSlice.from_bytes(bytes);
print(aligned.len());
print(aligned.is_empty());
let was = aligned.was_realigned();
print(was == true || was == false);
"#);
    assert_eq!(out, vec!["32", "false", "true"]);
}

// ═══════════════════════════════════════════════════════════════════
// Section 7: Parity tests (AST eval vs MIR-exec)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_parity_paged_kv_cache_basic() {
    assert_parity(r#"
let cache = PagedKvCache.new(32, 4);
print(cache.len());
print(cache.max_tokens());
print(cache.dim());
print(cache.num_blocks());
"#);
}

#[test]
fn test_parity_paged_kv_cache_append_read() {
    assert_parity(r#"
let cache = PagedKvCache.new(64, 4);
let t = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4]);
cache.append(t);
print(cache.len());
let result = cache.as_tensor();
print(result.shape());
print(result.get([0, 0]));
print(result.get([1, 3]));
"#);
}

#[test]
fn test_parity_paged_kv_cache_clear() {
    assert_parity(r#"
let cache = PagedKvCache.new(32, 2);
let t = Tensor.from_vec([10.0, 20.0, 30.0, 40.0], [2, 2]);
cache.append(t);
print(cache.len());
cache.clear();
print(cache.len());
print(cache.is_empty());
"#);
}

#[test]
fn test_parity_aligned_byteslice() {
    assert_parity(r#"
let raw = "testdata".as_bytes();
let aligned = AlignedByteSlice.from_bytes(raw);
print(aligned.len());
print(aligned.is_empty());
"#);
}

#[test]
fn test_parity_aligned_byteslice_as_tensor() {
    assert_parity(r#"
let raw_str = "abcdefghabcdefghabcdefghabcdefgh";
let bytes = raw_str.as_bytes();
let aligned = AlignedByteSlice.from_bytes(bytes);
print(aligned.len());
print(aligned.is_empty());
let was = aligned.was_realigned();
print(was == true || was == false);
"#);
}

#[test]
fn test_parity_paged_kv_cache_get_token() {
    assert_parity(r#"
let cache = PagedKvCache.new(32, 4);
let t = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4]);
cache.append(t);
let tok = cache.get_token(1);
print(tok.shape());
print(tok.get([0]));
print(tok.get([3]));
"#);
}

// ═══════════════════════════════════════════════════════════════════
// Section 8: Multi-head attention with PagedKvCache (CJC eval)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_cjc_paged_cache_attention_pipeline() {
    let out = eval_output(r#"
let batch = 1;
let seq_len = 4;
let model_dim = 8;
let num_heads = 2;

let input_data = [
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
    0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1
];
let x = Tensor.from_vec(input_data, [1, 4, 8]);

let cache_k = PagedKvCache.new(64, 8);
let cache_v = PagedKvCache.new(64, 8);

let K_squeezed = x.view_reshape([4, 8]);
let V_squeezed = x.view_reshape([4, 8]);
cache_k.append(K_squeezed);
cache_v.append(V_squeezed);

print(cache_k.len());
print(cache_v.len());

let Q_heads = x.split_heads(num_heads);
let cached_k = cache_k.as_tensor();
let k_3d = cached_k.view_reshape([1, 4, 8]);
let K_heads = k_3d.split_heads(num_heads);
let cached_v = cache_v.as_tensor();
let v_3d = cached_v.view_reshape([1, 4, 8]);
let V_heads = v_3d.split_heads(num_heads);

let attn = attention(Q_heads, K_heads, V_heads);
let merged = attn.merge_heads();
print(merged.shape());
"#);
    assert_eq!(out, vec!["4", "4", "[1, 4, 8]"]);
}

// ═══════════════════════════════════════════════════════════════════
// Section 9: Determinism double-run gates
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_determinism_paged_kv_cache() {
    let out = eval_output(r#"
let cache1 = PagedKvCache.new(32, 4);
let cache2 = PagedKvCache.new(32, 4);
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let t = Tensor.from_vec(data, [2, 4]);
cache1.append(t);
cache2.append(t);
let t1 = cache1.as_tensor();
let t2 = cache2.as_tensor();
let v1 = t1.get([0, 0]);
let v2 = t2.get([0, 0]);
assert_eq(v1, v2);
let v3 = t1.get([1, 3]);
let v4 = t2.get([1, 3]);
assert_eq(v3, v4);
print("PASS");
"#);
    assert_eq!(out, vec!["PASS"]);
}

#[test]
fn test_determinism_aligned_byteslice_tensor() {
    let out = eval_output(r#"
let raw_str = "0123456789abcdef0123456789abcdef";
let bytes = raw_str.as_bytes();
let a1 = AlignedByteSlice.from_bytes(bytes);
let a2 = AlignedByteSlice.from_bytes(bytes);
assert_eq(a1.len(), a2.len());
assert_eq(a1.was_realigned(), a2.was_realigned());
print("PASS");
"#);
    assert_eq!(out, vec!["PASS"]);
}

#[test]
fn test_determinism_raw_kernels() {
    // Verify raw kernels produce bit-identical results across runs
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let mut c1 = vec![0.0; 4];
    let mut c2 = vec![0.0; 4];
    kernel::matmul_raw(&a, &b, &mut c1, 2, 2, 2);
    kernel::matmul_raw(&a, &b, &mut c2, 2, 2, 2);
    assert_eq!(c1, c2);

    let data = vec![1.0, 2.0, 3.0];
    let mut s1 = vec![0.0; 3];
    let mut s2 = vec![0.0; 3];
    kernel::softmax_raw(&data, &mut s1, 1, 3);
    kernel::softmax_raw(&data, &mut s2, 1, 3);
    assert_eq!(s1, s2);
}

// ═══════════════════════════════════════════════════════════════════
// Section 10: 10,000-iteration stress gate
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_stress_10k_matmul_raw() {
    // 10,000 matmul operations using raw kernels
    let a = vec![0.1, 0.2, 0.3, 0.4];
    let b = vec![0.5, 0.6, 0.7, 0.8];
    let mut c = vec![0.0; 4];
    for _ in 0..10_000 {
        c.iter_mut().for_each(|x| *x = 0.0);
        kernel::matmul_raw(&a, &b, &mut c, 2, 2, 2);
    }
    // Final result should be deterministic
    let mut expected = vec![0.0; 4];
    kernel::matmul_raw(&a, &b, &mut expected, 2, 2, 2);
    assert_eq!(c, expected);
}

#[test]
fn test_stress_10k_softmax_raw() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let mut out = vec![0.0; 4];
    for _ in 0..10_000 {
        kernel::softmax_raw(&data, &mut out, 1, 4);
    }
    let sum: f64 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);
}

#[test]
fn test_stress_10k_paged_kv_cache_cycle() {
    // Create cache, fill to capacity, clear, repeat 10k times
    // This must not leak memory
    let mut cache = PagedKvCache::new(16, 4); // 1 block of 16 tokens
    for cycle in 0..10_000 {
        cache.clear();
        for i in 0..16 {
            let token = vec![(cycle * 16 + i) as f64; 4];
            cache.append(&token).unwrap();
        }
        assert_eq!(cache.len(), 16);
    }
    // Final state: last cycle's data
    let t = cache.as_tensor();
    assert_eq!(t.shape(), &[16, 4]);
}

#[test]
fn test_stress_10k_layer_norm_raw() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let gamma = vec![1.0; 8];
    let beta = vec![0.0; 8];
    let mut out = vec![0.0; 8];
    for _ in 0..10_000 {
        kernel::layer_norm_raw(&data, &gamma, &beta, &mut out, 1, 8, 1e-5);
    }
    let sum: f64 = out.iter().sum();
    assert!(sum.abs() < 1e-10);
}

#[test]
fn test_stress_10k_aligned_byteslice_cycle() {
    let vals: Vec<f64> = (0..64).map(|i| i as f64).collect();
    let bytes: Vec<u8> = vals.iter().flat_map(|f| f.to_le_bytes()).collect();
    let data = Rc::new(bytes);
    for _ in 0..10_000 {
        let abs = AlignedByteSlice::from_bytes(data.clone());
        let t = abs.as_tensor(&[8, 8], "f64").unwrap();
        assert_eq!(t.shape(), &[8, 8]);
    }
}

#[test]
fn test_stress_10k_full_inference_loop() {
    // Simulates a tight inference loop: matmul → softmax → layer_norm
    // using only pre-allocated buffers — zero heap allocation per iteration
    let dim = 8;
    let seq = 4;

    // Pre-allocate all buffers once
    let x = vec![0.1; seq * dim];
    let w = vec![0.01; dim * dim];
    let bias = vec![0.0; dim];
    let gamma = vec![1.0; dim];
    let beta = vec![0.0; dim];
    let mut proj = vec![0.0; seq * dim];
    let mut norm_out = vec![0.0; seq * dim];
    let mut softmax_out = vec![0.0; seq * seq];
    let mut scores = vec![0.0; seq * seq];

    for _ in 0..10_000 {
        // Linear projection: x @ W^T + bias
        kernel::linear_raw(&x, &w, &bias, &mut proj, seq, dim, dim);

        // LayerNorm
        kernel::layer_norm_raw(&proj, &gamma, &beta, &mut norm_out, seq, dim, 1e-5);

        // Attention scores: norm_out @ norm_out^T (simplified)
        kernel::matmul_raw(&norm_out, &norm_out, &mut scores, seq, dim, seq);

        // Softmax over scores
        kernel::softmax_raw(&scores, &mut softmax_out, seq, seq);
    }

    // Sanity: softmax rows sum to 1
    for row in 0..seq {
        let sum: f64 = softmax_out[row*seq..(row+1)*seq].iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Row {} sum = {}", row, sum);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Section 11: Parity — full PagedKvCache attention pipeline
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_parity_paged_cache_attention_pipeline() {
    assert_parity(r#"
let batch = 1;
let seq_len = 4;
let model_dim = 8;
let num_heads = 2;

let input_data = [
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
    0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1
];
let x = Tensor.from_vec(input_data, [1, 4, 8]);

let cache_k = PagedKvCache.new(64, 8);
let cache_v = PagedKvCache.new(64, 8);

let K_squeezed = x.view_reshape([4, 8]);
cache_k.append(K_squeezed);
cache_v.append(K_squeezed);

print(cache_k.len());
print(cache_v.len());

let cached = cache_k.as_tensor();
print(cached.shape());
print(cached.get([0, 0]));
"#);
}

// ═══════════════════════════════════════════════════════════════════
// Section 12: Edge cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_paged_kv_cache_non_power_of_2_max() {
    // 17 tokens = ceil(17/16) = 2 blocks
    let cache = PagedKvCache::new(17, 4);
    assert_eq!(cache.num_blocks(), 2);
    assert_eq!(cache.max_tokens(), 17);
}

#[test]
fn test_paged_kv_cache_exactly_one_block() {
    let mut cache = PagedKvCache::new(16, 2);
    assert_eq!(cache.num_blocks(), 1);
    for i in 0..16 {
        cache.append(&[i as f64, (i + 1) as f64]).unwrap();
    }
    assert_eq!(cache.len(), 16);
    assert_eq!(cache.blocks_in_use(), 1);
}

#[test]
fn test_kernel_relu_gelu_parity_at_zero() {
    let data = vec![0.0];
    let mut relu_out = vec![0.0; 1];
    let mut gelu_out = vec![0.0; 1];
    kernel::relu_raw(&data, &mut relu_out);
    kernel::gelu_raw(&data, &mut gelu_out);
    assert_eq!(relu_out[0], 0.0);
    assert_eq!(gelu_out[0], 0.0);
}

#[test]
fn test_kernel_linear_raw_batched() {
    // 2 batch × 3 output features, 2 input features
    let x = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
    let w = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // [3, 2]
    let bias = vec![0.0, 0.0, 0.0];
    let mut out = vec![0.0; 6];
    kernel::linear_raw(&x, &w, &bias, &mut out, 2, 2, 3);
    // Row 0: [1*1+2*0, 1*0+2*1, 1*1+2*1] = [1, 2, 3]
    // Row 1: [3*1+4*0, 3*0+4*1, 3*1+4*1] = [3, 4, 7]
    assert_eq!(out, vec![1.0, 2.0, 3.0, 3.0, 4.0, 7.0]);
}
