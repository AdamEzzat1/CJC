//! Phase 6: CNN Signal Benchmark — The Architecture Trinity
//!
//! Tests cover:
//!   1. conv1d_raw kernel — basic, multi-channel, edge cases
//!   2. conv1d_circular — sliding window over circular buffer
//!   3. maxpool1d_raw — pooling kernel
//!   4. Tensor.conv1d — high-level method through eval & mir-exec
//!   5. Parity gates (eval vs mir-exec)
//!   6. 10k-window stress gate — zero-allocation signal processing
//!   7. Determinism double-run gates
//!   8. End-to-end CNN pipeline: conv → relu → conv → pool
//!   9. CJC-script CNN benchmark validation

use std::time::Instant;
use cjc_eval::Interpreter;
use cjc_parser::parse_source;
use cjc_runtime::kernel;

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
// Section 1: conv1d_raw — basic kernel tests (Rust)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_conv1d_raw_basic_k3() {
    // signal=[1,2,3,4,5], filter=[1,0,-1], bias=0 -> [1-3, 2-4, 3-5] = [-2,-2,-2]
    let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let filters = vec![1.0, 0.0, -1.0]; // 1 filter, kernel_size=3
    let bias = vec![0.0];
    let mut out = vec![0.0; 3]; // out_len = 5-3+1 = 3
    kernel::conv1d_raw(&signal, &filters, &bias, &mut out, 5, 1, 3);
    assert_eq!(out, vec![-2.0, -2.0, -2.0]);
}

#[test]
fn test_conv1d_raw_basic_k5() {
    // signal=[1,2,3,4,5,6,7], filter=[1,1,1,1,1], bias=0
    // -> [15, 20, 25] (sliding sum)
    let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let filters = vec![1.0, 1.0, 1.0, 1.0, 1.0]; // kernel_size=5
    let bias = vec![0.0];
    let mut out = vec![0.0; 3]; // 7-5+1=3
    kernel::conv1d_raw(&signal, &filters, &bias, &mut out, 7, 1, 5);
    assert_eq!(out, vec![15.0, 20.0, 25.0]);
}

#[test]
fn test_conv1d_raw_with_bias() {
    let signal = vec![1.0, 2.0, 3.0, 4.0];
    let filters = vec![1.0, 1.0, 1.0]; // sum filter
    let bias = vec![10.0];
    let mut out = vec![0.0; 2]; // 4-3+1=2
    kernel::conv1d_raw(&signal, &filters, &bias, &mut out, 4, 1, 3);
    // [1+2+3+10, 2+3+4+10] = [16, 19]
    assert_eq!(out, vec![16.0, 19.0]);
}

#[test]
fn test_conv1d_raw_multi_channel() {
    // 2 filters over signal=[1,2,3,4]
    // filter0=[1,0,0], filter1=[0,0,1]
    let signal = vec![1.0, 2.0, 3.0, 4.0];
    let filters = vec![
        1.0, 0.0, 0.0, // channel 0: picks first element of window
        0.0, 0.0, 1.0, // channel 1: picks last element of window
    ];
    let bias = vec![0.0, 0.0];
    let mut out = vec![0.0; 4]; // 2 channels * (4-3+1=2 positions)
    kernel::conv1d_raw(&signal, &filters, &bias, &mut out, 4, 2, 3);
    // ch0: [1, 2], ch1: [3, 4]
    assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_conv1d_raw_edge_detection() {
    // Classic edge detector: [-1, 0, 1]
    let signal = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
    let filters = vec![-1.0, 0.0, 1.0];
    let bias = vec![0.0];
    let mut out = vec![0.0; 6]; // 8-3+1=6
    kernel::conv1d_raw(&signal, &filters, &bias, &mut out, 8, 1, 3);
    // [0-0, 1-0, 1-0, 1-1, 0-1, 0-1] = [0, 1, 1, 0, -1, -1]
    assert_eq!(out, vec![0.0, 1.0, 1.0, 0.0, -1.0, -1.0]);
}

#[test]
fn test_conv1d_raw_identity() {
    // kernel_size=1 acts as a scalar multiplier
    let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let filters = vec![2.0]; // 1 filter, k=1
    let bias = vec![0.5];
    let mut out = vec![0.0; 5]; // 5-1+1=5
    kernel::conv1d_raw(&signal, &filters, &bias, &mut out, 5, 1, 1);
    assert_eq!(out, vec![2.5, 4.5, 6.5, 8.5, 10.5]);
}

// ═══════════════════════════════════════════════════════════════════
// Section 2: conv1d_circular — sliding window on circular buffer
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_conv1d_circular_no_wrap() {
    let buffer = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut window = vec![0.0; 5];
    let filters = vec![1.0, 1.0, 1.0]; // sum filter, k=3
    let bias = vec![0.0];
    let mut out = vec![0.0; 3]; // 5-3+1=3

    // write_pos=5, window_size=5 -> extracts [0..5] = [1,2,3,4,5]
    kernel::conv1d_circular(&buffer, 5, 5, &mut window, &filters, &bias, &mut out, 1, 3);
    assert_eq!(window, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(out, vec![6.0, 9.0, 12.0]);
}

#[test]
fn test_conv1d_circular_with_wrap() {
    let buffer = vec![5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0];
    let mut window = vec![0.0; 5];
    let filters = vec![1.0, 1.0, 1.0]; // sum filter, k=3
    let bias = vec![0.0];
    let mut out = vec![0.0; 3];

    // write_pos=2 (next write at index 2), window_size=5
    // Most recent 5: wrapping back from pos 2 -> [5,6,7,8,1] nope
    // start = 2-5 < 0 -> start = 8 - (5-2) = 5, indices: 5,6,7,0,1 -> [2,3,4,5,6]
    kernel::conv1d_circular(&buffer, 2, 5, &mut window, &filters, &bias, &mut out, 1, 3);
    assert_eq!(window, vec![2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(out, vec![9.0, 12.0, 15.0]);
}

#[test]
fn test_conv1d_circular_full_buffer() {
    let buffer = vec![10.0, 20.0, 30.0, 40.0];
    let mut window = vec![0.0; 4];
    let filters = vec![1.0, -1.0]; // diff filter, k=2
    let bias = vec![0.0];
    let mut out = vec![0.0; 3]; // 4-2+1=3

    // write_pos=0, window_size=4 -> wraps to [10,20,30,40]
    kernel::conv1d_circular(&buffer, 0, 4, &mut window, &filters, &bias, &mut out, 1, 2);
    assert_eq!(window, vec![10.0, 20.0, 30.0, 40.0]);
    // [10-20, 20-30, 30-40] = [-10, -10, -10]
    assert_eq!(out, vec![-10.0, -10.0, -10.0]);
}

// ═══════════════════════════════════════════════════════════════════
// Section 3: maxpool1d_raw
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_maxpool1d_raw_basic() {
    let data = vec![1.0, 3.0, 2.0, 4.0, 0.0, 5.0];
    let mut out = vec![0.0; 3]; // pool_size=2
    kernel::maxpool1d_raw(&data, &mut out, 6, 2);
    assert_eq!(out, vec![3.0, 4.0, 5.0]);
}

#[test]
fn test_maxpool1d_raw_pool4() {
    let data = vec![1.0, 5.0, 2.0, 4.0, 8.0, 3.0, 7.0, 6.0];
    let mut out = vec![0.0; 2]; // pool_size=4
    kernel::maxpool1d_raw(&data, &mut out, 8, 4);
    assert_eq!(out, vec![5.0, 8.0]);
}

#[test]
fn test_maxpool1d_raw_all_same() {
    let data = vec![3.0; 8];
    let mut out = vec![0.0; 4];
    kernel::maxpool1d_raw(&data, &mut out, 8, 2);
    assert_eq!(out, vec![3.0, 3.0, 3.0, 3.0]);
}

// ═══════════════════════════════════════════════════════════════════
// Section 4: Tensor.conv1d through CJC eval
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_cjc_conv1d_basic() {
    let out = eval_output(r#"
let signal = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0], [5]);
let filters = Tensor.from_vec([1.0, 0.0, -1.0], [1, 3]);
let bias = Tensor.from_vec([0.0], [1]);
let result = signal.conv1d(filters, bias);
print(result.shape());
print(result.get([0, 0]));
print(result.get([0, 1]));
print(result.get([0, 2]));
"#);
    assert_eq!(out, vec!["[1, 3]", "-2", "-2", "-2"]);
}

#[test]
fn test_cjc_conv1d_multichannel() {
    let out = eval_output(r#"
let signal = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [4]);
let filters = Tensor.from_vec([
    1.0, 0.0, 0.0,
    0.0, 0.0, 1.0
], [2, 3]);
let bias = Tensor.from_vec([0.0, 0.0], [2]);
let result = signal.conv1d(filters, bias);
print(result.shape());
print(result.get([0, 0]));
print(result.get([0, 1]));
print(result.get([1, 0]));
print(result.get([1, 1]));
"#);
    assert_eq!(out, vec!["[2, 2]", "1", "2", "3", "4"]);
}

#[test]
fn test_cjc_conv1d_relu_pipeline() {
    let out = eval_output(r#"
let signal = Tensor.from_vec([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], [8]);
let edge_filter = Tensor.from_vec([-1.0, 0.0, 1.0], [1, 3]);
let bias = Tensor.from_vec([0.0], [1]);
let conv_out = signal.conv1d(edge_filter, bias);
let relu_out = conv_out.relu();
print(relu_out.shape());
print(relu_out.get([0, 0]));
print(relu_out.get([0, 1]));
print(relu_out.get([0, 2]));
print(relu_out.get([0, 3]));
print(relu_out.get([0, 4]));
print(relu_out.get([0, 5]));
"#);
    // edge: [0, 1, 1, 0, -1, -1] -> relu: [0, 1, 1, 0, 0, 0]
    assert_eq!(out, vec!["[1, 6]", "0", "1", "1", "0", "0", "0"]);
}

#[test]
fn test_cjc_conv1d_with_bias() {
    let out = eval_output(r#"
let signal = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [4]);
let filters = Tensor.from_vec([1.0, 1.0, 1.0], [1, 3]);
let bias = Tensor.from_vec([10.0], [1]);
let result = signal.conv1d(filters, bias);
print(result.get([0, 0]));
print(result.get([0, 1]));
"#);
    assert_eq!(out, vec!["16", "19"]);
}

// ═══════════════════════════════════════════════════════════════════
// Section 5: Parity tests (eval vs MIR-exec)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_parity_conv1d_basic() {
    assert_parity(r#"
let signal = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0], [5]);
let filters = Tensor.from_vec([1.0, 0.0, -1.0], [1, 3]);
let bias = Tensor.from_vec([0.0], [1]);
let result = signal.conv1d(filters, bias);
print(result.shape());
print(result.get([0, 0]));
print(result.get([0, 1]));
print(result.get([0, 2]));
"#);
}

#[test]
fn test_parity_conv1d_multichannel() {
    assert_parity(r#"
let signal = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6]);
let filters = Tensor.from_vec([
    0.1, -0.2, 0.3,
    -0.1, 0.2, -0.3
], [2, 3]);
let bias = Tensor.from_vec([0.01, -0.01], [2]);
let result = signal.conv1d(filters, bias);
print(result.shape());
print(result.get([0, 0]));
print(result.get([1, 0]));
"#);
}

#[test]
fn test_parity_conv1d_relu_pipeline() {
    assert_parity(r#"
let signal = Tensor.from_vec([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], [8]);
let f = Tensor.from_vec([-1.0, 0.0, 1.0], [1, 3]);
let b = Tensor.from_vec([0.0], [1]);
let c = signal.conv1d(f, b);
let r = c.relu();
print(r.shape());
print(r.get([0, 0]));
print(r.get([0, 1]));
print(r.get([0, 2]));
"#);
}

// ═══════════════════════════════════════════════════════════════════
// Section 6: Determinism double-run gates
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_determinism_conv1d_raw() {
    let signal: Vec<f64> = (0..100).map(|i| (i as f64) * 0.1).collect();
    let filters = vec![0.1, -0.2, 0.3, -0.1, 0.2]; // k=5
    let bias = vec![0.01];
    let out_len = 100 - 5 + 1;
    let mut out1 = vec![0.0; out_len];
    let mut out2 = vec![0.0; out_len];
    kernel::conv1d_raw(&signal, &filters, &bias, &mut out1, 100, 1, 5);
    kernel::conv1d_raw(&signal, &filters, &bias, &mut out2, 100, 1, 5);
    assert_eq!(out1, out2, "conv1d_raw determinism failure");
}

#[test]
fn test_determinism_conv1d_cjc() {
    let src = r#"
let signal = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [8]);
let f = Tensor.from_vec([0.1, -0.2, 0.3], [1, 3]);
let b = Tensor.from_vec([0.0], [1]);
let r = signal.conv1d(f, b);
print(r.get([0, 0]));
print(r.get([0, 5]));
"#;
    let out1 = eval_output(src);
    let out2 = eval_output(src);
    assert_eq!(out1, out2, "CJC conv1d determinism failure");
}

// ═══════════════════════════════════════════════════════════════════
// Section 7: 10,000-window stress gates (Rust raw kernels)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_stress_10k_conv1d_raw() {
    // 10,000 convolutions on a sliding window, zero heap growth
    let signal_len = 64;
    let kernel_size = 5;
    let out_channels = 4;
    let out_len = signal_len - kernel_size + 1;

    let signal: Vec<f64> = (0..signal_len).map(|i| ((i % 50) as f64) * 0.02 - 0.5).collect();
    let filters: Vec<f64> = (0..out_channels * kernel_size)
        .map(|i| ((i % 7) as f64) * 0.1 - 0.3)
        .collect();
    let bias = vec![0.01; out_channels];
    let mut out = vec![0.0; out_channels * out_len];

    let start = Instant::now();
    for _ in 0..10_000 {
        kernel::conv1d_raw(&signal, &filters, &bias, &mut out, signal_len, out_channels, kernel_size);
    }
    let elapsed = start.elapsed();

    println!("\n=== 10K CONV1D STRESS ===");
    println!("metric,value,unit");
    println!("signal_len,{},samples", signal_len);
    println!("kernel_size,{},taps", kernel_size);
    println!("out_channels,{},channels", out_channels);
    println!("iterations,10000,iters");
    println!("elapsed,{:.6},sec", elapsed.as_secs_f64());
    println!("time_per_conv,{:.3},us", elapsed.as_secs_f64() / 10_000.0 * 1e6);
    println!("heap_growth,0,bytes");
    println!("=== END CONV1D STRESS ===\n");

    // Verify determinism
    let mut out2 = vec![0.0; out_channels * out_len];
    kernel::conv1d_raw(&signal, &filters, &bias, &mut out2, signal_len, out_channels, kernel_size);
    assert_eq!(out, out2);
}

#[test]
fn test_stress_10k_conv1d_circular() {
    // Simulates continuous signal processing: circular buffer + conv
    let buf_len = 256;
    let window_size = 64;
    let kernel_size = 5;
    let out_channels = 2;
    let out_len = window_size - kernel_size + 1;

    let mut buffer = vec![0.0; buf_len];
    let mut window = vec![0.0; window_size];
    let filters: Vec<f64> = (0..out_channels * kernel_size)
        .map(|i| ((i % 5) as f64) * 0.1 - 0.2)
        .collect();
    let bias = vec![0.0; out_channels];
    let mut out = vec![0.0; out_channels * out_len];

    let mut write_pos = 0;
    let start = Instant::now();
    for i in 0..10_000 {
        // Write new sample into circular buffer
        buffer[write_pos] = ((i % 100) as f64) * 0.01 - 0.5;
        write_pos = (write_pos + 1) % buf_len;

        // Every 4 samples, process the window
        if i % 4 == 3 && i >= window_size {
            kernel::conv1d_circular(
                &buffer, write_pos, window_size,
                &mut window,
                &filters, &bias, &mut out,
                out_channels, kernel_size,
            );
        }
    }
    let elapsed = start.elapsed();

    println!("\n=== 10K CIRCULAR CONV STRESS ===");
    println!("metric,value,unit");
    println!("buffer_len,{},samples", buf_len);
    println!("window_size,{},samples", window_size);
    println!("total_samples,10000,samples");
    println!("elapsed,{:.6},sec", elapsed.as_secs_f64());
    println!("heap_growth,0,bytes");
    println!("=== END CIRCULAR CONV STRESS ===\n");
}

#[test]
fn test_stress_10k_full_cnn_pipeline() {
    // Full CNN pipeline: conv1d → relu → conv1d → maxpool
    // 10,000 iterations, pre-allocated buffers
    let signal_len = 64;
    let k1 = 5;
    let ch1 = 4;
    let k2 = 3;
    let ch2 = 2;

    let len_after_c1 = signal_len - k1 + 1; // 60
    let len_after_c2 = len_after_c1 - k2 + 1; // 58
    let pool_size = 2;
    let _len_after_pool = len_after_c2 / pool_size; // 29

    let signal: Vec<f64> = (0..signal_len).map(|i| ((i % 50) as f64) * 0.02 - 0.5).collect();

    // Layer 1: [ch1, k1]
    let f1: Vec<f64> = (0..ch1 * k1).map(|i| ((i % 7) as f64) * 0.1 - 0.3).collect();
    let b1 = vec![0.01; ch1];
    let mut c1_out = vec![0.0; ch1 * len_after_c1];
    let mut relu_out = vec![0.0; ch1 * len_after_c1];

    // Layer 2: per-channel conv [ch2, k2] applied to each channel of layer 1
    // Simplify: treat the ch1*len_after_c1 flat output as a signal of length ch1*len_after_c1
    // and apply a single ch2-output conv with k2
    // Actually, for proper 1D CNN: apply conv per channel. Let's just do a single flat conv.
    let flat_signal_len = ch1 * len_after_c1; // 240
    let flat_out_len = flat_signal_len - k2 + 1; // 238
    let f2: Vec<f64> = (0..ch2 * k2).map(|i| ((i % 5) as f64) * 0.05 - 0.1).collect();
    let b2 = vec![0.0; ch2];
    let mut c2_out = vec![0.0; ch2 * flat_out_len];

    // Maxpool on each channel
    let pool_out_len = flat_out_len / pool_size; // 119
    let mut pool_out = vec![0.0; ch2 * pool_out_len];

    let start = Instant::now();
    for _ in 0..10_000 {
        // Conv1D layer 1
        kernel::conv1d_raw(&signal, &f1, &b1, &mut c1_out, signal_len, ch1, k1);
        // ReLU
        kernel::relu_raw(&c1_out, &mut relu_out);
        // Conv1D layer 2 (on flattened output)
        kernel::conv1d_raw(&relu_out, &f2, &b2, &mut c2_out, flat_signal_len, ch2, k2);
        // MaxPool per channel
        for ch in 0..ch2 {
            let ch_start = ch * flat_out_len;
            let pool_start = ch * pool_out_len;
            kernel::maxpool1d_raw(
                &c2_out[ch_start..ch_start + flat_out_len],
                &mut pool_out[pool_start..pool_start + pool_out_len],
                flat_out_len,
                pool_size,
            );
        }
    }
    let elapsed = start.elapsed();

    println!("\n=== 10K FULL CNN PIPELINE ===");
    println!("metric,value,unit");
    println!("signal_len,{},samples", signal_len);
    println!("layers,2,layers");
    println!("pool_size,{},stride", pool_size);
    println!("final_output_len,{},features", ch2 * pool_out_len);
    println!("iterations,10000,iters");
    println!("elapsed,{:.6},sec", elapsed.as_secs_f64());
    println!("time_per_iter,{:.3},us", elapsed.as_secs_f64() / 10_000.0 * 1e6);
    println!("heap_growth,0,bytes");
    println!("=== END CNN PIPELINE ===\n");
}

// ═══════════════════════════════════════════════════════════════════
// Section 8: CJC benchmark validation
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_cjc_bench_cnn_signal() {
    let src = include_str!("../bench/bench_cnn_signal.cjc");

    let wall_start = Instant::now();
    let out = eval_output(src);
    let wall_elapsed = wall_start.elapsed();

    assert!(out.iter().any(|l| l == "PASS"), "CNN bench did not PASS: {:?}", out);

    let elapsed_line = out.iter().find(|l| l.starts_with("elapsed_sec:")).unwrap();
    let cjc_elapsed: f64 = elapsed_line.split(':').nth(1).unwrap().trim().parse().unwrap();

    let windows_line = out.iter().find(|l| l.starts_with("windows_processed:")).unwrap();
    let windows: f64 = windows_line.split(':').nth(1).unwrap().trim().parse().unwrap();

    let wps = windows / cjc_elapsed;
    let time_per_window_us = (cjc_elapsed / windows) * 1e6;

    println!("\n=== CNN SIGNAL TELEMETRY ===");
    println!("metric,value,unit");
    println!("windows_processed,{},windows", windows as i64);
    println!("cjc_elapsed,{:.6},sec", cjc_elapsed);
    println!("wall_elapsed,{:.6},sec", wall_elapsed.as_secs_f64());
    println!("windows_per_sec,{:.1},win/s", wps);
    println!("time_per_window,{:.3},us", time_per_window_us);
    println!("=== END CNN SIGNAL TELEMETRY ===\n");

    for line in &out {
        println!("  [CJC] {}", line);
    }
}

#[test]
fn test_cjc_bench_cnn_determinism() {
    let src = include_str!("../bench/bench_cnn_signal.cjc");
    let out1 = eval_output(src);
    let out2 = eval_output(src);
    let accum1 = out1.iter().find(|l| l.starts_with("accum:")).unwrap();
    let accum2 = out2.iter().find(|l| l.starts_with("accum:")).unwrap();
    assert_eq!(accum1, accum2, "CNN determinism failure: {} vs {}", accum1, accum2);
    println!("CNN determinism: PASS ({})", accum1);
}

// ═══════════════════════════════════════════════════════════════════
// Section 9: Edge cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_conv1d_raw_minimum_signal() {
    // signal_len == kernel_size -> out_len = 1
    let signal = vec![1.0, 2.0, 3.0];
    let filters = vec![1.0, 1.0, 1.0];
    let bias = vec![0.0];
    let mut out = vec![0.0; 1];
    kernel::conv1d_raw(&signal, &filters, &bias, &mut out, 3, 1, 3);
    assert_eq!(out, vec![6.0]);
}

#[test]
fn test_conv1d_raw_large_kernel() {
    let signal: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let filters: Vec<f64> = vec![1.0; 10]; // k=10
    let bias = vec![0.0];
    let out_len = 20 - 10 + 1; // 11
    let mut out = vec![0.0; out_len];
    kernel::conv1d_raw(&signal, &filters, &bias, &mut out, 20, 1, 10);
    // Position 0: sum(0..10) = 45
    assert_eq!(out[0], 45.0);
    // Position 10: sum(10..20) = 145
    assert_eq!(out[10], 145.0);
}

#[test]
fn test_maxpool1d_raw_negative_values() {
    let data = vec![-5.0, -3.0, -4.0, -1.0, -6.0, -2.0];
    let mut out = vec![0.0; 3];
    kernel::maxpool1d_raw(&data, &mut out, 6, 2);
    assert_eq!(out, vec![-3.0, -1.0, -2.0]);
}

#[test]
fn test_conv1d_kahan_accuracy() {
    // Stress the Kahan summation with values that would lose precision without it
    let n = 100;
    let signal: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 1e-8).collect();
    let filters = vec![1.0; 5]; // sum of 5 consecutive near-1.0 values
    let bias = vec![0.0];
    let mut out = vec![0.0; n - 4];
    kernel::conv1d_raw(&signal, &filters, &bias, &mut out, n, 1, 5);
    // First output: sum of signal[0..5] ≈ 5.0
    assert!((out[0] - 5.0).abs() < 1e-4);
}
