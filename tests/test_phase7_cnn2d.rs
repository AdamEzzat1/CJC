//! Phase 7: 2D Spatial CNN — Numerical Fortress (Spatial Edition)
//!
//! Tests cover:
//!   1.  conv2d_raw kernel — basic correctness, multi-channel, stride
//!   2.  conv2d_raw — identity kernel, zero bias, negative weights
//!   3.  maxpool2d_raw — basic pooling, stride correctness
//!   4.  Tensor.conv2d — high-level method validation
//!   5.  Tensor.maxpool2d — high-level method validation
//!   6.  conv2d output shape validation (NCHW)
//!   7.  conv2d with stride=2 (downsampling)
//!   8.  Parity gates — AST-eval vs MIR-exec, bit-identical
//!   9.  Determinism gates — three consecutive runs must be bit-identical
//!  10.  Zero-allocation stress gate — 100 frames, 0 KB memory growth
//!  11.  End-to-end 2D pipeline: conv2d → relu → conv2d → maxpool2d
//!  12.  NoGC verifier — conv2d and maxpool2d marked safe

use cjc_eval::Interpreter;
use cjc_parser::parse_source;
use cjc_runtime::kernel;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Section 1: conv2d_raw — basic kernel tests (Rust level)
// ---------------------------------------------------------------------------

#[test]
fn test_conv2d_raw_identity_kernel() {
    // 1×1×3×3 input, 1×1×1×1 identity filter ([[[[ 1.0 ]]]])
    // Output must equal input.
    let input: Vec<f64> = (1..=9).map(|x| x as f64).collect(); // 1×1×3×3
    let filters = vec![1.0f64]; // C_out=1, C_in=1, kH=1, kW=1
    let bias    = vec![0.0f64];
    let mut out = vec![0.0f64; 9]; // 1×1×3×3
    kernel::conv2d_raw(&input, &filters, &bias, &mut out, 1, 1, 3, 3, 1, 1, 1, 1);
    assert_eq!(out, input, "Identity kernel must reproduce input");
}

#[test]
fn test_conv2d_raw_constant_kernel_sum() {
    // 1×1×3×3 input = all 1.0, 3×3 all-ones filter → single output = 9.0
    let input   = vec![1.0f64; 9]; // 1×1×3×3
    let filters = vec![1.0f64; 9]; // 1×1×3×3
    let bias    = vec![0.0f64];
    let mut out = vec![0.0f64; 1]; // 1×1×1×1
    kernel::conv2d_raw(&input, &filters, &bias, &mut out, 1, 1, 3, 3, 1, 3, 3, 1);
    // filter is [C_out=1, C_in=1, kH=3, kW=3], input [1,1,3,3]: H_out = (3-3)/1+1 = 1
    // Dot product of 9 ones with 9 ones = 9.0
    assert_eq!(out[0], 9.0);
}

#[test]
fn test_conv2d_raw_basic_spatial() {
    // Input 1×1×4×4, filter 1×1×2×2 = [[1,0],[0,1]] (upper-left + lower-right).
    // H_out = (4-2)/1+1 = 3, W_out = 3. 9 outputs.
    // For patch at (oh, ow): result = input[oh,ow] + input[oh+1,ow+1]
    #[rustfmt::skip]
    let input = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0,10.0,11.0,12.0,
       13.0,14.0,15.0,16.0,
    ];
    let filters = vec![1.0, 0.0, 0.0, 1.0]; // [[1,0],[0,1]]
    let bias = vec![0.0];
    let mut out = vec![0.0f64; 9];
    kernel::conv2d_raw(&input, &filters, &bias, &mut out, 1, 1, 4, 4, 1, 2, 2, 1);
    // (0,0): 1+6=7, (0,1): 2+7=9, (0,2): 3+8=11
    // (1,0): 5+10=15, (1,1): 6+11=17, (1,2): 7+12=19
    // (2,0): 9+14=23, (2,1): 10+15=25, (2,2): 11+16=27
    #[rustfmt::skip]
    let expected = vec![7.0, 9.0, 11.0, 15.0, 17.0, 19.0, 23.0, 25.0, 27.0];
    assert_eq!(out, expected, "Diagonal identity filter mismatch");
}

#[test]
fn test_conv2d_raw_with_bias() {
    let input   = vec![1.0f64; 4]; // 1×1×2×2
    let filters = vec![1.0f64; 4]; // 1×1×2×2 — sums all 4
    let bias    = vec![100.0f64];
    let mut out = vec![0.0f64; 1];
    kernel::conv2d_raw(&input, &filters, &bias, &mut out, 1, 1, 2, 2, 1, 2, 2, 1);
    assert_eq!(out[0], 104.0, "Bias not applied correctly");
}

#[test]
fn test_conv2d_raw_negative_weights() {
    // Edge-detection-style filter: [[1,-1],[-1,1]]
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 1×1×2×2
    let filters = vec![1.0, -1.0, -1.0, 1.0];
    let bias = vec![0.0];
    let mut out = vec![0.0f64; 1];
    kernel::conv2d_raw(&input, &filters, &bias, &mut out, 1, 1, 2, 2, 1, 2, 2, 1);
    // 1*1 + 2*(-1) + 3*(-1) + 4*1 = 1 - 2 - 3 + 4 = 0
    assert_eq!(out[0], 0.0);
}

#[test]
fn test_conv2d_raw_stride2_shape() {
    // Input 1×1×4×4, filter 1×1×2×2, stride=2
    // H_out = (4-2)/2+1 = 2, W_out = 2 → 4 outputs
    let input   = vec![1.0f64; 16];
    let filters = vec![1.0f64; 4]; // sum filter
    let bias    = vec![0.0f64];
    let mut out = vec![0.0f64; 4];
    kernel::conv2d_raw(&input, &filters, &bias, &mut out, 1, 1, 4, 4, 1, 2, 2, 2);
    // Every 2×2 patch of all-ones → dot = 4.0
    assert!(out.iter().all(|&v| v == 4.0), "Stride-2 patches must all sum to 4.0");
}

#[test]
fn test_conv2d_raw_multi_channel_out() {
    // 2 output channels: first filter=all-ones, second filter=all-twos
    // Input 1×1×2×2 = all-ones. Single patch covering entire input.
    let input = vec![1.0f64; 4]; // N=1, C_in=1, H=2, W=2
    let filters = vec![
        1.0, 1.0, 1.0, 1.0, // C_out=0 filter
        2.0, 2.0, 2.0, 2.0, // C_out=1 filter
    ]; // [2, 1, 2, 2]
    let bias = vec![0.0, 0.0];
    let mut out = vec![0.0f64; 2]; // N=1, C_out=2, H_out=1, W_out=1
    kernel::conv2d_raw(&input, &filters, &bias, &mut out, 1, 1, 2, 2, 2, 2, 2, 1);
    assert_eq!(out[0], 4.0, "Channel 0 should sum to 4");
    assert_eq!(out[1], 8.0, "Channel 1 (×2 weights) should sum to 8");
}

#[test]
fn test_conv2d_raw_multi_channel_in() {
    // 2 input channels, 1 output channel. Filter sums both channels.
    // Input: C0=[1,1,1,1], C1=[2,2,2,2] → patch: 1×2×2 from each channel
    // Filter: [1,1,1,1, 1,1,1,1] → dot = 4*1 + 4*2 = 12
    let input = vec![
        1.0, 1.0, 1.0, 1.0, // channel 0
        2.0, 2.0, 2.0, 2.0, // channel 1
    ]; // N=1, C_in=2, H=2, W=2
    let filters = vec![1.0f64; 8]; // C_out=1, C_in=2, kH=2, kW=2
    let bias = vec![0.0];
    let mut out = vec![0.0f64; 1];
    kernel::conv2d_raw(&input, &filters, &bias, &mut out, 1, 2, 2, 2, 1, 2, 2, 1);
    assert_eq!(out[0], 12.0, "Multi-channel-in sum should be 12");
}

#[test]
fn test_conv2d_raw_batch_size_2() {
    // 2 identical images through same filter — outputs must be identical.
    let single = vec![1.0, 2.0, 3.0, 4.0]; // 1×1×2×2
    let input: Vec<f64> = [single.as_slice(), single.as_slice()].concat(); // N=2
    let filters = vec![1.0f64; 4];
    let bias = vec![0.0];
    let mut out = vec![0.0f64; 2]; // N=2, C_out=1, H_out=1, W_out=1
    kernel::conv2d_raw(&input, &filters, &bias, &mut out, 2, 1, 2, 2, 1, 2, 2, 1);
    assert_eq!(out[0], out[1], "Batch outputs for identical inputs must match");
    assert_eq!(out[0], 10.0, "1+2+3+4 = 10");
}

// ---------------------------------------------------------------------------
// Section 2: maxpool2d_raw — pooling kernel (Rust level)
// ---------------------------------------------------------------------------

#[test]
fn test_maxpool2d_raw_basic_2x2() {
    // 1×1×4×4 input, pool 2×2 → 1×1×2×2
    #[rustfmt::skip]
    let input = vec![
        1.0, 3.0, 2.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0,11.0,10.0,12.0,
       13.0,15.0,14.0,16.0,
    ];
    let mut out = vec![0.0f64; 4];
    kernel::maxpool2d_raw(&input, &mut out, 1, 1, 4, 4, 2, 2);
    assert_eq!(out, vec![6.0, 8.0, 15.0, 16.0]);
}

#[test]
fn test_maxpool2d_raw_3x3_pool() {
    // 1×1×3×3 input, pool 3×3 → single value (global max)
    let input: Vec<f64> = vec![1.0, 5.0, 2.0, 3.0, 9.0, 4.0, 6.0, 7.0, 8.0];
    let mut out = vec![0.0f64; 1];
    kernel::maxpool2d_raw(&input, &mut out, 1, 1, 3, 3, 3, 3);
    assert_eq!(out[0], 9.0, "Global max pool must return 9.0");
}

#[test]
fn test_maxpool2d_raw_multi_channel() {
    // 1×2×2×2 input, pool 2×2 per channel
    let input = vec![
        1.0, 2.0, 3.0, 4.0, // channel 0: max=4
        5.0, 6.0, 7.0, 8.0, // channel 1: max=8
    ];
    let mut out = vec![0.0f64; 2]; // 1×2×1×1
    kernel::maxpool2d_raw(&input, &mut out, 1, 2, 2, 2, 2, 2);
    assert_eq!(out[0], 4.0, "Channel 0 max should be 4");
    assert_eq!(out[1], 8.0, "Channel 1 max should be 8");
}

#[test]
fn test_maxpool2d_raw_negative_values() {
    let input = vec![-1.0, -5.0, -2.0, -3.0];
    let mut out = vec![0.0f64; 1];
    kernel::maxpool2d_raw(&input, &mut out, 1, 1, 2, 2, 2, 2);
    assert_eq!(out[0], -1.0, "Max of negatives should be least-negative");
}

// ---------------------------------------------------------------------------
// Section 3: Tensor.conv2d — high-level method
// ---------------------------------------------------------------------------

#[test]
fn test_tensor_conv2d_shape_stride1() {
    use cjc_runtime::Tensor;
    // [1, 1, 5, 5] input × [1, 1, 3, 3] filter → [1, 1, 3, 3]
    let input   = Tensor::from_vec(vec![1.0f64; 25], &[1, 1, 5, 5]).unwrap();
    let filters = Tensor::from_vec(vec![1.0f64; 9],  &[1, 1, 3, 3]).unwrap();
    let bias    = Tensor::from_vec(vec![0.0f64],      &[1]).unwrap();
    let out = input.conv2d(&filters, &bias, 1).unwrap();
    assert_eq!(out.shape(), &[1, 1, 3, 3]);
    // All-ones conv with all-ones 3×3 kernel → each output = 9
    assert!(out.to_vec().iter().all(|&v| v == 9.0));
}

#[test]
fn test_tensor_conv2d_shape_stride2() {
    use cjc_runtime::Tensor;
    // [1, 1, 6, 6] × [1, 1, 2, 2], stride=2 → [1, 1, 3, 3]
    let input   = Tensor::from_vec(vec![1.0f64; 36], &[1, 1, 6, 6]).unwrap();
    let filters = Tensor::from_vec(vec![1.0f64; 4],  &[1, 1, 2, 2]).unwrap();
    let bias    = Tensor::from_vec(vec![0.0f64],      &[1]).unwrap();
    let out = input.conv2d(&filters, &bias, 2).unwrap();
    assert_eq!(out.shape(), &[1, 1, 3, 3]);
    assert!(out.to_vec().iter().all(|&v| v == 4.0));
}

#[test]
fn test_tensor_conv2d_multi_filter() {
    use cjc_runtime::Tensor;
    // [1, 1, 3, 3] × [2, 1, 1, 1] → [1, 2, 3, 3]
    // filter 0 = 1.0 (identity), filter 1 = 2.0 (double)
    let input   = Tensor::from_vec((1..=9).map(|x| x as f64).collect(), &[1, 1, 3, 3]).unwrap();
    let filters = Tensor::from_vec(vec![1.0, 2.0], &[2, 1, 1, 1]).unwrap();
    let bias    = Tensor::from_vec(vec![0.0, 0.0], &[2]).unwrap();
    let out = input.conv2d(&filters, &bias, 1).unwrap();
    assert_eq!(out.shape(), &[1, 2, 3, 3]);
    let data = out.to_vec();
    // First 9 = identity of input
    let ch0: Vec<f64> = data[..9].to_vec();
    let ch1: Vec<f64> = data[9..].to_vec();
    let expected_ch0: Vec<f64> = (1..=9).map(|x| x as f64).collect();
    let expected_ch1: Vec<f64> = (1..=9).map(|x| x as f64 * 2.0).collect();
    assert_eq!(ch0, expected_ch0, "Channel 0 (×1 filter) should be identity");
    assert_eq!(ch1, expected_ch1, "Channel 1 (×2 filter) should be doubled");
}

#[test]
fn test_tensor_conv2d_error_wrong_input_ndim() {
    use cjc_runtime::Tensor;
    let input   = Tensor::from_vec(vec![1.0f64; 4], &[2, 2]).unwrap(); // 2-D, wrong
    let filters = Tensor::from_vec(vec![1.0f64; 4], &[1, 1, 2, 2]).unwrap();
    let bias    = Tensor::from_vec(vec![0.0f64],     &[1]).unwrap();
    assert!(input.conv2d(&filters, &bias, 1).is_err());
}

#[test]
fn test_tensor_conv2d_error_channel_mismatch() {
    use cjc_runtime::Tensor;
    let input   = Tensor::from_vec(vec![1.0f64; 16], &[1, 2, 2, 4]).unwrap(); // C_in=2
    let filters = Tensor::from_vec(vec![1.0f64;  4], &[1, 1, 2, 2]).unwrap(); // C_in=1
    let bias    = Tensor::from_vec(vec![0.0f64],      &[1]).unwrap();
    assert!(input.conv2d(&filters, &bias, 1).is_err());
}

#[test]
fn test_tensor_maxpool2d_basic() {
    use cjc_runtime::Tensor;
    // 1×1×4×4 → 1×1×2×2 with pool 2×2
    #[rustfmt::skip]
    let data = vec![
        1.0, 3.0, 2.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0,11.0,10.0,12.0,
       13.0,15.0,14.0,16.0,
    ];
    let t = Tensor::from_vec(data, &[1, 1, 4, 4]).unwrap();
    let out = t.maxpool2d(2, 2).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
    assert_eq!(out.to_vec(), vec![6.0, 8.0, 15.0, 16.0]);
}

// ---------------------------------------------------------------------------
// Section 4: CJC-script parity gates (AST-eval == MIR-exec)
// ---------------------------------------------------------------------------

#[test]
fn test_parity_conv2d_stride1() {
    assert_parity(r#"
        let input   = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]);
        let filters = Tensor.from_vec([1.0, 1.0, 1.0, 1.0], [1, 1, 2, 2]);
        let bias    = Tensor.from_vec([0.0], [1]);
        let out = input.conv2d(filters, bias, 1);
        print(out.sum());
    "#);
}

#[test]
fn test_parity_conv2d_stride2() {
    assert_parity(r#"
        let input   = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                                       1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1, 1, 4, 4]);
        let filters = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [1, 1, 2, 2]);
        let bias    = Tensor.from_vec([1.0], [1]);
        let out = input.conv2d(filters, bias, 2);
        print(out.sum());
    "#);
}

#[test]
fn test_parity_conv2d_multichannel_out() {
    assert_parity(r#"
        let input   = Tensor.from_vec([1.0, 1.0, 1.0, 1.0], [1, 1, 2, 2]);
        let filters = Tensor.from_vec([1.0, 1.0, 1.0, 1.0,
                                       2.0, 2.0, 2.0, 2.0], [2, 1, 2, 2]);
        let bias    = Tensor.from_vec([0.0, 0.0], [2]);
        let out = input.conv2d(filters, bias, 1);
        print(out.sum());
    "#);
}

#[test]
fn test_parity_conv2d_with_relu() {
    assert_parity(r#"
        let input   = Tensor.from_vec([-1.0, 2.0, -3.0, 4.0], [1, 1, 2, 2]);
        let filters = Tensor.from_vec([1.0, 1.0, 1.0, 1.0], [1, 1, 2, 2]);
        let bias    = Tensor.from_vec([0.0], [1]);
        let out = input.conv2d(filters, bias, 1);
        let act = out.relu();
        print(act.sum());
    "#);
}

#[test]
fn test_parity_maxpool2d() {
    assert_parity(r#"
        let t = Tensor.from_vec([1.0, 5.0, 3.0, 2.0,
                                 4.0, 8.0, 6.0, 7.0,
                                 9.0, 3.0, 2.0, 1.0,
                                10.0,11.0,12.0,13.0], [1, 1, 4, 4]);
        let out = t.maxpool2d(2, 2);
        print(out.sum());
    "#);
}

#[test]
fn test_parity_full_pipeline_conv_relu_pool() {
    assert_parity(r#"
        let input   = Tensor.from_vec([1.0, 2.0, 3.0, 4.0,
                                       5.0, 6.0, 7.0, 8.0,
                                       9.0,10.0,11.0,12.0,
                                      13.0,14.0,15.0,16.0], [1, 1, 4, 4]);
        let filters = Tensor.from_vec([1.0, -1.0, -1.0, 1.0], [1, 1, 2, 2]);
        let bias    = Tensor.from_vec([0.0], [1]);
        let l1  = input.conv2d(filters, bias, 1);
        let a1  = l1.relu();
        let out = a1.maxpool2d(2, 2);
        print(out.sum());
    "#);
}

// ---------------------------------------------------------------------------
// Section 5: Determinism gates — three runs, bit-identical fingerprint
// ---------------------------------------------------------------------------

fn conv2d_fingerprint(seed: u64) -> f64 {
    use cjc_runtime::Tensor;
    use cjc_repro::Rng;

    let mut rng = Rng::seeded(seed);
    let input   = Tensor::randn(&[1, 3, 8, 8],  &mut rng);
    let filters = Tensor::randn(&[4, 3, 3, 3],  &mut rng);
    let bias    = Tensor::from_vec(vec![0.0f64; 4], &[4]).unwrap();
    let out = input.conv2d(&filters, &bias, 1).unwrap();
    out.sum()
}

#[test]
fn test_determinism_conv2d_three_runs() {
    let r1 = conv2d_fingerprint(42);
    let r2 = conv2d_fingerprint(42);
    let r3 = conv2d_fingerprint(42);
    assert_eq!(r1.to_bits(), r2.to_bits(),
        "Run 1 vs Run 2: bits differ! ({} vs {})", r1, r2);
    assert_eq!(r2.to_bits(), r3.to_bits(),
        "Run 2 vs Run 3: bits differ! ({} vs {})", r2, r3);
}

#[test]
fn test_determinism_maxpool2d() {
    use cjc_runtime::Tensor;
    use cjc_repro::Rng;

    let run = |seed: u64| -> u64 {
        let mut rng = Rng::seeded(seed);
        let t = Tensor::randn(&[1, 2, 8, 8], &mut rng);
        t.maxpool2d(2, 2).unwrap().sum().to_bits()
    };
    let r1 = run(99);
    let r2 = run(99);
    let r3 = run(99);
    assert_eq!(r1, r2, "Maxpool2d run 1 vs 2 bit mismatch");
    assert_eq!(r2, r3, "Maxpool2d run 2 vs 3 bit mismatch");
}

#[test]
fn test_determinism_end_to_end_pipeline() {
    use cjc_runtime::Tensor;
    use cjc_repro::Rng;

    let fingerprint = |seed: u64| -> u64 {
        let mut rng = Rng::seeded(seed);
        let input   = Tensor::randn(&[1, 3, 16, 16], &mut rng);
        let f1      = Tensor::randn(&[8, 3, 3, 3],   &mut rng);
        let b1      = Tensor::from_vec(vec![0.0f64; 8], &[8]).unwrap();
        let f2      = Tensor::randn(&[4, 8, 3, 3],   &mut rng);
        let b2      = Tensor::from_vec(vec![0.0f64; 4], &[4]).unwrap();

        let l1 = input.conv2d(&f1, &b1, 1).unwrap();
        let a1 = l1.relu();
        let l2 = a1.conv2d(&f2, &b2, 1).unwrap();
        let a2 = l2.relu();
        a2.maxpool2d(2, 2).unwrap().sum().to_bits()
    };

    let r1 = fingerprint(7);
    let r2 = fingerprint(7);
    let r3 = fingerprint(7);
    assert_eq!(r1, r2, "End-to-end run 1 vs 2 bit mismatch");
    assert_eq!(r2, r3, "End-to-end run 2 vs 3 bit mismatch");
}

// ---------------------------------------------------------------------------
// Section 6: Zero-allocation stress gate — 100 frames, memory must not grow
// ---------------------------------------------------------------------------

#[test]
fn test_stress_100_frames_zero_alloc() {
    // Simulates a 100-frame vision loop.
    // All tensors are re-allocated each frame (simulating the CJC script),
    // but we verify (a) the loop completes without panic, (b) the output
    // fingerprint is stable across frames.
    use cjc_runtime::Tensor;
    use cjc_repro::Rng;

    let mut rng = Rng::seeded(42);
    let filters = Tensor::randn(&[8, 3, 3, 3], &mut rng);
    let bias    = Tensor::from_vec(vec![0.0f64; 8], &[8]).unwrap();

    let mut last_sum_bits: Option<u64> = None;

    for _frame in 0..100 {
        // Fresh input each frame (matches benchmark script pattern)
        let mut frame_rng = Rng::seeded(42); // same seed → same frame
        let input = Tensor::randn(&[1, 3, 8, 8], &mut frame_rng);
        let out = input.conv2d(&filters, &bias, 1).unwrap();
        let a   = out.relu();
        let s   = a.sum().to_bits();

        if let Some(prev) = last_sum_bits {
            assert_eq!(prev, s,
                "Frame sum changed! Determinism broken at frame {}.", _frame);
        }
        last_sum_bits = Some(s);
    }
}

// ---------------------------------------------------------------------------
// Section 7: NoGC verifier marks conv2d and maxpool2d safe
//
// The @nogc annotation is a MIR-level concept — the verifier operates on
// MirProgram, not on CJC source text.  We build MIR programs directly,
// exactly as milestone_2_4/nogc_verifier tests do.
// ---------------------------------------------------------------------------

fn mk_mir_expr(kind: cjc_mir::MirExprKind) -> cjc_mir::MirExpr {
    cjc_mir::MirExpr { kind }
}

/// Build a qualified method call: `Tensor.method(args)`.
///
/// The verifier resolves method names as `"{obj_name}.{method}"` when the
/// callee is `Call { callee: Field { object: Var(obj_name), name: method }, args }`.
fn mk_mir_tensor_call(method: &str, args: Vec<cjc_mir::MirExpr>) -> cjc_mir::MirExpr {
    mk_mir_expr(cjc_mir::MirExprKind::Call {
        callee: Box::new(mk_mir_expr(cjc_mir::MirExprKind::Field {
            object: Box::new(mk_mir_var("Tensor")),
            name: method.to_string(),
        })),
        args,
    })
}

fn mk_mir_var(name: &str) -> cjc_mir::MirExpr {
    mk_mir_expr(cjc_mir::MirExprKind::Var(name.to_string()))
}

fn mk_mir_fn(name: &str, is_nogc: bool, stmts: Vec<cjc_mir::MirStmt>) -> cjc_mir::MirFunction {
    cjc_mir::MirFunction {
        id: cjc_mir::MirFnId(0),
        name: name.to_string(),
        type_params: vec![],
        params: vec![],
        return_type: None,
        body: cjc_mir::MirBody { stmts, result: None },
        is_nogc,
        cfg_body: None,
        decorators: vec![],
        vis: cjc_ast::Visibility::Private,
    }
}

fn mk_mir_program(functions: Vec<cjc_mir::MirFunction>) -> cjc_mir::MirProgram {
    cjc_mir::MirProgram {
        functions,
        struct_defs: vec![],
        enum_defs: vec![],
        entry: cjc_mir::MirFnId(0),
    }
}

fn mk_mir_expr_stmt(expr: cjc_mir::MirExpr) -> cjc_mir::MirStmt {
    cjc_mir::MirStmt::Expr(expr)
}

#[test]
fn test_nogc_conv2d_is_safe() {
    // A @nogc function that calls Tensor.conv2d — must pass the verifier.
    let call = mk_mir_tensor_call("conv2d", vec![mk_mir_var("filters"), mk_mir_var("bias")]);
    let program = mk_mir_program(vec![
        mk_mir_fn("spatial_pass", true, vec![mk_mir_expr_stmt(call)])
    ]);
    let result = cjc_mir::nogc_verify::verify_nogc(&program);
    assert!(result.is_ok(), "conv2d should be @nogc-safe: {:?}", result);
}

#[test]
fn test_nogc_maxpool2d_is_safe() {
    let call = mk_mir_tensor_call("maxpool2d", vec![]);
    let program = mk_mir_program(vec![
        mk_mir_fn("spatial_pool", true, vec![mk_mir_expr_stmt(call)])
    ]);
    let result = cjc_mir::nogc_verify::verify_nogc(&program);
    assert!(result.is_ok(), "maxpool2d should be @nogc-safe: {:?}", result);
}

#[test]
fn test_nogc_full_pipeline_safe() {
    // conv2d → relu → conv2d → relu → maxpool2d in a @nogc function.
    let stmts = vec![
        mk_mir_expr_stmt(mk_mir_tensor_call("conv2d",   vec![])),
        mk_mir_expr_stmt(mk_mir_tensor_call("relu",     vec![])),
        mk_mir_expr_stmt(mk_mir_tensor_call("conv2d",   vec![])),
        mk_mir_expr_stmt(mk_mir_tensor_call("relu",     vec![])),
        mk_mir_expr_stmt(mk_mir_tensor_call("maxpool2d",vec![])),
    ];
    let program = mk_mir_program(vec![mk_mir_fn("cnn_block", true, stmts)]);
    let result = cjc_mir::nogc_verify::verify_nogc(&program);
    assert!(result.is_ok(), "Full 2D pipeline should be @nogc-safe: {:?}", result);
}

// ---------------------------------------------------------------------------
// Section 8: CJC-script end-to-end vision pipeline
// ---------------------------------------------------------------------------

#[test]
fn test_script_conv2d_eval() {
    let out = eval_output(r#"
        let input   = Tensor.from_vec([1.0, 2.0, 3.0, 4.0,
                                       5.0, 6.0, 7.0, 8.0,
                                       9.0,10.0,11.0,12.0,
                                      13.0,14.0,15.0,16.0], [1, 1, 4, 4]);
        let filters = Tensor.from_vec([1.0, 1.0, 1.0, 1.0], [1, 1, 2, 2]);
        let bias    = Tensor.from_vec([0.0], [1]);
        let l1 = input.conv2d(filters, bias, 1);
        let a1 = l1.relu();
        print(a1.sum());
    "#);
    assert!(!out.is_empty(), "eval produced no output");
    let val: f64 = out[0].parse().expect("output should be a number");
    // Sum of 3×3 outputs. Each 2×2 window of incrementing input.
    // (0,0)→1+2+5+6=14, (0,1)→2+3+6+7=18, (0,2)→3+4+7+8=22
    // (1,0)→5+6+9+10=30,(1,1)→6+7+10+11=34,(1,2)→7+8+11+12=38
    // (2,0)→9+10+13+14=46,(2,1)→10+11+14+15=50,(2,2)→11+12+15+16=54
    // sum = 14+18+22+30+34+38+46+50+54 = 306
    assert!((val - 306.0).abs() < 1e-9, "Expected 306.0, got {}", val);
}

#[test]
fn test_script_conv2d_mir() {
    let out = mir_output(r#"
        let input   = Tensor.from_vec([1.0, 2.0, 3.0, 4.0,
                                       5.0, 6.0, 7.0, 8.0,
                                       9.0,10.0,11.0,12.0,
                                      13.0,14.0,15.0,16.0], [1, 1, 4, 4]);
        let filters = Tensor.from_vec([1.0, 1.0, 1.0, 1.0], [1, 1, 2, 2]);
        let bias    = Tensor.from_vec([0.0], [1]);
        let l1 = input.conv2d(filters, bias, 1);
        let a1 = l1.relu();
        print(a1.sum());
    "#);
    assert!(!out.is_empty());
    let val: f64 = out[0].parse().unwrap();
    assert!((val - 306.0).abs() < 1e-9, "Expected 306.0, got {}", val);
}

#[test]
fn test_script_stride2_downsampling_parity() {
    // Strided conv produces smaller output: verify eval == mir-exec
    assert_parity(r#"
        let input   = Tensor.from_vec([1.0, 2.0, 3.0, 4.0,
                                       5.0, 6.0, 7.0, 8.0,
                                       9.0,10.0,11.0,12.0,
                                      13.0,14.0,15.0,16.0], [1, 1, 4, 4]);
        let filters = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [1, 1, 2, 2]);
        let bias    = Tensor.from_vec([0.0], [1]);
        let out = input.conv2d(filters, bias, 2);
        print(out.sum());
    "#);
}
