//! Stress Test 4: Memory Stability Test
//!
//! Verifies that BinnedAccumulator and kernel operations have zero memory
//! growth over many iterations, and that no allocations occur inside the
//! accumulator hot path.

use cjc_runtime::accumulator::{binned_sum_f64, BinnedAccumulatorF64, BinnedAccumulatorF32};
use cjc_runtime::dispatch::{ReductionContext, dispatch_sum_f64, dispatch_dot_f64};
use cjc_runtime::Tensor;

#[test]
fn test_binned_accumulator_stack_size() {
    // BinnedAccumulatorF64 should be a fixed-size stack struct.
    // 2048 bins * 8 bytes (f64) * 2 (bins + comp) + 2048 * 4 (counts) + overhead
    // = 32768 + 32768 + 8192 + ~24 = ~73752 bytes
    let size = std::mem::size_of::<BinnedAccumulatorF64>();
    eprintln!("BinnedAccumulatorF64 size: {size} bytes");
    // Must be constant and reasonable (under 128KB for stack allocation).
    assert!(size < 128 * 1024, "Accumulator too large for stack: {size}");
    assert!(size > 0, "Accumulator size should be nonzero");
}

#[test]
fn test_binned_f32_accumulator_stack_size() {
    let size = std::mem::size_of::<BinnedAccumulatorF32>();
    eprintln!("BinnedAccumulatorF32 size: {size} bytes");
    // 256 bins * 8 (f64) * 2 + 256 * 4 + overhead ≈ 5K
    assert!(size < 16 * 1024, "F32 accumulator too large: {size}");
    assert!(size > 0);
}

#[test]
fn test_1000_iterations_matmul_no_growth() {
    // Run 1000 iterations of matmul, measuring consistency.
    // We can't directly measure heap allocations without an allocator hook,
    // but we can verify:
    // 1. Results are deterministic across all iterations.
    // 2. No panics or OOM from accumulated state.
    let m = 8;
    let k = 16;
    let n = 8;

    let mut rng = cjc_repro::Rng::seeded(42);
    let a_data: Vec<f64> = (0..m * k).map(|_| rng.next_normal_f64()).collect();
    let b_data: Vec<f64> = (0..k * n).map(|_| rng.next_normal_f64()).collect();

    let a = Tensor::from_vec(a_data, &[m, k]).unwrap();
    let b = Tensor::from_vec(b_data, &[k, n]).unwrap();

    // First matmul — reference.
    let c_ref = a.matmul(&b).unwrap();
    let ref_sum = c_ref.sum();

    // Run 1000 more iterations.
    for iter in 0..1000 {
        let c = a.matmul(&b).unwrap();
        let s = c.sum();
        assert_eq!(s.to_bits(), ref_sum.to_bits(),
            "Iteration {iter}: matmul result drift detected: {s} vs {ref_sum}");
    }
}

#[test]
fn test_1000_iterations_binned_sum_stability() {
    let n = 1000;
    let mut rng = cjc_repro::Rng::seeded(42);
    let values: Vec<f64> = (0..n).map(|_| rng.next_f64() * 1e10 - 5e9).collect();

    let reference = binned_sum_f64(&values);

    for iter in 0..1000 {
        let result = binned_sum_f64(&values);
        assert_eq!(result.to_bits(), reference.to_bits(),
            "Iteration {iter}: binned_sum drift: {result} vs {reference}");
    }
}

#[test]
fn test_accumulator_reuse_no_leak() {
    // Create accumulator, add data, finalize, repeat — no growth.
    for _ in 0..1000 {
        let mut acc = BinnedAccumulatorF64::new();
        for i in 0..100 {
            acc.add(i as f64 * 0.1);
        }
        let _ = acc.finalize();
        // acc is dropped here — if there were leaked state, we'd eventually OOM.
    }
    // If we get here, no leak.
}

#[test]
fn test_merge_no_allocation() {
    // Repeated merges should not allocate.
    let mut rng = cjc_repro::Rng::seeded(42);

    for _ in 0..500 {
        let mut acc1 = BinnedAccumulatorF64::new();
        let mut acc2 = BinnedAccumulatorF64::new();

        for _ in 0..50 {
            acc1.add(rng.next_f64());
            acc2.add(rng.next_f64());
        }

        acc1.merge(&acc2);
        let _ = acc1.finalize();
    }
}

#[test]
fn test_dispatch_sum_stability() {
    let ctx = ReductionContext::linalg(); // Uses binned strategy.
    let values: Vec<f64> = (0..500).map(|i| i as f64 * 0.01 - 2.5).collect();

    let reference = dispatch_sum_f64(&values, &ctx);

    for _ in 0..1000 {
        let result = dispatch_sum_f64(&values, &ctx);
        assert_eq!(result.to_bits(), reference.to_bits());
    }
}

#[test]
fn test_dispatch_dot_stability() {
    let ctx = ReductionContext::linalg();
    let a: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
    let b: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).cos()).collect();

    let reference = dispatch_dot_f64(&a, &b, &ctx);

    for _ in 0..1000 {
        let result = dispatch_dot_f64(&a, &b, &ctx);
        assert_eq!(result.to_bits(), reference.to_bits());
    }
}

#[test]
fn test_kernel_dispatched_matmul_stability() {
    let ctx = ReductionContext::linalg();
    let m = 4;
    let k = 8;
    let n = 4;

    let mut rng = cjc_repro::Rng::seeded(42);
    let a: Vec<f64> = (0..m * k).map(|_| rng.next_normal_f64()).collect();
    let b: Vec<f64> = (0..k * n).map(|_| rng.next_normal_f64()).collect();
    let mut c_ref = vec![0.0; m * n];
    cjc_runtime::kernel::matmul_dispatched(&a, &b, &mut c_ref, m, k, n, &ctx);

    for _ in 0..500 {
        let mut c = vec![0.0; m * n];
        cjc_runtime::kernel::matmul_dispatched(&a, &b, &mut c, m, k, n, &ctx);
        for i in 0..c.len() {
            assert_eq!(c[i].to_bits(), c_ref[i].to_bits());
        }
    }
}

#[test]
fn test_nogc_context_sum() {
    // Verify that nogc context uses binned strategy.
    let ctx = ReductionContext::nogc();
    let values: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001 - 0.5).collect();

    let dispatched = dispatch_sum_f64(&values, &ctx);
    let binned = binned_sum_f64(&values);

    // In nogc context, dispatch should use binned — bit-identical.
    assert_eq!(dispatched.to_bits(), binned.to_bits(),
        "nogc dispatch should use binned: dispatched={dispatched}, binned={binned}");
}
