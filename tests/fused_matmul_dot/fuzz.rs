//! Bolero fuzz targets — `fused_matmul_dot` survives arbitrary inputs.
//!
//! 1. `fuzz_fused_matmul_dot_values` — random bounded-finite tensors of fixed
//!    shape; assert fused == unfused and stays finite.
//! 2. `fuzz_fused_matmul_dot_shapes` — random tensor shapes; matching shapes
//!    succeed, mismatched shapes `Err` gracefully (no panic, dispatch survives).

use bolero::check;

use cjc_repro::KahanAccumulatorF64;
use cjc_runtime::accumulator::{binned_sum_f64, fused_matmul_dot_kernel};
use cjc_runtime::builtins::dispatch_builtin;
use cjc_runtime::tensor::Tensor;
use cjc_runtime::value::Value;

/// Unfused reference at the kernel level (matches the design doc's contract).
fn unfused_reference(a: &[f64], w: &[f64], v: &[f64], m: usize, n: usize) -> f64 {
    let mut intermediate = Vec::with_capacity(n);
    for j in 0..n {
        let mut k = KahanAccumulatorF64::new();
        for i in 0..m {
            k.add(a[i] * w[i * n + j]);
        }
        intermediate.push(k.finalize());
    }
    let products: Vec<f64> = intermediate
        .iter()
        .zip(v.iter())
        .map(|(x, y)| x * y)
        .collect();
    binned_sum_f64(&products)
}

#[test]
fn fuzz_fused_matmul_dot_values() {
    // Fixed shape (m=2, n=3): a[2], w[2*3=6], v[3] → 11 floats.
    check!().with_type::<[f64; 11]>().for_each(|arr: &[f64; 11]| {
        // Bound inputs to a civil range so accumulation stays well-conditioned.
        if arr.iter().any(|x| !x.is_finite() || x.abs() > 1.0e3) {
            return;
        }
        let a = arr[0..2].to_vec();
        let w = arr[2..8].to_vec();
        let v = arr[8..11].to_vec();

        let fused = fused_matmul_dot_kernel(&a, &w, &v, 2, 3);
        let unfused = unfused_reference(&a, &w, &v, 2, 3);
        assert_eq!(
            fused.to_bits(),
            unfused.to_bits(),
            "fused != unfused for {arr:?}: fused={fused:e}, unfused={unfused:e}"
        );
        assert!(fused.is_finite(), "non-finite output {fused} for {arr:?}");
    });
}

#[test]
fn fuzz_fused_matmul_dot_shapes() {
    // Random shapes — match (success) or mismatch (graceful error).
    check!().with_type::<[u8; 3]>().for_each(|dims: &[u8; 3]| {
        let m = (dims[0] % 8) as usize + 1;
        let n = (dims[1] % 8) as usize + 1;
        // Force shape mismatch when dims[2] is odd, otherwise match.
        let v_len = if dims[2] % 2 == 0 {
            n
        } else {
            n.wrapping_add(1).max(1)
        };

        let a = Tensor::from_vec(vec![1.5; m], &[m]).unwrap();
        let w = Tensor::from_vec(vec![2.5; m * n], &[m, n]).unwrap();
        let v = Tensor::from_vec(vec![0.5; v_len], &[v_len]).unwrap();

        let res = dispatch_builtin(
            "fused_matmul_dot",
            &[Value::Tensor(a), Value::Tensor(w), Value::Tensor(v)],
        );

        if v_len == n {
            assert!(
                matches!(res, Ok(Some(Value::Float(_)))),
                "matching shape ({m}, {n}, v={v_len}) should succeed, got {res:?}",
            );
        } else {
            assert!(
                res.is_err(),
                "mismatched shape (n={n} vs v_len={v_len}) should Err, got {res:?}",
            );
        }

        // Dispatch must remain usable afterwards.
        let alive = dispatch_builtin("runtime_policy_batch_size", &[]);
        assert!(
            matches!(alive, Ok(Some(Value::Int(_)))),
            "dispatch corrupted: {alive:?}"
        );
    });
}
