//! Bolero fuzz targets — `fused_matmul_norm` survives arbitrary inputs.

use bolero::check;

use cjc_repro::KahanAccumulatorF64;
use cjc_runtime::accumulator::{binned_sum_f64, fused_matmul_norm_kernel};
use cjc_runtime::builtins::dispatch_builtin;
use cjc_runtime::tensor::Tensor;
use cjc_runtime::value::Value;

fn unfused_reference(a: &[f64], w: &[f64], ord: i64, m: usize, k: usize, n: usize) -> f64 {
    let mut intermediate = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            let mut acc = KahanAccumulatorF64::new();
            for kk in 0..k {
                acc.add(a[i * k + kk] * w[kk * n + j]);
            }
            intermediate.push(acc.finalize());
        }
    }
    match ord {
        1 => binned_sum_f64(&intermediate.iter().map(|x| x.abs()).collect::<Vec<_>>()),
        2 => binned_sum_f64(&intermediate.iter().map(|x| x * x).collect::<Vec<_>>()).sqrt(),
        p => {
            let pf = p as f64;
            binned_sum_f64(&intermediate.iter().map(|x| x.abs().powf(pf)).collect::<Vec<_>>())
                .powf(1.0 / pf)
        }
    }
}

#[test]
fn fuzz_fused_matmul_norm_values() {
    // Fixed shape (m=k=n=2): a[4] + w[4] = 8 floats.
    check!().with_type::<[f64; 8]>().for_each(|arr: &[f64; 8]| {
        if arr.iter().any(|x| !x.is_finite() || x.abs() > 1.0e3) {
            return;
        }
        let a = arr[0..4].to_vec();
        let w = arr[4..8].to_vec();

        let fused = fused_matmul_norm_kernel(&a, &w, 2, 2, 2, 2);
        let unfused = unfused_reference(&a, &w, 2, 2, 2, 2);
        assert_eq!(
            fused.to_bits(),
            unfused.to_bits(),
            "fused != unfused for {arr:?}"
        );
        assert!(fused >= 0.0 && fused.is_finite(), "L2 norm = {fused}");
    });
}

#[test]
fn fuzz_fused_matmul_norm_shapes() {
    // Random shapes; matching dims succeed, mismatched dims Err gracefully.
    check!().with_type::<[u8; 4]>().for_each(|dims: &[u8; 4]| {
        let m = (dims[0] % 6) as usize + 1;
        let k_a = (dims[1] % 6) as usize + 1;
        let k_w = if dims[2] % 2 == 0 {
            k_a
        } else {
            k_a.wrapping_add(1).max(1)
        };
        let n = (dims[3] % 6) as usize + 1;

        let a = Tensor::from_vec(vec![0.7; m * k_a], &[m, k_a]).unwrap();
        let w = Tensor::from_vec(vec![1.3; k_w * n], &[k_w, n]).unwrap();

        let res = dispatch_builtin(
            "fused_matmul_norm",
            &[Value::Tensor(a), Value::Tensor(w)],
        );

        if k_a == k_w {
            assert!(
                matches!(res, Ok(Some(Value::Float(_)))),
                "matching ({m}x{k_a}, {k_w}x{n}) should succeed, got {res:?}"
            );
        } else {
            assert!(
                res.is_err(),
                "mismatched ({k_a} vs {k_w}) should Err, got {res:?}"
            );
        }

        // Dispatch survives.
        let alive = dispatch_builtin("runtime_policy_batch_size", &[]);
        assert!(
            matches!(alive, Ok(Some(Value::Int(_)))),
            "dispatch corrupted: {alive:?}"
        );
    });
}
