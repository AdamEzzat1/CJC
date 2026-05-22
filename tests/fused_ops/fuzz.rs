//! GC-06 Phase 3a — bolero fuzz targets for the fused elementwise ops.
//!
//! 1. `fuzz_fused_values` — random bounded-finite inputs; assert the fused
//!    dispatch path equals the unfused sequence (determinism under fuzzing) and
//!    stays finite.
//! 2. `fuzz_fused_shapes` — random tensor lengths; same shape must succeed,
//!    mismatched shapes must `Err` gracefully (never panic, dispatch survives).

use bolero::check;

use cjc_runtime::builtins::dispatch_builtin;
use cjc_runtime::tensor::Tensor;
use cjc_runtime::value::Value;

#[test]
fn fuzz_fused_values() {
    check!().with_type::<[f64; 6]>().for_each(|arr: &[f64; 6]| {
        // Bound inputs so the comparison stays in a civil range.
        if arr.iter().any(|x| !x.is_finite() || x.abs() > 1.0e6) {
            return;
        }
        let a = Tensor::from_vec(arr[0..2].to_vec(), &[2]).unwrap();
        let b = Tensor::from_vec(arr[2..4].to_vec(), &[2]).unwrap();
        let c = Tensor::from_vec(arr[4..6].to_vec(), &[2]).unwrap();

        let got = match dispatch_builtin(
            "fused_mul_sub",
            &[Value::Tensor(a.clone()), Value::Tensor(b.clone()), Value::Tensor(c.clone())],
        ) {
            Ok(Some(Value::Tensor(t))) => t.to_vec(),
            other => panic!("fused_mul_sub returned {other:?} for {arr:?}"),
        };
        let unfused = a.mul_elem(&b).unwrap().sub(&c).unwrap().to_vec();
        assert_eq!(got, unfused, "fused != unfused for {arr:?}");
        for v in &got {
            assert!(v.is_finite(), "non-finite fused output {v} for {arr:?}");
        }
    });
}

#[test]
fn fuzz_fused_shapes() {
    check!().with_type::<[u8; 2]>().for_each(|lens: &[u8; 2]| {
        let la = (lens[0] % 8) as usize + 1;
        let lb = (lens[1] % 8) as usize + 1;
        let a = Tensor::from_vec(vec![1.5; la], &[la]).unwrap();
        let b = Tensor::from_vec(vec![2.5; lb], &[lb]).unwrap();
        let res = dispatch_builtin(
            "fused_sub_sq",
            &[Value::Tensor(a), Value::Tensor(b)],
        );
        if la == lb {
            assert!(
                matches!(res, Ok(Some(Value::Tensor(_)))),
                "same shape ({la}) should succeed, got {res:?}",
            );
        } else {
            assert!(res.is_err(), "shape mismatch ({la} vs {lb}) should Err, got {res:?}");
        }
        // Dispatch must remain usable afterwards.
        let alive = dispatch_builtin("runtime_policy_batch_size", &[]);
        assert!(matches!(alive, Ok(Some(Value::Int(_)))), "dispatch corrupted: {alive:?}");
    });
}
