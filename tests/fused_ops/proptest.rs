//! GC-06 Phase 3a — property tests for fused elementwise ops.
//!
//! The core determinism guarantee: each fused single-pass kernel produces
//! **byte-identical** output to the equivalent unfused sequence of tensor ops.
//! Fusion changes memory traffic and allocation count, never the bits. This is
//! what makes the memory-efficiency win free of any numerical cost.

use proptest::prelude::*;

use cjc_runtime::builtins::dispatch_builtin;
use cjc_runtime::tensor::Tensor;
use cjc_runtime::value::Value;

/// SplitMix64 (same mix as `cjc_repro::Rng`) for deterministic test data.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Deterministic finite floats in [-4, 4] (well-conditioned for the kernels).
fn det_floats(seed: u64, n: usize) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| (splitmix64(&mut s) as f64 / u64::MAX as f64) * 8.0 - 4.0)
        .collect()
}

fn tensor(data: Vec<f64>) -> Tensor {
    let n = data.len();
    Tensor::from_vec(data, &[n]).unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        ..ProptestConfig::default()
    })]

    /// `a*b + c` fused == unfused `mul_elem` then `add`.
    #[test]
    fn fused_mul_add_equals_unfused(n in 1usize..64, seed in any::<u64>()) {
        let a = tensor(det_floats(seed, n));
        let b = tensor(det_floats(seed.wrapping_add(1), n));
        let c = tensor(det_floats(seed.wrapping_add(2), n));
        let fused = a.fused_mul_add(&b, &c).unwrap();
        let unfused = a.mul_elem(&b).unwrap().add(&c).unwrap();
        prop_assert_eq!(fused.to_vec(), unfused.to_vec());
    }

    /// `alpha*x + y` fused == unfused `scalar_mul` then `add`.
    #[test]
    fn fused_axpy_equals_unfused(
        n in 1usize..64,
        seed in any::<u64>(),
        alpha in -4.0f64..4.0,
    ) {
        let x = tensor(det_floats(seed, n));
        let y = tensor(det_floats(seed.wrapping_add(1), n));
        let fused = x.fused_axpy(alpha, &y).unwrap();
        let unfused = x.scalar_mul(alpha).add(&y).unwrap();
        prop_assert_eq!(fused.to_vec(), unfused.to_vec());
    }

    /// `a*b - c` fused == unfused `mul_elem` then `sub`.
    #[test]
    fn fused_mul_sub_equals_unfused(n in 1usize..64, seed in any::<u64>()) {
        let a = tensor(det_floats(seed, n));
        let b = tensor(det_floats(seed.wrapping_add(1), n));
        let c = tensor(det_floats(seed.wrapping_add(2), n));
        let fused = a.fused_mul_sub(&b, &c).unwrap();
        let unfused = a.mul_elem(&b).unwrap().sub(&c).unwrap();
        prop_assert_eq!(fused.to_vec(), unfused.to_vec());
    }

    /// `(a-b)^2` fused == unfused `sub` then `mul_elem(self)`.
    #[test]
    fn fused_sub_sq_equals_unfused(n in 1usize..64, seed in any::<u64>()) {
        let a = tensor(det_floats(seed, n));
        let b = tensor(det_floats(seed.wrapping_add(1), n));
        let fused = a.fused_sub_sq(&b).unwrap();
        let d = a.sub(&b).unwrap();
        let unfused = d.mul_elem(&d).unwrap();
        prop_assert_eq!(fused.to_vec(), unfused.to_vec());
    }

    /// The builtin dispatch path matches the method (wiring is value-preserving).
    #[test]
    fn fused_builtins_match_methods(
        n in 1usize..64,
        seed in any::<u64>(),
        alpha in -4.0f64..4.0,
    ) {
        let a = tensor(det_floats(seed, n));
        let b = tensor(det_floats(seed.wrapping_add(1), n));
        let c = tensor(det_floats(seed.wrapping_add(2), n));

        let axpy_b = match dispatch_builtin(
            "fused_axpy",
            &[Value::Float(alpha), Value::Tensor(a.clone()), Value::Tensor(b.clone())],
        ) {
            Ok(Some(Value::Tensor(t))) => t.to_vec(),
            other => panic!("fused_axpy dispatch returned {other:?}"),
        };
        prop_assert_eq!(axpy_b, a.fused_axpy(alpha, &b).unwrap().to_vec());

        let ms_b = match dispatch_builtin(
            "fused_mul_sub",
            &[Value::Tensor(a.clone()), Value::Tensor(b.clone()), Value::Tensor(c.clone())],
        ) {
            Ok(Some(Value::Tensor(t))) => t.to_vec(),
            other => panic!("fused_mul_sub dispatch returned {other:?}"),
        };
        prop_assert_eq!(ms_b, a.fused_mul_sub(&b, &c).unwrap().to_vec());

        let sq_b = match dispatch_builtin(
            "fused_sub_sq",
            &[Value::Tensor(a.clone()), Value::Tensor(b.clone())],
        ) {
            Ok(Some(Value::Tensor(t))) => t.to_vec(),
            other => panic!("fused_sub_sq dispatch returned {other:?}"),
        };
        prop_assert_eq!(sq_b, a.fused_sub_sq(&b).unwrap().to_vec());
    }
}
