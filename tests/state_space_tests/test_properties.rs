//! Property tests for `state_space_*` primitives.
//!
//! Each property generates random inputs and asserts an invariant. Properties:
//!   1. Same seed → bit-identical hidden state after N steps.
//!   2. Different seeds → different trajectories (non-equality).
//!   3. Scanning T-step sequence equals stepping one row at a time.
//!   4. Reset + replay equals fresh `init`.
//!   5. Snapshot + restore preserves future outputs.
//!   6. Different input sequences → different hidden states.

use proptest::prelude::*;

use crate::harness::*;
use cjc_runtime::tensor::Tensor;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Generate a deterministic Vec<f64> in `[-1, 1]` from a u64 seed.
fn det_vec(seed: u64, n: usize) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            let bits = splitmix64(&mut s);
            (bits as f64 / u64::MAX as f64) * 2.0 - 1.0
        })
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 128,
        ..ProptestConfig::default()
    })]

    /// Same `(input_dim, hidden_dim, output_dim, seed)` and same input sequence
    /// must produce a bit-identical hidden-state trajectory.
    #[test]
    fn same_seed_same_input_bit_identical(
        seed in 1u64..1_000_000,
        steps in 1usize..8,
        input_dim in 1usize..5,
        hidden_dim in 1usize..8,
        output_dim in 1usize..5,
    ) {
        clear();
        let h_a = ssm_init(input_dim as i64, hidden_dim as i64, output_dim as i64, seed as i64);
        let h_b = ssm_init(input_dim as i64, hidden_dim as i64, output_dim as i64, seed as i64);
        for step in 0..steps {
            let x = det_vec(seed.wrapping_add(step as u64), input_dim);
            let xt = Tensor::from_vec(x.clone(), &[input_dim]).unwrap();
            let y_a = ssm_step(h_a, xt.clone());
            let y_b = ssm_step(h_b, xt);
            prop_assert_eq!(y_a.to_vec(), y_b.to_vec());
        }
        prop_assert_eq!(ssm_state(h_a).to_vec(), ssm_state(h_b).to_vec());
    }

    /// Different seeds → at least one differing component in the hidden state
    /// after one nonzero-input step. (With zero inputs and zero initial state
    /// the recurrence trivially yields zeros regardless of seed; a nonzero
    /// input rules that out.)
    #[test]
    fn different_seed_diverges(
        seed_a in 1u64..1_000,
        delta in 1u64..1_000,
        input_dim in 1usize..4,
        hidden_dim in 2usize..6,
    ) {
        clear();
        let seed_b = seed_a.wrapping_add(delta);
        let h_a = ssm_init(input_dim as i64, hidden_dim as i64, 1, seed_a as i64);
        let h_b = ssm_init(input_dim as i64, hidden_dim as i64, 1, seed_b as i64);
        let x = vec![1.0_f64; input_dim];
        let xt = Tensor::from_vec(x, &[input_dim]).unwrap();
        let _ = ssm_step(h_a, xt.clone());
        let _ = ssm_step(h_b, xt);
        prop_assert_ne!(ssm_state(h_a).to_vec(), ssm_state(h_b).to_vec());
    }

    /// Scanning a T-row matrix == stepping one row at a time, same cell setup.
    #[test]
    fn scan_equals_repeated_step(
        seed in 1u64..10_000,
        t in 1usize..6,
        input_dim in 1usize..4,
        hidden_dim in 1usize..6,
        output_dim in 1usize..4,
    ) {
        clear();
        let h_scan = ssm_init(input_dim as i64, hidden_dim as i64, output_dim as i64, seed as i64);
        let h_step = ssm_init(input_dim as i64, hidden_dim as i64, output_dim as i64, seed as i64);
        let xs_data = det_vec(seed.wrapping_mul(7), t * input_dim);
        let xs = Tensor::from_vec(xs_data.clone(), &[t, input_dim]).unwrap();
        let ys = ssm_scan(h_scan, xs);
        let mut acc = Vec::new();
        for step in 0..t {
            let row = xs_data[step * input_dim..(step + 1) * input_dim].to_vec();
            let xt = Tensor::from_vec(row, &[input_dim]).unwrap();
            let y = ssm_step(h_step, xt);
            acc.extend(y.to_vec());
        }
        prop_assert_eq!(ys.to_vec(), acc);
    }

    /// Reset + step from fresh state must agree with stepping a never-stepped cell.
    #[test]
    fn reset_equals_fresh(
        seed in 1u64..10_000,
        input_dim in 1usize..4,
        hidden_dim in 1usize..6,
    ) {
        clear();
        let h_a = ssm_init(input_dim as i64, hidden_dim as i64, 2, seed as i64);
        // Disturb cell A
        for i in 0..3 {
            let x = det_vec(seed.wrapping_add(i), input_dim);
            let _ = ssm_step(h_a, Tensor::from_vec(x, &[input_dim]).unwrap());
        }
        ssm_reset(h_a);
        let h_b = ssm_init(input_dim as i64, hidden_dim as i64, 2, seed as i64);
        let x = det_vec(seed.wrapping_mul(13), input_dim);
        let xt = Tensor::from_vec(x, &[input_dim]).unwrap();
        let y_a = ssm_step(h_a, xt.clone());
        let y_b = ssm_step(h_b, xt);
        prop_assert_eq!(y_a.to_vec(), y_b.to_vec());
    }

    /// Snapshot + restore preserves the entire future.
    #[test]
    fn snapshot_restore_preserves_future(
        seed in 1u64..10_000,
        future_steps in 1usize..5,
        input_dim in 1usize..4,
        hidden_dim in 1usize..6,
    ) {
        clear();
        let h = ssm_init(input_dim as i64, hidden_dim as i64, 2, seed as i64);
        // Step forward to mid-game
        for i in 0..2 {
            let x = det_vec(seed.wrapping_add(i), input_dim);
            let _ = ssm_step(h, Tensor::from_vec(x, &[input_dim]).unwrap());
        }
        let snap = ssm_snapshot(h);
        // Future-A: deterministic sequence
        let mut future_a = Vec::new();
        for i in 0..future_steps {
            let x = det_vec(seed.wrapping_mul(101).wrapping_add(i as u64), input_dim);
            let xt = Tensor::from_vec(x, &[input_dim]).unwrap();
            let y = ssm_step(h, xt);
            future_a.extend(y.to_vec());
        }
        // Disturb, then restore and replay the same future sequence
        let _ = ssm_step(h, Tensor::from_vec(vec![0.5_f64; input_dim], &[input_dim]).unwrap());
        ssm_restore(h, snap);
        let mut future_b = Vec::new();
        for i in 0..future_steps {
            let x = det_vec(seed.wrapping_mul(101).wrapping_add(i as u64), input_dim);
            let xt = Tensor::from_vec(x, &[input_dim]).unwrap();
            let y = ssm_step(h, xt);
            future_b.extend(y.to_vec());
        }
        prop_assert_eq!(future_a, future_b);
    }

    /// Different input sequences → different hidden states (with overwhelming
    /// probability for a randomly initialised cell). Generate two sequences
    /// guaranteed to differ at index 0 by sign, and check resulting states.
    #[test]
    fn different_input_diverges(
        seed in 1u64..10_000,
        input_dim in 1usize..4,
        hidden_dim in 2usize..6,
    ) {
        clear();
        let h_a = ssm_init(input_dim as i64, hidden_dim as i64, 1, seed as i64);
        let h_b = ssm_init(input_dim as i64, hidden_dim as i64, 1, seed as i64);
        let x_a = vec![1.0_f64; input_dim];
        let x_b = vec![-1.0_f64; input_dim];
        let _ = ssm_step(h_a, Tensor::from_vec(x_a, &[input_dim]).unwrap());
        let _ = ssm_step(h_b, Tensor::from_vec(x_b, &[input_dim]).unwrap());
        prop_assert_ne!(ssm_state(h_a).to_vec(), ssm_state(h_b).to_vec());
    }
}
