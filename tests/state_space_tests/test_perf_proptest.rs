//! Property tests for Phase-2 performance primitives.
//!
//! Properties:
//!   1. `tensor_concat_1d` is associative under composition.
//!   2. `tensor_concat_1d` length is exact (`a.len + b.len`).
//!   3. `state_space_step_with_readout` is bit-equivalent to step + state + readout.
//!   4. `state_space_step_batched` with B identical rows yields B identical output rows.
//!   5. Reconstructing one step from extracted weights matches the cell's own step.

use proptest::prelude::*;

use crate::harness::*;
use cjc_runtime::state_space::{dispatch_state_space, tensor_concat_1d};
use cjc_runtime::tensor::Tensor;
use cjc_runtime::value::Value;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn det_vec(seed: u64, n: usize) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| (splitmix64(&mut s) as f64 / u64::MAX as f64) * 2.0 - 1.0)
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 128,
        ..ProptestConfig::default()
    })]

    /// Length of concat is exactly the sum of input lengths.
    #[test]
    fn concat_length_is_sum(
        seed in 1u64..1_000_000,
        n_a in 0usize..32,
        n_b in 0usize..32,
    ) {
        let a = Tensor::from_vec(det_vec(seed, n_a), &[n_a]).unwrap();
        let b = Tensor::from_vec(det_vec(seed.wrapping_add(1), n_b), &[n_b]).unwrap();
        let c = tensor_concat_1d(&a, &b).unwrap();
        prop_assert_eq!(c.shape(), &[n_a + n_b]);
        prop_assert_eq!(c.len(), n_a + n_b);
    }

    /// `concat(concat(a,b), c) == concat(a, concat(b,c))` element-wise.
    #[test]
    fn concat_is_associative(
        seed in 1u64..1_000_000,
        n_a in 1usize..16,
        n_b in 1usize..16,
        n_c in 1usize..16,
    ) {
        let a = Tensor::from_vec(det_vec(seed, n_a), &[n_a]).unwrap();
        let b = Tensor::from_vec(det_vec(seed.wrapping_add(1), n_b), &[n_b]).unwrap();
        let c = Tensor::from_vec(det_vec(seed.wrapping_add(2), n_c), &[n_c]).unwrap();
        let left = tensor_concat_1d(&tensor_concat_1d(&a, &b).unwrap(), &c).unwrap();
        let right = tensor_concat_1d(&a, &tensor_concat_1d(&b, &c).unwrap()).unwrap();
        prop_assert_eq!(left.to_vec(), right.to_vec());
    }

    /// Concat preserves data: first n_a elements equal `a`, last n_b equal `b`.
    #[test]
    fn concat_preserves_data(
        seed in 1u64..1_000_000,
        n_a in 1usize..16,
        n_b in 1usize..16,
    ) {
        let a_d = det_vec(seed, n_a);
        let b_d = det_vec(seed.wrapping_add(1), n_b);
        let a = Tensor::from_vec(a_d.clone(), &[n_a]).unwrap();
        let b = Tensor::from_vec(b_d.clone(), &[n_b]).unwrap();
        let c = tensor_concat_1d(&a, &b).unwrap();
        let cd = c.to_vec();
        prop_assert_eq!(cd[..n_a].to_vec(), a_d);
        prop_assert_eq!(cd[n_a..].to_vec(), b_d);
    }

    /// `state_space_step_with_readout` is bit-equivalent to calling step,
    /// then reading state, then computing readout.
    #[test]
    fn fused_step_matches_split_path(
        seed in 1u64..10_000,
        input_dim in 1usize..5,
        hidden_dim in 1usize..6,
        output_dim in 1usize..5,
    ) {
        clear();
        let h_fused = ssm_init(input_dim as i64, hidden_dim as i64, output_dim as i64, seed as i64);
        let h_split = ssm_init(input_dim as i64, hidden_dim as i64, output_dim as i64, seed as i64);
        let x = Tensor::from_vec(det_vec(seed.wrapping_mul(7), input_dim), &[input_dim]).unwrap();

        let v = dispatch_state_space(
            "state_space_step_with_readout",
            &[Value::Int(h_fused), Value::Tensor(x.clone())],
        )
        .unwrap()
        .unwrap();
        let arr = match v {
            Value::Array(a) => a,
            _ => panic!("expected Array"),
        };
        let y_fused = match &arr[0] {
            Value::Tensor(t) => t.clone(),
            _ => panic!(),
        };
        let h_after_fused = match &arr[1] {
            Value::Tensor(t) => t.clone(),
            _ => panic!(),
        };

        let _ = ssm_step(h_split, x);
        let h_after_split = ssm_state(h_split);
        let y_split = ssm_readout(h_split);

        prop_assert_eq!(y_fused.to_vec(), y_split.to_vec());
        prop_assert_eq!(h_after_fused.to_vec(), h_after_split.to_vec());
    }

    /// Batched step with B identical rows must yield B identical output rows
    /// (because each row starts from a zero hidden state).
    #[test]
    fn batched_identical_rows_yield_identical_outputs(
        seed in 1u64..10_000,
        b in 2usize..6,
        input_dim in 1usize..4,
        hidden_dim in 1usize..6,
        output_dim in 1usize..4,
    ) {
        clear();
        let h = ssm_init(input_dim as i64, hidden_dim as i64, output_dim as i64, seed as i64);
        let row = det_vec(seed.wrapping_mul(11), input_dim);
        let mut xs_data = Vec::with_capacity(b * input_dim);
        for _ in 0..b {
            xs_data.extend_from_slice(&row);
        }
        let xs = Tensor::from_vec(xs_data, &[b, input_dim]).unwrap();
        let ys = unwrap_tensor(
            dispatch_state_space("state_space_step_batched", &[Value::Int(h), Value::Tensor(xs)])
                .unwrap(),
        );
        let yd = ys.to_vec();
        let row0 = &yd[0..output_dim];
        for r in 1..b {
            let rk = &yd[r * output_dim..(r + 1) * output_dim];
            prop_assert_eq!(row0, rk);
        }
    }

    /// Single-step reconstruction from extracted weights matches the cell.
    /// We start from a freshly-reset cell (`h = 0`), so:
    ///   h_new[i] = tanh(sum_j B[i,j] * x[j])
    /// (the A·h term vanishes). This isolates B and tanh from A — testing the
    /// extractors return matrices the recurrence actually uses.
    #[test]
    fn reconstructed_first_step_from_extracted_b(
        seed in 1u64..10_000,
        input_dim in 1usize..4,
        hidden_dim in 1usize..5,
    ) {
        clear();
        let h = ssm_init(input_dim as i64, hidden_dim as i64, 1, seed as i64);
        let b_t = unwrap_tensor(dispatch_state_space("state_space_get_B", &[Value::Int(h)]).unwrap());

        let x = det_vec(seed.wrapping_mul(7), input_dim);
        let xt = Tensor::from_vec(x.clone(), &[input_dim]).unwrap();
        let _ = ssm_step(h, xt);
        let h_after = ssm_state(h).to_vec();

        let b = b_t.to_vec();
        let mut recon = vec![0.0; hidden_dim];
        for i in 0..hidden_dim {
            let mut acc = 0.0;
            for j in 0..input_dim {
                acc += b[i * input_dim + j] * x[j];
            }
            recon[i] = acc.tanh();
        }
        prop_assert_eq!(h_after, recon);
    }
}
