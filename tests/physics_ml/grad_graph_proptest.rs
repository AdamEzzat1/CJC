//! Phase 3c — proptests for `grad_graph_*` builtin wrappers.
//!
//! Each property generates random inputs and asserts that the value flowing
//! out of the language-level `dispatch_grad_graph` path is bit-identical to
//! the value computed by the underlying `cjc_runtime::Tensor` op (for the
//! ones that map cleanly) or by `cjc_ad::GradGraph` directly. This is the
//! "wrapper preserves values" invariant — the satellite dispatch must not
//! drop precision, reorder reductions, or accidentally promote scalar types.
//!
//! Five properties (per the Phase 3c brief): add, mul, matmul, mlp_layer,
//! tanh. Each runs the proptest default 256 cases.

use proptest::prelude::*;

use cjc_ad::dispatch_grad_graph;
use cjc_ad::pinn::Activation;
use cjc_ad::GradGraph;
use cjc_runtime::tensor::Tensor;
use cjc_runtime::value::Value;

/// SplitMix64 hand-rolled here so proptest can deterministically derive a
/// `Vec<f64>` from a single `u64` seed without standing up the full RNG
/// infrastructure. Identical mix step as `cjc_repro::Rng`.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn det_floats(seed: u64, n: usize) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            // Map to a finite, well-conditioned range to keep tanh and matmul
            // numerically civil. Domain [-2, 2] avoids tanh saturation; matmul
            // values stay bounded.
            let bits = splitmix64(&mut s);
            let f = (bits as f64 / u64::MAX as f64) * 4.0 - 2.0;
            f
        })
        .collect()
}

fn t_value(t: Tensor) -> Value {
    Value::Tensor(t)
}

fn idx(v: Value) -> Value {
    v
}

fn unwrap_idx(v: Option<Value>) -> i64 {
    match v {
        Some(Value::Int(i)) => i,
        other => panic!("expected Int, got {:?}", other),
    }
}

fn unwrap_tensor(v: Option<Value>) -> Tensor {
    match v {
        Some(Value::Tensor(t)) => t,
        other => panic!("expected Tensor, got {:?}", other),
    }
}

fn forward(node: i64) -> Tensor {
    unwrap_tensor(
        dispatch_grad_graph("grad_graph_forward", &[Value::Int(node)])
            .unwrap(),
    )
}

fn input_node(t: Tensor) -> i64 {
    unwrap_idx(
        dispatch_grad_graph("grad_graph_input", &[t_value(t)])
            .unwrap(),
    )
}

fn reset() {
    let _ = dispatch_grad_graph("grad_graph_new", &[]).unwrap();
}

proptest! {
    #![proptest_config(ProptestConfig {
        // 256 cases per property — what the brief specifies as the floor.
        cases: 256,
        ..ProptestConfig::default()
    })]

    /// `grad_graph_add(a,b)` forward = element-wise a+b.
    #[test]
    fn add_forward_matches_direct(
        n in 1usize..16,
        seed in any::<u64>(),
    ) {
        let a_data = det_floats(seed, n);
        let b_data = det_floats(seed.wrapping_add(1), n);
        let a = Tensor::from_vec(a_data.clone(), &[n]).unwrap();
        let b = Tensor::from_vec(b_data.clone(), &[n]).unwrap();

        // Direct: element-wise add via Vec arithmetic. This is the ground
        // truth; we are checking that the dispatch path doesn't reorder or
        // drop bits.
        let expected: Vec<f64> = a_data.iter()
            .zip(b_data.iter())
            .map(|(x, y)| x + y)
            .collect();

        reset();
        let ai = input_node(a);
        let bi = input_node(b);
        let s = unwrap_idx(
            dispatch_grad_graph("grad_graph_add",
                &[Value::Int(ai), Value::Int(bi)]).unwrap(),
        );
        let got = forward(s).to_vec();

        prop_assert_eq!(got, expected);
    }

    /// `grad_graph_mul(a,b)` forward = element-wise a*b.
    #[test]
    fn mul_forward_matches_direct(
        n in 1usize..16,
        seed in any::<u64>(),
    ) {
        let a_data = det_floats(seed, n);
        let b_data = det_floats(seed.wrapping_add(1), n);
        let a = Tensor::from_vec(a_data.clone(), &[n]).unwrap();
        let b = Tensor::from_vec(b_data.clone(), &[n]).unwrap();

        let expected: Vec<f64> = a_data.iter()
            .zip(b_data.iter())
            .map(|(x, y)| x * y)
            .collect();

        reset();
        let ai = input_node(a);
        let bi = input_node(b);
        let s = unwrap_idx(
            dispatch_grad_graph("grad_graph_mul",
                &[Value::Int(ai), Value::Int(bi)]).unwrap(),
        );
        let got = forward(s).to_vec();

        prop_assert_eq!(got, expected);
    }

    /// `grad_graph_matmul(a,b)` matches `Tensor::matmul`. Bit-equality is
    /// the bar — both call the same deterministic kernel underneath.
    #[test]
    fn matmul_node_matches_direct_call(
        m in 1usize..6,
        k in 1usize..6,
        n in 1usize..6,
        seed in any::<u64>(),
    ) {
        let a_data = det_floats(seed, m * k);
        let b_data = det_floats(seed.wrapping_add(1), k * n);
        let a = Tensor::from_vec(a_data, &[m, k]).unwrap();
        let b = Tensor::from_vec(b_data, &[k, n]).unwrap();

        // Direct call.
        let expected = a.matmul(&b).unwrap();

        // Dispatch path.
        reset();
        let ai = input_node(a);
        let bi = input_node(b);
        let mi = unwrap_idx(
            dispatch_grad_graph("grad_graph_matmul",
                &[Value::Int(ai), Value::Int(bi)]).unwrap(),
        );
        let got = forward(mi);

        prop_assert_eq!(got.shape().to_vec(), expected.shape().to_vec());
        prop_assert_eq!(got.to_vec(), expected.to_vec());
    }

    /// `grad_graph_mlp_layer(x, w, b, "tanh")` matches building the same
    /// fused layer through `cjc_ad::GradGraph::mlp_layer` directly. The
    /// extra wrapper is the dispatch + activation-string parser.
    #[test]
    fn mlp_layer_dispatch_matches_direct_graph(
        in_features in 1usize..5,
        out_features in 1usize..5,
        seed in any::<u64>(),
    ) {
        let x_data = det_floats(seed, in_features);
        let w_data = det_floats(seed.wrapping_add(1), out_features * in_features);
        let b_data = det_floats(seed.wrapping_add(2), out_features);

        // Direct: fresh GradGraph, call mlp_layer.
        let mut g = GradGraph::new();
        let xi = g.input(Tensor::from_vec(x_data.clone(), &[1, in_features]).unwrap());
        let wi = g.parameter(Tensor::from_vec(w_data.clone(), &[out_features, in_features]).unwrap());
        let bi = g.parameter(Tensor::from_vec(b_data.clone(), &[out_features]).unwrap());
        let yi = g.mlp_layer(xi, wi, bi, Activation::Tanh);
        let expected = g.tensor(yi);

        // Dispatch: same inputs through the language-level path.
        reset();
        let xn = input_node(Tensor::from_vec(x_data, &[1, in_features]).unwrap());
        let wn = unwrap_idx(
            dispatch_grad_graph("grad_graph_param",
                &[t_value(Tensor::from_vec(w_data, &[out_features, in_features]).unwrap())]).unwrap(),
        );
        let bn = unwrap_idx(
            dispatch_grad_graph("grad_graph_param",
                &[t_value(Tensor::from_vec(b_data, &[out_features]).unwrap())]).unwrap(),
        );
        let yn = unwrap_idx(
            dispatch_grad_graph("grad_graph_mlp_layer",
                &[Value::Int(xn), Value::Int(wn), Value::Int(bn),
                  Value::String(std::rc::Rc::new("tanh".to_string()))]).unwrap(),
        );
        let got = forward(yn);

        prop_assert_eq!(got.shape().to_vec(), expected.shape().to_vec());
        prop_assert_eq!(got.to_vec(), expected.to_vec());
    }

    /// `grad_graph_tanh(a)` matches element-wise `f64::tanh`.
    #[test]
    fn tanh_forward_matches_direct(
        n in 1usize..16,
        seed in any::<u64>(),
    ) {
        let a_data = det_floats(seed, n);
        let a = Tensor::from_vec(a_data.clone(), &[n]).unwrap();

        let expected: Vec<f64> = a_data.iter().map(|x| x.tanh()).collect();

        reset();
        let ai = input_node(a);
        let ti = unwrap_idx(
            dispatch_grad_graph("grad_graph_tanh", &[Value::Int(ai)]).unwrap(),
        );
        let got = forward(ti).to_vec();

        prop_assert_eq!(got, expected);
    }
}

// `idx` helper prevents an unused-import warning on `Value::Int` paths
// that get inlined; some compilers flag the helper as dead. Keep it.
#[allow(dead_code)]
fn _idx_marker() -> Value {
    idx(Value::Int(0))
}
