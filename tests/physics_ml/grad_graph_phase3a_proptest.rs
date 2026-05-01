//! Phase 3a — proptests for transformer-backbone `grad_graph_*` wrappers.
//!
//! For each new dispatch arm, generate a random tensor input, run the value
//! through the language-level `dispatch_grad_graph` path, and assert it is
//! bit-identical to the value computed by calling the underlying
//! `cjc_ad::GradGraph` Rust method directly. This is the
//! **wrapper-preserves-values** invariant — the dispatch must not drop
//! precision, reorder Kahan sums, or accidentally promote types.
//!
//! Six properties: softmax, cross_entropy, layer_norm, gelu, silu, reshape.
//! Default 256 cases each.

use proptest::prelude::*;

use cjc_ad::dispatch_grad_graph;
use cjc_ad::GradGraph;
use cjc_runtime::tensor::Tensor;
use cjc_runtime::value::Value;

/// SplitMix64 — same mixer as `cjc_repro::Rng`, lifted here so proptest
/// can derive a `Vec<f64>` from a single `u64` seed.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// `n` finite f64 values in `[-2, 2]` — well-conditioned for tanh/exp/log
/// without saturating the activations.
fn det_floats(seed: u64, n: usize) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            let bits = splitmix64(&mut s);
            (bits as f64 / u64::MAX as f64) * 4.0 - 2.0
        })
        .collect()
}

/// Reset the ambient graph and create one input node holding `data` of
/// shape `shape`. Returns the node index.
fn fresh_input(data: Vec<f64>, shape: &[usize]) -> i64 {
    dispatch_grad_graph("grad_graph_new", &[]).unwrap();
    let v = dispatch_grad_graph(
        "grad_graph_input",
        &[Value::Tensor(Tensor::from_vec(data, shape).unwrap())],
    )
    .unwrap()
    .unwrap();
    match v {
        Value::Int(i) => i,
        _ => unreachable!(),
    }
}

fn forward_to_vec(idx: i64) -> Vec<f64> {
    let v = dispatch_grad_graph("grad_graph_forward", &[Value::Int(idx)])
        .unwrap()
        .unwrap();
    match v {
        Value::Tensor(t) => t.to_vec(),
        _ => unreachable!(),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn prop_softmax_dispatch_eq_direct(seed in any::<u64>(), n in 1usize..=16) {
        let data = det_floats(seed, n);

        // Direct call.
        let mut g = GradGraph::new();
        let inp = g.input(Tensor::from_vec(data.clone(), &[n]).unwrap());
        let out = g.softmax(inp);
        let direct = g.tensor(out).to_vec();

        // Dispatch call.
        let inp_d = fresh_input(data, &[n]);
        let out_d = match dispatch_grad_graph("grad_graph_softmax", &[Value::Int(inp_d)])
            .unwrap()
            .unwrap()
        {
            Value::Int(i) => i,
            _ => unreachable!(),
        };
        let dispatched = forward_to_vec(out_d);

        prop_assert_eq!(
            direct.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            dispatched.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn prop_cross_entropy_dispatch_eq_direct(seed in any::<u64>(), n in 1usize..=16) {
        let logits = det_floats(seed, n);
        // Build a one-hot-ish targets vector by softmax-ing a different seed.
        let raw_targets = det_floats(seed.wrapping_add(1), n);
        let max_t = raw_targets.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_t: Vec<f64> = raw_targets.iter().map(|&x| (x - max_t).exp()).collect();
        let s_t: f64 = exp_t.iter().sum();
        let targets: Vec<f64> = exp_t.iter().map(|&x| x / s_t).collect();

        // Direct.
        let mut g = GradGraph::new();
        let lg = g.input(Tensor::from_vec(logits.clone(), &[n]).unwrap());
        let tg = g.input(Tensor::from_vec(targets.clone(), &[n]).unwrap());
        let loss = g.cross_entropy(lg, tg);
        let direct = g.tensor(loss).to_vec();

        // Dispatch.
        dispatch_grad_graph("grad_graph_new", &[]).unwrap();
        let lg_d = match dispatch_grad_graph(
            "grad_graph_input",
            &[Value::Tensor(Tensor::from_vec(logits, &[n]).unwrap())],
        ).unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() };
        let tg_d = match dispatch_grad_graph(
            "grad_graph_input",
            &[Value::Tensor(Tensor::from_vec(targets, &[n]).unwrap())],
        ).unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() };
        let loss_d = match dispatch_grad_graph(
            "grad_graph_cross_entropy",
            &[Value::Int(lg_d), Value::Int(tg_d)],
        ).unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() };
        let dispatched = forward_to_vec(loss_d);

        prop_assert_eq!(
            direct.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            dispatched.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn prop_layer_norm_dispatch_eq_direct(seed in any::<u64>(), n in 2usize..=32) {
        let data = det_floats(seed, n);

        let mut g = GradGraph::new();
        let inp = g.input(Tensor::from_vec(data.clone(), &[n]).unwrap());
        let out = g.layer_norm(inp);
        let direct = g.tensor(out).to_vec();

        let inp_d = fresh_input(data, &[n]);
        let out_d = match dispatch_grad_graph("grad_graph_layer_norm", &[Value::Int(inp_d)])
            .unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() };
        let dispatched = forward_to_vec(out_d);

        prop_assert_eq!(
            direct.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            dispatched.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn prop_gelu_dispatch_eq_direct(seed in any::<u64>(), n in 1usize..=16) {
        let data = det_floats(seed, n);

        let mut g = GradGraph::new();
        let inp = g.input(Tensor::from_vec(data.clone(), &[n]).unwrap());
        let out = g.gelu(inp);
        let direct = g.tensor(out).to_vec();

        let inp_d = fresh_input(data, &[n]);
        let out_d = match dispatch_grad_graph("grad_graph_gelu", &[Value::Int(inp_d)])
            .unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() };
        let dispatched = forward_to_vec(out_d);

        prop_assert_eq!(
            direct.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            dispatched.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn prop_silu_dispatch_eq_direct(seed in any::<u64>(), n in 1usize..=16) {
        let data = det_floats(seed, n);

        let mut g = GradGraph::new();
        let inp = g.input(Tensor::from_vec(data.clone(), &[n]).unwrap());
        let out = g.silu(inp);
        let direct = g.tensor(out).to_vec();

        let inp_d = fresh_input(data, &[n]);
        let out_d = match dispatch_grad_graph("grad_graph_silu", &[Value::Int(inp_d)])
            .unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() };
        let dispatched = forward_to_vec(out_d);

        prop_assert_eq!(
            direct.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            dispatched.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn prop_reshape_preserves_data_in_dispatch(
        seed in any::<u64>(),
        rows in 1usize..=8, cols in 1usize..=8,
    ) {
        let n = rows * cols;
        let data = det_floats(seed, n);

        // Direct: reshape from [rows, cols] to [n].
        let mut g = GradGraph::new();
        let inp = g.input(Tensor::from_vec(data.clone(), &[rows, cols]).unwrap());
        let out = g.reshape(inp, &[n]);
        let direct = g.tensor(out).to_vec();

        // Dispatch.
        let inp_d = fresh_input(data, &[rows, cols]);
        let shape_arr = Value::Array(std::rc::Rc::new(vec![Value::Int(n as i64)]));
        let out_d = match dispatch_grad_graph(
            "grad_graph_reshape",
            &[Value::Int(inp_d), shape_arr],
        ).unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() };
        let dispatched = forward_to_vec(out_d);

        prop_assert_eq!(
            direct.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            dispatched.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        );
    }
}
