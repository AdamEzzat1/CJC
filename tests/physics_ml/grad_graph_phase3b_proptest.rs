//! Phase 3b — proptests for the array-arg & state-recovery `grad_graph_*`
//! wrappers. Same wrapper-preserves-values invariant as Phase 3a: the
//! dispatch path output bits must equal the direct `GradGraph` Rust call
//! output bits.
//!
//! Five properties: batch_norm, gather, cat, reforward, backward_collect.
//! Default 256 cases each.

use proptest::prelude::*;

use cjc_ad::dispatch_grad_graph;
use cjc_ad::GradGraph;
use cjc_runtime::tensor::Tensor;
use cjc_runtime::value::Value;

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
            let bits = splitmix64(&mut s);
            (bits as f64 / u64::MAX as f64) * 4.0 - 2.0
        })
        .collect()
}

fn fresh_input(data: Vec<f64>, shape: &[usize]) -> i64 {
    dispatch_grad_graph("grad_graph_new", &[]).unwrap();
    match dispatch_grad_graph(
        "grad_graph_input",
        &[Value::Tensor(Tensor::from_vec(data, shape).unwrap())],
    )
    .unwrap()
    .unwrap()
    {
        Value::Int(i) => i,
        _ => unreachable!(),
    }
}

fn forward_to_vec(idx: i64) -> Vec<f64> {
    match dispatch_grad_graph("grad_graph_forward", &[Value::Int(idx)])
        .unwrap()
        .unwrap()
    {
        Value::Tensor(t) => t.to_vec(),
        _ => unreachable!(),
    }
}

fn int_array(values: &[i64]) -> Value {
    Value::Array(std::rc::Rc::new(values.iter().map(|&i| Value::Int(i)).collect()))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn prop_batch_norm_dispatch_eq_direct(seed in any::<u64>(), n in 2usize..=32) {
        let data = det_floats(seed, n);

        let mut g = GradGraph::new();
        let inp = g.input(Tensor::from_vec(data.clone(), &[n]).unwrap());
        let out = g.batch_norm(inp);
        let direct = g.tensor(out).to_vec();

        let inp_d = fresh_input(data, &[n]);
        let out_d = match dispatch_grad_graph("grad_graph_batch_norm", &[Value::Int(inp_d)])
            .unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() };
        let dispatched = forward_to_vec(out_d);

        prop_assert_eq!(
            direct.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            dispatched.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn prop_gather_1d_dispatch_eq_direct(
        seed in any::<u64>(),
        n in 2usize..=16,
        k in 1usize..=8,
    ) {
        let data = det_floats(seed, n);
        // Pick `k` indices in [0, n) deterministically from the seed.
        let mut s = seed.wrapping_add(0xA5A5);
        let indices: Vec<usize> = (0..k).map(|_| (splitmix64(&mut s) as usize) % n).collect();

        let mut g = GradGraph::new();
        let inp = g.input(Tensor::from_vec(data.clone(), &[n]).unwrap());
        let out = g.gather(inp, &indices, 0);
        let direct = g.tensor(out).to_vec();

        let inp_d = fresh_input(data, &[n]);
        let idx_arr = int_array(&indices.iter().map(|&i| i as i64).collect::<Vec<_>>());
        let out_d = match dispatch_grad_graph(
            "grad_graph_gather",
            &[Value::Int(inp_d), idx_arr, Value::Int(0)],
        )
        .unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() };
        let dispatched = forward_to_vec(out_d);

        prop_assert_eq!(
            direct.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            dispatched.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn prop_cat_1d_dispatch_eq_direct(
        seed_a in any::<u64>(),
        seed_b in any::<u64>(),
        na in 1usize..=8, nb in 1usize..=8,
    ) {
        let data_a = det_floats(seed_a, na);
        let data_b = det_floats(seed_b, nb);

        let mut g = GradGraph::new();
        let a = g.input(Tensor::from_vec(data_a.clone(), &[na]).unwrap());
        let b = g.input(Tensor::from_vec(data_b.clone(), &[nb]).unwrap());
        let out = g.cat(&[a, b], 0);
        let direct = g.tensor(out).to_vec();

        // Dispatch.
        dispatch_grad_graph("grad_graph_new", &[]).unwrap();
        let a_d = match dispatch_grad_graph(
            "grad_graph_input",
            &[Value::Tensor(Tensor::from_vec(data_a, &[na]).unwrap())],
        ).unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() };
        let b_d = match dispatch_grad_graph(
            "grad_graph_input",
            &[Value::Tensor(Tensor::from_vec(data_b, &[nb]).unwrap())],
        ).unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() };
        let inputs = int_array(&[a_d, b_d]);
        let out_d = match dispatch_grad_graph(
            "grad_graph_cat",
            &[inputs, Value::Int(0)],
        ).unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() };
        let dispatched = forward_to_vec(out_d);

        prop_assert_eq!(
            direct.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            dispatched.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn prop_reforward_after_set_tensor_eq_direct(
        seed in any::<u64>(),
    ) {
        // Build x*y, set_tensor(x), reforward — direct vs dispatch.
        let x_init = det_floats(seed, 3);
        let y_init = det_floats(seed.wrapping_add(1), 3);
        let x_new = det_floats(seed.wrapping_add(2), 3);

        // Direct.
        let mut g = GradGraph::new();
        let x = g.parameter(Tensor::from_vec(x_init.clone(), &[3]).unwrap());
        let y = g.parameter(Tensor::from_vec(y_init.clone(), &[3]).unwrap());
        let xy = g.mul(x, y);
        g.set_tensor(x, Tensor::from_vec(x_new.clone(), &[3]).unwrap());
        g.reforward(xy, xy);
        let direct = g.tensor(xy).to_vec();

        // Dispatch.
        dispatch_grad_graph("grad_graph_new", &[]).unwrap();
        let x_d = match dispatch_grad_graph(
            "grad_graph_param",
            &[Value::Tensor(Tensor::from_vec(x_init, &[3]).unwrap())],
        ).unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() };
        let y_d = match dispatch_grad_graph(
            "grad_graph_param",
            &[Value::Tensor(Tensor::from_vec(y_init, &[3]).unwrap())],
        ).unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() };
        let xy_d = match dispatch_grad_graph(
            "grad_graph_mul",
            &[Value::Int(x_d), Value::Int(y_d)],
        ).unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() };
        dispatch_grad_graph(
            "grad_graph_set_tensor",
            &[
                Value::Int(x_d),
                Value::Tensor(Tensor::from_vec(x_new, &[3]).unwrap()),
            ],
        ).unwrap();
        dispatch_grad_graph(
            "grad_graph_reforward",
            &[Value::Int(xy_d), Value::Int(xy_d)],
        ).unwrap();
        let dispatched = forward_to_vec(xy_d);

        prop_assert_eq!(
            direct.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            dispatched.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn prop_backward_collect_dispatch_eq_direct(
        seed_x in any::<u64>(),
        seed_y in any::<u64>(),
        n in 1usize..=8,
    ) {
        let x_data = det_floats(seed_x, n);
        let y_data = det_floats(seed_y, n);

        // Direct: loss = sum(x*y).
        let mut g = GradGraph::new();
        let x = g.parameter(Tensor::from_vec(x_data.clone(), &[n]).unwrap());
        let y = g.parameter(Tensor::from_vec(y_data.clone(), &[n]).unwrap());
        let xy = g.mul(x, y);
        let loss = g.sum(xy);
        let grads = g.backward_collect(loss, &[x, y]);
        let direct_x = grads[0].as_ref().unwrap().to_vec();
        let direct_y = grads[1].as_ref().unwrap().to_vec();

        // Dispatch.
        dispatch_grad_graph("grad_graph_new", &[]).unwrap();
        let xd = match dispatch_grad_graph(
            "grad_graph_param",
            &[Value::Tensor(Tensor::from_vec(x_data, &[n]).unwrap())],
        ).unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() };
        let yd = match dispatch_grad_graph(
            "grad_graph_param",
            &[Value::Tensor(Tensor::from_vec(y_data, &[n]).unwrap())],
        ).unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() };
        let xyd = match dispatch_grad_graph(
            "grad_graph_mul",
            &[Value::Int(xd), Value::Int(yd)],
        ).unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() };
        let lossd = match dispatch_grad_graph("grad_graph_sum", &[Value::Int(xyd)])
            .unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() };
        let arr = match dispatch_grad_graph(
            "grad_graph_backward_collect",
            &[Value::Int(lossd), int_array(&[xd, yd])],
        ).unwrap().unwrap() {
            Value::Array(rc) => rc,
            _ => unreachable!(),
        };
        let disp_x = match &arr[0] {
            Value::Tensor(t) => t.to_vec(),
            _ => unreachable!(),
        };
        let disp_y = match &arr[1] {
            Value::Tensor(t) => t.to_vec(),
            _ => unreachable!(),
        };

        prop_assert_eq!(
            direct_x.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            disp_x.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        );
        prop_assert_eq!(
            direct_y.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            disp_y.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        );
    }
}
