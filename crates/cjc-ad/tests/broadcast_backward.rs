//! Reverse-mode gradients through broadcasting element-wise ops.
//!
//! `add`/`sub`/`mul`/`div` broadcast their operands (NumPy rules), so the
//! gradient reaching a broadcast operand must be summed back to its shape.
//! Before this was done, `x @ W + b` with a `[1, n]` bias panicked in
//! `backward` (`Tensor::add_assign shape mismatch`), which is what
//! `examples/01_mlp_xor.cjcl` hit.
//!
//! Each case is checked against central finite differences, in both
//! `backward` and `backward_with_seed`, and for bit-identical repeat runs.

use cjc_ad::{reduce_to_shape, GradGraph, GradOp};
use cjc_runtime::Tensor;

#[derive(Clone, Copy, Debug)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
}
const OPS: [Op; 4] = [Op::Add, Op::Sub, Op::Mul, Op::Div];

/// (a shape, b shape): broadcast on either side, by a missing leading axis,
/// by size-1 axes, and to a scalar-shaped `[1, 1]`.
const SHAPES: &[(&[usize], &[usize])] = &[
    (&[4, 3], &[1, 3]),
    (&[1, 3], &[4, 3]),
    (&[4, 3], &[3]),
    (&[4, 1], &[1, 1]),
    (&[2, 3, 4], &[3, 1]),
    (&[3, 1], &[1, 4]), // both sides broadcast
    (&[2, 3], &[2, 3]), // same shape: the unchanged path
];

fn data(shape: &[usize], salt: f64) -> Vec<f64> {
    let n: usize = shape.iter().product();
    // Deterministic, non-uniform, bounded away from 0 (for `div`).
    (0..n).map(|i| 1.5 + (0.7 * i as f64 + salt).sin()).collect()
}

/// loss = sum(op(a, b) * w), with `w` fixed and non-uniform, so every output
/// element's gradient differs.
fn build(op: Op, a: &[f64], sa: &[usize], b: &[f64], sb: &[usize]) -> (GradGraph, usize, usize, usize) {
    let mut g = GradGraph::new();
    let ai = g.parameter(Tensor::from_vec(a.to_vec(), sa).unwrap());
    let bi = g.parameter(Tensor::from_vec(b.to_vec(), sb).unwrap());
    let y = match op {
        Op::Add => g.add(ai, bi),
        Op::Sub => g.sub(ai, bi),
        Op::Mul => g.mul(ai, bi),
        Op::Div => g.div(ai, bi),
    };
    let ys = g.tensor(y).shape().to_vec();
    let wi = g.input(Tensor::from_vec(data(&ys, 3.0), &ys).unwrap());
    let yw = g.mul(y, wi);
    let loss = g.sum(yw);
    (g, ai, bi, loss)
}

fn loss_at(op: Op, a: &[f64], sa: &[usize], b: &[f64], sb: &[usize]) -> f64 {
    let (g, _, _, loss) = build(op, a, sa, b, sb);
    g.value(loss)
}

fn finite_diff(op: Op, a: &[f64], sa: &[usize], b: &[f64], sb: &[usize], wrt_a: bool) -> Vec<f64> {
    let h = 1e-6;
    let n = if wrt_a { a.len() } else { b.len() };
    (0..n)
        .map(|k| {
            let (mut ap, mut am, mut bp, mut bm) = (a.to_vec(), a.to_vec(), b.to_vec(), b.to_vec());
            if wrt_a {
                ap[k] += h;
                am[k] -= h;
            } else {
                bp[k] += h;
                bm[k] -= h;
            }
            (loss_at(op, &ap, sa, &bp, sb) - loss_at(op, &am, sa, &bm, sb)) / (2.0 * h)
        })
        .collect()
}

fn assert_close(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = 1e-5 * (1.0 + w.abs());
        assert!((g - w).abs() <= tol, "{what}[{k}]: analytic {g}, finite-difference {w}");
    }
}

#[test]
fn broadcast_gradients_match_finite_differences() {
    for op in OPS {
        for &(sa, sb) in SHAPES {
            let (a, b) = (data(sa, 0.0), data(sb, 1.0));
            let (mut g, ai, bi, loss) = build(op, &a, sa, &b, sb);
            g.backward(loss);
            let (ga, gb) = (g.grad(ai).unwrap(), g.grad(bi).unwrap());
            let what = format!("{op:?} {sa:?} {sb:?}");
            assert_eq!(ga.shape(), sa, "{what}: grad a shape");
            assert_eq!(gb.shape(), sb, "{what}: grad b shape");
            assert_close(&ga.to_vec(), &finite_diff(op, &a, sa, &b, sb, true), &format!("{what} d/da"));
            assert_close(&gb.to_vec(), &finite_diff(op, &a, sa, &b, sb, false), &format!("{what} d/db"));
        }
    }
}

#[test]
fn backward_with_seed_reduces_broadcast_gradients_too() {
    for op in OPS {
        for &(sa, sb) in SHAPES {
            let (a, b) = (data(sa, 0.0), data(sb, 1.0));
            let (mut g1, ai, bi, loss) = build(op, &a, sa, &b, sb);
            g1.backward(loss);
            let (mut g2, _, _, _) = build(op, &a, sa, &b, sb);
            g2.backward_with_seed(loss, &Tensor::ones(&[1]));
            for p in [ai, bi] {
                let (x, y) = (g1.grad(p).unwrap(), g2.grad(p).unwrap());
                assert_eq!(x.shape(), y.shape(), "{op:?} {sa:?} {sb:?} node {p}");
                assert_close(&y.to_vec(), &x.to_vec(), &format!("{op:?} {sa:?} {sb:?} seeded node {p}"));
            }
        }
    }
}

#[test]
fn broadcast_gradients_are_bit_identical_across_runs() {
    for op in OPS {
        for &(sa, sb) in SHAPES {
            let bits = || {
                let (a, b) = (data(sa, 0.0), data(sb, 1.0));
                let (mut g, _, _, loss) = build(op, &a, sa, &b, sb);
                g.backward(loss);
                g.state_bits()
            };
            assert_eq!(bits(), bits(), "{op:?} {sa:?} {sb:?}");
        }
    }
}

#[test]
fn a_bias_gradient_is_the_column_sum_of_the_upstream_gradient() {
    // The MLP case exactly: z = h + b, h [4, 1], b [1, 1]; d(sum z)/db = 4.
    let mut g = GradGraph::new();
    let h = g.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4, 1]).unwrap());
    let b = g.parameter(Tensor::from_vec(vec![0.5], &[1, 1]).unwrap());
    let z = g.add(h, b);
    let s = g.sum(z);
    g.backward(s);
    let gb = g.grad(b).unwrap();
    assert_eq!((gb.shape(), gb.to_vec()), (&[1usize, 1][..], vec![4.0]));
    // Accumulates over two backward passes, in the slot's own shape.
    g.backward(s);
    assert_eq!(g.grad(b).unwrap().to_vec(), vec![8.0]);
}

#[test]
fn reduce_to_shape_sums_broadcast_axes_only() {
    let t = Tensor::from_vec((1..=24).map(f64::from).collect(), &[2, 3, 4]).unwrap();
    assert_eq!(reduce_to_shape(&t, &[2, 3, 4]).to_vec(), t.to_vec());
    let r = reduce_to_shape(&t, &[3, 1]);
    assert_eq!(r.shape(), &[3, 1]);
    // row j sums t[i, j, k] over i and k
    let want: Vec<f64> = (0..3)
        .map(|j| (0..2).flat_map(|i| (0..4).map(move |k| (i * 12 + j * 4 + k + 1) as f64)).sum())
        .collect();
    assert_eq!(r.to_vec(), want);
}

#[test]
#[should_panic(expected = "is not a broadcast of")]
fn reduce_to_shape_rejects_a_shape_that_was_not_broadcast() {
    let t = Tensor::zeros(&[4, 3]);
    let _ = reduce_to_shape(&t, &[4, 2]);
}

#[test]
fn container_access_gradients_land_on_the_accessed_chunk() {
    let mut g = GradGraph::new();
    let parent = g.parameter(Tensor::from_vec(vec![3.0, 7.0], &[2]).unwrap());
    let f = g.push_node(
        GradOp::StructField { parent, field_index: 1, total_fields: 2 },
        Tensor::from_vec(vec![7.0], &[1]).unwrap(),
        None,
    );
    let map = g.parameter(Tensor::from_vec(vec![5.0, 10.0, 15.0], &[3]).unwrap());
    let m = g.push_node(
        GradOp::MapLookup { map_node: map, key_index: 1, total_keys: 3 },
        Tensor::from_vec(vec![10.0], &[1]).unwrap(),
        None,
    );
    let s = g.add(f, m);
    let loss = g.sum(s);
    g.backward(loss);
    assert_eq!(g.grad(parent).unwrap().to_vec(), vec![0.0, 1.0]);
    assert_eq!(g.grad(map).unwrap().to_vec(), vec![0.0, 1.0, 0.0]);
}
