//! Regression tests for two GradGraph bugs:
//!
//! * P1 — `hessian_diag` (and `hessian` / `double_backward`) used stale
//!   intermediate forward values after perturbing the parameter, and
//!   `reforward` silently kept stale values for ops it did not handle.
//! * P2 — the backward pass did not reduce broadcast gradients back to the
//!   input's shape (`sum(x[2,3] + b[3])` gave `grad(b)` of shape `[2,3]`).
//!
//! Every expected value here comes from an independent oracle: closed-form
//! derivatives, or finite differences of a plain-f64 re-implementation of the
//! loss that does not touch `Tensor` or `GradGraph`.

use cjc_ad::{GradGraph, GradOp};
use cjc_runtime::Tensor;

// ── helpers ────────────────────────────────────────────────────────────

fn t(data: &[f64], shape: &[usize]) -> Tensor {
    Tensor::from_vec_unchecked(data.to_vec(), shape)
}

fn hd(g: &mut GradGraph, loss: usize, p: usize, eps: f64) -> Vec<f64> {
    g.hessian_diag(loss, p, eps).expect("hessian_diag").to_vec()
}

fn close(got: f64, want: f64, tol: f64) -> bool {
    (got - want).abs() <= tol * (1.0 + want.abs())
}

fn assert_vec_close(label: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length {} vs {}", got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(close(g, w, tol), "{label}[{i}]: got {g}, want {w} (all got {got:?}, want {want:?})");
    }
}

/// Second central difference of a plain-f64 function, per coordinate.
fn fd_hess_diag(f: &dyn Fn(&[f64]) -> f64, p: &[f64]) -> Vec<f64> {
    let h = 1e-4;
    (0..p.len())
        .map(|i| {
            let mut pp = p.to_vec();
            pp[i] += h;
            let mut pm = p.to_vec();
            pm[i] -= h;
            (f(&pp) - 2.0 * f(p) + f(&pm)) / (h * h)
        })
        .collect()
}

// ── P1: hessian_diag on composed losses ───────────────────────────────

#[test]
fn hessian_diag_p4_composed() {
    // loss = (p*p)*(p*p) = p^4, d2/dp2 = 12 p^2
    for &p0 in &[1.0_f64, 1.5, -0.7] {
        let mut g = GradGraph::new();
        let p = g.parameter(t(&[p0], &[1]));
        let y = g.mul(p, p);
        let loss = g.mul(y, y);
        g.backward(loss);
        let h = hd(&mut g, loss, p, 1e-5);
        assert_vec_close("p^4", &h, &[12.0 * p0 * p0], 1e-6);
    }
}

#[test]
fn hessian_diag_exp_sin_vector() {
    // loss = sum(exp(sin(p))), d2/dp_i2 = e^{sin p}(cos^2 p - sin p)
    let p0 = [0.3, 0.7, -1.2];
    let mut g = GradGraph::new();
    let p = g.parameter(t(&p0, &[3]));
    let s = g.sin(p);
    let e = g.exp(s);
    let loss = g.sum(e);
    let h = hd(&mut g, loss, p, 1e-5);
    let want: Vec<f64> = p0
        .iter()
        .map(|&x| x.sin().exp() * (x.cos() * x.cos() - x.sin()))
        .collect();
    assert_vec_close("exp(sin(p))", &h, &want, 1e-6);
}

#[test]
fn hessian_diag_chain_with_intermediate() {
    // u = p*w + sin(p) (intermediate used twice), loss = sum(u*u)
    // dL/dp = 2u(w + cos p); d2L/dp2 = 2(w + cos p)^2 - 2u sin p
    let p0 = [0.4, -0.9];
    let w0 = [2.0, -3.0];
    let mut g = GradGraph::new();
    let p = g.parameter(t(&p0, &[2]));
    let w = g.input(t(&w0, &[2]));
    let pw = g.mul(p, w);
    let sp = g.sin(p);
    let u = g.add(pw, sp);
    let uu = g.mul(u, u);
    let loss = g.sum(uu);
    let h = hd(&mut g, loss, p, 1e-5);
    let want: Vec<f64> = p0
        .iter()
        .zip(w0.iter())
        .map(|(&x, &wi)| {
            let u = x * wi + x.sin();
            2.0 * (wi + x.cos()).powi(2) - 2.0 * u * x.sin()
        })
        .collect();
    assert_vec_close("chain", &h, &want, 1e-6);
}

#[test]
fn hessian_diag_restores_forward_state() {
    // After hessian_diag the graph's forward values and the parameter must
    // be exactly what they were before.
    let mut g = GradGraph::new();
    let p = g.parameter(t(&[1.25, -0.5], &[2]));
    let y = g.mul(p, p);
    let z = g.sin(y);
    let loss = g.sum(z);
    let before: Vec<Vec<u64>> = (0..=loss)
        .map(|i| g.tensor(i).to_vec().iter().map(|v| v.to_bits()).collect())
        .collect();
    let _ = hd(&mut g, loss, p, 1e-5);
    let after: Vec<Vec<u64>> = (0..=loss)
        .map(|i| g.tensor(i).to_vec().iter().map(|v| v.to_bits()).collect())
        .collect();
    assert_eq!(before, after);
}

/// For each op that the old `reforward` silently skipped, build
/// `loss = sum(w * op(...)^2)` and compare hessian_diag with a finite
/// difference of an independent plain-f64 implementation.
fn weighted_sq_loss(g: &mut GradGraph, y: usize, w: &[f64]) -> usize {
    let wn = g.input(t(w, &[w.len()]));
    let yy = g.mul(y, y);
    let yw = g.mul(yy, wn);
    g.sum(yw)
}

fn plain_softmax(v: &[f64]) -> Vec<f64> {
    let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = v.iter().map(|x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

fn plain_norm(v: &[f64]) -> Vec<f64> {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    let sd = (var + 1e-5).sqrt();
    v.iter().map(|x| (x - mean) / sd).collect()
}

fn wsq(y: &[f64], w: &[f64]) -> f64 {
    y.iter().zip(w).map(|(a, b)| b * a * a).sum()
}

#[test]
fn hessian_diag_log2() {
    let p0: [f64; 3] = [0.8, 1.7, 3.1];
    let w0 = [1.0, -2.0, 0.5];
    let mut g = GradGraph::new();
    let p = g.parameter(t(&p0, &[3]));
    let y = g.log2(p);
    let loss = weighted_sq_loss(&mut g, y, &w0);
    let h = hd(&mut g, loss, p, 1e-5);
    // d2/dp2 (log2 p)^2 = 2(1 - ln p) / (p^2 ln^2 2)
    let l2 = std::f64::consts::LN_2;
    let want: Vec<f64> = p0
        .iter()
        .zip(&w0)
        .map(|(&x, &wi)| wi * 2.0 * (1.0 - x.ln()) / (x * x * l2 * l2))
        .collect();
    assert_vec_close("log2", &h, &want, 1e-6);
}

#[test]
fn hessian_diag_softmax() {
    let p0 = [0.2, -0.4, 1.1];
    let w0 = [1.0, 2.0, -1.5];
    let mut g = GradGraph::new();
    let p = g.parameter(t(&p0, &[3]));
    let y = g.softmax(p);
    let loss = weighted_sq_loss(&mut g, y, &w0);
    let h = hd(&mut g, loss, p, 1e-5);
    let f = |v: &[f64]| wsq(&plain_softmax(v), &w0);
    assert_vec_close("softmax", &h, &fd_hess_diag(&f, &p0), 1e-5);
}

#[test]
fn hessian_diag_cross_entropy() {
    let p0 = [0.2, -0.4, 1.1];
    let t0 = [0.0, 1.0, 0.0];
    let mut g = GradGraph::new();
    let p = g.parameter(t(&p0, &[3]));
    let tg = g.input(t(&t0, &[3]));
    let loss = g.cross_entropy(p, tg);
    let h = hd(&mut g, loss, p, 1e-5);
    // d2/dp_i2 CE = s_i (1 - s_i)
    let s = plain_softmax(&p0);
    let want: Vec<f64> = s.iter().map(|&si| si * (1.0 - si)).collect();
    assert_vec_close("cross_entropy", &h, &want, 1e-6);
}

#[test]
fn hessian_diag_layer_norm_and_batch_norm() {
    let p0 = [0.2, -0.4, 1.1, 0.5];
    let w0 = [1.0, 2.0, -1.5, 0.25];
    for which in 0..2 {
        let mut g = GradGraph::new();
        let p = g.parameter(t(&p0, &[4]));
        let y = if which == 0 { g.layer_norm(p) } else { g.batch_norm(p) };
        let loss = weighted_sq_loss(&mut g, y, &w0);
        let h = hd(&mut g, loss, p, 1e-5);
        let f = |v: &[f64]| wsq(&plain_norm(v), &w0);
        assert_vec_close(
            if which == 0 { "layer_norm" } else { "batch_norm" },
            &h,
            &fd_hess_diag(&f, &p0),
            1e-5,
        );
    }
}

#[test]
fn hessian_diag_where_cat_gather() {
    let p0 = [0.3, -0.8, 1.4];
    // where(c, sin p, p*p)
    {
        let c0 = [1.0, 0.0, 1.0];
        let w0 = [1.0, -2.0, 0.5];
        let mut g = GradGraph::new();
        let p = g.parameter(t(&p0, &[3]));
        let c = g.input(t(&c0, &[3]));
        let sp = g.sin(p);
        let pp = g.mul(p, p);
        let y = g.where_cond(c, sp, pp);
        let loss = weighted_sq_loss(&mut g, y, &w0);
        let h = hd(&mut g, loss, p, 1e-5);
        let f = |v: &[f64]| {
            let y: Vec<f64> = v
                .iter()
                .zip(&c0)
                .map(|(&x, &ci)| if ci != 0.0 { x.sin() } else { x * x })
                .collect();
            wsq(&y, &w0)
        };
        assert_vec_close("where", &h, &fd_hess_diag(&f, &p0), 1e-5);
    }
    // cat([sin p, p*p], 0)
    {
        let w0 = [1.0, -2.0, 0.5, 0.7, 1.3, -0.4];
        let mut g = GradGraph::new();
        let p = g.parameter(t(&p0, &[3]));
        let sp = g.sin(p);
        let pp = g.mul(p, p);
        let y = g.cat(&[sp, pp], 0);
        let loss = weighted_sq_loss(&mut g, y, &w0);
        let h = hd(&mut g, loss, p, 1e-5);
        let f = |v: &[f64]| {
            let mut y: Vec<f64> = v.iter().map(|x| x.sin()).collect();
            y.extend(v.iter().map(|x| x * x));
            wsq(&y, &w0)
        };
        assert_vec_close("cat", &h, &fd_hess_diag(&f, &p0), 1e-5);
    }
    // gather(sin p, [2, 0, 0, 1])
    {
        let idx = [2usize, 0, 0, 1];
        let w0 = [1.0, -2.0, 0.5, 0.7];
        let mut g = GradGraph::new();
        let p = g.parameter(t(&p0, &[3]));
        let sp = g.sin(p);
        let y = g.gather(sp, &idx, 0);
        let loss = weighted_sq_loss(&mut g, y, &w0);
        let h = hd(&mut g, loss, p, 1e-5);
        let f = |v: &[f64]| {
            let y: Vec<f64> = idx.iter().map(|&i| v[i].sin()).collect();
            wsq(&y, &w0)
        };
        assert_vec_close("gather", &h, &fd_hess_diag(&f, &p0), 1e-5);
    }
}

#[test]
fn hessian_and_double_backward_log2_chain() {
    // loss = sum(w * log2(p)^2): separable, so the Hessian is diagonal with
    // H_ii = w_i * 2(1 - ln p_i) / (p_i^2 ln^2 2).
    let p0: [f64; 2] = [0.8, 1.7];
    let w0 = [1.0, -2.0];
    let l2 = std::f64::consts::LN_2;
    let want: Vec<f64> = p0
        .iter()
        .zip(&w0)
        .map(|(&x, &wi)| wi * 2.0 * (1.0 - x.ln()) / (x * x * l2 * l2))
        .collect();

    let mut g = GradGraph::new();
    let p = g.parameter(t(&p0, &[2]));
    let y = g.log2(p);
    let loss = weighted_sq_loss(&mut g, y, &w0);
    let h = g.hessian(loss, p).expect("hessian").to_vec();
    assert_vec_close("hessian", &h, &[want[0], 0.0, 0.0, want[1]], 1e-5);

    let mut g = GradGraph::new();
    let p = g.parameter(t(&p0, &[2]));
    let y = g.log2(p);
    let loss = weighted_sq_loss(&mut g, y, &w0);
    let d = g.double_backward(loss, p).expect("double_backward").to_vec();
    assert_vec_close("double_backward", &d, &want, 1e-5);
}

#[test]
fn reforward_errors_on_unrecomputable_op() {
    // A StructField node's tensor is supplied from outside the graph, so
    // reforward has no rule to recompute it: it must report an error instead
    // of silently leaving the old value.
    let mut g = GradGraph::new();
    let p = g.parameter(t(&[2.0], &[1]));
    let sf = g.push_node(
        GradOp::StructField { parent: p, field_index: 0, total_fields: 1 },
        t(&[2.0], &[1]),
        None,
    );
    let y = g.mul(sf, sf);
    let loss = g.sum(y);
    assert!(g.reforward(sf, loss).is_err(), "reforward must reject StructField");
    let r = g.hessian_diag(loss, p, 1e-5);
    assert!(r.is_err(), "hessian_diag must propagate the reforward error");
    assert!(g.hessian(loss, p).is_err());
    assert!(g.double_backward(loss, p).is_err());
    // The parameter is restored even on the error path.
    assert_eq!(g.tensor(p).to_vec(), vec![2.0]);
}

#[test]
fn reforward_ignores_unrecomputable_op_not_downstream_of_param() {
    // hessian_diag only needs to recompute nodes that depend on the
    // perturbed parameter; an opaque node elsewhere in the graph is fine.
    let mut g = GradGraph::new();
    let q = g.parameter(t(&[3.0], &[1]));
    let sf = g.push_node(
        GradOp::StructField { parent: q, field_index: 0, total_fields: 1 },
        t(&[3.0], &[1]),
        None,
    );
    let p = g.parameter(t(&[1.0], &[1]));
    let y = g.mul(p, p);
    let y2 = g.mul(y, y);
    let z = g.mul(y2, sf);
    let loss = g.sum(z);
    let h = hd(&mut g, loss, p, 1e-5);
    assert_vec_close("p^4 * 3", &h, &[36.0], 1e-6);
}

// ── P2: broadcast gradients reduced to the input shape ────────────────

#[derive(Clone, Copy, Debug)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

fn apply(op: Op, a: f64, b: f64) -> f64 {
    match op {
        Op::Add => a + b,
        Op::Sub => a - b,
        Op::Mul => a * b,
        Op::Div => a / b,
    }
}

/// Plain-f64 broadcast read of a tensor of shape `shape` at output
/// position (r, c) of an [N, M] result.
fn bread(data: &[f64], shape: &[usize], r: usize, c: usize) -> f64 {
    match shape {
        [1] => data[0],
        [m] => data[c % m],
        [1, m] => data[c % m],
        [n, 1] => data[r % n],
        [_, m] => data[r * m + c],
        _ => unreachable!(),
    }
}

const N: usize = 2;
const M: usize = 3;

/// loss = sum(w * op(L, R)) computed with plain f64 loops.
fn plain_loss(op: Op, l: &[f64], ls: &[usize], r: &[f64], rs: &[usize], w: &[f64]) -> f64 {
    let mut acc = 0.0;
    for i in 0..N {
        for j in 0..M {
            acc += w[i * M + j] * apply(op, bread(l, ls, i, j), bread(r, rs, i, j));
        }
    }
    acc
}

fn fd_grad(f: &dyn Fn(&[f64]) -> f64, v: &[f64]) -> Vec<f64> {
    let h = 1e-6;
    (0..v.len())
        .map(|i| {
            let mut a = v.to_vec();
            a[i] += h;
            let mut b = v.to_vec();
            b[i] -= h;
            (f(&a) - f(&b)) / (2.0 * h)
        })
        .collect()
}

fn small_shapes() -> Vec<Vec<usize>> {
    vec![vec![M], vec![1, M], vec![N, 1], vec![1]]
}

fn data_for(shape: &[usize], base: f64) -> Vec<f64> {
    let n: usize = shape.iter().product();
    (0..n).map(|k| base + 0.37 * k as f64).collect()
}

/// Returns a list of failure descriptions (empty = pass) so a red run shows
/// every broken combination at once.
fn check_broadcast_op(op: Op) -> Vec<String> {
    let mut failures = Vec::new();
    let big_shape = vec![N, M];
    let big = data_for(&big_shape, 1.1);
    let w: Vec<f64> = (0..N * M).map(|k| 0.5 + 0.25 * k as f64).collect();
    for small_shape in small_shapes() {
        let small = data_for(&small_shape, 1.7);
        for big_left in [true, false] {
            let (ld, ls, rd, rs) = if big_left {
                (big.clone(), big_shape.clone(), small.clone(), small_shape.clone())
            } else {
                (small.clone(), small_shape.clone(), big.clone(), big_shape.clone())
            };
            let label = format!("{op:?} {ls:?} op {rs:?}");

            let mut g = GradGraph::new();
            let l = g.parameter(t(&ld, &ls));
            let r = g.parameter(t(&rd, &rs));
            let y = match op {
                Op::Add => g.add(l, r),
                Op::Sub => g.sub(l, r),
                Op::Mul => g.mul(l, r),
                Op::Div => g.div(l, r),
            };
            let wn = g.input(t(&w, &[N, M]));
            let yw = g.mul(y, wn);
            let loss = g.sum(yw);
            g.backward(loss);

            let gl = g.grad(l).expect("grad l");
            let gr = g.grad(r).expect("grad r");
            if gl.shape() != ls.as_slice() {
                failures.push(format!("{label}: grad(left) shape {:?}, want {ls:?}", gl.shape()));
            }
            if gr.shape() != rs.as_slice() {
                failures.push(format!("{label}: grad(right) shape {:?}, want {rs:?}", gr.shape()));
            }
            let fl = |v: &[f64]| plain_loss(op, v, &ls, &rd, &rs, &w);
            let fr = |v: &[f64]| plain_loss(op, &ld, &ls, v, &rs, &w);
            let want_l = fd_grad(&fl, &ld);
            let want_r = fd_grad(&fr, &rd);
            for (side, got, want) in [("left", gl.to_vec(), want_l), ("right", gr.to_vec(), want_r)] {
                if got.len() != want.len()
                    || got.iter().zip(&want).any(|(&a, &b)| !close(a, b, 1e-6))
                {
                    failures.push(format!("{label}: grad({side}) = {got:?}, want {want:?}"));
                }
            }
        }
    }
    failures
}

#[test]
fn broadcast_grad_add() {
    let f = check_broadcast_op(Op::Add);
    assert!(f.is_empty(), "{} failures:\n{}", f.len(), f.join("\n"));
}

#[test]
fn broadcast_grad_sub() {
    let f = check_broadcast_op(Op::Sub);
    assert!(f.is_empty(), "{} failures:\n{}", f.len(), f.join("\n"));
}

#[test]
fn broadcast_grad_mul() {
    let f = check_broadcast_op(Op::Mul);
    assert!(f.is_empty(), "{} failures:\n{}", f.len(), f.join("\n"));
}

#[test]
fn broadcast_grad_div() {
    let f = check_broadcast_op(Op::Div);
    assert!(f.is_empty(), "{} failures:\n{}", f.len(), f.join("\n"));
}

#[test]
fn broadcast_grad_jacobian_backward_with_seed() {
    // y = x[2,3] + b[3]; dy_{r,c}/db_j = [c == j]. jacobian() goes through
    // backward_with_seed, a separate backward implementation.
    let mut g = GradGraph::new();
    let x = g.input(t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let b = g.parameter(t(&[0.0, 0.0, 0.0], &[3]));
    let y = g.add(x, b);
    let jac = g.jacobian(y, b);
    assert_eq!(jac.shape(), &[6, 3]);
    let mut want = vec![0.0; 18];
    for row in 0..6 {
        want[row * 3 + row % 3] = 1.0;
    }
    assert_eq!(jac.to_vec(), want);
}

#[test]
fn broadcast_grad_mlp_layer_bias_not_reduced_twice() {
    // The fused MlpLayer already sums its bias gradient over the batch.
    // loss = sum(input @ W^T + b) with Activation::None:
    // dL/db_j = batch, dL/dW_jk = sum_r input_rk.
    let mut g = GradGraph::new();
    let input = g.input(t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]));
    let wt = g.parameter(t(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[3, 2]));
    let b = g.parameter(t(&[0.0, 0.0, 0.0], &[3]));
    let y = g.mlp_layer(input, wt, b, cjc_ad::pinn::Activation::None);
    let loss = g.sum(y);
    g.backward(loss);
    let gb = g.grad(b).unwrap();
    assert_eq!(gb.shape(), &[3]);
    assert_eq!(gb.to_vec(), vec![4.0, 4.0, 4.0]);
    let gw = g.grad(wt).unwrap();
    assert_eq!(gw.shape(), &[3, 2]);
    assert_eq!(gw.to_vec(), vec![16.0, 20.0, 16.0, 20.0, 16.0, 20.0]);
}

#[test]
fn broadcast_grad_shared_input_accumulates() {
    // b[3] is used twice: once broadcast against [2,3] and once same-shape.
    // loss = sum(x + b) + sum(b * b) -> dL/db = 2 + 2b.
    let mut g = GradGraph::new();
    let x = g.input(t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let b = g.parameter(t(&[0.5, -1.0, 2.0], &[3]));
    let s = g.add(x, b);
    let l1 = g.sum(s);
    let bb = g.mul(b, b);
    let l2 = g.sum(bb);
    let loss = g.add(l1, l2);
    g.backward(loss);
    let gb = g.grad(b).unwrap();
    assert_eq!(gb.shape(), &[3]);
    assert_eq!(gb.to_vec(), vec![3.0, 0.0, 6.0]);
}

#[test]
fn broadcast_grad_deterministic_bits() {
    let run = || {
        let mut g = GradGraph::new();
        let x = g.parameter(t(&(0..60).map(|k| (k as f64 * 0.731).sin()).collect::<Vec<_>>(), &[20, 3]));
        let b = g.parameter(t(&[0.1, 1e-9, 3.3], &[1, 3]));
        let c = g.parameter(t(&(0..20).map(|k| 1.0 + k as f64 * 1e-3).collect::<Vec<_>>(), &[20, 1]));
        let y = g.mul(x, b);
        let z = g.div(y, c);
        let loss = g.sum(z);
        g.backward(loss);
        let mut bits: Vec<u64> = Vec::new();
        for id in [x, b, c] {
            bits.extend(g.grad(id).unwrap().to_vec().iter().map(|v| v.to_bits()));
        }
        bits
    };
    let first = run();
    for _ in 0..5 {
        assert_eq!(run(), first);
    }
}

// ── unbroadcast helper ────────────────────────────────────────────────

#[test]
fn unbroadcast_reduces_each_axis_kind() {
    // g[r, c] = 10 r + c over shape [2, 3]
    let g = t(&[0.0, 1.0, 2.0, 10.0, 11.0, 12.0], &[2, 3]);
    let r = cjc_ad::unbroadcast(&g, &[3]);
    assert_eq!((r.shape().to_vec(), r.to_vec()), (vec![3], vec![10.0, 12.0, 14.0]));
    let r = cjc_ad::unbroadcast(&g, &[1, 3]);
    assert_eq!((r.shape().to_vec(), r.to_vec()), (vec![1, 3], vec![10.0, 12.0, 14.0]));
    let r = cjc_ad::unbroadcast(&g, &[2, 1]);
    assert_eq!((r.shape().to_vec(), r.to_vec()), (vec![2, 1], vec![3.0, 33.0]));
    let r = cjc_ad::unbroadcast(&g, &[1]);
    assert_eq!((r.shape().to_vec(), r.to_vec()), (vec![1], vec![36.0]));
    let r = cjc_ad::unbroadcast(&g, &[1, 1]);
    assert_eq!((r.shape().to_vec(), r.to_vec()), (vec![1, 1], vec![36.0]));
    let r = cjc_ad::unbroadcast(&g, &[2, 3]);
    assert_eq!(r.to_vec(), g.to_vec());
    // rank-3: [2, 2, 3] -> [2, 1, 3]
    let g3 = t(&(0..12).map(|k| k as f64).collect::<Vec<_>>(), &[2, 2, 3]);
    let r = cjc_ad::unbroadcast(&g3, &[2, 1, 3]);
    assert_eq!((r.shape().to_vec(), r.to_vec()), (vec![2, 1, 3], vec![3.0, 5.0, 7.0, 15.0, 17.0, 19.0]));
    // smaller-rank gradient into a size-1-padded target is a reshape
    let r = cjc_ad::unbroadcast(&t(&[1.0, 2.0, 3.0], &[3]), &[1, 3]);
    assert_eq!((r.shape().to_vec(), r.to_vec()), (vec![1, 3], vec![1.0, 2.0, 3.0]));
}

#[test]
fn unbroadcast_is_compensated() {
    // 1e16 + 1 + 1 + ... (naive left-to-right summation loses every 1.0).
    let mut data = vec![1e16];
    data.extend(std::iter::repeat(1.0).take(99));
    let g = t(&data, &[100, 1]);
    let r = cjc_ad::unbroadcast(&g, &[1]);
    assert_eq!(r.to_vec(), vec![1e16 + 99.0]);
}

#[test]
#[should_panic(expected = "unbroadcast")]
fn unbroadcast_rejects_incompatible_shape() {
    let g = t(&[0.0; 6], &[2, 3]);
    let _ = cjc_ad::unbroadcast(&g, &[2]);
}
