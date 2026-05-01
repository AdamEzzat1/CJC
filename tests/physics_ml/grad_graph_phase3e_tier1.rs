//! Phase 3e Tier 1 — native higher-order autodiff for `Sum`, `Mean`,
//! and the new `BroadcastScalar` primitive.
//!
//! Tier 1 expands the polynomial subset (Phase 3d) to cover the
//! reductions used in PINN demos and any vector-loss training: `Sum`
//! and `Mean`, plus the helper op `BroadcastScalar` introduced to
//! express their gradient sub-graphs.
//!
//! `Sum` / `Mean` and `BroadcastScalar` are closed under
//! differentiation:
//!
//!   - d(Sum(a))/d(a_i) = upstream broadcast to a's shape  →  uses BroadcastScalar
//!   - d(BroadcastScalar(s, [n]))/d(s) = Σ upstream         →  uses Sum
//!   - d(Mean(a))/d(a_i) = (upstream/N) broadcast            →  uses ScalarMul + BroadcastScalar
//!
//! Adding all three together preserves the closure-under-
//! differentiation property the polynomial subset already had, so
//! second- and higher-order derivatives compose.
//!
//! ## Test categories
//!
//! 1. **Wiring (eval ↔ MIR byte-equal)** — `grad_graph_grad_of` over
//!    Sum/Mean snippets returns the same node, evaluated bit-equal
//!    across the two backends.
//! 2. **Analytic correctness (Rust-direct)** — closed-form derivatives
//!    of polynomial-vector expressions, e.g. d(Σ x_i²)/dx_i = 2x_i.
//! 3. **Second-order** — `d²(Σ x_i²)/dx_i² = 2` (constant).
//! 4. **Cross-check vs `backward()`** — gradients match the existing
//!    tensor-accumulating path bit-equal.
//! 5. **BroadcastScalar wiring** — verify the primitive itself is
//!    callable through the (Rust-side) GradGraph API and that its
//!    backward (sum of upstream) matches an analytic expectation.

#![allow(clippy::needless_raw_string_hashes)]

use cjc_ad::dispatch_grad_graph;
use cjc_ad::GradGraph;
use cjc_runtime::tensor::Tensor;
use cjc_runtime::value::Value;

#[derive(Clone, Copy, Debug)]
enum Backend {
    Eval,
    Mir,
}

fn run(backend: Backend, body: &str, seed: u64) -> Vec<String> {
    let src = format!("fn main() {{\n{body}\n}}\n");
    let (program, diags) = cjc_parser::parse_source(&src);
    assert!(
        !diags.has_errors(),
        "parse errors:\n{:#?}\nsource:\n{src}",
        diags.diagnostics,
    );
    match backend {
        Backend::Eval => {
            let mut interp = cjc_eval::Interpreter::new(seed);
            interp
                .exec(&program)
                .unwrap_or_else(|e| panic!("eval failed for snippet:\n{src}\nerror: {e:?}"));
            interp.output
        }
        Backend::Mir => {
            let (_v, exec) = cjc_mir_exec::run_program_with_executor(&program, seed)
                .unwrap_or_else(|e| panic!("MIR-exec failed for snippet:\n{src}\nerror: {e:?}"));
            exec.output
        }
    }
}

fn assert_parity(label: &str, body: &str) {
    let eval_out = run(Backend::Eval, body, 42);
    let mir_out = run(Backend::Mir, body, 42);
    assert_eq!(
        eval_out, mir_out,
        "[{label}] AST↔MIR parity violation\n  eval: {eval_out:?}\n  mir : {mir_out:?}",
    );
}

// ─── Wiring (eval ↔ MIR byte-equal) ────────────────────────────────

#[test]
fn wiring_grad_of_sum_yields_ones_broadcast() {
    // f = Σ x_i where x = [a, b, c]. df/dx_i = 1 for all i.
    assert_parity(
        "grad_of(sum(x), x) = ones",
        r#"
        grad_graph_new();
        let x = grad_graph_param(Tensor.from_vec([1.0, 2.0, 3.0], [3]));
        let s = grad_graph_sum(x);
        let dx = grad_graph_grad_of(s, x);
        print(grad_graph_forward(dx));
        "#,
    );
}

#[test]
fn wiring_grad_of_mean_yields_inv_n_broadcast() {
    // f = mean(x) where x = [a, b, c]. df/dx_i = 1/3 for all i.
    assert_parity(
        "grad_of(mean(x), x) = 1/n broadcast",
        r#"
        grad_graph_new();
        let x = grad_graph_param(Tensor.from_vec([1.0, 2.0, 3.0], [3]));
        let m = grad_graph_mean(x);
        let dx = grad_graph_grad_of(m, x);
        print(grad_graph_forward(dx));
        "#,
    );
}

#[test]
fn wiring_grad_of_sum_of_squares_yields_2x() {
    // f = Σ x_i². df/dx_i = 2 x_i.
    assert_parity(
        "grad_of(sum(x*x), x) = 2x",
        r#"
        grad_graph_new();
        let x = grad_graph_param(Tensor.from_vec([1.5, -0.5, 2.0], [3]));
        let xx = grad_graph_mul(x, x);
        let s = grad_graph_sum(xx);
        let dx = grad_graph_grad_of(s, x);
        print(grad_graph_forward(dx));
        "#,
    );
}

#[test]
fn wiring_second_order_d2_of_sum_x_squared_is_2_broadcast() {
    // d²(Σ x_i²)/dx² = 2 for each element. The killer test:
    // first-grad is "2x" (a function of x), and the second-grad should
    // evaluate to a vector of constants 2.0 — proving the gradient
    // sub-graph is itself differentiable.
    assert_parity(
        "grad_of(grad_of(sum(x*x), x), x) = 2-vector",
        r#"
        grad_graph_new();
        let x = grad_graph_param(Tensor.from_vec([0.7, 1.3, -0.4], [3]));
        let xx = grad_graph_mul(x, x);
        let s = grad_graph_sum(xx);
        let dx = grad_graph_grad_of(s, x);
        let d2x = grad_graph_grad_of(dx, x);
        print(grad_graph_forward(d2x));
        "#,
    );
}

// ─── Analytic correctness (Rust-direct) ────────────────────────────

fn forward_to_vec(g: &mut GradGraph, idx: usize) -> Vec<f64> {
    g.tensor(idx).to_vec()
}

#[test]
fn analytic_grad_of_sum_x_is_ones() {
    let mut g = GradGraph::new();
    let x = g.parameter(Tensor::from_vec(vec![5.0, -3.0, 7.5, 0.1], &[4]).unwrap());
    let s = g.sum(x);
    let dx = g.grad_of(s, x).unwrap();
    let actual = forward_to_vec(&mut g, dx);
    assert_eq!(actual, vec![1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn analytic_grad_of_mean_x_is_inv_n() {
    let mut g = GradGraph::new();
    let x = g.parameter(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap());
    let m = g.mean(x);
    let dx = g.grad_of(m, x).unwrap();
    let actual = forward_to_vec(&mut g, dx);
    let expected = vec![0.2; 5]; // 1/5 = 0.2 (exact in f64)
    assert_eq!(actual, expected);
}

#[test]
fn analytic_grad_of_sum_x_squared_is_2x() {
    for x_init in [
        vec![1.0, 2.0, 3.0],
        vec![-0.5, 0.0, 0.5, 1.5],
        vec![10.0, -10.0],
    ] {
        let n = x_init.len();
        let mut g = GradGraph::new();
        let x = g.parameter(Tensor::from_vec(x_init.clone(), &[n]).unwrap());
        let xx = g.mul(x, x);
        let s = g.sum(xx);
        let dx = g.grad_of(s, x).unwrap();
        let actual = forward_to_vec(&mut g, dx);
        let expected: Vec<f64> = x_init.iter().map(|&v| 2.0 * v).collect();
        // Polynomials evaluate exactly in f64.
        let bits = |v: &[f64]| -> Vec<u64> { v.iter().map(|x| x.to_bits()).collect() };
        assert_eq!(
            bits(&actual),
            bits(&expected),
            "grad_of(Σx², x) for x={x_init:?}: expected {expected:?}, got {actual:?}"
        );
    }
}

#[test]
fn analytic_second_order_d2_of_sum_x_squared_is_2_broadcast() {
    let mut g = GradGraph::new();
    let x = g.parameter(Tensor::from_vec(vec![0.7, -1.3, 2.5, 0.1], &[4]).unwrap());
    let xx = g.mul(x, x);
    let s = g.sum(xx);
    let dx = g.grad_of(s, x).unwrap();    // 2x
    let d2x = g.grad_of(dx, x).unwrap();  // 2 (broadcast)
    let actual = forward_to_vec(&mut g, d2x);
    // d²(Σx²)/dx² = 2 for each element, regardless of x's value.
    let expected = vec![2.0; 4];
    assert_eq!(actual, expected);
}

#[test]
fn analytic_grad_of_mean_x_squared_is_2x_over_n() {
    let mut g = GradGraph::new();
    let x_init = vec![3.0, -1.5, 4.5, 0.0];
    let x = g.parameter(Tensor::from_vec(x_init.clone(), &[4]).unwrap());
    let xx = g.mul(x, x);
    let m = g.mean(xx);
    let dx = g.grad_of(m, x).unwrap();
    let actual = forward_to_vec(&mut g, dx);
    // d(mean(x²))/dx_i = 2x_i / N. ULP-tolerance because the grad_of path
    // computes (1/N) * 2 * x rather than 2*x/N, which can differ by one ULP.
    let expected: Vec<f64> = x_init.iter().map(|&v| 2.0 * v / 4.0).collect();
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-12, "got {a}, expected ≈ {e}");
    }
}

// ─── Cross-check vs backward() ─────────────────────────────────────

#[test]
fn grad_of_agrees_with_backward_on_sum_polynomial() {
    // f = Σ (x_i * y_i). df/dx_i = y_i; df/dy_i = x_i.
    let n = 4;
    let mut g = GradGraph::new();
    let x_data = vec![1.0, 2.0, 3.0, 4.0];
    let y_data = vec![5.0, 6.0, 7.0, 8.0];
    let x = g.parameter(Tensor::from_vec(x_data.clone(), &[n]).unwrap());
    let y = g.parameter(Tensor::from_vec(y_data.clone(), &[n]).unwrap());
    let xy = g.mul(x, y);
    let s = g.sum(xy);

    // grad_of path
    let dfdx = g.grad_of(s, x).unwrap();
    let dfdy = g.grad_of(s, y).unwrap();
    let dfdx_val = forward_to_vec(&mut g, dfdx);
    let dfdy_val = forward_to_vec(&mut g, dfdy);

    // backward path
    g.zero_grad();
    g.backward(s);
    let bw_dfdx = g.grad(x).unwrap().to_vec();
    let bw_dfdy = g.grad(y).unwrap().to_vec();

    let bits = |v: &[f64]| -> Vec<u64> { v.iter().map(|x| x.to_bits()).collect() };
    assert_eq!(bits(&dfdx_val), bits(&bw_dfdx));
    assert_eq!(bits(&dfdy_val), bits(&bw_dfdy));
    assert_eq!(dfdx_val, y_data);
    assert_eq!(dfdy_val, x_data);
}

#[test]
fn grad_of_agrees_with_backward_on_mean_chain() {
    // f = mean(2x + 3y) where x, y are vectors.
    // df/dx_i = 2/N; df/dy_i = 3/N.
    //
    // The cross-check (grad_of bits == backward bits) is the strongest
    // signal here. The analytic equality is only checked via ULP-tolerance
    // because (1/N) * scalar produces a different bit pattern than the
    // f64 literal of the simplified ratio (e.g., (1/5)*3 != 0.6 bit-for-bit).
    let n = 5;
    let mut g = GradGraph::new();
    let x = g.parameter(Tensor::from_vec(vec![1.0; n], &[n]).unwrap());
    let y = g.parameter(Tensor::from_vec(vec![2.0; n], &[n]).unwrap());
    let two_x = g.scalar_mul(x, 2.0);
    let three_y = g.scalar_mul(y, 3.0);
    let sum = g.add(two_x, three_y);
    let m = g.mean(sum);

    let dfdx = g.grad_of(m, x).unwrap();
    let dfdy = g.grad_of(m, y).unwrap();
    let dfdx_val = forward_to_vec(&mut g, dfdx);
    let dfdy_val = forward_to_vec(&mut g, dfdy);

    g.zero_grad();
    g.backward(m);
    let bw_dfdx = g.grad(x).unwrap().to_vec();
    let bw_dfdy = g.grad(y).unwrap().to_vec();

    // grad_of and backward must agree bit-for-bit (same path, same arithmetic).
    let bits = |v: &[f64]| -> Vec<u64> { v.iter().map(|x| x.to_bits()).collect() };
    assert_eq!(bits(&dfdx_val), bits(&bw_dfdx));
    assert_eq!(bits(&dfdy_val), bits(&bw_dfdy));
    // Analytic equality up to f64 ULP — the closed-form value 0.4 / 0.6
    // and the computed (1/N)*scalar may differ by one ULP.
    for v in &dfdx_val {
        assert!((v - 0.4).abs() < 1e-12, "dfdx_val element = {v}, expected ≈ 0.4");
    }
    for v in &dfdy_val {
        assert!((v - 0.6).abs() < 1e-12, "dfdy_val element = {v}, expected ≈ 0.6");
    }
}

// ─── BroadcastScalar primitive itself ──────────────────────────────

#[test]
fn broadcast_scalar_forward_replicates_value() {
    let mut g = GradGraph::new();
    let s = g.parameter(Tensor::from_vec(vec![3.7], &[1]).unwrap());
    let bc = g.broadcast_scalar(s, &[5]);
    assert_eq!(g.tensor(bc).to_vec(), vec![3.7; 5]);
}

#[test]
fn broadcast_scalar_backward_sums_upstream() {
    // Build f = Σ broadcast_scalar(s, [4]). df/ds = 4 (length of broadcast).
    let mut g = GradGraph::new();
    let s = g.parameter(Tensor::from_vec(vec![1.0], &[1]).unwrap());
    let bc = g.broadcast_scalar(s, &[4]);
    let f = g.sum(bc);

    g.zero_grad();
    g.backward(f);
    let bw = g.grad(s).unwrap().to_vec();
    assert_eq!(bw, vec![4.0]);
}

#[test]
fn broadcast_scalar_grad_of_round_trip() {
    // grad_of(BroadcastScalar(s, [n]), s) should be a node holding [n].
    // Sum of a length-n broadcast is n, so its grad is the scalar n.
    let mut g = GradGraph::new();
    let s = g.parameter(Tensor::from_vec(vec![1.0], &[1]).unwrap());
    let bc = g.broadcast_scalar(s, &[3]);
    let summed = g.sum(bc); // we want a scalar f to call grad_of on
    let ds = g.grad_of(summed, s).unwrap();
    let val = forward_to_vec(&mut g, ds);
    assert_eq!(val, vec![3.0]);
}

// ─── Dispatch boundary smoke test ──────────────────────────────────

#[test]
fn dispatch_grad_of_sum_yields_ones_via_value_boundary() {
    let _ = dispatch_grad_graph("grad_graph_new", &[]).unwrap();
    let xv = dispatch_grad_graph(
        "grad_graph_param",
        &[Value::Tensor(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap())],
    )
    .unwrap()
    .unwrap();
    let x_idx = match xv {
        Value::Int(i) => i,
        _ => unreachable!(),
    };
    let s = dispatch_grad_graph("grad_graph_sum", &[Value::Int(x_idx)])
        .unwrap()
        .unwrap();
    let s_idx = match s {
        Value::Int(i) => i,
        _ => unreachable!(),
    };
    let dx = dispatch_grad_graph(
        "grad_graph_grad_of",
        &[Value::Int(s_idx), Value::Int(x_idx)],
    )
    .unwrap()
    .unwrap();
    let dx_idx = match dx {
        Value::Int(i) => i,
        _ => unreachable!(),
    };
    let val = dispatch_grad_graph("grad_graph_forward", &[Value::Int(dx_idx)])
        .unwrap()
        .unwrap();
    let t = match val {
        Value::Tensor(t) => t,
        _ => unreachable!(),
    };
    assert_eq!(t.to_vec(), vec![1.0, 1.0, 1.0]);
}
