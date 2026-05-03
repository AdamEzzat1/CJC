//! Phase 3d — native higher-order autodiff.
//!
//! Tests that `grad_graph_grad_of(f, x)` builds a graph node representing
//! `dF/dx` as a function of the existing parameters, and that the result
//! is itself differentiable (giving native second- and higher-order
//! derivatives).
//!
//! Phase 3d ships the **polynomial-arithmetic op subset**: Input,
//! Parameter, Add, Sub, Mul, ScalarMul, Neg. These are sufficient to
//! express any polynomial in the parameters; the test cases exercise
//! polynomial derivatives where the analytic answer is known exactly,
//! so any rounding error reveals an indexing bug.
//!
//! Three categories of test:
//!
//! 1. **Wiring (eval ↔ MIR byte-equal)** — same `.cjcl` snippet runs
//!    through both executors with identical printed output.
//! 2. **Analytic correctness** — `grad_of(x², x) = 2x`,
//!    `grad_of(grad_of(x³, x), x) = 6x`, etc. evaluated against the
//!    closed-form derivative.
//! 3. **Unsupported ops Err** — calling `grad_of` on an op outside the
//!    polynomial subset returns a clean error message, never panics.

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
fn wiring_grad_of_x_squared_yields_2x() {
    assert_parity(
        "grad_of(x², x) at x=3 = 6",
        r#"
        grad_graph_new();
        let x = grad_graph_param(Tensor.from_vec([3.0], [1]));
        let xx = grad_graph_mul(x, x);
        let dx = grad_graph_grad_of(xx, x);
        print(grad_graph_forward(dx));
        "#,
    );
}

#[test]
fn wiring_grad_of_x_cubed_yields_3x_squared() {
    assert_parity(
        "grad_of(x³, x) at x=2 = 12",
        r#"
        grad_graph_new();
        let x = grad_graph_param(Tensor.from_vec([2.0], [1]));
        let xx = grad_graph_mul(x, x);
        let xxx = grad_graph_mul(xx, x);
        let dx = grad_graph_grad_of(xxx, x);
        print(grad_graph_forward(dx));
        "#,
    );
}

#[test]
fn wiring_second_order_grad_of_x_cubed_yields_6x() {
    // d²(x³)/dx² = 6x. At x=2, expect 12.
    // This is the killer test — the gradient sub-graph from the first
    // grad_of must itself be differentiable, which is the entire point
    // of native HOAD.
    assert_parity(
        "grad_of(grad_of(x³, x), x) at x=2 = 12",
        r#"
        grad_graph_new();
        let x = grad_graph_param(Tensor.from_vec([2.0], [1]));
        let xx = grad_graph_mul(x, x);
        let xxx = grad_graph_mul(xx, x);
        let dx = grad_graph_grad_of(xxx, x);
        let d2x = grad_graph_grad_of(dx, x);
        print(grad_graph_forward(d2x));
        "#,
    );
}

#[test]
fn wiring_grad_of_add_distributes() {
    // f = x + y. df/dx = 1.
    assert_parity(
        "grad_of(x+y, x) = 1",
        r#"
        grad_graph_new();
        let x = grad_graph_param(Tensor.from_vec([5.0], [1]));
        let y = grad_graph_param(Tensor.from_vec([7.0], [1]));
        let s = grad_graph_add(x, y);
        let dx = grad_graph_grad_of(s, x);
        print(grad_graph_forward(dx));
        "#,
    );
}

#[test]
fn wiring_grad_of_sub_negates() {
    // f = x - y. df/dy = -1.
    assert_parity(
        "grad_of(x-y, y) = -1",
        r#"
        grad_graph_new();
        let x = grad_graph_param(Tensor.from_vec([5.0], [1]));
        let y = grad_graph_param(Tensor.from_vec([7.0], [1]));
        let s = grad_graph_sub(x, y);
        let dy = grad_graph_grad_of(s, y);
        print(grad_graph_forward(dy));
        "#,
    );
}

#[test]
fn wiring_grad_of_scalar_mul() {
    // f = 7*x. df/dx = 7.
    assert_parity(
        "grad_of(7x, x) = 7",
        r#"
        grad_graph_new();
        let x = grad_graph_param(Tensor.from_vec([3.0], [1]));
        let s = grad_graph_scalar_mul(x, 7.0);
        let dx = grad_graph_grad_of(s, x);
        print(grad_graph_forward(dx));
        "#,
    );
}

#[test]
fn wiring_grad_of_neg() {
    // f = -x. df/dx = -1.
    assert_parity(
        "grad_of(-x, x) = -1",
        r#"
        grad_graph_new();
        let x = grad_graph_param(Tensor.from_vec([5.0], [1]));
        let s = grad_graph_neg(x);
        let dx = grad_graph_grad_of(s, x);
        print(grad_graph_forward(dx));
        "#,
    );
}

// ─── Analytic correctness (Rust-direct) ────────────────────────────

fn forward_scalar(g: &mut GradGraph, idx: usize) -> f64 {
    g.tensor(idx).to_vec()[0]
}

#[test]
fn analytic_grad_of_x_squared() {
    let mut g = GradGraph::new();
    let x = g.parameter(Tensor::from_vec(vec![3.5], &[1]).unwrap());
    let xx = g.mul(x, x);
    let dx = g.grad_of(xx, x).unwrap();
    // d(x²)/dx = 2x. At x=3.5, expect 7.0.
    assert_eq!(forward_scalar(&mut g, dx), 7.0);
}

#[test]
fn analytic_grad_of_x_cubed_at_multiple_points() {
    for &x_val in &[1.0, 2.0, -1.5, 0.5, 4.0] {
        let mut g = GradGraph::new();
        let x = g.parameter(Tensor::from_vec(vec![x_val], &[1]).unwrap());
        let xx = g.mul(x, x);
        let xxx = g.mul(xx, x);
        let dx = g.grad_of(xxx, x).unwrap();
        let expected = 3.0 * x_val * x_val;
        let actual = forward_scalar(&mut g, dx);
        // f64::EPSILON-level: derivatives of polynomials are exact.
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "grad_of(x³, x) at x={x_val}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn analytic_second_order_d2_dx2_of_x_cubed_is_6x() {
    for &x_val in &[1.0, 2.0, -1.5, 0.5] {
        let mut g = GradGraph::new();
        let x = g.parameter(Tensor::from_vec(vec![x_val], &[1]).unwrap());
        let xx = g.mul(x, x);
        let xxx = g.mul(xx, x);
        let dx = g.grad_of(xxx, x).unwrap();
        let d2x = g.grad_of(dx, x).unwrap();
        let expected = 6.0 * x_val;
        let actual = forward_scalar(&mut g, d2x);
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "d²(x³)/dx² at x={x_val}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn analytic_third_order_d3_dx3_of_x_cubed_is_6() {
    let mut g = GradGraph::new();
    let x = g.parameter(Tensor::from_vec(vec![2.5], &[1]).unwrap());
    let xx = g.mul(x, x);
    let xxx = g.mul(xx, x);
    let d1 = g.grad_of(xxx, x).unwrap();
    let d2 = g.grad_of(d1, x).unwrap();
    let d3 = g.grad_of(d2, x).unwrap();
    // d³(x³)/dx³ = 6 (constant, independent of x).
    assert_eq!(forward_scalar(&mut g, d3), 6.0);
}

#[test]
fn analytic_grad_of_polynomial_x2_plus_5x_plus_3() {
    // f = x² + 5x + 3.
    // df/dx = 2x + 5.
    // At x=4, expect 13.
    let mut g = GradGraph::new();
    let x = g.parameter(Tensor::from_vec(vec![4.0], &[1]).unwrap());
    let three = g.input(Tensor::from_vec(vec![3.0], &[1]).unwrap());
    let xx = g.mul(x, x);
    let five_x = g.scalar_mul(x, 5.0);
    let sum1 = g.add(xx, five_x);
    let f = g.add(sum1, three);
    let df = g.grad_of(f, x).unwrap();
    assert_eq!(forward_scalar(&mut g, df), 13.0);

    // Second derivative: d²f/dx² = 2 (constant).
    let d2f = g.grad_of(df, x).unwrap();
    assert_eq!(forward_scalar(&mut g, d2f), 2.0);
}

#[test]
fn analytic_grad_of_x_minus_y_w_r_t_y_is_negative_one() {
    let mut g = GradGraph::new();
    let x = g.parameter(Tensor::from_vec(vec![10.0], &[1]).unwrap());
    let y = g.parameter(Tensor::from_vec(vec![3.0], &[1]).unwrap());
    let f = g.sub(x, y);
    let dy = g.grad_of(f, y).unwrap();
    assert_eq!(forward_scalar(&mut g, dy), -1.0);
}

#[test]
fn analytic_grad_of_neg_x_w_r_t_x_is_negative_one() {
    let mut g = GradGraph::new();
    let x = g.parameter(Tensor::from_vec(vec![7.0], &[1]).unwrap());
    let f = g.neg(x);
    let dx = g.grad_of(f, x).unwrap();
    assert_eq!(forward_scalar(&mut g, dx), -1.0);
}

// ─── Cross-check against backward() ────────────────────────────────

#[test]
fn grad_of_agrees_with_backward_on_polynomial() {
    // Compute grad via grad_of, then compute the same grad via backward,
    // and assert byte-equal. This validates that the gradient sub-graph's
    // forward evaluation produces the same value as the existing
    // tensor-accumulating backward path.
    let mut g = GradGraph::new();
    let x = g.parameter(Tensor::from_vec(vec![3.0], &[1]).unwrap());
    let y = g.parameter(Tensor::from_vec(vec![5.0], &[1]).unwrap());
    // f = x*y + 2*x = x(y + 2). df/dx = y + 2 = 7. df/dy = x = 3.
    let xy = g.mul(x, y);
    let two_x = g.scalar_mul(x, 2.0);
    let f = g.add(xy, two_x);

    // grad_of path
    let dfdx_node = g.grad_of(f, x).unwrap();
    let dfdy_node = g.grad_of(f, y).unwrap();
    let dfdx_val = forward_scalar(&mut g, dfdx_node);
    let dfdy_val = forward_scalar(&mut g, dfdy_node);

    // backward path
    g.zero_grad();
    g.backward(f);
    let bw_dfdx = g.grad(x).unwrap().to_vec()[0];
    let bw_dfdy = g.grad(y).unwrap().to_vec()[0];

    assert_eq!(dfdx_val.to_bits(), bw_dfdx.to_bits());
    assert_eq!(dfdy_val.to_bits(), bw_dfdy.to_bits());
    assert_eq!(dfdx_val, 7.0);
    assert_eq!(dfdy_val, 3.0);
}

// ─── Unsupported ops produce a clean Err ───────────────────────────

#[test]
fn unsupported_op_errs_not_panics() {
    // Originally this test used `tanh_act` to gate "unsupported op
    // produces clean Err." Phase 3e Tier 2 added native support for
    // `tanh_act`, so we now use `relu` — which remains deferred to
    // Tier 3 (piecewise activations need a Where-like graph op).
    // The test's purpose is unchanged: any *currently unsupported*
    // op must Err cleanly, never panic.
    let mut g = GradGraph::new();
    let x = g.parameter(Tensor::from_vec(vec![0.5], &[1]).unwrap());
    let r = g.relu(x);
    let result = g.grad_of(r, x);
    assert!(result.is_err(), "expected Err for relu; got {result:?}");
    let msg = result.unwrap_err();
    assert!(
        msg.contains("not yet supported"),
        "Err message should mention deferred status, got: {msg}"
    );
}

#[test]
fn x_unreachable_from_f_errs() {
    // x and f are independent; grad_of(f, x) is meaningless.
    let mut g = GradGraph::new();
    let _x = g.parameter(Tensor::from_vec(vec![1.0], &[1]).unwrap());
    let y = g.parameter(Tensor::from_vec(vec![2.0], &[1]).unwrap());
    let f = g.scalar_mul(y, 3.0); // f depends on y, not x
    // Use the FIRST `_x` we created (we just need its index for grad_of).
    // It's index 0; the parameter call returns it. Re-fetch:
    let x = g.parameter(Tensor::from_vec(vec![5.0], &[1]).unwrap());
    let result = g.grad_of(f, x);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not reachable"));
}

#[test]
fn dispatch_grad_of_via_value_boundary() {
    let _ = dispatch_grad_graph("grad_graph_new", &[]).unwrap();
    let xv = dispatch_grad_graph(
        "grad_graph_param",
        &[Value::Tensor(Tensor::from_vec(vec![2.0], &[1]).unwrap())],
    )
    .unwrap()
    .unwrap();
    let x_idx = match xv {
        Value::Int(i) => i,
        _ => unreachable!(),
    };
    let xx = dispatch_grad_graph(
        "grad_graph_mul",
        &[Value::Int(x_idx), Value::Int(x_idx)],
    )
    .unwrap()
    .unwrap();
    let xx_idx = match xx {
        Value::Int(i) => i,
        _ => unreachable!(),
    };
    let dx = dispatch_grad_graph(
        "grad_graph_grad_of",
        &[Value::Int(xx_idx), Value::Int(x_idx)],
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
    // d(x²)/dx at x=2 = 4.
    assert_eq!(t.to_vec(), vec![4.0]);
}
