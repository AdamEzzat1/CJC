//! Phase 3e Tier 2 — native HOAD for transcendentals + smooth activations.
//!
//! Adds nine new arms to `grad_of`:
//!
//! | GradOp     | Derivative formula              | Sub-graph composition         |
//! |------------|---------------------------------|-------------------------------|
//! | Exp(a)     | dF/da = upstream * exp(a)       | mul(upstream, current_node)   |
//! | Ln(a)      | dF/da = upstream / a            | div(upstream, a)              |
//! | Sqrt(a)    | dF/da = upstream / (2*sqrt(a))  | scalar_mul(div(upstream,i),0.5)|
//! | Sin(a)     | dF/da = upstream * cos(a)       | mul(upstream, cos(a))         |
//! | Cos(a)     | dF/da = -upstream * sin(a)      | neg(mul(upstream, sin(a)))    |
//! | Pow(a,n)   | dF/da = upstream * n*a^(n-1)    | mul(upstream, n*pow(a, n-1))  |
//! | Log2(a)    | dF/da = upstream / (a * ln(2))  | scalar_mul(div(up,a), 1/ln2)  |
//! | Sigmoid(a) | dF/da = upstream * σ(1-σ)       | mul(up, mul(σ, sub(ones, σ))) |
//! | TanhAct(a) | dF/da = upstream * (1 - tanh²)  | mul(up, sub(ones, mul(i, i))) |
//!
//! Each arm composes only ops from the polynomial subset (Tier 0) + the
//! Tier 1 reductions/broadcast — so the closure-under-differentiation
//! invariant is preserved. Second-order calls work for ops whose
//! derivative formula doesn't introduce ops outside the supported set.
//! For ops like Sigmoid where d²σ/dx² involves σ itself, second-order
//! works because Sigmoid is now supported.
//!
//! ## Deferred to Tier 3
//!
//! - Piecewise activations: Relu, Abs, Elu, Selu (need Where-like
//!   primitive to express derivative cleanly).
//! - Composite activations: Gelu, Silu (derivatives reference
//!   non-polynomial sub-expressions).
//! - Compound ops: Matmul, Div (in grad_of itself), MlpLayer, Softmax,
//!   LayerNorm, BatchNorm, CrossEntropy.
//!
//! ## Test categories
//!
//! 1. Wiring (eval ↔ MIR byte-equal) — small `.cjcl` snippets.
//! 2. Analytic correctness vs closed-form derivatives.
//! 3. Cross-check vs `backward()` — bit-equal on identical paths.
//! 4. Second-order derivatives — proves the gradient sub-graph is
//!    itself differentiable for ops whose derivative formula stays in
//!    the supported set.

#![allow(clippy::needless_raw_string_hashes)]

use cjc_ad::GradGraph;
use cjc_runtime::tensor::Tensor;

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

fn forward_to_vec(g: &mut GradGraph, idx: usize) -> Vec<f64> {
    g.tensor(idx).to_vec()
}

// ─── Wiring (eval ↔ MIR byte-equal) ────────────────────────────────

#[test]
fn wiring_grad_of_exp() {
    // f = exp(x), df/dx = exp(x). At x=1.0, expect e ≈ 2.718…
    assert_parity(
        "grad_of(exp(x), x) = exp(x)",
        r#"
        grad_graph_new();
        let x = grad_graph_param(Tensor.from_vec([1.0], [1]));
        let e = grad_graph_exp(x);
        let dx = grad_graph_grad_of(e, x);
        print(grad_graph_forward(dx));
        "#,
    );
}

#[test]
fn wiring_grad_of_ln() {
    // f = ln(x), df/dx = 1/x. At x=2, expect 0.5.
    assert_parity(
        "grad_of(ln(x), x) = 1/x",
        r#"
        grad_graph_new();
        let x = grad_graph_param(Tensor.from_vec([2.0], [1]));
        let l = grad_graph_ln(x);
        let dx = grad_graph_grad_of(l, x);
        print(grad_graph_forward(dx));
        "#,
    );
}

#[test]
fn wiring_grad_of_sqrt() {
    // f = sqrt(x), df/dx = 1/(2*sqrt(x)). At x=4, expect 0.25.
    assert_parity(
        "grad_of(sqrt(x), x) = 1/(2*sqrt(x))",
        r#"
        grad_graph_new();
        let x = grad_graph_param(Tensor.from_vec([4.0], [1]));
        let s = grad_graph_sqrt(x);
        let dx = grad_graph_grad_of(s, x);
        print(grad_graph_forward(dx));
        "#,
    );
}

#[test]
fn wiring_grad_of_sin_cos() {
    // f = sin(x), df/dx = cos(x). At x=0, expect 1.
    assert_parity(
        "grad_of(sin(x), x) = cos(x); grad_of(cos(x), x) = -sin(x)",
        r#"
        grad_graph_new();
        let x = grad_graph_param(Tensor.from_vec([0.0], [1]));
        let s = grad_graph_sin(x);
        let c = grad_graph_cos(x);
        let ds = grad_graph_grad_of(s, x);
        let dc = grad_graph_grad_of(c, x);
        print(grad_graph_forward(ds));
        print(grad_graph_forward(dc));
        "#,
    );
}

#[test]
fn wiring_grad_of_pow() {
    // f = x^4, df/dx = 4x³. At x=2, expect 32.
    assert_parity(
        "grad_of(x^4, x) = 4x³",
        r#"
        grad_graph_new();
        let x = grad_graph_param(Tensor.from_vec([2.0], [1]));
        let p = grad_graph_pow(x, 4.0);
        let dx = grad_graph_grad_of(p, x);
        print(grad_graph_forward(dx));
        "#,
    );
}

#[test]
fn wiring_grad_of_tanh() {
    // f = tanh(x), df/dx = 1 - tanh²(x). At x=0, tanh=0, so df/dx=1.
    assert_parity(
        "grad_of(tanh(x), x) at x=0 = 1",
        r#"
        grad_graph_new();
        let x = grad_graph_param(Tensor.from_vec([0.0], [1]));
        let t = grad_graph_tanh(x);
        let dx = grad_graph_grad_of(t, x);
        print(grad_graph_forward(dx));
        "#,
    );
}

#[test]
fn wiring_second_order_d2_of_x_squared_via_pow() {
    // d²(x^2)/dx² = 2 (constant). Built via Pow rather than Mul.
    assert_parity(
        "d²(x²)/dx² = 2 via Pow",
        r#"
        grad_graph_new();
        let x = grad_graph_param(Tensor.from_vec([3.5], [1]));
        let p = grad_graph_pow(x, 2.0);
        let dx = grad_graph_grad_of(p, x);
        let d2x = grad_graph_grad_of(dx, x);
        print(grad_graph_forward(d2x));
        "#,
    );
}

// ─── Analytic correctness (Rust-direct) ────────────────────────────

#[test]
fn analytic_grad_of_exp() {
    for &x_val in &[0.0, 0.5, -1.0, 2.0] {
        let mut g = GradGraph::new();
        let x = g.parameter(Tensor::from_vec(vec![x_val], &[1]).unwrap());
        let e = g.exp(x);
        let dx = g.grad_of(e, x).unwrap();
        let actual = forward_to_vec(&mut g, dx)[0];
        let expected = x_val.exp();
        assert!((actual - expected).abs() < 1e-12,
            "grad_of(exp(x), x) at x={x_val}: expected {expected}, got {actual}");
    }
}

#[test]
fn analytic_grad_of_ln() {
    for &x_val in &[0.5, 1.0, 2.0, 10.0] {
        let mut g = GradGraph::new();
        let x = g.parameter(Tensor::from_vec(vec![x_val], &[1]).unwrap());
        let l = g.ln(x);
        let dx = g.grad_of(l, x).unwrap();
        let actual = forward_to_vec(&mut g, dx)[0];
        let expected = 1.0 / x_val;
        assert!((actual - expected).abs() < 1e-12);
    }
}

#[test]
fn analytic_grad_of_sqrt() {
    for &x_val in &[0.25, 1.0, 4.0, 16.0] {
        let mut g = GradGraph::new();
        let x = g.parameter(Tensor::from_vec(vec![x_val], &[1]).unwrap());
        let s = g.sqrt(x);
        let dx = g.grad_of(s, x).unwrap();
        let actual = forward_to_vec(&mut g, dx)[0];
        let expected = 1.0 / (2.0 * x_val.sqrt());
        assert!((actual - expected).abs() < 1e-12);
    }
}

#[test]
fn analytic_grad_of_sin_yields_cos() {
    for &x_val in &[0.0, 0.5, 1.0, std::f64::consts::PI / 4.0] {
        let mut g = GradGraph::new();
        let x = g.parameter(Tensor::from_vec(vec![x_val], &[1]).unwrap());
        let s = g.sin(x);
        let dx = g.grad_of(s, x).unwrap();
        let actual = forward_to_vec(&mut g, dx)[0];
        let expected = x_val.cos();
        assert!((actual - expected).abs() < 1e-12);
    }
}

#[test]
fn analytic_grad_of_cos_yields_neg_sin() {
    for &x_val in &[0.0, 0.5, 1.0, std::f64::consts::PI / 3.0] {
        let mut g = GradGraph::new();
        let x = g.parameter(Tensor::from_vec(vec![x_val], &[1]).unwrap());
        let c = g.cos(x);
        let dx = g.grad_of(c, x).unwrap();
        let actual = forward_to_vec(&mut g, dx)[0];
        let expected = -x_val.sin();
        assert!((actual - expected).abs() < 1e-12);
    }
}

#[test]
fn analytic_grad_of_pow_with_various_exponents() {
    // d(x^n)/dx = n * x^(n-1).
    for &(x_val, n) in &[(2.0, 3.0), (1.5, 4.0), (0.5, 2.0), (3.0, 1.0)] {
        let mut g = GradGraph::new();
        let x = g.parameter(Tensor::from_vec(vec![x_val], &[1]).unwrap());
        let p = g.pow(x, n);
        let dx = g.grad_of(p, x).unwrap();
        let actual = forward_to_vec(&mut g, dx)[0];
        let expected = n * x_val.powf(n - 1.0);
        assert!((actual - expected).abs() < 1e-10,
            "grad_of(x^{n}, x) at x={x_val}: expected {expected}, got {actual}");
    }
}

#[test]
fn analytic_grad_of_log2() {
    // d(log2(x))/dx = 1/(x * ln(2)).
    for &x_val in &[0.5, 1.0, 2.0, 8.0] {
        let mut g = GradGraph::new();
        let x = g.parameter(Tensor::from_vec(vec![x_val], &[1]).unwrap());
        let l = g.log2(x);
        let dx = g.grad_of(l, x).unwrap();
        let actual = forward_to_vec(&mut g, dx)[0];
        let expected = 1.0 / (x_val * std::f64::consts::LN_2);
        assert!((actual - expected).abs() < 1e-12);
    }
}

#[test]
fn analytic_grad_of_sigmoid() {
    // d(σ(x))/dx = σ(x) * (1 - σ(x)).
    for &x_val in &[-2.0, -0.5, 0.0, 0.5, 2.0] {
        let mut g = GradGraph::new();
        let x = g.parameter(Tensor::from_vec(vec![x_val], &[1]).unwrap());
        let s = g.sigmoid(x);
        let dx = g.grad_of(s, x).unwrap();
        let actual = forward_to_vec(&mut g, dx)[0];
        let sigmoid_val = 1.0 / (1.0 + (-x_val).exp());
        let expected = sigmoid_val * (1.0 - sigmoid_val);
        assert!((actual - expected).abs() < 1e-12);
    }
}

#[test]
fn analytic_grad_of_tanh() {
    // d(tanh(x))/dx = 1 - tanh²(x).
    for &x_val in &[-2.0, -0.5, 0.0, 0.5, 2.0] {
        let mut g = GradGraph::new();
        let x = g.parameter(Tensor::from_vec(vec![x_val], &[1]).unwrap());
        let t = g.tanh_act(x);
        let dx = g.grad_of(t, x).unwrap();
        let actual = forward_to_vec(&mut g, dx)[0];
        let tanh_val = x_val.tanh();
        let expected = 1.0 - tanh_val * tanh_val;
        assert!((actual - expected).abs() < 1e-12);
    }
}

// ─── Second-order ──────────────────────────────────────────────────

#[test]
fn analytic_second_order_d2_of_exp_is_exp() {
    // d²(exp(x))/dx² = exp(x). Pure self-similar derivative — every
    // higher-order is exp(x).
    let mut g = GradGraph::new();
    let x = g.parameter(Tensor::from_vec(vec![0.5], &[1]).unwrap());
    let e = g.exp(x);
    let d1 = g.grad_of(e, x).unwrap();
    let d2 = g.grad_of(d1, x).unwrap();
    let actual = forward_to_vec(&mut g, d2)[0];
    let expected = 0.5_f64.exp();
    assert!((actual - expected).abs() < 1e-12,
        "d²(exp(x))/dx² at x=0.5: expected {expected}, got {actual}");
}

#[test]
fn analytic_second_order_d2_of_x_pow_3_is_6x() {
    // d²(x³)/dx² = 6x. Built via Pow (not Mul chain) — exercises the
    // Pow grad arm's second-order behavior.
    let mut g = GradGraph::new();
    let x = g.parameter(Tensor::from_vec(vec![2.5], &[1]).unwrap());
    let p = g.pow(x, 3.0);
    let d1 = g.grad_of(p, x).unwrap();
    let d2 = g.grad_of(d1, x).unwrap();
    let actual = forward_to_vec(&mut g, d2)[0];
    let expected = 6.0 * 2.5;
    assert!((actual - expected).abs() < 1e-10);
}

// ─── Cross-check vs backward() ─────────────────────────────────────

#[test]
fn grad_of_agrees_with_backward_on_chain() {
    // f = sum(exp(x) * sin(y)). df/dx = sum(exp(x)*sin(y)*1_x) → exp(x)*sin(y);
    // df/dy = exp(x)*cos(y).
    let mut g = GradGraph::new();
    let x = g.parameter(Tensor::from_vec(vec![0.5, 1.0], &[2]).unwrap());
    let y = g.parameter(Tensor::from_vec(vec![0.3, 0.7], &[2]).unwrap());
    let ex = g.exp(x);
    let sy = g.sin(y);
    let prod = g.mul(ex, sy);
    let s = g.sum(prod);

    let dfdx = g.grad_of(s, x).unwrap();
    let dfdy = g.grad_of(s, y).unwrap();
    let dfdx_v = forward_to_vec(&mut g, dfdx);
    let dfdy_v = forward_to_vec(&mut g, dfdy);

    g.zero_grad();
    g.backward(s);
    let bw_x = g.grad(x).unwrap().to_vec();
    let bw_y = g.grad(y).unwrap().to_vec();

    let bits = |v: &[f64]| -> Vec<u64> { v.iter().map(|x| x.to_bits()).collect() };
    assert_eq!(bits(&dfdx_v), bits(&bw_x));
    assert_eq!(bits(&dfdy_v), bits(&bw_y));

    // Sanity: at x=[0.5,1.0], y=[0.3,0.7]:
    // dfdx[i] = exp(x[i]) * sin(y[i])
    // dfdy[i] = exp(x[i]) * cos(y[i])
    let expected_dx: Vec<f64> = (0..2).map(|i| (0.5 + i as f64 * 0.5).exp() * (0.3 + i as f64 * 0.4).sin()).collect();
    let expected_dy: Vec<f64> = (0..2).map(|i| (0.5 + i as f64 * 0.5).exp() * (0.3 + i as f64 * 0.4).cos()).collect();
    for (a, e) in dfdx_v.iter().zip(expected_dx.iter()) {
        assert!((a - e).abs() < 1e-12);
    }
    for (a, e) in dfdy_v.iter().zip(expected_dy.iter()) {
        assert!((a - e).abs() < 1e-12);
    }
}

#[test]
fn grad_of_agrees_with_backward_on_sigmoid_loss() {
    // f = sum((sigmoid(x) - target)²). df/dx = 2*(σ-target)*σ*(1-σ).
    let mut g = GradGraph::new();
    let x = g.parameter(Tensor::from_vec(vec![0.5, -0.3], &[2]).unwrap());
    let target = g.input(Tensor::from_vec(vec![0.7, 0.4], &[2]).unwrap());
    let sx = g.sigmoid(x);
    let diff = g.sub(sx, target);
    let sq = g.mul(diff, diff);
    let s = g.sum(sq);

    let dfdx = g.grad_of(s, x).unwrap();
    let dfdx_v = forward_to_vec(&mut g, dfdx);

    g.zero_grad();
    g.backward(s);
    let bw = g.grad(x).unwrap().to_vec();

    let bits = |v: &[f64]| -> Vec<u64> { v.iter().map(|x| x.to_bits()).collect() };
    assert_eq!(bits(&dfdx_v), bits(&bw));
}

// ─── Unsupported ops still Err cleanly ─────────────────────────────

#[test]
fn unsupported_relu_errs() {
    let mut g = GradGraph::new();
    let x = g.parameter(Tensor::from_vec(vec![0.5], &[1]).unwrap());
    let r = g.relu(x);
    let result = g.grad_of(r, x);
    assert!(result.is_err());
    let msg = result.unwrap_err();
    assert!(msg.contains("Phase 3e Tier 3") || msg.contains("not yet supported"),
        "expected Err mentioning Tier 3 deferral; got: {msg}");
}

#[test]
fn unsupported_gelu_errs() {
    let mut g = GradGraph::new();
    let x = g.parameter(Tensor::from_vec(vec![0.5], &[1]).unwrap());
    let r = g.gelu(x);
    let result = g.grad_of(r, x);
    assert!(result.is_err());
}
