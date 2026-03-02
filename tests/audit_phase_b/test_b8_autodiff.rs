// ── Phase B8: Autodiff Engine Improvements ──────────────────────────
// Integration tests verifying new GradOp variants work through the
// cjc-ad reverse-mode engine. These are Rust-level tests (not CJC language
// tests) since autodiff is used programmatically.

use cjc_runtime::Tensor;
use cjc_ad::GradGraph;

fn t(vals: &[f64]) -> Tensor {
    Tensor::from_vec_unchecked(vals.to_vec(), &[vals.len()])
}

#[test]
fn b8_sin_gradient() {
    let mut g = GradGraph::new();
    let x = g.parameter(t(&[std::f64::consts::FRAC_PI_4]));
    let y = g.sin(x);
    g.backward(y);
    let gx = g.grad(x).unwrap().to_vec();
    let expected = std::f64::consts::FRAC_PI_4.cos();
    assert!((gx[0] - expected).abs() < 1e-10, "got {}, expected {expected}", gx[0]);
}

#[test]
fn b8_cos_gradient() {
    let mut g = GradGraph::new();
    let x = g.parameter(t(&[std::f64::consts::FRAC_PI_4]));
    let y = g.cos(x);
    g.backward(y);
    let gx = g.grad(x).unwrap().to_vec();
    let expected = -std::f64::consts::FRAC_PI_4.sin();
    assert!((gx[0] - expected).abs() < 1e-10, "got {}, expected {expected}", gx[0]);
}

#[test]
fn b8_sqrt_gradient() {
    let mut g = GradGraph::new();
    let x = g.parameter(t(&[9.0]));
    let y = g.sqrt(x);
    g.backward(y);
    let gx = g.grad(x).unwrap().to_vec();
    // d/dx sqrt(9) = 1/(2*3) = 1/6
    assert!((gx[0] - 1.0 / 6.0).abs() < 1e-10, "got {}", gx[0]);
}

#[test]
fn b8_pow_gradient() {
    let mut g = GradGraph::new();
    let x = g.parameter(t(&[3.0]));
    let y = g.pow(x, 2.0); // x^2
    g.backward(y);
    let gx = g.grad(x).unwrap().to_vec();
    // d/dx x^2 at x=3 = 2*3 = 6
    assert!((gx[0] - 6.0).abs() < 1e-10, "got {}", gx[0]);
}

#[test]
fn b8_sigmoid_gradient() {
    let mut g = GradGraph::new();
    let x = g.parameter(t(&[0.0]));
    let y = g.sigmoid(x);
    let loss = g.sum(y);
    g.backward(loss);
    let gx = g.grad(x).unwrap().to_vec();
    // sigmoid'(0) = 0.25
    assert!((gx[0] - 0.25).abs() < 1e-10, "got {}", gx[0]);
}

#[test]
fn b8_relu_gradient_positive() {
    let mut g = GradGraph::new();
    let x = g.parameter(t(&[5.0]));
    let y = g.relu(x);
    g.backward(y);
    let gx = g.grad(x).unwrap().to_vec();
    assert!((gx[0] - 1.0).abs() < 1e-10);
}

#[test]
fn b8_relu_gradient_negative() {
    let mut g = GradGraph::new();
    let x = g.parameter(t(&[-5.0]));
    let y = g.relu(x);
    g.backward(y);
    let gx = g.grad(x).unwrap().to_vec();
    assert!(gx[0].abs() < 1e-10);
}

#[test]
fn b8_tanh_gradient() {
    let mut g = GradGraph::new();
    let x = g.parameter(t(&[0.0]));
    let y = g.tanh_act(x);
    g.backward(y);
    let gx = g.grad(x).unwrap().to_vec();
    // tanh'(0) = 1
    assert!((gx[0] - 1.0).abs() < 1e-10, "got {}", gx[0]);
}

#[test]
fn b8_vector_ops() {
    // sin of a 3-element vector, then sum
    let mut g = GradGraph::new();
    let x = g.parameter(t(&[0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI]));
    let y = g.sin(x);
    let loss = g.sum(y);
    g.backward(loss);
    let gx = g.grad(x).unwrap().to_vec();
    // d/dx sin(x) = cos(x)
    assert!((gx[0] - 1.0).abs() < 1e-10); // cos(0) = 1
    assert!(gx[1].abs() < 1e-10); // cos(pi/2) = 0
    assert!((gx[2] - (-1.0)).abs() < 1e-10); // cos(pi) = -1
}

#[test]
fn b8_chain_sigmoid_pow() {
    // f(x) = sigmoid(x)^2, sum
    let mut g = GradGraph::new();
    let x = g.parameter(t(&[0.0]));
    let s = g.sigmoid(x);
    let p = g.pow(s, 2.0);
    let loss = g.sum(p);
    g.backward(loss);
    let gx = g.grad(x).unwrap().to_vec();
    // f'(x) = 2*sigmoid(x) * sigmoid'(x) = 2 * 0.5 * 0.25 = 0.25
    assert!((gx[0] - 0.25).abs() < 1e-10, "got {}", gx[0]);
}

#[test]
fn b8_determinism() {
    let run = || {
        let mut g = GradGraph::new();
        let x = g.parameter(t(&[1.5, -2.3, 0.7]));
        let s = g.sin(x);
        let r = g.relu(s);
        let loss = g.sum(r);
        g.backward(loss);
        g.grad(x).unwrap().to_vec()
    };
    let r1 = run();
    let r2 = run();
    for (a, b) in r1.iter().zip(r2.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn b8_mul_then_tanh() {
    // f(x) = tanh(2*x), f'(x) = 2 * (1 - tanh(2x)^2)
    let mut g = GradGraph::new();
    let x = g.parameter(t(&[0.0]));
    let two = g.input(t(&[2.0]));
    let prod = g.mul(x, two);
    let y = g.tanh_act(prod);
    g.backward(y);
    let gx = g.grad(x).unwrap().to_vec();
    // tanh(0) = 0, f'(0) = 2 * (1 - 0) = 2
    assert!((gx[0] - 2.0).abs() < 1e-10, "got {}", gx[0]);
}

#[test]
fn b8_sqrt_chain() {
    // f(x) = sqrt(x^2) = |x|, f'(x) = sign(x) for x > 0
    let mut g = GradGraph::new();
    let x = g.parameter(t(&[3.0]));
    let sq = g.pow(x, 2.0);
    let r = g.sqrt(sq);
    g.backward(r);
    let gx = g.grad(x).unwrap().to_vec();
    // d/dx sqrt(x^2) = x/|x| = 1.0 for x=3
    assert!((gx[0] - 1.0).abs() < 1e-8, "got {}", gx[0]);
}

#[test]
fn b8_cos_squared_plus_sin_squared() {
    // f(x) = sin(x)^2 + cos(x)^2 = 1, f'(x) = 0
    let mut g = GradGraph::new();
    let x = g.parameter(t(&[1.5]));
    let s = g.sin(x);
    let c = g.cos(x);
    let s2 = g.mul(s, s);
    let c2 = g.mul(c, c);
    let sum = g.add(s2, c2);
    g.backward(sum);
    let gx = g.grad(x).unwrap().to_vec();
    assert!(gx[0].abs() < 1e-10, "f'(x) should be 0, got {}", gx[0]);
}
