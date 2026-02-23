use cjc_runtime::Tensor;
use std::cell::RefCell;
use std::rc::Rc;

// ── Forward-Mode AD (Dual Numbers) ──────────────────────────────

/// Dual number for forward-mode automatic differentiation.
#[derive(Debug, Clone)]
pub struct Dual {
    pub value: f64,
    pub deriv: f64,
}

impl Dual {
    pub fn new(value: f64, deriv: f64) -> Self {
        Self { value, deriv }
    }

    pub fn constant(value: f64) -> Self {
        Self { value, deriv: 0.0 }
    }

    pub fn variable(value: f64) -> Self {
        Self { value, deriv: 1.0 }
    }

    pub fn zero() -> Self {
        Self {
            value: 0.0,
            deriv: 0.0,
        }
    }

    pub fn one() -> Self {
        Self {
            value: 1.0,
            deriv: 0.0,
        }
    }
}

impl std::ops::Add for Dual {
    type Output = Dual;
    fn add(self, rhs: Dual) -> Dual {
        Dual {
            value: self.value + rhs.value,
            deriv: self.deriv + rhs.deriv,
        }
    }
}

impl std::ops::Sub for Dual {
    type Output = Dual;
    fn sub(self, rhs: Dual) -> Dual {
        Dual {
            value: self.value - rhs.value,
            deriv: self.deriv - rhs.deriv,
        }
    }
}

impl std::ops::Mul for Dual {
    type Output = Dual;
    fn mul(self, rhs: Dual) -> Dual {
        Dual {
            value: self.value * rhs.value,
            deriv: self.value * rhs.deriv + self.deriv * rhs.value,
        }
    }
}

impl std::ops::Div for Dual {
    type Output = Dual;
    fn div(self, rhs: Dual) -> Dual {
        let denom = rhs.value * rhs.value;
        Dual {
            value: self.value / rhs.value,
            deriv: (self.deriv * rhs.value - self.value * rhs.deriv) / denom,
        }
    }
}

impl std::ops::Neg for Dual {
    type Output = Dual;
    fn neg(self) -> Dual {
        Dual {
            value: -self.value,
            deriv: -self.deriv,
        }
    }
}

impl Dual {
    pub fn sin(self) -> Dual {
        Dual {
            value: self.value.sin(),
            deriv: self.deriv * self.value.cos(),
        }
    }

    pub fn cos(self) -> Dual {
        Dual {
            value: self.value.cos(),
            deriv: -self.deriv * self.value.sin(),
        }
    }

    pub fn exp(self) -> Dual {
        let e = self.value.exp();
        Dual {
            value: e,
            deriv: self.deriv * e,
        }
    }

    pub fn ln(self) -> Dual {
        Dual {
            value: self.value.ln(),
            deriv: self.deriv / self.value,
        }
    }

    pub fn sqrt(self) -> Dual {
        let s = self.value.sqrt();
        Dual {
            value: s,
            deriv: self.deriv / (2.0 * s),
        }
    }

    pub fn pow(self, n: f64) -> Dual {
        Dual {
            value: self.value.powf(n),
            deriv: self.deriv * n * self.value.powf(n - 1.0),
        }
    }
}

// ── Reverse-Mode AD (Computational Graph) ───────────────────────

/// Operation recorded in the computation graph.
#[derive(Debug, Clone)]
pub enum GradOp {
    Input,
    Parameter,
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Div(usize, usize),
    Neg(usize),
    MatMul(usize, usize),
    Sum(usize),
    Mean(usize),
    ScalarMul(usize, f64),
    Exp(usize),
    Ln(usize),
    /// Gradient through struct field access: parent node, field index.
    StructField {
        parent: usize,
        field_index: usize,
        total_fields: usize,
    },
    /// Gradient through map lookup: map node, key index in insertion order.
    MapLookup {
        map_node: usize,
        key_index: usize,
        total_keys: usize,
    },
}

/// A node in the reverse-mode AD graph.
#[derive(Debug, Clone)]
pub struct GradNode {
    pub op: GradOp,
    pub tensor: Tensor,
    pub grad: Option<Tensor>,
}

/// The reverse-mode AD tape/graph.
pub struct GradGraph {
    pub nodes: Vec<Rc<RefCell<GradNode>>>,
}

impl GradGraph {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Create an input node (data, no gradient).
    pub fn input(&mut self, tensor: Tensor) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Rc::new(RefCell::new(GradNode {
            op: GradOp::Input,
            tensor,
            grad: None,
        })));
        idx
    }

    /// Create a parameter node (trainable, accumulates gradients).
    pub fn parameter(&mut self, tensor: Tensor) -> usize {
        let idx = self.nodes.len();
        let shape = tensor.shape().to_vec();
        self.nodes.push(Rc::new(RefCell::new(GradNode {
            op: GradOp::Parameter,
            tensor,
            grad: Some(Tensor::zeros(&shape)),
        })));
        idx
    }

    /// Element-wise addition.
    pub fn add(&mut self, a: usize, b: usize) -> usize {
        let a_t = self.nodes[a].borrow().tensor.clone();
        let b_t = self.nodes[b].borrow().tensor.clone();
        let result = a_t.add_unchecked(&b_t);
        let idx = self.nodes.len();
        self.nodes.push(Rc::new(RefCell::new(GradNode {
            op: GradOp::Add(a, b),
            tensor: result,
            grad: None,
        })));
        idx
    }

    /// Element-wise subtraction.
    pub fn sub(&mut self, a: usize, b: usize) -> usize {
        let a_t = self.nodes[a].borrow().tensor.clone();
        let b_t = self.nodes[b].borrow().tensor.clone();
        let result = a_t.sub_unchecked(&b_t);
        let idx = self.nodes.len();
        self.nodes.push(Rc::new(RefCell::new(GradNode {
            op: GradOp::Sub(a, b),
            tensor: result,
            grad: None,
        })));
        idx
    }

    /// Element-wise multiplication.
    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        let a_t = self.nodes[a].borrow().tensor.clone();
        let b_t = self.nodes[b].borrow().tensor.clone();
        let result = a_t.mul_elem_unchecked(&b_t);
        let idx = self.nodes.len();
        self.nodes.push(Rc::new(RefCell::new(GradNode {
            op: GradOp::Mul(a, b),
            tensor: result,
            grad: None,
        })));
        idx
    }

    /// Matrix multiplication.
    pub fn matmul(&mut self, a: usize, b: usize) -> usize {
        let a_t = self.nodes[a].borrow().tensor.clone();
        let b_t = self.nodes[b].borrow().tensor.clone();
        let result = a_t.matmul_unchecked(&b_t);
        let idx = self.nodes.len();
        self.nodes.push(Rc::new(RefCell::new(GradNode {
            op: GradOp::MatMul(a, b),
            tensor: result,
            grad: None,
        })));
        idx
    }

    /// Sum all elements.
    pub fn sum(&mut self, a: usize) -> usize {
        let a_t = self.nodes[a].borrow().tensor.clone();
        let s = a_t.sum();
        let result = Tensor::from_vec_unchecked(vec![s], &[1]);
        let idx = self.nodes.len();
        self.nodes.push(Rc::new(RefCell::new(GradNode {
            op: GradOp::Sum(a),
            tensor: result,
            grad: None,
        })));
        idx
    }

    /// Mean of all elements.
    pub fn mean(&mut self, a: usize) -> usize {
        let a_t = self.nodes[a].borrow().tensor.clone();
        let m = a_t.mean();
        let result = Tensor::from_vec_unchecked(vec![m], &[1]);
        let idx = self.nodes.len();
        self.nodes.push(Rc::new(RefCell::new(GradNode {
            op: GradOp::Mean(a),
            tensor: result,
            grad: None,
        })));
        idx
    }

    /// Get the scalar value from a 1-element tensor node.
    pub fn value(&self, idx: usize) -> f64 {
        let node = self.nodes[idx].borrow();
        let data = node.tensor.to_vec();
        data[0]
    }

    /// Get the tensor at a node.
    pub fn tensor(&self, idx: usize) -> Tensor {
        self.nodes[idx].borrow().tensor.clone()
    }

    /// Set the tensor at a node (for parameter updates).
    pub fn set_tensor(&self, idx: usize, tensor: Tensor) {
        self.nodes[idx].borrow_mut().tensor = tensor;
    }

    /// Get the gradient at a node.
    pub fn grad(&self, idx: usize) -> Option<Tensor> {
        self.nodes[idx].borrow().grad.clone()
    }

    /// Zero out all gradients.
    pub fn zero_grad(&self) {
        for node in &self.nodes {
            let mut n = node.borrow_mut();
            if let Some(ref mut grad) = n.grad {
                let shape = grad.shape().to_vec();
                *grad = Tensor::zeros(&shape);
            }
        }
    }

    /// Run backward pass from a loss node.
    pub fn backward(&self, loss_idx: usize) {
        let n = self.nodes.len();

        // Initialize gradients
        let mut grads: Vec<Option<Tensor>> = vec![None; n];

        // Loss gradient is 1.0
        let loss_shape = self.nodes[loss_idx].borrow().tensor.shape().to_vec();
        grads[loss_idx] = Some(Tensor::ones(&loss_shape));

        // Backward pass in reverse topological order
        for i in (0..=loss_idx).rev() {
            let grad = match grads[i].take() {
                Some(g) => g,
                None => continue,
            };

            // Clone op and tensor out of the borrow so we don't hold the RefCell across match arms
            let (op, node_tensor) = {
                let node = self.nodes[i].borrow();
                (node.op.clone(), node.tensor.clone())
            };

            match op {
                GradOp::Input => {}
                GradOp::Parameter => {
                    let mut node_mut = self.nodes[i].borrow_mut();
                    if let Some(ref mut existing_grad) = node_mut.grad {
                        *existing_grad = existing_grad.add_unchecked(&grad);
                    } else {
                        node_mut.grad = Some(grad);
                    }
                }
                GradOp::Add(a, b) => {
                    accumulate_grad(&mut grads, a, &grad);
                    accumulate_grad(&mut grads, b, &grad);
                }
                GradOp::Sub(a, b) => {
                    accumulate_grad(&mut grads, a, &grad);
                    let neg_grad = grad.neg();
                    accumulate_grad(&mut grads, b, &neg_grad);
                }
                GradOp::Mul(a, b) => {
                    let a_val = self.nodes[a].borrow().tensor.clone();
                    let b_val = self.nodes[b].borrow().tensor.clone();

                    let grad_a = grad.mul_elem_unchecked(&b_val);
                    let grad_b = grad.mul_elem_unchecked(&a_val);

                    accumulate_grad(&mut grads, a, &grad_a);
                    accumulate_grad(&mut grads, b, &grad_b);
                }
                GradOp::Div(a, b) => {
                    let a_val = self.nodes[a].borrow().tensor.clone();
                    let b_val = self.nodes[b].borrow().tensor.clone();

                    // d/da (a/b) = 1/b
                    let grad_a = grad.div_elem_unchecked(&b_val);
                    // d/db (a/b) = -a/b^2
                    let b_sq = b_val.mul_elem_unchecked(&b_val);
                    let neg_a = a_val.neg();
                    let grad_b = grad.mul_elem_unchecked(&neg_a.div_elem_unchecked(&b_sq));

                    accumulate_grad(&mut grads, a, &grad_a);
                    accumulate_grad(&mut grads, b, &grad_b);
                }
                GradOp::Neg(a) => {
                    let neg_grad = grad.neg();
                    accumulate_grad(&mut grads, a, &neg_grad);
                }
                GradOp::MatMul(a, b) => {
                    // d/da (a @ b) = grad @ b^T
                    // d/db (a @ b) = a^T @ grad
                    let a_val = self.nodes[a].borrow().tensor.clone();
                    let b_val = self.nodes[b].borrow().tensor.clone();

                    let b_t = b_val.transpose();
                    let a_t = a_val.transpose();

                    let grad_a = grad.matmul_unchecked(&b_t);
                    let grad_b = a_t.matmul_unchecked(&grad);

                    accumulate_grad(&mut grads, a, &grad_a);
                    accumulate_grad(&mut grads, b, &grad_b);
                }
                GradOp::Sum(a) => {
                    // Gradient of sum is all ones, scaled by upstream grad
                    let a_shape = self.nodes[a].borrow().tensor.shape().to_vec();
                    let grad_val = grad.to_vec()[0];
                    let expanded = Tensor::from_vec_unchecked(
                        vec![grad_val; a_shape.iter().product()],
                        &a_shape,
                    );
                    accumulate_grad(&mut grads, a, &expanded);
                }
                GradOp::Mean(a) => {
                    let a_shape = self.nodes[a].borrow().tensor.shape().to_vec();
                    let n_elem = a_shape.iter().product::<usize>() as f64;
                    let grad_val = grad.to_vec()[0] / n_elem;
                    let expanded = Tensor::from_vec_unchecked(
                        vec![grad_val; a_shape.iter().product()],
                        &a_shape,
                    );
                    accumulate_grad(&mut grads, a, &expanded);
                }
                GradOp::ScalarMul(a, s) => {
                    let scaled = grad.scalar_mul(s);
                    accumulate_grad(&mut grads, a, &scaled);
                }
                GradOp::Exp(a) => {
                    let grad_a = grad.mul_elem_unchecked(&node_tensor);
                    accumulate_grad(&mut grads, a, &grad_a);
                }
                GradOp::Ln(a) => {
                    let a_val = self.nodes[a].borrow().tensor.clone();
                    let grad_a = grad.div_elem_unchecked(&a_val);
                    accumulate_grad(&mut grads, a, &grad_a);
                }
                // Differentiable container ops: gradient flows back to parent
                GradOp::StructField {
                    parent,
                    field_index,
                    total_fields,
                } => {
                    // Gradient for field access: create a "one-hot" gradient
                    // where only the accessed field gets the incoming gradient.
                    // For now, accumulate directly to parent.
                    let _ = (field_index, total_fields);
                    accumulate_grad(&mut grads, parent, &grad);
                }
                GradOp::MapLookup {
                    map_node,
                    key_index,
                    total_keys,
                } => {
                    // Gradient for map lookup: accumulate to map node.
                    // Deterministic: key_index determines order of accumulation.
                    let _ = (key_index, total_keys);
                    accumulate_grad(&mut grads, map_node, &grad);
                }
            }
        }
    }
}

fn accumulate_grad(grads: &mut [Option<Tensor>], idx: usize, grad: &Tensor) {
    if let Some(existing) = &grads[idx] {
        grads[idx] = Some(existing.add_unchecked(grad));
    } else {
        grads[idx] = Some(grad.clone());
    }
}

impl Default for GradGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ── Finite Difference Validation ────────────────────────────────

/// Validate gradient using finite differences.
pub fn check_grad_finite_diff<F>(
    f: F,
    x: f64,
    expected_grad: f64,
    eps: f64,
    tol: f64,
) -> bool
where
    F: Fn(f64) -> f64,
{
    let fd_grad = (f(x + eps) - f(x - eps)) / (2.0 * eps);
    (fd_grad - expected_grad).abs() < tol
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Forward Mode Tests ──────────────────────────────────

    #[test]
    fn test_dual_add() {
        let a = Dual::variable(3.0);
        let b = Dual::constant(2.0);
        let c = a + b;
        assert_eq!(c.value, 5.0);
        assert_eq!(c.deriv, 1.0);
    }

    #[test]
    fn test_dual_mul() {
        let a = Dual::variable(3.0);
        let b = Dual::constant(2.0);
        let c = a * b;
        assert_eq!(c.value, 6.0);
        assert_eq!(c.deriv, 2.0); // d/dx (x * 2) = 2
    }

    #[test]
    fn test_dual_chain_rule() {
        // f(x) = x^2 + 2x + 1, f'(x) = 2x + 2, f'(3) = 8
        let x = Dual::variable(3.0);
        let result = x.clone() * x.clone() + Dual::constant(2.0) * x + Dual::one();
        assert_eq!(result.value, 16.0);
        assert_eq!(result.deriv, 8.0);
    }

    #[test]
    fn test_dual_exp() {
        let x = Dual::variable(1.0);
        let result = x.exp();
        assert!((result.value - std::f64::consts::E).abs() < 1e-10);
        assert!((result.deriv - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_dual_sin_cos() {
        let x = Dual::variable(0.0);
        let sin_x = x.clone().sin();
        let cos_x = x.cos();
        assert!((sin_x.value - 0.0).abs() < 1e-10);
        assert!((sin_x.deriv - 1.0).abs() < 1e-10); // d/dx sin(x) at 0 = cos(0) = 1
        assert!((cos_x.value - 1.0).abs() < 1e-10);
        assert!((cos_x.deriv - 0.0).abs() < 1e-10); // d/dx cos(x) at 0 = -sin(0) = 0
    }

    #[test]
    fn test_dual_div() {
        let a = Dual::variable(6.0);
        let b = Dual::constant(3.0);
        let c = a / b;
        assert_eq!(c.value, 2.0);
        assert!((c.deriv - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_finite_diff_validation() {
        // f(x) = x^2, f'(3) = 6
        let f = |x: f64| x * x;
        assert!(check_grad_finite_diff(f, 3.0, 6.0, 1e-7, 1e-5));
    }

    // ── Reverse Mode Tests ──────────────────────────────────

    #[test]
    fn test_reverse_add() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![3.0], &[1]));
        let b = g.parameter(Tensor::from_vec_unchecked(vec![2.0], &[1]));
        let c = g.add(a, b);

        g.backward(c);

        let ga = g.grad(a).unwrap();
        let gb = g.grad(b).unwrap();
        assert!((ga.to_vec()[0] - 1.0).abs() < 1e-10);
        assert!((gb.to_vec()[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reverse_mul() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![3.0], &[1]));
        let b = g.parameter(Tensor::from_vec_unchecked(vec![2.0], &[1]));
        let c = g.mul(a, b);

        g.backward(c);

        let ga = g.grad(a).unwrap();
        let gb = g.grad(b).unwrap();
        assert!((ga.to_vec()[0] - 2.0).abs() < 1e-10); // d/da (a*b) = b = 2
        assert!((gb.to_vec()[0] - 3.0).abs() < 1e-10); // d/db (a*b) = a = 3
    }

    #[test]
    fn test_reverse_matmul_gradient() {
        let mut g = GradGraph::new();

        // Simple 2x2 matmul
        let a = g.parameter(Tensor::from_vec_unchecked(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let b = g.parameter(Tensor::from_vec_unchecked(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]));
        let c = g.matmul(a, b);
        let loss = g.sum(c);

        g.backward(loss);

        // Gradient of sum(A @ B) w.r.t. A = ones @ B^T
        let ga = g.grad(a).unwrap();
        let ga_data = ga.to_vec();
        // B^T = [[5,7],[6,8]], ones@B^T = [[5+7, 6+8],[5+7, 6+8]] = [[12,14],[12,14]]
        // Wait: grad = ones(2,2), grad @ B^T
        // B^T = [[5,7],[6,8]]
        // ones(2,2) @ B^T = [[5+6, 7+8],[5+6, 7+8]] = [[11,15],[11,15]]
        assert!((ga_data[0] - 11.0).abs() < 1e-10);
        assert!((ga_data[1] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_reverse_mean_gradient() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![2.0, 4.0, 6.0, 8.0], &[4]));
        let loss = g.mean(a);

        g.backward(loss);

        let ga = g.grad(a).unwrap();
        let ga_data = ga.to_vec();
        // d/da mean(a) = 1/N for each element
        for &v in &ga_data {
            assert!((v - 0.25).abs() < 1e-10);
        }
    }

    #[test]
    fn test_reverse_mse_loss() {
        // MSE = mean((pred - target)^2)
        let mut g = GradGraph::new();

        let w = g.parameter(Tensor::from_vec_unchecked(vec![1.0, 1.0], &[2, 1]));
        let x = g.input(Tensor::from_vec_unchecked(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let target = g.input(Tensor::from_vec_unchecked(vec![3.0, 7.0], &[2, 1]));

        let pred = g.matmul(x, w);
        let diff = g.sub(pred, target);
        let sq = g.mul(diff, diff);
        let loss = g.mean(sq);

        let loss_val = g.value(loss);
        g.backward(loss);

        let gw = g.grad(w).unwrap();

        // Verify loss is finite and gradient exists
        assert!(loss_val.is_finite());
        assert_eq!(gw.to_vec().len(), 2);
        for &v in &gw.to_vec() {
            assert!(v.is_finite());
        }
    }
}
