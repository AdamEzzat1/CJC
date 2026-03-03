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
    // Phase B8: Transcendental & activation ops
    Sin(usize),
    Cos(usize),
    Sqrt(usize),
    Pow(usize, f64),
    Sigmoid(usize),
    Relu(usize),
    TanhAct(usize),
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

    // ── Phase B8: Transcendental & activation forward ops ──

    /// Element-wise sine.
    pub fn sin(&mut self, a: usize) -> usize {
        let a_t = self.nodes[a].borrow().tensor.clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| x.sin()).collect(),
            a_t.shape(),
        );
        let idx = self.nodes.len();
        self.nodes.push(Rc::new(RefCell::new(GradNode {
            op: GradOp::Sin(a),
            tensor: result,
            grad: None,
        })));
        idx
    }

    /// Element-wise cosine.
    pub fn cos(&mut self, a: usize) -> usize {
        let a_t = self.nodes[a].borrow().tensor.clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| x.cos()).collect(),
            a_t.shape(),
        );
        let idx = self.nodes.len();
        self.nodes.push(Rc::new(RefCell::new(GradNode {
            op: GradOp::Cos(a),
            tensor: result,
            grad: None,
        })));
        idx
    }

    /// Element-wise square root.
    pub fn sqrt(&mut self, a: usize) -> usize {
        let a_t = self.nodes[a].borrow().tensor.clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| x.sqrt()).collect(),
            a_t.shape(),
        );
        let idx = self.nodes.len();
        self.nodes.push(Rc::new(RefCell::new(GradNode {
            op: GradOp::Sqrt(a),
            tensor: result,
            grad: None,
        })));
        idx
    }

    /// Element-wise power with constant exponent.
    pub fn pow(&mut self, a: usize, n: f64) -> usize {
        let a_t = self.nodes[a].borrow().tensor.clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| x.powf(n)).collect(),
            a_t.shape(),
        );
        let idx = self.nodes.len();
        self.nodes.push(Rc::new(RefCell::new(GradNode {
            op: GradOp::Pow(a, n),
            tensor: result,
            grad: None,
        })));
        idx
    }

    /// Sigmoid activation: 1 / (1 + exp(-x)).
    pub fn sigmoid(&mut self, a: usize) -> usize {
        let a_t = self.nodes[a].borrow().tensor.clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
            a_t.shape(),
        );
        let idx = self.nodes.len();
        self.nodes.push(Rc::new(RefCell::new(GradNode {
            op: GradOp::Sigmoid(a),
            tensor: result,
            grad: None,
        })));
        idx
    }

    /// ReLU activation: max(0, x).
    pub fn relu(&mut self, a: usize) -> usize {
        let a_t = self.nodes[a].borrow().tensor.clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect(),
            a_t.shape(),
        );
        let idx = self.nodes.len();
        self.nodes.push(Rc::new(RefCell::new(GradNode {
            op: GradOp::Relu(a),
            tensor: result,
            grad: None,
        })));
        idx
    }

    /// Tanh activation.
    pub fn tanh_act(&mut self, a: usize) -> usize {
        let a_t = self.nodes[a].borrow().tensor.clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| x.tanh()).collect(),
            a_t.shape(),
        );
        let idx = self.nodes.len();
        self.nodes.push(Rc::new(RefCell::new(GradNode {
            op: GradOp::TanhAct(a),
            tensor: result,
            grad: None,
        })));
        idx
    }

    // ── Phase C1: Missing forward methods ──

    /// Element-wise division: a / b.
    /// GradOp::Div(a, b) already has backward implementation.
    pub fn div(&mut self, a: usize, b: usize) -> usize {
        let a_tensor = self.nodes[a].borrow().tensor.clone();
        let b_tensor = self.nodes[b].borrow().tensor.clone();
        let result = a_tensor.div_elem_unchecked(&b_tensor);
        let node = GradNode { op: GradOp::Div(a, b), tensor: result, grad: None };
        self.nodes.push(Rc::new(RefCell::new(node)));
        self.nodes.len() - 1
    }

    /// Element-wise negation: -a.
    /// GradOp::Neg(a) already has backward implementation.
    pub fn neg(&mut self, a: usize) -> usize {
        let a_tensor = self.nodes[a].borrow().tensor.clone();
        let result = a_tensor.neg();
        let node = GradNode { op: GradOp::Neg(a), tensor: result, grad: None };
        self.nodes.push(Rc::new(RefCell::new(node)));
        self.nodes.len() - 1
    }

    /// Scalar multiply: a * s (where s is an f64 constant).
    /// GradOp::ScalarMul(a, s) already has backward implementation.
    pub fn scalar_mul(&mut self, a: usize, s: f64) -> usize {
        let a_tensor = self.nodes[a].borrow().tensor.clone();
        let result = a_tensor.scalar_mul(s);
        let node = GradNode { op: GradOp::ScalarMul(a, s), tensor: result, grad: None };
        self.nodes.push(Rc::new(RefCell::new(node)));
        self.nodes.len() - 1
    }

    /// Element-wise exponential: exp(a).
    /// GradOp::Exp(a) already has backward implementation.
    pub fn exp(&mut self, a: usize) -> usize {
        let a_tensor = self.nodes[a].borrow().tensor.clone();
        let result = Tensor::from_vec_unchecked(
            a_tensor.to_vec().iter().map(|x| x.exp()).collect(),
            a_tensor.shape(),
        );
        let node = GradNode { op: GradOp::Exp(a), tensor: result, grad: None };
        self.nodes.push(Rc::new(RefCell::new(node)));
        self.nodes.len() - 1
    }

    /// Element-wise natural logarithm: ln(a).
    /// GradOp::Ln(a) already has backward implementation.
    pub fn ln(&mut self, a: usize) -> usize {
        let a_tensor = self.nodes[a].borrow().tensor.clone();
        let result = Tensor::from_vec_unchecked(
            a_tensor.to_vec().iter().map(|x| x.ln()).collect(),
            a_tensor.shape(),
        );
        let node = GradNode { op: GradOp::Ln(a), tensor: result, grad: None };
        self.nodes.push(Rc::new(RefCell::new(node)));
        self.nodes.len() - 1
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
                // Phase B8: Transcendental & activation backward
                GradOp::Sin(a) => {
                    let a_val = self.nodes[a].borrow().tensor.clone();
                    let cos_a = Tensor::from_vec_unchecked(
                        a_val.to_vec().iter().map(|&x| x.cos()).collect(),
                        a_val.shape(),
                    );
                    let grad_a = grad.mul_elem_unchecked(&cos_a);
                    accumulate_grad(&mut grads, a, &grad_a);
                }
                GradOp::Cos(a) => {
                    let a_val = self.nodes[a].borrow().tensor.clone();
                    let neg_sin_a = Tensor::from_vec_unchecked(
                        a_val.to_vec().iter().map(|&x| -x.sin()).collect(),
                        a_val.shape(),
                    );
                    let grad_a = grad.mul_elem_unchecked(&neg_sin_a);
                    accumulate_grad(&mut grads, a, &grad_a);
                }
                GradOp::Sqrt(a) => {
                    // d/dx sqrt(x) = 0.5 / sqrt(x) = 0.5 / node_tensor
                    let inv_2sqrt = Tensor::from_vec_unchecked(
                        node_tensor.to_vec().iter().map(|&x| 0.5 / x).collect(),
                        node_tensor.shape(),
                    );
                    let grad_a = grad.mul_elem_unchecked(&inv_2sqrt);
                    accumulate_grad(&mut grads, a, &grad_a);
                }
                GradOp::Pow(a, n) => {
                    let a_val = self.nodes[a].borrow().tensor.clone();
                    let coeff = Tensor::from_vec_unchecked(
                        a_val.to_vec().iter().map(|&x| n * x.powf(n - 1.0)).collect(),
                        a_val.shape(),
                    );
                    let grad_a = grad.mul_elem_unchecked(&coeff);
                    accumulate_grad(&mut grads, a, &grad_a);
                }
                GradOp::Sigmoid(a) => {
                    // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
                    let sig = &node_tensor;
                    let one_minus = Tensor::from_vec_unchecked(
                        sig.to_vec().iter().map(|&s| 1.0 - s).collect(),
                        sig.shape(),
                    );
                    let local = sig.mul_elem_unchecked(&one_minus);
                    let grad_a = grad.mul_elem_unchecked(&local);
                    accumulate_grad(&mut grads, a, &grad_a);
                }
                GradOp::Relu(a) => {
                    let a_val = self.nodes[a].borrow().tensor.clone();
                    let mask = Tensor::from_vec_unchecked(
                        a_val.to_vec().iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect(),
                        a_val.shape(),
                    );
                    let grad_a = grad.mul_elem_unchecked(&mask);
                    accumulate_grad(&mut grads, a, &grad_a);
                }
                GradOp::TanhAct(a) => {
                    // tanh'(x) = 1 - tanh(x)^2
                    let t = &node_tensor;
                    let one_minus_sq = Tensor::from_vec_unchecked(
                        t.to_vec().iter().map(|&x| 1.0 - x * x).collect(),
                        t.shape(),
                    );
                    let grad_a = grad.mul_elem_unchecked(&one_minus_sq);
                    accumulate_grad(&mut grads, a, &grad_a);
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

    // ── Phase B8: Reverse Mode Transcendental & Activation Tests ──

    #[test]
    fn test_reverse_sin() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![0.0], &[1]));
        let b = g.sin(a);
        g.backward(b);
        let ga = g.grad(a).unwrap();
        // d/dx sin(x) at x=0 = cos(0) = 1.0
        assert!((ga.to_vec()[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reverse_cos() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![0.0], &[1]));
        let b = g.cos(a);
        g.backward(b);
        let ga = g.grad(a).unwrap();
        // d/dx cos(x) at x=0 = -sin(0) = 0.0
        assert!(ga.to_vec()[0].abs() < 1e-10);
    }

    #[test]
    fn test_reverse_sqrt() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![4.0], &[1]));
        let b = g.sqrt(a);
        g.backward(b);
        let ga = g.grad(a).unwrap();
        // d/dx sqrt(x) at x=4 = 1/(2*2) = 0.25
        assert!((ga.to_vec()[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_reverse_pow() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![2.0], &[1]));
        let b = g.pow(a, 3.0); // x^3
        g.backward(b);
        let ga = g.grad(a).unwrap();
        // d/dx x^3 at x=2 = 3*4 = 12.0
        assert!((ga.to_vec()[0] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_reverse_sigmoid() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![0.0], &[1]));
        let b = g.sigmoid(a);
        g.backward(b);
        let ga = g.grad(a).unwrap();
        // sigmoid(0) = 0.5, sigmoid'(0) = 0.5 * 0.5 = 0.25
        assert!((ga.to_vec()[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_reverse_relu_positive() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![3.0], &[1]));
        let b = g.relu(a);
        g.backward(b);
        let ga = g.grad(a).unwrap();
        // relu'(3) = 1.0
        assert!((ga.to_vec()[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reverse_relu_negative() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![-2.0], &[1]));
        let b = g.relu(a);
        g.backward(b);
        let ga = g.grad(a).unwrap();
        // relu'(-2) = 0.0
        assert!(ga.to_vec()[0].abs() < 1e-10);
    }

    #[test]
    fn test_reverse_tanh() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![0.0], &[1]));
        let b = g.tanh_act(a);
        g.backward(b);
        let ga = g.grad(a).unwrap();
        // tanh'(0) = 1 - tanh(0)^2 = 1 - 0 = 1.0
        assert!((ga.to_vec()[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reverse_sin_cos_chain() {
        // f(x) = sin(cos(x)), f'(x) = cos(cos(x)) * (-sin(x))
        // at x=1: f'(1) = cos(cos(1)) * (-sin(1))
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![1.0], &[1]));
        let c = g.cos(a);
        let s = g.sin(c);
        g.backward(s);
        let ga = g.grad(a).unwrap();
        let expected = 1.0_f64.cos().cos() * (-1.0_f64.sin());
        assert!((ga.to_vec()[0] - expected).abs() < 1e-10, "got {}, expected {expected}", ga.to_vec()[0]);
    }

    #[test]
    fn test_reverse_sigmoid_sum() {
        // f(x) = sum(sigmoid(x)) for vector x
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![0.0, 1.0, -1.0], &[3]));
        let s = g.sigmoid(a);
        let loss = g.sum(s);
        g.backward(loss);
        let ga = g.grad(a).unwrap();
        let ga_data = ga.to_vec();
        // sigmoid'(0) = 0.25, sigmoid'(1) = sig(1)*(1-sig(1)), sigmoid'(-1) = sig(-1)*(1-sig(-1))
        let sig1 = 1.0 / (1.0 + (-1.0_f64).exp());
        let sig_neg1 = 1.0 / (1.0 + 1.0_f64.exp());
        assert!((ga_data[0] - 0.25).abs() < 1e-10);
        assert!((ga_data[1] - sig1 * (1.0 - sig1)).abs() < 1e-10);
        assert!((ga_data[2] - sig_neg1 * (1.0 - sig_neg1)).abs() < 1e-10);
    }

    #[test]
    fn test_b8_determinism() {
        let mut g1 = GradGraph::new();
        let a1 = g1.parameter(Tensor::from_vec_unchecked(vec![1.5], &[1]));
        let s1 = g1.sin(a1);
        g1.backward(s1);
        let ga1 = g1.grad(a1).unwrap().to_vec()[0];

        let mut g2 = GradGraph::new();
        let a2 = g2.parameter(Tensor::from_vec_unchecked(vec![1.5], &[1]));
        let s2 = g2.sin(a2);
        g2.backward(s2);
        let ga2 = g2.grad(a2).unwrap().to_vec()[0];

        assert_eq!(ga1.to_bits(), ga2.to_bits());
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

    // ── Phase C1: Reverse Mode Tests for New Forward Methods ──

    #[test]
    fn test_reverse_div() {
        // f(x) = x / 2, f'(x) = 0.5
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![6.0], &[1]));
        let b = g.input(Tensor::from_vec_unchecked(vec![2.0], &[1]));
        let c = g.div(a, b);
        g.backward(c);
        let ga = g.grad(a).unwrap();
        assert!((ga.to_vec()[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_reverse_neg() {
        // f(x) = -x, f'(x) = -1
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![3.0], &[1]));
        let c = g.neg(a);
        g.backward(c);
        let ga = g.grad(a).unwrap();
        assert!((ga.to_vec()[0] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_reverse_scalar_mul() {
        // f(x) = 3x, f'(x) = 3
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![2.0], &[1]));
        let c = g.scalar_mul(a, 3.0);
        g.backward(c);
        let ga = g.grad(a).unwrap();
        assert!((ga.to_vec()[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_reverse_exp() {
        // f(x) = exp(x), f'(x) = exp(x)
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![1.0], &[1]));
        let c = g.exp(a);
        g.backward(c);
        let ga = g.grad(a).unwrap();
        assert!((ga.to_vec()[0] - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_reverse_ln() {
        // f(x) = ln(x), f'(x) = 1/x, at x=2 → 0.5
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![2.0], &[1]));
        let c = g.ln(a);
        g.backward(c);
        let ga = g.grad(a).unwrap();
        assert!((ga.to_vec()[0] - 0.5).abs() < 1e-10);
    }
}
