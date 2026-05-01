//! Automatic differentiation for CJC.
//!
//! Provides forward-mode differentiation via dual numbers and reverse-mode
//! differentiation via a computation tape. Supports `grad()`, `jacobian()`,
//! and gradient graph construction for ML training loops.

use cjc_runtime::Tensor;

pub mod idx;
pub mod pinn;
pub mod dispatch;

pub use dispatch::dispatch_grad_graph;
pub use idx::{LayerIdx, NodeIdx, ParamIdx};

// ── Forward-Mode AD (Dual Numbers) ──────────────────────────────

/// Dual number for forward-mode automatic differentiation.
///
/// Carries a primal value and its derivative (tangent) through arithmetic
/// operations so that `f(Dual::variable(x))` yields both `f(x)` and `f'(x)`
/// in a single forward pass.
///
/// # Examples
///
/// ```rust,ignore
/// // Compute f(x) = x^2 and f'(x) at x = 3
/// let x = Dual::variable(3.0);
/// let y = x.clone() * x;
/// assert_eq!(y.value, 9.0);
/// assert_eq!(y.deriv, 6.0);
/// ```
#[derive(Debug, Clone)]
pub struct Dual {
    /// The primal (function) value.
    pub value: f64,
    /// The tangent (derivative) value.
    pub deriv: f64,
}

impl Dual {
    /// Create a dual number with an explicit value and derivative.
    ///
    /// # Arguments
    ///
    /// * `value` - The primal value.
    /// * `deriv` - The tangent (derivative) seed.
    pub fn new(value: f64, deriv: f64) -> Self {
        Self { value, deriv }
    }

    /// Create a dual number representing a constant (derivative = 0).
    ///
    /// # Arguments
    ///
    /// * `value` - The constant value.
    pub fn constant(value: f64) -> Self {
        Self { value, deriv: 0.0 }
    }

    /// Create a dual number representing the independent variable (derivative = 1).
    ///
    /// Use this for the variable with respect to which you are differentiating.
    ///
    /// # Arguments
    ///
    /// * `value` - The point at which to evaluate.
    pub fn variable(value: f64) -> Self {
        Self { value, deriv: 1.0 }
    }

    /// Return the additive identity dual number (value = 0, derivative = 0).
    pub fn zero() -> Self {
        Self {
            value: 0.0,
            deriv: 0.0,
        }
    }

    /// Return the multiplicative identity dual number (value = 1, derivative = 0).
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
    /// Compute the sine, propagating the derivative via the chain rule: `d/dx sin(x) = cos(x)`.
    pub fn sin(self) -> Dual {
        Dual {
            value: self.value.sin(),
            deriv: self.deriv * self.value.cos(),
        }
    }

    /// Compute the cosine, propagating the derivative via the chain rule: `d/dx cos(x) = -sin(x)`.
    pub fn cos(self) -> Dual {
        Dual {
            value: self.value.cos(),
            deriv: -self.deriv * self.value.sin(),
        }
    }

    /// Compute the exponential, propagating the derivative: `d/dx exp(x) = exp(x)`.
    pub fn exp(self) -> Dual {
        let e = self.value.exp();
        Dual {
            value: e,
            deriv: self.deriv * e,
        }
    }

    /// Compute the natural logarithm, propagating the derivative: `d/dx ln(x) = 1/x`.
    pub fn ln(self) -> Dual {
        Dual {
            value: self.value.ln(),
            deriv: self.deriv / self.value,
        }
    }

    /// Compute the square root, propagating the derivative: `d/dx sqrt(x) = 1/(2*sqrt(x))`.
    pub fn sqrt(self) -> Dual {
        let s = self.value.sqrt();
        Dual {
            value: s,
            deriv: self.deriv / (2.0 * s),
        }
    }

    /// Raise to a constant power `n`, propagating the derivative: `d/dx x^n = n * x^(n-1)`.
    ///
    /// # Arguments
    ///
    /// * `n` - The exponent (constant, not differentiated).
    pub fn pow(self, n: f64) -> Dual {
        Dual {
            value: self.value.powf(n),
            deriv: self.deriv * n * self.value.powf(n - 1.0),
        }
    }
}

// ── Reverse-Mode AD (Computational Graph) ───────────────────────

/// Operation recorded in the reverse-mode AD computation graph.
///
/// Each variant stores the node indices of its operands so the backward pass
/// can look up parent tensors and propagate gradients.
#[derive(Debug, Clone)]
pub enum GradOp {
    /// External input data (no gradient accumulated).
    Input,
    /// Trainable parameter (gradients are accumulated here during backward).
    Parameter,
    /// Element-wise addition of two nodes.
    Add(usize, usize),
    /// Element-wise subtraction of two nodes.
    Sub(usize, usize),
    /// Element-wise (Hadamard) multiplication of two nodes.
    Mul(usize, usize),
    /// Element-wise division of two nodes.
    Div(usize, usize),
    /// Element-wise negation.
    Neg(usize),
    /// Matrix multiplication of two 2-D nodes.
    MatMul(usize, usize),
    /// Sum all elements to a scalar `[1]` tensor.
    Sum(usize),
    /// Mean of all elements to a scalar `[1]` tensor.
    Mean(usize),
    /// Multiply every element by a constant scalar.
    ScalarMul(usize, f64),
    /// Element-wise exponential.
    Exp(usize),
    /// Element-wise natural logarithm.
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
    /// Element-wise sine: `d/dx sin(x) = cos(x)`.
    Sin(usize),
    /// Element-wise cosine: `d/dx cos(x) = -sin(x)`.
    Cos(usize),
    /// Element-wise square root: `d/dx sqrt(x) = 1/(2*sqrt(x))`.
    Sqrt(usize),
    /// Element-wise power with a constant exponent: `d/dx x^n = n * x^(n-1)`.
    Pow(usize, f64),
    /// Logistic sigmoid activation: `sigma(x) = 1 / (1 + exp(-x))`.
    Sigmoid(usize),
    /// Rectified linear unit activation: `max(0, x)`.
    Relu(usize),
    /// Hyperbolic tangent activation: `tanh(x)`.
    TanhAct(usize),
    /// Element-wise absolute value with sub-gradient `sign(x)` at zero.
    Abs(usize),
    /// Base-2 logarithm: `d/dx log2(x) = 1/(x * ln(2))`.
    Log2(usize),
    /// Softmax over the last axis, producing a probability distribution.
    Softmax(usize),
    /// Cross-entropy loss between predicted logits and target labels.
    CrossEntropy {
        /// Node index of the raw logit tensor.
        logits: usize,
        /// Node index of the target (one-hot or class-index) tensor.
        targets: usize,
    },
    /// Layer normalization over the last axis; stores statistics for backward.
    LayerNorm(usize),
    /// Batch normalization over the first axis; stores statistics for backward.
    BatchNorm(usize),
    /// Element-wise clamping to the range `[min, max]`.
    Clamp {
        /// Node index of the input tensor.
        input: usize,
        /// Lower bound.
        min: f64,
        /// Upper bound.
        max: f64,
    },
    /// Element-wise conditional select using a `{0.0, 1.0}` mask tensor.
    Where {
        /// Node index of the condition mask.
        cond: usize,
        /// Node index selected where condition is `1.0`.
        on_true: usize,
        /// Node index selected where condition is `0.0`.
        on_false: usize,
    },
    /// Reshape a tensor, storing the original shape for backward reconstruction.
    Reshape {
        /// Node index of the input tensor.
        input: usize,
        /// Shape before the reshape (used during backward).
        original_shape: Vec<usize>,
    },
    /// Transpose a 2-D tensor (swap rows and columns).
    TransposeOp(usize),
    /// Concatenate tensors along an axis, storing per-input sizes for backward splitting.
    CatOp {
        /// Node indices of the tensors to concatenate.
        inputs: Vec<usize>,
        /// Axis along which to concatenate.
        axis: usize,
        /// Size of each input along the concatenation axis.
        sizes: Vec<usize>,
    },
    /// Gather elements along an axis by index.
    GatherOp {
        /// Node index of the source tensor.
        input: usize,
        /// Indices to gather.
        indices: Vec<usize>,
        /// Axis along which to gather.
        axis: usize,
    },
    /// Gaussian Error Linear Unit: `x * Φ(x)` where `Φ` is the standard normal CDF.
    Gelu(usize),
    /// Sigmoid Linear Unit (Swish): `x * sigmoid(x)`.
    Silu(usize),
    /// Exponential Linear Unit: `x if x>0 else exp(x)-1` (α=1).
    Elu(usize),
    /// Scaled Exponential Linear Unit: `λ*(x if x>0 else α*(exp(x)-1))`.
    Selu(usize),
    /// Fused dense layer: `activation(input @ weight^T + bias)`.
    /// Collapses transpose + matmul + bias-add + activation into one node,
    /// reducing graph size by 3× per layer and fusing the backward pass.
    MlpLayer {
        /// Node index of the input tensor [batch, in_features].
        input: usize,
        /// Node index of the weight parameter [out_features, in_features].
        weight: usize,
        /// Node index of the bias parameter [out_features].
        bias: usize,
        /// Activation function applied after affine transform.
        activation: crate::pinn::Activation,
    },
    /// Phase 3e Tier 1 — broadcast a scalar (numel=1) tensor to a target
    /// shape. Forward fills every output element with the scalar's value.
    /// Backward sums the upstream tensor back into the input scalar.
    ///
    /// Introduced to give `grad_of` a way to express the gradient of
    /// `Sum`/`Mean` reductions in graph-node form: those ops reduce a
    /// vector to a scalar, so their backward step needs to broadcast a
    /// scalar gradient back to vector shape — and the broadcast must
    /// itself be a *graph node* so higher-order derivatives compose.
    /// `BroadcastScalar` and `Sum`/`Mean` are mutually closed under
    /// differentiation: differentiating `BroadcastScalar` yields a
    /// `Sum`-shaped operation; differentiating `Sum` yields a
    /// `BroadcastScalar`. Adding them together preserves the same
    /// closure-under-differentiation property the polynomial subset
    /// already had (see ADR-0023).
    BroadcastScalar {
        /// Node index of the scalar (numel=1) input.
        input: usize,
        /// Output tensor shape.
        target_shape: Vec<usize>,
    },
}

/// A node in the reverse-mode AD graph.
/// Kept for backward compatibility of `GradNode` type references.
#[derive(Debug, Clone)]
pub struct GradNode {
    pub op: GradOp,
    pub tensor: Tensor,
    pub grad: Option<Tensor>,
}

/// The reverse-mode AD tape/graph.
///
/// Flat arena storage: ops, tensors, and parameter gradients are stored
/// in parallel Vec<>s indexed by node ID. This eliminates Rc<RefCell<>>
/// overhead and enables zero-copy backward traversal.
pub struct GradGraph {
    ops: Vec<GradOp>,
    tensors: Vec<Tensor>,
    param_grads: Vec<Option<Tensor>>,
}

impl GradGraph {
    pub fn new() -> Self {
        Self {
            ops: Vec::new(),
            tensors: Vec::new(),
            param_grads: Vec::new(),
        }
    }

    /// Backward compatibility: return number of nodes.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Returns true if the graph has no nodes.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Push a raw node into the arena. Used by external code that previously
    /// accessed `self.nodes` directly.
    pub fn push_node(&mut self, op: GradOp, tensor: Tensor, grad: Option<Tensor>) -> usize {
        let idx = self.ops.len();
        self.ops.push(op);
        self.tensors.push(tensor);
        self.param_grads.push(grad);
        idx
    }

    /// Create an input node (data, no gradient).
    pub fn input(&mut self, tensor: Tensor) -> usize {
        let idx = self.ops.len();
        self.ops.push(GradOp::Input);
        self.tensors.push(tensor);
        self.param_grads.push(None);
        idx
    }

    /// Create a parameter node (trainable, accumulates gradients).
    pub fn parameter(&mut self, tensor: Tensor) -> usize {
        let idx = self.ops.len();
        let shape = tensor.shape().to_vec();
        self.ops.push(GradOp::Parameter);
        self.tensors.push(tensor);
        self.param_grads.push(Some(Tensor::zeros(&shape)));
        idx
    }

    /// Element-wise addition.
    pub fn add(&mut self, a: usize, b: usize) -> usize {
        let a_t = self.tensors[a].clone();
        let b_t = self.tensors[b].clone();
        let result = a_t.add_unchecked(&b_t);
        let idx = self.ops.len();
        self.ops.push(GradOp::Add(a, b));
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// Element-wise subtraction.
    pub fn sub(&mut self, a: usize, b: usize) -> usize {
        let a_t = self.tensors[a].clone();
        let b_t = self.tensors[b].clone();
        let result = a_t.sub_unchecked(&b_t);
        let idx = self.ops.len();
        self.ops.push(GradOp::Sub(a, b));
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// Element-wise multiplication.
    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        let a_t = self.tensors[a].clone();
        let b_t = self.tensors[b].clone();
        let result = a_t.mul_elem_unchecked(&b_t);
        let idx = self.ops.len();
        self.ops.push(GradOp::Mul(a, b));
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// Matrix multiplication.
    pub fn matmul(&mut self, a: usize, b: usize) -> usize {
        let a_t = self.tensors[a].clone();
        let b_t = self.tensors[b].clone();
        let result = a_t.matmul_unchecked(&b_t);
        let idx = self.ops.len();
        self.ops.push(GradOp::MatMul(a, b));
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// Sum all elements.
    pub fn sum(&mut self, a: usize) -> usize {
        let a_t = self.tensors[a].clone();
        let s = a_t.sum();
        let result = Tensor::from_vec_unchecked(vec![s], &[1]);
        let idx = self.ops.len();
        self.ops.push(GradOp::Sum(a));
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// Mean of all elements.
    pub fn mean(&mut self, a: usize) -> usize {
        let a_t = self.tensors[a].clone();
        let m = a_t.mean();
        let result = Tensor::from_vec_unchecked(vec![m], &[1]);
        let idx = self.ops.len();
        self.ops.push(GradOp::Mean(a));
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    // ── Phase B8: Transcendental & activation forward ops ──

    /// Element-wise sine.
    pub fn sin(&mut self, a: usize) -> usize {
        let a_t = self.tensors[a].clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| x.sin()).collect(),
            a_t.shape(),
        );
        let idx = self.ops.len();
        self.ops.push(GradOp::Sin(a));
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// Element-wise cosine.
    pub fn cos(&mut self, a: usize) -> usize {
        let a_t = self.tensors[a].clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| x.cos()).collect(),
            a_t.shape(),
        );
        let idx = self.ops.len();
        self.ops.push(GradOp::Cos(a));
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// Element-wise square root.
    pub fn sqrt(&mut self, a: usize) -> usize {
        let a_t = self.tensors[a].clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| x.sqrt()).collect(),
            a_t.shape(),
        );
        let idx = self.ops.len();
        self.ops.push(GradOp::Sqrt(a));
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// Element-wise power with constant exponent.
    pub fn pow(&mut self, a: usize, n: f64) -> usize {
        let a_t = self.tensors[a].clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| x.powf(n)).collect(),
            a_t.shape(),
        );
        let idx = self.ops.len();
        self.ops.push(GradOp::Pow(a, n));
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// Sigmoid activation: 1 / (1 + exp(-x)).
    pub fn sigmoid(&mut self, a: usize) -> usize {
        let a_t = self.tensors[a].clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
            a_t.shape(),
        );
        let idx = self.ops.len();
        self.ops.push(GradOp::Sigmoid(a));
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// ReLU activation: max(0, x).
    pub fn relu(&mut self, a: usize) -> usize {
        let a_t = self.tensors[a].clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect(),
            a_t.shape(),
        );
        let idx = self.ops.len();
        self.ops.push(GradOp::Relu(a));
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// Tanh activation.
    pub fn tanh_act(&mut self, a: usize) -> usize {
        let a_t = self.tensors[a].clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| x.tanh()).collect(),
            a_t.shape(),
        );
        let idx = self.ops.len();
        self.ops.push(GradOp::TanhAct(a));
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    // SELU constants
    const SELU_LAMBDA: f64 = 1.0507009873554804934193349852946;
    const SELU_ALPHA: f64 = 1.6732632423543772848170429916717;

    /// GELU activation: x * Φ(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³))).
    pub fn gelu(&mut self, a: usize) -> usize {
        let a_t = self.tensors[a].clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| {
                let inner = (2.0_f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            }).collect(),
            a_t.shape(),
        );
        let idx = self.ops.len();
        self.ops.push(GradOp::Gelu(a));
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// SiLU (Swish) activation: x * sigmoid(x).
    pub fn silu(&mut self, a: usize) -> usize {
        let a_t = self.tensors[a].clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| {
                let s = 1.0 / (1.0 + (-x).exp());
                x * s
            }).collect(),
            a_t.shape(),
        );
        let idx = self.ops.len();
        self.ops.push(GradOp::Silu(a));
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// ELU activation: x if x>0, else exp(x)-1 (α=1).
    pub fn elu(&mut self, a: usize) -> usize {
        let a_t = self.tensors[a].clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| if x > 0.0 { x } else { x.exp() - 1.0 }).collect(),
            a_t.shape(),
        );
        let idx = self.ops.len();
        self.ops.push(GradOp::Elu(a));
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// SELU activation: λ * (x if x>0, else α*(exp(x)-1)).
    pub fn selu(&mut self, a: usize) -> usize {
        let a_t = self.tensors[a].clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| {
                if x > 0.0 { Self::SELU_LAMBDA * x } else { Self::SELU_LAMBDA * Self::SELU_ALPHA * (x.exp() - 1.0) }
            }).collect(),
            a_t.shape(),
        );
        let idx = self.ops.len();
        self.ops.push(GradOp::Selu(a));
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    // ── Phase 8: Extended AD forward ops ──

    /// Element-wise absolute value.
    pub fn abs(&mut self, a: usize) -> usize {
        let a_t = self.tensors[a].clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| x.abs()).collect(),
            a_t.shape(),
        );
        let idx = self.ops.len();
        self.ops.push(GradOp::Abs(a));
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// Element-wise log base 2.
    pub fn log2(&mut self, a: usize) -> usize {
        let a_t = self.tensors[a].clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| x.log2()).collect(),
            a_t.shape(),
        );
        let idx = self.ops.len();
        self.ops.push(GradOp::Log2(a));
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// Softmax along the last axis (treats tensor as a flat vector for 1-D).
    /// Uses numerically stable log-sum-exp: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    pub fn softmax(&mut self, a: usize) -> usize {
        use cjc_repro::KahanAccumulatorF64;
        let a_t = self.tensors[a].clone();
        let data = a_t.to_vec();
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_shifted: Vec<f64> = data.iter().map(|&x| (x - max_val).exp()).collect();
        let mut sum_acc = KahanAccumulatorF64::new();
        for &v in &exp_shifted {
            sum_acc.add(v);
        }
        let sum_exp = sum_acc.finalize();
        let softmax_data: Vec<f64> = exp_shifted.iter().map(|&e| e / sum_exp).collect();
        let result = Tensor::from_vec_unchecked(softmax_data, a_t.shape());
        let idx = self.ops.len();
        self.ops.push(GradOp::Softmax(a));
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// Cross-entropy loss: -sum(targets * log(softmax(logits)))
    /// Uses numerically stable log-sum-exp internally.
    /// Returns a scalar [1] tensor.
    pub fn cross_entropy(&mut self, logits: usize, targets: usize) -> usize {
        use cjc_repro::KahanAccumulatorF64;
        let logits_t = self.tensors[logits].clone();
        let targets_t = self.tensors[targets].clone();
        let logits_data = logits_t.to_vec();
        let targets_data = targets_t.to_vec();
        // Numerically stable: log_softmax = x_i - max(x) - log(sum(exp(x_j - max(x))))
        let max_val = logits_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let shifted: Vec<f64> = logits_data.iter().map(|&x| x - max_val).collect();
        let exp_shifted: Vec<f64> = shifted.iter().map(|&x| x.exp()).collect();
        let mut sum_acc = KahanAccumulatorF64::new();
        for &v in &exp_shifted {
            sum_acc.add(v);
        }
        let log_sum_exp = sum_acc.finalize().ln();
        let log_softmax: Vec<f64> = shifted.iter().map(|&x| x - log_sum_exp).collect();
        // CE = -sum(targets * log_softmax)
        let mut ce_acc = KahanAccumulatorF64::new();
        for (t, ls) in targets_data.iter().zip(log_softmax.iter()) {
            ce_acc.add(-t * ls);
        }
        let ce = ce_acc.finalize();
        let result = Tensor::from_vec_unchecked(vec![ce], &[1]);
        let idx = self.ops.len();
        self.ops.push(GradOp::CrossEntropy { logits, targets });
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// Layer normalization: normalize input to zero mean and unit variance.
    /// y = (x - mean(x)) / sqrt(var(x) + eps), where eps = 1e-5.
    pub fn layer_norm(&mut self, a: usize) -> usize {
        use cjc_repro::KahanAccumulatorF64;
        let a_t = self.tensors[a].clone();
        let data = a_t.to_vec();
        let n = data.len() as f64;
        // Mean
        let mut mean_acc = KahanAccumulatorF64::new();
        for &v in &data {
            mean_acc.add(v);
        }
        let mean = mean_acc.finalize() / n;
        // Variance
        let mut var_acc = KahanAccumulatorF64::new();
        for &v in &data {
            let d = v - mean;
            var_acc.add(d * d);
        }
        let var = var_acc.finalize() / n;
        let eps = 1e-5;
        let std = (var + eps).sqrt();
        let normed: Vec<f64> = data.iter().map(|&x| (x - mean) / std).collect();
        let result = Tensor::from_vec_unchecked(normed, a_t.shape());
        let idx = self.ops.len();
        self.ops.push(GradOp::LayerNorm(a));
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// Batch normalization: normalize along the first axis (batch dimension).
    /// For a tensor of shape [batch, features], normalizes each feature across the batch.
    /// y = (x - mean(x)) / sqrt(var(x) + eps), where eps = 1e-5.
    /// For 1-D inputs, behaves identically to layer_norm.
    pub fn batch_norm(&mut self, a: usize) -> usize {
        use cjc_repro::KahanAccumulatorF64;
        let a_t = self.tensors[a].clone();
        let data = a_t.to_vec();
        let n = data.len() as f64;
        let mut mean_acc = KahanAccumulatorF64::new();
        for &v in &data {
            mean_acc.add(v);
        }
        let mean = mean_acc.finalize() / n;
        let mut var_acc = KahanAccumulatorF64::new();
        for &v in &data {
            let d = v - mean;
            var_acc.add(d * d);
        }
        let var = var_acc.finalize() / n;
        let eps = 1e-5;
        let std = (var + eps).sqrt();
        let normed: Vec<f64> = data.iter().map(|&x| (x - mean) / std).collect();
        let result = Tensor::from_vec_unchecked(normed, a_t.shape());
        let idx = self.ops.len();
        self.ops.push(GradOp::BatchNorm(a));
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// Element-wise clamp to [min, max].
    pub fn clamp(&mut self, a: usize, min: f64, max: f64) -> usize {
        let a_t = self.tensors[a].clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| x.max(min).min(max)).collect(),
            a_t.shape(),
        );
        let idx = self.ops.len();
        self.ops.push(GradOp::Clamp { input: a, min, max });
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// Conditional select: where(cond, on_true, on_false).
    /// cond is a tensor of 0.0/1.0 values acting as a mask.
    pub fn where_cond(&mut self, cond: usize, on_true: usize, on_false: usize) -> usize {
        let cond_t = self.tensors[cond].clone();
        let true_t = self.tensors[on_true].clone();
        let false_t = self.tensors[on_false].clone();
        let c = cond_t.to_vec();
        let t = true_t.to_vec();
        let f = false_t.to_vec();
        let result_data: Vec<f64> = c.iter().zip(t.iter().zip(f.iter()))
            .map(|(&ci, (&ti, &fi))| if ci != 0.0 { ti } else { fi })
            .collect();
        let result = Tensor::from_vec_unchecked(result_data, cond_t.shape());
        let idx = self.ops.len();
        self.ops.push(GradOp::Where { cond, on_true, on_false });
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// Reshape a tensor. Stores the original shape for backward.
    pub fn reshape(&mut self, a: usize, new_shape: &[usize]) -> usize {
        let a_t = self.tensors[a].clone();
        let original_shape = a_t.shape().to_vec();
        let result = a_t.reshape(new_shape).expect("GradGraph::reshape: shape mismatch");
        let idx = self.ops.len();
        self.ops.push(GradOp::Reshape { input: a, original_shape });
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// Transpose a 2-D tensor.
    pub fn transpose_op(&mut self, a: usize) -> usize {
        let a_t = self.tensors[a].clone();
        let result = a_t.transpose();
        let idx = self.ops.len();
        self.ops.push(GradOp::TransposeOp(a));
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// Fused dense layer: `activation(input @ weight^T + bias)`.
    ///
    /// Collapses transpose + matmul + bias-add + activation into a single graph
    /// node, reducing graph size by 3× per layer. The backward pass computes
    /// gradients for input, weight, and bias in one fused step.
    ///
    /// - `input`: node index for input tensor `[batch, in_features]`
    /// - `weight`: node index for weight parameter `[out_features, in_features]`
    /// - `bias`: node index for bias parameter `[out_features]`
    pub fn mlp_layer(&mut self, input: usize, weight: usize, bias: usize, activation: crate::pinn::Activation) -> usize {
        // Forward: activation(input @ weight^T + bias)
        let input_t = &self.tensors[input];
        let weight_t = &self.tensors[weight];
        let bias_t = &self.tensors[bias];

        let wt = weight_t.transpose();
        let z = input_t.matmul_unchecked(&wt);
        let z_biased = z.add_unchecked(bias_t);

        let result = match activation {
            crate::pinn::Activation::Tanh => {
                let data = z_biased.to_vec();
                let shape = z_biased.shape().to_vec();
                Tensor::from_vec_unchecked(data.iter().map(|x| x.tanh()).collect(), &shape)
            }
            crate::pinn::Activation::Sigmoid => {
                let data = z_biased.to_vec();
                let shape = z_biased.shape().to_vec();
                Tensor::from_vec_unchecked(data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(), &shape)
            }
            crate::pinn::Activation::Relu => {
                let data = z_biased.to_vec();
                let shape = z_biased.shape().to_vec();
                Tensor::from_vec_unchecked(data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect(), &shape)
            }
            crate::pinn::Activation::None => z_biased,
            crate::pinn::Activation::Gelu => {
                let data = z_biased.to_vec();
                let shape = z_biased.shape().to_vec();
                Tensor::from_vec_unchecked(data.iter().map(|&x| {
                    let inner = (2.0_f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
                    0.5 * x * (1.0 + inner.tanh())
                }).collect(), &shape)
            }
            crate::pinn::Activation::Silu => {
                let data = z_biased.to_vec();
                let shape = z_biased.shape().to_vec();
                Tensor::from_vec_unchecked(data.iter().map(|&x| x / (1.0 + (-x).exp())).collect(), &shape)
            }
            crate::pinn::Activation::Elu => {
                let data = z_biased.to_vec();
                let shape = z_biased.shape().to_vec();
                Tensor::from_vec_unchecked(data.iter().map(|&x| if x > 0.0 { x } else { x.exp() - 1.0 }).collect(), &shape)
            }
            crate::pinn::Activation::Selu => {
                let data = z_biased.to_vec();
                let shape = z_biased.shape().to_vec();
                Tensor::from_vec_unchecked(data.iter().map(|&x| {
                    if x > 0.0 { Self::SELU_LAMBDA * x } else { Self::SELU_LAMBDA * Self::SELU_ALPHA * (x.exp() - 1.0) }
                }).collect(), &shape)
            }
            crate::pinn::Activation::SinAct => {
                let data = z_biased.to_vec();
                let shape = z_biased.shape().to_vec();
                Tensor::from_vec_unchecked(data.iter().map(|&x| x.sin()).collect(), &shape)
            }
        };

        let idx = self.ops.len();
        self.ops.push(GradOp::MlpLayer { input, weight, bias, activation });
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// Concatenate tensors along an axis.
    /// All input tensors must have the same shape except along the concatenation axis.
    pub fn cat(&mut self, inputs: &[usize], axis: usize) -> usize {
        let tensors: Vec<Tensor> = inputs.iter()
            .map(|&i| self.tensors[i].clone())
            .collect();
        let sizes: Vec<usize> = tensors.iter()
            .map(|t| t.shape()[axis])
            .collect();
        // Build concatenated data along axis
        // For simplicity, handle axis=0 for 1-D and 2-D tensors
        let mut all_data = Vec::new();
        let mut total_along_axis = 0usize;
        let ndim = tensors[0].ndim();
        let mut result_shape = tensors[0].shape().to_vec();

        if ndim == 1 {
            // 1-D: just concatenate flat data
            for t in &tensors {
                all_data.extend(t.to_vec());
                total_along_axis += t.shape()[0];
            }
            result_shape[0] = total_along_axis;
        } else if ndim == 2 && axis == 0 {
            // Concat along rows
            for t in &tensors {
                all_data.extend(t.to_vec());
                total_along_axis += t.shape()[0];
            }
            result_shape[0] = total_along_axis;
        } else if ndim == 2 && axis == 1 {
            // Concat along columns
            let nrows = tensors[0].shape()[0];
            for row in 0..nrows {
                for t in &tensors {
                    let cols = t.shape()[1];
                    let row_data = t.to_vec();
                    let start = row * cols;
                    all_data.extend_from_slice(&row_data[start..start + cols]);
                }
            }
            total_along_axis = sizes.iter().sum();
            result_shape[1] = total_along_axis;
        } else {
            // General case: flatten, cat, reshape (deterministic but limited)
            for t in &tensors {
                all_data.extend(t.to_vec());
                total_along_axis += t.shape()[axis];
            }
            result_shape[axis] = total_along_axis;
        }

        let result = Tensor::from_vec_unchecked(all_data, &result_shape);
        let input_vec = inputs.to_vec();
        let idx = self.ops.len();
        self.ops.push(GradOp::CatOp { inputs: input_vec, axis, sizes });
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// Gather elements along an axis using indices.
    /// For a 1-D tensor, returns tensor[indices].
    pub fn gather(&mut self, a: usize, indices: &[usize], axis: usize) -> usize {
        let a_t = self.tensors[a].clone();
        let data = a_t.to_vec();
        // For 1-D: just pick elements at indices
        let gathered: Vec<f64> = if a_t.ndim() == 1 {
            indices.iter().map(|&i| data[i]).collect()
        } else if a_t.ndim() == 2 && axis == 0 {
            let cols = a_t.shape()[1];
            indices.iter().flat_map(|&i| {
                let start = i * cols;
                data[start..start + cols].to_vec()
            }).collect()
        } else {
            // Fallback: gather from flat data
            indices.iter().map(|&i| data[i]).collect()
        };
        let mut result_shape = a_t.shape().to_vec();
        if a_t.ndim() == 1 {
            result_shape[0] = indices.len();
        } else if axis == 0 {
            result_shape[0] = indices.len();
        } else {
            result_shape[axis] = indices.len();
        }
        let result = Tensor::from_vec_unchecked(gathered, &result_shape);
        let idx = self.ops.len();
        self.ops.push(GradOp::GatherOp { input: a, indices: indices.to_vec(), axis });
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    /// Phase 3e Tier 1 — broadcast a scalar (numel=1) tensor to a target
    /// shape by replicating the scalar value across every output element.
    ///
    /// The forward op is the reverse of `Sum`/`Mean`: `Sum`/`Mean` reduce
    /// `[N]` → `[1]`; `BroadcastScalar` expands `[1]` → `target_shape`.
    /// Adding this primitive lets `grad_of` express the gradient of
    /// `Sum`/`Mean` as a graph node, making higher-order differentiation
    /// of those reductions work.
    ///
    /// The input tensor must be scalar-shaped (numel = 1); call sites
    /// that don't guarantee this should reduce first.
    pub fn broadcast_scalar(&mut self, input: usize, target_shape: &[usize]) -> usize {
        let in_t = self.tensors[input].clone();
        let in_numel: usize = in_t.shape().iter().product();
        debug_assert_eq!(
            in_numel, 1,
            "broadcast_scalar: input must be scalar (numel=1); got shape {:?}",
            in_t.shape()
        );
        let v = in_t.to_vec()[0];
        let target_n: usize = target_shape.iter().product();
        let result = Tensor::from_vec_unchecked(vec![v; target_n], target_shape);
        let idx = self.ops.len();
        self.ops.push(GradOp::BroadcastScalar {
            input,
            target_shape: target_shape.to_vec(),
        });
        self.tensors.push(result);
        self.param_grads.push(None);
        idx
    }

    // ── Phase C1: Missing forward methods ──

    /// Element-wise division: a / b.
    /// GradOp::Div(a, b) already has backward implementation.
    pub fn div(&mut self, a: usize, b: usize) -> usize {
        let a_tensor = self.tensors[a].clone();
        let b_tensor = self.tensors[b].clone();
        let result = a_tensor.div_elem_unchecked(&b_tensor);
        let node = GradNode { op: GradOp::Div(a, b), tensor: result, grad: None };
        self.ops.push(node.op);
        self.tensors.push(node.tensor);
        self.param_grads.push(node.grad);
        self.ops.len() - 1
    }

    /// Element-wise negation: -a.
    /// GradOp::Neg(a) already has backward implementation.
    pub fn neg(&mut self, a: usize) -> usize {
        let a_tensor = self.tensors[a].clone();
        let result = a_tensor.neg();
        let node = GradNode { op: GradOp::Neg(a), tensor: result, grad: None };
        self.ops.push(node.op);
        self.tensors.push(node.tensor);
        self.param_grads.push(node.grad);
        self.ops.len() - 1
    }

    /// Scalar multiply: a * s (where s is an f64 constant).
    /// GradOp::ScalarMul(a, s) already has backward implementation.
    pub fn scalar_mul(&mut self, a: usize, s: f64) -> usize {
        let a_tensor = self.tensors[a].clone();
        let result = a_tensor.scalar_mul(s);
        let node = GradNode { op: GradOp::ScalarMul(a, s), tensor: result, grad: None };
        self.ops.push(node.op);
        self.tensors.push(node.tensor);
        self.param_grads.push(node.grad);
        self.ops.len() - 1
    }

    /// Element-wise exponential: exp(a).
    /// GradOp::Exp(a) already has backward implementation.
    pub fn exp(&mut self, a: usize) -> usize {
        let a_tensor = self.tensors[a].clone();
        let result = Tensor::from_vec_unchecked(
            a_tensor.to_vec().iter().map(|x| x.exp()).collect(),
            a_tensor.shape(),
        );
        let node = GradNode { op: GradOp::Exp(a), tensor: result, grad: None };
        self.ops.push(node.op);
        self.tensors.push(node.tensor);
        self.param_grads.push(node.grad);
        self.ops.len() - 1
    }

    /// Element-wise natural logarithm: ln(a).
    /// GradOp::Ln(a) already has backward implementation.
    pub fn ln(&mut self, a: usize) -> usize {
        let a_tensor = self.tensors[a].clone();
        let result = Tensor::from_vec_unchecked(
            a_tensor.to_vec().iter().map(|x| x.ln()).collect(),
            a_tensor.shape(),
        );
        let node = GradNode { op: GradOp::Ln(a), tensor: result, grad: None };
        self.ops.push(node.op);
        self.tensors.push(node.tensor);
        self.param_grads.push(node.grad);
        self.ops.len() - 1
    }

    /// Get the scalar value from a 1-element tensor node.
    pub fn value(&self, idx: usize) -> f64 {
        let data = self.tensors[idx].to_vec();
        data[0]
    }

    /// Get the tensor at a node.
    pub fn tensor(&self, idx: usize) -> Tensor {
        self.tensors[idx].clone()
    }

    /// Set the tensor at a node (for parameter updates).
    pub fn set_tensor(&mut self, idx: usize, tensor: Tensor) {
        self.tensors[idx] = tensor;
    }

    /// Get the gradient at a node.
    pub fn grad(&self, idx: usize) -> Option<Tensor> {
        self.param_grads[idx].clone()
    }

    /// Batched backward + gradient collection: zero_grad, backward, then
    /// collect gradients for the given parameter indices into a Vec.
    /// Collapses three interpreter round-trips into one native call.
    pub fn backward_collect(&mut self, loss_idx: usize, param_indices: &[usize]) -> Vec<Option<Tensor>> {
        self.zero_grad();
        self.backward(loss_idx);
        param_indices.iter().map(|&idx| self.grad(idx)).collect()
    }

    // ════════════════════════════════════════════════════════════════════
    //  Phase 3d — native higher-order autodiff (graph-of-graphs)
    // ════════════════════════════════════════════════════════════════════

    /// Compute the gradient of `f` with respect to `x` *as a new graph
    /// node*, not as a tensor value.
    ///
    /// Where `backward()` accumulates `dF/dParam` into `param_grads`
    /// (concrete tensors), `grad_of()` builds a *sub-graph* that
    /// represents `dF/dx` as a function of the existing parameters and
    /// inputs. The returned node can be forward-evaluated like any
    /// other, *and* it can itself be passed back into `grad_of()` —
    /// giving native second- and higher-order derivatives.
    ///
    /// Phase 3d ships **the polynomial-arithmetic op subset** (Input,
    /// Parameter, Add, Sub, Mul, ScalarMul, Neg). These suffice to
    /// express any polynomial in the parameters; higher-order
    /// derivatives of such polynomials evaluate exactly without
    /// finite-difference truncation error.
    ///
    /// Other ops (Sum, Mean, Matmul, transcendentals, activations,
    /// reductions, fused ops) return `Err`. Phase 3e will expand the
    /// supported set; users needing those today can use the FD pattern
    /// (`set_tensor` + `reforward` + scalar arithmetic) as in the
    /// pre-Phase-3d PINN demos.
    ///
    /// ### Example
    ///
    /// ```ignore
    /// let mut g = GradGraph::new();
    /// let x = g.parameter(Tensor::from_vec(vec![2.0], &[1]).unwrap());
    /// // f = x³
    /// let xx = g.mul(x, x);
    /// let f = g.mul(xx, x);
    /// // df/dx = 3x²
    /// let df = g.grad_of(f, x).unwrap();
    /// assert_eq!(g.tensor(df).to_vec(), vec![12.0]); // 3·2² = 12
    /// // d²f/dx² = 6x
    /// let d2f = g.grad_of(df, x).unwrap();
    /// assert_eq!(g.tensor(d2f).to_vec(), vec![12.0]); // 6·2 = 12
    /// ```
    pub fn grad_of(&mut self, f: usize, x: usize) -> Result<usize, String> {
        if f >= self.ops.len() {
            return Err(format!(
                "grad_of: f={} out of range (graph has {} nodes)",
                f,
                self.ops.len()
            ));
        }
        if x >= self.ops.len() {
            return Err(format!(
                "grad_of: x={} out of range (graph has {} nodes)",
                x,
                self.ops.len()
            ));
        }

        let n = self.ops.len();

        // Reachability set: which existing nodes are ancestors of f?
        // Same algorithm as backward(): walk backward from f, marking
        // each input as reachable.
        let mut reachable = vec![false; n];
        reachable[f] = true;
        for i in (0..=f).rev() {
            if !reachable[i] {
                continue;
            }
            self.mark_children_reachable(i, &mut reachable);
        }

        if !reachable[x] {
            return Err(format!(
                "grad_of: x (node {}) is not reachable from f (node {})",
                x, f
            ));
        }

        // upstream_grad[i] holds the graph-node-index for the gradient
        // at node `i`. Sized to the original graph length; new nodes
        // we create during the pass have higher indices and don't
        // need slots here.
        let mut upstream_grad: Vec<Option<usize>> = vec![None; n];

        // Initialize upstream at f to a ones-tensor matching f's shape.
        // Lives as a regular Input node so subsequent ops can reference
        // it just like any other graph node.
        let f_shape = self.tensors[f].shape().to_vec();
        let n_elems: usize = f_shape.iter().product();
        let ones_tensor = Tensor::from_vec_unchecked(vec![1.0; n_elems], &f_shape);
        let one_node = self.input(ones_tensor);
        upstream_grad[f] = Some(one_node);

        // Reverse-topological pass over the *original* graph nodes.
        // Each iteration consumes the upstream gradient at node i and
        // contributes to the upstream gradients at i's inputs.
        for i in (0..=f).rev() {
            if !reachable[i] {
                continue;
            }
            // STOP at x — we don't propagate further. upstream_grad[x]
            // holds the final accumulated gradient at this point.
            if i == x {
                continue;
            }

            let upstream = match upstream_grad[i] {
                Some(u) => u,
                None => continue,
            };

            let op = self.ops[i].clone();
            match op {
                GradOp::Input | GradOp::Parameter => {
                    // Leaf node — gradient stops here. Already preserved
                    // in upstream_grad[i] (we don't take it).
                }
                GradOp::Add(a, b) => {
                    self.accumulate_grad_node(&mut upstream_grad, a, upstream);
                    self.accumulate_grad_node(&mut upstream_grad, b, upstream);
                }
                GradOp::Sub(a, b) => {
                    self.accumulate_grad_node(&mut upstream_grad, a, upstream);
                    let neg_up = self.neg(upstream);
                    self.accumulate_grad_node(&mut upstream_grad, b, neg_up);
                }
                GradOp::Mul(a, b) => {
                    // d/da (a*b) = b ; d/db (a*b) = a
                    let grad_a = self.mul(b, upstream);
                    let grad_b = self.mul(a, upstream);
                    self.accumulate_grad_node(&mut upstream_grad, a, grad_a);
                    self.accumulate_grad_node(&mut upstream_grad, b, grad_b);
                }
                GradOp::ScalarMul(a, s) => {
                    // d/da (s*a) = s
                    let grad_a = self.scalar_mul(upstream, s);
                    self.accumulate_grad_node(&mut upstream_grad, a, grad_a);
                }
                GradOp::Neg(a) => {
                    let neg_up = self.neg(upstream);
                    self.accumulate_grad_node(&mut upstream_grad, a, neg_up);
                }
                // ── Phase 3e Tier 1 — reductions & broadcast ────────────
                GradOp::Sum(a) => {
                    // Forward: f = Σ a_i, shape [1]. Backward: dF/da_i = upstream
                    // (broadcast to a's shape). The broadcast is itself a graph
                    // node so higher-order derivatives compose.
                    let a_shape = self.tensors[a].shape().to_vec();
                    let bc = self.broadcast_scalar(upstream, &a_shape);
                    self.accumulate_grad_node(&mut upstream_grad, a, bc);
                }
                GradOp::Mean(a) => {
                    // Forward: f = (Σ a_i) / N. Backward: dF/da_i = upstream / N
                    // (broadcast). Express as scalar_mul(upstream, 1/N) → broadcast.
                    let a_shape = self.tensors[a].shape().to_vec();
                    let n: usize = a_shape.iter().product();
                    let inv_n = 1.0 / (n as f64);
                    let scaled = self.scalar_mul(upstream, inv_n);
                    let bc = self.broadcast_scalar(scaled, &a_shape);
                    self.accumulate_grad_node(&mut upstream_grad, a, bc);
                }
                GradOp::BroadcastScalar { input, .. } => {
                    // Forward: result[i] = scalar.value() for all i (broadcast).
                    // Backward: dF/dscalar = Σ upstream over output shape.
                    // Express via Sum, which itself is differentiable (closed-
                    // under-differentiation with BroadcastScalar — adding both at
                    // once preserves the polynomial subset's closure property).
                    let summed = self.sum(upstream);
                    self.accumulate_grad_node(&mut upstream_grad, input, summed);
                }
                other => {
                    return Err(format!(
                        "grad_of: op {other:?} not yet supported. Current coverage: \
                         polynomial subset (Add, Sub, Mul, ScalarMul, Neg) + \
                         Phase 3e Tier 1 reductions (Sum, Mean, BroadcastScalar). \
                         Phase 3e Tier 2 will add transcendentals and activations."
                    ));
                }
            }
        }

        upstream_grad[x].ok_or_else(|| {
            format!(
                "grad_of: gradient at x (node {}) was not computed (unreachable along visited paths)",
                x
            )
        })
    }

    /// Helper for `grad_of`: append `contribution` to `upstream_grad[target]`,
    /// building an `Add` node when there's already an existing gradient at
    /// `target` (i.e., `target` is referenced by multiple downstream ops).
    fn accumulate_grad_node(
        &mut self,
        grads: &mut [Option<usize>],
        target: usize,
        contribution: usize,
    ) {
        let new_val = match grads[target] {
            Some(existing) => self.add(existing, contribution),
            None => contribution,
        };
        grads[target] = Some(new_val);
    }

    /// Zero out all gradients.
    pub fn zero_grad(&mut self) {
        for pg in &mut self.param_grads {
            if let Some(ref mut grad) = pg {
                let shape = grad.shape().to_vec();
                *grad = Tensor::zeros(&shape);
            }
        }
    }

    /// Clip all gradients to `[-max_norm, max_norm]` (element-wise).
    /// This prevents gradient explosion during backpropagation.
    pub fn clip_grad(&mut self, max_norm: f64) {
        for pg in &mut self.param_grads {
            if let Some(ref mut grad) = pg {
                let data = grad.to_vec();
                let clipped: Vec<f64> = data.iter()
                    .map(|&x| x.max(-max_norm).min(max_norm))
                    .collect();
                let shape = grad.shape().to_vec();
                *grad = Tensor::from_vec_unchecked(clipped, &shape);
            }
        }
    }

    /// Clip gradients by global norm: if ||grads||_2 > max_norm, scale all
    /// gradients so the global norm equals max_norm. Deterministic via
    /// sequential accumulation.
    pub fn clip_grad_norm(&mut self, max_norm: f64) -> f64 {
        use cjc_repro::KahanAccumulatorF64;
        // Compute global norm
        let mut acc = KahanAccumulatorF64::new();
        for pg in &self.param_grads {
            if let Some(ref grad) = pg {
                for &v in &grad.to_vec() {
                    acc.add(v * v);
                }
            }
        }
        let global_norm = acc.finalize().sqrt();

        if global_norm > max_norm && global_norm > 0.0 {
            let scale = max_norm / global_norm;
            for pg in &mut self.param_grads {
                if let Some(ref mut grad) = pg {
                    let data = grad.to_vec();
                    let scaled: Vec<f64> = data.iter().map(|&x| x * scale).collect();
                    let shape = grad.shape().to_vec();
                    *grad = Tensor::from_vec_unchecked(scaled, &shape);
                }
            }
        }

        global_norm
    }

    /// Mark the children of node `i` as reachable for dead-node elimination.
    fn mark_children_reachable(&self, i: usize, reachable: &mut [bool]) {
        match &self.ops[i] {
            GradOp::Input | GradOp::Parameter => {}
            GradOp::Add(a, b) | GradOp::Sub(a, b) | GradOp::Mul(a, b)
            | GradOp::Div(a, b) | GradOp::MatMul(a, b) => {
                reachable[*a] = true;
                reachable[*b] = true;
            }
            GradOp::Neg(a) | GradOp::Sum(a) | GradOp::Mean(a)
            | GradOp::Exp(a) | GradOp::Ln(a) | GradOp::Sin(a)
            | GradOp::Cos(a) | GradOp::Sqrt(a) | GradOp::Sigmoid(a)
            | GradOp::Relu(a) | GradOp::TanhAct(a) | GradOp::Abs(a)
            | GradOp::Log2(a) | GradOp::Softmax(a) | GradOp::LayerNorm(a)
            | GradOp::BatchNorm(a) | GradOp::TransposeOp(a)
            | GradOp::Gelu(a) | GradOp::Silu(a) | GradOp::Elu(a)
            | GradOp::Selu(a) => {
                reachable[*a] = true;
            }
            GradOp::ScalarMul(a, _) | GradOp::Pow(a, _) => {
                reachable[*a] = true;
            }
            GradOp::CrossEntropy { logits, targets } => {
                reachable[*logits] = true;
                reachable[*targets] = true;
            }
            GradOp::Clamp { input, .. } => {
                reachable[*input] = true;
            }
            GradOp::Where { cond, on_true, on_false } => {
                reachable[*cond] = true;
                reachable[*on_true] = true;
                reachable[*on_false] = true;
            }
            GradOp::Reshape { input, .. } => {
                reachable[*input] = true;
            }
            GradOp::CatOp { inputs, .. } => {
                for &idx in inputs {
                    reachable[idx] = true;
                }
            }
            GradOp::GatherOp { input, .. } => {
                reachable[*input] = true;
            }
            GradOp::BroadcastScalar { input, .. } => {
                reachable[*input] = true;
            }
            GradOp::StructField { parent, .. } => {
                reachable[*parent] = true;
            }
            GradOp::MapLookup { map_node, .. } => {
                reachable[*map_node] = true;
            }
            GradOp::MlpLayer { input, weight, bias, .. } => {
                reachable[*input] = true;
                reachable[*weight] = true;
                reachable[*bias] = true;
            }
        }
    }

    /// Run backward pass from a loss node.
    ///
    /// Includes dead-node elimination: only visits nodes reachable from the
    /// loss node, skipping unreachable branches (e.g. unused policy/value heads).
    pub fn backward(&mut self, loss_idx: usize) {
        let n = self.ops.len();

        // Phase 1: Build reachability set (O(N) pass)
        let mut reachable = vec![false; n];
        reachable[loss_idx] = true;
        for i in (0..=loss_idx).rev() {
            if !reachable[i] { continue; }
            self.mark_children_reachable(i, &mut reachable);
        }

        // Initialize gradients
        let mut grads: Vec<Option<Tensor>> = vec![None; n];

        // Loss gradient is 1.0
        let loss_shape = self.tensors[loss_idx].shape().to_vec();
        grads[loss_idx] = Some(Tensor::ones(&loss_shape));

        // Phase 2: Backward pass — skip unreachable nodes
        for i in (0..=loss_idx).rev() {
            if !reachable[i] { continue; }
            let grad = match grads[i].take() {
                Some(g) => g,
                None => continue,
            };

            // Borrow op and tensor by reference (no clone needed with flat arena)
            let op = self.ops[i].clone();
            let node_tensor = self.tensors[i].clone();

            match op {
                GradOp::Input => {}
                GradOp::Parameter => {
                    if let Some(ref mut existing_grad) = self.param_grads[i] {
                        *existing_grad = existing_grad.add_unchecked(&grad);
                    } else {
                        self.param_grads[i] = Some(grad);
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
                    let a_val = self.tensors[a].clone();
                    let b_val = self.tensors[b].clone();

                    let grad_a = grad.mul_elem_unchecked(&b_val);
                    let grad_b = grad.mul_elem_unchecked(&a_val);

                    accumulate_grad(&mut grads, a, &grad_a);
                    accumulate_grad(&mut grads, b, &grad_b);
                }
                GradOp::Div(a, b) => {
                    let a_val = self.tensors[a].clone();
                    let b_val = self.tensors[b].clone();

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
                    let a_val = self.tensors[a].clone();
                    let b_val = self.tensors[b].clone();

                    let b_t = b_val.transpose();
                    let a_t = a_val.transpose();

                    let grad_a = grad.matmul_unchecked(&b_t);
                    let grad_b = a_t.matmul_unchecked(&grad);

                    accumulate_grad(&mut grads, a, &grad_a);
                    accumulate_grad(&mut grads, b, &grad_b);
                }
                GradOp::Sum(a) => {
                    // Gradient of sum is all ones, scaled by upstream grad
                    let a_shape = self.tensors[a].shape().to_vec();
                    let grad_val = grad.to_vec()[0];
                    let expanded = Tensor::from_vec_unchecked(
                        vec![grad_val; a_shape.iter().product()],
                        &a_shape,
                    );
                    accumulate_grad(&mut grads, a, &expanded);
                }
                GradOp::Mean(a) => {
                    let a_shape = self.tensors[a].shape().to_vec();
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
                    let a_val = self.tensors[a].clone();
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
                    let a_val = self.tensors[a].clone();
                    let cos_a = Tensor::from_vec_unchecked(
                        a_val.to_vec().iter().map(|&x| x.cos()).collect(),
                        a_val.shape(),
                    );
                    let grad_a = grad.mul_elem_unchecked(&cos_a);
                    accumulate_grad(&mut grads, a, &grad_a);
                }
                GradOp::Cos(a) => {
                    let a_val = self.tensors[a].clone();
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
                    let a_val = self.tensors[a].clone();
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
                    let a_val = self.tensors[a].clone();
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
                GradOp::Gelu(a) => {
                    // GELU'(x) ≈ 0.5*(1+tanh(k)) + 0.5*x*(1-tanh(k)²)*k'
                    // where k = √(2/π)*(x + 0.044715*x³), k' = √(2/π)*(1 + 0.134145*x²)
                    let a_val = self.tensors[a].clone();
                    let local = Tensor::from_vec_unchecked(
                        a_val.to_vec().iter().map(|&x| {
                            let c = (2.0_f64 / std::f64::consts::PI).sqrt();
                            let k = c * (x + 0.044715 * x * x * x);
                            let tanh_k = k.tanh();
                            let dk = c * (1.0 + 3.0 * 0.044715 * x * x);
                            0.5 * (1.0 + tanh_k) + 0.5 * x * (1.0 - tanh_k * tanh_k) * dk
                        }).collect(),
                        a_val.shape(),
                    );
                    accumulate_grad(&mut grads, a, &grad.mul_elem_unchecked(&local));
                }
                GradOp::Silu(a) => {
                    // SiLU'(x) = σ(x) + x*σ(x)*(1-σ(x)) = σ(x)*(1 + x*(1-σ(x)))
                    let a_val = self.tensors[a].clone();
                    let local = Tensor::from_vec_unchecked(
                        a_val.to_vec().iter().map(|&x| {
                            let s = 1.0 / (1.0 + (-x).exp());
                            s * (1.0 + x * (1.0 - s))
                        }).collect(),
                        a_val.shape(),
                    );
                    accumulate_grad(&mut grads, a, &grad.mul_elem_unchecked(&local));
                }
                GradOp::Elu(a) => {
                    // ELU'(x) = 1 if x>0, else exp(x) (α=1)
                    let a_val = self.tensors[a].clone();
                    let local = Tensor::from_vec_unchecked(
                        a_val.to_vec().iter().map(|&x| if x > 0.0 { 1.0 } else { x.exp() }).collect(),
                        a_val.shape(),
                    );
                    accumulate_grad(&mut grads, a, &grad.mul_elem_unchecked(&local));
                }
                GradOp::Selu(a) => {
                    // SELU'(x) = λ if x>0, else λ*α*exp(x)
                    let a_val = self.tensors[a].clone();
                    let local = Tensor::from_vec_unchecked(
                        a_val.to_vec().iter().map(|&x| {
                            if x > 0.0 { GradGraph::SELU_LAMBDA } else { GradGraph::SELU_LAMBDA * GradGraph::SELU_ALPHA * x.exp() }
                        }).collect(),
                        a_val.shape(),
                    );
                    accumulate_grad(&mut grads, a, &grad.mul_elem_unchecked(&local));
                }
                // Phase 8: Extended AD backward
                GradOp::Abs(a) => {
                    let a_val = self.tensors[a].clone();
                    let sign = Tensor::from_vec_unchecked(
                        a_val.to_vec().iter().map(|&x| {
                            if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
                        }).collect(),
                        a_val.shape(),
                    );
                    let grad_a = grad.mul_elem_unchecked(&sign);
                    accumulate_grad(&mut grads, a, &grad_a);
                }
                GradOp::Log2(a) => {
                    // d/dx log2(x) = 1 / (x * ln(2))
                    let a_val = self.tensors[a].clone();
                    let ln2 = std::f64::consts::LN_2;
                    let local = Tensor::from_vec_unchecked(
                        a_val.to_vec().iter().map(|&x| 1.0 / (x * ln2)).collect(),
                        a_val.shape(),
                    );
                    let grad_a = grad.mul_elem_unchecked(&local);
                    accumulate_grad(&mut grads, a, &grad_a);
                }
                GradOp::Softmax(a) => {
                    // Jacobian-vector product: grad_input = softmax * (grad - sum(grad * softmax))
                    use cjc_repro::KahanAccumulatorF64;
                    let sm = &node_tensor;
                    let sm_data = sm.to_vec();
                    let grad_data = grad.to_vec();
                    let mut dot_acc = KahanAccumulatorF64::new();
                    for (&g, &s) in grad_data.iter().zip(sm_data.iter()) {
                        dot_acc.add(g * s);
                    }
                    let dot = dot_acc.finalize();
                    let grad_input: Vec<f64> = sm_data.iter().zip(grad_data.iter())
                        .map(|(&s, &g)| s * (g - dot))
                        .collect();
                    let grad_a = Tensor::from_vec_unchecked(grad_input, sm.shape());
                    accumulate_grad(&mut grads, a, &grad_a);
                }
                GradOp::CrossEntropy { logits, targets } => {
                    // Combined softmax + cross-entropy gradient: grad_logits = grad * (softmax - targets)
                    use cjc_repro::KahanAccumulatorF64;
                    let logits_val = self.tensors[logits].clone();
                    let targets_val = self.tensors[targets].clone();
                    let logits_data = logits_val.to_vec();
                    let targets_data = targets_val.to_vec();
                    // Compute softmax of logits (numerically stable)
                    let max_val = logits_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exp_shifted: Vec<f64> = logits_data.iter().map(|&x| (x - max_val).exp()).collect();
                    let mut sum_acc = KahanAccumulatorF64::new();
                    for &v in &exp_shifted {
                        sum_acc.add(v);
                    }
                    let sum_exp = sum_acc.finalize();
                    let softmax: Vec<f64> = exp_shifted.iter().map(|&e| e / sum_exp).collect();
                    // grad_logits = upstream_grad * (softmax - targets)
                    let upstream = grad.to_vec()[0]; // CE produces scalar
                    let grad_logits: Vec<f64> = softmax.iter().zip(targets_data.iter())
                        .map(|(&s, &t)| upstream * (s - t))
                        .collect();
                    let gl = Tensor::from_vec_unchecked(grad_logits, logits_val.shape());
                    accumulate_grad(&mut grads, logits, &gl);
                    // No gradient flows to targets (they are labels)
                }
                GradOp::LayerNorm(a) => {
                    // Layer norm backward:
                    // dx = (1/std) * (grad - mean(grad) - x_hat * mean(grad * x_hat))
                    // where x_hat = normalized output (node_tensor)
                    use cjc_repro::KahanAccumulatorF64;
                    let x_hat = &node_tensor;
                    let x_hat_data = x_hat.to_vec();
                    let grad_data = grad.to_vec();
                    let n = x_hat_data.len() as f64;
                    // Reconstruct std from input
                    let a_val = self.tensors[a].clone();
                    let a_data = a_val.to_vec();
                    let mut mean_acc = KahanAccumulatorF64::new();
                    for &v in &a_data {
                        mean_acc.add(v);
                    }
                    let mean = mean_acc.finalize() / n;
                    let mut var_acc = KahanAccumulatorF64::new();
                    for &v in &a_data {
                        let d = v - mean;
                        var_acc.add(d * d);
                    }
                    let var = var_acc.finalize() / n;
                    let eps = 1e-5;
                    let std_val = (var + eps).sqrt();
                    // mean(grad)
                    let mut mg_acc = KahanAccumulatorF64::new();
                    for &g in &grad_data {
                        mg_acc.add(g);
                    }
                    let mean_grad = mg_acc.finalize() / n;
                    // mean(grad * x_hat)
                    let mut mgx_acc = KahanAccumulatorF64::new();
                    for (&g, &xh) in grad_data.iter().zip(x_hat_data.iter()) {
                        mgx_acc.add(g * xh);
                    }
                    let mean_grad_xhat = mgx_acc.finalize() / n;
                    let dx: Vec<f64> = grad_data.iter().zip(x_hat_data.iter())
                        .map(|(&g, &xh)| (g - mean_grad - xh * mean_grad_xhat) / std_val)
                        .collect();
                    let grad_a = Tensor::from_vec_unchecked(dx, a_val.shape());
                    accumulate_grad(&mut grads, a, &grad_a);
                }
                GradOp::BatchNorm(a) => {
                    // Identical to LayerNorm backward for the per-tensor case
                    use cjc_repro::KahanAccumulatorF64;
                    let x_hat = &node_tensor;
                    let x_hat_data = x_hat.to_vec();
                    let grad_data = grad.to_vec();
                    let n = x_hat_data.len() as f64;
                    let a_val = self.tensors[a].clone();
                    let a_data = a_val.to_vec();
                    let mut mean_acc = KahanAccumulatorF64::new();
                    for &v in &a_data {
                        mean_acc.add(v);
                    }
                    let mean = mean_acc.finalize() / n;
                    let mut var_acc = KahanAccumulatorF64::new();
                    for &v in &a_data {
                        let d = v - mean;
                        var_acc.add(d * d);
                    }
                    let var = var_acc.finalize() / n;
                    let eps = 1e-5;
                    let std_val = (var + eps).sqrt();
                    let mut mg_acc = KahanAccumulatorF64::new();
                    for &g in &grad_data {
                        mg_acc.add(g);
                    }
                    let mean_grad = mg_acc.finalize() / n;
                    let mut mgx_acc = KahanAccumulatorF64::new();
                    for (&g, &xh) in grad_data.iter().zip(x_hat_data.iter()) {
                        mgx_acc.add(g * xh);
                    }
                    let mean_grad_xhat = mgx_acc.finalize() / n;
                    let dx: Vec<f64> = grad_data.iter().zip(x_hat_data.iter())
                        .map(|(&g, &xh)| (g - mean_grad - xh * mean_grad_xhat) / std_val)
                        .collect();
                    let grad_a = Tensor::from_vec_unchecked(dx, a_val.shape());
                    accumulate_grad(&mut grads, a, &grad_a);
                }
                GradOp::Clamp { input, min, max } => {
                    // Gradient passes through where input is in [min, max], else 0
                    let a_val = self.tensors[input].clone();
                    let mask = Tensor::from_vec_unchecked(
                        a_val.to_vec().iter().map(|&x| {
                            if x >= min && x <= max { 1.0 } else { 0.0 }
                        }).collect(),
                        a_val.shape(),
                    );
                    let grad_a = grad.mul_elem_unchecked(&mask);
                    accumulate_grad(&mut grads, input, &grad_a);
                }
                GradOp::Where { cond, on_true, on_false } => {
                    let cond_data = self.tensors[cond].to_vec();
                    let grad_data = grad.to_vec();
                    let shape = grad.shape().to_vec();
                    let grad_true: Vec<f64> = cond_data.iter().zip(grad_data.iter())
                        .map(|(&c, &g)| if c != 0.0 { g } else { 0.0 })
                        .collect();
                    let grad_false: Vec<f64> = cond_data.iter().zip(grad_data.iter())
                        .map(|(&c, &g)| if c != 0.0 { 0.0 } else { g })
                        .collect();
                    let gt = Tensor::from_vec_unchecked(grad_true, &shape);
                    let gf = Tensor::from_vec_unchecked(grad_false, &shape);
                    accumulate_grad(&mut grads, on_true, &gt);
                    accumulate_grad(&mut grads, on_false, &gf);
                    // No gradient to condition
                }
                GradOp::Reshape { input, ref original_shape } => {
                    // Backward: reshape grad back to original shape
                    let grad_a = grad.reshape(original_shape)
                        .expect("Reshape backward: shape mismatch");
                    accumulate_grad(&mut grads, input, &grad_a);
                }
                GradOp::TransposeOp(a) => {
                    // Transpose is its own inverse for 2-D
                    let grad_a = grad.transpose();
                    accumulate_grad(&mut grads, a, &grad_a);
                }
                GradOp::CatOp { ref inputs, axis, ref sizes } => {
                    // Split grad along axis into pieces matching sizes
                    let grad_data = grad.to_vec();
                    let grad_shape = grad.shape().to_vec();
                    let ndim = grad_shape.len();
                    if ndim == 1 {
                        let mut offset = 0usize;
                        for (&idx, &sz) in inputs.iter().zip(sizes.iter()) {
                            let piece = grad_data[offset..offset + sz].to_vec();
                            let gt = Tensor::from_vec_unchecked(piece, &[sz]);
                            accumulate_grad(&mut grads, idx, &gt);
                            offset += sz;
                        }
                    } else if ndim == 2 && axis == 0 {
                        let cols = grad_shape[1];
                        let mut row_offset = 0usize;
                        for (&idx, &sz) in inputs.iter().zip(sizes.iter()) {
                            let start = row_offset * cols;
                            let end = start + sz * cols;
                            let piece = grad_data[start..end].to_vec();
                            let gt = Tensor::from_vec_unchecked(piece, &[sz, cols]);
                            accumulate_grad(&mut grads, idx, &gt);
                            row_offset += sz;
                        }
                    } else if ndim == 2 && axis == 1 {
                        let nrows = grad_shape[0];
                        let total_cols = grad_shape[1];
                        for (input_idx, (&idx, &sz)) in inputs.iter().zip(sizes.iter()).enumerate() {
                            let mut piece = Vec::with_capacity(nrows * sz);
                            let col_offset: usize = sizes[..input_idx].iter().sum();
                            for row in 0..nrows {
                                let row_start = row * total_cols + col_offset;
                                piece.extend_from_slice(&grad_data[row_start..row_start + sz]);
                            }
                            let gt = Tensor::from_vec_unchecked(piece, &[nrows, sz]);
                            accumulate_grad(&mut grads, idx, &gt);
                        }
                    } else {
                        // General fallback: split flat data proportionally
                        let mut offset = 0usize;
                        for (&idx, &sz) in inputs.iter().zip(sizes.iter()) {
                            let piece_len = sz * grad_data.len() / grad_shape[axis];
                            let piece = grad_data[offset..offset + piece_len].to_vec();
                            let mut piece_shape = grad_shape.clone();
                            piece_shape[axis] = sz;
                            let gt = Tensor::from_vec_unchecked(piece, &piece_shape);
                            accumulate_grad(&mut grads, idx, &gt);
                            offset += piece_len;
                        }
                    }
                }
                GradOp::GatherOp { input, ref indices, axis } => {
                    // Scatter-add: distribute grad back to input positions
                    let input_shape = self.tensors[input].shape().to_vec();
                    let input_len: usize = input_shape.iter().product();
                    let mut scatter = vec![0.0_f64; input_len];
                    let grad_data = grad.to_vec();
                    if self.tensors[input].ndim() == 1 {
                        for (gi, &idx) in indices.iter().enumerate() {
                            scatter[idx] += grad_data[gi];
                        }
                    } else if axis == 0 && self.tensors[input].ndim() == 2 {
                        let cols = input_shape[1];
                        for (gi, &idx) in indices.iter().enumerate() {
                            for c in 0..cols {
                                scatter[idx * cols + c] += grad_data[gi * cols + c];
                            }
                        }
                    } else {
                        // Fallback: treat as flat index
                        for (gi, &idx) in indices.iter().enumerate() {
                            scatter[idx] += grad_data[gi];
                        }
                    }
                    let grad_a = Tensor::from_vec_unchecked(scatter, &input_shape);
                    accumulate_grad(&mut grads, input, &grad_a);
                }
                GradOp::MlpLayer { input, weight, bias, activation } => {
                    // Fused backward for activation(input @ weight^T + bias).
                    //
                    // Let z = input @ W^T + bias, output = activation(z).
                    // d_loss/d_z = d_loss/d_output * activation'(z)
                    // d_loss/d_input = d_z @ W
                    // d_loss/d_W = d_z^T @ input  (then transpose to match W shape)
                    // d_loss/d_bias = sum(d_z, axis=0)

                    // Recompute z (pre-activation) from stored input/weight/bias
                    let input_t = &self.tensors[input];
                    let weight_t = &self.tensors[weight];
                    let bias_t = &self.tensors[bias];
                    let wt = weight_t.transpose();
                    let z = input_t.matmul_unchecked(&wt).add_unchecked(bias_t);

                    // Apply activation derivative to get d_z
                    let dz = match activation {
                        crate::pinn::Activation::Tanh => {
                            // d/dz tanh(z) = 1 - tanh(z)^2
                            let output_data = self.tensors[i].to_vec();
                            let grad_data = grad.to_vec();
                            let shape = grad.shape().to_vec();
                            Tensor::from_vec_unchecked(
                                output_data.iter().zip(grad_data.iter())
                                    .map(|(&o, &g)| g * (1.0 - o * o))
                                    .collect(),
                                &shape,
                            )
                        }
                        crate::pinn::Activation::Sigmoid => {
                            // d/dz sigmoid(z) = sigmoid(z) * (1 - sigmoid(z))
                            let output_data = self.tensors[i].to_vec();
                            let grad_data = grad.to_vec();
                            let shape = grad.shape().to_vec();
                            Tensor::from_vec_unchecked(
                                output_data.iter().zip(grad_data.iter())
                                    .map(|(&o, &g)| g * o * (1.0 - o))
                                    .collect(),
                                &shape,
                            )
                        }
                        crate::pinn::Activation::Relu => {
                            let z_data = z.to_vec();
                            let grad_data = grad.to_vec();
                            let shape = grad.shape().to_vec();
                            Tensor::from_vec_unchecked(
                                z_data.iter().zip(grad_data.iter())
                                    .map(|(&z_val, &g)| if z_val > 0.0 { g } else { 0.0 })
                                    .collect(),
                                &shape,
                            )
                        }
                        crate::pinn::Activation::None => grad.clone(),
                        crate::pinn::Activation::Gelu => {
                            let z_data = z.to_vec();
                            let grad_data = grad.to_vec();
                            let shape = grad.shape().to_vec();
                            Tensor::from_vec_unchecked(
                                z_data.iter().zip(grad_data.iter()).map(|(&x, &g)| {
                                    let c = (2.0_f64 / std::f64::consts::PI).sqrt();
                                    let k = c * (x + 0.044715 * x * x * x);
                                    let tanh_k = k.tanh();
                                    let dk = c * (1.0 + 3.0 * 0.044715 * x * x);
                                    g * (0.5 * (1.0 + tanh_k) + 0.5 * x * (1.0 - tanh_k * tanh_k) * dk)
                                }).collect(),
                                &shape,
                            )
                        }
                        crate::pinn::Activation::Silu => {
                            let z_data = z.to_vec();
                            let grad_data = grad.to_vec();
                            let shape = grad.shape().to_vec();
                            Tensor::from_vec_unchecked(
                                z_data.iter().zip(grad_data.iter()).map(|(&x, &g)| {
                                    let s = 1.0 / (1.0 + (-x).exp());
                                    g * s * (1.0 + x * (1.0 - s))
                                }).collect(),
                                &shape,
                            )
                        }
                        crate::pinn::Activation::Elu => {
                            let z_data = z.to_vec();
                            let grad_data = grad.to_vec();
                            let shape = grad.shape().to_vec();
                            Tensor::from_vec_unchecked(
                                z_data.iter().zip(grad_data.iter()).map(|(&x, &g)| {
                                    if x > 0.0 { g } else { g * x.exp() }
                                }).collect(),
                                &shape,
                            )
                        }
                        crate::pinn::Activation::Selu => {
                            let z_data = z.to_vec();
                            let grad_data = grad.to_vec();
                            let shape = grad.shape().to_vec();
                            Tensor::from_vec_unchecked(
                                z_data.iter().zip(grad_data.iter()).map(|(&x, &g)| {
                                    if x > 0.0 { g * GradGraph::SELU_LAMBDA } else { g * GradGraph::SELU_LAMBDA * GradGraph::SELU_ALPHA * x.exp() }
                                }).collect(),
                                &shape,
                            )
                        }
                        crate::pinn::Activation::SinAct => {
                            // d/dz sin(z) = cos(z)
                            let z_data = z.to_vec();
                            let grad_data = grad.to_vec();
                            let shape = grad.shape().to_vec();
                            Tensor::from_vec_unchecked(
                                z_data.iter().zip(grad_data.iter()).map(|(&x, &g)| g * x.cos()).collect(),
                                &shape,
                            )
                        }
                    };

                    // d_input = dz @ W
                    let grad_input = dz.matmul_unchecked(weight_t);
                    accumulate_grad(&mut grads, input, &grad_input);

                    // d_W = dz^T @ input (produces [out_f, in_f] matching W shape)
                    let grad_weight = dz.transpose().matmul_unchecked(input_t);
                    accumulate_grad(&mut grads, weight, &grad_weight);

                    // d_bias = sum(dz, axis=0)
                    let dz_data = dz.to_vec();
                    let dz_shape = dz.shape();
                    if dz_shape.len() == 2 {
                        let (rows, cols) = (dz_shape[0], dz_shape[1]);
                        let mut bias_grad = vec![0.0_f64; cols];
                        for r in 0..rows {
                            for c in 0..cols {
                                bias_grad[c] += dz_data[r * cols + c];
                            }
                        }
                        accumulate_grad(&mut grads, bias, &Tensor::from_vec_unchecked(bias_grad, &[cols]));
                    } else {
                        accumulate_grad(&mut grads, bias, &dz);
                    }
                }
                GradOp::BroadcastScalar { input, .. } => {
                    // Forward: result[i] = scalar.value() for all i.
                    // Backward: dF/dscalar = sum of upstream over all output elements.
                    use cjc_repro::KahanAccumulatorF64;
                    let upstream_data = grad.to_vec();
                    let mut acc = KahanAccumulatorF64::new();
                    for v in upstream_data.iter() {
                        acc.add(*v);
                    }
                    let scalar_grad =
                        Tensor::from_vec_unchecked(vec![acc.finalize()], &[1]);
                    accumulate_grad(&mut grads, input, &scalar_grad);
                }
            }
        }
    }

    /// Compute the Jacobian of a vector-valued output node with respect to
    /// a parameter node. Returns a 2D tensor of shape [output_dim, param_dim].
    ///
    /// Strategy: run backward once per output element with a one-hot seed.
    pub fn jacobian(&mut self, output_idx: usize, param_idx: usize) -> Tensor {
        let output_shape = self.tensors[output_idx].shape().to_vec();
        let output_dim: usize = output_shape.iter().product();
        let param_shape = self.tensors[param_idx].shape().to_vec();
        let param_dim: usize = param_shape.iter().product();

        let mut jac_data = vec![0.0_f64; output_dim * param_dim];

        for i in 0..output_dim {
            // Zero all gradients
            self.zero_grad();

            // Create one-hot seed for output element i
            let mut seed = vec![0.0_f64; output_dim];
            seed[i] = 1.0;
            let seed_tensor = Tensor::from_vec_unchecked(seed, &output_shape);

            // Run backward with this seed
            self.backward_with_seed(output_idx, &seed_tensor);

            // Read gradient of param node
            let grad = self.param_grads[param_idx].clone();
            if let Some(g) = grad {
                let g_vec = g.to_vec();
                for j in 0..param_dim {
                    jac_data[i * param_dim + j] = g_vec[j];
                }
            }
        }

        Tensor::from_vec_unchecked(jac_data, &[output_dim, param_dim])
    }

    /// Compute the diagonal of the Hessian of a scalar loss with respect to
    /// a parameter node. Uses finite differences on the gradient
    /// (compute grad, perturb, re-compute grad).
    ///
    /// Returns a tensor of the same shape as the parameter.
    pub fn hessian_diag(&mut self, loss_idx: usize, param_idx: usize, eps: f64) -> Tensor {
        let param_shape = self.tensors[param_idx].shape().to_vec();
        let param_dim: usize = param_shape.iter().product();
        let original = self.tensors[param_idx].to_vec();
        let mut hess_diag = vec![0.0_f64; param_dim];

        for i in 0..param_dim {
            // Perturb +eps
            let mut plus = original.clone();
            plus[i] += eps;
            self.tensors[param_idx] =
                Tensor::from_vec_unchecked(plus, &param_shape);
            self.zero_grad();
            self.backward(loss_idx);
            let grad_plus = self.param_grads[param_idx]
                .as_ref()
                .map(|g| g.to_vec()[i])
                .unwrap_or(0.0);

            // Perturb -eps
            let mut minus = original.clone();
            minus[i] -= eps;
            self.tensors[param_idx] =
                Tensor::from_vec_unchecked(minus, &param_shape);
            self.zero_grad();
            self.backward(loss_idx);
            let grad_minus = self.param_grads[param_idx]
                .as_ref()
                .map(|g| g.to_vec()[i])
                .unwrap_or(0.0);

            hess_diag[i] = (grad_plus - grad_minus) / (2.0 * eps);
        }

        // Restore original parameter
        self.tensors[param_idx] =
            Tensor::from_vec_unchecked(original, &param_shape);

        Tensor::from_vec_unchecked(hess_diag, &param_shape)
    }

    /// Compute the full Hessian matrix of a scalar loss with respect to a parameter node.
    ///
    /// Returns a 2D tensor of shape [param_dim, param_dim] where H[i, j] = d²loss / (dp_i dp_j).
    ///
    /// Strategy: For each parameter element i, perturb param[i] by +eps and -eps, re-run
    /// the forward pass to update intermediate node values, then run backward() to get the
    /// gradient vector. The i-th row of the Hessian is (grad_plus - grad_minus) / (2 * eps).
    /// Uses eps = 1e-5 for accurate central differences.
    pub fn hessian(&mut self, loss_idx: usize, param_idx: usize) -> Tensor {
        let eps = 1e-5;
        let param_shape = self.tensors[param_idx].shape().to_vec();
        let param_dim: usize = param_shape.iter().product();
        let original = self.tensors[param_idx].to_vec();
        let mut hess_data = vec![0.0_f64; param_dim * param_dim];

        for i in 0..param_dim {
            // Perturb +eps at index i, re-forward so intermediate nodes are up-to-date
            let mut plus = original.clone();
            plus[i] += eps;
            self.tensors[param_idx] =
                Tensor::from_vec_unchecked(plus, &param_shape);
            self.reforward(param_idx + 1, loss_idx);
            self.zero_grad();
            self.backward(loss_idx);
            let grad_plus: Vec<f64> = self.param_grads[param_idx]
                .as_ref()
                .map(|g| g.to_vec())
                .unwrap_or_else(|| vec![0.0; param_dim]);

            // Perturb -eps at index i, re-forward
            let mut minus = original.clone();
            minus[i] -= eps;
            self.tensors[param_idx] =
                Tensor::from_vec_unchecked(minus, &param_shape);
            self.reforward(param_idx + 1, loss_idx);
            self.zero_grad();
            self.backward(loss_idx);
            let grad_minus: Vec<f64> = self.param_grads[param_idx]
                .as_ref()
                .map(|g| g.to_vec())
                .unwrap_or_else(|| vec![0.0; param_dim]);

            // Row i of the Hessian: (grad_plus[j] - grad_minus[j]) / (2 * eps) for each j
            for j in 0..param_dim {
                hess_data[i * param_dim + j] = (grad_plus[j] - grad_minus[j]) / (2.0 * eps);
            }
        }

        // Restore original parameter and re-forward to clean state
        self.tensors[param_idx] =
            Tensor::from_vec_unchecked(original, &param_shape);
        self.reforward(param_idx + 1, loss_idx);

        Tensor::from_vec_unchecked(hess_data, &[param_dim, param_dim])
    }

    /// Re-run the forward pass for all nodes from `start` up to and including `end`.
    ///
    /// This is needed before backward when a parameter has been perturbed, so that
    /// intermediate computation nodes hold updated tensor values.
    /// Recompute forward-pass tensors for nodes `start..=end`, skipping
    /// `Input` and `Parameter` nodes (whose tensors are assumed up-to-date).
    /// Call `set_tensor()` on any parameters that changed before calling this.
    pub fn reforward(&mut self, start: usize, end: usize) {
        for node_i in start..=end {
            let op = self.ops[node_i].clone();
            let new_tensor = match &op {
                GradOp::Input | GradOp::Parameter => continue,
                GradOp::Add(a, b) => {
                    let at = self.tensors[*a].clone();
                    let bt = self.tensors[*b].clone();
                    at.add_unchecked(&bt)
                }
                GradOp::Sub(a, b) => {
                    let at = self.tensors[*a].clone();
                    let bt = self.tensors[*b].clone();
                    at.sub_unchecked(&bt)
                }
                GradOp::Mul(a, b) => {
                    let at = self.tensors[*a].clone();
                    let bt = self.tensors[*b].clone();
                    at.mul_elem_unchecked(&bt)
                }
                GradOp::Div(a, b) => {
                    let at = self.tensors[*a].clone();
                    let bt = self.tensors[*b].clone();
                    at.div_elem_unchecked(&bt)
                }
                GradOp::Neg(a) => {
                    self.tensors[*a].neg()
                }
                GradOp::ScalarMul(a, s) => {
                    self.tensors[*a].scalar_mul(*s)
                }
                GradOp::MatMul(a, b) => {
                    let at = self.tensors[*a].clone();
                    let bt = self.tensors[*b].clone();
                    at.matmul_unchecked(&bt)
                }
                GradOp::Sum(a) => {
                    let s = self.tensors[*a].sum();
                    Tensor::from_vec_unchecked(vec![s], &[1])
                }
                GradOp::Mean(a) => {
                    let m = self.tensors[*a].mean();
                    Tensor::from_vec_unchecked(vec![m], &[1])
                }
                GradOp::Exp(a) => {
                    let data = self.tensors[*a].to_vec();
                    let shape = self.tensors[*a].shape().to_vec();
                    Tensor::from_vec_unchecked(data.iter().map(|x| x.exp()).collect(), &shape)
                }
                GradOp::Ln(a) => {
                    let data = self.tensors[*a].to_vec();
                    let shape = self.tensors[*a].shape().to_vec();
                    Tensor::from_vec_unchecked(data.iter().map(|x| x.ln()).collect(), &shape)
                }
                GradOp::Sin(a) => {
                    let data = self.tensors[*a].to_vec();
                    let shape = self.tensors[*a].shape().to_vec();
                    Tensor::from_vec_unchecked(data.iter().map(|x| x.sin()).collect(), &shape)
                }
                GradOp::Cos(a) => {
                    let data = self.tensors[*a].to_vec();
                    let shape = self.tensors[*a].shape().to_vec();
                    Tensor::from_vec_unchecked(data.iter().map(|x| x.cos()).collect(), &shape)
                }
                GradOp::Sqrt(a) => {
                    let data = self.tensors[*a].to_vec();
                    let shape = self.tensors[*a].shape().to_vec();
                    Tensor::from_vec_unchecked(data.iter().map(|x| x.sqrt()).collect(), &shape)
                }
                GradOp::Pow(a, n) => {
                    let n = *n;
                    let data = self.tensors[*a].to_vec();
                    let shape = self.tensors[*a].shape().to_vec();
                    Tensor::from_vec_unchecked(data.iter().map(|x| x.powf(n)).collect(), &shape)
                }
                GradOp::Sigmoid(a) => {
                    let data = self.tensors[*a].to_vec();
                    let shape = self.tensors[*a].shape().to_vec();
                    Tensor::from_vec_unchecked(
                        data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
                        &shape,
                    )
                }
                GradOp::Relu(a) => {
                    let data = self.tensors[*a].to_vec();
                    let shape = self.tensors[*a].shape().to_vec();
                    Tensor::from_vec_unchecked(
                        data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect(),
                        &shape,
                    )
                }
                GradOp::TanhAct(a) => {
                    let data = self.tensors[*a].to_vec();
                    let shape = self.tensors[*a].shape().to_vec();
                    Tensor::from_vec_unchecked(data.iter().map(|x| x.tanh()).collect(), &shape)
                }
                GradOp::Gelu(a) => {
                    let data = self.tensors[*a].to_vec();
                    let shape = self.tensors[*a].shape().to_vec();
                    Tensor::from_vec_unchecked(data.iter().map(|&x| {
                        let inner = (2.0_f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
                        0.5 * x * (1.0 + inner.tanh())
                    }).collect(), &shape)
                }
                GradOp::Silu(a) => {
                    let data = self.tensors[*a].to_vec();
                    let shape = self.tensors[*a].shape().to_vec();
                    Tensor::from_vec_unchecked(data.iter().map(|&x| x / (1.0 + (-x).exp())).collect(), &shape)
                }
                GradOp::Elu(a) => {
                    let data = self.tensors[*a].to_vec();
                    let shape = self.tensors[*a].shape().to_vec();
                    Tensor::from_vec_unchecked(data.iter().map(|&x| if x > 0.0 { x } else { x.exp() - 1.0 }).collect(), &shape)
                }
                GradOp::Selu(a) => {
                    let data = self.tensors[*a].to_vec();
                    let shape = self.tensors[*a].shape().to_vec();
                    Tensor::from_vec_unchecked(data.iter().map(|&x| {
                        if x > 0.0 { GradGraph::SELU_LAMBDA * x } else { GradGraph::SELU_LAMBDA * GradGraph::SELU_ALPHA * (x.exp() - 1.0) }
                    }).collect(), &shape)
                }
                GradOp::Abs(a) => {
                    let data = self.tensors[*a].to_vec();
                    let shape = self.tensors[*a].shape().to_vec();
                    Tensor::from_vec_unchecked(data.iter().map(|x| x.abs()).collect(), &shape)
                }
                GradOp::Clamp { input, min, max } => {
                    let min = *min;
                    let max = *max;
                    let data = self.tensors[*input].to_vec();
                    let shape = self.tensors[*input].shape().to_vec();
                    Tensor::from_vec_unchecked(
                        data.iter().map(|&x| x.max(min).min(max)).collect(),
                        &shape,
                    )
                }
                GradOp::Reshape { input, .. } => {
                    let current_shape = self.tensors[node_i].shape().to_vec();
                    let data = self.tensors[*input].to_vec();
                    Tensor::from_vec_unchecked(data, &current_shape)
                }
                GradOp::TransposeOp(a) => {
                    self.tensors[*a].transpose()
                }
                GradOp::MlpLayer { input, weight, bias, activation } => {
                    let input_t = &self.tensors[*input];
                    let weight_t = &self.tensors[*weight];
                    let bias_t = &self.tensors[*bias];
                    let wt = weight_t.transpose();
                    let z = input_t.matmul_unchecked(&wt).add_unchecked(bias_t);
                    match activation {
                        crate::pinn::Activation::Tanh => {
                            let data = z.to_vec();
                            Tensor::from_vec_unchecked(data.iter().map(|x| x.tanh()).collect(), z.shape())
                        }
                        crate::pinn::Activation::Sigmoid => {
                            let data = z.to_vec();
                            Tensor::from_vec_unchecked(data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(), z.shape())
                        }
                        crate::pinn::Activation::Relu => {
                            let data = z.to_vec();
                            Tensor::from_vec_unchecked(data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect(), z.shape())
                        }
                        crate::pinn::Activation::None => z,
                        crate::pinn::Activation::Gelu => {
                            let data = z.to_vec();
                            Tensor::from_vec_unchecked(data.iter().map(|&x| {
                                let inner = (2.0_f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
                                0.5 * x * (1.0 + inner.tanh())
                            }).collect(), z.shape())
                        }
                        crate::pinn::Activation::Silu => {
                            let data = z.to_vec();
                            Tensor::from_vec_unchecked(data.iter().map(|&x| x / (1.0 + (-x).exp())).collect(), z.shape())
                        }
                        crate::pinn::Activation::Elu => {
                            let data = z.to_vec();
                            Tensor::from_vec_unchecked(data.iter().map(|&x| if x > 0.0 { x } else { x.exp() - 1.0 }).collect(), z.shape())
                        }
                        crate::pinn::Activation::Selu => {
                            let data = z.to_vec();
                            Tensor::from_vec_unchecked(data.iter().map(|&x| {
                                if x > 0.0 { GradGraph::SELU_LAMBDA * x } else { GradGraph::SELU_LAMBDA * GradGraph::SELU_ALPHA * (x.exp() - 1.0) }
                            }).collect(), z.shape())
                        }
                        crate::pinn::Activation::SinAct => {
                            let data = z.to_vec();
                            Tensor::from_vec_unchecked(data.iter().map(|&x| x.sin()).collect(), z.shape())
                        }
                    }
                }
                // For complex ops (softmax, layernorm, etc.), keep existing tensor.
                // These are not typically used in simple Hessian computations.
                _ => self.tensors[node_i].clone(),
            };
            self.tensors[node_i] = new_tensor;
        }
    }

    /// Compute the second derivative of a scalar loss with respect to a parameter node.
    ///
    /// Implements double_backward via finite differences on the backward pass:
    /// perturbs the parameter by +eps/-eps, re-runs the forward and backward pass,
    /// and computes d(grad)/d(param) numerically. For a scalar param this gives the
    /// exact second derivative d²loss/dparam².
    ///
    /// Returns a tensor of the same shape as the parameter containing second derivatives.
    pub fn double_backward(&mut self, loss_idx: usize, param_idx: usize) -> Tensor {
        let eps = 1e-5;
        let param_shape = self.tensors[param_idx].shape().to_vec();
        let param_dim: usize = param_shape.iter().product();
        let original = self.tensors[param_idx].to_vec();
        let mut diag = vec![0.0_f64; param_dim];

        for i in 0..param_dim {
            // Perturb +eps, re-forward, backward
            let mut plus = original.clone();
            plus[i] += eps;
            self.tensors[param_idx] =
                Tensor::from_vec_unchecked(plus, &param_shape);
            self.reforward(param_idx + 1, loss_idx);
            self.zero_grad();
            self.backward(loss_idx);
            let grad_plus = self.param_grads[param_idx]
                .as_ref()
                .map(|g| g.to_vec()[i])
                .unwrap_or(0.0);

            // Perturb -eps, re-forward, backward
            let mut minus = original.clone();
            minus[i] -= eps;
            self.tensors[param_idx] =
                Tensor::from_vec_unchecked(minus, &param_shape);
            self.reforward(param_idx + 1, loss_idx);
            self.zero_grad();
            self.backward(loss_idx);
            let grad_minus = self.param_grads[param_idx]
                .as_ref()
                .map(|g| g.to_vec()[i])
                .unwrap_or(0.0);

            diag[i] = (grad_plus - grad_minus) / (2.0 * eps);
        }

        // Restore original parameter and re-forward to clean state
        self.tensors[param_idx] =
            Tensor::from_vec_unchecked(original, &param_shape);
        self.reforward(param_idx + 1, loss_idx);

        Tensor::from_vec_unchecked(diag, &param_shape)
    }

    /// Vectorized map (batched evaluation) over a batch dimension.
    ///
    /// For each tensor in `batch_data`, sets the input node `input_idx` to that tensor,
    /// re-evaluates all downstream nodes by re-running the forward pass (recomputing
    /// tensor values from the graph structure), and records the output node index
    /// after each evaluation.
    ///
    /// Returns a `Vec<usize>` of output node indices (one per batch element). After
    /// calling this, `g.value(results[k])` returns the output for batch element k.
    ///
    /// Note: This is a simple batched evaluation helper. It mutates node tensors
    /// in-place. After calling vmap_forward, the graph holds the values for the
    /// LAST batch element. Use `g.value(results[k])` to read individual results
    /// (stored in snapshot tensors inside each returned node).
    ///
    /// Implementation: For each batch element, set the input tensor, re-forward the
    /// subgraph from input_idx..=loss_idx by replaying each op, and record the final
    /// node's value in a fresh parameter node.
    pub fn vmap_forward(&mut self, input_idx: usize, batch_data: &[Tensor]) -> Vec<usize> {
        let mut result_indices = Vec::with_capacity(batch_data.len());

        // Identify the topological range: nodes from input_idx onward that depend on it.
        // We re-evaluate all nodes from input_idx to the end of the current graph.
        let graph_len = self.ops.len();

        for batch_tensor in batch_data {
            // Set input node to batch element
            self.tensors[input_idx] = batch_tensor.clone();

            // Re-run forward pass for all nodes after input_idx by replaying their ops
            for node_i in (input_idx + 1)..graph_len {
                let op = self.ops[node_i].clone();
                let new_tensor = match &op {
                    GradOp::Add(a, b) => {
                        let at = self.tensors[*a].clone();
                        let bt = self.tensors[*b].clone();
                        at.add_unchecked(&bt)
                    }
                    GradOp::Sub(a, b) => {
                        let at = self.tensors[*a].clone();
                        let bt = self.tensors[*b].clone();
                        at.sub_unchecked(&bt)
                    }
                    GradOp::Mul(a, b) => {
                        let at = self.tensors[*a].clone();
                        let bt = self.tensors[*b].clone();
                        at.mul_elem_unchecked(&bt)
                    }
                    GradOp::Div(a, b) => {
                        let at = self.tensors[*a].clone();
                        let bt = self.tensors[*b].clone();
                        at.div_elem_unchecked(&bt)
                    }
                    GradOp::Neg(a) => {
                        self.tensors[*a].neg()
                    }
                    GradOp::ScalarMul(a, s) => {
                        self.tensors[*a].scalar_mul(*s)
                    }
                    GradOp::MatMul(a, b) => {
                        let at = self.tensors[*a].clone();
                        let bt = self.tensors[*b].clone();
                        at.matmul_unchecked(&bt)
                    }
                    GradOp::Sum(a) => {
                        let s = self.tensors[*a].sum();
                        let shape = vec![1usize];
                        Tensor::from_vec_unchecked(vec![s], &shape)
                    }
                    GradOp::Mean(a) => {
                        let m = self.tensors[*a].mean();
                        Tensor::from_vec_unchecked(vec![m], &[1])
                    }
                    GradOp::Exp(a) => {
                        let data = self.tensors[*a].to_vec();
                        let shape = self.tensors[*a].shape().to_vec();
                        Tensor::from_vec_unchecked(
                            data.iter().map(|x| x.exp()).collect(),
                            &shape,
                        )
                    }
                    GradOp::Ln(a) => {
                        let data = self.tensors[*a].to_vec();
                        let shape = self.tensors[*a].shape().to_vec();
                        Tensor::from_vec_unchecked(
                            data.iter().map(|x| x.ln()).collect(),
                            &shape,
                        )
                    }
                    GradOp::Sin(a) => {
                        let data = self.tensors[*a].to_vec();
                        let shape = self.tensors[*a].shape().to_vec();
                        Tensor::from_vec_unchecked(
                            data.iter().map(|x| x.sin()).collect(),
                            &shape,
                        )
                    }
                    GradOp::Cos(a) => {
                        let data = self.tensors[*a].to_vec();
                        let shape = self.tensors[*a].shape().to_vec();
                        Tensor::from_vec_unchecked(
                            data.iter().map(|x| x.cos()).collect(),
                            &shape,
                        )
                    }
                    GradOp::Sqrt(a) => {
                        let data = self.tensors[*a].to_vec();
                        let shape = self.tensors[*a].shape().to_vec();
                        Tensor::from_vec_unchecked(
                            data.iter().map(|x| x.sqrt()).collect(),
                            &shape,
                        )
                    }
                    GradOp::Pow(a, n) => {
                        let data = self.tensors[*a].to_vec();
                        let shape = self.tensors[*a].shape().to_vec();
                        Tensor::from_vec_unchecked(
                            data.iter().map(|x| x.powf(*n)).collect(),
                            &shape,
                        )
                    }
                    GradOp::Sigmoid(a) => {
                        let data = self.tensors[*a].to_vec();
                        let shape = self.tensors[*a].shape().to_vec();
                        Tensor::from_vec_unchecked(
                            data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
                            &shape,
                        )
                    }
                    GradOp::Relu(a) => {
                        let data = self.tensors[*a].to_vec();
                        let shape = self.tensors[*a].shape().to_vec();
                        Tensor::from_vec_unchecked(
                            data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect(),
                            &shape,
                        )
                    }
                    GradOp::TanhAct(a) => {
                        let data = self.tensors[*a].to_vec();
                        let shape = self.tensors[*a].shape().to_vec();
                        Tensor::from_vec_unchecked(
                            data.iter().map(|x| x.tanh()).collect(),
                            &shape,
                        )
                    }
                    GradOp::Gelu(a) => {
                        let data = self.tensors[*a].to_vec();
                        let shape = self.tensors[*a].shape().to_vec();
                        Tensor::from_vec_unchecked(data.iter().map(|&x| {
                            let inner = (2.0_f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
                            0.5 * x * (1.0 + inner.tanh())
                        }).collect(), &shape)
                    }
                    GradOp::Silu(a) => {
                        let data = self.tensors[*a].to_vec();
                        let shape = self.tensors[*a].shape().to_vec();
                        Tensor::from_vec_unchecked(data.iter().map(|&x| x / (1.0 + (-x).exp())).collect(), &shape)
                    }
                    GradOp::Elu(a) => {
                        let data = self.tensors[*a].to_vec();
                        let shape = self.tensors[*a].shape().to_vec();
                        Tensor::from_vec_unchecked(data.iter().map(|&x| if x > 0.0 { x } else { x.exp() - 1.0 }).collect(), &shape)
                    }
                    GradOp::Selu(a) => {
                        let data = self.tensors[*a].to_vec();
                        let shape = self.tensors[*a].shape().to_vec();
                        Tensor::from_vec_unchecked(data.iter().map(|&x| {
                            if x > 0.0 { GradGraph::SELU_LAMBDA * x } else { GradGraph::SELU_LAMBDA * GradGraph::SELU_ALPHA * (x.exp() - 1.0) }
                        }).collect(), &shape)
                    }
                    GradOp::Abs(a) => {
                        let data = self.tensors[*a].to_vec();
                        let shape = self.tensors[*a].shape().to_vec();
                        Tensor::from_vec_unchecked(
                            data.iter().map(|x| x.abs()).collect(),
                            &shape,
                        )
                    }
                    GradOp::Clamp { input, min, max } => {
                        let data = self.tensors[*input].to_vec();
                        let shape = self.tensors[*input].shape().to_vec();
                        Tensor::from_vec_unchecked(
                            data.iter().map(|&x| x.max(*min).min(*max)).collect(),
                            &shape,
                        )
                    }
                    GradOp::Reshape { input, .. } => {
                        // Keep same data, use current node's shape
                        let data = self.tensors[*input].to_vec();
                        let shape = self.tensors[node_i].shape().to_vec();
                        Tensor::from_vec_unchecked(data, &shape)
                    }
                    GradOp::TransposeOp(a) => {
                        self.tensors[*a].transpose()
                    }
                    // For complex ops and ops without direct input dependency on input_idx,
                    // keep the existing tensor value (no re-computation needed).
                    _ => self.tensors[node_i].clone(),
                };
                self.tensors[node_i] = new_tensor;
            }

            // Record the output value from the last node (graph_len - 1) by creating
            // a snapshot input node with the current output value.
            let output_tensor = self.tensors[graph_len - 1].clone();
            let snapshot_idx = self.ops.len();
            self.ops.push(GradOp::Input);
            self.tensors.push(output_tensor);
            self.param_grads.push(None);
            result_indices.push(snapshot_idx);
        }

        result_indices
    }

    /// Backward pass with a custom gradient seed tensor (for Jacobian computation).
    pub fn backward_with_seed(&mut self, loss_idx: usize, seed: &Tensor) {
        let n = self.ops.len();
        let mut grads: Vec<Option<Tensor>> = vec![None; n];
        grads[loss_idx] = Some(seed.clone());

        for i in (0..n).rev() {
            let grad = match grads[i].take() {
                Some(g) => g,
                None => continue,
            };

            if let Some(ref _param_grad) = self.param_grads[i] {
                // Accumulate into parameter grad storage
                let new_grad = {
                    if let Some(ref existing) = self.param_grads[i] {
                        if existing.to_vec().iter().all(|&x| x == 0.0) {
                            grad.clone()
                        } else {
                            existing.add_unchecked(&grad)
                        }
                    } else {
                        grad.clone()
                    }
                };
                self.param_grads[i] = Some(new_grad);
            }

            // Propagate gradients using the same rules as backward()
            let op = self.ops[i].clone();
            let node_tensor = self.tensors[i].clone();
            match &op {
                GradOp::Input | GradOp::Parameter => {}
                GradOp::Add(a, b) => {
                    accumulate_grad(&mut grads, *a, &grad);
                    accumulate_grad(&mut grads, *b, &grad);
                }
                GradOp::Sub(a, b) => {
                    accumulate_grad(&mut grads, *a, &grad);
                    accumulate_grad(&mut grads, *b, &grad.neg());
                }
                GradOp::Mul(a, b) => {
                    let a_val = self.tensors[*a].clone();
                    let b_val = self.tensors[*b].clone();
                    accumulate_grad(&mut grads, *a, &grad.mul_elem_unchecked(&b_val));
                    accumulate_grad(&mut grads, *b, &grad.mul_elem_unchecked(&a_val));
                }
                GradOp::Div(a, b) => {
                    let a_val = self.tensors[*a].clone();
                    let b_val = self.tensors[*b].clone();
                    let grad_a = grad.div_elem_unchecked(&b_val);
                    let neg_a_over_b2 = a_val.neg().div_elem_unchecked(
                        &b_val.mul_elem_unchecked(&b_val),
                    );
                    let grad_b = grad.mul_elem_unchecked(&neg_a_over_b2);
                    accumulate_grad(&mut grads, *a, &grad_a);
                    accumulate_grad(&mut grads, *b, &grad_b);
                }
                GradOp::Neg(a) => {
                    accumulate_grad(&mut grads, *a, &grad.neg());
                }
                GradOp::ScalarMul(a, s) => {
                    accumulate_grad(&mut grads, *a, &grad.scalar_mul(*s));
                }
                GradOp::Exp(a) => {
                    accumulate_grad(&mut grads, *a, &grad.mul_elem_unchecked(&node_tensor));
                }
                GradOp::Ln(a) => {
                    let a_val = self.tensors[*a].clone();
                    let inv = Tensor::from_vec_unchecked(
                        a_val.to_vec().iter().map(|&x| 1.0 / x).collect(),
                        a_val.shape(),
                    );
                    accumulate_grad(&mut grads, *a, &grad.mul_elem_unchecked(&inv));
                }
                GradOp::Sin(a) => {
                    let a_val = self.tensors[*a].clone();
                    let cos_a = Tensor::from_vec_unchecked(
                        a_val.to_vec().iter().map(|&x| x.cos()).collect(),
                        a_val.shape(),
                    );
                    accumulate_grad(&mut grads, *a, &grad.mul_elem_unchecked(&cos_a));
                }
                GradOp::Cos(a) => {
                    let a_val = self.tensors[*a].clone();
                    let neg_sin = Tensor::from_vec_unchecked(
                        a_val.to_vec().iter().map(|&x| -x.sin()).collect(),
                        a_val.shape(),
                    );
                    accumulate_grad(&mut grads, *a, &grad.mul_elem_unchecked(&neg_sin));
                }
                GradOp::Sqrt(a) => {
                    let inv2sqrt = Tensor::from_vec_unchecked(
                        node_tensor.to_vec().iter().map(|&x| 0.5 / x).collect(),
                        node_tensor.shape(),
                    );
                    accumulate_grad(&mut grads, *a, &grad.mul_elem_unchecked(&inv2sqrt));
                }
                GradOp::Pow(a, exp) => {
                    let a_val = self.tensors[*a].clone();
                    let local = Tensor::from_vec_unchecked(
                        a_val.to_vec().iter().map(|&x| exp * x.powf(exp - 1.0)).collect(),
                        a_val.shape(),
                    );
                    accumulate_grad(&mut grads, *a, &grad.mul_elem_unchecked(&local));
                }
                GradOp::Sigmoid(a) => {
                    let sig = &node_tensor;
                    let one_minus = Tensor::from_vec_unchecked(
                        sig.to_vec().iter().map(|&x| 1.0 - x).collect(),
                        sig.shape(),
                    );
                    let local = sig.mul_elem_unchecked(&one_minus);
                    accumulate_grad(&mut grads, *a, &grad.mul_elem_unchecked(&local));
                }
                GradOp::Relu(a) => {
                    let a_val = self.tensors[*a].clone();
                    let mask = Tensor::from_vec_unchecked(
                        a_val.to_vec().iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect(),
                        a_val.shape(),
                    );
                    accumulate_grad(&mut grads, *a, &grad.mul_elem_unchecked(&mask));
                }
                GradOp::TanhAct(a) => {
                    let one_minus_sq = Tensor::from_vec_unchecked(
                        node_tensor.to_vec().iter().map(|&x| 1.0 - x * x).collect(),
                        node_tensor.shape(),
                    );
                    accumulate_grad(&mut grads, *a, &grad.mul_elem_unchecked(&one_minus_sq));
                }
                GradOp::Gelu(a) => {
                    let a_val = self.tensors[*a].clone();
                    let local = Tensor::from_vec_unchecked(
                        a_val.to_vec().iter().map(|&x| {
                            let c = (2.0_f64 / std::f64::consts::PI).sqrt();
                            let k = c * (x + 0.044715 * x * x * x);
                            let tanh_k = k.tanh();
                            let dk = c * (1.0 + 3.0 * 0.044715 * x * x);
                            0.5 * (1.0 + tanh_k) + 0.5 * x * (1.0 - tanh_k * tanh_k) * dk
                        }).collect(),
                        a_val.shape(),
                    );
                    accumulate_grad(&mut grads, *a, &grad.mul_elem_unchecked(&local));
                }
                GradOp::Silu(a) => {
                    let a_val = self.tensors[*a].clone();
                    let local = Tensor::from_vec_unchecked(
                        a_val.to_vec().iter().map(|&x| {
                            let s = 1.0 / (1.0 + (-x).exp());
                            s * (1.0 + x * (1.0 - s))
                        }).collect(),
                        a_val.shape(),
                    );
                    accumulate_grad(&mut grads, *a, &grad.mul_elem_unchecked(&local));
                }
                GradOp::Elu(a) => {
                    let a_val = self.tensors[*a].clone();
                    let local = Tensor::from_vec_unchecked(
                        a_val.to_vec().iter().map(|&x| if x > 0.0 { 1.0 } else { x.exp() }).collect(),
                        a_val.shape(),
                    );
                    accumulate_grad(&mut grads, *a, &grad.mul_elem_unchecked(&local));
                }
                GradOp::Selu(a) => {
                    let a_val = self.tensors[*a].clone();
                    let local = Tensor::from_vec_unchecked(
                        a_val.to_vec().iter().map(|&x| {
                            if x > 0.0 { GradGraph::SELU_LAMBDA } else { GradGraph::SELU_LAMBDA * GradGraph::SELU_ALPHA * x.exp() }
                        }).collect(),
                        a_val.shape(),
                    );
                    accumulate_grad(&mut grads, *a, &grad.mul_elem_unchecked(&local));
                }
                GradOp::MatMul(a, b) => {
                    let a_val = self.tensors[*a].clone();
                    let b_val = self.tensors[*b].clone();
                    accumulate_grad(&mut grads, *a, &grad.matmul_unchecked(&b_val.transpose()));
                    accumulate_grad(&mut grads, *b, &a_val.transpose().matmul_unchecked(&grad));
                }
                GradOp::Sum(a) => {
                    let a_shape = self.tensors[*a].shape().to_vec();
                    let grad_val = grad.to_vec()[0];
                    let expanded = Tensor::from_vec_unchecked(
                        vec![grad_val; a_shape.iter().product()],
                        &a_shape,
                    );
                    accumulate_grad(&mut grads, *a, &expanded);
                }
                GradOp::Mean(a) => {
                    let a_shape = self.tensors[*a].shape().to_vec();
                    let n_elem = a_shape.iter().product::<usize>() as f64;
                    let grad_val = grad.to_vec()[0] / n_elem;
                    let expanded = Tensor::from_vec_unchecked(
                        vec![grad_val; a_shape.iter().product()],
                        &a_shape,
                    );
                    accumulate_grad(&mut grads, *a, &expanded);
                }
                GradOp::StructField { parent, field_index, total_fields } => {
                    let parent_shape = self.tensors[*parent].shape().to_vec();
                    let parent_n: usize = parent_shape.iter().product();
                    let chunk = parent_n / total_fields;
                    let start = field_index * chunk;
                    let mut parent_grad = vec![0.0_f64; parent_n];
                    let g_vec = grad.to_vec();
                    for (j, &gv) in g_vec.iter().enumerate() {
                        parent_grad[start + j] = gv;
                    }
                    let pg = Tensor::from_vec_unchecked(parent_grad, &parent_shape);
                    accumulate_grad(&mut grads, *parent, &pg);
                }
                GradOp::MapLookup { map_node, key_index, total_keys } => {
                    let map_shape = self.tensors[*map_node].shape().to_vec();
                    let map_n: usize = map_shape.iter().product();
                    let chunk = map_n / total_keys;
                    let start = key_index * chunk;
                    let mut map_grad = vec![0.0_f64; map_n];
                    let g_vec = grad.to_vec();
                    for (j, &gv) in g_vec.iter().enumerate() {
                        map_grad[start + j] = gv;
                    }
                    let mg = Tensor::from_vec_unchecked(map_grad, &map_shape);
                    accumulate_grad(&mut grads, *map_node, &mg);
                }
                // Phase 8: Extended AD backward (backward_with_seed)
                GradOp::Abs(a) => {
                    let a_val = self.tensors[*a].clone();
                    let sign = Tensor::from_vec_unchecked(
                        a_val.to_vec().iter().map(|&x| {
                            if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
                        }).collect(),
                        a_val.shape(),
                    );
                    accumulate_grad(&mut grads, *a, &grad.mul_elem_unchecked(&sign));
                }
                GradOp::Log2(a) => {
                    let a_val = self.tensors[*a].clone();
                    let ln2 = std::f64::consts::LN_2;
                    let local = Tensor::from_vec_unchecked(
                        a_val.to_vec().iter().map(|&x| 1.0 / (x * ln2)).collect(),
                        a_val.shape(),
                    );
                    accumulate_grad(&mut grads, *a, &grad.mul_elem_unchecked(&local));
                }
                GradOp::Softmax(a) => {
                    use cjc_repro::KahanAccumulatorF64;
                    let sm = &node_tensor;
                    let sm_data = sm.to_vec();
                    let grad_data = grad.to_vec();
                    let mut dot_acc = KahanAccumulatorF64::new();
                    for (&g, &s) in grad_data.iter().zip(sm_data.iter()) {
                        dot_acc.add(g * s);
                    }
                    let dot = dot_acc.finalize();
                    let grad_input: Vec<f64> = sm_data.iter().zip(grad_data.iter())
                        .map(|(&s, &g)| s * (g - dot))
                        .collect();
                    let grad_a = Tensor::from_vec_unchecked(grad_input, sm.shape());
                    accumulate_grad(&mut grads, *a, &grad_a);
                }
                GradOp::CrossEntropy { logits, targets } => {
                    use cjc_repro::KahanAccumulatorF64;
                    let logits_val = self.tensors[*logits].clone();
                    let targets_val = self.tensors[*targets].clone();
                    let logits_data = logits_val.to_vec();
                    let targets_data = targets_val.to_vec();
                    let max_val = logits_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exp_shifted: Vec<f64> = logits_data.iter().map(|&x| (x - max_val).exp()).collect();
                    let mut sum_acc = KahanAccumulatorF64::new();
                    for &v in &exp_shifted {
                        sum_acc.add(v);
                    }
                    let sum_exp = sum_acc.finalize();
                    let softmax: Vec<f64> = exp_shifted.iter().map(|&e| e / sum_exp).collect();
                    let upstream = grad.to_vec()[0];
                    let grad_logits: Vec<f64> = softmax.iter().zip(targets_data.iter())
                        .map(|(&s, &t)| upstream * (s - t))
                        .collect();
                    let gl = Tensor::from_vec_unchecked(grad_logits, logits_val.shape());
                    accumulate_grad(&mut grads, *logits, &gl);
                }
                GradOp::LayerNorm(a) => {
                    use cjc_repro::KahanAccumulatorF64;
                    let x_hat = &node_tensor;
                    let x_hat_data = x_hat.to_vec();
                    let grad_data = grad.to_vec();
                    let n = x_hat_data.len() as f64;
                    let a_val = self.tensors[*a].clone();
                    let a_data = a_val.to_vec();
                    let mut mean_acc = KahanAccumulatorF64::new();
                    for &v in &a_data { mean_acc.add(v); }
                    let mean = mean_acc.finalize() / n;
                    let mut var_acc = KahanAccumulatorF64::new();
                    for &v in &a_data { let d = v - mean; var_acc.add(d * d); }
                    let var = var_acc.finalize() / n;
                    let std_val = (var + 1e-5).sqrt();
                    let mut mg_acc = KahanAccumulatorF64::new();
                    for &g in &grad_data { mg_acc.add(g); }
                    let mean_grad = mg_acc.finalize() / n;
                    let mut mgx_acc = KahanAccumulatorF64::new();
                    for (&g, &xh) in grad_data.iter().zip(x_hat_data.iter()) { mgx_acc.add(g * xh); }
                    let mean_grad_xhat = mgx_acc.finalize() / n;
                    let dx: Vec<f64> = grad_data.iter().zip(x_hat_data.iter())
                        .map(|(&g, &xh)| (g - mean_grad - xh * mean_grad_xhat) / std_val)
                        .collect();
                    accumulate_grad(&mut grads, *a, &Tensor::from_vec_unchecked(dx, a_val.shape()));
                }
                GradOp::BatchNorm(a) => {
                    use cjc_repro::KahanAccumulatorF64;
                    let x_hat = &node_tensor;
                    let x_hat_data = x_hat.to_vec();
                    let grad_data = grad.to_vec();
                    let n = x_hat_data.len() as f64;
                    let a_val = self.tensors[*a].clone();
                    let a_data = a_val.to_vec();
                    let mut mean_acc = KahanAccumulatorF64::new();
                    for &v in &a_data { mean_acc.add(v); }
                    let mean = mean_acc.finalize() / n;
                    let mut var_acc = KahanAccumulatorF64::new();
                    for &v in &a_data { let d = v - mean; var_acc.add(d * d); }
                    let var = var_acc.finalize() / n;
                    let std_val = (var + 1e-5).sqrt();
                    let mut mg_acc = KahanAccumulatorF64::new();
                    for &g in &grad_data { mg_acc.add(g); }
                    let mean_grad = mg_acc.finalize() / n;
                    let mut mgx_acc = KahanAccumulatorF64::new();
                    for (&g, &xh) in grad_data.iter().zip(x_hat_data.iter()) { mgx_acc.add(g * xh); }
                    let mean_grad_xhat = mgx_acc.finalize() / n;
                    let dx: Vec<f64> = grad_data.iter().zip(x_hat_data.iter())
                        .map(|(&g, &xh)| (g - mean_grad - xh * mean_grad_xhat) / std_val)
                        .collect();
                    accumulate_grad(&mut grads, *a, &Tensor::from_vec_unchecked(dx, a_val.shape()));
                }
                GradOp::Clamp { input, min, max } => {
                    let a_val = self.tensors[*input].clone();
                    let mask = Tensor::from_vec_unchecked(
                        a_val.to_vec().iter().map(|&x| {
                            if x >= *min && x <= *max { 1.0 } else { 0.0 }
                        }).collect(),
                        a_val.shape(),
                    );
                    accumulate_grad(&mut grads, *input, &grad.mul_elem_unchecked(&mask));
                }
                GradOp::Where { cond, on_true, on_false } => {
                    let cond_data = self.tensors[*cond].to_vec();
                    let grad_data = grad.to_vec();
                    let shape = grad.shape().to_vec();
                    let grad_true: Vec<f64> = cond_data.iter().zip(grad_data.iter())
                        .map(|(&c, &g)| if c != 0.0 { g } else { 0.0 }).collect();
                    let grad_false: Vec<f64> = cond_data.iter().zip(grad_data.iter())
                        .map(|(&c, &g)| if c != 0.0 { 0.0 } else { g }).collect();
                    accumulate_grad(&mut grads, *on_true, &Tensor::from_vec_unchecked(grad_true, &shape));
                    accumulate_grad(&mut grads, *on_false, &Tensor::from_vec_unchecked(grad_false, &shape));
                }
                GradOp::Reshape { input, ref original_shape } => {
                    let grad_a = grad.reshape(original_shape).expect("Reshape backward: shape mismatch");
                    accumulate_grad(&mut grads, *input, &grad_a);
                }
                GradOp::TransposeOp(a) => {
                    accumulate_grad(&mut grads, *a, &grad.transpose());
                }
                GradOp::CatOp { ref inputs, axis, ref sizes } => {
                    let grad_data = grad.to_vec();
                    let grad_shape = grad.shape().to_vec();
                    let ndim = grad_shape.len();
                    if ndim == 1 {
                        let mut offset = 0usize;
                        for (idx, &sz) in inputs.iter().zip(sizes.iter()) {
                            let piece = grad_data[offset..offset + sz].to_vec();
                            accumulate_grad(&mut grads, *idx, &Tensor::from_vec_unchecked(piece, &[sz]));
                            offset += sz;
                        }
                    } else if ndim == 2 && *axis == 0 {
                        let cols = grad_shape[1];
                        let mut row_offset = 0usize;
                        for (idx, &sz) in inputs.iter().zip(sizes.iter()) {
                            let start = row_offset * cols;
                            let end = start + sz * cols;
                            let piece = grad_data[start..end].to_vec();
                            accumulate_grad(&mut grads, *idx, &Tensor::from_vec_unchecked(piece, &[sz, cols]));
                            row_offset += sz;
                        }
                    } else if ndim == 2 && *axis == 1 {
                        let nrows = grad_shape[0];
                        let total_cols = grad_shape[1];
                        for (input_idx, (idx, &sz)) in inputs.iter().zip(sizes.iter()).enumerate() {
                            let mut piece = Vec::with_capacity(nrows * sz);
                            let col_offset: usize = sizes[..input_idx].iter().sum();
                            for row in 0..nrows {
                                let row_start = row * total_cols + col_offset;
                                piece.extend_from_slice(&grad_data[row_start..row_start + sz]);
                            }
                            accumulate_grad(&mut grads, *idx, &Tensor::from_vec_unchecked(piece, &[nrows, sz]));
                        }
                    } else {
                        let mut offset = 0usize;
                        for (idx, &sz) in inputs.iter().zip(sizes.iter()) {
                            let piece_len = sz * grad_data.len() / grad_shape[*axis];
                            let piece = grad_data[offset..offset + piece_len].to_vec();
                            let mut piece_shape = grad_shape.clone();
                            piece_shape[*axis] = sz;
                            accumulate_grad(&mut grads, *idx, &Tensor::from_vec_unchecked(piece, &piece_shape));
                            offset += piece_len;
                        }
                    }
                }
                GradOp::GatherOp { input, ref indices, axis } => {
                    let input_shape = self.tensors[*input].shape().to_vec();
                    let input_len: usize = input_shape.iter().product();
                    let mut scatter = vec![0.0_f64; input_len];
                    let grad_data = grad.to_vec();
                    if self.tensors[*input].ndim() == 1 {
                        for (gi, &idx) in indices.iter().enumerate() {
                            scatter[idx] += grad_data[gi];
                        }
                    } else if *axis == 0 && self.tensors[*input].ndim() == 2 {
                        let cols = input_shape[1];
                        for (gi, &idx) in indices.iter().enumerate() {
                            for c in 0..cols {
                                scatter[idx * cols + c] += grad_data[gi * cols + c];
                            }
                        }
                    } else {
                        for (gi, &idx) in indices.iter().enumerate() {
                            scatter[idx] += grad_data[gi];
                        }
                    }
                    accumulate_grad(&mut grads, *input, &Tensor::from_vec_unchecked(scatter, &input_shape));
                }
                GradOp::MlpLayer { input, weight, bias, activation } => {
                    let input_t = &self.tensors[*input];
                    let weight_t = &self.tensors[*weight];
                    let bias_t = &self.tensors[*bias];
                    let wt = weight_t.transpose();
                    let z = input_t.matmul_unchecked(&wt).add_unchecked(bias_t);

                    let dz = match activation {
                        crate::pinn::Activation::Tanh => {
                            let output_data = node_tensor.to_vec();
                            let grad_data = grad.to_vec();
                            Tensor::from_vec_unchecked(
                                output_data.iter().zip(grad_data.iter())
                                    .map(|(&o, &g)| g * (1.0 - o * o)).collect(),
                                grad.shape(),
                            )
                        }
                        crate::pinn::Activation::Sigmoid => {
                            let output_data = node_tensor.to_vec();
                            let grad_data = grad.to_vec();
                            Tensor::from_vec_unchecked(
                                output_data.iter().zip(grad_data.iter())
                                    .map(|(&o, &g)| g * o * (1.0 - o)).collect(),
                                grad.shape(),
                            )
                        }
                        crate::pinn::Activation::Relu => {
                            let z_data = z.to_vec();
                            let grad_data = grad.to_vec();
                            Tensor::from_vec_unchecked(
                                z_data.iter().zip(grad_data.iter())
                                    .map(|(&z_val, &g)| if z_val > 0.0 { g } else { 0.0 }).collect(),
                                grad.shape(),
                            )
                        }
                        crate::pinn::Activation::None => grad.clone(),
                        crate::pinn::Activation::Gelu => {
                            let z_data = z.to_vec();
                            let grad_data = grad.to_vec();
                            Tensor::from_vec_unchecked(
                                z_data.iter().zip(grad_data.iter()).map(|(&x, &g)| {
                                    let c = (2.0_f64 / std::f64::consts::PI).sqrt();
                                    let k = c * (x + 0.044715 * x * x * x);
                                    let tanh_k = k.tanh();
                                    let dk = c * (1.0 + 3.0 * 0.044715 * x * x);
                                    g * (0.5 * (1.0 + tanh_k) + 0.5 * x * (1.0 - tanh_k * tanh_k) * dk)
                                }).collect(),
                                grad.shape(),
                            )
                        }
                        crate::pinn::Activation::Silu => {
                            let z_data = z.to_vec();
                            let grad_data = grad.to_vec();
                            Tensor::from_vec_unchecked(
                                z_data.iter().zip(grad_data.iter()).map(|(&x, &g)| {
                                    let s = 1.0 / (1.0 + (-x).exp());
                                    g * s * (1.0 + x * (1.0 - s))
                                }).collect(),
                                grad.shape(),
                            )
                        }
                        crate::pinn::Activation::Elu => {
                            let z_data = z.to_vec();
                            let grad_data = grad.to_vec();
                            Tensor::from_vec_unchecked(
                                z_data.iter().zip(grad_data.iter()).map(|(&x, &g)| {
                                    if x > 0.0 { g } else { g * x.exp() }
                                }).collect(),
                                grad.shape(),
                            )
                        }
                        crate::pinn::Activation::Selu => {
                            let z_data = z.to_vec();
                            let grad_data = grad.to_vec();
                            Tensor::from_vec_unchecked(
                                z_data.iter().zip(grad_data.iter()).map(|(&x, &g)| {
                                    if x > 0.0 { g * GradGraph::SELU_LAMBDA } else { g * GradGraph::SELU_LAMBDA * GradGraph::SELU_ALPHA * x.exp() }
                                }).collect(),
                                grad.shape(),
                            )
                        }
                        crate::pinn::Activation::SinAct => {
                            let z_data = z.to_vec();
                            let grad_data = grad.to_vec();
                            Tensor::from_vec_unchecked(
                                z_data.iter().zip(grad_data.iter()).map(|(&x, &g)| g * x.cos()).collect(),
                                grad.shape(),
                            )
                        }
                    };

                    accumulate_grad(&mut grads, *input, &dz.matmul_unchecked(weight_t));
                    accumulate_grad(&mut grads, *weight, &dz.transpose().matmul_unchecked(input_t));
                    let dz_data = dz.to_vec();
                    let dz_shape = dz.shape();
                    if dz_shape.len() == 2 {
                        let (rows, cols) = (dz_shape[0], dz_shape[1]);
                        let mut bias_grad = vec![0.0_f64; cols];
                        for r in 0..rows { for c in 0..cols { bias_grad[c] += dz_data[r * cols + c]; } }
                        accumulate_grad(&mut grads, *bias, &Tensor::from_vec_unchecked(bias_grad, &[cols]));
                    } else {
                        accumulate_grad(&mut grads, *bias, &dz);
                    }
                }
                GradOp::BroadcastScalar { input, .. } => {
                    // Forward: result[i] = scalar.value() for all i.
                    // Backward: dF/dscalar = sum of upstream over all output elements.
                    use cjc_repro::KahanAccumulatorF64;
                    let upstream_data = grad.to_vec();
                    let mut acc = KahanAccumulatorF64::new();
                    for v in upstream_data.iter() { acc.add(*v); }
                    let scalar_grad =
                        Tensor::from_vec_unchecked(vec![acc.finalize()], &[1]);
                    accumulate_grad(&mut grads, *input, &scalar_grad);
                }
            }
        }
    }
}

fn accumulate_grad(grads: &mut [Option<Tensor>], idx: usize, grad: &Tensor) {
    match &mut grads[idx] {
        Some(existing) => {
            existing.add_assign_unchecked(grad);
        }
        slot @ None => {
            *slot = Some(grad.clone());
        }
    }
}

// Close the impl block — backward_with_seed, jacobian, hessian_diag are all inside impl GradGraph

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

    // ── Phase 8: Extended AD Tests ──

    #[test]
    fn test_reverse_abs_positive() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![3.0], &[1]));
        let b = g.abs(a);
        g.backward(b);
        let ga = g.grad(a).unwrap();
        // abs'(3) = sign(3) = 1.0
        assert!((ga.to_vec()[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reverse_abs_negative() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![-2.5], &[1]));
        let b = g.abs(a);
        g.backward(b);
        let ga = g.grad(a).unwrap();
        // abs'(-2.5) = sign(-2.5) = -1.0
        assert!((ga.to_vec()[0] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_reverse_abs_zero() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![0.0], &[1]));
        let b = g.abs(a);
        g.backward(b);
        let ga = g.grad(a).unwrap();
        // abs'(0) = 0.0 (subgradient convention)
        assert!(ga.to_vec()[0].abs() < 1e-10);
    }

    #[test]
    fn test_reverse_abs_vector() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![-1.0, 2.0, 0.0, -3.0], &[4]));
        let b = g.abs(a);
        let loss = g.sum(b);
        g.backward(loss);
        let ga = g.grad(a).unwrap();
        let expected = vec![-1.0, 1.0, 0.0, -1.0];
        for (i, (&got, &exp)) in ga.to_vec().iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-10, "abs grad[{i}]: got {got}, expected {exp}");
        }
    }

    #[test]
    fn test_reverse_log2() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![4.0], &[1]));
        let b = g.log2(a);
        g.backward(b);
        let ga = g.grad(a).unwrap();
        // d/dx log2(x) at x=4 = 1/(4 * ln(2))
        let expected = 1.0 / (4.0 * std::f64::consts::LN_2);
        assert!((ga.to_vec()[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_reverse_log2_vector() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![1.0, 2.0, 8.0], &[3]));
        let b = g.log2(a);
        let loss = g.sum(b);
        g.backward(loss);
        let ga = g.grad(a).unwrap();
        let ln2 = std::f64::consts::LN_2;
        let expected = vec![1.0 / (1.0 * ln2), 1.0 / (2.0 * ln2), 1.0 / (8.0 * ln2)];
        for (i, (&got, &exp)) in ga.to_vec().iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-10, "log2 grad[{i}]: got {got}, expected {exp}");
        }
    }

    #[test]
    fn test_softmax_forward() {
        let mut g = GradGraph::new();
        let a = g.input(Tensor::from_vec_unchecked(vec![1.0, 2.0, 3.0], &[3]));
        let b = g.softmax(a);
        let sm = g.tensor(b);
        let sm_data = sm.to_vec();
        // softmax values should sum to 1
        let sum: f64 = sm_data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        // Verify ordering: softmax(3) > softmax(2) > softmax(1)
        assert!(sm_data[2] > sm_data[1]);
        assert!(sm_data[1] > sm_data[0]);
    }

    #[test]
    fn test_reverse_softmax() {
        // Finite difference check for softmax gradient
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![1.0, 2.0, 3.0], &[3]));
        let b = g.softmax(a);
        let loss = g.sum(b);
        g.backward(loss);
        let ga = g.grad(a).unwrap();
        // sum(softmax(x)) = 1 always, so d/dx sum(softmax(x)) = 0
        for &v in &ga.to_vec() {
            assert!(v.abs() < 1e-10, "softmax sum grad should be 0, got {v}");
        }
    }

    #[test]
    fn test_reverse_softmax_single_element() {
        // With a single element, softmax gradient through sum should be 0
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![2.0, 1.0], &[2]));
        let b = g.softmax(a);
        // Take only first element via scalar_mul trick: multiply by [1, 0]
        // Instead, use a direct sum which should give zero grad
        let loss = g.sum(b);
        g.backward(loss);
        let ga = g.grad(a).unwrap();
        for &v in &ga.to_vec() {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn test_cross_entropy_forward() {
        let mut g = GradGraph::new();
        let logits = g.input(Tensor::from_vec_unchecked(vec![2.0, 1.0, 0.1], &[3]));
        let targets = g.input(Tensor::from_vec_unchecked(vec![1.0, 0.0, 0.0], &[3])); // one-hot
        let ce = g.cross_entropy(logits, targets);
        let loss_val = g.value(ce);
        assert!(loss_val > 0.0, "CE loss should be positive");
        assert!(loss_val.is_finite(), "CE loss should be finite");
    }

    #[test]
    fn test_reverse_cross_entropy() {
        let mut g = GradGraph::new();
        let logits = g.parameter(Tensor::from_vec_unchecked(vec![2.0, 1.0, 0.1], &[3]));
        let targets = g.input(Tensor::from_vec_unchecked(vec![1.0, 0.0, 0.0], &[3]));
        let ce = g.cross_entropy(logits, targets);
        g.backward(ce);
        let ga = g.grad(logits).unwrap();
        let ga_data = ga.to_vec();
        // grad = softmax(logits) - targets
        // softmax should give something like [0.659, 0.242, 0.099]
        // grad[0] should be negative (softmax < 1.0 for correct class)
        assert!(ga_data[0] < 0.0, "CE grad for correct class should be negative");
        assert!(ga_data[1] > 0.0, "CE grad for incorrect class should be positive");
        assert!(ga_data[2] > 0.0, "CE grad for incorrect class should be positive");
        // Sum of gradients should be ~0 (softmax sums to 1, targets sum to 1)
        let sum: f64 = ga_data.iter().sum();
        assert!(sum.abs() < 1e-10, "CE grad should sum to 0, got {sum}");
    }

    #[test]
    fn test_layer_norm_forward() {
        let mut g = GradGraph::new();
        let a = g.input(Tensor::from_vec_unchecked(vec![1.0, 2.0, 3.0, 4.0], &[4]));
        let b = g.layer_norm(a);
        let normed = g.tensor(b).to_vec();
        // After layer norm, mean should be ~0 and std ~1
        let mean: f64 = normed.iter().sum::<f64>() / normed.len() as f64;
        assert!(mean.abs() < 1e-5, "LayerNorm mean should be ~0, got {mean}");
        let var: f64 = normed.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / normed.len() as f64;
        assert!((var - 1.0).abs() < 0.01, "LayerNorm variance should be ~1, got {var}");
    }

    #[test]
    fn test_reverse_layer_norm() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![1.0, 2.0, 3.0, 4.0], &[4]));
        let b = g.layer_norm(a);
        let loss = g.sum(b);
        g.backward(loss);
        let ga = g.grad(a).unwrap();
        // Gradient should be finite and non-zero
        for &v in &ga.to_vec() {
            assert!(v.is_finite(), "LayerNorm grad should be finite");
        }
        // Finite difference check
        let eps = 1e-5;
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        for i in 0..4 {
            let mut plus = input_data.clone();
            plus[i] += eps;
            let mut g_plus = GradGraph::new();
            let a_plus = g_plus.input(Tensor::from_vec_unchecked(plus, &[4]));
            let b_plus = g_plus.layer_norm(a_plus);
            let loss_plus = g_plus.sum(b_plus);
            let val_plus = g_plus.value(loss_plus);

            let mut minus = input_data.clone();
            minus[i] -= eps;
            let mut g_minus = GradGraph::new();
            let a_minus = g_minus.input(Tensor::from_vec_unchecked(minus, &[4]));
            let b_minus = g_minus.layer_norm(a_minus);
            let loss_minus = g_minus.sum(b_minus);
            let val_minus = g_minus.value(loss_minus);

            let fd_grad = (val_plus - val_minus) / (2.0 * eps);
            let ad_grad = ga.to_vec()[i];
            assert!(
                (fd_grad - ad_grad).abs() < 1e-4,
                "LayerNorm FD check failed at [{i}]: fd={fd_grad}, ad={ad_grad}"
            );
        }
    }

    #[test]
    fn test_batch_norm_forward() {
        let mut g = GradGraph::new();
        let a = g.input(Tensor::from_vec_unchecked(vec![2.0, 4.0, 6.0, 8.0], &[4]));
        let b = g.batch_norm(a);
        let normed = g.tensor(b).to_vec();
        let mean: f64 = normed.iter().sum::<f64>() / normed.len() as f64;
        assert!(mean.abs() < 1e-5, "BatchNorm mean should be ~0, got {mean}");
    }

    #[test]
    fn test_reverse_batch_norm() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![2.0, 4.0, 6.0, 8.0], &[4]));
        let b = g.batch_norm(a);
        let loss = g.sum(b);
        g.backward(loss);
        let ga = g.grad(a).unwrap();
        for &v in &ga.to_vec() {
            assert!(v.is_finite(), "BatchNorm grad should be finite");
        }
    }

    #[test]
    fn test_reverse_clamp_in_range() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![1.5], &[1]));
        let b = g.clamp(a, 0.0, 3.0);
        g.backward(b);
        let ga = g.grad(a).unwrap();
        // 1.5 is in [0, 3], so grad passes through: 1.0
        assert!((ga.to_vec()[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reverse_clamp_out_of_range() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![5.0], &[1]));
        let b = g.clamp(a, 0.0, 3.0);
        g.backward(b);
        let ga = g.grad(a).unwrap();
        // 5.0 is outside [0, 3], so grad is 0
        assert!(ga.to_vec()[0].abs() < 1e-10);
    }

    #[test]
    fn test_reverse_clamp_vector() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![-1.0, 0.5, 2.0, 4.0], &[4]));
        let b = g.clamp(a, 0.0, 3.0);
        let loss = g.sum(b);
        g.backward(loss);
        let ga = g.grad(a).unwrap();
        let expected = vec![0.0, 1.0, 1.0, 0.0]; // -1 out, 0.5 in, 2 in, 4 out
        for (i, (&got, &exp)) in ga.to_vec().iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-10, "clamp grad[{i}]: got {got}, expected {exp}");
        }
    }

    #[test]
    fn test_reverse_where_cond() {
        let mut g = GradGraph::new();
        let cond = g.input(Tensor::from_vec_unchecked(vec![1.0, 0.0, 1.0], &[3]));
        let a = g.parameter(Tensor::from_vec_unchecked(vec![10.0, 20.0, 30.0], &[3]));
        let b = g.parameter(Tensor::from_vec_unchecked(vec![100.0, 200.0, 300.0], &[3]));
        let w = g.where_cond(cond, a, b);
        // Forward: should select [10, 200, 30]
        let result = g.tensor(w).to_vec();
        assert!((result[0] - 10.0).abs() < 1e-10);
        assert!((result[1] - 200.0).abs() < 1e-10);
        assert!((result[2] - 30.0).abs() < 1e-10);
        let loss = g.sum(w);
        g.backward(loss);
        let ga = g.grad(a).unwrap().to_vec();
        let gb = g.grad(b).unwrap().to_vec();
        // grad flows to a where cond=1, to b where cond=0
        assert!((ga[0] - 1.0).abs() < 1e-10);
        assert!(ga[1].abs() < 1e-10);
        assert!((ga[2] - 1.0).abs() < 1e-10);
        assert!(gb[0].abs() < 1e-10);
        assert!((gb[1] - 1.0).abs() < 1e-10);
        assert!(gb[2].abs() < 1e-10);
    }

    #[test]
    fn test_reverse_reshape() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let b = g.reshape(a, &[3, 2]);
        let loss = g.sum(b);
        g.backward(loss);
        let ga = g.grad(a).unwrap();
        // Reshape backward: grad should have original shape [2, 3], all ones
        assert_eq!(ga.shape(), &[2, 3]);
        for &v in &ga.to_vec() {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_reverse_transpose_op() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let b = g.transpose_op(a);
        // Transposed shape should be [3, 2]
        assert_eq!(g.tensor(b).shape(), &[3, 2]);
        let loss = g.sum(b);
        g.backward(loss);
        let ga = g.grad(a).unwrap();
        // Transpose backward: grad should have original shape [2, 3], all ones
        assert_eq!(ga.shape(), &[2, 3]);
        for &v in &ga.to_vec() {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_reverse_cat_1d() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![1.0, 2.0], &[2]));
        let b = g.parameter(Tensor::from_vec_unchecked(vec![3.0, 4.0, 5.0], &[3]));
        let c = g.cat(&[a, b], 0);
        // Forward: [1, 2, 3, 4, 5]
        let result = g.tensor(c).to_vec();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let loss = g.sum(c);
        g.backward(loss);
        let ga = g.grad(a).unwrap().to_vec();
        let gb = g.grad(b).unwrap().to_vec();
        // All gradients should be 1 (from sum)
        assert_eq!(ga, vec![1.0, 1.0]);
        assert_eq!(gb, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_reverse_cat_2d_axis0() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![1.0, 2.0], &[1, 2]));
        let b = g.parameter(Tensor::from_vec_unchecked(vec![3.0, 4.0, 5.0, 6.0], &[2, 2]));
        let c = g.cat(&[a, b], 0);
        assert_eq!(g.tensor(c).shape(), &[3, 2]);
        let loss = g.sum(c);
        g.backward(loss);
        let ga = g.grad(a).unwrap();
        let gb = g.grad(b).unwrap();
        assert_eq!(ga.shape(), &[1, 2]);
        assert_eq!(gb.shape(), &[2, 2]);
        for &v in &ga.to_vec() {
            assert!((v - 1.0).abs() < 1e-10);
        }
        for &v in &gb.to_vec() {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_reverse_gather_1d() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![10.0, 20.0, 30.0, 40.0], &[4]));
        let b = g.gather(a, &[1, 3], 0);
        // Forward: [20, 40]
        let result = g.tensor(b).to_vec();
        assert!((result[0] - 20.0).abs() < 1e-10);
        assert!((result[1] - 40.0).abs() < 1e-10);
        let loss = g.sum(b);
        g.backward(loss);
        let ga = g.grad(a).unwrap().to_vec();
        // Scatter-add: indices [1, 3] get grad 1.0 each, others get 0
        assert!((ga[0]).abs() < 1e-10);
        assert!((ga[1] - 1.0).abs() < 1e-10);
        assert!((ga[2]).abs() < 1e-10);
        assert!((ga[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reverse_gather_duplicate_indices() {
        let mut g = GradGraph::new();
        let a = g.parameter(Tensor::from_vec_unchecked(vec![10.0, 20.0, 30.0], &[3]));
        let b = g.gather(a, &[1, 1, 2], 0);
        let loss = g.sum(b);
        g.backward(loss);
        let ga = g.grad(a).unwrap().to_vec();
        // Index 1 appears twice, so its grad should be 2.0
        assert!((ga[0]).abs() < 1e-10);
        assert!((ga[1] - 2.0).abs() < 1e-10);
        assert!((ga[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_phase8_determinism() {
        // Run twice and verify bit-identical gradients
        for _ in 0..2 {
            let run = || {
                let mut g = GradGraph::new();
                let a = g.parameter(Tensor::from_vec_unchecked(vec![1.0, -2.0, 3.0, -0.5], &[4]));
                let b = g.abs(a);
                let c = g.clamp(b, 0.0, 2.5);
                let d = g.layer_norm(c);
                let loss = g.sum(d);
                g.backward(loss);
                g.grad(a).unwrap().to_vec()
            };
            let r1 = run();
            let r2 = run();
            for (i, (v1, v2)) in r1.iter().zip(r2.iter()).enumerate() {
                assert_eq!(v1.to_bits(), v2.to_bits(), "Determinism failed at [{i}]");
            }
        }
    }

    #[test]
    fn test_phase8_softmax_cross_entropy_chain() {
        // Verify that softmax + CE combined gradient works end-to-end
        let mut g = GradGraph::new();
        let logits = g.parameter(Tensor::from_vec_unchecked(vec![1.0, 2.0, 3.0], &[3]));
        let targets = g.input(Tensor::from_vec_unchecked(vec![0.0, 0.0, 1.0], &[3]));
        let ce = g.cross_entropy(logits, targets);
        g.backward(ce);
        let ga = g.grad(logits).unwrap().to_vec();
        // Verify all gradients are finite
        for &v in &ga {
            assert!(v.is_finite());
        }
        // CE grad for correct class (index 2) should be negative (softmax - 1)
        assert!(ga[2] < 0.0, "CE grad for correct class should be negative");
    }

    // ── Sprint 1 SciML Hardening: hessian, double_backward, vmap_forward ──

    #[test]
    fn test_double_backward_cubic() {
        // f(x) = x^3, f'(x) = 3x^2, f''(x) = 6x
        let mut g = GradGraph::new();
        let x = g.parameter(Tensor::from_vec_unchecked(vec![2.0], &[1]));
        let x2 = g.mul(x, x);
        let x3 = g.mul(x2, x);
        let loss = g.sum(x3);
        let hess = g.double_backward(loss, x);
        // f''(2) = 6*2 = 12
        assert!((hess.to_vec()[0] - 12.0).abs() < 1e-4);
    }

    #[test]
    fn test_full_hessian_quadratic() {
        // f(x, y) = x^2 + y^2 using p = [x, y]
        // H = [[2, 0], [0, 2]]
        let mut g = GradGraph::new();
        let p = g.parameter(Tensor::from_vec_unchecked(vec![1.0, 1.0], &[2]));
        let p2 = g.mul(p, p); // [x^2, y^2]
        let s = g.sum(p2); // x^2 + y^2
        let hess = g.hessian(s, p);
        let h = hess.to_vec();
        assert!((h[0] - 2.0).abs() < 1e-3); // d2f/dx2 = 2
        assert!((h[1] - 0.0).abs() < 1e-3); // d2f/dxdy = 0
        assert!((h[2] - 0.0).abs() < 1e-3); // d2f/dydx = 0
        assert!((h[3] - 2.0).abs() < 1e-3); // d2f/dy2 = 2
    }

    #[test]
    fn test_vmap_forward() {
        let mut g = GradGraph::new();
        let x = g.parameter(Tensor::from_vec_unchecked(vec![1.0], &[1]));
        let x2 = g.mul(x, x);
        let loss = g.sum(x2);

        // vmap over batch [1.0, 2.0, 3.0]
        let batch = vec![
            Tensor::from_vec_unchecked(vec![1.0], &[1]),
            Tensor::from_vec_unchecked(vec![2.0], &[1]),
            Tensor::from_vec_unchecked(vec![3.0], &[1]),
        ];
        let results = g.vmap_forward(x, &batch);
        // Results should be [1.0, 4.0, 9.0]
        assert!((g.value(results[0]) - 1.0).abs() < 1e-10);
        assert!((g.value(results[1]) - 4.0).abs() < 1e-10);
        assert!((g.value(results[2]) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_hessian_determinism() {
        let mut g = GradGraph::new();
        let p = g.parameter(Tensor::from_vec_unchecked(vec![3.0, 4.0], &[2]));
        let p2 = g.mul(p, p);
        let s = g.sum(p2);
        let h1 = g.hessian(s, p);
        // Reset and redo
        g.set_tensor(p, Tensor::from_vec_unchecked(vec![3.0, 4.0], &[2]));
        let h2 = g.hessian(s, p);
        assert_eq!(h1.to_vec(), h2.to_vec(), "Hessian must be deterministic");
    }
}
