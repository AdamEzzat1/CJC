//! Physics-Informed Neural Networks (PINN) and Physics-Informed ML (PIML)
//!
//! Provides deterministic, zero-dependency infrastructure for:
//! - MLP construction and forward passes over a GradGraph
//! - Physics residual computation (PDE residuals via autodiff)
//! - Composite loss functions (data + physics + boundary)
//! - Training loops with Adam / L-BFGS
//! - Deterministic sampling (uniform grid, LHS, Sobol)
//!
//! All floating-point reductions use Kahan summation.
//! All randomness is seeded via SplitMix64 (cjc_repro::Rng).

use cjc_runtime::tensor::Tensor;
use cjc_repro::KahanAccumulatorF64;
use cjc_runtime::ml::{AdamState, adam_step};

// ---------------------------------------------------------------------------
// MLP Architecture (built on GradGraph indices)
// ---------------------------------------------------------------------------

/// Describes one dense (fully-connected) layer.
#[derive(Debug, Clone)]
pub struct DenseLayer {
    /// GradGraph index for weight matrix [out_features, in_features].
    pub weight_idx: usize,
    /// GradGraph index for bias vector [out_features].
    pub bias_idx: usize,
    /// Activation to apply after affine transform.
    pub activation: Activation,
    /// Number of input features (columns of weight matrix).
    pub in_features: usize,
    /// Number of output features (rows of weight matrix).
    pub out_features: usize,
}

/// Supported activation functions (all differentiable in GradGraph).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Hyperbolic tangent activation, maps inputs to (-1, 1).
    Tanh,
    /// Sigmoid (logistic) activation, maps inputs to (0, 1).
    Sigmoid,
    /// Rectified linear unit, returns max(0, x).
    Relu,
    /// Identity (no activation), passes the value through unchanged.
    None,
    /// Gaussian Error Linear Unit: x * Φ(x), default in transformers.
    Gelu,
    /// Sigmoid Linear Unit (Swish): x * σ(x), preferred in modern CNNs.
    Silu,
    /// Exponential Linear Unit: x if x>0 else α*(exp(x)-1), α=1.0.
    Elu,
    /// Scaled Exponential Linear Unit: λ * (x if x>0 else α*(exp(x)-1)).
    /// Self-normalizing with λ≈1.0507, α≈1.6733.
    Selu,
    /// Periodic sine activation: sin(x). Key for PINNs with oscillatory solutions.
    SinAct,
}

/// Multi-layer perceptron specification.
///
/// Stores an ordered sequence of [`DenseLayer`]s that together define the
/// network architecture. Layer weights and biases live on an external
/// [`GradGraph`](crate::GradGraph) and are referenced by index.
#[derive(Debug, Clone)]
pub struct Mlp {
    /// Ordered dense layers from input to output.
    pub layers: Vec<DenseLayer>,
}

/// Initialize an MLP on a [`GradGraph`](crate::GradGraph) with Xavier-uniform weights.
///
/// Register weight and bias parameter nodes on the graph for every layer and
/// return the resulting [`Mlp`] specification together with the flat list of
/// parameter node indices (alternating weight, bias for each layer).
///
/// # Arguments
///
/// * `graph` - Mutable reference to the computation graph where parameters are registered.
/// * `layer_sizes` - Sequence of layer widths, e.g. `[1, 32, 32, 1]` for
///   input dim 1, two hidden layers of 32, and output dim 1.
/// * `hidden_activation` - Activation applied after every hidden layer.
/// * `output_activation` - Activation applied after the final layer
///   (typically [`Activation::None`] for regression tasks).
/// * `seed` - RNG seed for deterministic Xavier-uniform initialization.
///
/// # Returns
///
/// A tuple `(Mlp, Vec<usize>)` where the second element contains the
/// GradGraph parameter indices (weights and biases interleaved per layer).
///
/// # Panics
///
/// Panics if `layer_sizes` contains fewer than two entries.
pub fn mlp_init(
    graph: &mut crate::GradGraph,
    layer_sizes: &[usize],
    hidden_activation: Activation,
    output_activation: Activation,
    seed: u64,
) -> (Mlp, Vec<usize>) {
    assert!(layer_sizes.len() >= 2, "Need at least input + output sizes");
    let mut rng = cjc_repro::Rng::seeded(seed);
    let mut layers = Vec::new();
    let mut param_indices = Vec::new();

    for i in 0..layer_sizes.len() - 1 {
        let in_f = layer_sizes[i];
        let out_f = layer_sizes[i + 1];
        let is_last = i == layer_sizes.len() - 2;

        // Xavier uniform initialization: scale = sqrt(6 / (fan_in + fan_out))
        let scale = (6.0 / (in_f + out_f) as f64).sqrt();
        let n_weights = out_f * in_f;
        let mut w_data = Vec::with_capacity(n_weights);
        for _ in 0..n_weights {
            // Map uniform [0,1) -> [-scale, scale)
            let u = rng.next_f64();
            w_data.push(u * 2.0 * scale - scale);
        }
        let w_tensor = Tensor::from_vec_unchecked(w_data, &[out_f, in_f]);
        let w_idx = graph.parameter(w_tensor);
        param_indices.push(w_idx);

        let b_tensor = Tensor::zeros(&[out_f]);
        let b_idx = graph.parameter(b_tensor);
        param_indices.push(b_idx);

        let activation = if is_last { output_activation } else { hidden_activation };

        layers.push(DenseLayer {
            weight_idx: w_idx,
            bias_idx: b_idx,
            activation,
            in_features: in_f,
            out_features: out_f,
        });
    }

    (Mlp { layers }, param_indices)
}

/// Execute a forward pass of an MLP through the GradGraph.
///
/// For each layer the function computes `activation(x @ W^T + b)` and chains
/// the result into the next layer.
///
/// # Arguments
///
/// * `graph` - Mutable reference to the computation graph.
/// * `mlp` - MLP specification whose layer indices must already exist on `graph`.
/// * `input_idx` - GradGraph node index for the input tensor
///   (shape `[batch, in_features]` or `[in_features]`).
///
/// # Returns
///
/// The GradGraph node index of the final output tensor.
pub fn mlp_forward(
    graph: &mut crate::GradGraph,
    mlp: &Mlp,
    input_idx: usize,
) -> usize {
    let mut x = input_idx;

    for layer in &mlp.layers {
        // Fused dense layer: activation(input @ weight^T + bias)
        // Collapses transpose + matmul + add + activation into one graph node.
        x = graph.mlp_layer(x, layer.weight_idx, layer.bias_idx, layer.activation);
    }

    x
}

// ---------------------------------------------------------------------------
// Physics Loss Components
// ---------------------------------------------------------------------------

/// Compute the mean-squared-error data loss between predicted and target values.
///
/// Builds `mean((pred - target)^2)` on the graph and returns the resulting
/// scalar loss node.
///
/// # Arguments
///
/// * `graph` - Mutable reference to the computation graph.
/// * `pred_idx` - GradGraph node index for the predicted values.
/// * `target_idx` - GradGraph node index for the target (ground-truth) values.
///
/// # Returns
///
/// GradGraph node index of the scalar MSE loss.
pub fn data_loss_mse(
    graph: &mut crate::GradGraph,
    pred_idx: usize,
    target_idx: usize,
) -> usize {
    let diff = graph.sub(pred_idx, target_idx);
    let sq = graph.mul(diff, diff);
    graph.mean(sq)
}

/// Compute the physics residual for the simple harmonic ODE `u_xx + u = 0`.
///
/// Build the mean-squared residual `mean(r^2)` on the graph, where
/// `r = u_xx + u`. In the current implementation this is a placeholder that
/// squares and averages `u` directly; the full finite-difference workflow is
/// implemented inside [`pinn_harmonic_train`].
///
/// # Arguments
///
/// * `graph` - Mutable reference to the computation graph.
/// * `u_idx` - GradGraph node index for the network output `u(x)`.
/// * `x_idx` - GradGraph node index for the input `x` coordinates (currently unused).
/// * `n_points` - Number of collocation points (currently unused).
///
/// # Returns
///
/// GradGraph node index of the scalar mean-squared residual.
pub fn physics_residual_harmonic(
    graph: &mut crate::GradGraph,
    u_idx: usize,
    x_idx: usize,
    n_points: usize,
) -> usize {
    // For PINNs, we need du/dx and d²u/dx².
    // We compute this using finite differences on the graph's forward pass
    // because CJC's GradGraph supports Jacobian computation.
    //
    // However, for the graph-based approach, we compute the residual directly:
    //   r = u_xx + u  (should be zero for u_xx + u = 0)
    //
    // We approximate u_xx via the Jacobian/Hessian infrastructure or
    // via finite differences on the network itself.
    //
    // The simplest approach for a PINN: evaluate the network at x-eps, x, x+eps
    // and use central differences for u_xx. This is done outside the graph
    // and then the residual is added to the graph as a known tensor.
    //
    // For a fully graph-based approach, we'd need per-element Jacobian which
    // is expensive. The practical PINN approach is finite-difference u_xx.
    //
    // This function assumes u_xx has already been computed and placed in the graph.
    // See `compute_pinn_loss_harmonic` for the full workflow.

    // residual = u + u_xx (for u_xx + u = 0)
    // This is a placeholder — the real implementation uses the full workflow below.
    let _ = (x_idx, n_points);
    let residual_sq = graph.mul(u_idx, u_idx);
    graph.mean(residual_sq)
}

// ---------------------------------------------------------------------------
// Complete PINN Training Infrastructure
// ---------------------------------------------------------------------------

/// Configuration for a PINN training run.
///
/// Collect all hyper-parameters (network shape, learning rate, loss weights,
/// sampling counts, and finite-difference step size) into a single struct so
/// that training functions accept one argument instead of many.
#[derive(Debug, Clone)]
pub struct PinnConfig {
    /// MLP layer sizes, e.g. [1, 32, 32, 1].
    pub layer_sizes: Vec<usize>,
    /// Number of training epochs.
    pub epochs: usize,
    /// Learning rate for Adam optimizer.
    pub lr: f64,
    /// Weight for physics loss relative to data loss.
    pub physics_weight: f64,
    /// Weight for boundary condition loss.
    pub boundary_weight: f64,
    /// Random seed for initialization and sampling.
    pub seed: u64,
    /// Number of interior collocation points.
    pub n_collocation: usize,
    /// Number of training data points.
    pub n_data: usize,
    /// Finite-difference epsilon for computing second derivatives.
    pub fd_eps: f64,
}

impl Default for PinnConfig {
    fn default() -> Self {
        Self {
            layer_sizes: vec![1, 32, 32, 1],
            epochs: 2000,
            lr: 1e-3,
            physics_weight: 1.0,
            boundary_weight: 50.0,
            seed: 42,
            n_collocation: 50,
            n_data: 20,
            fd_eps: 1e-3,
        }
    }
}

/// Single training log entry recorded at the end of each epoch.
#[derive(Debug, Clone)]
pub struct TrainLog {
    /// Zero-based epoch index.
    pub epoch: usize,
    /// Weighted sum of data, physics, and boundary losses.
    pub total_loss: f64,
    /// Mean-squared-error between network predictions and observed data.
    pub data_loss: f64,
    /// Mean-squared PDE/ODE residual at collocation points.
    pub physics_loss: f64,
    /// Squared violation of boundary conditions.
    pub boundary_loss: f64,
    /// L2 norm of the total gradient vector.
    pub grad_norm: f64,
}

/// Aggregate result of a PINN or PIML training run.
///
/// Contains final parameter values, the complete loss history, and optional
/// accuracy metrics computed against an analytical solution.
#[derive(Debug, Clone)]
pub struct PinnResult {
    /// Final parameter values (flattened).
    pub final_params: Vec<f64>,
    /// Loss history per epoch.
    pub history: Vec<TrainLog>,
    /// L2 error against analytical solution (if provided).
    pub l2_error: Option<f64>,
    /// Max absolute error against analytical solution.
    pub max_error: Option<f64>,
    /// Mean physics residual at final params.
    pub mean_residual: f64,
}

// ---------------------------------------------------------------------------
// Problem A: PIML — 1D Steady-State Heat Equation
// ---------------------------------------------------------------------------
// Problem: u_xx = f(x) on [0, 1], u(0) = 0, u(1) = 0
// Source:  f(x) = -π² sin(πx)
// Exact:   u(x) = sin(πx)
//
// PIML approach: polynomial regression u_approx(x) = Σ aᵢ xⁱ
// with physics penalty: Σ (u_xx_approx - f(x))²

/// Generate synthetic data for the 1D heat equation.
///
/// Returns (x_data, u_data) with noise added to u.
pub fn heat_1d_generate_data(n: usize, noise_std: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = cjc_repro::Rng::seeded(seed);
    let mut x_data = Vec::with_capacity(n);
    let mut u_data = Vec::with_capacity(n);

    for i in 0..n {
        let x = (i as f64 + 0.5) / n as f64; // uniform grid on (0, 1)
        let u_exact = (std::f64::consts::PI * x).sin();
        // Box-Muller for Gaussian noise
        let u1 = rng.next_f64().max(1e-300);
        let u2 = rng.next_f64();
        let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        x_data.push(x);
        u_data.push(u_exact + noise_std * noise);
    }

    (x_data, u_data)
}

/// Evaluate a polynomial u(x) = a0 + a1*x + a2*x² + ... using Horner's method.
fn poly_eval(coeffs: &[f64], x: f64) -> f64 {
    let mut result = 0.0;
    for i in (0..coeffs.len()).rev() {
        result = result * x + coeffs[i];
    }
    result
}

/// Second derivative of a polynomial: u''(x).
fn poly_eval_dd(coeffs: &[f64], x: f64) -> f64 {
    // d²/dx² of Σ aᵢ xⁱ = Σ_{i≥2} i*(i-1)*aᵢ x^{i-2}
    if coeffs.len() < 3 {
        return 0.0;
    }
    let mut result = 0.0;
    for i in (2..coeffs.len()).rev() {
        result = result * x + coeffs[i] * (i * (i - 1)) as f64;
    }
    result
}

/// Source term f(x) = -π² sin(πx) for the heat equation.
fn heat_source(x: f64) -> f64 {
    -std::f64::consts::PI.powi(2) * (std::f64::consts::PI * x).sin()
}

/// Train a PIML polynomial model for the 1D heat equation.
///
/// Returns (coefficients, loss_history, l2_error, max_error, mean_residual).
pub fn piml_heat_1d_train(
    degree: usize,
    n_data: usize,
    n_colloc: usize,
    noise_std: f64,
    epochs: usize,
    lr: f64,
    physics_weight: f64,
    boundary_weight: f64,
    seed: u64,
) -> PinnResult {
    let (x_data, u_data) = heat_1d_generate_data(n_data, noise_std, seed);

    // Collocation points — uniform grid on [0, 1]
    let x_colloc: Vec<f64> = (0..n_colloc).map(|i| (i as f64 + 0.5) / n_colloc as f64).collect();

    // Initialize polynomial coefficients (degree+1 params)
    let n_params = degree + 1;
    let mut coeffs = vec![0.0f64; n_params];
    // Small random init
    let mut rng = cjc_repro::Rng::seeded(seed.wrapping_add(1));
    for c in coeffs.iter_mut() {
        *c = (rng.next_f64() - 0.5) * 0.1;
    }

    let mut adam = AdamState::new(n_params, lr);
    let mut history = Vec::with_capacity(epochs);

    for epoch in 0..epochs {
        // --- Data loss: MSE(u_approx(x_data), u_data) ---
        let mut data_acc = KahanAccumulatorF64::new();
        let mut data_grads = vec![0.0f64; n_params];
        for (j, &x) in x_data.iter().enumerate() {
            let u_pred = poly_eval(&coeffs, x);
            let err = u_pred - u_data[j];
            data_acc.add(err * err);
            // d(err²)/d(aᵢ) = 2*err * x^i
            let mut xi = 1.0;
            for g in data_grads.iter_mut() {
                *g += 2.0 * err * xi / n_data as f64;
                xi *= x;
            }
        }
        let data_loss = data_acc.finalize() / n_data as f64;

        // --- Physics loss: MSE(u_xx(x_colloc) - f(x_colloc)) ---
        let mut phys_acc = KahanAccumulatorF64::new();
        let mut phys_grads = vec![0.0f64; n_params];
        for &x in &x_colloc {
            let u_xx = poly_eval_dd(&coeffs, x);
            let f_x = heat_source(x);
            let residual = u_xx - f_x;
            phys_acc.add(residual * residual);
            // d(residual²)/d(aᵢ) = 2*residual * d(u_xx)/d(aᵢ)
            // d(u_xx)/d(aᵢ) = i*(i-1) * x^{i-2} for i >= 2, else 0
            for i in 2..n_params {
                let du_xx_dai = (i * (i - 1)) as f64 * x.powi(i as i32 - 2);
                phys_grads[i] += 2.0 * residual * du_xx_dai / n_colloc as f64;
            }
        }
        let physics_loss = phys_acc.finalize() / n_colloc as f64;

        // --- Boundary loss: u(0)² + u(1)² ---
        let u0 = poly_eval(&coeffs, 0.0);
        let u1 = poly_eval(&coeffs, 1.0);
        let boundary_loss = u0 * u0 + u1 * u1;
        let mut bnd_grads = vec![0.0f64; n_params];
        // d(u(0)²)/d(a0) = 2*u(0), d(u(0)²)/d(aᵢ) = 0 for i>0 (since 0^i=0)
        bnd_grads[0] += 2.0 * u0;
        // d(u(1)²)/d(aᵢ) = 2*u(1) * 1^i = 2*u(1)
        for g in bnd_grads.iter_mut() {
            *g += 2.0 * u1;
        }

        // --- Total gradient ---
        let mut total_grads = vec![0.0f64; n_params];
        let mut grad_norm_acc = KahanAccumulatorF64::new();
        for i in 0..n_params {
            total_grads[i] = data_grads[i]
                + physics_weight * phys_grads[i]
                + boundary_weight * bnd_grads[i];
            grad_norm_acc.add(total_grads[i] * total_grads[i]);
        }
        let grad_norm = grad_norm_acc.finalize().sqrt();

        let total_loss = data_loss + physics_weight * physics_loss + boundary_weight * boundary_loss;

        history.push(TrainLog {
            epoch,
            total_loss,
            data_loss,
            physics_loss,
            boundary_loss,
            grad_norm,
        });

        // Adam update
        adam_step(&mut coeffs, &total_grads, &mut adam);
    }

    // --- Evaluate final accuracy ---
    let n_eval = 100;
    let mut l2_acc = KahanAccumulatorF64::new();
    let mut max_err = 0.0f64;
    let mut res_acc = KahanAccumulatorF64::new();

    for i in 0..n_eval {
        let x = (i as f64 + 0.5) / n_eval as f64;
        let u_exact = (std::f64::consts::PI * x).sin();
        let u_pred = poly_eval(&coeffs, x);
        let err = (u_pred - u_exact).abs();
        l2_acc.add(err * err);
        if err > max_err {
            max_err = err;
        }

        let u_xx = poly_eval_dd(&coeffs, x);
        let residual = (u_xx - heat_source(x)).abs();
        res_acc.add(residual * residual);
    }

    let l2_error = (l2_acc.finalize() / n_eval as f64).sqrt();
    let mean_residual = (res_acc.finalize() / n_eval as f64).sqrt();

    PinnResult {
        final_params: coeffs,
        history,
        l2_error: Some(l2_error),
        max_error: Some(max_err),
        mean_residual,
    }
}

// ---------------------------------------------------------------------------
// Problem B: PINN — 1D Harmonic Oscillator u_xx + u = 0
// ---------------------------------------------------------------------------
// Domain: x ∈ [0, π]
// BCs: u(0) = 0, u(π) = 0
// Exact solution: u(x) = sin(x)

/// Train a PINN for the harmonic oscillator u_xx + u = 0.
///
/// Uses a small MLP with tanh activations, trained with Adam.
/// Physics loss is computed via finite-difference second derivatives.
pub fn pinn_harmonic_train(config: &PinnConfig) -> PinnResult {
    let domain = (0.0, std::f64::consts::PI);

    // --- Sampling ---
    let n_c = config.n_collocation;
    let n_d = config.n_data;
    let mut rng = cjc_repro::Rng::seeded(config.seed);

    // Training data from exact solution + noise
    let mut x_data = Vec::with_capacity(n_d);
    let mut u_data = Vec::with_capacity(n_d);
    for i in 0..n_d {
        let x = domain.0 + (i as f64 + 0.5) / n_d as f64 * (domain.1 - domain.0);
        let u_exact = x.sin();
        // Small noise
        let u1 = rng.next_f64().max(1e-300);
        let u2 = rng.next_f64();
        let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        x_data.push(x);
        u_data.push(u_exact + 0.01 * noise);
    }

    // Collocation points (interior)
    let x_colloc: Vec<f64> = (0..n_c)
        .map(|i| domain.0 + (i as f64 + 0.5) / n_c as f64 * (domain.1 - domain.0))
        .collect();

    // --- Build network graph ---
    // We'll rebuild the graph each epoch (simple approach).
    // Extract param count first.
    let mut temp_graph = crate::GradGraph::new();
    let (mlp_spec, param_indices) = mlp_init(
        &mut temp_graph,
        &config.layer_sizes,
        Activation::Tanh,
        Activation::None,
        config.seed,
    );

    // Extract initial parameter values
    let mut params: Vec<Vec<f64>> = param_indices
        .iter()
        .map(|&idx| temp_graph.tensor(idx).to_vec())
        .collect();

    let total_params: usize = params.iter().map(|p| p.len()).sum();
    let mut adam = AdamState::new(total_params, config.lr);
    let mut history = Vec::with_capacity(config.epochs);

    let eps = config.fd_eps;
    let base_bnd_weight = config.boundary_weight;

    // ── Build graph once ──────────────────────────────────────────
    // The graph topology is epoch-invariant: only parameter tensors and
    // the adaptive boundary weight change between epochs.  Build the
    // graph on the first iteration, then reuse via set_tensor + reforward.
    let mut graph = crate::GradGraph::new();
    let mut p_indices = Vec::new();
    for (li, layer) in mlp_spec.layers.iter().enumerate() {
        let w_tensor = Tensor::from_vec_unchecked(
            params[li * 2].clone(),
            &[layer.out_features, layer.in_features],
        );
        let b_tensor = Tensor::from_vec_unchecked(
            params[li * 2 + 1].clone(),
            &[layer.out_features],
        );
        p_indices.push(graph.parameter(w_tensor));
        p_indices.push(graph.parameter(b_tensor));
    }

    let mlp = Mlp {
        layers: mlp_spec
            .layers
            .iter()
            .enumerate()
            .map(|(li, l)| DenseLayer {
                weight_idx: p_indices[li * 2],
                bias_idx: p_indices[li * 2 + 1],
                activation: l.activation,
                in_features: l.in_features,
                out_features: l.out_features,
            })
            .collect(),
    };

    // ===== DATA LOSS =====
    let mut data_loss_nodes = Vec::new();
    for (j, &x) in x_data.iter().enumerate() {
        let x_in = graph.input(Tensor::from_vec_unchecked(vec![x], &[1, 1]));
        let u_pred = mlp_forward(&mut graph, &mlp, x_in);
        let u_target = graph.input(Tensor::from_vec_unchecked(vec![u_data[j]], &[1, 1]));
        let diff = graph.sub(u_pred, u_target);
        let sq = graph.mul(diff, diff);
        data_loss_nodes.push(sq);
    }

    let mut data_sum = data_loss_nodes[0];
    for &node in &data_loss_nodes[1..] {
        data_sum = graph.add(data_sum, node);
    }
    let n_data_node = graph.input(Tensor::from_vec_unchecked(
        vec![n_d as f64],
        &[1],
    ));
    let data_loss_node = graph.div(data_sum, n_data_node);

    // ===== PHYSICS LOSS (via finite differences for u_xx) =====
    let mut phys_loss_nodes = Vec::new();
    for &x in &x_colloc {
        let x_in = graph.input(Tensor::from_vec_unchecked(vec![x], &[1, 1]));
        let u_x = mlp_forward(&mut graph, &mlp, x_in);

        let x_minus = graph.input(Tensor::from_vec_unchecked(vec![x - eps], &[1, 1]));
        let u_minus = mlp_forward(&mut graph, &mlp, x_minus);

        let x_plus = graph.input(Tensor::from_vec_unchecked(vec![x + eps], &[1, 1]));
        let u_plus = mlp_forward(&mut graph, &mlp, x_plus);

        let two_u = graph.scalar_mul(u_x, 2.0);
        let sum_pm = graph.add(u_plus, u_minus);
        let numerator = graph.sub(sum_pm, two_u);
        let eps_sq_inv = 1.0 / (eps * eps);
        let u_xx = graph.scalar_mul(numerator, eps_sq_inv);

        let residual = graph.add(u_xx, u_x);
        let r_sq = graph.mul(residual, residual);
        phys_loss_nodes.push(r_sq);
    }

    let mut phys_sum = phys_loss_nodes[0];
    for &node in &phys_loss_nodes[1..] {
        phys_sum = graph.add(phys_sum, node);
    }
    let n_colloc_node = graph.input(Tensor::from_vec_unchecked(
        vec![n_c as f64],
        &[1],
    ));
    let phys_loss_node = graph.div(phys_sum, n_colloc_node);

    // ===== BOUNDARY LOSS: u(0)=0, u(π)=0 =====
    let x0_in = graph.input(Tensor::from_vec_unchecked(vec![0.0], &[1, 1]));
    let u0 = mlp_forward(&mut graph, &mlp, x0_in);
    let u0_sq = graph.mul(u0, u0);

    let xpi_in = graph.input(Tensor::from_vec_unchecked(
        vec![std::f64::consts::PI],
        &[1, 1],
    ));
    let upi = mlp_forward(&mut graph, &mlp, xpi_in);
    let upi_sq = graph.mul(upi, upi);

    let bnd_loss_node = graph.add(u0_sq, upi_sq);

    // ===== TOTAL LOSS (adaptive boundary weight is an Input node we update each epoch) =====
    let pw = graph.input(Tensor::from_vec_unchecked(
        vec![config.physics_weight],
        &[1],
    ));
    // Boundary weight node — updated via set_tensor each epoch
    let bw = graph.input(Tensor::from_vec_unchecked(
        vec![base_bnd_weight],
        &[1],
    ));
    let weighted_phys = graph.mul(pw, phys_loss_node);
    let weighted_bnd = graph.mul(bw, bnd_loss_node);
    let total_1 = graph.add(data_loss_node, weighted_phys);
    let total_loss_node = graph.add(total_1, weighted_bnd);

    // Record the first non-parameter node index for reforward range
    let reforward_start = p_indices.last().map_or(0, |&i| i + 1);

    for epoch in 0..config.epochs {
        // Cosine LR annealing
        let lr_min = config.lr * 0.01;
        let cos_decay = 0.5 * (1.0 + (std::f64::consts::PI * epoch as f64 / config.epochs as f64).cos());
        adam.lr = lr_min + (config.lr - lr_min) * cos_decay;

        // Ramp boundary weight
        let ramp = ((epoch as f64 + 1.0) / (config.epochs as f64 * 0.2)).min(1.0);
        let effective_bnd_weight = base_bnd_weight * ramp;

        if epoch > 0 {
            // Update parameter tensors with latest weights
            for (li, layer) in mlp_spec.layers.iter().enumerate() {
                graph.set_tensor(
                    p_indices[li * 2],
                    Tensor::from_vec_unchecked(
                        params[li * 2].clone(),
                        &[layer.out_features, layer.in_features],
                    ),
                );
                graph.set_tensor(
                    p_indices[li * 2 + 1],
                    Tensor::from_vec_unchecked(
                        params[li * 2 + 1].clone(),
                        &[layer.out_features],
                    ),
                );
            }
            // Update adaptive boundary weight
            graph.set_tensor(bw, Tensor::from_vec_unchecked(vec![effective_bnd_weight], &[1]));
            // Recompute all derived tensors
            graph.reforward(reforward_start, total_loss_node);
        }

        // Read loss values
        let total_loss_val = graph.value(total_loss_node);
        let data_loss_val = graph.value(data_loss_node);
        let phys_loss_val = graph.value(phys_loss_node);
        let bnd_loss_val = graph.value(bnd_loss_node);

        // Backward
        graph.zero_grad();
        graph.backward(total_loss_node);

        // Collect gradients and update params
        let mut flat_params = Vec::with_capacity(total_params);
        let mut flat_grads = Vec::with_capacity(total_params);
        for (pi, &idx) in p_indices.iter().enumerate() {
            let grad = graph.grad(idx).unwrap_or_else(|| {
                Tensor::zeros(&graph.tensor(idx).shape().to_vec())
            });
            let grad_vec = grad.to_vec();
            flat_params.extend_from_slice(&params[pi]);
            flat_grads.extend_from_slice(&grad_vec);
        }

        let mut grad_norm_acc = KahanAccumulatorF64::new();
        for &g in &flat_grads {
            grad_norm_acc.add(g * g);
        }
        let grad_norm = grad_norm_acc.finalize().sqrt();

        // Adam step on flattened params
        adam_step(&mut flat_params, &flat_grads, &mut adam);

        // Unflatten back
        let mut offset = 0;
        for pi in 0..params.len() {
            let n = params[pi].len();
            params[pi].copy_from_slice(&flat_params[offset..offset + n]);
            offset += n;
        }

        history.push(TrainLog {
            epoch,
            total_loss: total_loss_val,
            data_loss: data_loss_val,
            physics_loss: phys_loss_val,
            boundary_loss: bnd_loss_val,
            grad_norm,
        });
    }

    // --- Evaluate final accuracy ---
    let n_eval = 100;
    let mut l2_acc = KahanAccumulatorF64::new();
    let mut max_err = 0.0f64;
    let mut res_acc = KahanAccumulatorF64::new();

    for i in 0..n_eval {
        let x = (i as f64 + 0.5) / n_eval as f64 * std::f64::consts::PI;
        let u_exact = x.sin();

        // Evaluate network at x
        let mut eval_graph = crate::GradGraph::new();
        let mut ep_indices = Vec::new();
        for (li, layer) in mlp_spec.layers.iter().enumerate() {
            let w = Tensor::from_vec_unchecked(
                params[li * 2].clone(),
                &[layer.out_features, layer.in_features],
            );
            let b = Tensor::from_vec_unchecked(
                params[li * 2 + 1].clone(),
                &[layer.out_features],
            );
            ep_indices.push(eval_graph.parameter(w));
            ep_indices.push(eval_graph.parameter(b));
        }
        let eval_mlp = Mlp {
            layers: mlp_spec
                .layers
                .iter()
                .enumerate()
                .map(|(li, l)| DenseLayer {
                    weight_idx: ep_indices[li * 2],
                    bias_idx: ep_indices[li * 2 + 1],
                    activation: l.activation,
                    in_features: l.in_features,
                    out_features: l.out_features,
                })
                .collect(),
        };

        let x_in = eval_graph.input(Tensor::from_vec_unchecked(vec![x], &[1, 1]));
        let u_pred_idx = mlp_forward(&mut eval_graph, &eval_mlp, x_in);
        let u_pred = eval_graph.value(u_pred_idx);

        let err = (u_pred - u_exact).abs();
        l2_acc.add(err * err);
        if err > max_err {
            max_err = err;
        }

        // Compute residual: u_xx + u
        let x_m = eval_graph.input(Tensor::from_vec_unchecked(vec![x - eps], &[1, 1]));
        let u_m = mlp_forward(&mut eval_graph, &eval_mlp, x_m);
        let x_p = eval_graph.input(Tensor::from_vec_unchecked(vec![x + eps], &[1, 1]));
        let u_p = mlp_forward(&mut eval_graph, &eval_mlp, x_p);
        let u_xx_val = (eval_graph.value(u_p) - 2.0 * u_pred + eval_graph.value(u_m)) / (eps * eps);
        let residual = (u_xx_val + u_pred).abs();
        res_acc.add(residual * residual);
    }

    let l2_error = (l2_acc.finalize() / n_eval as f64).sqrt();
    let mean_residual = (res_acc.finalize() / n_eval as f64).sqrt();

    let final_params: Vec<f64> = params.iter().flat_map(|p| p.iter().copied()).collect();

    PinnResult {
        final_params,
        history,
        l2_error: Some(l2_error),
        max_error: Some(max_err),
        mean_residual,
    }
}

// ---------------------------------------------------------------------------
// Deterministic Sampling Utilities
// ---------------------------------------------------------------------------

/// Generate a uniform grid on [a, b] with n points.
pub fn uniform_grid(a: f64, b: f64, n: usize) -> Vec<f64> {
    (0..n).map(|i| a + (i as f64 + 0.5) / n as f64 * (b - a)).collect()
}

/// Generate LHS samples on [a, b] using `latin_hypercube_sample`.
pub fn lhs_grid_1d(a: f64, b: f64, n: usize, seed: u64) -> Vec<f64> {
    let lhs = cjc_runtime::distributions::latin_hypercube_sample(n, 1, seed);
    let data = lhs.to_vec();
    data.iter().map(|&u| a + u * (b - a)).collect()
}

// ---------------------------------------------------------------------------
// Visualization Helpers (ASCII)
// ---------------------------------------------------------------------------

/// Render an ASCII line plot of (x, y) data.
///
/// Returns a string with a simple ASCII chart suitable for terminal output.
pub fn ascii_plot(x: &[f64], y: &[f64], width: usize, height: usize, title: &str) -> String {
    if x.is_empty() || y.is_empty() || x.len() != y.len() {
        return format!("[empty plot: {}]", title);
    }

    let y_min = y.iter().copied().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let x_min = x.iter().copied().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let y_range = if (y_max - y_min).abs() < 1e-15 { 1.0 } else { y_max - y_min };
    let x_range = if (x_max - x_min).abs() < 1e-15 { 1.0 } else { x_max - x_min };

    // Build character grid
    let mut grid = vec![vec![' '; width]; height];

    for i in 0..x.len() {
        let col = ((x[i] - x_min) / x_range * (width - 1) as f64).round() as usize;
        let row = ((y_max - y[i]) / y_range * (height - 1) as f64).round() as usize;
        let col = col.min(width - 1);
        let row = row.min(height - 1);
        grid[row][col] = '*';
    }

    let mut out = String::new();
    out.push_str(&format!("  {}\n", title));
    out.push_str(&format!("  {:.4} |\n", y_max));
    for row in &grid {
        out.push_str("         |");
        for &ch in row {
            out.push(ch);
        }
        out.push('\n');
    }
    out.push_str(&format!("  {:.4} |", y_min));
    for _ in 0..width {
        out.push('-');
    }
    out.push('\n');
    out.push_str(&format!(
        "         {:.4}{}  {:.4}\n",
        x_min,
        " ".repeat(width.saturating_sub(12)),
        x_max,
    ));

    out
}

/// Render a training loss curve as ASCII.
pub fn plot_loss_history(history: &[TrainLog], width: usize, height: usize) -> String {
    if history.is_empty() {
        return "[no training data]".to_string();
    }

    let epochs: Vec<f64> = history.iter().map(|h| h.epoch as f64).collect();
    let losses: Vec<f64> = history.iter().map(|h| h.total_loss.ln().max(-20.0)).collect();

    ascii_plot(&epochs, &losses, width, height, "Training Loss (log scale)")
}

// ---------------------------------------------------------------------------
// Phase 2: Domain Geometry
// ---------------------------------------------------------------------------

/// Describes the spatial/temporal domain for a PDE problem.
#[derive(Debug, Clone)]
pub enum PinnDomain {
    /// 1D interval [a, b].
    Interval1D { a: f64, b: f64 },
    /// 2D rectangle [x0, x1] × [y0, y1].
    Rectangle2D {
        x_range: (f64, f64),
        y_range: (f64, f64),
    },
    /// 1D space + time: x ∈ [x0, x1], t ∈ [t0, t1].
    SpaceTime1D {
        x_range: (f64, f64),
        t_range: (f64, f64),
    },
    /// Unit disk centered at origin with given radius.
    Disk {
        center: (f64, f64),
        radius: f64,
    },
    /// L-shaped domain: [0,1]² minus [0.5,1]×[0.5,1].
    LShape,
    /// Convex polygon defined by vertices (ordered counterclockwise).
    Polygon {
        vertices: Vec<(f64, f64)>,
    },
    /// 3D cuboid [x0,x1] × [y0,y1] × [z0,z1].
    Cuboid3D {
        x_range: (f64, f64),
        y_range: (f64, f64),
        z_range: (f64, f64),
    },
    /// 2D space + time: (x,y) ∈ rect, t ∈ [t0,t1]. Input dim = 3.
    SpaceTime2D {
        x_range: (f64, f64),
        y_range: (f64, f64),
        t_range: (f64, f64),
    },
}

impl PinnDomain {
    /// Sample `n` interior points deterministically.
    ///
    /// Returns a flat `Vec<f64>` of coordinates: for 1D domains each entry is
    /// one scalar; for 2D/space-time domains each point is two consecutive
    /// values `[x, y]` or `[x, t]`.
    pub fn sample_interior(&self, n: usize, seed: u64) -> Vec<f64> {
        match self {
            PinnDomain::Interval1D { a, b } => {
                lhs_grid_1d(*a, *b, n, seed)
            }
            PinnDomain::Rectangle2D { x_range, y_range } => {
                let lhs = cjc_runtime::distributions::latin_hypercube_sample(n, 2, seed);
                let data = lhs.to_vec();
                let mut out = Vec::with_capacity(n * 2);
                for i in 0..n {
                    let u = data[i * 2];
                    let v = data[i * 2 + 1];
                    out.push(x_range.0 + u * (x_range.1 - x_range.0));
                    out.push(y_range.0 + v * (y_range.1 - y_range.0));
                }
                out
            }
            PinnDomain::SpaceTime1D { x_range, t_range } => {
                let lhs = cjc_runtime::distributions::latin_hypercube_sample(n, 2, seed);
                let data = lhs.to_vec();
                let mut out = Vec::with_capacity(n * 2);
                for i in 0..n {
                    let u = data[i * 2];
                    let v = data[i * 2 + 1];
                    out.push(x_range.0 + u * (x_range.1 - x_range.0));
                    out.push(t_range.0 + v * (t_range.1 - t_range.0));
                }
                out
            }
            PinnDomain::Disk { center, radius } => {
                // Rejection sampling in bounding box, deterministic via SplitMix64
                let mut rng = cjc_repro::Rng::seeded(seed);
                let mut out = Vec::with_capacity(n * 2);
                while out.len() / 2 < n {
                    let u = rng.next_f64() * 2.0 - 1.0;
                    let v = rng.next_f64() * 2.0 - 1.0;
                    if u * u + v * v < 1.0 {
                        out.push(center.0 + u * radius);
                        out.push(center.1 + v * radius);
                    }
                }
                out
            }
            PinnDomain::LShape => {
                // [0,1]² minus [0.5,1]×[0.5,1]: rejection sampling
                let mut rng = cjc_repro::Rng::seeded(seed);
                let mut out = Vec::with_capacity(n * 2);
                while out.len() / 2 < n {
                    let x = rng.next_f64();
                    let y = rng.next_f64();
                    if !(x > 0.5 && y > 0.5) {
                        out.push(x);
                        out.push(y);
                    }
                }
                out
            }
            PinnDomain::Polygon { vertices } => {
                // Bounding box + rejection via winding number
                let (mut xmin, mut xmax) = (f64::MAX, f64::MIN);
                let (mut ymin, mut ymax) = (f64::MAX, f64::MIN);
                for &(vx, vy) in vertices.iter() {
                    if vx < xmin { xmin = vx; }
                    if vx > xmax { xmax = vx; }
                    if vy < ymin { ymin = vy; }
                    if vy > ymax { ymax = vy; }
                }
                let mut rng = cjc_repro::Rng::seeded(seed);
                let mut out = Vec::with_capacity(n * 2);
                while out.len() / 2 < n {
                    let x = xmin + rng.next_f64() * (xmax - xmin);
                    let y = ymin + rng.next_f64() * (ymax - ymin);
                    if point_in_polygon(x, y, vertices) {
                        out.push(x);
                        out.push(y);
                    }
                }
                out
            }
            PinnDomain::Cuboid3D { x_range, y_range, z_range } => {
                let lhs = cjc_runtime::distributions::latin_hypercube_sample(n, 3, seed);
                let data = lhs.to_vec();
                let mut out = Vec::with_capacity(n * 3);
                for i in 0..n {
                    out.push(x_range.0 + data[i * 3] * (x_range.1 - x_range.0));
                    out.push(y_range.0 + data[i * 3 + 1] * (y_range.1 - y_range.0));
                    out.push(z_range.0 + data[i * 3 + 2] * (z_range.1 - z_range.0));
                }
                out
            }
            PinnDomain::SpaceTime2D { x_range, y_range, t_range } => {
                let lhs = cjc_runtime::distributions::latin_hypercube_sample(n, 3, seed);
                let data = lhs.to_vec();
                let mut out = Vec::with_capacity(n * 3);
                for i in 0..n {
                    out.push(x_range.0 + data[i * 3] * (x_range.1 - x_range.0));
                    out.push(y_range.0 + data[i * 3 + 1] * (y_range.1 - y_range.0));
                    out.push(t_range.0 + data[i * 3 + 2] * (t_range.1 - t_range.0));
                }
                out
            }
        }
    }

    /// Sample `n` points on the domain boundary deterministically.
    ///
    /// For 1D: returns the two endpoints.
    /// For 2D rect: distributes n/4 points on each edge.
    /// For SpaceTime1D: samples n points on the spatial boundaries at various times,
    /// plus the initial condition line.
    pub fn sample_boundary(&self, n: usize, _seed: u64) -> Vec<f64> {
        match self {
            PinnDomain::Interval1D { a, b } => {
                vec![*a, *b]
            }
            PinnDomain::Rectangle2D { x_range, y_range } => {
                let per_side = n.max(4) / 4;
                let mut out = Vec::with_capacity(per_side * 4 * 2);
                // Bottom edge: y = y0
                for i in 0..per_side {
                    let frac = (i as f64 + 0.5) / per_side as f64;
                    out.push(x_range.0 + frac * (x_range.1 - x_range.0));
                    out.push(y_range.0);
                }
                // Top edge: y = y1
                for i in 0..per_side {
                    let frac = (i as f64 + 0.5) / per_side as f64;
                    out.push(x_range.0 + frac * (x_range.1 - x_range.0));
                    out.push(y_range.1);
                }
                // Left edge: x = x0
                for i in 0..per_side {
                    let frac = (i as f64 + 0.5) / per_side as f64;
                    out.push(x_range.0);
                    out.push(y_range.0 + frac * (y_range.1 - y_range.0));
                }
                // Right edge: x = x1
                for i in 0..per_side {
                    let frac = (i as f64 + 0.5) / per_side as f64;
                    out.push(x_range.1);
                    out.push(y_range.0 + frac * (y_range.1 - y_range.0));
                }
                out
            }
            PinnDomain::SpaceTime1D { x_range, t_range } => {
                // Split: n/3 on x=x0, n/3 on x=x1, n/3 on t=t0 (IC)
                let per_part = n.max(3) / 3;
                let mut out = Vec::with_capacity(per_part * 3 * 2);
                // Left boundary: x = x0
                for i in 0..per_part {
                    let frac = (i as f64 + 0.5) / per_part as f64;
                    out.push(x_range.0);
                    out.push(t_range.0 + frac * (t_range.1 - t_range.0));
                }
                // Right boundary: x = x1
                for i in 0..per_part {
                    let frac = (i as f64 + 0.5) / per_part as f64;
                    out.push(x_range.1);
                    out.push(t_range.0 + frac * (t_range.1 - t_range.0));
                }
                // Initial condition: t = t0
                for i in 0..per_part {
                    let frac = (i as f64 + 0.5) / per_part as f64;
                    out.push(x_range.0 + frac * (x_range.1 - x_range.0));
                    out.push(t_range.0);
                }
                out
            }
            PinnDomain::Disk { center, radius } => {
                // Points on the circle
                let mut out = Vec::with_capacity(n * 2);
                for i in 0..n {
                    let theta = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                    out.push(center.0 + radius * theta.cos());
                    out.push(center.1 + radius * theta.sin());
                }
                out
            }
            PinnDomain::LShape => {
                // 5-segment boundary of L-shape
                // Segments: (0,0)→(1,0), (1,0)→(1,0.5), (1,0.5)→(0.5,0.5),
                //           (0.5,0.5)→(0.5,1), (0.5,1)→(0,1), (0,1)→(0,0)
                let segments: Vec<((f64,f64),(f64,f64))> = vec![
                    ((0.0,0.0),(1.0,0.0)), ((1.0,0.0),(1.0,0.5)),
                    ((1.0,0.5),(0.5,0.5)), ((0.5,0.5),(0.5,1.0)),
                    ((0.5,1.0),(0.0,1.0)), ((0.0,1.0),(0.0,0.0)),
                ];
                let per_seg = n.max(6) / 6;
                let mut out = Vec::with_capacity(per_seg * 6 * 2);
                for ((x0,y0),(x1,y1)) in &segments {
                    for i in 0..per_seg {
                        let frac = (i as f64 + 0.5) / per_seg as f64;
                        out.push(x0 + frac * (x1 - x0));
                        out.push(y0 + frac * (y1 - y0));
                    }
                }
                out
            }
            PinnDomain::Polygon { vertices } => {
                let n_edges = vertices.len();
                let per_edge = n.max(n_edges) / n_edges;
                let mut out = Vec::with_capacity(per_edge * n_edges * 2);
                for e in 0..n_edges {
                    let (x0, y0) = vertices[e];
                    let (x1, y1) = vertices[(e + 1) % n_edges];
                    for i in 0..per_edge {
                        let frac = (i as f64 + 0.5) / per_edge as f64;
                        out.push(x0 + frac * (x1 - x0));
                        out.push(y0 + frac * (y1 - y0));
                    }
                }
                out
            }
            PinnDomain::Cuboid3D { x_range, y_range, z_range } => {
                // 6 faces, n/6 points per face
                let per_face = n.max(6) / 6;
                let mut out = Vec::with_capacity(per_face * 6 * 3);
                let mut rng = cjc_repro::Rng::seeded(_seed);
                for face in 0..6 {
                    for _ in 0..per_face {
                        let u = rng.next_f64();
                        let v = rng.next_f64();
                        let (x, y, z) = match face {
                            0 => (x_range.0, y_range.0 + u * (y_range.1 - y_range.0), z_range.0 + v * (z_range.1 - z_range.0)),
                            1 => (x_range.1, y_range.0 + u * (y_range.1 - y_range.0), z_range.0 + v * (z_range.1 - z_range.0)),
                            2 => (x_range.0 + u * (x_range.1 - x_range.0), y_range.0, z_range.0 + v * (z_range.1 - z_range.0)),
                            3 => (x_range.0 + u * (x_range.1 - x_range.0), y_range.1, z_range.0 + v * (z_range.1 - z_range.0)),
                            4 => (x_range.0 + u * (x_range.1 - x_range.0), y_range.0 + v * (y_range.1 - y_range.0), z_range.0),
                            _ => (x_range.0 + u * (x_range.1 - x_range.0), y_range.0 + v * (y_range.1 - y_range.0), z_range.1),
                        };
                        out.push(x); out.push(y); out.push(z);
                    }
                }
                out
            }
            PinnDomain::SpaceTime2D { x_range, y_range, t_range } => {
                // Split: spatial boundary at various t + initial condition surface
                let per_part = n.max(5) / 5;
                let mut out = Vec::with_capacity(per_part * 5 * 3);
                for i in 0..per_part {
                    let frac = (i as f64 + 0.5) / per_part as f64;
                    // x = x0
                    out.push(x_range.0); out.push(y_range.0 + frac * (y_range.1 - y_range.0));
                    out.push(t_range.0 + frac * (t_range.1 - t_range.0));
                }
                for i in 0..per_part {
                    let frac = (i as f64 + 0.5) / per_part as f64;
                    // x = x1
                    out.push(x_range.1); out.push(y_range.0 + frac * (y_range.1 - y_range.0));
                    out.push(t_range.0 + frac * (t_range.1 - t_range.0));
                }
                for i in 0..per_part {
                    let frac = (i as f64 + 0.5) / per_part as f64;
                    // y = y0
                    out.push(x_range.0 + frac * (x_range.1 - x_range.0)); out.push(y_range.0);
                    out.push(t_range.0 + frac * (t_range.1 - t_range.0));
                }
                for i in 0..per_part {
                    let frac = (i as f64 + 0.5) / per_part as f64;
                    // y = y1
                    out.push(x_range.0 + frac * (x_range.1 - x_range.0)); out.push(y_range.1);
                    out.push(t_range.0 + frac * (t_range.1 - t_range.0));
                }
                // IC surface: t = t0
                let mut rng = cjc_repro::Rng::seeded(_seed);
                for _ in 0..per_part {
                    out.push(x_range.0 + rng.next_f64() * (x_range.1 - x_range.0));
                    out.push(y_range.0 + rng.next_f64() * (y_range.1 - y_range.0));
                    out.push(t_range.0);
                }
                out
            }
        }
    }

    /// Check whether a point lies on the boundary (within tolerance).
    pub fn is_boundary(&self, point: &[f64], tol: f64) -> bool {
        match self {
            PinnDomain::Interval1D { a, b } => {
                point.len() >= 1 && ((point[0] - a).abs() < tol || (point[0] - b).abs() < tol)
            }
            PinnDomain::Rectangle2D { x_range, y_range } => {
                if point.len() < 2 { return false; }
                let (x, y) = (point[0], point[1]);
                (x - x_range.0).abs() < tol
                    || (x - x_range.1).abs() < tol
                    || (y - y_range.0).abs() < tol
                    || (y - y_range.1).abs() < tol
            }
            PinnDomain::SpaceTime1D { x_range, t_range } => {
                if point.len() < 2 { return false; }
                let (x, t) = (point[0], point[1]);
                (x - x_range.0).abs() < tol
                    || (x - x_range.1).abs() < tol
                    || (t - t_range.0).abs() < tol
            }
            PinnDomain::Disk { center, radius } => {
                if point.len() < 2 { return false; }
                let dx = point[0] - center.0;
                let dy = point[1] - center.1;
                ((dx * dx + dy * dy).sqrt() - radius).abs() < tol
            }
            PinnDomain::LShape => {
                if point.len() < 2 { return false; }
                let (x, y) = (point[0], point[1]);
                // Boundary of L-shape: outer square edges + inner corner edges
                x.abs() < tol || y.abs() < tol
                    || ((x - 1.0).abs() < tol && y <= 0.5 + tol)
                    || ((y - 1.0).abs() < tol && x <= 0.5 + tol)
                    || ((x - 0.5).abs() < tol && y >= 0.5 - tol)
                    || ((y - 0.5).abs() < tol && x >= 0.5 - tol)
            }
            PinnDomain::Polygon { vertices } => {
                if point.len() < 2 { return false; }
                let (px, py) = (point[0], point[1]);
                for i in 0..vertices.len() {
                    let (x0, y0) = vertices[i];
                    let (x1, y1) = vertices[(i + 1) % vertices.len()];
                    let dist = point_to_segment_dist(px, py, x0, y0, x1, y1);
                    if dist < tol { return true; }
                }
                false
            }
            PinnDomain::Cuboid3D { x_range, y_range, z_range } => {
                if point.len() < 3 { return false; }
                let (x, y, z) = (point[0], point[1], point[2]);
                (x - x_range.0).abs() < tol || (x - x_range.1).abs() < tol
                    || (y - y_range.0).abs() < tol || (y - y_range.1).abs() < tol
                    || (z - z_range.0).abs() < tol || (z - z_range.1).abs() < tol
            }
            PinnDomain::SpaceTime2D { x_range, y_range, t_range } => {
                if point.len() < 3 { return false; }
                let (x, y, t) = (point[0], point[1], point[2]);
                (x - x_range.0).abs() < tol || (x - x_range.1).abs() < tol
                    || (y - y_range.0).abs() < tol || (y - y_range.1).abs() < tol
                    || (t - t_range.0).abs() < tol
            }
        }
    }

    /// Input dimensionality of this domain.
    pub fn input_dim(&self) -> usize {
        match self {
            PinnDomain::Interval1D { .. } => 1,
            PinnDomain::Rectangle2D { .. } | PinnDomain::SpaceTime1D { .. }
            | PinnDomain::Disk { .. } | PinnDomain::LShape
            | PinnDomain::Polygon { .. } => 2,
            PinnDomain::Cuboid3D { .. } | PinnDomain::SpaceTime2D { .. } => 3,
        }
    }
}

/// Point-in-polygon test using ray casting (deterministic).
fn point_in_polygon(px: f64, py: f64, vertices: &[(f64, f64)]) -> bool {
    let n = vertices.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = vertices[i];
        let (xj, yj) = vertices[j];
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Distance from point (px, py) to line segment (x0,y0)-(x1,y1).
fn point_to_segment_dist(px: f64, py: f64, x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-30 {
        return ((px - x0).powi(2) + (py - y0).powi(2)).sqrt();
    }
    let t = ((px - x0) * dx + (py - y0) * dy) / len_sq;
    let t = t.max(0.0).min(1.0);
    let proj_x = x0 + t * dx;
    let proj_y = y0 + t * dy;
    ((px - proj_x).powi(2) + (py - proj_y).powi(2)).sqrt()
}

// ---------------------------------------------------------------------------
// Hard Boundary Constraint Enforcement
// ---------------------------------------------------------------------------

/// Apply a hard Dirichlet BC transform on a 1D interval [a, b].
///
/// Builds `u(x) = g(x) + d(x) * NN(x)` on the graph where:
/// - `g(x) = (1 - t) * g_a + t * g_b`  with  `t = (x - a) / (b - a)`
///   linearly interpolates the boundary values
/// - `d(x) = (x - a) * (b - x) / ((b - a)/2)^2`  is zero at both endpoints
///
/// This guarantees the boundary conditions are satisfied exactly regardless
/// of what the network outputs.
pub fn hard_bc_1d(
    graph: &mut crate::GradGraph,
    nn_output_idx: usize,
    x_idx: usize,
    a: f64,
    b: f64,
    g_a: f64,
    g_b: f64,
) -> usize {
    // t = (x - a) / (b - a)
    let a_node = graph.input(Tensor::from_vec_unchecked(vec![a], &[1]));
    let b_node = graph.input(Tensor::from_vec_unchecked(vec![b], &[1]));
    let x_minus_a = graph.sub(x_idx, a_node);
    let b_minus_a = graph.sub(b_node, a_node);
    let t = graph.div(x_minus_a, b_minus_a);

    // g(x) = (1-t)*g_a + t*g_b
    let one = graph.input(Tensor::from_vec_unchecked(vec![1.0], &[1]));
    let one_minus_t = graph.sub(one, t);
    let g_a_scaled = graph.scalar_mul(one_minus_t, g_a);
    let g_b_scaled = graph.scalar_mul(t, g_b);
    let g_x = graph.add(g_a_scaled, g_b_scaled);

    // d(x) = (x - a) * (b - x)  (zero at both endpoints)
    let b_minus_x = graph.sub(b_node, x_idx);
    let d_x = graph.mul(x_minus_a, b_minus_x);

    // Normalize: d_norm = d(x) / ((b-a)/2)^2 so peak = 1 at midpoint
    let half_range_sq = ((b - a) / 2.0).powi(2);
    let d_norm = graph.scalar_mul(d_x, 1.0 / half_range_sq);

    // u(x) = g(x) + d_norm(x) * NN(x)
    let nn_contribution = graph.mul(d_norm, nn_output_idx);
    graph.add(g_x, nn_contribution)
}

// ---------------------------------------------------------------------------
// Residual-Based Adaptive Refinement (RAR)
// ---------------------------------------------------------------------------

/// Perform one round of residual-based adaptive refinement.
///
/// Given residual values at candidate points, select the top `fraction` of
/// points with the highest absolute residual and merge them with the existing
/// collocation set.
///
/// All operations are deterministic: sorting uses a total-order comparison
/// that handles NaN consistently.
///
/// # Arguments
///
/// * `existing` — Current collocation points (flat, `dim` values per point).
/// * `candidates` — New candidate points (flat, `dim` values per point).
/// * `residuals` — Absolute residual at each candidate point (one per point).
/// * `fraction` — Fraction of candidates to keep (0.0–1.0).
/// * `dim` — Spatial dimension of each point.
///
/// # Returns
///
/// Merged point set (flat, `dim` values per point).
pub fn adaptive_refine(
    existing: &[f64],
    candidates: &[f64],
    residuals: &[f64],
    fraction: f64,
    dim: usize,
) -> Vec<f64> {
    let n_cand = residuals.len();
    if n_cand == 0 || fraction <= 0.0 {
        return existing.to_vec();
    }

    // Build index-residual pairs and sort descending by |residual|
    let mut indexed: Vec<(usize, f64)> = residuals
        .iter()
        .enumerate()
        .map(|(i, &r)| (i, r.abs()))
        .collect();
    // Deterministic sort: total ordering via to_bits for NaN stability
    indexed.sort_by(|a, b| b.1.to_bits().cmp(&a.1.to_bits()));

    let keep = ((n_cand as f64 * fraction).ceil() as usize).min(n_cand);

    let mut merged = existing.to_vec();
    for &(idx, _) in &indexed[..keep] {
        let start = idx * dim;
        let end = start + dim;
        if end <= candidates.len() {
            merged.extend_from_slice(&candidates[start..end]);
        }
    }
    merged
}

// ---------------------------------------------------------------------------
// Boundary Condition Types
// ---------------------------------------------------------------------------

/// Types of boundary conditions supported by the PINN infrastructure.
///
/// Each variant carries the data needed to compute the BC loss contribution
/// at a set of boundary points.
#[derive(Debug, Clone)]
pub enum BoundaryCondition {
    /// Dirichlet: u(x_b) = g(x_b). Value is prescribed.
    Dirichlet {
        /// Boundary point coordinates (flat, `dim` values per point).
        points: Vec<f64>,
        /// Prescribed values at each boundary point.
        values: Vec<f64>,
        /// Spatial dimension per point.
        dim: usize,
    },
    /// Neumann: ∂u/∂n(x_b) = h(x_b). Normal derivative is prescribed.
    Neumann {
        /// Boundary point coordinates (flat, `dim` values per point).
        points: Vec<f64>,
        /// Prescribed normal derivative values.
        values: Vec<f64>,
        /// Outward normal vectors at each point (flat, `dim` values per point).
        normals: Vec<f64>,
        /// Spatial dimension per point.
        dim: usize,
        /// Finite-difference epsilon for derivative approximation.
        fd_eps: f64,
    },
    /// Robin: α·u(x_b) + β·∂u/∂n(x_b) = g(x_b). Mixed condition.
    Robin {
        /// Boundary point coordinates (flat, `dim` values per point).
        points: Vec<f64>,
        /// Right-hand side values g(x_b).
        values: Vec<f64>,
        /// Outward normal vectors (flat, `dim` values per point).
        normals: Vec<f64>,
        /// Coefficient α (multiplier on u).
        alpha: f64,
        /// Coefficient β (multiplier on ∂u/∂n).
        beta: f64,
        /// Spatial dimension per point.
        dim: usize,
        /// Finite-difference epsilon.
        fd_eps: f64,
    },
    /// Periodic: u(x_left) = u(x_right). Matching values at opposite boundaries.
    Periodic {
        /// Left boundary coordinates (flat, `dim` values per point).
        left_points: Vec<f64>,
        /// Right boundary coordinates (flat, `dim` values per point).
        right_points: Vec<f64>,
        /// Spatial dimension per point.
        dim: usize,
    },
}

/// Compute the boundary condition loss for a given BC specification.
///
/// For each boundary point, evaluates the network via `eval_fn` and computes
/// the appropriate loss term. Returns the mean-squared BC violation.
///
/// - **Dirichlet:** MSE(u(x_b) - g)
/// - **Neumann:** MSE(∂u/∂n(x_b) - h), using FD along normal direction
/// - **Robin:** MSE(α·u(x_b) + β·∂u/∂n(x_b) - g)
/// - **Periodic:** MSE(u(x_left) - u(x_right))
///
/// The `eval_fn` takes a flat coordinate slice and returns the scalar network
/// output at that point.
pub fn bc_loss<F>(bc: &BoundaryCondition, eval_fn: &F) -> f64
where
    F: Fn(&[f64]) -> f64,
{
    match bc {
        BoundaryCondition::Dirichlet { points, values, dim } => {
            let n = values.len();
            if n == 0 { return 0.0; }
            let mut acc = KahanAccumulatorF64::new();
            for i in 0..n {
                let pt = &points[i * dim..(i + 1) * dim];
                let u = eval_fn(pt);
                let err = u - values[i];
                acc.add(err * err);
            }
            acc.finalize() / n as f64
        }
        BoundaryCondition::Neumann { points, values, normals, dim, fd_eps } => {
            let n = values.len();
            if n == 0 { return 0.0; }
            let mut acc = KahanAccumulatorF64::new();
            for i in 0..n {
                let pt = &points[i * dim..(i + 1) * dim];
                let normal = &normals[i * dim..(i + 1) * dim];
                // FD approximation of ∂u/∂n ≈ (u(x + ε·n) - u(x - ε·n)) / (2ε)
                let mut pt_plus: Vec<f64> = pt.to_vec();
                let mut pt_minus: Vec<f64> = pt.to_vec();
                for d in 0..*dim {
                    pt_plus[d] += fd_eps * normal[d];
                    pt_minus[d] -= fd_eps * normal[d];
                }
                let du_dn = (eval_fn(&pt_plus) - eval_fn(&pt_minus)) / (2.0 * fd_eps);
                let err = du_dn - values[i];
                acc.add(err * err);
            }
            acc.finalize() / n as f64
        }
        BoundaryCondition::Robin { points, values, normals, alpha, beta, dim, fd_eps } => {
            let n = values.len();
            if n == 0 { return 0.0; }
            let mut acc = KahanAccumulatorF64::new();
            for i in 0..n {
                let pt = &points[i * dim..(i + 1) * dim];
                let normal = &normals[i * dim..(i + 1) * dim];
                let u = eval_fn(pt);
                let mut pt_plus: Vec<f64> = pt.to_vec();
                let mut pt_minus: Vec<f64> = pt.to_vec();
                for d in 0..*dim {
                    pt_plus[d] += fd_eps * normal[d];
                    pt_minus[d] -= fd_eps * normal[d];
                }
                let du_dn = (eval_fn(&pt_plus) - eval_fn(&pt_minus)) / (2.0 * fd_eps);
                let err = alpha * u + beta * du_dn - values[i];
                acc.add(err * err);
            }
            acc.finalize() / n as f64
        }
        BoundaryCondition::Periodic { left_points, right_points, dim } => {
            let n = left_points.len() / dim;
            if n == 0 { return 0.0; }
            let mut acc = KahanAccumulatorF64::new();
            for i in 0..n {
                let left = &left_points[i * dim..(i + 1) * dim];
                let right = &right_points[i * dim..(i + 1) * dim];
                let err = eval_fn(left) - eval_fn(right);
                acc.add(err * err);
            }
            acc.finalize() / n as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Problem C: PINN — 1D Burgers' Equation
// ---------------------------------------------------------------------------
// PDE:  u_t + u·u_x = ν·u_xx   on x ∈ [-1,1], t ∈ [0,1]
// IC:   u(x, 0) = -sin(πx)
// BCs:  u(-1, t) = 0, u(1, t) = 0
// ν = 0.01/π  (Raissi 2019)
// Exact: Cole-Hopf transform (not computed here — we validate IC/BC/residual)

/// Configuration for the Burgers equation PINN.
#[derive(Debug, Clone)]
pub struct BurgersConfig {
    /// MLP layer sizes, e.g. [2, 20, 20, 20, 1].
    pub layer_sizes: Vec<usize>,
    /// Number of training epochs.
    pub epochs: usize,
    /// Learning rate for Adam optimizer.
    pub lr: f64,
    /// Kinematic viscosity (default: 0.01/π).
    pub nu: f64,
    /// Weight for physics loss.
    pub physics_weight: f64,
    /// Weight for boundary/IC loss.
    pub boundary_weight: f64,
    /// Random seed.
    pub seed: u64,
    /// Number of interior collocation points.
    pub n_collocation: usize,
    /// Number of initial-condition data points.
    pub n_ic: usize,
    /// Number of boundary-condition points per side.
    pub n_bc: usize,
    /// Finite-difference epsilon.
    pub fd_eps: f64,
}

impl Default for BurgersConfig {
    fn default() -> Self {
        Self {
            layer_sizes: vec![2, 20, 20, 20, 1],
            epochs: 500,
            lr: 1e-3,
            nu: 0.01 / std::f64::consts::PI,
            physics_weight: 1.0,
            boundary_weight: 10.0,
            seed: 42,
            n_collocation: 64,
            n_ic: 50,
            n_bc: 25,
            fd_eps: 1e-4,
        }
    }
}

/// Train a PINN for the 1D viscous Burgers' equation.
///
/// Uses a multi-layer perceptron with 2D input `(x, t)` and 1D output `u`.
/// The physics residual is `r = u_t + u·u_x − ν·u_xx`, computed via central
/// finite differences on the network's forward pass.
pub fn pinn_burgers_train(config: &BurgersConfig) -> PinnResult {
    let x_range = (-1.0, 1.0);
    let t_range = (0.0, 1.0);
    let eps = config.fd_eps;
    let nu = config.nu;

    let domain = PinnDomain::SpaceTime1D { x_range, t_range };
    let interior_pts = domain.sample_interior(config.n_collocation, config.seed);

    // IC points: u(x, 0) = -sin(πx)
    let mut ic_x = Vec::with_capacity(config.n_ic);
    let mut ic_u = Vec::with_capacity(config.n_ic);
    for i in 0..config.n_ic {
        let x = x_range.0 + (i as f64 + 0.5) / config.n_ic as f64 * (x_range.1 - x_range.0);
        ic_x.push(x);
        ic_u.push(-(std::f64::consts::PI * x).sin());
    }

    // BC points: u(-1, t) = 0, u(1, t) = 0
    let mut bc_pts = Vec::with_capacity(config.n_bc * 2 * 2);
    let mut bc_vals = Vec::with_capacity(config.n_bc * 2);
    for i in 0..config.n_bc {
        let t = t_range.0 + (i as f64 + 0.5) / config.n_bc as f64 * (t_range.1 - t_range.0);
        bc_pts.push(x_range.0); bc_pts.push(t); bc_vals.push(0.0); // left
        bc_pts.push(x_range.1); bc_pts.push(t); bc_vals.push(0.0); // right
    }

    // --- Build network ---
    let mut temp_graph = crate::GradGraph::new();
    let (mlp_spec, _temp_params) = mlp_init(
        &mut temp_graph,
        &config.layer_sizes,
        Activation::Tanh,
        Activation::None,
        config.seed,
    );

    // Extract initial params
    let mut params: Vec<Vec<f64>> = mlp_spec.layers.iter().enumerate().flat_map(|(li, _)| {
        vec![
            temp_graph.tensor(_temp_params[li * 2]).to_vec(),
            temp_graph.tensor(_temp_params[li * 2 + 1]).to_vec(),
        ]
    }).collect();

    let total_params: usize = params.iter().map(|p| p.len()).sum();
    let mut adam = AdamState::new(total_params, config.lr);
    let mut history = Vec::with_capacity(config.epochs);

    for epoch in 0..config.epochs {
        // Cosine LR annealing
        let lr_min = config.lr * 0.01;
        let cos_decay = 0.5 * (1.0 + (std::f64::consts::PI * epoch as f64 / config.epochs as f64).cos());
        adam.lr = lr_min + (config.lr - lr_min) * cos_decay;

        // Build graph fresh each epoch (simpler than reforward for 2D input)
        let mut graph = crate::GradGraph::new();
        let mut p_indices = Vec::new();
        for (li, layer) in mlp_spec.layers.iter().enumerate() {
            let w = Tensor::from_vec_unchecked(
                params[li * 2].clone(),
                &[layer.out_features, layer.in_features],
            );
            let b = Tensor::from_vec_unchecked(
                params[li * 2 + 1].clone(),
                &[layer.out_features],
            );
            p_indices.push(graph.parameter(w));
            p_indices.push(graph.parameter(b));
        }
        let mlp = Mlp {
            layers: mlp_spec.layers.iter().enumerate().map(|(li, l)| DenseLayer {
                weight_idx: p_indices[li * 2],
                bias_idx: p_indices[li * 2 + 1],
                activation: l.activation,
                in_features: l.in_features,
                out_features: l.out_features,
            }).collect(),
        };

        // === IC Loss ===
        let mut ic_loss_nodes = Vec::new();
        for (j, &x) in ic_x.iter().enumerate() {
            let inp = graph.input(Tensor::from_vec_unchecked(vec![x, 0.0], &[1, 2]));
            let pred = mlp_forward(&mut graph, &mlp, inp);
            let target = graph.input(Tensor::from_vec_unchecked(vec![ic_u[j]], &[1, 1]));
            let diff = graph.sub(pred, target);
            let sq = graph.mul(diff, diff);
            ic_loss_nodes.push(sq);
        }
        let mut ic_sum = ic_loss_nodes[0];
        for &n in &ic_loss_nodes[1..] { ic_sum = graph.add(ic_sum, n); }
        let ic_n = graph.input(Tensor::from_vec_unchecked(vec![config.n_ic as f64], &[1]));
        let ic_loss = graph.div(ic_sum, ic_n);

        // === BC Loss ===
        let n_bc_total = config.n_bc * 2;
        let mut bc_loss_nodes = Vec::new();
        for j in 0..n_bc_total {
            let x = bc_pts[j * 2];
            let t = bc_pts[j * 2 + 1];
            let inp = graph.input(Tensor::from_vec_unchecked(vec![x, t], &[1, 2]));
            let pred = mlp_forward(&mut graph, &mlp, inp);
            let target = graph.input(Tensor::from_vec_unchecked(vec![bc_vals[j]], &[1, 1]));
            let diff = graph.sub(pred, target);
            let sq = graph.mul(diff, diff);
            bc_loss_nodes.push(sq);
        }
        let mut bc_sum = bc_loss_nodes[0];
        for &n in &bc_loss_nodes[1..] { bc_sum = graph.add(bc_sum, n); }
        let bc_n = graph.input(Tensor::from_vec_unchecked(vec![n_bc_total as f64], &[1]));
        let bc_loss = graph.div(bc_sum, bc_n);

        // === Physics Loss (PDE residual) ===
        let n_colloc = config.n_collocation;
        let mut phys_loss_nodes = Vec::new();
        for i in 0..n_colloc {
            let x = interior_pts[i * 2];
            let t = interior_pts[i * 2 + 1];

            // u(x, t)
            let inp_c = graph.input(Tensor::from_vec_unchecked(vec![x, t], &[1, 2]));
            let u_c = mlp_forward(&mut graph, &mlp, inp_c);

            // u(x+ε, t) and u(x-ε, t) for u_x, u_xx
            let inp_xp = graph.input(Tensor::from_vec_unchecked(vec![x + eps, t], &[1, 2]));
            let u_xp = mlp_forward(&mut graph, &mlp, inp_xp);
            let inp_xm = graph.input(Tensor::from_vec_unchecked(vec![x - eps, t], &[1, 2]));
            let u_xm = mlp_forward(&mut graph, &mlp, inp_xm);

            // u(x, t+ε) and u(x, t-ε) for u_t
            let inp_tp = graph.input(Tensor::from_vec_unchecked(vec![x, t + eps], &[1, 2]));
            let u_tp = mlp_forward(&mut graph, &mlp, inp_tp);
            let inp_tm = graph.input(Tensor::from_vec_unchecked(vec![x, t - eps], &[1, 2]));
            let u_tm = mlp_forward(&mut graph, &mlp, inp_tm);

            // u_t ≈ (u(x,t+ε) - u(x,t-ε)) / (2ε)
            let u_t_num = graph.sub(u_tp, u_tm);
            let u_t = graph.scalar_mul(u_t_num, 1.0 / (2.0 * eps));

            // u_x ≈ (u(x+ε,t) - u(x-ε,t)) / (2ε)
            let u_x_num = graph.sub(u_xp, u_xm);
            let u_x = graph.scalar_mul(u_x_num, 1.0 / (2.0 * eps));

            // u_xx ≈ (u(x+ε,t) - 2u(x,t) + u(x-ε,t)) / ε²
            let two_u = graph.scalar_mul(u_c, 2.0);
            let sum_xpm = graph.add(u_xp, u_xm);
            let u_xx_num = graph.sub(sum_xpm, two_u);
            let u_xx = graph.scalar_mul(u_xx_num, 1.0 / (eps * eps));

            // residual = u_t + u * u_x - ν * u_xx
            let u_ux = graph.mul(u_c, u_x);
            let nu_uxx = graph.scalar_mul(u_xx, nu);
            let r1 = graph.add(u_t, u_ux);
            let residual = graph.sub(r1, nu_uxx);
            let r_sq = graph.mul(residual, residual);
            phys_loss_nodes.push(r_sq);
        }
        let mut phys_sum = phys_loss_nodes[0];
        for &n in &phys_loss_nodes[1..] { phys_sum = graph.add(phys_sum, n); }
        let phys_n = graph.input(Tensor::from_vec_unchecked(vec![n_colloc as f64], &[1]));
        let phys_loss = graph.div(phys_sum, phys_n);

        // === Total Loss ===
        let pw = graph.input(Tensor::from_vec_unchecked(vec![config.physics_weight], &[1]));
        let bw_val = config.boundary_weight * ((epoch as f64 + 1.0) / (config.epochs as f64 * 0.2)).min(1.0);
        let bw = graph.input(Tensor::from_vec_unchecked(vec![bw_val], &[1]));
        let weighted_phys = graph.mul(pw, phys_loss);
        let ic_bc_sum = graph.add(ic_loss, bc_loss);
        let weighted_bnd = graph.mul(bw, ic_bc_sum);
        let total_loss_node = graph.add(weighted_phys, weighted_bnd);

        let total_loss_val = graph.value(total_loss_node);
        let data_loss_val = graph.value(ic_loss) + graph.value(bc_loss);
        let phys_loss_val = graph.value(phys_loss);

        // Backward
        graph.zero_grad();
        graph.backward(total_loss_node);

        // Collect gradients + Adam update
        let mut flat_params = Vec::with_capacity(total_params);
        let mut flat_grads = Vec::with_capacity(total_params);
        for (pi, &idx) in p_indices.iter().enumerate() {
            let grad = graph.grad(idx).unwrap_or_else(|| {
                Tensor::zeros(&graph.tensor(idx).shape().to_vec())
            });
            flat_params.extend_from_slice(&params[pi]);
            flat_grads.extend_from_slice(&grad.to_vec());
        }

        let mut grad_norm_acc = KahanAccumulatorF64::new();
        for &g in &flat_grads { grad_norm_acc.add(g * g); }
        let grad_norm = grad_norm_acc.finalize().sqrt();

        adam_step(&mut flat_params, &flat_grads, &mut adam);

        // Unflatten
        let mut offset = 0;
        for pi in 0..params.len() {
            let n = params[pi].len();
            params[pi].copy_from_slice(&flat_params[offset..offset + n]);
            offset += n;
        }

        history.push(TrainLog {
            epoch,
            total_loss: total_loss_val,
            data_loss: data_loss_val,
            physics_loss: phys_loss_val,
            boundary_loss: graph.value(ic_bc_sum),
            grad_norm,
        });
    }

    // --- Evaluate ---
    let final_params: Vec<f64> = params.iter().flat_map(|p| p.iter().copied()).collect();

    // Check IC accuracy
    let mut l2_acc = KahanAccumulatorF64::new();
    let mut max_err = 0.0f64;
    let n_eval = 50;
    for i in 0..n_eval {
        let x = x_range.0 + (i as f64 + 0.5) / n_eval as f64 * (x_range.1 - x_range.0);
        let u_exact = -(std::f64::consts::PI * x).sin();
        let mut eval_graph = crate::GradGraph::new();
        let mut ep = Vec::new();
        for (li, layer) in mlp_spec.layers.iter().enumerate() {
            let w = Tensor::from_vec_unchecked(params[li*2].clone(), &[layer.out_features, layer.in_features]);
            let b = Tensor::from_vec_unchecked(params[li*2+1].clone(), &[layer.out_features]);
            ep.push(eval_graph.parameter(w));
            ep.push(eval_graph.parameter(b));
        }
        let em = Mlp {
            layers: mlp_spec.layers.iter().enumerate().map(|(li, l)| DenseLayer {
                weight_idx: ep[li*2], bias_idx: ep[li*2+1],
                activation: l.activation, in_features: l.in_features, out_features: l.out_features,
            }).collect(),
        };
        let inp = eval_graph.input(Tensor::from_vec_unchecked(vec![x, 0.0], &[1, 2]));
        let pred = mlp_forward(&mut eval_graph, &em, inp);
        let err = (eval_graph.value(pred) - u_exact).abs();
        l2_acc.add(err * err);
        if err > max_err { max_err = err; }
    }

    let l2_error = (l2_acc.finalize() / n_eval as f64).sqrt();

    let mean_residual = history.last().map_or(0.0, |h| h.physics_loss);
    PinnResult {
        final_params,
        history,
        l2_error: Some(l2_error),
        max_error: Some(max_err),
        mean_residual,
    }
}

// ---------------------------------------------------------------------------
// Problem D: PINN — 2D Poisson Equation
// ---------------------------------------------------------------------------
// PDE:  u_xx + u_yy = f(x,y)  on (x,y) ∈ [0,1]²
// f(x,y) = -2π² sin(πx) sin(πy)
// BCs:  u = 0 on all boundaries
// Exact: u(x,y) = sin(πx) sin(πy)

/// Configuration for the 2D Poisson PINN.
#[derive(Debug, Clone)]
pub struct PoissonConfig {
    /// MLP layer sizes, e.g. [2, 20, 20, 1].
    pub layer_sizes: Vec<usize>,
    pub epochs: usize,
    pub lr: f64,
    pub physics_weight: f64,
    pub boundary_weight: f64,
    pub seed: u64,
    pub n_collocation: usize,
    pub n_boundary: usize,
    pub fd_eps: f64,
}

impl Default for PoissonConfig {
    fn default() -> Self {
        Self {
            layer_sizes: vec![2, 20, 20, 1],
            epochs: 500,
            lr: 1e-3,
            physics_weight: 1.0,
            boundary_weight: 10.0,
            seed: 42,
            n_collocation: 64,
            n_boundary: 40,
            fd_eps: 1e-4,
        }
    }
}

/// Train a PINN for the 2D Poisson equation.
pub fn pinn_poisson_2d_train(config: &PoissonConfig) -> PinnResult {
    let domain = PinnDomain::Rectangle2D {
        x_range: (0.0, 1.0),
        y_range: (0.0, 1.0),
    };
    let eps = config.fd_eps;
    let pi = std::f64::consts::PI;

    let interior_pts = domain.sample_interior(config.n_collocation, config.seed);
    let boundary_pts = domain.sample_boundary(config.n_boundary, config.seed);
    let n_bnd = boundary_pts.len() / 2;

    // Build network
    let mut temp_graph = crate::GradGraph::new();
    let (mlp_spec, _tp) = mlp_init(
        &mut temp_graph, &config.layer_sizes,
        Activation::Tanh, Activation::None, config.seed,
    );

    let mut params: Vec<Vec<f64>> = mlp_spec.layers.iter().enumerate().flat_map(|(li, _)| {
        vec![
            temp_graph.tensor(_tp[li * 2]).to_vec(),
            temp_graph.tensor(_tp[li * 2 + 1]).to_vec(),
        ]
    }).collect();

    let total_params: usize = params.iter().map(|p| p.len()).sum();
    let mut adam = AdamState::new(total_params, config.lr);
    let mut history = Vec::with_capacity(config.epochs);

    for epoch in 0..config.epochs {
        let lr_min = config.lr * 0.01;
        let cos_decay = 0.5 * (1.0 + (pi * epoch as f64 / config.epochs as f64).cos());
        adam.lr = lr_min + (config.lr - lr_min) * cos_decay;

        let mut graph = crate::GradGraph::new();
        let mut p_indices = Vec::new();
        for (li, layer) in mlp_spec.layers.iter().enumerate() {
            let w = Tensor::from_vec_unchecked(params[li*2].clone(), &[layer.out_features, layer.in_features]);
            let b = Tensor::from_vec_unchecked(params[li*2+1].clone(), &[layer.out_features]);
            p_indices.push(graph.parameter(w));
            p_indices.push(graph.parameter(b));
        }
        let mlp = Mlp {
            layers: mlp_spec.layers.iter().enumerate().map(|(li, l)| DenseLayer {
                weight_idx: p_indices[li*2], bias_idx: p_indices[li*2+1],
                activation: l.activation, in_features: l.in_features, out_features: l.out_features,
            }).collect(),
        };

        // === Physics Loss: u_xx + u_yy - f(x,y) ===
        let n_c = config.n_collocation;
        let mut phys_nodes = Vec::new();
        for i in 0..n_c {
            let x = interior_pts[i * 2];
            let y = interior_pts[i * 2 + 1];

            let inp_c = graph.input(Tensor::from_vec_unchecked(vec![x, y], &[1, 2]));
            let u_c = mlp_forward(&mut graph, &mlp, inp_c);

            // u_xx via FD in x
            let inp_xp = graph.input(Tensor::from_vec_unchecked(vec![x + eps, y], &[1, 2]));
            let u_xp = mlp_forward(&mut graph, &mlp, inp_xp);
            let inp_xm = graph.input(Tensor::from_vec_unchecked(vec![x - eps, y], &[1, 2]));
            let u_xm = mlp_forward(&mut graph, &mlp, inp_xm);

            let two_u = graph.scalar_mul(u_c, 2.0);
            let sum_x = graph.add(u_xp, u_xm);
            let diff_x = graph.sub(sum_x, two_u);
            let u_xx = graph.scalar_mul(diff_x, 1.0 / (eps * eps));

            // u_yy via FD in y
            let inp_yp = graph.input(Tensor::from_vec_unchecked(vec![x, y + eps], &[1, 2]));
            let u_yp = mlp_forward(&mut graph, &mlp, inp_yp);
            let inp_ym = graph.input(Tensor::from_vec_unchecked(vec![x, y - eps], &[1, 2]));
            let u_ym = mlp_forward(&mut graph, &mlp, inp_ym);

            let sum_y = graph.add(u_yp, u_ym);
            let diff_y = graph.sub(sum_y, two_u);
            let u_yy = graph.scalar_mul(diff_y, 1.0 / (eps * eps));

            // f(x,y) = -2π² sin(πx) sin(πy)
            let f_val = -2.0 * pi * pi * (pi * x).sin() * (pi * y).sin();
            let f_node = graph.input(Tensor::from_vec_unchecked(vec![f_val], &[1, 1]));

            // residual = u_xx + u_yy - f
            let laplacian = graph.add(u_xx, u_yy);
            let residual = graph.sub(laplacian, f_node);
            let r_sq = graph.mul(residual, residual);
            phys_nodes.push(r_sq);
        }

        let mut phys_sum = phys_nodes[0];
        for &n in &phys_nodes[1..] { phys_sum = graph.add(phys_sum, n); }
        let phys_n = graph.input(Tensor::from_vec_unchecked(vec![n_c as f64], &[1]));
        let phys_loss = graph.div(phys_sum, phys_n);

        // === Boundary Loss: u = 0 on all edges ===
        let mut bnd_nodes = Vec::new();
        for j in 0..n_bnd {
            let bx = boundary_pts[j * 2];
            let by = boundary_pts[j * 2 + 1];
            let inp = graph.input(Tensor::from_vec_unchecked(vec![bx, by], &[1, 2]));
            let pred = mlp_forward(&mut graph, &mlp, inp);
            let sq = graph.mul(pred, pred); // target is 0
            bnd_nodes.push(sq);
        }
        let mut bnd_sum = bnd_nodes[0];
        for &n in &bnd_nodes[1..] { bnd_sum = graph.add(bnd_sum, n); }
        let bnd_n = graph.input(Tensor::from_vec_unchecked(vec![n_bnd as f64], &[1]));
        let bnd_loss = graph.div(bnd_sum, bnd_n);

        // === Total Loss ===
        let pw = graph.input(Tensor::from_vec_unchecked(vec![config.physics_weight], &[1]));
        let bw_val = config.boundary_weight * ((epoch as f64 + 1.0) / (config.epochs as f64 * 0.2)).min(1.0);
        let bw = graph.input(Tensor::from_vec_unchecked(vec![bw_val], &[1]));
        let wp = graph.mul(pw, phys_loss);
        let wb = graph.mul(bw, bnd_loss);
        let total_loss_node = graph.add(wp, wb);

        let total_val = graph.value(total_loss_node);
        let phys_val = graph.value(phys_loss);
        let bnd_val = graph.value(bnd_loss);

        graph.zero_grad();
        graph.backward(total_loss_node);

        let mut flat_params = Vec::with_capacity(total_params);
        let mut flat_grads = Vec::with_capacity(total_params);
        for (pi, &idx) in p_indices.iter().enumerate() {
            let grad = graph.grad(idx).unwrap_or_else(|| {
                Tensor::zeros(&graph.tensor(idx).shape().to_vec())
            });
            flat_params.extend_from_slice(&params[pi]);
            flat_grads.extend_from_slice(&grad.to_vec());
        }

        let mut gn_acc = KahanAccumulatorF64::new();
        for &g in &flat_grads { gn_acc.add(g * g); }

        adam_step(&mut flat_params, &flat_grads, &mut adam);

        let mut offset = 0;
        for pi in 0..params.len() {
            let n = params[pi].len();
            params[pi].copy_from_slice(&flat_params[offset..offset + n]);
            offset += n;
        }

        history.push(TrainLog {
            epoch,
            total_loss: total_val,
            data_loss: 0.0,
            physics_loss: phys_val,
            boundary_loss: bnd_val,
            grad_norm: gn_acc.finalize().sqrt(),
        });
    }

    // Evaluate accuracy
    let n_eval = 25; // 25×25 grid
    let mut l2_acc = KahanAccumulatorF64::new();
    let mut max_err = 0.0f64;

    for ix in 0..n_eval {
        for iy in 0..n_eval {
            let x = (ix as f64 + 0.5) / n_eval as f64;
            let y = (iy as f64 + 0.5) / n_eval as f64;
            let u_exact = (pi * x).sin() * (pi * y).sin();

            let mut eg = crate::GradGraph::new();
            let mut ep = Vec::new();
            for (li, layer) in mlp_spec.layers.iter().enumerate() {
                let w = Tensor::from_vec_unchecked(params[li*2].clone(), &[layer.out_features, layer.in_features]);
                let b = Tensor::from_vec_unchecked(params[li*2+1].clone(), &[layer.out_features]);
                ep.push(eg.parameter(w));
                ep.push(eg.parameter(b));
            }
            let em = Mlp {
                layers: mlp_spec.layers.iter().enumerate().map(|(li, l)| DenseLayer {
                    weight_idx: ep[li*2], bias_idx: ep[li*2+1],
                    activation: l.activation, in_features: l.in_features, out_features: l.out_features,
                }).collect(),
            };
            let inp = eg.input(Tensor::from_vec_unchecked(vec![x, y], &[1, 2]));
            let pred = mlp_forward(&mut eg, &em, inp);
            let err = (eg.value(pred) - u_exact).abs();
            l2_acc.add(err * err);
            if err > max_err { max_err = err; }
        }
    }

    let total_eval = (n_eval * n_eval) as f64;
    let mean_residual = history.last().map_or(0.0, |h| h.physics_loss);
    PinnResult {
        final_params: params.iter().flat_map(|p| p.iter().copied()).collect(),
        history,
        l2_error: Some((l2_acc.finalize() / total_eval).sqrt()),
        max_error: Some(max_err),
        mean_residual,
    }
}

// ---------------------------------------------------------------------------
// Problem E: PINN — 1D Heat Equation (Neural Network version)
// ---------------------------------------------------------------------------
// PDE:  u_t = α · u_xx   on x ∈ [0,1], t ∈ [0,1]
// IC:   u(x, 0) = sin(πx)
// BCs:  u(0, t) = 0, u(1, t) = 0
// Exact: u(x, t) = exp(-α π² t) sin(πx)
// α = 0.01

/// Configuration for the 1D heat equation PINN.
#[derive(Debug, Clone)]
pub struct HeatConfig {
    /// MLP layer sizes, e.g. [2, 20, 20, 1].
    pub layer_sizes: Vec<usize>,
    pub epochs: usize,
    pub lr: f64,
    /// Thermal diffusivity.
    pub alpha: f64,
    pub physics_weight: f64,
    pub boundary_weight: f64,
    pub seed: u64,
    pub n_collocation: usize,
    pub n_ic: usize,
    pub n_bc: usize,
    pub fd_eps: f64,
}

impl Default for HeatConfig {
    fn default() -> Self {
        Self {
            layer_sizes: vec![2, 20, 20, 1],
            epochs: 500,
            lr: 1e-3,
            alpha: 0.01,
            physics_weight: 1.0,
            boundary_weight: 10.0,
            seed: 42,
            n_collocation: 64,
            n_ic: 50,
            n_bc: 25,
            fd_eps: 1e-4,
        }
    }
}

/// Train a PINN for the 1D heat equation u_t = α·u_xx.
pub fn pinn_heat_1d_nn_train(config: &HeatConfig) -> PinnResult {
    let x_range = (0.0, 1.0);
    let t_range = (0.0, 1.0);
    let eps = config.fd_eps;
    let alpha = config.alpha;
    let pi = std::f64::consts::PI;

    let domain = PinnDomain::SpaceTime1D { x_range, t_range };
    let interior_pts = domain.sample_interior(config.n_collocation, config.seed);

    // IC: u(x, 0) = sin(πx)
    let mut ic_x = Vec::with_capacity(config.n_ic);
    let mut ic_u = Vec::with_capacity(config.n_ic);
    for i in 0..config.n_ic {
        let x = (i as f64 + 0.5) / config.n_ic as f64;
        ic_x.push(x);
        ic_u.push((pi * x).sin());
    }

    // BC: u(0, t) = 0, u(1, t) = 0
    let mut bc_pts = Vec::with_capacity(config.n_bc * 2 * 2);
    for i in 0..config.n_bc {
        let t = (i as f64 + 0.5) / config.n_bc as f64;
        bc_pts.push(0.0); bc_pts.push(t); // left
        bc_pts.push(1.0); bc_pts.push(t); // right
    }

    // Build network
    let mut temp_graph = crate::GradGraph::new();
    let (mlp_spec, _tp) = mlp_init(
        &mut temp_graph, &config.layer_sizes,
        Activation::Tanh, Activation::None, config.seed,
    );

    let mut params: Vec<Vec<f64>> = mlp_spec.layers.iter().enumerate().flat_map(|(li, _)| {
        vec![
            temp_graph.tensor(_tp[li * 2]).to_vec(),
            temp_graph.tensor(_tp[li * 2 + 1]).to_vec(),
        ]
    }).collect();

    let total_params: usize = params.iter().map(|p| p.len()).sum();
    let mut adam = AdamState::new(total_params, config.lr);
    let mut history = Vec::with_capacity(config.epochs);

    for epoch in 0..config.epochs {
        let lr_min = config.lr * 0.01;
        let cos_decay = 0.5 * (1.0 + (pi * epoch as f64 / config.epochs as f64).cos());
        adam.lr = lr_min + (config.lr - lr_min) * cos_decay;

        let mut graph = crate::GradGraph::new();
        let mut p_indices = Vec::new();
        for (li, layer) in mlp_spec.layers.iter().enumerate() {
            let w = Tensor::from_vec_unchecked(params[li*2].clone(), &[layer.out_features, layer.in_features]);
            let b = Tensor::from_vec_unchecked(params[li*2+1].clone(), &[layer.out_features]);
            p_indices.push(graph.parameter(w));
            p_indices.push(graph.parameter(b));
        }
        let mlp = Mlp {
            layers: mlp_spec.layers.iter().enumerate().map(|(li, l)| DenseLayer {
                weight_idx: p_indices[li*2], bias_idx: p_indices[li*2+1],
                activation: l.activation, in_features: l.in_features, out_features: l.out_features,
            }).collect(),
        };

        // === IC Loss ===
        let mut ic_nodes = Vec::new();
        for (j, &x) in ic_x.iter().enumerate() {
            let inp = graph.input(Tensor::from_vec_unchecked(vec![x, 0.0], &[1, 2]));
            let pred = mlp_forward(&mut graph, &mlp, inp);
            let target = graph.input(Tensor::from_vec_unchecked(vec![ic_u[j]], &[1, 1]));
            let diff = graph.sub(pred, target);
            ic_nodes.push(graph.mul(diff, diff));
        }
        let mut ic_sum = ic_nodes[0];
        for &n in &ic_nodes[1..] { ic_sum = graph.add(ic_sum, n); }
        let ic_n = graph.input(Tensor::from_vec_unchecked(vec![config.n_ic as f64], &[1]));
        let ic_loss = graph.div(ic_sum, ic_n);

        // === BC Loss ===
        let n_bc_total = config.n_bc * 2;
        let mut bc_nodes = Vec::new();
        for j in 0..n_bc_total {
            let bx = bc_pts[j * 2];
            let bt = bc_pts[j * 2 + 1];
            let inp = graph.input(Tensor::from_vec_unchecked(vec![bx, bt], &[1, 2]));
            let pred = mlp_forward(&mut graph, &mlp, inp);
            bc_nodes.push(graph.mul(pred, pred)); // target=0
        }
        let mut bc_sum = bc_nodes[0];
        for &n in &bc_nodes[1..] { bc_sum = graph.add(bc_sum, n); }
        let bc_n = graph.input(Tensor::from_vec_unchecked(vec![n_bc_total as f64], &[1]));
        let bc_loss = graph.div(bc_sum, bc_n);

        // === Physics Loss: u_t - α·u_xx = 0 ===
        let n_c = config.n_collocation;
        let mut phys_nodes = Vec::new();
        for i in 0..n_c {
            let x = interior_pts[i * 2];
            let t = interior_pts[i * 2 + 1];

            let inp_c = graph.input(Tensor::from_vec_unchecked(vec![x, t], &[1, 2]));
            let u_c = mlp_forward(&mut graph, &mlp, inp_c);

            // u_t via FD
            let inp_tp = graph.input(Tensor::from_vec_unchecked(vec![x, t + eps], &[1, 2]));
            let u_tp = mlp_forward(&mut graph, &mlp, inp_tp);
            let inp_tm = graph.input(Tensor::from_vec_unchecked(vec![x, t - eps], &[1, 2]));
            let u_tm = mlp_forward(&mut graph, &mlp, inp_tm);
            let u_t_diff = graph.sub(u_tp, u_tm);
            let u_t = graph.scalar_mul(u_t_diff, 1.0 / (2.0 * eps));

            // u_xx via FD
            let inp_xp = graph.input(Tensor::from_vec_unchecked(vec![x + eps, t], &[1, 2]));
            let u_xp = mlp_forward(&mut graph, &mlp, inp_xp);
            let inp_xm = graph.input(Tensor::from_vec_unchecked(vec![x - eps, t], &[1, 2]));
            let u_xm = mlp_forward(&mut graph, &mlp, inp_xm);
            let two_u = graph.scalar_mul(u_c, 2.0);
            let sum_xpm = graph.add(u_xp, u_xm);
            let diff_xpm = graph.sub(sum_xpm, two_u);
            let u_xx = graph.scalar_mul(diff_xpm, 1.0 / (eps * eps));

            // residual = u_t - α·u_xx
            let a_uxx = graph.scalar_mul(u_xx, alpha);
            let residual = graph.sub(u_t, a_uxx);
            phys_nodes.push(graph.mul(residual, residual));
        }
        let mut phys_sum = phys_nodes[0];
        for &n in &phys_nodes[1..] { phys_sum = graph.add(phys_sum, n); }
        let phys_n = graph.input(Tensor::from_vec_unchecked(vec![n_c as f64], &[1]));
        let phys_loss = graph.div(phys_sum, phys_n);

        // === Total ===
        let pw = graph.input(Tensor::from_vec_unchecked(vec![config.physics_weight], &[1]));
        let bw_val = config.boundary_weight * ((epoch as f64 + 1.0) / (config.epochs as f64 * 0.2)).min(1.0);
        let bw = graph.input(Tensor::from_vec_unchecked(vec![bw_val], &[1]));
        let wp = graph.mul(pw, phys_loss);
        let ic_bc = graph.add(ic_loss, bc_loss);
        let wb = graph.mul(bw, ic_bc);
        let total_loss_node = graph.add(wp, wb);

        let total_val = graph.value(total_loss_node);
        let phys_val = graph.value(phys_loss);

        graph.zero_grad();
        graph.backward(total_loss_node);

        let mut flat_params = Vec::with_capacity(total_params);
        let mut flat_grads = Vec::with_capacity(total_params);
        for (pi, &idx) in p_indices.iter().enumerate() {
            let grad = graph.grad(idx).unwrap_or_else(|| {
                Tensor::zeros(&graph.tensor(idx).shape().to_vec())
            });
            flat_params.extend_from_slice(&params[pi]);
            flat_grads.extend_from_slice(&grad.to_vec());
        }

        let mut gn_acc = KahanAccumulatorF64::new();
        for &g in &flat_grads { gn_acc.add(g * g); }

        adam_step(&mut flat_params, &flat_grads, &mut adam);

        let mut offset = 0;
        for pi in 0..params.len() {
            let n = params[pi].len();
            params[pi].copy_from_slice(&flat_params[offset..offset + n]);
            offset += n;
        }

        history.push(TrainLog {
            epoch,
            total_loss: total_val,
            data_loss: graph.value(ic_loss) + graph.value(bc_loss),
            physics_loss: phys_val,
            boundary_loss: graph.value(ic_bc),
            grad_norm: gn_acc.finalize().sqrt(),
        });
    }

    // Evaluate vs analytical at t=0.5
    let n_eval = 50;
    let t_eval = 0.5;
    let mut l2_acc = KahanAccumulatorF64::new();
    let mut max_err = 0.0f64;
    let mut res_acc = KahanAccumulatorF64::new();

    for i in 0..n_eval {
        let x = (i as f64 + 0.5) / n_eval as f64;
        let u_exact = (-alpha * pi * pi * t_eval).exp() * (pi * x).sin();

        let mut eg = crate::GradGraph::new();
        let mut ep = Vec::new();
        for (li, layer) in mlp_spec.layers.iter().enumerate() {
            let w = Tensor::from_vec_unchecked(params[li*2].clone(), &[layer.out_features, layer.in_features]);
            let b = Tensor::from_vec_unchecked(params[li*2+1].clone(), &[layer.out_features]);
            ep.push(eg.parameter(w));
            ep.push(eg.parameter(b));
        }
        let em = Mlp {
            layers: mlp_spec.layers.iter().enumerate().map(|(li, l)| DenseLayer {
                weight_idx: ep[li*2], bias_idx: ep[li*2+1],
                activation: l.activation, in_features: l.in_features, out_features: l.out_features,
            }).collect(),
        };
        let inp = eg.input(Tensor::from_vec_unchecked(vec![x, t_eval], &[1, 2]));
        let pred = mlp_forward(&mut eg, &em, inp);
        let err = (eg.value(pred) - u_exact).abs();
        l2_acc.add(err * err);
        if err > max_err { max_err = err; }

        // Residual at this point
        let inp_tp = eg.input(Tensor::from_vec_unchecked(vec![x, t_eval + eps], &[1, 2]));
        let u_tp = mlp_forward(&mut eg, &em, inp_tp);
        let inp_tm = eg.input(Tensor::from_vec_unchecked(vec![x, t_eval - eps], &[1, 2]));
        let u_tm = mlp_forward(&mut eg, &em, inp_tm);
        let u_t_val = (eg.value(u_tp) - eg.value(u_tm)) / (2.0 * eps);

        let inp_xp = eg.input(Tensor::from_vec_unchecked(vec![x + eps, t_eval], &[1, 2]));
        let u_xp = mlp_forward(&mut eg, &em, inp_xp);
        let inp_xm = eg.input(Tensor::from_vec_unchecked(vec![x - eps, t_eval], &[1, 2]));
        let u_xm = mlp_forward(&mut eg, &em, inp_xm);
        let u_val = eg.value(pred);
        let u_xx_val = (eg.value(u_xp) - 2.0 * u_val + eg.value(u_xm)) / (eps * eps);
        let res = (u_t_val - alpha * u_xx_val).abs();
        res_acc.add(res * res);
    }

    PinnResult {
        final_params: params.iter().flat_map(|p| p.iter().copied()).collect(),
        history,
        l2_error: Some((l2_acc.finalize() / n_eval as f64).sqrt()),
        max_error: Some(max_err),
        mean_residual: (res_acc.finalize() / n_eval as f64).sqrt(),
    }
}

// ---------------------------------------------------------------------------
// Problem F: PINN — 1D Wave Equation
// ---------------------------------------------------------------------------
// PDE: u_tt = c² · u_xx on x∈[0,1], t∈[0,1]
// IC: u(x,0) = sin(πx), u_t(x,0) = 0
// BC: u(0,t) = u(1,t) = 0
// Exact: u(x,t) = sin(πx)·cos(cπt)

/// Configuration for the 1D wave equation PINN.
#[derive(Debug, Clone)]
pub struct WaveConfig {
    pub layer_sizes: Vec<usize>,
    pub epochs: usize,
    pub lr: f64,
    pub c: f64,
    pub physics_weight: f64,
    pub boundary_weight: f64,
    pub seed: u64,
    pub n_collocation: usize,
    pub n_ic: usize,
    pub n_bc: usize,
    pub fd_eps: f64,
}

impl Default for WaveConfig {
    fn default() -> Self {
        Self {
            layer_sizes: vec![2, 20, 20, 20, 1],
            epochs: 500, lr: 1e-3, c: 1.0,
            physics_weight: 1.0, boundary_weight: 10.0,
            seed: 42, n_collocation: 64, n_ic: 50, n_bc: 25, fd_eps: 1e-4,
        }
    }
}

/// Train a PINN for the 1D wave equation u_tt = c²·u_xx.
pub fn pinn_wave_train(config: &WaveConfig) -> PinnResult {
    let x_range = (0.0, 1.0);
    let t_range = (0.0, 1.0);
    let eps = config.fd_eps;
    let c2 = config.c * config.c;

    let domain = PinnDomain::SpaceTime1D { x_range, t_range };
    let interior_pts = domain.sample_interior(config.n_collocation, config.seed);

    // IC points: u(x,0) = sin(πx)
    let mut ic_x = Vec::with_capacity(config.n_ic);
    let mut ic_u = Vec::with_capacity(config.n_ic);
    for i in 0..config.n_ic {
        let x = (i as f64 + 0.5) / config.n_ic as f64;
        ic_x.push(x);
        ic_u.push((std::f64::consts::PI * x).sin());
    }

    // BC: u(0,t)=0, u(1,t)=0
    let mut bc_pts = Vec::new();
    let mut bc_vals = Vec::new();
    for i in 0..config.n_bc {
        let t = (i as f64 + 0.5) / config.n_bc as f64;
        bc_pts.push(0.0); bc_pts.push(t); bc_vals.push(0.0);
        bc_pts.push(1.0); bc_pts.push(t); bc_vals.push(0.0);
    }

    // Build network
    let mut temp_graph = crate::GradGraph::new();
    let (mlp_spec, _tp) = mlp_init(&mut temp_graph, &config.layer_sizes, Activation::Tanh, Activation::None, config.seed);
    let mut params: Vec<Vec<f64>> = mlp_spec.layers.iter().enumerate().flat_map(|(li, _)| {
        vec![temp_graph.tensor(_tp[li * 2]).to_vec(), temp_graph.tensor(_tp[li * 2 + 1]).to_vec()]
    }).collect();
    let total_params: usize = params.iter().map(|p| p.len()).sum();
    let mut adam = AdamState::new(total_params, config.lr);
    let mut history = Vec::with_capacity(config.epochs);

    for epoch in 0..config.epochs {
        let lr_min = config.lr * 0.01;
        let cos_decay = 0.5 * (1.0 + (std::f64::consts::PI * epoch as f64 / config.epochs as f64).cos());
        adam.lr = lr_min + (config.lr - lr_min) * cos_decay;

        let mut graph = crate::GradGraph::new();
        let mut p_indices = Vec::new();
        for (li, layer) in mlp_spec.layers.iter().enumerate() {
            let w = Tensor::from_vec_unchecked(params[li * 2].clone(), &[layer.out_features, layer.in_features]);
            let b = Tensor::from_vec_unchecked(params[li * 2 + 1].clone(), &[layer.out_features]);
            p_indices.push(graph.parameter(w));
            p_indices.push(graph.parameter(b));
        }
        let mlp = Mlp {
            layers: mlp_spec.layers.iter().enumerate().map(|(li, l)| DenseLayer {
                weight_idx: p_indices[li * 2], bias_idx: p_indices[li * 2 + 1],
                activation: l.activation, in_features: l.in_features, out_features: l.out_features,
            }).collect(),
        };

        // IC Loss
        let mut ic_loss_nodes = Vec::new();
        for (j, &x) in ic_x.iter().enumerate() {
            let inp = graph.input(Tensor::from_vec_unchecked(vec![x, 0.0], &[1, 2]));
            let pred = mlp_forward(&mut graph, &mlp, inp);
            let target = graph.input(Tensor::from_vec_unchecked(vec![ic_u[j]], &[1, 1]));
            let diff = graph.sub(pred, target);
            let sq = graph.mul(diff, diff);
            ic_loss_nodes.push(sq);
        }
        let mut ic_sum = ic_loss_nodes[0];
        for &n in &ic_loss_nodes[1..] { ic_sum = graph.add(ic_sum, n); }
        let ic_n = graph.input(Tensor::from_vec_unchecked(vec![config.n_ic as f64], &[1]));
        let ic_loss = graph.div(ic_sum, ic_n);

        // BC Loss
        let n_bc_total = config.n_bc * 2;
        let mut bc_loss_nodes = Vec::new();
        for j in 0..n_bc_total {
            let x = bc_pts[j * 2]; let t = bc_pts[j * 2 + 1];
            let inp = graph.input(Tensor::from_vec_unchecked(vec![x, t], &[1, 2]));
            let pred = mlp_forward(&mut graph, &mlp, inp);
            let target = graph.input(Tensor::from_vec_unchecked(vec![bc_vals[j]], &[1, 1]));
            let diff = graph.sub(pred, target);
            let sq = graph.mul(diff, diff);
            bc_loss_nodes.push(sq);
        }
        let mut bc_sum = bc_loss_nodes[0];
        for &n in &bc_loss_nodes[1..] { bc_sum = graph.add(bc_sum, n); }
        let bc_n = graph.input(Tensor::from_vec_unchecked(vec![n_bc_total as f64], &[1]));
        let bc_loss = graph.div(bc_sum, bc_n);

        // Physics Loss: u_tt - c²·u_xx = 0
        let mut phys_loss_nodes = Vec::new();
        for i in 0..config.n_collocation {
            let x = interior_pts[i * 2]; let t = interior_pts[i * 2 + 1];
            let _inp_u_center_ = graph.input(Tensor::from_vec_unchecked(vec![x, t], &[1, 2]));
            let u_center = mlp_forward(&mut graph, &mlp, _inp_u_center_);
            let _inp_u_xp_ = graph.input(Tensor::from_vec_unchecked(vec![x + eps, t], &[1, 2]));
            let u_xp = mlp_forward(&mut graph, &mlp, _inp_u_xp_);
            let _inp_u_xm_ = graph.input(Tensor::from_vec_unchecked(vec![x - eps, t], &[1, 2]));
            let u_xm = mlp_forward(&mut graph, &mlp, _inp_u_xm_);
            let _inp_u_tp_ = graph.input(Tensor::from_vec_unchecked(vec![x, t + eps], &[1, 2]));
            let u_tp = mlp_forward(&mut graph, &mlp, _inp_u_tp_);
            let _inp_u_tm_ = graph.input(Tensor::from_vec_unchecked(vec![x, t - eps], &[1, 2]));
            let u_tm = mlp_forward(&mut graph, &mlp, _inp_u_tm_);
            // u_xx = (u(x+ε) - 2u + u(x-ε)) / ε²
            let sum_x = graph.add(u_xp, u_xm);
            let two_u = graph.scalar_mul(u_center, 2.0);
            let diff_x = graph.sub(sum_x, two_u);
            let u_xx = graph.scalar_mul(diff_x, 1.0 / (eps * eps));
            // u_tt = (u(t+ε) - 2u + u(t-ε)) / ε²
            let sum_t = graph.add(u_tp, u_tm);
            let diff_t = graph.sub(sum_t, two_u);
            let u_tt = graph.scalar_mul(diff_t, 1.0 / (eps * eps));
            // residual = u_tt - c²·u_xx
            let c2_uxx = graph.scalar_mul(u_xx, c2);
            let residual = graph.sub(u_tt, c2_uxx);
            let sq = graph.mul(residual, residual);
            phys_loss_nodes.push(sq);
        }
        let mut phys_sum = phys_loss_nodes[0];
        for &n in &phys_loss_nodes[1..] { phys_sum = graph.add(phys_sum, n); }
        let phys_n = graph.input(Tensor::from_vec_unchecked(vec![config.n_collocation as f64], &[1]));
        let phys_loss = graph.div(phys_sum, phys_n);

        // Total loss
        let bw_frac = (epoch as f64 / (config.epochs as f64 * 0.2)).min(1.0);
        let phys_scaled = graph.scalar_mul(phys_loss, config.physics_weight);
        let bc_scaled = graph.scalar_mul(bc_loss, config.boundary_weight * bw_frac);
        let ic_scaled = graph.scalar_mul(ic_loss, config.boundary_weight * bw_frac);
        let sum1 = graph.add(phys_scaled, bc_scaled);
        let total = graph.add(sum1, ic_scaled);

        graph.backward(total);

        let mut flat_grads = Vec::with_capacity(total_params);
        for idx in &p_indices {
            if let Some(g) = graph.grad(*idx) { flat_grads.extend_from_slice(&g.to_vec()); }
            else { flat_grads.extend(std::iter::repeat(0.0).take(graph.tensor(*idx).to_vec().len())); }
        }
        let mut flat_params: Vec<f64> = params.iter().flat_map(|p| p.clone()).collect();
        let mut gnorm_acc = KahanAccumulatorF64::new();
        for &g in &flat_grads { gnorm_acc.add(g * g); }
        let tl = graph.tensor(total).to_vec()[0];

        history.push(TrainLog {
            epoch,
            total_loss: tl,
            data_loss: graph.tensor(ic_loss).to_vec()[0],
            physics_loss: graph.tensor(phys_loss).to_vec()[0],
            boundary_loss: graph.tensor(bc_loss).to_vec()[0],
            grad_norm: gnorm_acc.finalize().sqrt(),
        });

        adam_step(&mut flat_params, &flat_grads, &mut adam);
        let mut offset = 0;
        for p in params.iter_mut() {
            let _plen_ = p.len();
            p.copy_from_slice(&flat_params[offset..offset + _plen_]);
            offset += p.len();
        }
    }

    // Evaluate: exact = sin(πx)cos(cπt)
    let n_eval = 100;
    let mut l2_acc = KahanAccumulatorF64::new();
    let mut max_err = 0.0f64;
    let mut res_acc = KahanAccumulatorF64::new();
    let mut eval_graph = crate::GradGraph::new();
    let mut ep_indices = Vec::new();
    for (li, layer) in mlp_spec.layers.iter().enumerate() {
        let w = Tensor::from_vec_unchecked(params[li * 2].clone(), &[layer.out_features, layer.in_features]);
        let b = Tensor::from_vec_unchecked(params[li * 2 + 1].clone(), &[layer.out_features]);
        ep_indices.push(eval_graph.parameter(w));
        ep_indices.push(eval_graph.parameter(b));
    }
    let eval_mlp = Mlp {
        layers: mlp_spec.layers.iter().enumerate().map(|(li, l)| DenseLayer {
            weight_idx: ep_indices[li * 2], bias_idx: ep_indices[li * 2 + 1],
            activation: l.activation, in_features: l.in_features, out_features: l.out_features,
        }).collect(),
    };
    for i in 0..n_eval {
        let x = (i as f64 + 0.5) / n_eval as f64;
        let t = 0.5;
        let exact = (std::f64::consts::PI * x).sin() * (config.c * std::f64::consts::PI * t).cos();
        let inp = eval_graph.input(Tensor::from_vec_unchecked(vec![x, t], &[1, 2]));
        let pred_idx = mlp_forward(&mut eval_graph, &eval_mlp, inp);
        let pred = eval_graph.tensor(pred_idx).to_vec()[0];
        let err = (pred - exact).abs();
        l2_acc.add(err * err);
        if err > max_err { max_err = err; }
        res_acc.add(err * err);
    }

    PinnResult {
        history,
        final_params: params.into_iter().flatten().collect(),
        l2_error: Some((l2_acc.finalize() / n_eval as f64).sqrt()),
        max_error: Some(max_err),
        mean_residual: (res_acc.finalize() / n_eval as f64).sqrt(),
    }
}

// ---------------------------------------------------------------------------
// Problem G: PINN — 2D Helmholtz Equation
// ---------------------------------------------------------------------------
// PDE: u_xx + u_yy + k²·u = f(x,y) on [0,1]²
// BC: u = 0 on boundary

/// Configuration for the Helmholtz PINN.
#[derive(Debug, Clone)]
pub struct HelmholtzConfig {
    pub layer_sizes: Vec<usize>,
    pub epochs: usize,
    pub lr: f64,
    pub k: f64,
    pub physics_weight: f64,
    pub boundary_weight: f64,
    pub seed: u64,
    pub n_collocation: usize,
    pub n_bc: usize,
    pub fd_eps: f64,
}

impl Default for HelmholtzConfig {
    fn default() -> Self {
        Self {
            layer_sizes: vec![2, 20, 20, 20, 1],
            epochs: 500, lr: 1e-3, k: 1.0,
            physics_weight: 1.0, boundary_weight: 10.0,
            seed: 42, n_collocation: 64, n_bc: 25, fd_eps: 1e-4,
        }
    }
}

/// Train a PINN for the 2D Helmholtz equation ∇²u + k²u = f.
/// Source term chosen so exact solution is sin(πx)sin(πy).
pub fn pinn_helmholtz_train(config: &HelmholtzConfig) -> PinnResult {
    let domain = PinnDomain::Rectangle2D { x_range: (0.0, 1.0), y_range: (0.0, 1.0) };
    let interior = domain.sample_interior(config.n_collocation, config.seed);
    let boundary = domain.sample_boundary(config.n_bc * 4, config.seed);
    let eps = config.fd_eps;
    let k2 = config.k * config.k;

    // Source term for exact sol sin(πx)sin(πy): f = -(2π² - k²)·sin(πx)sin(πy)
    let source = |x: f64, y: f64| -> f64 {
        -(2.0 * std::f64::consts::PI * std::f64::consts::PI - k2) * (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sin()
    };

    let mut temp_graph = crate::GradGraph::new();
    let (mlp_spec, _tp) = mlp_init(&mut temp_graph, &config.layer_sizes, Activation::Tanh, Activation::None, config.seed);
    let mut params: Vec<Vec<f64>> = mlp_spec.layers.iter().enumerate().flat_map(|(li, _)| {
        vec![temp_graph.tensor(_tp[li * 2]).to_vec(), temp_graph.tensor(_tp[li * 2 + 1]).to_vec()]
    }).collect();
    let total_params: usize = params.iter().map(|p| p.len()).sum();
    let mut adam = AdamState::new(total_params, config.lr);
    let mut history = Vec::with_capacity(config.epochs);

    for epoch in 0..config.epochs {
        let lr_min = config.lr * 0.01;
        adam.lr = lr_min + (config.lr - lr_min) * 0.5 * (1.0 + (std::f64::consts::PI * epoch as f64 / config.epochs as f64).cos());

        let mut graph = crate::GradGraph::new();
        let mut p_indices = Vec::new();
        for (li, layer) in mlp_spec.layers.iter().enumerate() {
            let w = Tensor::from_vec_unchecked(params[li * 2].clone(), &[layer.out_features, layer.in_features]);
            let b = Tensor::from_vec_unchecked(params[li * 2 + 1].clone(), &[layer.out_features]);
            p_indices.push(graph.parameter(w));
            p_indices.push(graph.parameter(b));
        }
        let mlp = Mlp {
            layers: mlp_spec.layers.iter().enumerate().map(|(li, l)| DenseLayer {
                weight_idx: p_indices[li * 2], bias_idx: p_indices[li * 2 + 1],
                activation: l.activation, in_features: l.in_features, out_features: l.out_features,
            }).collect(),
        };

        // Physics: u_xx + u_yy + k²u - f = 0
        let mut phys_nodes = Vec::new();
        for i in 0..config.n_collocation {
            let x = interior[i * 2]; let y = interior[i * 2 + 1];
            let _inp_u_c_ = graph.input(Tensor::from_vec_unchecked(vec![x, y], &[1, 2]));
            let u_c = mlp_forward(&mut graph, &mlp, _inp_u_c_);
            let _inp_u_xp_ = graph.input(Tensor::from_vec_unchecked(vec![x + eps, y], &[1, 2]));
            let u_xp = mlp_forward(&mut graph, &mlp, _inp_u_xp_);
            let _inp_u_xm_ = graph.input(Tensor::from_vec_unchecked(vec![x - eps, y], &[1, 2]));
            let u_xm = mlp_forward(&mut graph, &mlp, _inp_u_xm_);
            let _inp_u_yp_ = graph.input(Tensor::from_vec_unchecked(vec![x, y + eps], &[1, 2]));
            let u_yp = mlp_forward(&mut graph, &mlp, _inp_u_yp_);
            let _inp_u_ym_ = graph.input(Tensor::from_vec_unchecked(vec![x, y - eps], &[1, 2]));
            let u_ym = mlp_forward(&mut graph, &mlp, _inp_u_ym_);

            let sum_x = graph.add(u_xp, u_xm);
            let two_u = graph.scalar_mul(u_c, 2.0);
            let diff_x = graph.sub(sum_x, two_u);
            let u_xx = graph.scalar_mul(diff_x, 1.0 / (eps * eps));
            let sum_y = graph.add(u_yp, u_ym);
            let diff_y = graph.sub(sum_y, two_u);
            let u_yy = graph.scalar_mul(diff_y, 1.0 / (eps * eps));

            let laplacian = graph.add(u_xx, u_yy);
            let k2u = graph.scalar_mul(u_c, k2);
            let lhs = graph.add(laplacian, k2u);
            let f_val = graph.input(Tensor::from_vec_unchecked(vec![source(x, y)], &[1, 1]));
            let residual = graph.sub(lhs, f_val);
            let sq = graph.mul(residual, residual);
            phys_nodes.push(sq);
        }
        let mut phys_sum = phys_nodes[0];
        for &n in &phys_nodes[1..] { phys_sum = graph.add(phys_sum, n); }
        let _tmp_pinn_1_ = graph.input(Tensor::from_vec_unchecked(vec![config.n_collocation as f64], &[1]));
        let phys_loss = graph.div(phys_sum, _tmp_pinn_1_);

        // BC
        let n_bc_pts = boundary.len() / 2;
        let mut bc_nodes = Vec::new();
        for j in 0..n_bc_pts {
            let inp = graph.input(Tensor::from_vec_unchecked(vec![boundary[j * 2], boundary[j * 2 + 1]], &[1, 2]));
            let pred = mlp_forward(&mut graph, &mlp, inp);
            let sq = graph.mul(pred, pred);
            bc_nodes.push(sq);
        }
        let mut bc_sum = bc_nodes[0];
        for &n in &bc_nodes[1..] { bc_sum = graph.add(bc_sum, n); }
        let _tmp_pinn_2_ = graph.input(Tensor::from_vec_unchecked(vec![n_bc_pts as f64], &[1]));
        let bc_loss = graph.div(bc_sum, _tmp_pinn_2_);

        let bw = (epoch as f64 / (config.epochs as f64 * 0.2)).min(1.0);
        let _phys_scaled_ = graph.scalar_mul(phys_loss, config.physics_weight);
        let _bc_scaled_ = graph.scalar_mul(bc_loss, config.boundary_weight * bw);
        let total = graph.add(_phys_scaled_, _bc_scaled_);

        graph.backward(total);
        let mut flat_grads = Vec::with_capacity(total_params);
        for idx in &p_indices {
            if let Some(g) = graph.grad(*idx) { flat_grads.extend_from_slice(&g.to_vec()); }
            else { flat_grads.extend(std::iter::repeat(0.0).take(graph.tensor(*idx).to_vec().len())); }
        }
        let mut flat_params: Vec<f64> = params.iter().flat_map(|p| p.clone()).collect();
        let mut gnorm_acc = KahanAccumulatorF64::new();
        for &g in &flat_grads { gnorm_acc.add(g * g); }

        history.push(TrainLog {
            epoch, total_loss: graph.tensor(total).to_vec()[0],
            data_loss: 0.0, physics_loss: graph.tensor(phys_loss).to_vec()[0],
            boundary_loss: graph.tensor(bc_loss).to_vec()[0], grad_norm: gnorm_acc.finalize().sqrt(),
        });

        adam_step(&mut flat_params, &flat_grads, &mut adam);
        let mut offset = 0;
        for p in params.iter_mut() {
            let _plen_ = p.len();
            p.copy_from_slice(&flat_params[offset..offset + _plen_]);
            offset += _plen_;
        }
    }

    let n_eval = 100;
    let mut l2_acc = KahanAccumulatorF64::new();
    let mut max_err = 0.0f64;
    let mut eval_graph = crate::GradGraph::new();
    let mut ep = Vec::new();
    for (li, layer) in mlp_spec.layers.iter().enumerate() {
        ep.push(eval_graph.parameter(Tensor::from_vec_unchecked(params[li * 2].clone(), &[layer.out_features, layer.in_features])));
        ep.push(eval_graph.parameter(Tensor::from_vec_unchecked(params[li * 2 + 1].clone(), &[layer.out_features])));
    }
    let eval_mlp = Mlp { layers: mlp_spec.layers.iter().enumerate().map(|(li, l)| DenseLayer {
        weight_idx: ep[li * 2], bias_idx: ep[li * 2 + 1], activation: l.activation,
        in_features: l.in_features, out_features: l.out_features,
    }).collect() };
    for i in 0..n_eval {
        let x = (i as f64 + 0.5) / n_eval as f64;
        let y = 0.5;
        let exact = (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sin();
        let inp = eval_graph.input(Tensor::from_vec_unchecked(vec![x, y], &[1, 2]));
        let pred_idx = mlp_forward(&mut eval_graph, &eval_mlp, inp);
        let pred = eval_graph.tensor(pred_idx).to_vec()[0];
        let err = (pred - exact).abs();
        l2_acc.add(err * err);
        if err > max_err { max_err = err; }
    }

    PinnResult {
        history, final_params: params.into_iter().flatten().collect(),
        l2_error: Some((l2_acc.finalize() / n_eval as f64).sqrt()),
        max_error: Some(max_err), mean_residual: max_err,
    }
}

// ---------------------------------------------------------------------------
// Problem H: PINN — Diffusion-Reaction Equation
// ---------------------------------------------------------------------------
// PDE: u_t = D·u_xx + R·u·(1-u) on x∈[0,1], t∈[0,1]
// IC: u(x,0) = exp(-50(x-0.5)²)
// BC: u_x(0,t) = u_x(1,t) = 0 (Neumann)

/// Configuration for the diffusion-reaction PINN.
#[derive(Debug, Clone)]
pub struct DiffReactConfig {
    pub layer_sizes: Vec<usize>,
    pub epochs: usize,
    pub lr: f64,
    pub diffusion: f64,
    pub reaction: f64,
    pub physics_weight: f64,
    pub boundary_weight: f64,
    pub seed: u64,
    pub n_collocation: usize,
    pub n_ic: usize,
    pub n_bc: usize,
    pub fd_eps: f64,
}

impl Default for DiffReactConfig {
    fn default() -> Self {
        Self {
            layer_sizes: vec![2, 20, 20, 20, 1],
            epochs: 500, lr: 1e-3, diffusion: 0.01, reaction: 1.0,
            physics_weight: 1.0, boundary_weight: 10.0,
            seed: 42, n_collocation: 64, n_ic: 50, n_bc: 25, fd_eps: 1e-4,
        }
    }
}

/// Train a PINN for the diffusion-reaction equation u_t = D·u_xx + R·u(1-u).
pub fn pinn_diffreact_train(config: &DiffReactConfig) -> PinnResult {
    let x_range = (0.0, 1.0);
    let t_range = (0.0, 1.0);
    let eps = config.fd_eps;
    let d_coeff = config.diffusion;
    let r_coeff = config.reaction;

    let domain = PinnDomain::SpaceTime1D { x_range, t_range };
    let interior = domain.sample_interior(config.n_collocation, config.seed);

    let mut ic_x = Vec::with_capacity(config.n_ic);
    let mut ic_u = Vec::with_capacity(config.n_ic);
    for i in 0..config.n_ic {
        let x = (i as f64 + 0.5) / config.n_ic as f64;
        ic_x.push(x);
        ic_u.push((-50.0 * (x - 0.5) * (x - 0.5)).exp());
    }

    let mut temp_graph = crate::GradGraph::new();
    let (mlp_spec, _tp) = mlp_init(&mut temp_graph, &config.layer_sizes, Activation::Tanh, Activation::None, config.seed);
    let mut params: Vec<Vec<f64>> = mlp_spec.layers.iter().enumerate().flat_map(|(li, _)| {
        vec![temp_graph.tensor(_tp[li * 2]).to_vec(), temp_graph.tensor(_tp[li * 2 + 1]).to_vec()]
    }).collect();
    let total_params: usize = params.iter().map(|p| p.len()).sum();
    let mut adam = AdamState::new(total_params, config.lr);
    let mut history = Vec::with_capacity(config.epochs);

    for epoch in 0..config.epochs {
        let lr_min = config.lr * 0.01;
        adam.lr = lr_min + (config.lr - lr_min) * 0.5 * (1.0 + (std::f64::consts::PI * epoch as f64 / config.epochs as f64).cos());

        let mut graph = crate::GradGraph::new();
        let mut p_indices = Vec::new();
        for (li, layer) in mlp_spec.layers.iter().enumerate() {
            let w = Tensor::from_vec_unchecked(params[li * 2].clone(), &[layer.out_features, layer.in_features]);
            let b = Tensor::from_vec_unchecked(params[li * 2 + 1].clone(), &[layer.out_features]);
            p_indices.push(graph.parameter(w));
            p_indices.push(graph.parameter(b));
        }
        let mlp = Mlp { layers: mlp_spec.layers.iter().enumerate().map(|(li, l)| DenseLayer {
            weight_idx: p_indices[li * 2], bias_idx: p_indices[li * 2 + 1], activation: l.activation,
            in_features: l.in_features, out_features: l.out_features,
        }).collect() };

        // IC Loss
        let mut ic_nodes = Vec::new();
        for (j, &x) in ic_x.iter().enumerate() {
            let inp = graph.input(Tensor::from_vec_unchecked(vec![x, 0.0], &[1, 2]));
            let pred = mlp_forward(&mut graph, &mlp, inp);
            let target = graph.input(Tensor::from_vec_unchecked(vec![ic_u[j]], &[1, 1]));
            let diff = graph.sub(pred, target);
            ic_nodes.push(graph.mul(diff, diff));
        }
        let mut ic_sum = ic_nodes[0];
        for &n in &ic_nodes[1..] { ic_sum = graph.add(ic_sum, n); }
        let _tmp_pinn_3_ = graph.input(Tensor::from_vec_unchecked(vec![config.n_ic as f64], &[1]));
        let ic_loss = graph.div(ic_sum, _tmp_pinn_3_);

        // Neumann BC: u_x(0,t) = 0, u_x(1,t) = 0
        let mut bc_nodes = Vec::new();
        for i in 0..config.n_bc {
            let t = (i as f64 + 0.5) / config.n_bc as f64;
            // Left: u_x(0, t) ≈ (u(ε, t) - u(-ε, t)) / (2ε)
            let _inp_u_lp_ = graph.input(Tensor::from_vec_unchecked(vec![eps, t], &[1, 2]));
            let u_lp = mlp_forward(&mut graph, &mlp, _inp_u_lp_);
            let _inp_u_lm_ = graph.input(Tensor::from_vec_unchecked(vec![-eps, t], &[1, 2]));
            let u_lm = mlp_forward(&mut graph, &mlp, _inp_u_lm_);
            let du_l = graph.sub(u_lp, u_lm);
            let ux_l = graph.scalar_mul(du_l, 1.0 / (2.0 * eps));
            bc_nodes.push(graph.mul(ux_l, ux_l));
            // Right: u_x(1, t)
            let _inp_u_rp_ = graph.input(Tensor::from_vec_unchecked(vec![1.0 + eps, t], &[1, 2]));
            let u_rp = mlp_forward(&mut graph, &mlp, _inp_u_rp_);
            let _inp_u_rm_ = graph.input(Tensor::from_vec_unchecked(vec![1.0 - eps, t], &[1, 2]));
            let u_rm = mlp_forward(&mut graph, &mlp, _inp_u_rm_);
            let du_r = graph.sub(u_rp, u_rm);
            let ux_r = graph.scalar_mul(du_r, 1.0 / (2.0 * eps));
            bc_nodes.push(graph.mul(ux_r, ux_r));
        }
        let mut bc_sum = bc_nodes[0];
        for &n in &bc_nodes[1..] { bc_sum = graph.add(bc_sum, n); }
        let _tmp_pinn_4_ = graph.input(Tensor::from_vec_unchecked(vec![(config.n_bc * 2) as f64], &[1]));
        let bc_loss = graph.div(bc_sum, _tmp_pinn_4_);

        // Physics: u_t - D·u_xx - R·u·(1-u) = 0
        let mut phys_nodes = Vec::new();
        for i in 0..config.n_collocation {
            let x = interior[i * 2]; let t = interior[i * 2 + 1];
            let _inp_u_c_ = graph.input(Tensor::from_vec_unchecked(vec![x, t], &[1, 2]));
            let u_c = mlp_forward(&mut graph, &mlp, _inp_u_c_);
            let _inp_u_xp_ = graph.input(Tensor::from_vec_unchecked(vec![x + eps, t], &[1, 2]));
            let u_xp = mlp_forward(&mut graph, &mlp, _inp_u_xp_);
            let _inp_u_xm_ = graph.input(Tensor::from_vec_unchecked(vec![x - eps, t], &[1, 2]));
            let u_xm = mlp_forward(&mut graph, &mlp, _inp_u_xm_);
            let _inp_u_tp_ = graph.input(Tensor::from_vec_unchecked(vec![x, t + eps], &[1, 2]));
            let u_tp = mlp_forward(&mut graph, &mlp, _inp_u_tp_);
            let _inp_u_tm_ = graph.input(Tensor::from_vec_unchecked(vec![x, t - eps], &[1, 2]));
            let u_tm = mlp_forward(&mut graph, &mlp, _inp_u_tm_);
            let sum_x = graph.add(u_xp, u_xm);
            let two_u = graph.scalar_mul(u_c, 2.0);
            let diff_x = graph.sub(sum_x, two_u);
            let u_xx = graph.scalar_mul(diff_x, 1.0 / (eps * eps));
            let diff_t = graph.sub(u_tp, u_tm);
            let u_t = graph.scalar_mul(diff_t, 1.0 / (2.0 * eps));
            // R·u·(1-u)
            let one = graph.input(Tensor::from_vec_unchecked(vec![1.0], &[1, 1]));
            let one_minus_u = graph.sub(one, u_c);
            let u_times = graph.mul(u_c, one_minus_u);
            let react = graph.scalar_mul(u_times, r_coeff);
            let diff_term = graph.scalar_mul(u_xx, d_coeff);
            let rhs = graph.add(diff_term, react);
            let residual = graph.sub(u_t, rhs);
            phys_nodes.push(graph.mul(residual, residual));
        }
        let mut phys_sum = phys_nodes[0];
        for &n in &phys_nodes[1..] { phys_sum = graph.add(phys_sum, n); }
        let _tmp_pinn_5_ = graph.input(Tensor::from_vec_unchecked(vec![config.n_collocation as f64], &[1]));
        let phys_loss = graph.div(phys_sum, _tmp_pinn_5_);

        let bw = (epoch as f64 / (config.epochs as f64 * 0.2)).min(1.0);
        let _phys_s_ = graph.scalar_mul(phys_loss, config.physics_weight);
        let _bc_s_ = graph.scalar_mul(bc_loss, config.boundary_weight * bw);
        let _ic_s_ = graph.scalar_mul(ic_loss, config.boundary_weight * bw);
        let _bc_ic_ = graph.add(_bc_s_, _ic_s_);
        let total = graph.add(_phys_s_, _bc_ic_);

        graph.backward(total);
        let mut flat_grads = Vec::with_capacity(total_params);
        for idx in &p_indices {
            if let Some(g) = graph.grad(*idx) { flat_grads.extend_from_slice(&g.to_vec()); }
            else { flat_grads.extend(std::iter::repeat(0.0).take(graph.tensor(*idx).to_vec().len())); }
        }
        let mut flat_params: Vec<f64> = params.iter().flat_map(|p| p.clone()).collect();
        let mut gnorm_acc = KahanAccumulatorF64::new();
        for &g in &flat_grads { gnorm_acc.add(g * g); }
        history.push(TrainLog {
            epoch, total_loss: graph.tensor(total).to_vec()[0], data_loss: graph.tensor(ic_loss).to_vec()[0],
            physics_loss: graph.tensor(phys_loss).to_vec()[0], boundary_loss: graph.tensor(bc_loss).to_vec()[0],
            grad_norm: gnorm_acc.finalize().sqrt(),
        });
        adam_step(&mut flat_params, &flat_grads, &mut adam);
        let mut offset = 0;
        for p in params.iter_mut() {
            let _plen_ = p.len();
            p.copy_from_slice(&flat_params[offset..offset + _plen_]);
            offset += _plen_;
        }
    }

    let mean_residual = history.last().map_or(0.0, |h| h.physics_loss);
    PinnResult { history, final_params: params.into_iter().flatten().collect(), l2_error: None, max_error: None, mean_residual }
}

// ---------------------------------------------------------------------------
// Problem I: PINN — Allen-Cahn Equation
// ---------------------------------------------------------------------------
// PDE: u_t = ε²·u_xx + u - u³ on x∈[-1,1], t∈[0,1]

/// Configuration for the Allen-Cahn PINN.
#[derive(Debug, Clone)]
pub struct AllenCahnConfig {
    pub layer_sizes: Vec<usize>,
    pub epochs: usize,
    pub lr: f64,
    pub epsilon: f64,
    pub physics_weight: f64,
    pub boundary_weight: f64,
    pub seed: u64,
    pub n_collocation: usize,
    pub n_ic: usize,
    pub n_bc: usize,
    pub fd_eps: f64,
}

impl Default for AllenCahnConfig {
    fn default() -> Self {
        Self {
            layer_sizes: vec![2, 20, 20, 20, 1],
            epochs: 500, lr: 1e-3, epsilon: 0.01,
            physics_weight: 1.0, boundary_weight: 10.0,
            seed: 42, n_collocation: 64, n_ic: 50, n_bc: 25, fd_eps: 1e-4,
        }
    }
}

/// Train a PINN for the Allen-Cahn equation u_t = ε²·u_xx + u - u³.
pub fn pinn_allen_cahn_train(config: &AllenCahnConfig) -> PinnResult {
    let x_range = (-1.0, 1.0);
    let t_range = (0.0, 1.0);
    let eps = config.fd_eps;
    let eps2 = config.epsilon * config.epsilon;

    let domain = PinnDomain::SpaceTime1D { x_range, t_range };
    let interior = domain.sample_interior(config.n_collocation, config.seed);

    // IC: u(x,0) = x² * cos(πx)
    let mut ic_x = Vec::with_capacity(config.n_ic);
    let mut ic_u = Vec::with_capacity(config.n_ic);
    for i in 0..config.n_ic {
        let x = x_range.0 + (i as f64 + 0.5) / config.n_ic as f64 * 2.0;
        ic_x.push(x);
        ic_u.push(x * x * (std::f64::consts::PI * x).cos());
    }

    let mut temp_graph = crate::GradGraph::new();
    let (mlp_spec, _tp) = mlp_init(&mut temp_graph, &config.layer_sizes, Activation::Tanh, Activation::None, config.seed);
    let mut params: Vec<Vec<f64>> = mlp_spec.layers.iter().enumerate().flat_map(|(li, _)| {
        vec![temp_graph.tensor(_tp[li * 2]).to_vec(), temp_graph.tensor(_tp[li * 2 + 1]).to_vec()]
    }).collect();
    let total_params: usize = params.iter().map(|p| p.len()).sum();
    let mut adam = AdamState::new(total_params, config.lr);
    let mut history = Vec::with_capacity(config.epochs);

    for epoch in 0..config.epochs {
        let lr_min = config.lr * 0.01;
        adam.lr = lr_min + (config.lr - lr_min) * 0.5 * (1.0 + (std::f64::consts::PI * epoch as f64 / config.epochs as f64).cos());

        let mut graph = crate::GradGraph::new();
        let mut p_indices = Vec::new();
        for (li, layer) in mlp_spec.layers.iter().enumerate() {
            let w = Tensor::from_vec_unchecked(params[li * 2].clone(), &[layer.out_features, layer.in_features]);
            let b = Tensor::from_vec_unchecked(params[li * 2 + 1].clone(), &[layer.out_features]);
            p_indices.push(graph.parameter(w));
            p_indices.push(graph.parameter(b));
        }
        let mlp = Mlp { layers: mlp_spec.layers.iter().enumerate().map(|(li, l)| DenseLayer {
            weight_idx: p_indices[li * 2], bias_idx: p_indices[li * 2 + 1], activation: l.activation,
            in_features: l.in_features, out_features: l.out_features,
        }).collect() };

        // IC
        let mut ic_nodes = Vec::new();
        for (j, &x) in ic_x.iter().enumerate() {
            let inp = graph.input(Tensor::from_vec_unchecked(vec![x, 0.0], &[1, 2]));
            let pred = mlp_forward(&mut graph, &mlp, inp);
            let target = graph.input(Tensor::from_vec_unchecked(vec![ic_u[j]], &[1, 1]));
            let diff = graph.sub(pred, target);
            ic_nodes.push(graph.mul(diff, diff));
        }
        let mut ic_sum = ic_nodes[0];
        for &n in &ic_nodes[1..] { ic_sum = graph.add(ic_sum, n); }
        let _tmp_pinn_6_ = graph.input(Tensor::from_vec_unchecked(vec![config.n_ic as f64], &[1]));
        let ic_loss = graph.div(ic_sum, _tmp_pinn_6_);

        // BC: periodic u(-1,t) = u(1,t)
        let mut bc_nodes = Vec::new();
        for i in 0..config.n_bc {
            let t = (i as f64 + 0.5) / config.n_bc as f64;
            let _inp_u_l_ = graph.input(Tensor::from_vec_unchecked(vec![-1.0, t], &[1, 2]));
            let u_l = mlp_forward(&mut graph, &mlp, _inp_u_l_);
            let _inp_u_r_ = graph.input(Tensor::from_vec_unchecked(vec![1.0, t], &[1, 2]));
            let u_r = mlp_forward(&mut graph, &mlp, _inp_u_r_);
            let diff = graph.sub(u_l, u_r);
            bc_nodes.push(graph.mul(diff, diff));
        }
        let mut bc_sum = bc_nodes[0];
        for &n in &bc_nodes[1..] { bc_sum = graph.add(bc_sum, n); }
        let _tmp_pinn_7_ = graph.input(Tensor::from_vec_unchecked(vec![config.n_bc as f64], &[1]));
        let bc_loss = graph.div(bc_sum, _tmp_pinn_7_);

        // Physics: u_t - ε²·u_xx - u + u³ = 0
        let mut phys_nodes = Vec::new();
        for i in 0..config.n_collocation {
            let x = interior[i * 2]; let t = interior[i * 2 + 1];
            let _inp_u_c_ = graph.input(Tensor::from_vec_unchecked(vec![x, t], &[1, 2]));
            let u_c = mlp_forward(&mut graph, &mlp, _inp_u_c_);
            let _inp_u_xp_ = graph.input(Tensor::from_vec_unchecked(vec![x + eps, t], &[1, 2]));
            let u_xp = mlp_forward(&mut graph, &mlp, _inp_u_xp_);
            let _inp_u_xm_ = graph.input(Tensor::from_vec_unchecked(vec![x - eps, t], &[1, 2]));
            let u_xm = mlp_forward(&mut graph, &mlp, _inp_u_xm_);
            let _inp_u_tp_ = graph.input(Tensor::from_vec_unchecked(vec![x, t + eps], &[1, 2]));
            let u_tp = mlp_forward(&mut graph, &mlp, _inp_u_tp_);
            let _inp_u_tm_ = graph.input(Tensor::from_vec_unchecked(vec![x, t - eps], &[1, 2]));
            let u_tm = mlp_forward(&mut graph, &mlp, _inp_u_tm_);
            let sum_x = graph.add(u_xp, u_xm);
            let two_u = graph.scalar_mul(u_c, 2.0);
            let diff_x = graph.sub(sum_x, two_u);
            let u_xx = graph.scalar_mul(diff_x, 1.0 / (eps * eps));
            let diff_t = graph.sub(u_tp, u_tm);
            let u_t = graph.scalar_mul(diff_t, 1.0 / (2.0 * eps));
            // u³
            let u_sq = graph.mul(u_c, u_c);
            let u_cubed = graph.mul(u_sq, u_c);
            // residual = u_t - ε²·u_xx - u + u³
            let eps2_uxx = graph.scalar_mul(u_xx, eps2);
            let rhs = graph.add(eps2_uxx, u_c);
            let rhs2 = graph.sub(rhs, u_cubed);
            let residual = graph.sub(u_t, rhs2);
            phys_nodes.push(graph.mul(residual, residual));
        }
        let mut phys_sum = phys_nodes[0];
        for &n in &phys_nodes[1..] { phys_sum = graph.add(phys_sum, n); }
        let _tmp_pinn_8_ = graph.input(Tensor::from_vec_unchecked(vec![config.n_collocation as f64], &[1]));
        let phys_loss = graph.div(phys_sum, _tmp_pinn_8_);

        let bw = (epoch as f64 / (config.epochs as f64 * 0.2)).min(1.0);
        let _phys_s_ = graph.scalar_mul(phys_loss, config.physics_weight);
        let _bc_s_ = graph.scalar_mul(bc_loss, config.boundary_weight * bw);
        let _ic_s_ = graph.scalar_mul(ic_loss, config.boundary_weight * bw);
        let _bc_ic_ = graph.add(_bc_s_, _ic_s_);
        let total = graph.add(_phys_s_, _bc_ic_);

        graph.backward(total);
        let mut flat_grads = Vec::with_capacity(total_params);
        for idx in &p_indices {
            if let Some(g) = graph.grad(*idx) { flat_grads.extend_from_slice(&g.to_vec()); }
            else { flat_grads.extend(std::iter::repeat(0.0).take(graph.tensor(*idx).to_vec().len())); }
        }
        let mut flat_params: Vec<f64> = params.iter().flat_map(|p| p.clone()).collect();
        let mut gnorm_acc = KahanAccumulatorF64::new();
        for &g in &flat_grads { gnorm_acc.add(g * g); }
        history.push(TrainLog {
            epoch, total_loss: graph.tensor(total).to_vec()[0], data_loss: graph.tensor(ic_loss).to_vec()[0],
            physics_loss: graph.tensor(phys_loss).to_vec()[0], boundary_loss: graph.tensor(bc_loss).to_vec()[0],
            grad_norm: gnorm_acc.finalize().sqrt(),
        });
        adam_step(&mut flat_params, &flat_grads, &mut adam);
        let mut offset = 0;
        for p in params.iter_mut() {
            let _plen_ = p.len();
            p.copy_from_slice(&flat_params[offset..offset + _plen_]);
            offset += _plen_;
        }
    }

    let mean_residual = history.last().map_or(0.0, |h| h.physics_loss);
    PinnResult { history, final_params: params.into_iter().flatten().collect(), l2_error: None, max_error: None, mean_residual }
}

// ---------------------------------------------------------------------------
// Problem J: PINN — KdV Equation
// ---------------------------------------------------------------------------
// PDE: u_t + 6·u·u_x + u_xxx = 0 on x∈[-5,5], t∈[0,1]

/// Configuration for the KdV PINN.
#[derive(Debug, Clone)]
pub struct KdvConfig {
    pub layer_sizes: Vec<usize>,
    pub epochs: usize,
    pub lr: f64,
    pub physics_weight: f64,
    pub boundary_weight: f64,
    pub seed: u64,
    pub n_collocation: usize,
    pub n_ic: usize,
    pub n_bc: usize,
    pub fd_eps: f64,
}

impl Default for KdvConfig {
    fn default() -> Self {
        Self {
            layer_sizes: vec![2, 20, 20, 20, 1],
            epochs: 500, lr: 1e-3,
            physics_weight: 1.0, boundary_weight: 10.0,
            seed: 42, n_collocation: 64, n_ic: 50, n_bc: 25, fd_eps: 1e-4,
        }
    }
}

/// Train a PINN for the KdV equation u_t + 6·u·u_x + u_xxx = 0.
/// IC: sech²(x) soliton.
pub fn pinn_kdv_train(config: &KdvConfig) -> PinnResult {
    let x_range = (-5.0, 5.0);
    let t_range = (0.0, 1.0);
    let eps = config.fd_eps;

    let domain = PinnDomain::SpaceTime1D { x_range, t_range };
    let interior = domain.sample_interior(config.n_collocation, config.seed);

    // IC: u(x,0) = 0.5·sech²(x/2) (single soliton)
    let sech2 = |x: f64| -> f64 { let c = x.cosh(); 1.0 / (c * c) };
    let mut ic_x = Vec::with_capacity(config.n_ic);
    let mut ic_u = Vec::with_capacity(config.n_ic);
    for i in 0..config.n_ic {
        let x = x_range.0 + (i as f64 + 0.5) / config.n_ic as f64 * (x_range.1 - x_range.0);
        ic_x.push(x);
        ic_u.push(0.5 * sech2(x / 2.0));
    }

    let mut temp_graph = crate::GradGraph::new();
    let (mlp_spec, _tp) = mlp_init(&mut temp_graph, &config.layer_sizes, Activation::Tanh, Activation::None, config.seed);
    let mut params: Vec<Vec<f64>> = mlp_spec.layers.iter().enumerate().flat_map(|(li, _)| {
        vec![temp_graph.tensor(_tp[li * 2]).to_vec(), temp_graph.tensor(_tp[li * 2 + 1]).to_vec()]
    }).collect();
    let total_params: usize = params.iter().map(|p| p.len()).sum();
    let mut adam = AdamState::new(total_params, config.lr);
    let mut history = Vec::with_capacity(config.epochs);

    for epoch in 0..config.epochs {
        let lr_min = config.lr * 0.01;
        adam.lr = lr_min + (config.lr - lr_min) * 0.5 * (1.0 + (std::f64::consts::PI * epoch as f64 / config.epochs as f64).cos());

        let mut graph = crate::GradGraph::new();
        let mut p_indices = Vec::new();
        for (li, layer) in mlp_spec.layers.iter().enumerate() {
            let w = Tensor::from_vec_unchecked(params[li * 2].clone(), &[layer.out_features, layer.in_features]);
            let b = Tensor::from_vec_unchecked(params[li * 2 + 1].clone(), &[layer.out_features]);
            p_indices.push(graph.parameter(w));
            p_indices.push(graph.parameter(b));
        }
        let mlp = Mlp { layers: mlp_spec.layers.iter().enumerate().map(|(li, l)| DenseLayer {
            weight_idx: p_indices[li * 2], bias_idx: p_indices[li * 2 + 1], activation: l.activation,
            in_features: l.in_features, out_features: l.out_features,
        }).collect() };

        // IC
        let mut ic_nodes = Vec::new();
        for (j, &x) in ic_x.iter().enumerate() {
            let inp = graph.input(Tensor::from_vec_unchecked(vec![x, 0.0], &[1, 2]));
            let pred = mlp_forward(&mut graph, &mlp, inp);
            let target = graph.input(Tensor::from_vec_unchecked(vec![ic_u[j]], &[1, 1]));
            let diff = graph.sub(pred, target);
            ic_nodes.push(graph.mul(diff, diff));
        }
        let mut ic_sum = ic_nodes[0];
        for &n in &ic_nodes[1..] { ic_sum = graph.add(ic_sum, n); }
        let _tmp_pinn_9_ = graph.input(Tensor::from_vec_unchecked(vec![config.n_ic as f64], &[1]));
        let ic_loss = graph.div(ic_sum, _tmp_pinn_9_);

        // Physics: u_t + 6·u·u_x + u_xxx = 0
        let mut phys_nodes = Vec::new();
        for i in 0..config.n_collocation {
            let x = interior[i * 2]; let t = interior[i * 2 + 1];
            let _inp_u_c_ = graph.input(Tensor::from_vec_unchecked(vec![x, t], &[1, 2]));
            let u_c = mlp_forward(&mut graph, &mlp, _inp_u_c_);
            let _inp_u_xp_ = graph.input(Tensor::from_vec_unchecked(vec![x + eps, t], &[1, 2]));
            let u_xp = mlp_forward(&mut graph, &mlp, _inp_u_xp_);
            let _inp_u_xm_ = graph.input(Tensor::from_vec_unchecked(vec![x - eps, t], &[1, 2]));
            let u_xm = mlp_forward(&mut graph, &mlp, _inp_u_xm_);
            let _inp_u_tp_ = graph.input(Tensor::from_vec_unchecked(vec![x, t + eps], &[1, 2]));
            let u_tp = mlp_forward(&mut graph, &mlp, _inp_u_tp_);
            let _inp_u_tm_ = graph.input(Tensor::from_vec_unchecked(vec![x, t - eps], &[1, 2]));
            let u_tm = mlp_forward(&mut graph, &mlp, _inp_u_tm_);
            // u_x = (u(x+ε) - u(x-ε)) / (2ε)
            let du = graph.sub(u_xp, u_xm);
            let u_x = graph.scalar_mul(du, 1.0 / (2.0 * eps));
            // u_t
            let dt = graph.sub(u_tp, u_tm);
            let u_t = graph.scalar_mul(dt, 1.0 / (2.0 * eps));
            // u_xxx via 2nd-order FD on u_x: need u(x+2ε), u(x-2ε)
            let _inp_u_x2p_ = graph.input(Tensor::from_vec_unchecked(vec![x + 2.0 * eps, t], &[1, 2]));
            let u_x2p = mlp_forward(&mut graph, &mlp, _inp_u_x2p_);
            let _inp_u_x2m_ = graph.input(Tensor::from_vec_unchecked(vec![x - 2.0 * eps, t], &[1, 2]));
            let u_x2m = mlp_forward(&mut graph, &mlp, _inp_u_x2m_);
            // u_xxx ≈ (u(x+2ε) - 2u(x+ε) + 2u(x-ε) - u(x-2ε)) / (2ε³)
            let _tmp_pinn_10_ = graph.scalar_mul(u_xp, 2.0);
            let a1 = graph.sub(u_x2p, _tmp_pinn_10_);
            let _tmp_pinn_11_ = graph.scalar_mul(u_xm, 2.0);
            let a2 = graph.add(a1, _tmp_pinn_11_);
            let a3 = graph.sub(a2, u_x2m);
            let u_xxx = graph.scalar_mul(a3, 1.0 / (2.0 * eps * eps * eps));
            // 6·u·u_x
            let _tmp_pinn_12_ = graph.mul(u_c, u_x);
            let six_u_ux = graph.scalar_mul(_tmp_pinn_12_, 6.0);
            // residual = u_t + 6·u·u_x + u_xxx
            let sum1 = graph.add(u_t, six_u_ux);
            let residual = graph.add(sum1, u_xxx);
            phys_nodes.push(graph.mul(residual, residual));
        }
        let mut phys_sum = phys_nodes[0];
        for &n in &phys_nodes[1..] { phys_sum = graph.add(phys_sum, n); }
        let _tmp_pinn_13_ = graph.input(Tensor::from_vec_unchecked(vec![config.n_collocation as f64], &[1]));
        let phys_loss = graph.div(phys_sum, _tmp_pinn_13_);

        let bw = (epoch as f64 / (config.epochs as f64 * 0.2)).min(1.0);
        let _phys_s_ = graph.scalar_mul(phys_loss, config.physics_weight);
        let _ic_s_ = graph.scalar_mul(ic_loss, config.boundary_weight * bw);
        let total = graph.add(_phys_s_, _ic_s_);

        graph.backward(total);
        let mut flat_grads = Vec::with_capacity(total_params);
        for idx in &p_indices {
            if let Some(g) = graph.grad(*idx) { flat_grads.extend_from_slice(&g.to_vec()); }
            else { flat_grads.extend(std::iter::repeat(0.0).take(graph.tensor(*idx).to_vec().len())); }
        }
        let mut flat_params: Vec<f64> = params.iter().flat_map(|p| p.clone()).collect();
        let mut gnorm_acc = KahanAccumulatorF64::new();
        for &g in &flat_grads { gnorm_acc.add(g * g); }
        history.push(TrainLog {
            epoch, total_loss: graph.tensor(total).to_vec()[0], data_loss: graph.tensor(ic_loss).to_vec()[0],
            physics_loss: graph.tensor(phys_loss).to_vec()[0], boundary_loss: 0.0,
            grad_norm: gnorm_acc.finalize().sqrt(),
        });
        adam_step(&mut flat_params, &flat_grads, &mut adam);
        let mut offset = 0;
        for p in params.iter_mut() {
            let _plen_ = p.len();
            p.copy_from_slice(&flat_params[offset..offset + _plen_]);
            offset += _plen_;
        }
    }

    let mean_residual = history.last().map_or(0.0, |h| h.physics_loss);
    PinnResult { history, final_params: params.into_iter().flatten().collect(), l2_error: None, max_error: None, mean_residual }
}

// ---------------------------------------------------------------------------
// Problem K: PINN — Schrödinger Equation (split real/imag)
// ---------------------------------------------------------------------------
// PDE: i·ψ_t + 0.5·ψ_xx + |ψ|²·ψ = 0 (NLS, focusing)
// Split: ψ = u + iv, real part: u_t + 0.5·v_xx + (u²+v²)v = 0
//                    imag part: v_t - 0.5·u_xx - (u²+v²)u = 0

/// Configuration for the Schrödinger PINN.
#[derive(Debug, Clone)]
pub struct SchrodingerConfig {
    pub layer_sizes: Vec<usize>,
    pub epochs: usize,
    pub lr: f64,
    pub physics_weight: f64,
    pub boundary_weight: f64,
    pub seed: u64,
    pub n_collocation: usize,
    pub n_ic: usize,
    pub n_bc: usize,
    pub fd_eps: f64,
}

impl Default for SchrodingerConfig {
    fn default() -> Self {
        Self {
            layer_sizes: vec![2, 30, 30, 30, 2], // 2 outputs: u(real), v(imag)
            epochs: 500, lr: 1e-3,
            physics_weight: 1.0, boundary_weight: 10.0,
            seed: 42, n_collocation: 64, n_ic: 50, n_bc: 25, fd_eps: 1e-4,
        }
    }
}

/// Train a PINN for the nonlinear Schrödinger equation (split real/imag).
/// Network outputs 2 values: [u(real), v(imag)] at each (x, t).
pub fn pinn_schrodinger_train(config: &SchrodingerConfig) -> PinnResult {
    let x_range = (-5.0, 5.0);
    let t_range = (0.0, std::f64::consts::PI / 2.0);
    let eps = config.fd_eps;

    let domain = PinnDomain::SpaceTime1D { x_range, t_range };
    let interior = domain.sample_interior(config.n_collocation, config.seed);

    // IC: ψ(x,0) = 2·sech(x) → u(x,0) = 2·sech(x), v(x,0) = 0
    let sech = |x: f64| -> f64 { 1.0 / x.cosh() };
    let mut ic_x = Vec::with_capacity(config.n_ic);
    let mut ic_u = Vec::with_capacity(config.n_ic);
    for i in 0..config.n_ic {
        let x = x_range.0 + (i as f64 + 0.5) / config.n_ic as f64 * (x_range.1 - x_range.0);
        ic_x.push(x);
        ic_u.push(2.0 * sech(x));
    }

    let mut temp_graph = crate::GradGraph::new();
    let (mlp_spec, _tp) = mlp_init(&mut temp_graph, &config.layer_sizes, Activation::Tanh, Activation::None, config.seed);
    let mut params: Vec<Vec<f64>> = mlp_spec.layers.iter().enumerate().flat_map(|(li, _)| {
        vec![temp_graph.tensor(_tp[li * 2]).to_vec(), temp_graph.tensor(_tp[li * 2 + 1]).to_vec()]
    }).collect();
    let total_params: usize = params.iter().map(|p| p.len()).sum();
    let mut adam = AdamState::new(total_params, config.lr);
    let mut history = Vec::with_capacity(config.epochs);

    // Helper to evaluate network and extract (u, v) components
    // Since the network outputs [1, 2], we split into real and imag

    for epoch in 0..config.epochs {
        let lr_min = config.lr * 0.01;
        adam.lr = lr_min + (config.lr - lr_min) * 0.5 * (1.0 + (std::f64::consts::PI * epoch as f64 / config.epochs as f64).cos());

        let mut graph = crate::GradGraph::new();
        let mut p_indices = Vec::new();
        for (li, layer) in mlp_spec.layers.iter().enumerate() {
            let w = Tensor::from_vec_unchecked(params[li * 2].clone(), &[layer.out_features, layer.in_features]);
            let b = Tensor::from_vec_unchecked(params[li * 2 + 1].clone(), &[layer.out_features]);
            p_indices.push(graph.parameter(w));
            p_indices.push(graph.parameter(b));
        }
        let mlp = Mlp { layers: mlp_spec.layers.iter().enumerate().map(|(li, l)| DenseLayer {
            weight_idx: p_indices[li * 2], bias_idx: p_indices[li * 2 + 1], activation: l.activation,
            in_features: l.in_features, out_features: l.out_features,
        }).collect() };

        // IC: u(x,0) = 2·sech(x), v(x,0) = 0
        let mut ic_nodes = Vec::new();
        for (j, &x) in ic_x.iter().enumerate() {
            let inp = graph.input(Tensor::from_vec_unchecked(vec![x, 0.0], &[1, 2]));
            let out = mlp_forward(&mut graph, &mlp, inp); // [1, 2]
            let out_data = graph.tensor(out).to_vec();
            let u_pred = out_data[0];
            let v_pred = out_data[1];
            // MSE on both components
            let u_err = u_pred - ic_u[j];
            let v_err = v_pred; // should be 0
            let err_node = graph.input(Tensor::from_vec_unchecked(vec![u_err * u_err + v_err * v_err], &[1, 1]));
            ic_nodes.push(err_node);
        }
        let mut ic_sum = ic_nodes[0];
        for &n in &ic_nodes[1..] { ic_sum = graph.add(ic_sum, n); }
        let _tmp_pinn_14_ = graph.input(Tensor::from_vec_unchecked(vec![config.n_ic as f64], &[1]));
        let ic_loss = graph.div(ic_sum, _tmp_pinn_14_);

        // Physics: compute FD residual for both real and imag equations
        // This requires 5 network evaluations per collocation point
        let mut phys_nodes = Vec::new();
        for i in 0..config.n_collocation {
            let x = interior[i * 2]; let t = interior[i * 2 + 1];

            let _inp_out_c_ = graph.input(Tensor::from_vec_unchecked(vec![x, t], &[1, 2]));
            let out_c = mlp_forward(&mut graph, &mlp, _inp_out_c_);
            let data_c = graph.tensor(out_c).to_vec();
            let (u_c, v_c) = (data_c[0], data_c[1]);
            let _inp_out_xp_ = graph.input(Tensor::from_vec_unchecked(vec![x + eps, t], &[1, 2]));
            let out_xp = mlp_forward(&mut graph, &mlp, _inp_out_xp_);
            let data_xp = graph.tensor(out_xp).to_vec();
            let (u_xp, v_xp) = (data_xp[0], data_xp[1]);
            let _inp_out_xm_ = graph.input(Tensor::from_vec_unchecked(vec![x - eps, t], &[1, 2]));
            let out_xm = mlp_forward(&mut graph, &mlp, _inp_out_xm_);
            let data_xm = graph.tensor(out_xm).to_vec();
            let (u_xm, v_xm) = (data_xm[0], data_xm[1]);
            let _inp_out_tp_ = graph.input(Tensor::from_vec_unchecked(vec![x, t + eps], &[1, 2]));
            let out_tp = mlp_forward(&mut graph, &mlp, _inp_out_tp_);
            let data_tp = graph.tensor(out_tp).to_vec();
            let (u_tp, v_tp) = (data_tp[0], data_tp[1]);
            let _inp_out_tm_ = graph.input(Tensor::from_vec_unchecked(vec![x, t - eps], &[1, 2]));
            let out_tm = mlp_forward(&mut graph, &mlp, _inp_out_tm_);
            let data_tm = graph.tensor(out_tm).to_vec();
            let (u_tm, v_tm) = (data_tm[0], data_tm[1]);

            let u_xx = (u_xp - 2.0 * u_c + u_xm) / (eps * eps);
            let v_xx = (v_xp - 2.0 * v_c + v_xm) / (eps * eps);
            let u_t = (u_tp - u_tm) / (2.0 * eps);
            let v_t = (v_tp - v_tm) / (2.0 * eps);
            let abs2 = u_c * u_c + v_c * v_c;

            // Real: u_t + 0.5·v_xx + |ψ|²·v = 0
            let r_real = u_t + 0.5 * v_xx + abs2 * v_c;
            // Imag: v_t - 0.5·u_xx - |ψ|²·u = 0
            let r_imag = v_t - 0.5 * u_xx - abs2 * u_c;

            let r2 = r_real * r_real + r_imag * r_imag;
            let r2_node = graph.input(Tensor::from_vec_unchecked(vec![r2], &[1, 1]));
            phys_nodes.push(r2_node);
        }
        let mut phys_sum = phys_nodes[0];
        for &n in &phys_nodes[1..] { phys_sum = graph.add(phys_sum, n); }
        let _tmp_pinn_15_ = graph.input(Tensor::from_vec_unchecked(vec![config.n_collocation as f64], &[1]));
        let phys_loss = graph.div(phys_sum, _tmp_pinn_15_);

        let bw = (epoch as f64 / (config.epochs as f64 * 0.2)).min(1.0);
        let _phys_s_ = graph.scalar_mul(phys_loss, config.physics_weight);
        let _ic_s_ = graph.scalar_mul(ic_loss, config.boundary_weight * bw);
        let total = graph.add(_phys_s_, _ic_s_);

        graph.backward(total);
        let mut flat_grads = Vec::with_capacity(total_params);
        for idx in &p_indices {
            if let Some(g) = graph.grad(*idx) { flat_grads.extend_from_slice(&g.to_vec()); }
            else { flat_grads.extend(std::iter::repeat(0.0).take(graph.tensor(*idx).to_vec().len())); }
        }
        let mut flat_params: Vec<f64> = params.iter().flat_map(|p| p.clone()).collect();
        let mut gnorm_acc = KahanAccumulatorF64::new();
        for &g in &flat_grads { gnorm_acc.add(g * g); }
        history.push(TrainLog {
            epoch, total_loss: graph.tensor(total).to_vec()[0], data_loss: graph.tensor(ic_loss).to_vec()[0],
            physics_loss: graph.tensor(phys_loss).to_vec()[0], boundary_loss: 0.0,
            grad_norm: gnorm_acc.finalize().sqrt(),
        });
        adam_step(&mut flat_params, &flat_grads, &mut adam);
        let mut offset = 0;
        for p in params.iter_mut() {
            let _plen_ = p.len();
            p.copy_from_slice(&flat_params[offset..offset + _plen_]);
            offset += _plen_;
        }
    }

    let mean_residual = history.last().map_or(0.0, |h| h.physics_loss);
    PinnResult { history, final_params: params.into_iter().flatten().collect(), l2_error: None, max_error: None, mean_residual }
}

// ---------------------------------------------------------------------------
// Problem L: PINN — 2D Navier-Stokes (stream function formulation)
// ---------------------------------------------------------------------------
// Steady NS: -ν·∇²ω + u·ω_x + v·ω_y = 0, where ω = -∇²ψ, u = ψ_y, v = -ψ_x
// Lid-driven cavity: u(top)=1, u(other walls)=0

/// Configuration for the NS PINN.
#[derive(Debug, Clone)]
pub struct NavierStokesConfig {
    pub layer_sizes: Vec<usize>,
    pub epochs: usize,
    pub lr: f64,
    pub nu: f64,
    pub physics_weight: f64,
    pub boundary_weight: f64,
    pub seed: u64,
    pub n_collocation: usize,
    pub n_bc: usize,
    pub fd_eps: f64,
}

impl Default for NavierStokesConfig {
    fn default() -> Self {
        Self {
            layer_sizes: vec![2, 30, 30, 30, 1], // outputs stream function ψ
            epochs: 500, lr: 1e-3, nu: 0.01,
            physics_weight: 1.0, boundary_weight: 10.0,
            seed: 42, n_collocation: 64, n_bc: 25, fd_eps: 1e-4,
        }
    }
}

/// Train a PINN for the 2D steady Navier-Stokes equations using stream function.
/// The network outputs ψ(x,y), and we derive u=ψ_y, v=-ψ_x via FD.
pub fn pinn_navier_stokes_train(config: &NavierStokesConfig) -> PinnResult {
    let domain = PinnDomain::Rectangle2D { x_range: (0.0, 1.0), y_range: (0.0, 1.0) };
    let interior = domain.sample_interior(config.n_collocation, config.seed);
    let boundary = domain.sample_boundary(config.n_bc * 4, config.seed);
    let eps = config.fd_eps;

    let mut temp_graph = crate::GradGraph::new();
    let (mlp_spec, _tp) = mlp_init(&mut temp_graph, &config.layer_sizes, Activation::Tanh, Activation::None, config.seed);
    let mut params: Vec<Vec<f64>> = mlp_spec.layers.iter().enumerate().flat_map(|(li, _)| {
        vec![temp_graph.tensor(_tp[li * 2]).to_vec(), temp_graph.tensor(_tp[li * 2 + 1]).to_vec()]
    }).collect();
    let total_params: usize = params.iter().map(|p| p.len()).sum();
    let mut adam = AdamState::new(total_params, config.lr);
    let mut history = Vec::with_capacity(config.epochs);

    for epoch in 0..config.epochs {
        let lr_min = config.lr * 0.01;
        adam.lr = lr_min + (config.lr - lr_min) * 0.5 * (1.0 + (std::f64::consts::PI * epoch as f64 / config.epochs as f64).cos());

        let mut graph = crate::GradGraph::new();
        let mut p_indices = Vec::new();
        for (li, layer) in mlp_spec.layers.iter().enumerate() {
            let w = Tensor::from_vec_unchecked(params[li * 2].clone(), &[layer.out_features, layer.in_features]);
            let b = Tensor::from_vec_unchecked(params[li * 2 + 1].clone(), &[layer.out_features]);
            p_indices.push(graph.parameter(w));
            p_indices.push(graph.parameter(b));
        }
        let mlp = Mlp { layers: mlp_spec.layers.iter().enumerate().map(|(li, l)| DenseLayer {
            weight_idx: p_indices[li * 2], bias_idx: p_indices[li * 2 + 1], activation: l.activation,
            in_features: l.in_features, out_features: l.out_features,
        }).collect() };

        // Physics: vorticity-stream function formulation
        let mut phys_nodes = Vec::new();
        for i in 0..config.n_collocation {
            let x = interior[i * 2]; let y = interior[i * 2 + 1];
            let _inp_idx_c_ = graph.input(Tensor::from_vec_unchecked(vec![x, y], &[1, 2]));
            let idx_c = mlp_forward(&mut graph, &mlp, _inp_idx_c_);
            let psi_c = graph.tensor(idx_c).to_vec()[0];
            let _inp_idx_xp_ = graph.input(Tensor::from_vec_unchecked(vec![x + eps, y], &[1, 2]));
            let idx_xp = mlp_forward(&mut graph, &mlp, _inp_idx_xp_);
            let psi_xp = graph.tensor(idx_xp).to_vec()[0];
            let _inp_idx_xm_ = graph.input(Tensor::from_vec_unchecked(vec![x - eps, y], &[1, 2]));
            let idx_xm = mlp_forward(&mut graph, &mlp, _inp_idx_xm_);
            let psi_xm = graph.tensor(idx_xm).to_vec()[0];
            let _inp_idx_yp_ = graph.input(Tensor::from_vec_unchecked(vec![x, y + eps], &[1, 2]));
            let idx_yp = mlp_forward(&mut graph, &mlp, _inp_idx_yp_);
            let psi_yp = graph.tensor(idx_yp).to_vec()[0];
            let _inp_idx_ym_ = graph.input(Tensor::from_vec_unchecked(vec![x, y - eps], &[1, 2]));
            let idx_ym = mlp_forward(&mut graph, &mlp, _inp_idx_ym_);
            let psi_ym = graph.tensor(idx_ym).to_vec()[0];

            // u = ψ_y, v = -ψ_x
            let u_vel = (psi_yp - psi_ym) / (2.0 * eps);
            let v_vel = -(psi_xp - psi_xm) / (2.0 * eps);

            // ω = -∇²ψ
            let psi_xx = (psi_xp - 2.0 * psi_c + psi_xm) / (eps * eps);
            let psi_yy = (psi_yp - 2.0 * psi_c + psi_ym) / (eps * eps);
            let omega = -(psi_xx + psi_yy);

            // Need ω_x, ω_y via FD on ω
            // ω at (x±ε, y) and (x, y±ε)
            let _inp_idx_xp_yp_ = graph.input(Tensor::from_vec_unchecked(vec![x + eps, y + eps], &[1, 2]));
            let idx_xp_yp = mlp_forward(&mut graph, &mlp, _inp_idx_xp_yp_);
            let psi_xp_yp = graph.tensor(idx_xp_yp).to_vec()[0];
            let _inp_idx_xp_ym_ = graph.input(Tensor::from_vec_unchecked(vec![x + eps, y - eps], &[1, 2]));
            let idx_xp_ym = mlp_forward(&mut graph, &mlp, _inp_idx_xp_ym_);
            let psi_xp_ym = graph.tensor(idx_xp_ym).to_vec()[0];
            let _inp_idx_xm_yp_ = graph.input(Tensor::from_vec_unchecked(vec![x - eps, y + eps], &[1, 2]));
            let idx_xm_yp = mlp_forward(&mut graph, &mlp, _inp_idx_xm_yp_);
            let psi_xm_yp = graph.tensor(idx_xm_yp).to_vec()[0];
            let _inp_idx_xm_ym_ = graph.input(Tensor::from_vec_unchecked(vec![x - eps, y - eps], &[1, 2]));
            let idx_xm_ym = mlp_forward(&mut graph, &mlp, _inp_idx_xm_ym_);
            let psi_xm_ym = graph.tensor(idx_xm_ym).to_vec()[0];
            let _inp_idx_x2p_ = graph.input(Tensor::from_vec_unchecked(vec![x + 2.0 * eps, y], &[1, 2]));
            let idx_x2p = mlp_forward(&mut graph, &mlp, _inp_idx_x2p_);
            let psi_x2p = graph.tensor(idx_x2p).to_vec()[0];
            let _inp_idx_x2m_ = graph.input(Tensor::from_vec_unchecked(vec![x - 2.0 * eps, y], &[1, 2]));
            let idx_x2m = mlp_forward(&mut graph, &mlp, _inp_idx_x2m_);
            let psi_x2m = graph.tensor(idx_x2m).to_vec()[0];
            let _inp_idx_y2p_ = graph.input(Tensor::from_vec_unchecked(vec![x, y + 2.0 * eps], &[1, 2]));
            let idx_y2p = mlp_forward(&mut graph, &mlp, _inp_idx_y2p_);
            let psi_y2p = graph.tensor(idx_y2p).to_vec()[0];
            let _inp_idx_y2m_ = graph.input(Tensor::from_vec_unchecked(vec![x, y - 2.0 * eps], &[1, 2]));
            let idx_y2m = mlp_forward(&mut graph, &mlp, _inp_idx_y2m_);
            let psi_y2m = graph.tensor(idx_y2m).to_vec()[0];

            // ∇²ψ at (x+ε, y)
            let lap_xp = (psi_x2p - 2.0 * psi_xp + psi_c) / (eps * eps)
                + (psi_xp_yp - 2.0 * psi_xp + psi_xp_ym) / (eps * eps);
            // ∇²ψ at (x-ε, y)
            let lap_xm = (psi_c - 2.0 * psi_xm + psi_x2m) / (eps * eps)
                + (psi_xm_yp - 2.0 * psi_xm + psi_xm_ym) / (eps * eps);

            let omega_x = -((lap_xp) - (lap_xm)) / (2.0 * eps);
            // Similarly for omega_y (simplified)
            let lap_yp = (psi_xp_yp - 2.0 * psi_yp + psi_xm_yp) / (eps * eps)
                + (psi_y2p - 2.0 * psi_yp + psi_c) / (eps * eps);
            let lap_ym = (psi_xp_ym - 2.0 * psi_ym + psi_xm_ym) / (eps * eps)
                + (psi_c - 2.0 * psi_ym + psi_y2m) / (eps * eps);
            let omega_y = -((lap_yp) - (lap_ym)) / (2.0 * eps);

            // Also need ∇²ω for diffusion: approximate using omega values
            let omega_xp = -lap_xp; let omega_xm = -lap_xm;
            let omega_yp = -lap_yp; let omega_ym = -lap_ym;
            let omega_xx = (omega_xp - 2.0 * omega + omega_xm) / (eps * eps);
            let omega_yy = (omega_yp - 2.0 * omega + omega_ym) / (eps * eps);

            // Residual: -ν·(ω_xx + ω_yy) + u·ω_x + v·ω_y = 0
            let r = -config.nu * (omega_xx + omega_yy) + u_vel * omega_x + v_vel * omega_y;
            let r2 = graph.input(Tensor::from_vec_unchecked(vec![r * r], &[1, 1]));
            phys_nodes.push(r2);
        }
        let mut phys_sum = phys_nodes[0];
        for &n in &phys_nodes[1..] { phys_sum = graph.add(phys_sum, n); }
        let _tmp_pinn_16_ = graph.input(Tensor::from_vec_unchecked(vec![config.n_collocation as f64], &[1]));
        let phys_loss = graph.div(phys_sum, _tmp_pinn_16_);

        // BC: ψ = 0 on all walls (no-slip → u=v=0 on bottom/left/right, u=1 on top)
        // For stream function: ψ=0 on all walls, and ψ_y(top)=1 for lid velocity
        let n_bc_pts = boundary.len() / 2;
        let mut bc_nodes = Vec::new();
        for j in 0..n_bc_pts {
            let bx = boundary[j * 2]; let by = boundary[j * 2 + 1];
            let _inp_bc_idx_ = graph.input(Tensor::from_vec_unchecked(vec![bx, by], &[1, 2]));
            let bc_idx = mlp_forward(&mut graph, &mlp, _inp_bc_idx_);
            let psi = graph.tensor(bc_idx).to_vec()[0];
            // ψ = 0 on boundary
            let r2 = graph.input(Tensor::from_vec_unchecked(vec![psi * psi], &[1, 1]));
            bc_nodes.push(r2);
            // On top wall (y≈1): ψ_y = 1 (lid velocity)
            if (by - 1.0).abs() < 0.01 {
                let _inp_bc_yp_idx_ = graph.input(Tensor::from_vec_unchecked(vec![bx, by + eps], &[1, 2]));
                let bc_yp_idx = mlp_forward(&mut graph, &mlp, _inp_bc_yp_idx_);
                let psi_yp = graph.tensor(bc_yp_idx).to_vec()[0];
                let _inp_bc_ym_idx_ = graph.input(Tensor::from_vec_unchecked(vec![bx, by - eps], &[1, 2]));
                let bc_ym_idx = mlp_forward(&mut graph, &mlp, _inp_bc_ym_idx_);
                let psi_ym = graph.tensor(bc_ym_idx).to_vec()[0];
                let psi_y = (psi_yp - psi_ym) / (2.0 * eps);
                let lid_err = psi_y - 1.0;
                let lid_r2 = graph.input(Tensor::from_vec_unchecked(vec![lid_err * lid_err], &[1, 1]));
                bc_nodes.push(lid_r2);
            }
        }
        let mut bc_sum = bc_nodes[0];
        for &n in &bc_nodes[1..] { bc_sum = graph.add(bc_sum, n); }
        let _tmp_pinn_17_ = graph.input(Tensor::from_vec_unchecked(vec![bc_nodes.len() as f64], &[1]));
        let bc_loss = graph.div(bc_sum, _tmp_pinn_17_);

        let bw = (epoch as f64 / (config.epochs as f64 * 0.2)).min(1.0);
        let _phys_s_ = graph.scalar_mul(phys_loss, config.physics_weight);
        let _bc_s_ = graph.scalar_mul(bc_loss, config.boundary_weight * bw);
        let total = graph.add(_phys_s_, _bc_s_);

        graph.backward(total);
        let mut flat_grads = Vec::with_capacity(total_params);
        for idx in &p_indices {
            if let Some(g) = graph.grad(*idx) { flat_grads.extend_from_slice(&g.to_vec()); }
            else { flat_grads.extend(std::iter::repeat(0.0).take(graph.tensor(*idx).to_vec().len())); }
        }
        let mut flat_params: Vec<f64> = params.iter().flat_map(|p| p.clone()).collect();
        let mut gnorm_acc = KahanAccumulatorF64::new();
        for &g in &flat_grads { gnorm_acc.add(g * g); }
        history.push(TrainLog {
            epoch, total_loss: graph.tensor(total).to_vec()[0], data_loss: 0.0,
            physics_loss: graph.tensor(phys_loss).to_vec()[0], boundary_loss: graph.tensor(bc_loss).to_vec()[0],
            grad_norm: gnorm_acc.finalize().sqrt(),
        });
        adam_step(&mut flat_params, &flat_grads, &mut adam);
        let mut offset = 0;
        for p in params.iter_mut() {
            let _plen_ = p.len();
            p.copy_from_slice(&flat_params[offset..offset + _plen_]);
            offset += _plen_;
        }
    }

    let mean_residual = history.last().map_or(0.0, |h| h.physics_loss);
    PinnResult { history, final_params: params.into_iter().flatten().collect(), l2_error: None, max_error: None, mean_residual }
}

// ---------------------------------------------------------------------------
// Problem M: PINN — 2D Burgers' Equation
// ---------------------------------------------------------------------------
// PDE: u_t + u·u_x + u·u_y = ν·(u_xx + u_yy)

/// Configuration for the 2D Burgers PINN.
#[derive(Debug, Clone)]
pub struct Burgers2DConfig {
    pub layer_sizes: Vec<usize>,
    pub epochs: usize,
    pub lr: f64,
    pub nu: f64,
    pub physics_weight: f64,
    pub boundary_weight: f64,
    pub seed: u64,
    pub n_collocation: usize,
    pub n_ic: usize,
    pub n_bc: usize,
    pub fd_eps: f64,
}

impl Default for Burgers2DConfig {
    fn default() -> Self {
        Self {
            layer_sizes: vec![3, 20, 20, 20, 1], // 3D input: (x, y, t)
            epochs: 500, lr: 1e-3, nu: 0.01 / std::f64::consts::PI,
            physics_weight: 1.0, boundary_weight: 10.0,
            seed: 42, n_collocation: 64, n_ic: 50, n_bc: 25, fd_eps: 1e-4,
        }
    }
}

/// Train a PINN for the 2D Burgers equation.
pub fn pinn_burgers_2d_train(config: &Burgers2DConfig) -> PinnResult {
    let domain = PinnDomain::SpaceTime2D {
        x_range: (-1.0, 1.0), y_range: (-1.0, 1.0), t_range: (0.0, 1.0),
    };
    let interior = domain.sample_interior(config.n_collocation, config.seed);
    let eps = config.fd_eps;

    // IC: u(x,y,0) = -sin(πx)·sin(πy)
    let mut ic_pts = Vec::with_capacity(config.n_ic * 3);
    let mut ic_vals = Vec::with_capacity(config.n_ic);
    let mut rng = cjc_repro::Rng::seeded(config.seed + 100);
    for _ in 0..config.n_ic {
        let x = -1.0 + 2.0 * rng.next_f64();
        let y = -1.0 + 2.0 * rng.next_f64();
        ic_pts.push(x); ic_pts.push(y); ic_pts.push(0.0);
        ic_vals.push(-(std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sin());
    }

    let mut temp_graph = crate::GradGraph::new();
    let (mlp_spec, _tp) = mlp_init(&mut temp_graph, &config.layer_sizes, Activation::Tanh, Activation::None, config.seed);
    let mut params: Vec<Vec<f64>> = mlp_spec.layers.iter().enumerate().flat_map(|(li, _)| {
        vec![temp_graph.tensor(_tp[li * 2]).to_vec(), temp_graph.tensor(_tp[li * 2 + 1]).to_vec()]
    }).collect();
    let total_params: usize = params.iter().map(|p| p.len()).sum();
    let mut adam = AdamState::new(total_params, config.lr);
    let mut history = Vec::with_capacity(config.epochs);

    for epoch in 0..config.epochs {
        let lr_min = config.lr * 0.01;
        adam.lr = lr_min + (config.lr - lr_min) * 0.5 * (1.0 + (std::f64::consts::PI * epoch as f64 / config.epochs as f64).cos());

        let mut graph = crate::GradGraph::new();
        let mut p_indices = Vec::new();
        for (li, layer) in mlp_spec.layers.iter().enumerate() {
            let w = Tensor::from_vec_unchecked(params[li * 2].clone(), &[layer.out_features, layer.in_features]);
            let b = Tensor::from_vec_unchecked(params[li * 2 + 1].clone(), &[layer.out_features]);
            p_indices.push(graph.parameter(w));
            p_indices.push(graph.parameter(b));
        }
        let mlp = Mlp { layers: mlp_spec.layers.iter().enumerate().map(|(li, l)| DenseLayer {
            weight_idx: p_indices[li * 2], bias_idx: p_indices[li * 2 + 1], activation: l.activation,
            in_features: l.in_features, out_features: l.out_features,
        }).collect() };

        // IC
        let mut ic_nodes = Vec::new();
        for j in 0..config.n_ic {
            let x = ic_pts[j * 3]; let y = ic_pts[j * 3 + 1];
            let _inp_ic_idx_ = graph.input(Tensor::from_vec_unchecked(vec![x, y, 0.0], &[1, 3]));
            let ic_idx = mlp_forward(&mut graph, &mlp, _inp_ic_idx_);
            let pred = graph.tensor(ic_idx).to_vec()[0];
            let err = pred - ic_vals[j];
            ic_nodes.push(graph.input(Tensor::from_vec_unchecked(vec![err * err], &[1, 1])));
        }
        let mut ic_sum = ic_nodes[0];
        for &n in &ic_nodes[1..] { ic_sum = graph.add(ic_sum, n); }
        let _tmp_pinn_18_ = graph.input(Tensor::from_vec_unchecked(vec![config.n_ic as f64], &[1]));
        let ic_loss = graph.div(ic_sum, _tmp_pinn_18_);

        // Physics: u_t + u·u_x + u·u_y - ν·(u_xx + u_yy) = 0
        let mut phys_nodes = Vec::new();
        for i in 0..config.n_collocation {
            let x = interior[i * 3]; let y = interior[i * 3 + 1]; let t = interior[i * 3 + 2];
            let _inp_phys_c_idx_ = graph.input(Tensor::from_vec_unchecked(vec![x, y, t], &[1, 3]));
            let phys_c_idx = mlp_forward(&mut graph, &mlp, _inp_phys_c_idx_);
            let u_c = graph.tensor(phys_c_idx).to_vec()[0];
            let _inp_phys_xp_idx_ = graph.input(Tensor::from_vec_unchecked(vec![x + eps, y, t], &[1, 3]));
            let phys_xp_idx = mlp_forward(&mut graph, &mlp, _inp_phys_xp_idx_);
            let u_xp = graph.tensor(phys_xp_idx).to_vec()[0];
            let _inp_phys_xm_idx_ = graph.input(Tensor::from_vec_unchecked(vec![x - eps, y, t], &[1, 3]));
            let phys_xm_idx = mlp_forward(&mut graph, &mlp, _inp_phys_xm_idx_);
            let u_xm = graph.tensor(phys_xm_idx).to_vec()[0];
            let _inp_phys_yp_idx_ = graph.input(Tensor::from_vec_unchecked(vec![x, y + eps, t], &[1, 3]));
            let phys_yp_idx = mlp_forward(&mut graph, &mlp, _inp_phys_yp_idx_);
            let u_yp = graph.tensor(phys_yp_idx).to_vec()[0];
            let _inp_phys_ym_idx_ = graph.input(Tensor::from_vec_unchecked(vec![x, y - eps, t], &[1, 3]));
            let phys_ym_idx = mlp_forward(&mut graph, &mlp, _inp_phys_ym_idx_);
            let u_ym = graph.tensor(phys_ym_idx).to_vec()[0];
            let _inp_phys_tp_idx_ = graph.input(Tensor::from_vec_unchecked(vec![x, y, t + eps], &[1, 3]));
            let phys_tp_idx = mlp_forward(&mut graph, &mlp, _inp_phys_tp_idx_);
            let u_tp = graph.tensor(phys_tp_idx).to_vec()[0];
            let _inp_phys_tm_idx_ = graph.input(Tensor::from_vec_unchecked(vec![x, y, t - eps], &[1, 3]));
            let phys_tm_idx = mlp_forward(&mut graph, &mlp, _inp_phys_tm_idx_);
            let u_tm = graph.tensor(phys_tm_idx).to_vec()[0];

            let u_x = (u_xp - u_xm) / (2.0 * eps);
            let u_y = (u_yp - u_ym) / (2.0 * eps);
            let u_t = (u_tp - u_tm) / (2.0 * eps);
            let u_xx = (u_xp - 2.0 * u_c + u_xm) / (eps * eps);
            let u_yy = (u_yp - 2.0 * u_c + u_ym) / (eps * eps);

            let r = u_t + u_c * u_x + u_c * u_y - config.nu * (u_xx + u_yy);
            phys_nodes.push(graph.input(Tensor::from_vec_unchecked(vec![r * r], &[1, 1])));
        }
        let mut phys_sum = phys_nodes[0];
        for &n in &phys_nodes[1..] { phys_sum = graph.add(phys_sum, n); }
        let _tmp_pinn_19_ = graph.input(Tensor::from_vec_unchecked(vec![config.n_collocation as f64], &[1]));
        let phys_loss = graph.div(phys_sum, _tmp_pinn_19_);

        let bw = (epoch as f64 / (config.epochs as f64 * 0.2)).min(1.0);
        let _phys_s_ = graph.scalar_mul(phys_loss, config.physics_weight);
        let _ic_s_ = graph.scalar_mul(ic_loss, config.boundary_weight * bw);
        let total = graph.add(_phys_s_, _ic_s_);

        graph.backward(total);
        let mut flat_grads = Vec::with_capacity(total_params);
        for idx in &p_indices {
            if let Some(g) = graph.grad(*idx) { flat_grads.extend_from_slice(&g.to_vec()); }
            else { flat_grads.extend(std::iter::repeat(0.0).take(graph.tensor(*idx).to_vec().len())); }
        }
        let mut flat_params: Vec<f64> = params.iter().flat_map(|p| p.clone()).collect();
        let mut gnorm_acc = KahanAccumulatorF64::new();
        for &g in &flat_grads { gnorm_acc.add(g * g); }
        history.push(TrainLog {
            epoch, total_loss: graph.tensor(total).to_vec()[0], data_loss: graph.tensor(ic_loss).to_vec()[0],
            physics_loss: graph.tensor(phys_loss).to_vec()[0], boundary_loss: 0.0,
            grad_norm: gnorm_acc.finalize().sqrt(),
        });
        adam_step(&mut flat_params, &flat_grads, &mut adam);
        let mut offset = 0;
        for p in params.iter_mut() {
            let _plen_ = p.len();
            p.copy_from_slice(&flat_params[offset..offset + _plen_]);
            offset += _plen_;
        }
    }

    let mean_residual = history.last().map_or(0.0, |h| h.physics_loss);
    PinnResult { history, final_params: params.into_iter().flatten().collect(), l2_error: None, max_error: None, mean_residual }
}

// ---------------------------------------------------------------------------
// Inverse Problem Infrastructure
// ---------------------------------------------------------------------------

/// Configuration for inverse PINN problems (parameter discovery).
///
/// Given observational data and a PDE with unknown coefficients, the PINN
/// learns both the solution field AND the unknown PDE parameters simultaneously.
#[derive(Debug, Clone)]
pub struct InversePinnConfig {
    pub layer_sizes: Vec<usize>,
    pub epochs: usize,
    pub lr: f64,
    pub param_lr: f64,
    pub physics_weight: f64,
    pub seed: u64,
    pub n_collocation: usize,
    pub fd_eps: f64,
}

impl Default for InversePinnConfig {
    fn default() -> Self {
        Self {
            layer_sizes: vec![2, 20, 20, 20, 1],
            epochs: 500, lr: 1e-3, param_lr: 1e-2,
            physics_weight: 1.0, seed: 42, n_collocation: 64, fd_eps: 1e-4,
        }
    }
}

/// Result of an inverse PINN problem.
#[derive(Debug, Clone)]
pub struct InversePinnResult {
    /// Standard PINN training result.
    pub pinn_result: PinnResult,
    /// Discovered parameter values.
    pub discovered_params: Vec<f64>,
    /// Parameter names.
    pub param_names: Vec<String>,
}

/// Train an inverse PINN for the diffusion equation u_t = λ·u_xx,
/// discovering the diffusion coefficient λ from data.
///
/// Given observation data `(x, t, u_obs)`, simultaneously learns the solution
/// field and the unknown λ.
pub fn inverse_diffusion_train(
    config: &InversePinnConfig,
    obs_x: &[f64],
    obs_t: &[f64],
    obs_u: &[f64],
) -> InversePinnResult {
    let x_range = (0.0, 1.0);
    let t_range = (0.0, 1.0);
    let eps = config.fd_eps;
    let n_obs = obs_x.len();

    let domain = PinnDomain::SpaceTime1D { x_range, t_range };
    let interior = domain.sample_interior(config.n_collocation, config.seed);

    let mut temp_graph = crate::GradGraph::new();
    let (mlp_spec, _tp) = mlp_init(&mut temp_graph, &config.layer_sizes, Activation::Tanh, Activation::None, config.seed);
    let mut params: Vec<Vec<f64>> = mlp_spec.layers.iter().enumerate().flat_map(|(li, _)| {
        vec![temp_graph.tensor(_tp[li * 2]).to_vec(), temp_graph.tensor(_tp[li * 2 + 1]).to_vec()]
    }).collect();
    let total_params: usize = params.iter().map(|p| p.len()).sum();
    let mut adam = AdamState::new(total_params + 1, config.lr); // +1 for λ

    // Unknown parameter: λ (initialized to 0.1)
    let mut lambda = 0.1f64;
    let mut history = Vec::with_capacity(config.epochs);

    for epoch in 0..config.epochs {
        let lr_min = config.lr * 0.01;
        adam.lr = lr_min + (config.lr - lr_min) * 0.5 * (1.0 + (std::f64::consts::PI * epoch as f64 / config.epochs as f64).cos());

        let mut graph = crate::GradGraph::new();
        let mut p_indices = Vec::new();
        for (li, layer) in mlp_spec.layers.iter().enumerate() {
            let w = Tensor::from_vec_unchecked(params[li * 2].clone(), &[layer.out_features, layer.in_features]);
            let b = Tensor::from_vec_unchecked(params[li * 2 + 1].clone(), &[layer.out_features]);
            p_indices.push(graph.parameter(w));
            p_indices.push(graph.parameter(b));
        }
        let mlp = Mlp { layers: mlp_spec.layers.iter().enumerate().map(|(li, l)| DenseLayer {
            weight_idx: p_indices[li * 2], bias_idx: p_indices[li * 2 + 1], activation: l.activation,
            in_features: l.in_features, out_features: l.out_features,
        }).collect() };

        // Data loss: MSE(u_nn(obs_x, obs_t) - obs_u)
        let mut data_loss_val = 0.0;
        for j in 0..n_obs {
            let inp = graph.input(Tensor::from_vec_unchecked(vec![obs_x[j], obs_t[j]], &[1, 2]));
            let pred = mlp_forward(&mut graph, &mlp, inp);
            let pred_val = graph.tensor(pred).to_vec()[0];
            let err = pred_val - obs_u[j];
            data_loss_val += err * err;
        }
        data_loss_val /= n_obs as f64;

        // Physics: u_t - λ·u_xx = 0 (FD residual, non-differentiable through graph for λ)
        let mut phys_loss_val = 0.0;
        let mut lambda_grad = 0.0;
        for i in 0..config.n_collocation {
            let x = interior[i * 2]; let t = interior[i * 2 + 1];
            let _inp_inv_c_idx_ = graph.input(Tensor::from_vec_unchecked(vec![x, t], &[1, 2]));
            let inv_c_idx = mlp_forward(&mut graph, &mlp, _inp_inv_c_idx_);
            let u_c = graph.tensor(inv_c_idx).to_vec()[0];
            let _inp_inv_xp_idx_ = graph.input(Tensor::from_vec_unchecked(vec![x + eps, t], &[1, 2]));
            let inv_xp_idx = mlp_forward(&mut graph, &mlp, _inp_inv_xp_idx_);
            let u_xp = graph.tensor(inv_xp_idx).to_vec()[0];
            let _inp_inv_xm_idx_ = graph.input(Tensor::from_vec_unchecked(vec![x - eps, t], &[1, 2]));
            let inv_xm_idx = mlp_forward(&mut graph, &mlp, _inp_inv_xm_idx_);
            let u_xm = graph.tensor(inv_xm_idx).to_vec()[0];
            let _inp_inv_tp_idx_ = graph.input(Tensor::from_vec_unchecked(vec![x, t + eps], &[1, 2]));
            let inv_tp_idx = mlp_forward(&mut graph, &mlp, _inp_inv_tp_idx_);
            let u_tp = graph.tensor(inv_tp_idx).to_vec()[0];
            let _inp_inv_tm_idx_ = graph.input(Tensor::from_vec_unchecked(vec![x, t - eps], &[1, 2]));
            let inv_tm_idx = mlp_forward(&mut graph, &mlp, _inp_inv_tm_idx_);
            let u_tm = graph.tensor(inv_tm_idx).to_vec()[0];
            let u_t = (u_tp - u_tm) / (2.0 * eps);
            let u_xx = (u_xp - 2.0 * u_c + u_xm) / (eps * eps);
            let r = u_t - lambda * u_xx;
            phys_loss_val += r * r;
            // d(r²)/d(λ) = 2·r·(-u_xx)
            lambda_grad += 2.0 * r * (-u_xx) / config.n_collocation as f64;
        }
        phys_loss_val /= config.n_collocation as f64;

        let total_loss = data_loss_val + config.physics_weight * phys_loss_val;

        // Backward for NN params (through data loss only — physics is FD)
        let data_node = graph.input(Tensor::from_vec_unchecked(vec![data_loss_val], &[1]));
        graph.backward(data_node);

        let mut flat_grads = Vec::with_capacity(total_params + 1);
        for idx in &p_indices {
            if let Some(g) = graph.grad(*idx) { flat_grads.extend_from_slice(&g.to_vec()); }
            else { flat_grads.extend(std::iter::repeat(0.0).take(graph.tensor(*idx).to_vec().len())); }
        }
        flat_grads.push(lambda_grad * config.physics_weight);

        let mut flat_params_all: Vec<f64> = params.iter().flat_map(|p| p.clone()).collect();
        flat_params_all.push(lambda);

        let mut gnorm_acc = KahanAccumulatorF64::new();
        for &g in &flat_grads { gnorm_acc.add(g * g); }
        history.push(TrainLog {
            epoch, total_loss, data_loss: data_loss_val,
            physics_loss: phys_loss_val, boundary_loss: 0.0,
            grad_norm: gnorm_acc.finalize().sqrt(),
        });

        adam_step(&mut flat_params_all, &flat_grads, &mut adam);
        let mut offset = 0;
        for p in params.iter_mut() {
            let _plen_ = p.len();
            p.copy_from_slice(&flat_params_all[offset..offset + _plen_]);
            offset += _plen_;
        }
        lambda = flat_params_all[offset];
    }

    InversePinnResult {
        pinn_result: PinnResult {
            history, final_params: params.into_iter().flatten().collect(),
            l2_error: None, max_error: None,
            mean_residual: 0.0,
        },
        discovered_params: vec![lambda],
        param_names: vec!["lambda (diffusion coefficient)".into()],
    }
}

// ---------------------------------------------------------------------------
// L-BFGS Optimizer
// ---------------------------------------------------------------------------

/// L-BFGS optimizer state for second-order optimization.
///
/// Stores the last `m` correction pairs `(s, y)` for the two-loop recursion
/// that approximates the inverse Hessian-vector product. Deterministic: no
/// HashMap, no thread-local state, all reductions use Kahan summation.
#[derive(Debug, Clone)]
pub struct LbfgsState {
    /// Maximum number of correction pairs to store.
    pub m: usize,
    /// Ring buffer of `s` vectors (parameter differences: x_{k+1} - x_k).
    s_history: Vec<Vec<f64>>,
    /// Ring buffer of `y` vectors (gradient differences: g_{k+1} - g_k).
    y_history: Vec<Vec<f64>>,
    /// Previous parameters (for computing s = x_new - x_old).
    prev_params: Option<Vec<f64>>,
    /// Previous gradient (for computing y = g_new - g_old).
    prev_grad: Option<Vec<f64>>,
    /// Learning rate (step size multiplier for the search direction).
    pub lr: f64,
    /// Maximum gradient norm for clipping.
    pub max_grad_norm: f64,
}

impl LbfgsState {
    /// Create a new L-BFGS state with `m` correction pairs.
    pub fn new(m: usize, lr: f64) -> Self {
        Self {
            m,
            s_history: Vec::with_capacity(m),
            y_history: Vec::with_capacity(m),
            prev_params: None,
            prev_grad: None,
            lr,
            max_grad_norm: 10.0,
        }
    }

    /// Kahan dot product of two vectors.
    fn dot_kahan(a: &[f64], b: &[f64]) -> f64 {
        let mut acc = KahanAccumulatorF64::new();
        for (&ai, &bi) in a.iter().zip(b.iter()) {
            acc.add(ai * bi);
        }
        acc.finalize()
    }

    /// Perform one L-BFGS step: compute search direction via two-loop recursion,
    /// then update params.
    ///
    /// Returns `true` if a valid step was taken, `false` if the curvature
    /// condition was violated (sTy <= 0) and a steepest-descent fallback was used.
    pub fn step(&mut self, params: &mut [f64], grad: &[f64]) -> bool {
        let n = params.len();

        // Clip gradient norm
        let mut gnorm_acc = KahanAccumulatorF64::new();
        for &g in grad.iter() {
            gnorm_acc.add(g * g);
        }
        let gnorm = gnorm_acc.finalize().sqrt();
        let scale = if gnorm > self.max_grad_norm { self.max_grad_norm / gnorm } else { 1.0 };
        let clipped_grad: Vec<f64> = grad.iter().map(|&g| g * scale).collect();

        // Compute correction pairs if we have previous state
        if let (Some(prev_p), Some(prev_g)) = (&self.prev_params, &self.prev_grad) {
            let s: Vec<f64> = params.iter().zip(prev_p.iter()).map(|(&x, &xp)| x - xp).collect();
            let y: Vec<f64> = clipped_grad.iter().zip(prev_g.iter()).map(|(&g, &gp)| g - gp).collect();
            let ys = Self::dot_kahan(&y, &s);

            // Only store if curvature condition holds (sTy > 0)
            if ys > 1e-10 {
                if self.s_history.len() == self.m {
                    self.s_history.remove(0);
                    self.y_history.remove(0);
                }
                self.s_history.push(s);
                self.y_history.push(y);
            }
        }

        // Two-loop recursion to compute H_k * grad
        let k = self.s_history.len();
        let mut q = clipped_grad.clone();
        let mut alpha = vec![0.0f64; k];
        let mut rho = vec![0.0f64; k];
        let mut used_bfgs = true;

        // First loop: newest to oldest
        for i in (0..k).rev() {
            let ys = Self::dot_kahan(&self.y_history[i], &self.s_history[i]);
            if ys.abs() < 1e-30 {
                used_bfgs = false;
                break;
            }
            rho[i] = 1.0 / ys;
            alpha[i] = rho[i] * Self::dot_kahan(&self.s_history[i], &q);
            for j in 0..n {
                q[j] -= alpha[i] * self.y_history[i][j];
            }
        }

        let mut r = if used_bfgs && k > 0 {
            // Scale initial Hessian approximation: H0 = (sTy / yTy) * I
            let last = k - 1;
            let ys = Self::dot_kahan(&self.y_history[last], &self.s_history[last]);
            let yy = Self::dot_kahan(&self.y_history[last], &self.y_history[last]);
            let gamma = if yy.abs() > 1e-30 { ys / yy } else { 1.0 };
            q.iter().map(|&qi| gamma * qi).collect::<Vec<_>>()
        } else {
            q // Fallback to steepest descent direction
        };

        if used_bfgs && k > 0 {
            // Second loop: oldest to newest
            for i in 0..k {
                let beta = rho[i] * Self::dot_kahan(&self.y_history[i], &r);
                for j in 0..n {
                    r[j] += self.s_history[i][j] * (alpha[i] - beta);
                }
            }
        }

        // Store current state for next iteration
        self.prev_params = Some(params.to_vec());
        self.prev_grad = Some(clipped_grad);

        // Update: x_{k+1} = x_k - lr * H_k * g_k
        for i in 0..n {
            params[i] -= self.lr * r[i];
        }

        used_bfgs && k > 0
    }
}

/// Two-stage optimizer: Adam for first phase, L-BFGS for second phase.
///
/// The Raissi (2019) pattern: Adam provides robust initial convergence,
/// then L-BFGS polishes with quasi-Newton steps for faster local convergence.
pub struct TwoStageOptimizer {
    /// Adam state for the first phase.
    pub adam: AdamState,
    /// L-BFGS state for the second phase.
    pub lbfgs: LbfgsState,
    /// Fraction of epochs to use Adam (e.g. 0.8 = 80% Adam, 20% L-BFGS).
    pub adam_fraction: f64,
    /// Total number of epochs.
    pub total_epochs: usize,
}

impl TwoStageOptimizer {
    /// Create a new two-stage optimizer.
    pub fn new(n_params: usize, lr: f64, total_epochs: usize, adam_fraction: f64) -> Self {
        Self {
            adam: AdamState::new(n_params, lr),
            lbfgs: LbfgsState::new(20, lr * 0.1), // L-BFGS uses lower LR
            adam_fraction,
            total_epochs,
        }
    }

    /// Perform one optimizer step. Returns `true` if using L-BFGS phase.
    pub fn step(&mut self, params: &mut [f64], grad: &[f64], epoch: usize) -> bool {
        let switch_epoch = (self.total_epochs as f64 * self.adam_fraction) as usize;
        if epoch < switch_epoch {
            adam_step(params, grad, &mut self.adam);
            false
        } else {
            self.lbfgs.step(params, grad);
            true
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poly_eval_quadratic() {
        // p(x) = 1 + 2x + 3x² => p(2) = 1 + 4 + 12 = 17
        let coeffs = [1.0, 2.0, 3.0];
        assert!((poly_eval(&coeffs, 2.0) - 17.0).abs() < 1e-12);
    }

    #[test]
    fn test_poly_eval_dd_quadratic() {
        // p(x) = 1 + 2x + 3x² => p''(x) = 6
        let coeffs = [1.0, 2.0, 3.0];
        assert!((poly_eval_dd(&coeffs, 0.5) - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_poly_eval_dd_cubic() {
        // p(x) = x³ => coeffs = [0, 0, 0, 1], p''(x) = 6x
        let coeffs = [0.0, 0.0, 0.0, 1.0];
        assert!((poly_eval_dd(&coeffs, 2.0) - 12.0).abs() < 1e-12);
    }

    #[test]
    fn test_heat_data_deterministic() {
        let (x1, u1) = heat_1d_generate_data(10, 0.01, 42);
        let (x2, u2) = heat_1d_generate_data(10, 0.01, 42);
        assert_eq!(x1, x2);
        assert_eq!(u1, u2);
    }

    #[test]
    fn test_uniform_grid_bounds() {
        let g = uniform_grid(0.0, 1.0, 10);
        assert_eq!(g.len(), 10);
        for &x in &g {
            assert!(x > 0.0 && x < 1.0);
        }
    }

    #[test]
    fn test_lhs_grid_1d_bounds() {
        let g = lhs_grid_1d(0.0, 1.0, 20, 42);
        assert_eq!(g.len(), 20);
        for &x in &g {
            assert!(x >= 0.0 && x < 1.0, "x={} out of bounds", x);
        }
    }

    #[test]
    fn test_lhs_grid_1d_deterministic() {
        let g1 = lhs_grid_1d(0.0, 1.0, 20, 42);
        let g2 = lhs_grid_1d(0.0, 1.0, 20, 42);
        assert_eq!(g1, g2);
    }

    #[test]
    fn test_mlp_init_shapes() {
        let mut graph = crate::GradGraph::new();
        let (mlp, params) = mlp_init(
            &mut graph,
            &[1, 8, 8, 1],
            Activation::Tanh,
            Activation::None,
            42,
        );
        // 3 layers, 2 params (W, b) each = 6 param nodes
        assert_eq!(mlp.layers.len(), 3);
        assert_eq!(params.len(), 6);
        // First weight: [8, 1]
        assert_eq!(graph.tensor(params[0]).shape(), &[8, 1]);
        // First bias: [8]
        assert_eq!(graph.tensor(params[1]).shape(), &[8]);
    }

    #[test]
    fn test_mlp_init_deterministic() {
        let mut g1 = crate::GradGraph::new();
        let (_, p1) = mlp_init(&mut g1, &[1, 8, 1], Activation::Tanh, Activation::None, 42);

        let mut g2 = crate::GradGraph::new();
        let (_, p2) = mlp_init(&mut g2, &[1, 8, 1], Activation::Tanh, Activation::None, 42);

        for (&i1, &i2) in p1.iter().zip(p2.iter()) {
            assert_eq!(g1.tensor(i1).to_vec(), g2.tensor(i2).to_vec());
        }
    }

    #[test]
    fn test_mlp_forward_produces_output() {
        let mut graph = crate::GradGraph::new();
        let (mlp, _) = mlp_init(
            &mut graph,
            &[1, 8, 1],
            Activation::Tanh,
            Activation::None,
            42,
        );
        let x = graph.input(Tensor::from_vec_unchecked(vec![0.5], &[1, 1]));
        let y = mlp_forward(&mut graph, &mlp, x);
        let val = graph.value(y);
        assert!(val.is_finite(), "MLP output should be finite");
    }

    #[test]
    fn test_mlp_forward_deterministic() {
        let eval = |seed: u64| -> f64 {
            let mut graph = crate::GradGraph::new();
            let (mlp, _) = mlp_init(
                &mut graph,
                &[1, 16, 16, 1],
                Activation::Tanh,
                Activation::None,
                seed,
            );
            let x = graph.input(Tensor::from_vec_unchecked(vec![0.5], &[1, 1]));
            let y = mlp_forward(&mut graph, &mlp, x);
            graph.value(y)
        };

        let v1 = eval(42);
        let v2 = eval(42);
        assert_eq!(v1, v2, "Same seed must produce identical output");

        let v3 = eval(99);
        assert_ne!(v1, v3, "Different seeds should produce different output");
    }

    #[test]
    fn test_mlp_backward_produces_gradients() {
        let mut graph = crate::GradGraph::new();
        let (mlp, params) = mlp_init(
            &mut graph,
            &[1, 8, 1],
            Activation::Tanh,
            Activation::None,
            42,
        );
        let x = graph.input(Tensor::from_vec_unchecked(vec![0.5], &[1, 1]));
        let y = mlp_forward(&mut graph, &mlp, x);

        let target = graph.input(Tensor::from_vec_unchecked(vec![1.0], &[1, 1]));
        let loss = data_loss_mse(&mut graph, y, target);

        graph.zero_grad();
        graph.backward(loss);

        // All parameters should have non-None gradients
        for &p in &params {
            let grad = graph.grad(p);
            assert!(grad.is_some(), "Parameter should have gradient");
            let g = grad.unwrap();
            let has_nonzero = g.to_vec().iter().any(|&v| v != 0.0);
            assert!(has_nonzero, "Gradient should have nonzero entries");
        }
    }

    #[test]
    fn test_piml_heat_1d_converges() {
        let result = piml_heat_1d_train(
            8,       // degree
            30,      // n_data
            50,      // n_colloc
            0.01,    // noise
            5000,    // epochs (more for convergence)
            1e-3,    // lr
            1.0,     // physics_weight (gentler)
            10.0,    // boundary_weight (gentler)
            42,      // seed
        );

        // Check that loss decreased significantly
        let first_loss = result.history.first().unwrap().total_loss;
        let last_loss = result.history.last().unwrap().total_loss;
        assert!(
            last_loss < first_loss * 0.5,
            "Loss should decrease: first={}, last={}",
            first_loss,
            last_loss,
        );

        // L2 error should be reasonable for a polynomial fit
        let l2 = result.l2_error.unwrap();
        assert!(
            l2 < 0.5,
            "L2 error too large: {} (expected < 0.5)",
            l2,
        );
    }

    #[test]
    fn test_piml_heat_1d_determinism() {
        let r1 = piml_heat_1d_train(6, 20, 30, 0.01, 500, 1e-3, 10.0, 100.0, 42);
        let r2 = piml_heat_1d_train(6, 20, 30, 0.01, 500, 1e-3, 10.0, 100.0, 42);

        assert_eq!(r1.final_params, r2.final_params, "Params must be bit-identical");

        for (h1, h2) in r1.history.iter().zip(r2.history.iter()) {
            assert_eq!(h1.total_loss, h2.total_loss, "Loss trajectory must be identical at epoch {}", h1.epoch);
        }
    }

    #[test]
    fn test_pinn_harmonic_trains() {
        let config = PinnConfig {
            layer_sizes: vec![1, 16, 16, 1],
            epochs: 100,
            lr: 1e-3,
            physics_weight: 1.0,
            boundary_weight: 50.0,
            seed: 42,
            n_collocation: 20,
            n_data: 15,
            fd_eps: 1e-3,
        };

        let result = pinn_harmonic_train(&config);

        // Loss should decrease
        let first = result.history.first().unwrap().total_loss;
        let last = result.history.last().unwrap().total_loss;
        assert!(
            last < first,
            "PINN loss should decrease: first={}, last={}",
            first,
            last,
        );
    }

    #[test]
    fn test_pinn_harmonic_determinism() {
        let config = PinnConfig {
            layer_sizes: vec![1, 8, 8, 1],
            epochs: 50,
            lr: 1e-3,
            physics_weight: 1.0,
            boundary_weight: 50.0,
            seed: 42,
            n_collocation: 10,
            n_data: 10,
            fd_eps: 1e-3,
        };

        let r1 = pinn_harmonic_train(&config);
        let r2 = pinn_harmonic_train(&config);

        assert_eq!(
            r1.final_params, r2.final_params,
            "PINN params must be bit-identical with same seed"
        );

        for (h1, h2) in r1.history.iter().zip(r2.history.iter()) {
            assert_eq!(
                h1.total_loss, h2.total_loss,
                "PINN loss trajectory must be identical at epoch {}",
                h1.epoch,
            );
        }
    }

    #[test]
    fn test_ascii_plot_basic() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let plot = ascii_plot(&x, &y, 40, 10, "x^2");
        assert!(plot.contains("x^2"));
        assert!(plot.contains("*"));
    }

    #[test]
    fn test_plot_loss_history() {
        let history = vec![
            TrainLog { epoch: 0, total_loss: 10.0, data_loss: 5.0, physics_loss: 3.0, boundary_loss: 2.0, grad_norm: 1.0 },
            TrainLog { epoch: 1, total_loss: 5.0, data_loss: 2.5, physics_loss: 1.5, boundary_loss: 1.0, grad_norm: 0.5 },
        ];
        let plot = plot_loss_history(&history, 40, 10);
        assert!(plot.contains("Training Loss"));
    }

    #[test]
    fn test_data_loss_mse_graph() {
        let mut graph = crate::GradGraph::new();
        let pred = graph.input(Tensor::from_vec_unchecked(vec![1.0, 2.0, 3.0], &[3]));
        let target = graph.input(Tensor::from_vec_unchecked(vec![1.0, 2.0, 3.0], &[3]));
        let loss = data_loss_mse(&mut graph, pred, target);
        let val = graph.value(loss);
        assert!((val - 0.0).abs() < 1e-12, "MSE of identical should be 0, got {}", val);
    }

    #[test]
    fn test_data_loss_mse_nonzero() {
        let mut graph = crate::GradGraph::new();
        let pred = graph.input(Tensor::from_vec_unchecked(vec![1.0, 2.0, 3.0], &[3]));
        let target = graph.input(Tensor::from_vec_unchecked(vec![0.0, 0.0, 0.0], &[3]));
        let loss = data_loss_mse(&mut graph, pred, target);
        let val = graph.value(loss);
        // MSE = (1+4+9)/3 = 14/3 ≈ 4.6667
        assert!((val - 14.0 / 3.0).abs() < 1e-10, "MSE wrong: {}", val);
    }

    #[test]
    fn test_heat_source_known_values() {
        // f(0) = -π² * sin(0) = 0
        assert!((heat_source(0.0) - 0.0).abs() < 1e-12);
        // f(0.5) = -π² * sin(π/2) = -π²
        assert!((heat_source(0.5) - (-std::f64::consts::PI.powi(2))).abs() < 1e-10);
    }
}
