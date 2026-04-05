//! QML — Quantum Machine Learning via Data Re-Uploading (QC-REUP).
//!
//! MPS-based quantum neural network with:
//! - Parameterized Rx/Ry/Rz rotations (data encoding + trainable biases)
//! - Adjacent CNOT entanglement (1D MPS-friendly)
//! - Multiple re-upload passes
//! - Softmax classification head via Z expectation values
//! - Parameter-shift gradient training
//!
//! # Determinism
//!
//! - All complex arithmetic uses `mul_fixed` (no FMA)
//! - Reductions use ascending index order
//! - RNG via SplitMix64 with explicit seed threading
//! - Same seed = bit-identical training trajectory

use cjc_runtime::complex::ComplexF64;
use crate::mps::Mps;
use crate::vqe::{transfer_matrix_identity, transfer_matrix_z};

// ---------------------------------------------------------------------------
// Configuration & Result Types
// ---------------------------------------------------------------------------

/// Loss function for QML training.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QmlLoss {
    /// Mean squared error: sum_c (p[c] - one_hot[c])^2 / n_classes.
    Mse,
    /// Cross-entropy: -log(p[true_class]).
    CrossEntropy,
}

/// Configuration for a QC-REUP model.
pub struct QmlConfig {
    /// Number of qubits (must equal input feature dimension).
    pub n_qubits: usize,
    /// Number of data re-upload passes (circuit depth).
    pub n_reupload_passes: usize,
    /// Number of classification classes.
    pub n_classes: usize,
    /// Maximum MPS bond dimension (controls accuracy vs. memory).
    pub max_bond: usize,
    /// Qubit indices used for Z-expectation readout.
    pub readout_qubits: Vec<usize>,
    /// Gradient descent step size.
    pub learning_rate: f64,
    /// Number of training epochs.
    pub epochs: usize,
    /// Mini-batch size for gradient computation.
    pub batch_size: usize,
    /// Loss function to optimize.
    pub loss: QmlLoss,
    /// Seed for deterministic parameter initialization and shuffling.
    pub seed: u64,
}

/// Result of QML training.
pub struct QmlResult {
    /// Optimized trainable parameters.
    pub params: Vec<f64>,
    /// Loss value at each epoch (including initial).
    pub loss_history: Vec<f64>,
    /// Classification accuracy at each epoch (including initial).
    pub accuracy_history: Vec<f64>,
    /// Accuracy after the final epoch.
    pub final_accuracy: f64,
    /// Number of training epochs completed.
    pub epochs_completed: usize,
}

/// A labeled dataset for classification.
pub struct QmlDataset {
    /// Input feature vectors (one per sample, length = n_qubits).
    pub samples: Vec<Vec<f64>>,
    /// Class labels for each sample.
    pub labels: Vec<usize>,
    /// Total number of distinct classes.
    pub n_classes: usize,
}

// ---------------------------------------------------------------------------
// Rotation Matrix Builders
// ---------------------------------------------------------------------------

fn rx_mat(theta: f64) -> [[ComplexF64; 2]; 2] {
    let c = ComplexF64::real((theta / 2.0).cos());
    let s = ComplexF64::new(0.0, -(theta / 2.0).sin());
    [[c, s], [s, c]]
}

fn ry_mat(theta: f64) -> [[ComplexF64; 2]; 2] {
    let c = ComplexF64::real((theta / 2.0).cos());
    let s = ComplexF64::real((theta / 2.0).sin());
    [[c, ComplexF64::real(-s.re)], [s, c]]
}

fn rz_mat(theta: f64) -> [[ComplexF64; 2]; 2] {
    let half = theta / 2.0;
    let e_neg = ComplexF64::new(half.cos(), -half.sin());
    let e_pos = ComplexF64::new(half.cos(), half.sin());
    [[e_neg, ComplexF64::ZERO], [ComplexF64::ZERO, e_pos]]
}

// ---------------------------------------------------------------------------
// Parameter Layout
// ---------------------------------------------------------------------------

/// Total number of trainable parameters.
///
/// Per qubit per re-upload pass: 6 params (3 weights + 3 biases for Rx, Ry, Rz).
pub fn total_params(config: &QmlConfig) -> usize {
    6 * config.n_qubits * config.n_reupload_passes
}

/// Index into the flat parameter vector.
/// gate: 0=w_rx, 1=w_ry, 2=w_rz, 3=b_rx, 4=b_ry, 5=b_rz
fn param_index(layer: usize, qubit: usize, gate: usize, n_qubits: usize) -> usize {
    (layer * n_qubits * 6) + (qubit * 6) + gate
}

// ---------------------------------------------------------------------------
// Single-Qubit Z Expectation
// ---------------------------------------------------------------------------

/// Compute ⟨ψ|Z_q|ψ⟩ for a single qubit in an MPS state.
pub fn mps_single_z_expectation(mps: &Mps, qubit: usize) -> f64 {
    let n = mps.n_qubits;
    assert!(qubit < n, "qubit {} out of range for {}-qubit MPS", qubit, n);

    // Start with 1x1 identity environment
    let mut env = vec![vec![ComplexF64::ONE]];

    // Contract from left to qubit-1 with identity
    for site in 0..qubit {
        env = transfer_matrix_identity(&mps.tensors[site], &env);
    }

    // Apply Z at the target qubit
    env = transfer_matrix_z(&mps.tensors[qubit], &env);

    // Contract remaining sites with identity
    for site in (qubit + 1)..n {
        env = transfer_matrix_identity(&mps.tensors[site], &env);
    }

    env[0][0].re
}

// ---------------------------------------------------------------------------
// Forward Pass: Build QML Circuit
// ---------------------------------------------------------------------------

/// Build the MPS state for a single input sample using QC-REUP.
///
/// Each re-upload pass applies:
/// 1. Rx(w_rx * x + b_rx), Ry(w_ry * x + b_ry), Rz(w_rz * x + b_rz) per qubit
/// 2. CNOT chain: (0,1), (1,2), ..., (N-2, N-1)
pub fn build_qml_circuit(
    config: &QmlConfig,
    params: &[f64],
    input: &[f64],
) -> Mps {
    assert_eq!(input.len(), config.n_qubits,
        "input length {} != n_qubits {}", input.len(), config.n_qubits);
    assert_eq!(params.len(), total_params(config),
        "params length {} != expected {}", params.len(), total_params(config));

    let mut mps = Mps::with_max_bond(config.n_qubits, config.max_bond);

    for layer in 0..config.n_reupload_passes {
        // Data encoding + trainable rotation layer
        for q in 0..config.n_qubits {
            let w_rx = params[param_index(layer, q, 0, config.n_qubits)];
            let w_ry = params[param_index(layer, q, 1, config.n_qubits)];
            let w_rz = params[param_index(layer, q, 2, config.n_qubits)];
            let b_rx = params[param_index(layer, q, 3, config.n_qubits)];
            let b_ry = params[param_index(layer, q, 4, config.n_qubits)];
            let b_rz = params[param_index(layer, q, 5, config.n_qubits)];

            let angle_rx = w_rx * input[q] + b_rx;
            let angle_ry = w_ry * input[q] + b_ry;
            let angle_rz = w_rz * input[q] + b_rz;

            mps.apply_single_qubit(q, rx_mat(angle_rx));
            mps.apply_single_qubit(q, ry_mat(angle_ry));
            mps.apply_single_qubit(q, rz_mat(angle_rz));
        }

        // Entanglement layer: CNOT chain
        if config.n_qubits > 1 {
            for i in 0..(config.n_qubits - 1) {
                mps.apply_cnot_adjacent(i, i + 1);
            }
        }
    }

    mps
}

// ---------------------------------------------------------------------------
// Classification Head
// ---------------------------------------------------------------------------

/// Compute class probabilities from MPS state via softmax over Z expectation values.
pub fn classify(
    mps: &Mps,
    readout_qubits: &[usize],
    n_classes: usize,
) -> Vec<f64> {
    assert!(readout_qubits.len() >= n_classes,
        "need at least {} readout qubits, got {}", n_classes, readout_qubits.len());

    // Compute Z expectations for readout qubits
    let z_values: Vec<f64> = (0..n_classes)
        .map(|c| mps_single_z_expectation(mps, readout_qubits[c]))
        .collect();

    // Softmax: p_c = exp(z_c - max) / sum(exp(z_j - max))
    let max_z = z_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_values: Vec<f64> = z_values.iter().map(|z| (z - max_z).exp()).collect();
    let sum_exp: f64 = exp_values.iter().sum();

    exp_values.iter().map(|e| e / sum_exp).collect()
}

// ---------------------------------------------------------------------------
// Loss Functions
// ---------------------------------------------------------------------------

/// Cross-entropy loss: -log(p[true_class]).
fn cross_entropy_loss(probs: &[f64], label: usize) -> f64 {
    let p = probs[label].max(1e-15); // clamp to avoid log(0)
    -p.ln()
}

/// MSE loss: sum_c (p[c] - one_hot[c])^2 / n_classes.
fn mse_loss(probs: &[f64], label: usize, n_classes: usize) -> f64 {
    let mut sum = 0.0;
    for c in 0..n_classes {
        let target = if c == label { 1.0 } else { 0.0 };
        let diff = probs[c] - target;
        sum += diff * diff;
    }
    sum / n_classes as f64
}

/// Compute loss for a single sample.
fn sample_loss(config: &QmlConfig, params: &[f64], input: &[f64], label: usize) -> f64 {
    let mps = build_qml_circuit(config, params, input);
    let probs = classify(&mps, &config.readout_qubits, config.n_classes);
    match config.loss {
        QmlLoss::CrossEntropy => cross_entropy_loss(&probs, label),
        QmlLoss::Mse => mse_loss(&probs, label, config.n_classes),
    }
}

/// Compute average loss over a batch of samples using the configured loss function.
pub fn batch_loss(
    config: &QmlConfig,
    params: &[f64],
    samples: &[Vec<f64>],
    labels: &[usize],
) -> f64 {
    assert_eq!(samples.len(), labels.len());
    if samples.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    for i in 0..samples.len() {
        total += sample_loss(config, params, &samples[i], labels[i]);
    }
    total / samples.len() as f64
}

/// Predict the class label (argmax of probabilities) for a single input sample.
pub fn predict(config: &QmlConfig, params: &[f64], input: &[f64]) -> usize {
    let mps = build_qml_circuit(config, params, input);
    let probs = classify(&mps, &config.readout_qubits, config.n_classes);
    let mut best = 0;
    let mut best_p = probs[0];
    for c in 1..config.n_classes {
        if probs[c] > best_p {
            best_p = probs[c];
            best = c;
        }
    }
    best
}

/// Compute classification accuracy (fraction correct) over a dataset.
pub fn compute_accuracy(config: &QmlConfig, params: &[f64], dataset: &QmlDataset) -> f64 {
    if dataset.samples.is_empty() {
        return 0.0;
    }
    let mut correct = 0usize;
    for i in 0..dataset.samples.len() {
        if predict(config, params, &dataset.samples[i]) == dataset.labels[i] {
            correct += 1;
        }
    }
    correct as f64 / dataset.samples.len() as f64
}

// ---------------------------------------------------------------------------
// Parameter-Shift Gradient
// ---------------------------------------------------------------------------

/// Compute gradient of average batch loss w.r.t. all parameters.
///
/// Uses central finite-difference: ∂L/∂θ_k ≈ (L(θ_k + ε) - L(θ_k - ε)) / (2ε).
///
/// Note: The parameter-shift rule (pi/2 shift) is exact for bias parameters
/// (direct rotation angles), but weight parameters multiply with input data,
/// making the effective angle shift input-dependent. Finite difference works
/// correctly for all parameter types.
pub fn qml_gradient(
    config: &QmlConfig,
    params: &[f64],
    samples: &[Vec<f64>],
    labels: &[usize],
) -> Vec<f64> {
    let eps = 1e-4;
    let n_params = params.len();
    let mut grads = vec![0.0; n_params];

    for k in 0..n_params {
        let mut params_plus = params.to_vec();
        let mut params_minus = params.to_vec();
        params_plus[k] += eps;
        params_minus[k] -= eps;

        let loss_plus = batch_loss(config, &params_plus, samples, labels);
        let loss_minus = batch_loss(config, &params_minus, samples, labels);

        grads[k] = (loss_plus - loss_minus) / (2.0 * eps);
    }

    grads
}

// ---------------------------------------------------------------------------
// Data Preprocessing
// ---------------------------------------------------------------------------

/// 1D snake (boustrophedon) ordering of a 2D image grid.
///
/// Row 0: left-to-right, row 1: right-to-left, etc.
/// Returns (row, col) coordinates in traversal order.
pub fn snake_order(width: usize, height: usize) -> Vec<(usize, usize)> {
    let mut coords = Vec::with_capacity(width * height);
    for row in 0..height {
        if row % 2 == 0 {
            for col in 0..width {
                coords.push((row, col));
            }
        } else {
            for col in (0..width).rev() {
                coords.push((row, col));
            }
        }
    }
    coords
}

/// Downsample an image to n_qubits features using snake ordering + average pooling.
///
/// Input: grayscale pixels (0-255), row-major.
/// Output: n_qubits values in [0, pi].
pub fn preprocess_image(
    pixels: &[u8],
    width: usize,
    height: usize,
    n_qubits: usize,
) -> Vec<f64> {
    assert_eq!(pixels.len(), width * height, "pixel count mismatch");
    assert!(n_qubits > 0, "n_qubits must be > 0");

    let order = snake_order(width, height);
    let n_pixels = order.len();

    // Average-pool into n_qubits bins
    let mut bins = vec![0.0f64; n_qubits];
    let mut counts = vec![0usize; n_qubits];

    for (i, &(row, col)) in order.iter().enumerate() {
        let bin = i * n_qubits / n_pixels;
        let bin = bin.min(n_qubits - 1);
        bins[bin] += pixels[row * width + col] as f64;
        counts[bin] += 1;
    }

    // Normalize: average, then scale to [0, pi]
    for i in 0..n_qubits {
        if counts[i] > 0 {
            bins[i] = (bins[i] / counts[i] as f64) / 255.0 * std::f64::consts::PI;
        }
    }

    bins
}

/// Build a `QmlDataset` from raw grayscale image bytes and labels.
///
/// Images are preprocessed via snake ordering and average pooling to `n_qubits` features.
pub fn load_dataset(
    image_data: &[u8],
    labels: &[u8],
    width: usize,
    height: usize,
    n_samples: usize,
    n_qubits: usize,
    n_classes: usize,
) -> QmlDataset {
    let img_size = width * height;
    let n = n_samples.min(image_data.len() / img_size).min(labels.len());

    let mut samples = Vec::with_capacity(n);
    let mut labs = Vec::with_capacity(n);

    for i in 0..n {
        let start = i * img_size;
        let pixels = &image_data[start..start + img_size];
        samples.push(preprocess_image(pixels, width, height, n_qubits));
        labs.push(labels[i] as usize % n_classes);
    }

    QmlDataset {
        samples,
        labels: labs,
        n_classes,
    }
}

// ---------------------------------------------------------------------------
// Training Loop
// ---------------------------------------------------------------------------

/// Fisher-Yates shuffle with deterministic RNG.
fn shuffle_indices(indices: &mut [usize], rng_state: &mut u64) {
    let n = indices.len();
    for i in (1..n).rev() {
        let j = (crate::splitmix64(rng_state) as usize) % (i + 1);
        indices.swap(i, j);
    }
}

/// Train the QML model using mini-batch gradient descent with finite-difference gradients.
///
/// Returns a `QmlResult` with optimized parameters and training history.
/// Same seed produces bit-identical training trajectories.
pub fn qml_train(config: &QmlConfig, dataset: &QmlDataset) -> QmlResult {
    let n_params = total_params(config);
    let mut rng_state = config.seed;

    // Initialize parameters: small random values
    let mut params: Vec<f64> = (0..n_params)
        .map(|_| {
            let r = crate::rand_f64(&mut rng_state);
            (r - 0.5) * 0.1
        })
        .collect();

    let n_samples = dataset.samples.len();
    let mut loss_history = Vec::with_capacity(config.epochs);
    let mut accuracy_history = Vec::with_capacity(config.epochs);

    // Record initial metrics
    let init_loss = batch_loss(config, &params, &dataset.samples, &dataset.labels);
    loss_history.push(init_loss);
    let init_acc = compute_accuracy(config, &params, dataset);
    accuracy_history.push(init_acc);

    let mut indices: Vec<usize> = (0..n_samples).collect();

    for _epoch in 0..config.epochs {
        // Shuffle dataset
        shuffle_indices(&mut indices, &mut rng_state);

        // Mini-batch gradient descent
        let batch_size = config.batch_size.min(n_samples).max(1);
        let mut batch_start = 0;

        while batch_start < n_samples {
            let batch_end = (batch_start + batch_size).min(n_samples);
            let batch_samples: Vec<Vec<f64>> = (batch_start..batch_end)
                .map(|i| dataset.samples[indices[i]].clone())
                .collect();
            let batch_labels: Vec<usize> = (batch_start..batch_end)
                .map(|i| dataset.labels[indices[i]])
                .collect();

            let grads = qml_gradient(config, &params, &batch_samples, &batch_labels);

            for k in 0..n_params {
                params[k] -= config.learning_rate * grads[k];
            }

            batch_start = batch_end;
        }

        // Record epoch metrics
        let epoch_loss = batch_loss(config, &params, &dataset.samples, &dataset.labels);
        loss_history.push(epoch_loss);
        let epoch_acc = compute_accuracy(config, &params, dataset);
        accuracy_history.push(epoch_acc);
    }

    let final_accuracy = *accuracy_history.last().unwrap_or(&0.0);

    QmlResult {
        params,
        loss_history,
        accuracy_history,
        final_accuracy,
        epochs_completed: config.epochs,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config(n_qubits: usize, n_classes: usize) -> QmlConfig {
        QmlConfig {
            n_qubits,
            n_reupload_passes: 2,
            n_classes,
            max_bond: 16,
            readout_qubits: (0..n_classes).collect(),
            learning_rate: 0.1,
            epochs: 5,
            batch_size: 4,
            loss: QmlLoss::CrossEntropy,
            seed: 42,
        }
    }

    // --- Rotation matrices ---

    #[test]
    fn test_rx_identity_at_zero() {
        let m = rx_mat(0.0);
        assert!((m[0][0].re - 1.0).abs() < 1e-12);
        assert!(m[0][1].norm_sq() < 1e-24);
    }

    #[test]
    fn test_ry_identity_at_zero() {
        let m = ry_mat(0.0);
        assert!((m[0][0].re - 1.0).abs() < 1e-12);
        assert!(m[0][1].norm_sq() < 1e-24);
    }

    #[test]
    fn test_rz_identity_at_zero() {
        let m = rz_mat(0.0);
        assert!((m[0][0].re - 1.0).abs() < 1e-12);
        assert!(m[1][1].re - 1.0 < 1e-12);
    }

    // --- Snake ordering ---

    #[test]
    fn test_snake_order_4x4() {
        let order = snake_order(4, 4);
        assert_eq!(order.len(), 16);
        // Row 0: left-to-right
        assert_eq!(order[0], (0, 0));
        assert_eq!(order[3], (0, 3));
        // Row 1: right-to-left
        assert_eq!(order[4], (1, 3));
        assert_eq!(order[7], (1, 0));
        // Row 2: left-to-right
        assert_eq!(order[8], (2, 0));
    }

    #[test]
    fn test_snake_order_3x2() {
        let order = snake_order(3, 2);
        assert_eq!(order.len(), 6);
        assert_eq!(order[0], (0, 0));
        assert_eq!(order[2], (0, 2));
        assert_eq!(order[3], (1, 2));
        assert_eq!(order[5], (1, 0));
    }

    // --- Preprocessing ---

    #[test]
    fn test_preprocess_all_zeros() {
        let pixels = vec![0u8; 16];
        let result = preprocess_image(&pixels, 4, 4, 4);
        assert_eq!(result.len(), 4);
        for &v in &result {
            assert!(v.abs() < 1e-12, "zero pixels should give zero angles");
        }
    }

    #[test]
    fn test_preprocess_all_255() {
        let pixels = vec![255u8; 16];
        let result = preprocess_image(&pixels, 4, 4, 4);
        assert_eq!(result.len(), 4);
        for &v in &result {
            assert!((v - std::f64::consts::PI).abs() < 1e-12,
                "255 pixels should give pi, got {}", v);
        }
    }

    #[test]
    fn test_preprocess_normalization_range() {
        let mut pixels = vec![0u8; 64];
        for i in 0..64 {
            pixels[i] = (i * 4) as u8;
        }
        let result = preprocess_image(&pixels, 8, 8, 8);
        for &v in &result {
            assert!(v >= 0.0 && v <= std::f64::consts::PI,
                "value {} out of [0, pi]", v);
        }
    }

    // --- Single-qubit Z expectation ---

    #[test]
    fn test_single_z_ground_state() {
        let mps = Mps::new(4);
        for q in 0..4 {
            let z = mps_single_z_expectation(&mps, q);
            assert!((z - 1.0).abs() < 1e-10,
                "Z expectation of |0> should be +1, got {} for qubit {}", z, q);
        }
    }

    #[test]
    fn test_single_z_excited_state() {
        let mut mps = Mps::new(2);
        // Apply X to qubit 0 to get |10>
        let x_mat = [[ComplexF64::ZERO, ComplexF64::ONE],
                      [ComplexF64::ONE, ComplexF64::ZERO]];
        mps.apply_single_qubit(0, x_mat);
        assert!((mps_single_z_expectation(&mps, 0) - (-1.0)).abs() < 1e-10,
            "Z of |1> should be -1");
        assert!((mps_single_z_expectation(&mps, 1) - 1.0).abs() < 1e-10,
            "Z of |0> should be +1");
    }

    #[test]
    fn test_single_z_plus_state() {
        let mut mps = Mps::new(1);
        let isq2 = 1.0 / 2.0f64.sqrt();
        let h = [[ComplexF64::real(isq2), ComplexF64::real(isq2)],
                  [ComplexF64::real(isq2), ComplexF64::real(-isq2)]];
        mps.apply_single_qubit(0, h);
        let z = mps_single_z_expectation(&mps, 0);
        assert!(z.abs() < 1e-10, "Z of |+> should be 0, got {}", z);
    }

    // --- Circuit building ---

    #[test]
    fn test_build_circuit_zero_params_gives_product_state() {
        let config = default_config(4, 2);
        let params = vec![0.0; total_params(&config)];
        let input = vec![0.0; 4];
        let mps = build_qml_circuit(&config, &params, &input);
        // All zero angles → identity gates → |0000>
        for q in 0..4 {
            let z = mps_single_z_expectation(&mps, q);
            assert!((z - 1.0).abs() < 1e-10,
                "zero params + zero input should give |0>, got Z={} for qubit {}", z, q);
        }
    }

    // --- Classification ---

    #[test]
    fn test_classify_product_state_uniform() {
        let mps = Mps::new(4);
        let readout = vec![0, 1];
        let probs = classify(&mps, &readout, 2);
        assert_eq!(probs.len(), 2);
        // All Z values are +1, so softmax should give uniform
        assert!((probs[0] - 0.5).abs() < 1e-10, "p0={}", probs[0]);
        assert!((probs[1] - 0.5).abs() < 1e-10, "p1={}", probs[1]);
    }

    // --- Loss functions ---

    #[test]
    fn test_cross_entropy_known_values() {
        let probs = vec![0.7, 0.2, 0.1];
        let loss = cross_entropy_loss(&probs, 0);
        assert!((loss - (-0.7f64.ln())).abs() < 1e-10);
    }

    #[test]
    fn test_mse_loss_perfect_prediction() {
        let probs = vec![1.0, 0.0];
        let loss = mse_loss(&probs, 0, 2);
        assert!(loss.abs() < 1e-12, "perfect prediction should have 0 MSE");
    }

    #[test]
    fn test_mse_loss_worst_case() {
        let probs = vec![0.0, 1.0];
        let loss = mse_loss(&probs, 0, 2);
        // (0-1)^2 + (1-0)^2 = 2, / 2 = 1.0
        assert!((loss - 1.0).abs() < 1e-12);
    }

    // --- Gradient: finite-difference parity ---

    #[test]
    fn test_gradient_consistency() {
        let config = QmlConfig {
            n_qubits: 3,
            n_reupload_passes: 1,
            n_classes: 2,
            max_bond: 8,
            readout_qubits: vec![0, 1],
            learning_rate: 0.1,
            epochs: 1,
            batch_size: 2,
            loss: QmlLoss::Mse,
            seed: 42,
        };
        let params: Vec<f64> = (0..total_params(&config))
            .map(|i| (i as f64) * 0.1 - 0.5)
            .collect();
        let samples = vec![
            vec![0.5, 1.0, 0.2],
            vec![1.5, 0.3, 2.0],
        ];
        let labels = vec![0, 1];

        let grads = qml_gradient(&config, &params, &samples, &labels);

        // Cross-check: smaller epsilon finite-difference should agree
        let eps = 1e-6;
        for k in 0..params.len().min(6) {
            let mut p_plus = params.clone();
            let mut p_minus = params.clone();
            p_plus[k] += eps;
            p_minus[k] -= eps;
            let fd_grad = (batch_loss(&config, &p_plus, &samples, &labels)
                - batch_loss(&config, &p_minus, &samples, &labels))
                / (2.0 * eps);
            assert!((grads[k] - fd_grad).abs() < 1e-3,
                "param {}: grad={}, fd={}", k, grads[k], fd_grad);
        }

        // Gradient should be nonzero for at least some parameters
        let any_nonzero = grads.iter().any(|g| g.abs() > 1e-10);
        assert!(any_nonzero, "gradients should not all be zero");
    }

    // --- Training: loss decreases ---

    #[test]
    fn test_qml_train_loss_decreases() {
        // Trivial 2-class problem with distinct inputs
        let config = QmlConfig {
            n_qubits: 3,
            n_reupload_passes: 2,
            n_classes: 2,
            max_bond: 8,
            readout_qubits: vec![0, 1],
            learning_rate: 0.05,
            epochs: 15,
            batch_size: 4,
            loss: QmlLoss::CrossEntropy,
            seed: 42,
        };
        let dataset = QmlDataset {
            samples: vec![
                vec![0.1, 0.1, 0.1],
                vec![0.2, 0.1, 0.15],
                vec![2.8, 2.9, 3.0],
                vec![2.9, 3.0, 2.8],
            ],
            labels: vec![0, 0, 1, 1],
            n_classes: 2,
        };

        let result = qml_train(&config, &dataset);
        let initial = result.loss_history[0];
        // Check that loss decreased at some point during training
        let min_loss = result.loss_history.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(min_loss < initial,
            "loss should decrease at some point: initial={}, min={}", initial, min_loss);
    }

    // --- Determinism ---

    #[test]
    fn test_qml_deterministic() {
        let config = QmlConfig {
            n_qubits: 3,
            n_reupload_passes: 1,
            n_classes: 2,
            max_bond: 8,
            readout_qubits: vec![0, 1],
            learning_rate: 0.1,
            epochs: 3,
            batch_size: 2,
            loss: QmlLoss::CrossEntropy,
            seed: 123,
        };
        let dataset = QmlDataset {
            samples: vec![
                vec![0.1, 0.2, 0.3],
                vec![2.0, 1.5, 1.0],
            ],
            labels: vec![0, 1],
            n_classes: 2,
        };

        let r1 = qml_train(&config, &dataset);
        let r2 = qml_train(&config, &dataset);

        assert_eq!(r1.params.len(), r2.params.len());
        for k in 0..r1.params.len() {
            assert_eq!(r1.params[k].to_bits(), r2.params[k].to_bits(),
                "param[{}]: {} vs {}", k, r1.params[k], r2.params[k]);
        }
        for (i, (l1, l2)) in r1.loss_history.iter().zip(&r2.loss_history).enumerate() {
            assert_eq!(l1.to_bits(), l2.to_bits(),
                "loss_history[{}]: {} vs {}", i, l1, l2);
        }
    }

    // --- Memory budget ---

    #[test]
    fn test_qml_50_qubit_memory() {
        let config = QmlConfig {
            n_qubits: 50,
            n_reupload_passes: 3,
            n_classes: 2,
            max_bond: 16,
            readout_qubits: vec![0, 1],
            learning_rate: 0.1,
            epochs: 1,
            batch_size: 1,
            loss: QmlLoss::Mse,
            seed: 42,
        };
        let params = vec![0.01; total_params(&config)];
        let input = vec![1.0; 50];
        let mps = build_qml_circuit(&config, &params, &input);
        let mem = mps.memory_bytes();
        assert!(mem < 2_000_000_000, "memory {} exceeds 2GB", mem);
        // Should be well under: 50 * 2 * 16 * 16 * 16 bytes ~ 400KB
        assert!(mem < 1_000_000, "memory {} should be under 1MB for chi=16", mem);
    }

    // --- Load dataset ---

    #[test]
    fn test_load_dataset() {
        // Synthetic 4x4 images
        let n_images = 10;
        let img_size = 16;
        let image_data: Vec<u8> = (0..(n_images * img_size))
            .map(|i| (i % 256) as u8)
            .collect();
        let labels: Vec<u8> = (0..n_images).map(|i| (i % 3) as u8).collect();

        let ds = load_dataset(&image_data, &labels, 4, 4, 10, 4, 3);
        assert_eq!(ds.samples.len(), 10);
        assert_eq!(ds.labels.len(), 10);
        assert_eq!(ds.n_classes, 3);
        for s in &ds.samples {
            assert_eq!(s.len(), 4);
            for &v in s {
                assert!(v >= 0.0 && v <= std::f64::consts::PI);
            }
        }
    }
}
