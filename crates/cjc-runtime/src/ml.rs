//! ML Toolkit — loss functions, optimizers, activations, metrics.
//!
//! # Determinism Contract
//! - All functions are deterministic (no randomness except seeded kfold).
//! - Kahan summation for all reductions.
//! - Stable sort for AUC-ROC with index tie-breaking.

use cjc_repro::KahanAccumulatorF64;

use crate::accumulator::BinnedAccumulatorF64;
use crate::error::RuntimeError;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// Loss functions
// ---------------------------------------------------------------------------

/// Mean Squared Error: sum((pred - target)^2) / n.
pub fn mse_loss(pred: &[f64], target: &[f64]) -> Result<f64, String> {
    if pred.len() != target.len() {
        return Err("mse_loss: arrays must have same length".into());
    }
    if pred.is_empty() {
        return Err("mse_loss: empty data".into());
    }
    let mut acc = KahanAccumulatorF64::new();
    for i in 0..pred.len() {
        let d = pred[i] - target[i];
        acc.add(d * d);
    }
    Ok(acc.finalize() / pred.len() as f64)
}

/// Cross-entropy loss: -sum(target * ln(pred + eps)) / n.
pub fn cross_entropy_loss(pred: &[f64], target: &[f64]) -> Result<f64, String> {
    if pred.len() != target.len() {
        return Err("cross_entropy_loss: arrays must have same length".into());
    }
    if pred.is_empty() {
        return Err("cross_entropy_loss: empty data".into());
    }
    let eps = 1e-12;
    let mut acc = KahanAccumulatorF64::new();
    for i in 0..pred.len() {
        acc.add(-target[i] * (pred[i] + eps).ln());
    }
    Ok(acc.finalize() / pred.len() as f64)
}

/// Binary cross-entropy: -sum(t*ln(p) + (1-t)*ln(1-p)) / n.
pub fn binary_cross_entropy(pred: &[f64], target: &[f64]) -> Result<f64, String> {
    if pred.len() != target.len() {
        return Err("binary_cross_entropy: arrays must have same length".into());
    }
    if pred.is_empty() {
        return Err("binary_cross_entropy: empty data".into());
    }
    let eps = 1e-12;
    let mut acc = KahanAccumulatorF64::new();
    for i in 0..pred.len() {
        let p = pred[i].max(eps).min(1.0 - eps);
        acc.add(-(target[i] * p.ln() + (1.0 - target[i]) * (1.0 - p).ln()));
    }
    Ok(acc.finalize() / pred.len() as f64)
}

/// Huber loss: quadratic for small errors, linear for large.
pub fn huber_loss(pred: &[f64], target: &[f64], delta: f64) -> Result<f64, String> {
    if pred.len() != target.len() {
        return Err("huber_loss: arrays must have same length".into());
    }
    if pred.is_empty() {
        return Err("huber_loss: empty data".into());
    }
    let mut acc = KahanAccumulatorF64::new();
    for i in 0..pred.len() {
        let d = (pred[i] - target[i]).abs();
        if d <= delta {
            acc.add(0.5 * d * d);
        } else {
            acc.add(delta * (d - 0.5 * delta));
        }
    }
    Ok(acc.finalize() / pred.len() as f64)
}

/// Hinge loss: sum(max(0, 1 - target * pred)) / n.
pub fn hinge_loss(pred: &[f64], target: &[f64]) -> Result<f64, String> {
    if pred.len() != target.len() {
        return Err("hinge_loss: arrays must have same length".into());
    }
    if pred.is_empty() {
        return Err("hinge_loss: empty data".into());
    }
    let mut acc = KahanAccumulatorF64::new();
    for i in 0..pred.len() {
        acc.add((1.0 - target[i] * pred[i]).max(0.0));
    }
    Ok(acc.finalize() / pred.len() as f64)
}

// ---------------------------------------------------------------------------
// Optimizers
// ---------------------------------------------------------------------------

/// SGD optimizer state.
pub struct SgdState {
    pub lr: f64,
    pub momentum: f64,
    pub velocity: Vec<f64>,
}

impl SgdState {
    pub fn new(n_params: usize, lr: f64, momentum: f64) -> Self {
        Self { lr, momentum, velocity: vec![0.0; n_params] }
    }
}

/// SGD step: sequential, deterministic.
pub fn sgd_step(params: &mut [f64], grads: &[f64], state: &mut SgdState) {
    for i in 0..params.len() {
        state.velocity[i] = state.momentum * state.velocity[i] + grads[i];
        params[i] -= state.lr * state.velocity[i];
    }
}

/// Adam optimizer state.
pub struct AdamState {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub t: u64,
    pub m: Vec<f64>,
    pub v: Vec<f64>,
}

impl AdamState {
    pub fn new(n_params: usize, lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
        }
    }
}

/// Adam step: sequential, deterministic.
pub fn adam_step(params: &mut [f64], grads: &[f64], state: &mut AdamState) {
    state.t += 1;
    let t = state.t as f64;
    for i in 0..params.len() {
        state.m[i] = state.beta1 * state.m[i] + (1.0 - state.beta1) * grads[i];
        state.v[i] = state.beta2 * state.v[i] + (1.0 - state.beta2) * grads[i] * grads[i];
        let m_hat = state.m[i] / (1.0 - state.beta1.powf(t));
        let v_hat = state.v[i] / (1.0 - state.beta2.powf(t));
        params[i] -= state.lr * m_hat / (v_hat.sqrt() + state.eps);
    }
}

// ---------------------------------------------------------------------------
// Classification metrics (Sprint 6)
// ---------------------------------------------------------------------------

/// Binary confusion matrix.
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    pub tp: usize,
    pub fp: usize,
    pub tn: usize,
    pub fn_count: usize,
}

/// Build confusion matrix from predicted and actual boolean labels.
pub fn confusion_matrix(predicted: &[bool], actual: &[bool]) -> ConfusionMatrix {
    let mut tp = 0;
    let mut fp = 0;
    let mut tn = 0;
    let mut fn_count = 0;
    for i in 0..predicted.len().min(actual.len()) {
        match (predicted[i], actual[i]) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_count += 1,
            (false, false) => tn += 1,
        }
    }
    ConfusionMatrix { tp, fp, tn, fn_count }
}

/// Precision: TP / (TP + FP).
pub fn precision(cm: &ConfusionMatrix) -> f64 {
    let denom = cm.tp + cm.fp;
    if denom == 0 { 0.0 } else { cm.tp as f64 / denom as f64 }
}

/// Recall / sensitivity: TP / (TP + FN).
pub fn recall(cm: &ConfusionMatrix) -> f64 {
    let denom = cm.tp + cm.fn_count;
    if denom == 0 { 0.0 } else { cm.tp as f64 / denom as f64 }
}

/// F1 score: 2 * (precision * recall) / (precision + recall).
pub fn f1_score(cm: &ConfusionMatrix) -> f64 {
    let p = precision(cm);
    let r = recall(cm);
    if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
}

/// Accuracy: (TP + TN) / total.
pub fn accuracy(cm: &ConfusionMatrix) -> f64 {
    let total = cm.tp + cm.fp + cm.tn + cm.fn_count;
    if total == 0 { 0.0 } else { (cm.tp + cm.tn) as f64 / total as f64 }
}

/// AUC-ROC via trapezoidal rule.
/// DETERMINISM: sort by score with stable sort + index tie-breaking.
pub fn auc_roc(scores: &[f64], labels: &[bool]) -> Result<f64, String> {
    if scores.len() != labels.len() {
        return Err("auc_roc: scores and labels must have same length".into());
    }
    let n = scores.len();
    if n == 0 {
        return Err("auc_roc: empty data".into());
    }
    // Sort by score descending, stable with index tie-breaking
    let mut indexed: Vec<(usize, f64, bool)> = scores.iter().zip(labels.iter())
        .enumerate()
        .map(|(i, (&s, &l))| (i, s, l))
        .collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

    let pos_count = labels.iter().filter(|&&l| l).count();
    let neg_count = n - pos_count;
    if pos_count == 0 || neg_count == 0 {
        return Err("auc_roc: need both positive and negative labels".into());
    }

    let mut auc = 0.0;
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut prev_fpr = 0.0;
    let mut prev_tpr = 0.0;

    for &(_, _, label) in &indexed {
        if label { tp += 1.0; } else { fp += 1.0; }
        let tpr = tp / pos_count as f64;
        let fpr = fp / neg_count as f64;
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
        prev_fpr = fpr;
        prev_tpr = tpr;
    }
    Ok(auc)
}

/// K-fold cross-validation indices.
/// DETERMINISM: uses seeded RNG (Fisher-Yates).
pub fn kfold_indices(n: usize, k: usize, seed: u64) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut rng = cjc_repro::Rng::seeded(seed);
    // Fisher-Yates shuffle of [0..n]
    let mut indices: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        indices.swap(i, j);
    }
    let fold_size = n / k;
    let mut folds = Vec::with_capacity(k);
    for fold in 0..k {
        let start = fold * fold_size;
        let end = if fold == k - 1 { n } else { start + fold_size };
        let test: Vec<usize> = indices[start..end].to_vec();
        let train: Vec<usize> = indices[..start].iter()
            .chain(indices[end..].iter())
            .copied()
            .collect();
        folds.push((train, test));
    }
    folds
}

/// Train/test split indices.
pub fn train_test_split(n: usize, test_fraction: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut rng = cjc_repro::Rng::seeded(seed);
    let mut indices: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        indices.swap(i, j);
    }
    let test_size = ((n as f64) * test_fraction).round() as usize;
    let test = indices[..test_size].to_vec();
    let train = indices[test_size..].to_vec();
    (train, test)
}

// ---------------------------------------------------------------------------
// Phase B4: ML Training Extensions
// ---------------------------------------------------------------------------

/// Batch normalization (inference mode).
/// y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta.
pub fn batch_norm(
    x: &[f64],
    running_mean: &[f64],
    running_var: &[f64],
    gamma: &[f64],
    beta: &[f64],
    eps: f64,
) -> Result<Vec<f64>, String> {
    let n = x.len();
    if running_mean.len() != n || running_var.len() != n || gamma.len() != n || beta.len() != n {
        return Err("batch_norm: all arrays must have same length".into());
    }
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let normed = (x[i] - running_mean[i]) / (running_var[i] + eps).sqrt();
        result.push(gamma[i] * normed + beta[i]);
    }
    Ok(result)
}

/// Dropout mask generation using seeded RNG for determinism.
/// Returns mask of 0.0 and scale values (1/(1-p)) using inverted dropout.
pub fn dropout_mask(n: usize, drop_prob: f64, seed: u64) -> Vec<f64> {
    let mut rng = cjc_repro::Rng::seeded(seed);
    let scale = if drop_prob < 1.0 { 1.0 / (1.0 - drop_prob) } else { 0.0 };
    let mut mask = Vec::with_capacity(n);
    for _ in 0..n {
        let r = (rng.next_u64() as f64) / (u64::MAX as f64);
        if r < drop_prob {
            mask.push(0.0);
        } else {
            mask.push(scale);
        }
    }
    mask
}

/// Apply dropout: element-wise multiply data by mask.
pub fn apply_dropout(data: &[f64], mask: &[f64]) -> Result<Vec<f64>, String> {
    if data.len() != mask.len() {
        return Err("apply_dropout: data and mask must have same length".into());
    }
    Ok(data.iter().zip(mask.iter()).map(|(&d, &m)| d * m).collect())
}

/// Learning rate schedule: step decay.
/// lr = initial_lr * decay_rate^(floor(epoch / step_size))
pub fn lr_step_decay(initial_lr: f64, decay_rate: f64, epoch: usize, step_size: usize) -> f64 {
    initial_lr * decay_rate.powi((epoch / step_size) as i32)
}

/// Learning rate schedule: cosine annealing.
/// lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * epoch / total_epochs))
pub fn lr_cosine(max_lr: f64, min_lr: f64, epoch: usize, total_epochs: usize) -> f64 {
    let ratio = epoch as f64 / total_epochs as f64;
    min_lr + 0.5 * (max_lr - min_lr) * (1.0 + (std::f64::consts::PI * ratio).cos())
}

/// Learning rate schedule: linear warmup.
/// lr = initial_lr * min(1.0, epoch / warmup_epochs).
pub fn lr_linear_warmup(initial_lr: f64, epoch: usize, warmup_epochs: usize) -> f64 {
    if warmup_epochs == 0 {
        return initial_lr;
    }
    initial_lr * (epoch as f64 / warmup_epochs as f64).min(1.0)
}

/// L1 regularization penalty: lambda * sum(|params|).
pub fn l1_penalty(params: &[f64], lambda: f64) -> f64 {
    let mut acc = KahanAccumulatorF64::new();
    for &p in params {
        acc.add(p.abs());
    }
    lambda * acc.finalize()
}

/// L2 regularization penalty: 0.5 * lambda * sum(params^2).
pub fn l2_penalty(params: &[f64], lambda: f64) -> f64 {
    let mut acc = KahanAccumulatorF64::new();
    for &p in params {
        acc.add(p * p);
    }
    0.5 * lambda * acc.finalize()
}

/// L1 regularization gradient: lambda * sign(params).
pub fn l1_grad(params: &[f64], lambda: f64) -> Vec<f64> {
    params.iter().map(|&p| {
        if p > 0.0 { lambda } else if p < 0.0 { -lambda } else { 0.0 }
    }).collect()
}

/// L2 regularization gradient: lambda * params.
pub fn l2_grad(params: &[f64], lambda: f64) -> Vec<f64> {
    params.iter().map(|&p| lambda * p).collect()
}

/// Early stopping state tracker.
pub struct EarlyStoppingState {
    pub patience: usize,
    pub min_delta: f64,
    pub best_loss: f64,
    pub wait: usize,
    pub stopped: bool,
}

impl EarlyStoppingState {
    pub fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            best_loss: f64::INFINITY,
            wait: 0,
            stopped: false,
        }
    }

    /// Check if training should stop.
    pub fn check(&mut self, current_loss: f64) -> bool {
        if current_loss < self.best_loss - self.min_delta {
            self.best_loss = current_loss;
            self.wait = 0;
        } else {
            self.wait += 1;
        }
        if self.wait >= self.patience {
            self.stopped = true;
        }
        self.stopped
    }
}

// ---------------------------------------------------------------------------
// Phase 3C: PCA (Principal Component Analysis)
// ---------------------------------------------------------------------------

/// Principal Component Analysis via SVD of centered data.
///
/// `data` is a 2D Tensor of shape (n_samples, n_features).
/// `n_components` is the number of principal components to keep.
///
/// Returns (transformed_data, components, explained_variance_ratio):
/// - `transformed_data`: (n_samples, n_components) — data projected onto principal components
/// - `components`: (n_components, n_features) — principal component directions (rows)
/// - `explained_variance_ratio`: Vec<f64> of length n_components — fraction of variance per component
///
/// **Determinism contract:** All reductions use `BinnedAccumulatorF64`.
pub fn pca(
    data: &Tensor,
    n_components: usize,
) -> Result<(Tensor, Tensor, Vec<f64>), RuntimeError> {
    if data.ndim() != 2 {
        return Err(RuntimeError::InvalidOperation(
            "PCA requires a 2D data matrix".to_string(),
        ));
    }
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    if n_samples == 0 || n_features == 0 {
        return Err(RuntimeError::InvalidOperation(
            "PCA: empty data matrix".to_string(),
        ));
    }
    if n_components == 0 || n_components > n_features.min(n_samples) {
        return Err(RuntimeError::InvalidOperation(format!(
            "PCA: n_components ({}) must be in [1, min(n_samples, n_features) = {}]",
            n_components,
            n_features.min(n_samples)
        )));
    }

    let raw = data.to_vec();

    // Step 1: Compute column means using BinnedAccumulatorF64
    let mut means = vec![0.0f64; n_features];
    for j in 0..n_features {
        let mut acc = BinnedAccumulatorF64::new();
        for i in 0..n_samples {
            acc.add(raw[i * n_features + j]);
        }
        means[j] = acc.finalize() / n_samples as f64;
    }

    // Step 2: Center the data
    let mut centered = vec![0.0f64; n_samples * n_features];
    for i in 0..n_samples {
        for j in 0..n_features {
            centered[i * n_features + j] = raw[i * n_features + j] - means[j];
        }
    }
    let centered_tensor = Tensor::from_vec(centered, &[n_samples, n_features])?;

    // Step 3: SVD of centered data
    let (u, s, vt) = centered_tensor.svd()?;
    let k = n_components.min(s.len());

    // Step 4: Components = first k rows of Vt
    let vt_data = vt.to_vec();
    let vt_cols = vt.shape()[1]; // n_features
    let mut components = vec![0.0f64; k * n_features];
    for i in 0..k {
        for j in 0..n_features {
            components[i * n_features + j] = vt_data[i * vt_cols + j];
        }
    }

    // Step 5: Explained variance = s_i^2 / (n_samples - 1)
    // Total variance = sum of all s_i^2 / (n_samples - 1)
    let denom = if n_samples > 1 {
        (n_samples - 1) as f64
    } else {
        1.0
    };

    let mut total_var_acc = BinnedAccumulatorF64::new();
    for &si in &s {
        total_var_acc.add(si * si / denom);
    }
    let total_var = total_var_acc.finalize();

    let explained_variance_ratio: Vec<f64> = if total_var > 1e-15 {
        s[..k]
            .iter()
            .map(|&si| (si * si / denom) / total_var)
            .collect()
    } else {
        vec![0.0; k]
    };

    // Step 6: Transformed data = U_k @ diag(S_k)
    let u_data = u.to_vec();
    let u_cols = u.shape()[1];
    let mut transformed = vec![0.0f64; n_samples * k];
    for i in 0..n_samples {
        for j in 0..k {
            transformed[i * k + j] = u_data[i * u_cols + j] * s[j];
        }
    }

    Ok((
        Tensor::from_vec(transformed, &[n_samples, k])?,
        Tensor::from_vec(components, &[k, n_features])?,
        explained_variance_ratio,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_zero() {
        let pred = [1.0, 2.0, 3.0];
        let target = [1.0, 2.0, 3.0];
        assert_eq!(mse_loss(&pred, &target).unwrap(), 0.0);
    }

    #[test]
    fn test_mse_basic() {
        let pred = [1.0, 2.0, 3.0];
        let target = [2.0, 3.0, 4.0];
        assert_eq!(mse_loss(&pred, &target).unwrap(), 1.0);
    }

    #[test]
    fn test_huber_loss_quadratic() {
        let pred = [1.0];
        let target = [1.5];
        let h = huber_loss(&pred, &target, 1.0).unwrap();
        // |0.5| < 1.0, so quadratic: 0.5 * 0.25 = 0.125
        assert!((h - 0.125).abs() < 1e-12);
    }

    #[test]
    fn test_sgd_step() {
        let mut params = [1.0, 2.0];
        let grads = [0.1, 0.2];
        let mut state = SgdState::new(2, 0.1, 0.0);
        sgd_step(&mut params, &grads, &mut state);
        assert!((params[0] - 0.99).abs() < 1e-12);
        assert!((params[1] - 1.98).abs() < 1e-12);
    }

    #[test]
    fn test_adam_step() {
        let mut params = [1.0, 2.0];
        let grads = [0.1, 0.2];
        let mut state = AdamState::new(2, 0.001);
        adam_step(&mut params, &grads, &mut state);
        // After one step, params should be slightly different
        assert!(params[0] < 1.0);
        assert!(params[1] < 2.0);
    }

    #[test]
    fn test_confusion_matrix() {
        let pred = [true, true, false, false, true];
        let actual = [true, false, true, false, true];
        let cm = confusion_matrix(&pred, &actual);
        assert_eq!(cm.tp, 2);
        assert_eq!(cm.fp, 1);
        assert_eq!(cm.fn_count, 1);
        assert_eq!(cm.tn, 1);
    }

    #[test]
    fn test_precision_recall_f1() {
        let cm = ConfusionMatrix { tp: 5, fp: 2, tn: 8, fn_count: 1 };
        assert!((precision(&cm) - 5.0 / 7.0).abs() < 1e-12);
        assert!((recall(&cm) - 5.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_auc_perfect() {
        let scores = [0.9, 0.8, 0.2, 0.1];
        let labels = [true, true, false, false];
        let auc = auc_roc(&scores, &labels).unwrap();
        assert!((auc - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_kfold_deterministic() {
        let f1 = kfold_indices(100, 5, 42);
        let f2 = kfold_indices(100, 5, 42);
        for i in 0..5 {
            assert_eq!(f1[i].0, f2[i].0);
            assert_eq!(f1[i].1, f2[i].1);
        }
    }

    #[test]
    fn test_train_test_split_coverage() {
        let (train, test) = train_test_split(100, 0.2, 42);
        assert_eq!(train.len() + test.len(), 100);
        assert_eq!(test.len(), 20);
    }

    // --- B4: ML Training Extensions tests ---

    #[test]
    fn test_batch_norm_identity() {
        // mean=0, var=1, gamma=1, beta=0, eps=0 → input unchanged
        let x = vec![1.0, 2.0, 3.0];
        let mean = vec![0.0, 0.0, 0.0];
        let var = vec![1.0, 1.0, 1.0];
        let gamma = vec![1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0];
        let result = batch_norm(&x, &mean, &var, &gamma, &beta, 0.0).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-12);
        assert!((result[1] - 2.0).abs() < 1e-12);
        assert!((result[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_batch_norm_shift_scale() {
        let x = vec![0.0];
        let mean = vec![1.0]; // shift: x - 1 = -1
        let var = vec![4.0];  // scale: -1/sqrt(4) = -0.5
        let gamma = vec![2.0]; // multiply: 2 * -0.5 = -1
        let beta = vec![3.0]; // add: -1 + 3 = 2
        let result = batch_norm(&x, &mean, &var, &gamma, &beta, 0.0).unwrap();
        assert!((result[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_dropout_mask_seed_determinism() {
        let m1 = dropout_mask(100, 0.5, 42);
        let m2 = dropout_mask(100, 0.5, 42);
        assert_eq!(m1, m2);
    }

    #[test]
    fn test_dropout_mask_different_seeds() {
        let m1 = dropout_mask(100, 0.5, 42);
        let m2 = dropout_mask(100, 0.5, 99);
        assert_ne!(m1, m2);
    }

    #[test]
    fn test_lr_step_decay_schedule() {
        let lr0 = lr_step_decay(0.1, 0.5, 0, 10);
        assert!((lr0 - 0.1).abs() < 1e-12);
        let lr10 = lr_step_decay(0.1, 0.5, 10, 10);
        assert!((lr10 - 0.05).abs() < 1e-12);
        let lr20 = lr_step_decay(0.1, 0.5, 20, 10);
        assert!((lr20 - 0.025).abs() < 1e-12);
    }

    #[test]
    fn test_lr_cosine_endpoints() {
        let lr0 = lr_cosine(0.1, 0.001, 0, 100);
        assert!((lr0 - 0.1).abs() < 1e-10);
        let lr_end = lr_cosine(0.1, 0.001, 100, 100);
        assert!((lr_end - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_lr_linear_warmup() {
        let lr0 = lr_linear_warmup(0.1, 0, 10);
        assert!((lr0).abs() < 1e-12);
        let lr5 = lr_linear_warmup(0.1, 5, 10);
        assert!((lr5 - 0.05).abs() < 1e-12);
        let lr15 = lr_linear_warmup(0.1, 15, 10);
        assert!((lr15 - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_l1_penalty_known() {
        let params = [1.0, -2.0, 3.0];
        let p = l1_penalty(&params, 0.1);
        assert!((p - 0.6).abs() < 1e-12);
    }

    #[test]
    fn test_l2_penalty_known() {
        let params = [1.0, -2.0, 3.0];
        let p = l2_penalty(&params, 0.1);
        // 0.5 * 0.1 * (1 + 4 + 9) = 0.5 * 0.1 * 14 = 0.7
        assert!((p - 0.7).abs() < 1e-12);
    }

    #[test]
    fn test_early_stopping_triggers() {
        let mut es = EarlyStoppingState::new(3, 0.01);
        assert!(!es.check(1.0)); // best_loss=1.0, wait=0
        assert!(!es.check(1.0)); // no improvement, wait=1
        assert!(!es.check(1.0)); // no improvement, wait=2
        assert!(es.check(1.0));  // no improvement, wait=3 >= patience
    }

    #[test]
    fn test_early_stopping_resets() {
        let mut es = EarlyStoppingState::new(3, 0.01);
        es.check(1.0);
        es.check(1.0); // wait=1
        assert!(!es.check(0.5)); // improvement, wait=0
        assert!(!es.check(0.5)); // wait=1
    }

    // --- Phase 3C: PCA tests ---

    #[test]
    fn test_pca_basic_2d() {
        // 4 samples, 2 features — data lies mostly along first axis
        let data = Tensor::from_vec(
            vec![
                1.0, 0.1,
                2.0, 0.2,
                3.0, 0.3,
                4.0, 0.4,
            ],
            &[4, 2],
        )
        .unwrap();
        let (transformed, components, evr) = pca(&data, 2).unwrap();
        assert_eq!(transformed.shape(), &[4, 2]);
        assert_eq!(components.shape(), &[2, 2]);
        assert_eq!(evr.len(), 2);
        // Explained variance ratios should sum to ~1.0
        let total: f64 = evr.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-8,
            "explained variance ratios sum to {} (expected ~1.0)",
            total
        );
        // First component should explain most variance
        assert!(evr[0] > 0.9, "first component explains {} of variance", evr[0]);
    }

    #[test]
    fn test_pca_single_component() {
        let data = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
            ],
            &[3, 3],
        )
        .unwrap();
        let (transformed, components, evr) = pca(&data, 1).unwrap();
        assert_eq!(transformed.shape(), &[3, 1]);
        assert_eq!(components.shape(), &[1, 3]);
        assert_eq!(evr.len(), 1);
        assert!(evr[0] > 0.0 && evr[0] <= 1.0);
    }

    #[test]
    fn test_pca_explained_variance_ratio_bounded() {
        let data = Tensor::from_vec(
            vec![
                1.0, 0.0, 0.5,
                0.0, 1.0, 0.5,
                1.0, 1.0, 1.0,
                2.0, 0.0, 1.0,
                0.0, 2.0, 1.0,
            ],
            &[5, 3],
        )
        .unwrap();
        let (_, _, evr) = pca(&data, 3).unwrap();
        let total: f64 = evr.iter().sum();
        assert!(
            total <= 1.0 + 1e-10,
            "explained variance ratios sum to {} (should be <= 1.0)",
            total
        );
        for &r in &evr {
            assert!(r >= -1e-10, "negative explained variance ratio: {}", r);
        }
    }

    #[test]
    fn test_pca_deterministic() {
        let data = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        )
        .unwrap();
        let (t1, c1, e1) = pca(&data, 2).unwrap();
        let (t2, c2, e2) = pca(&data, 2).unwrap();
        assert_eq!(t1.to_vec(), t2.to_vec(), "PCA transformed not deterministic");
        assert_eq!(c1.to_vec(), c2.to_vec(), "PCA components not deterministic");
        assert_eq!(e1, e2, "PCA explained variance not deterministic");
    }

    #[test]
    fn test_pca_invalid_n_components() {
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert!(pca(&data, 0).is_err(), "n_components=0 should fail");
        assert!(pca(&data, 3).is_err(), "n_components > min(n,p) should fail");
    }
}
