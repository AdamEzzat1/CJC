//! ML Toolkit — loss functions, optimizers, activations, metrics.
//!
//! # Determinism Contract
//! - All functions are deterministic (no randomness except seeded kfold).
//! - Kahan summation for all reductions.
//! - Stable sort for AUC-ROC with index tie-breaking.

use cjc_repro::KahanAccumulatorF64;

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
}
