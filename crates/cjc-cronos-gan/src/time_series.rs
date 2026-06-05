//! Core temporal data types: [`TimeStep`], [`TimeSeries`], [`TemporalBatch`],
//! [`SequenceMask`], [`ForecastWindow`], [`TemporalLoss`].
//!
//! These types are deliberately concrete (not generic on storage backend)
//! to keep the Phase 1 surface auditable. Every value is laid out
//! contiguously in row-major order and every operation iterates in
//! row-major order so reductions are reproducible across platforms.
//!
//! All numerical content is `f64`. f32 / quantized variants are deferred to
//! a later phase along with autodiff integration.

use crate::error::CronosGanError;

/// Index of a discrete timestep within a sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TimeStep {
    pub index: usize,
}

impl TimeStep {
    pub fn new(index: usize) -> Self {
        Self { index }
    }
}

/// A single time series: `n_steps × n_dim` row-major matrix of f64 values.
///
/// Storage is `Vec<f64>` of length `n_steps * n_dim`. `data[t * n_dim + d]`
/// is the value of dimension `d` at timestep `t`.
#[derive(Clone, Debug, PartialEq)]
pub struct TimeSeries {
    data: Vec<f64>,
    n_steps: usize,
    n_dim: usize,
}

impl TimeSeries {
    /// Construct from a row-major `data` vector. Returns
    /// [`CronosGanError::DimensionMismatch`] if `data.len() != n_steps *
    /// n_dim`. Returns [`CronosGanError::NonFiniteInput`] if any value is
    /// NaN or ±∞ — fail-loudly is the policy here.
    pub fn new(data: Vec<f64>, n_steps: usize, n_dim: usize) -> Result<Self, CronosGanError> {
        if n_dim == 0 {
            return Err(CronosGanError::InvalidConfig {
                detail: "TimeSeries n_dim must be >= 1".to_string(),
            });
        }
        let expected = n_steps.checked_mul(n_dim).ok_or_else(|| {
            CronosGanError::InvalidConfig {
                detail: "n_steps * n_dim overflows usize".to_string(),
            }
        })?;
        if data.len() != expected {
            return Err(CronosGanError::DimensionMismatch {
                detail: format!(
                    "TimeSeries::new: data.len()={} but n_steps*n_dim={}",
                    data.len(),
                    expected
                ),
            });
        }
        for (i, &v) in data.iter().enumerate() {
            if !v.is_finite() {
                return Err(CronosGanError::NonFiniteInput {
                    detail: format!("TimeSeries::new: data[{}] = {} is non-finite", i, v),
                });
            }
        }
        Ok(Self { data, n_steps, n_dim })
    }

    /// Number of timesteps in the series.
    pub fn n_steps(&self) -> usize {
        self.n_steps
    }

    /// Number of feature dimensions per timestep.
    pub fn n_dim(&self) -> usize {
        self.n_dim
    }

    /// View the underlying contiguous `f64` buffer.
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Borrow the slice for timestep `t`. Panics if `t >= n_steps` — index
    /// out of bounds is a programmer error, not a runtime data error.
    pub fn step(&self, t: usize) -> &[f64] {
        let start = t * self.n_dim;
        &self.data[start..start + self.n_dim]
    }

    /// Borrow a forecast window. Returns
    /// [`CronosGanError::ForecastHorizonOutOfRange`] if the window does
    /// not fit within the series.
    pub fn window(&self, w: &ForecastWindow) -> Result<&[f64], CronosGanError> {
        let end = w.start_step.checked_add(w.horizon).ok_or_else(|| {
            CronosGanError::ForecastHorizonOutOfRange {
                start_step: w.start_step,
                horizon: w.horizon,
                n_steps: self.n_steps,
            }
        })?;
        if end > self.n_steps {
            return Err(CronosGanError::ForecastHorizonOutOfRange {
                start_step: w.start_step,
                horizon: w.horizon,
                n_steps: self.n_steps,
            });
        }
        Ok(&self.data[w.start_step * self.n_dim..end * self.n_dim])
    }
}

/// A batch of `TimeSeries` for adversarial training.
///
/// All series in a batch must share `n_dim`, but per-series `n_steps` may
/// differ if `masks` is supplied: each mask declares which prefix of its
/// series is valid for the batch's purposes.
#[derive(Clone, Debug)]
pub struct TemporalBatch {
    series: Vec<TimeSeries>,
    masks: Option<Vec<SequenceMask>>,
    n_dim: usize,
}

impl TemporalBatch {
    /// Construct a batch from a non-empty `series` vector. If `masks` is
    /// `Some`, it must have the same length as `series` and each mask's
    /// `valid` length must equal the corresponding series's `n_steps`.
    /// All series must share the same `n_dim`.
    pub fn new(
        series: Vec<TimeSeries>,
        masks: Option<Vec<SequenceMask>>,
    ) -> Result<Self, CronosGanError> {
        if series.is_empty() {
            return Err(CronosGanError::InvalidConfig {
                detail: "TemporalBatch::new: series must be non-empty".to_string(),
            });
        }
        let n_dim = series[0].n_dim();
        for (i, s) in series.iter().enumerate() {
            if s.n_dim() != n_dim {
                return Err(CronosGanError::DimensionMismatch {
                    detail: format!(
                        "TemporalBatch::new: series[{}].n_dim={} but series[0].n_dim={}",
                        i,
                        s.n_dim(),
                        n_dim
                    ),
                });
            }
        }
        if let Some(ref ms) = masks {
            if ms.len() != series.len() {
                return Err(CronosGanError::DimensionMismatch {
                    detail: format!(
                        "TemporalBatch::new: masks.len()={} but series.len()={}",
                        ms.len(),
                        series.len()
                    ),
                });
            }
            for (i, (s, m)) in series.iter().zip(ms.iter()).enumerate() {
                if m.valid.len() != s.n_steps() {
                    return Err(CronosGanError::MaskLengthMismatch {
                        mask_len: m.valid.len(),
                        series_len: s.n_steps(),
                    });
                }
                let _ = i;
            }
        }
        Ok(Self { series, masks, n_dim })
    }

    pub fn len(&self) -> usize {
        self.series.len()
    }

    pub fn is_empty(&self) -> bool {
        self.series.is_empty()
    }

    pub fn n_dim(&self) -> usize {
        self.n_dim
    }

    pub fn series(&self) -> &[TimeSeries] {
        &self.series
    }

    pub fn masks(&self) -> Option<&[SequenceMask]> {
        self.masks.as_deref()
    }
}

/// Variable-length-sequence mask. `valid[t] == true` iff timestep `t` is
/// part of the sequence's effective length; trailing `false` entries are
/// padding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceMask {
    pub valid: Vec<bool>,
}

impl SequenceMask {
    /// Construct a mask of `n` valid timesteps followed by zero padding —
    /// the common "no padding" case.
    pub fn all_valid(n: usize) -> Self {
        Self { valid: vec![true; n] }
    }

    /// Construct a mask with `n_valid` leading valid timesteps and
    /// `n_pad` trailing invalid (padding) timesteps. Useful for batching
    /// variable-length sequences.
    pub fn with_padding(n_valid: usize, n_pad: usize) -> Self {
        let mut valid = vec![true; n_valid];
        valid.extend(std::iter::repeat(false).take(n_pad));
        Self { valid }
    }

    /// Number of valid (non-padding) timesteps.
    pub fn n_valid(&self) -> usize {
        self.valid.iter().filter(|v| **v).count()
    }
}

/// A window of timesteps to forecast / hold out / score.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ForecastWindow {
    pub start_step: usize,
    pub horizon: usize,
}

impl ForecastWindow {
    pub fn new(start_step: usize, horizon: usize) -> Self {
        Self { start_step, horizon }
    }
}

/// Loss function applied to a `(prediction, target)` pair of row-major
/// `f64` slices of identical length.
///
/// All variants use `cjc_repro::KahanAccumulatorF64` for the reduction so
/// the reported loss is byte-identical across platforms even on long
/// sequences.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TemporalLoss {
    Mse,
    Mae,
    Huber { delta: f64 },
}

impl TemporalLoss {
    /// Compute the loss between `pred` and `target`. Returns
    /// [`CronosGanError::DimensionMismatch`] if the slice lengths differ
    /// or [`CronosGanError::NonFiniteInput`] if any value is non-finite.
    pub fn evaluate(&self, pred: &[f64], target: &[f64]) -> Result<f64, CronosGanError> {
        if pred.len() != target.len() {
            return Err(CronosGanError::DimensionMismatch {
                detail: format!(
                    "TemporalLoss::evaluate: pred.len()={} target.len()={}",
                    pred.len(),
                    target.len()
                ),
            });
        }
        for (i, &v) in pred.iter().enumerate() {
            if !v.is_finite() {
                return Err(CronosGanError::NonFiniteInput {
                    detail: format!("TemporalLoss::evaluate: pred[{}] is non-finite", i),
                });
            }
        }
        for (i, &v) in target.iter().enumerate() {
            if !v.is_finite() {
                return Err(CronosGanError::NonFiniteInput {
                    detail: format!("TemporalLoss::evaluate: target[{}] is non-finite", i),
                });
            }
        }
        let mut acc = cjc_repro::KahanAccumulatorF64::new();
        let n = pred.len() as f64;
        match self {
            TemporalLoss::Mse => {
                for (p, t) in pred.iter().zip(target.iter()) {
                    let d = p - t;
                    acc.add(d * d);
                }
                Ok(acc.finalize() / n)
            }
            TemporalLoss::Mae => {
                for (p, t) in pred.iter().zip(target.iter()) {
                    acc.add((p - t).abs());
                }
                Ok(acc.finalize() / n)
            }
            TemporalLoss::Huber { delta } => {
                let d = *delta;
                if !d.is_finite() || d <= 0.0 {
                    return Err(CronosGanError::InvalidConfig {
                        detail: format!("Huber delta must be > 0 and finite, got {}", d),
                    });
                }
                for (p, t) in pred.iter().zip(target.iter()) {
                    let r = (p - t).abs();
                    if r <= d {
                        acc.add(0.5 * r * r);
                    } else {
                        acc.add(d * (r - 0.5 * d));
                    }
                }
                Ok(acc.finalize() / n)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_series_construction_happy_path() {
        let ts = TimeSeries::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).unwrap();
        assert_eq!(ts.n_steps(), 3);
        assert_eq!(ts.n_dim(), 2);
        assert_eq!(ts.step(1), &[3.0, 4.0]);
    }

    #[test]
    fn time_series_rejects_wrong_size_data() {
        let err = TimeSeries::new(vec![1.0, 2.0, 3.0], 2, 2).unwrap_err();
        assert!(matches!(err, CronosGanError::DimensionMismatch { .. }));
    }

    #[test]
    fn time_series_rejects_nan() {
        let err = TimeSeries::new(vec![1.0, f64::NAN], 1, 2).unwrap_err();
        assert!(matches!(err, CronosGanError::NonFiniteInput { .. }));
    }

    #[test]
    fn temporal_batch_construction_requires_same_n_dim() {
        let a = TimeSeries::new(vec![1.0, 2.0], 1, 2).unwrap();
        let b = TimeSeries::new(vec![1.0, 2.0, 3.0], 1, 3).unwrap();
        let err = TemporalBatch::new(vec![a, b], None).unwrap_err();
        assert!(matches!(err, CronosGanError::DimensionMismatch { .. }));
    }

    #[test]
    fn temporal_batch_mask_length_must_match_series_length() {
        let a = TimeSeries::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let err = TemporalBatch::new(
            vec![a],
            Some(vec![SequenceMask::all_valid(3)]), // wrong: series has 2 steps
        )
        .unwrap_err();
        assert!(matches!(err, CronosGanError::MaskLengthMismatch { .. }));
    }

    #[test]
    fn sequence_mask_padding_n_valid_correct() {
        let m = SequenceMask::with_padding(3, 2);
        assert_eq!(m.n_valid(), 3);
        assert_eq!(m.valid.len(), 5);
    }

    #[test]
    fn forecast_window_oob_returns_error() {
        let ts = TimeSeries::new(vec![1.0, 2.0, 3.0], 3, 1).unwrap();
        let w = ForecastWindow::new(2, 5);
        let err = ts.window(&w).unwrap_err();
        assert!(matches!(
            err,
            CronosGanError::ForecastHorizonOutOfRange { .. }
        ));
    }

    #[test]
    fn temporal_loss_mse_known_value() {
        let pred = [1.0, 2.0, 3.0];
        let target = [1.0, 4.0, 3.0];
        let mse = TemporalLoss::Mse.evaluate(&pred, &target).unwrap();
        // squared errors: 0, 4, 0; mean = 4/3
        assert!((mse - (4.0 / 3.0)).abs() < 1e-15);
    }

    #[test]
    fn temporal_loss_huber_rejects_bad_delta() {
        let err = TemporalLoss::Huber { delta: -1.0 }
            .evaluate(&[1.0], &[2.0])
            .unwrap_err();
        assert!(matches!(err, CronosGanError::InvalidConfig { .. }));
    }
}
