//! Phase 4: synthetic temporal datasets for the Cronos GAN experiment
//! harness.
//!
//! Five deterministic generators mapping `(seed, n_steps)` to
//! `(inputs, targets)`:
//!
//! 1. [`smooth_sine`]            — `x_t = sin(ω t)`. The control: both
//!                                 networks should fit this well.
//! 2. [`noisy_sine`]              — smooth sine + Gaussian noise. Tests
//!                                 whether the SSM's stable bias acts as
//!                                 implicit regularisation.
//! 3. [`regime_shift`]            — two AR(1) processes with different
//!                                 dynamics glued at the midpoint.
//!                                 The canonical regime-shift test case.
//! 4. [`step_change_anomaly`]     — mostly flat with one isolated step
//!                                 jump. Tests anomaly localisation.
//! 5. [`chaotic_spike`]           — sine baseline + sparse large
//!                                 deterministic spikes. Tests whether
//!                                 the Liquid net's reactive gates
//!                                 surface the spikes more than the
//!                                 SSM's smoother dynamics.
//!
//! All generators take a [`CronosSeed`] and route their randomness
//! through `seed.substream(SALT)` with the dataset-specific salt
//! documented next to each function. Same seed ⇒ same dataset.
//!
//! Each generator returns 1-D inputs and 1-D targets where
//! `targets[t] = inputs[t+1]` (next-step prediction task) — the simplest
//! supervised setup that makes both networks' forecasts directly
//! comparable.

use crate::error::CronosGanError;
use crate::seed::CronosSeed;
use cjc_repro::Rng;

/// Smooth sine wave: `x_t = sin(ω t)` with default angular frequency
/// `ω = 0.4` rad/step. Inputs and targets are 1-D.
///
/// Deterministic without any RNG draw (no noise) — the `seed` parameter
/// is accepted for API uniformity but never read.
pub fn smooth_sine(_seed: CronosSeed, n_steps: usize) -> Result<(Vec<f64>, Vec<f64>), CronosGanError> {
    validate_n_steps(n_steps, "smooth_sine")?;
    let omega = 0.4_f64;
    let series: Vec<f64> = (0..(n_steps + 1)).map(|t| (omega * t as f64).sin()).collect();
    Ok(split_next_step(series, n_steps))
}

/// Sine baseline + Gaussian noise with σ = 0.15.
///
/// Sub-stream salt: `"dataset.noisy_sine"`.
pub fn noisy_sine(seed: CronosSeed, n_steps: usize) -> Result<(Vec<f64>, Vec<f64>), CronosGanError> {
    validate_n_steps(n_steps, "noisy_sine")?;
    let mut rng = seed.substream("dataset.noisy_sine");
    let omega = 0.4_f64;
    let sigma = 0.15_f64;
    let series: Vec<f64> = (0..(n_steps + 1))
        .map(|t| (omega * t as f64).sin() + sigma * standard_normal(&mut rng))
        .collect();
    Ok(split_next_step(series, n_steps))
}

/// Two AR(1) processes glued at the midpoint.
///
/// First half: `x_{t+1} = 0.7 x_t + ε_t, ε ~ N(0, 0.2²)`.
/// Second half: `x_{t+1} = -0.3 x_t + ε_t, ε ~ N(0, 0.5²)`.
/// The autoregressive coefficient flips sign AND the noise scale jumps,
/// so the Liquid net's reactive gates should fire harder around the
/// midpoint than the SSM's smoother evolution.
///
/// Sub-stream salt: `"dataset.regime_shift"`.
pub fn regime_shift(seed: CronosSeed, n_steps: usize) -> Result<(Vec<f64>, Vec<f64>), CronosGanError> {
    validate_n_steps(n_steps, "regime_shift")?;
    let mut rng = seed.substream("dataset.regime_shift");
    let half = (n_steps + 1) / 2;
    let mut series: Vec<f64> = Vec::with_capacity(n_steps + 1);
    series.push(0.0);
    for t in 1..=n_steps {
        let prev = series[t - 1];
        let (phi, sigma) = if t <= half { (0.7, 0.2) } else { (-0.3, 0.5) };
        let next = phi * prev + sigma * standard_normal(&mut rng);
        series.push(next);
    }
    Ok(split_next_step(series, n_steps))
}

/// Flat baseline ≈ 0 with a single deterministic step up to +1 at step
/// `n_steps / 2` that persists for the remainder. No randomness — the
/// step is exactly localised so anomaly-localisation tests are
/// reproducible.
///
/// The `seed` argument is accepted for uniformity but unused.
pub fn step_change_anomaly(
    _seed: CronosSeed,
    n_steps: usize,
) -> Result<(Vec<f64>, Vec<f64>), CronosGanError> {
    validate_n_steps(n_steps, "step_change_anomaly")?;
    let step_at = n_steps / 2;
    let series: Vec<f64> = (0..(n_steps + 1))
        .map(|t| if t >= step_at { 1.0 } else { 0.0 })
        .collect();
    Ok(split_next_step(series, n_steps))
}

/// Sine baseline with deterministic large spikes injected every 10
/// timesteps (offset by 3). Spike magnitude is +3.0 (about 20× the
/// baseline sine amplitude) — large enough that any reasonable network
/// will be surprised, sparse enough that the underlying sine dynamics
/// remain learnable.
///
/// Pure determinism: the spike times are integer-divisor-based, no RNG
/// is involved. The `seed` argument is accepted for uniformity but unused.
pub fn chaotic_spike(
    _seed: CronosSeed,
    n_steps: usize,
) -> Result<(Vec<f64>, Vec<f64>), CronosGanError> {
    validate_n_steps(n_steps, "chaotic_spike")?;
    let omega = 0.4_f64;
    let series: Vec<f64> = (0..(n_steps + 1))
        .map(|t| {
            let base = (omega * t as f64).sin();
            let spike = if t > 3 && (t - 3) % 10 == 0 { 3.0 } else { 0.0 };
            base + spike
        })
        .collect();
    Ok(split_next_step(series, n_steps))
}

/// Stable identifier for each shipped dataset — used by the experiment
/// harness's content-addressed run ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CronosDataset {
    SmoothSine,
    NoisySine,
    RegimeShift,
    StepChangeAnomaly,
    ChaoticSpike,
}

impl CronosDataset {
    /// Stable string label.
    pub fn label(self) -> &'static str {
        match self {
            CronosDataset::SmoothSine => "smooth_sine",
            CronosDataset::NoisySine => "noisy_sine",
            CronosDataset::RegimeShift => "regime_shift",
            CronosDataset::StepChangeAnomaly => "step_change_anomaly",
            CronosDataset::ChaoticSpike => "chaotic_spike",
        }
    }

    /// Generate inputs + targets for this dataset.
    pub fn generate(
        self,
        seed: CronosSeed,
        n_steps: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), CronosGanError> {
        match self {
            CronosDataset::SmoothSine => smooth_sine(seed, n_steps),
            CronosDataset::NoisySine => noisy_sine(seed, n_steps),
            CronosDataset::RegimeShift => regime_shift(seed, n_steps),
            CronosDataset::StepChangeAnomaly => step_change_anomaly(seed, n_steps),
            CronosDataset::ChaoticSpike => chaotic_spike(seed, n_steps),
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────

fn validate_n_steps(n: usize, name: &str) -> Result<(), CronosGanError> {
    if n < 2 {
        return Err(CronosGanError::InvalidConfig {
            detail: format!("{}: n_steps must be >= 2, got {}", name, n),
        });
    }
    Ok(())
}

/// Box-Muller standard normal — same helper used in ssm.rs and liquid.rs.
fn standard_normal(rng: &mut Rng) -> f64 {
    let mut u1 = rng.next_f64();
    while u1 == 0.0 {
        u1 = rng.next_f64();
    }
    let u2 = rng.next_f64();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f64::consts::TAU * u2;
    r * theta.cos()
}

/// Split `series[0..=n_steps]` into `inputs = series[0..n_steps]` and
/// `targets = series[1..=n_steps]` so the prediction task is next-step.
fn split_next_step(series: Vec<f64>, n_steps: usize) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(series.len(), n_steps + 1);
    let inputs = series[..n_steps].to_vec();
    let targets = series[1..].to_vec();
    (inputs, targets)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_shape(inputs: &[f64], targets: &[f64], n_steps: usize) {
        assert_eq!(inputs.len(), n_steps, "inputs length");
        assert_eq!(targets.len(), n_steps, "targets length");
        for &v in inputs.iter().chain(targets.iter()) {
            assert!(v.is_finite(), "non-finite dataset value: {}", v);
        }
    }

    #[test]
    fn smooth_sine_deterministic() {
        let (i1, t1) = smooth_sine(CronosSeed(1), 20).unwrap();
        let (i2, t2) = smooth_sine(CronosSeed(2), 20).unwrap();
        // smooth_sine ignores seed → identical regardless.
        assert_eq!(i1, i2);
        assert_eq!(t1, t2);
        check_shape(&i1, &t1, 20);
    }

    #[test]
    fn noisy_sine_seed_dependent() {
        let (i1, _) = noisy_sine(CronosSeed(1), 20).unwrap();
        let (i2, _) = noisy_sine(CronosSeed(2), 20).unwrap();
        assert_ne!(i1, i2, "different seeds should give different noisy sines");
        // Same seed twice ⇒ identical.
        let (i1b, _) = noisy_sine(CronosSeed(1), 20).unwrap();
        assert_eq!(i1, i1b);
    }

    #[test]
    fn regime_shift_glues_at_midpoint() {
        let (inputs, _) = regime_shift(CronosSeed(42), 40).unwrap();
        check_shape(&inputs, &inputs, 40);
        // The first half should be lower-magnitude on average than the
        // second half because the second regime has σ=0.5 vs σ=0.2.
        let first_mean_abs: f64 =
            inputs[..20].iter().map(|v| v.abs()).sum::<f64>() / 20.0;
        let second_mean_abs: f64 = inputs[20..].iter().map(|v| v.abs()).sum::<f64>() / 20.0;
        assert!(
            second_mean_abs > first_mean_abs,
            "regime shift should produce larger-magnitude second half, got {} vs {}",
            second_mean_abs, first_mean_abs,
        );
    }

    #[test]
    fn step_change_anomaly_is_localised() {
        let (inputs, _) = step_change_anomaly(CronosSeed(42), 20).unwrap();
        check_shape(&inputs, &inputs, 20);
        // First 10 steps are 0.0, last 10 are 1.0.
        for v in &inputs[..10] {
            assert_eq!(*v, 0.0);
        }
        for v in &inputs[10..] {
            assert_eq!(*v, 1.0);
        }
    }

    #[test]
    fn chaotic_spike_pattern_correct() {
        let (inputs, _) = chaotic_spike(CronosSeed(42), 30).unwrap();
        check_shape(&inputs, &inputs, 30);
        // Spikes at t=13, t=23 (t > 3 and (t-3) % 10 == 0).
        for (t, &v) in inputs.iter().enumerate() {
            let spike_expected = t > 3 && (t - 3) % 10 == 0;
            if spike_expected {
                assert!(
                    v.abs() > 2.0,
                    "expected spike at t={}, got value {}",
                    t, v
                );
            } else {
                assert!(
                    v.abs() < 2.0,
                    "expected non-spike at t={}, got value {}",
                    t, v
                );
            }
        }
    }

    #[test]
    fn dataset_enum_dispatch() {
        let n = 10;
        for ds in [
            CronosDataset::SmoothSine,
            CronosDataset::NoisySine,
            CronosDataset::RegimeShift,
            CronosDataset::StepChangeAnomaly,
            CronosDataset::ChaoticSpike,
        ] {
            let (inputs, targets) = ds.generate(CronosSeed(42), n).unwrap();
            check_shape(&inputs, &targets, n);
            assert!(!ds.label().is_empty());
        }
    }

    #[test]
    fn validate_n_steps_rejects_tiny() {
        let err = smooth_sine(CronosSeed(0), 1).unwrap_err();
        assert!(matches!(err, CronosGanError::InvalidConfig { .. }));
    }
}
