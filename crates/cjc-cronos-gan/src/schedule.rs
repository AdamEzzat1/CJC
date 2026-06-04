//! Phase 4d: λ-schedules for the asymmetric-mode challenger weight.
//!
//! `lambda_disagreement` used to be a flat `f64` carried on
//! [`crate::TemporalGanConfig`]. Phase 4d generalises it to
//! [`LambdaSchedule`] so the challenger weight can vary across training
//! steps. Each non-constant variant stores its own `n_train_steps`
//! horizon, so [`LambdaSchedule::lambda_at`] is fully self-contained —
//! the trainer just feeds in its own step counter.
//!
//! ## Variants
//!
//! - [`LambdaSchedule::Constant`] — equivalent to the pre-Phase-4d
//!   `f64` field; convenience constructors keep wrapping plain scalars
//!   into this variant so existing call sites don't break.
//! - [`LambdaSchedule::Linear`] — linear interpolation from `start`
//!   (step 0) to `end` (step `n_train_steps`), clamped to `end`
//!   afterwards. Useful for regularisation warmup that decays into
//!   vanilla supervised behaviour.
//! - [`LambdaSchedule::ExponentialDecay`] —
//!   `start · exp(-decay_rate · t / n_train_steps)`. Smooth monotonic
//!   decay; never reaches 0 exactly. Clamped to its terminal value past
//!   `n_train_steps`.
//! - [`LambdaSchedule::WarmupThenLinear`] — `start` for the first
//!   `warmup_steps` calls, then linear interpolation from `start` to
//!   `end` over `[warmup_steps, n_train_steps]`. Lets the predictor
//!   stabilise before the challenger pressure ramps up.
//!
//! ## Determinism contract
//!
//! [`LambdaSchedule::canonical_bytes`] emits a fixed-layout byte string
//! with a distinct tag per variant and the bit pattern of every field.
//! Two schedules with the same tag but different field values produce
//! different bytes — this is what makes the schedule safe to feed into
//! `TemporalGanConfig::canonical_bytes` and downstream replay hashes.

/// Schedule for the asymmetric-mode challenger weight λ.
///
/// See module docs for variant semantics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LambdaSchedule {
    /// λ is the same value on every training step.
    Constant(f64),
    /// λ decays linearly from `start` at step 0 to `end` at step
    /// `n_train_steps`. Clamped to `end` for any step beyond.
    Linear {
        start: f64,
        end: f64,
        n_train_steps: u64,
    },
    /// λ_t = `start` · exp(-`decay_rate` · t / `n_train_steps`).
    /// `decay_rate = 0` collapses to `Constant(start)`. Clamped to its
    /// step-`n_train_steps` value beyond that horizon.
    ExponentialDecay {
        start: f64,
        decay_rate: f64,
        n_train_steps: u64,
    },
    /// λ stays at `start` for the first `warmup_steps` calls, then
    /// linearly interpolates from `start` to `end` over
    /// `[warmup_steps, n_train_steps]`. Clamped to `end` afterwards.
    WarmupThenLinear {
        start: f64,
        end: f64,
        warmup_steps: u64,
        n_train_steps: u64,
    },
}

impl Default for LambdaSchedule {
    /// `Constant(0.0)` — symmetric mode default.
    fn default() -> Self {
        LambdaSchedule::Constant(0.0)
    }
}

impl From<f64> for LambdaSchedule {
    /// `f64 → Constant(value)`. Powers the back-compat shim from the
    /// pre-Phase-4d `lambda_disagreement: f64` API.
    fn from(value: f64) -> Self {
        LambdaSchedule::Constant(value)
    }
}

impl LambdaSchedule {
    /// λ value at training step `step` (0-indexed). For schedules with
    /// a bounded `n_train_steps`, callers asking past that horizon
    /// receive the terminal value (clamped).
    pub fn lambda_at(&self, step: u64) -> f64 {
        match *self {
            LambdaSchedule::Constant(v) => v,
            LambdaSchedule::Linear {
                start,
                end,
                n_train_steps,
            } => {
                if n_train_steps == 0 {
                    return end;
                }
                if step >= n_train_steps {
                    return end;
                }
                let t = step as f64;
                let n = n_train_steps as f64;
                start + (end - start) * (t / n)
            }
            LambdaSchedule::ExponentialDecay {
                start,
                decay_rate,
                n_train_steps,
            } => {
                if n_train_steps == 0 {
                    // Degenerate: every step "is" the terminal step.
                    return start * (-decay_rate).exp();
                }
                let t_clamped = step.min(n_train_steps) as f64;
                let n = n_train_steps as f64;
                start * (-decay_rate * t_clamped / n).exp()
            }
            LambdaSchedule::WarmupThenLinear {
                start,
                end,
                warmup_steps,
                n_train_steps,
            } => {
                if step < warmup_steps {
                    return start;
                }
                if step >= n_train_steps || n_train_steps <= warmup_steps {
                    return end;
                }
                let t = (step - warmup_steps) as f64;
                let n = (n_train_steps - warmup_steps) as f64;
                start + (end - start) * (t / n)
            }
        }
    }

    /// Validate that every λ value this schedule will emit is finite and
    /// non-negative — the precondition for the asymmetric-mode
    /// challenger loss to be well-defined.
    ///
    /// Returns `Err(detail)` if any field is non-finite or if the
    /// schedule could emit a negative λ (e.g. `Linear { start: -0.1 }`).
    pub fn validate_non_negative_and_finite(&self) -> Result<(), String> {
        match *self {
            LambdaSchedule::Constant(v) => {
                if !v.is_finite() || v < 0.0 {
                    Err(format!(
                        "Constant λ must be finite and >= 0, got {}",
                        v
                    ))
                } else {
                    Ok(())
                }
            }
            LambdaSchedule::Linear {
                start, end, ..
            } => {
                if !start.is_finite() || start < 0.0 {
                    return Err(format!(
                        "Linear λ start must be finite and >= 0, got {}",
                        start
                    ));
                }
                if !end.is_finite() || end < 0.0 {
                    return Err(format!(
                        "Linear λ end must be finite and >= 0, got {}",
                        end
                    ));
                }
                Ok(())
            }
            LambdaSchedule::ExponentialDecay {
                start,
                decay_rate,
                ..
            } => {
                if !start.is_finite() || start < 0.0 {
                    return Err(format!(
                        "ExponentialDecay λ start must be finite and >= 0, got {}",
                        start
                    ));
                }
                if !decay_rate.is_finite() {
                    return Err(format!(
                        "ExponentialDecay decay_rate must be finite, got {}",
                        decay_rate
                    ));
                }
                Ok(())
            }
            LambdaSchedule::WarmupThenLinear {
                start, end, ..
            } => {
                if !start.is_finite() || start < 0.0 {
                    return Err(format!(
                        "WarmupThenLinear λ start must be finite and >= 0, got {}",
                        start
                    ));
                }
                if !end.is_finite() || end < 0.0 {
                    return Err(format!(
                        "WarmupThenLinear λ end must be finite and >= 0, got {}",
                        end
                    ));
                }
                Ok(())
            }
        }
    }

    /// Canonical byte layout for replay-hash composition. Distinct tag
    /// per variant + LE bit pattern of every field. Same variant +
    /// different fields ⇒ different bytes.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(40);
        match *self {
            LambdaSchedule::Constant(v) => {
                bytes.push(0x00);
                bytes.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            LambdaSchedule::Linear {
                start,
                end,
                n_train_steps,
            } => {
                bytes.push(0x01);
                bytes.extend_from_slice(&start.to_bits().to_le_bytes());
                bytes.extend_from_slice(&end.to_bits().to_le_bytes());
                bytes.extend_from_slice(&n_train_steps.to_le_bytes());
            }
            LambdaSchedule::ExponentialDecay {
                start,
                decay_rate,
                n_train_steps,
            } => {
                bytes.push(0x02);
                bytes.extend_from_slice(&start.to_bits().to_le_bytes());
                bytes.extend_from_slice(&decay_rate.to_bits().to_le_bytes());
                bytes.extend_from_slice(&n_train_steps.to_le_bytes());
            }
            LambdaSchedule::WarmupThenLinear {
                start,
                end,
                warmup_steps,
                n_train_steps,
            } => {
                bytes.push(0x03);
                bytes.extend_from_slice(&start.to_bits().to_le_bytes());
                bytes.extend_from_slice(&end.to_bits().to_le_bytes());
                bytes.extend_from_slice(&warmup_steps.to_le_bytes());
                bytes.extend_from_slice(&n_train_steps.to_le_bytes());
            }
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── § Constant ──────────────────────────────────────────────────────

    #[test]
    fn constant_returns_same_value_every_step() {
        let s = LambdaSchedule::Constant(0.42);
        for step in [0_u64, 1, 5, 100, 10_000, u64::MAX] {
            assert_eq!(s.lambda_at(step).to_bits(), 0.42_f64.to_bits());
        }
    }

    // ─── § Linear ────────────────────────────────────────────────────────

    #[test]
    fn linear_interpolates_between_start_and_end() {
        let s = LambdaSchedule::Linear {
            start: 1.0,
            end: 0.0,
            n_train_steps: 10,
        };
        assert_eq!(s.lambda_at(0).to_bits(), 1.0_f64.to_bits());
        assert_eq!(s.lambda_at(10).to_bits(), 0.0_f64.to_bits());
        // Midpoint
        assert!((s.lambda_at(5) - 0.5).abs() < 1e-12);
        // 1/4 of the way
        assert!((s.lambda_at(2) - 0.8).abs() < 1e-12);
    }

    #[test]
    fn linear_clamps_past_n_train_steps() {
        let s = LambdaSchedule::Linear {
            start: 1.0,
            end: 0.0,
            n_train_steps: 10,
        };
        // Past the end → terminal value.
        for step in [10_u64, 11, 100, u64::MAX] {
            assert_eq!(s.lambda_at(step).to_bits(), 0.0_f64.to_bits());
        }
    }

    #[test]
    fn linear_with_zero_n_train_steps_returns_end() {
        let s = LambdaSchedule::Linear {
            start: 1.0,
            end: 0.5,
            n_train_steps: 0,
        };
        for step in [0_u64, 1, 100] {
            assert_eq!(s.lambda_at(step).to_bits(), 0.5_f64.to_bits());
        }
    }

    // ─── § ExponentialDecay ──────────────────────────────────────────────

    #[test]
    fn exponential_decay_matches_closed_form() {
        let s = LambdaSchedule::ExponentialDecay {
            start: 1.0,
            decay_rate: 2.0_f64.ln(), // exp(-ln 2) = 1/2 per `n_train_steps`
            n_train_steps: 10,
        };
        assert_eq!(s.lambda_at(0).to_bits(), 1.0_f64.to_bits());
        // At n_train_steps: start * exp(-ln 2) = 0.5
        assert!((s.lambda_at(10) - 0.5).abs() < 1e-12);
        // Halfway: start * exp(-ln 2 / 2) = 1/sqrt(2)
        assert!((s.lambda_at(5) - 0.5_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn exponential_decay_clamps_past_n_train_steps() {
        let s = LambdaSchedule::ExponentialDecay {
            start: 1.0,
            decay_rate: 2.0_f64.ln(),
            n_train_steps: 10,
        };
        let v_n = s.lambda_at(10);
        for step in [11_u64, 100, u64::MAX] {
            assert_eq!(s.lambda_at(step).to_bits(), v_n.to_bits());
        }
    }

    #[test]
    fn exponential_decay_with_zero_decay_rate_is_constant() {
        let s = LambdaSchedule::ExponentialDecay {
            start: 0.3,
            decay_rate: 0.0,
            n_train_steps: 100,
        };
        for step in [0_u64, 50, 100, 1000] {
            assert!((s.lambda_at(step) - 0.3).abs() < 1e-14);
        }
    }

    // ─── § WarmupThenLinear ──────────────────────────────────────────────

    #[test]
    fn warmup_then_linear_stays_constant_during_warmup() {
        let s = LambdaSchedule::WarmupThenLinear {
            start: 0.2,
            end: 0.0,
            warmup_steps: 5,
            n_train_steps: 15,
        };
        for step in [0_u64, 1, 2, 3, 4] {
            assert_eq!(s.lambda_at(step).to_bits(), 0.2_f64.to_bits());
        }
        // At warmup boundary, still `start` (the linear phase begins HERE).
        assert_eq!(s.lambda_at(5).to_bits(), 0.2_f64.to_bits());
        // At n_train_steps, `end`.
        assert_eq!(s.lambda_at(15).to_bits(), 0.0_f64.to_bits());
        // Halfway through the linear phase: 0.2 + (0 - 0.2) * 0.5 = 0.1
        assert!((s.lambda_at(10) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn warmup_then_linear_clamps_past_n_train_steps() {
        let s = LambdaSchedule::WarmupThenLinear {
            start: 0.2,
            end: 0.0,
            warmup_steps: 5,
            n_train_steps: 15,
        };
        for step in [15_u64, 16, 100, u64::MAX] {
            assert_eq!(s.lambda_at(step).to_bits(), 0.0_f64.to_bits());
        }
    }

    #[test]
    fn warmup_at_or_past_n_train_steps_degenerates_to_end() {
        // Bizarre but legal config: warmup_steps >= n_train_steps.
        // Interpretation: skip the linear phase entirely.
        let s = LambdaSchedule::WarmupThenLinear {
            start: 0.5,
            end: 0.0,
            warmup_steps: 20,
            n_train_steps: 10,
        };
        // During the warmup window (which extends past n_train_steps),
        // we still emit `start` for steps below warmup.
        assert_eq!(s.lambda_at(0).to_bits(), 0.5_f64.to_bits());
        assert_eq!(s.lambda_at(19).to_bits(), 0.5_f64.to_bits());
        // At/past warmup, jump to `end`.
        assert_eq!(s.lambda_at(20).to_bits(), 0.0_f64.to_bits());
    }

    // ─── § Conversions & defaults ────────────────────────────────────────

    #[test]
    fn from_f64_wraps_into_constant() {
        let s: LambdaSchedule = 0.7_f64.into();
        assert_eq!(s, LambdaSchedule::Constant(0.7));
        assert_eq!(s.lambda_at(0).to_bits(), 0.7_f64.to_bits());
    }

    #[test]
    fn default_is_constant_zero() {
        let s = LambdaSchedule::default();
        assert_eq!(s, LambdaSchedule::Constant(0.0));
    }

    // ─── § Canonical bytes ───────────────────────────────────────────────

    #[test]
    fn canonical_bytes_distinct_per_variant_with_same_fields() {
        // Same numeric "0.0" everywhere — the tag byte must be the
        // disambiguator.
        let c = LambdaSchedule::Constant(0.0).canonical_bytes();
        let l = LambdaSchedule::Linear {
            start: 0.0,
            end: 0.0,
            n_train_steps: 0,
        }
        .canonical_bytes();
        let e = LambdaSchedule::ExponentialDecay {
            start: 0.0,
            decay_rate: 0.0,
            n_train_steps: 0,
        }
        .canonical_bytes();
        let w = LambdaSchedule::WarmupThenLinear {
            start: 0.0,
            end: 0.0,
            warmup_steps: 0,
            n_train_steps: 0,
        }
        .canonical_bytes();
        assert_ne!(c, l);
        assert_ne!(c, e);
        assert_ne!(c, w);
        assert_ne!(l, e);
        assert_ne!(l, w);
        assert_ne!(e, w);
    }

    #[test]
    fn canonical_bytes_distinct_per_field_within_variant() {
        // Two Linear schedules with the same tag but different fields
        // must produce different bytes — this is the gate that protects
        // replay_hash from "same tag, different ramp" collisions.
        let a = LambdaSchedule::Linear {
            start: 0.1,
            end: 0.0,
            n_train_steps: 100,
        }
        .canonical_bytes();
        let b = LambdaSchedule::Linear {
            start: 0.2,
            end: 0.0,
            n_train_steps: 100,
        }
        .canonical_bytes();
        let c = LambdaSchedule::Linear {
            start: 0.1,
            end: 0.0,
            n_train_steps: 200,
        }
        .canonical_bytes();
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
    }

    #[test]
    fn canonical_bytes_byte_identical_for_equal_schedules() {
        let s1 = LambdaSchedule::ExponentialDecay {
            start: 0.5,
            decay_rate: 1.5,
            n_train_steps: 200,
        };
        let s2 = LambdaSchedule::ExponentialDecay {
            start: 0.5,
            decay_rate: 1.5,
            n_train_steps: 200,
        };
        assert_eq!(s1.canonical_bytes(), s2.canonical_bytes());
    }

    // ─── § Validation ────────────────────────────────────────────────────

    #[test]
    fn validate_rejects_negative_constant() {
        let s = LambdaSchedule::Constant(-0.1);
        assert!(s.validate_non_negative_and_finite().is_err());
    }

    #[test]
    fn validate_rejects_nan_constant() {
        let s = LambdaSchedule::Constant(f64::NAN);
        assert!(s.validate_non_negative_and_finite().is_err());
    }

    #[test]
    fn validate_rejects_negative_linear_start_or_end() {
        let neg_start = LambdaSchedule::Linear {
            start: -0.1,
            end: 0.0,
            n_train_steps: 10,
        };
        let neg_end = LambdaSchedule::Linear {
            start: 0.1,
            end: -0.05,
            n_train_steps: 10,
        };
        assert!(neg_start.validate_non_negative_and_finite().is_err());
        assert!(neg_end.validate_non_negative_and_finite().is_err());
    }

    #[test]
    fn validate_accepts_well_formed_schedules() {
        let cases = [
            LambdaSchedule::Constant(0.0),
            LambdaSchedule::Constant(0.1),
            LambdaSchedule::Linear {
                start: 0.2,
                end: 0.0,
                n_train_steps: 100,
            },
            LambdaSchedule::ExponentialDecay {
                start: 0.5,
                decay_rate: 1.0,
                n_train_steps: 200,
            },
            LambdaSchedule::WarmupThenLinear {
                start: 0.1,
                end: 0.0,
                warmup_steps: 10,
                n_train_steps: 100,
            },
        ];
        for s in cases {
            assert!(
                s.validate_non_negative_and_finite().is_ok(),
                "expected ok for {:?}",
                s
            );
        }
    }
}
