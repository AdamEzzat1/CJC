//! Phase 7b — eval-window analysis primitives.
//!
//! Two derived signals computed over the held-out eval rollouts that
//! `run_eval` produces:
//!
//! 1. **Predictive uncertainty** ([`compute_predictive_uncertainty`]):
//!    per-step `(mean, variance)` interpreting the SSM-Liquid pair as
//!    a 2-element ensemble over inductive biases. Variance is
//!    *epistemic* — it measures "how much our prediction depends on
//!    which inductive bias we picked", not noise in the underlying
//!    data.
//!
//! 2. **Disagreement segmentation** ([`segment_trajectory`]): walks a
//!    per-step gap trajectory with hysteresis thresholds, emitting
//!    `(start_step, end_step, Stable|Transitional)` segments.
//!    Hysteresis (enter > exit) prevents bouncing on borderline
//!    values.
//!
//! Both are pure functions of the inputs — no `Instant::now`, no
//! `HashMap`, no random draws — and produce byte-identical output for
//! the same input.
//!
//! ## Why "epistemic over inductive biases"
//!
//! Cronos GAN's two networks ARE an ensemble of 2 with deliberately
//! different inductive biases (linear-time-invariant SSM vs
//! nonlinear-time-varying Liquid NN). When they agree, the prediction
//! is bias-invariant — more trustworthy. When they disagree, the
//! prediction depends on which bias you chose. That's the textbook
//! definition of epistemic uncertainty.
//!
//! Aleatoric uncertainty (noise in the underlying data) requires
//! explicit noise modeling; this crate doesn't do that. Calling our
//! variance "epistemic over inductive biases" is the honest framing.

// ─── Per-step predictive uncertainty ─────────────────────────────────────

/// Compute per-step `(mean, variance)` from the SSM and Liquid eval
/// outputs, interpreting the pair as a 2-element ensemble.
///
/// For each timestep `t` and output dimension `d`:
/// ```text
/// mean(t, d)     = (ssm(t, d) + liq(t, d)) / 2
/// variance(t, d) = (ssm(t, d) - liq(t, d))^2 / 2
/// ```
///
/// The variance formula is the sample variance of a 2-point sample
/// around its own mean — `Σᵢ (xᵢ - x̄)² / (n - 1)` with `n = 2`.
///
/// When `output_dim > 1`, the returned vector is row-major
/// `[n_steps × output_dim]` of `(mean, variance)` tuples — same
/// layout as `ssm_outputs` / `liq_outputs`. For the typical 1-D
/// scalar case used by every shipped dataset, the vector is length
/// `n_steps`.
///
/// # Panics
///
/// Panics in debug builds if `ssm_outputs.len() != liq_outputs.len()`
/// or if the length is not a multiple of `output_dim`. Release-mode
/// builds are silently undefined on those inputs — the caller
/// (`run_eval`) guarantees the contract by construction.
pub fn compute_predictive_uncertainty(
    ssm_outputs: &[f64],
    liq_outputs: &[f64],
    output_dim: usize,
) -> Vec<(f64, f64)> {
    debug_assert_eq!(ssm_outputs.len(), liq_outputs.len());
    debug_assert!(output_dim > 0);
    debug_assert_eq!(ssm_outputs.len() % output_dim, 0);

    let n = ssm_outputs.len();
    let mut out = Vec::with_capacity(n);
    for (s, l) in ssm_outputs.iter().zip(liq_outputs.iter()) {
        let mean = (s + l) * 0.5;
        let diff = s - l;
        let variance = diff * diff * 0.5;
        out.push((mean, variance));
    }
    out
}

/// Convenience: extract the per-step absolute gap `|ssm - liq|` —
/// the natural per-step disagreement signal used by
/// [`segment_trajectory`]. For multi-D outputs, returns the per-step
/// L¹ gap aggregated across output dimensions; for 1-D, simply
/// `|ssm(t) - liq(t)|`.
pub fn compute_gap_trajectory(
    ssm_outputs: &[f64],
    liq_outputs: &[f64],
    output_dim: usize,
) -> Vec<f64> {
    debug_assert_eq!(ssm_outputs.len(), liq_outputs.len());
    debug_assert!(output_dim > 0);
    debug_assert_eq!(ssm_outputs.len() % output_dim, 0);

    let n_steps = ssm_outputs.len() / output_dim;
    let mut out = Vec::with_capacity(n_steps);
    for t in 0..n_steps {
        let start = t * output_dim;
        let end = start + output_dim;
        let mut acc = cjc_repro::KahanAccumulatorF64::new();
        for (s, l) in ssm_outputs[start..end]
            .iter()
            .zip(liq_outputs[start..end].iter())
        {
            acc.add((s - l).abs());
        }
        out.push(acc.finalize());
    }
    out
}

// ─── Segmentation ────────────────────────────────────────────────────────

/// Label for a contiguous run of eval-window timesteps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SegmentLabel {
    /// Gap is below `exit_threshold` (and never crossed
    /// `enter_threshold` since the last transition out of
    /// Transitional). The two networks agree.
    Stable,
    /// Gap crossed above `enter_threshold` and hasn't yet dropped
    /// below `exit_threshold`. The two networks disagree — candidate
    /// regime-shift region.
    Transitional,
}

impl SegmentLabel {
    pub fn as_str(self) -> &'static str {
        match self {
            SegmentLabel::Stable => "stable",
            SegmentLabel::Transitional => "transitional",
        }
    }
}

/// One contiguous run of timesteps assigned the same label.
/// `start_step` is inclusive, `end_step` is exclusive — Rust's
/// half-open range convention. The trajectory `[start_step,
/// end_step)` contains `end_step - start_step` timesteps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    pub start_step: usize,
    pub end_step: usize,
    pub label: SegmentLabel,
}

impl Segment {
    pub fn len(&self) -> usize {
        self.end_step.saturating_sub(self.start_step)
    }

    pub fn is_empty(&self) -> bool {
        self.end_step <= self.start_step
    }
}

/// Tuning knobs for [`segment_trajectory`]. The default values are
/// the same constants used by `run_eval` to populate
/// [`crate::EvalReport::segments`] — see those for the rationale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegmentationConfig {
    /// A timestep's gap above this threshold flips the state from
    /// `Stable` to `Transitional`.
    pub enter_threshold: f64,
    /// A timestep's gap below this threshold flips the state from
    /// `Transitional` to `Stable`. Must be strictly less than
    /// `enter_threshold` for the hysteresis property to hold.
    pub exit_threshold: f64,
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            enter_threshold: DEFAULT_SEGMENTATION_ENTER,
            exit_threshold: DEFAULT_SEGMENTATION_EXIT,
        }
    }
}

/// Default segmentation entry threshold. A timestep's per-step
/// absolute gap above this value flips the segment state to
/// `Transitional`. Picked against Phase 4c eval-window observations
/// (`eval_absolute_gap` typically in `[0.1, 1.7]` across cells).
pub const DEFAULT_SEGMENTATION_ENTER: f64 = 0.5;

/// Default segmentation exit threshold (strictly less than
/// [`DEFAULT_SEGMENTATION_ENTER`] for hysteresis). Falling below
/// this flips back to `Stable`.
pub const DEFAULT_SEGMENTATION_EXIT: f64 = 0.3;

/// Walk a per-step gap trajectory with hysteresis thresholds and
/// emit `(start_step, end_step, label)` segments.
///
/// Semantics:
/// - Start in `Stable` state.
/// - For each step `t` with gap `g`:
///   - If state is `Stable` and `g >= enter`: open a new
///     `Transitional` segment at `t`.
///   - If state is `Transitional` and `g < exit`: open a new
///     `Stable` segment at `t`.
///   - Else: continue the current segment.
/// - At the end of the trajectory, close the last open segment.
///
/// Hysteresis (enter > exit) prevents bouncing on values in the
/// `[exit, enter)` band — those values continue the current state
/// without transition.
///
/// Returns an empty `Vec` for an empty input.
///
/// # Edge cases
///
/// - `enter_threshold <= exit_threshold` is malformed; the function
///   still terminates and returns a segmentation, but the hysteresis
///   no-bounce property is degraded. Callers are expected to enforce
///   the invariant.
/// - NaN gap values: treated as "below `exit_threshold`" (i.e. they
///   do not enter Transitional). This matches the convention that
///   NaN-on-eval means "no signal available."
pub fn segment_trajectory(gaps: &[f64], cfg: SegmentationConfig) -> Vec<Segment> {
    if gaps.is_empty() {
        return Vec::new();
    }
    let mut segments = Vec::new();
    let mut current_label = SegmentLabel::Stable;
    let mut current_start: usize = 0;

    for (t, &g) in gaps.iter().enumerate() {
        let g_for_compare = if g.is_nan() { 0.0 } else { g };
        let next_label = match current_label {
            SegmentLabel::Stable => {
                if g_for_compare >= cfg.enter_threshold {
                    SegmentLabel::Transitional
                } else {
                    SegmentLabel::Stable
                }
            }
            SegmentLabel::Transitional => {
                if g_for_compare < cfg.exit_threshold {
                    SegmentLabel::Stable
                } else {
                    SegmentLabel::Transitional
                }
            }
        };
        if next_label != current_label {
            segments.push(Segment {
                start_step: current_start,
                end_step: t,
                label: current_label,
            });
            current_label = next_label;
            current_start = t;
        }
    }
    // Close the final open segment at the end of the trajectory.
    segments.push(Segment {
        start_step: current_start,
        end_step: gaps.len(),
        label: current_label,
    });
    segments
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // § compute_predictive_uncertainty ──────────────────────────────

    #[test]
    fn uncertainty_zero_when_predictions_match_exactly() {
        let ssm = vec![1.0, 2.0, 3.0, 4.0];
        let liq = vec![1.0, 2.0, 3.0, 4.0];
        let u = compute_predictive_uncertainty(&ssm, &liq, 1);
        for (mean, var) in u {
            assert!(var == 0.0, "var should be 0 when predictions match");
            assert!(mean >= 1.0 && mean <= 4.0);
        }
    }

    #[test]
    fn uncertainty_matches_two_point_sample_variance() {
        // Two predictions x1, x2: sample mean = (x1+x2)/2, sample
        // variance with n-1 = 1 is (x1 - mean)² + (x2 - mean)² =
        // 2 * ((x1-x2)/2)² = (x1-x2)²/2.
        let ssm = vec![1.0, 5.0];
        let liq = vec![3.0, 9.0];
        let u = compute_predictive_uncertainty(&ssm, &liq, 1);
        // Step 0: ssm=1, liq=3, mean=2, var=(1-3)²/2=2.
        assert_eq!(u[0].0, 2.0);
        assert_eq!(u[0].1, 2.0);
        // Step 1: ssm=5, liq=9, mean=7, var=(5-9)²/2=8.
        assert_eq!(u[1].0, 7.0);
        assert_eq!(u[1].1, 8.0);
    }

    #[test]
    fn uncertainty_length_matches_input() {
        let ssm = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let liq = vec![0.15, 0.18, 0.31, 0.45, 0.49];
        let u = compute_predictive_uncertainty(&ssm, &liq, 1);
        assert_eq!(u.len(), 5);
    }

    // § compute_gap_trajectory ──────────────────────────────────────

    #[test]
    fn gap_trajectory_returns_per_step_abs_diff_for_1d_output() {
        let ssm = vec![1.0, 2.0, 5.0];
        let liq = vec![1.5, 1.5, 7.0];
        let g = compute_gap_trajectory(&ssm, &liq, 1);
        assert_eq!(g.len(), 3);
        assert_eq!(g[0], 0.5);
        assert_eq!(g[1], 0.5);
        assert_eq!(g[2], 2.0);
    }

    #[test]
    fn gap_trajectory_aggregates_l1_across_output_dim() {
        // output_dim = 2, so per-step gap = |s1-l1| + |s2-l2|.
        let ssm = vec![1.0, 2.0, /* step 1 */ 3.0, 4.0];
        let liq = vec![1.5, 1.5, /* step 1 */ 3.0, 5.0];
        let g = compute_gap_trajectory(&ssm, &liq, 2);
        assert_eq!(g.len(), 2);
        // step 0: |1-1.5| + |2-1.5| = 0.5 + 0.5 = 1.0
        assert_eq!(g[0], 1.0);
        // step 1: |3-3| + |4-5| = 0 + 1 = 1.0
        assert_eq!(g[1], 1.0);
    }

    // § segment_trajectory ──────────────────────────────────────────

    #[test]
    fn segmentation_empty_input_returns_empty_segments() {
        let segs = segment_trajectory(&[], SegmentationConfig::default());
        assert!(segs.is_empty());
    }

    #[test]
    fn segmentation_all_low_gaps_returns_single_stable_segment() {
        let gaps = vec![0.0, 0.1, 0.2, 0.1, 0.05];
        let segs = segment_trajectory(&gaps, SegmentationConfig::default());
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].label, SegmentLabel::Stable);
        assert_eq!(segs[0].start_step, 0);
        assert_eq!(segs[0].end_step, 5);
    }

    #[test]
    fn segmentation_all_high_gaps_starts_stable_then_transitional() {
        // At step 0, state is Stable; gap=1.0 ≥ enter=0.5 ⇒
        // transition to Transitional at step 0. So the first segment
        // is Stable[0,0) (empty) and the rest is Transitional. To
        // keep segments non-empty, we collapse an empty leading
        // Stable below — actually current impl always emits the
        // closing transition, so we expect:
        //   Stable[0, 0), Transitional[0, n)
        // Test the SECOND segment is the meaningful one.
        let gaps = vec![1.0, 0.8, 0.9, 0.7];
        let segs = segment_trajectory(&gaps, SegmentationConfig::default());
        assert!(!segs.is_empty());
        let last = segs.last().unwrap();
        assert_eq!(last.label, SegmentLabel::Transitional);
        assert_eq!(last.end_step, 4);
    }

    #[test]
    fn segmentation_low_high_low_yields_three_segments() {
        // Steps 0-1: low (Stable), 2-4: high (Transitional), 5-6: low (Stable).
        let gaps = vec![0.0, 0.1, 0.6, 0.7, 0.55, 0.2, 0.0];
        let segs = segment_trajectory(&gaps, SegmentationConfig::default());
        // Expect three non-empty segments in order.
        let non_empty: Vec<&Segment> = segs.iter().filter(|s| !s.is_empty()).collect();
        assert_eq!(non_empty.len(), 3);
        assert_eq!(non_empty[0].label, SegmentLabel::Stable);
        assert_eq!(non_empty[0].start_step, 0);
        assert_eq!(non_empty[0].end_step, 2);
        assert_eq!(non_empty[1].label, SegmentLabel::Transitional);
        assert_eq!(non_empty[1].start_step, 2);
        assert_eq!(non_empty[1].end_step, 5);
        assert_eq!(non_empty[2].label, SegmentLabel::Stable);
        assert_eq!(non_empty[2].start_step, 5);
        assert_eq!(non_empty[2].end_step, 7);
    }

    #[test]
    fn segmentation_hysteresis_band_does_not_bounce() {
        // Values in the [exit=0.3, enter=0.5) band should NOT cause a
        // transition. Once Transitional, only g < 0.3 ends it.
        let gaps = vec![0.6, 0.4, 0.4, 0.4, 0.35, 0.4];
        let segs = segment_trajectory(&gaps, SegmentationConfig::default());
        let non_empty: Vec<&Segment> = segs.iter().filter(|s| !s.is_empty()).collect();
        // Expected: Transitional[0, 6) — once we entered Transitional
        // at step 0, all subsequent values are in the hysteresis band
        // and we never drop below 0.3.
        assert_eq!(non_empty.len(), 1);
        assert_eq!(non_empty[0].label, SegmentLabel::Transitional);
        assert_eq!(non_empty[0].end_step, 6);
    }

    #[test]
    fn segmentation_nan_gap_treated_as_below_exit() {
        // NaN means "no signal." In Stable state, NaN stays Stable.
        let gaps = vec![0.0, f64::NAN, 0.1, f64::NAN, 0.05];
        let segs = segment_trajectory(&gaps, SegmentationConfig::default());
        let non_empty: Vec<&Segment> = segs.iter().filter(|s| !s.is_empty()).collect();
        assert_eq!(non_empty.len(), 1);
        assert_eq!(non_empty[0].label, SegmentLabel::Stable);
    }

    #[test]
    fn segmentation_is_deterministic_across_runs() {
        let gaps = vec![0.0, 0.4, 0.7, 0.6, 0.2, 0.55, 0.8, 0.1];
        let a = segment_trajectory(&gaps, SegmentationConfig::default());
        let b = segment_trajectory(&gaps, SegmentationConfig::default());
        assert_eq!(a, b);
    }

    #[test]
    fn segments_cover_full_range_when_non_empty() {
        // Property: segments[0].start_step == 0 and the union of all
        // segments == [0, gaps.len()).
        let gaps = vec![0.1, 0.6, 0.2, 0.7, 0.1];
        let segs = segment_trajectory(&gaps, SegmentationConfig::default());
        assert_eq!(segs.first().unwrap().start_step, 0);
        assert_eq!(segs.last().unwrap().end_step, gaps.len());
        // Each segment's end_step == next segment's start_step.
        for w in segs.windows(2) {
            assert_eq!(w[0].end_step, w[1].start_step);
        }
    }
}
