//! CANA ↔ NSS pressure-density bridge.
//!
//! Translates a [`CompressionReport`] into a per-kind delta on a
//! [`PressureDensityState`], modelling the physical effect of executing
//! the report's plan:
//!
//! - **Memory pressure ↓** by the *compression reward* — successfully
//!   compressing pass histories and motifs reduces in-memory footprint.
//!   We scale by the ratio between original and compressed bytes, so a
//!   2× ratio gives `Δmemory = -0.5 * (compressed/original)` per entry.
//! - **Throughput pressure ↓** by a small amount per validated lossless
//!   entry — fewer bytes to traverse during downstream NSS replay means
//!   throughput pressure relaxes.
//! - **Memory pressure ↑** for malformed/decode-failed entries — a
//!   broken roundtrip increases reconstruction work that the runtime
//!   would have to absorb.
//! - **Thermal pressure ↑** by `α · observed_error` for advisory entries
//!   that survived their tolerance check — the lossy reconstruction may
//!   require additional compute downstream.
//! - **Thermal pressure ↑** by a saturating constant for entries that
//!   exceeded tolerance (those plans would *not* be applied, but the
//!   bridge logs the "risk if applied" for the energy ranker).
//!
//! These deltas are *advisory*: nothing in the bridge changes MIR.
//! Downstream consumers (the energy ranker) read the resulting
//! [`PressureDensityState`] / [`PressureCorrelationSummary`] and
//! re-rank candidate plans accordingly.
//!
//! ## Determinism contract
//!
//! - Iteration over [`CompressionReport::entries`] is slot-order — the
//!   plan already sorted by candidate ID, so the bridge inherits that.
//! - All deltas are computed as plain `f64` arithmetic with no
//!   accumulator state across entries (deltas are applied in-place to
//!   the state).
//! - The returned [`CompressionPressureDelta`] is a deterministic
//!   summary of *what the bridge did*; its hash is FNV-1a over the
//!   final density state's canonical bytes.

use cjc_nss::{PressureCorrelationSummary, PressureDensityState, PressureKind};

use crate::candidate::{CompressionKind, CriticalityTag};
use crate::report::{CompressionReport, EntryStatus};

/// Per-entry-cost coefficients used when translating a compression
/// report into a pressure-density delta.
///
/// All values are non-negative `f64`. The defaults are conservative
/// hand-picked weights from the prompt's `energy = costs - rewards`
/// shape; users tuning for a specific workload can override
/// individual coefficients before calling [`compression_pressure_delta`].
#[derive(Debug, Clone, Copy)]
pub struct BridgeCoefficients {
    /// Multiplier on `(original - compressed) / original` for the
    /// memory-pressure reduction. Range `[0, 1]`; default `0.6`.
    pub memory_reward_scale: f64,
    /// Constant per validated lossless entry subtracted from throughput
    /// pressure. Default `0.02`.
    pub throughput_reward_per_validated: f64,
    /// Multiplier on `observed_error` for the thermal-pressure increase
    /// of validated advisory entries. Default `0.1`.
    pub thermal_advisory_scale: f64,
    /// Penalty added to memory pressure for malformed lossless
    /// round-trips. Default `0.1`.
    pub memory_malformed_penalty: f64,
    /// Penalty added to thermal pressure for tolerance-exceeded
    /// advisory entries. Default `0.2`.
    pub thermal_tolerance_exceeded_penalty: f64,
}

impl Default for BridgeCoefficients {
    fn default() -> Self {
        Self {
            memory_reward_scale: 0.6,
            throughput_reward_per_validated: 0.02,
            thermal_advisory_scale: 0.1,
            memory_malformed_penalty: 0.1,
            thermal_tolerance_exceeded_penalty: 0.2,
        }
    }
}

impl BridgeCoefficients {
    /// All-zero coefficients: the bridge becomes a no-op (useful for
    /// tests that want to verify the bridge mutates nothing if asked
    /// not to).
    pub fn zero() -> Self {
        Self {
            memory_reward_scale: 0.0,
            throughput_reward_per_validated: 0.0,
            thermal_advisory_scale: 0.0,
            memory_malformed_penalty: 0.0,
            thermal_tolerance_exceeded_penalty: 0.0,
        }
    }
}

/// Output of [`compression_pressure_delta`].
#[derive(Debug, Clone)]
pub struct CompressionPressureDelta {
    /// The new density state after applying the per-entry deltas.
    pub updated: PressureDensityState,
    /// Summary view of `updated`.
    pub summary: PressureCorrelationSummary,
    /// Aggregate Δmemory_pressure (signed; negative means pressure ↓).
    pub delta_memory: f64,
    /// Aggregate Δthermal_pressure.
    pub delta_thermal: f64,
    /// Aggregate Δthroughput_pressure.
    pub delta_throughput: f64,
    /// Number of entries that contributed a memory reward.
    pub rewarded_entries: u32,
    /// Number of entries that triggered a penalty (malformed or
    /// tolerance-exceeded).
    pub penalised_entries: u32,
}

/// Apply a compression report's deltas to a baseline pressure-density
/// state and return the updated state + summary.
///
/// `baseline` is taken by value because the returned `updated` is a
/// modified copy; callers who want to apply deltas in-place can `clone`
/// before passing and replace afterwards.
pub fn compression_pressure_delta(
    baseline: PressureDensityState,
    report: &CompressionReport,
    coefficients: BridgeCoefficients,
) -> CompressionPressureDelta {
    let mut updated = baseline;
    let mut delta_memory: f64 = 0.0;
    let mut delta_thermal: f64 = 0.0;
    let mut delta_throughput: f64 = 0.0;
    let mut rewarded_entries: u32 = 0;
    let mut penalised_entries: u32 = 0;

    for entry in report.entries() {
        match &entry.status {
            EntryStatus::Validated => {
                let is_lossless = matches!(
                    entry.kind,
                    CompressionKind::LosslessTrace | CompressionKind::MotifDictionary
                );
                // Memory-pressure relaxation proportional to ratio savings.
                let ratio = entry.ratio();
                if entry.original_len > 0 && ratio <= 1.0 {
                    let saving = 1.0 - ratio; // 0..1
                    let dmem = -coefficients.memory_reward_scale * saving;
                    delta_memory += dmem;
                    rewarded_entries += 1;
                }
                if is_lossless {
                    // Lossless wins also relax throughput a small amount.
                    delta_throughput -= coefficients.throughput_reward_per_validated;
                }
                // Advisory entries' observed_error is a thermal-pressure
                // tax (reconstruction does more downstream work).
                if matches!(entry.criticality_tag, CriticalityTag::AdvisoryOnly)
                    && entry.observed_error > 0.0
                {
                    delta_thermal += coefficients.thermal_advisory_scale * entry.observed_error;
                }
            }
            EntryStatus::MalformedRoundTrip => {
                delta_memory += coefficients.memory_malformed_penalty;
                penalised_entries += 1;
            }
            EntryStatus::ToleranceExceeded { .. } => {
                delta_thermal += coefficients.thermal_tolerance_exceeded_penalty;
                penalised_entries += 1;
            }
            EntryStatus::DecodeFailed { .. } => {
                // A bad payload tells us nothing about pressure — it's
                // a control-plane error, not a workload signal. We log it
                // in `penalised_entries` for visibility but don't move
                // pressure.
                penalised_entries += 1;
            }
        }
    }

    // Apply aggregate deltas to the state.
    if delta_memory != 0.0 {
        updated.apply_delta(PressureKind::Memory, delta_memory);
    }
    if delta_thermal != 0.0 {
        updated.apply_delta(PressureKind::Thermal, delta_thermal);
    }
    if delta_throughput != 0.0 {
        updated.apply_delta(PressureKind::Throughput, delta_throughput);
    }

    let summary = updated.summary();
    CompressionPressureDelta {
        updated,
        summary,
        delta_memory,
        delta_thermal,
        delta_throughput,
        rewarded_entries,
        penalised_entries,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::{CandidateId, CompressionKind as Kind, CriticalityTag};
    use crate::report::{CompressionReport, EntryStatus, ReportEntry};

    fn baseline_with(mag: f64) -> PressureDensityState {
        let mut state = PressureDensityState::empty();
        state.apply_delta(PressureKind::Memory, mag);
        state.apply_delta(PressureKind::Thermal, mag * 0.5);
        state
    }

    fn validated_lossless_entry(slot: u32, original: usize, compressed: usize) -> ReportEntry {
        ReportEntry {
            slot,
            candidate_id: CandidateId(slot as u64 + 1),
            candidate_label: format!("e{slot}"),
            kind: Kind::LosslessTrace,
            criticality_tag: CriticalityTag::SemanticCritical,
            declared_tolerance: 0.0,
            original_len: original,
            compressed_len: compressed,
            input_hash: 0xA,
            summary_hash: 0xB,
            reconstructed_hash: 0xA,
            observed_error: 0.0,
            status: EntryStatus::Validated,
        }
    }

    fn validated_advisory_entry(
        slot: u32,
        original: usize,
        compressed: usize,
        observed: f64,
    ) -> ReportEntry {
        ReportEntry {
            slot,
            candidate_id: CandidateId(slot as u64 + 1),
            candidate_label: format!("a{slot}"),
            kind: Kind::LowRankAdvisory,
            criticality_tag: CriticalityTag::AdvisoryOnly,
            declared_tolerance: 0.1,
            original_len: original,
            compressed_len: compressed,
            input_hash: 0xA,
            summary_hash: 0xB,
            reconstructed_hash: 0xC,
            observed_error: observed,
            status: EntryStatus::Validated,
        }
    }

    #[test]
    fn empty_report_does_not_change_state() {
        let baseline = baseline_with(0.4);
        let pre_hash = baseline.stable_hash();
        let report = CompressionReport::new(0, vec![]);
        let result = compression_pressure_delta(baseline, &report, BridgeCoefficients::default());
        assert_eq!(result.updated.stable_hash(), pre_hash);
        assert_eq!(result.delta_memory, 0.0);
        assert_eq!(result.delta_thermal, 0.0);
        assert_eq!(result.delta_throughput, 0.0);
        assert_eq!(result.rewarded_entries, 0);
        assert_eq!(result.penalised_entries, 0);
    }

    #[test]
    fn validated_lossless_relaxes_memory_and_throughput() {
        let baseline = baseline_with(0.8);
        let pre_mem = baseline.magnitude(PressureKind::Memory);
        let report = CompressionReport::new(
            0,
            vec![validated_lossless_entry(0, 1000, 200)], // ratio 0.2 → saving 0.8
        );
        let result = compression_pressure_delta(baseline, &report, BridgeCoefficients::default());
        assert!(result.delta_memory < 0.0);
        assert!(result.delta_throughput < 0.0);
        assert_eq!(result.rewarded_entries, 1);
        assert!(result.updated.magnitude(PressureKind::Memory) < pre_mem);
    }

    #[test]
    fn validated_advisory_increases_thermal_by_observed_error() {
        let baseline = baseline_with(0.2);
        let pre_thermal = baseline.magnitude(PressureKind::Thermal);
        let entry = validated_advisory_entry(0, 1000, 500, 0.05);
        let report = CompressionReport::new(0, vec![entry]);
        let result = compression_pressure_delta(
            baseline,
            &report,
            BridgeCoefficients {
                // Zero out the memory reward so we isolate the thermal path.
                memory_reward_scale: 0.0,
                throughput_reward_per_validated: 0.0,
                ..BridgeCoefficients::default()
            },
        );
        assert!(result.delta_thermal > 0.0);
        assert!(result.updated.magnitude(PressureKind::Thermal) > pre_thermal);
    }

    #[test]
    fn malformed_round_trip_penalises_memory() {
        let baseline = baseline_with(0.1);
        let mut entry = validated_lossless_entry(0, 100, 80);
        entry.status = EntryStatus::MalformedRoundTrip;
        let report = CompressionReport::new(0, vec![entry]);
        let result =
            compression_pressure_delta(baseline.clone(), &report, BridgeCoefficients::default());
        assert!(result.delta_memory > 0.0);
        assert_eq!(result.penalised_entries, 1);
        // The penalised entry is NOT also rewarded.
        assert_eq!(result.rewarded_entries, 0);
    }

    #[test]
    fn tolerance_exceeded_penalises_thermal() {
        let baseline = baseline_with(0.2);
        let mut entry = validated_advisory_entry(0, 1000, 500, 0.5);
        entry.status = EntryStatus::ToleranceExceeded {
            declared: 0.1,
            observed: 0.5,
        };
        let report = CompressionReport::new(0, vec![entry]);
        let result = compression_pressure_delta(baseline, &report, BridgeCoefficients::default());
        assert!(result.delta_thermal > 0.0);
        assert_eq!(result.penalised_entries, 1);
    }

    #[test]
    fn zero_coefficients_makes_bridge_a_no_op() {
        let baseline = baseline_with(0.5);
        let pre_hash = baseline.stable_hash();
        let report = CompressionReport::new(
            0,
            vec![
                validated_lossless_entry(0, 1000, 200),
                validated_advisory_entry(1, 1000, 500, 0.05),
            ],
        );
        let result = compression_pressure_delta(baseline, &report, BridgeCoefficients::zero());
        assert_eq!(result.updated.stable_hash(), pre_hash);
        assert_eq!(result.delta_memory, 0.0);
        assert_eq!(result.delta_thermal, 0.0);
        assert_eq!(result.delta_throughput, 0.0);
    }

    #[test]
    fn bridge_is_deterministic() {
        let report = CompressionReport::new(
            42,
            vec![
                validated_lossless_entry(0, 1000, 200),
                validated_advisory_entry(1, 800, 400, 0.03),
            ],
        );
        let result1 =
            compression_pressure_delta(baseline_with(0.3), &report, BridgeCoefficients::default());
        let result2 =
            compression_pressure_delta(baseline_with(0.3), &report, BridgeCoefficients::default());
        assert_eq!(result1.updated.stable_hash(), result2.updated.stable_hash());
        assert_eq!(
            result1.delta_memory.to_bits(),
            result2.delta_memory.to_bits()
        );
        assert_eq!(
            result1.delta_thermal.to_bits(),
            result2.delta_thermal.to_bits()
        );
        assert_eq!(result1.summary, result2.summary);
    }

    #[test]
    fn bridge_summary_carries_post_delta_collapse_risk() {
        // Start with very high memory pressure; compression should
        // lower collapse_risk if it provides a reward.
        let mut baseline = PressureDensityState::empty();
        baseline.apply_delta(PressureKind::Memory, 0.95);
        let pre_risk = baseline.summary().collapse_risk;
        let report = CompressionReport::new(
            0,
            vec![validated_lossless_entry(0, 1000, 100)], // 90% saving
        );
        let result = compression_pressure_delta(baseline, &report, BridgeCoefficients::default());
        assert!(
            result.summary.collapse_risk < pre_risk,
            "expected pressure reduction; pre={pre_risk}, post={}",
            result.summary.collapse_risk
        );
    }

    #[test]
    fn bridge_counts_rewards_and_penalties_separately() {
        let mut entries = vec![validated_lossless_entry(0, 100, 50)];
        let mut bad = validated_lossless_entry(1, 100, 50);
        bad.status = EntryStatus::MalformedRoundTrip;
        entries.push(bad);
        let mut exceeded = validated_advisory_entry(2, 100, 50, 0.5);
        exceeded.status = EntryStatus::ToleranceExceeded {
            declared: 0.1,
            observed: 0.5,
        };
        entries.push(exceeded);

        let report = CompressionReport::new(0, entries);
        let result =
            compression_pressure_delta(baseline_with(0.5), &report, BridgeCoefficients::default());
        assert_eq!(result.rewarded_entries, 1);
        assert_eq!(result.penalised_entries, 2);
    }
}
