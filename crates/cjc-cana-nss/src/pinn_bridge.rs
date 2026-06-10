//! PINN v1 → NSS pressure-kind bridge.
//!
//! Maps a [`PhysicalCostEstimate`] (the deterministic physical-cost
//! model's output, see `cjc_cana::physical_cost`) onto NSS
//! [`PressureKind`] deltas, per the Phase-A handoff §4.4 table:
//!
//! | `PhysicalCostEstimate` field | NSS `PressureKind` | weight |
//! |---|---|---|
//! | `thermal_pressure` | `Thermal` | 1.0 |
//! | `memory_pressure` | `Memory` | 1.0 |
//! | `locality_risk` | `Memory` | 0.25 (working-set saturation echo) |
//! | `bandwidth_pressure` | `Throughput` | 1.0 |
//! | `bandwidth_pressure` | `Io` | 0.5 (v1 can't split disk from memory traffic; conservative echo) |
//! | `energy_estimate` | `Cpu` | 0.5 (clamped composite) |
//! | `confidence` | — | not a pressure; not emitted |
//!
//! ## Determinism contract
//!
//! Pure function of the estimate. Output is a fixed-shape `Vec` in a
//! fixed kind order (`Cpu`, `Memory`, `Io`, `Thermal`, `Throughput`),
//! every delta clamped to `[0, 1]`. Same estimate → byte-identical
//! output across runs and platforms. No FMA: every product binds to a
//! named intermediate before any addition.

use cjc_cana::physical_cost::PhysicalCostEstimate;
use cjc_nss::PressureKind;

/// Weight applied to `locality_risk` when folding it into the
/// `Memory` delta — locality risk is a working-set saturation signal,
/// quarter-weighted so it nudges rather than dominates the direct
/// memory-pressure term.
pub const LOCALITY_TO_MEMORY_WEIGHT: f64 = 0.25;

/// Weight applied to `bandwidth_pressure` when echoing it into `Io`.
/// v1's byte estimates don't distinguish disk traffic from memory
/// traffic, so `Io` receives a conservative half-weight echo of the
/// bandwidth signal rather than a dedicated estimate.
pub const BANDWIDTH_TO_IO_WEIGHT: f64 = 0.5;

/// Weight applied to `energy_estimate` (clamped) when mapping the
/// composite energy proxy onto `Cpu` saturation.
pub const ENERGY_TO_CPU_WEIGHT: f64 = 0.5;

/// Map a physical-cost estimate onto NSS pressure deltas.
///
/// Returns five `(kind, delta)` pairs in fixed order (`Cpu`, `Memory`,
/// `Io`, `Thermal`, `Throughput`), each delta in `[0, 1]`. Returns an
/// empty `Vec` when the estimate fails
/// [`PhysicalCostEstimate::is_valid`] — an invalid estimate abstains
/// rather than injecting garbage pressure.
pub fn physical_estimate_to_pressure_deltas(
    est: &PhysicalCostEstimate,
) -> Vec<(PressureKind, f64)> {
    if !est.is_valid() {
        return Vec::new();
    }

    let energy_clamped = est.energy_estimate.clamp(0.0, 1.0);
    let cpu_delta = ENERGY_TO_CPU_WEIGHT * energy_clamped;

    let locality_echo = LOCALITY_TO_MEMORY_WEIGHT * est.locality_risk;
    let memory_raw = est.memory_pressure + locality_echo;
    let memory_delta = memory_raw.clamp(0.0, 1.0);

    let io_delta = BANDWIDTH_TO_IO_WEIGHT * est.bandwidth_pressure;

    vec![
        (PressureKind::Cpu, cpu_delta.clamp(0.0, 1.0)),
        (PressureKind::Memory, memory_delta),
        (PressureKind::Io, io_delta.clamp(0.0, 1.0)),
        (PressureKind::Thermal, est.thermal_pressure),
        (PressureKind::Throughput, est.bandwidth_pressure),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn estimate(
        thermal: f64,
        memory: f64,
        bandwidth: f64,
        energy: f64,
        locality: f64,
    ) -> PhysicalCostEstimate {
        PhysicalCostEstimate {
            thermal_pressure: thermal,
            memory_pressure: memory,
            bandwidth_pressure: bandwidth,
            energy_estimate: energy,
            locality_risk: locality,
            confidence: 1.0,
        }
    }

    #[test]
    fn all_zero_estimate_maps_to_all_zero_deltas() {
        let deltas = physical_estimate_to_pressure_deltas(&estimate(0.0, 0.0, 0.0, 0.0, 0.0));
        assert_eq!(deltas.len(), 5);
        for (_, d) in &deltas {
            assert_eq!(*d, 0.0);
        }
    }

    #[test]
    fn mapping_table_is_honored() {
        let deltas = physical_estimate_to_pressure_deltas(&estimate(0.8, 0.4, 0.6, 0.5, 0.4));
        let get = |kind: PressureKind| {
            deltas
                .iter()
                .find(|(k, _)| *k == kind)
                .map(|(_, d)| *d)
                .unwrap()
        };
        assert!((get(PressureKind::Thermal) - 0.8).abs() < 1e-15);
        // memory 0.4 + 0.25 * locality 0.4 = 0.5
        assert!((get(PressureKind::Memory) - 0.5).abs() < 1e-15);
        assert!((get(PressureKind::Throughput) - 0.6).abs() < 1e-15);
        // io = 0.5 * bandwidth 0.6 = 0.3
        assert!((get(PressureKind::Io) - 0.3).abs() < 1e-15);
        // cpu = 0.5 * energy 0.5 = 0.25
        assert!((get(PressureKind::Cpu) - 0.25).abs() < 1e-15);
    }

    #[test]
    fn unbounded_energy_is_clamped_before_cpu_mapping() {
        let deltas = physical_estimate_to_pressure_deltas(&estimate(0.0, 0.0, 0.0, 1e9, 0.0));
        let cpu = deltas
            .iter()
            .find(|(k, _)| *k == PressureKind::Cpu)
            .map(|(_, d)| *d)
            .unwrap();
        assert!((cpu - ENERGY_TO_CPU_WEIGHT).abs() < 1e-15);
    }

    #[test]
    fn memory_plus_locality_echo_clamps_at_one() {
        let deltas = physical_estimate_to_pressure_deltas(&estimate(0.0, 0.95, 0.0, 0.0, 1.0));
        let memory = deltas
            .iter()
            .find(|(k, _)| *k == PressureKind::Memory)
            .map(|(_, d)| *d)
            .unwrap();
        assert_eq!(memory, 1.0);
    }

    #[test]
    fn invalid_estimate_abstains() {
        let bad = PhysicalCostEstimate {
            thermal_pressure: f64::NAN,
            memory_pressure: 0.0,
            bandwidth_pressure: 0.0,
            energy_estimate: 0.0,
            locality_risk: 0.0,
            confidence: 1.0,
        };
        assert!(physical_estimate_to_pressure_deltas(&bad).is_empty());
        let oob = estimate(1.5, 0.0, 0.0, 0.0, 0.0);
        assert!(physical_estimate_to_pressure_deltas(&oob).is_empty());
    }

    #[test]
    fn output_order_and_values_are_deterministic() {
        let est = estimate(0.3, 0.2, 0.7, 2.0, 0.1);
        let first = physical_estimate_to_pressure_deltas(&est);
        for _ in 0..50 {
            let again = physical_estimate_to_pressure_deltas(&est);
            assert_eq!(first.len(), again.len());
            for (a, b) in first.iter().zip(again.iter()) {
                assert_eq!(a.0, b.0);
                assert_eq!(a.1.to_bits(), b.1.to_bits());
            }
        }
        let kinds: Vec<PressureKind> = first.iter().map(|(k, _)| *k).collect();
        assert_eq!(
            kinds,
            vec![
                PressureKind::Cpu,
                PressureKind::Memory,
                PressureKind::Io,
                PressureKind::Thermal,
                PressureKind::Throughput,
            ]
        );
    }

    #[test]
    fn every_delta_is_unit_bounded_for_valid_estimates() {
        // Grid over the valid input space.
        let vals = [0.0, 0.25, 0.5, 0.75, 1.0];
        let energies = [0.0, 0.5, 1.0, 100.0];
        for &t in &vals {
            for &m in &vals {
                for &b in &vals {
                    for &l in &vals {
                        for &e in &energies {
                            let deltas =
                                physical_estimate_to_pressure_deltas(&estimate(t, m, b, e, l));
                            assert_eq!(deltas.len(), 5);
                            for (kind, d) in &deltas {
                                assert!(
                                    d.is_finite() && (0.0..=1.0).contains(d),
                                    "{kind:?} delta {d} out of range"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
