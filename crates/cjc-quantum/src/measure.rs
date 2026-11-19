//! Measurement — probabilistic collapse of quantum states.
//!
//! Measurement uses SplitMix64 PRNG with explicit seed threading for
//! deterministic outcomes: same seed = same measurement results.
//!
//! # Determinism
//!
//! - Probability accumulation uses Kahan summation
//! - PRNG is SplitMix64 with explicit `&mut u64` state threading
//! - Post-measurement renormalization uses Kahan summation
//! - Basis states processed in ascending index order

use crate::statevector::Statevector;
use cjc_repro::KahanAccumulatorF64;
use cjc_runtime::complex::ComplexF64;

/// Measure a single qubit, collapsing the statevector.
///
/// Returns the measurement outcome (0 or 1) and mutates the statevector
/// to the post-measurement state (renormalized).
///
/// # Determinism
///
/// The `rng_state` parameter is a mutable SplitMix64 state. Same seed and
/// same statevector produce identical outcomes across runs and platforms:
/// sampling uses only integer RNG arithmetic and Kahan-summed probabilities,
/// and the statevector itself is platform-independent because gate matrices
/// use `cjc_repro::dmath` (ADR-0046).
pub fn measure_qubit(
    sv: &mut Statevector,
    qubit: usize,
    rng_state: &mut u64,
) -> Result<u8, String> {
    sv.validate_qubit(qubit)?;

    let n = sv.n_states();
    let bit = 1usize << qubit;

    // Compute P(qubit=0) using Kahan summation
    let mut prob0_acc = KahanAccumulatorF64::new();
    for i in 0..n {
        if i & bit == 0 {
            prob0_acc.add(sv.amplitudes[i].norm_sq());
        }
    }
    let prob0 = prob0_acc.finalize();

    // Sample outcome
    let r = crate::rand_f64(rng_state);
    let outcome = if r < prob0 { 0u8 } else { 1u8 };

    // Collapse: zero out amplitudes inconsistent with outcome
    let keep_bit = if outcome == 0 { 0 } else { bit };
    for i in 0..n {
        if (i & bit) != keep_bit {
            sv.amplitudes[i] = ComplexF64::ZERO;
        }
    }

    // Renormalize the surviving amplitudes
    sv.normalize();

    Ok(outcome)
}

/// Measure all qubits, collapsing to a single basis state.
///
/// Returns a vector of measurement outcomes (0 or 1) for each qubit,
/// from qubit 0 to qubit n-1.
///
/// Each qubit is measured sequentially, with the statevector collapsing
/// after each measurement. The order is ascending (qubit 0 first).
pub fn measure_all(sv: &mut Statevector, rng_state: &mut u64) -> Result<Vec<u8>, String> {
    let n_qubits = sv.n_qubits();
    let mut outcomes = Vec::with_capacity(n_qubits);

    for q in 0..n_qubits {
        let outcome = measure_qubit(sv, q, rng_state)?;
        outcomes.push(outcome);
    }

    Ok(outcomes)
}

/// Sample from the probability distribution without collapsing.
///
/// Returns the index of the sampled basis state. Useful for repeated
/// sampling without modifying the state.
pub fn sample_basis_state(sv: &Statevector, rng_state: &mut u64) -> usize {
    let r = crate::rand_f64(rng_state);
    let n = sv.n_states();

    let mut cumulative = KahanAccumulatorF64::new();
    for i in 0..n {
        cumulative.add(sv.amplitudes[i].norm_sq());
        if cumulative.finalize() > r {
            return i;
        }
    }

    // Fallback to last state (handles floating-point edge cases)
    n - 1
}

/// Draw `n_shots` samples; returns exactly what `n_shots` calls to
/// [`sample_basis_state`] with the same `rng_state` would, in
/// O(2ⁿ + shots · n) instead of O(shots · 2ⁿ).
///
/// [`sample_basis_state`] returns the first index whose Kahan-finalized
/// running sum exceeds the draw `r`. That running sum is not guaranteed to be
/// monotone (Kahan's compensation can lower it by an ulp when a zero term is
/// added), so a plain binary search over it could disagree. The first index
/// where the *running maximum* of those sums exceeds `r` is always the same
/// index, and the running maximum is monotone, so it is binary-searched.
pub fn sample_basis_states(sv: &Statevector, n_shots: usize, rng_state: &mut u64) -> Vec<usize> {
    let n = sv.n_states();
    let mut running_max = Vec::with_capacity(n);
    let mut acc = KahanAccumulatorF64::new();
    let mut m = f64::NEG_INFINITY;
    for a in &sv.amplitudes {
        acc.add(a.norm_sq());
        let c = acc.finalize();
        if c > m {
            m = c;
        }
        running_max.push(m);
    }
    (0..n_shots)
        .map(|_| {
            let r = crate::rand_f64(rng_state);
            let i = running_max.partition_point(|&c| c <= r);
            if i < n {
                i
            } else {
                n - 1
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::Gate;

    const TOL: f64 = 1e-12;

    #[test]
    fn batch_sampler_matches_per_shot_sampler_bit_for_bit() {
        // Random states with many exact zeros (where Kahan's running sum can
        // dip), unnormalised states (the fallback), and several seeds.
        let mut s = 99u64;
        let mut next = || {
            s = s.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        };
        for trial in 0..60 {
            let n_q = 1 + trial % 9;
            let amps: Vec<ComplexF64> = (0..1usize << n_q)
                .map(|k| {
                    if next() % 3 == 0 {
                        ComplexF64::ZERO
                    } else {
                        let x = (next() >> 11) as f64 / (1u64 << 53) as f64;
                        let tiny = if k % 5 == 0 { 1e-300 } else { 1.0 };
                        ComplexF64::new(x * tiny, 0.0)
                    }
                })
                .collect();
            let mut sv = Statevector::from_amplitudes(amps).unwrap();
            if trial % 4 != 0 {
                sv.normalize();
            }
            for seed in [0u64, 1, 12345, u64::MAX] {
                let (mut a, mut b) = (seed, seed);
                let per_shot: Vec<usize> = (0..257).map(|_| sample_basis_state(&sv, &mut a)).collect();
                let batch = sample_basis_states(&sv, 257, &mut b);
                assert_eq!(per_shot, batch, "trial {} seed {}", trial, seed);
                assert_eq!(a, b, "rng state must advance identically");
            }
        }
    }

    #[test]
    fn test_measure_basis_state_0() {
        // |0⟩ always measures 0
        let mut sv = Statevector::new(1);
        let mut rng = 42u64;
        let outcome = measure_qubit(&mut sv, 0, &mut rng).unwrap();
        assert_eq!(outcome, 0);
        assert!(sv.is_normalized(TOL));
    }

    #[test]
    fn test_measure_basis_state_1() {
        // |1⟩ always measures 1
        let mut sv = Statevector::new(1);
        Gate::X(0).apply(&mut sv).unwrap();
        let mut rng = 42u64;
        let outcome = measure_qubit(&mut sv, 0, &mut rng).unwrap();
        assert_eq!(outcome, 1);
        assert!(sv.is_normalized(TOL));
    }

    #[test]
    fn test_measure_deterministic_same_seed() {
        // Same seed → same outcome
        for seed in 0..20u64 {
            let mut sv1 = Statevector::new(1);
            Gate::H(0).apply(&mut sv1).unwrap();
            let mut sv2 = sv1.clone();

            let mut rng1 = seed;
            let mut rng2 = seed;

            let o1 = measure_qubit(&mut sv1, 0, &mut rng1).unwrap();
            let o2 = measure_qubit(&mut sv2, 0, &mut rng2).unwrap();
            assert_eq!(o1, o2, "Determinism failure at seed {}", seed);
        }
    }

    #[test]
    fn test_measure_collapses_state() {
        // After measurement, state should be in the measured basis state
        let mut sv = Statevector::new(1);
        Gate::H(0).apply(&mut sv).unwrap();
        let mut rng = 42u64;
        let outcome = measure_qubit(&mut sv, 0, &mut rng).unwrap();

        if outcome == 0 {
            assert!((sv.amplitudes[0].norm_sq() - 1.0).abs() < TOL);
            assert!((sv.amplitudes[1].norm_sq()).abs() < TOL);
        } else {
            assert!((sv.amplitudes[0].norm_sq()).abs() < TOL);
            assert!((sv.amplitudes[1].norm_sq() - 1.0).abs() < TOL);
        }
    }

    #[test]
    fn test_measure_all_bell_state() {
        // Bell state: measurement of both qubits should be correlated
        let mut sv = Statevector::new(2);
        Gate::H(0).apply(&mut sv).unwrap();
        Gate::CNOT(0, 1).apply(&mut sv).unwrap();

        let mut rng = 42u64;
        let outcomes = measure_all(&mut sv, &mut rng).unwrap();
        assert_eq!(outcomes.len(), 2);
        // In Bell state (|00⟩+|11⟩)/√2, both qubits should agree
        assert_eq!(
            outcomes[0], outcomes[1],
            "Bell state qubits must be correlated"
        );
    }

    #[test]
    fn test_measure_statistics() {
        // H|0⟩ → measure many times, should get roughly 50/50
        let mut count0 = 0usize;
        let mut count1 = 0usize;
        for seed in 0..1000u64 {
            let mut sv = Statevector::new(1);
            Gate::H(0).apply(&mut sv).unwrap();
            let mut rng = seed;
            let outcome = measure_qubit(&mut sv, 0, &mut rng).unwrap();
            if outcome == 0 {
                count0 += 1;
            } else {
                count1 += 1;
            }
        }
        // Should be roughly 50/50 (within 10% tolerance)
        let ratio = count0 as f64 / 1000.0;
        assert!(
            ratio > 0.4 && ratio < 0.6,
            "H|0⟩ measurement ratio: {} (expected ~0.5)",
            ratio
        );
    }

    #[test]
    fn test_sample_basis_state_deterministic() {
        let sv = Statevector::new(2);
        let mut rng1 = 42u64;
        let mut rng2 = 42u64;

        for _ in 0..100 {
            let s1 = sample_basis_state(&sv, &mut rng1);
            let s2 = sample_basis_state(&sv, &mut rng2);
            assert_eq!(s1, s2);
        }
    }

    #[test]
    fn test_measure_invalid_qubit() {
        let mut sv = Statevector::new(2);
        let mut rng = 42u64;
        assert!(measure_qubit(&mut sv, 3, &mut rng).is_err());
    }

    #[test]
    fn test_measure_preserves_normalization() {
        let mut sv = Statevector::new(3);
        Gate::H(0).apply(&mut sv).unwrap();
        Gate::H(1).apply(&mut sv).unwrap();
        Gate::CNOT(0, 2).apply(&mut sv).unwrap();

        let mut rng = 42u64;
        measure_qubit(&mut sv, 0, &mut rng).unwrap();
        assert!(sv.is_normalized(TOL), "Norm after partial measurement");

        measure_qubit(&mut sv, 1, &mut rng).unwrap();
        assert!(sv.is_normalized(TOL), "Norm after second measurement");
    }
}
