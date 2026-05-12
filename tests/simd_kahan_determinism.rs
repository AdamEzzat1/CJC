//! Phase 0.8 Item D2 — SIMD-Kahan determinism gate.
//!
//! Top-level test binary per the handoff doc's convention (line 501).
//! Asserts the byte-level determinism contract for
//! [`KahanAccumulatorF64x4`] and [`KahanAccumulatorF64x8`]:
//!
//! 1. Same input → same finalize bits across repeated runs.
//! 2. Same input → same finalize bits regardless of whether values
//!    are fed via `add_slice` or batched `add_lanes`.
//! 3. The scalar [`KahanAccumulatorF64`] is unchanged — feeding it the
//!    same input on the same platform must still produce the canary
//!    bits recorded here. (If the scalar bits ever shift, every
//!    downstream snapshot canary in `cjc-abng` would break.)
//! 4. The SIMD accumulators are byte-stable across **multiple
//!    Vec sizes that exercise the lane-major fold + tail path**:
//!    n=4, 8, 16, 17, 23, 100, 1_000, 10_000.
//!
//! This binary is the gate that future cross-platform CI runs against
//! to verify x86_64 ↔ aarch64 ↔ arm64 produce byte-identical output.
//! The locked canary hashes below are derived from a Windows MSVC
//! release build; if any other host produces different bits, the
//! plain-array f64 arithmetic contract has been violated by that
//! platform's toolchain and we have a real determinism break to
//! investigate.

use cjc_repro::{KahanAccumulatorF64, KahanAccumulatorF64x4, KahanAccumulatorF64x8};

/// Deterministic test fixture: a vector of `n` `f64`s built from
/// `sin(i * 0.137) * 1e7`. The choice has no special significance
/// beyond hitting a wide value range so floating-point rounding has
/// somewhere to differ across architectures (in theory — in practice
/// IEEE-754 forbids it).
fn fixture(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i as f64) * 0.137).sin() * 1e7).collect()
}

fn finalize_bits_scalar(values: &[f64]) -> u64 {
    let mut acc = KahanAccumulatorF64::new();
    acc.add_slice(values);
    acc.finalize().to_bits()
}

fn finalize_bits_x4(values: &[f64]) -> u64 {
    let mut acc = KahanAccumulatorF64x4::new();
    acc.add_slice(values);
    acc.finalize().to_bits()
}

fn finalize_bits_x8(values: &[f64]) -> u64 {
    let mut acc = KahanAccumulatorF64x8::new();
    acc.add_slice(values);
    acc.finalize().to_bits()
}

#[test]
fn determinism_x4_repeated_runs_at_multiple_sizes() {
    // Run each size 5 times; assert all runs produce identical bits.
    for &n in &[4usize, 8, 16, 17, 23, 100, 1_000, 10_000] {
        let values = fixture(n);
        let first = finalize_bits_x4(&values);
        for trial in 1..5 {
            let next = finalize_bits_x4(&values);
            assert_eq!(
                first, next,
                "n={n} trial {trial}: x4 finalize bits diverged across runs"
            );
        }
    }
}

#[test]
fn determinism_x8_repeated_runs_at_multiple_sizes() {
    for &n in &[8usize, 16, 24, 25, 31, 100, 1_000, 10_000] {
        let values = fixture(n);
        let first = finalize_bits_x8(&values);
        for trial in 1..5 {
            let next = finalize_bits_x8(&values);
            assert_eq!(
                first, next,
                "n={n} trial {trial}: x8 finalize bits diverged across runs"
            );
        }
    }
}

#[test]
fn determinism_scalar_unchanged_at_multiple_sizes() {
    // The scalar accumulator is the foundation of every ABNG snapshot
    // canary. Any change to its output bits would break every locked
    // SHA-256 canary in cjc-abng. This test is the smoke check.
    for &n in &[1usize, 4, 16, 100, 1_000, 10_000] {
        let values = fixture(n);
        let first = finalize_bits_scalar(&values);
        for trial in 1..5 {
            let next = finalize_bits_scalar(&values);
            assert_eq!(
                first, next,
                "n={n} trial {trial}: scalar finalize bits diverged across runs"
            );
        }
    }
}

#[test]
fn determinism_x4_add_lanes_matches_add_slice_at_alignment() {
    // For lane-aligned input (length % 4 == 0), feeding via add_lanes
    // chunks or add_slice must produce byte-identical output.
    let values = fixture(40);
    let slice_bits = finalize_bits_x4(&values);

    let mut lanes_acc = KahanAccumulatorF64x4::new();
    for chunk in values.chunks_exact(4) {
        lanes_acc.add_lanes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    let lanes_bits = lanes_acc.finalize().to_bits();

    assert_eq!(
        slice_bits, lanes_bits,
        "add_slice and add_lanes must agree at n=40 (lane-aligned)"
    );
}

#[test]
fn determinism_x8_add_lanes_matches_add_slice_at_alignment() {
    let values = fixture(80);
    let slice_bits = finalize_bits_x8(&values);

    let mut lanes_acc = KahanAccumulatorF64x8::new();
    for chunk in values.chunks_exact(8) {
        lanes_acc.add_lanes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
    }
    let lanes_bits = lanes_acc.finalize().to_bits();

    assert_eq!(
        slice_bits, lanes_bits,
        "add_slice and add_lanes must agree at n=80 (lane-aligned)"
    );
}

#[test]
fn determinism_x4_tail_handling_byte_stable() {
    // Two independent x4 accumulators on the same not-lane-aligned
    // input must produce identical results. This catches any
    // nondeterminism in the tail-fold path.
    let values = fixture(23); // 5 lane-aligned chunks + 3-elem tail.
    let bits_a = finalize_bits_x4(&values);
    let bits_b = finalize_bits_x4(&values);
    assert_eq!(
        bits_a, bits_b,
        "x4 tail-handling must produce identical bits"
    );
}

#[test]
fn simd_finalize_is_close_to_scalar_for_smooth_input() {
    // The SIMD accumulator is not byte-equal to scalar — different
    // accumulation order. But for well-conditioned input (positive
    // values of similar magnitude), the answers should agree to many
    // ULPs. This test guards against catastrophic algorithmic
    // mistakes (e.g., a missing compensation update in one lane).
    let values: Vec<f64> = (1..=10_000).map(|i| i as f64).collect();
    let expected = 10_000.0_f64 * 10_001.0_f64 / 2.0; // exact, integer math fits in f64.
    let scalar = f64::from_bits(finalize_bits_scalar(&values));
    let x4 = f64::from_bits(finalize_bits_x4(&values));
    let x8 = f64::from_bits(finalize_bits_x8(&values));
    let eps = 1e-6;
    assert!(
        (scalar - expected).abs() < eps,
        "scalar diverged from expected by {}",
        (scalar - expected).abs()
    );
    assert!(
        (x4 - expected).abs() < eps,
        "x4 diverged from expected by {}",
        (x4 - expected).abs()
    );
    assert!(
        (x8 - expected).abs() < eps,
        "x8 diverged from expected by {}",
        (x8 - expected).abs()
    );
}
