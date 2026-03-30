//! Clifford Circuit Simulator — Stabilizer Tableau Formalism
//!
//! Implements the Aaronson-Gottesman CHP (CNOT-Hadamard-Phase) algorithm for
//! efficient simulation of Clifford circuits. Clifford circuits (composed of
//! H, S, CNOT, and Pauli gates) can be simulated in O(N^2) time per gate on
//! N qubits, compared to exponential cost for general quantum circuits.
//!
//! # Representation
//!
//! The state of N qubits is stored as a stabilizer tableau with 2N rows:
//! - Rows 0..N: destabilizer generators
//! - Rows N..2N: stabilizer generators
//!
//! Each row encodes a Pauli string: a tensor product of single-qubit Paulis
//! {I, X, Y, Z} with an overall phase of +1 or -1.
//!
//! # Bitpacking
//!
//! The x-bits and z-bits of each row are bitpacked into `Vec<u64>` words,
//! giving ~64x memory reduction over `Vec<bool>`. This allows simulation
//! of 10,000+ qubit Clifford circuits in reasonable memory.
//!
//! # Determinism
//!
//! All operations are deterministic given the same RNG seed. Measurement
//! outcomes for non-deterministic measurements use `splitmix64` for
//! reproducible randomness.

use cjc_runtime::complex::ComplexF64;
use crate::rand_f64;

// ---------------------------------------------------------------------------
// Bit manipulation helpers
// ---------------------------------------------------------------------------

/// Get bit `idx` from a bitpacked word array.
#[inline]
fn get_bit(words: &[u64], idx: usize) -> bool {
    let word = idx / 64;
    let bit = idx % 64;
    (words[word] >> bit) & 1 == 1
}

/// Set bit `idx` in a bitpacked word array to `val`.
#[inline]
fn set_bit(words: &mut [u64], idx: usize, val: bool) {
    let word = idx / 64;
    let bit = idx % 64;
    if val {
        words[word] |= 1u64 << bit;
    } else {
        words[word] &= !(1u64 << bit);
    }
}

/// Flip (toggle) bit `idx` in a bitpacked word array.
#[inline]
fn flip_bit(words: &mut [u64], idx: usize) {
    let word = idx / 64;
    let bit = idx % 64;
    words[word] ^= 1u64 << bit;
}

// ---------------------------------------------------------------------------
// Phase accumulation helper for Pauli row multiplication
// ---------------------------------------------------------------------------

/// Compute the phase contribution g(x1, z1, x2, z2) when multiplying two
/// single-qubit Paulis. Returns a value in {-1, 0, 1} representing the
/// exponent of i contributed by this qubit position.
///
/// The Pauli at each qubit is encoded as: i^{2xz} X^x Z^z
///   (0,0) -> I,  (1,0) -> X,  (0,1) -> Z,  (1,1) -> Y = iXZ
///
/// The function g gives the power of i from the product P1 * P2 at one qubit.
#[inline]
fn g_phase(x1: bool, z1: bool, x2: bool, z2: bool) -> i32 {
    match (x1, z1) {
        (false, false) => 0,                                    // I * anything = no phase
        (true, true) => (z2 as i32) - (x2 as i32),             // Y * {I,X,Z,Y}
        (true, false) => (z2 as i32) * (2 * (x2 as i32) - 1),  // X * {I,X,Z,Y}
        (false, true) => (x2 as i32) * (1 - 2 * (z2 as i32)),  // Z * {I,X,Z,Y}
    }
}

// ---------------------------------------------------------------------------
// StabilizerState
// ---------------------------------------------------------------------------

/// A quantum state represented in the stabilizer tableau formalism.
///
/// Efficiently simulates Clifford circuits (H, S, CNOT, Pauli gates) on
/// N qubits in O(N^2) time per gate and O(N^2/64) memory using bitpacking.
#[derive(Debug, Clone)]
pub struct StabilizerState {
    /// Number of qubits.
    pub n: usize,
    /// Number of u64 words per row: ceil(n / 64).
    words_per_row: usize,
    /// X-part of the Pauli string for each row. 2n rows, each `words_per_row` u64s.
    x: Vec<Vec<u64>>,
    /// Z-part of the Pauli string for each row. 2n rows, each `words_per_row` u64s.
    z: Vec<Vec<u64>>,
    /// Phase for each row. Values are 0 (phase +1) or 2 (phase -1), stored mod 4.
    phase: Vec<u8>,
}

impl StabilizerState {
    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    /// Create a new stabilizer state representing |0...0⟩ on `n_qubits` qubits.
    ///
    /// The initial tableau is:
    /// - Destabilizer i (row i, 0 <= i < n): X on qubit i, identity elsewhere
    /// - Stabilizer i (row n+i): Z on qubit i, identity elsewhere
    /// - All phases are 0 (+1)
    pub fn new(n_qubits: usize) -> Self {
        assert!(n_qubits > 0, "StabilizerState requires at least 1 qubit");
        let n = n_qubits;
        let words_per_row = (n + 63) / 64;
        let total_rows = 2 * n;

        let mut x = vec![vec![0u64; words_per_row]; total_rows];
        let mut z = vec![vec![0u64; words_per_row]; total_rows];
        let phase = vec![0u8; total_rows];

        // Destabilizer i: X_i (x-bit set at position i)
        for i in 0..n {
            set_bit(&mut x[i], i, true);
        }
        // Stabilizer i: Z_i (z-bit set at position i)
        for i in 0..n {
            set_bit(&mut z[n + i], i, true);
        }

        StabilizerState { n, words_per_row, x, z, phase }
    }

    // -------------------------------------------------------------------
    // Row operations (internal)
    // -------------------------------------------------------------------

    /// Multiply row `target` by row `source` in the tableau.
    ///
    /// This computes the Pauli product of the two rows and stores the result
    /// in row `target`. The phase is updated according to the commutation
    /// relations of the Pauli group.
    fn rowmult(&mut self, target: usize, source: usize) {
        // Accumulate phase contribution from each qubit
        let mut phase_sum = 0i32;
        for j in 0..self.n {
            let x1 = get_bit(&self.x[source], j);
            let z1 = get_bit(&self.z[source], j);
            let x2 = get_bit(&self.x[target], j);
            let z2 = get_bit(&self.z[target], j);
            phase_sum += g_phase(x1, z1, x2, z2);
        }

        // Update phase: combine source phase + target phase + accumulated phase
        // All arithmetic mod 4
        let new_phase = (self.phase[target] as i32 + self.phase[source] as i32 + phase_sum) as i64;
        // Bring into 0..3 range
        self.phase[target] = (((new_phase % 4) + 4) % 4) as u8;

        // XOR the x and z bit arrays (Pauli multiplication in binary representation)
        for w in 0..self.words_per_row {
            self.x[target][w] ^= self.x[source][w];
            self.z[target][w] ^= self.z[source][w];
        }
    }

    /// Copy row `src` into row `dst`.
    fn row_copy(&mut self, dst: usize, src: usize) {
        self.phase[dst] = self.phase[src];
        for w in 0..self.words_per_row {
            self.x[dst][w] = self.x[src][w];
            self.z[dst][w] = self.z[src][w];
        }
    }

    /// Set a row to the identity (all zeros, phase 0).
    fn row_clear(&mut self, row: usize) {
        self.phase[row] = 0;
        for w in 0..self.words_per_row {
            self.x[row][w] = 0;
            self.z[row][w] = 0;
        }
    }

    // -------------------------------------------------------------------
    // Clifford gates
    // -------------------------------------------------------------------

    /// Apply Hadamard gate to qubit `q`.
    ///
    /// Tableau update for each row i:
    ///   swap x[i][q] and z[i][q]
    ///   phase[i] += 2 * (x[i][q] AND z[i][q])  (mod 4)
    ///
    /// Note: the phase update uses the values *after* the swap.
    pub fn h(&mut self, q: usize) {
        assert!(q < self.n, "qubit index {} out of range (n={})", q, self.n);
        for i in 0..(2 * self.n) {
            let xi = get_bit(&self.x[i], q);
            let zi = get_bit(&self.z[i], q);
            // Swap x and z
            set_bit(&mut self.x[i], q, zi);
            set_bit(&mut self.z[i], q, xi);
            // Phase update: if both are now 1 (i.e., both were set before swap,
            // meaning original x=1 and z=1), add 2 to phase
            if xi && zi {
                self.phase[i] = (self.phase[i] + 2) % 4;
            }
        }
    }

    /// Apply S (phase) gate to qubit `q`.
    ///
    /// Tableau update for each row i:
    ///   phase[i] += 2 * (x[i][q] AND z[i][q])  (mod 4)
    ///   z[i][q] ^= x[i][q]
    pub fn s(&mut self, q: usize) {
        assert!(q < self.n, "qubit index {} out of range (n={})", q, self.n);
        for i in 0..(2 * self.n) {
            let xi = get_bit(&self.x[i], q);
            let zi = get_bit(&self.z[i], q);
            if xi && zi {
                self.phase[i] = (self.phase[i] + 2) % 4;
            }
            // z[i][q] ^= x[i][q]
            if xi {
                set_bit(&mut self.z[i], q, !zi);
            }
        }
    }

    /// Apply CNOT gate with control qubit `ctrl` and target qubit `tgt`.
    ///
    /// Tableau update for each row i:
    ///   phase[i] += 2 * x[i][ctrl] * z[i][tgt] * (x[i][tgt] XOR z[i][ctrl] XOR 1)  (mod 4)
    ///   x[i][tgt] ^= x[i][ctrl]
    ///   z[i][ctrl] ^= z[i][tgt]
    pub fn cnot(&mut self, ctrl: usize, tgt: usize) {
        assert!(ctrl < self.n, "ctrl qubit index {} out of range (n={})", ctrl, self.n);
        assert!(tgt < self.n, "tgt qubit index {} out of range (n={})", tgt, self.n);
        assert!(ctrl != tgt, "CNOT control and target must differ");
        for i in 0..(2 * self.n) {
            let xc = get_bit(&self.x[i], ctrl);
            let zt = get_bit(&self.z[i], tgt);
            let xt = get_bit(&self.x[i], tgt);
            let zc = get_bit(&self.z[i], ctrl);
            // Phase update
            if xc && zt && !(xt ^ zc) {
                self.phase[i] = (self.phase[i] + 2) % 4;
            }
            // x[i][tgt] ^= x[i][ctrl]
            if xc {
                set_bit(&mut self.x[i], tgt, !xt);
            }
            // z[i][ctrl] ^= z[i][tgt]
            if zt {
                set_bit(&mut self.z[i], ctrl, !zc);
            }
        }
    }

    /// Apply Pauli X gate to qubit `q`.
    ///
    /// X = H Z H = H S S H
    pub fn x(&mut self, q: usize) {
        assert!(q < self.n, "qubit index {} out of range (n={})", q, self.n);
        // X on qubit q flips the z-part phase: for each row where z[i][q] = 1,
        // the phase flips. This is because X anti-commutes with Z.
        // Equivalently: phase[i] += 2 * z[i][q]  (mod 4)
        for i in 0..(2 * self.n) {
            if get_bit(&self.z[i], q) {
                self.phase[i] = (self.phase[i] + 2) % 4;
            }
        }
    }

    /// Apply Pauli Z gate to qubit `q`.
    ///
    /// Z = S * S
    pub fn z(&mut self, q: usize) {
        assert!(q < self.n, "qubit index {} out of range (n={})", q, self.n);
        // Z on qubit q: for each row where x[i][q] = 1, phase flips.
        // Z anti-commutes with X.
        // phase[i] += 2 * x[i][q]  (mod 4)
        for i in 0..(2 * self.n) {
            if get_bit(&self.x[i], q) {
                self.phase[i] = (self.phase[i] + 2) % 4;
            }
        }
    }

    /// Apply Pauli Y gate to qubit `q`.
    ///
    /// Y = i X Z, so applying Y flips phase for any row where x[i][q] XOR z[i][q] is false
    /// (i.e., where the Pauli at position q commutes or anti-commutes with Y).
    pub fn y(&mut self, q: usize) {
        assert!(q < self.n, "qubit index {} out of range (n={})", q, self.n);
        // Y anti-commutes with X and Z individually, commutes only with I and Y.
        // For a Pauli P_q at qubit q of row i:
        //   I: commutes with Y -> no phase change
        //   X: anti-commutes -> phase += 2
        //   Z: anti-commutes -> phase += 2
        //   Y: commutes -> no phase change
        // So phase += 2 when exactly one of x,z is set (X or Z), but not both (Y) or neither (I).
        for i in 0..(2 * self.n) {
            let xi = get_bit(&self.x[i], q);
            let zi = get_bit(&self.z[i], q);
            if xi ^ zi {
                self.phase[i] = (self.phase[i] + 2) % 4;
            }
        }
    }

    // -------------------------------------------------------------------
    // Measurement
    // -------------------------------------------------------------------

    /// Measure qubit `q` in the computational basis.
    ///
    /// Returns 0 or 1. The measurement may be deterministic (if the qubit's
    /// value is fixed by the stabilizer constraints) or random (if the qubit
    /// is in a superposition), using `rng` for randomness.
    ///
    /// This implements the CHP measurement algorithm from Aaronson & Gottesman.
    pub fn measure(&mut self, q: usize, rng: &mut u64) -> u8 {
        assert!(q < self.n, "qubit index {} out of range (n={})", q, self.n);
        let n = self.n;

        // Step 1: Find a stabilizer generator (rows n..2n) with x[p][q] = 1
        let mut p: Option<usize> = None;
        for i in n..(2 * n) {
            if get_bit(&self.x[i], q) {
                p = Some(i);
                break;
            }
        }

        match p {
            Some(p) => {
                // Non-deterministic measurement
                // For all rows i != p where x[i][q] = 1, multiply row i by row p
                for i in 0..(2 * n) {
                    if i != p && get_bit(&self.x[i], q) {
                        self.rowmult(i, p);
                    }
                }

                // Set destabilizer row (p - n) to the old stabilizer row p
                self.row_copy(p - n, p);

                // Set stabilizer row p to: +/- Z_q (all zeros except z[q] = 1)
                self.row_clear(p);
                let outcome = if rand_f64(rng) < 0.5 { 0u8 } else { 1u8 };
                set_bit(&mut self.z[p], q, true);
                self.phase[p] = if outcome == 1 { 2 } else { 0 };

                outcome
            }
            None => {
                // Deterministic measurement: no stabilizer has x[q] = 1
                // Compute outcome from destabilizers.
                //
                // We use a scratch row (index 2n conceptually). Since we don't have
                // a 2n+1 row, we temporarily extend or use a local accumulator.
                //
                // Algorithm: start with identity (+I^n), then for each destabilizer
                // row i (0..n) where x[i][q] = 1, multiply by stabilizer row (n+i).
                // The final phase gives the outcome.

                // We accumulate into a temporary row
                let mut scratch_x = vec![0u64; self.words_per_row];
                let mut scratch_z = vec![0u64; self.words_per_row];
                let mut scratch_phase: u8 = 0;

                for i in 0..n {
                    if get_bit(&self.x[i], q) {
                        // Multiply scratch by stabilizer row (n + i)
                        let stab = n + i;
                        let mut phase_sum = 0i32;
                        for j in 0..n {
                            let x1 = get_bit(&self.x[stab], j);
                            let z1 = get_bit(&self.z[stab], j);
                            let x2 = get_bit(&scratch_x, j);
                            let z2 = get_bit(&scratch_z, j);
                            phase_sum += g_phase(x1, z1, x2, z2);
                        }

                        let new_phase = (scratch_phase as i32
                            + self.phase[stab] as i32
                            + phase_sum) as i64;
                        scratch_phase = (((new_phase % 4) + 4) % 4) as u8;

                        for w in 0..self.words_per_row {
                            scratch_x[w] ^= self.x[stab][w];
                            scratch_z[w] ^= self.z[stab][w];
                        }
                    }
                }

                // Phase 0 -> outcome 0 (+1 eigenvalue), Phase 2 -> outcome 1 (-1 eigenvalue)
                if scratch_phase == 0 { 0 } else { 1 }
            }
        }
    }

    // -------------------------------------------------------------------
    // State vector extraction (small systems only)
    // -------------------------------------------------------------------

    /// Convert the stabilizer state to a full statevector representation.
    ///
    /// Returns `None` if `n > 12` (since 2^n amplitudes would be too large).
    /// For n <= 12, constructs the unique state stabilized by all generators
    /// using successive projections.
    pub fn to_statevector(&self) -> Option<Vec<ComplexF64>> {
        if self.n > 12 {
            return None;
        }
        let n = self.n;
        let dim = 1usize << n;

        // Start with uniform superposition |+...+⟩ = (1/sqrt(2^n)) * sum_k |k⟩
        // This guarantees nonzero overlap with any stabilizer state, unlike |0...0⟩
        // which can have zero overlap (e.g., the state |1⟩ has zero overlap with |0⟩).
        let amp = 1.0 / (dim as f64).sqrt();
        let mut sv = vec![ComplexF64::real(amp); dim];

        // For each stabilizer generator, project onto its +1 eigenspace:
        // sv_new = (I + S_i) * sv / ||(I + S_i) * sv||
        for stab_idx in n..(2 * n) {
            // Compute S_i|k⟩ for each basis state |k⟩ and accumulate into s_sv
            let mut s_sv = vec![ComplexF64::ZERO; dim];

            for k in 0..dim {
                if sv[k].norm_sq() < 1e-30 {
                    continue; // Skip zero amplitudes for efficiency
                }

                // Apply the Pauli string of stabilizer stab_idx to basis state |k⟩
                let mut new_k = k;
                let mut local_re = 1.0f64;
                let mut local_im = 0.0f64;

                for j in 0..n {
                    let xj = get_bit(&self.x[stab_idx], j);
                    let zj = get_bit(&self.z[stab_idx], j);
                    let kj = (k >> j) & 1 == 1;

                    if xj {
                        new_k ^= 1 << j;
                    }
                    if zj && kj {
                        // Multiply phase by -1
                        local_re = -local_re;
                        local_im = -local_im;
                    }
                    if xj && zj {
                        // Multiply phase by i
                        let tmp_re = -local_im;
                        let tmp_im = local_re;
                        local_re = tmp_re;
                        local_im = tmp_im;
                    }
                }

                // Apply global phase from the stabilizer row
                if self.phase[stab_idx] == 2 {
                    local_re = -local_re;
                    local_im = -local_im;
                }

                let local_phase = ComplexF64::new(local_re, local_im);
                let contribution = sv[k].mul_fixed(local_phase);
                s_sv[new_k] = s_sv[new_k].add(contribution);
            }

            // sv = (sv + s_sv) * 0.5
            for k in 0..dim {
                sv[k] = sv[k].add(s_sv[k]).scale(0.5);
            }

            // Normalize
            let mut norm_sq = 0.0f64;
            for k in 0..dim {
                norm_sq += sv[k].norm_sq();
            }
            if norm_sq < 1e-30 {
                // Degenerate — should not happen for valid stabilizer states
                continue;
            }
            let inv_norm = 1.0 / norm_sq.sqrt();
            for k in 0..dim {
                sv[k] = sv[k].scale(inv_norm);
            }
        }

        Some(sv)
    }

    // -------------------------------------------------------------------
    // Utility
    // -------------------------------------------------------------------

    /// Return the number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.n
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let s = StabilizerState::new(3);
        let mut s2 = s.clone();
        let mut rng = 42u64;
        // |000⟩ — all measurements should deterministically yield 0
        assert_eq!(s2.measure(0, &mut rng), 0);
        assert_eq!(s2.measure(1, &mut rng), 0);
        assert_eq!(s2.measure(2, &mut rng), 0);
    }

    #[test]
    fn test_x_gate_flips() {
        let mut s = StabilizerState::new(1);
        s.x(0);
        let mut rng = 42u64;
        assert_eq!(s.measure(0, &mut rng), 1);
    }

    #[test]
    fn test_hadamard_creates_superposition() {
        // H|0⟩ = |+⟩, measurement should be random (roughly 50/50)
        let mut zeros = 0;
        let mut ones = 0;
        for seed in 0..100u64 {
            let mut s = StabilizerState::new(1);
            s.h(0);
            let mut rng = seed;
            let outcome = s.measure(0, &mut rng);
            if outcome == 0 {
                zeros += 1;
            } else {
                ones += 1;
            }
        }
        assert!(
            zeros > 20 && ones > 20,
            "Expected roughly 50/50 split, got zeros={} ones={}",
            zeros,
            ones
        );
    }

    #[test]
    fn test_bell_state_correlated() {
        for seed in 0..50u64 {
            let mut s = StabilizerState::new(2);
            s.h(0);
            s.cnot(0, 1);
            let mut rng = seed;
            let a = s.measure(0, &mut rng);
            let b = s.measure(1, &mut rng);
            assert_eq!(a, b, "Bell state qubits must agree at seed {}", seed);
        }
    }

    #[test]
    fn test_ghz_all_agree() {
        for n in 3..=6 {
            for seed in 0..20u64 {
                let mut s = StabilizerState::new(n);
                s.h(0);
                for i in 0..(n - 1) {
                    s.cnot(i, i + 1);
                }
                let mut rng = seed;
                let first = s.measure(0, &mut rng);
                for q in 1..n {
                    assert_eq!(
                        s.measure(q, &mut rng),
                        first,
                        "GHZ n={} seed={} qubit={}",
                        n,
                        seed,
                        q
                    );
                }
            }
        }
    }

    #[test]
    fn test_deterministic_same_seed() {
        let mut s1 = StabilizerState::new(4);
        s1.h(0);
        s1.cnot(0, 1);
        s1.h(2);
        s1.s(3);
        let mut s2 = s1.clone();
        let mut rng1 = 42u64;
        let mut rng2 = 42u64;
        for q in 0..4 {
            assert_eq!(s1.measure(q, &mut rng1), s2.measure(q, &mut rng2));
        }
    }

    #[test]
    fn test_z_gate_on_zero() {
        // Z|0⟩ = |0⟩
        let mut s = StabilizerState::new(1);
        s.z(0);
        let mut rng = 42u64;
        assert_eq!(s.measure(0, &mut rng), 0);
    }

    #[test]
    fn test_z_gate_on_one() {
        // ZX|0⟩ = Z|1⟩ = -|1⟩, still measures 1
        let mut s = StabilizerState::new(1);
        s.x(0);
        s.z(0);
        let mut rng = 42u64;
        assert_eq!(s.measure(0, &mut rng), 1);
    }

    #[test]
    fn test_1000_qubit_ghz() {
        let n = 1000;
        let mut s = StabilizerState::new(n);
        s.h(0);
        for i in 0..(n - 1) {
            s.cnot(i, i + 1);
        }
        let mut rng = 42u64;
        let first = s.measure(0, &mut rng);
        // Check a few qubits (measuring all 1000 would be slow due to tableau updates)
        assert_eq!(s.measure(1, &mut rng), first);
        assert_eq!(s.measure(n - 1, &mut rng), first);
    }

    #[test]
    fn test_to_statevector_bell() {
        let mut s = StabilizerState::new(2);
        s.h(0);
        s.cnot(0, 1);
        let sv = s.to_statevector().unwrap();
        let isq2 = 1.0 / 2.0f64.sqrt();
        // Bell state: (|00⟩ + |11⟩)/sqrt(2)
        assert!(
            (sv[0].re - isq2).abs() < 1e-10,
            "sv[00] = ({}, {})",
            sv[0].re,
            sv[0].im
        );
        assert!(sv[1].norm_sq() < 1e-10, "sv[01] should be 0");
        assert!(sv[2].norm_sq() < 1e-10, "sv[10] should be 0");
        assert!(
            (sv[3].re - isq2).abs() < 1e-10,
            "sv[11] = ({}, {})",
            sv[3].re,
            sv[3].im
        );
    }

    #[test]
    fn test_teleportation() {
        // Quantum teleportation protocol
        for seed in 0..20u64 {
            let mut s = StabilizerState::new(3);
            // Prepare qubit 0 in |1⟩
            s.x(0);
            // Create Bell pair on qubits 1, 2
            s.h(1);
            s.cnot(1, 2);
            // Bell measurement on qubits 0, 1
            s.cnot(0, 1);
            s.h(0);
            let mut rng = seed;
            let m0 = s.measure(0, &mut rng);
            let m1 = s.measure(1, &mut rng);
            // Corrections on qubit 2
            if m1 == 1 {
                s.x(2);
            }
            if m0 == 1 {
                s.z(2);
            }
            // Qubit 2 should now be |1⟩
            assert_eq!(
                s.measure(2, &mut rng),
                1,
                "Teleportation failed at seed {}",
                seed
            );
        }
    }

    #[test]
    fn test_memory_10k_qubits() {
        // Verify 10,000 qubit allocation succeeds without panic
        let s = StabilizerState::new(10_000);
        assert_eq!(s.n, 10_000);
    }

    #[test]
    fn test_to_statevector_single_qubit_zero() {
        let s = StabilizerState::new(1);
        let sv = s.to_statevector().unwrap();
        assert!((sv[0].re - 1.0).abs() < 1e-10);
        assert!(sv[1].norm_sq() < 1e-10);
    }

    #[test]
    fn test_to_statevector_single_qubit_one() {
        let mut s = StabilizerState::new(1);
        s.x(0);
        let sv = s.to_statevector().unwrap();
        assert!(sv[0].norm_sq() < 1e-10);
        assert!((sv[1].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_to_statevector_plus_state() {
        let mut s = StabilizerState::new(1);
        s.h(0);
        let sv = s.to_statevector().unwrap();
        let isq2 = 1.0 / 2.0f64.sqrt();
        assert!(
            (sv[0].re - isq2).abs() < 1e-10,
            "sv[0] = ({}, {})",
            sv[0].re,
            sv[0].im
        );
        assert!(
            (sv[1].re - isq2).abs() < 1e-10,
            "sv[1] = ({}, {})",
            sv[1].re,
            sv[1].im
        );
    }

    #[test]
    fn test_to_statevector_returns_none_for_large() {
        let s = StabilizerState::new(13);
        assert!(s.to_statevector().is_none());
    }

    #[test]
    fn test_s_gate_phase() {
        // S|+⟩ = (|0⟩ + i|1⟩)/sqrt(2)
        // The projection method may introduce a global phase, so check:
        // 1. Both amplitudes have magnitude 1/sqrt(2)
        // 2. The relative phase sv[1]/sv[0] = i
        let mut s = StabilizerState::new(1);
        s.h(0);
        s.s(0);
        let sv = s.to_statevector().unwrap();
        let half = 0.5;
        assert!(
            (sv[0].norm_sq() - half).abs() < 1e-10,
            "|sv[0]|^2 = {}, expected 0.5",
            sv[0].norm_sq()
        );
        assert!(
            (sv[1].norm_sq() - half).abs() < 1e-10,
            "|sv[1]|^2 = {}, expected 0.5",
            sv[1].norm_sq()
        );
        // Relative phase: sv[1]/sv[0] should be i
        // sv[1]/sv[0] = sv[1] * conj(sv[0]) / |sv[0]|^2
        let ratio = sv[1].mul_fixed(sv[0].conj()).scale(1.0 / sv[0].norm_sq());
        assert!(
            ratio.re.abs() < 1e-10 && (ratio.im - 1.0).abs() < 1e-10,
            "sv[1]/sv[0] = ({}, {}), expected (0, 1)",
            ratio.re,
            ratio.im
        );
    }

    #[test]
    fn test_y_gate() {
        // Y|0⟩ = i|1⟩
        let mut s = StabilizerState::new(1);
        s.y(0);
        let sv = s.to_statevector().unwrap();
        assert!(sv[0].norm_sq() < 1e-10, "sv[0] should be 0");
        // sv[1] should be i (up to global phase)
        assert!(
            (sv[1].norm_sq() - 1.0).abs() < 1e-10,
            "sv[1] should have magnitude 1"
        );
    }

    #[test]
    fn test_double_hadamard_is_identity() {
        let mut s = StabilizerState::new(1);
        s.h(0);
        s.h(0);
        let sv = s.to_statevector().unwrap();
        // Should be back to |0⟩
        assert!((sv[0].re - 1.0).abs() < 1e-10);
        assert!(sv[1].norm_sq() < 1e-10);
    }

    #[test]
    fn test_x_squared_is_identity() {
        let mut s = StabilizerState::new(1);
        s.x(0);
        s.x(0);
        let mut rng = 42u64;
        assert_eq!(s.measure(0, &mut rng), 0);
    }

    #[test]
    fn test_cnot_control_zero_no_flip() {
        // CNOT with control=|0⟩ should not flip target
        let mut s = StabilizerState::new(2);
        s.cnot(0, 1);
        let mut rng = 42u64;
        assert_eq!(s.measure(0, &mut rng), 0);
        assert_eq!(s.measure(1, &mut rng), 0);
    }

    #[test]
    fn test_cnot_control_one_flips_target() {
        // CNOT with control=|1⟩ should flip target
        let mut s = StabilizerState::new(2);
        s.x(0);
        s.cnot(0, 1);
        let mut rng = 42u64;
        assert_eq!(s.measure(0, &mut rng), 1);
        assert_eq!(s.measure(1, &mut rng), 1);
    }

    #[test]
    fn test_bit_helpers() {
        let mut words = vec![0u64; 2];
        assert!(!get_bit(&words, 0));
        set_bit(&mut words, 0, true);
        assert!(get_bit(&words, 0));
        set_bit(&mut words, 65, true);
        assert!(get_bit(&words, 65));
        flip_bit(&mut words, 65);
        assert!(!get_bit(&words, 65));
        flip_bit(&mut words, 3);
        assert!(get_bit(&words, 3));
    }
}
