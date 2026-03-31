//! Pure CJC Quantum Backend — "CJC all the way down"
//!
//! This module implements quantum simulation algorithms using only basic
//! data structures (Vec<f64>, Vec<u64>) that map directly to CJC arrays.
//! All state is inspectable and modifiable from CJC programs.
//!
//! The pure backend is an alternative to the optimized Rust backend.
//! It trades speed for:
//! - **Inspectability**: Users can print/examine quantum state internals
//! - **Modifiability**: Researchers can tweak algorithms without recompiling
//! - **Educational value**: Algorithms are transparent and readable
//! - **AD integration**: CJC's autodiff can differentiate through operations
//!
//! # Determinism
//!
//! Same seed = bit-identical output across runs, guaranteed by:
//! - Fixed iteration order in all loops
//! - Kahan summation for floating-point reductions
//! - No FMA (fused multiply-add)
//! - SplitMix64 PRNG with explicit seed threading

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use cjc_runtime::det_map::DetMap;
use cjc_runtime::value::Value;

/// Helper: create a Value::Map from key-value pairs.
fn make_map(pairs: Vec<(&str, Value)>) -> Value {
    let mut map = DetMap::new();
    for (k, v) in pairs {
        map.insert(Value::String(Rc::new(k.into())), v);
    }
    Value::Map(Rc::new(RefCell::new(map)))
}

// ═══════════════════════════════════════════════════════════════════
// Complex arithmetic — (re, im) pairs
// ═══════════════════════════════════════════════════════════════════

type C = (f64, f64);

const CZERO: C = (0.0, 0.0);
const CONE: C = (1.0, 0.0);

#[inline]
fn c_re(a: C) -> f64 { a.0 }

#[inline]
fn c_im(a: C) -> f64 { a.1 }

#[inline]
fn c_add(a: C, b: C) -> C { (a.0 + b.0, a.1 + b.1) }

#[inline]
fn c_sub(a: C, b: C) -> C { (a.0 - b.0, a.1 - b.1) }

#[inline]
fn c_mul(a: C, b: C) -> C {
    // Fixed sequence: no FMA for determinism
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

#[inline]
fn c_conj(a: C) -> C { (a.0, -a.1) }

#[inline]
fn c_abs2(a: C) -> f64 { a.0 * a.0 + a.1 * a.1 }

#[inline]
fn c_abs(a: C) -> f64 { c_abs2(a).sqrt() }

#[inline]
fn c_scale(s: f64, a: C) -> C { (s * a.0, s * a.1) }

#[inline]
fn c_neg(a: C) -> C { (-a.0, -a.1) }

/// Kahan-compensated sum for deterministic f64 accumulation.
fn kahan_sum(vals: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    let mut comp = 0.0f64;
    for &v in vals {
        let y = v - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    sum
}

// ═══════════════════════════════════════════════════════════════════
// Small matrix SVD — Jacobi one-sided (complex)
// ═══════════════════════════════════════════════════════════════════
//
// Used for MPS bond truncation. Handles matrices up to ~64x64.
// Deterministic: fixed sweep order, no randomization.

/// Compute thin SVD of an m×n complex matrix A (m >= n).
/// Returns (U: m×k, S: k, V: n×k) where k = min(m,n).
/// A is stored column-major: A[row + col*m] = complex value.
pub fn jacobi_svd(a: &[C], m: usize, n: usize) -> (Vec<C>, Vec<f64>, Vec<C>) {
    let k = m.min(n);
    // Work on a copy
    let mut w: Vec<C> = a.to_vec();
    // V = identity
    let mut v: Vec<C> = vec![CZERO; n * n];
    for i in 0..n {
        v[i + i * n] = CONE;
    }

    let max_sweeps = 100;
    let eps = 1e-15;

    for _sweep in 0..max_sweeps {
        let mut converged = true;

        for p in 0..n {
            for q in (p + 1)..n {
                // Compute 2x2 Gram subproblem: G = [alpha, gamma; conj(gamma), beta]
                let mut alpha = 0.0f64;
                let mut beta = 0.0f64;
                let mut gamma = CZERO;

                for i in 0..m {
                    let wp = w[i + p * m];
                    let wq = w[i + q * m];
                    alpha += c_abs2(wp);
                    beta += c_abs2(wq);
                    gamma = c_add(gamma, c_mul(c_conj(wp), wq));
                }

                let gamma_abs = c_abs(gamma);
                if gamma_abs < eps * (alpha * beta).sqrt().max(eps) {
                    continue;
                }
                converged = false;

                // Compute Jacobi rotation
                let tau = (beta - alpha) / (2.0 * gamma_abs);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let cos = 1.0 / (1.0 + t * t).sqrt();
                let sin = t * cos;

                // Phase factor
                let phase = if gamma_abs > eps {
                    (gamma.0 / gamma_abs, gamma.1 / gamma_abs)
                } else {
                    CONE
                };
                let conj_phase = c_conj(phase);

                // Apply rotation to W columns p, q
                for i in 0..m {
                    let wp = w[i + p * m];
                    let wq = w[i + q * m];
                    w[i + p * m] = c_add(c_scale(cos, wp), c_mul(c_scale(sin, conj_phase), wq));
                    w[i + q * m] = c_add(c_scale(-sin, c_mul(phase, wp)), c_scale(cos, wq));
                }

                // Apply rotation to V columns p, q
                for i in 0..n {
                    let vp = v[i + p * n];
                    let vq = v[i + q * n];
                    v[i + p * n] = c_add(c_scale(cos, vp), c_mul(c_scale(sin, conj_phase), vq));
                    v[i + q * n] = c_add(c_scale(-sin, c_mul(phase, vp)), c_scale(cos, vq));
                }
            }
        }

        if converged {
            break;
        }
    }

    // Extract singular values (column norms of W) and sort descending
    let mut sigma: Vec<(f64, usize)> = (0..k)
        .map(|j| {
            let norm2: f64 = (0..m).map(|i| c_abs2(w[i + j * m])).sum();
            (norm2.sqrt(), j)
        })
        .collect();
    sigma.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Build U, S, V in sorted order
    let mut s_out = vec![0.0f64; k];
    let mut u_out = vec![CZERO; m * k];
    let mut v_out = vec![CZERO; n * k];

    for (out_idx, &(sv, orig_idx)) in sigma.iter().enumerate() {
        s_out[out_idx] = sv;
        if sv > eps {
            for i in 0..m {
                u_out[i + out_idx * m] = c_scale(1.0 / sv, w[i + orig_idx * m]);
            }
        }
        for i in 0..n {
            v_out[i + out_idx * n] = v[i + orig_idx * n];
        }
    }

    (u_out, s_out, v_out)
}

// ═══════════════════════════════════════════════════════════════════
// Pure MPS — Matrix Product State (50+ qubits)
// ═══════════════════════════════════════════════════════════════════
//
// Each qubit has a rank-3 tensor: T[phys][bond_left][bond_right]
// Physical index: 0 or 1 (qubit state)
// Bond indices: connect adjacent qubits

/// Pure MPS tensor for one qubit site.
#[derive(Clone, Debug)]
pub struct PureMpsTensor {
    pub bond_left: usize,
    pub bond_right: usize,
    /// Flat storage: data[phys * bl * br + r * br + c] = complex (re, im)
    pub data: Vec<C>,
}

impl PureMpsTensor {
    fn new_zero_state(bl: usize, br: usize) -> Self {
        let mut data = vec![CZERO; 2 * bl * br];
        // |0⟩ state: T[0][0][0] = 1, rest = 0
        data[0] = CONE;
        PureMpsTensor { bond_left: bl, bond_right: br, data }
    }

    fn get(&self, phys: usize, r: usize, c: usize) -> C {
        self.data[phys * self.bond_left * self.bond_right + r * self.bond_right + c]
    }

    fn set(&mut self, phys: usize, r: usize, c: usize, val: C) {
        self.data[phys * self.bond_left * self.bond_right + r * self.bond_right + c] = val;
    }
}

/// Pure MPS state — inspectable from CJC.
#[derive(Clone, Debug)]
pub struct PureMps {
    pub n_qubits: usize,
    pub max_bond: usize,
    pub tensors: Vec<PureMpsTensor>,
}

impl PureMps {
    pub fn new(n_qubits: usize, max_bond: usize) -> Self {
        let tensors: Vec<PureMpsTensor> = (0..n_qubits)
            .map(|_| PureMpsTensor::new_zero_state(1, 1))
            .collect();
        PureMps { n_qubits, max_bond, tensors }
    }

    /// Apply a 2x2 unitary gate to qubit q (in-place).
    pub fn apply_single_qubit(&mut self, q: usize, u: [[C; 2]; 2]) {
        assert!(q < self.n_qubits, "Qubit index out of range");
        let t = &mut self.tensors[q];
        let bl = t.bond_left;
        let br = t.bond_right;
        for r in 0..bl {
            for c in 0..br {
                let old0 = t.get(0, r, c);
                let old1 = t.get(1, r, c);
                t.set(0, r, c, c_add(c_mul(u[0][0], old0), c_mul(u[0][1], old1)));
                t.set(1, r, c, c_add(c_mul(u[1][0], old0), c_mul(u[1][1], old1)));
            }
        }
    }

    /// Apply CNOT between adjacent qubits using SVD truncation.
    pub fn apply_cnot(&mut self, ctrl: usize, targ: usize) {
        assert!(ctrl < self.n_qubits && targ < self.n_qubits);
        let diff = (ctrl as isize - targ as isize).unsigned_abs();
        assert_eq!(diff, 1, "Pure MPS CNOT requires adjacent qubits");

        let (left_idx, right_idx) = if ctrl < targ { (ctrl, targ) } else { (targ, ctrl) };
        let tl = self.tensors[left_idx].clone();
        let tr = self.tensors[right_idx].clone();

        let bl = tl.bond_left;
        let bm = tl.bond_right; // = tr.bond_left
        let br = tr.bond_right;
        assert_eq!(bm, tr.bond_left);

        // Contract: combined[sl][sr][l][r] = sum_m T_left[sl][l][m] * T_right[sr][m][r]
        // Then apply CNOT gate permutation
        let gate_map: [[usize; 2]; 4] = if ctrl < targ {
            [[0, 0], [0, 1], [1, 1], [1, 0]] // |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩
        } else {
            [[0, 0], [1, 1], [1, 0], [0, 1]] // target before control
        };

        // Build combined matrix: rows = (sl, l), cols = (sr, r)
        // dims: (2 * bl) × (2 * br)
        let rows = 2 * bl;
        let cols = 2 * br;
        let mut combined = vec![CZERO; rows * cols]; // column-major for SVD

        for sl in 0..2usize {
            for sr in 0..2usize {
                let [new_sl, new_sr] = gate_map[sl * 2 + sr];
                for l in 0..bl {
                    for r in 0..br {
                        let mut val = CZERO;
                        for m in 0..bm {
                            val = c_add(val, c_mul(
                                tl.get(new_sl, l, m),
                                tr.get(new_sr, m, r),
                            ));
                        }
                        let row = sl * bl + l;
                        let col = sr * br + r;
                        combined[row + col * rows] = val; // column-major
                    }
                }
            }
        }

        // SVD and truncate
        let (u, s, v) = jacobi_svd(&combined, rows, cols);
        let k = s.iter().take_while(|&&sv| sv > 1e-14).count().max(1).min(self.max_bond);

        // Rebuild left tensor: T_left[sl][l][bond_new] = U[sl*bl+l, bond_new] * sqrt(S[bond_new])
        let new_bl_left = tl.bond_left;
        let new_br_left = k;
        let mut new_left = PureMpsTensor {
            bond_left: new_bl_left,
            bond_right: new_br_left,
            data: vec![CZERO; 2 * new_bl_left * new_br_left],
        };
        for sl in 0..2 {
            for l in 0..bl {
                for j in 0..k {
                    let row = sl * bl + l;
                    let sqrt_s = s[j].sqrt();
                    new_left.set(sl, l, j, c_scale(sqrt_s, u[row + j * rows]));
                }
            }
        }

        // Rebuild right tensor: T_right[sr][bond_new][r] = sqrt(S[bond_new]) * V^H[bond_new, sr*br+r]
        let new_bl_right = k;
        let new_br_right = tr.bond_right;
        let mut new_right = PureMpsTensor {
            bond_left: new_bl_right,
            bond_right: new_br_right,
            data: vec![CZERO; 2 * new_bl_right * new_br_right],
        };
        for sr in 0..2 {
            for r in 0..br {
                for j in 0..k {
                    let col = sr * br + r;
                    let sqrt_s = s[j].sqrt();
                    // V^H[j, col] = conj(V[col, j]) — V is column-major n×k
                    let vh = c_conj(v[col + j * cols]);
                    new_right.set(sr, j, r, c_scale(sqrt_s, vh));
                }
            }
        }

        self.tensors[left_idx] = new_left;
        self.tensors[right_idx] = new_right;
    }

    /// Compute ⟨ψ|Z_q|ψ⟩ via left-to-right contraction.
    pub fn z_expectation(&self, q: usize) -> f64 {
        let n = self.n_qubits;
        assert!(q < n);

        // Environment: env[r1][r2] = contraction of sites 0..site-1
        // where r1 indexes bra bond, r2 indexes ket bond
        let mut env: Vec<C> = vec![CONE]; // 1×1 identity

        let mut env_rows = 1usize;
        let mut env_cols = 1usize;

        for site in 0..n {
            let t = &self.tensors[site];
            let bl = t.bond_left;
            let br = t.bond_right;

            let new_rows = br;
            let new_cols = br;
            let mut new_env = vec![CZERO; new_rows * new_cols];

            // sigma_z factor: +1 for phys=0, -1 for phys=1 (at site q)
            for phys in 0..2usize {
                let z_factor = if site == q {
                    if phys == 0 { 1.0 } else { -1.0 }
                } else {
                    1.0
                };

                for r1 in 0..br {
                    for r2 in 0..br {
                        let mut val = CZERO;
                        for l1 in 0..bl {
                            for l2 in 0..bl {
                                let bra = c_conj(t.get(phys, l1, r1));
                                let ket = t.get(phys, l2, r2);
                                let env_val = env[l1 * env_cols + l2];
                                val = c_add(val, c_scale(z_factor, c_mul(c_mul(bra, env_val), ket)));
                            }
                        }
                        new_env[r1 * new_cols + r2] = c_add(
                            new_env[r1 * new_cols + r2],
                            val,
                        );
                    }
                }
            }

            env = new_env;
            env_rows = new_rows;
            env_cols = new_cols;
        }

        // Final environment should be 1×1
        let _ = env_rows; // used implicitly
        c_re(env[0])
    }

    /// Total memory in bytes (for CJC-level reporting).
    pub fn memory_bytes(&self) -> usize {
        self.tensors.iter()
            .map(|t| t.data.len() * 16) // 16 bytes per complex (2 × f64)
            .sum()
    }

    /// Convert to CJC-inspectable Value::Map.
    pub fn to_value_map(&self) -> Value {
        let tensors: Vec<Value> = self.tensors.iter().map(|t| {
            let data_re: Vec<Value> = t.data.iter().map(|c| Value::Float(c.0)).collect();
            let data_im: Vec<Value> = t.data.iter().map(|c| Value::Float(c.1)).collect();
            make_map(vec![
                ("bond_left", Value::Int(t.bond_left as i64)),
                ("bond_right", Value::Int(t.bond_right as i64)),
                ("data_re", Value::Array(Rc::new(data_re))),
                ("data_im", Value::Array(Rc::new(data_im))),
            ])
        }).collect();

        make_map(vec![
            ("_backend", Value::String(Rc::new("pure".into()))),
            ("_type", Value::String(Rc::new("mps".into()))),
            ("n_qubits", Value::Int(self.n_qubits as i64)),
            ("max_bond", Value::Int(self.max_bond as i64)),
            ("tensors", Value::Array(Rc::new(tensors))),
        ])
    }
}

// ═══════════════════════════════════════════════════════════════════
// Pure Stabilizer — CHP Clifford simulator (1000+ qubits)
// ═══════════════════════════════════════════════════════════════════
//
// Tableau representation: 2n+1 rows of (x-bits, z-bits, phase).
// Row i (0..n): stabilizer generator i
// Row i (n..2n): destabilizer i
// Row 2n: scratch row for measurement
// Bits stored as u64 words for efficiency.

#[derive(Clone, Debug)]
pub struct PureStabilizer {
    pub n: usize,
    pub words_per_row: usize,
    /// x-tableau: x[row][word]
    pub x: Vec<Vec<u64>>,
    /// z-tableau: z[row][word]
    pub z: Vec<Vec<u64>>,
    /// Phase: 0 or 1 (represents +1 or -1)
    pub phase: Vec<u8>,
}

impl PureStabilizer {
    pub fn new(n: usize) -> Self {
        let words = (n + 63) / 64;
        let rows = 2 * n + 1;
        let mut x = vec![vec![0u64; words]; rows];
        let mut z = vec![vec![0u64; words]; rows];
        let phase = vec![0u8; rows];

        // Initial state |0...0⟩:
        // Stabilizers: Z_i for each qubit
        // Destabilizers: X_i for each qubit
        for i in 0..n {
            let w = i / 64;
            let b = i % 64;
            z[i][w] = 1u64 << b;          // stabilizer i = Z_i
            x[n + i][w] = 1u64 << b;      // destabilizer i = X_i
        }

        PureStabilizer { n, words_per_row: words, x, z, phase }
    }

    fn get_x(&self, row: usize, qubit: usize) -> bool {
        let w = qubit / 64;
        let b = qubit % 64;
        (self.x[row][w] >> b) & 1 == 1
    }

    fn get_z(&self, row: usize, qubit: usize) -> bool {
        let w = qubit / 64;
        let b = qubit % 64;
        (self.z[row][w] >> b) & 1 == 1
    }

    fn set_x(&mut self, row: usize, qubit: usize, val: bool) {
        let w = qubit / 64;
        let b = qubit % 64;
        if val {
            self.x[row][w] |= 1u64 << b;
        } else {
            self.x[row][w] &= !(1u64 << b);
        }
    }

    fn set_z(&mut self, row: usize, qubit: usize, val: bool) {
        let w = qubit / 64;
        let b = qubit % 64;
        if val {
            self.z[row][w] |= 1u64 << b;
        } else {
            self.z[row][w] &= !(1u64 << b);
        }
    }

    /// Row multiplication: row[target] *= row[source]
    /// Updates phase according to commutation rules.
    fn rowmult(&mut self, target: usize, source: usize) {
        // Word-level phase accumulation
        let mut phase_sum = 0i64;
        for w in 0..self.words_per_row {
            let x1 = self.x[source][w];
            let z1 = self.z[source][w];
            let x2 = self.x[target][w];
            let z2 = self.z[target][w];

            // Count +i and -i contributions from Pauli multiplication
            let pos = (x1 & !z1 & x2 & z2)       // X * Y → +i
                    | (!x1 & z1 & x2 & !z2)       // Z * X → +i
                    | (x1 & z1 & z2 & !x2);       // Y * Z → +i
            let neg = (x1 & !z1 & !x2 & z2)       // X * Z → -i
                    | (!x1 & z1 & x2 & z2)         // Z * Y → -i
                    | (x1 & z1 & x2 & !z2);        // Y * X → -i

            phase_sum += pos.count_ones() as i64;
            phase_sum -= neg.count_ones() as i64;
        }

        // Update phase
        let total_phase = (self.phase[target] as i64 + self.phase[source] as i64) * 2 + phase_sum;
        self.phase[target] = ((total_phase % 4 + 4) % 4 / 2) as u8;

        // XOR the bit rows
        for w in 0..self.words_per_row {
            self.x[target][w] ^= self.x[source][w];
            self.z[target][w] ^= self.z[source][w];
        }
    }

    pub fn h(&mut self, q: usize) {
        for row in 0..(2 * self.n + 1) {
            let xi = self.get_x(row, q);
            let zi = self.get_z(row, q);
            // Phase update: if both X and Z are set, flip phase
            if xi && zi {
                self.phase[row] ^= 1;
            }
            // Swap X and Z
            self.set_x(row, q, zi);
            self.set_z(row, q, xi);
        }
    }

    pub fn s(&mut self, q: usize) {
        for row in 0..(2 * self.n + 1) {
            let xi = self.get_x(row, q);
            let zi = self.get_z(row, q);
            if xi && zi {
                self.phase[row] ^= 1;
            }
            // Z = Z XOR X
            self.set_z(row, q, xi ^ zi);
        }
    }

    pub fn x(&mut self, q: usize) {
        // X gate: flips phase of any row with Z_q set (but not X_q)
        for row in 0..(2 * self.n + 1) {
            if self.get_z(row, q) {
                self.phase[row] ^= 1;
            }
        }
    }

    pub fn y(&mut self, q: usize) {
        for row in 0..(2 * self.n + 1) {
            let xi = self.get_x(row, q);
            let zi = self.get_z(row, q);
            if xi ^ zi {
                self.phase[row] ^= 1;
            }
        }
    }

    pub fn z(&mut self, q: usize) {
        for row in 0..(2 * self.n + 1) {
            if self.get_x(row, q) {
                self.phase[row] ^= 1;
            }
        }
    }

    pub fn cnot(&mut self, ctrl: usize, targ: usize) {
        for row in 0..(2 * self.n + 1) {
            let xc = self.get_x(row, ctrl);
            let zc = self.get_z(row, ctrl);
            let xt = self.get_x(row, targ);
            let zt = self.get_z(row, targ);

            // Phase update
            if xc && zt && (xt ^ zc ^ true) {
                self.phase[row] ^= 1;
            }

            // X propagation: X_targ ^= X_ctrl
            self.set_x(row, targ, xt ^ xc);
            // Z propagation: Z_ctrl ^= Z_targ
            self.set_z(row, ctrl, zc ^ zt);
        }
    }

    /// Measure qubit q. Returns 0 or 1.
    pub fn measure(&mut self, q: usize, rng: &mut u64) -> u8 {
        let n = self.n;

        // Check if outcome is random: look for stabilizer with X_q set
        let mut p_idx: Option<usize> = None;
        for i in 0..n {
            if self.get_x(i, q) {
                p_idx = Some(i);
                break;
            }
        }

        if let Some(p) = p_idx {
            // Random outcome
            // For all rows that anticommute with the measurement, multiply by row p
            for i in 0..(2 * n) {
                if i != p && self.get_x(i, q) {
                    self.rowmult(i, p);
                }
            }
            // Move stabilizer p to destabilizer
            self.x[n + p] = self.x[p].clone();
            self.z[n + p] = self.z[p].clone();
            self.phase[n + p] = self.phase[p];

            // Set stabilizer p to ±Z_q
            for w in 0..self.words_per_row {
                self.x[p][w] = 0;
                self.z[p][w] = 0;
            }
            let wq = q / 64;
            let bq = q % 64;
            self.z[p][wq] = 1u64 << bq;

            // Random outcome
            let outcome = (splitmix64(rng) & 1) as u8;
            self.phase[p] = outcome;
            outcome
        } else {
            // Deterministic outcome
            // Use scratch row (2n) to accumulate
            let scratch = 2 * n;
            for w in 0..self.words_per_row {
                self.x[scratch][w] = 0;
                self.z[scratch][w] = 0;
            }
            self.phase[scratch] = 0;

            for i in 0..n {
                if self.get_x(n + i, q) {
                    self.rowmult(scratch, i);
                }
            }

            self.phase[scratch]
        }
    }

    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// Convert to CJC-inspectable Value::Map.
    pub fn to_value_map(&self) -> Value {
        let x_arr: Vec<Value> = self.x.iter().map(|row| {
            Value::Array(Rc::new(row.iter().map(|&w| Value::Int(w as i64)).collect()))
        }).collect();
        let z_arr: Vec<Value> = self.z.iter().map(|row| {
            Value::Array(Rc::new(row.iter().map(|&w| Value::Int(w as i64)).collect()))
        }).collect();
        let phase_arr: Vec<Value> = self.phase.iter().map(|&p| Value::Int(p as i64)).collect();

        make_map(vec![
            ("_backend", Value::String(Rc::new("pure".into()))),
            ("_type", Value::String(Rc::new("stabilizer".into()))),
            ("n", Value::Int(self.n as i64)),
            ("x", Value::Array(Rc::new(x_arr))),
            ("z", Value::Array(Rc::new(z_arr))),
            ("phase", Value::Array(Rc::new(phase_arr))),
        ])
    }
}

// ═══════════════════════════════════════════════════════════════════
// Pure Density Matrix — Mixed states + noise channels
// ═══════════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct PureDensity {
    pub n_qubits: usize,
    pub dim: usize,
    /// Flat row-major: data[row * dim + col] = complex value
    pub data: Vec<C>,
}

impl PureDensity {
    pub fn new(n_qubits: usize) -> Self {
        let dim = 1 << n_qubits;
        let mut data = vec![CZERO; dim * dim];
        data[0] = CONE; // |0⟩⟨0|
        PureDensity { n_qubits, dim, data }
    }

    fn get(&self, r: usize, c: usize) -> C {
        self.data[r * self.dim + c]
    }

    fn set(&mut self, r: usize, c: usize, val: C) {
        self.data[r * self.dim + c] = val;
    }

    /// Apply a 2×2 unitary gate to qubit q: ρ → U_q ρ U_q†
    pub fn apply_gate_2x2(&mut self, q: usize, u: [[C; 2]; 2]) {
        let dim = self.dim;
        let mask = 1 << q;
        let u_dag = [
            [c_conj(u[0][0]), c_conj(u[1][0])],
            [c_conj(u[0][1]), c_conj(u[1][1])],
        ];

        // ρ' = U ρ U†
        // Step 1: ρ → U ρ (act on row index)
        let mut temp = vec![CZERO; dim * dim];
        for i in 0..dim {
            let i0 = i & !mask;
            let i1 = i | mask;
            let bit = if i & mask != 0 { 1 } else { 0 };
            for j in 0..dim {
                let v0 = self.get(i0, j);
                let v1 = self.get(i1, j);
                temp[i * dim + j] = c_add(c_mul(u[bit][0], v0), c_mul(u[bit][1], v1));
            }
        }

        // Step 2: (U ρ) → (U ρ) U† (act on column index)
        for i in 0..dim {
            for j in 0..dim {
                let j0 = j & !mask;
                let j1 = j | mask;
                let bit = if j & mask != 0 { 1 } else { 0 };
                let v0 = temp[i * dim + j0];
                let v1 = temp[i * dim + j1];
                self.data[i * dim + j] = c_add(c_mul(v0, u_dag[0][bit]), c_mul(v1, u_dag[1][bit]));
            }
        }
    }

    /// Apply CNOT gate.
    pub fn apply_cnot(&mut self, ctrl: usize, targ: usize) {
        let dim = self.dim;
        let ctrl_mask = 1 << ctrl;
        let targ_mask = 1 << targ;

        // CNOT permutation: |c,t⟩ → |c, c⊕t⟩
        // Apply to both row and column indices
        let mut new_data = vec![CZERO; dim * dim];
        for i in 0..dim {
            let new_i = if i & ctrl_mask != 0 { i ^ targ_mask } else { i };
            for j in 0..dim {
                let new_j = if j & ctrl_mask != 0 { j ^ targ_mask } else { j };
                new_data[new_i * dim + new_j] = self.data[i * dim + j];
            }
        }
        self.data = new_data;
    }

    /// Apply depolarizing noise channel on qubit q: ρ → (1-p)ρ + p/3(XρX + YρY + ZρZ)
    pub fn apply_depolarize(&mut self, q: usize, p: f64) {
        let dim = self.dim;
        let mask = 1 << q;

        let mut x_rho_x = vec![CZERO; dim * dim];
        let mut y_rho_y = vec![CZERO; dim * dim];
        let mut z_rho_z = vec![CZERO; dim * dim];

        for i in 0..dim {
            for j in 0..dim {
                let val = self.get(i, j);
                let fi = i ^ mask;
                let fj = j ^ mask;

                // X ρ X†
                x_rho_x[fi * dim + fj] = val;

                // Z ρ Z†
                let zi = if i & mask != 0 { -1.0 } else { 1.0 };
                let zj = if j & mask != 0 { -1.0 } else { 1.0 };
                z_rho_z[i * dim + j] = c_scale(zi * zj, val);

                // Y ρ Y† = (iXZ) ρ (iXZ)† = XZ ρ ZX (phases cancel)
                let yi = if i & mask != 0 { 1.0 } else { -1.0 };
                let yj = if j & mask != 0 { 1.0 } else { -1.0 };
                y_rho_y[fi * dim + fj] = c_scale(yi * yj, val);
            }
        }

        let c1 = 1.0 - p;
        let c2 = p / 3.0;
        for idx in 0..dim * dim {
            self.data[idx] = c_add(
                c_scale(c1, self.data[idx]),
                c_scale(c2, c_add(x_rho_x[idx], c_add(y_rho_y[idx], z_rho_z[idx]))),
            );
        }
    }

    /// Apply dephasing channel on qubit q: ρ → (1-p)ρ + p·ZρZ
    pub fn apply_dephase(&mut self, q: usize, p: f64) {
        let dim = self.dim;
        let mask = 1 << q;
        let c1 = 1.0 - p;
        let c2 = p;
        for i in 0..dim {
            for j in 0..dim {
                let zi = if i & mask != 0 { -1.0 } else { 1.0 };
                let zj = if j & mask != 0 { -1.0 } else { 1.0 };
                let val = self.get(i, j);
                let z_val = c_scale(zi * zj, val);
                self.set(i, j, c_add(c_scale(c1, val), c_scale(c2, z_val)));
            }
        }
    }

    /// Apply amplitude damping channel on qubit q.
    pub fn apply_amplitude_damp(&mut self, q: usize, gamma: f64) {
        let dim = self.dim;
        let mask = 1 << q;
        let sg = gamma.sqrt();
        let s1g = (1.0 - gamma).sqrt();

        // Kraus: K0 = |0⟩⟨0| + sqrt(1-γ)|1⟩⟨1|, K1 = sqrt(γ)|0⟩⟨1|
        // ρ' = K0 ρ K0† + K1 ρ K1†
        let mut new_data = vec![CZERO; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                let val = self.get(i, j);
                let ib = (i >> q) & 1;
                let jb = (j >> q) & 1;

                // K0 contribution
                let k0i = if ib == 0 { 1.0 } else { s1g };
                let k0j = if jb == 0 { 1.0 } else { s1g };
                new_data[i * dim + j] = c_add(new_data[i * dim + j], c_scale(k0i * k0j, val));

                // K1 contribution: K1|1⟩ = sqrt(γ)|0⟩, K1|0⟩ = 0
                if ib == 1 && jb == 1 {
                    let ni = i ^ mask; // flip qubit to 0
                    let nj = j ^ mask;
                    new_data[ni * dim + nj] = c_add(new_data[ni * dim + nj], c_scale(gamma, val));
                }
            }
        }
        self.data = new_data;
    }

    pub fn trace(&self) -> f64 {
        let vals: Vec<f64> = (0..self.dim).map(|i| c_re(self.get(i, i))).collect();
        kahan_sum(&vals)
    }

    pub fn purity(&self) -> f64 {
        // Tr(ρ²) = sum_{i,j} |ρ_{ij}|²
        let vals: Vec<f64> = self.data.iter().map(|c| c_abs2(*c)).collect();
        kahan_sum(&vals)
    }

    pub fn von_neumann_entropy(&self) -> f64 {
        // Eigenvalues of ρ via diagonalization, then -sum(λ log λ)
        let eigenvalues = self.eigenvalues_hermitian();
        let mut entropy = 0.0f64;
        for &lam in &eigenvalues {
            if lam > 1e-15 {
                entropy -= lam * lam.ln();
            }
        }
        entropy
    }

    pub fn probabilities(&self) -> Vec<f64> {
        (0..self.dim).map(|i| c_re(self.get(i, i)).max(0.0)).collect()
    }

    /// Compute eigenvalues of Hermitian matrix using Jacobi eigenvalue algorithm.
    fn eigenvalues_hermitian(&self) -> Vec<f64> {
        let n = self.dim;
        // Work on real part of diagonal and off-diagonal
        // For a Hermitian matrix, use Jacobi rotation
        let mut a: Vec<Vec<C>> = (0..n)
            .map(|i| (0..n).map(|j| self.get(i, j)).collect())
            .collect();

        let max_iter = 100 * n * n;
        let eps = 1e-15;

        for _ in 0..max_iter {
            // Find largest off-diagonal element
            let mut max_val = 0.0f64;
            let mut p = 0;
            let mut q = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    let v = c_abs2(a[i][j]);
                    if v > max_val {
                        max_val = v;
                        p = i;
                        q = j;
                    }
                }
            }
            if max_val.sqrt() < eps {
                break;
            }

            // Compute Jacobi rotation to zero a[p][q]
            let app = c_re(a[p][p]);
            let aqq = c_re(a[q][q]);
            let apq = a[p][q];

            let tau = (aqq - app) / (2.0 * c_abs(apq));
            let t = if tau >= 0.0 {
                1.0 / (tau + (1.0 + tau * tau).sqrt())
            } else {
                -1.0 / (-tau + (1.0 + tau * tau).sqrt())
            };
            let cos = 1.0 / (1.0 + t * t).sqrt();
            let sin = t * cos;

            let phase = if c_abs(apq) > eps {
                let a = c_abs(apq);
                (apq.0 / a, apq.1 / a)
            } else {
                CONE
            };
            let cphase = c_conj(phase);

            // Apply rotation
            for k in 0..n {
                let akp = a[k][p];
                let akq = a[k][q];
                a[k][p] = c_add(c_scale(cos, akp), c_mul(c_scale(sin, cphase), akq));
                a[k][q] = c_add(c_scale(-sin, c_mul(phase, akp)), c_scale(cos, akq));
            }
            for k in 0..n {
                let apk = a[p][k];
                let aqk = a[q][k];
                a[p][k] = c_add(c_scale(cos, apk), c_mul(c_scale(sin, phase), aqk));
                a[q][k] = c_add(c_scale(-sin, c_mul(cphase, apk)), c_scale(cos, aqk));
            }
        }

        (0..n).map(|i| c_re(a[i][i])).collect()
    }

    /// Convert to CJC-inspectable Value::Map.
    pub fn to_value_map(&self) -> Value {
        let data_re: Vec<Value> = self.data.iter().map(|c| Value::Float(c.0)).collect();
        let data_im: Vec<Value> = self.data.iter().map(|c| Value::Float(c.1)).collect();

        make_map(vec![
            ("_backend", Value::String(Rc::new("pure".into()))),
            ("_type", Value::String(Rc::new("density".into()))),
            ("n_qubits", Value::Int(self.n_qubits as i64)),
            ("dim", Value::Int(self.dim as i64)),
            ("data_re", Value::Array(Rc::new(data_re))),
            ("data_im", Value::Array(Rc::new(data_im))),
        ])
    }
}

// ═══════════════════════════════════════════════════════════════════
// Pure Circuit + Statevector
// ═══════════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
pub enum PureGate {
    H(usize), X(usize), Y(usize), Z(usize),
    S(usize), T(usize),
    Rx(usize, f64), Ry(usize, f64), Rz(usize, f64),
    CNOT(usize, usize), CZ(usize, usize), SWAP(usize, usize),
    Toffoli(usize, usize, usize),
}

#[derive(Clone, Debug)]
pub struct PureCircuit {
    pub n_qubits: usize,
    pub gates: Vec<PureGate>,
}

impl PureCircuit {
    pub fn new(n: usize) -> Self {
        PureCircuit { n_qubits: n, gates: vec![] }
    }

    pub fn add(&mut self, gate: PureGate) {
        self.gates.push(gate);
    }

    pub fn n_gates(&self) -> usize {
        self.gates.len()
    }

    /// Execute circuit → statevector.
    pub fn execute(&self) -> PureStatevector {
        let dim = 1 << self.n_qubits;
        let mut sv = vec![CZERO; dim];
        sv[0] = CONE; // |0...0⟩

        for gate in &self.gates {
            match gate {
                PureGate::H(q) => self.apply_1q(&mut sv, *q, &h_matrix()),
                PureGate::X(q) => self.apply_1q(&mut sv, *q, &x_matrix()),
                PureGate::Y(q) => self.apply_1q(&mut sv, *q, &y_matrix()),
                PureGate::Z(q) => self.apply_1q(&mut sv, *q, &z_matrix()),
                PureGate::S(q) => self.apply_1q(&mut sv, *q, &s_matrix()),
                PureGate::T(q) => self.apply_1q(&mut sv, *q, &t_matrix()),
                PureGate::Rx(q, t) => self.apply_1q(&mut sv, *q, &rx_matrix(*t)),
                PureGate::Ry(q, t) => self.apply_1q(&mut sv, *q, &ry_matrix(*t)),
                PureGate::Rz(q, t) => self.apply_1q(&mut sv, *q, &rz_matrix(*t)),
                PureGate::CNOT(c, t) => self.apply_cnot(&mut sv, *c, *t),
                PureGate::CZ(a, b) => self.apply_cz(&mut sv, *a, *b),
                PureGate::SWAP(a, b) => self.apply_swap(&mut sv, *a, *b),
                PureGate::Toffoli(a, b, c) => self.apply_toffoli(&mut sv, *a, *b, *c),
            }
        }

        PureStatevector { n_qubits: self.n_qubits, amplitudes: sv }
    }

    fn apply_1q(&self, sv: &mut [C], q: usize, u: &[[C; 2]; 2]) {
        let mask = 1usize << q;
        let dim = sv.len();
        let mut i = 0;
        while i < dim {
            if i & mask == 0 {
                let j = i | mask;
                let a0 = sv[i];
                let a1 = sv[j];
                sv[i] = c_add(c_mul(u[0][0], a0), c_mul(u[0][1], a1));
                sv[j] = c_add(c_mul(u[1][0], a0), c_mul(u[1][1], a1));
            }
            i += 1;
        }
    }

    fn apply_cnot(&self, sv: &mut [C], ctrl: usize, targ: usize) {
        let dim = sv.len();
        let cm = 1 << ctrl;
        let tm = 1 << targ;
        for i in 0..dim {
            if i & cm != 0 && i & tm == 0 {
                let j = i | tm;
                sv.swap(i, j);
            }
        }
    }

    fn apply_cz(&self, sv: &mut [C], a: usize, b: usize) {
        let dim = sv.len();
        let am = 1 << a;
        let bm = 1 << b;
        for i in 0..dim {
            if i & am != 0 && i & bm != 0 {
                sv[i] = c_neg(sv[i]);
            }
        }
    }

    fn apply_swap(&self, sv: &mut [C], a: usize, b: usize) {
        let dim = sv.len();
        let am = 1 << a;
        let bm = 1 << b;
        for i in 0..dim {
            let ba = (i >> a) & 1;
            let bb = (i >> b) & 1;
            if ba != bb && ba < bb {
                let j = (i ^ am) ^ bm;
                sv.swap(i, j);
            }
        }
    }

    fn apply_toffoli(&self, sv: &mut [C], c1: usize, c2: usize, t: usize) {
        let dim = sv.len();
        let c1m = 1 << c1;
        let c2m = 1 << c2;
        let tm = 1 << t;
        for i in 0..dim {
            if i & c1m != 0 && i & c2m != 0 && i & tm == 0 {
                let j = i | tm;
                sv.swap(i, j);
            }
        }
    }

    /// Execute and measure all qubits.
    pub fn execute_and_measure(&self, rng: &mut u64) -> Vec<u8> {
        let sv = self.execute();
        sv.measure_all(rng)
    }
}

#[derive(Clone, Debug)]
pub struct PureStatevector {
    pub n_qubits: usize,
    pub amplitudes: Vec<C>,
}

impl PureStatevector {
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|a| c_abs2(*a)).collect()
    }

    pub fn measure_all(&self, rng: &mut u64) -> Vec<u8> {
        let probs = self.probabilities();
        let r = splitmix64_f64(rng);
        let mut cum = 0.0f64;
        let mut outcome = 0usize;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if r < cum {
                outcome = i;
                break;
            }
        }
        // Decompose outcome into bits
        (0..self.n_qubits)
            .map(|q| ((outcome >> q) & 1) as u8)
            .collect()
    }

    pub fn sample(&self, n_shots: usize, rng: &mut u64) -> Vec<usize> {
        let probs = self.probabilities();
        (0..n_shots).map(|_| {
            let r = splitmix64_f64(rng);
            let mut cum = 0.0;
            let mut out = 0;
            for (i, &p) in probs.iter().enumerate() {
                cum += p;
                if r < cum {
                    out = i;
                    break;
                }
            }
            out
        }).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Gate matrices
// ═══════════════════════════════════════════════════════════════════

pub fn h_matrix() -> [[C; 2]; 2] {
    let isq2 = 1.0 / 2.0f64.sqrt();
    [[(isq2, 0.0), (isq2, 0.0)],
     [(isq2, 0.0), (-isq2, 0.0)]]
}

pub fn x_matrix() -> [[C; 2]; 2] {
    [[CZERO, CONE], [CONE, CZERO]]
}

pub fn y_matrix() -> [[C; 2]; 2] {
    [[CZERO, (0.0, -1.0)], [(0.0, 1.0), CZERO]]
}

pub fn z_matrix() -> [[C; 2]; 2] {
    [[CONE, CZERO], [CZERO, (-1.0, 0.0)]]
}

pub fn s_matrix() -> [[C; 2]; 2] {
    [[CONE, CZERO], [CZERO, (0.0, 1.0)]]
}

pub fn t_matrix() -> [[C; 2]; 2] {
    let v = 1.0 / 2.0f64.sqrt();
    [[CONE, CZERO], [CZERO, (v, v)]]
}

pub fn rx_matrix(theta: f64) -> [[C; 2]; 2] {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [[(c, 0.0), (0.0, -s)], [(0.0, -s), (c, 0.0)]]
}

pub fn ry_matrix(theta: f64) -> [[C; 2]; 2] {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [[(c, 0.0), (-s, 0.0)], [(s, 0.0), (c, 0.0)]]
}

pub fn rz_matrix(theta: f64) -> [[C; 2]; 2] {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [[(c, -s), CZERO], [CZERO, (c, s)]]
}

// ═══════════════════════════════════════════════════════════════════
// PRNG — SplitMix64 (deterministic, same as Rust backend)
// ═══════════════════════════════════════════════════════════════════

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn splitmix64_f64(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

// ═══════════════════════════════════════════════════════════════════
// Inspect helper — extract CJC-inspectable map from QuantumState
// ═══════════════════════════════════════════════════════════════════

pub fn quantum_inspect(val: &Value) -> Result<Value, String> {
    match val {
        Value::QuantumState(rc) => {
            let borrow = rc.borrow();
            if let Some(mps) = borrow.downcast_ref::<PureMps>() {
                Ok(mps.to_value_map())
            } else if let Some(stab) = borrow.downcast_ref::<PureStabilizer>() {
                Ok(stab.to_value_map())
            } else if let Some(dm) = borrow.downcast_ref::<PureDensity>() {
                Ok(dm.to_value_map())
            } else {
                Err("quantum_inspect: not a pure backend state (use Rust-backend states with native tools)".into())
            }
        }
        _ => Err("quantum_inspect: expected QuantumState".into()),
    }
}

// ═══════════════════════════════════════════════════════════════════
// Wrap helpers
// ═══════════════════════════════════════════════════════════════════

pub fn wrap_pure<T: Any + 'static>(val: T) -> Value {
    Value::QuantumState(Rc::new(RefCell::new(val)))
}

/// Check if a value is a pure backend state of a given type.
pub fn is_pure<T: Any + 'static>(val: &Value) -> bool {
    match val {
        Value::QuantumState(rc) => rc.borrow().downcast_ref::<T>().is_some(),
        _ => false,
    }
}

/// Check if the last argument is the string "pure".
pub fn has_pure_flag(args: &[Value]) -> bool {
    matches!(args.last(), Some(Value::String(s)) if s.as_ref() == "pure")
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_arithmetic() {
        let a = (1.0, 2.0);
        let b = (3.0, 4.0);
        let c = c_mul(a, b);
        assert!((c.0 - (-5.0)).abs() < 1e-10);
        assert!((c.1 - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_jacobi_svd_identity() {
        let a = vec![CONE, CZERO, CZERO, CONE]; // 2x2 identity, column-major
        let (u, s, v) = jacobi_svd(&a, 2, 2);
        assert!((s[0] - 1.0).abs() < 1e-10);
        assert!((s[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_jacobi_svd_simple() {
        // [[3, 0], [0, 2]] column-major
        let a = vec![(3.0, 0.0), CZERO, CZERO, (2.0, 0.0)];
        let (_, s, _) = jacobi_svd(&a, 2, 2);
        assert!((s[0] - 3.0).abs() < 1e-10);
        assert!((s[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_pure_mps_zero_state() {
        let mps = PureMps::new(3, 4);
        assert_eq!(mps.n_qubits, 3);
        let z0 = mps.z_expectation(0);
        assert!((z0 - 1.0).abs() < 1e-10, "|0⟩ Z-exp should be 1, got {}", z0);
    }

    #[test]
    fn test_pure_mps_hadamard() {
        let mut mps = PureMps::new(3, 4);
        mps.apply_single_qubit(0, h_matrix());
        let z = mps.z_expectation(0);
        assert!(z.abs() < 1e-10, "H|0⟩ Z-exp should be 0, got {}", z);
    }

    #[test]
    fn test_pure_mps_x_gate() {
        let mut mps = PureMps::new(3, 4);
        mps.apply_single_qubit(1, x_matrix());
        let z = mps.z_expectation(1);
        assert!((z - (-1.0)).abs() < 1e-10, "X|0⟩ Z-exp should be -1, got {}", z);
    }

    #[test]
    fn test_pure_mps_cnot() {
        let mut mps = PureMps::new(2, 4);
        mps.apply_single_qubit(0, h_matrix());
        mps.apply_cnot(0, 1);
        let z0 = mps.z_expectation(0);
        assert!(z0.abs() < 1e-10, "Bell Z0 should be 0, got {}", z0);
    }

    #[test]
    fn test_pure_mps_50q() {
        let mut mps = PureMps::new(50, 16);
        mps.apply_single_qubit(0, h_matrix());
        mps.apply_single_qubit(25, h_matrix());
        let z = mps.z_expectation(0);
        assert!(z.abs() < 1e-10, "50q H|0⟩ Z-exp should be 0, got {}", z);
    }

    #[test]
    fn test_pure_stabilizer_zero_state() {
        let s = PureStabilizer::new(3);
        assert_eq!(s.num_qubits(), 3);
    }

    #[test]
    fn test_pure_stabilizer_x_measure() {
        let mut s = PureStabilizer::new(3);
        s.x(0);
        let mut rng = 42u64;
        let result = s.measure(0, &mut rng);
        assert_eq!(result, 1, "X|0⟩ should measure 1");
    }

    #[test]
    fn test_pure_stabilizer_1000q() {
        let mut s = PureStabilizer::new(1000);
        s.h(0);
        s.cnot(0, 1);
        s.x(999);
        let mut rng = 42u64;
        let result = s.measure(999, &mut rng);
        assert_eq!(result, 1, "X|999⟩ should measure 1");
    }

    #[test]
    fn test_pure_density_trace() {
        let d = PureDensity::new(2);
        let tr = d.trace();
        assert!((tr - 1.0).abs() < 1e-10, "Trace should be 1, got {}", tr);
    }

    #[test]
    fn test_pure_density_gate_purity() {
        let mut d = PureDensity::new(2);
        d.apply_gate_2x2(0, h_matrix());
        let p = d.purity();
        assert!((p - 1.0).abs() < 1e-10, "Pure state purity should be 1, got {}", p);
    }

    #[test]
    fn test_pure_density_noise() {
        let mut d = PureDensity::new(2);
        d.apply_gate_2x2(0, h_matrix());
        d.apply_depolarize(0, 0.1);
        let p = d.purity();
        assert!(p < 1.0 && p > 0.0, "Noisy purity should be < 1, got {}", p);
    }

    #[test]
    fn test_pure_circuit_bell() {
        let mut c = PureCircuit::new(2);
        c.add(PureGate::H(0));
        c.add(PureGate::CNOT(0, 1));
        let sv = c.execute();
        let probs = sv.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[3] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_pure_determinism_mps() {
        let run = || {
            let mut mps = PureMps::new(10, 8);
            mps.apply_single_qubit(0, h_matrix());
            mps.apply_single_qubit(5, ry_matrix(0.7));
            mps.z_expectation(5)
        };
        assert_eq!(run().to_bits(), run().to_bits(), "Must be bit-identical");
    }

    #[test]
    fn test_pure_determinism_stabilizer() {
        let run = || {
            let mut s = PureStabilizer::new(100);
            s.h(0);
            s.cnot(0, 1);
            let mut rng = 42u64;
            s.measure(0, &mut rng)
        };
        assert_eq!(run(), run(), "Must be deterministic");
    }

    #[test]
    fn test_inspect_mps() {
        let mps = PureMps::new(3, 4);
        let map = mps.to_value_map();
        assert_eq!(map.type_name(), "Map");
    }
}
