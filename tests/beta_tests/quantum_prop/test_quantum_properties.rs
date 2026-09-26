//! Property tests for cjc-quantum (proptest).
//!
//! Each property compares against an independent oracle, usually the dense
//! statevector simulator, or checks an exact invariant. Tolerances are stated
//! next to each assertion. "Bits" means `f64::to_bits` equality.

use cjc_quantum::density::{
    amplitude_damping_channel, dephasing_channel, depolarizing_channel, DensityMatrix,
};
use cjc_quantum::mps::Mps;
use cjc_quantum::stabilizer::StabilizerState;
use cjc_quantum::{Circuit, Gate};
use cjc_runtime::complex::ComplexF64;
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

/// Any of the 13 circuit gates on `n` qubits (n ≥ 3), distinct operands.
fn gate(n: usize) -> impl Strategy<Value = Gate> {
    let q = 0..n;
    let angle = -4.0 * std::f64::consts::PI..4.0 * std::f64::consts::PI;
    prop_oneof![
        q.clone().prop_map(Gate::H),
        q.clone().prop_map(Gate::X),
        q.clone().prop_map(Gate::Y),
        q.clone().prop_map(Gate::Z),
        q.clone().prop_map(Gate::S),
        q.clone().prop_map(Gate::T),
        (q.clone(), angle.clone()).prop_map(|(a, t)| Gate::Rx(a, t)),
        (q.clone(), angle.clone()).prop_map(|(a, t)| Gate::Ry(a, t)),
        (q.clone(), angle).prop_map(|(a, t)| Gate::Rz(a, t)),
        (q.clone(), 1..n).prop_map(move |(a, d)| Gate::CNOT(a, (a + d) % n)),
        (q.clone(), 1..n).prop_map(move |(a, d)| Gate::CZ(a, (a + d) % n)),
        (q.clone(), 1..n).prop_map(move |(a, d)| Gate::SWAP(a, (a + d) % n)),
        (q, 1..n, 1..n)
            .prop_filter("distinct", move |(_, d1, d2)| d1 != d2)
            .prop_map(move |(a, d1, d2)| Gate::Toffoli(a, (a + d1) % n, (a + d2) % n)),
    ]
}

fn circuit() -> impl Strategy<Value = (usize, Vec<Gate>)> {
    (3usize..=6).prop_flat_map(|n| (Just(n), prop::collection::vec(gate(n), 0..30)))
}

fn build(n: usize, gates: &[Gate]) -> Circuit {
    let mut c = Circuit::new(n);
    for g in gates {
        c.add(g.clone());
    }
    c
}

/// Inverse of a gate, up to global phase (S† and T† are Rz(−π/2), Rz(−π/4)).
fn inverse(g: &Gate) -> Gate {
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};
    match *g {
        Gate::S(q) => Gate::Rz(q, -FRAC_PI_2),
        Gate::T(q) => Gate::Rz(q, -FRAC_PI_4),
        Gate::Rx(q, t) => Gate::Rx(q, -t),
        Gate::Ry(q, t) => Gate::Ry(q, -t),
        Gate::Rz(q, t) => Gate::Rz(q, -t),
        ref other => other.clone(), // H X Y Z CNOT CZ SWAP Toffoli are involutions
    }
}

fn probs(n: usize, gates: &[Gate]) -> Vec<f64> {
    build(n, gates).execute().unwrap().probabilities()
}

fn ry(theta: f64) -> [[ComplexF64; 2]; 2] {
    let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    [
        [ComplexF64::real(c), ComplexF64::real(-s)],
        [ComplexF64::real(s), ComplexF64::real(c)],
    ]
}

fn z_expectation(probs: &[f64], q: usize) -> f64 {
    probs
        .iter()
        .enumerate()
        .map(|(k, p)| if (k >> q) & 1 == 0 { *p } else { -*p })
        .sum()
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Unitary gates preserve the norm. Tolerance: generous ulp budget for
    /// ≤30 gates on ≤64 amplitudes.
    #[test]
    fn prop_gates_preserve_normalization((n, gates) in circuit()) {
        let p = probs(n, &gates);
        let total: f64 = p.iter().sum();
        prop_assert!((total - 1.0).abs() < 1e-12, "sum of probabilities = {}", total);
        prop_assert!(p.iter().all(|x| *x >= 0.0 && x.is_finite()));
    }

    /// U followed by U† returns |0…0⟩ (probabilities; S†/T† only up to phase).
    #[test]
    fn prop_inverse_circuit_restores_ground_state((n, gates) in circuit()) {
        let mut all = gates.clone();
        all.extend(gates.iter().rev().map(inverse));
        let p = probs(n, &all);
        prop_assert!((p[0] - 1.0).abs() < 1e-10, "P(|0…0⟩) = {}", p[0]);
    }

    /// Same circuit + same seed ⇒ bit-identical probabilities, samples, and
    /// terminal measurements. Different executions must not drift.
    #[test]
    fn prop_replay_is_bit_identical((n, gates) in circuit(), seed in any::<u64>()) {
        let c = build(n, &gates);
        let (p1, p2) = (c.execute().unwrap().probabilities(), c.execute().unwrap().probabilities());
        prop_assert!(p1.iter().zip(&p2).all(|(a, b)| a.to_bits() == b.to_bits()));
        let (mut r1, mut r2) = (seed, seed);
        prop_assert_eq!(c.sample(32, &mut r1).unwrap(), c.sample(32, &mut r2).unwrap());
        let (mut m1, mut m2) = (seed, seed);
        prop_assert_eq!(
            c.execute_and_measure(&mut m1).unwrap().0,
            c.execute_and_measure(&mut m2).unwrap().0
        );
    }

    /// MPS with an exact bond dimension agrees with the dense statevector on
    /// every ⟨Z_q⟩ for {H, X, Ry, adjacent CNOT} circuits.
    #[test]
    fn prop_mps_matches_dense(
        n in 2usize..=7,
        ops in prop::collection::vec((0u8..4, 0usize..7, -3.0f64..3.0), 0..25),
    ) {
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let hm = [[ComplexF64::real(h), ComplexF64::real(h)], [ComplexF64::real(h), ComplexF64::real(-h)]];
        let xm = [[ComplexF64::ZERO, ComplexF64::ONE], [ComplexF64::ONE, ComplexF64::ZERO]];
        let mut mps = Mps::with_max_bond(n, 64); // 2^(7/2) < 64: exact
        let mut gates = Vec::new();
        for &(kind, q, t) in &ops {
            let q = q % n;
            match kind {
                0 => { mps.apply_single_qubit(q, hm); gates.push(Gate::H(q)); }
                1 => { mps.apply_single_qubit(q, xm); gates.push(Gate::X(q)); }
                2 => { mps.apply_single_qubit(q, ry(t)); gates.push(Gate::Ry(q, t)); }
                _ => {
                    if q + 1 < n {
                        mps.apply_cnot_adjacent(q, q + 1);
                        gates.push(Gate::CNOT(q, q + 1));
                    }
                }
            }
        }
        let p = probs(n, &gates);
        for q in 0..n {
            let zm = cjc_quantum::qml::mps_single_z_expectation(&mps, q);
            let zd = z_expectation(&p, q);
            prop_assert!((zm - zd).abs() < 1e-10, "qubit {}: mps {} dense {}", q, zm, zd);
        }
    }

    /// Noise-free density matrix = |ψ⟩⟨ψ| of the dense simulation.
    #[test]
    fn prop_density_matches_dense_without_noise((n, gates) in (3usize..=4).prop_flat_map(|n| (Just(n), prop::collection::vec(gate(n), 0..20)))) {
        let mut rho = DensityMatrix::new(n);
        for g in &gates {
            rho.apply_gate(g);
        }
        let pd = rho.probabilities();
        let ps = probs(n, &gates);
        for (a, b) in pd.iter().zip(&ps) {
            prop_assert!((a - b).abs() < 1e-12, "density {} vs dense {}", a, b);
        }
    }

    /// Channels keep trace 1 and probabilities non-negative for any p ∈ [0, 1].
    #[test]
    fn prop_density_channels_preserve_trace(
        ops in prop::collection::vec((0u8..6, 0usize..3, 0.0f64..=1.0), 0..25),
    ) {
        let n = 3;
        let mut rho = DensityMatrix::new(n);
        rho.apply_gate(&Gate::H(0));
        rho.apply_gate(&Gate::CNOT(0, 1));
        for &(kind, q, p) in &ops {
            match kind {
                0 => rho.apply_single_qubit_channel(q, &depolarizing_channel(p)),
                1 => rho.apply_single_qubit_channel(q, &dephasing_channel(p)),
                2 => rho.apply_single_qubit_channel(q, &amplitude_damping_channel(p)),
                3 => rho.apply_gate(&Gate::H(q)),
                4 => rho.apply_gate(&Gate::Ry(q, p * 3.0)),
                _ => rho.apply_gate(&Gate::CNOT(q, (q + 1) % n)),
            }
        }
        prop_assert!((rho.trace() - 1.0).abs() < 1e-12, "trace = {}", rho.trace());
        prop_assert!(rho.probabilities().iter().all(|x| *x > -1e-12));
        let purity = rho.purity();
        prop_assert!(purity <= 1.0 + 1e-12 && purity >= 1.0 / 8.0 - 1e-12, "purity = {}", purity);
    }

    /// CHP stabilizer simulation agrees with the dense statevector on random
    /// Clifford circuits (compared as probability distributions).
    #[test]
    fn prop_stabilizer_matches_dense(
        n in 2usize..=6,
        ops in prop::collection::vec((0u8..6, 0usize..6, 1usize..6), 0..40),
    ) {
        let mut st = StabilizerState::new(n);
        let mut gates = Vec::new();
        for &(kind, q, d) in &ops {
            let q = q % n;
            match kind {
                0 => { st.h(q); gates.push(Gate::H(q)); }
                1 => { st.s(q); gates.push(Gate::S(q)); }
                2 => { st.x(q); gates.push(Gate::X(q)); }
                3 => { st.y(q); gates.push(Gate::Y(q)); }
                4 => { st.z(q); gates.push(Gate::Z(q)); }
                _ => {
                    let t = (q + d % n.max(2)) % n;
                    if t != q {
                        st.cnot(q, t);
                        gates.push(Gate::CNOT(q, t));
                    }
                }
            }
        }
        let sv = st.to_statevector().expect("n ≤ 12 must convert");
        let ps = probs(n, &gates);
        for (k, (a, b)) in sv.iter().map(|c| c.norm_sq()).zip(&ps).enumerate() {
            prop_assert!((a - b).abs() < 1e-10, "basis {}: stabilizer {} dense {}", k, a, b);
        }
    }

    /// Generated `.cjcl` gate programs give identical output under cjc-eval and
    /// cjc-mir-exec (parity), compared as rendered values.
    #[test]
    fn prop_eval_mir_parity_on_generated_programs(
        n in 1usize..=4,
        ops in prop::collection::vec((0u8..8, 0usize..4, 1usize..4, -3.0f64..3.0), 0..12),
        seed in 0i64..1000,
    ) {
        let mut body = format!("    let c = qubits({});\n", n);
        for &(kind, q, d, t) in &ops {
            let q = q % n;
            let r = (q + d % n.max(1)) % n;
            let line = match kind {
                0 => format!("q_h(c, {})", q),
                1 => format!("q_x(c, {})", q),
                2 => format!("q_s(c, {})", q),
                3 => format!("q_t(c, {})", q),
                4 => format!("q_rx(c, {}, {:?})", q, t),
                5 => format!("q_ry(c, {}, {:?})", q, t),
                6 => format!("q_rz(c, {}, {:?})", q, t),
                _ if r != q => format!("q_cx(c, {}, {})", q, r),
                _ => format!("q_z(c, {})", q),
            };
            body.push_str(&format!("    let c = {};\n", line));
        }
        body.push_str(&format!("    [q_probs(c), q_sample(c, 8, {})]\n", seed));
        let src = format!("fn main() -> Any {{\n{}}}\n", body);
        let (prog, diags) = cjc_parser::parse_source(&src);
        prop_assert!(!diags.has_errors(), "parse errors in generated program:\n{}", src);
        let e = format!("{}", cjc_eval::Interpreter::new(42).exec(&prog).unwrap());
        let (v, _) = cjc_mir_exec::run_program_with_executor(&prog, 42).unwrap();
        prop_assert_eq!(e, format!("{}", v));
    }
}
