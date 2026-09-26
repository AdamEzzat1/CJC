//! Canonical seeded workload generator (BENCHMARK_PLAN §3).
//!
//! One gate list per case, emitted as `.cjcl` source, OpenQASM 2.0 (via
//! `cjc_quantum::qasm`, the same code users call), and Stim text for Clifford
//! cases. Every form comes from the same list, so they describe the same
//! circuit by construction; the runner additionally checks that the Rust
//! (QASM-imported) path and the `.cjcl` paths produce identical output bits.

use cjc_quantum::circuit::Circuit;
use cjc_quantum::gates::Gate;

/// SplitMix64, the constants used across CJC (`cjc_repro::Rng`).
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1) with 53 random bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    /// Fisher–Yates permutation of 0..n.
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
        v
    }
}

/// Which simulator family (and therefore which observable) a case uses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Family {
    /// Dense statevector; observe the probability vector and 1,000 shots.
    Dense,
    /// Stabilizer; observe peek_z for every qubit (Rust) and a seeded
    /// measure-all record (all paths).
    Clifford,
    /// MPS with bond cap `chi`; observe ⟨Z_i⟩ for every qubit.
    Mps { chi: usize },
}

#[derive(Clone, Debug)]
pub struct Case {
    pub id: String,
    pub workload: &'static str,
    pub family: Family,
    pub n: usize,
    pub depth: usize,
    pub seed: u64,
    pub gates: Vec<Gate>,
    /// Dense only: the `.cjcl` program also calls `q_sample` and `q_measure`
    /// on the same circuit (W6), the pattern the execution cache targets.
    pub multi_observe: bool,
}

pub const SHOTS: usize = 1000;
pub const SAMPLE_SEED: u64 = 7;

// ---------------------------------------------------------------------------
// Workloads
// ---------------------------------------------------------------------------

/// W1: GHZ via H(0), CX(0, k).
pub fn w1_ghz(n: usize) -> Case {
    let mut g = vec![Gate::H(0)];
    g.extend((1..n).map(|k| Gate::CNOT(0, k)));
    Case { id: format!("W1_n{}", n), workload: "W1_ghz_dense", family: Family::Dense, n, depth: 1, seed: 0, gates: g, multi_observe: false }
}

/// Random perfect matching of 0..n (the last qubit idles when n is odd).
fn matching(rng: &mut SplitMix64, n: usize) -> Vec<(usize, usize)> {
    let p = rng.permutation(n);
    p.chunks_exact(2).map(|c| (c[0], c[1])).collect()
}

/// W2: per layer, a random gate from {H, S, T, Rx, Ry, Rz} on every qubit,
/// then CX on a random perfect matching.
pub fn w2_random(n: usize, depth: usize, seed: u64) -> Case {
    let mut rng = SplitMix64(seed ^ 0x5732_0000);
    let mut g = Vec::new();
    for _ in 0..depth {
        for q in 0..n {
            let angle = rng.next_f64() * std::f64::consts::TAU;
            g.push(match rng.below(6) {
                0 => Gate::H(q),
                1 => Gate::S(q),
                2 => Gate::T(q),
                3 => Gate::Rx(q, angle),
                4 => Gate::Ry(q, angle),
                _ => Gate::Rz(q, angle),
            });
        }
        for (a, b) in matching(&mut rng, n) {
            g.push(Gate::CNOT(a, b));
        }
    }
    Case {
        id: format!("W2_n{}_d{}_s{}", n, depth, seed),
        workload: "W2_dense_random",
        family: Family::Dense,
        n,
        depth,
        seed,
        gates: g,
        multi_observe: false,
    }
}

/// W6: a W2 circuit observed three times from `.cjcl` (`q_probs`, then
/// `q_sample` and `q_measure` on the same circuit value).
pub fn w6_multi_observe(n: usize, depth: usize, seed: u64) -> Case {
    let mut c = w2_random(n, depth, seed);
    c.id = format!("W6_n{}_d{}_s{}", n, depth, seed);
    c.workload = "W6_multi_observe";
    c.multi_observe = true;
    c
}

/// W3: per layer, a random Clifford from {H, S, X, Y, Z} on every qubit, then
/// CX on a random perfect matching.
pub fn w3_clifford(n: usize, depth: usize, seed: u64) -> Case {
    let mut rng = SplitMix64(seed ^ 0x5733_0000);
    let mut g = Vec::new();
    for _ in 0..depth {
        for q in 0..n {
            g.push(match rng.below(5) {
                0 => Gate::H(q),
                1 => Gate::S(q),
                2 => Gate::X(q),
                3 => Gate::Y(q),
                _ => Gate::Z(q),
            });
        }
        for (a, b) in matching(&mut rng, n) {
            g.push(Gate::CNOT(a, b));
        }
    }
    Case {
        id: format!("W3_n{}_d{}_s{}", n, depth, seed),
        workload: "W3_clifford",
        family: Family::Clifford,
        n,
        depth,
        seed,
        gates: g,
        multi_observe: false,
    }
}

/// W4a: GHZ chain with adjacent CNOTs (bond dimension 2 is exact).
pub fn w4_ghz_chain(n: usize, chi: usize) -> Case {
    let mut g = vec![Gate::H(0)];
    g.extend((1..n).map(|k| Gate::CNOT(k - 1, k)));
    Case {
        id: format!("W4ghz_n{}_chi{}", n, chi),
        workload: "W4_mps_ghz",
        family: Family::Mps { chi },
        n,
        depth: 1,
        seed: 0,
        gates: g,
        multi_observe: false,
    }
}

/// W4b: brickwork of Ry(random) on every qubit, then adjacent CNOTs on even
/// pairs (even layers) or odd pairs (odd layers).
pub fn w4_brickwork(n: usize, depth: usize, chi: usize, seed: u64) -> Case {
    let mut rng = SplitMix64(seed ^ 0x5734_0000);
    let mut g = Vec::new();
    for layer in 0..depth {
        for q in 0..n {
            g.push(Gate::Ry(q, rng.next_f64() * std::f64::consts::TAU));
        }
        let mut a = layer % 2;
        while a + 1 < n {
            g.push(Gate::CNOT(a, a + 1));
            a += 2;
        }
    }
    Case {
        id: format!("W4brick_n{}_d{}_chi{}_s{}", n, depth, chi, seed),
        workload: "W4_mps_brickwork",
        family: Family::Mps { chi },
        n,
        depth,
        seed,
        gates: g,
        multi_observe: false,
    }
}

// ---------------------------------------------------------------------------
// Emitters
// ---------------------------------------------------------------------------

pub fn to_circuit(case: &Case) -> Circuit {
    let mut c = Circuit::new(case.n);
    for g in &case.gates {
        c.add(g.clone());
    }
    c
}

pub fn to_qasm(case: &Case) -> String {
    cjc_quantum::qasm::to_qasm(&to_circuit(case))
}

/// Stim text for Clifford cases (H S X Y Z CX only).
pub fn to_stim(case: &Case) -> Option<String> {
    let mut s = String::new();
    for g in &case.gates {
        let line = match *g {
            Gate::H(q) => format!("H {}", q),
            Gate::S(q) => format!("S {}", q),
            Gate::X(q) => format!("X {}", q),
            Gate::Y(q) => format!("Y {}", q),
            Gate::Z(q) => format!("Z {}", q),
            Gate::CNOT(a, b) => format!("CX {} {}", a, b),
            _ => return None,
        };
        s.push_str(&line);
        s.push('\n');
    }
    Some(s)
}

fn fmt_f64(x: f64) -> String {
    // Shortest round-trip form; CJC float literals need a '.' or exponent.
    let s = format!("{:?}", x);
    if s.contains('.') {
        s
    } else if let Some(i) = s.find('e') {
        format!("{}.0{}", &s[..i], &s[i..])
    } else {
        format!("{}.0", s)
    }
}

/// `.cjcl` program whose `main` returns the case's observable.
pub fn to_cjcl(case: &Case) -> String {
    let mut s = String::from("fn main() -> Any {\n");
    match case.family {
        Family::Dense => {
            s.push_str(&format!("    let c = qubits({});\n", case.n));
            for g in &case.gates {
                s.push_str("    c = ");
                s.push_str(&match *g {
                    Gate::H(q) => format!("q_h(c, {})", q),
                    Gate::X(q) => format!("q_x(c, {})", q),
                    Gate::Y(q) => format!("q_y(c, {})", q),
                    Gate::Z(q) => format!("q_z(c, {})", q),
                    Gate::S(q) => format!("q_s(c, {})", q),
                    Gate::T(q) => format!("q_t(c, {})", q),
                    Gate::Rx(q, t) => format!("q_rx(c, {}, {})", q, fmt_f64(t)),
                    Gate::Ry(q, t) => format!("q_ry(c, {}, {})", q, fmt_f64(t)),
                    Gate::Rz(q, t) => format!("q_rz(c, {}, {})", q, fmt_f64(t)),
                    Gate::CNOT(a, b) => format!("q_cx(c, {}, {})", a, b),
                    Gate::CZ(a, b) => format!("q_cz(c, {}, {})", a, b),
                    Gate::SWAP(a, b) => format!("q_swap(c, {}, {})", a, b),
                    Gate::Toffoli(a, b, t) => format!("q_toffoli(c, {}, {}, {})", a, b, t),
                });
                s.push_str(";\n");
            }
            if case.multi_observe {
                s.push_str(&format!(
                    "    let p = q_probs(c);\n    let shots = q_sample(c, {}, {});\n    let m = q_measure(c, 3);\n    p\n",
                    SHOTS, SAMPLE_SEED
                ));
            } else {
                s.push_str("    q_probs(c)\n");
            }
        }
        Family::Clifford => {
            s.push_str(&format!("    let s = stabilizer_new({});\n", case.n));
            for g in &case.gates {
                s.push_str("    s = ");
                s.push_str(&match *g {
                    Gate::H(q) => format!("stabilizer_h(s, {})", q),
                    Gate::S(q) => format!("stabilizer_s(s, {})", q),
                    Gate::X(q) => format!("stabilizer_x(s, {})", q),
                    Gate::Y(q) => format!("stabilizer_y(s, {})", q),
                    Gate::Z(q) => format!("stabilizer_z(s, {})", q),
                    Gate::CNOT(a, b) => format!("stabilizer_cnot(s, {}, {})", a, b),
                    _ => unreachable!("non-Clifford gate in a Clifford case"),
                });
                s.push_str(";\n");
            }
            // Measure qubit q with seed q, in order (mirrors the Rust path).
            s.push_str(&format!(
                "    let out = [];\n    let q = 0;\n    while q < {} {{\n        out = array_push(out, stabilizer_measure(s, q, q));\n        q = q + 1;\n    }}\n    out\n",
                case.n
            ));
        }
        Family::Mps { chi } => {
            s.push_str(&format!("    let m = mps_new({}, {});\n", case.n, chi));
            for g in &case.gates {
                s.push_str("    m = ");
                s.push_str(&match *g {
                    Gate::H(q) => format!("mps_h(m, {})", q),
                    Gate::X(q) => format!("mps_x(m, {})", q),
                    Gate::Ry(q, t) => format!("mps_ry(m, {}, {})", q, fmt_f64(t)),
                    Gate::CNOT(a, b) => format!("mps_cnot(m, {}, {})", a, b),
                    _ => unreachable!("gate not expressible on the MPS surface"),
                });
                s.push_str(";\n");
            }
            s.push_str(&format!(
                "    let out = [];\n    let q = 0;\n    while q < {} {{\n        out = array_push(out, mps_z_expectation(m, q));\n        q = q + 1;\n    }}\n    out\n",
                case.n
            ));
        }
    }
    s.push_str("}\n");
    s
}

// ---------------------------------------------------------------------------
// Suites
// ---------------------------------------------------------------------------

/// Tiny cases for the harness's own tests.
pub fn suite_smoke() -> Vec<Case> {
    vec![
        w1_ghz(4),
        w2_random(5, 3, 1),
        w3_clifford(12, 6, 1),
        w4_ghz_chain(8, 4),
        w4_brickwork(6, 2, 8, 1),
    ]
}

/// First real comparison run (BENCHMARK_PLAN §9 steps 1–2: W1–W4).
pub fn suite_baseline() -> Vec<Case> {
    let mut v = Vec::new();
    for n in [2, 4, 8, 12, 16, 20, 22, 24, 26] {
        v.push(w1_ghz(n));
    }
    for n in [16, 18, 20, 22] {
        for depth in [10, 20] {
            v.push(w2_random(n, depth, 1));
        }
    }
    for n in [100, 250, 500, 1000] {
        v.push(w3_clifford(n, n, 1));
    }
    for n in [16, 18, 20] {
        v.push(w6_multi_observe(n, 10, 1));
    }
    for n in [50, 100, 250, 500, 1000] {
        v.push(w4_ghz_chain(n, 16));
    }
    for n in [50, 100] {
        for depth in [2, 4, 8] {
            for chi in [16, 32] {
                v.push(w4_brickwork(n, depth, chi, 1));
            }
        }
    }
    v
}
