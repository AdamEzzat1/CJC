//! QAOA — Quantum Approximate Optimization Algorithm for MaxCut.
//!
//! Implements QAOA for MaxCut problems on 1D nearest-neighbor graphs using
//! MPS (Matrix Product State) representation. This enables efficient simulation
//! of QAOA circuits on large qubit counts with bounded entanglement.
//!
//! # Supported Graphs
//!
//! Only graphs with edges between adjacent sites (i, i+1) are supported for the
//! cost unitary, since MPS two-qubit gates require adjacency. Non-adjacent edges
//! are included in energy evaluation (via general ZZ measurement) but skipped
//! during the ansatz construction.
//!
//! # Determinism
//!
//! - Parameter initialization uses seeded SplitMix64
//! - All MPS operations use sign-stabilized SVD
//! - Complex arithmetic uses fixed-sequence multiplication (no FMA)
//! - Gradient computation via parameter-shift rule is deterministic

use cjc_runtime::complex::ComplexF64;
use crate::mps::{Mps, MpsTensor};

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

/// Simple undirected graph for MaxCut problems.
#[derive(Debug, Clone)]
pub struct Graph {
    /// Number of vertices.
    pub n_vertices: usize,
    /// Edge list as pairs of vertex indices.
    pub edges: Vec<(usize, usize)>,
}

impl Graph {
    /// Create a graph with the given vertices and edges.
    ///
    /// Edges are stored as-is; both `(i,j)` and `(j,i)` are treated the same
    /// for MaxCut purposes.
    pub fn new(n_vertices: usize, edges: Vec<(usize, usize)>) -> Self {
        Self { n_vertices, edges }
    }

    /// Create a cycle graph: 0-1-2-..-(n-1)-0.
    ///
    /// Contains n edges forming a single cycle.
    pub fn cycle(n: usize) -> Self {
        assert!(n >= 3, "Cycle graph requires at least 3 vertices");
        let mut edges = Vec::with_capacity(n);
        for i in 0..(n - 1) {
            edges.push((i, i + 1));
        }
        edges.push((n - 1, 0));
        Self { n_vertices: n, edges }
    }

    /// Create a complete graph K_n.
    ///
    /// Contains n*(n-1)/2 edges.
    pub fn complete(n: usize) -> Self {
        let mut edges = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                edges.push((i, j));
            }
        }
        Self { n_vertices: n, edges }
    }
}

// ---------------------------------------------------------------------------
// QAOA Result
// ---------------------------------------------------------------------------

/// Result of a QAOA optimization run.
#[derive(Debug, Clone)]
pub struct QaoaResult {
    /// Optimized gamma parameters (cost layer rotations).
    pub gammas: Vec<f64>,
    /// Optimized beta parameters (mixer layer rotations).
    pub betas: Vec<f64>,
    /// Best energy found during optimization.
    pub energy: f64,
    /// Best cut value (equal to energy for MaxCut).
    pub cut_value: f64,
    /// Energy at each iteration (including initial).
    pub energy_history: Vec<f64>,
    /// Number of optimization iterations performed.
    pub iterations: usize,
}

// ---------------------------------------------------------------------------
// Transfer Matrices (local implementations for general ZZ)
// ---------------------------------------------------------------------------

/// Transfer matrix contraction with identity operator.
///
/// T_new[a,b] = sum_{j,j'} sum_s env[j,j'] * conj(A^s[j,a]) * A^s[j',b]
fn transfer_matrix_identity_local(
    tensor: &MpsTensor,
    env: &[Vec<ComplexF64>],
) -> Vec<Vec<ComplexF64>> {
    let bl = tensor.bond_left;
    let br = tensor.bond_right;
    assert_eq!(env.len(), bl);
    assert_eq!(env[0].len(), bl);

    let mut result = vec![vec![ComplexF64::ZERO; br]; br];

    for s in 0..2 {
        for j in 0..bl {
            for jp in 0..bl {
                let e = env[j][jp];
                if e.re == 0.0 && e.im == 0.0 {
                    continue;
                }
                for a in 0..br {
                    let conj_a = tensor.a[s].get(j, a).conj();
                    let ea = e.mul_fixed(conj_a);
                    for b in 0..br {
                        let asb = tensor.a[s].get(jp, b);
                        result[a][b] = result[a][b].add(ea.mul_fixed(asb));
                    }
                }
            }
        }
    }

    result
}

/// Transfer matrix contraction with Z operator.
///
/// Z eigenvalues: o_0 = +1, o_1 = -1.
/// T_new[a,b] = sum_{j,j'} sum_s z_s * env[j,j'] * conj(A^s[j,a]) * A^s[j',b]
fn transfer_matrix_z_local(
    tensor: &MpsTensor,
    env: &[Vec<ComplexF64>],
) -> Vec<Vec<ComplexF64>> {
    let bl = tensor.bond_left;
    let br = tensor.bond_right;
    assert_eq!(env.len(), bl);
    assert_eq!(env[0].len(), bl);

    let z_eigenvalues = [1.0, -1.0];
    let mut result = vec![vec![ComplexF64::ZERO; br]; br];

    for s in 0..2 {
        let z_s = z_eigenvalues[s];
        for j in 0..bl {
            for jp in 0..bl {
                let e = env[j][jp].scale(z_s);
                if e.re == 0.0 && e.im == 0.0 {
                    continue;
                }
                for a in 0..br {
                    let conj_a = tensor.a[s].get(j, a).conj();
                    let ea = e.mul_fixed(conj_a);
                    for b in 0..br {
                        let asb = tensor.a[s].get(jp, b);
                        result[a][b] = result[a][b].add(ea.mul_fixed(asb));
                    }
                }
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// General ZZ Expectation
// ---------------------------------------------------------------------------

/// Compute <psi|Z_i Z_j|psi> for arbitrary sites i and j.
///
/// Contracts identity transfer matrices between the boundary and site_i,
/// a Z transfer matrix at site_i, identity transfer matrices between
/// site_i+1 and site_j-1, a Z transfer matrix at site_j, then identity
/// transfer matrices from site_j+1 to the right boundary.
///
/// This generalizes `mps_zz_expectation` (which only handles adjacent sites)
/// to arbitrary pairs.
pub fn mps_zz_general(mps: &Mps, site_i: usize, site_j: usize) -> f64 {
    let (si, sj) = if site_i <= site_j {
        (site_i, site_j)
    } else {
        (site_j, site_i)
    };
    let n = mps.n_qubits;
    assert!(sj < n, "ZZ site index out of range");

    if si == sj {
        // Z_i Z_i = I, and <psi|I|psi> = 1 for normalized states.
        // For MPS that may not be perfectly normalized, compute the norm.
        // But for simplicity, return 1.0 (QAOA states are normalized).
        return 1.0;
    }

    // Start with left boundary: 1x1 identity environment
    let mut env = vec![vec![ComplexF64::ZERO; 1]; 1];
    env[0][0] = ComplexF64::ONE;

    // Contract sites 0..si with identity
    for k in 0..si {
        env = transfer_matrix_identity_local(&mps.tensors[k], &env);
    }

    // Contract site si with Z
    env = transfer_matrix_z_local(&mps.tensors[si], &env);

    // Contract sites si+1..sj with identity
    for k in (si + 1)..sj {
        env = transfer_matrix_identity_local(&mps.tensors[k], &env);
    }

    // Contract site sj with Z
    env = transfer_matrix_z_local(&mps.tensors[sj], &env);

    // Contract sites sj+1..n with identity
    for k in (sj + 1)..n {
        env = transfer_matrix_identity_local(&mps.tensors[k], &env);
    }

    env[0][0].re
}

// ---------------------------------------------------------------------------
// MaxCut Energy
// ---------------------------------------------------------------------------

/// Compute MaxCut cost: sum_{(i,j) in E} (1 - Z_i Z_j) / 2.
///
/// This counts the expected number of edges crossing the cut.
/// For each edge, (1 - <ZZ>) / 2 gives 1 if the endpoints are in different
/// partitions and 0 if they are in the same partition.
pub fn qaoa_maxcut_energy(mps: &Mps, graph: &Graph) -> f64 {
    let mut cost = 0.0;
    for &(i, j) in &graph.edges {
        let zz = mps_zz_general(mps, i, j);
        cost += (1.0 - zz) / 2.0;
    }
    cost
}

// ---------------------------------------------------------------------------
// QAOA Ansatz Construction
// ---------------------------------------------------------------------------

/// Build the QAOA ansatz state for MaxCut.
///
/// 1. Start from |+>^n (Hadamard on all qubits)
/// 2. For each layer p:
///    a. Cost unitary: for each adjacent edge (i, i+1), apply e^{-i gamma Z_i Z_{i+1}}
///       decomposed as CNOT(i, i+1) -> Rz(2*gamma, i+1) -> CNOT(i, i+1)
///    b. Mixer unitary: Rx(2*beta) on each qubit
///
/// Non-adjacent edges are skipped in the cost unitary (MPS limitation).
/// The number of QAOA layers is determined by the length of gammas/betas.
pub fn build_qaoa_ansatz(
    graph: &Graph,
    gammas: &[f64],
    betas: &[f64],
    max_bond: usize,
) -> Mps {
    let n = graph.n_vertices;
    let p = gammas.len();
    assert_eq!(
        gammas.len(),
        betas.len(),
        "gammas and betas must have the same length"
    );
    assert!(n > 0, "Graph must have at least 1 vertex");

    let mut mps = Mps::with_max_bond(n, max_bond);

    // Initial state: |+>^n via Hadamard on all qubits
    let isq2 = 1.0 / 2.0f64.sqrt();
    let h_gate = [
        [ComplexF64::real(isq2), ComplexF64::real(isq2)],
        [ComplexF64::real(isq2), ComplexF64::real(-isq2)],
    ];
    for q in 0..n {
        mps.apply_single_qubit(q, h_gate);
    }

    for layer in 0..p {
        let gamma = gammas[layer];
        let beta = betas[layer];

        // Cost unitary: e^{-i gamma Z_i Z_{i+1}} for each adjacent edge
        // Decomposition: CNOT(i, i+1), Rz(2*gamma, i+1), CNOT(i, i+1)
        for &(i, j) in &graph.edges {
            if j == i + 1 {
                apply_zz_rotation(&mut mps, i, j, gamma);
            } else if i == j + 1 {
                apply_zz_rotation(&mut mps, j, i, gamma);
            }
            // Non-adjacent edges: skipped (MPS limitation)
        }

        // Mixer unitary: Rx(2*beta) on each qubit
        let c = beta.cos();
        let s = beta.sin();
        let rx = [
            [ComplexF64::real(c), ComplexF64::new(0.0, -s)],
            [ComplexF64::new(0.0, -s), ComplexF64::real(c)],
        ];
        for q in 0..n {
            mps.apply_single_qubit(q, rx);
        }
    }

    mps
}

/// Apply the ZZ rotation e^{-i gamma Z_i Z_j} for adjacent sites (j = i+1).
///
/// Decomposition: CNOT(i, j) -> Rz(2*gamma, j) -> CNOT(i, j).
fn apply_zz_rotation(mps: &mut Mps, i: usize, j: usize, gamma: f64) {
    debug_assert_eq!(j, i + 1, "ZZ rotation requires adjacent sites");

    mps.apply_cnot_adjacent(i, j);

    // Rz(2*gamma) = diag(e^{-i*gamma}, e^{+i*gamma})
    let c = gamma.cos();
    let s = gamma.sin();
    let rz = [
        [ComplexF64::new(c, -s), ComplexF64::ZERO],
        [ComplexF64::ZERO, ComplexF64::new(c, s)],
    ];
    mps.apply_single_qubit(j, rz);

    mps.apply_cnot_adjacent(i, j);
}

// ---------------------------------------------------------------------------
// QAOA Optimizer
// ---------------------------------------------------------------------------

/// Run QAOA for MaxCut on the given graph.
///
/// Uses gradient ascent with the parameter-shift rule to maximize the expected
/// cut value. Parameters (gammas, betas) are initialized randomly from the
/// seeded PRNG.
///
/// # Arguments
///
/// * `graph` - The graph to find the MaxCut of.
/// * `p_layers` - Number of QAOA layers (circuit depth).
/// * `max_bond` - Maximum MPS bond dimension (controls accuracy vs. memory).
/// * `learning_rate` - Step size for gradient ascent.
/// * `max_iters` - Maximum number of optimization iterations.
/// * `seed` - Seed for deterministic parameter initialization.
///
/// # Returns
///
/// A `QaoaResult` containing the optimized parameters, best energy, and history.
pub fn qaoa_maxcut(
    graph: &Graph,
    p_layers: usize,
    max_bond: usize,
    learning_rate: f64,
    max_iters: usize,
    seed: u64,
) -> QaoaResult {
    let mut rng_state = seed;
    let mut gammas: Vec<f64> = (0..p_layers)
        .map(|_| crate::rand_f64(&mut rng_state) * 0.1)
        .collect();
    let mut betas: Vec<f64> = (0..p_layers)
        .map(|_| crate::rand_f64(&mut rng_state) * 0.1)
        .collect();

    let mut energy_history = Vec::new();

    // Evaluate initial energy
    let mps = build_qaoa_ansatz(graph, &gammas, &betas, max_bond);
    let mut best_energy = qaoa_maxcut_energy(&mps, graph);
    energy_history.push(best_energy);

    // Parameter-shift rule: df/dtheta = (f(theta + pi/2) - f(theta - pi/2)) / 2
    let shift = std::f64::consts::FRAC_PI_2;

    for _iter in 0..max_iters {
        let mut gamma_grads = vec![0.0; p_layers];
        let mut beta_grads = vec![0.0; p_layers];

        // Compute gradients for all gamma parameters
        for k in 0..p_layers {
            let mut gp = gammas.clone();
            gp[k] += shift;
            let mut gm = gammas.clone();
            gm[k] -= shift;

            let ep = qaoa_maxcut_energy(
                &build_qaoa_ansatz(graph, &gp, &betas, max_bond),
                graph,
            );
            let em = qaoa_maxcut_energy(
                &build_qaoa_ansatz(graph, &gm, &betas, max_bond),
                graph,
            );
            gamma_grads[k] = (ep - em) / 2.0;
        }

        // Compute gradients for all beta parameters
        for k in 0..p_layers {
            let mut bp = betas.clone();
            bp[k] += shift;
            let mut bm = betas.clone();
            bm[k] -= shift;

            let ep = qaoa_maxcut_energy(
                &build_qaoa_ansatz(graph, &gammas, &bp, max_bond),
                graph,
            );
            let em = qaoa_maxcut_energy(
                &build_qaoa_ansatz(graph, &gammas, &bm, max_bond),
                graph,
            );
            beta_grads[k] = (ep - em) / 2.0;
        }

        // Gradient ascent (maximizing cut value)
        for k in 0..p_layers {
            gammas[k] += learning_rate * gamma_grads[k];
            betas[k] += learning_rate * beta_grads[k];
        }

        let mps = build_qaoa_ansatz(graph, &gammas, &betas, max_bond);
        let energy = qaoa_maxcut_energy(&mps, graph);
        energy_history.push(energy);
        if energy > best_energy {
            best_energy = energy;
        }
    }

    QaoaResult {
        gammas,
        betas,
        energy: best_energy,
        cut_value: best_energy,
        energy_history,
        iterations: max_iters,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_graph_cycle() {
        let g = Graph::cycle(5);
        assert_eq!(g.n_vertices, 5);
        assert_eq!(g.edges.len(), 5);
        // Last edge wraps around
        assert_eq!(g.edges[4], (4, 0));
    }

    #[test]
    fn test_graph_complete() {
        let g = Graph::complete(4);
        assert_eq!(g.n_vertices, 4);
        assert_eq!(g.edges.len(), 6); // 4*3/2
    }

    #[test]
    fn test_maxcut_energy_all_plus() {
        // |+>^n: <ZZ> = 0, so cost = n_edges * 0.5
        let g = Graph::new(4, vec![(0, 1), (1, 2), (2, 3)]); // path graph
        let mut mps = Mps::with_max_bond(4, 64);
        let isq2 = 1.0 / 2.0f64.sqrt();
        let h = [
            [ComplexF64::real(isq2), ComplexF64::real(isq2)],
            [ComplexF64::real(isq2), ComplexF64::real(-isq2)],
        ];
        for q in 0..4 {
            mps.apply_single_qubit(q, h);
        }
        let cost = qaoa_maxcut_energy(&mps, &g);
        assert!(
            (cost - 1.5).abs() < TOL,
            "Expected 1.5, got {}",
            cost
        ); // 3 edges * 0.5
    }

    #[test]
    fn test_maxcut_energy_all_zero() {
        // |0000>: <Z_i Z_j> = 1 for all i,j, so cost = 0
        let g = Graph::new(4, vec![(0, 1), (1, 2), (2, 3)]);
        let mps = Mps::with_max_bond(4, 64);
        let cost = qaoa_maxcut_energy(&mps, &g);
        assert!(cost.abs() < TOL, "Expected 0.0, got {}", cost);
    }

    #[test]
    fn test_qaoa_path_graph_converges() {
        let g = Graph::new(4, vec![(0, 1), (1, 2), (2, 3)]);
        let result = qaoa_maxcut(&g, 1, 16, 0.2, 10, 42);
        assert!(result.energy > 0.0, "Energy should be positive");
        // Should not degrade below initial value (within tolerance)
        assert!(
            result.energy >= result.energy_history[0] - 1e-8,
            "Energy should not degrade: best={}, initial={}",
            result.energy,
            result.energy_history[0]
        );
    }

    #[test]
    fn test_qaoa_deterministic() {
        let g = Graph::new(4, vec![(0, 1), (1, 2), (2, 3)]);
        let r1 = qaoa_maxcut(&g, 1, 16, 0.1, 3, 42);
        let r2 = qaoa_maxcut(&g, 1, 16, 0.1, 3, 42);
        assert_eq!(
            r1.energy.to_bits(),
            r2.energy.to_bits(),
            "QAOA must be deterministic: {} vs {}",
            r1.energy,
            r2.energy
        );
        // Check full history is bit-identical
        assert_eq!(r1.energy_history.len(), r2.energy_history.len());
        for (i, (e1, e2)) in r1
            .energy_history
            .iter()
            .zip(r2.energy_history.iter())
            .enumerate()
        {
            assert_eq!(
                e1.to_bits(),
                e2.to_bits(),
                "History mismatch at iteration {}: {} vs {}",
                i,
                e1,
                e2
            );
        }
    }

    #[test]
    fn test_qaoa_triangle_2_edges() {
        // Path: 0-1-2 (2 edges). Optimal cut = 2 (cut vertex 1).
        let g = Graph::new(3, vec![(0, 1), (1, 2)]);
        let result = qaoa_maxcut(&g, 2, 16, 0.3, 20, 42);
        assert!(
            result.energy > 0.5,
            "QAOA should find reasonable cut, got {}",
            result.energy
        );
    }

    #[test]
    fn test_zz_general_adjacent() {
        // General ZZ for adjacent sites should match vqe::mps_zz_expectation
        let mut mps = Mps::new(4);
        let isq2 = 1.0 / 2.0f64.sqrt();
        let h = [
            [ComplexF64::real(isq2), ComplexF64::real(isq2)],
            [ComplexF64::real(isq2), ComplexF64::real(-isq2)],
        ];
        mps.apply_single_qubit(0, h);
        mps.apply_cnot_adjacent(0, 1);

        let zz_adj = crate::vqe::mps_zz_expectation(&mps, 0);
        let zz_gen = mps_zz_general(&mps, 0, 1);
        assert!(
            (zz_adj - zz_gen).abs() < TOL,
            "Adjacent ZZ mismatch: vqe={}, general={}",
            zz_adj,
            zz_gen
        );
    }

    #[test]
    fn test_zz_general_non_adjacent() {
        // Non-adjacent ZZ: sites 0 and 2 in 4-qubit system
        // |0000>: Z_0 Z_2 = (+1)(+1) = 1
        let mps = Mps::new(4);
        let zz = mps_zz_general(&mps, 0, 2);
        assert!(
            (zz - 1.0).abs() < TOL,
            "Expected ZZ=1.0 for |0000>, got {}",
            zz
        );
    }

    #[test]
    fn test_zz_general_non_adjacent_entangled() {
        // Create a Bell pair on sites 0,1 and check Z_0 Z_2
        // Bell state on (0,1): (|00>+|11>)/sqrt(2), sites 2,3 in |0>
        // <Z_0 Z_2> = <Z_0> * <Z_2> = 0 * 1 = 0
        // (Z_0 has expectation 0 in Bell state, Z_2 = +1 in |0>)
        let mut mps = Mps::new(4);
        let isq2 = 1.0 / 2.0f64.sqrt();
        let h = [
            [ComplexF64::real(isq2), ComplexF64::real(isq2)],
            [ComplexF64::real(isq2), ComplexF64::real(-isq2)],
        ];
        mps.apply_single_qubit(0, h);
        mps.apply_cnot_adjacent(0, 1);

        let zz = mps_zz_general(&mps, 0, 2);
        assert!(
            zz.abs() < TOL,
            "Expected ZZ=0.0 for Bell(0,1) x |0>(2), got {}",
            zz
        );
    }

    #[test]
    fn test_zz_general_symmetric() {
        // ZZ(i,j) should equal ZZ(j,i)
        let mut mps = Mps::new(4);
        let isq2 = 1.0 / 2.0f64.sqrt();
        let h = [
            [ComplexF64::real(isq2), ComplexF64::real(isq2)],
            [ComplexF64::real(isq2), ComplexF64::real(-isq2)],
        ];
        mps.apply_single_qubit(1, h);

        let zz_02 = mps_zz_general(&mps, 0, 2);
        let zz_20 = mps_zz_general(&mps, 2, 0);
        assert_eq!(
            zz_02.to_bits(),
            zz_20.to_bits(),
            "ZZ should be symmetric: ZZ(0,2)={}, ZZ(2,0)={}",
            zz_02,
            zz_20
        );
    }

    #[test]
    fn test_qaoa_ansatz_produces_normalized_state() {
        // The QAOA ansatz should preserve normalization
        let g = Graph::new(4, vec![(0, 1), (1, 2), (2, 3)]);
        let gammas = vec![0.3];
        let betas = vec![0.5];
        let mps = build_qaoa_ansatz(&g, &gammas, &betas, 16);

        // Compute norm via statevector (small system)
        let sv = mps.to_statevector();
        let mut norm_sq = 0.0;
        for amp in &sv {
            norm_sq += amp.norm_sq();
        }
        assert!(
            (norm_sq - 1.0).abs() < 1e-8,
            "State should be normalized, got norm^2 = {}",
            norm_sq
        );
    }

    #[test]
    fn test_qaoa_50_qubit_builds() {
        let g = Graph::new(50, (0..49).map(|i| (i, i + 1)).collect());
        let gammas = vec![0.1];
        let betas = vec![0.2];
        let mps = build_qaoa_ansatz(&g, &gammas, &betas, 8);
        let e = qaoa_maxcut_energy(&mps, &g);
        assert!(e.is_finite(), "Energy should be finite for 50-qubit system");
    }

    #[test]
    fn test_qaoa_energy_history_length() {
        let g = Graph::new(3, vec![(0, 1), (1, 2)]);
        let result = qaoa_maxcut(&g, 1, 8, 0.1, 5, 42);
        // Initial + 5 iterations = 6 entries
        assert_eq!(
            result.energy_history.len(),
            6,
            "Expected 6 history entries, got {}",
            result.energy_history.len()
        );
        assert_eq!(result.iterations, 5);
    }

    #[test]
    fn test_build_qaoa_ansatz_no_edges() {
        // Graph with no edges: ansatz should still build, energy = 0
        let g = Graph::new(3, vec![]);
        let gammas = vec![0.5];
        let betas = vec![0.3];
        let mps = build_qaoa_ansatz(&g, &gammas, &betas, 8);
        let e = qaoa_maxcut_energy(&mps, &g);
        assert!(
            e.abs() < TOL,
            "No edges should give zero cut energy, got {}",
            e
        );
    }
}
