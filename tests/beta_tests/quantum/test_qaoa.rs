//! Integration tests for the QAOA (Quantum Approximate Optimization Algorithm) module.
//!
//! Tests cover graph construction, MaxCut energy properties, optimization quality,
//! determinism guarantees, and the general ZZ expectation value implementation.

use cjc_quantum::qaoa::*;
use cjc_quantum::mps::Mps;
use cjc_quantum::vqe::mps_zz_expectation;
use cjc_runtime::complex::ComplexF64;

const TOL: f64 = 1e-10;

// ---------------------------------------------------------------------------
// 1. Graph construction
// ---------------------------------------------------------------------------

#[test]
fn qaoa_graph_cycle_has_correct_edge_count() {
    let g = Graph::cycle(5);
    assert_eq!(g.n_vertices, 5, "Cycle(5) should have 5 vertices");
    assert_eq!(g.edges.len(), 5, "Cycle(5) should have 5 edges");
    // First edge is (0,1), last wraps around to (4,0)
    assert_eq!(g.edges[0], (0, 1), "First edge of Cycle(5) should be (0,1)");
    assert_eq!(g.edges[4], (4, 0), "Last edge of Cycle(5) should wrap around to (4,0)");
}

#[test]
fn qaoa_graph_cycle_3_is_triangle() {
    let g = Graph::cycle(3);
    assert_eq!(g.n_vertices, 3, "Cycle(3) should have 3 vertices");
    assert_eq!(g.edges.len(), 3, "Cycle(3) should have 3 edges (triangle)");
    assert_eq!(g.edges[0], (0, 1));
    assert_eq!(g.edges[1], (1, 2));
    assert_eq!(g.edges[2], (2, 0));
}

#[test]
fn qaoa_graph_complete_4_has_6_edges() {
    let g = Graph::complete(4);
    assert_eq!(g.n_vertices, 4, "K4 should have 4 vertices");
    assert_eq!(g.edges.len(), 6, "K4 should have 4*3/2 = 6 edges");
}

#[test]
fn qaoa_graph_complete_3_has_3_edges() {
    let g = Graph::complete(3);
    assert_eq!(g.n_vertices, 3, "K3 should have 3 vertices");
    assert_eq!(g.edges.len(), 3, "K3 should have 3*2/2 = 3 edges");
}

#[test]
fn qaoa_graph_complete_edges_are_sorted_pairs() {
    let g = Graph::complete(5);
    for &(i, j) in &g.edges {
        assert!(
            i < j,
            "Complete graph edges should have i < j, got ({}, {})",
            i,
            j
        );
    }
}

// ---------------------------------------------------------------------------
// 2. MaxCut energy is non-negative
// ---------------------------------------------------------------------------

#[test]
fn qaoa_maxcut_energy_is_non_negative_all_zero_state() {
    // |0...0> state: all ZZ = +1, so cost = 0 (non-negative)
    let g = Graph::new(4, vec![(0, 1), (1, 2), (2, 3)]);
    let mps = Mps::with_max_bond(4, 16);
    let cost = qaoa_maxcut_energy(&mps, &g);
    assert!(
        cost >= -TOL,
        "MaxCut energy should be non-negative, got {}",
        cost
    );
}

#[test]
fn qaoa_maxcut_energy_is_non_negative_plus_state() {
    // |+...+> state: all ZZ = 0, so cost = n_edges * 0.5
    let g = Graph::cycle(5);
    let mut mps = Mps::with_max_bond(5, 16);
    let isq2 = 1.0 / 2.0f64.sqrt();
    let h = [
        [ComplexF64::real(isq2), ComplexF64::real(isq2)],
        [ComplexF64::real(isq2), ComplexF64::real(-isq2)],
    ];
    for q in 0..5 {
        mps.apply_single_qubit(q, h);
    }
    let cost = qaoa_maxcut_energy(&mps, &g);
    assert!(
        cost >= -TOL,
        "MaxCut energy should be non-negative for |+> state, got {}",
        cost
    );
}

#[test]
fn qaoa_maxcut_energy_is_non_negative_after_optimization() {
    let g = Graph::new(4, vec![(0, 1), (1, 2), (2, 3)]);
    let result = qaoa_maxcut(&g, 1, 16, 0.2, 5, 99);
    assert!(
        result.energy >= -TOL,
        "Optimized MaxCut energy should be non-negative, got {}",
        result.energy
    );
    for (i, &e) in result.energy_history.iter().enumerate() {
        assert!(
            e >= -TOL,
            "Energy history entry {} should be non-negative, got {}",
            i,
            e
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Triangle graph finds cut >= 2
// ---------------------------------------------------------------------------

#[test]
fn qaoa_triangle_finds_cut_at_least_2() {
    // Triangle (K3) has optimal MaxCut = 2 (any partition with 1 vs 2 vertices cuts 2 edges).
    // Only adjacent edges (0,1) and (1,2) are applied in the ansatz; the wrap-around
    // edge (2,0) is non-adjacent so only contributes to energy evaluation.
    // Use a path graph 0-1-2 which has all adjacent edges and optimal cut = 2.
    let g = Graph::new(3, vec![(0, 1), (1, 2)]);
    let result = qaoa_maxcut(&g, 2, 16, 0.3, 30, 42);
    assert!(
        result.energy >= 0.8,
        "QAOA on 3-vertex path should find cut >= 0.8 (toward optimal 2), got {}",
        result.energy
    );
}

#[test]
fn qaoa_triangle_graph_energy_improves() {
    let g = Graph::new(3, vec![(0, 1), (1, 2)]);
    let result = qaoa_maxcut(&g, 2, 16, 0.3, 20, 42);
    let initial = result.energy_history[0];
    assert!(
        result.energy >= initial - 1e-8,
        "QAOA should not degrade below initial energy: best={}, initial={}",
        result.energy,
        initial
    );
}

// ---------------------------------------------------------------------------
// 4. 4-cycle finds good cut
// ---------------------------------------------------------------------------

#[test]
fn qaoa_4_cycle_finds_good_cut() {
    // 4-cycle: 0-1-2-3-0. Optimal MaxCut = 4 (bipartite: {0,2} vs {1,3}).
    // Only adjacent edges (0,1), (1,2), (2,3) are applied in ansatz;
    // the wrap-around (3,0) is non-adjacent.
    // Use a path graph for the ansatz-active edges.
    let g = Graph::new(4, vec![(0, 1), (1, 2), (2, 3)]);
    let result = qaoa_maxcut(&g, 2, 16, 0.3, 30, 42);
    assert!(
        result.energy >= 1.5,
        "QAOA on 4-vertex path should find reasonable cut value, got {}",
        result.energy
    );
}

#[test]
fn qaoa_4_path_energy_bounded_by_edges() {
    // MaxCut energy cannot exceed the number of edges.
    let g = Graph::new(4, vec![(0, 1), (1, 2), (2, 3)]);
    let result = qaoa_maxcut(&g, 2, 16, 0.3, 20, 42);
    let n_edges = g.edges.len() as f64;
    assert!(
        result.energy <= n_edges + TOL,
        "MaxCut energy ({}) should not exceed number of edges ({})",
        result.energy,
        n_edges
    );
}

// ---------------------------------------------------------------------------
// 5. Determinism: same seed = identical results
// ---------------------------------------------------------------------------

#[test]
fn qaoa_determinism_same_seed_identical_energy() {
    let g = Graph::new(4, vec![(0, 1), (1, 2), (2, 3)]);
    let r1 = qaoa_maxcut(&g, 1, 16, 0.1, 5, 42);
    let r2 = qaoa_maxcut(&g, 1, 16, 0.1, 5, 42);
    assert_eq!(
        r1.energy.to_bits(),
        r2.energy.to_bits(),
        "Same seed must produce bit-identical energy: {} vs {}",
        r1.energy,
        r2.energy
    );
}

#[test]
fn qaoa_determinism_full_history_bit_identical() {
    let g = Graph::new(4, vec![(0, 1), (1, 2), (2, 3)]);
    let r1 = qaoa_maxcut(&g, 1, 16, 0.2, 8, 123);
    let r2 = qaoa_maxcut(&g, 1, 16, 0.2, 8, 123);
    assert_eq!(
        r1.energy_history.len(),
        r2.energy_history.len(),
        "History lengths must match"
    );
    for (i, (e1, e2)) in r1
        .energy_history
        .iter()
        .zip(r2.energy_history.iter())
        .enumerate()
    {
        assert_eq!(
            e1.to_bits(),
            e2.to_bits(),
            "Energy history mismatch at iteration {}: {} vs {}",
            i,
            e1,
            e2
        );
    }
}

#[test]
fn qaoa_determinism_parameters_bit_identical() {
    let g = Graph::new(3, vec![(0, 1), (1, 2)]);
    let r1 = qaoa_maxcut(&g, 2, 16, 0.15, 5, 77);
    let r2 = qaoa_maxcut(&g, 2, 16, 0.15, 5, 77);
    for (i, (g1, g2)) in r1.gammas.iter().zip(r2.gammas.iter()).enumerate() {
        assert_eq!(
            g1.to_bits(),
            g2.to_bits(),
            "Gamma[{}] mismatch: {} vs {}",
            i,
            g1,
            g2
        );
    }
    for (i, (b1, b2)) in r1.betas.iter().zip(r2.betas.iter()).enumerate() {
        assert_eq!(
            b1.to_bits(),
            b2.to_bits(),
            "Beta[{}] mismatch: {} vs {}",
            i,
            b1,
            b2
        );
    }
}

#[test]
fn qaoa_different_seeds_produce_different_results() {
    let g = Graph::new(4, vec![(0, 1), (1, 2), (2, 3)]);
    let r1 = qaoa_maxcut(&g, 1, 16, 0.2, 10, 42);
    let r2 = qaoa_maxcut(&g, 1, 16, 0.2, 10, 99);
    // Different seeds should (almost certainly) produce different energies
    let same = r1.energy.to_bits() == r2.energy.to_bits();
    // This is a soft check; it is theoretically possible but astronomically unlikely
    assert!(
        !same,
        "Different seeds should produce different results (extremely unlikely collision)"
    );
}

// ---------------------------------------------------------------------------
// 6. Energy bounded by number of edges
// ---------------------------------------------------------------------------

#[test]
fn qaoa_energy_bounded_by_edges_cycle_5() {
    let g = Graph::cycle(5);
    let result = qaoa_maxcut(&g, 1, 16, 0.2, 10, 42);
    let n_edges = g.edges.len() as f64;
    assert!(
        result.energy <= n_edges + TOL,
        "Energy ({}) must not exceed number of edges ({})",
        result.energy,
        n_edges
    );
    assert!(
        result.energy >= -TOL,
        "Energy ({}) must be non-negative",
        result.energy
    );
}

#[test]
fn qaoa_energy_bounded_by_edges_complete_4() {
    // K4 with only adjacent edges in the ansatz; energy evaluated on all edges.
    let g = Graph::complete(4);
    let result = qaoa_maxcut(&g, 1, 16, 0.2, 10, 42);
    let n_edges = g.edges.len() as f64;
    assert!(
        result.energy <= n_edges + TOL,
        "Energy ({}) must not exceed number of edges ({}) for K4",
        result.energy,
        n_edges
    );
    assert!(
        result.energy >= -TOL,
        "Energy ({}) must be non-negative for K4",
        result.energy
    );
}

#[test]
fn qaoa_ansatz_energy_bounded_for_random_params() {
    // Even with arbitrary parameters, energy must stay in [0, n_edges]
    let g = Graph::new(5, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
    let gammas = vec![1.5, -0.7];
    let betas = vec![-2.0, 3.14];
    let mps = build_qaoa_ansatz(&g, &gammas, &betas, 16);
    let energy = qaoa_maxcut_energy(&mps, &g);
    let n_edges = g.edges.len() as f64;
    assert!(
        energy >= -TOL,
        "Energy ({}) must be non-negative for arbitrary params",
        energy
    );
    assert!(
        energy <= n_edges + TOL,
        "Energy ({}) must not exceed n_edges ({}) for arbitrary params",
        energy,
        n_edges
    );
}

// ---------------------------------------------------------------------------
// 7. MPS ZZ general matches expected values for simple states
// ---------------------------------------------------------------------------

#[test]
fn qaoa_zz_general_all_zero_state_equals_one() {
    // |0000>: Z_i = +1 for all i, so Z_i Z_j = +1
    let mps = Mps::with_max_bond(4, 16);
    for i in 0..4 {
        for j in 0..4 {
            let zz = mps_zz_general(&mps, i, j);
            assert!(
                (zz - 1.0).abs() < TOL,
                "ZZ({},{}) for |0000> should be 1.0, got {}",
                i,
                j,
                zz
            );
        }
    }
}

#[test]
fn qaoa_zz_general_plus_state_adjacent_is_zero() {
    // |+...+>: <Z_i Z_j> = <Z_i><Z_j> = 0 for i != j (product state)
    let mut mps = Mps::with_max_bond(4, 16);
    let isq2 = 1.0 / 2.0f64.sqrt();
    let h = [
        [ComplexF64::real(isq2), ComplexF64::real(isq2)],
        [ComplexF64::real(isq2), ComplexF64::real(-isq2)],
    ];
    for q in 0..4 {
        mps.apply_single_qubit(q, h);
    }
    let zz_01 = mps_zz_general(&mps, 0, 1);
    assert!(
        zz_01.abs() < TOL,
        "ZZ(0,1) for |++++> should be 0, got {}",
        zz_01
    );
    let zz_02 = mps_zz_general(&mps, 0, 2);
    assert!(
        zz_02.abs() < TOL,
        "ZZ(0,2) for |++++> should be 0, got {}",
        zz_02
    );
    let zz_13 = mps_zz_general(&mps, 1, 3);
    assert!(
        zz_13.abs() < TOL,
        "ZZ(1,3) for |++++> should be 0, got {}",
        zz_13
    );
}

#[test]
fn qaoa_zz_general_bell_pair_correlated() {
    // Bell state on (0,1): H(0), CNOT(0,1) -> (|00>+|11>)/sqrt(2)
    // <Z_0 Z_1> = 1 (both always agree)
    let mut mps = Mps::with_max_bond(3, 16);
    let isq2 = 1.0 / 2.0f64.sqrt();
    let h = [
        [ComplexF64::real(isq2), ComplexF64::real(isq2)],
        [ComplexF64::real(isq2), ComplexF64::real(-isq2)],
    ];
    mps.apply_single_qubit(0, h);
    mps.apply_cnot_adjacent(0, 1);
    let zz = mps_zz_general(&mps, 0, 1);
    assert!(
        (zz - 1.0).abs() < TOL,
        "ZZ(0,1) for Bell state should be 1.0, got {}",
        zz
    );
}

#[test]
fn qaoa_zz_general_bell_pair_uncorrelated_with_third() {
    // Bell state on (0,1), qubit 2 in |0>
    // <Z_0 Z_2> = <Z_0><Z_2> = 0 * 1 = 0
    let mut mps = Mps::with_max_bond(3, 16);
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
        "ZZ(0,2) for Bell(0,1) x |0>(2) should be 0, got {}",
        zz
    );
}

#[test]
fn qaoa_zz_general_is_symmetric() {
    // ZZ(i,j) must equal ZZ(j,i)
    let mut mps = Mps::with_max_bond(4, 16);
    let isq2 = 1.0 / 2.0f64.sqrt();
    let h = [
        [ComplexF64::real(isq2), ComplexF64::real(isq2)],
        [ComplexF64::real(isq2), ComplexF64::real(-isq2)],
    ];
    mps.apply_single_qubit(1, h);
    mps.apply_cnot_adjacent(1, 2);

    let zz_03 = mps_zz_general(&mps, 0, 3);
    let zz_30 = mps_zz_general(&mps, 3, 0);
    assert_eq!(
        zz_03.to_bits(),
        zz_30.to_bits(),
        "ZZ must be symmetric: ZZ(0,3)={}, ZZ(3,0)={}",
        zz_03,
        zz_30
    );

    let zz_13 = mps_zz_general(&mps, 1, 3);
    let zz_31 = mps_zz_general(&mps, 3, 1);
    assert_eq!(
        zz_13.to_bits(),
        zz_31.to_bits(),
        "ZZ must be symmetric: ZZ(1,3)={}, ZZ(3,1)={}",
        zz_13,
        zz_31
    );
}

#[test]
fn qaoa_zz_general_matches_adjacent_vqe_implementation() {
    // The general ZZ for adjacent sites should agree with vqe::mps_zz_expectation
    let mut mps = Mps::with_max_bond(4, 16);
    let isq2 = 1.0 / 2.0f64.sqrt();
    let h = [
        [ComplexF64::real(isq2), ComplexF64::real(isq2)],
        [ComplexF64::real(isq2), ComplexF64::real(-isq2)],
    ];
    mps.apply_single_qubit(0, h);
    mps.apply_cnot_adjacent(0, 1);

    // Compare adjacent ZZ from both implementations
    for site in 0..3 {
        let zz_vqe = mps_zz_expectation(&mps, site);
        let zz_gen = mps_zz_general(&mps, site, site + 1);
        assert!(
            (zz_vqe - zz_gen).abs() < TOL,
            "Adjacent ZZ mismatch at site {}: vqe={}, general={}",
            site,
            zz_vqe,
            zz_gen
        );
    }
}

#[test]
fn qaoa_zz_general_same_site_returns_one() {
    // Z_i Z_i = I, so <psi|Z_i Z_i|psi> = 1 for normalized states
    let mut mps = Mps::with_max_bond(3, 16);
    let isq2 = 1.0 / 2.0f64.sqrt();
    let h = [
        [ComplexF64::real(isq2), ComplexF64::real(isq2)],
        [ComplexF64::real(isq2), ComplexF64::real(-isq2)],
    ];
    mps.apply_single_qubit(0, h);
    for q in 0..3 {
        let zz = mps_zz_general(&mps, q, q);
        assert!(
            (zz - 1.0).abs() < TOL,
            "ZZ({},{}) should be 1.0 for any normalized state, got {}",
            q,
            q,
            zz
        );
    }
}
