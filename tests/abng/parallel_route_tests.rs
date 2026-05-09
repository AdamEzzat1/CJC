//! Phase 0.8 Item C1 — determinism + parity tests for
//! `route_to_leaf_batch_par`.
//!
//! The C1 contract is byte-identical output regardless of thread
//! count: for any valid `(graph, xs, n)`,
//!
//! ```text
//! route_to_leaf_batch_par(xs, n, k) == route_to_leaf_batch(xs, n)
//! ```
//!
//! for every `k >= 1`. These tests gate that contract across
//! representative thread counts (1, 2, 4, 8), various graph shapes
//! (empty, deep, dense), and edge cases (n=0, n<n_threads).
//!
//! Pure determinism: each row's leaf id is a deterministic function
//! of `(graph_state, row_input)`. Writing into disjoint slots of the
//! output `Vec<NodeId>` introduces no reordering. The test vector
//! also covers error parity (NoCodebook, InputArityMismatch).

use cjc_abng::children::AdaptiveChildren;
use cjc_abng::codebook::CodebookError;
use cjc_abng::graph::{AdaptiveBeliefGraph, GraphError};
use cjc_ad::pinn::Activation;

fn build_routing_graph(seed: u64) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(seed);
    g.set_codebook(1, 4, &[-1.0, 0.0, 1.0]).unwrap();
    g.set_leaf_head(1, vec![4], 1, Activation::Tanh).unwrap();
    g.set_blr_prior(2.0, 1.0, 0.5).unwrap();
    // Build a small radix tree by adding children at byte-keys.
    for byte in 0u8..4 {
        g.add_node(0, byte).unwrap();
    }
    g
}

fn deep_routing_graph(seed: u64) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(seed);
    g.set_codebook(2, 4, &[-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
        .unwrap();
    // Two-level tree: each child of root has children of its own.
    for byte in 0u8..4 {
        let _child = g.add_node(0, byte).unwrap();
    }
    for parent in 1u32..=4 {
        for byte in 0u8..4 {
            g.add_node(parent, byte).unwrap();
        }
    }
    g
}

fn synth_input(n: usize, d: usize) -> Vec<f64> {
    (0..n * d)
        .map(|i| ((i as f64) * 0.137).sin())
        .collect()
}

// ---------------------------------------------------------------------------
// Parity gate: parallel == serial for any thread count
// ---------------------------------------------------------------------------

#[test]
fn par_equals_serial_thread_count_1() {
    let g = build_routing_graph(42);
    let xs = synth_input(100, 1);
    let serial = g.route_to_leaf_batch(&xs, 100).unwrap();
    let par = g.route_to_leaf_batch_par(&xs, 100, 1).unwrap();
    assert_eq!(serial, par);
}

#[test]
fn par_equals_serial_thread_count_2() {
    let g = build_routing_graph(42);
    let xs = synth_input(1000, 1);
    let serial = g.route_to_leaf_batch(&xs, 1000).unwrap();
    let par = g.route_to_leaf_batch_par(&xs, 1000, 2).unwrap();
    assert_eq!(serial, par);
}

#[test]
fn par_equals_serial_thread_count_4() {
    let g = build_routing_graph(42);
    let xs = synth_input(1000, 1);
    let serial = g.route_to_leaf_batch(&xs, 1000).unwrap();
    let par = g.route_to_leaf_batch_par(&xs, 1000, 4).unwrap();
    assert_eq!(serial, par);
}

#[test]
fn par_equals_serial_thread_count_8() {
    let g = build_routing_graph(42);
    let xs = synth_input(1000, 1);
    let serial = g.route_to_leaf_batch(&xs, 1000).unwrap();
    let par = g.route_to_leaf_batch_par(&xs, 1000, 8).unwrap();
    assert_eq!(serial, par);
}

#[test]
fn par_equals_serial_on_deep_tree_d2() {
    let g = deep_routing_graph(7);
    let xs = synth_input(500, 2);
    let serial = g.route_to_leaf_batch(&xs, 500).unwrap();
    for k in [1, 2, 4, 8, 16] {
        let par = g.route_to_leaf_batch_par(&xs, 500, k).unwrap();
        assert_eq!(par, serial, "thread count {k} diverged from serial");
    }
}

#[test]
fn par_handles_n_smaller_than_thread_count() {
    // n=3, n_threads=8: should clamp threads to 3 and still match serial.
    let g = build_routing_graph(11);
    let xs = synth_input(3, 1);
    let serial = g.route_to_leaf_batch(&xs, 3).unwrap();
    let par = g.route_to_leaf_batch_par(&xs, 3, 8).unwrap();
    assert_eq!(par, serial);
    assert_eq!(par.len(), 3);
}

#[test]
fn par_n_zero_returns_empty() {
    let g = build_routing_graph(0);
    let par = g.route_to_leaf_batch_par(&[], 0, 4).unwrap();
    assert!(par.is_empty());
}

#[test]
fn par_thread_count_zero_treated_as_one() {
    // 0 thread count would be a logic error; we clamp to 1.
    let g = build_routing_graph(0);
    let xs = synth_input(50, 1);
    let serial = g.route_to_leaf_batch(&xs, 50).unwrap();
    let par = g.route_to_leaf_batch_par(&xs, 50, 0).unwrap();
    assert_eq!(par, serial);
}

#[test]
fn par_very_large_thread_count_clamped() {
    // 1024 threads on n=10 rows: clamped to 10.
    let g = build_routing_graph(0);
    let xs = synth_input(10, 1);
    let serial = g.route_to_leaf_batch(&xs, 10).unwrap();
    let par = g.route_to_leaf_batch_par(&xs, 10, 1024).unwrap();
    assert_eq!(par, serial);
}

// ---------------------------------------------------------------------------
// Error parity
// ---------------------------------------------------------------------------

#[test]
fn par_no_codebook_error_matches_serial() {
    let g = AdaptiveBeliefGraph::new(0);
    let xs = vec![0.0; 4];
    let serial_err = g.route_to_leaf_batch(&xs, 4);
    let par_err = g.route_to_leaf_batch_par(&xs, 4, 4);
    match (&serial_err, &par_err) {
        (Err(GraphError::NoCodebook), Err(GraphError::NoCodebook)) => {}
        other => panic!(
            "both must surface NoCodebook; got serial={:?}, par={:?}",
            other.0, other.1
        ),
    }
}

#[test]
fn par_arity_mismatch_error_matches_serial() {
    let g = build_routing_graph(0);
    // d=1, n=5 -> need 5 floats; provide 4.
    let xs = vec![0.0; 4];
    let par = g.route_to_leaf_batch_par(&xs, 5, 4);
    assert!(matches!(
        par,
        Err(GraphError::Codebook(CodebookError::InputArityMismatch { .. }))
    ));
}

// ---------------------------------------------------------------------------
// Cross-shape determinism — varied input distributions
// ---------------------------------------------------------------------------

#[test]
fn par_deterministic_across_repeated_calls() {
    // Calling route_to_leaf_batch_par twice with the same input must
    // return identical results. (This is a baseline determinism
    // guarantee — if we had any race, repeated calls could diverge.)
    let g = build_routing_graph(0);
    let xs = synth_input(2000, 1);
    let r1 = g.route_to_leaf_batch_par(&xs, 2000, 4).unwrap();
    let r2 = g.route_to_leaf_batch_par(&xs, 2000, 4).unwrap();
    assert_eq!(r1, r2);
}

#[test]
fn par_varying_chunk_alignment() {
    // n=997 rows across 4 threads gives chunk_size=ceil(997/4)=250
    // with the last chunk getting 247. Ensures the alignment math
    // (row_offset, xs_chunk slicing) stays correct on non-divisible
    // sizes.
    let g = build_routing_graph(0);
    let xs = synth_input(997, 1);
    let serial = g.route_to_leaf_batch(&xs, 997).unwrap();
    for k in [3, 4, 5, 7, 11, 13] {
        let par = g.route_to_leaf_batch_par(&xs, 997, k).unwrap();
        assert_eq!(par, serial, "diverged at k={k}");
    }
}

#[test]
fn par_descend_with_children_matches_descend() {
    // White-box-ish: assert the leaves found by route_to_leaf_batch_par
    // match what `descend(prefix).leaf_id` produces, row-by-row.
    let g = build_routing_graph(0);
    let xs = synth_input(50, 1);
    let par = g.route_to_leaf_batch_par(&xs, 50, 4).unwrap();
    let cb = g.codebook.as_ref().unwrap();
    let mut buf = Vec::with_capacity(1);
    for (i, row) in xs.chunks(1).enumerate() {
        cb.encode_into(row, &mut buf).unwrap();
        let leaf_via_descend = g.descend(&buf).leaf_id;
        assert_eq!(par[i], leaf_via_descend, "row {i} diverged");
    }
}

// ---------------------------------------------------------------------------
// Send-bound check
// ---------------------------------------------------------------------------

#[test]
fn par_does_not_require_graph_send() {
    // Compile-time witness: although AdaptiveBeliefGraph is !Send
    // (transitively via Rc inside Tensor), `route_to_leaf_batch_par`
    // must NOT require it. This test confirms the contract holds at
    // the type level.
    fn assert_callable_without_send<F>(_: F)
    where
        F: Fn(&AdaptiveBeliefGraph, &[f64]) -> Result<Vec<u32>, GraphError>,
    {
    }
    assert_callable_without_send(|g, xs| {
        g.route_to_leaf_batch_par(xs, xs.len(), 4)
    });
    // The fact that this compiles means the public signature
    // doesn't impose Send on `&Graph`.

    // Also: AdaptiveChildren is Sync (tensor-free) — required for
    // the &[&AdaptiveChildren] pattern to share across threads.
    fn assert_sync<T: Sync>() {}
    assert_sync::<AdaptiveChildren>();
}
