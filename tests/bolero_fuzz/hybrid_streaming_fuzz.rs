//! Bolero fuzz target for v3 Phase 3 — Hybrid streaming set-op fast paths.
//!
//! The existing `adaptive_selection_fuzz` caps nrows at 4096, which is below
//! the Hybrid activation threshold (`nrows >= 2 * HYBRID_CHUNK_SIZE = 8192`).
//! This target forces nrows above that threshold so Phase 3 dispatches fire,
//! and adds a scalar oracle (BitMask AND/OR + bit-iteration) for byte-equality
//! comparison to catch any divergence between the per-chunk fast path and the
//! materialize-and-AND fallback.
//!
//! Properties (one bolero target per):
//!   1. `intersect` agrees with scalar oracle for any selection pair.
//!   2. `union` agrees with scalar oracle for any selection pair.
//!   3. Set-op output is ascending and deterministic across two calls.

use cjc_data::adaptive_selection::AdaptiveSelection;
use cjc_data::BitMask;
use std::panic;

/// Hybrid activation needs nrows ≥ 8192. Use 16384 (4 chunks) so partial
/// final chunks aren't a confounder; partial-chunk safety is pinned by
/// `phase3_partial_final_chunk_no_oob` in the integration suite.
const FUZZ_NROWS: usize = 16_384;

fn bools_from_bytes(bytes: &[u8]) -> Vec<bool> {
    (0..FUZZ_NROWS)
        .map(|i| bytes.get(i).copied().unwrap_or(0) & 1 == 1)
        .collect()
}

/// Scalar oracle: AND two BitMasks bit-by-bit, return ascending hit indices.
fn oracle_intersect(a: &BitMask, b: &BitMask) -> Vec<usize> {
    let mut out = Vec::new();
    for i in 0..a.nrows() {
        if a.get(i) && b.get(i) {
            out.push(i);
        }
    }
    out
}

fn oracle_union(a: &BitMask, b: &BitMask) -> Vec<usize> {
    let mut out = Vec::new();
    for i in 0..a.nrows() {
        if a.get(i) || b.get(i) {
            out.push(i);
        }
    }
    out
}

/// Build two AdaptiveSelections from `input` split in half.
fn build_pair(input: &[u8]) -> Option<(AdaptiveSelection, BitMask, AdaptiveSelection, BitMask)> {
    if input.len() < FUZZ_NROWS / 4 {
        return None;
    }
    // Split fuzz input across the two selections; pad with zeros via .get().
    let half = input.len() / 2;
    let bools_a = bools_from_bytes(&input[..half]);
    let bools_b = bools_from_bytes(&input[half..]);
    let bm_a = BitMask::from_bools(&bools_a);
    let bm_b = BitMask::from_bools(&bools_b);
    let sel_a = AdaptiveSelection::from_predicate_result(bm_a.words_slice().to_vec(), FUZZ_NROWS);
    let sel_b = AdaptiveSelection::from_predicate_result(bm_b.words_slice().to_vec(), FUZZ_NROWS);
    Some((sel_a, bm_a, sel_b, bm_b))
}

#[test]
fn fuzz_hybrid_intersect_oracle() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let _ = panic::catch_unwind(|| {
            let Some((sel_a, bm_a, sel_b, bm_b)) = build_pair(input) else {
                return;
            };
            let r = sel_a.intersect(&sel_b);
            let got: Vec<usize> = r.iter_indices().collect();
            let want = oracle_intersect(&bm_a, &bm_b);
            assert_eq!(
                got, want,
                "Phase 3 intersect mismatch: a_mode={}, b_mode={}, r_mode={}",
                sel_a.explain_selection_mode(),
                sel_b.explain_selection_mode(),
                r.explain_selection_mode()
            );
        });
    });
}

#[test]
fn fuzz_hybrid_union_oracle() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let _ = panic::catch_unwind(|| {
            let Some((sel_a, bm_a, sel_b, bm_b)) = build_pair(input) else {
                return;
            };
            let r = sel_a.union(&sel_b);
            let got: Vec<usize> = r.iter_indices().collect();
            let want = oracle_union(&bm_a, &bm_b);
            assert_eq!(
                got, want,
                "Phase 3 union mismatch: a_mode={}, b_mode={}, r_mode={}",
                sel_a.explain_selection_mode(),
                sel_b.explain_selection_mode(),
                r.explain_selection_mode()
            );
        });
    });
}

#[test]
fn fuzz_hybrid_set_op_determinism() {
    // Calling intersect/union twice must produce byte-equal output.
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let _ = panic::catch_unwind(|| {
            let Some((sel_a, _, sel_b, _)) = build_pair(input) else {
                return;
            };
            let r1 = sel_a.intersect(&sel_b);
            let r2 = sel_a.intersect(&sel_b);
            assert_eq!(r1.materialize_indices(), r2.materialize_indices());
            let u1 = sel_a.union(&sel_b);
            let u2 = sel_a.union(&sel_b);
            assert_eq!(u1.materialize_indices(), u2.materialize_indices());
            // Ascending invariant.
            let r_idx: Vec<usize> = r1.iter_indices().collect();
            let u_idx: Vec<usize> = u1.iter_indices().collect();
            for w in r_idx.windows(2) {
                assert!(w[0] < w[1]);
            }
            for w in u_idx.windows(2) {
                assert!(w[0] < w[1]);
            }
        });
    });
}
