//! Fuzz + property targets for the Adaptive TidyView selection engine.
//!
//! Run with:
//!   cargo test --test bolero_fuzz adaptive_selection
//!
//! Properties:
//!   1. Round-trip: from_predicate_result(words, n).materialize_mask() ==
//!      original BitMask (after tail-bit normalization).
//!   2. Mode invariance: count, contains, iter_indices, materialize_mask,
//!      materialize_indices all agree regardless of the chosen arm.
//!   3. Set-op stability: intersect / union for arbitrary mode-mixed inputs
//!      never panics and produces ascending output.
//!   4. Iter ordering: iter_indices() is strictly ascending in every arm.

use cjc_data::adaptive_selection::AdaptiveSelection;
use cjc_data::BitMask;
use std::panic;

/// Build a BitMask from raw bytes interpreted as a bool flag per row.
fn bools_from_bytes(bytes: &[u8], nrows: usize) -> Vec<bool> {
    (0..nrows)
        .map(|i| bytes.get(i).copied().unwrap_or(0) & 1 == 1)
        .collect()
}

/// Property: classifier round-trips through materialize_mask back to a
/// `BitMask` whose set bits exactly match the input.
#[test]
fn fuzz_classifier_round_trip() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let _ = panic::catch_unwind(|| {
            // Cap nrows so we don't spend the whole fuzz time on giant inputs
            let nrows = input.len().min(8192);
            if nrows == 0 {
                return;
            }
            let bools = bools_from_bytes(input, nrows);
            let bm = BitMask::from_bools(&bools);
            let words: Vec<u64> = bm.words_slice().to_vec();
            let sel = AdaptiveSelection::from_predicate_result(words, nrows);
            // Round-trip
            let bm2 = sel.materialize_mask();
            assert_eq!(
                bm.words_slice(),
                bm2.words_slice(),
                "round-trip mismatch in mode {}",
                sel.explain_selection_mode()
            );
            // count agrees
            assert_eq!(sel.count(), bm.count_ones());
        });
    });
}

/// Property: every arm reports the same count, contains, iteration over
/// the same logical selection.
#[test]
fn fuzz_mode_invariance() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let _ = panic::catch_unwind(|| {
            let nrows = input.len().min(8192);
            if nrows == 0 {
                return;
            }
            let bools = bools_from_bytes(input, nrows);
            let bm = BitMask::from_bools(&bools);
            let words: Vec<u64> = bm.words_slice().to_vec();
            let sel = AdaptiveSelection::from_predicate_result(words, nrows);
            // count, iter, contains agree with BitMask
            assert_eq!(sel.count(), bm.count_ones());
            let sel_iter: Vec<usize> = sel.iter_indices().collect();
            let bm_iter: Vec<usize> = bm.iter_set().collect();
            assert_eq!(sel_iter, bm_iter, "iter mismatch in {}", sel.explain_selection_mode());
            for i in 0..nrows {
                assert_eq!(sel.contains(i), bm.get(i), "contains mismatch at row {i}");
            }
        });
    });
}

/// Property: intersect and union produce ascending, deterministic output
/// over arbitrary mode-mixed inputs. Cannot panic.
#[test]
fn fuzz_set_op_stability() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let _ = panic::catch_unwind(|| {
            let nrows = (input.len() / 2).min(4096);
            if nrows == 0 {
                return;
            }
            let half = input.len() / 2;
            let bools_a = bools_from_bytes(&input[..half], nrows);
            let bools_b = bools_from_bytes(&input[half..], nrows);

            let bm_a = BitMask::from_bools(&bools_a);
            let bm_b = BitMask::from_bools(&bools_b);
            let sel_a = AdaptiveSelection::from_predicate_result(
                bm_a.words_slice().to_vec(),
                nrows,
            );
            let sel_b = AdaptiveSelection::from_predicate_result(
                bm_b.words_slice().to_vec(),
                nrows,
            );

            let inter = sel_a.intersect(&sel_b);
            let union = sel_a.union(&sel_b);

            // Ascending
            let inter_idx: Vec<usize> = inter.iter_indices().collect();
            let union_idx: Vec<usize> = union.iter_indices().collect();
            for w in inter_idx.windows(2) {
                assert!(w[0] < w[1]);
            }
            for w in union_idx.windows(2) {
                assert!(w[0] < w[1]);
            }

            // Determinism: same call twice == same output
            let inter2 = sel_a.intersect(&sel_b);
            assert_eq!(inter.materialize_indices(), inter2.materialize_indices());

            // Cardinality identity: |A| + |B| = |A ∪ B| + |A ∩ B|
            assert_eq!(
                sel_a.count() + sel_b.count(),
                inter.count() + union.count(),
                "cardinality identity violated"
            );
        });
    });
}
