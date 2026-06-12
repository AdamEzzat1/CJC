//! Phase E probe tests — wiring → unit → proptest → bolero (the
//! standing test-discipline contract).
//!
//! The serialization transforms are the only novel code in Phase E
//! (the codecs themselves are the committed, already-tested
//! `cjc-cana-compress` ones), so the weight here is on proving the
//! transforms bit-exact-invertible for ALL inputs, including NaN
//! payloads and extreme integers.

use std::rc::Rc;

use cana_compress_probe::*;
use cjc_nss::mir_adapter::MirTraceEvent;
use cjc_runtime::{Tensor, Value};

fn event(tick: u64, block: u32) -> MirTraceEvent {
    let mut e = MirTraceEvent::minimal(tick, block);
    e.register_pressure = 0.25;
    e.heap_bytes_in_use = 4096 * (tick % 7);
    e.call_depth = (tick % 5) as u32 + 1;
    e.branch_taken = tick % 2 == 0;
    e.instruction_count = 12;
    e.thermal_intensity = 0.5;
    e
}

// =============================================================================
// Wiring — real instrumented stream, real codecs, real tensors
// =============================================================================

#[test]
fn real_instrumented_stream_roundtrips_and_compresses() {
    // mem_grad_a1 is the smallest Phase-D subject; its instrumented
    // stream exercises the full pipeline (subjects -> compile ->
    // instrument -> serialize -> codecs -> roundtrip) end to end.
    let outcome = measure_trace_subject("mem_grad_a1").expect("trace subject must measure");
    assert!(outcome.total_events > 0, "instrumented run produced no events");
    assert_eq!(
        outcome.measured_events,
        outcome.total_events.min(TRACE_EVENT_CAP)
    );
    assert!(outcome.representations_bitexact);
    assert!(outcome.canonical.roundtrips_ok());
    assert!(outcome.delta.roundtrips_ok());
    // No compression CLAIM here — ratios are whatever the artifact
    // yields — but the bookkeeping must be coherent.
    assert!(outcome.canonical.original_bytes > 0);
    assert_eq!(outcome.canonical.original_bytes, outcome.delta.original_bytes);
}

#[test]
fn lowrank_search_finds_small_rank_on_lowrank_matrix() {
    // Build an exactly-rank-2 64x32 matrix: sum of two outer products.
    let (rows, cols) = (64usize, 32usize);
    let mut m = vec![0.0f64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let a = (r as f64 * 0.1).sin() * (c as f64 * 0.05).cos();
            let b = (r as f64 * 0.03 + 1.0) * (c as f64 * 0.02 - 0.5);
            m[r * cols + c] = 2.0 * a + 0.5 * b;
        }
    }
    let found = smallest_rank_within(&m, rows, cols, 0.05)
        .expect("search must not error")
        .expect("a rank-2 matrix must compress within 5%");
    let (rank, bytes, err) = found;
    assert!(rank <= 2, "rank-2 input needed rank {rank}");
    assert!(err <= 0.05);
    assert!(bytes < 8 * rows * cols, "payload must beat raw storage");
}

#[test]
fn lowrank_search_keeps_raw_when_tolerance_unreachable_cheaply() {
    // A full-rank-ish matrix (deterministic pseudo-noise) at a tight
    // tolerance either needs near-full rank (payload bigger than raw,
    // -> None) or genuinely meets it; both are valid, but the
    // invariants must hold in each case.
    let (rows, cols) = (16usize, 16usize);
    let mut m = vec![0.0f64; rows * cols];
    let mut state = 0x9E3779B97F4A7C15u64;
    for x in m.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *x = ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
    }
    match smallest_rank_within(&m, rows, cols, 1e-6).expect("must not error") {
        Some((rank, bytes, err)) => {
            assert!(err <= 1e-6);
            assert!(bytes < 8 * rows * cols);
            assert!(rank <= rows.min(cols));
        }
        None => {} // kept raw — the honest outcome for incompressible noise
    }
}

#[test]
fn collect_tensors_walks_nested_values() {
    let t1 = Tensor::from_vec_unchecked(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let t2 = Tensor::from_vec_unchecked(vec![5.0; 6], &[6]);
    let inner = Value::Array(Rc::new(vec![Value::Tensor(t2), Value::Int(7)]));
    let root = Value::Array(Rc::new(vec![Value::Tensor(t1), inner, Value::Bool(true)]));
    let mut out = Vec::new();
    collect_tensors(&root, "ckpt".to_string(), &mut out);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].0, "ckpt[0]");
    assert_eq!(out[1].0, "ckpt[1][0]");
    assert_eq!(out[0].1.shape(), &[2, 2]);
}

#[test]
fn disk_artifact_measurement_on_committed_corpus() {
    // profiles.cpdb is committed — this measurement is reproducible
    // from a clean checkout and proves the codec path on a real file.
    let d = measure_disk_artifact("bench_results/cana_ablation/profiles.cpdb")
        .expect("committed corpus must measure");
    assert!(d.outcome.original_bytes > 1_000_000, "corpus should be ~1MB");
    assert!(d.outcome.roundtrips_ok());
}

// =============================================================================
// Unit — serialization edges, decoder rejection, bookkeeping
// =============================================================================

#[test]
fn empty_and_single_event_roundtrip_both_encodings() {
    for events in [vec![], vec![event(42, 7)]] {
        let canon = events_to_canonical_bytes(&events);
        let delta = events_to_delta_bytes(&events);
        assert!(events_bitwise_equal(
            &events,
            &canonical_bytes_to_events(&canon).unwrap()
        ));
        assert!(events_bitwise_equal(
            &events,
            &delta_bytes_to_events(&delta).unwrap()
        ));
    }
}

#[test]
fn extreme_values_roundtrip_bit_exactly() {
    let mut weird = MirTraceEvent::minimal(u64::MAX, u32::MAX);
    weird.register_pressure = f64::from_bits(0x7FF8_0000_0000_1234); // NaN payload
    weird.heap_bytes_in_use = u64::MAX;
    weird.call_depth = u32::MAX;
    weird.branch_taken = true;
    weird.io_event = true;
    weird.gc_event = true;
    weird.instruction_count = u32::MAX;
    weird.thermal_intensity = -0.0;
    let events = vec![weird, event(0, 0), weird];

    let canon_back = canonical_bytes_to_events(&events_to_canonical_bytes(&events)).unwrap();
    let delta_back = delta_bytes_to_events(&events_to_delta_bytes(&events)).unwrap();
    assert!(events_bitwise_equal(&events, &canon_back));
    assert!(events_bitwise_equal(&events, &delta_back));
}

#[test]
fn decoders_reject_malformed_input() {
    let good = events_to_canonical_bytes(&[event(1, 2)]);
    // Wrong magic.
    let mut bad = good.clone();
    bad[0] = b'X';
    assert!(canonical_bytes_to_events(&bad).is_none());
    // Truncated body.
    assert!(canonical_bytes_to_events(&good[..good.len() - 1]).is_none());
    // Count larger than payload.
    let mut over = good.clone();
    over[4] = 99;
    assert!(canonical_bytes_to_events(&over).is_none());
    // Reserved flag bits set.
    let mut flagged = good.clone();
    let flags_offset = 12 + 8 + 4 + 8 + 8 + 4;
    flagged[flags_offset] = 0b1000;
    assert!(canonical_bytes_to_events(&flagged).is_none());
    // Same checks for the delta decoder.
    let dgood = events_to_delta_bytes(&[event(1, 2)]);
    let mut dbad = dgood.clone();
    dbad[0] = b'X';
    assert!(delta_bytes_to_events(&dbad).is_none());
    assert!(delta_bytes_to_events(&dgood[..dgood.len() - 1]).is_none());
    // Empty / garbage.
    assert!(canonical_bytes_to_events(&[]).is_none());
    assert!(delta_bytes_to_events(&[1, 2, 3]).is_none());
}

#[test]
fn lossless_measurement_bookkeeping() {
    // Highly repetitive input must compress under RLE; bookkeeping
    // must reflect it.
    let repetitive = vec![7u8; 64 * 1024];
    let o = measure_lossless(&repetitive);
    assert!(o.roundtrips_ok());
    assert!(o.rle_bytes < o.original_bytes);
    assert!(o.rle_ratio() > 1.0);

    // Empty input is legal and must roundtrip.
    let empty = measure_lossless(&[]);
    assert!(empty.roundtrips_ok());
}

// =============================================================================
// Proptest — transform invertibility over arbitrary inputs
// =============================================================================

mod props {
    use super::*;
    use proptest::prelude::*;

    fn arb_event() -> impl Strategy<Value = MirTraceEvent> {
        (
            any::<u64>(),
            any::<u32>(),
            any::<u64>(), // register_pressure bits
            any::<u64>(),
            any::<u32>(),
            any::<(bool, bool, bool)>(),
            any::<u32>(),
            any::<u64>(), // thermal bits
        )
            .prop_map(|(tick, block, rp, heap, depth, (b, io, gc), ic, th)| {
                let mut e = MirTraceEvent::minimal(tick, block);
                e.register_pressure = f64::from_bits(rp);
                e.heap_bytes_in_use = heap;
                e.call_depth = depth;
                e.branch_taken = b;
                e.io_event = io;
                e.gc_event = gc;
                e.instruction_count = ic;
                e.thermal_intensity = f64::from_bits(th);
                e
            })
    }

    proptest! {
        #[test]
        fn canonical_roundtrips_arbitrary_events(events in proptest::collection::vec(arb_event(), 0..64)) {
            let bytes = events_to_canonical_bytes(&events);
            let back = canonical_bytes_to_events(&bytes).unwrap();
            prop_assert!(events_bitwise_equal(&events, &back));
        }

        #[test]
        fn delta_roundtrips_arbitrary_events(events in proptest::collection::vec(arb_event(), 0..64)) {
            let bytes = events_to_delta_bytes(&events);
            let back = delta_bytes_to_events(&bytes).unwrap();
            prop_assert!(events_bitwise_equal(&events, &back));
        }

        #[test]
        fn lossless_codecs_roundtrip_arbitrary_bytes(input in proptest::collection::vec(any::<u8>(), 0..4096)) {
            let o = measure_lossless(&input);
            prop_assert!(o.roundtrips_ok());
            prop_assert_eq!(o.original_bytes, input.len());
        }
    }
}

// =============================================================================
// Bolero — decoders never panic on arbitrary bytes
// =============================================================================

#[test]
fn fuzz_canonical_decoder_never_panics() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|bytes| {
        let _ = canonical_bytes_to_events(bytes);
    });
}

#[test]
fn fuzz_delta_decoder_never_panics() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|bytes| {
        let _ = delta_bytes_to_events(bytes);
    });
}
