//! Property tests for the Seshat analysis engine. Each invariant runs over
//! randomly-generated traces (proptest default 256 cases) and encodes a
//! guarantee the determinism/soundness contract depends on.

use cjc_seshat::analyze::{self, pct_milli};
use cjc_seshat::{
    analyze_trace, diff, replay, serialize, FlameNode, FrameKind, OwnershipDomain, Trace,
};
use proptest::prelude::*;

const N_FRAMES: usize = 6;

fn kind_for(i: usize) -> FrameKind {
    match i % 5 {
        0 => FrameKind::Py,
        1 => FrameKind::Rust,
        2 => FrameKind::Native,
        3 => FrameKind::FfiBoundary,
        _ => FrameKind::AsyncTask,
    }
}

fn domain_for(i: usize) -> OwnershipDomain {
    match i % 8 {
        0 => OwnershipDomain::PyHeap,
        1 => OwnershipDomain::RustHeap,
        2 => OwnershipDomain::Mmap,
        3 => OwnershipDomain::NumPy,
        4 => OwnershipDomain::Arrow,
        5 => OwnershipDomain::Tensor,
        6 => OwnershipDomain::Gpu,
        _ => OwnershipDomain::NativeExt,
    }
}

fn sum_self(n: &FlameNode) -> u64 {
    n.self_count + n.children.iter().map(sum_self).sum::<u64>()
}

// strategy: a list of stacks, each a short vec of frame indices [0, N_FRAMES)
fn stacks_strategy() -> impl Strategy<Value = Vec<Vec<usize>>> {
    prop::collection::vec(
        prop::collection::vec(0..N_FRAMES, 0..5),
        0..40,
    )
}

fn trace_from_stacks(stacks: &[Vec<usize>]) -> Trace {
    let mut b = Trace::builder(0);
    let frames: Vec<u32> = (0..N_FRAMES)
        .map(|i| b.intern_frame(kind_for(i), &format!("f{i}"), "x.rs", i as u32))
        .collect();
    for st in stacks {
        let stack: Vec<u32> = st.iter().map(|&i| frames[i]).collect();
        b.sample_running(&stack);
    }
    b.finish()
}

proptest! {
    // ── Property 1: sample conservation ──
    #[test]
    fn p1_sample_conservation(stacks in stacks_strategy()) {
        let t = trace_from_stacks(&stacks);
        let fg = analyze::flamegraph(&t);
        let nonempty = stacks.iter().filter(|s| !s.is_empty()).count() as u64;
        prop_assert_eq!(fg.total_samples, nonempty);
        prop_assert_eq!(sum_self(&fg.root), nonempty);
        prop_assert_eq!(fg.frame_self.values().sum::<u64>(), nonempty);
    }

    // ── Property 2: ownership partition ──
    #[test]
    fn p2_ownership_partition(
        allocs in prop::collection::vec((0usize..8, 0u64..1_000_000), 0..50)
    ) {
        let mut b = Trace::builder(0);
        let f = b.intern_frame(FrameKind::Rust, "alloc_site", "a.rs", 1);
        let mut expected = 0u64;
        for (dom, bytes) in &allocs {
            b.alloc(domain_for(*dom), *bytes, f);
            expected += *bytes;
        }
        let t = b.finish();
        let r = analyze::ownership(&t);
        prop_assert_eq!(r.total_allocated, expected);
        prop_assert_eq!(r.per_domain_alloc.values().sum::<u64>(), expected);
    }

    // ── Property 3: merge order-invariance (the determinism backbone) ──
    #[test]
    fn p3_flamegraph_order_invariant(stacks in stacks_strategy()) {
        let forward = trace_from_stacks(&stacks);
        let mut rev = stacks.clone();
        rev.reverse();
        let backward = trace_from_stacks(&rev);
        prop_assert_eq!(
            analyze::flamegraph(&forward).content_hash(),
            analyze::flamegraph(&backward).content_hash()
        );
    }

    // ── Property 4: copy detector soundness (no false positives) ──
    #[test]
    fn p4_copy_soundness(
        copies in prop::collection::vec((0usize..8, 0usize..8, 1u64..1_000_000), 0..40)
    ) {
        let mut b = Trace::builder(0);
        let f = b.intern_frame(FrameKind::Rust, "copy_site", "c.rs", 1);
        let mut expected_pairs = std::collections::BTreeSet::new();
        let mut expected_bytes = 0u64;
        for (from, to, bytes) in &copies {
            let (fd, td) = (domain_for(*from), domain_for(*to));
            b.copy(fd, td, *bytes, f);
            expected_pairs.insert((fd.label().to_string(), td.label().to_string()));
            expected_bytes += *bytes;
        }
        let t = b.finish();
        let r = analyze::copy(&t);
        // every reported flow corresponds to a real (from,to) we generated
        for flow in &r.flows {
            prop_assert!(
                expected_pairs.contains(&(flow.from.clone(), flow.to.clone())),
                "reported a copy {}->{} that was never recorded", flow.from, flow.to
            );
        }
        prop_assert_eq!(r.total_bytes, expected_bytes);
    }

    // ── Property 5: diff identity ──
    #[test]
    fn p5_diff_identity(stacks in stacks_strategy()) {
        let t = trace_from_stacks(&stacks);
        let d = diff(&t, &t);
        prop_assert_eq!(d.num_changes(), 0);
        prop_assert_eq!(d.boundary_delta_milli, 0);
        prop_assert_eq!(d.copy_bytes_delta, 0);
    }

    // ── Property 6: serialize/replay round-trip preserves content hash ──
    #[test]
    fn p6_round_trip_content_hash(stacks in stacks_strategy()) {
        let t = trace_from_stacks(&stacks);
        let bytes = serialize(&t);
        let back = replay(&bytes).expect("our own serialization always replays");
        prop_assert_eq!(t.content_hash(), back.content_hash());
    }

    // ── Property 7: bounded numeric outputs (no NaN/Inf, percentages in range) ──
    #[test]
    fn p7_bounded_outputs(stacks in stacks_strategy()) {
        let t = trace_from_stacks(&stacks);
        let r = analyze_trace(&t);
        // boundary share is a valid percentage
        let share = pct_milli(r.boundary.boundary_samples, r.boundary.total_samples);
        prop_assert!(share <= 100_000);
        // contention components never exceed the sample total
        let c = &r.contention;
        prop_assert!(c.running + c.gil_wait + c.lock_wait + c.channel_wait + c.io_wait + c.async_idle == c.total);
        // analysing twice is byte-identical
        prop_assert_eq!(r.content_hash(), analyze_trace(&t).content_hash());
    }

    // ── pct_milli is always in [0, 100_000] when part <= total ──
    #[test]
    fn p8_pct_milli_in_range(total in 0u64..1_000_000, frac in 0u64..1_000_000) {
        let part = frac.min(total);
        let p = pct_milli(part, total);
        prop_assert!(p <= 100_000);
        if total == 0 { prop_assert_eq!(p, 0); }
    }
}
