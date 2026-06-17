//! Integration + property tests for `seshat merge` (Feature 13).

use cjc_seshat::{
    analyze_trace, diff, merge, replay, serialize, Event, FrameKind, MergeOptions, OwnershipDomain,
    Trace,
};
use proptest::prelude::*;

fn n_samples(t: &Trace) -> usize {
    t.events()
        .iter()
        .filter(|e| matches!(e, Event::Sample { .. }))
        .count()
}

fn py_host() -> Trace {
    let mut b = Trace::builder(1);
    let main = b.intern_frame(FrameKind::Py, "main", "app.py", 1);
    let bnd = b.intern_frame(FrameKind::FfiBoundary, "myext.run", "<native>", 0);
    for _ in 0..6 {
        b.sample_running(&[main, bnd]);
    }
    b.alloc(OwnershipDomain::PyHeap, 500, main);
    b.finish()
}

fn rust_native() -> Trace {
    let mut b = Trace::builder(2);
    let f = b.intern_frame(FrameKind::Rust, "kernel", "k.rs", 9);
    for _ in 0..3 {
        b.sample_running(&[f]);
    }
    b.alloc(OwnershipDomain::RustHeap, 700, f);
    b.finish()
}

#[test]
fn explicit_under_grafts_by_name() {
    let m = merge(
        &py_host(),
        &rust_native(),
        &MergeOptions { under: Some("myext".into()) },
    );
    let nested = m.events().iter().any(|e| match e {
        Event::Sample { stack, .. } => {
            let labels: Vec<String> = stack.iter().map(|&f| m.frame_label(f)).collect();
            labels.iter().any(|l| l.contains("myext.run"))
                && labels.iter().any(|l| l.contains("kernel"))
        }
        _ => false,
    });
    assert!(nested, "the rust kernel should nest under the named boundary");
}

#[test]
fn merged_trace_round_trips() {
    let m = merge(&py_host(), &rust_native(), &MergeOptions::default());
    let back = replay(&serialize(&m)).expect("a merged trace serializes + replays");
    assert_eq!(m.content_hash(), back.content_hash());
    assert_eq!(analyze_trace(&m).content_hash(), analyze_trace(&back).content_hash());
}

#[test]
fn diff_of_merge_with_itself_is_empty() {
    let m = merge(&py_host(), &rust_native(), &MergeOptions::default());
    assert_eq!(diff(&m, &m).num_changes(), 0);
}

#[test]
fn unmatched_under_falls_back_to_union() {
    // a boundary name that matches nothing → no graft, native unioned at root,
    // but both languages still present in one report.
    let m = merge(
        &py_host(),
        &rust_native(),
        &MergeOptions { under: Some("does-not-exist".into()) },
    );
    let r = analyze_trace(&m);
    assert!(r.flamegraph.frame_total.keys().any(|k| k.starts_with("py:")));
    assert!(r.flamegraph.frame_total.keys().any(|k| k.starts_with("rust:")));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Merge preserves every sample: |merged samples| == |host| + |native|.
    #[test]
    fn merge_conserves_sample_count(h in 0usize..20, n in 0usize..20) {
        let mut hb = Trace::builder(0);
        let main = hb.intern_frame(FrameKind::Py, "m", "a.py", 1);
        let bnd = hb.intern_frame(FrameKind::FfiBoundary, "ext.f", "<native>", 0);
        for _ in 0..h { hb.sample_running(&[main, bnd]); }
        let host = hb.finish();

        let mut nb = Trace::builder(0);
        let r = nb.intern_frame(FrameKind::Rust, "r", "r.rs", 1);
        for _ in 0..n { nb.sample_running(&[r]); }
        let native = nb.finish();

        let m = merge(&host, &native, &MergeOptions::default());
        prop_assert_eq!(n_samples(&m), h + n);
    }
}
