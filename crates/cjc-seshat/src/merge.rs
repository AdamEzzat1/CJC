//! Feature 13 — **trace merge**: stitch a Python `.seshat` and a Rust `.seshat`
//! into one unified cross-language trace.
//!
//! The Python recorder sees *the Py↔Rust boundary* (the native callee's name as
//! an [`FrameKind::FfiBoundary`] frame) but not the Rust functions running past
//! it; the Rust collector produces its own trace of that native work. [`merge`]
//! grafts the Rust trace **underneath the matching Python boundary frame**, so
//! the merged flamegraph reads `python → boundary → rust`, the ownership map
//! shows `PyHeap + RustHeap` together, and copies/zones from both sides union.
//!
//! Like every other analysis this is a **pure, deterministic function of its
//! inputs** (two recorded traces in, one trace out) — no wall-clock, no
//! collection. It reuses [`serialize`](crate::serialize)/[`replay`](crate::replay),
//! so it needs no `.seshat` format change.
//!
//! ## Correlation (v1: by name)
//!
//! The graft point is chosen by **name**: either an explicit
//! [`MergeOptions::under`] (matched against the host's boundary-frame labels) or,
//! by default, the boundary frame that appears in the most host samples (the
//! dominant seam). Precise *per-call-site* correlation — a runtime token written
//! by both sides — is future work and would need a format addition; the
//! name-based merge covers the common "one extension entry point" case and never
//! fabricates structure (unmatched → the Rust tree is unioned at the root).

use std::collections::{BTreeMap, BTreeSet};

use crate::trace::{CausalEdge, Event, FrameId, FrameKind, Trace, TraceBuilder};

/// Knobs for [`merge`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MergeOptions {
    /// Name (or substring) of the host boundary frame to graft the native trace
    /// under. `None` ⇒ auto-pick the most-sampled boundary frame.
    pub under: Option<String>,
}

/// Merge a `host` trace (e.g. Python) and a `native` trace (e.g. Rust) into one
/// unified [`Trace`]. The native trace is grafted beneath the chosen host
/// boundary frame; if no boundary is found, the native tree is unioned at the
/// root (still one combined report, just not nested).
///
/// Deterministic: `merge(a, b, o)` is a pure function of `(a, b, o)`.
pub fn merge(host: &Trace, native: &Trace, opts: &MergeOptions) -> Trace {
    let mut b = Trace::builder(host.run_id);

    // ── 1. copy the host verbatim (frames re-interned into the new table) ──
    let mut host_frames: BTreeMap<FrameId, FrameId> = BTreeMap::new();
    let mut host_handles: BTreeMap<u64, u64> = BTreeMap::new();
    copy_events(&mut b, host, &[], &mut host_frames, &mut host_handles);

    // ── 2. choose the graft prefix (path down to the boundary frame) ──
    let prefix = pick_boundary(host, opts.under.as_deref())
        .map(|b_old| graft_prefix(host, b_old, &host_frames))
        .unwrap_or_default();

    // ── 3. copy the native trace, nesting its samples under the prefix ──
    let mut native_frames: BTreeMap<FrameId, FrameId> = BTreeMap::new();
    let mut native_handles: BTreeMap<u64, u64> = BTreeMap::new();
    copy_events(&mut b, native, &prefix, &mut native_frames, &mut native_handles);

    b.set_wall_ns(host.wall_ns_total.saturating_add(native.wall_ns_total));
    b.finish()
}

/// Re-intern a source frame into the builder, memoizing old→new ids.
fn map_frame(
    b: &mut TraceBuilder,
    src: &Trace,
    cache: &mut BTreeMap<FrameId, FrameId>,
    fid: FrameId,
) -> FrameId {
    if let Some(&n) = cache.get(&fid) {
        return n;
    }
    let f = src.frame(fid);
    // own the strings so `src` is not borrowed across the `&mut b` call
    let name = src.string(f.name).to_string();
    let file = src.string(f.file).to_string();
    let n = b.intern_frame(f.kind, &name, &file, f.line);
    cache.insert(fid, n);
    n
}

/// Copy every event of `src` into `b`, remapping frames and (for samples)
/// prepending `prefix` so the source's stacks nest beneath it.
fn copy_events(
    b: &mut TraceBuilder,
    src: &Trace,
    prefix: &[FrameId],
    frames: &mut BTreeMap<FrameId, FrameId>,
    handles: &mut BTreeMap<u64, u64>,
) {
    for ev in src.events() {
        match ev {
            Event::Sample { thread, state, stack } => {
                let mut s: Vec<FrameId> = prefix.to_vec();
                for &f in stack {
                    s.push(map_frame(b, src, frames, f));
                }
                b.sample(*thread, *state, &s);
            }
            Event::Alloc { domain, bytes, frame } => {
                let fr = map_frame(b, src, frames, *frame);
                b.alloc(*domain, *bytes, fr);
            }
            Event::Free { domain, bytes, frame } => {
                let fr = map_frame(b, src, frames, *frame);
                b.free(*domain, *bytes, fr);
            }
            Event::Counter { thread, freq_mhz, cache_misses, ipc_milli } => {
                b.counter(*thread, *freq_mhz, *cache_misses, *ipc_milli);
            }
            Event::ZoneStart { name, handle } => {
                let name_s = src.string(*name).to_string();
                let nh = b.zone_start(&name_s);
                handles.insert(*handle, nh);
            }
            Event::ZoneStop { handle } => {
                if let Some(&nh) = handles.get(handle) {
                    b.zone_stop(nh);
                }
            }
            Event::Edge(edge) => copy_edge(b, src, frames, edge),
        }
    }
}

fn copy_edge(
    b: &mut TraceBuilder,
    src: &Trace,
    frames: &mut BTreeMap<FrameId, FrameId>,
    edge: &CausalEdge,
) {
    match edge {
        CausalEdge::Copy { from, to, bytes, frame } => {
            let fr = map_frame(b, src, frames, *frame);
            b.copy(*from, *to, *bytes, fr);
        }
        CausalEdge::BoundaryCross { boundary } => {
            let fr = map_frame(b, src, frames, *boundary);
            b.boundary_cross(fr);
        }
        CausalEdge::Wakeup { task, by } => {
            let task = map_frame(b, src, frames, *task);
            let by = map_frame(b, src, frames, *by);
            b.edge(CausalEdge::Wakeup { task, by });
        }
        CausalEdge::AwaitResume { task, waited_ticks } => {
            let task = map_frame(b, src, frames, *task);
            b.edge(CausalEdge::AwaitResume { task, waited_ticks: *waited_ticks });
        }
        CausalEdge::GilHandoff { from, to } => {
            let from = map_frame(b, src, frames, *from);
            let to = map_frame(b, src, frames, *to);
            b.edge(CausalEdge::GilHandoff { from, to });
        }
    }
}

/// Choose the host boundary frame to graft under. With `under`, match its
/// (lower-cased) text against boundary frame labels; otherwise pick the boundary
/// frame appearing in the most host samples (ties → lowest frame id, for
/// determinism). `None` if the host has no boundary frame at all.
fn pick_boundary(host: &Trace, under: Option<&str>) -> Option<FrameId> {
    // Inclusive sample counts per FfiBoundary frame.
    let mut counts: BTreeMap<FrameId, u64> = BTreeMap::new();
    for ev in host.events() {
        if let Event::Sample { stack, .. } = ev {
            let mut seen: BTreeSet<FrameId> = BTreeSet::new();
            for &fid in stack {
                if host.frame(fid).kind == FrameKind::FfiBoundary && seen.insert(fid) {
                    *counts.entry(fid).or_insert(0) += 1;
                }
            }
        }
    }
    // Boundary frames that appear only in BoundaryCross edges (e.g. a calls-mode
    // trace) still count as candidates.
    let mut edge_boundaries: Vec<FrameId> = Vec::new();
    for ev in host.events() {
        if let Event::Edge(CausalEdge::BoundaryCross { boundary }) = ev {
            if host.frame(*boundary).kind == FrameKind::FfiBoundary {
                edge_boundaries.push(*boundary);
                counts.entry(*boundary).or_insert(0);
            }
        }
    }

    match under {
        Some(needle) => {
            let needle = needle.to_ascii_lowercase();
            // deterministic scan in frame-id order
            counts
                .keys()
                .copied()
                .find(|&fid| host.frame_label(fid).to_ascii_lowercase().contains(&needle))
        }
        None => counts
            .into_iter()
            // most samples; tie-break by lowest frame id
            .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
            .map(|(fid, _)| fid),
    }
}

/// Build the new-id stack path from the host root down to and including the
/// chosen boundary frame `b_old`. Uses the first host sample that contains it;
/// if it appears in no sample (calls-mode boundary-cross only), the prefix is
/// just the boundary frame itself.
fn graft_prefix(
    host: &Trace,
    b_old: FrameId,
    host_frames: &BTreeMap<FrameId, FrameId>,
) -> Vec<FrameId> {
    for ev in host.events() {
        if let Event::Sample { stack, .. } = ev {
            if let Some(pos) = stack.iter().position(|&f| f == b_old) {
                return stack[..=pos]
                    .iter()
                    .filter_map(|f| host_frames.get(f).copied())
                    .collect();
            }
        }
    }
    host_frames.get(&b_old).copied().into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{FrameKind, OwnershipDomain};

    fn host_trace() -> Trace {
        // python: main → ffi:ext.process (boundary), plus a PyHeap alloc
        let mut b = Trace::builder(1);
        let main = b.intern_frame(FrameKind::Py, "main", "app.py", 1);
        let bnd = b.intern_frame(FrameKind::FfiBoundary, "ext.process", "<native>", 0);
        for _ in 0..5 {
            b.sample_running(&[main, bnd]);
        }
        b.alloc(OwnershipDomain::PyHeap, 1000, main);
        b.set_wall_ns(10);
        b.finish()
    }

    fn native_trace() -> Trace {
        // rust: inner → leaf, plus a RustHeap alloc
        let mut b = Trace::builder(2);
        let inner = b.intern_frame(FrameKind::Rust, "process_batch", "lib.rs", 88);
        let leaf = b.intern_frame(FrameKind::Rust, "dot", "blas.rs", 5);
        for _ in 0..4 {
            b.sample_running(&[inner, leaf]);
        }
        b.alloc(OwnershipDomain::RustHeap, 2000, leaf);
        b.set_wall_ns(20);
        b.finish()
    }

    #[test]
    fn rust_nests_under_python_boundary() {
        let m = merge(&host_trace(), &native_trace(), &MergeOptions::default());
        // some sample stack must read main → ext.process → process_batch → dot
        let labels: Vec<Vec<String>> = m
            .events()
            .iter()
            .filter_map(|e| match e {
                Event::Sample { stack, .. } => {
                    Some(stack.iter().map(|&f| m.frame_label(f)).collect())
                }
                _ => None,
            })
            .collect();
        let nested = labels.iter().any(|s| {
            s.len() >= 4
                && s[0].contains("main")
                && s[1].contains("ext.process")
                && s.iter().any(|l| l.contains("process_batch"))
                && s.iter().any(|l| l.contains("dot"))
        });
        assert!(nested, "rust must nest under the boundary; got {labels:?}");
    }

    #[test]
    fn ownership_unions_both_domains() {
        let m = merge(&host_trace(), &native_trace(), &MergeOptions::default());
        let r = crate::analyze_trace(&m);
        assert!(r.ownership.per_domain_alloc.contains_key("PyHeap"));
        assert!(r.ownership.per_domain_alloc.contains_key("RustHeap"));
        assert_eq!(r.ownership.total_allocated, 3000);
    }

    #[test]
    fn merge_is_deterministic() {
        let (h, n) = (host_trace(), native_trace());
        let h1 = crate::analyze_trace(&merge(&h, &n, &MergeOptions::default())).content_hash();
        for _ in 0..4 {
            assert_eq!(
                crate::analyze_trace(&merge(&h, &n, &MergeOptions::default())).content_hash(),
                h1
            );
        }
    }

    #[test]
    fn union_when_no_boundary() {
        // host with no boundary frame → native unioned at root, still one trace
        let mut hb = Trace::builder(0);
        let m0 = hb.intern_frame(FrameKind::Py, "main", "app.py", 1);
        hb.sample_running(&[m0]);
        let host = hb.finish();
        let m = merge(&host, &native_trace(), &MergeOptions::default());
        let r = crate::analyze_trace(&m);
        // both python and rust frames present
        assert!(r.flamegraph.frame_total.keys().any(|k| k.starts_with("py:")));
        assert!(r.flamegraph.frame_total.keys().any(|k| k.starts_with("rust:")));
    }
}
