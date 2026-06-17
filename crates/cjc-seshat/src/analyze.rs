//! The analysis engine — the 12 Seshat features, each a **pure, deterministic
//! reduction** over a [`Trace`]. No wall-clock, no `HashMap`, no float
//! summation in the hashed path: counts and bytes are integers, percentages are
//! expressed as integer *milli-percent* (`100% == 100_000`) so reports hash
//! bit-identically on every platform.
//!
//! Feature map:
//!  1 flamegraph · 2 boundary · 3 copy · 4 contention · 5 async-stall ·
//!  6 ownership · 7 peak · 8 advise (in `report.rs`) · 9 variance ·
//! 10 thermal · 11 pipeline · 12 regress

use std::collections::{BTreeMap, BTreeSet};

use crate::hash::SeshatHasher;
use crate::trace::{CausalEdge, Event, FrameKind, ThreadState, Trace};

/// Integer milli-percent: `part/total` scaled so 100% == 100_000. Deterministic
/// (single integer division), never NaN.
pub fn pct_milli(part: u64, total: u64) -> u64 {
    if total == 0 {
        0
    } else {
        // u128 to avoid overflow on large byte counts.
        ((part as u128 * 100_000u128) / total as u128) as u64
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Feature 1 — Cross-language flamegraph
// ════════════════════════════════════════════════════════════════════════════

/// A node in the merged cross-language call tree. Self-contained (carries the
/// resolved `label`) so it renders and hashes without the originating trace,
/// which is also what makes the hash independent of interning-id *values*.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlameNode {
    pub label: String,
    pub kind_tag: u8,
    pub self_count: u64,
    pub total_count: u64,
    pub children: Vec<FlameNode>,
}

/// Output of [`flamegraph`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlamegraphReport {
    /// Number of non-empty samples merged.
    pub total_samples: u64,
    /// Synthetic root; `root.children` are the outermost frames.
    pub root: FlameNode,
    /// label → self (leaf) sample count.
    pub frame_self: BTreeMap<String, u64>,
    /// label → inclusive sample count (appears anywhere in the stack).
    pub frame_total: BTreeMap<String, u64>,
}

struct Tmp {
    kind_tag: u8,
    self_count: u64,
    total_count: u64,
    children: BTreeMap<String, Tmp>,
}

fn tmp_to_node(label: String, t: Tmp) -> FlameNode {
    let children = t
        .children
        .into_iter()
        .map(|(k, v)| tmp_to_node(k, v))
        .collect();
    FlameNode {
        label,
        kind_tag: t.kind_tag,
        self_count: t.self_count,
        total_count: t.total_count,
        children,
    }
}

/// Build the merged call tree. Children are sorted by label (via `BTreeMap`), so
/// merging independent samples in any order yields the same tree — the
/// determinism backbone (`tests/prop.rs` property #3).
pub fn flamegraph(trace: &Trace) -> FlamegraphReport {
    let mut roots: BTreeMap<String, Tmp> = BTreeMap::new();
    let mut total: u64 = 0;
    let mut frame_self: BTreeMap<String, u64> = BTreeMap::new();
    let mut frame_total: BTreeMap<String, u64> = BTreeMap::new();

    for ev in trace.events() {
        let Event::Sample { stack, .. } = ev else { continue };
        if stack.is_empty() {
            continue;
        }
        total += 1;

        let mut seen: BTreeSet<String> = BTreeSet::new();
        let mut cur = &mut roots;
        let last = stack.len() - 1;
        for (i, &fid) in stack.iter().enumerate() {
            let label = trace.frame_label(fid);
            let kind_tag = trace.frame(fid).kind.tag();

            if seen.insert(label.clone()) {
                *frame_total.entry(label.clone()).or_insert(0) += 1;
            }

            let node = cur.entry(label.clone()).or_insert_with(|| Tmp {
                kind_tag,
                self_count: 0,
                total_count: 0,
                children: BTreeMap::new(),
            });
            node.total_count += 1;
            if i == last {
                node.self_count += 1;
                *frame_self.entry(label.clone()).or_insert(0) += 1;
            }
            cur = &mut node.children;
        }
    }

    let children: Vec<FlameNode> = roots.into_iter().map(|(k, v)| tmp_to_node(k, v)).collect();
    let root = FlameNode {
        label: "(root)".to_string(),
        kind_tag: 255,
        self_count: 0,
        total_count: total,
        children,
    };
    FlamegraphReport {
        total_samples: total,
        root,
        frame_self,
        frame_total,
    }
}

fn hash_node(h: &mut SeshatHasher, n: &FlameNode) {
    h.write_str(&n.label);
    h.write_u8(n.kind_tag);
    h.write_u64(n.self_count);
    h.write_u64(n.total_count);
    h.write_u64(n.children.len() as u64);
    for c in &n.children {
        hash_node(h, c);
    }
}

impl FlamegraphReport {
    pub fn hash_into(&self, h: &mut SeshatHasher) {
        h.write_u64(self.total_samples);
        hash_node(h, &self.root);
    }
    /// Standalone content hash of the merged tree. Independent of the order in
    /// which independent samples were recorded (property #3).
    pub fn content_hash(&self) -> u64 {
        let mut h = SeshatHasher::new();
        h.write(b"seshat-flamegraph-v1");
        self.hash_into(&mut h);
        h.finish()
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Feature 2 — Python↔Rust boundary cost
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundaryReport {
    pub total_samples: u64,
    /// Samples whose stack contains *any* FFI-boundary frame (inclusive cost).
    pub boundary_samples: u64,
    /// Samples whose *leaf* is an FFI-boundary frame (time spent in the crossing
    /// itself: conversion/refcount/GIL acquisition).
    pub boundary_self: u64,
    /// boundary frame label → inclusive sample count.
    pub per_boundary: BTreeMap<String, u64>,
    /// Number of explicit `BoundaryCross` edges.
    pub crossings: u64,
}

pub fn boundary(trace: &Trace) -> BoundaryReport {
    let mut total = 0u64;
    let mut bsamp = 0u64;
    let mut bself = 0u64;
    let mut per: BTreeMap<String, u64> = BTreeMap::new();
    let mut crossings = 0u64;

    for ev in trace.events() {
        match ev {
            Event::Sample { stack, .. } if !stack.is_empty() => {
                total += 1;
                let mut has = false;
                let mut seen: BTreeSet<String> = BTreeSet::new();
                for &fid in stack {
                    if trace.frame(fid).kind == FrameKind::FfiBoundary {
                        has = true;
                        let l = trace.frame_label(fid);
                        if seen.insert(l.clone()) {
                            *per.entry(l).or_insert(0) += 1;
                        }
                    }
                }
                if has {
                    bsamp += 1;
                }
                if let Some(&top) = stack.last() {
                    if trace.frame(top).kind == FrameKind::FfiBoundary {
                        bself += 1;
                    }
                }
            }
            Event::Edge(CausalEdge::BoundaryCross { .. }) => crossings += 1,
            _ => {}
        }
    }

    BoundaryReport {
        total_samples: total,
        boundary_samples: bsamp,
        boundary_self: bself,
        per_boundary: per,
        crossings,
    }
}

impl BoundaryReport {
    pub fn hash_into(&self, h: &mut SeshatHasher) {
        h.write_u64(self.total_samples);
        h.write_u64(self.boundary_samples);
        h.write_u64(self.boundary_self);
        h.write_u64(self.crossings);
        h.write_u64(self.per_boundary.len() as u64);
        for (k, v) in &self.per_boundary {
            h.write_str(k);
            h.write_u64(*v);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Feature 3 — Copy detector  (CONTRACT: no false positives)
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CopyFlow {
    pub from: String,
    pub to: String,
    pub bytes: u64,
    pub count: u64,
    /// True iff the (from, to) domains are zero-copy-compatible — i.e. this copy
    /// is *avoidable*.
    pub avoidable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CopyReport {
    pub total_bytes: u64,
    pub total_count: u64,
    pub avoidable_bytes: u64,
    /// Sorted by bytes desc, then (from, to) — every entry corresponds to a real
    /// `Copy` edge in the trace.
    pub flows: Vec<CopyFlow>,
}

pub fn copy(trace: &Trace) -> CopyReport {
    use crate::trace::OwnershipDomain;
    let mut agg: BTreeMap<(OwnershipDomain, OwnershipDomain), (u64, u64)> = BTreeMap::new();
    for ev in trace.events() {
        if let Event::Edge(CausalEdge::Copy { from, to, bytes, .. }) = ev {
            let e = agg.entry((*from, *to)).or_insert((0, 0));
            e.0 += *bytes;
            e.1 += 1;
        }
    }

    let mut flows: Vec<CopyFlow> = agg
        .into_iter()
        .map(|((from, to), (bytes, count))| CopyFlow {
            from: from.label().to_string(),
            to: to.label().to_string(),
            bytes,
            count,
            avoidable: OwnershipDomain::zero_copy_compatible(from, to),
        })
        .collect();
    // bytes desc, then label order for stable ties.
    flows.sort_by(|a, b| {
        b.bytes
            .cmp(&a.bytes)
            .then_with(|| a.from.cmp(&b.from))
            .then_with(|| a.to.cmp(&b.to))
    });

    let total_bytes = flows.iter().map(|f| f.bytes).sum();
    let total_count = flows.iter().map(|f| f.count).sum();
    let avoidable_bytes = flows.iter().filter(|f| f.avoidable).map(|f| f.bytes).sum();

    CopyReport {
        total_bytes,
        total_count,
        avoidable_bytes,
        flows,
    }
}

impl CopyReport {
    pub fn hash_into(&self, h: &mut SeshatHasher) {
        h.write_u64(self.total_bytes);
        h.write_u64(self.total_count);
        h.write_u64(self.avoidable_bytes);
        h.write_u64(self.flows.len() as u64);
        for f in &self.flows {
            h.write_str(&f.from);
            h.write_str(&f.to);
            h.write_u64(f.bytes);
            h.write_u64(f.count);
            h.write_u8(f.avoidable as u8);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Feature 4 — GIL contention + Rust thread contention
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ContentionReport {
    pub running: u64,
    pub gil_wait: u64,
    pub lock_wait: u64,
    pub channel_wait: u64,
    pub io_wait: u64,
    pub async_idle: u64,
    pub total: u64,
}

pub fn contention(trace: &Trace) -> ContentionReport {
    let mut r = ContentionReport::default();
    for ev in trace.events() {
        if let Event::Sample { state, .. } = ev {
            r.total += 1;
            match state {
                ThreadState::Running => r.running += 1,
                ThreadState::GilWait => r.gil_wait += 1,
                ThreadState::LockWait => r.lock_wait += 1,
                ThreadState::ChannelWait => r.channel_wait += 1,
                ThreadState::IoWait => r.io_wait += 1,
                ThreadState::AsyncIdle => r.async_idle += 1,
            }
        }
    }
    r
}

impl ContentionReport {
    /// Samples blocked on Rust-side synchronization (mutex/channel).
    pub fn rust_blocked(&self) -> u64 {
        self.lock_wait + self.channel_wait
    }
    pub fn hash_into(&self, h: &mut SeshatHasher) {
        for v in [
            self.running,
            self.gil_wait,
            self.lock_wait,
            self.channel_wait,
            self.io_wait,
            self.async_idle,
            self.total,
        ] {
            h.write_u64(v);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Feature 5 — Async-aware profiling
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct AsyncTaskStat {
    /// `AsyncIdle` samples whose leaf is this task (parked at an await).
    pub stall_ticks: u64,
    pub wakeups: u64,
    pub resumes: u64,
    pub max_wait_ticks: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct AsyncReport {
    pub tasks: BTreeMap<String, AsyncTaskStat>,
    pub total_wakeups: u64,
    pub total_resumes: u64,
    pub total_stall_ticks: u64,
}

pub fn async_stall(trace: &Trace) -> AsyncReport {
    let mut r = AsyncReport::default();
    for ev in trace.events() {
        match ev {
            Event::Sample { state: ThreadState::AsyncIdle, stack, .. } if !stack.is_empty() => {
                let top = *stack.last().unwrap();
                if trace.frame(top).kind == FrameKind::AsyncTask {
                    let l = trace.frame_label(top);
                    r.tasks.entry(l).or_default().stall_ticks += 1;
                    r.total_stall_ticks += 1;
                }
            }
            Event::Edge(CausalEdge::Wakeup { task, .. }) => {
                let l = trace.frame_label(*task);
                r.tasks.entry(l).or_default().wakeups += 1;
                r.total_wakeups += 1;
            }
            Event::Edge(CausalEdge::AwaitResume { task, waited_ticks }) => {
                let l = trace.frame_label(*task);
                let s = r.tasks.entry(l).or_default();
                s.resumes += 1;
                if *waited_ticks > s.max_wait_ticks {
                    s.max_wait_ticks = *waited_ticks;
                }
                r.total_resumes += 1;
            }
            _ => {}
        }
    }
    r
}

impl AsyncReport {
    pub fn hash_into(&self, h: &mut SeshatHasher) {
        h.write_u64(self.total_wakeups);
        h.write_u64(self.total_resumes);
        h.write_u64(self.total_stall_ticks);
        h.write_u64(self.tasks.len() as u64);
        for (k, s) in &self.tasks {
            h.write_str(k);
            h.write_u64(s.stall_ticks);
            h.write_u64(s.wakeups);
            h.write_u64(s.resumes);
            h.write_u64(s.max_wait_ticks);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Feature 6 — Memory ownership map  (PROPERTY: partitions all tracked bytes)
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct OwnershipReport {
    pub total_allocated: u64,
    pub per_domain_alloc: BTreeMap<String, u64>,
    pub per_domain_freed: BTreeMap<String, u64>,
    /// alloc − freed per domain (net live bytes; may be negative if a trace
    /// frees more than it allocated within the window).
    pub per_domain_live: BTreeMap<String, i64>,
    /// "From->To" domain → bytes transferred via copy edges (ownership handoff).
    pub transfers: BTreeMap<String, u64>,
}

pub fn ownership(trace: &Trace) -> OwnershipReport {
    let mut r = OwnershipReport::default();
    for ev in trace.events() {
        match ev {
            Event::Alloc { domain, bytes, .. } => {
                *r.per_domain_alloc.entry(domain.label().to_string()).or_insert(0) += *bytes;
                r.total_allocated += *bytes;
            }
            Event::Free { domain, bytes, .. } => {
                *r.per_domain_freed.entry(domain.label().to_string()).or_insert(0) += *bytes;
            }
            Event::Edge(CausalEdge::Copy { from, to, bytes, .. }) => {
                let k = format!("{}->{}", from.label(), to.label());
                *r.transfers.entry(k).or_insert(0) += *bytes;
            }
            _ => {}
        }
    }
    // net live per domain (union of alloc/free keys)
    let mut domains: BTreeSet<String> = BTreeSet::new();
    domains.extend(r.per_domain_alloc.keys().cloned());
    domains.extend(r.per_domain_freed.keys().cloned());
    for d in domains {
        let a = *r.per_domain_alloc.get(&d).unwrap_or(&0) as i64;
        let f = *r.per_domain_freed.get(&d).unwrap_or(&0) as i64;
        r.per_domain_live.insert(d, a - f);
    }
    r
}

impl OwnershipReport {
    pub fn hash_into(&self, h: &mut SeshatHasher) {
        h.write_u64(self.total_allocated);
        for (label, map) in [
            ("alloc", &self.per_domain_alloc),
            ("freed", &self.per_domain_freed),
        ] {
            h.write_str(label);
            h.write_u64(map.len() as u64);
            for (k, v) in map {
                h.write_str(k);
                h.write_u64(*v);
            }
        }
        h.write_u64(self.per_domain_live.len() as u64);
        for (k, v) in &self.per_domain_live {
            h.write_str(k);
            h.write_i64(*v);
        }
        h.write_u64(self.transfers.len() as u64);
        for (k, v) in &self.transfers {
            h.write_str(k);
            h.write_u64(*v);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Feature 7 — Peak-memory explanation
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PeakContributor {
    pub domain: String,
    pub frame: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PeakReport {
    pub peak_bytes: u64,
    /// Logical seq at which the peak live-set occurred.
    pub peak_seq: u64,
    /// Top contributors to the live-set at the peak, bytes desc.
    pub contributors: Vec<PeakContributor>,
    /// Deterministic human narrative.
    pub narrative: String,
}

pub fn peak(trace: &Trace) -> PeakReport {
    let mut live: i64 = 0;
    let mut peak: i64 = 0;
    let mut peak_seq: u64 = 0;
    // (domain, frame-label) -> live bytes
    let mut bucket: BTreeMap<(String, String), i64> = BTreeMap::new();
    let mut peak_bucket: BTreeMap<(String, String), i64> = BTreeMap::new();

    for (seq, ev) in trace.events().iter().enumerate() {
        match ev {
            Event::Alloc { domain, bytes, frame } => {
                live += *bytes as i64;
                let key = (domain.label().to_string(), trace.frame_label(*frame));
                *bucket.entry(key).or_insert(0) += *bytes as i64;
            }
            Event::Free { domain, bytes, frame } => {
                live -= *bytes as i64;
                let key = (domain.label().to_string(), trace.frame_label(*frame));
                *bucket.entry(key).or_insert(0) -= *bytes as i64;
            }
            _ => continue,
        }
        if live > peak {
            peak = live;
            peak_seq = seq as u64;
            peak_bucket = bucket.clone();
        }
    }

    let mut contributors: Vec<PeakContributor> = peak_bucket
        .into_iter()
        .filter(|(_, v)| *v > 0)
        .map(|((domain, frame), v)| PeakContributor {
            domain,
            frame,
            bytes: v as u64,
        })
        .collect();
    contributors.sort_by(|a, b| {
        b.bytes
            .cmp(&a.bytes)
            .then_with(|| a.domain.cmp(&b.domain))
            .then_with(|| a.frame.cmp(&b.frame))
    });
    contributors.truncate(5);

    let narrative = if peak <= 0 {
        "No tracked allocations; peak live-set is zero.".to_string()
    } else {
        let mut s = format!(
            "Peak live-set of {} bytes at logical seq {}.",
            peak, peak_seq
        );
        if let Some(top) = contributors.first() {
            s.push_str(&format!(
                " Largest contributor: {} bytes owned by {} (allocated at {}).",
                top.bytes, top.domain, top.frame
            ));
        }
        if contributors.len() > 1 {
            s.push_str(&format!(
                " {} distinct (domain, frame) owners were simultaneously live.",
                contributors.len()
            ));
        }
        s
    };

    PeakReport {
        peak_bytes: peak.max(0) as u64,
        peak_seq,
        contributors,
        narrative,
    }
}

impl PeakReport {
    pub fn hash_into(&self, h: &mut SeshatHasher) {
        h.write_u64(self.peak_bytes);
        h.write_u64(self.peak_seq);
        h.write_u64(self.contributors.len() as u64);
        for c in &self.contributors {
            h.write_str(&c.domain);
            h.write_str(&c.frame);
            h.write_u64(c.bytes);
        }
        // narrative is derived from the above; excluded from hash (display only)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Feature 10 — Thermal / power-aware mode
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThermalReport {
    pub counters_available: bool,
    pub baseline_mhz: u32,
    pub min_mhz: u32,
    pub max_cache_misses: u64,
    pub throttle_detected: bool,
    pub oversubscription: bool,
    pub distinct_threads: u32,
    pub cores: u32,
    pub evidence: Vec<String>,
}

/// `cores` is the configured physical core count for oversubscription detection
/// (0 = unknown → never flag oversubscription).
pub fn thermal(trace: &Trace, cores: u32) -> ThermalReport {
    let mut baseline: Option<u32> = None;
    let mut min_mhz = u32::MAX;
    let mut max_misses = 0u64;
    let mut have_counters = false;
    let mut threads: BTreeSet<u32> = BTreeSet::new();

    for ev in trace.events() {
        match ev {
            Event::Counter { freq_mhz, cache_misses, .. } => {
                have_counters = true;
                if baseline.is_none() {
                    baseline = Some(*freq_mhz);
                }
                if *freq_mhz < min_mhz {
                    min_mhz = *freq_mhz;
                }
                if *cache_misses > max_misses {
                    max_misses = *cache_misses;
                }
            }
            Event::Sample { thread, .. } => {
                threads.insert(*thread);
            }
            _ => {}
        }
    }

    let baseline_mhz = baseline.unwrap_or(0);
    let min_mhz = if have_counters { min_mhz } else { 0 };
    // Throttle: sustained freq drop ≥10% below the first observed frequency.
    let throttle_detected =
        have_counters && baseline_mhz > 0 && min_mhz < baseline_mhz * 9 / 10;
    let distinct_threads = threads.len() as u32;
    let oversubscription = cores > 0 && distinct_threads > cores;

    let mut evidence = Vec::new();
    if !have_counters {
        evidence.push("No hardware counter samples in trace (thermal mode needs `collect-live` counters).".to_string());
    } else {
        evidence.push(format!(
            "Frequency ranged {} → {} MHz (baseline {}).",
            baseline_mhz, min_mhz, baseline_mhz
        ));
        if throttle_detected {
            evidence.push(format!(
                "Throttle: min frequency {} MHz is >10% below baseline {} MHz.",
                min_mhz, baseline_mhz
            ));
        }
        evidence.push(format!("Peak cache-miss sample: {}.", max_misses));
    }
    if oversubscription {
        evidence.push(format!(
            "Oversubscription: {} logical threads active on {} cores.",
            distinct_threads, cores
        ));
    }

    ThermalReport {
        counters_available: have_counters,
        baseline_mhz,
        min_mhz,
        max_cache_misses: max_misses,
        throttle_detected,
        oversubscription,
        distinct_threads,
        cores,
        evidence,
    }
}

impl ThermalReport {
    pub fn hash_into(&self, h: &mut SeshatHasher) {
        h.write_u8(self.counters_available as u8);
        h.write_u32(self.baseline_mhz);
        h.write_u32(self.min_mhz);
        h.write_u64(self.max_cache_misses);
        h.write_u8(self.throttle_detected as u8);
        h.write_u8(self.oversubscription as u8);
        h.write_u32(self.distinct_threads);
        h.write_u32(self.cores);
        // evidence strings derive from the above; display-only.
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Feature 11 — Data-pipeline profiler
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct PipelineReport {
    /// stage (zone) name → samples attributed to it (innermost open zone wins).
    pub per_stage: BTreeMap<String, u64>,
    pub total_samples: u64,
}

pub fn pipeline(trace: &Trace) -> PipelineReport {
    let mut r = PipelineReport::default();
    // stack of (handle, name) for currently-open zones
    let mut zones: Vec<(u64, String)> = Vec::new();
    for ev in trace.events() {
        match ev {
            Event::ZoneStart { name, handle } => {
                zones.push((*handle, trace.string(*name).to_string()));
            }
            Event::ZoneStop { handle } => {
                // pop the matching zone (and anything opened after it that was
                // left dangling — robust to malformed nesting)
                if let Some(pos) = zones.iter().rposition(|(h, _)| h == handle) {
                    zones.truncate(pos);
                }
            }
            Event::Sample { stack, .. } if !stack.is_empty() => {
                r.total_samples += 1;
                let stage = zones
                    .last()
                    .map(|(_, n)| n.clone())
                    .unwrap_or_else(|| "(unzoned)".to_string());
                *r.per_stage.entry(stage).or_insert(0) += 1;
            }
            _ => {}
        }
    }
    r
}

impl PipelineReport {
    pub fn hash_into(&self, h: &mut SeshatHasher) {
        h.write_u64(self.total_samples);
        h.write_u64(self.per_stage.len() as u64);
        for (k, v) in &self.per_stage {
            h.write_str(k);
            h.write_u64(*v);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Feature 9 — Determinism / reproducibility profiling (multi-trace)
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VariantFrame {
    pub label: String,
    pub min_pct_milli: u64,
    pub max_pct_milli: u64,
    /// max − min, in milli-percent. Larger ⇒ more run-to-run variance.
    pub spread_milli: u64,
    pub suspected_cause: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VarianceReport {
    pub runs: usize,
    pub total_samples: Vec<u64>,
    /// Frames whose inclusive-share spread across runs exceeds the threshold,
    /// sorted by spread desc.
    pub variant_frames: Vec<VariantFrame>,
    pub threshold_milli: u64,
}

fn suspected_cause(label: &str) -> String {
    let lo = label.to_ascii_lowercase();
    if lo.starts_with("async:") {
        "async race / wakeup timing".to_string()
    } else if lo.contains("hash") || lo.contains("map") {
        "hash-map iteration order".to_string()
    } else if lo.contains("rng") || lo.contains("rand") || lo.contains("seed") {
        "RNG seed sensitivity".to_string()
    } else if lo.contains("lock") || lo.contains("mutex") || lo.contains("sched") {
        "thread scheduling variance".to_string()
    } else {
        "scheduling / cache instability".to_string()
    }
}

/// Compare ≥2 recorded traces of the *same* program and isolate which frames
/// vary in their share of the profile. `threshold_milli` is the spread (in
/// milli-percent) above which a frame is flagged (e.g. 5_000 = 5 percentage
/// points).
pub fn variance(traces: &[Trace], threshold_milli: u64) -> VarianceReport {
    let flames: Vec<FlamegraphReport> = traces.iter().map(flamegraph).collect();
    let total_samples: Vec<u64> = flames.iter().map(|f| f.total_samples).collect();

    // union of all frame labels
    let mut labels: BTreeSet<String> = BTreeSet::new();
    for f in &flames {
        labels.extend(f.frame_total.keys().cloned());
    }

    let mut variant_frames = Vec::new();
    for label in labels {
        let mut min = u64::MAX;
        let mut max = 0u64;
        for (f, total) in flames.iter().zip(&total_samples) {
            let inc = *f.frame_total.get(&label).unwrap_or(&0);
            let p = pct_milli(inc, *total);
            min = min.min(p);
            max = max.max(p);
        }
        let spread = max.saturating_sub(min);
        if spread >= threshold_milli {
            variant_frames.push(VariantFrame {
                label: label.clone(),
                min_pct_milli: min,
                max_pct_milli: max,
                spread_milli: spread,
                suspected_cause: suspected_cause(&label),
            });
        }
    }
    variant_frames.sort_by(|a, b| {
        b.spread_milli
            .cmp(&a.spread_milli)
            .then_with(|| a.label.cmp(&b.label))
    });

    VarianceReport {
        runs: traces.len(),
        total_samples,
        variant_frames,
        threshold_milli,
    }
}

impl VarianceReport {
    pub fn hash_into(&self, h: &mut SeshatHasher) {
        h.write_u64(self.runs as u64);
        h.write_u64(self.threshold_milli);
        for t in &self.total_samples {
            h.write_u64(*t);
        }
        h.write_u64(self.variant_frames.len() as u64);
        for v in &self.variant_frames {
            h.write_str(&v.label);
            h.write_u64(v.min_pct_milli);
            h.write_u64(v.max_pct_milli);
            h.write_u64(v.spread_milli);
            h.write_str(&v.suspected_cause);
        }
    }
    pub fn content_hash(&self) -> u64 {
        let mut h = SeshatHasher::new();
        h.write(b"seshat-variance-v1");
        self.hash_into(&mut h);
        h.finish()
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Feature 12 — "What changed?" regression profiler (two-trace diff)
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameMove {
    pub label: String,
    pub baseline_pct_milli: u64,
    pub candidate_pct_milli: u64,
    /// candidate − baseline, in milli-percent (signed).
    pub delta_milli: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegressionReport {
    pub baseline_samples: u64,
    pub candidate_samples: u64,
    /// Frames whose share moved, |delta| desc. Empty ⇒ no change ⇒ `diff(t, t)`.
    pub movers: Vec<FrameMove>,
    /// candidate − baseline boundary share, milli-percent.
    pub boundary_delta_milli: i64,
    /// candidate − baseline total copy bytes (signed).
    pub copy_bytes_delta: i64,
    pub summary: String,
}

/// Compare two traces and explain the delta. `diff(t, t)` yields zero movers
/// (`tests/prop.rs` property #5).
pub fn diff(baseline: &Trace, candidate: &Trace) -> RegressionReport {
    let bf = flamegraph(baseline);
    let cf = flamegraph(candidate);
    let bb = boundary(baseline);
    let cb = boundary(candidate);
    let bc = copy(baseline);
    let cc = copy(candidate);

    let mut labels: BTreeSet<String> = BTreeSet::new();
    labels.extend(bf.frame_total.keys().cloned());
    labels.extend(cf.frame_total.keys().cloned());

    let mut movers = Vec::new();
    for label in labels {
        let bp = pct_milli(*bf.frame_total.get(&label).unwrap_or(&0), bf.total_samples);
        let cp = pct_milli(*cf.frame_total.get(&label).unwrap_or(&0), cf.total_samples);
        let delta = cp as i64 - bp as i64;
        if delta != 0 {
            movers.push(FrameMove {
                label,
                baseline_pct_milli: bp,
                candidate_pct_milli: cp,
                delta_milli: delta,
            });
        }
    }
    movers.sort_by(|a, b| {
        b.delta_milli
            .abs()
            .cmp(&a.delta_milli.abs())
            .then_with(|| a.label.cmp(&b.label))
    });

    let boundary_delta_milli = pct_milli(cb.boundary_samples, cb.total_samples) as i64
        - pct_milli(bb.boundary_samples, bb.total_samples) as i64;
    let copy_bytes_delta = cc.total_bytes as i64 - bc.total_bytes as i64;

    let summary = if movers.is_empty() && boundary_delta_milli == 0 && copy_bytes_delta == 0 {
        "No significant change between baseline and candidate.".to_string()
    } else {
        let mut s = String::new();
        if let Some(top) = movers.first() {
            let dir = if top.delta_milli > 0 { "increased" } else { "decreased" };
            s.push_str(&format!(
                "Largest shift: {} {} by {}.{:03} pp.",
                top.label,
                dir,
                top.delta_milli.abs() / 1000,
                top.delta_milli.abs() % 1000
            ));
        }
        if boundary_delta_milli != 0 {
            let dir = if boundary_delta_milli > 0 { "more" } else { "less" };
            s.push_str(&format!(
                " Py↔Rust boundary share is {} ({}.{:03} pp).",
                dir,
                boundary_delta_milli.abs() / 1000,
                boundary_delta_milli.abs() % 1000
            ));
        }
        if copy_bytes_delta != 0 {
            s.push_str(&format!(" Copy bytes delta: {}.", copy_bytes_delta));
        }
        s
    };

    RegressionReport {
        baseline_samples: bf.total_samples,
        candidate_samples: cf.total_samples,
        movers,
        boundary_delta_milli,
        copy_bytes_delta,
        summary,
    }
}

impl RegressionReport {
    pub fn hash_into(&self, h: &mut SeshatHasher) {
        h.write_u64(self.baseline_samples);
        h.write_u64(self.candidate_samples);
        h.write_i64(self.boundary_delta_milli);
        h.write_i64(self.copy_bytes_delta);
        h.write_u64(self.movers.len() as u64);
        for m in &self.movers {
            h.write_str(&m.label);
            h.write_u64(m.baseline_pct_milli);
            h.write_u64(m.candidate_pct_milli);
            h.write_i64(m.delta_milli);
        }
    }
    pub fn content_hash(&self) -> u64 {
        let mut h = SeshatHasher::new();
        h.write(b"seshat-regression-v1");
        self.hash_into(&mut h);
        h.finish()
    }
    /// Number of frames whose share changed (regressions + improvements).
    pub fn num_changes(&self) -> usize {
        self.movers.len()
    }
}
