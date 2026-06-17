//! The `.seshat` trace model — the deterministic, content-addressed data model
//! that *every* analysis is a pure function of.
//!
//! ## The central determinism decision
//!
//! A profiler measures time; time is nondeterministic; CJC-Lang demands
//! bit-identical output. The resolution (see
//! `docs/seshat/SESHAT_IMPLEMENTATION_PROMPT.md`):
//!
//! - **Ordering & structure use a LOGICAL clock.** The "time" of an event is
//!   simply its index in [`Trace::events`] — a monotonic `seq: u64`. Analyses
//!   order, group, and hash by `seq` and interned ids, never by wall-clock.
//! - **Reproducible attribution uses COUNTS, not durations.** "boundary is 38%
//!   of the profile" is computed from sample counts, which are stable for a
//!   recorded trace, and is in the determinism gate.
//! - **Wall-clock is a single advisory scalar** ([`Trace::wall_ns_total`]),
//!   recorded for display but EXCLUDED from [`Trace::content_hash`].
//!
//! The collectors (feature `collect-live`) are the only place wall-clock or OS
//! thread ids may appear; they map them to logical `seq` / `u32` thread ids
//! before anything reaches this model.

use std::collections::BTreeMap;

use crate::hash::SeshatHasher;

/// Index into the trace's frame table.
pub type FrameId = u32;
/// Index into the trace's string table.
pub type StrId = u32;

/// What language/layer a stack frame belongs to. The whole point of Seshat is
/// that these coexist in one merged stack.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum FrameKind {
    /// A CPython interpreter frame.
    Py,
    /// A Rust frame.
    Rust,
    /// A native C/C++ frame (e.g. a C extension, libc, BLAS).
    Native,
    /// A Python↔Rust / FFI boundary marker (PyO3, ctypes, cffi).
    FfiBoundary,
    /// An async task root (Tokio task or asyncio coroutine).
    AsyncTask,
}

impl FrameKind {
    /// Stable tag byte for hashing/serialization. Never reorder these.
    pub fn tag(self) -> u8 {
        match self {
            FrameKind::Py => 0,
            FrameKind::Rust => 1,
            FrameKind::Native => 2,
            FrameKind::FfiBoundary => 3,
            FrameKind::AsyncTask => 4,
        }
    }

    /// Reconstruct from a tag byte (serialization). Returns `None` if unknown.
    pub fn from_tag(t: u8) -> Option<FrameKind> {
        Some(match t {
            0 => FrameKind::Py,
            1 => FrameKind::Rust,
            2 => FrameKind::Native,
            3 => FrameKind::FfiBoundary,
            4 => FrameKind::AsyncTask,
            _ => return None,
        })
    }

    /// Short human label used as a prefix in frame labels.
    pub fn label(self) -> &'static str {
        match self {
            FrameKind::Py => "py",
            FrameKind::Rust => "rust",
            FrameKind::Native => "native",
            FrameKind::FfiBoundary => "ffi",
            FrameKind::AsyncTask => "async",
        }
    }

    /// Parse from a `.cjcl`/Python-facing string (case-insensitive).
    pub fn from_str(s: &str) -> Option<FrameKind> {
        Some(match s.to_ascii_lowercase().as_str() {
            "py" | "python" => FrameKind::Py,
            "rust" => FrameKind::Rust,
            "native" | "c" | "cpp" => FrameKind::Native,
            "ffi" | "boundary" => FrameKind::FfiBoundary,
            "async" | "task" => FrameKind::AsyncTask,
            _ => return None,
        })
    }
}

/// Where a byte originates / is owned. Seshat's unique angle vs Memray is
/// tracking *ownership transfer* of these domains across the Py↔Rust seam.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum OwnershipDomain {
    /// CPython object heap.
    PyHeap,
    /// Rust global allocator.
    RustHeap,
    /// `mmap`-backed memory.
    Mmap,
    /// A NumPy `ndarray` buffer.
    NumPy,
    /// An Apache Arrow buffer.
    Arrow,
    /// A CJC-Lang / framework tensor buffer.
    Tensor,
    /// GPU device memory.
    Gpu,
    /// A native extension's own heap.
    NativeExt,
}

impl OwnershipDomain {
    /// Stable tag byte. Never reorder.
    pub fn tag(self) -> u8 {
        match self {
            OwnershipDomain::PyHeap => 0,
            OwnershipDomain::RustHeap => 1,
            OwnershipDomain::Mmap => 2,
            OwnershipDomain::NumPy => 3,
            OwnershipDomain::Arrow => 4,
            OwnershipDomain::Tensor => 5,
            OwnershipDomain::Gpu => 6,
            OwnershipDomain::NativeExt => 7,
        }
    }

    /// Reconstruct from a tag byte. `None` if unknown.
    pub fn from_tag(t: u8) -> Option<OwnershipDomain> {
        Some(match t {
            0 => OwnershipDomain::PyHeap,
            1 => OwnershipDomain::RustHeap,
            2 => OwnershipDomain::Mmap,
            3 => OwnershipDomain::NumPy,
            4 => OwnershipDomain::Arrow,
            5 => OwnershipDomain::Tensor,
            6 => OwnershipDomain::Gpu,
            7 => OwnershipDomain::NativeExt,
            _ => return None,
        })
    }

    /// Canonical label used as a map key in reports.
    pub fn label(self) -> &'static str {
        match self {
            OwnershipDomain::PyHeap => "PyHeap",
            OwnershipDomain::RustHeap => "RustHeap",
            OwnershipDomain::Mmap => "Mmap",
            OwnershipDomain::NumPy => "NumPy",
            OwnershipDomain::Arrow => "Arrow",
            OwnershipDomain::Tensor => "Tensor",
            OwnershipDomain::Gpu => "Gpu",
            OwnershipDomain::NativeExt => "NativeExt",
        }
    }

    /// Parse from a `.cjcl`/Python-facing string (case-insensitive).
    pub fn from_str(s: &str) -> Option<OwnershipDomain> {
        Some(match s.to_ascii_lowercase().as_str() {
            "pyheap" | "python" | "py" => OwnershipDomain::PyHeap,
            "rustheap" | "rust" => OwnershipDomain::RustHeap,
            "mmap" => OwnershipDomain::Mmap,
            "numpy" | "np" => OwnershipDomain::NumPy,
            "arrow" => OwnershipDomain::Arrow,
            "tensor" => OwnershipDomain::Tensor,
            "gpu" | "cuda" => OwnershipDomain::Gpu,
            "nativeext" | "native" => OwnershipDomain::NativeExt,
            _ => return None,
        })
    }

    /// Pairs that can be handed off zero-copy. A `Copy` whose `(from, to)` is in
    /// this set (either direction) is flagged "avoidable" by the copy detector.
    pub fn zero_copy_compatible(a: OwnershipDomain, b: OwnershipDomain) -> bool {
        use OwnershipDomain::*;
        let pair = (a, b);
        matches!(
            pair,
            (NumPy, Arrow)
                | (Arrow, NumPy)
                | (Arrow, RustHeap)
                | (RustHeap, Arrow)
                | (NumPy, Tensor)
                | (Tensor, NumPy)
                | (RustHeap, Tensor)
                | (Tensor, RustHeap)
                | (NumPy, RustHeap)
                | (RustHeap, NumPy)
        )
    }
}

/// What a thread was doing when a sample was taken. Drives the contention
/// (GIL vs lock) and async-stall analyses — all from sample *counts*.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum ThreadState {
    /// On-CPU, doing work.
    Running,
    /// Blocked waiting to acquire the Python GIL.
    GilWait,
    /// Blocked on a Rust mutex / `RwLock`.
    LockWait,
    /// Blocked on a channel send/recv.
    ChannelWait,
    /// Blocked on I/O.
    IoWait,
    /// An async task parked at an `.await` point.
    AsyncIdle,
}

impl ThreadState {
    pub fn tag(self) -> u8 {
        match self {
            ThreadState::Running => 0,
            ThreadState::GilWait => 1,
            ThreadState::LockWait => 2,
            ThreadState::ChannelWait => 3,
            ThreadState::IoWait => 4,
            ThreadState::AsyncIdle => 5,
        }
    }
    pub fn from_tag(t: u8) -> Option<ThreadState> {
        Some(match t {
            0 => ThreadState::Running,
            1 => ThreadState::GilWait,
            2 => ThreadState::LockWait,
            3 => ThreadState::ChannelWait,
            4 => ThreadState::IoWait,
            5 => ThreadState::AsyncIdle,
            _ => return None,
        })
    }
    pub fn from_str(s: &str) -> Option<ThreadState> {
        Some(match s.to_ascii_lowercase().as_str() {
            "running" | "run" => ThreadState::Running,
            "gil" | "gilwait" => ThreadState::GilWait,
            "lock" | "lockwait" => ThreadState::LockWait,
            "channel" | "channelwait" => ThreadState::ChannelWait,
            "io" | "iowait" => ThreadState::IoWait,
            "async" | "asyncidle" | "idle" => ThreadState::AsyncIdle,
            _ => return None,
        })
    }
}

/// A stack frame in the interned frame table.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Frame {
    pub kind: FrameKind,
    pub name: StrId,
    pub file: StrId,
    pub line: u32,
}

/// A causal edge connecting two points in the logical timeline. These are what
/// make Seshat *causal* rather than just a flat profile.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum CausalEdge {
    /// `by` woke up async task `task`.
    Wakeup { task: FrameId, by: FrameId },
    /// `task` resumed from an `.await` after stalling `waited_ticks` logical
    /// ticks.
    AwaitResume { task: FrameId, waited_ticks: u64 },
    /// The GIL was handed from `from` to `to`.
    GilHandoff { from: FrameId, to: FrameId },
    /// `bytes` were copied from one ownership domain to another at `frame`.
    Copy {
        from: OwnershipDomain,
        to: OwnershipDomain,
        bytes: u64,
        frame: FrameId,
    },
    /// Control crossed an FFI boundary frame.
    BoundaryCross { boundary: FrameId },
}

/// One event on the logical timeline. The event's `seq` is its index in
/// [`Trace::events`]; there is no separate timestamp field by design.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Event {
    /// A periodic stack sample. `stack[0]` is outermost, `stack[last]` is the
    /// leaf (on-CPU) frame.
    Sample {
        thread: u32,
        state: ThreadState,
        stack: Vec<FrameId>,
    },
    /// An allocation of `bytes` in `domain`, attributed to `frame`.
    Alloc {
        domain: OwnershipDomain,
        bytes: u64,
        frame: FrameId,
    },
    /// A free of `bytes` from `domain`, attributed to `frame`.
    Free {
        domain: OwnershipDomain,
        bytes: u64,
        frame: FrameId,
    },
    /// A hardware/OS counter sample (perf_event / RDPMC). Drives thermal mode.
    /// `ipc_milli` is instructions-per-cycle × 1000 (integer to stay
    /// deterministic; floats never enter the hashed model).
    Counter {
        thread: u32,
        freq_mhz: u32,
        cache_misses: u64,
        ipc_milli: u32,
    },
    /// Open a named profiling zone (pipeline stage). Returns `handle` to the
    /// emitter; matched by [`Event::ZoneStop`].
    ZoneStart { name: StrId, handle: u64 },
    /// Close the zone opened with `handle`.
    ZoneStop { handle: u64 },
    /// A causal edge.
    Edge(CausalEdge),
}

/// A complete, content-addressed `.seshat` trace.
///
/// Construct via [`Trace::builder`]. The string and frame tables are interned;
/// events reference them by id. [`Trace::content_hash`] is a pure function of
/// the tables + event stream and **excludes** the advisory wall-clock scalar.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Trace {
    pub(crate) strings: Vec<String>,
    pub(crate) frames: Vec<Frame>,
    pub(crate) events: Vec<Event>,
    /// Advisory only — total wall-clock nanoseconds of the recording. Excluded
    /// from [`content_hash`](Trace::content_hash). Never used for ordering.
    pub wall_ns_total: u64,
    /// Advisory run label (e.g. a seed). Excluded from `content_hash`.
    pub run_id: u64,
}

impl Trace {
    /// Start building a trace. `run_id` is an advisory label.
    pub fn builder(run_id: u64) -> TraceBuilder {
        TraceBuilder::new(run_id)
    }

    /// Number of events (= the max logical `seq` + 1).
    pub fn num_events(&self) -> usize {
        self.events.len()
    }

    /// Read-only view of the event stream.
    pub fn events(&self) -> &[Event] {
        &self.events
    }

    /// Resolve a string id. Out-of-range ids resolve to `"<?>"` so analyses
    /// never panic on a hand-built or partially-decoded trace.
    pub fn string(&self, id: StrId) -> &str {
        self.strings.get(id as usize).map(|s| s.as_str()).unwrap_or("<?>")
    }

    /// Resolve a frame id. Out-of-range ids yield a sentinel frame.
    pub fn frame(&self, id: FrameId) -> Frame {
        self.frames
            .get(id as usize)
            .cloned()
            .unwrap_or(Frame {
                kind: FrameKind::Native,
                name: u32::MAX,
                file: u32::MAX,
                line: 0,
            })
    }

    /// A stable, human + machine label for a frame: `"kind:name (file:line)"`.
    /// This is the key analyses group by, so it is also what makes report
    /// hashes independent of interning-id values.
    pub fn frame_label(&self, id: FrameId) -> String {
        let f = self.frame(id);
        let name = if f.name == u32::MAX { "<?>" } else { self.string(f.name) };
        let file = if f.file == u32::MAX { "<?>" } else { self.string(f.file) };
        format!("{}:{} ({}:{})", f.kind.label(), name, file, f.line)
    }

    /// Deterministic 64-bit content hash. Excludes `wall_ns_total` and `run_id`
    /// (the advisory channel). Two traces with identical tables + event stream
    /// hash identically on every platform.
    pub fn content_hash(&self) -> u64 {
        let mut h = SeshatHasher::new();
        h.write(b"seshat-trace-v1");
        h.write_u64(self.strings.len() as u64);
        for s in &self.strings {
            h.write_str(s);
        }
        h.write_u64(self.frames.len() as u64);
        for f in &self.frames {
            h.write_u8(f.kind.tag());
            h.write_u32(f.name);
            h.write_u32(f.file);
            h.write_u32(f.line);
        }
        h.write_u64(self.events.len() as u64);
        for e in &self.events {
            hash_event(&mut h, e);
        }
        h.finish()
    }
}

/// Hash a single event (used by both `content_hash` and serialization sanity).
fn hash_event(h: &mut SeshatHasher, e: &Event) {
    match e {
        Event::Sample { thread, state, stack } => {
            h.write_u8(0);
            h.write_u32(*thread);
            h.write_u8(state.tag());
            h.write_u64(stack.len() as u64);
            for &f in stack {
                h.write_u32(f);
            }
        }
        Event::Alloc { domain, bytes, frame } => {
            h.write_u8(1);
            h.write_u8(domain.tag());
            h.write_u64(*bytes);
            h.write_u32(*frame);
        }
        Event::Free { domain, bytes, frame } => {
            h.write_u8(2);
            h.write_u8(domain.tag());
            h.write_u64(*bytes);
            h.write_u32(*frame);
        }
        Event::Counter { thread, freq_mhz, cache_misses, ipc_milli } => {
            h.write_u8(3);
            h.write_u32(*thread);
            h.write_u32(*freq_mhz);
            h.write_u64(*cache_misses);
            h.write_u32(*ipc_milli);
        }
        Event::ZoneStart { name, handle } => {
            h.write_u8(4);
            h.write_u32(*name);
            h.write_u64(*handle);
        }
        Event::ZoneStop { handle } => {
            h.write_u8(5);
            h.write_u64(*handle);
        }
        Event::Edge(edge) => {
            h.write_u8(6);
            hash_edge(h, edge);
        }
    }
}

fn hash_edge(h: &mut SeshatHasher, e: &CausalEdge) {
    match e {
        CausalEdge::Wakeup { task, by } => {
            h.write_u8(0);
            h.write_u32(*task);
            h.write_u32(*by);
        }
        CausalEdge::AwaitResume { task, waited_ticks } => {
            h.write_u8(1);
            h.write_u32(*task);
            h.write_u64(*waited_ticks);
        }
        CausalEdge::GilHandoff { from, to } => {
            h.write_u8(2);
            h.write_u32(*from);
            h.write_u32(*to);
        }
        CausalEdge::Copy { from, to, bytes, frame } => {
            h.write_u8(3);
            h.write_u8(from.tag());
            h.write_u8(to.tag());
            h.write_u64(*bytes);
            h.write_u32(*frame);
        }
        CausalEdge::BoundaryCross { boundary } => {
            h.write_u8(4);
            h.write_u32(*boundary);
        }
    }
}

/// Incrementally builds a [`Trace`]. Strings and frames are interned (deduped);
/// `seq` is implicit (event index). Zone handles are assigned from a
/// deterministic per-builder counter.
#[derive(Debug, Clone)]
pub struct TraceBuilder {
    strings: Vec<String>,
    str_index: BTreeMap<String, StrId>,
    frames: Vec<Frame>,
    frame_index: BTreeMap<Frame, FrameId>,
    events: Vec<Event>,
    wall_ns_total: u64,
    run_id: u64,
    next_handle: u64,
}

impl TraceBuilder {
    /// Fresh builder. `run_id` is an advisory label (excluded from hashing).
    pub fn new(run_id: u64) -> Self {
        Self {
            strings: Vec::new(),
            str_index: BTreeMap::new(),
            frames: Vec::new(),
            frame_index: BTreeMap::new(),
            events: Vec::new(),
            wall_ns_total: 0,
            run_id,
            next_handle: 1,
        }
    }

    /// Intern a string, returning its stable id. Deduped — the same string
    /// always returns the same id within a builder.
    pub fn intern_str(&mut self, s: &str) -> StrId {
        if let Some(&id) = self.str_index.get(s) {
            return id;
        }
        let id = self.strings.len() as StrId;
        self.strings.push(s.to_string());
        self.str_index.insert(s.to_string(), id);
        id
    }

    /// Intern a frame, returning its stable id. Deduped by (kind, name, file,
    /// line).
    pub fn intern_frame(&mut self, kind: FrameKind, name: &str, file: &str, line: u32) -> FrameId {
        let name = self.intern_str(name);
        let file = self.intern_str(file);
        let f = Frame { kind, name, file, line };
        if let Some(&id) = self.frame_index.get(&f) {
            return id;
        }
        let id = self.frames.len() as FrameId;
        self.frames.push(f.clone());
        self.frame_index.insert(f, id);
        id
    }

    /// Record a stack sample with an explicit thread id and state.
    pub fn sample(&mut self, thread: u32, state: ThreadState, stack: &[FrameId]) {
        self.events.push(Event::Sample {
            thread,
            state,
            stack: stack.to_vec(),
        });
    }

    /// Convenience: a `Running` sample on thread 0.
    pub fn sample_running(&mut self, stack: &[FrameId]) {
        self.sample(0, ThreadState::Running, stack);
    }

    /// Record an allocation.
    pub fn alloc(&mut self, domain: OwnershipDomain, bytes: u64, frame: FrameId) {
        self.events.push(Event::Alloc { domain, bytes, frame });
    }

    /// Record a free.
    pub fn free(&mut self, domain: OwnershipDomain, bytes: u64, frame: FrameId) {
        self.events.push(Event::Free { domain, bytes, frame });
    }

    /// Record a hardware/OS counter sample.
    pub fn counter(&mut self, thread: u32, freq_mhz: u32, cache_misses: u64, ipc_milli: u32) {
        self.events.push(Event::Counter {
            thread,
            freq_mhz,
            cache_misses,
            ipc_milli,
        });
    }

    /// Open a profiling zone; returns a handle to pass to [`zone_stop`].
    pub fn zone_start(&mut self, name: &str) -> u64 {
        let name = self.intern_str(name);
        let handle = self.next_handle;
        self.next_handle += 1;
        self.events.push(Event::ZoneStart { name, handle });
        handle
    }

    /// Close a previously-opened zone.
    pub fn zone_stop(&mut self, handle: u64) {
        self.events.push(Event::ZoneStop { handle });
    }

    /// Record a causal edge.
    pub fn edge(&mut self, edge: CausalEdge) {
        self.events.push(Event::Edge(edge));
    }

    /// Convenience: record a cross-domain copy.
    pub fn copy(&mut self, from: OwnershipDomain, to: OwnershipDomain, bytes: u64, frame: FrameId) {
        self.edge(CausalEdge::Copy { from, to, bytes, frame });
    }

    /// Convenience: record an FFI boundary crossing.
    pub fn boundary_cross(&mut self, boundary: FrameId) {
        self.edge(CausalEdge::BoundaryCross { boundary });
    }

    /// Set the advisory wall-clock total (nanoseconds).
    pub fn set_wall_ns(&mut self, ns: u64) {
        self.wall_ns_total = ns;
    }

    /// Number of events recorded so far (introspection for builtins/tests).
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Finalize into an immutable [`Trace`].
    pub fn finish(self) -> Trace {
        Trace {
            strings: self.strings,
            frames: self.frames,
            events: self.events,
            wall_ns_total: self.wall_ns_total,
            run_id: self.run_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interning_dedups() {
        let mut b = Trace::builder(0);
        let a1 = b.intern_str("foo");
        let a2 = b.intern_str("foo");
        assert_eq!(a1, a2);
        let f1 = b.intern_frame(FrameKind::Rust, "f", "a.rs", 1);
        let f2 = b.intern_frame(FrameKind::Rust, "f", "a.rs", 1);
        assert_eq!(f1, f2);
        let f3 = b.intern_frame(FrameKind::Rust, "f", "a.rs", 2);
        assert_ne!(f1, f3);
    }

    #[test]
    fn content_hash_excludes_wall_clock() {
        let mut a = Trace::builder(0);
        let fa = a.intern_frame(FrameKind::Rust, "f", "a.rs", 1);
        a.sample_running(&[fa]);
        a.set_wall_ns(1_000);
        let ta = a.finish();

        let mut b = Trace::builder(999); // different advisory run_id
        let fb = b.intern_frame(FrameKind::Rust, "f", "a.rs", 1);
        b.sample_running(&[fb]);
        b.set_wall_ns(9_999_999); // different advisory wall-clock
        let tb = b.finish();

        assert_eq!(ta.content_hash(), tb.content_hash());
    }

    #[test]
    fn content_hash_sensitive_to_events() {
        let mut a = Trace::builder(0);
        let fa = a.intern_frame(FrameKind::Rust, "f", "a.rs", 1);
        a.sample_running(&[fa]);
        let ta = a.finish();

        let mut b = Trace::builder(0);
        let fb = b.intern_frame(FrameKind::Rust, "g", "a.rs", 1);
        b.sample_running(&[fb]);
        let tb = b.finish();

        assert_ne!(ta.content_hash(), tb.content_hash());
    }

    #[test]
    fn zone_handles_are_sequential_and_deterministic() {
        let mut b = Trace::builder(0);
        let h1 = b.zone_start("parse");
        let h2 = b.zone_start("validate");
        assert_eq!(h1, 1);
        assert_eq!(h2, 2);
    }
}
