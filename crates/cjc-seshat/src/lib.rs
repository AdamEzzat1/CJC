//! # Seshat — a cross-language causal profiler for Python/Rust systems
//!
//! Seshat explains, in one trace, where a mixed Python/Rust program spends CPU,
//! memory, copies, async stalls, GIL contention, Rust lock contention, and
//! deterministic performance variance — and *why*, causally, rather than as a
//! flat list of self-times.
//!
//! ## Architecture: collection ⟂ analysis
//!
//! A profiler measures time, and time is nondeterministic; CJC-Lang demands
//! bit-identical output. Seshat resolves this the way `cjc-cana` does — by
//! splitting the system in two:
//!
//! - **Collection** (feature `collect-live`, the `collect` module): the
//!   nondeterministic, platform-specific probes (Rust `GlobalAlloc` shim,
//!   CPython frame walk, perf_event counters). They only *produce* a
//!   [`Trace`]; nothing else reads wall-clock.
//! - **Analysis** (this crate's default surface): a **pure, deterministic
//!   function of a recorded [`Trace`]**. Same trace → byte-identical
//!   [`SeshatReport`], on every platform, every run.
//!
//! The unit of reproducibility is the `.seshat` trace ([`serialize`] /
//! [`replay`]). Live re-recording of the same program is *not* expected to be
//! byte-identical — that is exactly what [`analyze::variance`] measures.
//!
//! ## The 12 features
//!
//! | # | Feature | Entry point |
//! |---|---------|-------------|
//! | 1 | Cross-language flamegraph | [`analyze::flamegraph`] |
//! | 2 | Python↔Rust boundary cost | [`analyze::boundary`] |
//! | 3 | Copy detector | [`analyze::copy`] |
//! | 4 | GIL + Rust lock contention | [`analyze::contention`] |
//! | 5 | Async-aware stalls | [`analyze::async_stall`] |
//! | 6 | Memory ownership map | [`analyze::ownership`] |
//! | 7 | Peak-memory explanation | [`analyze::peak`] |
//! | 8 | Recommendations with evidence | [`analyze_trace`] → `recommendations` |
//! | 9 | Determinism variance | [`analyze::variance`] |
//! | 10 | Thermal / power-aware | [`analyze::thermal`] |
//! | 11 | Data-pipeline profiler | [`analyze::pipeline`] |
//! | 12 | "What changed?" regression | [`diff`] |
//!
//! ## Quickstart
//!
//! ```
//! use cjc_seshat::{Trace, FrameKind, OwnershipDomain, analyze_trace};
//!
//! let mut b = Trace::builder(/* run_id */ 42);
//! let main = b.intern_frame(FrameKind::Py, "main", "app.py", 10);
//! let bnd  = b.intern_frame(FrameKind::FfiBoundary, "pyo3::call", "ffi.rs", 1);
//! let work = b.intern_frame(FrameKind::Rust, "process_batch", "lib.rs", 88);
//! for _ in 0..38 { b.sample_running(&[main, bnd, work]); } // 38% in boundary
//! for _ in 0..62 { b.sample_running(&[main, work]); }
//! b.alloc(OwnershipDomain::RustHeap, 1 << 20, work);
//! b.copy(OwnershipDomain::RustHeap, OwnershipDomain::NumPy, 1 << 20, bnd);
//! let trace = b.finish();
//!
//! let report = analyze_trace(&trace);
//! assert_eq!(report.flamegraph.total_samples, 100);
//! // deterministic: same trace → same hash, every run, every platform
//! assert_eq!(report.content_hash(), analyze_trace(&trace).content_hash());
//! ```

pub mod analyze;
pub mod dispatch;
pub mod hash;
mod merge;
pub mod render;
pub mod report;
pub mod serialize;
pub mod trace;

#[cfg(feature = "collect-live")]
pub mod collect;

// ─── Primary public surface ──────────────────────────────────────────────────

pub use trace::{
    CausalEdge, Event, Frame, FrameId, FrameKind, OwnershipDomain, StrId, ThreadState, Trace,
    TraceBuilder,
};

pub use analyze::{
    diff, variance, AsyncReport, BoundaryReport, ContentionReport, CopyFlow, CopyReport,
    FlameNode, FlamegraphReport, OwnershipReport, PeakReport, PipelineReport, RegressionReport,
    ThermalReport, VarianceReport,
};

pub use merge::{merge, MergeOptions};

pub use report::{analyze_trace, analyze_trace_with, AnalyzeOptions, Recommendation, SeshatReport};

pub use serialize::{replay, serialize, DecodeError};

pub use dispatch::{dispatch_seshat, is_seshat_builtin, SESHAT_BUILTINS};

pub use hash::SeshatHasher;
