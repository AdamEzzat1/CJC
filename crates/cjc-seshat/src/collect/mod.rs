//! Live collectors (feature `collect-live`) — the nondeterministic, quarantined
//! capture layer. **These only produce [`Trace`](crate::Trace)s**; the analysis
//! engine never reads wall-clock.
//!
//! What's here today (minimal Rust in-process collector):
//! - [`SeshatAlloc`] — a `#[global_allocator]` shim capturing real heap traffic.
//! - [`Recorder`] — a recording session with a background zone-stack sampler.
//! - [`zone`] / [`Zone`] — RAII pipeline-stage scopes for attribution.
//! - [`mark_copy`] / [`mark_boundary`] — live causal-edge markers.
//!
//! ```ignore
//! // In your binary:
//! #[global_allocator]
//! static GLOBAL: cjc_seshat::collect::SeshatAlloc = cjc_seshat::collect::SeshatAlloc;
//!
//! let rec = cjc_seshat::collect::Recorder::start();
//! {
//!     let _z = cjc_seshat::collect::zone("compute");
//!     // ... real Rust work; allocations are captured automatically ...
//! }
//! let trace = rec.finish();                 // a real .seshat trace
//! let report = cjc_seshat::analyze_trace(&trace);
//! ```
//!
//! Honest scope: the sampler records the *zone* stack, not native unwound
//! frames, so the flamegraph is a timeline of your declared regions (zero-dep,
//! cross-platform). Allocation capture is automatic and exact. CPython frame
//! walking, perf_event counters, and the PyO3 boundary probe remain deferred.

mod alloc;
mod recorder;

pub use alloc::SeshatAlloc;
pub use recorder::{
    mark_boundary, mark_copy, mark_host, native_sample, zone, CaptureConfig, Recorder, Zone,
};
