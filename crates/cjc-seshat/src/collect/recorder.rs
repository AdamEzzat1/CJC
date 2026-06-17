//! The recording session API: [`Recorder`] (start/finish), [`Zone`] (RAII
//! pipeline-stage scopes), and the live `mark_*` markers. Turns the captured
//! [`RawEvent`] stream into a real [`Trace`].
//!
//! Nondeterministic by nature (wall-clock-sampled), behind `collect-live`. The
//! produced trace is then analyzed by the deterministic engine.

use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::collect::alloc::{Guard, RawEvent, COLLECTOR, NO_ZONE};
use crate::trace::{FrameId, FrameKind, OwnershipDomain, Trace, TraceBuilder};

/// Recording configuration.
#[derive(Clone, Copy, Debug)]
pub struct CaptureConfig {
    /// Sampling interval in milliseconds (clamped to ≥1).
    pub interval_ms: u64,
    /// Capture a native call stack at each allocation/free (dhat/Memray-style)
    /// so memory is attributed to the **real Rust function**, automatically,
    /// instead of just the open `zone(...)`. Off by default — it adds real
    /// overhead (one backtrace per allocation).
    pub alloc_stacks: bool,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        CaptureConfig { interval_ms: 1, alloc_stacks: false }
    }
}

/// An active recording session. Drop or call [`finish`](Recorder::finish) to
/// stop sampling and produce the trace.
///
/// The sampler runs on a background thread that, every `interval`, records a
/// [`Sample`](RawEvent::Sample) of the currently-open [`Zone`] stack. Sample
/// *counts* per zone are the count-based time proxy the deterministic engine
/// analyzes; the wall-clock interval itself is advisory.
pub struct Recorder {
    sampler: Option<JoinHandle<()>>,
    start: Instant,
}

impl Recorder {
    /// Start recording with a 1 ms sampling interval and zone-based attribution.
    pub fn start() -> Recorder {
        Recorder::start_with_config(CaptureConfig::default())
    }

    /// Start recording with a custom sampling interval (clamped to ≥1 ms).
    pub fn start_with_interval_ms(interval_ms: u64) -> Recorder {
        Recorder::start_with_config(CaptureConfig { interval_ms, ..CaptureConfig::default() })
    }

    /// Start recording with an explicit [`CaptureConfig`]. With
    /// `alloc_stacks = true`, allocations are attributed to real Rust functions
    /// via native unwinding (automatic — no manual zones needed for memory).
    pub fn start_with_config(cfg: CaptureConfig) -> Recorder {
        COLLECTOR.set_capture_stacks(cfg.alloc_stacks);
        COLLECTOR.enable();
        let interval = Duration::from_millis(cfg.interval_ms.max(1));
        let sampler = thread::Builder::new()
            .name("seshat-sampler".to_string())
            .spawn(move || {
                while COLLECTOR.is_enabled() {
                    thread::sleep(interval);
                    COLLECTOR.record_sample();
                }
            })
            .ok();
        Recorder {
            sampler,
            start: Instant::now(),
        }
    }

    /// Stop recording, join the sampler, and build the trace.
    pub fn finish(mut self) -> Trace {
        COLLECTOR.disable();
        if let Some(h) = self.sampler.take() {
            let _ = h.join();
        }
        let wall_ns = self.start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let (events, names) = COLLECTOR.drain();
        build_trace(events, names, wall_ns)
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        // If `finish` was not called, still stop the sampler cleanly.
        COLLECTOR.disable();
        if let Some(h) = self.sampler.take() {
            let _ = h.join();
        }
    }
}

/// A RAII pipeline-stage scope. Allocations and samples taken while it is alive
/// are attributed to `name`. Nest them freely.
///
/// ```ignore
/// let _z = cjc_seshat::collect::zone("parse");
/// // ... work attributed to "parse" ...
/// ```
pub struct Zone {
    _private: (),
}

/// Open a named zone; the returned guard closes it on drop.
pub fn zone(name: &str) -> Zone {
    COLLECTOR.enter_zone(name);
    Zone { _private: () }
}

impl Drop for Zone {
    fn drop(&mut self) {
        COLLECTOR.exit_zone();
    }
}

/// Record a cross-domain copy live (e.g. a Rust→NumPy buffer handoff).
pub fn mark_copy(from: OwnershipDomain, to: OwnershipDomain, bytes: u64) {
    COLLECTOR.record_copy(from, to, bytes);
}

/// Record an FFI / Py↔Rust boundary crossing live.
pub fn mark_boundary(name: &str) {
    COLLECTOR.record_boundary(name);
}

/// Re-intern the raw capture into a deterministic [`Trace`].
fn build_trace(events: Vec<RawEvent>, names: Vec<String>, wall_ns: u64) -> Trace {
    // Hold the guard so this builder's own allocations are not re-captured (the
    // recorder is already disabled, but belt-and-suspenders against an in-flight
    // sampler tick).
    let _g = Guard::enter();

    let mut b = Trace::builder(0);
    let zone_frames: Vec<FrameId> = names
        .iter()
        .map(|n| b.intern_frame(FrameKind::Rust, n, "<zone>", 0))
        .collect();
    let unzoned = b.intern_frame(FrameKind::Rust, "rust_alloc", "<heap>", 0);
    let frame_for = |zone: u32, zone_frames: &[FrameId]| -> FrameId {
        if zone == NO_ZONE {
            unzoned
        } else {
            zone_frames
                .get(zone as usize)
                .copied()
                .unwrap_or(unzoned)
        }
    };

    let mut handles: Vec<u64> = Vec::new();
    for ev in events {
        match ev {
            RawEvent::Alloc { bytes, zone, ips } => {
                let frame = if ips.is_empty() {
                    frame_for(zone, &zone_frames)
                } else {
                    resolve_alloc_frame(&mut b, &ips)
                        .unwrap_or_else(|| frame_for(zone, &zone_frames))
                };
                b.alloc(OwnershipDomain::RustHeap, bytes, frame);
            }
            RawEvent::Free { bytes, zone, ips } => {
                let frame = if ips.is_empty() {
                    frame_for(zone, &zone_frames)
                } else {
                    resolve_alloc_frame(&mut b, &ips)
                        .unwrap_or_else(|| frame_for(zone, &zone_frames))
                };
                b.free(OwnershipDomain::RustHeap, bytes, frame);
            }
            RawEvent::Sample { stack } => {
                let s: Vec<FrameId> = stack
                    .iter()
                    .filter_map(|&i| zone_frames.get(i as usize).copied())
                    .collect();
                b.sample_running(&s);
            }
            RawEvent::ZoneStart { zone } => {
                let name = names.get(zone as usize).map(|s| s.as_str()).unwrap_or("<?>");
                let h = b.zone_start(name);
                handles.push(h);
            }
            RawEvent::ZoneStop => {
                if let Some(h) = handles.pop() {
                    b.zone_stop(h);
                }
            }
            RawEvent::Copy { from, to, bytes } => {
                b.copy(from, to, bytes, unzoned);
            }
            RawEvent::Boundary { name } => {
                let n = names.get(name as usize).map(|s| s.as_str()).unwrap_or("<?>");
                let f = b.intern_frame(FrameKind::FfiBoundary, n, "<ffi>", 0);
                b.boundary_cross(f);
            }
        }
    }

    b.set_wall_ns(wall_ns);
    b.finish()
}

/// Symbolize a captured IP stack and intern the **first user frame** as a real
/// Rust frame. Walks outermost-leaf first, skipping the allocator/collector/
/// unwinder machinery, so the allocation lands on the function that actually
/// asked for memory. Best-effort: returns `None` if nothing resolves (then the
/// caller falls back to zone attribution).
fn resolve_alloc_frame(b: &mut TraceBuilder, ips: &[usize]) -> Option<FrameId> {
    for &ip in ips {
        let mut found: Option<(String, String, u32)> = None;
        backtrace::resolve(ip as *mut std::ffi::c_void, |sym| {
            if found.is_some() {
                return;
            }
            let name = sym.name().map(|n| n.to_string()).unwrap_or_default();
            if name.is_empty() || is_runtime_frame(&name) {
                return;
            }
            let file = sym
                .filename()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| "<native>".to_string());
            let line = sym.lineno().unwrap_or(0);
            found = Some((name, file, line));
        });
        if let Some((name, file, line)) = found {
            return Some(b.intern_frame(FrameKind::Rust, &name, &file, line));
        }
    }
    None
}

/// True for frames belonging to the allocator / collector / unwinder plumbing —
/// skipped so attribution reaches the user's real allocating function.
fn is_runtime_frame(name: &str) -> bool {
    const SKIP: &[&str] = &[
        "backtrace::",
        "cjc_seshat::collect",
        "__rust_alloc",
        "__rg_alloc",
        "__rdl_alloc",
        "alloc::alloc::",
        "alloc::raw_vec::",
        "alloc::vec::", // Vec construction plumbing (incl. spec_from_iter/in_place)
        "core::alloc::",
        "GlobalAlloc",
        "SeshatAlloc",
    ];
    SKIP.iter().any(|s| name.contains(s))
}
