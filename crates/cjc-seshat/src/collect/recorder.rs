//! The recording session API: [`Recorder`] (start/finish), [`Zone`] (RAII
//! pipeline-stage scopes), and the live `mark_*` markers. Turns the captured
//! [`RawEvent`] stream into a real [`Trace`].
//!
//! Nondeterministic by nature (wall-clock-sampled), behind `collect-live`. The
//! produced trace is then analyzed by the deterministic engine.

use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::collect::alloc::{Guard, RawEvent, COLLECTOR, NO_ZONE};
use crate::trace::{FrameId, FrameKind, OwnershipDomain, Trace};

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
    /// Start recording with a 1 ms sampling interval.
    pub fn start() -> Recorder {
        Recorder::start_with_interval_ms(1)
    }

    /// Start recording with a custom sampling interval (clamped to ≥1 ms).
    pub fn start_with_interval_ms(interval_ms: u64) -> Recorder {
        COLLECTOR.enable();
        let interval = Duration::from_millis(interval_ms.max(1));
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
            RawEvent::Alloc { bytes, zone } => {
                b.alloc(OwnershipDomain::RustHeap, bytes, frame_for(zone, &zone_frames));
            }
            RawEvent::Free { bytes, zone } => {
                b.free(OwnershipDomain::RustHeap, bytes, frame_for(zone, &zone_frames));
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
