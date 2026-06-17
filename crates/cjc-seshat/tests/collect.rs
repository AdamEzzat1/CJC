//! Live-collector tests (feature `collect-live`). These assert the produced
//! trace is **well-formed** — it captures real heap traffic, round-trips, and
//! analyzes without panic — NOT specific timing/sample values, which are
//! inherently nondeterministic for a live recording.
//!
//! Run with: `cargo test -p cjc-seshat --features collect-live --test collect`

#![cfg(feature = "collect-live")]

// The whole test binary profiles itself through the Seshat allocator.
#[global_allocator]
static GLOBAL: cjc_seshat::collect::SeshatAlloc = cjc_seshat::collect::SeshatAlloc;

use cjc_seshat::collect::{mark_boundary, mark_copy, zone, CaptureConfig, Recorder};
use cjc_seshat::{analyze_trace, replay, serialize, OwnershipDomain};

// The collector is a process-global singleton (one `#[global_allocator]` + one
// COLLECTOR), so two recordings must never run concurrently — cargo runs tests
// in parallel by default. Every test in this binary holds this lock for its
// duration, serializing them. `into_inner` recovers from a poisoned lock if a
// prior test panicked.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn live_recording_captures_real_heap_and_round_trips() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let rec = Recorder::start_with_interval_ms(1);

    {
        let _z = zone("build");
        // real allocations: a growing Vec reallocates several times
        let mut v: Vec<u64> = Vec::new();
        for i in 0..200_000u64 {
            v.push(i.wrapping_mul(2654435761));
        }
        std::hint::black_box(&v);
    }
    {
        let _z = zone("handoff");
        mark_boundary("pyo3::handoff");
        mark_copy(OwnershipDomain::RustHeap, OwnershipDomain::NumPy, 1_600_000);
    }

    let trace = rec.finish();

    // ── well-formedness, not timing ──
    assert!(trace.num_events() > 0, "captured at least one event");

    // round-trips through the on-disk format
    let bytes = serialize(&trace);
    let back = replay(&bytes).expect("a live trace must round-trip");
    assert_eq!(trace.content_hash(), back.content_hash());

    // analysis runs and sees the real heap traffic + the explicit markers
    let report = analyze_trace(&trace);
    assert!(
        report.ownership.total_allocated > 0,
        "the allocator shim captured real allocations"
    );
    assert_eq!(report.copy.total_bytes, 1_600_000, "the marked copy was captured");
    assert!(report.copy.flows[0].avoidable, "RustHeap→NumPy is zero-copy compatible");
    assert_eq!(report.boundary.crossings, 1, "the boundary marker was captured");

    // re-analyzing the SAME (now fixed) trace is deterministic, even though the
    // capture that produced it was not.
    assert_eq!(report.content_hash(), analyze_trace(&trace).content_hash());
}

/// A uniquely-named, never-inlined function so its symbol is unmistakable in the
/// captured backtrace. It makes one big allocation that stays live to the end.
#[inline(never)]
fn seshat_unwind_probe_alloc() -> Vec<u64> {
    let mut v: Vec<u64> = Vec::with_capacity(1_000_000); // one ~8 MB block
    for i in 0..1_000_000u64 {
        v.push(i.wrapping_mul(2654435761));
    }
    std::hint::black_box(&v);
    v
}

#[test]
fn alloc_stacks_attribute_to_real_functions() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // With native unwinding ON, allocations are attributed to the REAL Rust
    // function — no manual `zone(...)` needed (the dhat/Memray capability).
    let rec = Recorder::start_with_config(CaptureConfig { interval_ms: 1, alloc_stacks: true });
    let v = seshat_unwind_probe_alloc();
    let trace = rec.finish();
    assert!(v.len() == 1_000_000); // keep the buffer live past `finish`

    // still well-formed + round-trips, like the zone-only path
    let back = replay(&serialize(&trace)).expect("a live trace must round-trip");
    assert_eq!(trace.content_hash(), back.content_hash());
    let report = analyze_trace(&trace);
    assert!(report.ownership.total_allocated > 0);

    // The big block is attributed to the probe function by name — proving the
    // backtrace was captured and symbolized, not just bucketed into a zone.
    let attributed = report
        .peak
        .contributors
        .iter()
        .any(|c| c.frame.contains("seshat_unwind_probe_alloc"));
    assert!(
        attributed,
        "expected the 8 MB allocation attributed to `seshat_unwind_probe_alloc`; \
         peak contributors = {:?}",
        report.peak.contributors
    );
}
