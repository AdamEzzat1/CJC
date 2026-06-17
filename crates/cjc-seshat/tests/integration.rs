//! Seshat integration + determinism + golden tests.
//!
//! These build a rich synthetic cross-language trace (no live collector) and
//! assert each of the 12 features produces the expected structure, that the
//! report is byte-stable across repeated analyses (determinism gate), and that
//! the report survives a serialize→replay round-trip unchanged.

use cjc_seshat::{
    analyze_trace, diff, replay, serialize, variance, FrameKind, OwnershipDomain, ThreadState,
    Trace,
};

/// A deliberately seam-heavy workload: a Python `main` calling through a PyO3
/// boundary into Rust `process_batch`, which allocates, copies Rust→NumPy,
/// contends on the GIL and a lock, runs an async task that stalls, and is
/// observed under a frequency-throttling counter stream — all inside named
/// pipeline zones.
fn workload() -> Trace {
    let mut b = Trace::builder(42);
    let main = b.intern_frame(FrameKind::Py, "main", "app.py", 10);
    let bnd = b.intern_frame(FrameKind::FfiBoundary, "pyo3::process", "ffi.rs", 1);
    let work = b.intern_frame(FrameKind::Rust, "process_batch", "lib.rs", 88);
    let conv = b.intern_frame(FrameKind::Rust, "py_to_struct", "convert.rs", 20);
    let task = b.intern_frame(FrameKind::AsyncTask, "fetch_task", "io.rs", 5);

    // ── pipeline stage: parse ──
    let z_parse = b.zone_start("parse");
    for _ in 0..10 {
        b.sample_running(&[main]);
    }
    b.zone_stop(z_parse);

    // ── pipeline stage: convert (Py→Rust boundary heavy) ──
    let z_conv = b.zone_start("convert");
    for _ in 0..38 {
        // boundary frame on the stack → counts toward boundary inclusive cost
        b.sample_running(&[main, bnd, conv]);
    }
    b.boundary_cross(bnd);
    b.zone_stop(z_conv);

    // ── pipeline stage: compute ──
    let z_comp = b.zone_start("compute");
    for _ in 0..40 {
        b.sample_running(&[main, work]);
    }
    // memory: Rust allocates a big buffer, then copies it into a NumPy array
    // (avoidable — zero-copy compatible), both stay live → peak.
    b.alloc(OwnershipDomain::RustHeap, 4_000_000, work);
    b.alloc(OwnershipDomain::NumPy, 4_000_000, conv);
    b.copy(OwnershipDomain::RustHeap, OwnershipDomain::NumPy, 4_000_000, bnd);
    b.zone_stop(z_comp);

    // ── contention: GIL + Rust lock waits ──
    for _ in 0..8 {
        b.sample(1, ThreadState::GilWait, &[main]);
    }
    for _ in 0..4 {
        b.sample(2, ThreadState::LockWait, &[work]);
    }

    // ── async: a task that stalls then resumes ──
    for _ in 0..3 {
        b.sample(3, ThreadState::AsyncIdle, &[task]);
    }
    b.edge(cjc_seshat::CausalEdge::AwaitResume {
        task,
        waited_ticks: 12,
    });

    // ── thermal: frequency drops 3600 → 3000 MHz (>10% → throttle) ──
    b.counter(0, 3600, 1_000, 2_500);
    b.counter(0, 3000, 9_000, 1_200);

    b.set_wall_ns(123_456_789);
    b.finish()
}

#[test]
fn flamegraph_conserves_samples() {
    let t = workload();
    let r = analyze_trace(&t);
    // 10 parse + 38 convert + 40 compute + 8 gil + 4 lock + 3 async = 103 samples
    assert_eq!(r.flamegraph.total_samples, 103);
    let self_sum: u64 = r.flamegraph.frame_self.values().sum();
    assert_eq!(self_sum, 103, "every sample lands in exactly one leaf");
}

#[test]
fn boundary_cost_is_measured() {
    let r = analyze_trace(&workload());
    // 38 of 103 samples cross the FFI boundary.
    assert_eq!(r.boundary.boundary_samples, 38);
    assert_eq!(r.boundary.crossings, 1);
    let share = cjc_seshat::analyze::pct_milli(r.boundary.boundary_samples, r.boundary.total_samples);
    assert!(share > 36_000 && share < 38_000, "≈36.9%, got {share} milli-%");
}

#[test]
fn copy_detector_flags_avoidable_no_false_positive() {
    let r = analyze_trace(&workload());
    assert_eq!(r.copy.total_bytes, 4_000_000);
    assert_eq!(r.copy.flows.len(), 1, "exactly one copy edge was recorded");
    assert!(r.copy.flows[0].avoidable, "RustHeap→NumPy is zero-copy compatible");
    assert_eq!(r.copy.avoidable_bytes, 4_000_000);
}

#[test]
fn contention_splits_gil_vs_lock() {
    let r = analyze_trace(&workload());
    assert_eq!(r.contention.gil_wait, 8);
    assert_eq!(r.contention.lock_wait, 4);
    assert_eq!(r.contention.rust_blocked(), 4);
}

#[test]
fn async_stalls_attributed_to_task() {
    let r = analyze_trace(&workload());
    assert_eq!(r.async_stall.total_stall_ticks, 3);
    assert_eq!(r.async_stall.total_resumes, 1);
    let stat = r
        .async_stall
        .tasks
        .values()
        .find(|s| s.stall_ticks == 3)
        .expect("the async task accumulated stall ticks");
    assert_eq!(stat.max_wait_ticks, 12);
}

#[test]
fn ownership_partitions_all_bytes() {
    let r = analyze_trace(&workload());
    assert_eq!(r.ownership.total_allocated, 8_000_000);
    let domain_sum: u64 = r.ownership.per_domain_alloc.values().sum();
    assert_eq!(domain_sum, r.ownership.total_allocated, "domains partition total");
    assert_eq!(*r.ownership.transfers.get("RustHeap->NumPy").unwrap(), 4_000_000);
}

#[test]
fn peak_is_explained() {
    let r = analyze_trace(&workload());
    assert_eq!(r.peak.peak_bytes, 8_000_000);
    assert!(!r.peak.contributors.is_empty());
    assert!(r.peak.narrative.contains("Peak live-set"));
}

#[test]
fn thermal_detects_throttle() {
    let r = analyze_trace(&workload());
    assert!(r.thermal.counters_available);
    assert!(r.thermal.throttle_detected, "3600→3000 MHz is >10% drop");
    assert_eq!(r.thermal.baseline_mhz, 3600);
    assert_eq!(r.thermal.min_mhz, 3000);
}

#[test]
fn pipeline_rolls_up_stages() {
    let r = analyze_trace(&workload());
    assert_eq!(*r.pipeline.per_stage.get("parse").unwrap(), 10);
    assert_eq!(*r.pipeline.per_stage.get("convert").unwrap(), 38);
    assert_eq!(*r.pipeline.per_stage.get("compute").unwrap(), 40);
    // the contention/async samples fall outside any zone
    assert_eq!(*r.pipeline.per_stage.get("(unzoned)").unwrap(), 15);
}

#[test]
fn recommendations_are_grounded() {
    let r = analyze_trace(&workload());
    let codes: Vec<&str> = r.recommendations.iter().map(|x| x.code.as_str()).collect();
    assert!(codes.contains(&"SES-BOUNDARY-HOT"), "got {codes:?}");
    assert!(codes.contains(&"SES-COPY-AVOIDABLE"), "got {codes:?}");
    assert!(codes.contains(&"SES-THROTTLE"), "got {codes:?}");
    // every recommendation carries evidence text
    for rec in &r.recommendations {
        assert!(!rec.evidence.is_empty(), "rec {} lacks evidence", rec.code);
    }
}

// ── determinism gate ────────────────────────────────────────────────────────

#[test]
fn report_is_deterministic_across_runs() {
    let t = workload();
    let h1 = analyze_trace(&t).content_hash();
    for _ in 0..8 {
        assert_eq!(analyze_trace(&t).content_hash(), h1);
    }
}

#[test]
fn report_survives_serialize_round_trip() {
    let t = workload();
    let bytes = serialize(&t);
    let back = replay(&bytes).expect("valid trace replays");
    assert_eq!(t.content_hash(), back.content_hash());
    assert_eq!(analyze_trace(&t).content_hash(), analyze_trace(&back).content_hash());
}

/// Golden hash — guards against *unintended* report changes across edits. If a
/// deliberate format/semantics change lands, update this constant in the same
/// commit (and say why in the message).
#[test]
fn golden_report_hash_is_stable() {
    let h = analyze_trace(&workload()).content_hash();
    const GOLDEN: u64 = 0xa4cd_a136_9275_d1ff;
    assert_eq!(h, GOLDEN, "report hash drifted; got {h:#018x}");
}

// ── feature 12: regression diff identity ────────────────────────────────────

#[test]
fn diff_of_identical_traces_is_empty() {
    let t = workload();
    let d = diff(&t, &t);
    assert_eq!(d.num_changes(), 0);
    assert_eq!(d.boundary_delta_milli, 0);
    assert_eq!(d.copy_bytes_delta, 0);
    assert!(d.summary.contains("No significant change"));
}

#[test]
fn diff_detects_a_shift() {
    let base = workload();
    // candidate: same but with extra boundary samples
    let mut b = Trace::builder(42);
    let main = b.intern_frame(FrameKind::Py, "main", "app.py", 10);
    let bnd = b.intern_frame(FrameKind::FfiBoundary, "pyo3::process", "ffi.rs", 1);
    for _ in 0..50 {
        b.sample_running(&[main, bnd]);
    }
    let cand = b.finish();
    let d = diff(&base, &cand);
    assert!(d.num_changes() > 0);
    assert!(d.boundary_delta_milli > 0, "candidate has more boundary share");
}

// ── feature 9: variance across runs ─────────────────────────────────────────

#[test]
fn variance_flags_a_drifting_frame() {
    // two "runs" of the same program where a hashmap-iteration frame takes a
    // very different share of the profile.
    let mk = |hot: usize, cold: usize| {
        let mut b = Trace::builder(0);
        let f_main = b.intern_frame(FrameKind::Rust, "main", "m.rs", 1);
        let f_map = b.intern_frame(FrameKind::Rust, "hashmap_scan", "m.rs", 2);
        for _ in 0..hot {
            b.sample_running(&[f_main, f_map]);
        }
        for _ in 0..cold {
            b.sample_running(&[f_main]);
        }
        b.finish()
    };
    let runs = vec![mk(80, 20), mk(20, 80)];
    let v = variance(&runs, 5_000); // 5pp threshold
    assert_eq!(v.runs, 2);
    let drift = v
        .variant_frames
        .iter()
        .find(|f| f.label.contains("hashmap_scan"))
        .expect("the hashmap frame should be flagged");
    assert!(drift.suspected_cause.contains("hash-map"));
}

#[test]
fn renderers_are_deterministic_and_nonempty() {
    use cjc_seshat::render;
    let r = analyze_trace(&workload());
    let j1 = render::json(&r);
    let j2 = render::json(&r);
    assert_eq!(j1, j2, "JSON render is deterministic");
    assert!(j1.contains("\"content_hash\""));
    assert!(render::text(&r).contains("Seshat report"));
    let svg = render::flamegraph_svg(&r);
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("process_batch"));
}
