//! The aggregate [`SeshatReport`] over a single trace, plus **Feature 8** —
//! optimization recommendations as a *deterministic rule table* (a predicate
//! over the report → a templated message whose slots are filled from report
//! facts), so the same report always yields the same advice in the same order.

use crate::analyze::{
    self, AsyncReport, BoundaryReport, ContentionReport, CopyReport, FlamegraphReport,
    OwnershipReport, PeakReport, PipelineReport, ThermalReport,
};
use crate::hash::SeshatHasher;
use crate::trace::Trace;

/// Knobs for analysis. Defaults are conservative; all thresholds are in the
/// same integer units as the reports (milli-percent or bytes), so configuring
/// them never introduces float nondeterminism.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnalyzeOptions {
    /// Physical core count for oversubscription detection (0 = unknown).
    pub cores: u32,
    /// Boundary-share (milli-percent) at/above which to recommend buffer-passing.
    pub boundary_hot_milli: u64,
    /// Avoidable-copy bytes at/above which to recommend zero-copy.
    pub copy_avoidable_bytes: u64,
    /// GIL-wait share (milli-percent) at/above which to recommend releasing it.
    pub gil_bound_milli: u64,
    /// Rust lock/channel-wait share (milli-percent) to flag lock contention.
    pub lock_bound_milli: u64,
}

impl Default for AnalyzeOptions {
    fn default() -> Self {
        Self {
            cores: 0,
            boundary_hot_milli: 20_000, // 20%
            copy_avoidable_bytes: 1 << 20, // 1 MiB
            gil_bound_milli: 20_000,    // 20%
            lock_bound_milli: 15_000,   // 15%
        }
    }
}

/// A single evidence-grounded recommendation (Feature 8).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recommendation {
    /// Stable rule code, e.g. `"SES-BOUNDARY-HOT"`.
    pub code: String,
    pub title: String,
    pub detail: String,
    /// The trace facts that triggered the rule (the "grounded in traces" part).
    pub evidence: String,
}

/// The full single-trace report — features 1–8, 10, 11. (Variance/regression
/// are multi-trace; see [`analyze::variance`] / [`analyze::diff`].)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeshatReport {
    pub flamegraph: FlamegraphReport,
    pub boundary: BoundaryReport,
    pub copy: CopyReport,
    pub contention: ContentionReport,
    pub async_stall: AsyncReport,
    pub ownership: OwnershipReport,
    pub peak: PeakReport,
    pub thermal: ThermalReport,
    pub pipeline: PipelineReport,
    pub recommendations: Vec<Recommendation>,
    /// Advisory only — echoed from the trace, excluded from `content_hash`.
    pub wall_ns_total: u64,
}

impl SeshatReport {
    /// Deterministic content hash of every gated field. **Excludes** advisory
    /// wall-clock and all display-only narrative/evidence strings. Two identical
    /// traces produce byte-identical reports and therefore identical hashes.
    pub fn content_hash(&self) -> u64 {
        let mut h = SeshatHasher::new();
        h.write(b"seshat-report-v1");
        self.flamegraph.hash_into(&mut h);
        self.boundary.hash_into(&mut h);
        self.copy.hash_into(&mut h);
        self.contention.hash_into(&mut h);
        self.async_stall.hash_into(&mut h);
        self.ownership.hash_into(&mut h);
        self.peak.hash_into(&mut h);
        self.thermal.hash_into(&mut h);
        self.pipeline.hash_into(&mut h);
        // Recommendation codes are deterministic facts (gated); the prose detail
        // is derived from gated numbers so hashing the code+evidence suffices.
        h.write_u64(self.recommendations.len() as u64);
        for r in &self.recommendations {
            h.write_str(&r.code);
            h.write_str(&r.evidence);
        }
        h.finish()
    }
}

/// Run every single-trace analysis with default options.
pub fn analyze_trace(trace: &Trace) -> SeshatReport {
    analyze_trace_with(trace, &AnalyzeOptions::default())
}

/// Run every single-trace analysis with explicit options.
pub fn analyze_trace_with(trace: &Trace, opts: &AnalyzeOptions) -> SeshatReport {
    let flamegraph = analyze::flamegraph(trace);
    let boundary = analyze::boundary(trace);
    let copy = analyze::copy(trace);
    let contention = analyze::contention(trace);
    let async_stall = analyze::async_stall(trace);
    let ownership = analyze::ownership(trace);
    let peak = analyze::peak(trace);
    let thermal = analyze::thermal(trace, opts.cores);
    let pipeline = analyze::pipeline(trace);

    let recommendations = recommend(
        opts,
        &boundary,
        &copy,
        &contention,
        &peak,
        &thermal,
        &ownership,
    );

    SeshatReport {
        flamegraph,
        boundary,
        copy,
        contention,
        async_stall,
        ownership,
        peak,
        thermal,
        pipeline,
        recommendations,
        wall_ns_total: trace.wall_ns_total,
    }
}

/// Feature 8 — the deterministic rule table. Rules are evaluated in a fixed
/// order; each emits at most one recommendation when its predicate holds.
fn recommend(
    opts: &AnalyzeOptions,
    boundary: &BoundaryReport,
    copy: &CopyReport,
    contention: &ContentionReport,
    peak: &PeakReport,
    thermal: &ThermalReport,
    ownership: &OwnershipReport,
) -> Vec<Recommendation> {
    let mut out = Vec::new();

    // Rule 1 — Py↔Rust boundary is hot.
    let bpct = analyze::pct_milli(boundary.boundary_samples, boundary.total_samples);
    if bpct >= opts.boundary_hot_milli {
        out.push(Recommendation {
            code: "SES-BOUNDARY-HOT".to_string(),
            title: "Python↔Rust boundary dominates runtime".to_string(),
            detail: "A large share of samples are spent crossing the FFI boundary \
                     (object conversion, GIL acquisition, refcount churn). Consider \
                     accepting Arrow/NumPy buffers directly instead of converting \
                     Python objects element-by-element."
                .to_string(),
            evidence: format!(
                "boundary={}.{:03}% of {} samples; {} crossings",
                bpct / 1000,
                bpct % 1000,
                boundary.total_samples,
                boundary.crossings
            ),
        });
    }

    // Rule 2 — avoidable copies.
    if copy.avoidable_bytes >= opts.copy_avoidable_bytes {
        let top = copy
            .flows
            .iter()
            .find(|f| f.avoidable)
            .map(|f| format!("{}→{} ({} bytes)", f.from, f.to, f.bytes))
            .unwrap_or_default();
        out.push(Recommendation {
            code: "SES-COPY-AVOIDABLE".to_string(),
            title: "Avoidable cross-domain copies".to_string(),
            detail: "Buffers are being copied between zero-copy-compatible domains. \
                     Share the buffer (e.g. NumPy↔Arrow↔Rust slice) instead of \
                     materializing a new allocation."
                .to_string(),
            evidence: format!(
                "{} avoidable bytes; largest avoidable flow {}",
                copy.avoidable_bytes, top
            ),
        });
    }

    // Rule 3 — GIL-bound.
    let gpct = analyze::pct_milli(contention.gil_wait, contention.total);
    if gpct >= opts.gil_bound_milli {
        out.push(Recommendation {
            code: "SES-GIL-BOUND".to_string(),
            title: "Workload is GIL-bound".to_string(),
            detail: "Threads spend a large share blocked acquiring the Python GIL. \
                     Release the GIL around heavy Rust compute (PyO3 `allow_threads`) \
                     so Rust work runs in parallel. NOTE: from the pure-Python \
                     recorder this GIL-wait share is a sampling heuristic (a thread \
                     frozen at one frame while another progresses), not an exact \
                     GIL-acquisition measurement; treat it as directional."
                .to_string(),
            evidence: format!(
                "gil_wait={}.{:03}% of {} samples",
                gpct / 1000,
                gpct % 1000,
                contention.total
            ),
        });
    }

    // Rule 4 — Rust lock contention.
    let lpct = analyze::pct_milli(contention.rust_blocked(), contention.total);
    if lpct >= opts.lock_bound_milli {
        out.push(Recommendation {
            code: "SES-LOCK-BOUND".to_string(),
            title: "Rust lock/channel contention".to_string(),
            detail: "Threads spend a large share blocked on Rust mutexes/channels. \
                     Consider sharding the contended structure, using per-thread \
                     accumulators, or a lock-free queue."
                .to_string(),
            evidence: format!(
                "lock+channel wait={}.{:03}% of {} samples",
                lpct / 1000,
                lpct % 1000,
                contention.total
            ),
        });
    }

    // Rule 5 — peak memory held by a dual-owned buffer (ownership transfer).
    if !ownership.transfers.is_empty() && peak.contributors.len() >= 2 {
        // Find the largest transfer (deterministic: BTreeMap iter is sorted; pick
        // max bytes, tie-broken by key).
        let mut best: Option<(&String, &u64)> = None;
        for (k, v) in &ownership.transfers {
            best = match best {
                None => Some((k, v)),
                Some((_, bv)) if v > bv => Some((k, v)),
                other => other,
            };
        }
        if let Some((k, v)) = best {
            out.push(Recommendation {
                code: "SES-PEAK-DUAL-OWNED".to_string(),
                title: "Peak driven by buffers live in two domains at once".to_string(),
                detail: "A buffer was copied across the ownership boundary and both \
                         copies stayed live, inflating peak memory. Free the source \
                         after handoff, or share instead of copying."
                    .to_string(),
                evidence: format!(
                    "peak={} bytes; largest ownership transfer {} = {} bytes",
                    peak.peak_bytes, k, v
                ),
            });
        }
    }

    // Rule 6 — thermal throttle.
    if thermal.throttle_detected {
        out.push(Recommendation {
            code: "SES-THROTTLE".to_string(),
            title: "CPU frequency throttling observed".to_string(),
            detail: "Performance dropped alongside a CPU-frequency decline. Reduce \
                     sustained all-core load or check thermal headroom before \
                     attributing the slowdown to the algorithm."
                .to_string(),
            evidence: format!(
                "freq {}→{} MHz",
                thermal.baseline_mhz, thermal.min_mhz
            ),
        });
    }

    // Rule 7 — thread oversubscription.
    if thermal.oversubscription {
        out.push(Recommendation {
            code: "SES-OVERSUB".to_string(),
            title: "Thread oversubscription".to_string(),
            detail: "More active threads than physical cores increases scheduling \
                     overhead and cache pressure. Match the worker pool to the core \
                     count."
                .to_string(),
            evidence: format!(
                "{} threads on {} cores",
                thermal.distinct_threads, thermal.cores
            ),
        });
    }

    out
}
