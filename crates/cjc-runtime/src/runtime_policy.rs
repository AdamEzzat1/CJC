//! Runtime Policy Layer — deterministic, thermally-bounded execution policy.
//!
//! This module is the "green compute" control surface for CJC-Lang. It lets a
//! run declare *how much machine* it is willing to use — thread caps, batch
//! sizing, audit depth — and exposes a **deterministic** energy estimate so a
//! program can reason about joules-per-result instead of merely wall-clock
//! seconds.
//!
//! The guiding philosophy: do not let CJC-Lang blindly saturate the CPU. Use
//! *deterministic bounded execution*. Thermal/energy limits are made explicit
//! and deterministic rather than left to the OS scheduler.
//!
//! # Builtins (registered in [`crate::builtins`])
//!
//! Policy query / mutate:
//! - `runtime_policy_thermal_mode() -> String`
//! - `runtime_policy_set_thermal_mode(mode: String) -> String`
//! - `runtime_policy_threads() -> Int`           (resolved effective cap)
//! - `runtime_policy_set_threads(n: Int) -> Int`
//! - `runtime_policy_batch_size() -> Int`
//! - `runtime_policy_set_batch_size(n: Int) -> Int`
//! - `runtime_policy_audit_mode() -> String`
//! - `runtime_policy_set_audit_mode(mode: String) -> String`
//! - `runtime_policy_numeric_mode() -> String`
//! - `runtime_policy_set_numeric_mode(mode: String) -> String`
//! - `runtime_policy_reset() -> Int`
//! - `runtime_policy_summary() -> String`
//!
//! Energy model:
//! - `energy_estimate(flops: Int, bytes: Int) -> Float`   (joules)
//! - `energy_per_flop() -> Float`
//! - `energy_per_byte() -> Float`
//!
//! # Determinism story
//!
//! Two invariants make this layer safe under Prime Directive #3
//! (same seed = bit-identical output):
//!
//! 1. **Thread count never changes results.** The parallel kernels in
//!    [`crate::tensor`] reduce with Kahan / [`crate::accumulator`] binned
//!    summation over a *fixed chunk order*, so the numeric output is identical
//!    regardless of how many rayon workers are live. The thermal mode and
//!    thread cap therefore move *only* the performance/heat axis, never the
//!    answer axis. Capping threads is pure "deterministic bounded execution".
//!
//! 2. **Energy is estimated from workload counts, never from wall time.**
//!    [`energy_estimate_joules`] is a pure function of integer FLOP and byte
//!    counts times fixed documented constants. Wall-clock time is explicitly
//!    *not* an input, because it varies run-to-run and would poison
//!    determinism. Same program + same seed → same FLOP count → same joule
//!    estimate, bit-for-bit. The two multiplies are kept in separate `let`
//!    bindings so the compiler cannot contract them into a single FMA (the
//!    same no-FMA discipline the SIMD kernels follow).
//!
//! The policy itself lives in a thread-local `RefCell<RuntimePolicy>`, mirroring
//! the [`crate::profile`] counter sink. The interpreter thread reads and writes
//! it; the actual rayon thread cap is applied once per process by
//! [`apply_thread_cap`] (the CLI calls this at startup) because rayon's global
//! pool can only be configured once. No RNG is touched. No `HashMap` is used.

use std::cell::RefCell;

// ── Mode enums ───────────────────────────────────────────────────────────

/// Determinism guarantee level. `Strict` is the only mode that ships today;
/// `Relaxed` is reserved so the field exists in the policy surface without
/// implying it weakens any current guarantee.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Determinism {
    /// Bit-identical output across runs and platforms (the default).
    Strict,
    /// Reserved — does not currently relax any guarantee.
    Relaxed,
}

impl Determinism {
    pub fn as_str(self) -> &'static str {
        match self {
            Determinism::Strict => "strict",
            Determinism::Relaxed => "relaxed",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "strict" => Some(Determinism::Strict),
            "relaxed" => Some(Determinism::Relaxed),
            _ => None,
        }
    }
}

/// Floating-point reduction strategy. Maps to the existing accumulator family;
/// every variant preserves determinism (none enables FMA or random ordering).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericMode {
    /// Compensated (Kahan) summation — the default scalar strategy.
    Kahan,
    /// Order-invariant binned accumulator — best for parallel reductions.
    Binned,
    /// Fixed pairwise reduction tree — deterministic divide-and-conquer.
    FixedTree,
}

impl NumericMode {
    pub fn as_str(self) -> &'static str {
        match self {
            NumericMode::Kahan => "kahan",
            NumericMode::Binned => "binned",
            NumericMode::FixedTree => "fixed-tree",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "kahan" => Some(NumericMode::Kahan),
            "binned" => Some(NumericMode::Binned),
            "fixed-tree" | "fixedtree" | "fixed_tree" => Some(NumericMode::FixedTree),
            _ => None,
        }
    }
}

/// Audit/forensics depth. Controls how much *cold-path* work (logs, Merkle
/// trees, full lineage) runs alongside the hot numerical path. Deeper modes
/// trade speed and energy for traceability; they never change numeric output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuditMode {
    /// Cheapest — aggregate summaries only.
    Summary,
    /// Per-operation audit records.
    Full,
    /// Maximum traceability (full lineage + hashes). Most expensive.
    Forensic,
}

impl AuditMode {
    pub fn as_str(self) -> &'static str {
        match self {
            AuditMode::Summary => "summary",
            AuditMode::Full => "full",
            AuditMode::Forensic => "forensic",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "summary" => Some(AuditMode::Summary),
            "full" => Some(AuditMode::Full),
            "forensic" => Some(AuditMode::Forensic),
            _ => None,
        }
    }
}

/// Thermal/energy execution profile. This is the headline "green" knob: it
/// bounds how aggressively a run uses the CPU so a laptop does not cook itself
/// sustaining turbo across all cores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThermalMode {
    /// Gentle on the machine — quarter of cores, small batches, summary audit.
    Cool,
    /// Laptop-safe default — half the cores, normal batches, full audit.
    Balanced,
    /// Benchmark mode — all cores, large batches, minimal audit overhead.
    MaxPerf,
}

impl ThermalMode {
    pub fn as_str(self) -> &'static str {
        match self {
            ThermalMode::Cool => "cool",
            ThermalMode::Balanced => "balanced",
            ThermalMode::MaxPerf => "max-perf",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "cool" => Some(ThermalMode::Cool),
            "balanced" => Some(ThermalMode::Balanced),
            "max-perf" | "maxperf" | "max_perf" => Some(ThermalMode::MaxPerf),
            _ => None,
        }
    }

    /// Preset batch size for this thermal mode. Smaller batches under `Cool`
    /// keep sustained heat spikes down; larger batches under `MaxPerf` favor
    /// throughput.
    pub fn preset_batch_size(self) -> usize {
        match self {
            ThermalMode::Cool => 32,
            ThermalMode::Balanced => 128,
            ThermalMode::MaxPerf => 512,
        }
    }

    /// Preset audit depth for this thermal mode. `Cool` and `MaxPerf` both pick
    /// `Summary` (minimal cold-path overhead — one to be gentle, the other to
    /// keep benchmark timing clean); `Balanced` keeps `Full` audit for normal
    /// operation.
    pub fn preset_audit_mode(self) -> AuditMode {
        match self {
            ThermalMode::Cool => AuditMode::Summary,
            ThermalMode::Balanced => AuditMode::Full,
            ThermalMode::MaxPerf => AuditMode::Summary,
        }
    }
}

// ── Energy model ──────────────────────────────────────────────────────────

/// Estimated energy cost of a single double-precision FLOP, in joules.
///
/// This is an order-of-magnitude *representative* figure for a modern CPU
/// (~100 pJ per useful FLOP once issue/fetch overhead is amortized), not a
/// measured value for any specific chip. It exists so programs can compute a
/// *relative, deterministic* "joules per result" metric. Treat the absolute
/// number as an estimate; treat ratios between two CJC-Lang runs as meaningful.
pub const ENERGY_PER_FLOP_JOULES: f64 = 1.0e-10;

/// Estimated energy cost of moving one byte through the memory hierarchy, in
/// joules (~100 pJ/byte, representative of DRAM traffic). Memory traffic is a
/// dominant energy consumer, which is why TidyView's sparse/dictionary-encoded
/// layouts matter for the green story. Same caveat as [`ENERGY_PER_FLOP_JOULES`]:
/// an estimate for relative comparison, not a calibrated absolute.
pub const ENERGY_PER_BYTE_JOULES: f64 = 1.0e-10;

/// Deterministic energy estimate in joules for a workload of `flops`
/// floating-point operations and `bytes` of memory traffic.
///
/// Pure function of the (non-negative) integer counts and the two fixed
/// constants above — **no wall-clock time, no RNG, no FMA**. Negative inputs
/// are clamped to zero so the result is always non-negative and finite.
pub fn energy_estimate_joules(flops: i64, bytes: i64) -> f64 {
    let f = flops.max(0) as f64;
    let b = bytes.max(0) as f64;
    // Kept in separate bindings so the two multiplies cannot be contracted
    // into a single fused-multiply-add (preserves bit-identical results).
    let flop_energy = f * ENERGY_PER_FLOP_JOULES;
    let byte_energy = b * ENERGY_PER_BYTE_JOULES;
    flop_energy + byte_energy
}

// ── The policy struct ───────────────────────────────────────────────────────

/// A fully-resolved runtime execution policy.
///
/// `max_threads == 0` means "auto" — resolve the effective cap from
/// [`ThermalMode`] and the detected core count via [`effective_threads`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimePolicy {
    pub determinism: Determinism,
    pub numeric_mode: NumericMode,
    pub thermal_mode: ThermalMode,
    /// Hard thread cap; `0` = auto (derive from `thermal_mode`).
    pub max_threads: usize,
    /// Advisory batch size for chunked workloads (training, ABNG, TidyView).
    pub batch_size: usize,
    pub audit_mode: AuditMode,
    /// Race-to-idle scheduling: when `true`, parallel work runs at *full* width
    /// for a short burst and only throttles to the thermal cap once load is
    /// *sustained* (see [`run_parallel`]). Recovers burst performance while
    /// keeping the sustained thermal bound. When `false`, the cap applies
    /// uniformly (a fixed, reproducible schedule). Moot when the cap equals the
    /// core count (`max-perf`). Never affects results — only the schedule.
    pub adaptive: bool,
}

impl RuntimePolicy {
    /// Build the policy implied by a thermal profile: the profile sets the
    /// thermal mode, its preset batch size, its preset audit depth, and leaves
    /// the thread count on `auto` (0). Determinism stays `Strict` and the
    /// numeric mode stays `Kahan` — those are orthogonal to thermal behavior.
    pub fn for_thermal_mode(mode: ThermalMode) -> Self {
        Self {
            determinism: Determinism::Strict,
            numeric_mode: NumericMode::Kahan,
            thermal_mode: mode,
            max_threads: 0,
            batch_size: mode.preset_batch_size(),
            audit_mode: mode.preset_audit_mode(),
            adaptive: true,
        }
    }

    /// One-line, deterministic, BTreeMap-free summary for reporting.
    pub fn summary(&self) -> String {
        format!(
            "runtime_policy: thermal={} threads={} batch={} audit={} numeric={} determinism={} adaptive={}",
            self.thermal_mode.as_str(),
            effective_threads(self, detect_cores()),
            self.batch_size,
            self.audit_mode.as_str(),
            self.numeric_mode.as_str(),
            self.determinism.as_str(),
            self.adaptive,
        )
    }
}

impl Default for RuntimePolicy {
    /// The laptop-safe default is `Balanced`, not "max all cores forever".
    fn default() -> Self {
        Self::for_thermal_mode(ThermalMode::Balanced)
    }
}

// ── Thread resolution ─────────────────────────────────────────────────────

/// Detected logical core count (`>= 1`). Falls back to 1 if the platform
/// cannot report parallelism. This is the only place we read machine topology.
pub fn detect_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Resolve the effective thread cap for a policy given a detected core count.
///
/// An explicit `max_threads > 0` wins (clamped to the detected cores so we
/// never over-subscribe). Otherwise the thermal mode derives the cap:
/// `Cool` ≈ quarter of cores, `Balanced` ≈ half, `MaxPerf` = all. The result
/// is always in `1..=cores`. Pure function — deterministic given its inputs.
pub fn effective_threads(policy: &RuntimePolicy, detected_cores: usize) -> usize {
    let cores = detected_cores.max(1);
    if policy.max_threads > 0 {
        policy.max_threads.min(cores)
    } else {
        match policy.thermal_mode {
            ThermalMode::Cool => (cores / 4).max(1),
            ThermalMode::Balanced => (cores / 2).max(1),
            ThermalMode::MaxPerf => cores,
        }
    }
}

/// Pre-warm the throttle pool for a thread cap. Returns the live worker count.
///
/// **Phase 2 (race-to-idle) changed the model.** The global rayon pool is left
/// at its default (all cores) so that *bursts* can use full parallelism; the
/// thermal cap is enforced per-operation by [`run_parallel`], which `install`s
/// sustained work into a smaller, cached pool. This call just pre-builds that
/// capped pool so the first sustained op doesn't pay the build cost. `n == 0`
/// or `n >= cores` means "no cap" and builds nothing.
#[cfg(feature = "parallel")]
pub fn apply_thread_cap(n: usize) -> usize {
    let full = detect_cores();
    if n > 0 && n < full {
        let _ = capped_pool(n); // pre-build; OnceLock fixes the size
    }
    rayon::current_num_threads()
}

/// No-parallel fallback: there is exactly one thread of execution.
#[cfg(not(feature = "parallel"))]
pub fn apply_thread_cap(_n: usize) -> usize {
    1
}

// ── Race-to-idle adaptive scheduling (Phase 2) ─────────────────────────────

/// Burst budget: parallel work runs at full width for this long before a
/// *sustained* workload is throttled to the thermal cap. Sized to the thermal
/// time constant — sub-second bursts don't heat-soak the package, so they run
/// free; only multi-second sustained load (the kind that actually throttles a
/// laptop) gets capped.
#[cfg(feature = "parallel")]
const SUSTAIN_WINDOW: std::time::Duration = std::time::Duration::from_millis(2000);

/// An idle gap longer than this resets the burst timer, so a fresh burst after
/// a pause again gets full width.
#[cfg(feature = "parallel")]
const IDLE_RESET: std::time::Duration = std::time::Duration::from_millis(500);

#[cfg(feature = "parallel")]
#[derive(Default)]
struct AdaptiveState {
    burst_start: Option<std::time::Instant>,
    last_op: Option<std::time::Instant>,
}

#[cfg(feature = "parallel")]
thread_local! {
    static ADAPTIVE: RefCell<AdaptiveState> = RefCell::new(AdaptiveState::default());
}

/// Pure burst/sustained decision — separated from the clock so it is unit
/// testable with explicit instants. Returns `true` (throttle) once the current
/// burst has been active for at least `window`; an idle gap beyond `idle`
/// starts a fresh burst. Mutates `state` to record the burst start / last op.
#[cfg(feature = "parallel")]
fn decide_sustained(
    state: &mut AdaptiveState,
    now: std::time::Instant,
    window: std::time::Duration,
    idle: std::time::Duration,
) -> bool {
    if let Some(last) = state.last_op {
        if now.duration_since(last) > idle {
            state.burst_start = None;
        }
    }
    let start = *state.burst_start.get_or_insert(now);
    state.last_op = Some(now);
    now.duration_since(start) >= window
}

#[cfg(feature = "parallel")]
fn is_sustained_now() -> bool {
    let now = std::time::Instant::now();
    ADAPTIVE.with(|s| decide_sustained(&mut s.borrow_mut(), now, SUSTAIN_WINDOW, IDLE_RESET))
}

/// Reset the burst timer (e.g. between test runs on a reused thread).
#[cfg(feature = "parallel")]
fn reset_adaptive_state() {
    ADAPTIVE.with(|s| *s.borrow_mut() = AdaptiveState::default());
}

#[cfg(not(feature = "parallel"))]
fn reset_adaptive_state() {}

/// The cap-sized throttle pool. `OnceLock` fixes the size at first use (the cap
/// is stable per process — set once by the CLI at startup). `None` if rayon
/// failed to build it, in which case [`run_parallel`] degrades to no throttle.
#[cfg(feature = "parallel")]
static CAPPED_POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();

#[cfg(feature = "parallel")]
fn capped_pool(cap: usize) -> Option<&'static rayon::ThreadPool> {
    CAPPED_POOL
        .get_or_init(|| rayon::ThreadPoolBuilder::new().num_threads(cap).build().ok())
        .as_ref()
}

/// Run `work` under the active thermal policy, throttling parallelism to the
/// thermal cap only when appropriate.
///
/// Wrap a parallel kernel's body in this. The rules:
/// - cap ≥ cores (`max-perf` / `--threads ≥ N`): run on the full global pool.
/// - already inside a rayon worker (nested call): run inline on the current
///   pool — never nest-`install` (avoids surprising thread fan-out / blocking).
/// - `adaptive` (default): full width during a burst, throttle once load is
///   sustained ([`decide_sustained`]).
/// - `adaptive == false`: always throttle to the cap (fixed, reproducible).
///
/// Throttling means `install`-ing into the [`capped_pool`], inside which
/// `rayon::current_num_threads()` reports the cap — so existing chunkers that
/// size their work to the live thread count auto-scale. **Determinism is
/// preserved:** the choice of pool changes only how many bands/rows run
/// concurrently, never the per-element math (reductions keep their fixed
/// within-row order), so output is bit-identical regardless of this decision.
#[cfg(feature = "parallel")]
pub fn run_parallel<R, F>(work: F) -> R
where
    R: Send,
    F: FnOnce() -> R + Send,
{
    // Nested call from within a pool worker: run inline, don't re-install.
    if rayon::current_thread_index().is_some() {
        return work();
    }
    let policy = get();
    let cap = effective_threads(&policy, detect_cores());
    if cap >= detect_cores() {
        return work(); // no cap — full global pool
    }
    let throttle = if policy.adaptive { is_sustained_now() } else { true };
    if !throttle {
        return work(); // burst — full global pool
    }
    match capped_pool(cap) {
        Some(pool) => pool.install(work),
        None => work(),
    }
}

/// No-parallel fallback: run inline.
#[cfg(not(feature = "parallel"))]
pub fn run_parallel<R, F>(work: F) -> R
where
    F: FnOnce() -> R,
{
    work()
}

// ── Thread-local policy state ─────────────────────────────────────────────

thread_local! {
    /// The active runtime policy for this thread. The interpreter runs on one
    /// thread and reads/writes this; rayon workers honor the thread cap via the
    /// global pool, not via this cell.
    pub(crate) static POLICY: RefCell<RuntimePolicy> = RefCell::new(RuntimePolicy::default());
}

/// Snapshot the current policy.
pub fn get() -> RuntimePolicy {
    POLICY.with(|c| *c.borrow())
}

/// Reset to the laptop-safe `Balanced` default. Tests and the REPL call this
/// to avoid cross-run leakage on a reused thread. Also clears the race-to-idle
/// burst timer so a fresh run starts in the burst regime.
pub fn reset() {
    POLICY.with(|c| *c.borrow_mut() = RuntimePolicy::default());
    reset_adaptive_state();
}

/// Adopt a thermal profile wholesale: sets the thermal mode plus its preset
/// batch size and audit depth, and resets the thread cap to `auto`. Explicit
/// per-field setters called *after* this win (the CLI applies the profile
/// first, then individual `--threads` / `--batch-size` / `--audit-mode` overrides).
pub fn set_thermal_mode(mode: ThermalMode) {
    POLICY.with(|c| {
        let mut p = c.borrow_mut();
        p.thermal_mode = mode;
        p.batch_size = mode.preset_batch_size();
        p.audit_mode = mode.preset_audit_mode();
        p.max_threads = 0;
    });
}

/// Set an explicit thread cap (`0` = auto).
pub fn set_threads(n: usize) {
    POLICY.with(|c| c.borrow_mut().max_threads = n);
}

/// Set the advisory batch size.
pub fn set_batch_size(n: usize) {
    POLICY.with(|c| c.borrow_mut().batch_size = n);
}

/// Set the audit depth.
pub fn set_audit_mode(mode: AuditMode) {
    POLICY.with(|c| c.borrow_mut().audit_mode = mode);
}

/// Set the numeric reduction mode.
pub fn set_numeric_mode(mode: NumericMode) {
    POLICY.with(|c| c.borrow_mut().numeric_mode = mode);
}

/// Set the determinism level.
pub fn set_determinism(d: Determinism) {
    POLICY.with(|c| c.borrow_mut().determinism = d);
}

/// Enable/disable race-to-idle adaptive scheduling. `false` = fixed cap
/// (reproducible schedule); `true` = burst-then-throttle (the default).
pub fn set_adaptive(on: bool) {
    POLICY.with(|c| c.borrow_mut().adaptive = on);
}

/// Resolved effective thread cap for the current policy on this machine.
pub fn current_effective_threads() -> usize {
    effective_threads(&get(), detect_cores())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_balanced() {
        let p = RuntimePolicy::default();
        assert_eq!(p.thermal_mode, ThermalMode::Balanced);
        assert_eq!(p.determinism, Determinism::Strict);
        assert_eq!(p.numeric_mode, NumericMode::Kahan);
        assert_eq!(p.max_threads, 0);
        assert_eq!(p.batch_size, 128);
        assert_eq!(p.audit_mode, AuditMode::Full);
        assert!(p.adaptive, "adaptive (race-to-idle) is on by default");
    }

    #[test]
    fn set_adaptive_round_trip() {
        reset();
        assert!(get().adaptive);
        set_adaptive(false);
        assert!(!get().adaptive);
        reset();
        assert!(get().adaptive, "reset restores adaptive default");
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn decide_sustained_burst_then_throttle() {
        use std::time::{Duration, Instant};
        let mut st = AdaptiveState::default();
        let t0 = Instant::now();
        let win = Duration::from_millis(2000);
        let idle = Duration::from_millis(500);
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        // Frequent ops (300ms gaps < the 500ms idle threshold) so the burst
        // timer accumulates rather than resetting.
        assert!(!decide_sustained(&mut st, t0, win, idle), "burst starts");
        let mut ms = 300;
        while ms < 2000 {
            assert!(!decide_sustained(&mut st, at(ms), win, idle), "still burst at {ms}ms");
            ms += 300;
        }
        // Past the 2s window with continuous activity → throttle.
        assert!(decide_sustained(&mut st, at(2100), win, idle), "sustained past window");
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn decide_sustained_idle_resets_burst() {
        use std::time::{Duration, Instant};
        let mut st = AdaptiveState::default();
        let t0 = Instant::now();
        let win = Duration::from_millis(2000);
        let idle = Duration::from_millis(500);
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        // Drive to sustained with frequent ops.
        assert!(!decide_sustained(&mut st, t0, win, idle));
        let mut ms = 300;
        while ms <= 2100 {
            decide_sustained(&mut st, at(ms), win, idle);
            ms += 300;
        }
        assert!(decide_sustained(&mut st, at(2400), win, idle), "is sustained");
        // A pause longer than the idle threshold (600ms > 500ms) starts a fresh burst.
        assert!(
            !decide_sustained(&mut st, at(3000), win, idle),
            "idle gap should reset the burst → full speed again"
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn run_parallel_preserves_value_under_throttle() {
        // The throttle decision must never change results: run a tiny reduction
        // both ways and assert byte-identical output. (Concurrency differs;
        // the per-element math does not.)
        reset();
        set_thermal_mode(ThermalMode::Cool);
        set_adaptive(false); // force throttle path
        let throttled: f64 = run_parallel(|| (0..1000).map(|i| i as f64).sum());
        set_adaptive(true);
        reset_adaptive_state(); // burst path
        let burst: f64 = run_parallel(|| (0..1000).map(|i| i as f64).sum());
        assert_eq!(throttled, burst);
        reset();
    }

    #[test]
    fn thermal_presets_distinct() {
        assert_eq!(ThermalMode::Cool.preset_batch_size(), 32);
        assert_eq!(ThermalMode::Balanced.preset_batch_size(), 128);
        assert_eq!(ThermalMode::MaxPerf.preset_batch_size(), 512);
        assert_eq!(ThermalMode::Cool.preset_audit_mode(), AuditMode::Summary);
        assert_eq!(ThermalMode::Balanced.preset_audit_mode(), AuditMode::Full);
        assert_eq!(ThermalMode::MaxPerf.preset_audit_mode(), AuditMode::Summary);
    }

    #[test]
    fn effective_threads_monotonic_in_thermal_mode() {
        let cores = 8;
        let cool = effective_threads(&RuntimePolicy::for_thermal_mode(ThermalMode::Cool), cores);
        let bal = effective_threads(&RuntimePolicy::for_thermal_mode(ThermalMode::Balanced), cores);
        let max = effective_threads(&RuntimePolicy::for_thermal_mode(ThermalMode::MaxPerf), cores);
        assert!(cool <= bal, "cool {cool} should not exceed balanced {bal}");
        assert!(bal <= max, "balanced {bal} should not exceed max-perf {max}");
        assert_eq!(cool, 2);
        assert_eq!(bal, 4);
        assert_eq!(max, 8);
    }

    #[test]
    fn effective_threads_always_in_range() {
        for cores in [1usize, 2, 3, 7, 16, 64] {
            for mode in [ThermalMode::Cool, ThermalMode::Balanced, ThermalMode::MaxPerf] {
                let t = effective_threads(&RuntimePolicy::for_thermal_mode(mode), cores);
                assert!(t >= 1, "threads must be >= 1 (cores={cores}, mode={mode:?})");
                assert!(t <= cores, "threads {t} must be <= cores {cores}");
            }
        }
    }

    #[test]
    fn explicit_thread_cap_wins_and_clamps() {
        let mut p = RuntimePolicy::for_thermal_mode(ThermalMode::MaxPerf);
        p.max_threads = 3;
        assert_eq!(effective_threads(&p, 8), 3, "explicit cap should be honored");
        p.max_threads = 100;
        assert_eq!(effective_threads(&p, 8), 8, "cap clamps to detected cores");
    }

    #[test]
    fn effective_threads_zero_cores_is_one() {
        let p = RuntimePolicy::default();
        assert_eq!(effective_threads(&p, 0), 1);
    }

    #[test]
    fn energy_is_non_negative_and_zero_at_zero() {
        assert_eq!(energy_estimate_joules(0, 0), 0.0);
        assert!(energy_estimate_joules(-5, -7) >= 0.0);
        assert_eq!(energy_estimate_joules(-5, -7), 0.0, "negatives clamp to zero");
    }

    #[test]
    fn energy_is_monotonic() {
        let a = energy_estimate_joules(1000, 1000);
        let b = energy_estimate_joules(2000, 1000);
        let c = energy_estimate_joules(2000, 2000);
        assert!(b > a, "more flops => more energy");
        assert!(c > b, "more bytes => more energy");
    }

    #[test]
    fn energy_is_additive_in_components() {
        let flop_only = energy_estimate_joules(1_000_000, 0);
        let byte_only = energy_estimate_joules(0, 1_000_000);
        let both = energy_estimate_joules(1_000_000, 1_000_000);
        assert_eq!(both, flop_only + byte_only);
    }

    #[test]
    fn energy_is_deterministic_across_calls() {
        let first = energy_estimate_joules(123_456, 789);
        for _ in 0..1000 {
            assert_eq!(energy_estimate_joules(123_456, 789), first);
        }
    }

    #[test]
    fn enum_round_trips() {
        for m in [ThermalMode::Cool, ThermalMode::Balanced, ThermalMode::MaxPerf] {
            assert_eq!(ThermalMode::from_str(m.as_str()), Some(m));
        }
        for m in [NumericMode::Kahan, NumericMode::Binned, NumericMode::FixedTree] {
            assert_eq!(NumericMode::from_str(m.as_str()), Some(m));
        }
        for m in [AuditMode::Summary, AuditMode::Full, AuditMode::Forensic] {
            assert_eq!(AuditMode::from_str(m.as_str()), Some(m));
        }
        for m in [Determinism::Strict, Determinism::Relaxed] {
            assert_eq!(Determinism::from_str(m.as_str()), Some(m));
        }
    }

    #[test]
    fn invalid_mode_strings_return_none() {
        assert_eq!(ThermalMode::from_str("blazing"), None);
        assert_eq!(NumericMode::from_str(""), None);
        assert_eq!(AuditMode::from_str("paranoid"), None);
        assert_eq!(Determinism::from_str("yolo"), None);
    }

    #[test]
    fn set_get_round_trip_and_reset() {
        reset();
        set_thermal_mode(ThermalMode::Cool);
        let p = get();
        assert_eq!(p.thermal_mode, ThermalMode::Cool);
        assert_eq!(p.batch_size, 32);
        assert_eq!(p.audit_mode, AuditMode::Summary);

        set_threads(2);
        set_batch_size(64);
        set_audit_mode(AuditMode::Forensic);
        set_numeric_mode(NumericMode::Binned);
        let p = get();
        assert_eq!(p.max_threads, 2);
        assert_eq!(p.batch_size, 64);
        assert_eq!(p.audit_mode, AuditMode::Forensic);
        assert_eq!(p.numeric_mode, NumericMode::Binned);

        reset();
        assert_eq!(get(), RuntimePolicy::default());
    }

    #[test]
    fn profile_then_override_precedence() {
        reset();
        // CLI order: profile first, then explicit override.
        set_thermal_mode(ThermalMode::MaxPerf);
        assert_eq!(get().batch_size, 512);
        set_batch_size(16);
        assert_eq!(get().batch_size, 16, "explicit override wins over profile preset");
        assert_eq!(get().thermal_mode, ThermalMode::MaxPerf, "mode unchanged by batch override");
        reset();
    }

    #[test]
    fn apply_thread_cap_never_panics_and_is_positive() {
        // Does not assert an exact count: in the test process rayon's pool may
        // already be initialized, so build_global may be a no-op. We only
        // require a sane positive worker count and no panic.
        let n = apply_thread_cap(2);
        assert!(n >= 1);
    }

    #[test]
    fn summary_is_stable() {
        reset();
        let s1 = get().summary();
        let s2 = get().summary();
        assert_eq!(s1, s2);
        assert!(s1.contains("thermal=balanced"));
        assert!(s1.contains("determinism=strict"));
        reset();
    }
}
