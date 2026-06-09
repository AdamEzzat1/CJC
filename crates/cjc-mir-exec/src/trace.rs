//! Per-thread trace collector for instrumented MIR-exec runs.
//!
//! **PR 1 of Option B** — see `docs/cana/OPTION_B_DESIGN.md` for the full
//! design. This module provides the foundation that future PRs will
//! populate with actual instrumentation sites. PR 1 ships:
//!
//!   - [`TraceCollector`] — per-thread buffer of `MirTraceEvent`s
//!   - [`TRACE_COLLECTOR`] — TLS slot the executor reads/writes
//!   - [`with_trace`] — convenience accessor for instrumentation sites
//!   - The instrumented entry point (`run_program_instrumented` in
//!     `lib.rs`) parallel to the existing `run_program_with_executor`
//!
//! **What this PR does NOT do:** wire any actual emit sites in the
//! executor. The instrumentation locations (basic-block entry,
//! call entry/exit, FP-op count, branch dispatch, GC sweep, etc.)
//! land in PRs 2-4. After PR 1, calling `run_program_instrumented`
//! returns an empty `Vec<MirTraceEvent>` — and the parity gate
//! confirms the program output is byte-identical to the
//! uninstrumented run. That's the foundation guarantee.
//!
//! # Why TLS
//!
//! Matches the [`cjc_ad::GRAD_GRAPH`] pattern already used by the
//! grad_graph_* builtins. Both executors share the same TLS slot on
//! the same thread, so any code path (eval or mir-exec) can emit
//! into the collector without explicit plumbing through call stacks.
//!
//! # Determinism contract
//!
//! See [`OPTION_B_DESIGN.md`] §4. This module MUST avoid:
//!
//!   - `std::time::Instant` / `SystemTime` (system clock — not
//!     deterministic).
//!   - `std::sync::atomic::*` (cross-thread ordering — not
//!     deterministic). TLS removes the need.
//!   - `std::collections::HashMap` iteration (random hash seed — not
//!     deterministic).
//!   - Time-based sampling (`if elapsed > X`). All sampling is
//!     event-count-based (`if self.tick % N == 0`).
//!
//! The `MirTraceEvent.tick` field is set by the collector from its
//! own monotonic counter — callers cannot influence it. Two runs of
//! the same program on the same seed produce byte-identical event
//! sequences.

use std::cell::RefCell;

use cjc_nss::mir_adapter::MirTraceEvent;

/// Per-thread event buffer. One instance per executor invocation;
/// the TLS slot ([`TRACE_COLLECTOR`]) holds the active one.
///
/// Default state is `enabled = false` — the executor's existing
/// entry points (`run_program_with_executor`, `run_program_optimized`,
/// etc.) leave the collector disabled, so any future instrumentation
/// emit sites are no-ops on those paths. Only the new
/// `run_program_instrumented` entry point enables it.
pub struct TraceCollector {
    events: Vec<MirTraceEvent>,
    /// Monotonic event index. Set on the event in [`Self::emit`]
    /// regardless of what the caller passed; this guarantees a coherent
    /// sequence even when emit sites are added independently in
    /// different PRs and can't coordinate tick numbering.
    tick: u64,
    enabled: bool,
}

impl TraceCollector {
    /// New, disabled, zero-capacity collector.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            tick: 0,
            enabled: false,
        }
    }

    /// New, disabled collector with pre-allocated event capacity.
    /// Useful when the executor can estimate event count from the
    /// program's basic-block count (typically ~1 event per BB).
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            events: Vec::with_capacity(cap),
            tick: 0,
            enabled: false,
        }
    }

    /// Begin recording. Subsequent [`emit`](Self::emit) calls append
    /// to the event buffer and increment the tick counter.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Stop recording. Subsequent [`emit`](Self::emit) calls become
    /// no-ops; existing events are retained until [`take`](Self::take)
    /// or [`reset`](Self::reset).
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Append one trace event. The `event.tick` field is overwritten
    /// with the collector's own monotonic counter — callers should
    /// not depend on the value they pass through. No-op when disabled.
    ///
    /// Cost: one branch (`if self.enabled`) when disabled — typically
    /// inlines to ~3 instructions on hot paths. When enabled: one
    /// branch + one `Vec::push` + one `saturating_add`.
    pub fn emit(&mut self, mut event: MirTraceEvent) {
        if !self.enabled {
            return;
        }
        event.tick = self.tick;
        self.events.push(event);
        self.tick = self.tick.saturating_add(1);
    }

    /// Consume the recorded events. Resets the tick counter so the
    /// collector can be re-used for another instrumented run on the
    /// same thread without leaking ticks across runs.
    ///
    /// Does NOT change the enabled flag — that's the caller's job
    /// (typically the `run_program_instrumented` entry point disables
    /// the collector immediately after `take()`).
    pub fn take(&mut self) -> Vec<MirTraceEvent> {
        self.tick = 0;
        std::mem::take(&mut self.events)
    }

    /// Reset to the post-`new()` state: empty buffer, tick=0, disabled.
    pub fn reset(&mut self) {
        self.events.clear();
        self.tick = 0;
        self.enabled = false;
    }

    /// Number of events currently buffered.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Next tick the collector will stamp on a successful emit.
    pub fn tick(&self) -> u64 {
        self.tick
    }
}

impl Default for TraceCollector {
    fn default() -> Self {
        Self::new()
    }
}

thread_local! {
    /// Per-thread active trace collector. The instrumented entry
    /// point in `lib.rs` enables/disables it around the executor
    /// invocation; uninstrumented entry points leave it disabled,
    /// so any future instrumentation sites in the executor become
    /// no-ops on those paths.
    ///
    /// **DO NOT** access this directly from outside the executor
    /// crate — use [`with_trace`] so the borrow_mut() bookkeeping
    /// stays centralized.
    pub static TRACE_COLLECTOR: RefCell<TraceCollector> =
        RefCell::new(TraceCollector::new());
}

/// Convenience accessor for instrumentation sites in the executor.
/// Wraps the TLS borrow_mut() — callers just hand it a closure that
/// mutates the collector.
///
/// Example (from a future PR):
///
/// ```ignore
/// // In a basic-block entry instrumentation site:
/// with_trace(|c| c.emit(MirTraceEvent {
///     tick: 0, // overwritten by emit()
///     block_id: current_block_id,
///     register_pressure: sample_register_pressure(executor),
///     heap_bytes_in_use: sample_heap(executor),
///     call_depth: executor.call_stack().len() as u32,
///     branch_taken: incoming_branch_taken,
///     io_event: false,
///     gc_event: false,
///     instruction_count: stmts_since_last_event,
/// }));
/// ```
pub fn with_trace<F, R>(f: F) -> R
where
    F: FnOnce(&mut TraceCollector) -> R,
{
    TRACE_COLLECTOR.with(|c| f(&mut c.borrow_mut()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_event(block_id: u32) -> MirTraceEvent {
        MirTraceEvent {
            tick: 999, // intentionally non-zero — emit() must overwrite
            block_id,
            register_pressure: 0.5,
            heap_bytes_in_use: 1024,
            call_depth: 2,
            branch_taken: false,
            io_event: false,
            gc_event: false,
            instruction_count: 8,
        }
    }

    #[test]
    fn collector_starts_disabled_and_empty() {
        let c = TraceCollector::new();
        assert!(!c.is_enabled());
        assert_eq!(c.event_count(), 0);
        assert_eq!(c.tick(), 0);
    }

    #[test]
    fn emit_is_noop_when_disabled() {
        let mut c = TraceCollector::new();
        c.emit(sample_event(1));
        c.emit(sample_event(2));
        assert_eq!(
            c.event_count(),
            0,
            "events must not be recorded while disabled",
        );
        assert_eq!(c.tick(), 0, "tick must not advance while disabled");
    }

    #[test]
    fn emit_records_when_enabled() {
        let mut c = TraceCollector::new();
        c.enable();
        c.emit(sample_event(1));
        c.emit(sample_event(2));
        c.emit(sample_event(3));
        assert_eq!(c.event_count(), 3);
        assert_eq!(c.tick(), 3);
    }

    #[test]
    fn emit_overrides_caller_supplied_tick() {
        // The collector sets tick from its own counter — the caller's
        // tick field is IGNORED. This keeps events monotonic across
        // the program regardless of which emit site issued them, and
        // means future PRs that add new emit sites don't need to
        // coordinate tick numbering with existing sites.
        let mut c = TraceCollector::new();
        c.enable();
        c.emit(sample_event(10));
        c.emit(sample_event(20));
        c.emit(sample_event(30));
        let events = c.take();
        assert_eq!(events[0].tick, 0, "first event gets tick 0");
        assert_eq!(events[1].tick, 1, "second event gets tick 1");
        assert_eq!(events[2].tick, 2, "third event gets tick 2");
    }

    #[test]
    fn take_returns_events_and_resets_tick_but_not_enabled_flag() {
        let mut c = TraceCollector::new();
        c.enable();
        c.emit(sample_event(1));
        c.emit(sample_event(2));
        let evs = c.take();
        assert_eq!(evs.len(), 2);
        assert_eq!(c.event_count(), 0, "events drained");
        assert_eq!(c.tick(), 0, "tick reset after take");
        assert!(
            c.is_enabled(),
            "take() must NOT disable — that's the caller's job",
        );
    }

    #[test]
    fn reset_clears_everything_including_enabled_flag() {
        let mut c = TraceCollector::new();
        c.enable();
        c.emit(sample_event(1));
        c.reset();
        assert!(!c.is_enabled());
        assert_eq!(c.event_count(), 0);
        assert_eq!(c.tick(), 0);
    }

    #[test]
    fn with_capacity_pre_allocates_without_recording() {
        let c = TraceCollector::with_capacity(1024);
        // pre-alloc affects internal capacity but NOT user-visible state.
        assert!(!c.is_enabled());
        assert_eq!(c.event_count(), 0);
        assert_eq!(c.tick(), 0);
    }

    #[test]
    fn with_trace_accesses_tls_slot() {
        // Reset the TLS slot first — other tests in this binary may
        // have touched it and cfg(test) ordering is not guaranteed.
        with_trace(|c| c.reset());

        with_trace(|c| c.enable());
        with_trace(|c| c.emit(sample_event(7)));
        with_trace(|c| c.emit(sample_event(8)));

        let evs = with_trace(|c| c.take());
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].block_id, 7);
        assert_eq!(evs[1].block_id, 8);
        assert_eq!(evs[0].tick, 0);
        assert_eq!(evs[1].tick, 1);

        // Leave the TLS slot in clean state for any subsequent tests.
        with_trace(|c| c.reset());
    }

    #[test]
    fn deterministic_tick_sequence() {
        // The key determinism witness: same emit sequence produces the
        // same tick sequence, every time. If anything in the collector
        // ever consults a clock or atomic counter, this assertion can
        // become flaky — make it a hard test of the contract.
        let mut c = TraceCollector::new();
        c.enable();
        for i in 0..100 {
            c.emit(sample_event(i as u32));
        }
        let evs = c.take();
        for (i, ev) in evs.iter().enumerate() {
            assert_eq!(
                ev.tick, i as u64,
                "tick {} != index {} — non-monotonic event ordering",
                ev.tick, i,
            );
        }
    }

    #[test]
    fn enable_disable_idempotent() {
        let mut c = TraceCollector::new();
        c.enable();
        c.enable();
        c.enable();
        assert!(c.is_enabled());
        c.disable();
        c.disable();
        assert!(!c.is_enabled());
    }

    #[test]
    fn disable_then_enable_continues_tick_sequence() {
        // tick is collector-state, not enable-state. Re-enabling after
        // a disable should continue from the previous tick value, NOT
        // reset to 0. (Only `take` and `reset` zero the tick.)
        let mut c = TraceCollector::new();
        c.enable();
        c.emit(sample_event(1));
        c.emit(sample_event(2));
        c.disable();
        c.emit(sample_event(3)); // no-op
        c.enable();
        c.emit(sample_event(4));
        let evs = c.take();
        assert_eq!(evs.len(), 3, "only the 3 enabled emits recorded");
        assert_eq!(evs[0].tick, 0);
        assert_eq!(evs[1].tick, 1);
        assert_eq!(
            evs[2].tick, 2,
            "tick continued through disable/enable cycle",
        );
    }
}
