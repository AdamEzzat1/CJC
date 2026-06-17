//! Live Rust heap collector — a `GlobalAlloc` shim that records real
//! allocations/frees into a global buffer, plus the reentrancy guard that makes
//! recording-while-allocating safe.
//!
//! **This is nondeterministic by nature** (it observes real heap traffic and is
//! sampled in wall-clock time) and lives behind the `collect-live` feature. It
//! only *produces* a [`Trace`](crate::Trace); the analysis engine never sees
//! wall-clock.
//!
//! ## Reentrancy
//!
//! Recording an allocation requires data structures that themselves allocate.
//! Without care, `alloc → record → push to Vec → alloc → …` recurses forever
//! (and deadlocks on the buffer `Mutex`). The fix is the classic one (dhat,
//! Memray use it): a thread-local [`Guard`]. While a thread is inside the
//! recording path, the guard is set and the allocator forwards straight to the
//! system allocator without recording. Every collector method that locks the
//! buffer first ensures the guard is held, so a nested `alloc()` on the same
//! thread skips recording and never re-locks.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};

use crate::trace::OwnershipDomain;

/// Sentinel zone id for "no zone open".
pub(crate) const NO_ZONE: u32 = u32::MAX;

/// A recorded raw event, in capture order. Interning into a [`Trace`] is
/// deferred to [`Recorder::finish`](crate::collect::Recorder::finish).
#[derive(Clone, Debug)]
pub(crate) enum RawEvent {
    Alloc { bytes: u64, zone: u32 },
    Free { bytes: u64, zone: u32 },
    Sample { stack: Vec<u32> },
    ZoneStart { zone: u32 },
    ZoneStop,
    Copy { from: OwnershipDomain, to: OwnershipDomain, bytes: u64 },
    Boundary { name: u32 },
}

struct Inner {
    events: Vec<RawEvent>,
    names: Vec<String>,
    name_idx: BTreeMap<String, u32>,
    zone_stack: Vec<u32>,
}

impl Inner {
    const fn new() -> Self {
        Inner {
            events: Vec::new(),
            names: Vec::new(),
            name_idx: BTreeMap::new(),
            zone_stack: Vec::new(),
        }
    }
    fn reset(&mut self) {
        self.events.clear();
        self.names.clear();
        self.name_idx.clear();
        self.zone_stack.clear();
    }
    fn intern(&mut self, name: &str) -> u32 {
        if let Some(&i) = self.name_idx.get(name) {
            return i;
        }
        let i = self.names.len() as u32;
        self.names.push(name.to_string());
        self.name_idx.insert(name.to_string(), i);
        i
    }
}

/// The process-global collector. A single instance shared by all threads.
pub(crate) struct Collector {
    enabled: AtomicBool,
    inner: Mutex<Inner>,
}

pub(crate) static COLLECTOR: Collector = Collector {
    enabled: AtomicBool::new(false),
    inner: Mutex::new(Inner::new()),
};

impl Collector {
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(crate) fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    /// Reset the buffer and begin recording.
    pub(crate) fn enable(&self) {
        let _g = Guard::enter();
        self.lock().reset();
        self.enabled.store(true, Ordering::SeqCst);
    }

    /// Stop recording (does not drain).
    pub(crate) fn disable(&self) {
        self.enabled.store(false, Ordering::SeqCst);
    }

    /// Take the captured events + name table, leaving the buffer empty.
    pub(crate) fn drain(&self) -> (Vec<RawEvent>, Vec<String>) {
        let _g = Guard::enter();
        let mut inner = self.lock();
        let events = std::mem::take(&mut inner.events);
        let names = std::mem::take(&mut inner.names);
        inner.name_idx.clear();
        inner.zone_stack.clear();
        (events, names)
    }

    // ── allocator path (caller already holds the Guard) ──
    fn record_alloc(&self, bytes: u64) {
        let mut inner = self.lock();
        let zone = inner.zone_stack.last().copied().unwrap_or(NO_ZONE);
        inner.events.push(RawEvent::Alloc { bytes, zone });
    }
    fn record_free(&self, bytes: u64) {
        let mut inner = self.lock();
        let zone = inner.zone_stack.last().copied().unwrap_or(NO_ZONE);
        inner.events.push(RawEvent::Free { bytes, zone });
    }

    // ── user-facing path (establish the Guard so bookkeeping allocs are not
    //    recorded and can never re-lock the buffer on this thread) ──
    pub(crate) fn enter_zone(&self, name: &str) {
        let _g = Guard::enter();
        let mut inner = self.lock();
        let i = inner.intern(name);
        inner.zone_stack.push(i);
        inner.events.push(RawEvent::ZoneStart { zone: i });
    }
    pub(crate) fn exit_zone(&self) {
        let _g = Guard::enter();
        let mut inner = self.lock();
        if inner.zone_stack.pop().is_some() {
            inner.events.push(RawEvent::ZoneStop);
        }
    }
    pub(crate) fn record_sample(&self) {
        let _g = Guard::enter();
        let mut inner = self.lock();
        if inner.zone_stack.is_empty() {
            return;
        }
        let stack = inner.zone_stack.clone();
        inner.events.push(RawEvent::Sample { stack });
    }
    pub(crate) fn record_copy(&self, from: OwnershipDomain, to: OwnershipDomain, bytes: u64) {
        let _g = Guard::enter();
        self.lock().events.push(RawEvent::Copy { from, to, bytes });
    }
    pub(crate) fn record_boundary(&self, name: &str) {
        let _g = Guard::enter();
        let mut inner = self.lock();
        let i = inner.intern(name);
        inner.events.push(RawEvent::Boundary { name: i });
    }
}

// ─── reentrancy guard ───────────────────────────────────────────────────────

thread_local! {
    static IN_RECORD: Cell<bool> = const { Cell::new(false) };
}

/// RAII guard that marks "this thread is inside the recording path". Created
/// only when it was the one to flip the flag, so nested `Guard::enter()` calls
/// return `None` and do not prematurely clear the flag on drop.
pub(crate) struct Guard;

impl Guard {
    pub(crate) fn enter() -> Option<Guard> {
        IN_RECORD
            .try_with(|f| {
                if f.get() {
                    None
                } else {
                    f.set(true);
                    Some(Guard)
                }
            })
            .unwrap_or(None)
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = IN_RECORD.try_with(|f| f.set(false));
    }
}

// ─── the global allocator shim ──────────────────────────────────────────────

/// A `#[global_allocator]`-compatible wrapper around the system allocator that
/// records real heap traffic when a [`Recorder`](crate::collect::Recorder) is
/// active. Install it in the binary you want to profile:
///
/// ```ignore
/// #[global_allocator]
/// static GLOBAL: cjc_seshat::collect::SeshatAlloc = cjc_seshat::collect::SeshatAlloc;
/// ```
///
/// When no recording is active, overhead is a single relaxed atomic load.
pub struct SeshatAlloc;

unsafe impl GlobalAlloc for SeshatAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() && COLLECTOR.is_enabled() {
            if let Some(_g) = Guard::enter() {
                COLLECTOR.record_alloc(layout.size() as u64);
            }
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if COLLECTOR.is_enabled() {
            if let Some(_g) = Guard::enter() {
                COLLECTOR.record_free(layout.size() as u64);
            }
        }
        System.dealloc(ptr, layout);
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc_zeroed(layout);
        if !ptr.is_null() && COLLECTOR.is_enabled() {
            if let Some(_g) = Guard::enter() {
                COLLECTOR.record_alloc(layout.size() as u64);
            }
        }
        ptr
    }

    // `realloc` intentionally uses the default `GlobalAlloc` implementation,
    // which routes through `alloc` + `dealloc` above, so a reallocation is
    // recorded as a free of the old block plus an allocation of the new one.
}
