//! Peak heap bytes of a closure, for `postulate/runReport`.
//!
//! [`CountingAlloc`] wraps the system allocator with two atomic counters (live
//! bytes, peak live bytes). The `cjc-analyzer` binary installs it; a library
//! user or test that does not install it gets `None` from [`measure`], never a
//! made-up number.
//!
//! The figure is **measured, not reproducible**: the LSP transport threads
//! allocate concurrently, so `runReport` lists `peak_bytes` under `"measured"`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);

/// Install with `#[global_allocator] static A: CountingAlloc = CountingAlloc;`.
pub struct CountingAlloc;

fn grew(n: usize) {
    let live = LIVE.fetch_add(n as u64, Relaxed) + n as u64;
    PEAK.fetch_max(live, Relaxed);
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size() as u64, Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            LIVE.fetch_sub(layout.size() as u64, Relaxed);
            grew(new_size);
        }
        p
    }
}

/// Run `f`; return its result and the peak live bytes above the starting
/// level while it ran, or `None` when [`CountingAlloc`] is not installed.
pub fn measure<R>(f: impl FnOnce() -> R) -> (R, Option<u64>) {
    let base = LIVE.load(Relaxed);
    if base == 0 {
        // Nothing has ever been counted: the meter is not installed.
        return (f(), None);
    }
    PEAK.store(base, Relaxed);
    let r = f();
    (r, Some(PEAK.load(Relaxed).saturating_sub(base)))
}
