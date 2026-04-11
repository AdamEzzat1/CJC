//! Deterministic profile counters for Tier 2 of the Chess RL v2.3 upgrade.
//!
//! This module provides a minimal, write-only profiling sink that the
//! CJC-Lang program can use to time named zones inside a hot loop. It is
//! the smallest possible surface that makes the v2.2 bottleneck measurable
//! without perturbing program state.
//!
//! # Builtins
//!
//! - `profile_zone_start(name: String) -> i64`
//! - `profile_zone_stop(handle: i64) -> f64`
//! - `profile_dump(path: String) -> i64`
//!
//! All three dispatch arms live in [`crate::builtins`]; this module owns
//! only the thread-local state and the pure helper functions that operate
//! on it.
//!
//! # Determinism story
//!
//! The counter state lives in a thread-local `RefCell<ProfileState>`. The
//! program can **observe** the identity of a zone handle (to pair start/
//! stop calls), but the *integer value* of the handle must not feed into
//! program logic, RNG draws, tensor math, or control flow. The Chess RL
//! v2.3 parity test asserts that an instrumented rollout produces a
//! weight hash identical to an uninstrumented rollout.
//!
//! No floating-point math is done on the counters until `profile_dump`
//! renders the CSV, by which point the nanosecond counters are committed.
//! All iteration over zones uses [`BTreeMap`] ordering so the CSV layout
//! is reproducible across runs.
//!
//! No RNG is touched. No tensor math is touched. No cross-thread state
//! is accessed. No external crates are pulled in — only
//! `std::time::Instant`, `std::collections::BTreeMap`, and
//! `std::cell::RefCell`.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::Instant;

/// Per-zone aggregated statistics.
#[derive(Clone, Debug, Default)]
pub struct ZoneStats {
    /// Number of times the zone has been stopped.
    pub count: u64,
    /// Total nanoseconds spent in the zone.
    pub total_ns: u128,
    /// Minimum single-call nanoseconds.
    pub min_ns: u128,
    /// Maximum single-call nanoseconds.
    pub max_ns: u128,
    /// Sum of `ns^2` across calls, for stddev without FMA.
    pub sum_sq_ns: u128,
}

impl ZoneStats {
    fn update(&mut self, ns: u128) {
        if self.count == 0 {
            self.min_ns = ns;
            self.max_ns = ns;
        } else {
            if ns < self.min_ns {
                self.min_ns = ns;
            }
            if ns > self.max_ns {
                self.max_ns = ns;
            }
        }
        self.count += 1;
        self.total_ns = self.total_ns.saturating_add(ns);
        // Squaring a nanosecond count can overflow u64, so we stay in u128.
        let sq = (ns as u128).saturating_mul(ns as u128);
        self.sum_sq_ns = self.sum_sq_ns.saturating_add(sq);
    }

    /// Integer mean nanoseconds (`total_ns / count`). Returns 0 when count is 0.
    pub fn mean_ns(&self) -> u128 {
        if self.count == 0 {
            0
        } else {
            self.total_ns / (self.count as u128)
        }
    }

    /// Standard deviation in nanoseconds, computed as
    /// `sqrt(max(0, sum_sq/count - mean^2))`. Uses f64 at the last step only
    /// — no FMA, no Kahan, because the result is for reporting only.
    pub fn stddev_ns(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let mean = self.mean_ns() as f64;
        let mean_sq = mean * mean;
        let var_raw = (self.sum_sq_ns / (self.count as u128)) as f64 - mean_sq;
        if var_raw <= 0.0 {
            0.0
        } else {
            var_raw.sqrt()
        }
    }
}

/// Internal profiler state — one instance per thread.
pub struct ProfileState {
    /// Stopped zones, keyed by name for deterministic iteration.
    pub zones: BTreeMap<String, ZoneStats>,
    /// Currently-open zones, keyed by handle.
    pub active: BTreeMap<i64, (String, Instant)>,
    /// Monotonically increasing handle counter.
    pub next_handle: i64,
}

impl ProfileState {
    /// Fresh empty state.
    pub fn new() -> Self {
        Self {
            zones: BTreeMap::new(),
            active: BTreeMap::new(),
            next_handle: 0,
        }
    }

    /// Clear all state, as if the profiler had just been created.
    pub fn reset(&mut self) {
        self.zones.clear();
        self.active.clear();
        self.next_handle = 0;
    }
}

impl Default for ProfileState {
    fn default() -> Self {
        Self::new()
    }
}

thread_local! {
    /// Thread-local profiler state. Each thread keeps its own counters;
    /// no cross-thread coordination is done or needed.
    pub(crate) static PROFILE: RefCell<ProfileState> = RefCell::new(ProfileState::new());
}

/// Start a zone and return an opaque handle. The handle is monotonically
/// increasing within the current thread.
pub fn zone_start(name: &str) -> i64 {
    PROFILE.with(|cell| {
        let mut state = cell.borrow_mut();
        let handle = state.next_handle;
        state.next_handle = state.next_handle.wrapping_add(1);
        state
            .active
            .insert(handle, (name.to_string(), Instant::now()));
        handle
    })
}

/// Stop a zone previously started by [`zone_start`]. Updates the aggregated
/// [`ZoneStats`] and returns the elapsed seconds as f64. Returns `-1.0` for
/// an unknown handle (never panics).
pub fn zone_stop(handle: i64) -> f64 {
    PROFILE.with(|cell| {
        let mut state = cell.borrow_mut();
        let Some((name, start)) = state.active.remove(&handle) else {
            return -1.0;
        };
        let elapsed = start.elapsed();
        let ns = elapsed.as_nanos();
        let entry = state.zones.entry(name).or_default();
        entry.update(ns);
        // Report elapsed seconds for convenience; the v2.3 parity test
        // asserts that ignoring this value yields a bit-identical weight
        // hash, which is the determinism contract.
        ns as f64 / 1.0e9
    })
}

/// Serialize the aggregated zone statistics to CSV, sorted by `total_ns`
/// descending so the hot zones appear at the top. Returns the number of
/// data rows written (not including the header). Resets the profiler
/// state after writing.
///
/// CSV schema:
/// ```text
/// zone_name,count,total_ns,min_ns,max_ns,mean_ns,stddev_ns
/// ```
pub fn dump_to_path(path: &str) -> Result<i64, String> {
    let csv = PROFILE.with(|cell| {
        let mut state = cell.borrow_mut();

        // Collect rows from the BTreeMap (ordered by name).
        let mut rows: Vec<(String, ZoneStats)> = state
            .zones
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        // Sort by total_ns descending. Ties fall back to the BTreeMap name
        // ordering (which is what `stable_sort_by` inherits from `iter`).
        rows.sort_by(|a, b| b.1.total_ns.cmp(&a.1.total_ns));

        let mut out = String::new();
        out.push_str("zone_name,count,total_ns,min_ns,max_ns,mean_ns,stddev_ns\n");
        for (name, stats) in &rows {
            // Round stddev to the nearest integer ns to keep CSV integer-clean.
            let stddev_ns = stats.stddev_ns().round() as u128;
            out.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                name,
                stats.count,
                stats.total_ns,
                stats.min_ns,
                stats.max_ns,
                stats.mean_ns(),
                stddev_ns,
            ));
        }

        // Clear the state so subsequent runs start fresh.
        let row_count = rows.len() as i64;
        state.reset();
        (out, row_count)
    });

    std::fs::write(path, &csv.0).map_err(|e| format!("profile_dump error: {e}"))?;
    Ok(csv.1)
}

/// Test-only helper: snapshot the current zone stats without clearing
/// state. Used by unit tests that need to inspect counters mid-run.
#[doc(hidden)]
pub fn snapshot_zones() -> BTreeMap<String, ZoneStats> {
    PROFILE.with(|cell| cell.borrow().zones.clone())
}

/// Test-only helper: clear all state without writing a file.
#[doc(hidden)]
pub fn reset_for_test() {
    PROFILE.with(|cell| cell.borrow_mut().reset());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_stop_round_trip() {
        reset_for_test();
        let h = zone_start("zone_a");
        assert_eq!(h, 0);
        let elapsed = zone_stop(h);
        assert!(elapsed >= 0.0);
        let snap = snapshot_zones();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap["zone_a"].count, 1);
    }

    #[test]
    fn handle_is_monotonic() {
        reset_for_test();
        let a = zone_start("a");
        let b = zone_start("b");
        let c = zone_start("c");
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(c, 2);
        zone_stop(a);
        zone_stop(b);
        zone_stop(c);
    }

    #[test]
    fn nested_zones_accumulate_independently() {
        reset_for_test();
        let outer = zone_start("outer");
        let inner = zone_start("inner");
        zone_stop(inner);
        zone_stop(outer);
        let snap = snapshot_zones();
        assert_eq!(snap.len(), 2);
        assert_eq!(snap["outer"].count, 1);
        assert_eq!(snap["inner"].count, 1);
    }

    #[test]
    fn repeated_zone_accumulates_count() {
        reset_for_test();
        for _ in 0..10 {
            let h = zone_start("hot");
            zone_stop(h);
        }
        let snap = snapshot_zones();
        assert_eq!(snap["hot"].count, 10);
    }

    #[test]
    fn unknown_handle_returns_negative_one() {
        reset_for_test();
        let e = zone_stop(9999);
        assert!(e < 0.0);
    }

    #[test]
    fn dump_resets_state() {
        reset_for_test();
        let h = zone_start("zone_x");
        zone_stop(h);
        let tmp = std::env::temp_dir().join("cjc_profile_dump_resets_state.csv");
        let rows = dump_to_path(tmp.to_str().unwrap()).unwrap();
        assert_eq!(rows, 1);
        assert!(snapshot_zones().is_empty());
        let content = std::fs::read_to_string(&tmp).unwrap();
        assert!(content.starts_with(
            "zone_name,count,total_ns,min_ns,max_ns,mean_ns,stddev_ns\n"
        ));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn dump_csv_format_integer_columns() {
        reset_for_test();
        for _ in 0..3 {
            let h = zone_start("z");
            zone_stop(h);
        }
        let tmp = std::env::temp_dir().join("cjc_profile_dump_csv_format.csv");
        let rows = dump_to_path(tmp.to_str().unwrap()).unwrap();
        assert_eq!(rows, 1);
        let content = std::fs::read_to_string(&tmp).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);
        let fields: Vec<&str> = lines[1].split(',').collect();
        assert_eq!(fields.len(), 7);
        assert_eq!(fields[0], "z");
        // All numeric columns should parse as integers.
        for f in &fields[1..] {
            assert!(
                f.parse::<u128>().is_ok(),
                "column {f} is not an integer in v2.3 CSV"
            );
        }
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn dump_sort_order_hot_first() {
        reset_for_test();
        // "cold" has 1 call; "hot" has a longer explicit stats injection.
        let h_cold = zone_start("cold");
        zone_stop(h_cold);
        // Inject a synthetic hot zone via direct state manipulation (keeps
        // the test under 1 ms).
        PROFILE.with(|cell| {
            let mut state = cell.borrow_mut();
            let entry = state.zones.entry("hot".to_string()).or_default();
            entry.update(10_000_000_000); // 10s
        });
        let tmp = std::env::temp_dir().join("cjc_profile_dump_sort_order.csv");
        dump_to_path(tmp.to_str().unwrap()).unwrap();
        let content = std::fs::read_to_string(&tmp).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[1].starts_with("hot,"), "hot zone should be first");
        assert!(lines[2].starts_with("cold,"), "cold zone should be second");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn empty_dump_writes_header_only() {
        reset_for_test();
        let tmp = std::env::temp_dir().join("cjc_profile_empty_dump.csv");
        let rows = dump_to_path(tmp.to_str().unwrap()).unwrap();
        assert_eq!(rows, 0);
        let content = std::fs::read_to_string(&tmp).unwrap();
        assert_eq!(
            content,
            "zone_name,count,total_ns,min_ns,max_ns,mean_ns,stddev_ns\n"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn zone_stats_update_tracks_min_max() {
        let mut s = ZoneStats::default();
        s.update(100);
        s.update(50);
        s.update(200);
        assert_eq!(s.count, 3);
        assert_eq!(s.min_ns, 50);
        assert_eq!(s.max_ns, 200);
        assert_eq!(s.total_ns, 350);
    }

    #[test]
    fn mean_and_stddev_sane() {
        let mut s = ZoneStats::default();
        for ns in [100u128, 200, 300] {
            s.update(ns);
        }
        assert_eq!(s.mean_ns(), 200);
        // sum_sq = 10000 + 40000 + 90000 = 140000; /3 ≈ 46666; - 40000 = 6666
        // sqrt(6666) ≈ 81.6
        let sd = s.stddev_ns();
        assert!(sd > 70.0 && sd < 100.0, "unexpected stddev: {sd}");
    }
}
