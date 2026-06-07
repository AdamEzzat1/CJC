//! In-memory recorder of compiler-pass outcomes.
//!
//! Phase 1 collects pass records but never reads them — the Phase-1 cost
//! model is [`crate::NullCostModel`] which has no opinion to inform. The
//! point of shipping `PassHistory` now is to give Phase 2 a stable
//! call-site shape: any compiler pass already knows how to call
//! `history.record(...)` after running.
//!
//! ## What Phase 1 records
//!
//! For each pass invocation:
//!
//! - The input [`ProgramHash`] (before the pass ran)
//! - The pass name (`"constant_fold"`, `"dce"`, etc.)
//! - The output [`ProgramHash`] (after the pass ran)
//! - A small structured `outcome` (changed/no-op/error)
//!
//! Phase 5 (profile-guided) will extend `PassRecord` with measured runtime
//! and memory; the schema is forward-compatible.

use std::collections::VecDeque;

use crate::hash::ProgramHash;

// ---------------------------------------------------------------------------
// PassRecord
// ---------------------------------------------------------------------------

/// A single compiler-pass outcome record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassRecord {
    /// Name of the pass that ran (e.g., `"constant_fold"`, `"dce"`).
    pub pass_name: String,
    /// Program hash *before* the pass ran.
    pub input_hash: ProgramHash,
    /// Program hash *after* the pass ran.
    pub output_hash: ProgramHash,
    /// Structured outcome category.
    pub outcome: PassOutcome,
}

/// What happened when a pass ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassOutcome {
    /// The pass mutated the program (input_hash != output_hash).
    Changed,
    /// The pass ran but produced an identical program (true no-op).
    NoOp,
    /// The pass was skipped before running. `reason` is a stable enum value
    /// (not a free-form string) so Phase-2 analytics can aggregate over it.
    Skipped(SkipReason),
}

/// Why a pass was skipped — kept structured for downstream aggregation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// The legality gate rejected the recommendation.
    LegalityGate,
    /// The pass was explicitly disabled (CLI flag, attribute, etc.).
    UserDisabled,
    /// The cost model predicted negative or zero net benefit.
    CostBelowThreshold,
    /// Some other compiler-internal condition.
    Other,
}

impl PassRecord {
    /// True if `input_hash == output_hash`. Useful for cheap "did this pass
    /// do anything?" checks.
    pub fn is_noop(&self) -> bool {
        matches!(self.outcome, PassOutcome::NoOp) || self.input_hash == self.output_hash
    }
}

// ---------------------------------------------------------------------------
// PassHistory — bounded in-memory ring
// ---------------------------------------------------------------------------

/// Bounded in-memory record of recent pass outcomes.
///
/// Uses a [`VecDeque`] under the hood so that exceeding `capacity` drops
/// the oldest record. The default capacity is `4096` — large enough for any
/// realistic compilation session, small enough to fit comfortably in memory.
///
/// `PassHistory` is `Send + Sync`-safe via standard mutex idioms; Phase 1
/// only uses it single-threaded, so we don't ship a lock type yet.
#[derive(Debug, Clone)]
pub struct PassHistory {
    capacity: usize,
    records: VecDeque<PassRecord>,
}

impl PassHistory {
    /// Construct a history with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            records: VecDeque::new(),
        }
    }

    /// Construct a history with the default capacity (`4096`).
    pub fn new() -> Self {
        Self::with_capacity(4096)
    }

    /// Push a new record. If `len() == capacity`, the oldest record is
    /// dropped.
    pub fn record(&mut self, record: PassRecord) {
        if self.records.len() >= self.capacity {
            self.records.pop_front();
        }
        self.records.push_back(record);
    }

    /// Number of records currently held.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// True if no records have been pushed.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Iterate records in insertion order (oldest first).
    pub fn iter(&self) -> impl Iterator<Item = &PassRecord> {
        self.records.iter()
    }

    /// Most recent record (`None` if empty).
    pub fn latest(&self) -> Option<&PassRecord> {
        self.records.back()
    }
}

impl Default for PassHistory {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(name: &str, before: u64, after: u64) -> PassRecord {
        PassRecord {
            pass_name: name.to_string(),
            input_hash: ProgramHash(before),
            output_hash: ProgramHash(after),
            outcome: if before == after {
                PassOutcome::NoOp
            } else {
                PassOutcome::Changed
            },
        }
    }

    #[test]
    fn empty_history_is_empty() {
        let h = PassHistory::new();
        assert_eq!(h.len(), 0);
        assert!(h.is_empty());
        assert!(h.latest().is_none());
    }

    #[test]
    fn record_grows_and_latest_reports_back() {
        let mut h = PassHistory::with_capacity(4);
        h.record(rec("cf", 1, 2));
        h.record(rec("dce", 2, 3));
        assert_eq!(h.len(), 2);
        assert_eq!(h.latest().unwrap().pass_name, "dce");
    }

    #[test]
    fn ring_buffer_drops_oldest() {
        let mut h = PassHistory::with_capacity(3);
        for i in 0..5u64 {
            h.record(rec("cf", i, i + 1));
        }
        assert_eq!(h.len(), 3);
        // Oldest two were dropped — first surviving is i=2.
        let names: Vec<_> = h
            .iter()
            .map(|r| r.input_hash.0)
            .collect();
        assert_eq!(names, vec![2, 3, 4]);
    }

    #[test]
    fn capacity_one_works() {
        let mut h = PassHistory::with_capacity(1);
        h.record(rec("a", 1, 2));
        h.record(rec("b", 2, 3));
        assert_eq!(h.len(), 1);
        assert_eq!(h.latest().unwrap().pass_name, "b");
    }

    #[test]
    fn zero_capacity_clamps_to_one() {
        // Defensive: capacity 0 would lock the structure useless;
        // we promote it to 1.
        let mut h = PassHistory::with_capacity(0);
        h.record(rec("a", 0, 1));
        assert_eq!(h.len(), 1);
    }

    #[test]
    fn noop_detection() {
        let r = rec("cf", 7, 7);
        assert!(r.is_noop());
        let r2 = rec("dce", 1, 2);
        assert!(!r2.is_noop());
    }

    #[test]
    fn skip_reasons_are_distinct() {
        // Compile-time check that every variant is constructible.
        let _ = PassOutcome::Skipped(SkipReason::LegalityGate);
        let _ = PassOutcome::Skipped(SkipReason::UserDisabled);
        let _ = PassOutcome::Skipped(SkipReason::CostBelowThreshold);
        let _ = PassOutcome::Skipped(SkipReason::Other);
    }
}
