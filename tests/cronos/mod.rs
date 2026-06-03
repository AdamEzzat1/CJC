//! cjc-cronos integration / proptest / fuzz tests.
//!
//! **Status:** SCAFFOLDING. Only smoke tests live here today; the
//! per-method submodules (ETS, ARIMA, Kalman, STL) land across subsequent
//! implementation sessions per the handoff at
//! `CJC-Lang_Obsidian_Vault/10_Roadmap_and_Open_Questions/New Crate Stack — Cronos, Causal, Tempest.md` §6.2.
//!
//! Required minimums before v0.1 ships (handoff §6.1):
//!
//! | Bucket            | Count |
//! | ----------------- | ----- |
//! | Unit              | ≥ 30  |
//! | Integration       | ≥ 15  |
//! | Proptest          | ≥ 5   |
//! | Bolero fuzz       | ≥ 3   |
//!
//! Wired into the workspace's `[[test]]` table in the root `Cargo.toml`
//! so `cargo test --test cronos` runs everything here.

use cjc_cronos::{CronosError, FingerprintId, Frequency, TimeSeries};

#[test]
fn scaffold_reaches_crate() {
    // Foundational re-exports resolve from the integration-test boundary.
    let f = Frequency::Daily;
    assert_eq!(f.default_seasonal_period(), 7);

    let id = FingerprintId(0xCAFE_BABE);
    assert_eq!(format!("{}", id), "00000000cafebabe");
}

#[test]
fn cronos_error_display_is_stable() {
    let err = CronosError::UnknownColumn { name: "value".to_string() };
    assert_eq!(err.to_string(), "unknown column: value");
}

#[test]
fn time_series_construction_basic() {
    // Three-point monotonically increasing series.
    let time = vec![1, 2, 3];
    let values = vec![10.0, 20.0, 30.0];
    let ts = TimeSeries::new(time, values, Frequency::Daily).unwrap();
    assert_eq!(ts.len(), 3);
    assert!(!ts.is_empty());
    assert_eq!(ts.frequency(), Frequency::Daily);
    assert_eq!(ts.values()[1], 20.0);
}

#[test]
fn unsorted_time_index_is_rejected() {
    let time = vec![1, 3, 2]; // not monotonic at index 2
    let values = vec![10.0, 20.0, 30.0];
    let err = TimeSeries::new(time, values, Frequency::Daily).unwrap_err();
    assert!(matches!(err, CronosError::UnsortedTimeIndex { first_offending_row: 2 }));
}
