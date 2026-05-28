//! Locke integration / property / fuzz tests.
//!
//! Convention: each submodule is a self-contained test file. The whole
//! module is wired into the workspace's `[[test]]` table in the root
//! `Cargo.toml` so `cargo test --test locke` runs everything here.

mod algebra_tests;
mod validation_tests;
mod drift_tests;
mod lineage_tests;
mod belief_tests;
mod categorical_tests;
mod causal_tests;
mod column_summary_tests;
mod determinism_tests;
mod locke_proptest;
mod locke_fuzz;
mod language_builtins;
mod multiclass_leakage_tests;
mod pii_tests;
mod seasonality_tests;
mod shape_tests;
mod snapshot_tests;
mod ground_truth_tests;
mod scale_benchmark;
