//! CJC v0.1 Hardening Audit — Test Suite
//!
//! This module organizes the hardening tests into five categories:
//! - unit/       — Focused local correctness tests
//! - prop/       — Property-based invariant tests (proptest)
//! - fuzz/       — Bolero fuzzing harnesses (Windows-compatible)
//! - integration/ — Cross-layer wiring verification
//! - determinism/ — Reproducibility and deterministic behavior

pub mod helpers;
