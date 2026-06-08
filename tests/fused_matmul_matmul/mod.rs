//! CANA Phase 3.5d — `fused_matmul_matmul` test suite.
//!
//! Third tensor-level fused primitive. The chain `matmul(matmul(A, B), C)`
//! is the canonical 3-matrix product that appears in attention layers,
//! deep MLPs, and tensor-train decompositions. Math is left-associative.

pub mod unit;
pub mod wiring;
pub mod proptest;
pub mod cana_integration;
