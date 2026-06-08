//! CANA Phase 3.5a-prime — `fused_matmul_norm` test suite.
//!
//! Second fused tensor primitive following the Phase 3.5a pattern. Unlike
//! `fused_matmul_dot`, the unfused chain `norm(matmul(A, W))` composes
//! naturally in CJC-Lang — both ops accept 2D tensors. This makes
//! `fused_matmul_norm` the first auto-rewriteable target for the (future)
//! Phase 3.5c MIR rewriter.

pub mod unit;
pub mod wiring;
pub mod proptest;
pub mod fuzz;
pub mod cana_integration;
