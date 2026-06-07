//! CANA Phase 3.5a — `fused_matmul_dot` test suite.
//!
//! Four flavors:
//!   - unit      → kernel-level correctness vs unfused reference
//!   - wiring    → builtin byte-identical across cjc-eval and cjc-mir-exec
//!   - proptest  → fused == unfused chain bit-for-bit over random shapes
//!   - fuzz      → bolero: dispatch survives arbitrary inputs without panic
//!
//! See [docs/cana/CANA_PHASE_3_5_FUSION_CODEGEN_DESIGN.md](../../docs/cana/CANA_PHASE_3_5_FUSION_CODEGEN_DESIGN.md).

pub mod unit;
pub mod wiring;
pub mod proptest;
pub mod fuzz;
pub mod cana_integration;
