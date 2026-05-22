//! GC-06 Phase 3a — fused elementwise op test suite.
//!
//! Four flavors:
//!   - unit      → explicit values, non-contiguous fallback, shape errors
//!   - wiring    → fused builtins byte-identical across cjc-eval and cjc-mir-exec
//!   - proptest  → each fused kernel == its unfused sequence (the core guarantee)
//!   - fuzz      → bolero: dispatch survives arbitrary values/shapes
//!
//! Wired via `[[test]] name = "fused_ops" path = "tests/fused_ops/mod.rs"`.

pub mod unit;
pub mod wiring;
pub mod proptest;
pub mod fuzz;
