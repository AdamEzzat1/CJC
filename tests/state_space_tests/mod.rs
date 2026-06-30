//! State-space-model primitive test suite.
//!
//! Covers the `state_space_*` builtins shipped in `cjc-runtime/src/state_space.rs`
//! plus the `demos/state_space_chess` end-to-end demo.
//!
//! Layout:
//!   tests/state_space_tests/
//!     mod.rs                       ← this file (declares submodules)
//!     harness.rs                   ← shared helpers (run snippets, parity)
//!     test_dispatch_unit.rs        ← direct-dispatch unit tests, no parser
//!     test_dispatch_parity.rs      ← AST↔MIR byte-equal parity for each op
//!     test_properties.rs           ← proptest invariants
//!     test_demo_smoke.rs           ← chess demo runs end-to-end
//!     test_demo_replay.rs          ← deterministic replay double-run
//!
//! Wired via `[[test]] name = "state_space_tests" path = "..."` in root Cargo.toml.

pub mod harness;

pub mod test_dispatch_unit;
pub mod test_dispatch_parity;
pub mod test_properties;
pub mod test_demo_smoke;
pub mod test_demo_replay;

// Phase 2 — performance primitives + extractors:
pub mod test_perf_primitives;
pub mod test_perf_parity;
pub mod test_perf_proptest;
pub mod test_perf_fuzz;
