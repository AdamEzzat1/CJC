//! Runtime Policy Layer — integration test suite (green compute).
//!
//! Four flavors, matching the project's test discipline:
//!   - unit tests           → inline in `cjc-runtime/src/runtime_policy.rs`
//!   - wiring tests         → `wiring.rs`   (every builtin AST↔MIR byte-equal)
//!   - property tests       → `proptest.rs` (energy monotone/additive, round-trips)
//!   - fuzz tests           → `fuzz.rs`     (bolero: dispatch survives arbitrary input)
//!
//! Wired via `[[test]] name = "runtime_policy" path = "tests/runtime_policy/mod.rs"`
//! in the root Cargo.toml.

pub mod wiring;
pub mod proptest;
pub mod fuzz;
