// Milestone 2.4 — Test Suite Entry Point
//
// Gate mapping:
//   G-8: tests/milestone_2_4/nogc_verifier/ — smuggle GC into nogc tests
//   G-10: full integration suite with optimizations enabled (parity + shape)
//
// Run milestone tests only:   cargo test --test test_milestone_2_4
// Run full suite (opts on):   cargo test

pub mod nogc_verifier;
pub mod optimizer;
pub mod parity;
pub mod shape;
