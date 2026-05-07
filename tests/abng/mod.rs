//! Integration tests for `cjc-abng`.
//!
//! Phase 0.1 modules:
//!
//! * [`unit`] — pure-Rust unit tests of `NodeStats`, `AdaptiveBeliefGraph`,
//!   and audit-chain primitives.
//! * [`replay`] — round-trip and double-run determinism tests at the
//!   serializer/replay boundary.
//! * [`dispatch`] — every Phase 0.1 `abng_*` builtin exercised via
//!   [`cjc_abng::dispatch_abng`] with happy-path and Err-path coverage.
//! * [`parity`] — AST eval ↔ MIR exec parity on `.cjcl` source that
//!   exercises the public Phase 0.1 builtins.
//!
//! Phase 0.2 modules:
//!
//! * [`multinode`] — multi-node arena, children promotion, codebook freeze,
//!   descend routing, per-node stats chain decoupling.
//! * [`dispatch_p2`] — the new Phase 0.2 builtins (add_node, set_codebook,
//!   encode_prefix, descend, route_path, etc.) at the dispatch layer.
//! * [`parity_p2`] — AST eval ↔ MIR exec parity for the Phase 0.2 builtins.

mod blr_tests;
mod decision_tests;
mod dispatch;
mod dispatch_p2;
mod dispatch_p3a;
mod dispatch_p3b;
mod dispatch_p3c;
mod dispatch_p3d;
mod expected_epistemic_tests;
mod leaf_head_tests;
mod maturity_signature_tests;
mod multinode;
mod parity;
mod parity_p2;
mod parity_p3a;
mod parity_p3b;
mod parity_p3c;
mod parity_p3d;
mod replay;
mod uncertainty_tests;
mod unit;
