//! CANA Phase 3.5c — end-to-end MIR rewriter tests.
//!
//! The unit tests in `crates/cjc-mir/src/fusion_rewrite.rs` exercise the
//! pattern matcher against hand-built MIR. This suite drives the rewriter
//! through the real pipeline:
//!
//!   parse_source → AST → MIR → fusion_rewrite → MIR-exec
//!
//! and proves byte-identical output to the unrewritten chain in both
//! executors. That is the rewriter's correctness contract.

pub mod parity;
pub mod identification;
