//! Final Phase Hardening Before VM — Comprehensive test suite
//!
//! Tests all features implemented in the pre-VM hardening pass:
//! - Tier 1: Module visibility, trait/impl dispatch, cross-file diagnostics
//! - Tier 2: Pattern exhaustiveness, string builtins, monomorphization
//! - Tier 3: Parallel stubs, package scaffold, REPL improvements
//! - AD: Gradient clipping
//! - Parity: AST-eval vs MIR-exec agreement on all new features

mod test_module_visibility;
mod test_trait_impl;
mod test_diagnostics;
mod test_exhaustiveness;
mod test_string_builtins;
mod test_ad_clip_grad;
mod test_parity;
