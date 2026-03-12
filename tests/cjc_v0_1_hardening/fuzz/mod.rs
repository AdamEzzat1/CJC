//! Bolero fuzzing harnesses for CJC v0.1 hardening.
//!
//! On Windows, Bolero runs via proptest backend (no coverage-guided fuzzing).
//! Run: cargo test --test test_cjc_v0_1_hardening -- fuzz
//!
//! On Linux CI, promote to coverage-guided:
//!   cargo bolero test test_cjc_v0_1_hardening::fuzz::fuzz_*

mod test_fuzz_hardening;
