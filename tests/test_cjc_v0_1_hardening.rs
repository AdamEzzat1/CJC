//! CJC v0.1 Hardening Audit — Master Test File
//!
//! This file includes all hardening tests organized by category:
//! - unit/       — Focused local correctness tests
//! - prop/       — Property-based invariant tests (proptest)
//! - fuzz/       — Bolero fuzzing harnesses (Windows-compatible)
//! - integration/ — Cross-layer wiring verification
//! - determinism/ — Reproducibility and deterministic behavior
//!
//! Run all hardening tests:
//!   cargo test --test test_cjc_v0_1_hardening
//!
//! Run a specific category:
//!   cargo test --test test_cjc_v0_1_hardening unit::
//!   cargo test --test test_cjc_v0_1_hardening prop::
//!   cargo test --test test_cjc_v0_1_hardening fuzz::
//!   cargo test --test test_cjc_v0_1_hardening integration::
//!   cargo test --test test_cjc_v0_1_hardening determinism::

#[path = "cjc_v0_1_hardening/mod.rs"]
mod cjc_v0_1_hardening;

// Re-export submodules so Cargo discovers all #[test] functions
mod unit {
    #[path = "../cjc_v0_1_hardening/unit/test_lexer_hardening.rs"]
    pub mod test_lexer_hardening;

    #[path = "../cjc_v0_1_hardening/unit/test_parser_hardening.rs"]
    pub mod test_parser_hardening;

    #[path = "../cjc_v0_1_hardening/unit/test_type_checker_hardening.rs"]
    pub mod test_type_checker_hardening;

    #[path = "../cjc_v0_1_hardening/unit/test_runtime_builtins_hardening.rs"]
    pub mod test_runtime_builtins_hardening;

    #[path = "../cjc_v0_1_hardening/unit/test_eval_hardening.rs"]
    pub mod test_eval_hardening;

    #[path = "../cjc_v0_1_hardening/unit/test_mir_exec_hardening.rs"]
    pub mod test_mir_exec_hardening;

    #[path = "../cjc_v0_1_hardening/unit/test_dispatch_hardening.rs"]
    pub mod test_dispatch_hardening;

    #[path = "../cjc_v0_1_hardening/unit/test_data_hardening.rs"]
    pub mod test_data_hardening;

    #[path = "../cjc_v0_1_hardening/unit/test_snap_hardening.rs"]
    pub mod test_snap_hardening;

    #[path = "../cjc_v0_1_hardening/unit/test_regex_hardening.rs"]
    pub mod test_regex_hardening;

    #[path = "../cjc_v0_1_hardening/unit/test_repro_hardening.rs"]
    pub mod test_repro_hardening;
}

mod prop {
    #[path = "../cjc_v0_1_hardening/prop/test_lexer_props.rs"]
    pub mod test_lexer_props;

    #[path = "../cjc_v0_1_hardening/prop/test_eval_props.rs"]
    pub mod test_eval_props;

    #[path = "../cjc_v0_1_hardening/prop/test_snap_props.rs"]
    pub mod test_snap_props;

    #[path = "../cjc_v0_1_hardening/prop/test_repro_props.rs"]
    pub mod test_repro_props;

    #[path = "../cjc_v0_1_hardening/prop/test_dispatch_props.rs"]
    pub mod test_dispatch_props;
}

mod fuzz {
    #[path = "../cjc_v0_1_hardening/fuzz/test_fuzz_hardening.rs"]
    pub mod test_fuzz_hardening;
}

mod integration {
    #[path = "../cjc_v0_1_hardening/integration/test_wiring_parity.rs"]
    pub mod test_wiring_parity;

    #[path = "../cjc_v0_1_hardening/integration/test_wiring_builtins.rs"]
    pub mod test_wiring_builtins;

    #[path = "../cjc_v0_1_hardening/integration/test_wiring_hir_mir.rs"]
    pub mod test_wiring_hir_mir;
}

mod determinism {
    #[path = "../cjc_v0_1_hardening/determinism/test_execution_determinism.rs"]
    pub mod test_execution_determinism;

    #[path = "../cjc_v0_1_hardening/determinism/test_numerical_determinism.rs"]
    pub mod test_numerical_determinism;
}
