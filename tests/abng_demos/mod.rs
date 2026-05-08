//! CJC-Lang demos for ABNG.
//!
//! Each demo lives as a `pub const SOURCE: &str = r#"..."#` const string
//! in its own module here; the matching root-level `tests/test_abng_*_cjcl.rs`
//! file uses `harness` to run the source through both `cjc-eval` (AST
//! interpreter) and `cjc-mir-exec` (MIR register-machine), and asserts the
//! printed output is byte-identical across executors.
//!
//! Why these demos exist alongside the Rust-side ones:
//!
//! * The Rust demos (`tests/test_abng_pinn_uncertainty.rs`,
//!   `tests/test_abng_tabular_gp.rs`,
//!   `tests/test_abng_lineage_attestation.rs`) exercise the cjc-abng
//!   crate's primitives byte-for-byte. They prove correctness at the
//!   crate boundary.
//! * The CJC-Lang demos here exercise the *language-level surface* —
//!   the `abng_*` builtins as called from `.cjcl` source. Every
//!   assertion runs through the entire pipeline (lexer → parser → AST →
//!   eval AND lexer → parser → AST → HIR → MIR → register-machine).
//!   AST↔MIR byte-equality is the strongest determinism gate in the
//!   project.
//!
//! The two layers are complementary: the Rust demos catch regressions
//! in the crate; the CJC-Lang demos catch regressions in the language
//! pipeline that affect ABNG behavior.

pub mod harness;
pub mod lineage_source;
pub mod pinn_source;
pub mod tabular_source;
pub mod ood_source;
pub mod adaptive_source;
pub mod calibration_source;
pub mod drift_source;
pub mod compact_source;
pub mod maturity_source;
