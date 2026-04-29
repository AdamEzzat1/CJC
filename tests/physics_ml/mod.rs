//! Physics ML benchmark suite — Phases 1 + 2 + 3.
//!
//! Each PDE has its own submodule. Tests verify (a) accuracy against the
//! analytical or external reference, (b) bit-identical replay under fixed
//! seed, and (c) numerical sanity (no NaN/Inf, residual bounded).
//!
//! Layout:
//!   tests/physics_ml/
//!     mod.rs                ← this file (declares submodules)
//!     common.rs             ← shared helpers (bit-hash, threshold defs)
//!     pinn_heat_1d.rs       ← 1D heat equation benchmark    (Phase 1)
//!     pinn_wave_1d.rs       ← 1D wave equation benchmark    (Phase 2)
//!     pinn_burgers_1d.rs    ← 1D viscous Burgers benchmark  (Phase 2)
//!     pinn_allen_cahn_1d.rs ← 1D Allen-Cahn benchmark       (Phase 3)
//!     pinn_kdv_1d.rs        ← 1D KdV soliton benchmark      (Phase 3)
//!
//! Wired via `[[test]] name = "physics_ml" path = "tests/physics_ml/mod.rs"`
//! in the root Cargo.toml.

pub mod common;
pub mod pinn_heat_1d;
pub mod pinn_wave_1d;
pub mod pinn_burgers_1d;
pub mod pinn_allen_cahn_1d;
pub mod pinn_kdv_1d;
pub mod grad_graph_wiring;
pub mod grad_graph_proptest;
pub mod grad_graph_fuzz;
pub mod heat_1d_pure_cjcl_parity;
