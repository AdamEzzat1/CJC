//! Chess RL v2 — the upgraded flagship demo.
//!
//! This module builds a more serious reinforcement learning chess demo on top
//! of CJC-Lang's current capabilities. See `docs/chess_rl_v2/README.md` for
//! the full architecture writeup.
//!
//! Layout:
//!   - `source`  — CJC-Lang source as `pub const` strings (engine, features,
//!                 model, training, eval). Concatenated into a single program.
//!   - `harness` — Rust-side test helpers (run program, parse outputs).
//!   - `test_*`  — Rust integration tests organized by concern.

pub mod harness;
pub mod source;

pub mod test_engine;
pub mod test_model;
pub mod test_training;
pub mod test_parity;
