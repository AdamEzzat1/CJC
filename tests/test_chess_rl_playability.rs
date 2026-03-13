//! Chess RL Playability — Engine Interface Hardening Tests.
//!
//! These tests validate the chess engine's correctness at the boundary
//! where interactive play depends on it: move legality, board state
//! consistency, terminal detection, encoding determinism, and parity
//! between CJC engine and the expected behavior.

mod chess_rl_project;
mod chess_rl_playability;
