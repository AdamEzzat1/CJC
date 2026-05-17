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

mod baseline_wisconsin_bc;
mod batch_observe_tests;
mod demo_a2_fused_training;
mod demo_a3_merkle_proof;
mod demo_b3_zstd_snapshot;
mod demo_c2_parallel_verify;
mod blr_cache_tests;
mod blr_checkpoint_tests;
mod blr_feature_version_tests;
mod blr_numerical_rescue_tests;
mod blr_predict_fallback_tests;
mod blr_tests;
mod compact_log_tests;
mod decide_step_canary_tests;
mod decision_tests;
mod dispatch;
mod dispatch_p2;
mod dispatch_p3a;
mod dispatch_p3b;
mod dispatch_p3c;
mod dispatch_p3d;
mod expected_epistemic_tests;
mod leaf_head_tests;
mod leaf_params_batch_tests;
mod maturity_signature_tests;
mod merge_math_tests;
mod merkle_tests;
mod multinode;
mod observe_validation_tests;
mod parallel_route_tests;
mod parity;
mod per_thread_arena_tests;
mod parity_p2;
mod parity_p3a;
mod parity_p3b;
mod parity_p3c;
mod parity_p3d;
mod provenance_tests;
mod replay;
mod replay_invariant_tests;
mod serialize_compressed;
mod serialize_mmap;
mod serialize_streaming;
mod route_cache_tests;
mod route_entropy_grow_tests;
mod route_trace_tests;
mod split_nll_gate_tests;
mod train_step_v14_tests;
mod uncertainty_tests;
mod unit;
mod verify_par_tests;

// Phase 0.9.5 COMMIT 3 — categorical subsystem property + fuzz suites.
mod categorical_fuzz;
mod categorical_proptest;

// Phase 0.9.5 COMMIT 5 — Dataset-A (Diabetes-130) benchmark harness.
mod dataset_a_diabetes130;
