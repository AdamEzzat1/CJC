// Phase 1: Deterministic ML Training Stack — DatasetPlan test harness.
//
// Run standalone:
//   cargo test --test test_ml_training
//
// Sub-suites:
//   unit:    cargo test --test test_ml_training dataset_plan_unit
//   props:   cargo test --test test_ml_training dataset_plan_props
//   parity:  cargo test --test test_ml_training dataset_plan_parity
//   fuzz:    cargo test --test test_ml_training dataset_plan_fuzz

pub mod ml_training;
