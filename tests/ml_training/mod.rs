// Phase 1 — DatasetPlan test suite.
//
// Sub-modules:
//   - common:                shared test fixtures
//   - dataset_plan_unit:     construction, splits, batching, materialization
//   - dataset_plan_props:    property tests (added in step 4b)
//   - dataset_plan_parity:   determinism / cross-construction byte-equality (4c)
//   - dataset_plan_fuzz:     Bolero structural fuzz with naive oracle (4d)

pub mod common;
pub mod dataset_plan_unit;
pub mod dataset_plan_props;
pub mod dataset_plan_parity;
pub mod dataset_plan_fuzz;
