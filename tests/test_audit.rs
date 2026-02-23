// CJC Audit Test Suite — Reality Check Phase
//
// Run all audit tests:
//   cargo test --test test_audit
//
// Sub-suites:
//   cargo test --test test_audit trait_dispatch
//   cargo test --test test_audit module_system
//   cargo test --test test_audit match_exhaustiveness
//   cargo test --test test_audit mir_form
//   cargo test --test test_audit type_error_spans
//   cargo test --test test_audit float_const_folding
//   cargo test --test test_audit matmul_path
//   cargo test --test test_audit parallelism_absence
//   cargo test --test test_audit datatype_inventory

mod audit_tests;
