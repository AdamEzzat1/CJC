mod test_audit_trait_dispatch;
mod test_audit_module_system;
mod test_audit_match_exhaustiveness;
mod test_audit_mir_form;
mod test_audit_type_error_spans;
mod test_audit_float_const_folding;
mod test_audit_matmul_path;
mod test_audit_parallelism_absence;
mod test_audit_datatype_inventory_smoke;

// Phase 2 Core Hardening tests
mod test_phase2_mutable_binding;
mod test_phase2_const_exprs;
mod test_phase2_tco;
mod test_phase2_fstring;
mod test_phase2_result_option;
mod test_phase2_impl_trait_syntax;
mod test_phase2_monomorphization;

// F5 Complex Numbers audit tests
mod test_complex_f64_runtime;
