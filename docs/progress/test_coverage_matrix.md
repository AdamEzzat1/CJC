# CJC Test Coverage Matrix

**Last Updated:** 2025-01-01
**Total Tests:** 1,692+ (audit baseline: 535 at Stage 2.4)

This matrix maps each CJC feature to the test files that cover it, organized by test category.

## Coverage Categories

| Category | Description |
|----------|-------------|
| **Unit** | Tests within a single crate or focused on one component |
| **Integration** | Tests that exercise multiple crates together |
| **Audit** | Tests added during the Phase 2 audit / hardening cycle |
| **Milestone** | Tests added for specific milestone gates (2.4, 2.5, 2.6) |
| **Proptest** | Property-based tests using the `proptest` crate |
| **Fixture** | End-to-end golden-file tests in `tests/fixtures/` |

---

## Feature Coverage Table

| Feature | Unit Tests | Integration / Milestone | Audit Tests | Proptest | Fixture |
|---------|-----------|------------------------|-------------|---------|---------|
| **Lexer** | `tests/test_lexer.rs` | — | `audit_tests/test_phase2_fstring.rs` | `prop_tests/parser_props.rs` | `fixtures/basic/` |
| **Parser** | `tests/test_parser.rs` | — | `audit_tests/test_phase2_fstring.rs`, `audit_tests/test_phase2_const_exprs.rs` | `prop_tests/parser_props.rs` | `fixtures/` |
| **Type Checker** | `tests/test_types.rs` | — | `audit_tests/test_phase2_mutable_binding.rs`, `audit_tests/test_phase2_impl_trait_syntax.rs` | `prop_tests/type_checker_props.rs` | `fixtures/error_cases/` |
| **HIR Lowering** | `tests/test_hir.rs` | `milestone_2_4/parity/` | — | — | — |
| **MIR** | `tests/test_mir.rs` | `milestone_2_4/` | `audit_tests/test_audit_mir_form.rs`, `audit_tests/test_audit_cfg_ssa.rs` | — | — |
| **MIR Executor** | `tests/test_mir_exec.rs` | `milestone_2_4/parity/` | — | `prop_tests/round_trip_props.rs` | `fixtures/` |
| **AST Evaluator** | `tests/test_eval.rs` | `milestone_2_4/parity/` | — | — | — |
| **Parity (eval==mir)** | — | `milestone_2_4/parity/` | — | `prop_tests/round_trip_props.rs` | — |
| **Tail-Call Optimization** | — | — | `audit_tests/test_phase2_tco.rs` | — | `fixtures/tco/` |
| **Closures** | `tests/test_closures.rs` | `milestone_2_4/parity/` | — | — | `fixtures/closures/` |
| **Match Patterns** | `tests/test_match_patterns.rs` | `milestone_2_6/exhaustiveness.rs` | `audit_tests/test_audit_match_exhaustiveness.rs` | — | `fixtures/match_patterns/` |
| **For Loops** | `tests/test_for_loops.rs` | — | — | — | `fixtures/for_loops/` |
| **Structs** | — | — | `audit_tests/test_audit_datatype_inventory_smoke.rs` | — | `fixtures/structs/` |
| **Enums** | — | `milestone_2_6/enums.rs` | `audit_tests/test_audit_datatype_inventory_smoke.rs` | — | `fixtures/enums/` |
| **Generics / Monomorphization** | — | `milestone_2_5/monomorph.rs`, `milestone_2_6/monomorph.rs` | `audit_tests/test_phase2_monomorphization.rs` | — | `fixtures/generics/` |
| **F-Strings** | — | — | `audit_tests/test_phase2_fstring.rs` | — | `fixtures/fstring/` |
| **Const Expressions** | — | — | `audit_tests/test_phase2_const_exprs.rs` | — | `fixtures/const_expr/` |
| **Mutable Bindings (E0150)** | — | — | `audit_tests/test_phase2_mutable_binding.rs` | `prop_tests/type_checker_props.rs` | `fixtures/error_cases/` |
| **Trait Bounds (E0300)** | — | — | `audit_tests/test_phase2_impl_trait_syntax.rs` | — | — |
| **Option/Result Builtins** | — | `milestone_2_6/option_result.rs` | `audit_tests/test_phase2_result_option.rs` | — | `fixtures/match_patterns/` |
| **Matmul (serial)** | `tests/test_runtime.rs` | `hardening_tests/test_h5_matmul_alloc_free.rs` | `audit_tests/test_audit_matmul_path.rs` | — | `fixtures/numeric/` |
| **Parallel Matmul** | — | — | `audit_tests/test_audit_parallel_matmul.rs` | — | — |
| **Tensor Ops** | `tests/test_runtime.rs` | `tests/test_numerical_fortress.rs` | — | — | `fixtures/numeric/` |
| **Shape Inference** | — | `milestone_2_4/shape/` | `audit_tests/test_audit_shape_inference.rs` | — | — |
| **NoGC Verifier** | — | `milestone_2_4/nogc_verifier/` | — | — | — |
| **Optimizer (CF+DCE)** | — | `milestone_2_4/optimizer/` | — | — | — |
| **CFG / Phi Nodes / Use-Def** | — | — | `audit_tests/test_audit_cfg_ssa.rs`, `hardening_tests/test_h4_mir_cfg.rs` | — | — |
| **Vec COW (Array/Tuple)** | — | `milestone_2_4/parity/` | `audit_tests/test_audit_cow_array.rs` | — | — |
| **Numeric Types (i8..u128,f16)** | — | — | `audit_tests/test_audit_numeric_types.rs` | `prop_tests/type_checker_props.rs` | — |
| **Collections (Set, Queue)** | — | — | `audit_tests/test_audit_collections.rs` | — | — |
| **ML Types (DType, QuantTensor)** | — | — | `audit_tests/test_audit_ml_types.rs` | — | — |
| **SparseMatrix Methods** | — | `milestone_2_5/sparse.rs` | `audit_tests/test_audit_ml_types.rs` | — | — |
| **Auto-Diff** | `tests/test_ad.rs` | — | — | — | `fixtures/ad/` |
| **DataFrame / CSV** | `tests/test_data.rs`, `tests/test_phase8_data_logistics.rs` | `tidy_tests/` | `audit_tests/test_audit_module_system.rs` | — | — |
| **Regex** | `tests/test_regex.rs` | — | — | — | — |
| **Bytes / Strings** | `tests/test_bytes_strings.rs` | — | `audit_tests/test_audit_type_error_spans.rs` | — | `fixtures/basic/` |
| **Determinism / Repro** | `tests/test_determinism.rs`, `tests/test_repro.rs` | — | `audit_tests/test_audit_parallelism_absence.rs` | — | — |
| **Bf16 / F16** | `tests/test_f16_precision.rs` | `milestone_2_6/bf16.rs` | — | — | — |
| **Complex Numbers** | `tests/test_complex_blas.rs` | — | — | — | — |
| **Quantized BLAS** | `tests/test_quantized_blas.rs` | — | — | — | — |
| **Transformer** | `tests/test_transformer.rs` | — | — | — | — |
| **CNN** | `tests/test_phase6_cnn.rs`, `tests/test_phase7_cnn2d.rs` | — | — | — | — |
| **Diagnostics** | `tests/test_diag.rs` | — | `audit_tests/test_audit_type_error_spans.rs` | — | `fixtures/error_cases/` |

---

## Coverage Gaps

The following features have **no** property-based or fixture coverage:

| Feature | Gap | Priority |
|---------|-----|----------|
| MIR CFG/SSA | No proptest for CFG builder correctness | P2 |
| Generics | No fixture for complex generic programs | P2 |
| Optimizer | No fixture testing optimizer output | P2 |
| NoGC Verifier | No proptest for annotation parsing | P3 |
| Regex | No fixture for regex patterns | P3 |
| Transformer Kernel | No fixture test (too large) | P3 |

---

## Test Count by Category

| Category | File Count | Test Count (approx) |
|----------|-----------|---------------------|
| Root-level tests | 50 | ~1,200 |
| Audit tests | 17+ | ~130 |
| Hardening tests | 6 | ~51 |
| Milestone 2.4 | 5 | ~62 |
| Milestone 2.5 | 13 | ~100 |
| Milestone 2.6 | 8 | ~80 |
| Tidy tests | 28 | ~100 |
| Proptest | 3 | ~60 (runs 256 cases each) |
| Fixtures | 20+ | 20 |
| **Total** | **150+** | **~1,800+** |
