# CJC v0.1 Hardening — Test Matrix

## Test Categories

### Unit Tests (179 tests)
Tests individual components in isolation.

| Component | File | Tests | Coverage Focus |
|-----------|------|-------|----------------|
| Lexer | test_lexer_hardening.rs | 15 | Empty input, boundaries, edge chars, Unicode, error recovery |
| Parser | test_parser_hardening.rs | 27 | All statement/expression forms, error paths, deeply nested |
| Type Checker | test_type_checker_hardening.rs | 13 | Literal inference, operators, functions, closures, structs |
| Builtins | test_runtime_builtins_hardening.rs | ~25 | Math, array, bitwise, statistics, constants |
| Eval | test_eval_hardening.rs | 22 | All language features through AST interpreter |
| MIR-exec | test_mir_exec_hardening.rs | 24 | All language features through MIR register machine |
| Dispatch | test_dispatch_hardening.rs | 11 | Operators via program execution (not direct API) |
| Data | test_data_hardening.rs | 8 | DataFrame construction, column types, edge cases |
| Snap | test_snap_hardening.rs | 8 | Roundtrip all types, SHA-256, error paths |
| Regex | test_regex_hardening.rs | 15 | Patterns, classes, anchors, NFA safety, flags |
| Repro | test_repro_hardening.rs | 11 | RNG determinism, Kahan accuracy, fork |

### Property Tests (25 tests)
proptest-based randomized testing.

| Component | File | Tests | Properties Verified |
|-----------|------|-------|---------------------|
| Lexer | test_lexer_props.rs | 4 | No panics on UTF-8/ASCII, determinism, nonempty output |
| Eval | test_eval_props.rs | 4 | Literal correctness, arithmetic in both executors |
| Snap | test_snap_props.rs | 6 | Roundtrip int/float/bool/string, determinism, no crash |
| Repro | test_repro_props.rs | 5 | RNG seed determinism, f64 range, fork, Kahan |
| Dispatch | test_dispatch_props.rs | 6 | Commutativity, identity, zero, eval-MIR parity |

### Fuzz Tests (8 tests)
Bolero-based (runs as proptest on Windows, coverage-guided on Linux).

| Target | Tests | Focus |
|--------|-------|-------|
| Lexer | 1 | UTF-8 crash resistance |
| Parser | 1 | UTF-8 crash resistance |
| MIR pipeline | 1 | Full pipeline crash resistance |
| Eval pipeline | 1 | Full pipeline crash resistance |
| Snap roundtrip | 1 | Encode/decode integrity |
| Regex | 1 | Pattern+input crash resistance |
| RNG | 1 | Seed determinism |
| Kahan | 1 | Accumulator determinism |

### Integration Tests (53 tests)
Cross-component verification.

| Area | File | Tests | Focus |
|------|------|-------|-------|
| Parity | test_wiring_parity.rs | 22 | Eval vs MIR-exec identical output |
| Builtins | test_wiring_builtins.rs | 15 | Builtins work through full pipeline |
| HIR/MIR | test_wiring_hir_mir.rs | 16 | AST→HIR→MIR lowering correctness |

### Determinism Tests (28 tests)
Verify bit-identical results across runs.

| Area | File | Tests | Focus |
|------|------|-------|-------|
| Execution | test_execution_determinism.rs | 14 | All features produce identical output across 10+ runs |
| Numerical | test_numerical_determinism.rs | 14 | Float ops, tensors, special values bit-identical |
