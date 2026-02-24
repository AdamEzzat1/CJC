# CJC Phase F5→F6 Audit: Hardening + De-Dup + Optimizer + Type Parity + QA

## Baseline Snapshot (Step 0)

**Date:** 2026-02-23
**Workspace:** 17 crates, ~92K LOC

### Test Baseline
- All workspace tests: **PASS** (0 failures)
- Total test suites: 83 `test result:` lines
- Ignored tests: 8 (3 doc-tests requiring external resources, 5 in integration)
- Fixture tests: 14 .cjc files, all passing via MIR-exec runner

### Fixture Inventory (14 files)
1. tests/fixtures/basic/hello_world.cjc
2. tests/fixtures/basic/arithmetic.cjc
3. tests/fixtures/numeric/float_ops.cjc
4. tests/fixtures/for_loops/range_loop.cjc
5. tests/fixtures/tco/tail_call.cjc
6. tests/fixtures/fstring/interpolation.cjc
7. tests/fixtures/enums/option_basic.cjc
8. tests/fixtures/error_cases/type_error.cjc
9. tests/fixtures/structs/basic_struct.cjc
10. tests/fixtures/closures/basic_closure.cjc
11. tests/fixtures/regex/basic_match.cjc
12. tests/fixtures/regex/regex_variables.cjc
13. tests/fixtures/bytes/byte_strings.cjc
14. tests/fixtures/complex/complex_basic.cjc

### Fuzz Targets (3)
1. fuzz/fuzz_targets/lexer.rs
2. fuzz/fuzz_targets/parser.rs
3. fuzz/fuzz_targets/complex_eval.rs

### Known Issues at Baseline
- 6 compiler warnings in cjc-mir-exec (unreachable patterns, unused variables)
- cjc-cli: 0 tests
- gc.rs, tensor.rs, value.rs: 0 inline tests
- Type::Complex not in Type enum (Value::Complex exists)
- Type::I32/F32 defined but no corresponding Value variants used
- Type::Range/Slice exist but no Value::Range/Value::Slice
- Runtime-only values (Scratchpad, PagedKvCache, AlignedBytes) have no Type equivalents
- ~1700 LOC duplicated between cjc-eval and cjc-mir-exec
- cjc-data/src/lib.rs: 5927 LOC monolith
- MIR optimizer: only CF + DCE + CF (missing SR, CSE, LICM)

---

## Step-by-Step Implementation Log

### Step 1: FStringLit MIR-exec Parity
**Status:** ALREADY WORKING
- FStringLit is desugared in HIR (cjc-hir) into Binary::Add chains with to_string() calls
- MIR-exec handles these through standard Binary/Call evaluation
- Existing fixture tests/fixtures/fstring/interpolation.cjc passes via MIR-exec
- No code changes needed for basic parity
- Added: additional parity test fixtures for edge cases

