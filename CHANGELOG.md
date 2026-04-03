# Changelog

All notable changes to CJC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.1] — 2026-04-02

### Quantum Hardening Phase 3

- **Jordan-Wigner transformation** (`fermion` module): Pauli algebra, `PauliTerm` with multiply/weight, `jw_one_body`/`jw_two_body` transforms, pre-built H₂ and LiH molecular Hamiltonians, Kahan-summed expectation values
- **Suzuki-Trotter time evolution** (`trotter` module): 1st-order (Lie-Trotter) and 2nd-order (Strang splitting) product formulas, diagonal Pauli rotation optimization, Trotter error bounds
- **Zero-Noise Extrapolation** (`mitigation` module): Richardson extrapolation with Vandermonde solve (partial pivoting), linear 2-point extrapolation, noise scaling for depolarizing/dephasing/amplitude-damping channels, `run_zne` workflow helper
- **MPS canonical form**: `left_canonicalize`, `right_canonicalize`, `mixed_canonicalize` via SVD-based orthogonalization with sign stabilization
- **MPS SWAP networks**: `apply_gate_swap_network` for arbitrary-distance 2-qubit gates, `apply_two_qubit_gate` for 4×4 unitary on adjacent sites
- **Pure CJC backends**: `PureFermionicHamiltonian`, `pure_h2_hamiltonian`, `pure_trotter_evolve`, `pure_richardson_extrapolate` — all inspectable, AD-compatible
- **Dispatch wiring**: `q_fermion_h2`, `q_fermion_lih`, `q_fermion_expectation`, `q_trotter_evolve`, `q_trotter_error`, `q_zne_mitigate`, `q_zne_linear`, `q_scale_noise`, `mps_left_canonicalize`, `mps_right_canonicalize`, `mps_mixed_canonicalize`, `mps_swap`
- **52 new unit tests** across fermion, trotter, and mitigation modules
- **46 new integration tests** in `tests/beta_tests/hardening/` (fermion, trotter, mitigation, dual-mode parity)
- **2 new Bolero fuzz targets**: `fuzz_fermion_expectation_determinism`, `fuzz_zne_richardson_determinism`

### AST Evolution v0.3

- **AST Visitor trait** (`visit` module): read-only `AstVisitor` trait with `walk_*` functions covering all 35 `ExprKind`, 9 `StmtKind`, 11 `DeclKind`, and 9 `PatternKind` variants
- **AST Metrics** (`metrics` module): single-pass structural statistics via visitor — node counts, depth measurements, operator frequencies, feature-presence flags
- **AST Validation** (`validate` module): lightweight structural checks — break/continue outside loop, duplicate params/fields, empty match, unreachable code after return, nesting depth limit
- **AST Inspect** (`inspect` module): deterministic text dumps — `dump_ast_summary`, `dump_ast_metrics`, `dump_validation_report`, `dump_expr_tree`
- **Node utility methods** (`node_utils` module): `Expr::child_count`, `is_literal`, `is_place`, `is_compound`; `Block::is_empty`, `stmt_count`, `has_trailing_expr`; `Program::function_count`, `struct_count`, `has_main_function`
- **27 new unit tests** in cjc-ast across 5 modules
- **25 new integration tests** for visitor, metrics, validation, and inspect
- **2 Bolero fuzz targets**: `fuzz_ast_validator`, `fuzz_ast_metrics`

### MIR/CFG/SSA Evolution v0.3

- **SchedulePlan metadata** on loops: descriptive-only execution schedule hints (`SequentialStrict`, `DescriptiveTiled`, `DescriptiveVectorized`, `DescriptiveMaterializeBoundary`, `DescriptiveStaticPartition`) — non-semantic
- **AccumulatorSemantics enum** on reductions: classifies required accumulator type (`Plain`, `Kahan`, `Binned`, `RuntimeDefined`)
- **Enriched LoopInfo**: `is_countable`, `trip_count_hint`, `num_exits`, `schedule` fields
- **Enriched ReductionInfo**: `reassociation_forbidden`, `strict_order_required`, `accumulator_semantics` fields
- **SSA loop/reduction overlay** (`ssa_loop_overlay` module): maps SSA definitions to loops, identifies loop-carried variables, cross-references reductions with SSA accumulators
- **Inspect/diagnostics module** (`inspect` module): deterministic text dumps for loop trees, reduction reports, legality reports, and schedule summaries
- **Expanded legality verifier**: schedule metadata consistency checks, reduction metadata cross-validation
- **14 new integration tests** for schedule metadata, inspect diagnostics, and enriched metadata
- **9 new unit tests** across `ssa_loop_overlay` and `inspect` modules

### CLI Phase 3 — Compiler Visibility & Verification (10 new commands)

- `cjc emit` — dump AST/HIR/MIR intermediate representations with `--opt` and `--diff`
- `cjc explain` — show desugared HIR forms with function signatures and NoGC annotations
- `cjc gc` — GC analysis with allocation timeline and stability checks
- `cjc nogc` — static NoGC verification per function
- `cjc audit` — numerical hygiene analysis (naive summation, float equality, division-by-zero)
- `cjc precision` — f64 vs f32 precision analysis with relative error reporting
- `cjc lock` — deterministic lockfile generation and cross-platform verification
- `cjc parity` — dual-executor parity checker (eval vs mir-exec)
- `cjc test` — native test runner discovering `test_` functions and `@test` decorators
- `cjc ci` — full CI diagnostic suite (doctor + proof + parity + test + nogc)

### Enhanced CLI Commands

- `cjc mem` — added `--nogc`, `--mir`, `--eval` flags
- `cjc bench` — added `--nogc`, `--mir`, `--eval` flags
- `cjc proof` — added `--seeds` multi-seed mode
- `cjc doctor` — added `--strict` flag for CI pipelines
- `cjc view` — hash display now opt-in via `--hash`

### Design Decisions

- All Pauli terms stored in `Vec` (not HashMap) for deterministic iteration
- Complex arithmetic via `mul_fixed` (no FMA) throughout
- Kahan summation for all floating-point reductions
- MPS canonical form via sign-stabilized SVD preserves bit-identical determinism
- SchedulePlan is strictly non-semantic — the executor ignores it entirely
- All new AST/MIR modules are additive overlays — no existing code paths modified
- Phase 3 CLI commands are read-only analysis tools — they do not modify source files

## [0.1.0] — 2026-03-27

### Added

- **Core language:** functions with type annotations, closures, while/for loops, if/else (statement and expression), pattern matching, structs, enums, traits, generics
- **Type system** with inference, 25+ value types, and generic support
- **Dual execution:** AST tree-walk interpreter (`cjc-eval`) and MIR register-machine executor (`cjc-mir-exec`) with full parity
- **MIR optimization pipeline:** constant folding, strength reduction, DCE, CSE, LICM
- **SSA construction** (Cytron minimal) with 6 SSA optimization passes
- **Loop analysis**, reduction analysis, and legality verification infrastructure
- **282+ builtin functions** across math, string, array, tensor, statistics, and I/O
- **Deterministic execution:** SplitMix64 RNG, Kahan/Binned accumulators, BTreeMap everywhere — same seed = bit-identical output
- **Automatic differentiation:** forward-mode dual numbers and reverse-mode tape with gradient graphs
- **DataFrame library** (tidyverse-inspired): filter, select, mutate, group_by, summarize, join
- **Tensor operations** with SIMD acceleration (no FMA for bit-reproducibility)
- **Sparse linear algebra:** CSR format, SpMV, basic iterative solvers
- **NFA-based regex engine** (zero external dependencies)
- **Binary serialization** (`cjc-snap`)
- **Grammar-of-graphics visualization** (`cjc-vizor`)
- **NoGC verification** pass for allocation-free code paths
- **Escape analysis** for automatic memory tier selection (Stack/Arena/Rc)
- **CLI** with run, repl, check, lex, parse, ast, hir, mir, inspect, schema, bench, and diagnostic commands
- **5,600+ tests** with determinism verification

### Known Limitations

- Single-file programs only (module system is incomplete)
- Visibility modifiers (`pub`/`priv`) parsed but not enforced
- Parser recovery limited to top-level declarations
- Range type exists in type system but no `Value::Range` at runtime
- Some CLI subcommands are stubs (marked experimental)
