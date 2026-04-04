# Changelog

All notable changes to CJC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.2] — 2026-04-04

### Data Science Foundation & Statistical Completeness Phase

#### NA Type (Missing Values)
- **`NA` literal** across full compiler pipeline (lexer, parser, AST, HIR, MIR, eval, mir-exec, snap, types)
- **SQL-style NA semantics:** `NA == NA → false`, `NA != NA → true`, arithmetic propagation (`NA + x → NA`)
- **Logical propagation:** `NA && true → NA`, `NA || false → NA`
- **Builtins:** `is_na(val)`, `drop_na(arr)` for NA detection and removal
- **`coalesce()` upgraded** to support both scalar mode (`coalesce(NA, 42) → 42`) and array mode
- **Snap serialization:** `TAG_NA = 0x18` for binary round-trip, JSON encodes as `null`

#### Array Higher-Order Functions
- **11 new HOFs** implemented in BOTH executors (eval + mir-exec) for full parity:
  - `range(end)`, `range(start, end)`, `range(start, end, step)` — integer range generation
  - `array_map(arr, fn)` — transform each element
  - `array_filter(arr, fn)` — keep elements matching predicate
  - `array_reduce(arr, init, fn)` — fold with accumulator
  - `array_any(arr, fn)` / `array_all(arr, fn)` — short-circuit predicates
  - `array_find(arr, fn)` — first match or `NA`
  - `array_enumerate(arr)` — index-value tuples
  - `array_zip(arr_a, arr_b)` — pairwise tuples
  - `array_sort_by(arr, key_fn)` — deterministic key-based sort
  - `array_unique(arr)` — deduplicate preserving order
- All HOFs support both named functions and closures with captured environments

#### Function-as-Value
- **Named function references:** `fn double(x: i64) -> i64 { ... }` can be passed as `array_map(arr, double)`
- Eval executor now resolves function names to `Value::Fn` when used as expressions (mir-exec already supported this)

#### Categorical / Factor Data
- **6 new builtins** for categorical data manipulation:
  - `as_factor(string_arr)` — encode string array to Factor struct (levels + codes)
  - `factor_levels(f)` / `factor_codes(f)` — extract components
  - `fct_relevel(f, new_order)` — reorder factor levels
  - `fct_lump(f, n)` — lump rare levels into "Other"
  - `fct_count(f)` — count observations per level
- Factor represented as `Struct { name: "Factor", levels: [...], codes: [...] }`
- Deterministic: level order preserved from first appearance, BTreeMap for frequency counting

#### DataFrame Inspection Methods
- **7 new TidyView methods** for data exploration:
  - `.head(n)` / `.tail(n)` — formatted first/last N rows (default 10)
  - `.shape()` — returns `(nrows, ncols)` tuple
  - `.columns()` — alias for column_names, returns string array
  - `.dtypes()` — returns Struct mapping column names to type strings
  - `.describe()` — statistical summary (count, mean, std, min, 25%, 50%, 75%, max for numeric; count, unique, top for string/categorical)
  - `.glimpse()` — transposed column view with types and first 8 values
- `describe()` uses Kahan summation for numerically stable mean/std computation

#### Tests
- **65 new integration tests** across 3 test files:
  - `test_na_type.rs` — 43 tests: NA semantics, array HOFs, categorical builtins, parity + determinism
  - `test_dataframe_inspect.rs` — 12 tests: shape, columns, dtypes, head, tail, describe, glimpse, determinism
  - `test_e2e_pipelines.rs` — 10 tests: realistic data science workflows (survey cleaning, feature engineering, category encoding, NA-aware aggregation, paired data, sorting, functional composition, mixed types, data validation, determinism stress)
- All tests verify eval/mir-exec **parity** (identical output from both executors)
- All tests verify **determinism** (repeated runs produce bit-identical output)
- **All 6,500+ existing tests continue to pass with zero regressions** (4,780 in `cargo test` summary lines)

#### CLI Enhancements
- **`cjc eval "expr"` command** — evaluate single expressions from the command line
- **`--format` flag** — uniform `--format plain|json|csv` across `cjc run` and `cjc eval`
- **Differentiated exit codes** — 0=success, 1=runtime, 2=parse, 3=type/check, 4=parity

#### Statistical Tests (7 new)
- **Normality tests:** `jarque_bera(data)`, `anderson_darling(data)`, `ks_test(data)` — test if data follows normal distribution
- **Variance tests:** `levene_test(groups...)`, `bartlett_test(groups...)` — test equality of variances
- **Effect sizes:** `cohens_d(x, y)`, `eta_squared(groups...)`, `cramers_v(table, r, c)` — measure effect magnitude
- All tests use Kahan summation for numerical stability
- Full chi-squared survival function with regularized incomplete gamma (Lanczos approximation)

#### Sampling & Cross-Validation (7 builtins)
- **`latin_hypercube(n, dims, seed)`** — space-filling quasi-random sampling (already existed, now exposed)
- **`sobol_sequence(n, dims)`** — low-discrepancy sequence for numerical integration
- **`train_test_split(n, test_frac, seed)`** — deterministic train/test splitting
- **`kfold_indices(n, k, seed)`** — k-fold cross-validation index generation
- **`bootstrap(data, n_resamples, stat_fn, seed)`** — bootstrap CI with point estimate, 95% CI, and SE
- **`permutation_test(x, y, n_perms, seed)`** — non-parametric permutation test for group differences
- **`stratified_split(labels, test_frac, seed)`** — stratified train/test split preserving class proportions
- All samplers use deterministic seeded shuffling (Fisher-Yates)

#### REPL Enhancements
- **`:vars`** — alias for `:env`, show current variable bindings
- **`:time <expr>`** — time an expression and display elapsed duration
- **`:describe <expr>`** — statistical summary (count, mean, std, min, 25%, 50%, 75%, max) for numeric arrays
- **`:save <file>`** — save REPL session to file
- **`:load <file>`** — load and execute a CJC source file in the REPL

#### Additional Tests
- **26 new integration tests** in `test_phase3_stats_sampling.rs`:
  - Normality tests with known normal and skewed data
  - Effect size validation (large/small/no effect)
  - Variance test correctness
  - Sampling shape and determinism verification
  - Bootstrap CI and permutation test validation
  - Stratified split class proportion preservation
  - All tests verify eval/mir-exec parity

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
