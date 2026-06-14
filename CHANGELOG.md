# Changelog

All notable changes to CJC-Lang (Computational Jacobian Core) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Fixes

#### Executor parity: calling a closure/function value bound to a local (`cjc-eval`)
- **Closed an AST-eval ↔ MIR-exec parity gap** (determinism invariant 7). Calling a closure held in a local — `let f = |x: i64| x + offset; f(i)` — raised `runtime error: undefined function \`f\`` under the AST tree-walk interpreter (`cjcl run`), while the MIR executor (`cjcl run --mir-opt`) evaluated it correctly. Both executors now produce byte-identical output.
- **Cause:** `cjc-eval::dispatch_call` resolved a callee name only against registered functions and builtins; it never consulted the scope chain for a variable holding a `Value::Closure` / `Value::Fn`. `cjc-mir-exec::dispatch_call` already did. The fix mirrors that branch: when a name is neither a user function nor a known builtin, look it up in scope and dispatch through it (prepending the captured env for closures). The `eval_call` catch-all (arbitrary-expression callees) likewise gained the `Value::Closure` arm for symmetry.
- **Higher-order forms** — passing a closure or a named function value as an argument and calling it inside the callee — are covered by the same path and verified at parity.
- **Tests:** 6 new dual-executor parity tests in `tests/test_builtin_parity.rs` (direct call in a loop, no-capture lambda, repeated invocation, closure-as-argument, named-`fn`-value-as-argument, multiple closures in scope). The parity gate now catches this form.

#### Executor parity: lexical capture for escaping closures (`cjc-eval`)
- **Closed the deeper half of the same parity gap.** A closure that *escapes* its defining scope (e.g. returned from a factory) diverged between executors: cjc-eval lowered every lambda to a plain `Value::Fn` and resolved free variables by **dynamic scoping** (whatever was live at the *call* site), while cjc-mir-exec does proper **lexical capture** (the values live at the *creation* site). For `make_adder(5)` returning `|x| x + n`, eval produced `1003`/`undefined variable n` where MIR produced the correct `8`.
- **Fix:** cjc-eval's lambda evaluation now performs closure conversion, mirroring cjc-hir's analysis. It lifts the lambda to a function whose parameters are `[captured…, explicit…]` and snapshots the captured variables' current values into a `Value::Closure { fn_name, env, arity }`. A free name is captured iff it resolves to a live local binding at creation time — so function names, builtins, enum variants, and names `let`-bound inside the body are *not* captured (resolved by name at call time), matching MIR-exec's capture set. Lambdas with no captures still lower to a plain `Value::Fn` (unchanged). The AST free-variable collectors are ported into cjc-eval so the v1 interpreter takes no dependency on the HIR/MIR pipeline.
- **Now-live code:** eval's `Value::Closure` arms in `array_map` / `array_filter` / `array_reduce` / `array_any` / `array_all` (previously unreachable, since eval never produced a `Closure`) now correctly prepend the captured env to per-element arguments.
- **Tests:** 7 further parity tests (escaping factory, post-creation mutation is a snapshot, multiple instances with distinct captures, multi-variable capture, factory closure called in a loop, capturing closure through `array_map`/`array_reduce`). Regression: 460+ tests green across eval, parity, closures, decorators, variadic, vizor, physics_ml, and chess_rl_v2; 0 failures.

## [0.1.10] — 2026-05-23

### Green Compute — Runtime Policy Layer, Adaptive Scheduling & Fused Kernels

A green-compute layer that bounds thermal/CPU load deterministically. The heat/speed knobs **provably never change output** — same seed, same bits, across every profile, thread count, and both executors.

#### Runtime Policy Layer (`cjc-runtime`)
- **`RuntimePolicy`** — thermal profiles via `--profile cool | balanced | max-perf`, a thread cap via `--threads N`, an advisory batch size, audit depth, and a numeric-mode hint. The default is `balanced` (≈ ½ cores), deliberately not "all cores forever."
- **Deterministic energy estimate** — `energy_estimate(flops, bytes)` is a pure function of the workload, never of wall-clock time: same seed → same joules, bit-for-bit. Plus `energy_per_flop` / `energy_per_byte`.
- Policy query/set builtins (`runtime_policy_*`), routed through the shared dispatch so both executors agree.

#### Race-to-Idle Adaptive Scheduling
- Parallel kernels run at full width during short bursts and throttle to the cap only once load is **sustained** — recovering most of the throughput a fixed cap costs while keeping the thermal bound. `--no-adaptive` applies the cap uniformly for reproducible benchmarking.
- **Determinism preserved:** only the *schedule* adapts; thread count, pool size, and chunk count never change the numeric result.

#### Fused Elementwise Kernels (GC-06 Phase 3a)
- `fused_axpy(α, x, y)`, `fused_mul_sub(a, b, c)`, `fused_sub_sq(a, b)` — single-pass kernels that eliminate intermediate tensor allocations (~40% less memory traffic, one allocation instead of two). Bit-identical to the unfused sequence (software two-rounding, **no hardware FMA**).

#### CLI
- New flags applied once at startup across all commands: `--profile`, `--threads`, `--batch-size`, `--audit-mode`, `--no-adaptive`.

#### Fixes
- Renamed the runtime-policy `--audit` flag to `--audit-mode` to avoid colliding with the `abng inspect --audit` subcommand flag.

#### Tests
- 80 new tests (unit, dual-executor parity, property, and fuzz) covering the policy/scheduling layer and the fused kernels.

## [0.1.5] — 2026-04-14

### Physics-Informed Neural Networks (PINNs) & Scientific Machine Learning

CJC-Lang can now solve partial differential equations using neural networks — entirely from scratch, zero external dependencies.

#### PINN Infrastructure (`cjc-ad`)
- **Full MLP architecture** — `DenseLayer`, `Mlp` struct with Xavier-uniform initialization, deterministic forward pass on `GradGraph`
- **8 activation functions** — `Tanh`, `Sigmoid`, `ReLU`, `GELU`, `SiLU`, `ELU`, `SELU`, `SinAct` (sinusoidal activation for periodic PDEs)
- **Physics loss components** — `data_loss_mse()`, `physics_residual_harmonic()`, central finite differences for spatial derivatives
- **`PinnConfig` + `PinnResult` + `TrainLog`** — full training configuration, convergence history, and error metrics

#### 12 PDE Problem Solvers
Each solver trains a neural network to satisfy both boundary conditions and the governing PDE:

| Problem | Type | Domain |
|---------|------|--------|
| Harmonic Oscillator | 1D ODE | Time domain |
| Heat Equation | 1D Parabolic | Spatial |
| Burgers Equation | 1D Hyperbolic | Spatial-temporal |
| Poisson Equation | 2D Elliptic | Spatial |
| Wave Equation | 2D Hyperbolic | Spatial-temporal |
| Helmholtz Equation | 2D Helmholtz | Spatial |
| Diffusion-Reaction | 2D Coupled | Spatial |
| Allen-Cahn | Reaction-diffusion | Phase field |
| Korteweg-de Vries | 1D Dispersive | Soliton dynamics |
| Schrodinger (NLS) | Quantum wave | Complex field |
| Navier-Stokes 2D | Fluid dynamics | Velocity-pressure |
| Burgers 2D | 2D Extension | Multi-dimensional |

#### Optimization Methods
- **Adam optimizer** — β1=0.9, β2=0.999, element-wise with native `adam_step` builtin
- **L-BFGS** — Two-loop recursion for second-order convergence
- **TwoStageOptimizer** — Adam (80% of epochs) → L-BFGS (final 20%) for fast convergence + fine-tuning

#### Graph Reuse for Multi-Epoch Training
- **`reforward()` method** — re-evaluate computation graph with updated inputs without rebuilding
- **`set_tensor()` + `reforward()`** — PINN training builds graph once, updates collocation points per epoch
- Eliminates O(epochs × graph_size) allocations

#### Tests
- **`pinn_correctness.rs`** — L2 error bounds, residual convergence, gradient flow
- **`pinn_expansion_tests.rs`** — 537 tests across all 12 PDE solvers
- **`pinn_parity.rs`** — 188 tests verifying eval/mir-exec produce identical PINN results
- **`pinn_pde_problems.rs`** — 621 tests for PDE-specific validation

### Autodiff Engine Rewrite (v2.4–v2.5)

Major performance overhaul of the reverse-mode automatic differentiation engine:

#### Arena-Based GradGraph (v2.4)
- **Replaced `Vec<Rc<RefCell<GradNode>>>` with flat arrays** — `ops: Vec<GradOp>`, `tensors: Vec<Tensor>`, `param_grads: Vec<Option<Tensor>>`
- **Dead node elimination** — reachability analysis before backward pass; unreachable nodes skipped (20-30% for multi-head networks)
- **Result:** 1.89× backward pass speedup (47% time reduction)

#### In-Place Gradient Accumulation (v2.5)
- **`Tensor::add_assign_unchecked`** — mutates gradient tensors in-place, eliminating ~N/2 tensor allocations per backward pass
- **`backward_collect()`** — batches zero_grad + backward + gradient collection into one call
- **Fused MLP layer** — `GradOp::MlpLayer` collapses transpose + matmul + bias-add + activation into one graph node (3× fewer nodes per layer)
- **Result:** 2.76× total speedup from v2.3 baseline

### Chess RL v2.1–v2.5

In-place upgrade of the chess reinforcement learning benchmark with advanced training infrastructure:

- **Adam optimizer** — native `adam_step` builtin (9× speedup over CJC-Lang scalar loop)
- **A2C with GAE** — Advantage Actor-Critic with Generalized Advantage Estimation
- **Advantage/return whitening, temperature annealing, resignation threshold**
- **31-tensor checkpoint bundle** + CSV training log + Elo-lite rating system
- **PGN export** (long-algebraic notation) + Vizor training curve SVGs
- **Native hot-path kernels** — `encode_state_fast`, `score_moves_batch` (7.7× rollout speedup)
- **Profiling infrastructure** — `profile_zone_start`/`profile_zone_stop`/`profile_dump` builtins
- **10 new builtins:** `adam_step`, `file_append`, `profile_zone_start`, `profile_zone_stop`, `profile_dump`, `encode_state_fast`, `score_moves_batch`, `categorical_sample`, `log`, `exp`

### virtual-frame / TidyView Engine (companion crate)

The TidyView columnar data engine (`virtual-frame` on crates.io) was expanded from 7 modules to 23:

- **Adaptive Dictionary Encoding** — automatic encoding decisions based on cardinality thresholds (4.3× filter speedup, 7.2× memory reduction for low-cardinality columns)
- **Vectorized Expression Kernels** — word-level (64-row) predicate evaluation with AND/OR/NOT fast paths; expression-to-kernel gap reduced from 4.7× to 1.71×
- **Deterministic Join Planner** — automatic strategy selection (SortMerge / BTreeMapHash / NestedLoop) based on sortedness and column types
- **11 new subsystems:** Query Plan Optimizer, Columnar Compression (RLE/Delta/BitPack), Zone Maps, Null-Aware Engine, Parallel Deterministic Execution, .tvf Storage Format, Streaming/Out-of-Core, Advanced Joins (sort-merge/semi/anti/cross), Row Lineage/Provenance

---

## [0.1.4] — 2026-04-06

### Rebrand: CJC → CJC-Lang
- **Project renamed from CJC to CJC-Lang** — the acronym still stands for Computational Jacobian Core
- **CLI command changed from `cjc` to `cjcl`** — all 30 commands now use `cjcl` prefix (e.g., `cjcl run`, `cjcl flow`)
- **File extension changed from `.cjc` to `.cjcl`** — source files, test fixtures, and examples all updated
- **Install command: `cargo install cjc-lang`** — installs the `cjcl` binary
- **Package name on crates.io: `cjc-lang`** (previously `cjc`)
- All documentation, blog posts, and help text updated to reflect the new name

## [0.1.3] — 2026-04-05

### Install Fix
- **`cargo install cjc` now works** — root package now includes binary entry point (v0.1.2 was library-only on crates.io)
- `cjc-cli` refactored to mixed lib+bin crate, exposing `cli_main()` for the root binary

## [0.1.2] — 2026-04-05

### Data Science Foundation, Statistical Completeness & Hardening Phase

#### Hardening & Safety Fixes
- **Eliminated 28 `.unwrap()` calls on GradGraph downcasts** — 14 in `cjc-eval`, 14 in `cjc-mir-exec`; all converted to proper `ok_or_else` error propagation with descriptive `EvalError::Runtime` / `MirExecError` messages
- **Replaced `HashSet` with `BTreeSet`** in `cjc-data/tidy_dispatch.rs` — eliminates last non-deterministic iteration order in the codebase

#### ML Infrastructure (3 new primitives)
- **`embedding(weights, indices)` builtin** — lookup rows from a weight matrix by index; wired in all three places (runtime, eval, mir-exec)
- **`avgpool2d(kh, kw, sh, sw)` tensor method** — 2D average pooling with configurable kernel and stride; complementary to existing `maxpool2d`
- **`batch_indices(dataset_size, batch_size, seed)` builtin** — deterministic mini-batch index generation using SplitMix64; returns array of `[start, end]` pairs

#### New Test Suites (5 files)
- **`test_mir_exec_coverage.rs`** — 20+ parity tests covering MIR executor edge cases
- **`test_parity_stress.rs`** — 50-seed stress tests across program templates
- **`test_ml_infrastructure.rs`** — embedding, avgpool2d, and batch_indices tests with parity verification
- **`test_type_system_props.rs`** — property-based tests (proptest) for type system invariants
- **`test_mir_fuzz.rs`** — fuzz testing for MIR pipeline robustness

#### Language Ergonomic Gaps (6 features)
- **`tanh(x)` standalone builtin** — unified scalar (Float/Int) and Tensor support; `tanh_scalar` retained as alias
- **`relu(x)` standalone builtin** — scalar (Float returns max(0,x), Int returns max(0,x)) and Tensor support
- **`reshape(tensor, shape)` builtin** — reshape tensors to new dimensions (uses existing `Tensor::reshape`)
- **`as` type casting** — `x as f64`, `x as i64`, `x as bool`, `x as String`; desugars to builtin calls in HIR; high precedence (like Rust); supports chained casts (`x as i64 as f64`)
- **Tuple field access** — `t.0`, `t.1`, etc. for direct tuple element access without `match` destructuring; works in both eval and mir-exec
- **`tensor_slice(t, starts, ends)` and `slice(t, dim, start, end)` builtins** — zero-copy tensor slicing along all dims or a single dim
- **`int()` / `float()` accept Bool** — `int(true) → 1`, `float(false) → 0.0` (enables `as` casting from bool)
- **40 new integration tests** in `test_feature_gaps.rs` — all with eval/mir-exec parity verification and determinism checks

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
