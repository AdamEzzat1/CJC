# Handoff — Next Session (v4)

**Date:** 2026-06-09
**Branch:** `claude/peaceful-moore-92fb41`
**Worktree:** `C:\Users\adame\CJC\.claude\worktrees\peaceful-moore-92fb41`
**Supersedes:** `HANDOFF_NEXT_SESSION_v3.md`

This v4 doc lands after a sustained session that worked through every
item v3 flagged as "tractable in one session" plus several
adjacent-language items. The next session is **adding something new
to the compiler**, so this doc emphasizes architecture pointers (the
contracts and patterns a new feature has to fit into) over the
deferred-item list (which is documented but not load-bearing for
new-feature work).

---

## 0. TL;DR — what shipped this session

**8 commits** landing real implementations + scaffolds + 4 design docs.
Every commit preserved AST/MIR byte-identity. ~3,800 LOC added across
the workspace. No regressions in any pre-existing suite.

```
c4ba20e  Lanczos eigenvector recovery + tensor_elementwise SIMD dispatch
966d908  §4.4 sparse_lanczos/sparse_arnoldi dispatch + §4.5 decorators design
7a01d1f  §3.1 Option B PR 1 (TraceCollector) + §3.2 impl designs
7f2b1b5  §3.2 scaffolds (vectorize/specialize/monomorphize) + §4.1 audit + §3.1 design
a36aea8  §3.3 re-lock 6 audit-chain canaries (ABNG 0.9.5 drift)
d26f4f8  §4.3 reserve ode_step / pde_step / symbolic_derivative
3d333d8  §3.2 loop_unroll MVP + Universal tier + v6 trained coefficients
```

---

## 1. The contracts every new feature must honour

If you're adding ANYTHING to the compiler, the following invariants
have to survive your change. They are encoded as CI gates — breaking
any one of them is a blocking failure.

### 1.1 AST/MIR byte-identity (the hardest invariant)

Run: `cargo test --test fixtures --release`

This gate runs every program in `tests/fixtures/` through BOTH:
- `cjcl run` (AST tree-walk interpreter — `cjc_eval`)
- `cjcl run --mir-opt` (MIR register-machine executor — `cjc_mir_exec` with optimizer enabled)

…and asserts the stdout is **byte-identical**. Every feature in CJC-Lang
must work the same in both executors. A passing parity gate is the
single strongest signal your change is correct.

If you add:
- **A new builtin** → both executors auto-route to `cjc_runtime::dispatch_builtin` via fallback, so wiring in one place is enough.
- **A new MIR optimizer pass** → must produce semantically-equivalent MIR (the parity test will catch divergence).
- **A new language feature** (syntax, expr kind, stmt kind) → BOTH executors need handling. See `crates/cjc-eval/src/lib.rs` and `crates/cjc-mir-exec/src/lib.rs`.

### 1.2 Determinism contract

Same seed → bit-identical output across runs, OS, and CPU. This means:
- All FP reductions use `cjc_repro::kahan_sum_f64` or `BinnedAccumulator`. NEVER naive `iter().sum::<f64>()`.
- RNG is `SplitMix64` with explicit seed threading. NEVER `rand::thread_rng()` or `Instant::now()` as entropy.
- Maps are `BTreeMap`/`BTreeSet` everywhere. NEVER `HashMap`/`HashSet` (random hash seed).
- SIMD kernels do NOT use FMA (would change bit patterns vs scalar).

The `cjc_repro` crate is the canonical home for determinism helpers.

### 1.3 Three-place wiring pattern (for builtins)

Per `CLAUDE.md` — every new builtin function needs registration in **three** places, but the executor wiring is usually automatic:

1. `crates/cjc-runtime/src/builtins.rs` — add a `match` arm in `dispatch_builtin` (REQUIRED).
2. `crates/cjc-eval/src/lib.rs` — the existing `Try shared (stateless) builtins` fallback at line ~2449 auto-routes (NO CHANGES needed for pure stateless builtins).
3. `crates/cjc-mir-exec/src/lib.rs` — same fallback at line ~2205 (NO CHANGES needed for pure stateless builtins).

Only if the builtin needs executor-specific state (e.g. accesses the call stack, profiling data, TLS arenas) do you need to add per-executor handling. The `grad_graph_*` and `quantum_*` families are precedent for this — they live in `cjc-ad::dispatch_grad_graph` and `cjc-quantum::dispatch_quantum` respectively, called from both executors at the same routing point as the runtime dispatch.

### 1.4 NoGC verification

If your feature touches code in NoGC-marked functions (`@nogc`), it cannot introduce hidden allocations. Run `cargo test -p cjc-mir nogc_verify` to verify. The verifier is conservative — refusing to compile is the right answer if your feature COULD allocate even if it usually doesn't.

### 1.5 Backward compatibility

The existing test corpus is large (~5,300 tests). Every change must not regress any test. The session's discipline was: run the relevant subset locally before committing, then trust the per-commit parity gate.

---

## 2. Compiler architecture — where to put a new feature

### 2.1 The pipeline

```
.cjcl source
    ↓  cjc-lexer (tokenize)
    ↓  cjc-parser (Pratt parser → AST)
    ↓  cjc-types (optional type-check)
    ↓  cjc-hir (AST → HIR lowering)
    ↓  cjc-mir (HIR → MIR lowering + optimize)
    ↓
    ├─→ cjc-eval (tree-walk, v1) — runs the AST directly
    └─→ cjc-mir-exec (register-machine, v2) — runs the MIR
              ↑
              └─ optional: optimize via cjc-cana (PassRanker)
                 + thermal-aware cost model (cjc-cana-nss)
```

### 2.2 Where to add new features

| You want to add… | Touch these files | Don't forget |
|---|---|---|
| A new builtin function | `cjc-runtime/src/builtins.rs` (one `match` arm) | Validate arity + types; add unit tests + a parity-relevant fixture |
| A new MIR optimizer pass | `cjc-mir/src/optimize.rs` (add to `DEFAULT_PASS_SEQUENCE` + `apply_pass_with_diagnostics` dispatch); `cjc-cana/src/pass_ranker.rs::CANONICAL_PASSES`; `cjc-cana/src/legality.rs::pass_safety_tier`; `cjc-cana/src/linear_cost_model.rs` (both default and trained tables) | Drift-guard test (`canonical_pass_count_is_N`); `bench/cana_train_cost_model/main.rs::TARGET_PASSES` if you want trained coefficients |
| A new expression kind | `cjc-ast::ExprKind`, `cjc-hir::HirExprKind`, `cjc-mir::MirExprKind`, lowering paths in `cjc-hir/src/lib.rs` and `cjc-mir/src/lib.rs`, eval handlers in `cjc-eval` and `cjc-mir-exec` | New tests in `tests/` AND a fixture for the parity gate |
| A new statement kind | Same as expression kind, but `StmtKind` | Same |
| A new type | `cjc-types`, type inference paths, MIR representation | Test type errors are good (E-codes) |
| A new value variant | `cjc-runtime/src/value.rs::Value` — **THIS IS A HARD RULE per CLAUDE.md.** Adding a variant touches every `match Value::*` in the codebase. ~50+ files. Don't take lightly. | All pattern matches; serialization (`cjc_snap`); dispatch tables; `type_name()`; tests for each touch point |
| A new attribute / decorator | Use the existing `@name` decorator infrastructure. Parser already handles `@name` and `@name(args)`. AST/HIR/MIR fields exist. Runtime handlers are in eval + mir-exec call dispatch. See `docs/DECORATORS_DESIGN.md` for the user-defined-decorator gap. | If your decorator name is unknown to the runtime, dispatch silently ignores it today — adding user-defined decorator semantics is a separate work item |
| A new compiler diagnostic | `cjc-diag/src/lib.rs::ErrorCode` (assign next E-code in the right family); render in `cjc-diag/src/render.rs` | Test the error fires on the trigger case |
| A new file format / serializer | `cjc-snap` (binary), or new module — KEEP DETERMINISTIC | Hash-canary test |

### 2.3 The CANA layer (only relevant if your feature interacts with the optimizer)

The compiler has a sophisticated optimization-decision layer in `cjc-cana`:

- `cjc-cana/src/features.rs` — featurizer: extracts per-function `expr_count`, `loop_depth`, `branch_count`, `alloc_sites`, `reductions`, etc. from MIR.
- `cjc-cana/src/cost_model.rs` — `CostModel` trait: predicts per-pass benefit and compile-time cost.
- `cjc-cana/src/linear_cost_model.rs` — hand-tuned + trained coefficient implementations.
- `cjc-cana/src/legality.rs` — `LegalityGate`: refuses dangerous pass recommendations (e.g. CSE on a function with strict reductions). Per-pass safety tiers: `Universal` vs `NoStrictReductions`.
- `cjc-cana/src/pass_ranker.rs` — orchestrates featurizer + cost model + legality gate → produces a per-function `PassPlan`.
- `cjc-cana/src/thermal_cost_model.rs` — wraps a base cost model to penalize "thermally aggressive" passes on hot functions.
- `cjc-cana-nss/src/lib.rs` — `NssPressurePredictor` that maps MIR features to per-function thermal pressure (current Option C synthesizes; Option B real instrumentation is in flight — see §3.1 of prior handoff).

If your new feature is a MIR pass:
1. Add the pass to `CANONICAL_PASSES` in `cjc-cana::pass_ranker`.
2. Classify it in `cjc-cana::legality::pass_safety_tier`. Default to `NoStrictReductions` unless you can prove it preserves strict-reduction bit-exact ordering.
3. Add default coefficients in `cjc-cana::linear_cost_model::default_pass_coefficients`.
4. Add placeholder trained coefficients (or run `cargo run --release -p cana-train-cost-model` after extending the corpus).
5. Decide if it goes in `THERMALLY_AGGRESSIVE_PASSES` (in `thermal_cost_model.rs`).

The scaffold pattern is well-established now — `vectorize`, `specialize`, `monomorphize` all shipped as no-op scaffolds in commit `7f2b1b5`. Follow that template.

---

## 3. Test state at session end

| Suite | Count | Notes |
|---|---|---|
| cjc-mir lib | 187/187 | +13 from this session |
| cjc-cana lib | 144/144 | +1 §6.4 unknown-pass-blocked test (kept from prior session) |
| cjc-cana-nss | 14/14 | unchanged |
| cjc-runtime lib | ~785/785 | +24 across the session (sparse_eigen +6, builtins +18) |
| cjc-mir-exec trace | 11/11 | new module this session |
| cjc-runtime sparse_eigen | 11/11 | +6 eigenvector recovery tests |
| cjc-runtime builtins | 60/60 | +18 dispatch tests |
| tests/test_decorators | 12/12 | existing suite, no regressions |
| tests/test_mir_exec_instrumented | 5/5 | new this session |
| AST/MIR parity (fixtures) | 1/1 | byte-identity preserved across all 8 commits |
| 6 re-locked ABNG canaries | 6/6 | hash drift documented + locked |

15 pre-existing failures broken down:
- 6 ABNG audit-chain canaries — RE-LOCKED with ABNG 0.9.5 R0/R1 drift explanation (commit `a36aea8`).
- 4 PINN-parity tests (#12-14, #15 from v3 doc) — NOT failing anymore. 3 self-skip with "missing artifact" (the file was never committed in this worktree); #15 was already fixed somewhere.
- 5 non-canary serializer/replay tests — STILL FAILING. Not re-locked because they're not canary-shaped; they need real serializer root-cause analysis. Documented in `docs/cana/CANA_PHASE_1_REGRESSION_FAILURES.md`.

---

## 4. Files added this session

```
crates/cjc-mir-exec/src/trace.rs               — NEW (TraceCollector for Option B PR 1)
crates/cjc-mir-exec/src/lib.rs                 — added mod trace + run_program_instrumented
crates/cjc-mir-exec/Cargo.toml                 — added cjc-nss dep

crates/cjc-mir/src/optimize.rs                 — loop_unroll + 3 scaffolds + tests
crates/cjc-cana/src/legality.rs                — loop_unroll Universal + 3 scaffold classifications
crates/cjc-cana/src/pass_ranker.rs             — loop_unroll in CANONICAL_PASSES
crates/cjc-cana/src/linear_cost_model.rs       — loop_unroll coefficients (default + trained)
bench/cana_train_cost_model/main.rs            — loop_unroll added to TARGET_PASSES
bench/cana_train_cost_model/programs.rs        — corpus expansion for loop_unroll

crates/cjc-runtime/src/builtins.rs             — 3 solver stubs + sparse_lanczos / sparse_arnoldi
                                                  + sparse_lanczos_with_vectors
                                                  + 8 tensor_elementwise_* dispatch arms
                                                  + ~33 new tests
crates/cjc-runtime/src/sparse_eigen.rs         — lanczos_eigsh_with_vectors
                                                  + tridiagonal_qr_with_vectors
                                                  + 6 new tests

tests/test_mir_exec_instrumented.rs            — NEW (5 parity tests for run_program_instrumented)

tests/test_abng_lineage_attestation.rs         — canary re-lock
tests/test_abng_pinn_uncertainty.rs            — canary re-lock
tests/test_abng_tabular_gp.rs                  — canary re-lock
tests/test_abng_lineage_attestation_cjcl.rs    — canary re-lock
tests/test_abng_pinn_uncertainty_cjcl.rs       — canary re-lock
tests/test_abng_tabular_gp_cjcl.rs             — canary re-lock

docs/cana/CANA_PHASE_1_REGRESSION_FAILURES.md  — 2026-06-09 re-check section
docs/AD_MIR_COVERAGE_AUDIT.md                  — NEW (autodiff MIR coverage report)
docs/cana/OPTION_B_DESIGN.md                   — NEW (Option B 5-PR sequence design)
docs/cana/PASS_IMPLEMENTATION_DESIGNS.md       — NEW (vectorize/specialize/monomorphize designs)
docs/DECORATORS_DESIGN.md                      — NEW (user-defined decorator design)
docs/cana/HANDOFF_NEXT_SESSION_v4.md           — NEW (this doc)
```

---

## 5. Quick reference — verification commands

```bash
cd /c/Users/adame/CJC/.claude/worktrees/peaceful-moore-92fb41

# Most critical: AST/MIR byte-identity (the load-bearing contract)
cargo test --test fixtures --release

# Per-crate sweeps
cargo test -p cjc-runtime --release --lib
cargo test -p cjc-mir --release --lib
cargo test -p cjc-mir-exec --release --lib
cargo test -p cjc-cana --release --lib
cargo test -p cjc-cana-nss --release

# Specific suites added this session
cargo test --test test_mir_exec_instrumented --release
cargo test --test test_decorators --release
cargo test --release --test test_abng_lineage_attestation chain_head_canary_locked

# The new builtins (sparse eigen, tensor SIMD)
cargo test -p cjc-runtime --release --lib sparse_eigen::
cargo test -p cjc-runtime --release --lib builtins::tests::

# The CANA benches (verify thermal-aware routing still works on PINN)
cargo run --release -p cana-ab-pinn     # ~5 min @ N=10
cargo run --release -p cana-pinn-thermal-probe  # ~10 sec

# Cost-model training regeneration (only if you change TARGET_PASSES
# or the corpus in bench/cana_train_cost_model/programs.rs)
cargo run --release -p cana-train-cost-model   # ~2-5 min
```

---

## 6. Deferred items (for visibility, not for action)

The next session is on a new feature. These are open but should not
drive the next session's planning — they're listed so you know what's
still on the radar.

**Sparse / numerical:**
- Lanczos restart for tight memory budgets (~150 LOC, 1 day)
- General eigenvalue problem `A x = λ B x` (~200 LOC, 1-2 days)
- Real ODE/PDE/symbolic_derivative implementations behind the stubs (closure-as-arg plumbing is the blocker)

**CANA + Optimizer:**
- Option B PRs 2-5: actual instrumentation sites + heap accounting + NSS consumer (1.5-2 weeks, design doc in `docs/cana/OPTION_B_DESIGN.md`)
- monomorphize real implementation (3-5 days, design in `docs/cana/PASS_IMPLEMENTATION_DESIGNS.md` §2)
- specialize real implementation (4-6 days, ibid §3)
- vectorize MIR loop-rewrite pass (1 week for slice 1's MIR half; the SIMD builtin half shipped in `c4ba20e`)

**Language features:**
- User-defined decorators — strategy A in `docs/DECORATORS_DESIGN.md` (~500 LOC, 3-4 days)

**Test debt:**
- 5 non-canary serializer/replay regression failures need real root-cause analysis (multi-hour per failure, ~documented in `docs/cana/CANA_PHASE_1_REGRESSION_FAILURES.md`)

---

## 7. Notes for adding a new feature

These are the patterns that paid off repeatedly this session — they're
also good defaults for new-feature work.

### 7.1 Survey before implementing

The handoff said "decorators is the largest single item" but a code
survey showed lexer/parser/AST/HIR/MIR + 35 tests + working
@memoize/@trace were already shipped. Same with sparse eigen — the
algorithms were already there with Kahan summation + a determinism
test, just not exposed via `dispatch_builtin`. Always
`grep -rn "<feature_name>" crates/` before assuming greenfield.

### 7.2 Ship the parity gate test FIRST

If your feature is observable in program output, write a parity test
(`tests/fixtures/your_feature.rs`) BEFORE writing the implementation.
The parity test is the strongest signal of correctness; if it passes,
your two executors agree, and most semantic bugs reveal themselves.

### 7.3 Conservative legality classifications

For new MIR passes: default to `NoStrictReductions` in
`pass_safety_tier`, with a comment explaining what would need to be
true for promotion to `Universal`. This session shipped 3 new passes
that way (`vectorize`, `specialize`, `monomorphize`). The promotion
becomes a small focused commit later with a structural argument.

### 7.4 Document gaps in commit messages, not just docs

If your commit explicitly defers a piece of work, name it in the
commit message under "What this commit explicitly does NOT do". Future
agents grep `git log` looking for "what's next on X" and the commit
message is where they find it. The session's commits all follow this
pattern — see `c4ba20e` for the most explicit example.

### 7.5 Treat the handoff as advice, not truth

This document, like its predecessors, will decay. Numbers go stale,
"failing" tests get fixed silently, "deferred" items get partially
landed by other agents. When you re-verify, the empirical state on
disk wins.

---

## 8. About the new feature you're adding

This handoff doesn't know what feature you're adding. But based on the
architecture patterns this session reinforced, here are the questions
to answer before you start writing code:

1. **Does it touch the type system?** If yes, expect to update
   `cjc-types` + the type-check gate. ~250-500 LOC for non-trivial
   changes.

2. **Is it a new dispatchable name?** If yes, use the `dispatch_builtin`
   pattern. Both executors get it for free.

3. **Is it a new language construct (syntax)?** If yes, you'll touch
   lexer + parser + AST + HIR + MIR + both executors. Plan ~5-7 files,
   ~500-1500 LOC. Run the parity gate often.

4. **Does it allocate?** If yes, decide if it can run in a `@nogc`
   block. If no, you don't need the verifier dance.

5. **Is it deterministic?** If you're tempted to use `Instant::now()`
   or `HashMap` iteration, find another way. The determinism contract
   is non-negotiable.

6. **Does it have an output the user sees?** Add a fixture for the
   parity gate.

7. **Is it a new pass?** Then §1.5 of this doc applies — register in
   CANA's legality + cost-model + ranker tables.

Whatever the feature is, the architecture above will absorb it cleanly
if you follow the patterns. Good luck.

---

*Generated 2026-06-09 as the closing artifact of the session that
shipped 8 commits across §3.1-§3.3 and §4.1-§4.5 of the v3 handoff.*
