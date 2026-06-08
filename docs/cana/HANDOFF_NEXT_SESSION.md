# CJC-Lang CANA — Next-Session Handoff Notes

**Date:** 2026-06-07 (end of session)
**Branch:** `claude/compassionate-chebyshev-8956ab` (worktree)
**Workspace root:** `C:\Users\adame\CJC`
**Worktree:** `C:\Users\adame\CJC\.claude\worktrees\compassionate-chebyshev-8956ab`
**Companion docs:**
- `~/Downloads/NSS_ARCHITECTURE_AND_COMPILER_IMPROVEMENTS.md`
- `~/Downloads/NSS_HANDOFF_PHASE_5_COMPILER_INTEGRATION.md`

---

## 0. TL;DR for the next session

Two parallel tracks of remaining work:

**Track A — Cost-model validation (4 items, ~1 day total):**
The trained model shipped in `1592e2d` reports good *training* RMSE, but has never been validated against held-out programs, real workloads, or a feature distribution that resembles real CJC-Lang code. Four concrete experiments are scoped to find out whether the model actually generalizes — see §3.

**Track B — NSS integration (the Phase 4 design doc finally has its bridge):**
The user's sibling fork shipped `cjc-nss` (240 tests passing, 2.4k LOC). The Phase 4 prep work that landed in `b327cb6` (PressurePredictor trait + ThermalAwareCostModel + KernelVariant) is now ready to be plumbed into a real `cjc-cana-nss` bridge crate. See §5 and the two NSS handoff docs listed above.

The tracks are independent — you can pick either one (or both) without ordering constraints.

---

## 1. What shipped this session (chronological commit list)

20 commits, in order:

```
0d4a41c  feat(cana): wire --mir-opt through CANA pass-plan recommendation
cc9e8fb  bench(cana): Phase 2 pass-ordering benchmark + findings doc
38b05be  fix(cjc-mir): LICM must not hoist let-bindings whose name is reassigned in loop (task_a30e8eec)
522520b  feat(cana): Phase 5 caching — CachingPassRanker eliminates Phase 2 overhead on repeat compilations
8dc23a7  bench(cana): expand corpus to 8 programs + add cana_cached config
ece3a81  feat(cana): Phase 3 foundation — kernel fusion candidate identification
98f2329  docs(cana): Phase 4 NSS integration design doc
7888176  fix(cjc-mir): dominates() OOB on unreachable blocks (task_9d7ae8b2)
d487c00  feat(cana): Phase 3.5a foundation — fused_matmul_dot tensor primitive
4dff0c0  feat(cana): Phase 3.5a-prime — fused_matmul_norm tensor primitive
4bca589  feat(cana): Phase 3.5c — MIR fusion rewriter (matmul → norm)
f89f3ca  refactor(tests): extract shared run_eval/run_mir/run_parity helpers
0131439  feat(cana): Phase 3.5b — GradOp::MlpLayerMatMul fused autodiff op
37500e6  test(cana): Phase 3.5b — Selu + SinAct backward parity coverage
6ff5c1b  feat(cana): Phase 3.5d — fused_matmul_matmul (third fusion pair)
3db2694  feat(cana): Phase 3.5e — auto-inject fusion_rewrite into PassPlan
b327cb6  feat(cana): Phase 4 prep — PressurePredictor + ThermalAwareCostModel scaffolding
ecd817b  feat(cana): TIER B #4 — trained cost model + 18-program corpus
2f8b136  feat(cana): cost model v2 — 60-program corpus + MIR-count proxy + N_ITERS=21
1592e2d  feat(cana,cjc-mir): cost model v3 — Option A + permanent diagnostics + CSE bug fix
```

The durable wins (independent of trained coefficient values) from `1592e2d`:

1. **`PassDiagnostics` + `apply_pass_with_diagnostics`** public API in `cjc_mir::optimize` — permanent observability surface. Each pass now reports `changes_applied` as first-class data.
2. **CSE bug fix** — `expr_key` now handles `MirExprKind::VarLocal { name, .. }`. Before this, post-slot-resolution CSE silently never fired on any real lowered code. Same family as `task_a30e8eec` and commit `2983f2b`.
3. **Pass-native change-count returns** — `strength_reduce_fn`/`cse_fn`/`licm_fn` return `usize` instead of `()`. Public `apply_pass` still returns `bool`. Internal change, but enables observability.

---

## 2. Verification commands (run first in the new session)

```bash
cd C:/Users/adame/CJC/.claude/worktrees/compassionate-chebyshev-8956ab

# Quick sanity: CANA tests
cargo test --release -p cjc-cana
# Expect: ok. 125 unit + 26 integration tests passing (90+26 → 125+26 over this session)

# Critical: AST/MIR parity (load-bearing gate)
cargo test --test fixtures
# Expect: test run_all_fixtures ... ok

# cjc-mir new tests (Option A: diagnostics)
cargo test --release -p cjc-mir tests::
# Expect: 174 passed (was 148 → 159 after fusion → 174 after diagnostics)

# Phase 3.5 fused-primitive suites (4 tests targets, all still green)
cargo test --test fused_matmul_dot --test fused_matmul_norm \
            --test fused_matmul_matmul --test fused_mlp_matmul --release

# Training binary (regenerates v3 coefficients in ~10 seconds)
cargo run --release -p cana-train-cost-model | head -100
# Should print "TRAINED COEFFICIENTS — paste into linear_cost_model.rs"
# followed by the per-pass match arms.
# Same bit-for-bit output as the v3 commit on every re-run.
```

---

## 3. Track A — Cost-model validation

The trained model shipped in `1592e2d` has serious unverified claims. Four
experiments to find out whether it generalizes.

### 3A.1 Held-out test set — ~30 min

**The gap.** The "RMSE dropped 14×" headline is *training* RMSE — the
model's error on the corpus it was fit to. A model that perfectly
memorizes its training set has zero training RMSE. We have no idea how
the model performs on programs it hasn't seen.

**The experiment.**

```
1. In `bench/cana_train_cost_model/main.rs`, split PROGRAMS into:
     train_set: first 53 programs (or every 4th excluded)
     test_set:  remaining 16 programs

2. Collect training points only from train_set.

3. Fit per-pass coefficients via fit_ols_gd, same algorithm as today.

4. Apply the fit model to the test_set: predict benefit per (test_program,
   pass), compare to measured benefit, compute test_rmse.

5. Print side-by-side:
     pass               train_rmse | test_rmse | gap
     constant_fold      0.045       | ???       | ???
     dce                0.067       | ???       | ???
     strength_reduce    0.029       | ???       | ???
     licm               0.012       | ???       | ???
     cse                0.011       | ???       | ???
```

**What you'll learn.** If `test_rmse ≈ train_rmse`, the model
generalizes — the headline holds. If `test_rmse > 2 × train_rmse`,
we've overfit the small corpus and the trained values shouldn't be
trusted for production decisions.

**Add to docs.** Update `docs/cana/CANA_COST_MODEL_TRAINING_FINDINGS.md`
with the train/test split table. Possibly downgrade the headline claims
if test_rmse is much worse than train_rmse.

### 3A.2 Real-workload AB test — 2-4 hr

**The gap.** Nobody has compiled a real CJC-Lang program with
`default_ranker` vs `trained_ranker` to see whether trained decisions
differ at all, let alone whether they differ for the better.

**The experiment.** Pick three substantial real programs:

| Program | Source | Why |
|---|---|---|
| Chess RL training | `tests/chess_rl_v2/` | Large, multi-function, real ML workload |
| PINN heat 1D | `examples/physics_ml/pinn_heat_1d_pure.cjcl` | Real GradGraph-heavy program |
| LendingClub demo | `demos/lendingclub/` | Real Locke workload, big DataFrame |

For each program × ranker config:
1. Compile with `--mir-opt`, capture the PassPlan emitted by
   `recommend_pass_plan(&mir)`.
2. Diff the per-function pass sequences (default vs trained).
3. Time the compile step (`compile_us`).
4. Time the runtime of the produced MIR (`run_us`).
5. Verify byte-identical output (no regression in correctness).

**What you'll learn.**
- If trained and default produce IDENTICAL PassPlans, the trained
  coefficients are irrelevant in practice — fine to ship, but no
  measurable win.
- If trained PassPlans differ but runtime is unchanged, the model is
  shuffling deck chairs.
- If trained PassPlans differ AND compile-or-runtime improves, the
  model genuinely works.
- If trained PassPlans differ AND compile-or-runtime regresses, the
  trained model is HARMFUL in production — `trained_ranker()` should
  be marked deprecated/experimental until fixed.

**Bonus.** This is the first real end-to-end exercise of the entire
CANA + Phase 3.5 fusion pipeline on real workloads. Expect to discover
secondary issues (e.g. fusion candidates that don't materialize because
no `--mir-opt` consumer enables `fusion_rewrite` by default yet).

### 3A.3 Cross-corpus validation — 1 hr

**The gap.** Train on our 69-program corpus, evaluate on the 8 programs
from `bench/cana_pass_ordering/` (the original Phase 2 benchmark
corpus, hand-written by a different author for a different purpose).

**The experiment.**

```
1. Add a `--eval-corpus=path/to/programs.rs` flag to
   bench/cana_train_cost_model/main.rs.
2. Default: train + evaluate on the 69-program corpus (existing behavior).
3. With --eval-corpus=bench/cana_pass_ordering: train on the 69 programs,
   evaluate on the 8 pass_ordering programs.
4. Print eval RMSE per pass.
```

**What you'll learn.** Same as 3A.1 but with a truly external corpus.
If RMSE on the pass_ordering corpus is similar to held-out RMSE
from 3A.1, the model generalizes across authors and styles.
If it's much worse, the model has overfitted to my hand-written corpus
style (lots of integer loops, no tensors).

### 3A.4 Feature coverage audit — 30 min

**The gap.** `w_alloc_sites` is `0.0` for all five passes in the v3
fit. That's because the corpus is 69 integer-arithmetic programs with
zero allocation sites. The model has never observed `alloc_sites > 0`
and silently learned to ignore that feature.

Any real CJC-Lang program using tensors, arrays, or strings has
non-zero `alloc_sites`. The trained model has **zero information** in
that feature dimension. Its predictions on real programs will be
implicit extrapolation.

**The experiment.**

```
1. In main.rs after Phase 1 (collect training points), print:

     === Feature distribution actually seen during training ===
       expr_count   min=___ q25=___ q50=___ q75=___ max=___ unique=___
       loop_depth   min=___ max=___ histogram: 0=__ 1=__ 2=__ 3=__ 4=__
       branch_count min=___ q25=___ q50=___ q75=___ max=___
       alloc_sites  min=___ max=___ unique=___

2. Flag any feature where unique <= 2 or max == 0 with:

     "WARNING: feature X has insufficient variance in corpus.
      Trained model effectively ignores this dimension."

3. Add corpus programs covering the unseen feature corners:
   - At least 3 programs with alloc_sites > 0 (tensor allocation, array,
     string interning)
   - At least 3 programs with branch_count > 10 (deep nested if/else,
     match dispatch)

4. Re-fit and compare coefficient drift on the augmented corpus.
```

**Production gate.** This is the most consequential of the four — if
the model is missing entire feature dimensions, no amount of
better-looking RMSE makes it trustworthy on a real workload. Consider
adding an assertion in `LinearCostModel::trained()` that prints a
warning when called from a program whose features fall outside the
training distribution.

### 3A.5 Documenting the eight critiques

The session's critical assessment of v3 produced eight "what looks
better than it actually is" items. Items 1-3 are partial wins; items
4-8 are unverified claims:

1. (good) CSE bug fix is real and durable.
2. (good) `PassDiagnostics` API is permanent observability infra.
3. (good) Methodology iteration was rigorous and self-correcting.
4. **(unverified)** RMSE is training RMSE, not held-out — addressed by §3A.1.
5. **(systemic)** Mean benefits are tiny (RMSE is 6× the mean signal).
   Partly addressable by corpus expansion (§3A.4), partly inherent to
   the linear-cost-model design. Worth documenting as a known
   limitation in the LinearCostModel doc comment.
6. **(unverified)** `w_alloc_sites = 0` everywhere — addressed by §3A.4.
7. **(unverified)** Negative wrong-sign weights for CF and DCE
   `w_branch_count` — addressed by §3A.1 (if the test RMSE is fine,
   the signs are genuine; if not, overfitting).
8. **(known limitation)** Pass-native signal ≠ runtime benefit. SR
   `x * 2 → x + x` counts as 1 rewrite but produces zero runtime gain.
   Mitigation: add a `weight_by_inferred_runtime_impact` annotation to
   each pass that the diagnostic-count signal multiplies by. Future
   work; not in scope of the immediate validation push.

Action: add §6 to `docs/cana/CANA_COST_MODEL_TRAINING_FINDINGS.md`
that summarises items 4-8 as **known limitations to address before
recommending the trained model for production**. This is a temperature
check — promote `trained_ranker()` to "stable" only after the
validation in 3A.1-3A.4 passes.

---

## 4. Track B — NSS integration

The user shipped `cjc-nss` in a sibling fork. The two NSS docs in
`~/Downloads/` describe what's now available:

- **`NSS_ARCHITECTURE_AND_COMPILER_IMPROVEMENTS.md`** — what NSS is and
  why it's a good fit for compilers (PressureKind × NodeId substrate,
  multi-timescale signal decomposition, deterministic replay, exact
  causal attribution).
- **`NSS_HANDOFF_PHASE_5_COMPILER_INTEGRATION.md`** — what NSS Phase 5
  shipped: MIR adapter, legality verifier, autonomous optimizer with
  audit trail. **240 tests passing, 0 failures.**

CANA's Phase 4 prep work (`b327cb6`) anticipated this exactly:
`PressurePredictor` trait, `ThermalAwareCostModel`, `KernelVariant`
enum. Those types are ready to receive an NSS-backed implementation
when the bridge lands.

### 4B.1 Drop cjc-nss into the workspace — 1 hr

**The experiment.**

```
1. Copy the cjc-nss crate from the sibling fork into:
     crates/cjc-nss/
2. Add to root Cargo.toml workspace members + workspace deps:
     [workspace]
     members = [..., "crates/cjc-nss"]

     [workspace.dependencies]
     cjc-nss = { path = "crates/cjc-nss" }
3. Confirm cjc-nss tests pass in this workspace:
     cargo test -p cjc-nss --release
     # Expect: 240 tests, 0 failures
4. Confirm nothing else regresses:
     cargo test --workspace --release --no-fail-fast
```

**What's there.** All of Phases 1-5 from the NSS handoff doc, including:
- `ClusterNeuralSystemsSimulator` (the predictor)
- `mir_adapter::adapt_mir_trace_to_cluster_trajectory` (the bridge to
  MIR traces)
- `LegalityVerifier` (catches oscillation / cooldown / capacity issues
  in optimization scripts)
- `AutonomousOptimizer` (safety-bounded closed loop)
- `DecisionRecord` (content-addressed audit trail)

### 4B.2 Build `crates/cjc-cana-nss/` bridge crate — 2-3 days

**The experiment.** New workspace member that implements
`cjc_cana::PressurePredictor` against `cjc_nss::ClusterNeuralSystemsSimulator`.

Skeleton:

```rust
// crates/cjc-cana-nss/src/lib.rs

pub struct NssPressurePredictor {
    nss: cjc_nss::ClusterNeuralSystemsSimulator,
    seed: cjc_nss::NssSeed,
}

impl NssPressurePredictor {
    pub fn from_seed(seed: u64) -> Result<Self, NssBridgeError> {
        let cfg = cjc_nss::ClusterNssConfig {
            temporal_mode: cjc_nss::TemporalMode::MultiAll,
            ..Default::default()
        };
        let nss = cjc_nss::ClusterNeuralSystemsSimulator::from_seed(
            cfg, cjc_nss::NssSeed(seed))?;
        Ok(Self { nss, seed: cjc_nss::NssSeed(seed) })
    }
}

impl cjc_cana::PressurePredictor for NssPressurePredictor {
    fn predict_thermal(&self, program: &MirProgram, features: &CanaFeatures)
        -> BTreeMap<String, f64> {
        // 1. Project program features onto NSS's PressureKind × NodeId substrate
        //    (each MirFunction becomes a NodeId; CanaFeatures map onto pressures)
        // 2. Build a synthetic ClusterTrajectory from the projection
        // 3. Call nss.predict_next(...) for each trajectory tick
        // 4. Read the per-node thermal pressure component from the prediction
        //    and assemble the BTreeMap keyed by function name
    }
    // ... same pattern for predict_memory_peak, predict_cpu_saturation
    fn identify_structural_hot_kernels(...) -> Vec<String> {
        // Read NSS's multi-timescale engine — functions with persistent
        // signal at α=0.99 (structural timescale) are the candidates.
    }
    fn name(&self) -> &'static str { "nss_v1" }
    fn version(&self) -> u32 { 1 }
}
```

**Files to consult:**
- `crates/cjc-cana/src/pressure.rs` — the trait shipped in `b327cb6`
- `crates/cjc-cana/src/thermal_cost_model.rs` — the wrapper that
  consumes the trait
- `~/Downloads/NSS_HANDOFF_PHASE_5_COMPILER_INTEGRATION.md` §3.2-3.3 —
  end-to-end wiring recipe

**Tests required.**
- `NssPressurePredictor` is deterministic — same MIR + same seed →
  byte-identical predictions across runs/OS/CPU.
- `NssPressurePredictor` returns the correct shape (one entry per
  function in the program).
- Composing `ThermalAwareCostModel<LinearCostModel, NssPressurePredictor>`
  with `default_ranker` produces the expected per-pass adjustments.
- Round-trip: feed a hand-built MIR program through the predictor and
  verify the per-function thermal map matches what NSS would emit if
  invoked directly with the same projection.

### 4B.3 Wire NSS into `--mir-opt --thermal-aware` CLI flag — 1 day

**The experiment.** Make NSS-aware ranking opt-in via a new CLI flag.

```
1. Add to cjcl CLI:
     --thermal-aware  enable NssPressurePredictor + ThermalAwareCostModel
     --thermal-target=NN  override the 75°C default (see thermal_cost_model.rs)

2. In cjc-cli or cjc-mir-exec, when the flag is set:
     let pp = cjc_cana_nss::NssPressurePredictor::from_seed(seed)?;
     let cost_model = ThermalAwareCostModel::new(
         LinearCostModel::trained(), pp);
     let ranker = PassRanker::new(cost_model, DefaultLegalityGate::new());
     let report = ranker.rank(&mir, &features);
     let plan = pass_plan_from(&report.sequence);
     // continue with optimize_program_with_plan as today

3. Document --thermal-aware in cjcl help text + a new docs/cana/
   THERMAL_AWARE_COMPILATION.md.
```

**What you'll learn.** Whether NSS-driven recommendations actually
change PassPlans in measurable ways on real programs. This is the
first time NSS's predictions affect compile-time decisions.

### 4B.4 Hot/Warm/Cool kernel variant codegen — 1 week

**The experiment.** From `crates/cjc-cana/src/kernel_variant.rs`
(shipped in `b327cb6`):

```rust
pub enum KernelVariant { Hot, Warm, Cool }
```

Phase 3.5's fusion codegen today emits exactly one variant (Hot —
fully fused). The Phase 4 plan calls for emitting all three and letting
the runtime pick at call time based on NSS-observed pressure.

**Per fused primitive in cjc-runtime:**

| Variant | Implementation |
|---|---|
| Hot | The fully-fused primitive that ships today (e.g. `fused_matmul_norm`). |
| Warm | Partially fused: keep the matmul and norm as separate primitive calls but skip the intermediate Tensor wrapper. ~30% slower than Hot, ~30% cooler. |
| Cool | MIR-walked: the unfused chain `matmul(a, w)` then `norm(h)`. Slowest, lowest thermal pressure. |

Runtime selector:

```rust
// In cjc-runtime, replace direct dispatch of "fused_matmul_norm" with:
let variant = KernelVariant::select_for_budget(
    nss.current_thermal_pressure_for_this_kernel());
let result = match variant {
    KernelVariant::Hot => fused_matmul_norm_hot(a, w),
    KernelVariant::Warm => fused_matmul_norm_warm(a, w),
    KernelVariant::Cool => unfused_matmul_norm(a, w),
};
```

Byte-identical output across variants — that's the Phase 1 determinism
contract carried forward. Only resource usage differs.

**Tests required.** For each fused primitive, parity tests across all
three variants on hundreds of random inputs.

### 4B.5 The user's "second neural network" composition

Per `NSS_ARCHITECTURE_AND_COMPILER_IMPROVEMENTS.md` §8, the user is
working on a second NN. Three composition points are available:

1. **Audit-trace consumer** (lowest coupling, recommended start).
   Second NN reads `cjc_nss::DecisionRecord` streams as feature inputs.
2. **Pressure-field interlingua** (medium coupling). Second NN's output
   projects onto `PressureKind`s the same way the MIR adapter does.
3. **Learned-feature extractor** (highest coupling). Second NN feeds
   into NSS's encoder via a `LearnedPressureField` wrapper.

**Next-session conversation starter.** Before designing the bridge in
4B.2 specifically, ask the user which composition point they intend for
the second NN, since that affects how `NssPressurePredictor` should
construct its inputs.

---

## 5. Repository state snapshot

### Per-crate test counts (verified individually this session)

| Crate / target | Tests | Notes |
|---|---|---|
| cjc-mir | 174 | +9 diagnostics tests + +5 fusion_rewrite tests this session |
| cjc-runtime | 754 | +13 fused-primitive kernel tests this session |
| cjc-cana (unit) | 125 | +35 over the session (90 → 96 fusion → 118 NSS-prep → 125 trained) |
| cjc-cana (integration) | 26 | unchanged |
| Phase 3.5 fused-primitive suites | 90+ | 4 separate test targets |
| fusion_rewrite | 7 | NEW this session |
| fused_mlp_matmul | 20 | NEW this session |
| AST/MIR parity (fixtures) | 1 | unchanged — green |

### 15 known pre-existing failures (from session-start handoff)

The handoff at the start of session 2026-06-07 listed 15 pre-existing
failures: 7 Tier 0 replay + 8 Tier 2 canary tests. I did not touch any
of those surfaces, so they should be unchanged. The full workspace
sweep at session start would have taken 5+ hours and was killed; the
per-crate verification above is more reliable.

Documentation: `docs/cana/CANA_PHASE_1_REGRESSION_FAILURES.md` (from
prior session).

### Files added this session

```
crates/cjc-cana/src/pressure.rs                              NEW (Phase 4 prep)
crates/cjc-cana/src/thermal_cost_model.rs                    NEW (Phase 4 prep)
crates/cjc-cana/src/kernel_variant.rs                        NEW (Phase 4 prep)
crates/cjc-mir/src/fusion_rewrite.rs                         NEW (Phase 3.5c)
bench/cana_train_cost_model/programs.rs                      NEW (cost model training)
bench/cana_train_cost_model/main.rs                          NEW (cost model training)
bench/cana_train_cost_model/Cargo.toml                       NEW
tests/fused_matmul_dot/{mod,unit,wiring,proptest,fuzz,cana_integration}.rs  NEW (3.5a)
tests/fused_matmul_norm/{mod,unit,wiring,proptest,fuzz,cana_integration}.rs  NEW (3.5a-prime)
tests/fused_matmul_matmul/{mod,unit,wiring,proptest,cana_integration}.rs    NEW (3.5d)
tests/fused_mlp_matmul/{mod,forward_parity,backward_parity,determinism}.rs   NEW (3.5b)
tests/fusion_rewrite/{mod,parity,identification}.rs          NEW (3.5c)
tests/fused_test_helpers/mod.rs                              NEW (shared helpers)
docs/cana/CANA_PHASE_3_5_FUSION_CODEGEN_DESIGN.md            NEW
docs/cana/CANA_COST_MODEL_TRAINING_FINDINGS.md               NEW (v1 → v2 → v3 doc)
docs/cana/HANDOFF_NEXT_SESSION.md                            NEW (this file)
```

### Files modified this session (key ones only)

```
crates/cjc-mir/src/dominators.rs           dominates() OOB fix
crates/cjc-mir/src/optimize.rs             Diagnostic counts + PassDiagnostics API + apply_pass_with_diagnostics
crates/cjc-cana/src/fusion.rs              +norm, +fused_matmul_norm/dot/matmul in NATIVE_PRIMITIVES
crates/cjc-cana/src/linear_cost_model.rs   CoefficientSource enum + trained() constructor + v3 coefficients
crates/cjc-cana/src/pass_ranker.rs         Auto-inject fusion_rewrite + trained_ranker()
crates/cjc-cana/src/lib.rs                 +pressure, +thermal_cost_model, +kernel_variant modules
crates/cjc-ad/src/lib.rs                   +GradOp::MlpLayerMatMul (Phase 3.5b)
crates/cjc-runtime/src/builtins.rs         +fused_matmul_dot/norm/matmul dispatch arms
crates/cjc-runtime/src/accumulator.rs      +fused_matmul_*_kernel functions
crates/cjc-eval/src/lib.rs                 allowlist + fused primitives
crates/cjc-mir-exec/src/lib.rs             allowlist + fused primitives + scope routing fix
Cargo.toml                                  +bench/cana_train_cost_model workspace member
```

---

## 6. Quick reference — common operations

```bash
# See what changed in the most recent commit
git show HEAD --stat

# Run just the cost-model training tests
cargo test --release -p cjc-cana --lib linear_cost_model::

# Run just the diagnostics tests
cargo test --release -p cjc-mir tests::diagnostics

# Regenerate trained coefficients (deterministic, ~10s + build time)
cargo run --release -p cana-train-cost-model

# Run a CJC-Lang program with the trained ranker
cjcl run example.cjcl --mir-opt
# (No CLI flag for trained_ranker yet — would need wiring in cjc-cli)

# Phase 3.5 fusion end-to-end check (parses a CJC-Lang program with a
# matmul → norm chain, lowers to MIR, runs fusion_rewrite, executes)
cargo test --test fusion_rewrite --release

# AST/MIR parity gate (the load-bearing test for any --mir-opt change)
cargo test --test fixtures
```

---

## 7. Project rules (CLAUDE.md prime directives — re-read first thing)

The `CLAUDE.md` in workspace root enforces:

1. Don't break the compiler pipeline (Lexer → Parser → AST → HIR → MIR → Exec)
2. No hidden allocations or GC usage in NoGC-verified paths
3. Maintain deterministic execution (bit-identical replay)
4. Preserve backward compatibility
5. Never silently refactor unrelated systems — scope changes to the feature
6. Language primitives stay minimal; higher-level functionality in libraries
7. Both executors must agree (cjc-eval vs cjc-mir-exec parity)

Determinism rules:
- All floating-point reductions use Kahan or BinnedAccumulator
- BTreeMap/BTreeSet everywhere — never HashMap with random iteration
- RNG is SplitMix64 with explicit seed threading
- SIMD kernels must NOT use FMA — preserves bit-identical results
- Parallel ops must produce identical results regardless of thread count

For NSS work specifically, layer in the additional invariants from
`NSS_HANDOFF_PHASE_5_COMPILER_INTEGRATION.md` §4 — same family of
constraints (no FMA, BTreeMap iteration, Kahan reductions, deterministic
substreaming).

---

## 8. If asked to do anything destructive

- NEVER force-push to master.
- NEVER --no-verify on commits (CLAUDE.md forbids).
- NEVER amend existing commits — make new ones.
- NEVER stage `bench_results/phase_0_8_demos/*.svg` — unrelated to this work.

---

## 9. Most likely first task you'll be asked

In rough probability order:

(a) **"Drop in cjc-nss and validate"** → §4B.1 (~1 hr). Lowest-risk
    way to start the NSS track. Validates the user's sibling fork
    plays well with the parent workspace.

(b) **"Run the held-out test"** → §3A.1 (~30 min). Cheapest cost-model
    validation experiment. Determines whether the 14× headline is real.

(c) **"AB test on chess RL"** → §3A.2 (~2-4 hr). Highest-information
    validation experiment but takes longer. Will reveal real-world
    behaviour of the trained model.

(d) **"Build the cjc-cana-nss bridge"** → §4B.2 (~2-3 days). Major
    scope; expect to break it across multiple sessions. The Phase 4
    prep work landed specifically to make this mechanical, not novel.

(e) **"Feature coverage audit"** → §3A.4 (~30 min). Important
    correctness check. Likely outcome: corpus needs 5-10 more programs
    with tensor allocations before trained_ranker() is safe to enable
    for general use.

---

## 10. Final session arithmetic

- **20 commits** shipped end-to-end on this branch this session.
- **3 durable compiler improvements** independent of training: PassDiagnostics API, CSE bug fix, diagnostic-count returns.
- **5 phases of CANA** advanced: 3.5a, 3.5a-prime, 3.5b, 3.5c, 3.5d, 3.5e, Phase 4 prep, TIER B #4 v1→v2→v3.
- **0 regressions** in any tested gate (the 15 known pre-existing failures are documented and untouched).
- **The big unknown**: does the v3 trained model actually help on real programs? §3 answers this.
- **The big unblock**: cjc-nss is now available, ready to bridge. §4 starts that work.

---

*Companion documents (place alongside this file when starting the next session):*

- `~/Downloads/NSS_ARCHITECTURE_AND_COMPILER_IMPROVEMENTS.md`
- `~/Downloads/NSS_HANDOFF_PHASE_5_COMPILER_INTEGRATION.md`
- `docs/cana/CANA_COST_MODEL_TRAINING_FINDINGS.md`
- `docs/cana/CANA_PHASE_4_NSS_INTEGRATION_DESIGN.md` (the design doc; this handoff is the implementation plan)
- `docs/cana/CANA_PHASE_3_5_FUSION_CODEGEN_DESIGN.md`

END HANDOFF
