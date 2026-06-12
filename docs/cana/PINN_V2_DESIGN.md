# PINN v2 — Design Note (§2.2 objective decision + feature work)

**Date:** 2026-06-10 · **Status:** SHIPPED — shadow gate verdict: PROMOTE
**Prerequisite reading:** `docs/cana/HANDOFF_PINN_V2.md` (the §2.1
data-sanity mandate), `docs/cana/PINN_V1_DESIGN.md`.

## 1. What the §2.1 data-sanity pass found (tool: `bench/cana_train_pinn`)

Run against the 1,474-row corpus (`bench_results/cana_ablation/
profiles.cpdb`, 134 programs × 11 configs):

| Finding | Evidence | Consequence |
|---|---|---|
| Energy (`score`) signal is thin | 39/1,474 rows ≠ 1.0; 17 distinct values; range [0.967, 1.009] | No neural energy head — insufficient label variance |
| Linear saturates the energy signal | corr(log-flops, score) = +0.996 on divergent rows; OLS R²(test) = 0.85 from 30 train rows | The handoff's warning confirmed: MLP adds nothing for score |
| Pressure labels are per-program | bitwise-identical across configs (134/134 after label fix) | Effective n = 134, not 804 |
| Recorded thermal NOT linearly predictable | OLS R²(test) = −0.05 from workload estimates | Investigated: see next row |
| **Root cause: feature set is type-blind** | `estimated_flops = expr_count × loop_amp` counts ALL exprs; `FnFeatures` has no FP-specific field | **Information gap, not model-class gap.** No model can predict FP density from these inputs |
| Corpus label bug | `ends_with("_rec")` missed `_t50`/`_c80`/`_c60` → 402 rows carried Option-A labels | Fixed (membership test); corpus regenerated |

## 2. §2.2 objective decision

Train v2 to predict the OBSERVED outcomes (recorded thermal / cpu /
memory at program granularity + measured baseline-relative energy at
row granularity) from workload estimates — per the handoff
recommendation — **but only after closing the information gap**:

1. **New static analysis `TypeMix`** (`cjc-cana/src/type_mix.rs`):
   conservative intra-function float propagation producing
   `float_binop_count` / `binop_count` / `float_param_count`.
   Mirrors the executor's runtime FP-binop counter
   (`cjc-mir-exec/src/lib.rs:1347-1367`: ANY binary op with ≥1 float
   operand, comparisons included) so the static proxy and the recorded
   thermal label measure the same phenomenon.
2. **`PhysicalCostQuery.float_ops_estimate`** (additive field):
   `float_binop_count × loop_amp × amp.flops`. Additive so PINN v1's
   active closed form is untouched (v1 is in report hashes; changing
   its outputs would be a silent version break).
3. **Profile DB schema v2**: new `estimated_float_ops` column;
   `PROFILE_SCHEMA_VERSION` 1 → 2. Corpus is regenerated
   deterministically (~35 s), so no migration path is needed.
4. **Model class decided by re-run sanity numbers**, not by fiat: if
   linear-on-{log features + FP density} reaches high held-out R² for
   thermal, v2 ships trained linear coefficients; an MLP is justified
   only if the residual is demonstrably nonlinear.

### Float propagation rules (conservative)

- Seed: params with `ty_name == "f64"`.
- `FloatLit` is float; `Var`/`VarLocal` is float iff in the
  float-var set.
- `Add/Sub/Mul/Div/Mod/Pow` produce float iff either operand is float;
  comparisons produce bool (but still COUNT as FP ops when an operand
  is float, matching the runtime counter).
- `Let`/`Assign` of a float-valued expr marks the binding float;
  iterate to fixpoint (loop-carried floats converge; cap at 8 rounds —
  the lattice is monotone two-level so 2 rounds suffice in practice).
- Calls conservatively non-float (under-count; future work could
  consult return annotations program-wide).

### Hashing impact

`TypeMix::feed` joins `FnFeatures::feed` (new tag `0xC0`), so
`FeatureHash` changes for every program. This is a content-addressed
fingerprint doing its job — the features genuinely changed. No test
asserts literal hash values (verified by grep). Sidecar/row hashes
change accordingly; the corpus regenerates.

## 3. Training plan (§2.3–2.5, updated by evidence)

- **Thermal/cpu/memory heads:** program-granularity dataset (n=134),
  features = log1p workload estimates + FP-density ratio. Linear
  (deterministic OLS) vs tiny MLP (GradGraph + `adam_step`, offline,
  seeded) — winner by held-out R², deterministic by-program split.
- **Energy head:** insufficient label variance for any trained model
  (39 informative rows). v2 scope: keep the closed energy form;
  recommend a follow-up harness extension (forced-plan configs) to
  generate energy-label variance before training an energy head.
- **Persistence:** CPB0 bundle (handoff §2.4) once a trained model
  beats the v1 closed form in shadow mode (§2.5).

## 4. Verification

- New unit tests in `type_mix.rs` (propagation, counting, comparisons,
  fixpoint through loops).
- `cargo test --test fixtures --release` (parity gate) before/after.
- `cargo test -p cjc-cana --release --lib` (169 + new).
- Corpus regeneration gates (parity 100%, row-hash double-run stable,
  ≥4 thermal bands, ≥1000 rows).
- Sanity tool re-run: thermal OLS R²(test) is the decision number.

## 5. Results (measured, 2026-06-10)

### Information-gap closure

After `TypeMix` + `estimated_float_ops` landed and the corpus was
regenerated:

| metric | before FP feature | after |
|---|---|---|
| corr(static FP signal, recorded thermal) | n/a (no feature) | **+0.9472** |
| thermal OLS R²(test), by-program split | **−0.0503** | **+0.9558** |

The failure was an information gap, not a model-class gap — confirmed.

### Training (`cargo run --release -p cana-train-pinn -- train`)

Deterministic ridge OLS, 103 train / 31 held-out programs (FNV%5
split). R²(train) 0.9817, **R²(test) 0.9524**, MAE(test) 0.0336
(pre-clamp). Physics checks: standardized FP-density coefficient
**+0.1936 > 0** (asserted); zero-workload raw prediction −0.81 →
clamps to 0 (cold programs predicted cold). Bundle double-write is
byte-identical.

### Shadow gate (`-- shadow`, §2.5)

Ground truth: recorded per-program thermal labels. v2 predictions
through the real clamped head API.

| cohort | v1 MAE | v1 corr | v2 MAE | v2 corr |
|---|---|---|---|---|
| train (103) | 0.1417 | +0.4328 | 0.0158 | +0.9930 |
| **held-out (31)** | 0.1896 | +0.1821 | **0.0208** | **+0.9827** |
| overall (134) | 0.1528 | +0.3676 | 0.0169 | +0.9900 |

**Verdict: PROMOTE** (9× MAE reduction held-out). Attach via
`PinnPhysicalCostModel::with_thermal_head(bundle.head)`; the model then
reports `pinn_thermal_v2` v2 into report hashes.

### Shipped surface

| artifact | location |
|---|---|
| `TypeMix` static analysis | `crates/cjc-cana/src/type_mix.rs` (in `FnFeatures`, feeds `FeatureHash`, tag `0xC0`) |
| `float_ops_estimate` | `crates/cjc-cana/src/physical_cost.rs` (additive; v1 closed form does not read it) |
| Profile DB schema v2 | `crates/cjc-cana-compress/src/profile_db.rs` (`estimated_float_ops`) |
| `PinnThermalV2` head | `crates/cjc-cana/src/pinn_thermal_v2.rs` (7-feature basis, `features_from_query`) |
| Swap-in | `crates/cjc-cana/src/pinn_cost_model.rs` (`with_thermal_head`; NSS max-blend still applies; identity flips to v2) |
| CPB0 bundle codec | `crates/cjc-cana-compress/src/pinn_bundle.rs` |
| Trainer + shadow harness | `bench/cana_train_pinn/` (modes: `sanity`, `train`, `shadow`) |
| Trained weights | `bench_results/cana_train_pinn/pinn_thermal_v2.cpb` |
| Corpus (regenerated, label-fixed, schema v2) | `bench_results/cana_ablation/profiles.cpdb` |

### Out of scope / follow-ups

1. ~~**Energy head**: needs label variance first~~ — **variance landed**
   (see §6); the trained energy head itself remains future work.
2. **cpu/memory heads**: recorded label ranges are too small to train
   on (std 0.013 / 0.0007); v1 closed form stays.
3. ~~**Harness ablation config with the trained head**~~ — DONE, §6.
4. ~~**CLI loading of CPB0 bundles**~~ — DONE, §6. The Phase-6
   `--compression-report` flag remains open (pre-existing follow-up).
5. **Cross-function float propagation** in `TypeMix` (calls are
   conservatively non-float today).

## 6. Second arc (same day): activation + energy-variance groundwork

Shipped after the merge-to-master reconciliation (the merge was a pure
ancestry join — `27c8d6f`'s tree contributed nothing the branch didn't
already have; `git diff` between pre-merge HEAD and the merge commit is
empty).

### `full_pinn_v2_rec` ablation config (plan-level consequences)

The harness loads the committed CPB0 bundle and runs the trained head
as a 12th ranked config. Result: **14/134 plans differ from
`full_pinn_rec`** (3/134 scores differ). The v2 head withholds passes
on FP-dense functions (`float/polynomial`, `fp_hot/horner`,
`fp_hot/__main`) where v1's closed-form thermal was structurally
near-zero (max ≈ 0.0001) — promotion changes real decisions, not just
metrics.

### Forced-plan configs (energy-label variance)

Eight `force_*` configs apply a fixed pass list to EVERY function
(per-`(function, pass)` legality checks retained — 282/2,680 rows show
gate-filtered pairs). `force_none` pins the unoptimized anchor.
Corpus: **134 × 20 = 2,680 rows**; parity 100%; row-hash double-run
stable. Energy signal: **295 informative rows** (was 39), 49 distinct
scores, range **[0.909, 11.16]** (was [0.967, 1.009]). Notably
`force_unroll`/`force_all` sometimes BEAT the ranked baseline (0.909)
— direct evidence the linear benefit model mis-ranks, i.e. learnable
signal. Honest negative: linear OLS on the diverged rows now
catastrophically fails held-out (R²(test) ≈ −32 vs train 0.79) — the
energy head needs a log-ratio target and/or nonlinearity; THAT is the
evidence-based §2.2-style verdict for the future energy session.

### `cjcl run --pinn-weights PATH`

CLI flag (both `PATH` and `=PATH` forms; implies `--mir-opt`; takes
precedence over `--thermal-aware` per the don't-double-wrap rule).
Loads a CPB0 bundle via `cjc-cana-compress::pinn_bundle` (missing or
corrupt bundle = hard CLI error, never silent degradation) and runs
`cjc_mir_exec::run_program_optimized_pinn_v2[_with_executor]` — new
entry points mirroring the thermal-aware pair, same
`PerPassLegalityGate`. Locked by `tests/test_pinn_v2_runner.rs`
(4 tests: AST↔MIR output parity on int/FP/nested-FP programs with the
committed bundle + determinism) and an end-to-end CLI smoke
(`--pinn-weights` output byte-equal to the plain run; missing bundle →
exit 1).

## 7. Phase A1 (2026-06-11): tensor blindness — measured, fixed, retrained (model v3)

### The experiment (`bench/cana_tensor_probe`, read-only)

The next-arc handoff ranked "the thermal head is tensor-blind" as the
highest-information hypothesis. The probe instrumented 4 tensor
programs (matmul / element-wise / method-form reduction / mix; each
running ≥ 409,600 analytic FP ops) plus two controls, and separated
three comparisons the shadow gate conflates: head-vs-label,
label-vs-analytic-FP, and static-features-vs-both.

**Measured (pre-fix): blindness was DUAL — worse than hypothesized.**

| instrument | result on tensor programs |
|---|---|
| recorded thermal label | **0.0000** on all 4 (dense-scalar control: 1.0000) |
| runtime trace FP total | equal to the SCALAR subset to **0.00% error** — 409,600+ tensor FP ops invisible |
| static `est_float_ops` | 16–24 (scalar builder ops only) |
| `tensor_heavy_ops` | missed element-wise binops entirely AND method forms (`.sum()` → 0); only free-call `matmul(a,b)` counted |
| v2 head vs label | \|Δ\| ≈ 0.42 — did not even match the blind label out-of-distribution |

Root cause, code-verified: the runtime counter increments only at the
three scalar Float-binop dispatch arms; the tensor arms
(`cjc-mir-exec`, `(Tensor, …)` binary dispatch) and tensor
builtin/method dispatch never touched it. A "head accurately predicts
the label" result on tensor workloads would have been agreement
between two blind instruments — the probe's analytic lower bounds
(loop trip counts × tensor shapes) are what broke the tie.

**Bonus finding:** the per-window 1.0 intensity cap
(`thermal_raw.min(1.0)`) clips ~23% of FP density on multi-FP-op
STATEMENTS (the scalar control recovers only 76.6% through
`Σ intensity·instr`). This also bounds the harness energy formula's FP
term. Left as-is deliberately — intensity is a density in [0, 1] and
the label semantics want saturation — but recorded for Phase B, whose
energy target reconstructs FP through the same capped field.

### The fix (dual-side, mirror-semantics preserved)

- **Runtime (label):** `trace_fp_ops` widened u32→u64; tensor binops
  count element-wise FP work pre-dispatch (`tensor_binop_fp_work`),
  matmul counts `2·m·k·n` from runtime shapes, curated FP-heavy
  builtins/methods (`sum`, `mean`, `dot`, `adam_step`, …) count
  element work at both free-call and method dispatch choke points.
  Pure data movement (`transpose`, `reshape`, `get`) deliberately
  counts 0; under-counting stays the safe direction. All accounting
  gated on `trace_enabled`.
- **Static (features):** `TypeMix` propagates a second monotone
  binding set (tensor-ness; seeds: `Tensor` params, `TensorLit`,
  tensor-returning calls/constructors/methods) and reports
  `tensor_binop_count` — tensor-involving binops are EXCLUDED from
  `float_binop_count`, mirroring the runtime arm precedence
  (`2.0 * t` is a tensor op, not a scalar FP op). `MemoryProxy` now
  classifies method-call callees (`.sum()` counts; `Tensor.from_vec`
  is an alloc site). `build_physical_query` prices tensor sites at
  `TENSOR_FP_PER_OP = 128` elements into `float_ops_estimate`.
- **Basis (model v3):** the density feature is capped at 1.0 to mirror
  the label's intensity cap (tensor functions push the raw ratio to
  5–10 against a label pinned at 1.0); magnitude survives in
  `ln(1+float_ops)`. `PINN_V2_MODEL_VERSION` 2 → 3.
- **Corpus:** 9-program `tensor_` family added to the ablation
  harness (4 hot-loop shapes covering each blind path + graded
  `tensor_tg_k{0..4}` sweeping tensor-vs-scalar FP share). Without
  it, the new signal would have had zero training variance — the
  pre-A1 corpus contained no tensor ops at all, so all 134 existing
  programs' rows are label-identical before/after the runtime fix.

### Post-fix measurements

Probe re-run: all 4 tensor programs' labels flipped 0.0000 → 1.0000;
static `est_float_ops` 16 → 1,040–3,088; the committed probe now
asserts the post-fix state (a return to "CONFIRMED blind" = regression).

Corpus: **143 × 20 = 2,860 rows**, parity 100% (tensor programs are
byte-identical across eval/MIR-exec — locked in
`tests/test_tensor_fp_accounting.rs` BEFORE the family entered the
harness), row-hash double-run stable. Recorded thermal: std 0.0
→ **0.3495** across programs, 15 programs in the saturated band.

Retrain + shadow (model v3, 111 train / 32 held-out):

| cohort | v1 MAE | v1 corr | v2(v3) MAE | v2(v3) corr |
|---|---|---|---|---|
| train (111) | 0.2035 | +0.2972 | 0.0312 | +0.9873 |
| **held-out (32)** | 0.2150 | +0.1626 | **0.0319** | **+0.9819** |
| overall (143) | 0.2061 | +0.2692 | 0.0314 | +0.9861 |

**Verdict: PROMOTE** (R²(test) 0.9564; FP-density coefficient +0.7444
> 0 asserted). Train→regen fixed point verified: regenerating the
corpus under the v3 bundle and retraining reproduces the bundle
byte-for-byte (labels feed from `nss_rec` rows, which don't depend on
the head). Plan-level consequence grew: 23/143 plans differ between
`full_pinn_v2_rec` and `full_pinn_rec` (was 14/134).

### Test surface added

- `tests/test_tensor_fp_accounting.rs` — 6 wiring tests (label reads
  hot end-to-end; counter exceeds scalar-only bound; static pricing;
  instrumented-output identity; eval↔MIR parity on tensor programs).
- `tests/test_tensor_typemix_props.rs` — 3 proptest properties (exact
  graded counting, order invariance, density bounds) + 1 bolero
  structural fuzz (copy-chain totality/consistency).
- `cjc-cana` lib: +10 unit tests (TypeMix tensor propagation,
  MemoryProxy method classification, physical-cost tensor pricing,
  density cap) → 195 lib tests.

## 8. Phase A items 2–7 (same session): schema v3, families, gates, holdout

Full sequencing record in `docs/cana/HANDOFF_PHASE_A.md`; this section
is the numbers ledger.

### Schema v3 + final corpus

`FnProfile` per-function records (workload estimates + the
previously-unstored `countable_loop_count`/`max_loop_depth` + per-fn
cpu/memory/thermal labels). `PROFILE_SCHEMA_VERSION = 3`. Corpus:
**158 programs × 20 configs = 3,160 rows** (134 prior + 9 `tensor_` +
5 `mem_grad_` + 10 `holdout_`); parity 100%; row-hash double-run
byte-identical. **Per-function labels: 360** (vs 158 per-program) —
fn-granularity thermal std 0.3214 (usable), cpu 0.0128, memory 0.0008.

### A4 measured (negative result, mechanism-confirmed)

`mem_grad_a{1..5}` sweeps ×4 arena-let executions per step. Measured
program-level memory label: max **0.0078** at `mem_grad_a5` — exactly
the mechanism's prediction (≈ mean of cumulative 64 B × 65,536
arena-let executions / 1 GiB), std 0.0009. The label is structurally
blind to Rc memory (`gc_alloc` objects + arena-let executions are ALL
it sees); no honest program family can reach the research doc's
std > 0.05 expectation. Phase F starts with a label-side fix.

### A5 measured (bound refuted and re-set)

First gated regen: the research doc's 1.5× code-size bound tripped on
the ranked BASELINE plan — full unroll of a countable 8-trip loop grew
`grad_f10_d2_n64` 97 → 605 nodes (**6.24×, by design**). Cap re-set to
16× (runaway-duplication scale), with a measured corpus-max printed
every regen. NoGC + MIR-legality verifiers now run on every optimized
program in every config (3,160/3,160 green).

### Retrain on v3 corpus (114 train / 34 FNV-test / 10 holdout-frozen)

R²(train) 0.9587, **R²(test) 0.9552**, FP-density coefficient +0.7212.
Train↔regen fixed point re-verified after the coherence regen.

### Shadow (the frozen-holdout debut)

| cohort | v1 MAE | v1 corr | v2(v3) MAE | v2(v3) corr |
|---|---|---|---|---|
| train (114) | 0.1982 | +0.3014 | 0.0315 | +0.9865 |
| held-out FNV (34) | 0.2023 | +0.1735 | 0.0314 | +0.9820 |
| **holdout-frozen (10)** | 0.4761 | −0.1038 | **0.1885** | **+0.8107** |
| overall (158) | 0.2166 | +0.2419 | 0.0414 | +0.9690 |

**Verdict: PROMOTE** — and the frozen cohort immediately justified its
existence: true never-seen generalization is MAE 0.19/corr +0.81, a
6× degradation from the eroding FNV split's 0.031 (still 2.5× better
than v1). Report the holdout line, not the FNV line, when making
external claims about head accuracy. Plan-level: 38/158 plans differ
between `full_pinn_v2_rec` and `full_pinn_rec` post-retrain.

## 9. Phase B (2026-06-11, same session): the trained energy head — PROMOTE

### Data sanity settled the hypothesis (`-- sanity-energy`)

Working rows 2,960 (holdout's 200 frozen), 306 diverged, score range
[0.496, 11.16] — note the new sub-1.0 tail: some plans now HALVE
modeled energy on tensor programs. The grid (program-level split):

| recipe | R²(test) |
|---|---|
| REPLICATION: config one-hots + raw score, diverged rows | **−16.98** (the −32-class failure reproduces on v3) |
| raw-score target, any features, any rows | −1.3 … −18 |
| ln(score), diverged, base features | −0.36 |
| ln(score), diverged, +loop features | +0.45 |
| ln(score), diverged, +structural | +0.34 |
| **ln(score), diverged, +loops+structural** | **+0.8207** |

The research doc's 0.65–0.75 expectation is **settled: exceeded** —
the Numerics diagnosis (collinear one-hots + missing loop features +
heavy-tailed raw target) was correct on all three counts.

### The regret-vs-R² fork (the load-bearing finding)

The deployed consumer is a plan SELECTOR; its metric is **regret**
(measured score of the predicted-cheapest plan minus the true
minimum). Two fits compete:

| fit rows | R²(test, diverged, ln) | test regret | holdout regret |
|---|---|---|---|
| diverged only | **0.82** | +0.0509 (30/34) — *worse than always-baseline* | +0.0504 (9/10) |
| **ALL rows (ties included)** | 0.21 | **+0.0014 (32/34 exact-best)** | **0.0000 (10/10)** |

The R²-best model is the regret-WORST option: trained only on
divergent plans, it never learned where ties live and confidently
mispredicts no-change plans. **Shipped recipe: all-rows fit, ln
target, +loops+structural, no one-hots** — chosen by the deployment
metric. Baselines on test: always-baseline +0.0332,
structural-argmin +0.0081 (28/34). On holdout the structural
heuristic also reaches 0.0 — the separation shows on the test cohort.

### Shipped surface

- `crates/cjc-cana/src/pinn_energy_v1.rs` — `PinnEnergyV1`
  (variable-length basis: 10 workload + P pass counts + 4 tail;
  the pass vocabulary is part of the trained artifact),
  `EnergyQuery` + `features_from_query` as THE single basis
  definition (trainer and future selector both lift into it),
  `predict_ln_score` (fixed-order, no FMA, dimension-guarded).
- `crates/cjc-cana-compress/src/energy_bundle.rs` — CPB1 codec
  (magic `CPB1`; vocabulary persisted; adversarial-length guards;
  corruption matrix + double-write determinism unit tests; proptest
  round-trip ∀ vocabulary; bolero decoder fuzz).
- Trainer modes `train-energy` / `shadow-energy`; weights at
  `bench_results/cana_train_pinn/pinn_energy_v1.cpb` (21 features:
  vocabulary = the 7 corpus passes). Baseline-plan sanity
  prediction +0.031 ≈ tie.
- `tests/test_energy_head.rs` — 5 wiring tests on the committed
  bundle (identity, bit-determinism, finite across the plan space,
  plan-discrimination, unknown-pass alignment).

### Shadow gate (through the persisted CPB1 artifact)

Diverged FNV-test R²(ln) 0.2091 (positive floor passes; the 0.82
belongs to the regret-losing fit — both recorded). Regret: test
+0.00140 (exact 32/34) beats always-baseline (+0.03316) and
structural-argmin (+0.00806); holdout 0.00000 (10/10) ties
structural. **Verdict: PROMOTE — the trained head is Phase C's
selector criterion.** Caveat recorded: regret was evaluated over the
20-config plan space as a proxy; Phase C re-validates on its actual
10-candidate-per-function space before any activation.

## 10. Phase C (2026-06-11, same session): PassPlanSelector — the energy layer's first measured outcome effect

### What shipped

`crates/cjc-cana/src/plan_selector.rs` — `PassPlanSelector` per the
research-doc architecture (choose-among-plans wrapping the ranker's
decide-a-plan, NOT inside `EnergyAwarePassRanker`). Candidate set per
function, fixed IDs: ranked plan (explicit `DEFAULT_PASS_SEQUENCE`
when the ranked plan has no entry — the absence trap), none, all 7
canonical passes, and each canonical singleton = 10. Per-(function,
pass) legality filtering through the SAME `PerPassLegalityGate` the
forced configs use; scoring via the committed CPB1 head over
per-function `EnergyQuery`s with the post-plan node count obtained by
ACTUALLY applying each candidate (deterministic, cached per distinct
filtered plan); `(predicted, candidate_id)` argmin under
`f64::total_cmp` (`select_argmin` is public for the property tests).
Identity: `energy_selector_v1` v1 joins row identity as
`energy_selector_v1+pinn_energy_v1`.

Gates (`tests/test_cana_energy_selector.rs`, 7 tests): the four QA
gates (independent legality re-verification of every selected pass;
double-run determinism; selector-on output parity vs AST eval on
int/FP/tensor/multi-fn programs; never-worse-than-ranked on the
predicted criterion) + the explicit-entries trap test + argmin
total-order proptest + committed-head totality fuzz (bolero over u64
query extremes). Plus 7 in-crate unit tests (neutral-head tie-break
keeps the ranked plan; absence-means-default made explicit; etc.).

Feedback-loop guard: `selector_rec` rows are EXCLUDED from energy-head
training and evaluation (`ENERGY_EXCLUDED_CONFIGS` in the trainer) —
a model must never train on its own decisions. Verified: after the
selector regen, re-running BOTH trainers reproduces BOTH bundles
byte-identically.

### The measured exit criterion (`selector_rec`, 158 × 21 = 3,318 rows)

| config | mean measured score | divergent rows | range |
|---|---|---|---|
| baseline plan | 1.00000 | 0 | — |
| `full_pinn_v2_rec` (ranked incumbent) | 1.00329 | 8 | [0.967, 1.508] |
| **`selector_rec`** | **0.98230** | 22 | [**0.496**, 1.143] |

- **Exit criterion MET**: selector beats the baseline plan on
  6/158 programs on measured energy (the requirement was >0);
  parity 100% on all 3,318 rows; verifiers green; row-hash stable.
- **First nonzero outcome effect for the energy layer, ever** — every
  prior ablation measured exactly zero because the layer had a
  one-candidate space.
- **Honest texture**: the selector makes bold bets. 6 large wins
  (down to 0.496 = −50% modeled energy) against 16 modest regressions
  (worst +14.3%); vs the ranked incumbent it is better on 7, worse on
  11, but **mean −2.1 points** (0.982 vs 1.003). The regressions are
  the out-of-distribution effect Phase B's caveat predicted: the head
  scores novel pass combinations (e.g. 5-pass reordered plans) it
  never saw in training.
- **The wins, NAMED (post-commit report addition)**: `mem_grad_a{1..5}`
  + `holdout_alloc_pulse`, all ≈0.496 — every win is the
  allocation-churn shape (dead per-iteration allocations in loops),
  i.e. ONE mechanism: the selector recommends DCE where the ranked
  stack under-recommends it. Strong part: `holdout_alloc_pulse` is a
  FROZEN-HOLDOUT program — the mechanism generalized to code neither
  head trained on. Weak part: dead-alloc loops are maximally
  DCE-friendly synthetic shapes; win DIVERSITY is 1 family. Recorded
  so Phase D's A/B and any external claim carry the right caveat.

### NOT done / next

The selector is ablation-grade, not default-on: 16 measured
regressions disqualify unconditional activation. Candidate guardrails
for a future session (in evidence order): (a) margin gating — keep
the ranked plan unless the predicted gain exceeds a threshold
calibrated on the corpus; (b) add selector-produced plan shapes to
training via a SEPARATE exploration config whose rows are
head-independent (forced versions of the selector's candidate set),
closing the OOD gap without the feedback loop; (c) Phase D wall-clock
validation before any of this matters to users.

## 11. Phase D (2026-06-11, follow-on session): silicon validation — the wins hold

Full record: `docs/cana/PHASE_D_DIAGNOSTICS.md`; harness:
`bench/cana_diagnostics`; artifacts:
`bench_results/cana_diagnostics/{REPORT.md,phases.csv,plans/}`.

- **The §10 caveat resolved in the wins' favor**: 5 of the 6 named
  selector wins hold on wall-clock with the ENTIRE conservative
  median-of-5 band below 1.0 — `mem_grad_a2..a5` at 0.287–0.371
  median, `holdout_alloc_pulse` (frozen holdout) at 0.301.
  `mem_grad_a1` direction-consistent (median 0.668) but
  noise-inconclusive. Byte-identical outputs everywhere (gate before
  any timing), corpus scores reproduced to 1e-9 in the measured build.
- **Measured beats modeled** (0.29–0.37 vs 0.496): the energy formula
  prices every non-FP statement at 1, but the DCE'd statements are
  ALLOCATIONS (~2–3× an average interpreter statement). Formula
  recalibration direction: allocation-statement weight analogous to
  `FP_ENERGY_WEIGHT`.
- **Modeled ties measure as ties**: all thermal-family and
  tensor-family subjects (modeled 1.000) land inconclusive, except one
  borderline noise-suspect regression (`tensor_tg_k3`, band lo 1.017).
  The thermal-aware stack shows NO wall-clock effect on the Track-3
  subjects — its plans tie baseline on executed work.
- **Real-program guard**: `examples/08_pinn_heat_equation.cjcl` under
  the selector plan — output byte-identical, no measurable wall/RSS
  delta. Also exposed a 1.63 GB selector candidate-probing RSS spike
  (planning-time, not execution) → fixed same day on master
  (`e3f631b`, `optimize_function_with_passes`; plan identity
  corpus-gated, so the timed plans are unaffected).
- RSS: plan choice does not move peak memory (≈0.999 ratios), as
  expected for freed-within-iteration allocations.
