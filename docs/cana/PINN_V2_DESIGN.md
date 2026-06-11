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
