# Handoff — PINN v2 Neural Training (everything not finished)

**Date:** 2026-06-10
**Branch:** `claude/objective-davinci-62975a` (worktree `C:\Users\adame\CJC\.claude\worktrees\objective-davinci-62975a`)
**HEAD:** `ee7b341`
**Supersedes:** `docs/cana_compress/PHASE_A_HANDOFF_V2.md` (its Tracks 1-2 and
all v2 prerequisites are DONE; only its §4.3/§5.4 v2-neural material is live)
**Sibling:** `docs/cana/HANDOFF_NEXT_SESSION_v4.md` (general new-feature
architecture guidance — still accurate)

---

## 0. THE RULE THAT PAID FOR ITSELF SEVEN TIMES

**Verify every handoff/design claim against the code before building on
it.** This session's predecessor docs were wrong in seven separate,
implementation-blocking ways, every one caught by reading the code first:

1. The Phase-6 "11 commits shipped" had **never been committed** — 8.8K
   LOC sat uncommitted in this worktree while memory claimed it shipped.
2. `CostEstimate::Unknown` does NOT mean "do nothing" — `PassRanker`
   KEEPS unknown-benefit passes (`UnknownButKeptConservatively`). Hard
   rejection = `Estimated { value: 0.0, confidence: 1.0 }`.
3. `VerifierOutcome` / `ParityOutcome` types never existed anywhere.
4. A4's prescribed location (`cjc-cana/profile_db.rs`) required a
   dependency cycle.
5. A5's prescribed design ("extend trait methods to take an extra
   param") cannot compile; wrapper pattern used instead.
6. The NSS adapter **never loaded `PressureKind::Thermal` at all** —
   `predict_thermal` returned 0.0 always, pre-`89a3aa9`. The design's
   `thermal_intensity` field was specified but never added.
7. The instrumentation under-counted straight-lined code 24× (residual
   instruction window never drained) — caught only because the new
   energy score surfaced a physically impossible 0.0417 ratio.

This document is itself subject to the rule. **`git log` and the code
are authoritative; this doc is advice.**

---

## 1. State at handoff — what IS done

Seven commits, every one with the AST/MIR parity gate green:

```
ee7b341  Track-3 prerequisites: energy gate score + thermal gradient + 1474-row corpus
89a3aa9  Option B PRs 2-5: real executor instrumentation (+ Thermal pipeline fix)
a11578b  Phase A4-A6: profile DB + compression wiring + 5-way ablation
f614c56  PINN v1: deterministic physical-cost layer
66b65bd  cjc-quantum wide-matrix SVD fix + fmt sweep
e0e819b  Phase 6 baseline: cjc-cana-compress + cjc-nss::density (the
         previously-uncommitted work, verified then committed)
37adaf7  (base) v4 handoff
```

The full advisory stack is functionally complete end-to-end:

- **PINN v1** (`cjc-cana/src/{physical_cost,pinn_cost_model}.rs`):
  deterministic coefficient model, active, in report hashes as
  `pinn_coeffs_v1` v1.
- **Real instrumentation** (`cjc-mir-exec/src/{trace.rs,lib.rs}`):
  emit sites at function entry / while-iteration / if-branch / FP
  binop / print / gc_collect, + residual drain at `exec()` exit.
  Instrumentation is semantically transparent (locked by
  `tests/test_mir_exec_instrumented.rs`, 5 tests).
- **Recorded predictor**
  (`cjc-cana-nss::RecordedPressurePredictor::from_recorded_events`):
  real traces → real per-function pressures. Thermal genuinely
  differentiates (gradient family spans 0.091→0.818 in 5 bands).
- **Training corpus**: `bench_results/cana_ablation/profiles.cpdb` —
  **1,474 rows** (134 programs × 11 configs), 113 plan divergences,
  parity 100%, row-hash double-run stable. Regenerate any time with
  `cargo run --release -p cana-ablation` (~34 s, deterministic).
- **Profile DB API**
  (`cjc-cana-compress/src/profile_db.rs`): `read_all(path)` →
  `Vec<CompilationProfile>`; `row_hash()` excludes wall-clock.

Plan-divergence proof points (the machinery works):
- `nss_rec` on fp_hot: thermal 1.0 > 0.80 → `loop_unroll` withheld.
- `full_pinn_rec` on fp_hot: 1.0 > 0.95 hard cap → plan emptied.
- The c80/c60 cap variants + t50 threshold variant trip on different
  slices of the thermal gradient → label variance across configs.

---

## 2. NOT FINISHED — PINN v2 neural training (the actual work)

This is the entire remaining scope of the original plan. Five steps,
roughly two sessions:

### 2.1 Data-sanity pass (DO THIS FIRST, before any training code)

Read the 1,474 rows (`cjc_cana_compress::profile_db::read_all`) and
check:
- Label distributions: `score`, `nss_predicted_*`, `pinn_predicted_*`,
  `mir_nodes_*`, energy-relevant fields. How much variance is there
  really, per config and per program family?
- Feature↔label correlation: if the linear signal already saturates
  (plausible — `LinearCostModel::trained()` exists for a reason), a
  2-layer MLP may add nothing. Know this BEFORE writing the loop.
- The `score` field is baseline-relative energy (lower = better);
  `pass_sequence` is the plan; `config_id` distinguishes the 11 stacks.

### 2.2 Training objective (the one open decision)

Recommended (from the closing analysis of the prerequisite session):
**predict the OBSERVED outcomes (recorded thermal/memory/cpu + measured
relative energy) from the workload estimates** (`estimated_flops`,
`estimated_bytes_*`, `estimated_working_set`, plus pass-plan features).
That directly replaces `predict_physical`'s closed form with learned
coefficients AND fixes the gate's hand-tuned `FP_ENERGY_WEIGHT = 3.0`
(in `bench/cana_ablation/main.rs`) at the same time.

Physics-residual soft losses from the Phase-A handoff §4.3 (work
conservation, arithmetic intensity / roofline, Fourier law, Little's
law) — as weighted penalty terms in the objective, NOT hard
constraints.

### 2.3 Training loop

- `cjc-ad::GradGraph` + the native `adam_step` builtin — the exact
  pattern proven in the Phase-3c PINN demos
  (`crates/cjc-ad/src/pinn.rs`, `examples/physics_ml/`).
- ~2-layer MLP, ~64 hidden units (handoff suggestion; let the
  data-sanity pass inform this).
- Offline only (bench harness, never during compilation). Fixed seed,
  Kahan reductions, deterministic iteration order — all the usual
  invariants apply to TRAINING as well.
- Where: new bench crate `bench/cana_train_pinn/` (workspace member),
  reading the cpdb, emitting a weight bundle.

### 2.4 Weight persistence

`CPB0` bundle (CANA PINN Bundle v0) built on `LosslessTracePayload` —
the Phase-A handoff §8 Q2 decision. Magic + schema_version + weights +
`model_id`/`model_version` (must flow into report hashes; the v1
precedent is `PINN_V1_MODEL_ID` in `cjc-cana/src/pinn_cost_model.rs`).

### 2.5 Swap-in + shadow mode → active promotion

- v2 keeps the `PinnPhysicalCostModel<M, P>` surface; only
  `predict_physical`'s internals change (load weights instead of the
  coefficient table). Callers don't move.
- **Shadow mode is REQUIRED for v2** (trained weights — unlike v1,
  which skipped it by having none). Run v2 in parallel with v1 in the
  ablation harness; record both predictions per row; promote only if
  v2's predictions beat v1's against measured labels.
- ⚠ The §5.4 shadow-mode spec in PHASE_A_HANDOFF_V2.md is **the last
  large untested section of that handoff**. Per §0 of this doc: verify
  its assumptions against the code before building it.

### 2.6 Known measurement caveats to carry into v2

- The §5.2 gate currently reads 0/134 wins for full_pinn_rec — this is
  an honest measurement: thermal withholding breaks even on modeled
  energy (best case −3.3%); its value is the cap itself, invisible to
  a scalar energy score until the energy model has trained
  coefficients. Don't "fix" the gate by hand-tuning; fix it by
  training.
- `nss_quantum_rec` never diverges from `nss_quantum` (0 score
  differences): the energy re-ranker reorders within recommendation
  lists but doesn't change the plan SET. If v2 wants the quantum layer
  to matter for outcomes, the ranker would need to influence
  recommendations, not just their order — that's an architecture
  question, not a bug.
- Loop-iteration emits dominate event volume. Fine at this corpus
  scale (~34 s total); if v2 adds much bigger programs, consider
  sampling (`tick % N == 0`) — event-count-based only, never
  time-based.

---

## 3. NOT FINISHED — housekeeping (independent of v2)

1. **Merge to master.** This branch is 7 commits ahead of base
   `37adaf7`; master has since moved to `27c8d6f` (a merge of
   `compassionate-chebyshev`). Likely conflict points: workspace
   `Cargo.toml` member list, `Cargo.lock`. The branches
   `peaceful-moore-92fb41` / `cranky-shamir-4915bd` / `modest-cerf`
   still point at `37adaf7` — only this branch carries the new work.
2. **`cjcl --compression-report` CLI flag** — Phase 6 follow-up, never
   wired.
3. **Streaming compression API** — current API is batch-only.
4. **5 non-canary serializer/replay test failures** — pre-existing,
   documented in `docs/cana/CANA_PHASE_1_REGRESSION_FAILURES.md`, need
   real root-cause work.
5. **The v4 handoff's unrelated queue** (not this arc): user-defined
   decorators (design in `docs/DECORATORS_DESIGN.md`), monomorphize /
   specialize / vectorize real implementations
   (`docs/cana/PASS_IMPLEMENTATION_DESIGNS.md`), Lanczos restart,
   generalized eigenproblem `A x = λ B x`, real ODE/PDE/symbolic
   implementations behind the §4.3 stubs.
6. **rustfmt + `#[path]` gotcha**: `cana_ablation` path-includes
   `bench/cana_train_cost_model/programs.rs`; running rustfmt on the
   ablation crate reformats the included file too. Cosmetic, but
   surprising — already swept once in `ee7b341`.

---

## 4. File map (this arc only)

```
crates/cjc-cana/src/physical_cost.rs        PhysicalCost{Query,Estimate,Coefficients,Constraints}, build_physical_query, predict_physical
crates/cjc-cana/src/pinn_cost_model.rs      PinnPhysicalCostModel<M,P> (v2 swaps internals here)
crates/cjc-cana-nss/src/lib.rs              NssPressurePredictor (Option A) + RecordedPressurePredictor (Option B)
crates/cjc-cana-nss/src/pinn_bridge.rs      PhysicalCostEstimate → PressureKind deltas
crates/cjc-cana-compress/src/profile_db.rs  CompilationProfile + cpdb append/read
crates/cjc-cana-compress/src/energy.rs      11-field EnergyComponents
crates/cjc-cana-compress/src/bridge.rs      CompressionAwarePressurePredictor (A5)
crates/cjc-nss/src/mir_adapter.rs           MirTraceEvent (incl. thermal_intensity) + adapter
crates/cjc-mir-exec/src/trace.rs            TraceCollector + TLS + with_trace
crates/cjc-mir-exec/src/lib.rs              emit sites + trace_node_assignments() + residual drain
bench/cana_ablation/                        the harness; regenerates the corpus deterministically
bench_results/cana_ablation/profiles.cpdb   1,474-row training corpus (committed)
tests/test_mir_exec_instrumented.rs         instrumentation transparency gate (5 tests)
docs/cana/PINN_V1_DESIGN.md                 v1 design note (incl. corrected rejection semantics)
docs/cana/PHYSICS_INFORMED_CANA_NSS.md      user-facing doc for the physical layer
docs/cana/OPTION_B_DESIGN.md                instrumentation design (PRs all landed)
docs/cana_compress/PHASE_A_HANDOFF_V2.md    original plan; only §4.3/§5.4 v2 material still live
```

## 5. Verification commands (run before AND after any change)

```bash
cd /c/Users/adame/CJC/.claude/worktrees/objective-davinci-62975a

cargo test --test fixtures --release                 # THE load-bearing parity gate
cargo test --test test_mir_exec_instrumented --release  # instrumentation transparency
cargo test -p cjc-cana --release --lib               # 169 expected
cargo test -p cjc-cana-compress --release            # 173 lib + integration suites
cargo test -p cjc-cana-nss --release                 # 21 expected
cargo test -p cjc-nss --release --lib                # 219 expected
cargo test -p cjc-mir-exec --release --lib           # 28 expected
cargo run --release -p cana-ablation                 # ~34 s; asserts parity/hash/bands/rows
```

## 6. Determinism invariants (unchanged, non-negotiable)

BTreeMap everywhere; Kahan/Binned for FP reductions; no FMA; SplitMix64
seeded RNG only; FNV-1a hashing; `f64::total_cmp`; no wall-clock or
atomics in decision paths (wall-clock allowed only as diagnostic
metadata, excluded from `row_hash`); model_id+version in report hashes;
training offline only; shadow mode for trained weights; legality gate
retains final authority.

---

*Closing artifact of the 2026-06-10 session (7 commits, e0e819b..ee7b341).
Next session: §2.1 data-sanity pass first, then the v2 training loop.*
