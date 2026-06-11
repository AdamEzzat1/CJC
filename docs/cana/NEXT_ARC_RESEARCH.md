# Next Arc Research — Turning CANA/NSS/PINN Into Measured Wins

**Date:** 2026-06-11 · **Method:** six-role stacked-team research sweep
(Architect, Numerics, Runtime, NSS, Determinism Auditor, QA), read-only
over the post-PINN-v2 tree (`9502620`). This document is the synthesis;
every claim below was grounded in file:line evidence by the role agents.

## 0. The unifying finding

Each of the three goals (speed, memory, thermal) has a concrete path,
and the three technologies divide cleanly:

| Goal | Mechanism | Status |
|---|---|---|
| **Speed** | Energy-SELECTOR (quantum layer finally gets a candidate space) + trained energy head as criterion | Architecture designed; data exists; head needs training |
| **Memory** | Compression on artifacts (checkpoints/traces/sidecars) + memory-gradient corpus → memory head + compile→runtime hints | Tier-1 wins prototype-ready; labels need variance first |
| **Thermal** | Trained head SHIPPED; missing piece is hardware-diagnostic A/B proof | Harness designed; fp_hot/grad family are the subjects |

## 1. Speed — activate the quantum layer via a PassPlanSelector

**Architect findings:**
- New `PassPlanSelector<M, G>` in `crates/cjc-cana/src/plan_selector.rs`
  wrapping PassRanker (NOT inside EnergyAwarePassRanker — separation of
  decide-set vs choose-among-sets). ~400–500 LOC total.
- Deterministic candidate set per function: **10 plans** = ranked + 
  force_none + force_all + 7 canonical singletons. Mirrors the forced
  configs that already PROVED mis-ranking headroom (force_unroll/
  force_all sometimes beat ranked baseline on measured energy).
- Cost: ~8µs/function (legality gate dominates at ~70 checks/function;
  scoring ~100ns/candidate). Negligible.
- Tie-breaking: `(energy, CandidateId)` ascending via `f64::total_cmp`
  (precedent: `cjc-cana-compress/src/energy.rs`).
- Selector identity (`selector_id`, `selector_version`) must join
  report hashes (precedent: PINN_V2_MODEL_ID flip).
- Plan-absence semantics trap: functions ABSENT from PassPlan get
  DEFAULT_PASS_SEQUENCE; present-with-empty get nothing
  (`cjc-mir/src/optimize.rs:342-356`). Candidates must always insert
  explicit per-function entries.

**Numerics findings (the energy head that scores candidates):**
- The measured R²(test) ≈ −32 failure is collinearity + rank-deficient
  one-hots, not absent signal. Prescription: ridge on the 7-feature v2
  basis + 2 NEW features (`countable_loop_count`, `max_loop_depth`
  from CfgMetrics — exist but not in profile rows → schema v3),
  pass-count features, predict score directly. Expected R²(test)
  0.65–0.75 — **a hypothesis to test, not a promise** (the −32
  measurement is the only empirical datum so far).
- MLP (GradGraph + adam_step) only if ridge < 0.65 held-out; weights
  as CPB1 bundle; shadow gate mandatory either way. Baseline to beat:
  hand-tuned FP_ENERGY_WEIGHT = 3.0.
- Per-function energy attribution is NOT possible from current traces
  (no function identity on events; heap is program-wide max). ~2–3 day
  instrumentation extension if per-function energy is ever needed;
  near-term workaround: program-level prediction apportioned by node
  counts.
- Physics post-fit checks: FP-density coefficient > 0 (Fourier-
  flavored); work-conservation decomposition (fit per-term models,
  coefficients must approximately sum to full model's).

## 2. Memory — three prototype-ready wins + the label problem

**Runtime findings (Tier 1, measurable now):**
1. **Trace-event stream compression** — loop-iteration events dominate
   volume and are highly repetitive; LosslessTrace RLE / MotifDictionary
   are production-ready. Plausible 5–28× event-stream reduction;
   measurable as bytes before/after with in-hash integrity (the codec
   embeds input hashes).
2. **Checkpoint compression (cjc-snap)** — chess-RL weight checkpoints
   (~1.1 MB) via `compress_low_rank` under `AdvisoryOnly{tolerance}`;
   plausible 2–3× with bounded Frobenius error; diagnostic checkpoints
   only, never training-resumption paths.
3. **Profile DB / sidecar disk artifacts** — 2–5× plausible, tangible
   file sizes, zero determinism risk.
- **COW-buffer runtime compression: FANTASY** (honest verdict) — would
  re-architect the COW contract. Only ephemeral-on-recycle compression
  is even conceivable, later.
- **Memory-gradient corpus family** (~5 programs: parametric alloc per
  iteration, tensor churn, COW-write stress): expected to move memory
  label std from 0.0007 to ~0.1+, unblocking a PinnMemoryV2 head on
  the thermal-head template.
- **Compile→runtime hint channel EXISTS for allocation** (escape
  analysis `AllocHint` flows MIR→executor already); an NSS-prediction→
  runtime-policy channel is design-ready but unbuilt (new MirFunction
  field + scheduler TLS). Peak-memory reduction claims are speculative
  until measured.

## 3. Thermal — prove the shipped head on silicon (diagnostics only)

**Auditor findings:**
- New `bench/cana_diagnostics/` crate. **Contract: determinism gate
  FIRST** (byte-identical outputs between arms, hard error otherwise),
  THEN diagnostics. Protocol: 2 warm-up runs, interleaved A/B/A/B,
  ~5 s sustained-load phases (thermal time constants are seconds to
  tens of seconds), median-of-5 wall-clock, peak RSS via
  GetProcessMemoryInfo.
- Windows signal reality: wall-clock (±100 ns) and peak RSS (±1%)
  reliable; CPU frequency ~1 Hz sampling best-effort; CPU temperature
  via WMI is `Option<f64>` (many machines lack sensors); thermal-
  throttle state NOT reliably detectable — report frequency TRENDS
  within runs, never absolute cross-machine temps.
- Best A/B subjects (sustained FP load + deterministic): `fp_hot`,
  the `grad_f9*` family, `examples/08_pinn_heat_equation.cjcl`.
- MVP: extend cana_ablation with peak-RSS + wall-clock pairs (~3 h);
  production harness ~8–15 h.
- **Hard wall: diagnostics never feed back into decisions or hashes.**

## 4. NSS — cheap data multipliers, and what to park

**NSS findings:**
- **Per-function pressure labels: implement immediately.** Per-node
  pressures already flow (`cjc-cana-nss/src/lib.rs:462-476`); the
  per-program MAX is a reporting choice in the harness. Schema v3
  (per-fn maps) ≈ 100 LOC → effective corpus 134 → ~536 samples.
- **Post-optimization pressure labels:** full per-config re-
  instrumentation is ~17× harness cost; scope to full_pinn configs
  (~3×) or train a delta-correction model on forced-plan rows instead.
- **Counterfactual-as-plan-scorer: park.** Compile decisions are
  pre-execution; fork-and-observe solves the wrong problem; static
  scoring is 100× cheaper with no demonstrated accuracy deficit.
- **Trace sampling (`tick % N`):** design already reserves it; ~50 LOC;
  prerequisite for instrumenting chess-RL / physics_ml scale workloads.
- **Multi-timescale SSM, density correlations, advisory/autonomous
  controllers: park** (density already serves energy tie-breaking;
  the rest is cluster-ops machinery with no compile-time consumer).

## 5. QA — gates before plans get aggressive

- Four new selector gates (legality-of-selected, selector determinism,
  selector-on/off output parity, never-worse-than-baseline sanity):
  ~22 min CI total, in `tests/test_cana_energy_selector.rs`.
- **Run the EXISTING NoGC verifier + MIR legality verifier on optimized
  output inside the ablation harness** (~20 LOC, ~2 s) — they exist and
  are currently NOT exercised there; mandatory before forced/selected
  unrolling gets more aggressive. Add a code-size bound
  (nodes_after/nodes_before ≤ 1.5) for unroll explosions.
- Generalize the shadow harness (incumbent vs candidate vs labels) to
  model-agnostic form (~200 LOC) when the SECOND trained head arrives —
  not before.
- Macro-workload CI subsets: chess-RL smoke+determinism ~7 min,
  physics_ml grad-graph + 1 PINN canary ~8 min, with weight-hash
  anchors.

## 6. Recommended sequence

| Phase | Work | Goal served | Effort | Measured exit criterion |
|---|---|---|---|---|
| **A** | Per-function labels (schema v3) + loop-count features + memory-gradient programs + NoGC/MIR-verify gates in harness | all (data+safety) | ~1 session | corpus ≥ 5k usable labels; memory std > 0.05; gates green |
| **B** | Energy head training (ridge → escalate only if < 0.65) + CPB1 + shadow gate vs FP_ENERGY_WEIGHT=3.0 | speed | ~1 session | held-out R² + shadow PROMOTE/REJECT verdict |
| **C** | PassPlanSelector + selector gates + ablation config; corpus re-run | speed (quantum activated) | ~1 session | selected plans beat ranked baseline on measured energy on >0 programs, parity 100% |
| **D** | Diagnostics harness (MVP→production); fp_hot/grad A/B; wall-clock + RSS (+best-effort thermal) | thermal + speed proof | ~1 session | byte-equal outputs + reported deltas with confidence bands |
| **E** | Compression prototypes: checkpoint low-rank, trace-stream RLE, sidecar disk | memory | ~1 session | before/after bytes at bounded reconstruction error |
| **F** | Memory head (if Phase A variance confirms) + NSS→runtime hint channel design note | memory | later | shadow gate verdict |

**Parked with reasons:** counterfactual plan scoring, COW-buffer
compression, multi-timescale/advisory/autonomous NSS machinery.

**Honest flags:** the energy-head R² 0.65–0.75 expectation contradicts
the only measurement so far (−32); Phase B exists to settle it
empirically. The 20–30% peak-memory-reduction figure for the runtime
scheduler is unmeasured speculation. Cross-machine thermal comparisons
are methodologically invalid; only within-run trends count.
