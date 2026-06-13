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

## 7. Hypotheses settled since this synthesis (update log)

- **2026-06-11 — "thermal head is tensor-blind": SETTLED, CONFIRMED,
  FIXED** (Phase A item 1; full record in `PINN_V2_DESIGN.md` §7).
  Measured worse than hypothesized: blindness was DUAL — the recorded
  LABEL (runtime FP counter counts scalar binops only) and the static
  basis both read tensor workloads as cold (label 0.0000 on programs
  running 409,600+ FP ops; trace totals matched the scalar subset to
  0.00% error). Fixed on both sides (element-count runtime accounting
  + TypeMix tensor propagation + method-call classification), corpus
  extended with a 9-program `tensor_` family (143 × 20 = 2,860 rows),
  head retrained as model v3: shadow MAE held-out 0.0319 vs v1 0.2150,
  corr +0.98, PROMOTE. New bonus finding for Phase B: the per-window
  intensity cap clips ~23% of FP density on multi-FP-op statements —
  the energy formula's FP term inherits this bound.
- **2026-06-11 — "memory-gradient programs reach label std ~0.1+":
  REFUTED** (Phase A item 4; `PINN_V2_DESIGN.md` §8). The memory label
  is structurally blind to Rc memory: `heap_bytes_in_use` = gc_alloc'd
  objects × 4096 + executed arena-classified `Let`s × 64 (flat,
  size-blind, cumulative) — arrays/tensors/strings never register.
  The `mem_grad_a{1..5}` family hit the mechanism-exact ceiling
  (max 0.0078, std 0.0009). §2's "0.0007 → ~0.1+" expectation cannot
  be met by programs; Phase F starts with a label-side fix.
- **2026-06-11 — "code-size bound nodes_after/before ≤ 1.5": REFUTED
  on first contact** (Phase A item 5). The ranked BASELINE plan fully
  unrolls countable 8-trip loops at 6.24× node growth by design
  (`grad_f10_d2_n64`: 97 → 605). Gate shipped at 16× (runaway scale)
  with a measured corpus-max report each regen.
- **2026-06-11 — FNV-split erosion quantified** (Phase A item 7): the
  frozen holdout's debut shadow shows true never-seen generalization
  at MAE 0.1885 / corr +0.8107 vs the FNV split's 0.0314 / +0.9820 —
  a 6× optimism gap. Promotion gates keep PASSing (v1 is at 0.4761
  on the same cohort), but external accuracy claims must quote the
  frozen-holdout line.
- **2026-06-11 — "ridge + loop features reaches energy R² 0.65–0.75":
  SETTLED, EXCEEDED** (Phase B; `PINN_V2_DESIGN.md` §9). On diverged
  rows with ln(score) target + loop + structural features:
  R²(test) **0.8207** (the −32-class failure replicated at −16.98
  with the old recipe — the §1 collinearity diagnosis was right on
  all three counts). NEW finding the hypothesis didn't anticipate:
  the R²-best fit (diverged-only) is the regret-WORST selector
  criterion (+0.051 test regret, worse than always-baseline +0.033);
  the all-rows fit wins deployment (+0.0014, 32/34 exact-best picks,
  10/10 frozen holdout) at R² 0.21. The shipped `pinn_energy_v1`
  CPB1 head uses the regret-chosen recipe; shadow verdict PROMOTE.
  Phase C must re-validate regret on its actual 10-candidate space.
- **2026-06-11 — "selector finds energy wins on real programs":
  SETTLED, CONFIRMED with honest texture** (Phase C;
  `PINN_V2_DESIGN.md` §10). `selector_rec` is the FIRST config in
  project history with mean measured energy below baseline: 0.98230
  (ranked incumbent: 1.00329). Exit criterion met — beats the
  baseline plan on 6/158 programs, parity 100% across 3,318 rows.
  Texture: 6 large wins (to −50%) vs 16 modest regressions (worst
  +14%) — the predicted out-of-distribution effect on novel pass
  combinations. Ablation-grade, NOT default-on; margin gating /
  head-independent exploration configs / Phase-D wall-clock are the
  paths forward. Feedback-loop guard live: selector rows excluded
  from energy training, both bundle fixed points byte-verified.
- **2026-06-11 — "modeled-energy wins appear on wall-clock": SETTLED,
  CONFIRMED — and the model UNDERSELLS them** (Phase D;
  `PHASE_D_DIAGNOSTICS.md`, harness `bench/cana_diagnostics`). 5 of
  the 6 named selector wins hold on silicon with the ENTIRE
  conservative median-of-5 band below 1.0: `mem_grad_a2..a5` at
  0.287–0.371 median wall ratio, `holdout_alloc_pulse` (frozen
  holdout) at 0.301; `mem_grad_a1` direction-consistent but
  noise-inconclusive. Byte-identical outputs on all 23 subjects;
  corpus scores reproduced to 1e-9 in the measured build before any
  clock was read. NEW finding: measured reduction EXCEEDS modeled
  (0.29–0.37 vs 0.496) because the formula prices every non-FP
  statement at 1 while the DCE'd statements are allocations (~2–3×
  an interpreter statement) — alloc-statement weight is the
  recalibration sketch. Modeled ties measured as ties (thermal +
  tensor families inconclusive; one borderline noise-suspect
  regression `tensor_tg_k3` band-lo 1.017). Bonus finding: selector
  candidate-probing peaked 1.63 GB RSS on the real example program
  (planning-time) — per-function optimize API now evidence-backed.
- **2026-06-12 — Phase E compression plausibility bands: SETTLED in
  all three lanes, two surprises** (`PHASE_E_COMPRESSION.md`;
  `bench/cana_compress_probe`). (1) Trace streams: "plausible 5–28×"
  EXCEEDED — 35–43× lossless via delta/XOR-columnar + motif,
  bit-exact roundtrip proven; the representation transform matters
  more than the codec. (2) Checkpoint low-rank "2–3×": measured
  1.38× at ≤5% rel-Frobenius on the real (near-init, 60-episode)
  chess-RL checkpoint — full-rank init-like matrices correctly kept
  raw; band unresolved for converged models. Format correction: the
  checkpoint is `tensor_snap` (CJCT), not cjc-snap as §2 claimed.
  (3) Disk artifacts "2–5×": motif hits 8.34× on profiles.cpdb while
  byte-RLE EXPANDS it (0.96×) — codec/representation pairing rule now
  measured, not assumed.
- **2026-06-12 — "memory label fixable via creation-site byte
  counting": SETTLED, CONFIRMED — and the blocker moved one layer
  down** (Phase F0; `PHASE_F0_MEMORY_LABEL.md`). Label std 0.0009 →
  0.1083 (trainability bar 0.05 CLEARED) via
  `alloc_bytes_in_window` + cumulative adapter term; capacity
  calibrated from regen evidence (64 MiB guess → 4 MiB measured).
  Plans/scores/bundles all byte-identical — pure label signal. NEW
  finding: the static feature set now fails where the label used to
  (R²(test) 0.048 vs 0.77 train) — the A1/§2.1 information-gap
  pattern on the memory axis. F1 = static creation-site alloc
  estimate (FeatureHash ripple, own session), THEN the memory head.
- **2026-06-13 — "the F0 feature-side blocker is fixable with a static
  creation-volume feature": SETTLED, PARTIALLY CONFIRMED + memory head
  PROMOTE** (Phase F1; `PHASE_F1_MEMORY_HEAD.md`). New
  `MemoryProxy::lit_elem_slots` → `creation_alloc_bytes_estimate`
  (schema v4) lifted R²(test) 0.019 → 0.088 (~4.6×) over its
  no-creation ablation — the feature is load-bearing (beats ablation on
  every cohort) but does NOT fully close the gap; the memory signal is
  harder than thermal because allocation volume is trip-count-dominated
  and static loop-amplification only approximates it. `pinn_memory_v1`
  (CPB2, 8-feature linear) SHADOW verdict PROMOTE: beats train-mean
  climatology AND the ablation on held-out (MAE 0.0412) and overall
  (0.0196) MAE, frozen-holdout corr +0.96, creation coeff +0.18.
  Ships shadow-only. KEY safety result: thermal+energy bundles retrain
  BYTE-IDENTICAL on the v4 corpus (invisible to them), both still
  PROMOTE, plans byte-identical. F2 lever: trip-count-aware
  amplification (sharpens creation AND flops/bytes; own FeatureHash
  ripple).
- **2026-06-13 — "the selector's 16 regressions are fixable by margin
  gating": SETTLED, PARTIALLY — half of them, for free** (Phase G;
  `PHASE_G_MARGIN_GATING.md`). `with_margin(τ)` keeps the ranked plan
  unless a switch's predicted advantage clears τ. Calibrated τ=0.02
  STRICTLY DOMINATES the ungated selector: regressions 16 → 7 (−56%),
  wins 6 → 7, mean 0.98230 → 0.98186, frozen-holdout win preserved,
  selector_rec + energy bundle byte-identical. NEW findings: optimal τ
  is SMALL (0.02, not the hypothesized ~0.70 — the head gates on
  per-function advantages compressed by the program→function
  granularity mismatch; above 0.05 wins get gated away and mean climbs
  >1.0); and a regression FLOOR of 7 survives any τ — confident head
  MISPREDICTIONS, not marginal switches. Margin gating structurally
  can't reach the floor; it needs head-independent exploration configs
  (handoff §3b) to retrain the head on the OOD pass combinations. The
  selector is closer to default-on but not there.
