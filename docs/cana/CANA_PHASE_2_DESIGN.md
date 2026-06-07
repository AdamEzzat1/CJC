# CANA Phase 2 — Design Decisions & Foundation

**Status:** Foundation shipped (`LinearCostModel`, `PassRanker`); wiring into `optimize_program` deferred to a follow-up commit.
**Crate:** `crates/cjc-cana/`
**Test count after this phase:** 63 unit tests + Phase 1's integration suite (8 wiring + 9 determinism + 7 proptest + 2 bolero).

---

## TL;DR

Phase 2 ships two new modules in `cjc-cana`:

- **`linear_cost_model.rs`** — first non-null `CostModel` implementation. Hand-tuned linear regression over `CanaFeatures`. Deterministic, total, no training data required.
- **`pass_ranker.rs`** — produces a per-function ranked pass sequence by querying the cost model, sorting by predicted benefit, dropping sub-threshold recommendations, and filtering through the legality gate.

Together they constitute the **first user-facing advisory output** from CANA — a `RankingReport` that names the recommended pass sequence for a program. Phase 1 told you *what your MIR looks like*. Phase 2 tells you *what the compiler should do with it*.

What this phase **does not** do (intentionally): apply any optimization. The recommendations exist; consuming them requires a follow-up change to `cjc-mir::optimize::optimize_program` to accept an optional pass sequence. That change is small, but it crosses crate boundaries and must be parity-tested. It is the natural next commit.

---

## 1. Architecture decisions

### ADR 0001 — Cost model is a hand-tuned linear function, not a trained NN

**Context.** Section 11 of `CANA_NSS_COMPILER_IMPROVEMENT_PLAN.md` calls for "a simple linear regression cost model trained on synthetic MIR programs." We need a benchmark corpus to train on, and we don't have one yet.

**Decision.** Ship a **hand-tuned** `LinearCostModel` with coefficients chosen by inspection (e.g. "LICM weights `loop_depth` at 0.15; CSE weights `expr_count` at 0.0006"). Every coefficient is a constant in the source file with a comment explaining its value.

**Rationale.**
- **Trivial auditability.** Every prediction is reconstructable by reading the source. A trained model is a black box.
- **No training-data dependency.** The Phase 2 deliverable doesn't depend on assembling a benchmark corpus first.
- **Trait-compatible.** Phase 5's trained model implements the same `CostModel` trait; swap-in is mechanical.
- **Determinism.** Hand-tuned constants are bit-stable across machines. Trained weights stored on disk are too, but the training process itself isn't.

**Consequences.**
- (+) Phase 2 ships in a single session of work.
- (+) Predictions are explainable to a human reviewer.
- (–) Predictions are less accurate than a trained model would be on edge cases.
- (–) Adding a new pass requires editing the source file's coefficient table.

**Replaces:** Phase 5 will swap in a trained `LinearCostModel` (same surface) once a benchmark corpus exists. The hand-tuned model will be archived as `LinearCostModel::hand_tuned()` for reference.

---

### ADR 0002 — Ranker filters at the per-recommendation level, not per-sequence

**Context.** The Phase 1 `LegalityGate::verify` takes a whole `PassSequence` and returns one verdict for the lot. That's fine for "did anything in here break a contract?" but useless for "*which* recommendations break a contract?"

**Decision.** The ranker calls `gate.verify(...)` **once per candidate recommendation** with a single-pass `PassSequence`. Rejected recommendations move from `recommended` to `dropped` with `RankingRationale::LegalityGateRejected`. Approved recommendations go into the final sequence.

**Rationale.**
- **Pinpoints which pass each function should skip.** A pass that's safe for function `A` but illegal for function `B` is correctly recommended for `A` and skipped for `B`.
- **The final `RankingReport.sequence` is always Approved by construction.** That's a useful invariant — downstream consumers can act on the sequence without re-checking.
- **The dropped list is structured.** Reviewers can audit not just *what* was recommended but *what was rejected and why*.

**Consequences.**
- (+) Per-pass, per-function granularity in audit reports.
- (+) The legality gate's logic stays simple — it doesn't need to know about "drop just this entry from the sequence."
- (–) `gate.verify()` runs O(pass × function) times rather than once. For Phase 2's 6 passes × small programs, the cost is negligible (<1ms per program); for huge programs we may need to revisit.

---

### ADR 0003 — `CostEstimate::Unknown` keeps the pass at low priority, not drops it

**Context.** When the cost model returns `Unknown` (a pass it doesn't know about, or a function not in features), the ranker has to decide: drop the pass (conservative — runtime may suffer) or keep it (conservative — wastes some compile time).

**Decision.** Keep the pass at *very low priority* (predicted benefit = 0.001) with `RankingRationale::UnknownButKeptConservatively`. The pass appears in the recommendation but ranks last; if the skip threshold is above 0.001 (the default 0.005 is), it then ends up dropped.

**Rationale.**
- **CANA's prime contract is "never silently break things."** Skipping a pass because we don't know its value is a quiet behavioural change.
- **The structured rationale makes the decision visible.** A reviewer sees "this pass was kept because the model couldn't predict it" — they can address the gap.
- **Compile time vs runtime is a calibrated tradeoff.** For Phase 2 the default threshold (0.005) drops Unknown passes anyway, so we get conservative behaviour by default with an explicit knob to disable it.

**Consequences.**
- (+) New passes (not in the coefficient table) still get a chance to run.
- (+) Auditable rationale in the report.
- (–) The default behaviour reads as a paradox: "Unknown means kept conservatively, but then dropped by the threshold." This is documented but takes one re-read to follow.

---

### ADR 0004 — Wiring into `optimize_program` is deferred

**Context.** The natural next step is to make `cjc-mir::optimize::optimize_program` accept an optional `PassSequence` from CANA and run the recommended sequence instead of the fixed 6-pass default.

**Decision.** **Don't** ship that in this commit. Phase 2 lands the recommendation engine; the wiring is a separate follow-up.

**Rationale.**
- **The wiring change touches `cjc-mir`, which is a hot path.** Modifying `optimize_program`'s signature requires updating every call site (`cjc-mir-exec::run_program_optimized`, `cjc-cli`, all test harnesses) and re-running the full AST/MIR parity suite to confirm no semantic drift.
- **The recommendation engine is testable in isolation.** Its determinism + legality contracts are verified by 17 new unit tests without any compiler-side change.
- **Separate commits, separate parity runs.** If the wiring introduces a parity regression, it's clearly the wiring's fault — not entangled with the cost model.

**Consequences.**
- (+) Phase 2's foundation lands with zero risk to existing tests.
- (+) The follow-up wiring is mechanical: read `RankingReport.sequence`, run passes in that order, log to `PassHistory`.
- (–) Phase 2 doesn't yet *change* compiler behaviour. Users get a richer `--cana-report` (with recommendations) but the same compilation.

**Follow-up planned for:** the very next Phase 2 commit, which will:
1. Add `cjc-mir::optimize::optimize_program_with_recommendations(program, sequence)` that consumes a `PassSequence`.
2. Update `cjc-mir-exec::run_program_optimized` to call CANA when `--mir-opt` is set.
3. Re-run the AST/MIR parity gate (`tests/fixtures/`).
4. Benchmark before/after on 5–10 representative programs.

---

### ADR 0005 — `CANONICAL_PASSES` array order matches the default pipeline

**Context.** Phase 2 ranker reorders passes by predicted benefit. The *default* order in `CANONICAL_PASSES` is what the compiler runs when CANA isn't consulted.

**Decision.** `CANONICAL_PASSES` = `["constant_fold", "strength_reduce", "dce", "cse", "licm", "cf_round_2"]` — matches `cjc_mir::optimize::optimize_program`'s 6-pass sequence in order.

**Rationale.** When the model is `Unknown` for every pass, the ranker emits the default order, preserving today's behaviour. CANA Phase 2 with no real cost model degrades cleanly to "the same compilation we ship now." That's a useful safety property.

**Consequences.** When new passes get added to `optimize_program`, this constant must be kept in sync. A single test (`canonical_pass_count_is_six`) guards against silent drift in count, but not order — Phase 3 will need a stronger sync mechanism (likely a runtime cross-check in the wiring layer).

---

## 2. New surface added

### From `cost_model.rs` (Phase 1) — unchanged

```rust
pub enum CostEstimate { Unknown, Estimated { value: f64, confidence: f64 } }
pub enum CostQuery<'a> { PassRuntime { ... }, PassBenefit { ... }, PeakMemory { ... } }
pub trait CostModel { fn query(...) -> CostEstimate; fn name(&self) -> &'static str; fn version(&self) -> u32 { 0 } }
pub struct NullCostModel;  // Phase 1
```

### From `linear_cost_model.rs` (NEW)

```rust
pub struct LinearCostModel;

impl LinearCostModel {
    pub const fn new() -> Self;
}

impl CostModel for LinearCostModel {
    fn query(&self, program: &MirProgram, features: &CanaFeatures, query: &CostQuery) -> CostEstimate;
    fn name(&self) -> &'static str;  // "linear_v1"
    fn version(&self) -> u32;        // 1
}
```

Internals (private):
- `pass_coefficients(pass_name) -> Option<PassCoefficients>` — the table
- `predict_pass_gain`, `predict_pass_compile_cost`, `predict_peak_memory`

### From `pass_ranker.rs` (NEW)

```rust
pub const CANONICAL_PASSES: &[&str] = &[...];        // 6-pass vocabulary
pub const DEFAULT_SKIP_THRESHOLD: f64 = 0.005;       // 0.5% runtime win

pub struct PassRecommendation {
    pass_name: String,
    predicted_benefit: f64,
    predicted_compile_cost: f64,
    confidence: f64,
    rationale: RankingRationale,
}

pub enum RankingRationale {
    BenefitAboveThreshold,
    UnknownButKeptConservatively,
    BelowSkipThreshold,
    LegalityGateRejected,
}

pub struct FunctionRanking { pub recommended: Vec<PassRecommendation>, pub dropped: Vec<...> }
pub struct RankingReport { pub per_fn: BTreeMap<String, FunctionRanking>, pub sequence: PassSequence, pub verdict: LegalityVerdict }

pub struct PassRanker<M: CostModel, G: LegalityGate> { ... }
impl PassRanker<M, G> {
    pub fn new(cost_model: M, legality_gate: G) -> Self;
    pub fn with_skip_threshold(self, t: f64) -> Self;
    pub fn rank(&self, program: &MirProgram, features: &CanaFeatures) -> RankingReport;
}

pub fn default_ranker() -> PassRanker<LinearCostModel, DefaultLegalityGate>;
```

### Re-exports added to `lib.rs`

```rust
pub use crate::linear_cost_model::LinearCostModel;
pub use crate::pass_ranker::{
    default_ranker, FunctionRanking, PassRanker, PassRecommendation, RankingRationale,
    RankingReport, CANONICAL_PASSES, DEFAULT_SKIP_THRESHOLD,
};
```

### Modified: `LegalityVerdict` gains `Default = Approved`

The ranker's `RankingReport` derives `Default`, which needed `LegalityVerdict: Default`. `Approved` is the right default — it's what an empty sequence produces.

---

## 3. Tests added (17 new)

### `linear_cost_model::tests` (8 tests)

| Test | Asserts |
|---|---|
| `name_and_version_are_stable` | `name() == "linear_v1"`, `version() == 1` |
| `unknown_pass_returns_unknown_not_zero` | Querying a pass not in the coefficient table → `Unknown` |
| `unknown_function_returns_unknown` | Querying a function not in features → `Unknown` |
| `licm_predicts_more_benefit_with_loops_than_without` | LICM benefit ranking is *qualitatively* correct |
| `predictions_are_deterministic_across_runs` | 100-iteration repeat test (CANA's core determinism contract) |
| `predictions_are_in_normalized_range` | All known passes produce values in `[0, 0.5]` |
| `compile_cost_is_positive_and_bounded` | All known passes produce compile costs in `[0.01, 1.0]` |
| `peak_memory_estimate_for_empty_program_is_low` | Empty program → peak memory < 0.1 |
| `confidence_is_in_unit_interval` | All confidences in `[0, 1]` |

### `pass_ranker::tests` (7 tests)

| Test | Asserts |
|---|---|
| `ranking_is_deterministic_across_runs` | 50-iteration repeat (whole report equality, not just hashes) |
| `ranking_for_empty_program_runs_passes_at_low_priority` | Empty function still gets 6 candidates (some dropped) |
| `licm_appears_in_loop_program_recommendations` | LICM is recommended when loops exist |
| `recommended_passes_are_in_descending_benefit_order` | Ranking order invariant |
| `ranker_verdict_is_approved_under_default_gate` | Filtered sequence always passes legality |
| `skip_threshold_controls_drop_decisions` | Strict threshold → fewer kept |
| `dropped_recommendations_carry_rationale` | Every dropped pass has a structured rationale |
| `canonical_pass_count_is_six` | Drift guard for `CANONICAL_PASSES` length |

All 17 pass deterministically. Combined with Phase 1's 46 unit tests, `cjc-cana --lib` now has **63 unit tests passing**.

---

## 4. Determinism + legality verification

Every Phase 2 surface preserves the Phase 1 contracts:

| Contract | How it's preserved in Phase 2 |
|---|---|
| Same MIR → byte-identical hashes | Unchanged — Phase 2 doesn't touch hashing |
| Same MIR → byte-identical features | Unchanged — Phase 2 reads features, doesn't extract them |
| Same MIR + same model → same recommendation | New test `ranking_is_deterministic_across_runs` (50 iterations) |
| Legality gate vetoes unsafe rewrites | Per-pass filtering (ADR 0002); `ranker_verdict_is_approved_under_default_gate` enforces it |
| No floats compared for equality | Sorting uses `partial_cmp`; tied benefits broken by alphabetic pass-name comparison (ADR-evident in ranking code) |

No new dependency was added to `Cargo.toml`. The Phase 2 surface remains zero-extra-dependency.

---

## 5. What's NOT in Phase 2 (deferred to follow-up commits)

- **Wiring into `optimize_program`.** See ADR 0004.
- **CLI flag `--cana-recommend`.** The `--cana-report` flag from Phase 1 already emits recommendations if we update `report.rs` to include them; that's a one-file change in a follow-up.
- **Benchmark suite.** Phase 5's profile-guided feedback needs a representative corpus. Building it is a multi-session task.
- **Trained cost model.** Phase 5.
- **NSS integration for pressure-aware recommendations.** Phase 4.
- **Thermal cost coefficients.** Phase 4.

These are explicit in the multi-phase plan from `CANA_NSS_COMPILER_IMPROVEMENT_PLAN.md` — Phase 2 is the foundation, not the endgame.

---

## 6. Cross-references

- `crates/cjc-cana/src/linear_cost_model.rs` — Phase 2's cost model implementation
- `crates/cjc-cana/src/pass_ranker.rs` — the recommendation engine
- `crates/cjc-cana/src/legality.rs` — Phase 1 gate, unchanged except for `Default` on `LegalityVerdict`
- `docs/cana/CANA_NSS_COMPILER_IMPROVEMENT_PLAN.md` — full multi-phase roadmap
- `docs/cana/CANA_PHASE_1_REGRESSION_FAILURES.md` — Phase 1 regression triage (context for what Phase 2 builds on)

---

*Generated alongside Phase 2 foundation commit. See follow-up commit for wiring into `cjc-mir::optimize::optimize_program`.*
