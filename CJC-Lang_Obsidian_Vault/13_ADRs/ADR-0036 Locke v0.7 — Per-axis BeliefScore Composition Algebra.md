# ADR-0036 Locke v0.7 — Per-axis BeliefScore Composition Algebra

- **Status:** Accepted (v0.7 part 1, 2026-05-28; v0.7 part 2 shipped 2026-05-29)
- **Crate:** `cjc-locke` (extended from ADRs 0028–0035)
- **Companion docs:** [[Locke Belief Reports]], [[Locke Roadmap]] §v0.7

## Context

The v0.5 blog post (`adamezzat1.github.io/blog/posts/cjc-locke-v0.1-released/`) and subsequent ADRs 0033 / 0034 documented `(BeliefScore, meet, ⊤)` as a commutative idempotent monoid with identity — a **meet-semilattice** under component-wise `min`. Five laws were stated:

| Law | Statement |
|---|---|
| Identity | `meet(b, ⊤) = b` for `⊤ = [1, …, 1]` |
| Idempotence | `meet(b, b) = b` |
| Commutativity | `meet(b, c) = meet(c, b)` |
| Associativity | `meet(meet(a, b), c) = meet(a, meet(b, c))` |
| Monotonicity | `meet(b, c) ≤ b` (component-wise partial order) |

In the code, the composition operator was implicit — applied ad-hoc per call site (`api::belief_report_from_locke_with_model`, `cjcl locke gate` diff scoring, the per-leaf belief experiment in `tests/abng/per_leaf_belief.rs`). The formal model had no in-code embodiment, the laws had no property tests, and adding a per-axis composition variant (needed for the v0.7 per-leaf experiment) would have meant another ad-hoc pass.

This ADR ships the algebra as a first-class module with:

1. A formal `CompositionRule` enum with four variants (`Min` / `Max` / `GeometricMean` / `ArithmeticMean`) and explicit per-variant law guarantees.
2. A `BeliefAxisRules` struct selecting one rule per `BeliefScore` axis.
3. `compose(a, b, rules)`, `compose_many(scores, rules)`, `compose_many_arithmetic(scores)`, and `compose_weighted(scores, weights)` API.
4. Order relations `le_componentwise` and `eq_componentwise` for the partial order under which `Min` is a meet.
5. Identity elements `top()` (for `Min`) and `bottom()` (for `Max`).
6. Proptest-locked laws verifying all 5 meet-semilattice properties on `BeliefAxisRules::default()`, plus a boundedness law (`[0, 1]^8` is closed under all four rules).
7. Bolero structural fuzz on the compose API: arbitrary 16-float input + every rule combination produces axis values in `[0, 1]` and never panics.

## Decisions

### 1. New module `crates/cjc-locke/src/algebra.rs` (~410 LOC)

Public API:

```rust
pub enum CompositionRule { Min, Max, GeometricMean, ArithmeticMean }

impl CompositionRule {
    pub fn apply(self, a: f64, b: f64) -> f64;                 // clamps inputs to [0,1]
    pub fn is_idempotent_everywhere(self) -> bool;             // for law-scoping
    pub fn is_monotonic_down(self) -> bool;
    pub fn is_associative(self) -> bool;
}

pub struct BeliefAxisRules {
    pub schema, missingness, drift, leakage,
        lineage, sample, duplication, constraint: CompositionRule,
}

impl Default for BeliefAxisRules { /* all-Min */ }
impl BeliefAxisRules {
    pub fn all_min() -> Self;
    pub fn all_max() -> Self;
    pub fn all_arithmetic_mean() -> Self;
    pub fn is_idempotent_everywhere(&self) -> bool;
    pub fn is_associative(&self) -> bool;
}

pub fn top() -> BeliefScore;            // identity for Min
pub fn bottom() -> BeliefScore;         // identity for Max

pub fn compose(a, b, rules) -> BeliefScore;
pub fn compose_many(scores, rules) -> Option<BeliefScore>;       // chained pairwise
pub fn compose_many_arithmetic(scores) -> Option<BeliefScore>;   // single-pass mean
pub fn compose_weighted(scores, weights) -> Option<BeliefScore>; // weighted by e.g. n_rows

pub fn le_componentwise(a, b, eps) -> bool;   // partial order
pub fn eq_componentwise(a, b, eps) -> bool;
```

### 2. Per-rule law guarantees

| Rule | Identity | Idempotent | Commutative | Associative | Monotonic-down |
|---|:---:|:---:|:---:|:---:|:---:|
| `Min` | `⊤ = 1` | ✓ | ✓ | ✓ | ✓ |
| `Max` | `⊥ = 0` | ✓ | ✓ | ✓ | dual ↑ |
| `GeometricMean` | `1` (vacuous, since √(b·1) = √b ≠ b unless b ∈ {0, 1}) | only at {0, 1} | ✓ | ✓ | ✓ (for inputs ≤ 1) |
| `ArithmeticMean` | none (mean with `1` is not identity) | ✓ (since (x+x)/2 = x) | ✓ | ✗ (chaining diverges) | ✗ |

The meet-semilattice algebra requires the union (all 5 laws). Only the all-`Min` rule satisfies it. The other rules are first-class combinators with documented-weaker guarantees — useful for use cases (per-leaf aggregation, average-across-views) where `Min` is the wrong semantics.

### 3. Use-case map

| Need | Rule | Why |
|---|---|---|
| Compose parent and child views in a lineage DAG | `Min` (default) | Belief never improves under derivation. The v0.5 blog's "the rightmost node's vector is component-wise ≤ the leftmost" claim. |
| Build a per-leaf belief summary across many leaves | `ArithmeticMean` via `compose_many_arithmetic` | True "average per-leaf belief"; pairwise chaining gives the wrong answer because mean is non-associative. |
| Build a dataset-level belief from leaf beliefs weighted by leaf row count | `compose_weighted` | Per-leaf belief contributes proportionally to its sample size. |
| Diagnostic: "best-case across alternative views" | `Max` (or `all_max()`) | Dual semilattice — shows the upper envelope of belief; not the operational decision but useful for triage. |
| Independent evidence units multiplying down | `GeometricMean` | When each axis represents a probabilistic factor. |

### 4. Wiring

The existing `api::belief_report_from_locke_with_model` and `gate::diff_reports` code paths are **not** changed in this batch — they were already computing axis scores correctly via the existing one-shot path, just without referencing the algebra explicitly. v0.7 part 2 (deferred) will migrate them to use `compose` directly, after the diabetes-130 experiment confirms the per-leaf composition story holds at scale.

The per-leaf belief experiment in `tests/abng/per_leaf_belief.rs` and the new `tests/abng/per_leaf_belief_diabetes130.rs` use the algebra directly via `compose_many_arithmetic` and `compose_weighted` to aggregate per-leaf scores.

### 5. Test infrastructure

| Layer | Location | New count |
|---|---|---|
| Unit (in-module) | `algebra.rs::tests` | 21 |
| Integration | `tests/locke/algebra_tests.rs` (7 cases including real validate() pipelines) | 7 |
| Property (proptest) | `tests/locke/locke_proptest.rs` | 6 (5 meet-semilattice laws + boundedness) |
| Bolero structural fuzz | `tests/locke/locke_fuzz.rs` | 1 (all rules × arbitrary 16-float input) |

Final totals after v0.7 part 1: **cjc-locke 268 lib** (was 247 v0.6.3, +21) + **tests/locke 176** (was 161, +15) + **cjc-cli 154** (no regressions). Workspace builds clean.

## Consequences

1. **The epistemic-layer claim now has formal-model code backing.** The blog post and ADRs documented the meet-semilattice algebra; this ADR ships it. Proptest verifies the laws hold on arbitrary `BeliefScore` inputs over hundreds of cases per property.
2. **Per-axis composition is now expressible**, which the v0.7 per-leaf and weighted-aggregation experiments need. Previously the only composition was the implicit `Min` baked into ad-hoc call sites.
3. **The algebra is the public surface** — callers consume `compose` / `compose_many` / `compose_weighted` rather than reimplementing the math. The `gate::diff_reports` migration to this API is the v0.7 part 2 follow-up.
4. **Determinism is preserved** — every rule's `apply()` is pure floating-point arithmetic with deterministic clamps; Bolero fuzz confirms no panic and no out-of-range axis values across arbitrary inputs.

## v0.7 part 2 (shipped 2026-05-29)

Closes the two consumer-migration items deferred by part 1.

### What shipped

1. **`api::belief_report_from_locke_with_model` migrated to `algebra::compose_many`.** The final `BeliefScore` is now built from 8 per-axis partials (each carrying one axis's computed value with the other 7 set to the meet identity `⊤ = 1.0`) reduced under `BeliefAxisRules::default()` (all-`Min`). The migration is *byte-identical* to the pre-migration direct `BeliefScore::from_dimensions(...)` construction because under all-Min, composing partials with seven 1.0 axes reduces axis-wise to `min(v, 1, 1, ..., 1) = v` — and the final compose step calls `from_dimensions` on the same 8-tuple in the same order as the pre-migration path.

   - The pre-migration inline implementation is preserved as `#[doc(hidden)] pub fn __belief_report_from_locke_inline_for_regression_test(...)` exclusively as a byte-identity reference oracle for the regression proptest. It will be removed once the migration has shipped for at least one stable release without regression.
   - Per-axis derivation (`belief_axis_scores_from_report`) and surrounding report assembly (`finish_belief_report`) are extracted as private helpers, shared between the migrated path and the oracle — so the proptest exercises only the construction-mode divergence (compose vs from_dimensions), not the per-axis derivation logic.

2. **`gate::diff_reports` extended to surface the meet-semilattice partial order.** A new field `belief_partial_order: BeliefPartialOrder` is added to `ReportDiff`. It uses `algebra::le_componentwise` (with `DEFAULT_BELIEF_COMPARISON_EPS = 1e-12`) to classify the relationship between the reference and current reports' belief scores under the v0.7 part 1 partial order. A `BeliefDirection` enum classifies the result into `Equal` / `MonotonicDecrease` / `MonotonicIncrease` / `Incomparable`. The CLI emit (`emit_diff_text`) gains one stable line:

   ```
   belief: direction=monotonic_decrease ref_overall=0.951 cur_overall=0.847
   ```

   - The five pre-migration fields on `ReportDiff` (appeared, disappeared, unchanged, ref_run_id, cur_run_id) are unchanged. The new field is additive; existing tests against the finding-set diff continue to pass without modification.
   - `diff_reports` uses `BeliefPenalty::default()` for the belief computation to keep its signature unchanged. Callers needing a custom penalty model derive belief scores separately and call `BeliefPartialOrder::from_scores(reference, current, eps)` directly.

### New public surface

```rust
pub const DEFAULT_BELIEF_COMPARISON_EPS: f64 = 1e-12;

pub struct BeliefPartialOrder {
    pub reference_belief: BeliefScore,
    pub current_belief: BeliefScore,
    pub current_le_reference: bool,
    pub reference_le_current: bool,
}

impl BeliefPartialOrder {
    pub fn from_scores(reference, current, eps) -> Self;
    pub fn direction(&self) -> BeliefDirection;
    pub fn is_monotonic_decrease(&self) -> bool;
    pub fn is_monotonic_increase(&self) -> bool;
    pub fn is_equal(&self) -> bool;
    pub fn is_incomparable(&self) -> bool;
}

pub enum BeliefDirection {
    Equal,
    MonotonicDecrease,
    MonotonicIncrease,
    Incomparable,
}

impl BeliefDirection { pub fn label(self) -> &'static str; }

impl ReportDiff {
    pub fn belief_direction(&self) -> BeliefDirection;
    // existing methods unchanged
}
```

`BeliefPartialOrder` / `BeliefDirection` / `DEFAULT_BELIEF_COMPARISON_EPS` are re-exported from the crate root for ergonomic CLI / consumer access.

### Test infrastructure (v0.7 part 2)

| Layer | Location | New count |
|---|---|---|
| Unit (in-module) | `gate.rs::tests` — `BeliefPartialOrder` semantics + `emit_diff_text` belief-line | 7 |
| Property (proptest) | `tests/locke/locke_proptest.rs` — algebra-level byte-identity + end-to-end migration byte-identity | 2 |

The algebra-level proptest (`algebra_path_is_byte_identical_to_direct_from_dimensions`) generates arbitrary 8-tuples in `[0, 1]^8` and asserts bit-equal `to_bits()` on every axis and `overall` between `compose_many(per_axis_partials, all_min)` and `from_dimensions(tuple)`. This is the algebraic invariant that makes the migration safe — locked at the f64 bit-pattern level, not just numerically close.

The end-to-end proptest (`belief_report_migrated_path_is_byte_identical_to_inline_oracle`) feeds arbitrary float-column DataFrames through `validate(...)` and asserts bit-equal `BeliefReport` between the migrated path and the preserved inline oracle.

### Net delta after v0.7 part 2

- cjc-locke `--lib`: 284 → **291** (+7 unit tests in `gate.rs::tests`)
- tests/locke: 194 → **196** (+2 proptest properties × 256 cases each)
- ABNG suite: 629 (unchanged — no ABNG surface changed)
- Workspace builds clean.

## Out of scope (still deferred)

- **Continuous-domain semantics for `transform_factor`** — the v0.5 blog mentioned parameterising rule selectivity by a continuous parameter (e.g. `filter(predicate)` with predicate selectivity `s ∈ [0, 1]`). Not in this batch.
- **A *join* operator alongside `meet`** — Max is *the* join under the dual semilattice but it's not currently the operational composition for any use case; would land when alternative-view-merging becomes a primary workflow.
- **Full diabetes-130 per-leaf run.** Extending `tests/abng/per_leaf_belief.rs` to the real dataset is independent of the algebra migration and tracked separately under [[Locke Roadmap]] §v0.7 part 2.

## Out of scope for v0.7 part 1 (historical, kept for traceability)

The two items below were deferred from part 1 and *shipped in part 2 above*:

- ~~**Migrate `api::belief_report_from_locke_with_model` to use `compose` directly.**~~ ✓ Shipped 2026-05-29.
- ~~**Migrate `gate::diff_reports` to use `le_componentwise`** for the diff partial order.~~ ✓ Shipped 2026-05-29.

## Open questions

- Should `BeliefAxisRules` be carried inside `BeliefReport` so the report itself records which composition algebra was used to produce it? Currently the rules are stateless and applied at compose-time; recording would tighten the audit story but adds a field. v0.7.1 question.
- Should the algebra also include a `Median` or `Trimmed Mean` rule for robustness against outlier leaves? Both are non-associative; would need a `compose_many_*` variant rather than pairwise.
- The `is_associative()` and `is_idempotent_everywhere()` methods could become a proper trait if the rules grew into a plugin system. Currently three call sites and an enum is fine.

Related: [[ADR-0028 Locke Data Skepticism Layer]], [[ADR-0032 Locke v0.5 Capabilities]], [[ADR-0033 Locke v0.6 Categorical and Drift Capabilities]], [[ADR-0034 Locke v0.6 Batch 2 — PII, Drift, Label-Encoding, Confidence Summary, Seasonality]], [[ADR-0035 Locke v0.6.3 — Distribution Shape, Multi-class Leakage, CategoricalAdaptive]], [[Locke Belief Reports]], [[Locke Roadmap]].
