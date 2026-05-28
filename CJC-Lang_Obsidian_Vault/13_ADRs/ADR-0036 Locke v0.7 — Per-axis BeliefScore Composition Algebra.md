# ADR-0036 Locke v0.7 — Per-axis BeliefScore Composition Algebra

- **Status:** Accepted (v0.7 part 1, 2026-05-28)
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

## Out of scope for v0.7 part 1

Deferred to part 2:

- **Migrate `api::belief_report_from_locke_with_model` to use `compose` directly.** Currently it computes per-axis scores in-line via `penalty_from_findings_with_model`. The migration is mechanical but touches multiple call sites — out of scope for the algebra-formalisation ADR.
- **Migrate `gate::diff_reports` to use `le_componentwise`** for the diff partial order. Same rationale.
- **Continuous-domain semantics for `transform_factor`** — the v0.5 blog mentioned parameterising rule selectivity by a continuous parameter (e.g. `filter(predicate)` with predicate selectivity `s ∈ [0, 1]`). Not in this batch.
- **A *join* operator alongside `meet`** — Max is *the* join under the dual semilattice but it's not currently the operational composition for any use case; would land when alternative-view-merging becomes a primary workflow.

## Open questions

- Should `BeliefAxisRules` be carried inside `BeliefReport` so the report itself records which composition algebra was used to produce it? Currently the rules are stateless and applied at compose-time; recording would tighten the audit story but adds a field. v0.7.1 question.
- Should the algebra also include a `Median` or `Trimmed Mean` rule for robustness against outlier leaves? Both are non-associative; would need a `compose_many_*` variant rather than pairwise.
- The `is_associative()` and `is_idempotent_everywhere()` methods could become a proper trait if the rules grew into a plugin system. Currently three call sites and an enum is fine.

Related: [[ADR-0028 Locke Data Skepticism Layer]], [[ADR-0032 Locke v0.5 Capabilities]], [[ADR-0033 Locke v0.6 Categorical and Drift Capabilities]], [[ADR-0034 Locke v0.6 Batch 2 — PII, Drift, Label-Encoding, Confidence Summary, Seasonality]], [[ADR-0035 Locke v0.6.3 — Distribution Shape, Multi-class Leakage, CategoricalAdaptive]], [[Locke Belief Reports]], [[Locke Roadmap]].
