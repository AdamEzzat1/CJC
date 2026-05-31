# Locke Causality Guardrails

> Locke can flag weak causal reasoning, but it does not certify causal truth.

The `causal` module is the most *conservative* part of Locke v0. It exists to push back against the everyday slippage from "X and Y are correlated" into "X causes Y."

## What v0 does

| Warning kind                            | Trigger |
|------------------------------------------|---------|
| `StrongCorrelationNoIntervention`        | \|r\| ≥ 0.7 (default) on a column pair |
| `LikelyConfounder`                       | column Z is correlated ≥ 0.4 with both feature X and target Y |
| `CausalLanguageInLabel`                  | a configured keyword (`causes`, `causal`, `leads_to`, ...) appears in metadata text |
| `ObservationalOnly`                      | user explicitly declared `CausalMode::ObservationalOnly` |
| `ModelExplanationAsCausal`               | caller passes `interpret_model_explanation_as_causal = true` |

Every report carries a **disclaimer** as a top-level field:

> Locke does not infer causal effects. Every warning below is an association-based risk indicator, not a causal claim.

## Flow

```mermaid
flowchart TD
  DF[DataFrame] --> N[numeric columns only]
  N --> P[pairwise Pearson r]
  P --> S{|r| >= 0.7?}
  S -->|yes| W1[StrongCorrelation warning]
  P --> T{target given?}
  T -->|yes| CON[find confounders]
  CON --> W2[LikelyConfounder warnings]
  OBS[observational-only mode] --> W3[ObservationalOnly warning]
  LBL[label text scan] --> W4[CausalLanguageInLabel warnings]
  EXP[caller flag] --> W5[ModelExplanationAsCausal warning]
  W1 & W2 & W3 & W4 & W5 --> R[CausalGuardrailReport]
```

## Confounder detection

A column `Z` is flagged as a *candidate confounder* of `(X, target)` if:

- `|r(Z, target)| ≥ confounder_threshold` (default 0.4) **and**
- `|r(Z, X)| ≥ confounder_threshold`

This is **only association**. Locke explicitly does not claim Z is a confounder, only that **it could be** — every emitted `ConfounderHint` carries the assumption "association does not imply confounding without a causal model."

## Why "observational-only" is a separate mode

Users running Locke on cohort studies, observational EHR data, A/A analyses without randomisation, or any non-experimental dataset get a different default: every strong correlation also gets an `ObservationalOnly` standing warning, raising visibility that no intervention was performed.

```rust
let mut cfg = CausalConfig::default();
cfg.mode = CausalMode::ObservationalOnly;
let report = causal_guardrail(&df, Some("y"), &cfg, None, false);
```

## Causal-language scan

Locke scans an optional `label_text` for keywords listed in `CausalConfig::causal_keywords`:

```text
"causes", "causal", "leads_to", "due_to", "because_of", "drives"
```

A match triggers `CausalLanguageInLabel` — the offending substring is reported in the `b` field of the warning. This is intentionally a heuristic: false positives are acceptable because the cost of a missed causal claim is much higher than the cost of a notice.

## User-declared causal DAG (v0.2)

The `CausalDag` type lets the user declare a partial causal graph upfront:

```rust
use cjc_locke::{CausalConfig, CausalDag, CausalMode};

let mut dag = CausalDag::new();
dag.add_edge("treatment_arm", "5y_survival").unwrap();
dag.add_edge("age", "5y_survival").unwrap();
// dag.add_edge("5y_survival", "age").unwrap_err(); // would be a cycle

let mut cfg = CausalConfig::default();
cfg.mode = CausalMode::ObservationalOnly;
cfg.assumed_dag = dag;
```

**What this does**: when `audit_correlations` finds a strong correlation `|r| ≥ threshold` between two columns that are already related in the DAG (directly *or* transitively), the resulting `StrongCorrelationNoIntervention` warning is **annotated** with "acknowledged by causal DAG" rather than firing as a fresh warning. The warning still appears — Locke never *hides* findings — but its message and assumptions list make clear the user has already declared the pathway.

**What this does NOT do**: it does not turn correlation into causation. It does not check the user's DAG against the data. It does not run d-separation, do-calculus, or any identifiability test. It's a **registry of stated assumptions**, not a verifier of them.

Cycle introduction is rejected at `add_edge` time:

```rust
dag.add_edge("a", "b")?;
dag.add_edge("b", "c")?;
dag.add_edge("c", "a")  // Err(CausalDagError::CycleIntroduced { ... })
```

Self-loops are also rejected (`CausalDagError::SelfLoop`).

## User-declared claims

A caller can pre-register their own `CausalClaim` so Locke records the claim *as a claim*, with a deterministic ID and rationale:

```rust
let claim = CausalClaim::new(
    /* cause= */ "treatment_arm",
    /* effect= */ "5y_survival",
    CausalDirection::AtoB,
    /* rationale= */ "randomised at enrollment (n=4200)",
    /* user_confidence= */ 0.8,
);
```

Locke does not *verify* the claim — it just makes the assumption visible in the audit trail.

## Limitations (loud)

- No do-calculus, no DAG-based identifiability check, no propensity scoring, no instrumental-variable detection.
- Confounder detection sees **only associations** — variables that don't appear in the dataframe cannot be flagged.
- Pearson correlation is linear; non-linear dependencies are invisible.
- Causal language detection is keyword-based — paraphrased causal claims slip through.

v0.2 will explore a DAG-based assumption registry; see [[Locke Roadmap]].

## Tests

- `crates/cjc-locke/src/causal.rs` — 8 unit tests covering each warning kind plus disclaimer presence
- `tests/locke/causal_tests.rs` — 5 integration tests over realistic patterns
- `tests/locke/locke_fuzz.rs::fuzz_causal_guardrail_arbitrary_label_text` — Bolero fuzz over UTF-8 labels
