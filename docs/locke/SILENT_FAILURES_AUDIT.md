# Locke "Silently Misses What Feels Obvious" Audit

**Date:** 2026-05-28
**Scope:** find patterns where Locke's default behaviour reports "no concern" while a human reader can immediately see one.

Motivated by the Phase 0.10 §4.D Part 1 finding — `weight` is 96.9% `?` but Locke's `missingness_score = 1.0000` ("perfect"). The user-supplied `NullMaskMap` workaround should not be a precondition for catching the diabetes-130 missingness pattern.

## Findings

### Finding 1 — `detect_missingness` treats only `f64::NAN` as missing (the canonical case)

**Where:** `crates/cjc-locke/src/validation.rs:detect_missingness` (line ~203).
**What:** For `Column::Str`, `Column::Int`, `Column::Bool`, `Column::Categorical`, `Column::DateTime` — Locke requires a caller-supplied `NullMask`. If the caller doesn't supply one, the column emits only an info-level E9002 "no null mask supplied" diagnostic. **No actual missingness is counted.**
**Why this fails on the wild:** medical, financial, and scientific CSVs commonly store missingness as the literal strings `?` (UCI convention), `NA` (R convention), `NULL` (SQL convention), `-`, `N/A`, empty string, etc. Locke is blind to all of them.
**Fix:** add `ValidationConfig.auto_detect_sentinels: bool` (default true) + `additional_sentinels: Vec<String>` (default empty); pre-scan `Str` columns for built-in sentinels; emit a NEW **E9008** info finding for every column where a sentinel was detected, and union the auto-mask with any user-supplied mask before `detect_missingness` runs.

### Finding 2 — `detect_conditional_missingness` has the same Float-only bias

**Where:** `crates/cjc-locke/src/validation.rs:detect_conditional_missingness` (line ~1090).
**What:** The pairwise NaN-implication check (E9070) at line 1090 has `if let Column::Float(v) = col` — only float NaN positions count. String sentinels are invisible, *even if the caller did supply a `NullMaskMap`* (the function doesn't consult the mask).
**Why this fails:** the analogous pattern to Finding 1 — exists in a second detector that should consume the same missing-set.
**Fix:** make `detect_conditional_missingness` accept the (auto + user) `NullMaskMap`, fold its rows into the per-column missing-set alongside `Float`-NaN positions.

### Finding 3 — E9064 missing (per-level deterministic leakage)

**Where:** `crates/cjc-locke/src/leakage.rs` — no per-level conditional-probability detector exists.
**What:** E9063 (multi-class leakage) uses per-feature ROC-AUC, which misses leakage that hides at the *level* of a column. E.g., `discharge_disposition_id = 11` deterministically predicts `readmitted = NO` on diabetes-130 because dead patients can't be readmitted. The column-wide rank statistic doesn't catch this because death codes are interspersed numerically with non-death codes; the column AUC stays under the 0.85 warning threshold.
**Fix:** new **E9064** Error detector: for each `(column, level, target_class)` triple, compute `P(target=class | column=level)`. Flag when `P(class|level) ≥ 0.99` with ≥ `min_support` rows AND unconditional `P(class) < 0.99` (i.e., the level adds information beyond the base rate).
**Validated by:** the diabetes-130 dataset's discharge codes 11/13/14/19/20/21.

### Finding 4 — `leakage_score` and `drift_score` default to 1.0 when no signal is supplied

**Where:** `crates/cjc-locke/src/api.rs:belief_report_from_locke_with_model` (lines 207–209).
**What:** When `validate()` runs without a comparison DataFrame (no drift signal) or without a target column (no leakage signal), these axes are set to `1.0` ("perfect"). The score's `assumptions` field disclaims this, but the *number itself* is misleading — the BeliefScore inflates the overall score with axes that weren't checked.
**Decision:** *not fixed this round.* Changing the default would propagate to many tests and to `BeliefReport` API consumers. Document the issue in the BeliefReport docs, leave the score behaviour as-is, and consider adding a `BeliefScore::Unknown(axis)` variant in v0.7+. Recorded as a roadmap item.

### Finding 5 — `multiclass_max_classes` default = 20 may silently skip multi-class targets

**Where:** `crates/cjc-locke/src/leakage.rs:LeakageConfig::default()` line 50–57.
**What:** `multiclass_max_classes: 20` skips columns with more than 20 distinct values to avoid treating continuous data as multi-class. But some real multi-class targets (e.g., ICD-9 chapters, postal-code zones) have 30–100 classes. The detector silently skips them.
**Decision:** *not fixed this round.* The constant is already configurable via `LeakageConfig`. Document the limitation in the leakage module rustdoc; consider raising to 50 in a future release.

## Scope decision for v0.6.4

**In scope this session:** Findings 1, 2, 3 — all directly traced to the §4.B / §4.D / §4.E Part 1 work in the Phase 0.10 blog post.

**Deferred:** Findings 4, 5 — recorded for future versions (v0.7+).

**Out of scope (per user instruction, "≥ 2 sessions of focused work each"):**
- Text drift (vocabulary KS, token-entropy drift, language distribution)
- Ontology / taxonomy consistency (hyphen/underscore variants, hierarchy fragmentation)
- Per-value category lineage (raw → normalized → grouped → encoded → embedding chain)
- Governance workflows (suppression files, owner annotations, required-finding policies)
- Per-axis BeliefScore composition rules (formalising the v0.2 plan with property tests)

## Test plan for in-scope work

| Item | Unit | Integration | Proptest | Bolero fuzz |
|---|---|---|---|---|
| F1 — auto-sentinel detection | bottom of validation.rs | tests/locke/sentinel_tests.rs (new) | tests/locke/locke_proptest.rs | tests/locke/locke_fuzz.rs |
| F2 — conditional-missingness NullMaskMap consumption | bottom of validation.rs | extend the existing tests in tests/locke/ | extend proptest file | extend fuzz file |
| F3 — E9064 detector | bottom of leakage.rs | extend tests/locke/multiclass_leakage_tests.rs (or per_level_leakage_tests.rs) | extend proptest file | extend fuzz file |

## Code references

- ValidationConfig: [`crates/cjc-locke/src/validation.rs:30`](../../crates/cjc-locke/src/validation.rs#L30)
- detect_missingness: [`crates/cjc-locke/src/validation.rs:203`](../../crates/cjc-locke/src/validation.rs#L203)
- detect_conditional_missingness: [`crates/cjc-locke/src/validation.rs:1077`](../../crates/cjc-locke/src/validation.rs#L1077)
- LeakageConfig: [`crates/cjc-locke/src/leakage.rs:30`](../../crates/cjc-locke/src/leakage.rs#L30)
- validate_dataframe: [`crates/cjc-locke/src/validation.rs:1682`](../../crates/cjc-locke/src/validation.rs#L1682)
