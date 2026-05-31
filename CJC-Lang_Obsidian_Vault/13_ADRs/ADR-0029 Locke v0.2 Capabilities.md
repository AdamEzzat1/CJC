# ADR-0029 Locke v0.2 Capabilities

- **Status:** Accepted (v0.2, 2026-05-27)
- **Crate:** `cjc-locke` (extended from ADR-0028)

## Context

[[ADR-0028 Locke Data Skepticism Layer]] shipped a deterministic, evidence-aware analytics layer with seven explicit deferrals (the "v0.2 priorities"). v0.2 closes those gaps without changing the v0.1 type vocabulary or breaking any v0.1 caller.

## Decisions

### 1. Null mask for non-float columns

- New types `NullMask` (`BTreeSet<usize>` of null row indices) and `NullMaskMap` (`BTreeMap<String, NullMask>`).
- `ValidateOptions::null_masks` lets callers mark missing rows in any column.
- `detect_missingness(df, cfg, null_masks)` gains the new parameter.
- For `Column::Float`, NaN positions are **unioned** with mask positions.
- For other types, the mask is the *only* source of missingness.
- Out-of-bounds indices flagged as **E9006** (Warning) and skipped.
- E9002 stays as a downgraded Info-level "no mask supplied" acknowledgement.

### 2. Exact KS D-statistic (E9039) replaces PSI as default numeric drift

- New `cjc_locke::stats::ks_d_statistic(xs, ys) -> Option<f64>`.
- Merge-walk over two sorted samples; NaN-filtered; `f64::total_cmp` for determinism.
- Default thresholds: `ks_d_warn=0.10`, `ks_d_error=0.20`.
- PSI helper retained as an internal function for the `locke_psi` builtin; v0.1's E9033 finding is no longer emitted by default.

### 3. Auto-instrumented lineage (`TracedDataFrame`)

- New wrapper in `cjc_locke::traced`.
- `observe(builder, source, df)` registers an `Impression` and returns a wrapper.
- `filter`, `select`, `with_column`, `rename`, `transform`, `concat` emit `LockeIdea` + edges automatically.
- Wrapper holds `&mut LineageBuilder`, so lifetime is scoped to the pipeline.
- Determinism: each transformation's parameters are stored in a sorted `BTreeMap`, so the resulting `Idea` id is content-addressed identically across runs.

### 4. CJC-Lang language-level builtins (satellite dispatch)

- New `cjc-locke/src/dispatch.rs` with `dispatch_locke(name, args) -> Result<Option<Value>, String>`.
- 6 builtins exposed: `locke_missing_count`, `locke_missing_rate`, `locke_ks_d`, `locke_psi`, `locke_sample_score`, `locke_belief_overall`.
- Wired into `cjc-eval` and `cjc-mir-exec` following the `cjc_ad::dispatch_grad_graph` / `cjc_abng::dispatch_abng` / `cjc_quantum::dispatch_quantum` precedent.
- Adds a `CallDispatch::Locke` arm to the MIR executor's call cache.
- Builtins take and return primitive `Value` variants (`Int`, `Float`, `Array`) — no `Value::DataFrame` introduced (deferred to v0.3).

### 5. DAG-based causal assumption registry

- New `CausalDag` type with directed edges + cycle detection at construction.
- `CausalConfig::assumed_dag` field; default is an empty DAG (no change in behavior).
- `audit_correlations` annotates strong-correlation warnings between declared (or reachable) pairs with "acknowledged by causal DAG" — warning is **annotated**, not removed. Locke never hides findings.
- Cycle / self-loop rejected at `add_edge` time with `CausalDagError::CycleIntroduced` / `CausalDagError::SelfLoop`.

### 6. User-tunable belief weights

- New `BeliefWeights` struct, 8 fields, default = `1.0` everywhere.
- New `BeliefScore::from_dimensions_weighted(..., &weights)`.
- Negative / NaN weights clamped to 0; all-zero weights → `overall = 0.0`.
- v0.1's `from_dimensions(...)` unchanged — bit-equivalent to v0.2 with default weights.

### 7. insta CLI snapshot tests

- Added `insta = "1"` as a workspace `[dev-dependencies]`.
- New `tests/locke/snapshot_tests.rs` with 4 snapshots: `validate_default`, `belief_explain`, `drift_default`, `lineage_minimal`.
- Content-addressed fingerprints redacted to `[FP]` so snapshots survive unrelated evidence-format tweaks.

## Test counts

| Bucket                          | v0.1 | v0.2 | Δ   |
|---------------------------------|------|------|-----|
| `cargo test -p cjc-locke --lib` | 65   | 101  | +36 |
| `cargo test --test locke`       | 44   | 63   | +19 |
| `cargo test -p cjc-cli locke`   | 6    | 6    | 0   |
| **Total**                       | **115** | **170** | **+55** |

Adjacent regression: `cjc-data` 258/258, `cjc-eval` 28/28, `cjc-mir-exec` 17/17 — no regression.

## Consequences

**Positive**

- Non-float missingness is now first-class; close one of the loudest v0.1 limitations.
- KS D-statistic gives a proper distributional drift signal (not just a bin-grid proxy).
- `TracedDataFrame` removes the manual lineage-building boilerplate.
- Locke is now callable from `.cjcl` source — opens the door to user-defined analytics pipelines.
- Belief weights let downstream tools (CI gates, dashboards) tune the score to their priorities.
- insta snapshots pin the CLI's text emit, catching accidental reformatting regressions.

**Negative**

- `detect_missingness` signature changed (added `null_masks` parameter) — any direct caller of the function must update. The high-level `validate()` facade is source-compatible because `ValidateOptions::null_masks` defaults to empty.
- E9039 replaces E9033 in the default path; consumers that parsed v0.1's E9033 messages need to update to E9039.
- `cjc-locke` now depends on `cjc-runtime` (for the `Value` enum); the dep graph grows but no cycle is introduced because `cjc-runtime` does not depend on `cjc-locke`.

## Related

- [[ADR-0028 Locke Data Skepticism Layer]] (v0.1 base)
- [[Locke Overview]], [[Locke Data Skepticism]], [[Locke Induction Risk]], [[Locke Causality Guardrails]], [[Locke Lineage and Provenance]], [[Locke Belief Reports]], [[Locke Roadmap]]
