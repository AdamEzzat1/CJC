# ADR-0028 Locke Data Skepticism Layer

- **Status:** Accepted (v0.1.10)
- **Date:** 2026-05-27
- **Crate:** `cjc-locke` (new)

## Context

CJC-Lang has a strong **runtime** determinism story (Kahan accumulators, `BTreeMap` discipline, content-addressed reports in `cjc-abng`, lineage in research crates) but lacks a unified surface for **dataset-level** skepticism: missingness, duplicates, schema validation, drift, lineage, belief reports, and causality guardrails. Users currently roll their own per-project — losing determinism and comparability across runs.

ABNG already ships `drift`, `audit`, `merkle`, `stats` modules, but ABNG is **tree-node-oriented** (belief radix graph) — not the column/row-oriented model needed for tabular skepticism. Direct reuse would either (a) bloat ABNG with unrelated functionality, or (b) force Locke to inherit ABNG's node-graph semantics for data that doesn't naturally fit them.

## Decision

Ship Locke as a **dedicated workspace crate `cjc-locke`** with its own type vocabulary, depending only on `cjc-data` and `cjc-repro`. Expose through a CLI shim in `cjc-cli/src/commands/locke.rs` registering `cjcl locke …`. Reserve **error codes E9000–E9099** for Locke findings.

### What's in scope (v0.1.10)

- 9 modules: `id`, `report`, `stats`, `validation`, `drift`, `lineage`, `belief`, `causal`, `api`
- Deterministic 64-bit content fingerprints (SplitMix64-based, domain-salted) for all IDs
- 7 validators emitting structured findings (E9001–E9022)
- Drift compare with PSI / TVD / mean/std/range / missingness shift / small-sample warning
- `LockeImpression` vs `LockeIdea` semantic split + lineage DAG + audit chain
- 8-dimensional `BeliefScore` with explainable per-dimension breakdown
- 5 causal warning kinds + observational-only mode + user-declared claims
- CLI: `validate`, `drift`, `belief`, `lineage`, `causal` with `--json`, `--fail-on`, `--target`, `--observational-only`
- 115 tests (65 unit + 44 integration + 6 CLI), 6 proptest properties, 5 Bolero fuzz targets

### What's deferred (v0.2)

See [[Locke Roadmap]]. Notable items: null mask for non-float columns, exact KS test, auto-instrumented lineage from `cjc-data`, CJC-Lang language-level `locke.validate(df)` builtin.

## Alternatives considered

### A — Embed in `cjc-data`

Pros: zero dependency boundary, immediate access from any DataFrame consumer.
Cons: pollutes the tabular DSL with skepticism types; `cjc-data` already has a public surface that consumers depend on at the type level, and adding `LockeReport` / `BeliefScore` etc. would force every downstream of `cjc-data` to ship with Locke's API. Rejected.

### B — Embed in `cjc-abng`

Pros: reuse `drift.rs`, `audit.rs`, `merkle.rs`.
Cons: ABNG's primitives operate on tree nodes with a Welford density tracker per node; Locke's primitives operate on columns. Forcing the same shape would either degrade ABNG's invariants or twist Locke's data flow. Rejected.

### C — A pure Rust library outside the workspace

Pros: clean isolation.
Cons: loses workspace `Cargo.lock` reproducibility, makes CI gating awkward, can't easily share `cjc-repro` Kahan helpers. Rejected.

### D (chosen) — New workspace crate `cjc-locke`

Pros: matches the precedent of `cjc-quantum`, `cjc-dharht`, `cjc-abng`; clear ownership boundary; shared workspace lockfile; CLI integration via the existing `commands/<name>.rs` pattern.
Cons: one more crate to publish.

## Determinism contract

1. All IDs are content-addressed via a SplitMix64-derived 64-bit fingerprint with domain-salt separation (Finding, Impression, Idea, LineageNode, LineageEdge, AuditEvent, CausalClaim).
2. All maps in emitted output use `BTreeMap`/`BTreeSet`; no `HashMap` iteration anywhere.
3. All float reductions use `cjc_repro::KahanAccumulatorF64`. No FMA, no `f64::mul_add`.
4. Missing-value semantics are explicit: NaN = missing for `Column::Float`; non-float columns emit E9002 (limitation note) rather than silent sentinel-value handling.
5. No wall-clock timestamps in audit events; monotonic `seq` numbers scoped to a `run_label`.

These invariants are tested by 5 dedicated `determinism_tests.rs` cases plus 6 proptest properties.

## Consequences

**Positive**

- Locke can be a CI gate: `cjcl locke drift train.csv prod.csv --fail-on warning` blocks deploys on distribution shift.
- The lineage DAG provides a deterministic basis for future ML-metadata standards (Croissant, etc.).
- The hard split between Impressions and Ideas gives downstream tools a vocabulary for talking about "observed vs derived" without ambiguity.

**Negative**

- One more crate to maintain in the workspace.
- The 8-dimension belief score is an *opinion*; users with different preferences will need v0.2 user-weighted aggregation.
- v0 has known limitations (non-float missingness, no exact KS, no auto-lineage) that need v0.2 work.

## Related

- [[Locke Overview]] — user-facing introduction
- [[Locke Architecture]] — module layout + integration points
- [[Locke Belief Reports]] — score-breakdown rationale
- [[Locke Causality Guardrails]] — why v0 is conservative
- [[Locke Roadmap]] — v0.2+ priorities
