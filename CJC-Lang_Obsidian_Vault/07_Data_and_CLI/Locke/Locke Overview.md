# Locke Overview

> Evidence-aware analytics and data skepticism for CJC-Lang.

## What Locke is

Locke is the data-skepticism layer of CJC-Lang. It separates **observed facts** from **derived claims**, validates data quality, surfaces induction risks (train→test drift, leakage hints), preserves lineage, and produces **confidence-aware reports** with an explainable score breakdown.

All Locke output is **deterministic**: repeated runs over the same inputs produce byte-identical findings, IDs, and reports — making Locke usable as a CI gate.

## What Locke v0 is NOT

Locke v0 deliberately does *not* claim to:

- prove causality
- guarantee data truth
- solve dataset bias
- fully prevent leakage
- certify ML models

Instead, Locke v0:

- **detects** common data risks
- **flags** assumptions
- **tracks** evidence
- **makes** inference boundaries explicit
- **produces** reproducible belief reports

## Core types

| Type                       | Purpose                                                  |
|---------------------------|----------------------------------------------------------|
| `LockeReport`             | top-level container of findings + summaries              |
| `ValidationFinding`       | one structured finding (id, severity, evidence, etc.)    |
| `FindingSeverity`         | `Info < Notice < Warning < Error`                        |
| `BeliefScore`             | per-dimension breakdown (schema, missing, drift, ...)    |
| `BeliefReport`            | score + assumptions + recommended next steps             |
| `InductionRiskReport`     | train/test drift findings                                |
| `CausalGuardrailReport`   | correlation→causation warnings, confounder hints         |
| `LockeImpression`         | a *raw* observed fact (column, schema, row)              |
| `LockeIdea`               | a *derived* value with parent lineage references         |
| `LineageGraph`            | deterministic DAG over impressions + ideas               |

## Quick start

```rust
use cjc_locke::api::{validate, ValidateOptions};
use cjc_locke::validation::ValidationConfig;

let opts = ValidateOptions {
    dataset_label: "train.csv".into(),
    config: ValidationConfig::default(),
    ..Default::default()
};
let report = validate(&df, &opts);
println!("worst severity: {}", report.worst_severity());
```

## CLI

```bash
cjcl locke validate data.csv
cjcl locke drift train.csv test.csv
cjcl locke belief data.csv
cjcl locke lineage data.csv
cjcl locke causal data.csv --target y --observational-only
```

See [[Locke CLI]] for full flag reference.

## Documents in this folder

- [[Locke Architecture]]
- [[Locke Data Skepticism]]
- [[Locke Induction Risk]]
- [[Locke Causality Guardrails]]
- [[Locke Lineage and Provenance]]
- [[Locke Impressions vs Ideas]]
- [[Locke Belief Reports]]
- [[Locke Testing and Verification]]
- [[Locke CLI]]
- [[Locke Roadmap]]

## Status

- Locke **v0.4** (2026-05-27). **238 tests passing** across unit + integration + proptest + Bolero + insta + CLI + ground-truth + language-builtin + JSON-emit + gate + HTML + streaming + Parquet-recognition buckets.
- Crate: `cjc-locke` (added to the workspace alongside `cjc-quantum`, `cjc-abng`, `cjc-dharht`).
- Error-code range: **E9000–E9099**.

### v0.4 additions

- **`cjcl locke gate <reference.json> <current>`** — snapshot a known-good report, diff against fresh data, fail CI on appeared findings.
- **`emit_locke_report_json` + `parse_locke_report_json`** — hand-written, zero-dep canonical JSON serializer with schema versioning.
- **Outlier detection (E9040 Notice, E9041 Warning)** — IQR + modified-Z (Iglewicz & Hoaglin), per numeric column.
- **Sentinel-value detection (E9007 Info)** — heuristic candidates for `-1`, `-9999`, `"NA"`, etc.
- **HTML report (`--html PATH`)** — single self-contained file, severity-color-coded, ~5KB.
- **No-reconstruction streaming** — Welford running mean+variance + `BTreeMap<u64,u64>` ECDF for incremental KS. Bit-identical to single-shot KS D.
- **Parquet structural recognition** — clear diagnostics distinguish "not Parquet" from "Parquet but unsupported yet."

### v0.3 additions (close the v0.2 "thin or oversold" gaps)

- **Table-handle registry** — `.cjcl` source can finally do `let h = locke_table_new(); locke_table_add_float_col(h, ...); let r = locke_validate(h);`. 14 new builtins; no `Value::DataFrame` variant required.
- **Real causal-warning severity** — DAG-acknowledged correlations now drop one ordinal severity level programmatically (not just in message text).
- **`TracedDataFrame` covers the full DSL** — added `group_by`, `summarise`, `arrange`, `distinct`, `mutate`, `pivot`, `sample`, `join`, `bind_cols`.
- **`BeliefPenalty`** — swappable per-severity penalty model. Tunable ≠ calibrated, but plug-your-own-calibration is now possible.
- **Ground-truth corpus tests** — 8 fixtures with exactly known seeded properties (17 NaNs, 5 dup rows, KS D ≈ 0.5, severity-threshold ladder).
- **Streaming + multi-format** — `StreamingValidator` with running per-column state; `validate_view(&TidyView, opts)`; CLI auto-detects CSV/TSV/JSONL.

### v0.2 additions

- `NullMask` for non-float columns
- Exact KS D-statistic (E9039) replaced PSI as default numeric-drift signal
- `TracedDataFrame` lineage wrapper (extended in v0.3)
- 6 numeric `locke_*` builtins (extended to 22 in v0.3)
- `CausalDag` assumption registry
- `BeliefWeights`
- `insta` snapshot tests

See [[Locke Roadmap]] for v0.4 priorities.
