# ADR-0030 Locke v0.3 Capabilities

- **Status:** Accepted (v0.3, 2026-05-27)
- **Crate:** `cjc-locke` (extended from ADR-0028 / ADR-0029)

## Context

[[ADR-0029 Locke v0.2 Capabilities]] closed the v0.1 deferrals but left six "thin or oversold" gaps explicitly named in the v0.2 honest assessment:

1. `locke.validate(df)` from CJC-Lang source was numeric-primitive-only — no DataFrame-level API.
2. The causal DAG "downgrade" was message-text-only with no real severity drop.
3. `TracedDataFrame` covered `filter` / `select` / `with_column` / `rename` / `concat` only.
4. The belief penalty model (0.02 / 0.10 / 0.25) was an opinion baked into code, not configurable.
5. Test coverage was self-consistency only — no ground-truth fixtures with exact known properties.
6. Input was CSV-only + fully in-memory; no streaming, no JSONL.

v0.3 closes all six gaps. The user's directive "model Locke after TidyView in how it handles data" is honored by the streaming validator's design.

## Decisions

### 1. Table-handle registry for `locke.validate(df)`

- New `thread_local!` stores in `cjc-locke/src/dispatch.rs`: `MUT_TABLES`, `READY_TABLES`, `REPORTS`, `DRIFT_REPORTS`, `NEXT_HANDLE`.
- Pattern matches `cjc-ad::GradGraph` and `cjc-abng::dispatch_abng` precedent. **No `Value::DataFrame` variant introduced** — the `Value` enum stays untouched.
- 14 new builtins: `locke_table_new`, `locke_table_add_float_col`, `locke_table_add_int_col`, `locke_table_add_str_col`, `locke_table_add_bool_col`, `locke_table_freeze`, `locke_table_nrows`, `locke_table_ncols`, `locke_validate`, `locke_drift`, `locke_report_worst_severity`, `locke_report_n_findings`, `locke_report_count_by_severity`, `locke_report_overall_score`, `locke_drift_n_findings`, `locke_drift_worst_severity`.
- End-to-end CJC-Lang parity test verified through both `cjc-eval` and `cjc-mir-exec`.

### 2. Real `severity` field on `CausalWarning`

- New `pub severity: FindingSeverity` field on `CausalWarning`.
- `CausalWarningKind::default_severity()` returns the per-kind default (StrongCorrelationNoIntervention = Warning, LikelyConfounder = Warning, CausalLanguageInLabel = Notice, ObservationalOnly = Notice, ModelExplanationAsCausal = Warning).
- New `CausalWarning::new_with_severity(...)` constructor + `drop_severity_one_level(...)` helper.
- DAG-acknowledged correlations now produce warnings with severity dropped one ordinal level — programmatic consumers filtering by severity get the right answer without parsing message text. Severity also participates in the content fingerprint so downgraded warnings have distinct IDs.

### 3. Extended `TracedDataFrame`

Added 8 op-recording methods: `group_by`, `summarise`, `arrange`, `distinct`, `mutate`, `pivot`, `sample`, `join`, `bind_cols` (in addition to v0.2's `filter`, `select`, `with_column`, `rename`, `transform`, `concat`).

`TracedDataFrame` is now a genuine drop-in wrapper for the cjc-data DSL — any pipeline step that produces a new DataFrame can be recorded as a lineage edge with deterministic content-addressed parameters.

The "auto" framing is more honest in v0.3 docs: it's *one wrapper covering the full DSL*, not magical interception.

### 4. User-tunable `BeliefPenalty`

- New struct: `BeliefPenalty { info, notice, warning, error: f64 }` with `Default` matching v0.2's baked-in values.
- New `penalty_from_findings_with_model(...)` and `belief_report_from_locke_with_model(...)` variants.
- v0.2's `penalty_from_findings(...)` becomes a thin shim that calls the new function with `BeliefPenalty::default()` — every existing test stays green.
- Tunable ≠ calibrated. Users with calibration data can plug it in; users without it stay on the defaults.

### 5. Ground-truth corpus tests

- New `tests/locke/ground_truth_tests.rs` with 8 tests over synthetic fixtures with *exactly known* statistical properties:
  - `missingness_corpus(n, k, seed)` — exactly K NaN positions
  - `duplicates_corpus(n_unique, n_dup_pairs)` — exactly K dup rows in K groups
  - `drift_corpus_known_d()` — uniform shift with KS D ≈ 0.5
- Tests assert Locke detects *exactly* the seeded conditions, including severity-threshold transitions at 5% / 30% / 60% missingness.

### 6. Streaming + multi-format input, TidyView-modeled

- New `cjc-locke/src/streaming.rs` with `StreamingValidator` + `StreamingConfig`.
- Per-column running state: `KahanAccumulatorF64` for sums, NaN counters, `BTreeSet` for distinct values (capped), sample buffers (capped), exact row-canonical-bytes `BTreeMap` for dup detection (capped).
- Determinism: feeding chunks in row order produces a `LockeReport` byte-identical to single-shot `validate(&full_df)` (verified by `streaming_matches_single_shot_on_equal_chunks` test).
- `validate_view(view: &TidyView, opts)` — accepts a `cjc_data::TidyView` directly so pre-filtering with the TidyView DSL (which uses predicate bytecode + sparse-gather) is honored.
- CLI: `cjcl locke validate <path>` now auto-detects format via `cjc-cli::formats::detect_format`. CSV, TSV, JSONL all work. Parquet/Arrow remain metadata-only with clear error messages pointing to v0.4.
- Modeled on TidyView's design: lazy views (TidyView), running stats per column (analog of zone maps), and explicit memory caps (sample_cap, distinct_cap, duplicate_hash_cap) so the validator degrades gracefully on huge inputs rather than OOM-ing.

## Test counts

| Bucket                          | v0.2 | v0.3 | Δ   |
|---------------------------------|------|------|-----|
| `cargo test -p cjc-locke --lib` | 101  | 122  | +21 |
| `cargo test --test locke`       | 63   | 74   | +11 |
| `cargo test -p cjc-cli locke`   | 6    | 6    | 0   |
| **Total**                       | **170** | **202** | **+32** |

Adjacent regression: `cjc-data` 258/258, `cjc-eval` 28/28, `cjc-mir-exec` 17/17 — zero regression.

## Consequences

**Positive**

- `cjcl` users can now validate JSONL files (the most common ML metadata format).
- `.cjcl` source can finally do `let r = locke_validate(h); print(locke_report_worst_severity(r));` — real DataFrame-level skepticism in the language.
- The causal DAG downgrade actually flows through `severity` so dashboards and CI gates filtering by severity see the demotion.
- The belief score model is now a swappable component: teams with calibration data can plug in their own penalty values.
- Ground-truth fixtures move test confidence from "doesn't panic" to "catches *exactly* the seeded bugs at the threshold ladder."
- Streaming validation removes the in-memory limit — large datasets become tractable, with explicit caps that bound memory deterministically.

**Negative**

- The dispatch surface in `cjc-locke` doubled (6 → 22 builtins). The CallDispatch cache in `cjc-mir-exec` continues to handle this efficiently, but the surface is now non-trivial to memorize.
- The streaming validator's v0.3 implementation reconstructs a `DataFrame` from retained samples to delegate to single-shot `validate(...)`. This loses fidelity past `sample_cap` for KS-style checks. v0.4 will move every check into the streaming layer so no reconstruction is needed.
- Parquet/Arrow inputs still require conversion to CSV/JSONL — full readers are deferred to v0.4.

## Related

- [[ADR-0028 Locke Data Skepticism Layer]] (v0.1 base)
- [[ADR-0029 Locke v0.2 Capabilities]]
- [[ADR-0017 Adaptive TidyView Selection]] — the design Locke v0.3 streams against
- [[TidyView Architecture]] — the data-handling pattern Locke's streaming layer models on
- [[Locke Overview]], [[Locke Roadmap]]
