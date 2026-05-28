# Locke Roadmap

## v0.1.x (shipped)

- [x] `cjc-locke` crate scaffolded with 9 modules
- [x] core types: `LockeReport`, `ValidationFinding`, `FindingSeverity`, `FindingEvidence`, `BeliefScore`, `BeliefReport`, `LockeImpression`, `LockeIdea`, `LineageGraph`, `AuditEvent`, `InductionRiskReport`, `CausalGuardrailReport`
- [x] missingness (NaN-as-missing for Float)
- [x] duplicate-row + duplicate-key detection
- [x] impossible-value DSL (5 rule kinds)
- [x] constant + near-constant detection
- [x] schema mismatch (missing, type, extra)
- [x] suspicious cardinality heuristic
- [x] drift compare: mean/std/range/missingness/PSI/TVD/small-sample
- [x] lineage builder with content-addressed IDs and audit chain
- [x] belief score with 8 explainable sub-dimensions
- [x] causal-guardrail with 5 warning kinds + observational-only mode
- [x] CLI: `cjcl locke validate|drift|belief|lineage|causal` + `--json`, `--fail-on`, `--target`, `--observational-only`
- [x] 115 tests (65 unit + 44 integration + 6 CLI), deterministic across runs
- [x] Obsidian docs (11 files + ADR)

## v0.2 (shipped 2026-05-27)

- [x] **Null mask for non-float columns** — `NullMask::from_indices(...)`, `ValidateOptions::null_masks`. E9001 now fires for non-float columns when a mask is supplied; E9002 stays as a downgraded Info-level note when no mask is given. Out-of-bounds indices flagged as E9006 (Warning) and skipped.
- [x] **Exact KS D-statistic (E9039)** — replaces equal-width-binned PSI as the default numeric-drift signal. `cjc_locke::stats::ks_d_statistic(xs, ys)`. Default thresholds `ks_d_warn=0.10`, `ks_d_error=0.20`. PSI helper retained internally for the language-level `locke_psi` builtin.
- [x] **Auto-instrumented lineage** — `TracedDataFrame::observe(builder, "src.csv", df).filter(...).select(...).with_column(...)`. Each method emits a `LockeIdea` + lineage edge.
- [x] **CJC-Lang language-level builtins** — `locke_missing_count`, `locke_missing_rate`, `locke_ks_d`, `locke_psi`, `locke_sample_score`, `locke_belief_overall` via `cjc_locke::dispatch_locke` satellite dispatch routed from both `cjc-eval` and `cjc-mir-exec`. Bit-identical parity verified.
- [x] **DAG-based assumption registry** — `CausalDag` type with cycle detection. `CausalConfig::assumed_dag` lets users declare hypothesised causal pathways; strong-correlation warnings between related pairs are downgraded with annotated assumptions.
- [x] **User-tunable belief-score weights** — `BeliefWeights` struct + `BeliefScore::from_dimensions_weighted(...)`. Default = equal weights (bit-equivalent to v0.1). Negative / NaN weights are clamped to 0.
- [x] **Insta CLI snapshot tests** — `tests/locke/snapshot_tests.rs`. 4 snapshots cover validate / belief / drift / lineage CLI emit. Fingerprints redacted to `[FP]` so unrelated evidence-format tweaks don't cascade.

## v0.3 (shipped 2026-05-27)

- [x] **Table-handle registry** — `.cjcl` source can build, validate, and inspect DataFrames via `locke_table_*` and `locke_validate` builtins. No `Value::DataFrame` introduced; uses the thread-local handle pattern.
- [x] **Real `severity` on `CausalWarning`** — DAG downgrade actually drops the ordinal level so consumers filtering by severity see the demotion.
- [x] **`TracedDataFrame` full DSL coverage** — `group_by`, `summarise`, `arrange`, `distinct`, `mutate`, `pivot`, `sample`, `join`, `bind_cols` all emit lineage.
- [x] **`BeliefPenalty`** + `belief_report_from_locke_with_model(...)` — penalty model is now a swappable component, default = v0.2 baked-in.
- [x] **Ground-truth corpus tests** — `tests/locke/ground_truth_tests.rs` with 8 fixtures.
- [x] **`StreamingValidator` + multi-format input** — CLI accepts CSV/TSV/JSONL via `cjc-cli::formats::detect_format`; `validate_view(&TidyView, opts)` for lazy filtering.

## v0.4 (shipped 2026-05-27)

- [x] **`cjcl locke gate <ref.json> <current>`** — canonical JSON serializer + report diff command for CI gates.
- [x] **Outlier detection** (E9040 / E9041, IQR + modified-Z).
- [x] **Sentinel-value detection** (E9007).
- [x] **HTML report output** (`--html PATH`, self-contained, severity-color-coded).
- [x] **No-reconstruction streaming** — Welford + ECDF map. KS D bit-identical to single-shot at `sample_cap=0`.
- [x] **Parquet structural recognition** — better diagnostics; full decoder is v0.5.

## v0.5 (shipped 2026-05-28)

- [x] **Time-aware validation** — `temporal.rs` with sortedness/cutoff/gap/overlap checks. E9050-E9054.
- [x] **Target leakage detection** — `leakage.rs` with rank-sum AUC. E9060/E9061/E9062.
- [x] **Conditional missingness** — pairwise NaN-implication scan. E9070.
- [x] **Imbalanced-class warning** — E9071.
- [x] **ID-like cardinality hint** — E9072.
- [x] **Duplicate-key conditioning** — E9073.
- [x] **HTML cross-column correlation matrix** — inline-SVG heatmap via `emit_locke_report_html_with_df`.

## v0.6 (in progress, 2026-05-28)

### Shipped this release

- [x] **Categorical / string semantic-quality detectors** — 5 codes in `categorical.rs`:
  - **E9016** rare categories (long-tail / overfit risk)
  - **E9017** one-hot encoding-risk (cardinality > 50, ratio < 0.95)
  - **E9080** case-fold collisions ("Premium"/"premium"/"PREMIUM")
  - **E9081** whitespace + terminal-punctuation variants ("USA"/"USA.")
  - **E9082** near-duplicate categories via bounded Levenshtein ≤ 2
- [x] **Confusable / Unicode anti-spoofing** — 3 additional codes:
  - **E9083** mixed-script labels (Latin + Cyrillic / Greek / CJK in one string)
  - **E9084** mojibake detection (UTF-8 decoded as Latin-1, including Win-1252 smart-quote residue)
  - **E9085** transitive-cluster summary (when ≥ 2 of E9080/E9081/E9082 fire on same column)
- [x] **Wasserstein-1 drift evidence** — added as `Metric { label: "wasserstein_1" }` inside every E9039 KS finding. Unit-bearing complement to KS D's unit-free shape statistic. Computed via O((n+m) log(n+m)) merge integration over the two empirical CDFs with Kahan accumulation. Public API: `cjc_locke::wasserstein_1(&[f64], &[f64]) -> Option<f64>`.
- [x] **CLI `cjcl locke lineage --mermaid`** — emit Quarto/Markdown Mermaid `flowchart LR` block from a lineage graph. Impressions render as cylinder nodes, ideas as rounded nodes. Two runs over the same graph emit byte-identical bytes.
- [x] **CLI `cjcl locke verify --runs N`** — re-runs `validate` N times and asserts byte-identical canonical-JSON output. Exits 0 on agreement; non-zero with `status: DIVERGENT` line otherwise.

### Belief-axis mapping for the new codes

| Code | Axis weakened | Why |
|---|---|---|
| E9016 | `constraint` | rare categories are a distributional, not schema-shape, concern |
| E9017 | `schema` | one-hot explosion is an encoding-design defect of the effective alphabet |
| E9080–E9085 | `schema` | semantic-category fragmentation = effective-alphabet ambiguity |

The meet-semilattice composition algebra documented in [[Locke Belief Reports]] §formal-model is preserved — no new BeliefScore axes were added; v0.6 codes route to existing axes.

### Test infrastructure

- [x] **Unit tests** — 20 new in `categorical::tests` and `drift::tests` (Wasserstein).
- [x] **Integration tests** — 14 new in `tests/locke/categorical_tests.rs`.
- [x] **Property tests** — 4 new in `tests/locke/locke_proptest.rs` covering categorical determinism, Wasserstein non-negativity / symmetry, Mermaid emit determinism.
- [x] **Bolero fuzz targets** — 3 new in `tests/locke/locke_fuzz.rs` (arbitrary strings → no panic, arbitrary floats for Wasserstein → finite + non-negative, arbitrary labels for Mermaid → well-formed fenced block).
- [x] **CLI parser tests** — 4 new in `crates/cjc-cli/src/commands/locke.rs` (`--mermaid` flag, `verify` subcommand + `--runs N`).
- [x] **Insta snapshot** — `drift_default.snap` updated to include the new W1 evidence in the E9039 message.

Net delta: cjc-locke 217 lib (was 197, +20) + tests/locke 103 (was 89, +14) + cjc-cli 154 (no regressions). Workspace builds clean.

### v0.6 priorities still deferred

#### Data-skepticism upgrades (legacy entries kept for traceability)
- [x] ~~**Sentinel-value detection**~~ — shipped v0.4 as E9007.
- [x] ~~**Outlier heuristics**~~ — shipped v0.4 as E9040/E9041.
- [ ] **Distribution-shape diagnostics** — skew, kurtosis, top-k modes.

### Drift upgrades

- [ ] **Exact Kolmogorov–Smirnov D** — replace the bin-grid PSI approximation with a sorted-CDF KS statistic.
- [ ] **Time-aware splits** — `--time-col` enables ordering checks and future-leakage warnings.
- [ ] **Label-drift conditional on features** — opt-in metric requiring a user-supplied model or grouping.
- [ ] **Conditional shift** — slice drift by category levels.

### Lineage upgrades

- [ ] **Auto-instrumentation from `cjc-data`** — pipeline ops emit lineage edges automatically.
- [ ] **Binary serialise via `cjc-snap`** — `LineageGraph` round-trips to a `SnapBlob`.
- [ ] **Lineage diff** — structural comparison between two graphs.
- [ ] **Reachability queries** — "which Ideas depend on Impression X?"

### Belief upgrades

- [ ] **User-tunable weights** — `BeliefScore::weighted_mean(weights)`.
- [ ] **Evidence-weighted penalties** — scale by `violation_rate` instead of fixed steps.
- [ ] **Per-column belief** — `BeliefScore` per column, not just dataset-wide.

### Causality upgrades

- [ ] **DAG-based assumption registry** — user declares a partial DAG; Locke checks reports against it.
- [ ] **Multiple correlation kinds** — Spearman rank, mutual information.
- [ ] **Sensitivity-style placeholder** — `CounterfactualSensitivityCheck` becomes a real perturbation harness.

### CJC-Lang language integration

- [ ] **`locke.validate(df)` builtin** — invoke from `.cjcl` source.
- [ ] **`locke.belief(df)` builtin**.
- [ ] **`locke.lineage(...)` syntax** — annotation on pipeline functions.

### Tooling

- [ ] **`insta` snapshot tests** for CLI text emit.
- [ ] **Linux libfuzzer integration** for Bolero targets.
- [ ] **`cjcl locke gate <report.json>`** — re-evaluate a stored report against new thresholds without re-scanning data.

## Open questions

- Should belief reports support an export format compatible with [Croissant](https://github.com/mlcommons/croissant) / ML dataset metadata standards?
- Is there a useful interface to ABNG's `audit` and `merkle` infrastructure, or does Locke's column-oriented model diverge too much from ABNG's tree-node model to share code?
- Should `cjcl locke validate` learn to read JSONL / Parquet via the existing `formats` module in `cjc-cli`?

## Non-goals (loud)

Locke v0 + v0.2 explicitly will **not**:

- Infer causal effects from observational data.
- Run propensity scoring, instrumental variables, or do-calculus.
- Replace a dedicated experiment-tracking system (mlflow, wandb).
- Replace a dedicated data-validation framework like Great Expectations — Locke is the layer that *complements* such tools with content-addressed lineage and explainable belief reports.
