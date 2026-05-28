---
title: ADR Index
tags: [adr, index, decisions]
status: Index
---

# ADR Index

Architecture Decision Records for CJC-Lang. Each ADR captures a single decision, why it was made, and what it constrains. The ADRs live in `docs/adr/` — these vault notes are one-screen summaries that link back to the source and connect decisions to concept notes.

**Numbering gap:** ADR-0006, ADR-0007, ADR-0008 do not exist in `docs/adr/`. The numbering jumps 0005 → 0009.

## Accepted decisions (live architecture)

| ID | Title | Status | Date | Summary |
|---|---|---|---|---|
| [[ADR-0001 Tree-form MIR]] | Tree-form MIR with separate derived CFG | Accepted | 2024-01-15 | Keep MIR as nested expressions; derive CFG as a view |
| [[ADR-0002 Kahan Accumulator]] | `KahanAccumulatorF64` for serial reductions | Accepted | 2024-01-15 | O(ε) error bound; order-dependent (serial only) |
| [[ADR-0003 Backward-compatible run_program]] | Stable executor entry points per capability | Accepted | 2024-01-20 | `run_program(program, seed)` signature never changes |
| [[ADR-0004 SplitMix64 RNG]] | SplitMix64 as the canonical CJC RNG | Accepted | 2024-01-15 | Seeded, cross-platform, zero-dep, ~1 ns/sample |
| [[ADR-0005 Binned Accumulator]] | Exponent-binned accumulator for order-invariant summation | Accepted | 2024-01-20 | Commutative; used by parallel reductions |
| [[ADR-0014 MIR Analysis Infrastructure]] | Loop analysis, reductions, verifier modules | Accepted | 2026-03-23 | Additive overlays on tree-form MIR |
| [[ADR-0015 PINN PDE Problem Suite]] | FD residuals, domain geometry, hard BCs | Accepted | 2026-04-11 | Burgers/Poisson/Heat solvers with 64 tests |
| [[ADR-0016 Language-Level GradGraph Primitives]] | `grad_graph_*` builtins via satellite dispatch | Accepted | 2026-04-26 | 24 new builtins; ambient thread-local graph; flips PINN to pure-CJC-Lang |
| [[ADR-0017 Adaptive TidyView Selection]] | Five-arm `AdaptiveSelection` enum, density-classified | Accepted | 2026-04-26 | Empty/All/SelectionVector/VerbatimMask + reserved Hybrid; sparse joins no longer pay dense costs |
| [[ADR-0018 Deterministic Adaptive Dictionary Engine]] | Byte-first categorical engine: `BytePool` + `AdaptiveCodes` + `BTreeMap` lookup | Accepted | 2026-04-28 | Phase 1 of TidyView v3; row-axis (ADR-0017) was adaptive, column-axis (categoricals) now adaptive too |
| [[ADR-0024 Tier-0 Slot Resolution]] | `VarLocal` variant + `HirToMir` slot tracker + executor frame fast-path + closure/match slot resolution + single-source-of-truth cleanup | Accepted | 2026-05-20 | Stages 1+2+3+4+5a shipped. Stage 4: **15% speedup on chess_rl_v2**. Stage 5a: **sharper microbench win (lookup mir/eval 0.70 → 0.50)** but **chess_rl regressed 680s → ~950s** -- workload-specific, flagged for Stage 5b investigation |
| [[ADR-0025 Runtime Policy Layer]] | `RuntimePolicy` struct + thermal profiles + rayon thread cap + 4 CLI flags + 15 policy/energy builtins (green compute) | Accepted | 2026-05-20 | Deterministic thermally-bounded execution. Thread count never changes results; energy estimated from workload counts (not wall time). Laptop-safe `balanced` default. 54 tests (16 unit + 27 wiring + 8 proptest + 3 fuzz) |
| [[ADR-0026 Race-to-Idle Adaptive Scheduling]] | Burst-full / throttle-when-sustained scheduling; global pool full + cap-sized `install` pool (changes ADR-0025's cap mechanism) | Accepted | 2026-05-21 | Recovers burst performance, keeps the sustained thermal bound. Schedule adapts (wall-clock) but output stays bit-identical (concurrency ≠ results). `--no-adaptive` for reproducible benchmarking |
| [[ADR-0027 Fused Elementwise Kernels]] | Single-pass `fused_axpy` / `fused_mul_sub` / `fused_sub_sq` builtins (GC-06 Phase 3a) | Accepted | 2026-05-21 | Eliminate intermediate tensor allocs (~40% less traffic, 1 alloc not 2). Bit-identical to unfused (separate mul+add, no FMA). Compounds with race-to-idle. 17 tests |
| [[ADR-0028 Locke Data Skepticism Layer]] | New `cjc-locke` crate: validation + drift + lineage + belief + causal guardrails | Accepted | 2026-05-27 | Evidence-aware analytics for CJC-Lang. 9 modules, content-addressed IDs, `BTreeMap` discipline, Kahan-summed stats, conservative causal warnings. `cjcl locke validate \| drift \| belief \| lineage \| causal`. 115 tests (65 unit + 44 integration + 6 CLI) |
| [[ADR-0029 Locke v0.2 Capabilities]] | Locke v0.2: null masks, exact KS, traced lineage, language builtins, causal DAG, weighted belief, insta snapshots | Accepted | 2026-05-27 | Closes the seven v0.1 deferrals. `NullMask` + E9006, exact KS D-statistic (E9039 replaces PSI default), `TracedDataFrame`, `locke_*` language builtins via satellite dispatch, `CausalDag` assumption registry, `BeliefWeights`, `insta` golden tests. 170 tests total (+55), no regression in adjacent crates |
| [[ADR-0030 Locke v0.3 Capabilities]] | Locke v0.3: table-handle registry, real causal severity, full TracedDataFrame coverage, tunable BeliefPenalty, ground-truth corpus, streaming + JSONL | Accepted | 2026-05-27 | Closes the six v0.2 "thin or oversold" gaps. `.cjcl` source can finally `locke_validate(h)` a DataFrame via thread-local handle registry (preserves Value enum). DAG-downgraded warnings drop severity programmatically. TracedDataFrame covers 14 cjc-data ops. BeliefPenalty makes the model swappable. 8 ground-truth fixtures with known seeded properties. StreamingValidator + validate_view(&TidyView). CLI accepts CSV/TSV/JSONL. 202 tests total (+32), no regression |
| [[ADR-0031 Locke v0.4 Capabilities]] | Locke v0.4: gate command + JSON Schema, outlier + sentinel detection, HTML output, no-reconstruction streaming, Parquet structural recognition | Accepted | 2026-05-27 | Closes 5 of 6 v0.4 Tier-1 priorities. `cjcl locke gate ref.json current` for snapshot diff. E9040/E9041 outliers (IQR + modified-Z), E9007 sentinel detection. Self-contained HTML report. Welford + ECDF map → exact streaming KS D bit-identical to single-shot. Parquet structural skeleton (full decoder v0.5). 238 tests total (+36), no regression |
| [[ADR-0032 Locke v0.5 Capabilities]] | Locke v0.5: time-aware validation, target leakage AUC, conditional missingness, imbalanced class, ID-like cardinality, dup-key conditioning, HTML correlation matrix | Accepted | 2026-05-28 | Customer-churn project enablement. `--time-col`/`--max-timestamp`/`--gap-threshold` flags (E9050-E9054). Per-feature ROC AUC vs binary target (E9060/E9061), with `f64::total_cmp` rank-sum. Pairwise NaN-implication scan (E9070). E9071 imbalanced-class. E9072 ID-like cardinality. E9073 dup-key conditioning. Inline-SVG correlation heatmap in HTML (with-DataFrame overload). 264 tests total (+26), no regression |
| [[ADR-0033 Locke v0.6 Categorical and Drift Capabilities]] | Locke v0.6 batch 1: categorical / string semantic quality, mojibake / confusable-script detection, Wasserstein-1 in drift, Mermaid lineage CLI, reproducibility verify CLI | Accepted | 2026-05-28 | 8 new codes E9016 rare-category, E9017 one-hot risk, E9080 case-fold, E9081 whitespace, E9082 Levenshtein, E9083 mixed-script, E9084 mojibake, E9085 transitive cluster. Wasserstein-1 added as evidence inside E9039 KS. `cjcl locke lineage --mermaid` emits Quarto Mermaid block. `cjcl locke verify --runs N` asserts byte-identical reports. Belief axes: E9016 → constraint, E9017+E9080-85 → schema (meet-semilattice preserved). cjc-locke lib 217 (+20), tests/locke 103 (+14), cjc-cli 154 unchanged. 4 proptest + 3 bolero fuzz targets added |

## Proposed decisions (not yet implemented)

| ID | Title | Status | Date | Summary |
|---|---|---|---|---|
| [[ADR-0009 Vec COW Array]] | `Rc<Vec<Value>>` for `Value::Array` and `Value::Tuple` | Proposed | 2025-01-01 | O(1) array passing; COW via `Rc::make_mut` |
| [[ADR-0010 Scope Stack SmallVec]] | SmallVec-backed scope frames | Proposed | 2025-01-01 | Deferred pending profile data |
| [[ADR-0011 Parallel Matmul]] | Rayon parallel matmul (feature-gated) | Proposed | 2025-01-01 | Uses [[ADR-0005 Binned Accumulator]] for determinism |
| [[ADR-0012 CFG Phi Nodes]] | True CFG with phi nodes and use-def chains | Proposed | 2025-01-01 | Extends [[ADR-0001 Tree-form MIR]] |
| [[ADR-0013 Package Manager]] | Minimal `cjc.toml` package manager | Proposed | 2026-03-22 | Git-based deps, no diamonds in v1 |

## Decision graph

```
ADR-0001 (Tree-form MIR) ──┬──> ADR-0003 (stable entry points)
                           └──> ADR-0012 (CFG phi nodes — *extends*, does not supersede)

ADR-0002 (Kahan, serial) ─────> ADR-0011 (parallel matmul — switches to Binned)
ADR-0005 (Binned, commutative) ┘

ADR-0004 (SplitMix64) ────────> every deterministic RNG user

ADR-0014 (MIR analysis) ──────> future optimizer passes (LICM, CSE, …)
```

## Reading order for new contributors

1. [[ADR-0001 Tree-form MIR]] — the MIR data model everything else builds on
2. [[ADR-0003 Backward-compatible run_program]] — how the executor API stays stable
3. [[ADR-0002 Kahan Accumulator]] + [[ADR-0005 Binned Accumulator]] — the two-accumulator determinism story
4. [[ADR-0004 SplitMix64 RNG]] — the other half of determinism
5. [[ADR-0014 MIR Analysis Infrastructure]] — what's live on top of MIR today
6. Proposed ADRs last — they are forward-looking

## Related

- [[Compiler Source Map]]
- [[Runtime Source Map]]
- [[Determinism Contract]]
- [[MIR]]
