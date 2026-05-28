# ADR-0033 Locke v0.6 Categorical / String / Drift Capabilities

- **Status:** Accepted (v0.6 batch 1, 2026-05-28)
- **Crate:** `cjc-locke` (extended from ADRs 0028–0032)
- **Companion docs:** [[Locke Data Skepticism]] §v0.6, [[Locke CLI]] §v0.6 subcommands, [[Locke Roadmap]] §v0.6

## Context

The user-facing assessment at the close of v0.5 named three structural gaps that read as "Locke validates numbers well but is weak where most real-world failures actually happen":

1. **Categorical / string semantic quality** — case-fold collisions, whitespace / punctuation variants, near-duplicates by edit distance, rare categories, encoding-risk (one-hot explosion).
2. **Unicode / encoding anti-spoofing** — confusable-script attacks (Cyrillic 'а' impersonating Latin 'a'), mojibake (UTF-8 decoded as Latin-1).
3. **Stronger drift theory** — Wasserstein-1 as a unit-bearing complement to the existing KS D-statistic.

Plus two CLI ergonomic additions:

4. **`cjcl locke lineage --mermaid`** — emit Quarto/Markdown-friendly Mermaid for the lineage graph so it can be pasted directly into Locke blog posts and architecture docs.
5. **`cjcl locke verify --runs N`** — declarative reproducibility check (re-run N times, assert byte-identical canonical-JSON output).

These are the "cheap items" from a much larger v0.6/v0.7 roadmap. They share two properties that make them appropriate for one batch:

- Each is a small, additive detector or CLI flag — none requires new core types.
- The error-code namespace is the same `E9000–E9099` block; eight new codes fit in already-reserved sub-ranges.

## Decisions

### 1. New module `crates/cjc-locke/src/categorical.rs`

Eight detectors emit eight codes:

| Code | Severity | Detector |
|---|---|---|
| **E9016** | Notice | `detect_rare_categories` — categories with count < `rare_category_min_count` (default 5), requires ≥ `rare_category_min_rare_count` (default 2) rare categories before firing. Long-tail overfit / encoder-instability signal. |
| **E9017** | Notice | `detect_encoding_risk` — `n_distinct > one_hot_explosion_threshold` (default 50) AND `cardinality_ratio < 0.95` (so E9072 still owns the ID-leakage case). Suggests embedding instead of one-hot. |
| **E9080** | Warning | `detect_case_fold_collisions` — Rust `str::to_lowercase` group, fire on any group with > 1 distinct original spelling. |
| **E9081** | Notice | `detect_whitespace_punctuation_variants` — `trim` + strip terminal `.,!?;:` + `to_lowercase`, fire only when the lowercase forms differ (defers to E9080 otherwise). |
| **E9082** | Warning | `detect_near_duplicate_categories` — bounded Levenshtein distance ≤ 2 on strings of length ≥ 4; O(N²·L) scan capped at `max_categories_for_edit_distance` (default 200), emit Info note when the cap is exceeded. |
| **E9083** | Warning | `detect_confusable_scripts` — Unicode script-bucket counting; fire on strings of length ≥ 3 that span > `mixed_script_max_distinct` distinct buckets (default 1, i.e., any mixed-script string). Covers Latin / Greek / Cyrillic / Hebrew / Arabic / Devanagari / CJK / Hiragana / Katakana / Hangul plus a coarse "other" bucket. |
| **E9084** | Notice | `detect_mojibake` — Signature counting for the two classic UTF-8-decoded-as-Latin-1 / Win-1252 patterns: `'Ã'\|'Â'` followed by a Latin-1-supplement char (covers é/è/à/ç/ñ/©/®), and `'â'` followed by a Win-1252 special (covers smart quotes, em-dash, bullet). |
| **E9085** | Notice | `detect_transitive_clusters` — *aggregates* prior E9080/E9081/E9082 findings. Fires when ≥ `transitive_cluster_min_signals` (default 2) distinct channels agree on the same column. Sits alongside the per-channel findings; does not replace them. |

`CategoricalQualityConfig::default()` is sensible for most workloads; every threshold is a public field for callers that need to tune.

Wiring: `validate_dataframe(...)` now calls `crate::categorical::detect_all_categorical_quality(...)` after sentinel detection. Belief axes updated in `api::belief_report_from_locke_with_model`:

- **E9016 → `constraint_score`** (rare categories are a distributional concern, not a schema-shape one)
- **E9017, E9080–E9085 → `schema_score`** (semantic-category fragmentation = effective-alphabet ambiguity)

No new BeliefScore axes were added. The meet-semilattice composition algebra documented in [[Locke Belief Reports]] §formal-model is preserved.

### 2. Wasserstein-1 in `crates/cjc-locke/src/drift.rs`

- New public function `pub fn wasserstein_1(a: &[f64], b: &[f64]) -> Option<f64>` — exact 1-Wasserstein distance via two-pointer merge over sorted samples. O((n+m) log(n+m)). NaN excluded (matching `ks_d_statistic` semantics). Kahan accumulation for the integral. Returns `None` on empty input.
- Integrated into `numeric_ks_finding` (E9039): every KS finding's `evidence` now carries `Metric { label: "wasserstein_1", value: ... }` alongside `ks_d`. Reviewers can use both: KS catches "ECDFs differ in shape", Wasserstein catches "mass moved a measurable distance in the column's units".
- The KS finding's `message` reads `"numeric distribution shift in `x` (KS D = 0.5000, W1 = 50.0000)"`. The literal `W1` (not the subscript `W₁`) is used to keep the insta snapshot file ASCII-clean under Windows tooling that occasionally double-encodes Unicode subscripts.

### 3. Lineage Mermaid emit (`crates/cjc-locke/src/lineage.rs`)

- New public function `pub fn emit_lineage_mermaid(g: &LineageGraph) -> String`.
- Output: a fenced `\`\`\`{mermaid}` block with a `flowchart LR` body.
- Determinism: nodes iterate in BTreeMap (content-addressed-id sorted) order; edges iterate in Vec order. Two runs over the same graph produce byte-identical bytes.
- Quote escaping: any `"` in a label is replaced with `&quot;`; embedded newlines are squashed to spaces.
- CLI wiring: `cjcl locke lineage <data.csv> [--mermaid]` — the flag swaps `emit_lineage_text` for `emit_lineage_mermaid`. Mutually exclusive with `--json` (text and Mermaid are the only options; JSON for lineage is a v0.7 question).

### 4. Reproducibility verifier (`cmd_verify`)

- New CLI subcommand `cjcl locke verify <data.csv> [--runs N]` (default N = 3, minimum 2).
- Runs `validate(...)` N times, captures `emit_locke_report_json(...)` for each, compares byte-for-byte against the first run.
- Emits a `# Locke Reproducibility Verifier` block showing dataset, run_id, finding count, report byte count, and either `status: REPRODUCIBLE` or `status: DIVERGENT — k/N runs differed from run 0 (indices: ...)`.
- Exit code: 0 on agreement; 2 on parse/IO errors; 3 (via `--fail-on error` policy) when divergent.

## Test infrastructure

| Layer | Location | New count |
|---|---|---|
| Unit (in-module) | `crates/cjc-locke/src/categorical.rs`, `drift.rs`, `lineage.rs` | 20 |
| Integration | `tests/locke/categorical_tests.rs` | 14 |
| Property (proptest) | `tests/locke/locke_proptest.rs` | 4 |
| Bolero structural fuzz | `tests/locke/locke_fuzz.rs` | 3 |
| CLI parser (in-module) | `crates/cjc-cli/src/commands/locke.rs` | 4 |
| Insta snapshot | `tests/locke/snapshots/locke__snapshot_tests__drift_default.snap` | 1 updated |

Final totals: **cjc-locke 217 lib** (was 197, +20) + **tests/locke 103** (was 89, +14) + **cjc-cli 154** (no regressions). All `--release` workspace builds clean; no parity / dispatch regressions in adjacent crates.

## Out of scope (deferred to next v0.6 batch / v0.7)

- **PII detection** (E9090–E9093): email, phone, SSN, API-key regex heuristics — requires false-positive corpus before landing.
- **Category drift** — entropy shift, new-category set-difference, disappearing-category, exploding cardinality drift between train/test.
- **Label-encoding risk / ordinal-on-nominal**: detect Int columns that are nominally categorical.
- **Per-column confidence summary**: unified per-column report combining all findings into a single "Confidence: Moderate, Issues: …" emit.
- **Ontology / taxonomy consistency**: hyphen/underscore variants, common-prefix taxonomy inference.
- **Text drift**: vocabulary KS, token-entropy drift, language-distribution shift.
- **Category lineage**: per-value `raw → normalized → grouped → encoded → embedding` trails.
- **Governance workflows**: suppression files, owner annotations, required-finding policies.
- **Belief-report formal proof tests**: property-test the meet-semilattice algebra in code (it's documented but not test-locked).
- **CategoricalAdaptive variant** support: every v0.6 detector skips `Column::CategoricalAdaptive` for now.

These are explicitly named in [[Locke Roadmap]] as v0.6.x / v0.7 work.

## Consequences

1. **Locke v0.6 covers most real-world non-numeric data-quality failures** — case/whitespace/typo collisions, mixed-script confusables, mojibake — without needing a regex DSL. The detectors are deterministic, content-addressed-id stable, and route into existing belief axes.
2. **Wasserstein-1 alongside KS gives drift reviewers a second axis to triangulate on.** A small KS D + large Wasserstein-1 means "mass moved a small distance everywhere"; a large KS D + small Wasserstein-1 means "ECDFs cross but means are close".
3. **Mermaid lineage emit makes documentation generation trivial.** Locke blog posts and architecture docs can paste-in the live lineage graph without manual transcription.
4. **`verify` makes the reproducibility claim testable from CI** without callers writing their own diff harness.

## Open questions

- Should E9085 (transitive cluster) become a CLI-suppressible finding by default? Some users will want only the per-channel findings, not the aggregate.
- Is the `script_bucket` classifier too coarse? Combining Latin and Latin-Extended into one bucket would reduce E9083 false positives on legitimate accented text (currently a string like "café" doesn't trigger because `é` is in the Latin-1 range U+0080..U+024F → "latin_extended" bucket while "c"/"a"/"f" are "latin"). Default v0.6 behavior: a "café"-only string IS flagged as mixed-script if combined with ASCII context. We may need to merge `latin` and `latin_extended` in v0.6.1.
- Per-axis BeliefScore composition (v0.2 plan from the v0.5 blog) and continuous selectivity semantics for `transform_factor` remain deferred. The meet-semilattice algebra still uses uniform component-wise min.

Related: [[ADR-0028 Locke Data Skepticism Layer]], [[ADR-0032 Locke v0.5 Capabilities]], [[Locke Roadmap]], [[Locke Data Skepticism]], [[Locke CLI]].
