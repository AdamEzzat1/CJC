//! High-level facade used by the CLI and (later) the CJC-Lang binding.
//!
//! The facade composes validators + drift + belief + causal into a
//! single function-call surface, returning fully-deterministic reports.

use std::collections::BTreeMap;

use cjc_data::DataFrame;

use crate::algebra::{compose_many, BeliefAxisRules};
use crate::belief::{
    penalty_from_findings, penalty_from_findings_with_model, sample_score_from_n, BeliefPenalty,
    BeliefReport, BeliefScore,
};
use crate::causal::{audit_correlations, CausalConfig, CausalGuardrailReport, CorrelationFinding};
use crate::drift::{compare as compare_drift, DriftConfig, InductionRiskReport};
use crate::lineage::{LineageBuilder, LineageGraph, LockeImpression, ImpressionKind};
use crate::report::{
    ColumnBeliefReport, FindingSeverity, LockeInputSummary, LockeReport, ValidationFinding,
};
use crate::stats::pearson_correlation;
use crate::validation::{
    validate_dataframe, ExpectedSchema, ImpossibleValueRule, NullMaskMap, ValidationConfig,
};

/// User-facing options for a single `validate(...)` call.
///
/// `null_masks` (v0.2) lets the caller mark specific row indices as null
/// for non-float columns. Without a mask, non-float missingness is not
/// inferred — see [`NullMask`](crate::validation::NullMask).
///
/// `custom_detectors` (v0.8 / ADR-0041) registers user-defined detectors
/// that run after the built-in ones. They must declare a code in
/// `E9500..=E9999` and which belief axes they affect. See
/// [`crate::custom_detector`] for the trait + determinism contract.
#[derive(Clone, Debug, Default)]
pub struct ValidateOptions {
    pub dataset_label: String,
    pub config: ValidationConfig,
    pub impossible_rules: Vec<ImpossibleValueRule>,
    pub expected_schema: Option<ExpectedSchema>,
    pub primary_key: Option<String>,
    pub null_masks: NullMaskMap,
    pub custom_detectors: Vec<std::sync::Arc<dyn crate::custom_detector::CustomDetector>>,
}

/// Build a `LockeReport` from a single dataframe.
pub fn validate(df: &DataFrame, opts: &ValidateOptions) -> LockeReport {
    let mut findings = validate_dataframe(
        df,
        &opts.config,
        &opts.impossible_rules,
        opts.expected_schema.as_ref(),
        opts.primary_key.as_deref(),
        &opts.null_masks,
    );

    // v0.8 (ADR-0041) — invoke custom detectors after built-ins. They
    // produce findings in the E9500..=E9999 namespace and declare which
    // belief axes their findings affect. We merge their findings into
    // the main vector and stash the axis assignments + assumptions for
    // the LockeReport.
    let custom_outcome =
        crate::custom_detector::run_custom_detectors(df, &opts.custom_detectors);
    findings.extend(custom_outcome.findings.iter().cloned());

    // v0.6.4 — also build the auto-sentinel mask here so the per-column
    // `missingness_rate` reflects auto-detected `?` / `NA` / ... rows on
    // Str columns. Without this, validate_dataframe's E9001 fires
    // correctly (it sees the unioned mask internally) but the
    // ColumnBeliefReport rate computed below would still be 0 for any
    // Str column the user didn't manually mask. That's the missing half
    // of the §4.D Part 1 fix.
    let (auto_masks, _auto_findings) =
        crate::validation::detect_string_sentinels(df, &opts.config);
    let effective_masks =
        crate::validation::merge_null_mask_maps(&opts.null_masks, &auto_masks);

    // Pre-bucket findings by column once. The prior code re-scanned every
    // finding for each column — O(C·F). For wide dataframes with many
    // findings this is the dominant cost in the api wrapper (especially
    // visible in the 1M-row scale benchmark, where ns/row went super-linear).
    // Bucketing once: O(F) pass + per-bucket sort, identical sort_key ordering
    // → bit-identical column_reports[col].findings.
    let mut findings_by_col: BTreeMap<&str, Vec<ValidationFinding>> = BTreeMap::new();
    for f in &findings {
        if let Some(col_name) = f.column.as_deref() {
            findings_by_col.entry(col_name).or_default().push(f.clone());
        }
    }
    for v in findings_by_col.values_mut() {
        v.sort_by(|a, b| a.sort_key().cmp(&b.sort_key()));
    }

    let mut column_reports: BTreeMap<String, ColumnBeliefReport> = BTreeMap::new();
    for (name, col) in &df.columns {
        // `get().cloned()` rather than `remove()` so duplicate-named
        // columns (legal in cjc_data::DataFrame::from_columns) see the
        // same bucket on each visit — preserving the prior behaviour
        // where each iteration re-derived the full filtered list.
        let col_findings: Vec<ValidationFinding> = findings_by_col
            .get(name.as_str())
            .cloned()
            .unwrap_or_default();
        let n_total = col.len() as f64;
        // Missingness rate: Float NaN union with effective-mask
        // positions; non-Float uses effective mask only.
        let missing = {
            let col_len = col.len();
            let mask_count = effective_masks
                .get(name)
                .map(|m| m.null_rows.iter().filter(|i| **i < col_len).count())
                .unwrap_or(0);
            // Union: for Float, |NaN ∪ Mask| = |NaN| + |Mask positions that
            // are NOT also NaN|. Counted in O(n + m) with zero heap; the
            // previous implementation allocated a fresh BTreeSet<usize> per
            // column and scanned NaNs twice. Bit-identical union size.
            match col {
                cjc_data::Column::Float(v) => {
                    let n_nan = v.iter().filter(|x| x.is_nan()).count();
                    let extra = effective_masks
                        .get(name)
                        .map(|m| {
                            m.null_rows
                                .iter()
                                .filter(|&&r| r < v.len() && !v[r].is_nan())
                                .count()
                        })
                        .unwrap_or(0);
                    n_nan + extra
                }
                _ => mask_count,
            }
        } as f64;
        let missingness_rate = if n_total == 0.0 { 0.0 } else { missing / n_total };
        // Single-pass distinct + top-freq. Replaces two back-to-back full
        // column scans that each built their own per-value count map.
        let (distinct, top_freq) = crate::validation::distinct_count_and_top_freq(col);
        let constant = distinct <= 1;
        let near_constant = if n_total > 0.0 {
            top_freq as f64 / n_total >= opts.config.near_constant_threshold
        } else {
            false
        };
        column_reports.insert(
            name.clone(),
            ColumnBeliefReport {
                column: name.clone(),
                findings: col_findings,
                missingness_rate,
                distinct_count: distinct,
                constant,
                near_constant,
            },
        );
    }

    let column_types: BTreeMap<String, String> = df
        .columns
        .iter()
        .map(|(n, c)| (n.clone(), c.type_name().to_string()))
        .collect();
    let input = LockeInputSummary {
        dataset_label: opts.dataset_label.clone(),
        n_rows: df.nrows() as u64,
        n_cols: df.ncols() as u64,
        column_types,
    };
    let mut assumptions = vec![
        "NaN treated as missing for Float columns; other types report a limitation".into(),
        "missingness, drift, and belief use deterministic Kahan summation".into(),
        "duplicate detection is byte-canonical".into(),
    ];
    assumptions.extend(custom_outcome.assumptions.iter().cloned());

    let mut report = LockeReport::new(input, findings, column_reports, assumptions);
    report.custom_axis_assignments = custom_outcome.axis_assignments;

    // v0.7+ (A2-by-default) — optionally attach per-value canonicalisation
    // lineage. Default off (preserves byte-identical v0.7 reports); CLI
    // exposes this via `cjcl locke validate --with-trace`.
    if opts.config.collect_per_value_lineage {
        let lineage_cfg = crate::per_value_lineage::PerValueLineageConfig::default();
        let lineage = crate::per_value_lineage::build_per_value_lineage(df, &lineage_cfg);
        report = report.with_per_value_lineage(lineage);
    }

    report
}

/// Compute a `BeliefReport` from a `LockeReport`. Pure function of the
/// report — does not re-scan the dataframe.
pub fn belief_report_from_locke(report: &LockeReport) -> BeliefReport {
    belief_report_from_locke_with_model(report, &BeliefPenalty::default())
}

/// v0.3: user-tunable variant. Same shape as `belief_report_from_locke`
/// but takes an explicit `BeliefPenalty` model.
///
/// **v0.7 part 2 (ADR-0036)** — the final [`BeliefScore`] is constructed via
/// [`algebra::compose_many`](crate::algebra::compose_many) over 8 per-axis
/// partials under [`BeliefAxisRules::default()`] (all-`Min`), making the
/// meet-semilattice algebra explicit at the construction site. The result
/// is byte-identical to the pre-migration direct
/// [`BeliefScore::from_dimensions`] call on the same 8-tuple — proptest-locked
/// in `tests/locke/locke_proptest.rs`. The inline path is preserved as
/// [`__belief_report_from_locke_inline_for_regression_test`] for the
/// byte-identity regression gate.
pub fn belief_report_from_locke_with_model(
    report: &LockeReport,
    penalty: &BeliefPenalty,
) -> BeliefReport {
    let (
        schema_score,
        missingness_score,
        drift_score,
        leakage_score,
        lineage_score,
        sample_score,
        duplication_score,
        constraint_score,
    ) = belief_axis_scores_from_report(report, penalty);

    // v0.7 part 2 — route final BeliefScore construction through the
    // algebra. Each per-axis partial carries one axis's computed value
    // and 1.0 (the all-Min identity ⊤) on the other seven. Under
    // `BeliefAxisRules::default()` (all-Min), `compose_many` reduces
    // the 8 partials to a single BeliefScore whose i-th axis is
    // `min(axis_i, 1.0, 1.0, ..., 1.0) = axis_i`. The last step inside
    // `compose_many` calls `BeliefScore::from_dimensions` with that
    // 8-tuple, in the same order as the pre-migration direct call —
    // hence the result is byte-identical, not just numerically close.
    // Proof-of-invariant: `algebra_path_is_byte_identical_to_direct_from_dimensions`
    // in `tests/locke/locke_proptest.rs`.
    let rules = BeliefAxisRules::default();
    let partials = [
        BeliefScore::from_dimensions(schema_score, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        BeliefScore::from_dimensions(1.0, missingness_score, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        BeliefScore::from_dimensions(1.0, 1.0, drift_score, 1.0, 1.0, 1.0, 1.0, 1.0),
        BeliefScore::from_dimensions(1.0, 1.0, 1.0, leakage_score, 1.0, 1.0, 1.0, 1.0),
        BeliefScore::from_dimensions(1.0, 1.0, 1.0, 1.0, lineage_score, 1.0, 1.0, 1.0),
        BeliefScore::from_dimensions(1.0, 1.0, 1.0, 1.0, 1.0, sample_score, 1.0, 1.0),
        BeliefScore::from_dimensions(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, duplication_score, 1.0),
        BeliefScore::from_dimensions(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, constraint_score),
    ];
    let score = compose_many(&partials, &rules)
        .expect("8 non-empty partials always compose under any rule");

    finish_belief_report(report, score)
}

/// Compute the 8 per-axis scores from a `LockeReport` + penalty model.
/// Shared between the migrated and inline paths so the byte-identity
/// regression test exercises only the construction-mode divergence
/// (compose vs from_dimensions), not the per-axis derivation logic.
///
/// **v0.8 (ADR-0041)** — also routes custom-detector findings to their
/// declared belief axes via `report.custom_axis_assignments`. When no
/// custom detectors are registered, the map is empty and the per-axis
/// scores are byte-identical to pre-v0.8.
fn belief_axis_scores_from_report(
    report: &LockeReport,
    penalty: &BeliefPenalty,
) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    let n = report.input.n_rows;

    // Helper closure: does this code's custom-axis assignment contain
    // the named axis? Empty map → always false → backward-compatible.
    let custom_contains = |code: &str, axis: crate::custom_detector::BeliefAxisSet| -> bool {
        report
            .custom_axis_assignments
            .get(code)
            .map(|axes| axes.contains(axis))
            .unwrap_or(false)
    };

    let missingness_score = {
        // Built-in mean-rate computation across column_reports stays the
        // baseline. Custom findings on the missingness axis stack a
        // penalty on top via the standard model.
        let baseline = if report.column_reports.is_empty() {
            1.0
        } else {
            let total: f64 = report
                .column_reports
                .values()
                .map(|c| c.missingness_rate)
                .sum();
            1.0 - (total / report.column_reports.len() as f64)
        };
        let custom_penalty = penalty_from_findings_with_model(
            &report.findings,
            |code| custom_contains(code, crate::custom_detector::BeliefAxisSet::MISSINGNESS),
            penalty,
        );
        (baseline - custom_penalty).clamp(0.0, 1.0)
    };
    let duplication_score = 1.0
        - penalty_from_findings_with_model(
            &report.findings,
            |code| {
                code == "E9003"
                    || code == "E9004"
                    || custom_contains(code, crate::custom_detector::BeliefAxisSet::DUPLICATION)
            },
            penalty,
        );
    let schema_score = 1.0
        - penalty_from_findings_with_model(
            &report.findings,
            |code| {
                // True schema-shape codes
                code == "E9020" || code == "E9021" || code == "E9022"
                // v0.6 batch 2: label-encoding risk weakens schema (the
                // column's declared type is misleading for downstream).
                || code == "E9023"
                // v0.6: semantic-category fragmentation (encoding-risk
                // E9017, case-fold E9080, whitespace E9081, near-duplicate
                // E9082, confusable-script E9083, mojibake E9084,
                // transitive-cluster E9085, NFC/NFD E9086). These weaken
                // the schema axis because the column's effective alphabet
                // is ambiguous, even though the row-level types are fine.
                || code == "E9017"
                || code == "E9080" || code == "E9081" || code == "E9082"
                || code == "E9083" || code == "E9084" || code == "E9085"
                || code == "E9086"
                || custom_contains(code, crate::custom_detector::BeliefAxisSet::SCHEMA)
            },
            penalty,
        );
    let constraint_score = 1.0
        - penalty_from_findings_with_model(
            &report.findings,
            // E9014 (legacy constraint) + E9016 (rare-category long-tail
            // is a distributional constraint risk, not a schema-shape one)
            // + v0.6 batch 2 PII (E9090-E9093 — presence of PII is a
            // constraint violation per data-governance policy).
            |code| {
                code == "E9014" || code == "E9016"
                || code == "E9090" || code == "E9091" || code == "E9092" || code == "E9093"
                || custom_contains(code, crate::custom_detector::BeliefAxisSet::CONSTRAINT)
            },
            penalty,
        );
    // Drift / leakage / lineage scores are 1.0 by default in the
    // single-df flow; custom detectors are the FIRST way for these axes
    // to be touched without a compare/lineage pass. Apply the standard
    // penalty model where the custom map routes findings to the axis.
    let drift_score = 1.0
        - penalty_from_findings_with_model(
            &report.findings,
            |code| custom_contains(code, crate::custom_detector::BeliefAxisSet::DRIFT),
            penalty,
        );
    let leakage_score = 1.0
        - penalty_from_findings_with_model(
            &report.findings,
            |code| custom_contains(code, crate::custom_detector::BeliefAxisSet::LEAKAGE),
            penalty,
        );
    let lineage_score = 1.0
        - penalty_from_findings_with_model(
            &report.findings,
            |code| custom_contains(code, crate::custom_detector::BeliefAxisSet::LINEAGE),
            penalty,
        );
    let sample_score = {
        let baseline = sample_score_from_n(n);
        let custom_penalty = penalty_from_findings_with_model(
            &report.findings,
            |code| custom_contains(code, crate::custom_detector::BeliefAxisSet::SAMPLE),
            penalty,
        );
        (baseline - custom_penalty).clamp(0.0, 1.0)
    };

    (
        schema_score,
        missingness_score,
        drift_score,
        leakage_score,
        lineage_score,
        sample_score,
        duplication_score,
        constraint_score,
    )
}

/// Build the surrounding `BeliefReport` (assumptions, evidence summary,
/// recommended next steps) from a finished `BeliefScore`. Shared between
/// the migrated and inline paths.
fn finish_belief_report(report: &LockeReport, score: BeliefScore) -> BeliefReport {
    let mut assumptions = report.assumptions.clone();
    // If sub-scores defaulted to 1.0 due to absent evidence, say so.
    assumptions
        .push("drift_score = 1.0 by default (no comparison dataframe supplied)".into());
    assumptions
        .push("leakage_score = 1.0 by default (Locke v0 does not infer leakage automatically)".into());
    assumptions
        .push("lineage_score = 1.0 by default (no lineage graph supplied)".into());

    let evidence_summary: BTreeMap<String, String> = report
        .severity_counts
        .clone()
        .into_iter_named()
        .into_iter()
        .collect();

    let mut next = Vec::new();
    if score.missingness_score < 0.9 {
        next.push("address missingness with imputation or filtering".into());
    }
    if score.duplication_score < 0.9 {
        next.push("deduplicate before training or analysis".into());
    }
    if score.schema_score < 1.0 {
        next.push("align actual schema with expected schema".into());
    }
    if score.constraint_score < 1.0 {
        next.push("triage impossible-value violations".into());
    }

    BeliefReport::new(score, assumptions, evidence_summary, next)
}

/// **Test-only oracle.** Reproduces the pre-v0.7-part-2 inline implementation
/// of [`belief_report_from_locke_with_model`] — direct
/// [`BeliefScore::from_dimensions`] construction instead of the algebra
/// path. Exposed as `#[doc(hidden)] pub` exclusively as a byte-identity
/// reference oracle for the v0.7 part 2 algebra-migration regression
/// proptest (`tests/locke/locke_proptest.rs`).
///
/// Not part of the public Locke API; will be removed once the algebra
/// migration has shipped for at least one stable release without regression.
#[doc(hidden)]
pub fn __belief_report_from_locke_inline_for_regression_test(
    report: &LockeReport,
    penalty: &BeliefPenalty,
) -> BeliefReport {
    let (
        schema_score,
        missingness_score,
        drift_score,
        leakage_score,
        lineage_score,
        sample_score,
        duplication_score,
        constraint_score,
    ) = belief_axis_scores_from_report(report, penalty);

    let score = BeliefScore::from_dimensions(
        schema_score,
        missingness_score,
        drift_score,
        leakage_score,
        lineage_score,
        sample_score,
        duplication_score,
        constraint_score,
    );

    finish_belief_report(report, score)
}

impl crate::report::SeverityCounts {
    fn into_iter_named(self) -> Vec<(String, String)> {
        vec![
            ("info".into(), self.info.to_string()),
            ("notice".into(), self.notice.to_string()),
            ("warning".into(), self.warning.to_string()),
            ("error".into(), self.error.to_string()),
        ]
    }
}

/// Compose validate + drift + belief into one composite invocation.
///
/// **v0.7+ A1 follow-through** — this is a genuine second consumer of
/// [`algebra::compose`](crate::algebra::compose). Previously the train
/// belief and the drift signal were merged by directly overwriting the
/// drift axis on the belief score (`BeliefScore::from_dimensions(...,
/// drift_score, ...)`). Under the meet-semilattice algebra the same
/// behaviour is expressed declaratively as `compose(train_belief,
/// drift_only_partial, all_min)`: every other axis takes `min(x, 1.0) = x`
/// (unchanged), and the drift axis takes `min(train.drift, drift_score)`.
/// Train's drift score defaults to 1.0 when no comparison ran (see
/// [`belief_report_from_locke_with_model`]), so the result on the drift
/// axis is `drift_score` — byte-identical to the pre-migration path,
/// but now traceable through the algebra. Proptest-locked in
/// `tests/locke/locke_proptest.rs::validate_and_compare_drift_composition_is_byte_identical`.
///
/// **Cascade semantics (v0.7+ B5.4)**: the drift axis under the
/// meet-semilattice algebra is **monotonically downward** — composing
/// against another partial floor a previous value, it never recovers.
/// `validate_and_compare` as written always starts from a fresh
/// `belief_report_from_locke(val_report)`, so cascading happens only
/// across separate calls *with the same train data* (different test
/// data); each call independently rebuilds the train belief, so the
/// drift axis does not chain through this API directly. Callers that
/// stitch multiple `(train, test_i)` comparisons together by hand
/// — `compose(prior_belief, new_compare_result.belief, all_min)` —
/// will see the drift axis monotonically floor to the worst observed
/// `drift_score` across the chain; that is the intended
/// meet-semilattice contract, not a bug. See
/// `tests/locke/locke_proptest.rs` and `algebra::tests` for the
/// monotonicity laws.
pub fn validate_and_compare(
    train: &DataFrame,
    test: &DataFrame,
    opts: &ValidateOptions,
    drift_cfg: &DriftConfig,
) -> (LockeReport, InductionRiskReport, BeliefReport) {
    let val_report = validate(train, opts);
    let drift_report = compare_drift(train, test, drift_cfg);

    // Update belief to incorporate drift signal via the meet-semilattice
    // algebra. The drift signal is a "drift-only partial": top on every
    // axis except drift, which carries the derived score. Under all-Min
    // composition this collapses to "train belief, except drift axis =
    // min(train.drift, drift_score)" — i.e. the same construction as
    // the pre-A1 direct-assignment path, but visibly using the algebra.
    let mut belief = belief_report_from_locke(&val_report);
    let drift_penalty = penalty_from_findings(&drift_report.findings, |code| {
        code.starts_with("E903")
    });
    let drift_score = (1.0 - drift_penalty).clamp(0.0, 1.0);
    let drift_only_partial =
        BeliefScore::from_dimensions(1.0, 1.0, drift_score, 1.0, 1.0, 1.0, 1.0, 1.0);
    let rules = crate::algebra::BeliefAxisRules::default();
    belief.score = crate::algebra::compose(&belief.score, &drift_only_partial, &rules);
    belief
        .assumptions
        .retain(|a| !a.starts_with("drift_score = 1.0"));
    belief
        .assumptions
        .push(format!("drift_score derived from comparison ({} drift findings)", drift_report.findings.len()));
    (val_report, drift_report, belief)
}

/// Compute pairwise Pearson correlations between every pair of numeric
/// columns, then route through the causal-guardrail audit.
pub fn causal_guardrail(
    df: &DataFrame,
    target_column: Option<&str>,
    causal_cfg: &CausalConfig,
    label_text: Option<&str>,
    interpret_model_explanation_as_causal: bool,
) -> CausalGuardrailReport {
    let mut numeric_cols: Vec<(String, Vec<f64>)> = Vec::new();
    for (name, col) in &df.columns {
        let v: Option<Vec<f64>> = match col {
            cjc_data::Column::Float(v) => Some(v.clone()),
            cjc_data::Column::Int(v) => Some(v.iter().map(|x| *x as f64).collect()),
            _ => None,
        };
        if let Some(v) = v {
            numeric_cols.push((name.clone(), v));
        }
    }
    numeric_cols.sort_by(|a, b| a.0.cmp(&b.0));

    let mut correlations: Vec<CorrelationFinding> = Vec::new();
    for i in 0..numeric_cols.len() {
        for j in i + 1..numeric_cols.len() {
            let (an, av) = &numeric_cols[i];
            let (bn, bv) = &numeric_cols[j];
            if let Some(r) = pearson_correlation(av, bv) {
                correlations.push(CorrelationFinding {
                    a: an.clone(),
                    b: bn.clone(),
                    r,
                    n_pairs: av.len() as u64,
                });
            }
        }
    }
    audit_correlations(
        &correlations,
        target_column,
        causal_cfg,
        label_text,
        interpret_model_explanation_as_causal,
    )
}

/// Build a minimal lineage graph that records a single dataset
/// impression. Mostly a smoke-test helper for the CLI `lineage` command;
/// real pipelines call `LineageBuilder` directly.
pub fn lineage_for_dataset(label: &str, df: &DataFrame) -> LineageGraph {
    let mut b = LineageBuilder::new("validate-run");
    let names: Vec<String> = df.columns.iter().map(|(n, _)| n.clone()).collect();
    let imp = LockeImpression::new(label, ImpressionKind::Dataset, df.nrows() as u64, names);
    let _ = b.add_impression(imp);
    b.finish()
}

/// Worst-severity summary of all findings + drift findings.
pub fn worst_severity(report: &LockeReport, drift: Option<&InductionRiskReport>) -> FindingSeverity {
    let mut sev = report.worst_severity();
    if let Some(d) = drift {
        sev = sev.max(d.worst_severity());
    }
    sev
}

