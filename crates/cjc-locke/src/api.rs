//! High-level facade used by the CLI and (later) the CJC-Lang binding.
//!
//! The facade composes validators + drift + belief + causal into a
//! single function-call surface, returning fully-deterministic reports.

use std::collections::BTreeMap;

use cjc_data::DataFrame;

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
#[derive(Clone, Debug, Default)]
pub struct ValidateOptions {
    pub dataset_label: String,
    pub config: ValidationConfig,
    pub impossible_rules: Vec<ImpossibleValueRule>,
    pub expected_schema: Option<ExpectedSchema>,
    pub primary_key: Option<String>,
    pub null_masks: NullMaskMap,
}

/// Build a `LockeReport` from a single dataframe.
pub fn validate(df: &DataFrame, opts: &ValidateOptions) -> LockeReport {
    let findings = validate_dataframe(
        df,
        &opts.config,
        &opts.impossible_rules,
        opts.expected_schema.as_ref(),
        opts.primary_key.as_deref(),
        &opts.null_masks,
    );

    let mut column_reports: BTreeMap<String, ColumnBeliefReport> = BTreeMap::new();
    for (name, col) in &df.columns {
        let mut col_findings: Vec<ValidationFinding> = findings
            .iter()
            .filter(|f| f.column.as_deref() == Some(name.as_str()))
            .cloned()
            .collect();
        col_findings.sort_by(|a, b| a.sort_key().cmp(&b.sort_key()));
        let n_total = col.len() as f64;
        // Missingness rate: Float NaN union with null-mask positions; non-float
        // uses mask only.
        let missing = {
            let col_len = col.len();
            let mask_count = opts
                .null_masks
                .get(name)
                .map(|m| m.null_rows.iter().filter(|i| **i < col_len).count())
                .unwrap_or(0);
            let nan_count = match col {
                cjc_data::Column::Float(v) => v.iter().filter(|x| x.is_nan()).count(),
                _ => 0,
            };
            // Union estimate: for Float, true union via BTreeSet would be more
            // accurate, but for the per-column rate this small overcount is
            // bounded by the column length and acceptable.
            match col {
                cjc_data::Column::Float(v) => {
                    let mut s: std::collections::BTreeSet<usize> =
                        (0..v.len()).filter(|i| v[*i].is_nan()).collect();
                    if let Some(m) = opts.null_masks.get(name) {
                        for r in &m.null_rows {
                            if *r < v.len() {
                                s.insert(*r);
                            }
                        }
                    }
                    s.len()
                }
                _ => {
                    let _ = nan_count;
                    mask_count
                }
            }
        } as f64;
        let missingness_rate = if n_total == 0.0 { 0.0 } else { missing / n_total };
        let distinct = crate::validation::distinct_count(col);
        let constant = distinct <= 1;
        let near_constant = if n_total > 0.0 {
            crate::validation::top_value_freq(col) as f64 / n_total
                >= opts.config.near_constant_threshold
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
    let assumptions = vec![
        "NaN treated as missing for Float columns; other types report a limitation".into(),
        "missingness, drift, and belief use deterministic Kahan summation".into(),
        "duplicate detection is byte-canonical".into(),
    ];

    LockeReport::new(input, findings, column_reports, assumptions)
}

/// Compute a `BeliefReport` from a `LockeReport`. Pure function of the
/// report — does not re-scan the dataframe.
pub fn belief_report_from_locke(report: &LockeReport) -> BeliefReport {
    belief_report_from_locke_with_model(report, &BeliefPenalty::default())
}

/// v0.3: user-tunable variant. Same shape as `belief_report_from_locke`
/// but takes an explicit `BeliefPenalty` model.
pub fn belief_report_from_locke_with_model(
    report: &LockeReport,
    penalty: &BeliefPenalty,
) -> BeliefReport {
    let n = report.input.n_rows;

    let missingness_score = {
        // Mean missingness rate across columns reported.
        if report.column_reports.is_empty() {
            1.0
        } else {
            let total: f64 = report
                .column_reports
                .values()
                .map(|c| c.missingness_rate)
                .sum();
            1.0 - (total / report.column_reports.len() as f64)
        }
    };
    let duplication_score = 1.0
        - penalty_from_findings_with_model(
            &report.findings,
            |code| code == "E9003" || code == "E9004",
            penalty,
        );
    let schema_score = 1.0
        - penalty_from_findings_with_model(
            &report.findings,
            |code| {
                // True schema-shape codes
                code == "E9020" || code == "E9021" || code == "E9022"
                // v0.6: semantic-category fragmentation (encoding-risk,
                // case-fold, whitespace, near-duplicate). These weaken the
                // schema axis because the column's effective alphabet is
                // ambiguous, even though the row-level types are fine.
                || code == "E9011" || code == "E9080" || code == "E9081" || code == "E9082"
            },
            penalty,
        );
    let constraint_score = 1.0
        - penalty_from_findings_with_model(
            &report.findings,
            // E9014 (legacy constraint) + E9010 (rare-category long-tail
            // is a distributional constraint risk, not a schema-shape one).
            |code| code == "E9014" || code == "E9010",
            penalty,
        );
    // Drift / leakage / lineage scores are 1.0 here (no signal in single-df flow);
    // they get populated when the caller composes `validate` + `compare` + lineage.
    let drift_score = 1.0;
    let leakage_score = 1.0;
    let lineage_score = 1.0;
    let sample_score = sample_score_from_n(n);

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
pub fn validate_and_compare(
    train: &DataFrame,
    test: &DataFrame,
    opts: &ValidateOptions,
    drift_cfg: &DriftConfig,
) -> (LockeReport, InductionRiskReport, BeliefReport) {
    let val_report = validate(train, opts);
    let drift_report = compare_drift(train, test, drift_cfg);

    // Update belief to incorporate drift signal.
    let mut belief = belief_report_from_locke(&val_report);
    let drift_penalty = penalty_from_findings(&drift_report.findings, |code| {
        code.starts_with("E903")
    });
    let drift_score = (1.0 - drift_penalty).clamp(0.0, 1.0);
    belief.score = BeliefScore::from_dimensions(
        belief.score.schema_score,
        belief.score.missingness_score,
        drift_score,
        belief.score.leakage_score,
        belief.score.lineage_score,
        belief.score.sample_score,
        belief.score.duplication_score,
        belief.score.constraint_score,
    );
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

