//! # cjc-locke — evidence-aware analytics for CJC-Lang
//!
//! Locke is the data-skepticism layer for CJC-Lang. It separates *observed
//! facts* (impressions) from *derived claims* (ideas), validates data
//! quality, detects induction risks, preserves lineage, and produces
//! confidence-aware reports. **All output is deterministic** — repeated
//! runs over the same inputs produce byte-identical findings, IDs, and
//! reports.
//!
//! ## What Locke v0 does
//!
//! * detects missingness, duplicates, impossible values, schema mismatches,
//!   constant/near-constant columns, suspicious cardinality
//! * compares train vs test for mean / std / range / missingness shift,
//!   PSI-like numeric drift, TVD categorical drift
//! * builds an acyclic lineage DAG with deterministic content-addressed IDs
//! * produces a `BeliefReport` with explicit per-dimension score breakdown
//! * emits conservative causal warnings (never certifies causation)
//!
//! ## What Locke v0 does NOT do
//!
//! * causal inference (only flags correlation-as-causation risk)
//! * leakage detection beyond a hint heuristic
//! * model interpretability
//! * exact KS/CDF tests (deferred — see `drift` module docs)
//!
//! ## Quick start
//!
//! ```no_run
//! use cjc_locke::api::{validate, ValidateOptions};
//! use cjc_locke::validation::ValidationConfig;
//! # let df = cjc_data::DataFrame::new();
//! let opts = ValidateOptions {
//!     dataset_label: "train.csv".into(),
//!     config: ValidationConfig::default(),
//!     ..Default::default()
//! };
//! let report = validate(&df, &opts);
//! println!("worst severity: {}", report.worst_severity());
//! ```

pub mod api;
pub mod belief;
pub mod categorical;
pub mod causal;
pub mod dispatch;
pub mod drift;
pub mod gate;
pub mod html_emit;
pub mod id;
pub mod json_emit;
pub mod leakage;
pub mod parquet_reader;
pub mod lineage;
pub mod report;
pub mod stats;
pub mod streaming;
pub mod temporal;
pub mod traced;
pub mod validation;

pub use dispatch::dispatch_locke;
pub use gate::{diff_reports, emit_diff_text, ReportDiff};
pub use html_emit::{emit_locke_report_html, emit_locke_report_html_with_df};
pub use json_emit::{emit_locke_report_json, parse_locke_report_json};
pub use streaming::{
    validate_view, StreamingColumnSummary, StreamingConfig, StreamingValidator,
};

// Convenience re-exports — the most commonly used types.
pub use belief::{BeliefPenalty, BeliefReport, BeliefScore, BeliefWeights};
pub use causal::{
    CausalClaim, CausalConfig, CausalDag, CausalDagError, CausalDirection,
    CausalGuardrailReport, CausalMode, CausalWarning, ConfounderHint, CorrelationFinding,
};
pub use drift::{compare, DriftConfig, InductionRiskReport};
pub use id::{FingerprintId, IdDomain};
pub use lineage::{
    AuditEvent, ImpressionKind, LineageBuilder, LineageEdge, LineageGraph, LineageNode,
    LockeIdea, LockeImpression, TransformationRecord,
};
pub use traced::TracedDataFrame;
pub use report::{
    ColumnBeliefReport, DatasetSkepticismReport, FindingEvidence, FindingSeverity,
    LockeInputSummary, LockeReport, SeverityCounts, ValidationFinding,
};
pub use validation::{
    detect_conditional_missingness, detect_constant_and_near_constant,
    detect_duplicate_key_conditioning, detect_duplicate_keys, detect_duplicates_full_row,
    detect_high_cardinality_categorical, detect_imbalanced_target, detect_impossible_values,
    detect_missingness, detect_outliers, detect_schema_mismatch, detect_sentinel_values,
    validate_dataframe, ConditionalMissingnessConfig, ExpectedSchema, ImpossibleValueRule,
    NullMask, NullMaskMap, OutlierConfig, SentinelConfig, ValidationConfig,
};
pub use categorical::{
    detect_all_categorical_quality, detect_case_fold_collisions, detect_encoding_risk,
    detect_near_duplicate_categories, detect_rare_categories,
    detect_whitespace_punctuation_variants, CategoricalQualityConfig,
};
pub use api::{
    belief_report_from_locke, belief_report_from_locke_with_model, causal_guardrail,
    lineage_for_dataset, validate, validate_and_compare, worst_severity, ValidateOptions,
};

/// Locke version — exported as a string for inclusion in reports.
pub const LOCKE_VERSION: &str = env!("CARGO_PKG_VERSION");
