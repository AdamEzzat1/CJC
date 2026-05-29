//! Per-value category lineage (v0.7+ heavy, A2).
//!
//! Where [`crate::traced::TracedDataFrame`] records lineage at the
//! **DataFrame** level — "filter then group_by then summarise" as Idea
//! nodes parented by Impressions — this module records lineage at the
//! **per-distinct-value** level: for each `(column, original_value)`
//! pair, the chain of canonicalisation stages Locke would apply if the
//! user adopted its suggested normalisations.
//!
//! ## What this answers
//!
//! Concretely: "given value V in column C, what canonical forms did it
//! pass through (and were grouped with other distinct originals)?"
//! The diabetes-130 flagship case: the column `weight` is 96.9% `?` —
//! v0.6.4's E9008 auto-detects this and folds the rows into the null
//! mask, but until this module, a user could not *trace* what each
//! `?` became. Now they can: `cjcl locke trace-value weight ?` shows
//! `stage 1: sentinel_mask (E9008) → null`.
//!
//! ## What it does NOT do
//!
//! The lineage is computed from the *current* column contents; Locke is
//! still a skepticism layer, not a data-cleaning layer. The trace
//! describes *what would happen if* the user adopted the canonical
//! forms — it does not rewrite the DataFrame.
//!
//! ## Determinism contract
//!
//! - The output [`PerValueLineageMap`] is a `BTreeMap` keyed by
//!   `(column, original_value)` — iteration in sorted order across both
//!   keys.
//! - Each [`PerValueLineage::stages`] is a `Vec` in fixed code order
//!   (sentinel → case_fold → whitespace_punct → unicode_normalize →
//!   rare_candidate) — never permuted by collection iteration order.
//! - [`LineageStage::siblings`] is a `BTreeSet<String>` — sorted output.
//! - Canonicalisation helpers (`normalize_whitespace_punct`,
//!   `strip_combining_marks`) come from `crate::categorical`, so the
//!   per-value trace is *guaranteed* to apply the same transforms as
//!   the corresponding detector. No drift risk.
//!
//! ## Boundedness
//!
//! For columns with very wide alphabets, the lineage map size is bounded
//! by `cfg.max_distinct_per_column` to keep memory tractable. Columns
//! exceeding the limit emit a single skipped-column note in the lineage
//! map under the synthetic key `(column, "__skipped__")` (a transform
//! with no canonical and a single "too_many_distinct_values" stage).

use std::collections::{BTreeMap, BTreeSet};

use cjc_data::DataFrame;

use crate::categorical::{
    category_counts, normalize_whitespace_punct, strip_combining_marks,
};
use crate::validation::BUILTIN_STRING_SENTINELS;

// ─── Config ───────────────────────────────────────────────────────────────

/// Which canonicalisation stages to record for each value.
///
/// `Default::default()` enables all five.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PerValueStageSet {
    /// E9008 — exact-match against `BUILTIN_STRING_SENTINELS` plus
    /// `cfg.additional_sentinels`. Maps matching values to null.
    pub sentinel: bool,
    /// E9080 — Unicode-aware lowercase via `str::to_lowercase`.
    pub case_fold: bool,
    /// E9081 — trim + strip trailing characters in `cfg.trim_terminal_chars`
    /// + lowercase. Strict refinement of `case_fold`.
    pub whitespace_punct: bool,
    /// E9086 — strip Unicode combining marks + map Latin-1/Latin-Extended-A
    /// precomposed letters to their ASCII bases.
    pub unicode_normalize: bool,
    /// E9016 — tag values whose count is below `cfg.rare_threshold`.
    /// Not a transform; produces no canonical.
    pub rare_candidate: bool,
}

impl Default for PerValueStageSet {
    fn default() -> Self {
        Self {
            sentinel: true,
            case_fold: true,
            whitespace_punct: true,
            unicode_normalize: true,
            rare_candidate: true,
        }
    }
}

/// Knobs controlling per-value lineage construction.
#[derive(Clone, Debug)]
pub struct PerValueLineageConfig {
    pub stages: PerValueStageSet,
    /// Skip columns with more distinct values than this. Default `10_000`.
    /// Columns above the cap appear in the output under the synthetic key
    /// `(column, "__skipped__")` so the caller knows the column was
    /// considered but bypassed.
    pub max_distinct_per_column: usize,
    /// Forwarded to the sentinel stage. Empty by default.
    pub additional_sentinels: Vec<String>,
    /// Forwarded to the whitespace_punct stage. Default `".,!?;:"` to
    /// match `CategoricalQualityConfig::default()`.
    pub trim_terminal_chars: &'static str,
    /// Forwarded to the rare_candidate stage. Default `5` (matches
    /// `CategoricalQualityConfig::rare_category_min_count`).
    pub rare_threshold: u64,
}

impl Default for PerValueLineageConfig {
    fn default() -> Self {
        Self {
            stages: PerValueStageSet::default(),
            max_distinct_per_column: 10_000,
            additional_sentinels: Vec::new(),
            trim_terminal_chars: ".,!?;:",
            rare_threshold: 5,
        }
    }
}

// ─── Data model ───────────────────────────────────────────────────────────

/// Type of canonicalisation a single stage records.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueTransform {
    /// E9008 — value exactly matches a sentinel string; result is null.
    SentinelMask { sentinel: String },
    /// E9080 — Unicode-aware lowercase via `str::to_lowercase`.
    CaseFold,
    /// E9081 — trim + strip trailing punctuation + lowercase.
    WhitespacePunctStrip { terminal_chars: String },
    /// E9086 — strip Unicode combining marks + Latin precomposed → ASCII base.
    UnicodeNormalize,
    /// E9016 — value's count is below the configured rare threshold.
    /// Produces no canonical form; tag only.
    RareCandidate { count: u64, threshold: u64 },
    /// Synthetic stage emitted when a column is skipped because its
    /// distinct cardinality exceeds `cfg.max_distinct_per_column`.
    /// Produces no canonical form.
    TooManyDistinctValuesSkipped { n_distinct: usize, limit: usize },
}

impl ValueTransform {
    /// Stable transform name (used in emit + identifiers).
    pub fn name(&self) -> &'static str {
        match self {
            ValueTransform::SentinelMask { .. } => "sentinel_mask",
            ValueTransform::CaseFold => "case_fold",
            ValueTransform::WhitespacePunctStrip { .. } => "whitespace_punct_strip",
            ValueTransform::UnicodeNormalize => "unicode_normalize",
            ValueTransform::RareCandidate { .. } => "rare_candidate",
            ValueTransform::TooManyDistinctValuesSkipped { .. } => {
                "too_many_distinct_values_skipped"
            }
        }
    }

    /// Finding code the transform corresponds to (or `"-"` for synthetic stages).
    pub fn finding_code(&self) -> &'static str {
        match self {
            ValueTransform::SentinelMask { .. } => "E9008",
            ValueTransform::CaseFold => "E9080",
            ValueTransform::WhitespacePunctStrip { .. } => "E9081",
            ValueTransform::UnicodeNormalize => "E9086",
            ValueTransform::RareCandidate { .. } => "E9016",
            ValueTransform::TooManyDistinctValuesSkipped { .. } => "-",
        }
    }
}

/// One canonicalisation step in a per-value lineage chain.
#[derive(Clone, Debug, PartialEq)]
pub struct LineageStage {
    pub transform: ValueTransform,
    /// Canonical form produced. `None` when the transform produces null
    /// (sentinel mask) or no canonical (rare candidate, skipped).
    pub canonical: Option<String>,
    /// Other distinct originals from the same column that map to the
    /// same canonical form at this stage. Empty if the canonicalisation
    /// is unique to this value (no merge / collision).
    pub siblings: BTreeSet<String>,
}

/// Lineage chain for one `(column, original_value)` pair.
#[derive(Clone, Debug, PartialEq)]
pub struct PerValueLineage {
    pub column: String,
    pub original_value: String,
    pub stages: Vec<LineageStage>,
}

/// Per-value lineage map keyed by `(column, original_value)`.
/// `BTreeMap` → deterministic iteration order.
pub type PerValueLineageMap = BTreeMap<(String, String), PerValueLineage>;

// ─── Build the lineage ────────────────────────────────────────────────────

/// Build the per-value lineage map for every applicable column in `df`.
///
/// Iterates `df.columns` in insertion order. For each Str / Categorical /
/// CategoricalAdaptive column with ≤ `cfg.max_distinct_per_column`
/// distinct values, materialises every distinct value's lineage chain
/// under the configured [`PerValueStageSet`]. Columns exceeding the
/// limit emit one synthetic skipped entry; non-categorical columns are
/// silently skipped (Locke does not canonicalise numeric data).
pub fn build_per_value_lineage(
    df: &DataFrame,
    cfg: &PerValueLineageConfig,
) -> PerValueLineageMap {
    let mut out: PerValueLineageMap = BTreeMap::new();
    for (col_name, col) in &df.columns {
        let Some(counts) = category_counts(col) else { continue };
        if counts.len() > cfg.max_distinct_per_column {
            out.insert(
                (col_name.clone(), "__skipped__".into()),
                PerValueLineage {
                    column: col_name.clone(),
                    original_value: "__skipped__".into(),
                    stages: vec![LineageStage {
                        transform: ValueTransform::TooManyDistinctValuesSkipped {
                            n_distinct: counts.len(),
                            limit: cfg.max_distinct_per_column,
                        },
                        canonical: None,
                        siblings: BTreeSet::new(),
                    }],
                },
            );
            continue;
        }
        // Precompute the canonicalisation tables once per column so each
        // value's siblings can be derived in O(1) from a BTreeMap lookup.
        let case_fold_groups = group_by(&counts, |s| s.to_lowercase());
        let whitespace_groups = group_by(&counts, |s| {
            normalize_whitespace_punct(s, cfg.trim_terminal_chars)
        });
        let unicode_groups = group_by(&counts, |s| strip_combining_marks(s));

        for (original, &count) in &counts {
            let stages = lineage_stages_for(
                original,
                count,
                &counts,
                cfg,
                &case_fold_groups,
                &whitespace_groups,
                &unicode_groups,
            );
            // Always insert, even when stages is empty — the trace_value
            // CLI distinguishes "value exists but no canonicalisation
            // applied" (empty stages) from "value not present in column"
            // (returns None).
            out.insert(
                (col_name.clone(), original.clone()),
                PerValueLineage {
                    column: col_name.clone(),
                    original_value: original.clone(),
                    stages,
                },
            );
        }
    }
    out
}

/// Trace a single `(column, value)` pair through the same pipeline as
/// [`build_per_value_lineage`]. Returns `None` when the column doesn't
/// exist, isn't categorical, or the value isn't present.
pub fn trace_value(
    df: &DataFrame,
    cfg: &PerValueLineageConfig,
    column: &str,
    value: &str,
) -> Option<PerValueLineage> {
    let (_, col) = df.columns.iter().find(|(n, _)| n == column)?;
    let counts = category_counts(col)?;
    if counts.len() > cfg.max_distinct_per_column {
        return Some(PerValueLineage {
            column: column.into(),
            original_value: "__skipped__".into(),
            stages: vec![LineageStage {
                transform: ValueTransform::TooManyDistinctValuesSkipped {
                    n_distinct: counts.len(),
                    limit: cfg.max_distinct_per_column,
                },
                canonical: None,
                siblings: BTreeSet::new(),
            }],
        });
    }
    let count = *counts.get(value)?;
    let case_fold_groups = group_by(&counts, |s| s.to_lowercase());
    let whitespace_groups =
        group_by(&counts, |s| normalize_whitespace_punct(s, cfg.trim_terminal_chars));
    let unicode_groups = group_by(&counts, |s| strip_combining_marks(s));
    let stages = lineage_stages_for(
        value,
        count,
        &counts,
        cfg,
        &case_fold_groups,
        &whitespace_groups,
        &unicode_groups,
    );
    Some(PerValueLineage {
        column: column.into(),
        original_value: value.into(),
        stages,
    })
}

/// Compute the lineage stages list for a single value. Shared between
/// `build_per_value_lineage` and `trace_value`.
fn lineage_stages_for(
    original: &str,
    count: u64,
    counts: &BTreeMap<String, u64>,
    cfg: &PerValueLineageConfig,
    case_fold_groups: &BTreeMap<String, BTreeSet<String>>,
    whitespace_groups: &BTreeMap<String, BTreeSet<String>>,
    unicode_groups: &BTreeMap<String, BTreeSet<String>>,
) -> Vec<LineageStage> {
    let mut stages: Vec<LineageStage> = Vec::new();

    // Stage 1 — sentinel mask (terminates the chain on match).
    if cfg.stages.sentinel {
        if let Some(s) = matching_sentinel(original, &cfg.additional_sentinels) {
            stages.push(LineageStage {
                transform: ValueTransform::SentinelMask { sentinel: s.to_string() },
                canonical: None,
                siblings: BTreeSet::new(),
            });
            return stages;
        }
    }

    // Stage 2 — case fold.
    if cfg.stages.case_fold {
        let canonical = original.to_lowercase();
        if let Some(stage) = stage_if_meaningful(
            ValueTransform::CaseFold,
            original,
            canonical,
            case_fold_groups,
        ) {
            stages.push(stage);
        }
    }

    // Stage 3 — whitespace + terminal-punctuation strip + lowercase.
    if cfg.stages.whitespace_punct {
        let canonical = normalize_whitespace_punct(original, cfg.trim_terminal_chars);
        if !canonical.is_empty() {
            if let Some(stage) = stage_if_meaningful(
                ValueTransform::WhitespacePunctStrip {
                    terminal_chars: cfg.trim_terminal_chars.to_string(),
                },
                original,
                canonical,
                whitespace_groups,
            ) {
                stages.push(stage);
            }
        }
    }

    // Stage 4 — Unicode normalisation.
    if cfg.stages.unicode_normalize {
        let canonical = strip_combining_marks(original);
        if let Some(stage) = stage_if_meaningful(
            ValueTransform::UnicodeNormalize,
            original,
            canonical,
            unicode_groups,
        ) {
            stages.push(stage);
        }
    }

    // Stage 5 — rare-candidate flag (no canonical, no siblings beyond
    // self; tag-only). Emitted when count < threshold and the column
    // has at least one value with count >= threshold (otherwise every
    // value is rare and the tag isn't useful).
    if cfg.stages.rare_candidate
        && count < cfg.rare_threshold
        && counts.values().any(|c| *c >= cfg.rare_threshold)
    {
        stages.push(LineageStage {
            transform: ValueTransform::RareCandidate {
                count,
                threshold: cfg.rare_threshold,
            },
            canonical: None,
            siblings: BTreeSet::new(),
        });
    }

    stages
}

/// Group distinct values by a canonicalisation key. Returns
/// `canonical_form → set_of_originals`.
fn group_by<F>(
    counts: &BTreeMap<String, u64>,
    canonicalise: F,
) -> BTreeMap<String, BTreeSet<String>>
where
    F: Fn(&str) -> String,
{
    let mut groups: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for original in counts.keys() {
        let key = canonicalise(original);
        groups.entry(key).or_default().insert(original.clone());
    }
    groups
}

/// Build a `LineageStage` only when the transform is *meaningful* for
/// this value — either the canonical differs from the original, OR
/// the canonical groups this value with at least one other distinct
/// original. Returns `None` if the stage would be a no-op solo.
fn stage_if_meaningful(
    transform: ValueTransform,
    original: &str,
    canonical: String,
    groups: &BTreeMap<String, BTreeSet<String>>,
) -> Option<LineageStage> {
    let siblings: BTreeSet<String> = groups
        .get(&canonical)
        .map(|set| {
            set.iter()
                .filter(|s| s.as_str() != original)
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    let canonical_differs = canonical != original;
    if canonical_differs || !siblings.is_empty() {
        Some(LineageStage {
            transform,
            canonical: Some(canonical),
            siblings,
        })
    } else {
        None
    }
}

/// Return the sentinel that exactly matches `value`, if any. Builtin
/// list is consulted first, then custom additions.
fn matching_sentinel<'a>(
    value: &'a str,
    additional: &'a [String],
) -> Option<&'a str> {
    if let Some(b) = BUILTIN_STRING_SENTINELS.iter().find(|s| **s == value) {
        return Some(b);
    }
    additional
        .iter()
        .find(|s| s.as_str() == value)
        .map(String::as_str)
}

// ─── Canonical text emit ──────────────────────────────────────────────────

/// Render a single [`PerValueLineage`] as canonical text. Stable
/// across runs (siblings sorted by BTreeSet iteration, stages in fixed
/// code order). Suitable for CLI emit and snapshot testing.
pub fn emit_value_trace_text(lineage: &PerValueLineage) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "trace: column={} value={:?}\n",
        lineage.column, lineage.original_value
    ));
    if lineage.stages.is_empty() {
        out.push_str("  no canonicalisation transforms triggered for this value.\n");
        return out;
    }
    for (i, stage) in lineage.stages.iter().enumerate() {
        out.push_str(&format!(
            "  stage {}: {} ({})\n",
            i + 1,
            stage.transform.name(),
            stage.transform.finding_code(),
        ));
        match &stage.transform {
            ValueTransform::SentinelMask { sentinel } => {
                out.push_str(&format!("    sentinel: {:?}\n", sentinel));
                out.push_str("    result: null\n");
            }
            ValueTransform::RareCandidate { count, threshold } => {
                out.push_str(&format!(
                    "    count: {} (below threshold {})\n",
                    count, threshold
                ));
            }
            ValueTransform::TooManyDistinctValuesSkipped { n_distinct, limit } => {
                out.push_str(&format!(
                    "    column has {} distinct values, exceeds limit {}\n",
                    n_distinct, limit
                ));
            }
            _ => {
                if let Some(c) = &stage.canonical {
                    out.push_str(&format!("    canonical: {:?}\n", c));
                }
                if !stage.siblings.is_empty() {
                    let sib_list: Vec<String> =
                        stage.siblings.iter().map(|s| format!("{:?}", s)).collect();
                    out.push_str(&format!(
                        "    siblings: [{}]\n",
                        sib_list.join(", ")
                    ));
                }
            }
        }
    }
    out
}

// ─── Unit tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_data::{Column, DataFrame};

    fn df_with_str(col: &str, values: &[&str]) -> DataFrame {
        DataFrame::from_columns(vec![(
            col.into(),
            Column::Str(values.iter().map(|s| s.to_string()).collect()),
        )])
        .unwrap()
    }

    #[test]
    fn sentinel_mask_terminates_the_chain() {
        let df = df_with_str("weight", &["?", "?", "5kg", "10kg"]);
        let lineage = trace_value(
            &df,
            &PerValueLineageConfig::default(),
            "weight",
            "?",
        )
        .unwrap();
        assert_eq!(lineage.stages.len(), 1);
        match &lineage.stages[0].transform {
            ValueTransform::SentinelMask { sentinel } => assert_eq!(sentinel, "?"),
            other => panic!("expected SentinelMask, got {:?}", other),
        }
        assert_eq!(lineage.stages[0].canonical, None);
    }

    #[test]
    fn case_fold_records_siblings() {
        let df = df_with_str(
            "tier",
            &["Premium", "premium", "PREMIUM", "Standard"],
        );
        let lineage = trace_value(
            &df,
            &PerValueLineageConfig::default(),
            "tier",
            "Premium",
        )
        .unwrap();
        // case_fold should fire; siblings are the other two spellings.
        let case_fold_stage = lineage
            .stages
            .iter()
            .find(|s| matches!(s.transform, ValueTransform::CaseFold))
            .expect("case_fold stage missing");
        assert_eq!(case_fold_stage.canonical.as_deref(), Some("premium"));
        let expected_siblings: BTreeSet<String> =
            ["premium", "PREMIUM"].iter().map(|s| s.to_string()).collect();
        assert_eq!(case_fold_stage.siblings, expected_siblings);
    }

    #[test]
    fn clean_value_produces_no_stages() {
        // Values that are already lowercase ASCII without trailing
        // punctuation or combining marks — every transform's canonical
        // equals the original AND no siblings exist, so no stages fire.
        let df = df_with_str("race", &["caucasian", "asian", "afro"]);
        let lineage = trace_value(
            &df,
            &PerValueLineageConfig::default(),
            "race",
            "caucasian",
        )
        .unwrap();
        // No canonicalisation change, no siblings, no rare candidacy
        // (every count is 1 — counts.values().any(>= 5) is false, so the
        // rare stage skips when every value is rare).
        assert!(
            lineage.stages.is_empty(),
            "expected empty stages, got {:?}",
            lineage.stages
        );
    }

    #[test]
    fn case_changes_emit_stages_even_without_siblings() {
        // "Caucasian" → "caucasian" — canonical differs from original
        // even though no other value collides. The stage fires because
        // it documents the transformation Locke *would* apply, not just
        // collisions. (Strict-mode-only would suppress this — current
        // default is "show transformations, not just collisions".)
        let df = df_with_str("race", &["Caucasian", "Asian"]);
        let lineage = trace_value(
            &df,
            &PerValueLineageConfig::default(),
            "race",
            "Caucasian",
        )
        .unwrap();
        let case_fold_stage = lineage
            .stages
            .iter()
            .find(|s| matches!(s.transform, ValueTransform::CaseFold))
            .expect("case_fold stage should fire");
        assert_eq!(case_fold_stage.canonical.as_deref(), Some("caucasian"));
        assert!(case_fold_stage.siblings.is_empty());
    }

    #[test]
    fn whitespace_punct_stage_only_when_canonical_changes() {
        let df = df_with_str("country", &["USA", "USA.", " USA "]);
        let lineage = trace_value(
            &df,
            &PerValueLineageConfig::default(),
            "country",
            "USA.",
        )
        .unwrap();
        // case_fold: "USA." → "usa." — but the case_fold group ["usa."]
        // has only one member (the original itself, lowercased differs
        // from siblings " USA " → " usa " and "USA" → "usa"); so the
        // case_fold stage fires because the canonical differs from the
        // original ("USA." → "usa.").
        let ws_stage = lineage
            .stages
            .iter()
            .find(|s| matches!(s.transform, ValueTransform::WhitespacePunctStrip { .. }))
            .expect("whitespace_punct stage missing");
        assert_eq!(ws_stage.canonical.as_deref(), Some("usa"));
        // Siblings: "USA" and " USA " also normalise to "usa".
        assert!(ws_stage.siblings.contains("USA"));
        assert!(ws_stage.siblings.contains(" USA "));
    }

    #[test]
    fn rare_candidate_fires_when_count_below_threshold_and_some_value_above() {
        let df = df_with_str(
            "city",
            &["NYC", "NYC", "NYC", "NYC", "NYC", "NYC", "Boston"],
        );
        // NYC count = 6 (above default threshold 5); Boston count = 1.
        let cfg = PerValueLineageConfig::default();
        let lineage_boston = trace_value(&df, &cfg, "city", "Boston").unwrap();
        let rare_stage = lineage_boston
            .stages
            .iter()
            .find(|s| matches!(s.transform, ValueTransform::RareCandidate { .. }))
            .expect("RareCandidate missing for Boston");
        if let ValueTransform::RareCandidate { count, threshold } = &rare_stage.transform
        {
            assert_eq!(*count, 1);
            assert_eq!(*threshold, 5);
        }
        // NYC is above threshold, no rare stage.
        let lineage_nyc = trace_value(&df, &cfg, "city", "NYC").unwrap();
        assert!(
            !lineage_nyc
                .stages
                .iter()
                .any(|s| matches!(s.transform, ValueTransform::RareCandidate { .. })),
            "NYC should not be rare-candidate"
        );
    }

    #[test]
    fn rare_candidate_quiet_when_every_value_is_rare() {
        // Every value below threshold → no signal to report.
        let df = df_with_str("token", &["a", "b", "c", "d"]);
        let cfg = PerValueLineageConfig::default();
        let lineage = trace_value(&df, &cfg, "token", "a").unwrap();
        assert!(
            !lineage
                .stages
                .iter()
                .any(|s| matches!(s.transform, ValueTransform::RareCandidate { .. })),
            "no RareCandidate when all rare"
        );
    }

    #[test]
    fn nonexistent_value_returns_none() {
        let df = df_with_str("c", &["x", "y"]);
        let cfg = PerValueLineageConfig::default();
        assert!(trace_value(&df, &cfg, "c", "z").is_none());
    }

    #[test]
    fn nonexistent_column_returns_none() {
        let df = df_with_str("c", &["x"]);
        let cfg = PerValueLineageConfig::default();
        assert!(trace_value(&df, &cfg, "other", "x").is_none());
    }

    #[test]
    fn build_lineage_is_deterministic() {
        let df = df_with_str(
            "tier",
            &["Premium", "premium", "PREMIUM", "Standard"],
        );
        let cfg = PerValueLineageConfig::default();
        let a = build_per_value_lineage(&df, &cfg);
        let b = build_per_value_lineage(&df, &cfg);
        assert_eq!(a, b);
    }

    #[test]
    fn build_lineage_skips_oversized_columns() {
        let cfg = PerValueLineageConfig {
            max_distinct_per_column: 3,
            ..Default::default()
        };
        let df = df_with_str("c", &["a", "b", "c", "d", "e"]);
        let map = build_per_value_lineage(&df, &cfg);
        // Should contain exactly one entry: the synthetic __skipped__ key.
        assert_eq!(map.len(), 1);
        let lineage = map.get(&("c".to_string(), "__skipped__".to_string())).unwrap();
        assert!(matches!(
            lineage.stages[0].transform,
            ValueTransform::TooManyDistinctValuesSkipped { .. }
        ));
    }

    #[test]
    fn additional_sentinels_extend_the_builtin_list() {
        let df = df_with_str("x", &["UNK", "1", "2"]);
        let cfg = PerValueLineageConfig {
            additional_sentinels: vec!["UNK".into()],
            ..Default::default()
        };
        let lineage = trace_value(&df, &cfg, "x", "UNK").unwrap();
        assert_eq!(lineage.stages.len(), 1);
        if let ValueTransform::SentinelMask { sentinel } = &lineage.stages[0].transform {
            assert_eq!(sentinel, "UNK");
        } else {
            panic!("expected SentinelMask for custom sentinel");
        }
    }

    #[test]
    fn emit_text_is_deterministic_and_well_formed() {
        let df = df_with_str(
            "tier",
            &["Premium", "premium", "PREMIUM"],
        );
        let lineage = trace_value(
            &df,
            &PerValueLineageConfig::default(),
            "tier",
            "Premium",
        )
        .unwrap();
        let s1 = emit_value_trace_text(&lineage);
        let s2 = emit_value_trace_text(&lineage);
        assert_eq!(s1, s2);
        assert!(s1.contains("trace: column=tier"));
        assert!(s1.contains("stage 1: case_fold (E9080)"));
        assert!(s1.contains("canonical: \"premium\""));
        assert!(s1.contains("siblings:"));
    }

    #[test]
    fn emit_text_for_clean_value_says_so() {
        let df = df_with_str("c", &["x", "y", "z"]);
        let lineage = trace_value(
            &df,
            &PerValueLineageConfig::default(),
            "c",
            "x",
        )
        .unwrap();
        let s = emit_value_trace_text(&lineage);
        assert!(s.contains("no canonicalisation transforms triggered"));
    }

    #[test]
    fn unicode_normalize_groups_nfc_nfd_variants() {
        // "café" in NFC (U+00E9) vs "café" in NFD (e + combining acute)
        let nfc = "caf\u{00E9}";
        let nfd = "cafe\u{0301}";
        assert_ne!(nfc, nfd, "test fixture invariant");
        let df = df_with_str("name", &[nfc, nfd, "Boston"]);
        let lineage = trace_value(
            &df,
            &PerValueLineageConfig::default(),
            "name",
            nfd,
        )
        .unwrap();
        let unorm = lineage
            .stages
            .iter()
            .find(|s| matches!(s.transform, ValueTransform::UnicodeNormalize))
            .expect("unicode_normalize stage missing");
        assert_eq!(unorm.canonical.as_deref(), Some("cafe"));
        // NFC sibling should appear.
        assert!(unorm.siblings.contains(nfc));
    }

    #[test]
    fn stage_set_can_disable_individual_stages() {
        let df = df_with_str("c", &["A", "a"]);
        let cfg = PerValueLineageConfig {
            stages: PerValueStageSet {
                sentinel: false,
                case_fold: false,
                whitespace_punct: false,
                unicode_normalize: false,
                rare_candidate: false,
            },
            ..Default::default()
        };
        let lineage = trace_value(&df, &cfg, "c", "A").unwrap();
        assert!(
            lineage.stages.is_empty(),
            "disabling all stages should suppress lineage"
        );
    }

    #[test]
    fn nonstring_columns_are_ignored() {
        let df = DataFrame::from_columns(vec![(
            "n".into(),
            Column::Int(vec![1, 2, 3]),
        )])
        .unwrap();
        let map = build_per_value_lineage(&df, &PerValueLineageConfig::default());
        assert!(map.is_empty(), "numeric column should not produce lineage");
        let lineage = trace_value(
            &df,
            &PerValueLineageConfig::default(),
            "n",
            "1",
        );
        assert!(lineage.is_none(), "numeric column trace returns None");
    }

    #[test]
    fn matching_sentinel_consults_builtin_then_additional() {
        assert_eq!(matching_sentinel("?", &[]), Some("?"));
        assert_eq!(matching_sentinel("UNK", &[]), None);
        let custom = vec!["UNK".to_string()];
        assert_eq!(matching_sentinel("UNK", &custom), Some("UNK"));
    }
}
