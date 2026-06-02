//! Target leakage detection (v0.5).
//!
//! The classic ML pipeline bug: a feature is *almost* perfectly
//! predictive of the target on the training set, because it was
//! computed *after* the event (e.g., "days since last login" measured
//! at score time leaks "the customer churned recently").
//!
//! Locke flags this via per-feature ROC AUC against a caller-supplied
//! target column. Codes:
//!
//! | Code  | Severity | When |
//! |-------|----------|------|
//! | E9060 | Error    | AUC ≥ 0.95 (almost certainly leakage) |
//! | E9061 | Warning  | 0.85 ≤ AUC < 0.95 (worth investigating) |
//!
//! Plus E9072 for ID-like cardinality (a hint that may compound with
//! the AUC signal): a column whose distinct/n_rows ratio is ≥ 0.95 is
//! probably a key column leaking as a feature.
//!
//! ## Limitations
//!
//! - Only binary targets are supported in v0.5. Multi-class would
//!   need one-vs-rest AUC; deferred.
//! - AUC is computed from the full dataset, not a held-out split.
//!   For a true "is this leakage" test, the user should run the same
//!   check on a separate validation set.

use cjc_data::{Column, DataFrame};

use crate::report::{FindingEvidence, FindingSeverity, ValidationFinding};

#[derive(Clone, Debug)]
pub struct LeakageConfig {
    pub auc_error_threshold: f64,
    pub auc_warn_threshold: f64,
    /// Minimum count of each class — below this AUC is unreliable.
    pub min_class_count: u64,
    /// Cardinality / n_rows above which a column is flagged as ID-like.
    pub id_like_cardinality_ratio: f64,
    /// Minimum row count before ID-like checks fire.
    pub id_like_min_rows: u64,
    /// v0.6.3: maximum number of distinct values an Int target can have
    /// before being treated as continuous and skipped. Default 20.
    pub multiclass_max_classes: u32,
}

impl Default for LeakageConfig {
    fn default() -> Self {
        Self {
            auc_error_threshold: 0.95,
            auc_warn_threshold: 0.85,
            min_class_count: 10,
            id_like_cardinality_ratio: 0.95,
            id_like_min_rows: 50,
            multiclass_max_classes: 20,
        }
    }
}

/// Extract a binary target column as `Vec<u8>` (0/1). Returns `None`
/// if the column doesn't look binary (Bool always works; Int with
/// exactly 2 distinct values also works).
fn extract_binary_target(df: &DataFrame, name: &str) -> Option<Vec<u8>> {
    let col = df.get_column(name)?;
    match col {
        Column::Bool(v) => Some(v.iter().map(|b| if *b { 1u8 } else { 0u8 }).collect()),
        Column::Int(v) => {
            // Accept any column whose values are exactly {0, 1} or a {a, b}
            // pair where the smaller is encoded 0 and the larger 1.
            let mut distinct: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
            for x in v {
                distinct.insert(*x);
                if distinct.len() > 2 {
                    return None;
                }
            }
            if distinct.len() != 2 {
                return None;
            }
            let mut iter = distinct.iter();
            let lo = *iter.next().unwrap();
            let hi = *iter.next().unwrap();
            Some(
                v.iter()
                    .map(|x| if *x == lo { 0u8 } else if *x == hi { 1u8 } else { 0 })
                    .collect(),
            )
        }
        _ => None,
    }
}

/// Extract feature values as f64 for AUC computation. Returns `None`
/// for non-numeric columns. NaN is excluded along with its target.
fn extract_feature(df: &DataFrame, name: &str) -> Option<Vec<f64>> {
    let col = df.get_column(name)?;
    match col {
        Column::Float(v) => Some(v.clone()),
        Column::Int(v) => Some(v.iter().map(|x| *x as f64).collect()),
        Column::Bool(v) => Some(v.iter().map(|b| if *b { 1.0 } else { 0.0 }).collect()),
        _ => None,
    }
}

/// Compute ROC AUC of `feature` against a binary `target` (0/1).
/// Returns `None` if either class has < `min_class_count` samples or
/// if no valid pairs remain after NaN filtering.
///
/// Algorithm: pair (feature, target), drop NaN rows, sort by feature,
/// compute rank-sum AUC. O(n log n) time, O(n) memory, fully
/// deterministic via `f64::total_cmp`.
pub fn binary_target_auc(
    feature: &[f64],
    target: &[u8],
    min_class_count: u64,
) -> Option<f64> {
    if feature.len() != target.len() {
        return None;
    }
    let mut pairs: Vec<(f64, u8)> = feature
        .iter()
        .zip(target.iter())
        .filter_map(|(f, t)| {
            if f.is_nan() {
                None
            } else {
                Some((*f, *t))
            }
        })
        .collect();
    let n_pos = pairs.iter().filter(|(_, t)| *t == 1).count() as u64;
    let n_neg = pairs.iter().filter(|(_, t)| *t == 0).count() as u64;
    if n_pos < min_class_count || n_neg < min_class_count {
        return None;
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Rank-sum AUC. We need average ranks for ties.
    let n = pairs.len();
    let mut sum_ranks_positive = 0.0f64;
    let mut i = 0usize;
    while i < n {
        // Find tie group [i, j)
        let mut j = i + 1;
        while j < n && pairs[j].0 == pairs[i].0 {
            j += 1;
        }
        // Average rank for this tie group (1-indexed).
        let avg_rank = (i as f64 + 1.0 + j as f64) / 2.0;
        for k in i..j {
            if pairs[k].1 == 1 {
                sum_ranks_positive += avg_rank;
            }
        }
        i = j;
    }

    let n_pos_f = n_pos as f64;
    let n_neg_f = n_neg as f64;
    let auc = (sum_ranks_positive - n_pos_f * (n_pos_f + 1.0) / 2.0) / (n_pos_f * n_neg_f);
    Some(auc)
}

// ─── Multi-class leakage (v0.6.3) ─────────────────────────────────────────

/// Extract an Int target with 3 to `max_classes` distinct values as a
/// `(Vec<u32>, Vec<i64>)` of (per-row class index, sorted distinct labels).
/// Returns `None` if the target isn't Int, has fewer than 3 distinct
/// values (binary or constant — handled elsewhere), or has more than
/// `max_classes` distinct values (likely continuous-ish).
fn extract_multiclass_target(
    df: &DataFrame,
    name: &str,
    max_classes: u32,
) -> Option<(Vec<u32>, Vec<i64>)> {
    let col = df.get_column(name)?;
    let Column::Int(values) = col else { return None };
    let mut distinct: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
    for v in values {
        distinct.insert(*v);
        if distinct.len() > max_classes as usize {
            return None;
        }
    }
    if distinct.len() < 3 {
        return None;
    }
    let labels: Vec<i64> = distinct.into_iter().collect();
    // `labels` is built from a `BTreeSet`, so it is already sorted. Use
    // `binary_search` directly instead of building a separate
    // `BTreeMap<i64, u32>` whose only purpose is "label -> index" lookup.
    // For n_classes ≤ 20 (the practical cap) binary search on a
    // contiguous `Vec` beats `BTreeMap` (better cache locality, zero
    // additional heap, identical O(log n) complexity). `expect` cannot
    // fire: every `v` in `values` was already inserted into `distinct`
    // → it is guaranteed to be in `labels`.
    let classes: Vec<u32> = values
        .iter()
        .map(|v| labels.binary_search(v).expect("value must be in distinct set") as u32)
        .collect();
    Some((classes, labels))
}

/// One-vs-rest AUC: for each class `c` in `0..n_classes`, compute AUC of
/// `feature` against the binary target `target == c`. Returns the **max**
/// per-class AUC (so a feature that perfectly identifies one class
/// surfaces, even if it's noise on the others). Returns `None` if no
/// class has enough samples on both sides for the AUC computation.
pub fn multiclass_max_one_vs_rest_auc(
    feature: &[f64],
    target: &[u32],
    n_classes: u32,
    min_class_count: u64,
) -> Option<f64> {
    compute_per_class_aucs(feature, target, n_classes, min_class_count)
        .into_iter()
        .map(|(_, a)| a)
        .fold(None, |best, a| Some(best.map_or(a, |b: f64| b.max(a))))
}

/// Compute per-class one-vs-rest |AUC| in a single pass. The detector
/// pipeline previously called `multiclass_max_one_vs_rest_auc` (which
/// internally iterates classes) and then iterated classes again to
/// build the top-3 display — N²-style double work in the multi-class
/// path. This helper computes both pieces from one iteration; max is a
/// trivial fold of the returned vec.
///
/// Returns an empty vec when no class has enough samples on both
/// sides.
pub(crate) fn compute_per_class_aucs(
    feature: &[f64],
    target: &[u32],
    n_classes: u32,
    min_class_count: u64,
) -> Vec<(u32, f64)> {
    if feature.len() != target.len() {
        return Vec::new();
    }
    let mut out: Vec<(u32, f64)> = Vec::with_capacity(n_classes as usize);
    for c in 0..n_classes {
        let binary: Vec<u8> = target
            .iter()
            .map(|t| if *t == c { 1u8 } else { 0u8 })
            .collect();
        if let Some(auc) = binary_target_auc(feature, &binary, min_class_count) {
            let abs_auc = auc.max(1.0 - auc);
            out.push((c, abs_auc));
        }
    }
    out
}

/// Multi-class variant of `detect_target_leakage`. Emits **E9063**
/// (Warning when max OVR |AUC| ≥ `auc_warn_threshold`, Error when ≥
/// `auc_error_threshold`). Same evidence shape as E9060/E9061 plus
/// the per-class breakdown.
pub fn detect_target_leakage_multiclass(
    df: &DataFrame,
    target_col: &str,
    cfg: &LeakageConfig,
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let Some((target_idx, labels)) = extract_multiclass_target(df, target_col, cfg.multiclass_max_classes) else {
        return out;
    };
    let n_classes = labels.len() as u32;
    let n_rows = df.nrows() as u64;
    for (name, _col) in &df.columns {
        if name == target_col {
            continue;
        }
        let Some(feat) = extract_feature(df, name) else {
            continue;
        };
        // v0.7+ B6.7: previously the per-class AUC loop ran twice —
        // once via `multiclass_max_one_vs_rest_auc` for the max, then
        // again here for the top-3 display. Each loop allocates a
        // length-N binary vector per class and sorts a length-N pair
        // vector inside `binary_target_auc`. Single-pass cuts the cost
        // by a factor of two.
        let mut per_class_aucs =
            compute_per_class_aucs(&feat, &target_idx, n_classes, cfg.min_class_count);
        if per_class_aucs.is_empty() {
            continue;
        }
        let max_auc = per_class_aucs
            .iter()
            .map(|(_, a)| *a)
            .fold(f64::MIN, f64::max);
        let (severity, marker) = if max_auc >= cfg.auc_error_threshold {
            (FindingSeverity::Error, "almost certainly leakage")
        } else if max_auc >= cfg.auc_warn_threshold {
            (FindingSeverity::Warning, "worth investigating")
        } else {
            continue;
        };
        per_class_aucs.sort_by(|a, b| b.1.total_cmp(&a.1));
        let per_class_str = per_class_aucs
            .iter()
            .take(3)
            .map(|(c, auc)| format!("class[{}]={:.3}", labels[*c as usize], auc))
            .collect::<Vec<_>>()
            .join(", ");

        out.push(ValidationFinding::new(
            "E9063",
            severity,
            format!(
                "feature `{}` has max OVR |AUC| = {:.4} against multi-class target `{}` ({}); {}",
                name, max_auc, target_col, n_classes, marker
            ),
            Some(name.clone()),
            None,
            vec![
                FindingEvidence::Metric {
                    label: "max_ovr_auc".into(),
                    value: max_auc,
                },
                FindingEvidence::Count {
                    label: "n_classes".into(),
                    value: n_classes as u64,
                },
                FindingEvidence::Sample {
                    label: "target".into(),
                    value: target_col.into(),
                },
                FindingEvidence::Sample {
                    label: "top_per_class_aucs".into(),
                    value: per_class_str,
                },
            ],
            n_rows,
            vec![
                "one-vs-rest AUC: for each class c, compute AUC of feature vs (target == c) and take the max".into(),
                "v0.6.3 multi-class extension; binary handled by E9060/E9061".into(),
                "high AUC on a single class often points to a class-specific leak (e.g. encoded outcome category)".into(),
            ],
            vec![
                "inspect which class drives the leak; it usually localises the upstream bug".into(),
                "verify the feature is not derived from a post-outcome value".into(),
            ],
        ));
    }
    out
}

/// For each feature column (excluding the target), compute AUC and
/// emit E9060 (Error, AUC ≥ 0.95) or E9061 (Warning, AUC ≥ 0.85).
pub fn detect_target_leakage(
    df: &DataFrame,
    target_col: &str,
    cfg: &LeakageConfig,
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let Some(target) = extract_binary_target(df, target_col) else {
        out.push(ValidationFinding::new(
            "E9062",
            FindingSeverity::Notice,
            format!(
                "target column `{}` is not a binary type (Bool or Int-with-exactly-2-distinct); target leakage check skipped",
                target_col
            ),
            Some(target_col.into()),
            None,
            vec![],
            df.nrows() as u64,
            vec![
                "v0.5 supports only binary targets for the per-feature AUC check".into(),
            ],
            vec![
                "encode the target as Bool or 0/1 Int".into(),
                "multi-class AUC support is on the v0.6 roadmap".into(),
            ],
        ));
        return out;
    };

    let n_rows = df.nrows() as u64;
    for (name, _col) in &df.columns {
        if name == target_col {
            continue;
        }
        let Some(feat) = extract_feature(df, name) else {
            continue;
        };
        let Some(auc) = binary_target_auc(&feat, &target, cfg.min_class_count) else {
            continue;
        };
        // Use max(AUC, 1-AUC) so we don't miss negatively-perfect features.
        let abs_auc = auc.max(1.0 - auc);
        let (severity, code) = if abs_auc >= cfg.auc_error_threshold {
            (FindingSeverity::Error, "E9060")
        } else if abs_auc >= cfg.auc_warn_threshold {
            (FindingSeverity::Warning, "E9061")
        } else {
            continue;
        };
        out.push(ValidationFinding::new(
            code,
            severity,
            format!(
                "feature `{}` has |AUC| = {:.4} against target `{}` — suspicious target leakage",
                name, abs_auc, target_col
            ),
            Some(name.clone()),
            None,
            vec![
                FindingEvidence::Metric {
                    label: "abs_auc".into(),
                    value: abs_auc,
                },
                FindingEvidence::Metric {
                    label: "raw_auc".into(),
                    value: auc,
                },
                FindingEvidence::Sample {
                    label: "target".into(),
                    value: target_col.into(),
                },
            ],
            n_rows,
            vec![
                "AUC near 0.5 means no signal; near 0 or 1 means the feature is nearly deterministic of the target".into(),
                "high AUC may be legitimate (a strong feature) OR leakage — Locke surfaces both".into(),
                "this AUC is computed on the full dataset; verify on a held-out split".into(),
            ],
            vec![
                "check whether this feature is computed BEFORE or AFTER the target event".into(),
                "if computed after, exclude from training".into(),
            ],
        ));
    }
    out
}

// ─── Per-level deterministic-leakage detector (E9064, v0.6.4) ───────────────

/// Config for [`detect_per_level_target_leakage`].
///
/// **Motivation.** [`detect_target_leakage_multiclass`] (E9063) computes
/// a per-feature ROC AUC against a multi-class target. AUC is a
/// column-wide rank statistic — it misses leakage that hides at the
/// *level* of a column. On UCI Diabetes-130, `discharge_disposition_id`
/// codes 11, 13, 14, 19, 20, 21 deterministically predict
/// `readmitted = NO` (a dead patient cannot be readmitted), but death
/// codes are interspersed *numerically* with non-death codes 12, 15,
/// 16, 17, 18 — so the column-wide AUC stays below 0.85 and E9063
/// stays silent. The Phase 0.10 §4.B work documented this miss and
/// motivated this detector.
///
/// **Detection.** For each `(column, level, target_class)` triple:
///
/// 1. Count rows where `column == level` — call this the *support*.
/// 2. Of those, count rows where `target == class` — the *concentration*.
/// 3. Compute `P(class | level) = concentration / support`.
/// 4. Compute the unconditional `P(class)` (the base rate) once per
///    target class.
/// 5. Emit [`"E9064"`](ValidationFinding) when
///    `P(class | level) ≥ conditional_threshold`, the support is at
///    least `min_support`, and `P(class) < conditional_threshold` (so
///    the level adds information beyond the base rate).
#[derive(Clone, Debug)]
pub struct PerLevelLeakageConfig {
    /// Threshold on `P(class | level)`. Default `0.99` — only flag
    /// near-deterministic outcomes.
    pub conditional_threshold: f64,
    /// Minimum rows where `column == level` for the (level, class)
    /// pair to be considered. Default `10` — below this the
    /// conditional estimate is unreliable.
    pub min_support: u64,
    /// Maximum distinct levels per column. Above this, the column is
    /// likely continuous-ish or high-cardinality nominal (e.g. an
    /// ICD-9 code); skip it. Default `1000`.
    pub max_levels: u64,
    /// Maximum distinct classes in the target. Above this, the target
    /// is likely continuous; skip it. Default `20` (matches
    /// [`LeakageConfig::multiclass_max_classes`]).
    pub max_classes: u32,
}

impl Default for PerLevelLeakageConfig {
    fn default() -> Self {
        Self {
            conditional_threshold: 0.99,
            min_support: 10,
            max_levels: 1000,
            max_classes: 20,
        }
    }
}

/// Typed BTreeMap key for the per-level leakage detector. Avoids the
/// per-row String allocation that the old `format_level`-keyed map
/// did — on diabetes-130-scale data that's 10⁵+ Strings per validate.
///
/// Determinism: the final `LockeReport` sorts findings by
/// `(severity, code, column, id)` — so the per-detector emit order is
/// invisible to output. Switching from String keys (lex order) to
/// typed keys (natural order, e.g. Int(2) < Int(10) instead of "10" <
/// "2") shifts emit order but the final report bytes are unchanged.
///
/// Float columns are skipped upstream at the per-feature filter so no
/// f64-key collision semantics are involved.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum LevelKey<'a> {
    Int(i64),
    Str(&'a str),
    Bool(bool),
    Cat(u32),
    Datetime(i64),
    /// `CategoricalAdaptive` historically collapsed to a single bucket
    /// via the "<adaptive>" sentinel; preserve that behaviour with a
    /// unit variant.
    Adaptive,
}

fn level_key<'a>(col: &'a Column, row: usize) -> LevelKey<'a> {
    match col {
        Column::Int(v) => LevelKey::Int(v[row]),
        Column::Str(v) => LevelKey::Str(v[row].as_str()),
        Column::Bool(v) => LevelKey::Bool(v[row]),
        Column::Categorical { codes, .. } => LevelKey::Cat(codes[row]),
        Column::DateTime(v) => LevelKey::Datetime(v[row]),
        Column::CategoricalAdaptive(_) => LevelKey::Adaptive,
        // Float is filtered out at the per-feature loop so this arm is
        // never reached during normal operation. Treat as Adaptive to
        // avoid panicking in case a future caller widens the surface.
        Column::Float(_) => LevelKey::Adaptive,
    }
}

/// Render a level value as a deterministic string for the finding
/// message. Float NaN is rendered as `NaN` so the message is stable
/// across runs.
fn format_level(col: &Column, row: usize) -> String {
    match col {
        Column::Str(v) => format!("{:?}", v[row]),
        Column::Int(v) => v[row].to_string(),
        Column::Bool(v) => v[row].to_string(),
        Column::Float(v) => {
            if v[row].is_nan() {
                "NaN".to_string()
            } else {
                format!("{:.6}", v[row])
            }
        }
        Column::Categorical { levels, codes } => {
            let c = codes[row] as usize;
            if c < levels.len() {
                format!("{:?}", levels[c])
            } else {
                "<oob>".into()
            }
        }
        Column::DateTime(v) => v[row].to_string(),
        Column::CategoricalAdaptive(_) => "<adaptive>".into(),
    }
}

/// Per-level deterministic-outcome leakage detector. Emits
/// [`"E9064"`](ValidationFinding) (Error severity) when at least one
/// `(column, level)` pair deterministically predicts the target.
///
/// Returns the empty vector when:
/// - the target column is missing or its type isn't `Bool` / `Int`
/// - the target has more than `cfg.max_classes` distinct values
/// - no feature column has any level meeting the thresholds
///
/// # Algorithm
///
/// One pass over the rows per feature column. O(n_rows × n_features)
/// time, O(n_levels_per_feature × n_classes) memory per column.
///
/// # Determinism
///
/// Levels iterate in `BTreeMap` order; classes iterate in the
/// canonical sorted-distinct order computed from `extract_multiclass_target`
/// (binary follows the same convention). Findings are appended
/// column-by-column, level-by-level; two runs over the same
/// `DataFrame` produce byte-identical IDs.
pub fn detect_per_level_target_leakage(
    df: &DataFrame,
    target_col: &str,
    cfg: &PerLevelLeakageConfig,
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();

    // Step 1 — decode the target. Accept either Bool, binary Int, or
    // multi-class Int (up to max_classes distinct).
    let target_col_data = match df.get_column(target_col) {
        Some(c) => c,
        None => return out,
    };
    let (target_codes, class_labels): (Vec<u32>, Vec<String>) = match target_col_data {
        Column::Bool(v) => {
            let codes: Vec<u32> = v.iter().map(|b| u32::from(*b)).collect();
            (codes, vec!["false".into(), "true".into()])
        }
        Column::Int(v) => {
            let mut distinct: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
            for x in v {
                distinct.insert(*x);
                if distinct.len() > cfg.max_classes as usize {
                    return out;
                }
            }
            if distinct.len() < 2 {
                return out;
            }
            let labels_i64: Vec<i64> = distinct.into_iter().collect();
            let mut lookup: std::collections::BTreeMap<i64, u32> =
                std::collections::BTreeMap::new();
            for (i, lbl) in labels_i64.iter().enumerate() {
                lookup.insert(*lbl, i as u32);
            }
            let codes: Vec<u32> = v.iter().map(|x| lookup[x]).collect();
            let labels: Vec<String> = labels_i64.iter().map(|x| x.to_string()).collect();
            (codes, labels)
        }
        _ => return out,
    };
    let n_classes = class_labels.len() as u32;
    let n_rows = target_codes.len();
    if n_rows == 0 {
        return out;
    }

    // Step 2 — compute base rates P(class).
    let mut class_counts: Vec<u64> = vec![0u64; n_classes as usize];
    for c in &target_codes {
        class_counts[*c as usize] += 1;
    }
    let base_rates: Vec<f64> = class_counts
        .iter()
        .map(|c| *c as f64 / n_rows as f64)
        .collect();

    // Step 3 — per-feature scan.
    for (name, col) in &df.columns {
        if name == target_col {
            continue;
        }
        if matches!(col, Column::Float(_)) {
            // Floats are continuous; binning is the caller's job. A
            // value-by-value match would be meaningless for f64.
            continue;
        }

        // v0.7+ B6.6: per-level bucketing previously called
        // `format_level` per (row, col) — one heap allocation per cell,
        // 10⁵+ on diabetes-130-scale data. The LevelKey enum uses a
        // typed key (Int/Str(&str)/Bool/...) that avoids the
        // allocation; format_level is still called once per *emitted*
        // finding for the human-readable level display.
        let mut level_buckets: std::collections::BTreeMap<LevelKey<'_>, (usize, Vec<u64>)> =
            std::collections::BTreeMap::new();
        // (representative_row_idx, per_class_count)
        for (row, tgt) in target_codes.iter().enumerate() {
            let key = level_key(col, row);
            let entry = level_buckets
                .entry(key)
                .or_insert_with(|| (row, vec![0u64; n_classes as usize]));
            entry.1[*tgt as usize] += 1;
            if level_buckets.len() > cfg.max_levels as usize {
                // High-cardinality column — bail. (BTreeMap can grow
                // past max_levels in one row; we cap.)
                break;
            }
        }
        if level_buckets.len() > cfg.max_levels as usize {
            continue;
        }

        // Step 4 — evaluate each level × class for the threshold.
        for (level_key_val, (rep_row, per_class)) in &level_buckets {
            let support: u64 = per_class.iter().sum();
            if support < cfg.min_support {
                continue;
            }
            // Find the most-concentrated class for this level.
            let (best_c, best_count) = per_class
                .iter()
                .enumerate()
                .max_by_key(|(_, c)| **c)
                .map(|(i, c)| (i as u32, *c))
                .unwrap_or((0, 0));
            let p_class_given_level = best_count as f64 / support as f64;
            if p_class_given_level < cfg.conditional_threshold {
                continue;
            }
            // Filter base-rate-dominated levels (the column doesn't
            // add information).
            let p_class_uncond = base_rates[best_c as usize];
            if p_class_uncond >= cfg.conditional_threshold {
                continue;
            }

            let pretty_level = format_level(col, *rep_row);
            let _ = level_key_val;
            let pretty_class = &class_labels[best_c as usize];

            out.push(ValidationFinding::new(
                "E9064",
                FindingSeverity::Error,
                format!(
                    "level `{}` of column `{}` deterministically predicts target class `{}` ({}/{} rows = {:.4}, base rate {:.4})",
                    pretty_level, name, pretty_class, best_count, support, p_class_given_level, p_class_uncond
                ),
                Some(name.clone()),
                None,
                vec![
                    FindingEvidence::Sample {
                        label: "level".into(),
                        value: pretty_level.clone(),
                    },
                    FindingEvidence::Sample {
                        label: "target_class".into(),
                        value: pretty_class.clone(),
                    },
                    FindingEvidence::Metric {
                        label: "p_class_given_level".into(),
                        value: p_class_given_level,
                    },
                    FindingEvidence::Metric {
                        label: "base_rate".into(),
                        value: p_class_uncond,
                    },
                    FindingEvidence::Count {
                        label: "support".into(),
                        value: support,
                    },
                ],
                n_rows as u64,
                vec![
                    "Per-level deterministic-outcome leakage: this specific value of the column predicts the target with ≥ threshold probability".into(),
                    "Diabetes-130 motivating example: discharge_disposition_id ∈ {11, 13, 14, 19, 20, 21} (death/hospice) → readmitted = NO".into(),
                    "Continuous features (Float) skipped — bin them before re-running if you want a per-bin check".into(),
                ],
                vec![
                    "Verify the level is not derived from a post-outcome event (e.g. a discharge code that records the outcome)".into(),
                    "If the level encodes a structural impossibility (death → no readmission), drop those rows from training".into(),
                    "Alternative: split the multi-class target so the impossible class is its own task".into(),
                ],
            ));
        }
    }

    out
}

/// ID-like cardinality hint — flags columns whose `distinct / n_rows`
/// ratio crosses the configured threshold. Often catches join keys or
/// row IDs accidentally left in the feature matrix.
pub fn detect_id_like_columns(df: &DataFrame, cfg: &LeakageConfig) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let n_rows = df.nrows() as u64;
    if n_rows < cfg.id_like_min_rows {
        return out;
    }
    for (name, col) in &df.columns {
        let distinct = crate::validation::distinct_count(col);
        let ratio = distinct as f64 / n_rows as f64;
        if ratio >= cfg.id_like_cardinality_ratio {
            out.push(ValidationFinding::new(
                "E9072",
                FindingSeverity::Notice,
                format!(
                    "column `{}` has cardinality {} / {} rows = {:.3}; may be an ID leaking as a feature",
                    name, distinct, n_rows, ratio
                ),
                Some(name.clone()),
                None,
                vec![
                    FindingEvidence::Count {
                        label: "distinct".into(),
                        value: distinct,
                    },
                    FindingEvidence::Ratio {
                        label: "distinct_per_row".into(),
                        value: ratio,
                    },
                ],
                n_rows,
                vec![
                    "ID-like columns memorise the training set and don't generalise".into(),
                    "high cardinality may also be legitimate (a free-text field) — Locke surfaces a hint, not a verdict".into(),
                ],
                vec![
                    "drop the column if it's a customer/row/transaction ID".into(),
                    "hash to a low-cardinality bucket if you must retain the value".into(),
                ],
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_data::{Column, DataFrame};

    fn df_with_perfect_leak() -> DataFrame {
        // Feature is literally the target — AUC will be exactly 1.0.
        let n = 100;
        let target: Vec<i64> = (0..n).map(|i| if i < n / 2 { 0 } else { 1 }).collect();
        let feature: Vec<f64> = target.iter().map(|t| *t as f64).collect();
        DataFrame::from_columns(vec![
            ("y".into(), Column::Int(target)),
            ("leaky".into(), Column::Float(feature)),
        ])
        .unwrap()
    }

    fn df_with_random_feature() -> DataFrame {
        use cjc_repro::Rng;
        let n = 200;
        let mut rng = Rng::seeded(42);
        let target: Vec<i64> = (0..n)
            .map(|i| if i < n / 2 { 0 } else { 1 })
            .collect();
        let feature: Vec<f64> = (0..n).map(|_| (rng.next_u64() % 1000) as f64).collect();
        DataFrame::from_columns(vec![
            ("y".into(), Column::Int(target)),
            ("noise".into(), Column::Float(feature)),
        ])
        .unwrap()
    }

    #[test]
    fn perfect_leak_yields_e9060_error() {
        let df = df_with_perfect_leak();
        let r = detect_target_leakage(&df, "y", &LeakageConfig::default());
        let f = r.iter().find(|f| f.code == "E9060").expect("E9060 for perfect leak");
        assert_eq!(f.severity, FindingSeverity::Error);
    }

    #[test]
    fn random_feature_yields_no_leakage_finding() {
        let df = df_with_random_feature();
        let r = detect_target_leakage(&df, "y", &LeakageConfig::default());
        assert!(r.iter().all(|f| f.code != "E9060" && f.code != "E9061"));
    }

    #[test]
    fn non_binary_target_emits_e9062_notice() {
        let df = DataFrame::from_columns(vec![
            ("y".into(), Column::Int(vec![1, 2, 3, 4, 5, 1, 2, 3])),
            ("x".into(), Column::Float(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])),
        ])
        .unwrap();
        let r = detect_target_leakage(&df, "y", &LeakageConfig::default());
        assert!(r.iter().any(|f| f.code == "E9062"));
    }

    #[test]
    fn id_like_column_emits_e9072() {
        // 100 rows, 100 distinct user_ids → ratio = 1.0
        let user_ids: Vec<i64> = (0..100).collect();
        let df = DataFrame::from_columns(vec![("user_id".into(), Column::Int(user_ids))]).unwrap();
        let r = detect_id_like_columns(&df, &LeakageConfig::default());
        assert!(r.iter().any(|f| f.code == "E9072"));
    }

    #[test]
    fn low_cardinality_column_does_not_fire_id_like() {
        let df = DataFrame::from_columns(vec![
            ("status".into(), Column::Int((0..100).map(|i| i % 3).collect())),
        ])
        .unwrap();
        let r = detect_id_like_columns(&df, &LeakageConfig::default());
        assert!(r.is_empty());
    }

    #[test]
    fn small_dataset_skips_id_like_check() {
        // Below the min_rows threshold.
        let df = DataFrame::from_columns(vec![
            ("id".into(), Column::Int((0..10).collect())),
        ])
        .unwrap();
        let r = detect_id_like_columns(&df, &LeakageConfig::default());
        assert!(r.is_empty());
    }

    #[test]
    fn auc_of_identical_data_is_one() {
        let feat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let tgt: Vec<u8> = vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        let auc = binary_target_auc(&feat, &tgt, 5).unwrap();
        assert!((auc - 1.0).abs() < 1e-12);
    }

    #[test]
    fn auc_of_reversed_is_zero() {
        let feat: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let tgt: Vec<u8> = vec![1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0];
        let auc = binary_target_auc(&feat, &tgt, 5).unwrap();
        assert!(auc < 1e-12);
    }

    #[test]
    fn detection_is_deterministic() {
        let df = df_with_perfect_leak();
        let cfg = LeakageConfig::default();
        let a = detect_target_leakage(&df, "y", &cfg);
        let b = detect_target_leakage(&df, "y", &cfg);
        assert_eq!(a, b);
    }

    // ─── v0.6.4 — E9064 per-level deterministic leakage ────────────────────

    /// Build a synthetic "discharge code → outcome" frame mirroring
    /// the diabetes-130 pattern: codes 11/13/14 (death/hospice)
    /// deterministically predict y=0; codes 1-10 (normal discharges)
    /// have a 50/50 split.
    fn df_with_per_level_leak() -> DataFrame {
        let mut discharge: Vec<i64> = Vec::new();
        let mut y: Vec<i64> = Vec::new();
        // 15 rows of death codes — all y=0
        for code in [11_i64, 13, 14] {
            for _ in 0..15 {
                discharge.push(code);
                y.push(0);
            }
        }
        // 100 rows of normal codes — 50/50 split
        for code in 1_i64..=10 {
            for r in 0..10 {
                discharge.push(code);
                y.push(if r < 5 { 0 } else { 1 });
            }
        }
        DataFrame::from_columns(vec![
            ("discharge".into(), Column::Int(discharge)),
            ("y".into(), Column::Int(y)),
        ])
        .unwrap()
    }

    #[test]
    fn per_level_leakage_fires_on_death_codes() {
        let df = df_with_per_level_leak();
        let cfg = PerLevelLeakageConfig::default();
        let r = detect_per_level_target_leakage(&df, "y", &cfg);
        assert!(!r.is_empty(), "expected at least one E9064 finding");
        // Each of {11, 13, 14} should produce an E9064 finding
        // (15 rows of class 0, p=1.0).
        for code in ["11", "13", "14"] {
            assert!(
                r.iter().any(|f| f.code == "E9064"
                    && f.column.as_deref() == Some("discharge")
                    && f.message.contains(&format!("level `{}`", code))),
                "missing E9064 for discharge={}",
                code
            );
        }
        // Codes 1-10 should NOT fire (50/50 split is well below 0.99).
        for code in 1..=10 {
            assert!(
                !r.iter().any(|f| f.message.contains(&format!("level `{}`", code))),
                "false positive on non-leaking discharge={}",
                code
            );
        }
    }

    #[test]
    fn per_level_leakage_respects_min_support() {
        // 3 rows of code 11 → y=0; min_support=10 → no finding.
        let df = DataFrame::from_columns(vec![
            ("discharge".into(), Column::Int(vec![11, 11, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])),
            ("y".into(),         Column::Int(vec![ 0,  0,  0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])),
        ])
        .unwrap();
        let cfg = PerLevelLeakageConfig {
            min_support: 10,
            ..Default::default()
        };
        let r = detect_per_level_target_leakage(&df, "y", &cfg);
        assert!(
            r.is_empty(),
            "min_support should suppress finding (got {:?})",
            r
        );
    }

    #[test]
    fn per_level_leakage_skips_base_rate_dominated_classes() {
        // 99% of the dataset is class 0; the level 11 → class 0
        // relationship adds NO information vs the base rate.
        let mut discharge = vec![11_i64; 200];
        let mut y = vec![0_i64; 200];
        // Sprinkle a few class-1 rows on code 99 (not on code 11)
        for i in 198..200 {
            discharge[i] = 99;
            y[i] = 1;
        }
        let df = DataFrame::from_columns(vec![
            ("discharge".into(), Column::Int(discharge)),
            ("y".into(), Column::Int(y)),
        ])
        .unwrap();
        let cfg = PerLevelLeakageConfig::default();
        let r = detect_per_level_target_leakage(&df, "y", &cfg);
        // Base rate of class 0 is 198/200 = 0.99 ≥ threshold; the
        // level adds nothing, so we should NOT fire.
        assert!(r.is_empty(), "base-rate filter should suppress (got {:?})", r);
    }

    #[test]
    fn per_level_leakage_handles_str_levels() {
        // String-level case: status "Deceased" → y=0 deterministically.
        let mut status: Vec<String> = Vec::new();
        let mut y: Vec<i64> = Vec::new();
        for _ in 0..15 {
            status.push("Deceased".into());
            y.push(0);
        }
        for _ in 0..30 {
            status.push("Alive".into());
            y.push(0);
        }
        for _ in 0..30 {
            status.push("Alive".into());
            y.push(1);
        }
        let df = DataFrame::from_columns(vec![
            ("status".into(), Column::Str(status)),
            ("y".into(), Column::Int(y)),
        ])
        .unwrap();
        let cfg = PerLevelLeakageConfig::default();
        let r = detect_per_level_target_leakage(&df, "y", &cfg);
        assert!(
            r.iter().any(|f| f.code == "E9064" && f.message.contains("Deceased")),
            "expected E9064 finding on Deceased level (got {:?})",
            r
        );
    }

    #[test]
    fn per_level_leakage_skips_float_columns() {
        // Float columns are continuous — we don't try to bin per-value.
        let df = DataFrame::from_columns(vec![
            (
                "temp".into(),
                Column::Float((0..40).map(|i| i as f64).collect()),
            ),
            ("y".into(), Column::Int(vec![0_i64; 40])),
        ])
        .unwrap();
        let r = detect_per_level_target_leakage(
            &df, "y", &PerLevelLeakageConfig::default(),
        );
        assert!(r.is_empty());
    }

    #[test]
    fn per_level_leakage_is_deterministic() {
        let df = df_with_per_level_leak();
        let cfg = PerLevelLeakageConfig::default();
        let a = detect_per_level_target_leakage(&df, "y", &cfg);
        let b = detect_per_level_target_leakage(&df, "y", &cfg);
        assert_eq!(a, b);
    }

    #[test]
    fn per_level_leakage_handles_bool_target() {
        // Bool target accepted.
        let df = DataFrame::from_columns(vec![
            ("status".into(), Column::Str(vec!["dead".into(); 15].into_iter().chain(vec!["ok".into(); 100]).collect())),
            ("alive".into(), Column::Bool(vec![false; 15].into_iter().chain(vec![true; 50]).chain(vec![false; 50]).collect())),
        ])
        .unwrap();
        let r = detect_per_level_target_leakage(
            &df, "alive", &PerLevelLeakageConfig::default(),
        );
        assert!(
            r.iter().any(|f| f.code == "E9064" && f.message.contains("dead")),
            "expected E9064 on dead level vs Bool target (got {:?})",
            r
        );
    }
}
