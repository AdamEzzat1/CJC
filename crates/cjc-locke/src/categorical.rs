//! Categorical / string-quality detectors.
//!
//! These checks run on `Column::Str` and `Column::Categorical` columns and
//! surface semantic / encoding problems that the numeric-first validators
//! miss. Eight codes:
//!
//! | Code  | Severity | What it flags |
//! |-------|----------|---------------|
//! | E9016 | Notice   | Rare categories below `rare_category_min_count` (long-tail / overfit risk) |
//! | E9017 | Notice   | High cardinality above `one_hot_explosion_threshold` (encoding choice risk) |
//! | E9080 | Warning  | Categories that collide under simple lowercase fold (e.g. "Premium" / "premium") |
//! | E9081 | Notice   | Categories that collide after trim + strip-terminal-punctuation (e.g. "USA" / "USA.") |
//! | E9082 | Warning  | Categories within Levenshtein distance ≤ `edit_distance_threshold` of each other |
//! | E9083 | Warning  | Mixed-script strings (e.g. Latin + Cyrillic in one label) — confusable-character risk |
//! | E9084 | Notice   | Mojibake — UTF-8 decoded as Latin-1 patterns (e.g. "Ã©" for "é", "â€™" for "'") |
//! | E9085 | Notice   | Transitive cluster summary — when ≥ 2 of E9080/E9081/E9082 fire on a column, also emit a unified cluster view |
//!
//! ## Determinism
//!
//! Every detector iterates columns in `df.columns` insertion order, groups
//! findings by sorted canonical keys, and emits at most one finding per
//! (column, group). The edit-distance detector skips columns whose
//! distinct-cardinality exceeds `max_categories_for_edit_distance`, since
//! the comparison is O(N²·L); the skip emits a single Info finding rather
//! than silently doing partial work.
//!
//! ## Scope
//!
//! These detectors do **not** rewrite the column — they surface candidate
//! semantic problems for human review. Locke remains a skepticism layer,
//! not a data-cleaning layer; users decide whether to consolidate "Premium"
//! / "premium" upstream or treat them as distinct.

use std::collections::BTreeMap;

use cjc_data::{Column, DataFrame};

use crate::report::{FindingEvidence, FindingSeverity, ValidationFinding};

// ─── Config ───────────────────────────────────────────────────────────────

/// Knobs for the categorical-quality detectors.
#[derive(Clone, Debug)]
pub struct CategoricalQualityConfig {
    /// Categories with count below this threshold are "rare" for E9010.
    pub rare_category_min_count: u64,
    /// E9010 fires only if at least this many distinct categories cross the
    /// rare threshold. Avoids spamming on tiny columns.
    pub rare_category_min_rare_count: u64,
    /// E9011 fires when distinct cardinality on a Str/Categorical column
    /// exceeds this (suggests an embedding rather than one-hot).
    pub one_hot_explosion_threshold: u64,
    /// Detectors are skipped on columns with fewer rows than this.
    pub min_rows_for_detection: u64,
    /// Edit-distance detector skips strings shorter than this — short
    /// strings produce too many spurious near-duplicates.
    pub min_string_len_for_edit_distance: usize,
    /// Edit-distance threshold for E9082. Default 2 (typo / single-char-add).
    pub edit_distance_threshold: usize,
    /// Edit-distance detector is O(N²·L). Columns with more distinct
    /// categories than this are skipped (and surface an Info note).
    pub max_categories_for_edit_distance: usize,
    /// E9081 strips these trailing characters before comparing.
    pub trim_terminal_chars: &'static str,
    /// E9083 fires when more than this many *distinct* Unicode scripts
    /// appear inside a single category string (default 1 → fire on any
    /// mixed-script category). Latin + ASCII-digits + ASCII-punct count
    /// as a single script bucket.
    pub mixed_script_max_distinct: usize,
    /// E9083 ignores strings shorter than this (single-char strings can't
    /// meaningfully mix scripts).
    pub mixed_script_min_len: usize,
    /// E9084 (mojibake) fires when a single category string contains at
    /// least this many mojibake-signature characters.
    pub mojibake_min_signature_count: usize,
    /// E9085 (transitive cluster) fires when at least this many of
    /// {E9080, E9081, E9082} fired on the same column. Default 2: at least
    /// two distinct normalization channels agree something is off.
    pub transitive_cluster_min_signals: usize,
}

impl Default for CategoricalQualityConfig {
    fn default() -> Self {
        Self {
            rare_category_min_count: 5,
            rare_category_min_rare_count: 2,
            one_hot_explosion_threshold: 50,
            min_rows_for_detection: 10,
            min_string_len_for_edit_distance: 4,
            edit_distance_threshold: 2,
            max_categories_for_edit_distance: 200,
            trim_terminal_chars: ".,!?;:",
            mixed_script_max_distinct: 1,
            mixed_script_min_len: 3,
            mojibake_min_signature_count: 1,
            transitive_cluster_min_signals: 2,
        }
    }
}

// ─── Shared categorical-column iteration ──────────────────────────────────

/// Per-distinct-category count, in sorted order (`BTreeMap`).
type CategoryCounts = BTreeMap<String, u64>;

/// Materialise per-category counts for a Str, Categorical, or
/// CategoricalAdaptive column. Returns `None` for non-categorical
/// columns (numeric / bool / datetime).
///
/// For `Column::CategoricalAdaptive` we walk the adaptive code stream and
/// look up each code's bytes in the column's `ByteDictionary`, converting
/// to `String` via `from_utf8_lossy` (bytes are UTF-8 by `ByteDictionary`'s
/// design — `lossy` only triggers on corrupted streams). The result is
/// semantically identical to materialising the column to `Column::Str`
/// first, at O(rows × log(dict_size)) cost. Wired in v0.6.3.
///
/// Crate-visible so `per_value_lineage` can iterate distinct categorical
/// values without duplicating the dispatch over `Column` variants.
pub(crate) fn category_counts(col: &Column) -> Option<CategoryCounts> {
    let mut counts: CategoryCounts = BTreeMap::new();
    match col {
        Column::Str(v) => {
            for s in v {
                *counts.entry(s.clone()).or_insert(0u64) += 1;
            }
            Some(counts)
        }
        Column::Categorical { levels, codes } => {
            for &c in codes {
                let Some(label) = levels.get(c as usize) else {
                    continue;
                };
                *counts.entry(label.clone()).or_insert(0u64) += 1;
            }
            Some(counts)
        }
        Column::CategoricalAdaptive(cc) => {
            let dict = cc.dictionary();
            for code in cc.codes().iter() {
                let Some(bytes) = dict.get(code) else { continue };
                let label = String::from_utf8_lossy(bytes).to_string();
                *counts.entry(label).or_insert(0u64) += 1;
            }
            Some(counts)
        }
        _ => None,
    }
}

// ─── E9016 — Rare categories ──────────────────────────────────────────────

/// Fire E9016 (Notice) when a categorical column has multiple categories
/// appearing fewer than `cfg.rare_category_min_count` times. Long-tail
/// categories are an overfit / instability signal for downstream encoders.
pub fn detect_rare_categories(
    df: &DataFrame,
    cfg: &CategoricalQualityConfig,
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let n_rows = df.nrows() as u64;
    if n_rows < cfg.min_rows_for_detection {
        return out;
    }
    for (name, col) in &df.columns {
        let Some(counts) = category_counts(col) else {
            continue;
        };
        let n_distinct = counts.len() as u64;
        if n_distinct < 2 {
            continue;
        }
        let rares: Vec<(&String, u64)> = counts
            .iter()
            .filter(|(_, &c)| c < cfg.rare_category_min_count)
            .map(|(k, &c)| (k, c))
            .collect();
        if (rares.len() as u64) < cfg.rare_category_min_rare_count {
            continue;
        }
        let total_rare_rows: u64 = rares.iter().map(|(_, c)| *c).sum();
        let rare_share = total_rare_rows as f64 / n_rows as f64;
        // Show up to 5 sample rare categories, sorted ascending by count.
        // v0.7+ deep-dive perf-fix: the previous version collected twice
        // (`.collect::<Vec<_>>().into_iter().map(...).collect()`) — a
        // copy-paste-refactor leftover that built an intermediate Vec for
        // no semantic reason. Direct collect saves one allocation per call.
        let mut sample: Vec<(&&String, &u64)> =
            rares.iter().map(|(k, c)| (k, c)).collect();
        sample.sort_by(|a, b| (a.1, a.0).cmp(&(b.1, b.0)));
        let sample_str = sample
            .iter()
            .take(5)
            .map(|(k, c)| format!("{:?}:{}", k, c))
            .collect::<Vec<_>>()
            .join(", ");

        out.push(ValidationFinding::new(
            "E9016",
            FindingSeverity::Notice,
            format!(
                "column `{}` has {} rare categories (count < {}) covering {} rows ({:.1}%)",
                name,
                rares.len(),
                cfg.rare_category_min_count,
                total_rare_rows,
                rare_share * 100.0
            ),
            Some(name.clone()),
            None,
            vec![
                FindingEvidence::Count {
                    label: "n_rare_categories".into(),
                    value: rares.len() as u64,
                },
                FindingEvidence::Count {
                    label: "total_rare_rows".into(),
                    value: total_rare_rows,
                },
                FindingEvidence::Ratio {
                    label: "rare_share".into(),
                    value: rare_share.clamp(0.0, 1.0),
                },
                FindingEvidence::Count {
                    label: "n_distinct".into(),
                    value: n_distinct,
                },
                FindingEvidence::Sample {
                    label: "sample".into(),
                    value: sample_str,
                },
            ],
            n_rows,
            vec![
                "rare categories destabilise embeddings and one-hot encoders".into(),
                "a category with count 1 in the training split contributes a leaf seen exactly once".into(),
            ],
            vec![
                "consider grouping rare categories into an `__other__` bucket".into(),
                "or treating long-tail categories as the column's missing-value sentinel".into(),
            ],
        ));
    }
    out
}

// ─── E9017 — Encoding-risk (one-hot explosion) ────────────────────────────

/// Fire E9017 (Notice) when a string/categorical column has cardinality
/// above `cfg.one_hot_explosion_threshold`. This isn't a leakage claim
/// (E9072 covers ID-like cardinality near 1.0) — it's a downstream-encoding
/// hint: 312 distinct countries probably wants an embedding, not one-hot.
pub fn detect_encoding_risk(
    df: &DataFrame,
    cfg: &CategoricalQualityConfig,
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let n_rows = df.nrows() as u64;
    if n_rows < cfg.min_rows_for_detection {
        return out;
    }
    for (name, col) in &df.columns {
        let Some(counts) = category_counts(col) else {
            continue;
        };
        let n_distinct = counts.len() as u64;
        if n_distinct <= cfg.one_hot_explosion_threshold {
            continue;
        }
        let cardinality_ratio = n_distinct as f64 / n_rows.max(1) as f64;
        // E9072 covers the >=0.95 case; stay below that to avoid double-firing.
        if cardinality_ratio >= 0.95 {
            continue;
        }
        out.push(ValidationFinding::new(
            "E9017",
            FindingSeverity::Notice,
            format!(
                "column `{}` has {} distinct values (> {} threshold); one-hot encoding may produce sparse unstable features",
                name, n_distinct, cfg.one_hot_explosion_threshold
            ),
            Some(name.clone()),
            None,
            vec![
                FindingEvidence::Count {
                    label: "n_distinct".into(),
                    value: n_distinct,
                },
                FindingEvidence::Count {
                    label: "one_hot_threshold".into(),
                    value: cfg.one_hot_explosion_threshold,
                },
                FindingEvidence::Ratio {
                    label: "cardinality_ratio".into(),
                    value: cardinality_ratio.clamp(0.0, 1.0),
                },
            ],
            n_rows,
            vec![
                "this is an encoding-design hint, not an ID-leakage claim (see E9072 for the ≥0.95 case)".into(),
                "the threshold is heuristic; raise it for legitimately wide categorical features".into(),
            ],
            vec![
                "consider target / mean encoding, hashing trick, or a small embedding".into(),
                "if downstream is tree-based, ordinal-encoding with care may be fine".into(),
            ],
        ));
    }
    out
}

// ─── E9080 — Case-fold collisions ─────────────────────────────────────────

/// Group categories whose lowercase form collides. Fire E9080 (Warning)
/// per column that has at least one such group.
pub fn detect_case_fold_collisions(
    df: &DataFrame,
    cfg: &CategoricalQualityConfig,
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let n_rows = df.nrows() as u64;
    if n_rows < cfg.min_rows_for_detection {
        return out;
    }
    for (name, col) in &df.columns {
        let Some(counts) = category_counts(col) else {
            continue;
        };
        if counts.len() < 2 {
            continue;
        }
        // Group originals by lowercase folded key.
        let mut groups: BTreeMap<String, Vec<(String, u64)>> = BTreeMap::new();
        for (orig, count) in &counts {
            let folded = orig.to_lowercase();
            groups
                .entry(folded)
                .or_default()
                .push((orig.clone(), *count));
        }
        // Keep groups with >1 distinct ORIGINAL spelling.
        let collision_groups: Vec<(&String, &Vec<(String, u64)>)> = groups
            .iter()
            .filter(|(_, members)| members.len() > 1)
            .collect();
        if collision_groups.is_empty() {
            continue;
        }
        let total_collision_rows: u64 = collision_groups
            .iter()
            .flat_map(|(_, m)| m.iter().map(|(_, c)| *c))
            .sum();
        let sample_str = collision_groups
            .iter()
            .take(3)
            .map(|(folded, members)| {
                let inner = members
                    .iter()
                    .map(|(s, c)| format!("{:?}:{}", s, c))
                    .collect::<Vec<_>>()
                    .join("|");
                format!("{:?}=>{}", folded, inner)
            })
            .collect::<Vec<_>>()
            .join("; ");

        out.push(ValidationFinding::new(
            "E9080",
            FindingSeverity::Warning,
            format!(
                "column `{}` has {} case-fold collision group(s) covering {} rows; e.g. \"Premium\"/\"premium\" treated as distinct",
                name,
                collision_groups.len(),
                total_collision_rows
            ),
            Some(name.clone()),
            None,
            vec![
                FindingEvidence::Count {
                    label: "n_collision_groups".into(),
                    value: collision_groups.len() as u64,
                },
                FindingEvidence::Count {
                    label: "rows_in_collisions".into(),
                    value: total_collision_rows,
                },
                FindingEvidence::Ratio {
                    label: "collision_row_share".into(),
                    value: (total_collision_rows as f64 / n_rows as f64).clamp(0.0, 1.0),
                },
                FindingEvidence::Sample {
                    label: "sample".into(),
                    value: sample_str,
                },
            ],
            n_rows,
            vec![
                "case-fold comparison uses Rust `str::to_lowercase` (Unicode-aware)".into(),
                "some columns intentionally distinguish case (e.g. product SKUs); Locke surfaces a hint, not a verdict".into(),
            ],
            vec![
                "normalize case at ingest if the column is meant to be canonical".into(),
                "or accept the distinction explicitly and document it".into(),
            ],
        ));
    }
    out
}

// ─── E9081 — Whitespace / terminal-punctuation variants ───────────────────

/// Fold helper for E9081: trim outer whitespace, strip trailing punctuation
/// listed in `cfg.trim_terminal_chars`, and lowercase. Returns the
/// canonical form used for grouping.
///
/// Crate-visible so `per_value_lineage` can reproduce the exact same
/// canonicalisation when tracing a single value through the pipeline.
pub(crate) fn normalize_whitespace_punct(s: &str, terminal_chars: &str) -> String {
    let trimmed = s.trim();
    let mut bytes_end = trimmed.len();
    // Walk from the end stripping any chars in terminal_chars.
    // Operate on chars, not bytes, to respect UTF-8.
    let chars: Vec<(usize, char)> = trimmed.char_indices().collect();
    for i in (0..chars.len()).rev() {
        let (offset, ch) = chars[i];
        if terminal_chars.contains(ch) || ch.is_whitespace() {
            bytes_end = offset;
        } else {
            break;
        }
    }
    trimmed[..bytes_end].to_lowercase()
}

/// Fire E9081 (Notice) on columns where categories collapse under
/// trim + strip-terminal-punctuation + lowercase but were stored as
/// distinct strings.
pub fn detect_whitespace_punctuation_variants(
    df: &DataFrame,
    cfg: &CategoricalQualityConfig,
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let n_rows = df.nrows() as u64;
    if n_rows < cfg.min_rows_for_detection {
        return out;
    }
    for (name, col) in &df.columns {
        let Some(counts) = category_counts(col) else {
            continue;
        };
        if counts.len() < 2 {
            continue;
        }
        let mut groups: BTreeMap<String, Vec<(String, u64)>> = BTreeMap::new();
        for (orig, count) in &counts {
            let folded = normalize_whitespace_punct(orig, cfg.trim_terminal_chars);
            // Skip cases where folded is empty (e.g. orig was " . " — that's
            // a different problem — sentinel-like, not whitespace-variant).
            if folded.is_empty() {
                continue;
            }
            groups
                .entry(folded)
                .or_default()
                .push((orig.clone(), *count));
        }
        // To avoid double-reporting with E9080 (case-fold), only fire when
        // at least one collision group has members whose lowercase forms
        // differ — i.e. the whitespace/punctuation is the actual cause.
        let collision_groups: Vec<(&String, &Vec<(String, u64)>)> = groups
            .iter()
            .filter(|(_, members)| {
                if members.len() < 2 {
                    return false;
                }
                let mut lowered = members.iter().map(|(s, _)| s.to_lowercase()).collect::<Vec<_>>();
                lowered.sort();
                lowered.dedup();
                lowered.len() > 1
            })
            .collect();
        if collision_groups.is_empty() {
            continue;
        }
        let total_collision_rows: u64 = collision_groups
            .iter()
            .flat_map(|(_, m)| m.iter().map(|(_, c)| *c))
            .sum();
        let sample_str = collision_groups
            .iter()
            .take(3)
            .map(|(folded, members)| {
                let inner = members
                    .iter()
                    .map(|(s, c)| format!("{:?}:{}", s, c))
                    .collect::<Vec<_>>()
                    .join("|");
                format!("{:?}=>{}", folded, inner)
            })
            .collect::<Vec<_>>()
            .join("; ");

        out.push(ValidationFinding::new(
            "E9081",
            FindingSeverity::Notice,
            format!(
                "column `{}` has {} whitespace/punctuation variant group(s) covering {} rows; e.g. \"USA\"/\"USA.\"/\" USA \"",
                name,
                collision_groups.len(),
                total_collision_rows
            ),
            Some(name.clone()),
            None,
            vec![
                FindingEvidence::Count {
                    label: "n_variant_groups".into(),
                    value: collision_groups.len() as u64,
                },
                FindingEvidence::Count {
                    label: "rows_in_variants".into(),
                    value: total_collision_rows,
                },
                FindingEvidence::Ratio {
                    label: "variant_row_share".into(),
                    value: (total_collision_rows as f64 / n_rows as f64).clamp(0.0, 1.0),
                },
                FindingEvidence::Sample {
                    label: "sample".into(),
                    value: sample_str,
                },
                FindingEvidence::Sample {
                    label: "terminal_chars".into(),
                    value: cfg.trim_terminal_chars.into(),
                },
            ],
            n_rows,
            vec![
                "fold uses str::trim + strip-terminal-punct + lowercase".into(),
                "this fires in addition to E9080 only when the lowercase forms also differ".into(),
            ],
            vec![
                "normalize whitespace + trailing punctuation at ingest".into(),
                "or canonicalise via a controlled-vocabulary lookup".into(),
            ],
        ));
    }
    out
}

// ─── E9082 — Near-duplicate categories (edit distance) ────────────────────

/// Bounded Levenshtein distance. Returns `threshold + 1` as a sentinel when
/// the distance exceeds the threshold (so callers compare with `<= threshold`).
///
/// Two-row DP, O(min(m,n) · L) bytes; bails early when the row minimum
/// exceeds the threshold.
fn bounded_edit_distance(a: &str, b: &str, threshold: usize) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let (a_chars, b_chars) = if a_chars.len() <= b_chars.len() {
        (a_chars, b_chars)
    } else {
        (b_chars, a_chars)
    };
    let m = a_chars.len();
    let n = b_chars.len();
    if n.saturating_sub(m) > threshold {
        return threshold + 1;
    }
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr: Vec<usize> = vec![0; m + 1];
    for j in 1..=n {
        curr[0] = j;
        let mut row_min = curr[0];
        for i in 1..=m {
            let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
            curr[i] = (prev[i - 1] + cost)
                .min(prev[i] + 1)
                .min(curr[i - 1] + 1);
            if curr[i] < row_min {
                row_min = curr[i];
            }
        }
        if row_min > threshold {
            return threshold + 1;
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

/// Fire E9082 (Warning) on columns with at least one pair of distinct
/// categories within `cfg.edit_distance_threshold`.
pub fn detect_near_duplicate_categories(
    df: &DataFrame,
    cfg: &CategoricalQualityConfig,
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let n_rows = df.nrows() as u64;
    if n_rows < cfg.min_rows_for_detection {
        return out;
    }
    for (name, col) in &df.columns {
        let Some(counts) = category_counts(col) else {
            continue;
        };
        let n_distinct = counts.len();
        if n_distinct < 2 {
            continue;
        }
        if n_distinct > cfg.max_categories_for_edit_distance {
            // Single Info note rather than partial work.
            out.push(ValidationFinding::new(
                "E9082",
                FindingSeverity::Info,
                format!(
                    "column `{}` has {} distinct values, above the edit-distance scan cap ({}); skipped",
                    name, n_distinct, cfg.max_categories_for_edit_distance
                ),
                Some(name.clone()),
                None,
                vec![
                    FindingEvidence::Count {
                        label: "n_distinct".into(),
                        value: n_distinct as u64,
                    },
                    FindingEvidence::Count {
                        label: "scan_cap".into(),
                        value: cfg.max_categories_for_edit_distance as u64,
                    },
                ],
                n_rows,
                vec!["edit-distance scan is O(N²·L); skipped to preserve responsiveness".into()],
                vec![
                    "raise `max_categories_for_edit_distance` if you need this column scanned".into(),
                ],
            ));
            continue;
        }
        // Keep only categories long enough that small edit distances are
        // meaningful (skip "M" vs "F" – they're 1 apart but legitimate).
        let mut keys: Vec<(&String, u64)> = counts
            .iter()
            .filter(|(k, _)| k.chars().count() >= cfg.min_string_len_for_edit_distance)
            .map(|(k, c)| (k, *c))
            .collect();
        if keys.len() < 2 {
            continue;
        }
        keys.sort_by(|a, b| a.0.cmp(b.0));

        let threshold = cfg.edit_distance_threshold;
        // Collect (a, b, distance, count_a, count_b) for pairs within threshold.
        // To avoid massive false-positive spam, only keep pairs where:
        //   - distance <= threshold
        //   - distance > 0 (skip exact match)
        //   - |len(a) - len(b)| <= threshold
        // To stay deterministic, iterate sorted pairs (i<j).
        struct Pair<'a> {
            a: &'a str,
            b: &'a str,
            distance: usize,
            count_a: u64,
            count_b: u64,
        }
        let mut pairs: Vec<Pair> = Vec::new();
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                let (a, ca) = (keys[i].0, keys[i].1);
                let (b, cb) = (keys[j].0, keys[j].1);
                let la = a.chars().count();
                let lb = b.chars().count();
                if la.abs_diff(lb) > threshold {
                    continue;
                }
                let d = bounded_edit_distance(a, b, threshold);
                if d == 0 || d > threshold {
                    continue;
                }
                pairs.push(Pair {
                    a,
                    b,
                    distance: d,
                    count_a: ca,
                    count_b: cb,
                });
            }
        }
        if pairs.is_empty() {
            continue;
        }
        // Stable order: by distance ascending, then alphabetical.
        pairs.sort_by(|x, y| {
            (x.distance, x.a, x.b).cmp(&(y.distance, y.a, y.b))
        });
        let sample_str = pairs
            .iter()
            .take(5)
            .map(|p| format!("{:?}~{:?}(d={})", p.a, p.b, p.distance))
            .collect::<Vec<_>>()
            .join("; ");
        let total_pair_rows: u64 = pairs.iter().map(|p| p.count_a + p.count_b).sum();

        out.push(ValidationFinding::new(
            "E9082",
            FindingSeverity::Warning,
            format!(
                "column `{}` has {} near-duplicate category pair(s) within edit distance ≤ {}",
                name,
                pairs.len(),
                threshold
            ),
            Some(name.clone()),
            None,
            vec![
                FindingEvidence::Count {
                    label: "n_near_dup_pairs".into(),
                    value: pairs.len() as u64,
                },
                FindingEvidence::Count {
                    label: "edit_distance_threshold".into(),
                    value: threshold as u64,
                },
                FindingEvidence::Count {
                    label: "rows_in_pairs".into(),
                    value: total_pair_rows,
                },
                FindingEvidence::Sample {
                    label: "sample".into(),
                    value: sample_str,
                },
            ],
            n_rows,
            vec![
                "Levenshtein distance is character-level; short strings produce noise (cap via min_string_len)".into(),
                "high-d small-edit pairs may be legitimate (e.g. \"USA\" vs \"USB\" — different products)".into(),
            ],
            vec![
                "review pairs; consolidate via controlled vocabulary if they are semantic duplicates".into(),
                "for typo-driven duplicates, normalize at ingest with a fuzzy-match lookup".into(),
            ],
        ));
    }
    out
}

// ─── E9083 — Confusable / mixed-script characters ─────────────────────────

/// Classify a `char` into a coarse script bucket. Latin + ASCII digits +
/// common ASCII punctuation are folded into a single bucket because mixing
/// them inside one label is normal (e.g. "USA-1"). Everything else groups
/// by `char::is_ascii` falsehood + the leading Unicode block boundaries
/// most likely to host confusables.
fn script_bucket(c: char) -> &'static str {
    if c.is_ascii() {
        // Latin block (uppercase/lowercase) + digits + punctuation + space.
        return "latin";
    }
    // Common confusable origins. We don't try to enumerate every script —
    // the goal is to detect "this label has chars from more than one
    // visually-confusable bucket".
    let code = c as u32;
    match code {
        0x0080..=0x024F => "latin_extended",
        0x0370..=0x03FF => "greek",
        0x0400..=0x04FF => "cyrillic",
        0x0500..=0x052F => "cyrillic_supp",
        0x0530..=0x058F => "armenian",
        0x0590..=0x05FF => "hebrew",
        0x0600..=0x06FF => "arabic",
        0x0900..=0x097F => "devanagari",
        0x4E00..=0x9FFF => "cjk",
        0x3040..=0x309F => "hiragana",
        0x30A0..=0x30FF => "katakana",
        0xAC00..=0xD7AF => "hangul",
        _ => "other",
    }
}

/// Fire E9083 (Warning) on category strings whose characters span more
/// than `cfg.mixed_script_max_distinct` script buckets. This is the
/// classic Unicode confusable attack vector ("раypal" with Cyrillic 'а' /
/// 'р' looks identical to "paypal" but compares unequal).
pub fn detect_confusable_scripts(
    df: &DataFrame,
    cfg: &CategoricalQualityConfig,
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let n_rows = df.nrows() as u64;
    if n_rows < cfg.min_rows_for_detection {
        return out;
    }
    for (name, col) in &df.columns {
        let Some(counts) = category_counts(col) else {
            continue;
        };
        // BTreeMap from offending label -> (n_buckets, row_count, script set).
        let mut hits: BTreeMap<String, (usize, u64, Vec<&'static str>)> = BTreeMap::new();
        for (orig, count) in &counts {
            let nchars = orig.chars().count();
            if nchars < cfg.mixed_script_min_len {
                continue;
            }
            let mut buckets: Vec<&'static str> =
                orig.chars().map(script_bucket).collect();
            buckets.sort_unstable();
            buckets.dedup();
            if buckets.len() > cfg.mixed_script_max_distinct {
                hits.insert(orig.clone(), (buckets.len(), *count, buckets));
            }
        }
        if hits.is_empty() {
            continue;
        }
        let total_hit_rows: u64 = hits.values().map(|(_, c, _)| *c).sum();
        let sample_str = hits
            .iter()
            .take(3)
            .map(|(s, (nb, c, buckets))| {
                format!("{:?}({}sc:[{}]):{}", s, nb, buckets.join(","), c)
            })
            .collect::<Vec<_>>()
            .join("; ");

        out.push(ValidationFinding::new(
            "E9083",
            FindingSeverity::Warning,
            format!(
                "column `{}` has {} category string(s) mixing > {} Unicode script(s); confusable-character risk",
                name,
                hits.len(),
                cfg.mixed_script_max_distinct
            ),
            Some(name.clone()),
            None,
            vec![
                FindingEvidence::Count {
                    label: "n_mixed_strings".into(),
                    value: hits.len() as u64,
                },
                FindingEvidence::Count {
                    label: "rows_in_mixed".into(),
                    value: total_hit_rows,
                },
                FindingEvidence::Ratio {
                    label: "mixed_row_share".into(),
                    value: (total_hit_rows as f64 / n_rows as f64).clamp(0.0, 1.0),
                },
                FindingEvidence::Sample {
                    label: "sample".into(),
                    value: sample_str,
                },
            ],
            n_rows,
            vec![
                "script buckets group ASCII as `latin`; Latin Extended / Greek / Cyrillic / CJK / etc. are separate buckets".into(),
                "mixed script can be legitimate (multilingual data) or an attack vector (homoglyph spoofing)".into(),
            ],
            vec![
                "if multilingual is expected, raise mixed_script_max_distinct".into(),
                "otherwise normalize to a canonical script at ingest".into(),
            ],
        ));
    }
    out
}

// ─── E9084 — Mojibake (UTF-8 decoded as Latin-1) ──────────────────────────

/// Count the mojibake-signature characters in a single string. The
/// signatures here are the canonical residue of "I decoded UTF-8 bytes as
/// Latin-1 / Windows-1252":
///
/// * UTF-8 2-byte sequences starting with `0xC3` (covers é/è/à/ç/ñ...)
///   become "Ã" + Latin-1-supplement char (U+0080..U+00FF).
/// * UTF-8 2-byte sequences starting with `0xC2` (covers ©/®/°/non-breaking
///   space...) become "Â" + Latin-1-supplement char.
/// * UTF-8 3-byte sequences for smart quotes / em-dash (U+2018..U+201F /
///   U+2013..U+2014) become "â" + Windows-1252 special (€/‚/„/™/etc.).
fn count_mojibake_signatures(s: &str) -> usize {
    let chars: Vec<char> = s.chars().collect();
    let mut n = 0;
    let mut i = 0;
    fn in_latin1_supp(c: char) -> bool {
        let code = c as u32;
        (0x0080..=0x00FF).contains(&code)
    }
    fn is_win1252_special(c: char) -> bool {
        // The classic Windows-1252 extras that show up in mojibake'd
        // smart quotes / em-dashes / etc.
        matches!(
            c,
            '\u{20AC}' // €
            | '\u{201A}' | '\u{201E}' | '\u{2026}' // ‚ „ …
            | '\u{2020}' | '\u{2021}' // † ‡
            | '\u{02C6}' | '\u{02DC}' // ˆ ˜
            | '\u{2030}'              // ‰
            | '\u{0160}' | '\u{0152}' // Š Œ
            | '\u{2018}' | '\u{2019}' // ‘ ’
            | '\u{201C}' | '\u{201D}' // “ ”
            | '\u{2022}'              // •
            | '\u{2013}' | '\u{2014}' // – —
            | '\u{2122}'              // ™
            | '\u{0161}' | '\u{0153}' // š œ
        )
    }
    while i + 1 < chars.len() {
        let (a, b) = (chars[i], chars[i + 1]);
        // Latin-1 high-byte residue from UTF-8 2-byte:
        //   Ã + Latin-1-supp char → was original é/è/à/ñ/...
        //   Â + Latin-1-supp char → was original ©/®/°/non-breaking-space/...
        if (a == 'Ã' || a == 'Â')
            && (in_latin1_supp(b) || b.is_ascii_alphabetic())
        {
            n += 1;
            i += 2;
            continue;
        }
        // 3-byte residue: â + Windows-1252 special → was smart-quote /
        // dash / bullet / etc.
        if a == 'â' && is_win1252_special(b) {
            n += 1;
            // Advance by 2 (the third char of the original triple may or
            // may not be present after Latin-1 → Win-1252 substitution).
            i += 2;
            continue;
        }
        i += 1;
    }
    n
}

/// Fire E9084 (Notice) on columns where at least one category string
/// shows mojibake signatures above `cfg.mojibake_min_signature_count`.
pub fn detect_mojibake(
    df: &DataFrame,
    cfg: &CategoricalQualityConfig,
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let n_rows = df.nrows() as u64;
    if n_rows < cfg.min_rows_for_detection {
        return out;
    }
    for (name, col) in &df.columns {
        let Some(counts) = category_counts(col) else {
            continue;
        };
        let mut hits: BTreeMap<String, (usize, u64)> = BTreeMap::new();
        for (orig, count) in &counts {
            let n = count_mojibake_signatures(orig);
            if n >= cfg.mojibake_min_signature_count {
                hits.insert(orig.clone(), (n, *count));
            }
        }
        if hits.is_empty() {
            continue;
        }
        let total_hit_rows: u64 = hits.values().map(|(_, c)| *c).sum();
        let sample_str = hits
            .iter()
            .take(3)
            .map(|(s, (n, c))| format!("{:?}(sig={}):{}", s, n, c))
            .collect::<Vec<_>>()
            .join("; ");
        out.push(ValidationFinding::new(
            "E9084",
            FindingSeverity::Notice,
            format!(
                "column `{}` has {} category string(s) showing mojibake signatures (UTF-8 decoded as Latin-1)",
                name,
                hits.len()
            ),
            Some(name.clone()),
            None,
            vec![
                FindingEvidence::Count {
                    label: "n_mojibake_strings".into(),
                    value: hits.len() as u64,
                },
                FindingEvidence::Count {
                    label: "rows_in_mojibake".into(),
                    value: total_hit_rows,
                },
                FindingEvidence::Ratio {
                    label: "mojibake_row_share".into(),
                    value: (total_hit_rows as f64 / n_rows as f64).clamp(0.0, 1.0),
                },
                FindingEvidence::Sample {
                    label: "sample".into(),
                    value: sample_str,
                },
            ],
            n_rows,
            vec![
                "signatures: \"Ã\"/\"Â\"+ASCII (Latin-1 residue of UTF-8) and \"â€\"+ASCII (UTF-8 smart-quote residue)".into(),
                "false-positives are possible on legitimate multilingual text containing these characters in context".into(),
            ],
            vec![
                "re-ingest from source with correct encoding (UTF-8 in, UTF-8 out)".into(),
                "or run a one-pass decode-twice fix (`text.encode('latin-1').decode('utf-8')`)".into(),
            ],
        ));
    }
    out
}

// ─── E9085 — Transitive cluster summary ───────────────────────────────────

/// When multiple semantic-duplicate channels (E9080 / E9081 / E9082) fire
/// on the same column, emit a unified summary that lists how many channels
/// agreed and the total share of rows in collisions. Doesn't replace the
/// per-channel findings — it sits alongside them for the per-column view.
pub fn detect_transitive_clusters(
    df: &DataFrame,
    cfg: &CategoricalQualityConfig,
    prior_findings: &[ValidationFinding],
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let n_rows = df.nrows() as u64;
    if n_rows < cfg.min_rows_for_detection {
        return out;
    }

    // Group prior findings by column where code ∈ {E9080, E9081, E9082}.
    let mut per_column: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for f in prior_findings {
        if let Some(col) = f.column.as_deref() {
            if matches!(f.code, "E9080" | "E9081" | "E9082") {
                per_column.entry(col).or_default().push(f.code);
            }
        }
    }
    // For each column that crosses the threshold, emit E9085.
    for (col, codes) in &per_column {
        let mut sorted_codes: Vec<&str> = codes.iter().copied().collect();
        sorted_codes.sort_unstable();
        sorted_codes.dedup();
        if sorted_codes.len() < cfg.transitive_cluster_min_signals {
            continue;
        }
        out.push(ValidationFinding::new(
            "E9085",
            FindingSeverity::Notice,
            format!(
                "column `{}` has semantic-duplicate signals from {} channels: {}",
                col,
                sorted_codes.len(),
                sorted_codes.join(", ")
            ),
            Some((*col).into()),
            None,
            vec![
                FindingEvidence::Count {
                    label: "n_channels".into(),
                    value: sorted_codes.len() as u64,
                },
                FindingEvidence::Sample {
                    label: "channels".into(),
                    value: sorted_codes.join(","),
                },
            ],
            n_rows,
            vec![
                "this finding is a *summary* of prior E9080/E9081/E9082 findings on this column".into(),
                "the per-channel findings remain authoritative for exact counts and pairs".into(),
            ],
            vec![
                "multiple channels agreeing strongly suggests upstream normalization is required".into(),
                "consider establishing a controlled vocabulary for this column".into(),
            ],
        ));
    }
    out
}

// ─── E9086 — Unicode NFC/NFD normalization variants ───────────────────────

/// Approximate normalisation for NFC/NFD collision detection: strip all
/// Unicode combining marks (U+0300..U+036F + the more obscure ranges)
/// AND map Latin-1 / Latin-Extended-A precomposed accented letters to
/// their ASCII bases. This sidesteps a full `unicode-normalization`
/// dependency while still catching the most common real-world case: a
/// precomposed `é` (U+00E9, NFC) coexisting with `e + ́`
/// (U+0065 U+0301, NFD).
///
/// Both forms map to the same ASCII string under this function, which
/// is what `detect_unicode_normalization_variants` uses for grouping.
///
/// Crate-visible so `per_value_lineage` can reproduce the exact same
/// canonicalisation when tracing a single value through the pipeline.
pub(crate) fn strip_combining_marks(s: &str) -> String {
    s.chars()
        .filter(|c| {
            let code = *c as u32;
            !matches!(code,
                0x0300..=0x036F
                | 0x1AB0..=0x1AFF
                | 0x1DC0..=0x1DFF
                | 0x20D0..=0x20FF
                | 0xFE20..=0xFE2F
            )
        })
        .map(latin_to_base_ascii)
        .collect()
}

/// Map a single Latin-1 / Latin-Extended-A precomposed letter to its
/// ASCII base (case-preserving). Returns the input unchanged for any
/// char outside these blocks. Hand-rolled small table — covers the
/// ~150 most common Western European accented letters without pulling
/// in a Unicode database.
fn latin_to_base_ascii(c: char) -> char {
    match c {
        // Lowercase Latin-1 supplement accented letters
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' | 'ă' | 'ą' => 'a',
        'ç' | 'ć' | 'č' | 'ĉ' | 'ċ' => 'c',
        'ď' | 'đ' => 'd',
        'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ĕ' | 'ė' | 'ę' | 'ě' => 'e',
        'ĝ' | 'ğ' | 'ġ' | 'ģ' => 'g',
        'ĥ' | 'ħ' => 'h',
        'ì' | 'í' | 'î' | 'ï' | 'ĩ' | 'ī' | 'ĭ' | 'į' | 'ı' => 'i',
        'ĵ' => 'j',
        'ķ' => 'k',
        'ĺ' | 'ļ' | 'ľ' | 'ŀ' | 'ł' => 'l',
        'ñ' | 'ń' | 'ņ' | 'ň' | 'ŉ' => 'n',
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ō' | 'ŏ' | 'ő' => 'o',
        'ŕ' | 'ŗ' | 'ř' => 'r',
        'ś' | 'ŝ' | 'ş' | 'š' => 's',
        'ţ' | 'ť' | 'ŧ' => 't',
        'ù' | 'ú' | 'û' | 'ü' | 'ũ' | 'ū' | 'ŭ' | 'ů' | 'ű' | 'ų' => 'u',
        'ŵ' => 'w',
        'ý' | 'ÿ' | 'ŷ' => 'y',
        'ź' | 'ż' | 'ž' => 'z',
        'ß' => 's',
        'æ' | 'œ' => 'a',
        // Uppercase
        'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'Å' | 'Ā' | 'Ă' | 'Ą' => 'A',
        'Ç' | 'Ć' | 'Č' | 'Ĉ' | 'Ċ' => 'C',
        'Ď' | 'Đ' => 'D',
        'È' | 'É' | 'Ê' | 'Ë' | 'Ē' | 'Ĕ' | 'Ė' | 'Ę' | 'Ě' => 'E',
        'Ĝ' | 'Ğ' | 'Ġ' | 'Ģ' => 'G',
        'Ĥ' | 'Ħ' => 'H',
        'Ì' | 'Í' | 'Î' | 'Ï' | 'Ĩ' | 'Ī' | 'Ĭ' | 'Į' | 'İ' => 'I',
        'Ĵ' => 'J',
        'Ķ' => 'K',
        'Ĺ' | 'Ļ' | 'Ľ' | 'Ŀ' | 'Ł' => 'L',
        'Ñ' | 'Ń' | 'Ņ' | 'Ň' => 'N',
        'Ò' | 'Ó' | 'Ô' | 'Õ' | 'Ö' | 'Ø' | 'Ō' | 'Ŏ' | 'Ő' => 'O',
        'Ŕ' | 'Ŗ' | 'Ř' => 'R',
        'Ś' | 'Ŝ' | 'Ş' | 'Š' => 'S',
        'Ţ' | 'Ť' | 'Ŧ' => 'T',
        'Ù' | 'Ú' | 'Û' | 'Ü' | 'Ũ' | 'Ū' | 'Ŭ' | 'Ů' | 'Ű' | 'Ų' => 'U',
        'Ŵ' => 'W',
        'Ý' | 'Ÿ' | 'Ŷ' => 'Y',
        'Ź' | 'Ż' | 'Ž' => 'Z',
        other => other,
    }
}

/// True iff the string contains at least one combining mark — i.e. it is
/// (probably) in NFD form for one of the affected ranges. Used to gate
/// E9086 below: we only fire when one of the colliding spellings is
/// actually decomposed.
fn has_combining_mark(s: &str) -> bool {
    s.chars().any(|c| {
        let code = c as u32;
        matches!(code,
            0x0300..=0x036F
            | 0x1AB0..=0x1AFF
            | 0x1DC0..=0x1DFF
            | 0x20D0..=0x20FF
            | 0xFE20..=0xFE2F
        )
    })
}

/// Fire E9086 (Warning) on columns where at least one pair of categories
/// becomes equal after stripping all combining marks, AND at least one
/// of them carries a combining mark in its stored form. This identifies
/// NFC vs NFD mixing — the most common cause of "the same word appearing
/// twice but compare-unequal" in datasets ingested from heterogeneous
/// sources (macOS HFS+ filenames in NFD, Linux files in NFC, copy-paste
/// from web in either form).
pub fn detect_unicode_normalization_variants(
    df: &DataFrame,
    cfg: &CategoricalQualityConfig,
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let n_rows = df.nrows() as u64;
    if n_rows < cfg.min_rows_for_detection {
        return out;
    }
    for (name, col) in &df.columns {
        let Some(counts) = category_counts(col) else {
            continue;
        };
        if counts.len() < 2 {
            continue;
        }
        // Group by combining-mark-stripped form.
        let mut groups: BTreeMap<String, Vec<(String, u64)>> = BTreeMap::new();
        for (orig, count) in &counts {
            let key = strip_combining_marks(orig);
            // Skip if the orig and stripped form are equal AND no other
            // category has a combining mark — that's a fully NFC-only
            // column where stripping is a no-op.
            groups.entry(key).or_default().push((orig.clone(), *count));
        }
        // Keep only groups with multiple originals AND at least one of
        // them carries a combining mark.
        let collision_groups: Vec<(&String, &Vec<(String, u64)>)> = groups
            .iter()
            .filter(|(_, members)| {
                members.len() > 1
                    && members.iter().any(|(s, _)| has_combining_mark(s))
            })
            .collect();
        if collision_groups.is_empty() {
            continue;
        }
        let total_collision_rows: u64 = collision_groups
            .iter()
            .flat_map(|(_, m)| m.iter().map(|(_, c)| *c))
            .sum();
        let sample_str = collision_groups
            .iter()
            .take(3)
            .map(|(stripped, members)| {
                let inner = members
                    .iter()
                    .map(|(s, c)| {
                        let tag = if has_combining_mark(s) { "NFD" } else { "NFC" };
                        format!("{:?}({}):{}", s, tag, c)
                    })
                    .collect::<Vec<_>>()
                    .join("|");
                format!("{:?}=>{}", stripped, inner)
            })
            .collect::<Vec<_>>()
            .join("; ");

        out.push(ValidationFinding::new(
            "E9086",
            FindingSeverity::Warning,
            format!(
                "column `{}` has {} NFC/NFD collision group(s) covering {} rows; the same word is stored in multiple Unicode normalisation forms",
                name,
                collision_groups.len(),
                total_collision_rows
            ),
            Some(name.clone()),
            None,
            vec![
                FindingEvidence::Count {
                    label: "n_normalization_groups".into(),
                    value: collision_groups.len() as u64,
                },
                FindingEvidence::Count {
                    label: "rows_in_normalization".into(),
                    value: total_collision_rows,
                },
                FindingEvidence::Ratio {
                    label: "normalization_row_share".into(),
                    value: (total_collision_rows as f64 / n_rows as f64).clamp(0.0, 1.0),
                },
                FindingEvidence::Sample {
                    label: "sample".into(),
                    value: sample_str,
                },
            ],
            n_rows,
            vec![
                "fold strips Unicode combining marks (U+0300-U+036F and related blocks)".into(),
                "this is a crude approximation of NFD-then-discard, not a full Unicode normalisation".into(),
                "false-positive rate is low because we require at least one member to carry a combining mark".into(),
            ],
            vec![
                "normalise to NFC at ingest (consider adding `unicode-normalization` dep at the source layer)".into(),
                "or store the column as its `.chars().nfc()` form before downstream use".into(),
            ],
        ));
    }
    out
}

// ─── Aggregate ─────────────────────────────────────────────────────────────

/// Run all nine categorical-quality detectors and concatenate results.
///
/// Order matters slightly: E9085 (transitive cluster) consumes the
/// findings of E9080/E9081/E9082, so we run those first.
pub fn detect_all_categorical_quality(
    df: &DataFrame,
    cfg: &CategoricalQualityConfig,
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    out.extend(detect_rare_categories(df, cfg));
    out.extend(detect_encoding_risk(df, cfg));
    out.extend(detect_case_fold_collisions(df, cfg));
    out.extend(detect_whitespace_punctuation_variants(df, cfg));
    out.extend(detect_near_duplicate_categories(df, cfg));
    out.extend(detect_confusable_scripts(df, cfg));
    out.extend(detect_mojibake(df, cfg));
    out.extend(detect_unicode_normalization_variants(df, cfg));
    // Transitive-cluster summary must see the prior findings.
    let cluster_summaries = detect_transitive_clusters(df, cfg, &out);
    out.extend(cluster_summaries);
    out
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_data::{Column, DataFrame};

    fn df_str(name: &str, values: &[&str]) -> DataFrame {
        DataFrame::from_columns(vec![(
            name.into(),
            Column::Str(values.iter().map(|s| (*s).into()).collect()),
        )])
        .unwrap()
    }

    fn df_categorical(name: &str, levels: &[&str], codes: &[u32]) -> DataFrame {
        DataFrame::from_columns(vec![(
            name.into(),
            Column::Categorical {
                levels: levels.iter().map(|s| (*s).into()).collect(),
                codes: codes.to_vec(),
            },
        )])
        .unwrap()
    }

    // ── E9016 ─────────────────────────────────────────────────────────

    #[test]
    fn e9016_fires_on_long_tail() {
        // 12 rows: 9 of "common", 1 each of three rare values.
        let mut values: Vec<&str> = vec!["common"; 9];
        values.extend(["rare_a", "rare_b", "rare_c"]);
        let df = df_str("plan", &values);
        let cfg = CategoricalQualityConfig::default();
        let f = detect_rare_categories(&df, &cfg);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].code, "E9016");
        assert_eq!(f[0].column.as_deref(), Some("plan"));
    }

    #[test]
    fn e9016_quiet_when_only_one_rare() {
        let mut values: Vec<&str> = vec!["a"; 5];
        values.extend(["b"; 5]);
        values.push("only_rare");
        let df = df_str("plan", &values);
        let cfg = CategoricalQualityConfig::default();
        // min_rare_count=2 by default — single rare should be quiet.
        assert!(detect_rare_categories(&df, &cfg).is_empty());
    }

    #[test]
    fn e9016_works_on_categorical_storage() {
        let codes: Vec<u32> = (0..12)
            .map(|i| if i < 9 { 0 } else { (i - 8) as u32 })
            .collect();
        let df = df_categorical("plan", &["common", "rare_a", "rare_b", "rare_c"], &codes);
        let f = detect_rare_categories(&df, &CategoricalQualityConfig::default());
        assert_eq!(f.len(), 1);
    }

    // ── E9017 ─────────────────────────────────────────────────────────

    #[test]
    fn e9017_fires_above_threshold() {
        // 100 rows, 60 distinct values — above default 50, below 0.95 ratio.
        let values: Vec<String> = (0..100).map(|i| format!("v{:02}", i % 60)).collect();
        let v_refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        let df = df_str("country", &v_refs);
        let f = detect_encoding_risk(&df, &CategoricalQualityConfig::default());
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].code, "E9017");
    }

    #[test]
    fn e9017_quiet_when_id_like() {
        // 100 rows, 99 distinct — that's E9072 territory, not E9011.
        let values: Vec<String> = (0..100).map(|i| format!("u{:03}", i % 99)).collect();
        let v_refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        let df = df_str("uid", &v_refs);
        // cardinality_ratio = 99/100 = 0.99 → skipped.
        assert!(detect_encoding_risk(&df, &CategoricalQualityConfig::default()).is_empty());
    }

    // ── E9080 ─────────────────────────────────────────────────────────

    #[test]
    fn e9080_fires_on_case_collision() {
        let values = vec![
            "Premium", "premium", "PREMIUM", "basic", "basic", "basic", "basic", "basic", "basic",
            "basic", "basic", "basic",
        ];
        let df = df_str("tier", &values);
        let f = detect_case_fold_collisions(&df, &CategoricalQualityConfig::default());
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].code, "E9080");
        assert_eq!(f[0].severity, FindingSeverity::Warning);
    }

    #[test]
    fn e9080_quiet_when_no_collision() {
        let df = df_str("tier", &["Premium", "basic", "trial", "trial", "trial", "trial", "trial", "trial", "trial", "trial"]);
        assert!(detect_case_fold_collisions(&df, &CategoricalQualityConfig::default()).is_empty());
    }

    // ── E9081 ─────────────────────────────────────────────────────────

    #[test]
    fn e9081_fires_on_punct_and_whitespace_variants() {
        // "USA", "USA.", " USA ", "USA!" all collapse under
        // (trim + strip-terminal-punct + lowercase) to "usa", but their
        // raw lowercase forms differ (" usa ", "usa", "usa!", "usa."), so
        // E9080 cannot catch them. E9081 should.
        let values = vec![
            "USA",
            "USA.",
            " USA ",
            "USA!",
            "Canada",
            "Canada",
            "Canada",
            "Canada",
            "Canada",
            "Canada",
        ];
        let df = df_str("country", &values);
        let f = detect_whitespace_punctuation_variants(&df, &CategoricalQualityConfig::default());
        assert_eq!(f.len(), 1, "expected one E9081 group, got {:?}", f);
        assert_eq!(f[0].code, "E9081");

        // And E9080 must be quiet on this same data — none of the
        // lowercase forms collide.
        let g = detect_case_fold_collisions(&df, &CategoricalQualityConfig::default());
        assert!(g.is_empty(), "E9080 should be quiet here, got {:?}", g);
    }

    #[test]
    fn e9081_quiet_when_terminal_strip_doesnt_collapse() {
        // "U.S.A." → strip terminal '.' → "U.S.A" → lowercase "u.s.a"
        // "USA." → strip terminal '.' → "USA" → lowercase "usa"
        // Distinct normalized forms → no E9081 collision.
        let values = vec![
            "U.S.A.",
            "USA.",
            "Mexico",
            "Mexico",
            "Mexico",
            "Mexico",
            "Mexico",
            "Mexico",
            "Mexico",
            "Mexico",
        ];
        let df = df_str("country", &values);
        let f = detect_whitespace_punctuation_variants(&df, &CategoricalQualityConfig::default());
        assert!(f.is_empty(), "expected E9081 quiet, got {:?}", f);
    }

    #[test]
    fn e9081_fires_on_spaced_variants() {
        // "California" vs "California " vs "California." all collapse under
        // trim + strip-terminal-punct + lowercase to "california". Lowercase
        // alone gives {"california", "california ", "california."} — three
        // distinct, so E9081 fires.
        let values = vec![
            "California",
            "California ",
            "California.",
            "Texas",
            "Texas",
            "Texas",
            "Texas",
            "Texas",
            "Texas",
            "Texas",
        ];
        let df = df_str("state", &values);
        let f = detect_whitespace_punctuation_variants(&df, &CategoricalQualityConfig::default());
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].code, "E9081");
    }

    // ── E9082 ─────────────────────────────────────────────────────────

    #[test]
    fn e9082_detects_typo_pair() {
        let values = vec![
            "enterprise",
            "enterprize",
            "starter",
            "starter",
            "starter",
            "starter",
            "starter",
            "starter",
            "starter",
            "starter",
        ];
        let df = df_str("plan", &values);
        let f = detect_near_duplicate_categories(&df, &CategoricalQualityConfig::default());
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].code, "E9082");
        assert_eq!(f[0].severity, FindingSeverity::Warning);
    }

    #[test]
    fn e9082_skips_short_strings() {
        let mut values = vec!["M", "F"]; // distance 1 but length 1.
        values.extend(["X"; 10]);
        let df = df_str("sex", &values);
        // Default min_string_len_for_edit_distance = 4 → skip.
        assert!(detect_near_duplicate_categories(&df, &CategoricalQualityConfig::default()).is_empty());
    }

    #[test]
    fn e9082_emits_info_when_capped() {
        // 250 distinct strings, cap is 200.
        let values: Vec<String> = (0..250).map(|i| format!("abcdef{:03}", i)).collect();
        let v_refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        let df = df_str("token", &v_refs);
        let f = detect_near_duplicate_categories(&df, &CategoricalQualityConfig::default());
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].code, "E9082");
        assert_eq!(f[0].severity, FindingSeverity::Info);
    }

    // ── Determinism ─────────────────────────────────────────────────────

    #[test]
    fn detect_all_is_deterministic() {
        let values = vec![
            "Premium", "premium", "PREMIUM", "enterprise", "enterprize",
            "basic", "basic", "basic", "basic", "basic", "basic", "basic",
        ];
        let df = df_str("tier", &values);
        let cfg = CategoricalQualityConfig::default();
        let a = detect_all_categorical_quality(&df, &cfg);
        let b = detect_all_categorical_quality(&df, &cfg);
        assert_eq!(a, b, "two runs must produce identical findings");
    }

    // ── bounded_edit_distance ───────────────────────────────────────────

    #[test]
    fn edit_distance_basic() {
        assert_eq!(bounded_edit_distance("kitten", "sitting", 3), 3);
        assert_eq!(bounded_edit_distance("flaw", "lawn", 2), 2);
        assert_eq!(bounded_edit_distance("foo", "foo", 2), 0);
        // Above threshold returns sentinel = threshold+1.
        assert_eq!(bounded_edit_distance("kitten", "sitting", 1), 2);
    }

    // ── E9083 ───────────────────────────────────────────────────────────

    #[test]
    fn e9083_fires_on_cyrillic_in_latin() {
        // "pаypal" — the second char is Cyrillic 'а' (U+0430), not Latin 'a' (U+0061).
        let mut values = vec!["pаypal"];
        values.extend(vec!["paypal"; 20]);
        let df = df_str("brand", &values);
        let f = detect_confusable_scripts(&df, &CategoricalQualityConfig::default());
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].code, "E9083");
        assert_eq!(f[0].severity, FindingSeverity::Warning);
    }

    #[test]
    fn e9083_quiet_on_pure_latin() {
        let mut values = vec!["paypal", "stripe", "amazon", "google"];
        values.extend(vec!["paypal"; 10]);
        let df = df_str("brand", &values);
        assert!(detect_confusable_scripts(&df, &CategoricalQualityConfig::default()).is_empty());
    }

    #[test]
    fn e9083_quiet_on_pure_cyrillic_text() {
        // All chars in same Cyrillic bucket → no mixed-script finding.
        let mut values: Vec<String> = vec!["привет".into()];
        values.extend((0..20).map(|_| "москва".to_string()));
        let v_refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        let df = df_str("city", &v_refs);
        assert!(detect_confusable_scripts(&df, &CategoricalQualityConfig::default()).is_empty());
    }

    #[test]
    fn e9083_skips_short_strings() {
        // Single char of mixed script doesn't fire (min_len = 3 default).
        let mut values = vec!["а"]; // single Cyrillic
        values.extend(vec!["a"; 20]);
        let df = df_str("c", &values);
        assert!(detect_confusable_scripts(&df, &CategoricalQualityConfig::default()).is_empty());
    }

    // ── E9084 ───────────────────────────────────────────────────────────

    #[test]
    fn e9084_fires_on_classic_mojibake() {
        // "café" decoded as Latin-1 produces "cafÃ©".
        let mut values = vec!["cafÃ©"];
        values.extend(vec!["bistro"; 20]);
        let df = df_str("venue", &values);
        let f = detect_mojibake(&df, &CategoricalQualityConfig::default());
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].code, "E9084");
        assert_eq!(f[0].severity, FindingSeverity::Notice);
    }

    #[test]
    fn e9084_quiet_on_clean_text() {
        let values = vec!["café", "bistro", "restaurant", "salon", "tavern"];
        let mut all = values;
        all.extend(vec!["café"; 20]);
        let df = df_str("venue", &all);
        assert!(detect_mojibake(&df, &CategoricalQualityConfig::default()).is_empty());
    }

    #[test]
    fn e9084_counts_signatures_correctly() {
        // Two signatures: "Ã©" and "Ã¨".
        assert_eq!(count_mojibake_signatures("cafÃ© et thÃ¨"), 2);
        // Smart quote sig: â€™.
        assert_eq!(count_mojibake_signatures("itâ€™s"), 1);
        // Pure ASCII / clean utf-8.
        assert_eq!(count_mojibake_signatures("café"), 0);
        assert_eq!(count_mojibake_signatures(""), 0);
    }

    // ── E9085 ───────────────────────────────────────────────────────────

    #[test]
    fn e9085_fires_when_two_channels_agree() {
        // Construct findings as if E9080 and E9081 both fired on "plan".
        let prior = vec![
            ValidationFinding::new(
                "E9080",
                FindingSeverity::Warning,
                "case-fold",
                Some("plan".into()),
                None,
                vec![],
                100,
                vec![],
                vec![],
            ),
            ValidationFinding::new(
                "E9081",
                FindingSeverity::Notice,
                "whitespace",
                Some("plan".into()),
                None,
                vec![],
                100,
                vec![],
                vec![],
            ),
        ];
        // We need a df only to read n_rows; build one with 100 rows.
        let values: Vec<String> = (0..100).map(|i| format!("x{}", i)).collect();
        let v_refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        let df = df_str("plan", &v_refs);
        let out = detect_transitive_clusters(&df, &CategoricalQualityConfig::default(), &prior);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].code, "E9085");
        assert_eq!(out[0].column.as_deref(), Some("plan"));
    }

    #[test]
    fn e9085_quiet_with_single_channel() {
        let prior = vec![ValidationFinding::new(
            "E9080",
            FindingSeverity::Warning,
            "only one",
            Some("plan".into()),
            None,
            vec![],
            100,
            vec![],
            vec![],
        )];
        let values: Vec<String> = (0..100).map(|i| format!("x{}", i)).collect();
        let v_refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        let df = df_str("plan", &v_refs);
        assert!(detect_transitive_clusters(&df, &CategoricalQualityConfig::default(), &prior).is_empty());
    }

    #[test]
    fn detect_all_includes_e9083_e9084_e9085() {
        // Multi-issue column: case-fold + edit-dist + mojibake + mixed-script.
        let mut values: Vec<&str> = vec!["Premium", "premium", "PREMIUM", "enterprise", "enterprize"];
        values.extend(["cafÃ©", "pаypal"]);
        values.extend(vec!["basic"; 20]);
        let df = df_str("tier", &values);
        let f = detect_all_categorical_quality(&df, &CategoricalQualityConfig::default());
        let codes: std::collections::BTreeSet<&str> = f.iter().map(|x| x.code).collect();
        for code in ["E9080", "E9082", "E9083", "E9084", "E9085"] {
            assert!(codes.contains(code), "expected {} in {:?}", code, codes);
        }
    }

    // ── script_bucket — direct unit test ────────────────────────────────

    #[test]
    fn script_bucket_classifies() {
        assert_eq!(script_bucket('a'), "latin");
        assert_eq!(script_bucket('A'), "latin");
        assert_eq!(script_bucket('1'), "latin");
        assert_eq!(script_bucket('.'), "latin");
        assert_eq!(script_bucket('а'), "cyrillic"); // U+0430
        assert_eq!(script_bucket('α'), "greek"); // U+03B1
        assert_eq!(script_bucket('中'), "cjk");
    }
}
