//! Personally Identifiable Information (PII) detection (v0.6 batch 2).
//!
//! Four lightweight pattern detectors that scan `Column::Str` values for
//! shapes that resemble PII and emit a finding when the share crosses a
//! threshold. Locke does **not** redact or modify the data — it surfaces
//! a structured warning so the consumer can decide whether to redact,
//! hash, or document the column.
//!
//! ## Codes
//!
//! | Code  | Severity | What it flags |
//! |-------|----------|---------------|
//! | E9090 | Warning  | email-like strings (`local@domain.tld`) |
//! | E9091 | Notice   | phone-number-like strings (NA / E.164 / parenthesized) |
//! | E9092 | Error    | SSN-like strings (`NNN-NN-NNNN`); high-confidence US-SSN pattern |
//! | E9093 | Warning  | API-key / token-like strings (high-entropy alphanumeric ≥ 24 chars) |
//!
//! ## Determinism + zero-dep design
//!
//! Locke is zero-dep at runtime by policy ([[Locke Architecture]] §scope).
//! These detectors therefore use hand-rolled byte-level pattern recognisers
//! rather than the `regex` crate. The patterns are deliberately conservative
//! — high precision over recall — to keep the false-positive rate low on
//! legitimate non-PII data (e.g. URL paths, ISBNs, internal product SKUs).
//!
//! All detectors iterate columns in `df.columns` insertion order and emit
//! samples sorted by canonical key, so two runs over the same input
//! produce byte-identical findings.

use std::collections::BTreeMap;

use cjc_data::{Column, DataFrame};

use crate::report::{FindingEvidence, FindingSeverity, ValidationFinding};

// ─── Config ───────────────────────────────────────────────────────────────

/// Knobs for the PII detectors.
#[derive(Clone, Debug)]
pub struct PiiConfig {
    /// Skip detectors on columns with fewer rows than this.
    pub min_rows: u64,
    /// Each detector fires only when the share of matching rows crosses
    /// this. Default 0.10 — fire when ≥ 10% of the column looks like PII.
    pub min_match_share: f64,
    /// API-key heuristic: minimum length of an all-alphanumeric run to be
    /// considered key-like. Default 24 (covers most modern API keys).
    pub api_key_min_len: usize,
    /// API-key heuristic: minimum Shannon entropy (in bits per char) of
    /// the candidate string to count as high-entropy. Default 3.5.
    pub api_key_min_entropy_bits: f64,
}

impl Default for PiiConfig {
    fn default() -> Self {
        Self {
            min_rows: 10,
            min_match_share: 0.10,
            api_key_min_len: 24,
            api_key_min_entropy_bits: 3.5,
        }
    }
}

// ─── Pattern recognisers ──────────────────────────────────────────────────

/// True iff `s` matches the canonical email shape: `local@domain.tld`
/// where `local` and each `domain` segment is a non-empty sequence of
/// `[a-zA-Z0-9._+-]`, and `tld` has length ≥ 2 and is all alphabetic.
pub fn looks_like_email(s: &str) -> bool {
    // Must have exactly one `@` and at least one `.` after it.
    let at_pos = match s.find('@') {
        Some(p) => p,
        None => return false,
    };
    if s[at_pos + 1..].find('@').is_some() {
        return false; // multiple `@`
    }
    let (local, rest) = s.split_at(at_pos);
    let domain = &rest[1..];
    if local.is_empty() || domain.is_empty() {
        return false;
    }
    // Local part: ASCII + `._+-`.
    if !local.chars().all(|c| {
        c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '+' | '-')
    }) {
        return false;
    }
    // Domain split on `.`; need at least one dot, every segment non-empty,
    // segments alphanumeric/-, last segment alphabetic and length ≥ 2.
    let segs: Vec<&str> = domain.split('.').collect();
    if segs.len() < 2 {
        return false;
    }
    for seg in &segs {
        if seg.is_empty()
            || !seg.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
        {
            return false;
        }
    }
    let tld = segs.last().unwrap();
    tld.len() >= 2 && tld.chars().all(|c| c.is_ascii_alphabetic())
}

/// True iff `s` matches a plausible phone-number shape. Covers:
/// * E.164: `+CCNNNNNNNNNNN` (with optional spaces/dashes inside).
/// * NA: `NNN-NNN-NNNN` or `(NNN) NNN-NNNN`.
/// * Length-10 to length-15 digit runs with at most 4 separator chars.
pub fn looks_like_phone(s: &str) -> bool {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return false;
    }
    let mut digits = 0;
    let mut seps = 0;
    let mut leading_plus = false;
    for (i, c) in trimmed.chars().enumerate() {
        if i == 0 && c == '+' {
            leading_plus = true;
            continue;
        }
        if c.is_ascii_digit() {
            digits += 1;
        } else if matches!(c, ' ' | '-' | '(' | ')' | '.') {
            seps += 1;
        } else {
            return false;
        }
    }
    if leading_plus {
        // E.164 — 8 to 15 digits.
        digits >= 8 && digits <= 15 && seps <= 5
    } else {
        // NA-style — 10 digits with at most 4 separators.
        digits == 10 && seps <= 4
    }
}

/// True iff `s` matches the canonical US SSN format `NNN-NN-NNNN` with
/// strictly 11 characters and two dashes at positions 3 and 6.
///
/// We deliberately do **not** match raw 9-digit runs — those produce too
/// many false positives on synthetic ids / order numbers / etc.
pub fn looks_like_ssn(s: &str) -> bool {
    if s.len() != 11 {
        return false;
    }
    let bytes = s.as_bytes();
    if bytes[3] != b'-' || bytes[6] != b'-' {
        return false;
    }
    for (i, &b) in bytes.iter().enumerate() {
        if i == 3 || i == 6 {
            continue;
        }
        if !b.is_ascii_digit() {
            return false;
        }
    }
    true
}

/// Shannon entropy of a string treated as a uniform draw over its
/// character histogram. Returned in **bits per character**.
fn shannon_entropy_bits(s: &str) -> f64 {
    let chars: Vec<char> = s.chars().collect();
    if chars.is_empty() {
        return 0.0;
    }
    let mut hist: BTreeMap<char, u64> = BTreeMap::new();
    for &c in &chars {
        *hist.entry(c).or_insert(0) += 1;
    }
    let n = chars.len() as f64;
    let mut acc = cjc_repro::KahanAccumulatorF64::new();
    for &count in hist.values() {
        let p = count as f64 / n;
        if p > 0.0 {
            acc.add(-p * p.log2());
        }
    }
    acc.finalize()
}

/// True iff `s` looks like an API key or access token: at least
/// `min_len` characters, all alphanumeric / `-` / `_`, and Shannon
/// entropy ≥ `min_entropy_bits` bits per character.
pub fn looks_like_api_key(s: &str, min_len: usize, min_entropy_bits: f64) -> bool {
    if s.chars().count() < min_len {
        return false;
    }
    if !s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return false;
    }
    shannon_entropy_bits(s) >= min_entropy_bits
}

// ─── Detectors ────────────────────────────────────────────────────────────

fn emit_pii_finding(
    code: &'static str,
    severity: FindingSeverity,
    label: &str,
    column: &str,
    hits: u64,
    n_rows: u64,
    samples: &[(&str, u64)],
    assumptions: Vec<String>,
    suggested: Vec<String>,
) -> ValidationFinding {
    let share = hits as f64 / n_rows.max(1) as f64;
    let sample_str = samples
        .iter()
        .take(3)
        .map(|(s, c)| format!("{:?}:{}", s, c))
        .collect::<Vec<_>>()
        .join("; ");
    ValidationFinding::new(
        code,
        severity,
        format!(
            "column `{}` has {} {}-like value(s) ({:.1}% of rows)",
            column,
            hits,
            label,
            share * 100.0
        ),
        Some(column.into()),
        None,
        vec![
            FindingEvidence::Count {
                label: "n_pii_hits".into(),
                value: hits,
            },
            FindingEvidence::Ratio {
                label: "pii_share".into(),
                value: share.clamp(0.0, 1.0),
            },
            FindingEvidence::Sample {
                label: "sample".into(),
                value: sample_str,
            },
            FindingEvidence::Sample {
                label: "pii_kind".into(),
                value: label.into(),
            },
        ],
        n_rows,
        assumptions,
        suggested,
    )
}

/// Run all four PII detectors over `Column::Str` columns and return all
/// findings concatenated.
pub fn detect_all_pii(df: &DataFrame, cfg: &PiiConfig) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let n_rows = df.nrows() as u64;
    if n_rows < cfg.min_rows {
        return out;
    }
    for (name, col) in &df.columns {
        let Column::Str(values) = col else { continue };
        let mut email_hits = 0u64;
        let mut phone_hits = 0u64;
        let mut ssn_hits = 0u64;
        let mut api_key_hits = 0u64;
        let mut email_samples: BTreeMap<String, u64> = BTreeMap::new();
        let mut phone_samples: BTreeMap<String, u64> = BTreeMap::new();
        let mut ssn_samples: BTreeMap<String, u64> = BTreeMap::new();
        let mut api_key_samples: BTreeMap<String, u64> = BTreeMap::new();
        for v in values {
            if looks_like_email(v) {
                email_hits += 1;
                *email_samples.entry(v.clone()).or_insert(0) += 1;
            }
            if looks_like_phone(v) {
                phone_hits += 1;
                *phone_samples.entry(v.clone()).or_insert(0) += 1;
            }
            if looks_like_ssn(v) {
                ssn_hits += 1;
                *ssn_samples.entry(v.clone()).or_insert(0) += 1;
            }
            if looks_like_api_key(v, cfg.api_key_min_len, cfg.api_key_min_entropy_bits) {
                api_key_hits += 1;
                *api_key_samples.entry(v.clone()).or_insert(0) += 1;
            }
        }

        let to_samples = |m: &BTreeMap<String, u64>| -> Vec<(String, u64)> {
            m.iter().map(|(k, v)| (k.clone(), *v)).collect()
        };
        let share = |h: u64| h as f64 / n_rows as f64;

        if email_hits > 0 && share(email_hits) >= cfg.min_match_share {
            let samples = to_samples(&email_samples);
            let sample_refs: Vec<(&str, u64)> =
                samples.iter().map(|(s, c)| (s.as_str(), *c)).collect();
            out.push(emit_pii_finding(
                "E9090",
                FindingSeverity::Warning,
                "email",
                name,
                email_hits,
                n_rows,
                &sample_refs,
                vec![
                    "email pattern: `local@domain.tld` with conservative char set".into(),
                    "false-positive rate is low; false-negative on intentionally-malformed addresses".into(),
                ],
                vec![
                    "if this column should not contain emails, redact at ingest".into(),
                    "if it should, document the column as PII for downstream compliance".into(),
                ],
            ));
        }
        if phone_hits > 0 && share(phone_hits) >= cfg.min_match_share {
            let samples = to_samples(&phone_samples);
            let sample_refs: Vec<(&str, u64)> =
                samples.iter().map(|(s, c)| (s.as_str(), *c)).collect();
            out.push(emit_pii_finding(
                "E9091",
                FindingSeverity::Notice,
                "phone",
                name,
                phone_hits,
                n_rows,
                &sample_refs,
                vec![
                    "phone heuristic matches E.164 (+...) and NA (NNN-NNN-NNNN or (NNN) NNN-NNNN) shapes".into(),
                    "Notice severity because phone-shaped values legitimately appear in many columns (ZIP+4 numbers, product codes)".into(),
                ],
                vec![
                    "if intentional, document the column as PII; if not, redact at ingest".into(),
                ],
            ));
        }
        if ssn_hits > 0 && share(ssn_hits) >= cfg.min_match_share {
            let samples = to_samples(&ssn_samples);
            let sample_refs: Vec<(&str, u64)> =
                samples.iter().map(|(s, c)| (s.as_str(), *c)).collect();
            out.push(emit_pii_finding(
                "E9092",
                FindingSeverity::Error,
                "ssn",
                name,
                ssn_hits,
                n_rows,
                &sample_refs,
                vec![
                    "SSN pattern: exactly `NNN-NN-NNNN`; we deliberately reject raw 9-digit runs to avoid false positives".into(),
                    "Error severity because US-SSN exposure is almost always a compliance violation".into(),
                ],
                vec![
                    "redact at ingest immediately; SSNs should never reach a training set".into(),
                    "consider a hashing or tokenisation scheme upstream of Locke".into(),
                ],
            ));
        }
        if api_key_hits > 0 && share(api_key_hits) >= cfg.min_match_share {
            let samples = to_samples(&api_key_samples);
            let sample_refs: Vec<(&str, u64)> =
                samples.iter().map(|(s, c)| (s.as_str(), *c)).collect();
            out.push(emit_pii_finding(
                "E9093",
                FindingSeverity::Warning,
                "api_key",
                name,
                api_key_hits,
                n_rows,
                &sample_refs,
                vec![
                    format!(
                        "API-key heuristic: ≥ {} chars, alphanumeric+_-, Shannon entropy ≥ {} bits/char",
                        cfg.api_key_min_len, cfg.api_key_min_entropy_bits
                    ),
                    "false-positives possible on UUIDs / hashes / random IDs without `-`".into(),
                ],
                vec![
                    "if this column carries secrets, revoke and rotate immediately".into(),
                    "redact at ingest; secrets should not appear in any analytic store".into(),
                ],
            ));
        }
    }
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

    // ── Pattern recognisers ────────────────────────────────────────────

    #[test]
    fn email_canonical_shape() {
        assert!(looks_like_email("alice@example.com"));
        assert!(looks_like_email("bob.smith+tag@sub.example.co.uk"));
        assert!(looks_like_email("a1@b2.cd"));
        assert!(!looks_like_email("no-at-sign"));
        assert!(!looks_like_email("two@@signs.com"));
        assert!(!looks_like_email("@noprefix.com"));
        assert!(!looks_like_email("nosuffix@"));
        assert!(!looks_like_email("nodot@nope"));
        assert!(!looks_like_email("trailing@dot."));
        assert!(!looks_like_email("bad@one_letter.x")); // TLD too short
    }

    #[test]
    fn phone_e164_and_na_shapes() {
        assert!(looks_like_phone("+14155552671"));
        assert!(looks_like_phone("+1 415 555 2671"));
        assert!(looks_like_phone("415-555-2671"));
        assert!(looks_like_phone("(415) 555-2671"));
        assert!(!looks_like_phone("415-555")); // too few digits
        assert!(!looks_like_phone("not a phone"));
        assert!(!looks_like_phone("4155552671A"));
    }

    #[test]
    fn ssn_exact_format() {
        assert!(looks_like_ssn("123-45-6789"));
        assert!(!looks_like_ssn("123456789")); // no dashes
        assert!(!looks_like_ssn("12-345-6789")); // wrong dash positions
        assert!(!looks_like_ssn("123-AB-6789")); // non-digit
        assert!(!looks_like_ssn("123-45-67890")); // too long
    }

    #[test]
    fn api_key_high_entropy_required() {
        let hi = "abcDEF123XYZ_456-PQR789mnop";
        assert!(looks_like_api_key(hi, 24, 3.5));
        let lo = "aaaaaaaaaaaaaaaaaaaaaaaaaaaa"; // long but low entropy
        assert!(!looks_like_api_key(lo, 24, 3.5));
        let short = "abcDEF123";
        assert!(!looks_like_api_key(short, 24, 3.5));
    }

    #[test]
    fn shannon_entropy_bounds() {
        // Uniform 4-char alphabet → 2 bits/char.
        let h = shannon_entropy_bits("abcdabcdabcd");
        assert!((h - 2.0).abs() < 1e-6, "got {}", h);
        // Single-char string → 0 bits.
        let h = shannon_entropy_bits("aaaaa");
        assert_eq!(h, 0.0);
        // Empty string defined as 0.
        assert_eq!(shannon_entropy_bits(""), 0.0);
    }

    // ── Detectors ──────────────────────────────────────────────────────

    #[test]
    fn e9090_fires_when_share_crosses_threshold() {
        let mut values = vec!["alice@example.com", "bob@example.com", "carol@example.com"];
        values.extend(vec!["plain text"; 17]);
        let df = df_str("note", &values);
        let f = detect_all_pii(&df, &PiiConfig::default());
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].code, "E9090");
        assert_eq!(f[0].severity, FindingSeverity::Warning);
    }

    #[test]
    fn e9091_fires_on_phone_column() {
        let mut values: Vec<&str> = vec!["+14155550100", "+14155550101", "+14155550102"];
        values.extend(vec!["something else"; 17]);
        let df = df_str("contact", &values);
        let f = detect_all_pii(&df, &PiiConfig::default());
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].code, "E9091");
        assert_eq!(f[0].severity, FindingSeverity::Notice);
    }

    #[test]
    fn e9092_fires_on_ssn_column() {
        let mut values: Vec<&str> = vec!["111-22-3333", "444-55-6666", "777-88-9999"];
        values.extend(vec!["non-ssn"; 17]);
        let df = df_str("id", &values);
        let f = detect_all_pii(&df, &PiiConfig::default());
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].code, "E9092");
        assert_eq!(f[0].severity, FindingSeverity::Error);
    }

    #[test]
    fn e9093_fires_on_api_key_column() {
        let mut values: Vec<&str> = vec![
            "abcDEF123XYZ_456-PQR789mnop",
            "qwerASDF456ZXC-1234abcdef98",
            "x4Q-AbCdEfGhIjKlMnOp1234567",
        ];
        values.extend(vec!["short"; 17]);
        let df = df_str("token", &values);
        let f = detect_all_pii(&df, &PiiConfig::default());
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].code, "E9093");
        assert_eq!(f[0].severity, FindingSeverity::Warning);
    }

    #[test]
    fn detect_all_pii_is_deterministic() {
        let values = vec![
            "alice@example.com",
            "111-22-3333",
            "+14155550100",
            "abcDEF123XYZ_456-PQR789mnop",
        ];
        let mut all = values.clone();
        all.extend(vec!["filler"; 30]);
        let df = df_str("mixed", &all);
        let a = detect_all_pii(&df, &PiiConfig::default());
        let b = detect_all_pii(&df, &PiiConfig::default());
        assert_eq!(a, b);
    }

    #[test]
    fn no_pii_on_purely_numeric_column() {
        let df = DataFrame::from_columns(vec![
            ("x".into(), Column::Float(vec![1.0; 30])),
        ])
        .unwrap();
        assert!(detect_all_pii(&df, &PiiConfig::default()).is_empty());
    }

    #[test]
    fn below_threshold_does_not_fire() {
        // 1 email out of 50 rows = 2%, below default 10% threshold.
        let mut values = vec!["alice@example.com"];
        values.extend(vec!["plain"; 49]);
        let df = df_str("note", &values);
        assert!(detect_all_pii(&df, &PiiConfig::default()).is_empty());
    }
}
