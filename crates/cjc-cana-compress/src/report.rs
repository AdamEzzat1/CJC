//! [`CompressionReport`] — the deterministic output of
//! [`crate::CompressionPlan::execute`].
//!
//! ## Why this matches `cjc_cana::CanaReport`'s shape
//!
//! `cjc_cana::CanaReport` hand-writes its own JSON sidecar instead of
//! depending on `serde`. We follow the same pattern for the same reasons:
//! a 100-LOC canonical writer with `BTreeMap`-equivalent ordering and an
//! integer schema-version stamp is auditable; pulling in serde for the
//! sake of one report type is not.
//!
//! ## Canonical bytes
//!
//! [`CompressionReport::canonical_bytes`] is the *foundation* of every
//! determinism property in the crate. Two reports with identical contents
//! produce byte-identical bytes; the wiring test "same input → same
//! report hash" passes by virtue of this function being deterministic.
//!
//! ## Per-entry status surface
//!
//! Every executed entry lands in one of four statuses:
//!
//! - **`Validated`** — round-trip / tolerance check passed.
//! - **`MalformedRoundTrip`** — lossless reconstruction's hash differed
//!   from the input's. Indicates a compressor bug; should never happen
//!   in production but the type-level surface refuses to assume away the
//!   possibility.
//! - **`ToleranceExceeded`** — observed reconstruction error exceeded
//!   the declared advisory tolerance.
//! - **`DecodeFailed`** — a low-level [`crate::candidate::CompressionError`]
//!   surfaced (bad magic, malformed shape, etc.).

use cjc_cana::hash::hash_bytes;

use crate::candidate::{CandidateId, CompressionError, CompressionKind, CriticalityTag};

// ---------------------------------------------------------------------------
// ReportHash — newtype wrapper to prevent confusion with other u64s
// ---------------------------------------------------------------------------

/// Content-addressed fingerprint of a [`CompressionReport`]. Same hash
/// type-newtype pattern as [`cjc_cana::hash::ProgramHash`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReportHash(pub u64);

impl ReportHash {
    /// Render as a lowercase 16-char hex string.
    pub fn to_hex(self) -> String {
        format!("{:016x}", self.0)
    }
}

// ---------------------------------------------------------------------------
// ReportEntry
// ---------------------------------------------------------------------------

/// Per-candidate execution outcome.
///
/// Every field is value-typed so the struct is `Clone`-able and equality-
/// checkable. The fields appear in *display order*, which is the same
/// order `canonical_bytes` writes them — this is the only ordering rule
/// the report needs because we keep one entry per row.
#[derive(Debug, Clone)]
pub struct ReportEntry {
    /// Slot index in the plan that produced this entry.
    pub slot: u32,
    /// Candidate ID.
    pub candidate_id: CandidateId,
    /// Human-readable candidate label.
    pub candidate_label: String,
    /// Kind of compression applied.
    pub kind: CompressionKind,
    /// Whether the input was semantic-critical or advisory.
    pub criticality_tag: CriticalityTag,
    /// Declared advisory tolerance (`0.0` for semantic-critical entries).
    pub declared_tolerance: f64,
    /// Length of the original payload in bytes.
    pub original_len: usize,
    /// Length of the compressed bytes (canonical bytes for lossy
    /// summaries; raw byte stream for lossless codecs).
    pub compressed_len: usize,
    /// FNV-1a hash of the original payload.
    pub input_hash: u64,
    /// FNV-1a hash of the compressed summary's canonical bytes.
    pub summary_hash: u64,
    /// FNV-1a hash of the reconstructed payload (for lossy compressors,
    /// of the reconstructed f64 matrix bit-pattern).
    pub reconstructed_hash: u64,
    /// Observed reconstruction error (`0.0` for lossless).
    pub observed_error: f64,
    /// Validation outcome.
    pub status: EntryStatus,
}

impl ReportEntry {
    /// True iff this entry's compression passed both the round-trip and
    /// (where applicable) tolerance check.
    pub fn is_validated(&self) -> bool {
        matches!(self.status, EntryStatus::Validated)
    }

    /// Compression ratio (compressed bytes / original bytes). `1.0` means
    /// no savings; values `< 1.0` are wins.
    pub fn ratio(&self) -> f64 {
        if self.original_len == 0 {
            0.0
        } else {
            (self.compressed_len as f64) / (self.original_len as f64)
        }
    }
}

// ---------------------------------------------------------------------------
// EntryStatus
// ---------------------------------------------------------------------------

/// Validation outcome for one report entry.
///
/// `PartialEq` + `Clone` because callers join across this enum frequently
/// (e.g. a test wants to assert `entry.status == EntryStatus::Validated`).
#[derive(Debug, Clone, PartialEq)]
pub enum EntryStatus {
    /// Compression produced a round-trip-correct output within tolerance.
    Validated,

    /// Lossless decompression returned data whose hash didn't match the
    /// input. Should be impossible if the codec is correct; the variant
    /// exists so a compressor bug surfaces in the report rather than
    /// silently corrupting downstream consumers.
    MalformedRoundTrip,

    /// Lossy compression's observed Frobenius error exceeded the
    /// candidate's declared tolerance.
    ToleranceExceeded {
        /// Tolerance the candidate declared.
        declared: f64,
        /// Error the compressor measured.
        observed: f64,
    },

    /// A [`CompressionError`] from the underlying compressor (bad magic,
    /// invalid shape, malformed back-reference, etc.).
    DecodeFailed {
        /// The underlying error.
        error: CompressionError,
    },
}

impl EntryStatus {
    /// Stable single-byte discriminator for [`CompressionReport::canonical_bytes`].
    pub const fn tag_byte(&self) -> u8 {
        match self {
            EntryStatus::Validated => 0,
            EntryStatus::MalformedRoundTrip => 1,
            EntryStatus::ToleranceExceeded { .. } => 2,
            EntryStatus::DecodeFailed { .. } => 3,
        }
    }

    /// Short label for display / JSON serialization.
    pub const fn short_label(&self) -> &'static str {
        match self {
            EntryStatus::Validated => "validated",
            EntryStatus::MalformedRoundTrip => "malformed_round_trip",
            EntryStatus::ToleranceExceeded { .. } => "tolerance_exceeded",
            EntryStatus::DecodeFailed { .. } => "decode_failed",
        }
    }
}

// ---------------------------------------------------------------------------
// CompressionReport
// ---------------------------------------------------------------------------

/// Top-level report for one plan execution.
#[derive(Debug, Clone)]
pub struct CompressionReport {
    schema_version: u32,
    plan_hash: u64,
    entries: Vec<ReportEntry>,
    /// Content-addressed hash of the canonical bytes.
    report_hash: ReportHash,
}

impl CompressionReport {
    /// Construct a report from a plan hash + per-entry outcomes. Computes
    /// `report_hash` from the canonical bytes.
    pub fn new(plan_hash: u64, entries: Vec<ReportEntry>) -> Self {
        let mut report = Self {
            schema_version: 1,
            plan_hash,
            entries,
            report_hash: ReportHash(0),
        };
        report.report_hash = ReportHash(hash_bytes(&report.canonical_bytes_internal()));
        report
    }

    /// Schema version. Bumped on incompatible JSON / bytes changes.
    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Plan hash this report executed.
    pub fn plan_hash(&self) -> u64 {
        self.plan_hash
    }

    /// Iterate entries in slot order.
    pub fn entries(&self) -> impl Iterator<Item = &ReportEntry> {
        self.entries.iter()
    }

    /// `true` iff every entry is `Validated`.
    pub fn all_validated(&self) -> bool {
        self.entries.iter().all(|e| e.is_validated())
    }

    /// `true` iff this report contains an entry of the given kind. Useful
    /// in tests that mix lossless + lossy candidates and want to assert
    /// both branches executed.
    pub fn contains_kind(&self, kind: CompressionKind) -> bool {
        self.entries.iter().any(|e| e.kind == kind)
    }

    /// Content-addressed report hash. Same bytes as
    /// [`CompressionReport::canonical_bytes`] would emit, but pre-computed
    /// at construction time so callers don't have to hash the bytes again.
    pub fn report_hash(&self) -> ReportHash {
        self.report_hash
    }

    /// Canonical byte representation. Deterministic for byte-identical
    /// input. Used by:
    ///
    /// - [`Self::report_hash`] (under the hood)
    /// - Wiring tests (snapshot-style equality)
    /// - The bridge to NSS pressure-density updates
    ///   (treats the report bytes as the audit fingerprint of the
    ///   compression decision)
    pub fn canonical_bytes(&self) -> Vec<u8> {
        self.canonical_bytes_internal()
    }

    /// Render a deterministic JSON sidecar — same format philosophy as
    /// [`cjc_cana::CanaReport::to_json`]: handwritten, BTreeMap-style key
    /// ordering, no serde dependency.
    pub fn to_json(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::with_capacity(256 + self.entries.len() * 256);
        s.push_str("{\n");
        writeln!(s, "  \"schema_version\": {},", self.schema_version).unwrap();
        writeln!(s, "  \"crate_version\": \"{}\",", crate::COMPRESS_VERSION).unwrap();
        writeln!(s, "  \"plan_hash\": \"{:016x}\",", self.plan_hash).unwrap();
        writeln!(s, "  \"report_hash\": \"{}\",", self.report_hash.to_hex()).unwrap();
        writeln!(s, "  \"entry_count\": {},", self.entries.len()).unwrap();
        s.push_str("  \"entries\": [\n");
        for (i, e) in self.entries.iter().enumerate() {
            s.push_str("    {\n");
            writeln!(s, "      \"slot\": {},", e.slot).unwrap();
            writeln!(s, "      \"candidate_id\": {},", e.candidate_id.0).unwrap();
            writeln!(
                s,
                "      \"candidate_label\": \"{}\",",
                json_escape_string(&e.candidate_label)
            )
            .unwrap();
            writeln!(s, "      \"kind\": \"{}\",", e.kind.short_label()).unwrap();
            writeln!(
                s,
                "      \"criticality\": \"{}\",",
                e.criticality_tag.short_label()
            )
            .unwrap();
            writeln!(
                s,
                "      \"declared_tolerance\": {},",
                f64_as_json_number(e.declared_tolerance)
            )
            .unwrap();
            writeln!(s, "      \"original_len\": {},", e.original_len).unwrap();
            writeln!(s, "      \"compressed_len\": {},", e.compressed_len).unwrap();
            writeln!(s, "      \"input_hash\": \"{:016x}\",", e.input_hash).unwrap();
            writeln!(s, "      \"summary_hash\": \"{:016x}\",", e.summary_hash).unwrap();
            writeln!(
                s,
                "      \"reconstructed_hash\": \"{:016x}\",",
                e.reconstructed_hash
            )
            .unwrap();
            writeln!(
                s,
                "      \"observed_error\": {},",
                f64_as_json_number(e.observed_error)
            )
            .unwrap();
            writeln!(s, "      \"status\": \"{}\"", e.status.short_label()).unwrap();
            if i + 1 < self.entries.len() {
                s.push_str("    },\n");
            } else {
                s.push_str("    }\n");
            }
        }
        s.push_str("  ]\n");
        s.push_str("}\n");
        s
    }

    // -----------------------------------------------------------------------
    // Internal byte writer
    // -----------------------------------------------------------------------

    fn canonical_bytes_internal(&self) -> Vec<u8> {
        // Magic + schema version + plan hash + entry count + entries.
        // Entries are written in slot order (which is the iteration order
        // of `self.entries`, which Plan::new sorted by ID).
        let mut out = Vec::with_capacity(64 + self.entries.len() * 64);
        out.extend_from_slice(b"CCR0"); // CANA Compression Report v0
        out.extend_from_slice(&self.schema_version.to_le_bytes());
        out.extend_from_slice(&self.plan_hash.to_le_bytes());
        out.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for e in &self.entries {
            // Tag bytes for stable, compact representation.
            out.extend_from_slice(&e.slot.to_le_bytes());
            out.extend_from_slice(&e.candidate_id.0.to_le_bytes());
            // Length-prefixed label.
            out.extend_from_slice(&(e.candidate_label.len() as u32).to_le_bytes());
            out.extend_from_slice(e.candidate_label.as_bytes());
            out.push(e.kind.tag_byte());
            out.push(e.criticality_tag.tag_byte());
            out.extend_from_slice(&e.declared_tolerance.to_bits().to_le_bytes());
            out.extend_from_slice(&(e.original_len as u64).to_le_bytes());
            out.extend_from_slice(&(e.compressed_len as u64).to_le_bytes());
            out.extend_from_slice(&e.input_hash.to_le_bytes());
            out.extend_from_slice(&e.summary_hash.to_le_bytes());
            out.extend_from_slice(&e.reconstructed_hash.to_le_bytes());
            out.extend_from_slice(&e.observed_error.to_bits().to_le_bytes());
            out.push(e.status.tag_byte());
            // For statuses that carry extra payload, append it after the
            // tag byte so older decoders can skip; current schema embeds
            // it inline.
            match &e.status {
                EntryStatus::Validated | EntryStatus::MalformedRoundTrip => {}
                EntryStatus::ToleranceExceeded { declared, observed } => {
                    out.extend_from_slice(&declared.to_bits().to_le_bytes());
                    out.extend_from_slice(&observed.to_bits().to_le_bytes());
                }
                EntryStatus::DecodeFailed { error } => {
                    let msg = format!("{error}");
                    out.extend_from_slice(&(msg.len() as u32).to_le_bytes());
                    out.extend_from_slice(msg.as_bytes());
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

fn json_escape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                use std::fmt::Write as _;
                write!(&mut out, "\\u{:04x}", c as u32).unwrap();
            }
            c => out.push(c),
        }
    }
    out
}

fn f64_as_json_number(x: f64) -> String {
    // JSON doesn't support NaN / ±Inf — render them as JSON strings so the
    // file remains parseable. This shouldn't happen in practice because
    // every report-emitting compressor clamps to finite values.
    if x.is_nan() {
        "\"NaN\"".to_string()
    } else if x.is_infinite() {
        if x.is_sign_positive() {
            "\"Inf\"".to_string()
        } else {
            "\"-Inf\"".to_string()
        }
    } else {
        format!("{x:.10e}")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::{CandidateId, CompressionKind, CriticalityTag};

    fn validated_entry(slot: u32, id: u64, label: &str) -> ReportEntry {
        ReportEntry {
            slot,
            candidate_id: CandidateId(id),
            candidate_label: label.to_string(),
            kind: CompressionKind::LosslessTrace,
            criticality_tag: CriticalityTag::SemanticCritical,
            declared_tolerance: 0.0,
            original_len: 10,
            compressed_len: 8,
            input_hash: 0xABC,
            summary_hash: 0xDEF,
            reconstructed_hash: 0xABC,
            observed_error: 0.0,
            status: EntryStatus::Validated,
        }
    }

    #[test]
    fn report_hash_is_deterministic() {
        let entries = vec![validated_entry(0, 1, "x"), validated_entry(1, 2, "y")];
        let r1 = CompressionReport::new(0xCAFE, entries.clone());
        let r2 = CompressionReport::new(0xCAFE, entries);
        assert_eq!(r1.report_hash(), r2.report_hash());
        assert_eq!(r1.canonical_bytes(), r2.canonical_bytes());
    }

    #[test]
    fn report_hash_differs_on_different_entries() {
        let r1 = CompressionReport::new(0xCAFE, vec![validated_entry(0, 1, "x")]);
        let r2 = CompressionReport::new(0xCAFE, vec![validated_entry(0, 1, "y")]);
        assert_ne!(r1.report_hash(), r2.report_hash());
    }

    #[test]
    fn report_hash_differs_on_different_plan_hash() {
        let entries = vec![validated_entry(0, 1, "x")];
        let r1 = CompressionReport::new(1, entries.clone());
        let r2 = CompressionReport::new(2, entries);
        assert_ne!(r1.report_hash(), r2.report_hash());
    }

    #[test]
    fn all_validated_returns_false_with_failed_entry() {
        let bad = ReportEntry {
            status: EntryStatus::MalformedRoundTrip,
            ..validated_entry(0, 1, "x")
        };
        let r = CompressionReport::new(0, vec![validated_entry(0, 0, "ok"), bad]);
        assert!(!r.all_validated());
    }

    #[test]
    fn report_hash_differs_with_tolerance_exceeded_payload() {
        let validated = validated_entry(0, 1, "x");
        let exceeded = ReportEntry {
            status: EntryStatus::ToleranceExceeded {
                declared: 0.1,
                observed: 0.2,
            },
            ..validated_entry(0, 1, "x")
        };
        let r_v = CompressionReport::new(0, vec![validated]);
        let r_e = CompressionReport::new(0, vec![exceeded]);
        assert_ne!(r_v.report_hash(), r_e.report_hash());
    }

    #[test]
    fn contains_kind_works() {
        let r = CompressionReport::new(0, vec![validated_entry(0, 1, "x")]);
        assert!(r.contains_kind(CompressionKind::LosslessTrace));
        assert!(!r.contains_kind(CompressionKind::MotifDictionary));
    }

    #[test]
    fn to_json_renders_required_fields() {
        let r = CompressionReport::new(0xCAFE, vec![validated_entry(0, 1, "x")]);
        let j = r.to_json();
        assert!(j.contains("\"schema_version\": 1"));
        assert!(j.contains("\"plan_hash\""));
        assert!(j.contains("\"report_hash\""));
        assert!(j.contains("\"entries\""));
        assert!(j.contains("\"status\": \"validated\""));
        assert!(j.contains("\"kind\": \"lossless_trace\""));
    }

    #[test]
    fn to_json_handles_special_chars_in_label() {
        let mut e = validated_entry(0, 1, "x\"y\\z");
        e.candidate_label = "x\"y\\z".to_string();
        let r = CompressionReport::new(0, vec![e]);
        let j = r.to_json();
        // Properly escaped — \" and \\.
        assert!(j.contains(r#"\"y\\z"#));
    }

    #[test]
    fn f64_renders_as_scientific_notation() {
        assert_eq!(f64_as_json_number(1.0), "1.0000000000e0");
        assert_eq!(f64_as_json_number(0.0), "0.0000000000e0");
        assert_eq!(f64_as_json_number(f64::NAN), "\"NaN\"");
        assert_eq!(f64_as_json_number(f64::INFINITY), "\"Inf\"");
        assert_eq!(f64_as_json_number(f64::NEG_INFINITY), "\"-Inf\"");
    }

    #[test]
    fn report_hash_to_hex_pads_to_16_chars() {
        assert_eq!(ReportHash(0).to_hex(), "0000000000000000");
        assert_eq!(ReportHash(u64::MAX).to_hex(), "ffffffffffffffff");
    }
}
