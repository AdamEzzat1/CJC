//! Compression candidates — the typed entry point to the compression layer.
//!
//! Every input to the compression layer is wrapped in a
//! [`CompressionCandidate`] that carries:
//!
//! 1. A stable [`CandidateId`] (for deterministic tie-breaking in the
//!    energy ranker).
//! 2. A [`CompressionKind`] discriminator (which scheme to apply).
//! 3. A [`Criticality`] tag (semantic-critical vs advisory-only).
//! 4. The raw bytes to compress.
//!
//! ## The hard rule (enforced at construction time)
//!
//! Semantic-critical facts MUST use a lossless [`CompressionKind`]. Any
//! attempt to construct a candidate with `Criticality::SemanticCritical`
//! and a lossy kind ([`CompressionKind::LowRankAdvisory`] or
//! [`CompressionKind::TensorTrainAdvisory`]) is rejected at the builder
//! level with [`CompressionError::LossyOnCritical`]. This makes "compress
//! a no-GC fact with low-rank" a *compile-time-shaped error* rather than
//! a soundness bug discovered at audit time.
//!
//! Determinism: every field uses owned/value types with stable byte
//! representations, and [`CompressionCandidate::canonical_bytes`] feeds
//! them to [`cjc_cana::CanaHasher`] in a fixed order.

use cjc_cana::CanaHasher;

// ---------------------------------------------------------------------------
// CandidateId
// ---------------------------------------------------------------------------

/// Stable identifier for a [`CompressionCandidate`].
///
/// The identifier is *the* tie-breaker in [`crate::EnergyRanker::rank`] when
/// two candidates have numerically equal energy. Callers MUST assign
/// monotonically increasing IDs in a deterministic order — typically the
/// `BTreeMap` iteration order over `(function_name, kind)` pairs.
///
/// We use `u64` so the address space is comfortably larger than any
/// realistic per-compilation plan count, and we expose `to_be_bytes` so
/// the bytes flow through the FNV-1a hasher in a fixed order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CandidateId(pub u64);

impl CandidateId {
    /// Construct a candidate ID. Trivially total.
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Byte representation used by [`CompressionCandidate::canonical_bytes`].
    /// Big-endian so larger IDs lexicographically sort after smaller ones —
    /// useful if a debugging hex dump is read by eye.
    pub fn to_bytes(self) -> [u8; 8] {
        self.0.to_be_bytes()
    }
}

// ---------------------------------------------------------------------------
// CompressionKind
// ---------------------------------------------------------------------------

/// Which compression scheme a candidate uses.
///
/// Each variant carries a fixed *lossiness* policy — the constructor checks
/// the policy against the candidate's [`Criticality`]. Adding a new variant
/// requires editing [`CompressionKind::is_lossy`] AND
/// [`CompressionKind::short_label`] (the latter feeds the canonical-bytes
/// stream).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CompressionKind {
    /// **Lossless.** Run-length + delta + dictionary header over a pass
    /// history (or any other linearly-ordered semantic-critical stream).
    /// Round-trips exactly.
    LosslessTrace,

    /// **Lossless.** Deterministic LZ77-style motif/dictionary compression
    /// over repeated feature sequences. Round-trips exactly. Suitable for
    /// MIR motif catalogs that the audit chain references.
    MotifDictionary,

    /// **Lossy (advisory).** Truncated-rank summary using sign-stabilized
    /// SVD against a Frobenius-norm error budget. Only valid on
    /// [`Criticality::AdvisoryOnly`] inputs.
    LowRankAdvisory,

    /// **Lossy (advisory).** Tensor-train / MPS-inspired summary that
    /// reuses [`cjc_quantum::mps`] truncation primitives. Only valid on
    /// [`Criticality::AdvisoryOnly`] inputs.
    TensorTrainAdvisory,
}

impl CompressionKind {
    /// `true` iff this kind is lossy (forbidden on semantic-critical inputs).
    ///
    /// Adding a new kind: update this AND any callsite that depends on the
    /// lossless/lossy partition. We deliberately list every variant so
    /// adding a kind without re-classifying it is a compile error.
    pub const fn is_lossy(self) -> bool {
        match self {
            CompressionKind::LosslessTrace => false,
            CompressionKind::MotifDictionary => false,
            CompressionKind::LowRankAdvisory => true,
            CompressionKind::TensorTrainAdvisory => true,
        }
    }

    /// Short stable label used by [`CompressionCandidate::canonical_bytes`].
    ///
    /// The strings here become part of the report's content-addressed hash;
    /// renaming a variant changes the hash, which is the right behaviour
    /// (a "lossless_trace" v1 report must not collide with a v2 report
    /// that uses a different format).
    pub const fn short_label(self) -> &'static str {
        match self {
            CompressionKind::LosslessTrace => "lossless_trace",
            CompressionKind::MotifDictionary => "motif_dict",
            CompressionKind::LowRankAdvisory => "low_rank",
            CompressionKind::TensorTrainAdvisory => "tensor_train",
        }
    }

    /// Stable single-byte discriminator for `CanaHasher::write_tag`. Order
    /// must match the variant declaration order so adding a kind never
    /// renumbers existing ones.
    pub const fn tag_byte(self) -> u8 {
        match self {
            CompressionKind::LosslessTrace => 0,
            CompressionKind::MotifDictionary => 1,
            CompressionKind::LowRankAdvisory => 2,
            CompressionKind::TensorTrainAdvisory => 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Criticality
// ---------------------------------------------------------------------------

/// Semantic-critical vs advisory tag on a compression candidate.
///
/// The tag is what makes lossless guarantees enforceable: rather than
/// trusting every caller to "remember" that pass histories must round-trip
/// exactly, the constructor refuses to build a `SemanticCritical` candidate
/// with a lossy [`CompressionKind`].
///
/// The reconstruction-tolerance carried inside
/// [`Criticality::AdvisoryOnly`] is also enforced at compression time —
/// see the per-kind compressor docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Criticality {
    /// Semantic-critical fact (pass history, alias/effect fact, no-GC
    /// fact, audit chain, exact shape fact, user-visible value). Must be
    /// lossless. Reconstruction must equal input exactly.
    SemanticCritical,

    /// Advisory-only fact (cost-model histogram, scratch feature vector,
    /// runtime pressure trajectory, kernel-decision history). May be
    /// compressed lossily as long as the observed reconstruction error
    /// stays below `tolerance_f`. The tolerance carries the relative
    /// Frobenius/L2 budget used by lossy compressors.
    AdvisoryOnly {
        /// Maximum tolerated reconstruction error (relative Frobenius
        /// norm `||M - M̂||_F / ||M||_F`). Must be `>= 0` and finite.
        tolerance_f: f64,
    },
}

/// A simplified tag for serialization / report rendering. Strips the
/// numeric tolerance so the report's "kind of input this was" answer is
/// independent of the budget that was actually requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CriticalityTag {
    /// Was [`Criticality::SemanticCritical`].
    SemanticCritical,
    /// Was [`Criticality::AdvisoryOnly`].
    AdvisoryOnly,
}

impl Criticality {
    /// Strip the tolerance to get the report-shaped tag.
    pub const fn tag(self) -> CriticalityTag {
        match self {
            Criticality::SemanticCritical => CriticalityTag::SemanticCritical,
            Criticality::AdvisoryOnly { .. } => CriticalityTag::AdvisoryOnly,
        }
    }

    /// Tolerance value, or 0.0 for semantic-critical (which has no budget).
    pub const fn tolerance(self) -> f64 {
        match self {
            Criticality::SemanticCritical => 0.0,
            Criticality::AdvisoryOnly { tolerance_f } => tolerance_f,
        }
    }

    fn validate(self) -> Result<(), CompressionError> {
        match self {
            Criticality::SemanticCritical => Ok(()),
            Criticality::AdvisoryOnly { tolerance_f } => {
                if !tolerance_f.is_finite() || tolerance_f < 0.0 {
                    return Err(CompressionError::InvalidTolerance { value: tolerance_f });
                }
                Ok(())
            }
        }
    }
}

impl CriticalityTag {
    /// Stable single-byte discriminator (for `CanaHasher::write_tag`).
    pub const fn tag_byte(self) -> u8 {
        match self {
            CriticalityTag::SemanticCritical => 0,
            CriticalityTag::AdvisoryOnly => 1,
        }
    }

    /// Short stable label for report rendering and canonical bytes.
    pub const fn short_label(self) -> &'static str {
        match self {
            CriticalityTag::SemanticCritical => "semantic_critical",
            CriticalityTag::AdvisoryOnly => "advisory_only",
        }
    }
}

// ---------------------------------------------------------------------------
// CompressionCandidate
// ---------------------------------------------------------------------------

/// A typed input to the compression layer.
///
/// Construct via [`CompressionCandidate::new`]. The builder enforces the
/// semantic-critical-must-be-lossless invariant; the resulting struct is
/// immutable.
///
/// Field visibility is module-only because all mutation paths must go
/// through [`CompressionCandidate::new`] (which validates) — exposing the
/// fields directly would let callers bypass the invariant.
#[derive(Debug, Clone)]
pub struct CompressionCandidate {
    id: CandidateId,
    kind: CompressionKind,
    criticality: Criticality,
    payload: Vec<u8>,
    /// Human-readable label, used in reports. Examples: `"pass_history.main"`,
    /// `"feature_vec.matmul_score"`.
    label: String,
}

impl CompressionCandidate {
    /// Construct + validate a candidate.
    ///
    /// Rejects:
    /// - Semantic-critical input with a lossy [`CompressionKind`]
    ///   ([`CompressionError::LossyOnCritical`]).
    /// - Advisory input with a non-finite or negative tolerance
    ///   ([`CompressionError::InvalidTolerance`]).
    /// - Empty payload ([`CompressionError::EmptyPayload`]) — an empty
    ///   input would compress to itself trivially and confuse hash equality
    ///   tests.
    pub fn new(
        id: CandidateId,
        kind: CompressionKind,
        criticality: Criticality,
        payload: Vec<u8>,
        label: impl Into<String>,
    ) -> Result<Self, CompressionError> {
        criticality.validate()?;
        if payload.is_empty() {
            return Err(CompressionError::EmptyPayload);
        }
        if kind.is_lossy() && matches!(criticality, Criticality::SemanticCritical) {
            return Err(CompressionError::LossyOnCritical { kind });
        }
        Ok(Self {
            id,
            kind,
            criticality,
            payload,
            label: label.into(),
        })
    }

    /// Stable identifier (tie-breaker for ranking).
    pub fn id(&self) -> CandidateId {
        self.id
    }

    /// Selected compression scheme.
    pub fn kind(&self) -> CompressionKind {
        self.kind
    }

    /// Criticality tag + tolerance.
    pub fn criticality(&self) -> Criticality {
        self.criticality
    }

    /// Raw payload bytes (the input to the compressor).
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    /// Human-readable label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Length of the payload in bytes. Cheaper than calling `payload().len()`
    /// in hot loops because it doesn't materialize the slice.
    pub fn payload_len(&self) -> usize {
        self.payload.len()
    }

    /// Canonical 64-bit FNV-1a hash of the candidate's identity + payload.
    /// Two candidates with identical `(id, kind, criticality, tolerance,
    /// label, payload)` produce the same hash regardless of construction
    /// order. The hash deliberately *excludes* the payload contents'
    /// position in any larger plan — only the candidate itself.
    pub fn canonical_hash(&self) -> u64 {
        let mut h = CanaHasher::new();
        // Fixed field order — must not change without bumping
        // `COMPRESS_VERSION`.
        h.write_u64(self.id.0);
        h.write_tag(self.kind.tag_byte());
        h.write_tag(self.criticality.tag().tag_byte());
        h.write_u64(self.criticality.tolerance().to_bits());
        h.write_str(&self.label);
        h.write_usize(self.payload.len());
        h.write(&self.payload);
        h.finish()
    }
}

// ---------------------------------------------------------------------------
// CompressionError
// ---------------------------------------------------------------------------

/// Errors surfaced by the compression layer.
///
/// Every variant carries enough context to identify the offending candidate
/// without re-running compression. The `Display` impl is stable enough for
/// log scraping.
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionError {
    /// Caller tried to wrap a semantic-critical input in a lossy
    /// [`CompressionKind`]. The hard rule.
    LossyOnCritical {
        /// The kind they asked for.
        kind: CompressionKind,
    },

    /// Advisory tolerance was non-finite or negative.
    InvalidTolerance {
        /// The bad value.
        value: f64,
    },

    /// Payload was empty. Empty inputs are rejected because their
    /// round-trip semantics are degenerate (`compress(empty) ==
    /// decompress(empty) == empty`), which would let a bug in the
    /// compressor go unnoticed.
    EmptyPayload,

    /// The compressor produced a reconstruction whose observed error
    /// exceeded the declared advisory tolerance. Reported by lossy
    /// compressors at validate time.
    ToleranceExceeded {
        /// Tolerance the candidate declared.
        declared: f64,
        /// Error the compressor measured.
        observed: f64,
    },

    /// The lossless decoder detected a malformed compressed payload
    /// (corrupted dictionary header, truncated literal, etc.). Carries
    /// the byte offset where the error was first detected.
    MalformedPayload {
        /// Offset in bytes where the decoder failed.
        at_byte: usize,
        /// Short human-readable reason.
        reason: &'static str,
    },

    /// The advisory compressor's input had a shape it couldn't compress
    /// (e.g. a low-rank request on a 1×N matrix, or a tensor-train request
    /// on input that doesn't reshape into the requested tensor shape).
    UnsupportedShape {
        /// Short human-readable reason.
        reason: &'static str,
    },
}

impl std::fmt::Display for CompressionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompressionError::LossyOnCritical { kind } => write!(
                f,
                "lossy compression kind `{}` rejected on semantic-critical input",
                kind.short_label()
            ),
            CompressionError::InvalidTolerance { value } => {
                write!(f, "advisory tolerance must be finite and >= 0, got {value}")
            }
            CompressionError::EmptyPayload => write!(f, "compression payload is empty"),
            CompressionError::ToleranceExceeded { declared, observed } => write!(
                f,
                "reconstruction error {observed} exceeded declared tolerance {declared}"
            ),
            CompressionError::MalformedPayload { at_byte, reason } => {
                write!(
                    f,
                    "malformed compressed payload at byte {at_byte}: {reason}"
                )
            }
            CompressionError::UnsupportedShape { reason } => {
                write!(f, "compressor cannot operate on this shape: {reason}")
            }
        }
    }
}

impl std::error::Error for CompressionError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_id_bytes_are_big_endian() {
        // Explicitly big-endian so a debug hex dump sorts naturally.
        assert_eq!(CandidateId(1).to_bytes(), [0, 0, 0, 0, 0, 0, 0, 1]);
        assert_eq!(
            CandidateId(0x0102030405060708).to_bytes(),
            [1, 2, 3, 4, 5, 6, 7, 8]
        );
    }

    #[test]
    fn kind_lossy_partition() {
        assert!(!CompressionKind::LosslessTrace.is_lossy());
        assert!(!CompressionKind::MotifDictionary.is_lossy());
        assert!(CompressionKind::LowRankAdvisory.is_lossy());
        assert!(CompressionKind::TensorTrainAdvisory.is_lossy());
    }

    #[test]
    fn kind_short_labels_are_distinct() {
        let labels = [
            CompressionKind::LosslessTrace.short_label(),
            CompressionKind::MotifDictionary.short_label(),
            CompressionKind::LowRankAdvisory.short_label(),
            CompressionKind::TensorTrainAdvisory.short_label(),
        ];
        let mut seen = std::collections::BTreeSet::new();
        for l in labels.iter() {
            assert!(seen.insert(*l), "duplicate label `{}`", l);
        }
    }

    #[test]
    fn kind_tag_bytes_are_distinct_and_stable() {
        // Stability matters: tag bytes feed CanaHasher, so re-ordering would
        // change every existing report's hash.
        assert_eq!(CompressionKind::LosslessTrace.tag_byte(), 0);
        assert_eq!(CompressionKind::MotifDictionary.tag_byte(), 1);
        assert_eq!(CompressionKind::LowRankAdvisory.tag_byte(), 2);
        assert_eq!(CompressionKind::TensorTrainAdvisory.tag_byte(), 3);
    }

    #[test]
    fn semantic_critical_rejects_lossy_kinds() {
        // The hard rule.
        for kind in [
            CompressionKind::LowRankAdvisory,
            CompressionKind::TensorTrainAdvisory,
        ] {
            let result = CompressionCandidate::new(
                CandidateId(1),
                kind,
                Criticality::SemanticCritical,
                vec![1, 2, 3],
                "critical",
            );
            match result {
                Err(CompressionError::LossyOnCritical { kind: k }) => assert_eq!(k, kind),
                other => panic!("expected LossyOnCritical for {kind:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn semantic_critical_accepts_lossless_kinds() {
        for kind in [
            CompressionKind::LosslessTrace,
            CompressionKind::MotifDictionary,
        ] {
            let result = CompressionCandidate::new(
                CandidateId(1),
                kind,
                Criticality::SemanticCritical,
                vec![1, 2, 3],
                "critical",
            );
            assert!(
                result.is_ok(),
                "lossless on critical must succeed: {kind:?}"
            );
        }
    }

    #[test]
    fn advisory_accepts_all_kinds() {
        for kind in [
            CompressionKind::LosslessTrace,
            CompressionKind::MotifDictionary,
            CompressionKind::LowRankAdvisory,
            CompressionKind::TensorTrainAdvisory,
        ] {
            let result = CompressionCandidate::new(
                CandidateId(1),
                kind,
                Criticality::AdvisoryOnly { tolerance_f: 0.1 },
                vec![1, 2, 3],
                "advisory",
            );
            assert!(result.is_ok(), "advisory on {kind:?} must succeed");
        }
    }

    #[test]
    fn advisory_rejects_non_finite_tolerance() {
        for bad in [f64::NAN, f64::INFINITY, -0.1] {
            let result = CompressionCandidate::new(
                CandidateId(1),
                CompressionKind::LowRankAdvisory,
                Criticality::AdvisoryOnly { tolerance_f: bad },
                vec![1, 2, 3],
                "advisory",
            );
            match result {
                Err(CompressionError::InvalidTolerance { value }) => {
                    // NaN won't compare equal — compare bit-pattern.
                    assert_eq!(value.to_bits(), bad.to_bits());
                }
                other => panic!("expected InvalidTolerance for {bad:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn empty_payload_rejected() {
        let r = CompressionCandidate::new(
            CandidateId(1),
            CompressionKind::LosslessTrace,
            Criticality::SemanticCritical,
            vec![],
            "empty",
        );
        assert!(matches!(r, Err(CompressionError::EmptyPayload)));
    }

    #[test]
    fn canonical_hash_is_deterministic() {
        let c = CompressionCandidate::new(
            CandidateId(42),
            CompressionKind::LosslessTrace,
            Criticality::SemanticCritical,
            vec![1, 2, 3, 4, 5],
            "label_x",
        )
        .unwrap();
        let h = c.canonical_hash();
        for _ in 0..50 {
            assert_eq!(c.canonical_hash(), h);
        }
    }

    #[test]
    fn canonical_hash_distinguishes_id() {
        let mk = |id: u64| {
            CompressionCandidate::new(
                CandidateId(id),
                CompressionKind::LosslessTrace,
                Criticality::SemanticCritical,
                vec![1, 2, 3],
                "x",
            )
            .unwrap()
        };
        assert_ne!(mk(1).canonical_hash(), mk(2).canonical_hash());
    }

    #[test]
    fn canonical_hash_distinguishes_kind() {
        let mk = |kind: CompressionKind| {
            CompressionCandidate::new(
                CandidateId(1),
                kind,
                Criticality::AdvisoryOnly { tolerance_f: 0.1 },
                vec![1, 2, 3],
                "x",
            )
            .unwrap()
        };
        assert_ne!(
            mk(CompressionKind::LosslessTrace).canonical_hash(),
            mk(CompressionKind::MotifDictionary).canonical_hash()
        );
        assert_ne!(
            mk(CompressionKind::LowRankAdvisory).canonical_hash(),
            mk(CompressionKind::TensorTrainAdvisory).canonical_hash()
        );
    }

    #[test]
    fn canonical_hash_distinguishes_payload() {
        let mk = |bytes: Vec<u8>| {
            CompressionCandidate::new(
                CandidateId(1),
                CompressionKind::LosslessTrace,
                Criticality::SemanticCritical,
                bytes,
                "x",
            )
            .unwrap()
        };
        assert_ne!(
            mk(vec![1, 2, 3]).canonical_hash(),
            mk(vec![1, 2, 4]).canonical_hash()
        );
    }

    #[test]
    fn canonical_hash_distinguishes_tolerance() {
        let mk = |tol: f64| {
            CompressionCandidate::new(
                CandidateId(1),
                CompressionKind::LowRankAdvisory,
                Criticality::AdvisoryOnly { tolerance_f: tol },
                vec![1, 2, 3],
                "x",
            )
            .unwrap()
        };
        assert_ne!(mk(0.1).canonical_hash(), mk(0.2).canonical_hash());
    }

    #[test]
    fn criticality_tag_strips_tolerance() {
        let sc = Criticality::SemanticCritical;
        assert_eq!(sc.tag(), CriticalityTag::SemanticCritical);
        assert_eq!(sc.tolerance(), 0.0);

        let adv = Criticality::AdvisoryOnly { tolerance_f: 0.15 };
        assert_eq!(adv.tag(), CriticalityTag::AdvisoryOnly);
        assert_eq!(adv.tolerance(), 0.15);
    }

    #[test]
    fn display_for_errors_is_stable() {
        // Used for log scraping; format string must not change silently.
        let e = CompressionError::LossyOnCritical {
            kind: CompressionKind::LowRankAdvisory,
        };
        assert!(format!("{e}").contains("low_rank"));
        assert!(format!("{e}").contains("semantic-critical"));

        let e = CompressionError::ToleranceExceeded {
            declared: 0.1,
            observed: 0.2,
        };
        assert!(format!("{e}").contains("0.1"));
        assert!(format!("{e}").contains("0.2"));

        let e = CompressionError::MalformedPayload {
            at_byte: 42,
            reason: "bad header",
        };
        assert!(format!("{e}").contains("42"));
        assert!(format!("{e}").contains("bad header"));
    }
}
