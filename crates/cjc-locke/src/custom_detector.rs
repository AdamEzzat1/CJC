//! Custom detector extension layer (v0.8 / ADR-0041).
//!
//! Lets users add their own findings to a Locke report *without* breaking
//! the determinism contract or the opinionated built-in baseline.
//!
//! # Why this exists
//!
//! Locke's built-in detectors (E9001-E9112) are opinionated. They catch
//! the strongest signals — high missingness, distributional drift,
//! |AUC|≥0.85 leakage, ID-like cardinality. Real-world data has
//! domain-specific issues that the built-ins cannot detect because they
//! are not visible from data alone. Examples:
//!
//! - LendingClub: any column whose name starts with `total_*` or
//!   `last_pymnt_*` is post-origination by domain definition, but Locke's
//!   |AUC| heuristic flags only the ones that happen to land above 0.85.
//! - Healthcare: any column with a `_at_admission` suffix is safe; with
//!   a `_at_discharge` suffix is leakage. Locke cannot infer the naming
//!   convention.
//! - Trading: certain symbols are wash trades by listing rules.
//!
//! These checks belong *in Locke* — they should produce findings, count
//! toward the belief score, and emit deterministically to JSON. Without
//! this layer, the analyst either runs them in code *next to* Locke (and
//! loses composition) or forks Locke (and loses upgrade path).
//!
//! # Determinism guarantees
//!
//! - **Detector order is canonical**: sorted by code, ties broken by
//!   registration index.
//! - **Emission order inside `run()` does not matter**: the framework
//!   sorts emitted findings by `sort_key()` after the detector returns,
//!   merging into the built-in findings under the same sort.
//! - **The sink is restricted**: only `emit()` may construct findings,
//!   which routes through the same canonical `ValidationFinding::new()`
//!   constructor the built-ins use. No raw mutation, no out-of-band
//!   fingerprint construction.
//! - **Namespace enforcement**: codes must lie in `E9500..=E9999`. Codes
//!   outside this range are silently dropped (and reported in
//!   `LockeReport.assumptions`).
//!
//! What we explicitly do *not* prevent:
//!
//! - A detector that uses `HashMap` internally and emits findings in
//!   randomized order — the framework sort fixes that.
//! - A detector that uses `Instant::now()` to gate emission — the user
//!   can do this, but the result is not deterministic. Locke calls this
//!   out in `LockeReport.assumptions`; it does not prevent it. The
//!   contract is structural, not behavioural.
//!
//! # Belief score composition
//!
//! Detectors declare which belief axes their findings affect via
//! `belief_axes()`. The framework records the mapping in
//! `LockeReport.custom_axis_assignments`, and the belief composition
//! routes findings to the right axis penalties. Returning
//! `BeliefAxisSet::NONE` produces findings that *do not* affect the
//! score — useful for advisory checks that should appear in the report
//! but not drag belief down.
//!
//! # Python wiring
//!
//! `cjc-locke-py` exposes a Python `CustomDetector` ABC (in
//! `cjc_locke/__init__.py`) and a wrapper that adapts Python subclasses
//! into Rust `Arc<dyn CustomDetector>`. The same determinism contract
//! applies — the framework sorts the Python-emitted findings before
//! merge, so Python's dict / set ordering quirks do not leak into the
//! report bytes.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use cjc_data::DataFrame;

use crate::report::{FindingEvidence, FindingSeverity, ValidationFinding};

/// Minimum numeric value of a custom detector's E-code. Custom codes
/// must be in `E9500..=E9999`. Codes `< E9500` are reserved for
/// built-in detectors.
pub const CUSTOM_E_CODE_MIN: u32 = 9500;

/// Maximum numeric value of a custom detector's E-code (inclusive).
/// Codes above this range are reserved for future framework use.
pub const CUSTOM_E_CODE_MAX: u32 = 9999;

/// Which belief-score axes a custom detector contributes to.
///
/// Bitset over the 8 belief axes. Combine with `union`; check
/// membership with `contains`. Use the named constants for clarity.
///
/// `NONE` is a valid declaration — it means the detector contributes
/// findings to the report but does not affect any belief score.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BeliefAxisSet(pub u16);

impl BeliefAxisSet {
    pub const NONE: Self = Self(0);
    pub const SCHEMA: Self = Self(1 << 0);
    pub const MISSINGNESS: Self = Self(1 << 1);
    pub const DRIFT: Self = Self(1 << 2);
    pub const LEAKAGE: Self = Self(1 << 3);
    pub const LINEAGE: Self = Self(1 << 4);
    pub const SAMPLE: Self = Self(1 << 5);
    pub const DUPLICATION: Self = Self(1 << 6);
    pub const CONSTRAINT: Self = Self(1 << 7);
    pub const ALL: Self = Self(0xFF);

    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub fn contains(self, other: Self) -> bool {
        other.0 != 0 && (self.0 & other.0) == other.0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Iterate over individual axis flags that are set, low-bit first.
    /// Order is deterministic by bit position.
    pub fn iter(self) -> impl Iterator<Item = BeliefAxis> {
        (0u8..8u8).filter_map(move |i| {
            let bit = 1u16 << i;
            if self.0 & bit != 0 {
                BeliefAxis::from_bit(bit)
            } else {
                None
            }
        })
    }

    /// Parse from a list of axis name strings. Unknown names produce
    /// `Err`. Used by the Python bridge.
    pub fn from_names(names: &[&str]) -> Result<Self, String> {
        let mut out = Self::NONE;
        for name in names {
            let bit = match *name {
                "schema" => Self::SCHEMA,
                "missingness" => Self::MISSINGNESS,
                "drift" => Self::DRIFT,
                "leakage" => Self::LEAKAGE,
                "lineage" => Self::LINEAGE,
                "sample" => Self::SAMPLE,
                "duplication" => Self::DUPLICATION,
                "constraint" => Self::CONSTRAINT,
                other => return Err(format!("unknown belief axis `{}`", other)),
            };
            out = out.union(bit);
        }
        Ok(out)
    }
}

impl fmt::Display for BeliefAxisSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&'static str> = self.iter().map(|a| a.as_str()).collect();
        write!(f, "{{{}}}", names.join(","))
    }
}

/// One of the 8 belief axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BeliefAxis {
    Schema,
    Missingness,
    Drift,
    Leakage,
    Lineage,
    Sample,
    Duplication,
    Constraint,
}

impl BeliefAxis {
    fn from_bit(bit: u16) -> Option<Self> {
        match bit {
            x if x == BeliefAxisSet::SCHEMA.0 => Some(BeliefAxis::Schema),
            x if x == BeliefAxisSet::MISSINGNESS.0 => Some(BeliefAxis::Missingness),
            x if x == BeliefAxisSet::DRIFT.0 => Some(BeliefAxis::Drift),
            x if x == BeliefAxisSet::LEAKAGE.0 => Some(BeliefAxis::Leakage),
            x if x == BeliefAxisSet::LINEAGE.0 => Some(BeliefAxis::Lineage),
            x if x == BeliefAxisSet::SAMPLE.0 => Some(BeliefAxis::Sample),
            x if x == BeliefAxisSet::DUPLICATION.0 => Some(BeliefAxis::Duplication),
            x if x == BeliefAxisSet::CONSTRAINT.0 => Some(BeliefAxis::Constraint),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            BeliefAxis::Schema => "schema",
            BeliefAxis::Missingness => "missingness",
            BeliefAxis::Drift => "drift",
            BeliefAxis::Leakage => "leakage",
            BeliefAxis::Lineage => "lineage",
            BeliefAxis::Sample => "sample",
            BeliefAxis::Duplication => "duplication",
            BeliefAxis::Constraint => "constraint",
        }
    }
}

/// Reasons a custom-detector registration or finding emission can fail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CustomDetectorError {
    /// Code is outside the allowed `E9500..=E9999` range.
    CodeOutOfNamespace {
        code: String,
        reason: &'static str,
    },
    /// Empty message would produce an unhelpful finding.
    EmptyMessage,
    /// Detector reports no belief axes AND emits non-`Info` findings.
    /// Visible-but-non-scoring findings should be `Info`-severity.
    NonAdvisoryAxesEmpty,
}

impl fmt::Display for CustomDetectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CustomDetectorError::CodeOutOfNamespace { code, reason } => {
                write!(f, "custom detector code `{}` rejected: {}", code, reason)
            }
            CustomDetectorError::EmptyMessage => {
                write!(f, "custom detector emitted a finding with empty message")
            }
            CustomDetectorError::NonAdvisoryAxesEmpty => write!(
                f,
                "detector declares no belief axes but emitted a non-Info finding"
            ),
        }
    }
}

impl std::error::Error for CustomDetectorError {}

/// Validate a custom detector's code at registration time.
///
/// Returns the parsed numeric portion if valid. Used by both the
/// Rust-side registration and the Python bridge so error messages are
/// uniform.
pub fn validate_custom_code(code: &str) -> Result<u32, CustomDetectorError> {
    if !code.starts_with('E') {
        return Err(CustomDetectorError::CodeOutOfNamespace {
            code: code.to_string(),
            reason: "must start with 'E'",
        });
    }
    let n: u32 = code[1..].parse().map_err(|_| CustomDetectorError::CodeOutOfNamespace {
        code: code.to_string(),
        reason: "suffix must be a non-negative integer",
    })?;
    if !(CUSTOM_E_CODE_MIN..=CUSTOM_E_CODE_MAX).contains(&n) {
        return Err(CustomDetectorError::CodeOutOfNamespace {
            code: code.to_string(),
            reason: "must be in E9500..=E9999 (built-in range is reserved)",
        });
    }
    Ok(n)
}

/// Sink passed to a custom detector's `run()`. The only way for the
/// detector to produce findings is via `emit()`.
///
/// All canonicalization (fingerprint composition, byte ordering)
/// happens inside `emit()`; the detector cannot construct findings
/// directly. This is what makes determinism a *structural* guarantee
/// rather than a trust contract.
pub struct FindingSink {
    findings: Vec<ValidationFinding>,
    code: &'static str,
    axes: BeliefAxisSet,
    errors: Vec<CustomDetectorError>,
}

impl FindingSink {
    pub fn new(code: &'static str, axes: BeliefAxisSet) -> Self {
        Self {
            findings: Vec::new(),
            code,
            axes,
            errors: Vec::new(),
        }
    }

    /// Emit a finding. Returns the running count.
    ///
    /// Emission is rejected (with the error recorded) if:
    /// - `message` is empty
    /// - `severity` is not `Info` and the detector declared no axes
    ///   (a finding that affects nothing should be advisory only)
    pub fn emit(
        &mut self,
        severity: FindingSeverity,
        message: impl Into<String>,
        column: Option<String>,
        row_range: Option<(usize, usize)>,
        evidence: Vec<FindingEvidence>,
        sample_size: u64,
    ) -> usize {
        let msg = message.into();
        if msg.is_empty() {
            self.errors.push(CustomDetectorError::EmptyMessage);
            return self.findings.len();
        }
        if self.axes.is_empty() && severity != FindingSeverity::Info {
            self.errors
                .push(CustomDetectorError::NonAdvisoryAxesEmpty);
            return self.findings.len();
        }
        let f = ValidationFinding::new(
            self.code,
            severity,
            msg,
            column,
            row_range,
            evidence,
            sample_size,
            vec!["custom-detector finding (see ADR-0041)".into()],
            Vec::new(),
        );
        self.findings.push(f);
        self.findings.len()
    }

    /// Number of findings emitted so far.
    pub fn len(&self) -> usize {
        self.findings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.findings.is_empty()
    }

    /// The detector's declared belief axes (read-only).
    pub fn axes(&self) -> BeliefAxisSet {
        self.axes
    }

    /// The code being emitted.
    pub fn code(&self) -> &'static str {
        self.code
    }

    /// Inspect emission errors (useful in tests).
    pub fn errors(&self) -> &[CustomDetectorError] {
        &self.errors
    }

    /// Consume the sink and return its findings + any errors.
    pub fn drain(self) -> (Vec<ValidationFinding>, Vec<CustomDetectorError>) {
        (self.findings, self.errors)
    }
}

/// Custom detector trait — the user implementation surface.
///
/// `code()` and `belief_axes()` are static configuration; `run()` is the
/// actual detection. The framework calls `run()` exactly once per
/// validate call, with a read-only `&DataFrame` and a fresh sink.
///
/// **Trait-object friendly**: `Send + Sync` so detectors can be stored
/// in `Arc<dyn CustomDetector>` and shared across threads. The
/// framework does not currently parallelise detectors, but the bound
/// keeps that option open.
///
/// **Debug requirement**: implementors must implement `Debug` so the
/// `ValidateOptions` debug print can include the detector list. A
/// short `{ code }`-style impl is fine.
pub trait CustomDetector: Send + Sync + fmt::Debug {
    /// E-code in `E9500..=E9999`. Validated at registration time;
    /// codes outside this range are silently skipped at run.
    fn code(&self) -> &'static str;

    /// Belief axes affected by findings from this detector. Findings
    /// contribute to these axes via the existing belief penalty model.
    fn belief_axes(&self) -> BeliefAxisSet;

    /// Human-readable label used in error messages.
    fn name(&self) -> &str {
        self.code()
    }

    /// Run the detector. Emit findings via `sink.emit(...)`.
    fn run(&self, df: &DataFrame, sink: &mut FindingSink);
}

/// Result of running every registered custom detector once.
///
/// Findings are sorted by canonical `sort_key()` so the merged report
/// remains byte-identical across runs. Axis assignments record which
/// custom code routes to which belief axes; the belief composition
/// reads this map to apply penalties correctly.
#[derive(Debug, Default)]
pub struct CustomDetectorOutcome {
    pub findings: Vec<ValidationFinding>,
    pub axis_assignments: BTreeMap<String, BeliefAxisSet>,
    pub assumptions: Vec<String>,
    pub registration_errors: Vec<CustomDetectorError>,
    pub emission_errors: Vec<CustomDetectorError>,
}

/// Execute every registered detector once over the dataframe.
///
/// Detectors are sorted by code before invocation so registration order
/// does not affect the output. Each detector's `code()` is validated;
/// failures are reported (not raised) and the detector is skipped.
/// Custom-detector findings are sorted by `sort_key()` before being
/// returned; the caller merges them with built-in findings and the
/// final sort produces a single canonical ordering.
pub fn run_custom_detectors(
    df: &DataFrame,
    detectors: &[Arc<dyn CustomDetector>],
) -> CustomDetectorOutcome {
    let mut outcome = CustomDetectorOutcome::default();

    // Sort detectors deterministically by code so registration order
    // doesn't leak into the report. Ties (impossible if codes are
    // unique) fall through to the natural slice order which is stable.
    let mut sorted: Vec<&Arc<dyn CustomDetector>> = detectors.iter().collect();
    sorted.sort_by(|a, b| a.code().cmp(b.code()));

    let mut seen_codes = std::collections::BTreeSet::new();
    for det in sorted {
        let code = det.code();
        match validate_custom_code(code) {
            Ok(_) => {}
            Err(e) => {
                outcome.registration_errors.push(e);
                continue;
            }
        }
        if !seen_codes.insert(code.to_string()) {
            // Duplicate code — only run the first detector with this
            // code so the report stays deterministic.
            outcome.registration_errors.push(CustomDetectorError::CodeOutOfNamespace {
                code: code.to_string(),
                reason: "duplicate code registration",
            });
            continue;
        }
        let axes = det.belief_axes();
        outcome.axis_assignments.insert(code.to_string(), axes);
        let mut sink = FindingSink::new(code, axes);
        det.run(df, &mut sink);
        let (findings, errors) = sink.drain();
        outcome.emission_errors.extend(errors);
        outcome.findings.extend(findings);
    }

    outcome.findings.sort_by(|a, b| a.sort_key().cmp(&b.sort_key()));

    if !outcome.findings.is_empty() {
        outcome
            .assumptions
            .push(format!(
                "custom detectors contributed {} finding(s); see ADR-0041 for the determinism contract",
                outcome.findings.len()
            ));
    }
    if !outcome.registration_errors.is_empty() {
        outcome.assumptions.push(format!(
            "{} custom detector(s) rejected at registration",
            outcome.registration_errors.len()
        ));
    }

    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_data::Column;

    // ── BeliefAxisSet basics ──────────────────────────────────────────

    #[test]
    fn axis_set_union_and_contains() {
        let a = BeliefAxisSet::LEAKAGE.union(BeliefAxisSet::DRIFT);
        assert!(a.contains(BeliefAxisSet::LEAKAGE));
        assert!(a.contains(BeliefAxisSet::DRIFT));
        assert!(!a.contains(BeliefAxisSet::SCHEMA));
        // Empty set contains-empty is FALSE — we treat NONE as "no
        // observation", not "trivially contained".
        assert!(!BeliefAxisSet::NONE.contains(BeliefAxisSet::NONE));
        assert!(!BeliefAxisSet::LEAKAGE.contains(BeliefAxisSet::NONE));
    }

    #[test]
    fn axis_set_iter_is_low_bit_first() {
        let s = BeliefAxisSet::CONSTRAINT
            .union(BeliefAxisSet::SCHEMA)
            .union(BeliefAxisSet::LEAKAGE);
        let axes: Vec<_> = s.iter().collect();
        assert_eq!(axes, vec![BeliefAxis::Schema, BeliefAxis::Leakage, BeliefAxis::Constraint]);
    }

    #[test]
    fn axis_set_from_names_round_trip() {
        let s = BeliefAxisSet::from_names(&["leakage", "schema"]).unwrap();
        assert!(s.contains(BeliefAxisSet::LEAKAGE));
        assert!(s.contains(BeliefAxisSet::SCHEMA));
        assert!(!s.contains(BeliefAxisSet::DRIFT));
        let err = BeliefAxisSet::from_names(&["nonsense"]).unwrap_err();
        assert!(err.contains("nonsense"));
    }

    // ── Code namespace ───────────────────────────────────────────────

    #[test]
    fn code_accepts_e9500_through_e9999() {
        assert!(validate_custom_code("E9500").is_ok());
        assert!(validate_custom_code("E9501").is_ok());
        assert!(validate_custom_code("E9999").is_ok());
    }

    #[test]
    fn code_rejects_built_in_range() {
        assert!(validate_custom_code("E9001").is_err());
        assert!(validate_custom_code("E9499").is_err());
        assert!(validate_custom_code("E0000").is_err());
    }

    #[test]
    fn code_rejects_above_e9999() {
        assert!(validate_custom_code("E10000").is_err());
        assert!(validate_custom_code("E99999").is_err());
    }

    #[test]
    fn code_rejects_malformed() {
        assert!(validate_custom_code("X9500").is_err());
        assert!(validate_custom_code("9500").is_err());
        assert!(validate_custom_code("Eabc").is_err());
        assert!(validate_custom_code("").is_err());
    }

    // ── FindingSink ──────────────────────────────────────────────────

    #[test]
    fn sink_emits_findings_with_canonical_constructor() {
        let mut sink = FindingSink::new("E9500", BeliefAxisSet::LEAKAGE);
        let n = sink.emit(
            FindingSeverity::Warning,
            "test finding",
            Some("col_a".to_string()),
            None,
            vec![],
            10,
        );
        assert_eq!(n, 1);
        let (findings, errors) = sink.drain();
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].code, "E9500");
        assert_eq!(findings[0].severity, FindingSeverity::Warning);
        assert_eq!(findings[0].column.as_deref(), Some("col_a"));
        assert_eq!(findings[0].sample_size, 10);
        assert!(errors.is_empty());
    }

    #[test]
    fn sink_rejects_empty_message() {
        let mut sink = FindingSink::new("E9500", BeliefAxisSet::LEAKAGE);
        sink.emit(
            FindingSeverity::Warning,
            "",
            None,
            None,
            vec![],
            0,
        );
        let (findings, errors) = sink.drain();
        assert!(findings.is_empty());
        assert_eq!(errors, vec![CustomDetectorError::EmptyMessage]);
    }

    #[test]
    fn sink_rejects_non_info_when_axes_empty() {
        let mut sink = FindingSink::new("E9500", BeliefAxisSet::NONE);
        sink.emit(
            FindingSeverity::Warning,
            "tries to score",
            None,
            None,
            vec![],
            0,
        );
        // Warning is rejected
        let (findings, errors) = sink.drain();
        assert!(findings.is_empty());
        assert_eq!(errors, vec![CustomDetectorError::NonAdvisoryAxesEmpty]);
    }

    #[test]
    fn sink_accepts_info_when_axes_empty() {
        let mut sink = FindingSink::new("E9500", BeliefAxisSet::NONE);
        sink.emit(
            FindingSeverity::Info,
            "purely advisory",
            None,
            None,
            vec![],
            0,
        );
        let (findings, errors) = sink.drain();
        assert_eq!(findings.len(), 1);
        assert!(errors.is_empty());
    }

    // ── End-to-end via run_custom_detectors ──────────────────────────

    #[derive(Debug)]
    struct ConstFlag {
        code: &'static str,
        axes: BeliefAxisSet,
        column: &'static str,
    }
    impl CustomDetector for ConstFlag {
        fn code(&self) -> &'static str {
            self.code
        }
        fn belief_axes(&self) -> BeliefAxisSet {
            self.axes
        }
        fn run(&self, _df: &DataFrame, sink: &mut FindingSink) {
            sink.emit(
                FindingSeverity::Warning,
                format!("flag on `{}`", self.column),
                Some(self.column.to_string()),
                None,
                vec![],
                1,
            );
        }
    }

    fn tiny_df() -> DataFrame {
        DataFrame::from_columns(vec![
            ("a".into(), Column::Float(vec![1.0, 2.0, 3.0])),
            ("b".into(), Column::Int(vec![10, 20, 30])),
        ])
        .unwrap()
    }

    #[test]
    fn run_two_detectors_produces_sorted_findings() {
        let df = tiny_df();
        let dets: Vec<Arc<dyn CustomDetector>> = vec![
            Arc::new(ConstFlag {
                code: "E9501",
                axes: BeliefAxisSet::LEAKAGE,
                column: "b",
            }),
            Arc::new(ConstFlag {
                code: "E9500",
                axes: BeliefAxisSet::SCHEMA,
                column: "a",
            }),
        ];
        let outcome = run_custom_detectors(&df, &dets);
        assert_eq!(outcome.findings.len(), 2);
        // sort_key is (severity, code, column, id) — same severity, so E9500 before E9501.
        assert_eq!(outcome.findings[0].code, "E9500");
        assert_eq!(outcome.findings[1].code, "E9501");
        assert_eq!(outcome.axis_assignments.len(), 2);
        assert_eq!(
            outcome.axis_assignments.get("E9500").copied(),
            Some(BeliefAxisSet::SCHEMA)
        );
        assert!(outcome.registration_errors.is_empty());
        assert!(outcome.emission_errors.is_empty());
    }

    #[test]
    fn run_skips_invalid_codes() {
        let df = tiny_df();
        let dets: Vec<Arc<dyn CustomDetector>> = vec![
            Arc::new(ConstFlag {
                code: "E9001", // built-in range, must be skipped
                axes: BeliefAxisSet::LEAKAGE,
                column: "a",
            }),
            Arc::new(ConstFlag {
                code: "E9500",
                axes: BeliefAxisSet::LEAKAGE,
                column: "a",
            }),
        ];
        let outcome = run_custom_detectors(&df, &dets);
        assert_eq!(outcome.findings.len(), 1);
        assert_eq!(outcome.findings[0].code, "E9500");
        assert_eq!(outcome.registration_errors.len(), 1);
    }

    #[test]
    fn run_deduplicates_codes() {
        let df = tiny_df();
        let dets: Vec<Arc<dyn CustomDetector>> = vec![
            Arc::new(ConstFlag {
                code: "E9500",
                axes: BeliefAxisSet::LEAKAGE,
                column: "a",
            }),
            Arc::new(ConstFlag {
                code: "E9500",
                axes: BeliefAxisSet::LEAKAGE,
                column: "b",
            }),
        ];
        let outcome = run_custom_detectors(&df, &dets);
        // Only the first registered E9500 detector runs; second is rejected.
        assert_eq!(outcome.findings.len(), 1);
        assert_eq!(outcome.registration_errors.len(), 1);
    }

    #[test]
    fn determinism_run_twice_same_findings() {
        let df = tiny_df();
        let dets: Vec<Arc<dyn CustomDetector>> = vec![
            Arc::new(ConstFlag {
                code: "E9500",
                axes: BeliefAxisSet::LEAKAGE,
                column: "a",
            }),
            Arc::new(ConstFlag {
                code: "E9501",
                axes: BeliefAxisSet::SCHEMA,
                column: "b",
            }),
        ];
        let a = run_custom_detectors(&df, &dets);
        let b = run_custom_detectors(&df, &dets);
        assert_eq!(a.findings, b.findings);
        assert_eq!(a.axis_assignments, b.axis_assignments);
    }
}
