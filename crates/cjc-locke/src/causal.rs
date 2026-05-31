//! Causality guardrails. Conservative by construction.
//!
//! Locke v0 does **not** infer causal effects. It surfaces *risks* that
//! a downstream consumer might over-interpret correlation as causation,
//! and it gives the user a place to register their own causal assumptions
//! so the assumption is visible in every report.
//!
//! Key principle: every type in this module is suffixed with "Warning"
//! or "Hint" or "Claim" — never "Result", "Effect", or "Cause". Locke
//! is not a causal-inference tool.

use std::collections::BTreeMap;

use crate::id::{fingerprint_compose, fingerprint_str, FingerprintId, IdDomain};

/// A user-declared causal claim. Locke records this *as a claim*, not as
/// a verified fact. The claim is content-addressed so an audit log can
/// reference it.
#[derive(Clone, Debug, PartialEq)]
pub struct CausalClaim {
    pub id: FingerprintId,
    pub cause: String,
    pub effect: String,
    pub direction: CausalDirection,
    pub rationale: String,
    /// User-supplied confidence (subjective, not derived).
    pub user_confidence: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CausalDirection {
    AtoB,
    BtoA,
    Bidirectional,
    Unknown,
}

impl CausalClaim {
    pub fn new(cause: &str, effect: &str, direction: CausalDirection, rationale: &str, user_confidence: f64) -> Self {
        let dir = match direction {
            CausalDirection::AtoB => "a2b",
            CausalDirection::BtoA => "b2a",
            CausalDirection::Bidirectional => "both",
            CausalDirection::Unknown => "unknown",
        };
        let id = fingerprint_compose(
            IdDomain::CausalClaim,
            "claim",
            &[
                fingerprint_str(IdDomain::CausalClaim, cause),
                fingerprint_str(IdDomain::CausalClaim, effect),
                fingerprint_str(IdDomain::CausalClaim, dir),
                fingerprint_str(IdDomain::CausalClaim, rationale),
            ],
        );
        Self {
            id,
            cause: cause.into(),
            effect: effect.into(),
            direction,
            rationale: rationale.into(),
            user_confidence: user_confidence.clamp(0.0, 1.0),
        }
    }
}

/// A flagged risk that correlation is being read as causation.
#[derive(Clone, Debug, PartialEq)]
pub struct CausalWarning {
    pub id: FingerprintId,
    pub kind: CausalWarningKind,
    /// Programmatic severity (v0.3). DAG-acknowledged warnings have this
    /// dropped one level. Callers filtering on severity get the right
    /// answer without parsing the message text.
    pub severity: crate::report::FindingSeverity,
    pub message: String,
    pub a: String,
    pub b: String,
    pub correlation: Option<f64>,
    pub assumptions: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CausalWarningKind {
    StrongCorrelationNoIntervention,
    LikelyConfounder,
    CausalLanguageInLabel,
    ObservationalOnly,
    ModelExplanationAsCausal,
}

impl CausalWarningKind {
    fn tag(self) -> &'static str {
        match self {
            CausalWarningKind::StrongCorrelationNoIntervention => "strong-corr",
            CausalWarningKind::LikelyConfounder => "confounder",
            CausalWarningKind::CausalLanguageInLabel => "causal-lang",
            CausalWarningKind::ObservationalOnly => "observational",
            CausalWarningKind::ModelExplanationAsCausal => "explain-as-cause",
        }
    }

    /// Default severity per warning kind (v0.3). Reflects the prose
    /// semantics the v0.1 / v0.2 warnings already conveyed.
    pub fn default_severity(self) -> crate::report::FindingSeverity {
        use crate::report::FindingSeverity::*;
        match self {
            CausalWarningKind::StrongCorrelationNoIntervention => Warning,
            CausalWarningKind::LikelyConfounder => Warning,
            CausalWarningKind::CausalLanguageInLabel => Notice,
            CausalWarningKind::ObservationalOnly => Notice,
            CausalWarningKind::ModelExplanationAsCausal => Warning,
        }
    }
}

/// Drop a severity by one ordinal level. `Info` stays `Info`. Used by
/// the causal-DAG acknowledgement path so a downgrade is real.
fn drop_severity_one_level(s: crate::report::FindingSeverity) -> crate::report::FindingSeverity {
    use crate::report::FindingSeverity::*;
    match s {
        Error => Warning,
        Warning => Notice,
        Notice => Info,
        Info => Info,
    }
}

impl CausalWarning {
    fn new(
        kind: CausalWarningKind,
        message: impl Into<String>,
        a: &str,
        b: &str,
        correlation: Option<f64>,
        assumptions: Vec<String>,
    ) -> Self {
        Self::new_with_severity(
            kind,
            kind.default_severity(),
            message,
            a,
            b,
            correlation,
            assumptions,
        )
    }

    fn new_with_severity(
        kind: CausalWarningKind,
        severity: crate::report::FindingSeverity,
        message: impl Into<String>,
        a: &str,
        b: &str,
        correlation: Option<f64>,
        assumptions: Vec<String>,
    ) -> Self {
        let message = message.into();
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(kind.tag().as_bytes());
        bytes.push(0);
        // Severity participates in the content fingerprint so the
        // downgraded warning has a different id than the full-severity
        // one (otherwise two consumers receiving "same kind + a + b"
        // could disagree on which they saw).
        bytes.push(severity as u8);
        bytes.push(0);
        bytes.extend_from_slice(message.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(a.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(b.as_bytes());
        bytes.push(0);
        if let Some(r) = correlation {
            bytes.extend_from_slice(&r.to_bits().to_le_bytes());
        }
        for a in &assumptions {
            bytes.extend_from_slice(a.as_bytes());
            bytes.push(0x1e);
        }
        let id = fingerprint_compose(
            IdDomain::CausalClaim,
            "warning",
            &[crate::id::fingerprint(IdDomain::CausalClaim, &bytes)],
        );
        Self {
            id,
            kind,
            severity,
            message,
            a: a.into(),
            b: b.into(),
            correlation,
            assumptions,
        }
    }
}

/// Pairwise correlation finding (a, b, r). Stored separately from
/// `CausalWarning` because a correlation by itself is *not* a warning —
/// it becomes one when its magnitude crosses the configured threshold.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrelationFinding {
    pub a: String,
    pub b: String,
    pub r: f64,
    pub n_pairs: u64,
}

/// A column that's plausibly a confounder of (`feature`, `target`).
///
/// "Plausibly" = correlated with both above a threshold. Locke does **not**
/// run an intervention; this is purely an association-based hint.
#[derive(Clone, Debug, PartialEq)]
pub struct ConfounderHint {
    pub candidate: String,
    pub feature: String,
    pub target: String,
    pub r_with_feature: f64,
    pub r_with_target: f64,
}

/// User-toggled mode declaring data is purely observational.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CausalMode {
    Default,
    /// User has declared no intervention/randomisation; raises the
    /// causal-warning severity on every correlation finding.
    ObservationalOnly,
}

/// A user-declared partial causal DAG (v0.2).
///
/// Nodes are column names; an edge `a → b` declares "user believes a causes b."
/// Cycles are rejected at construction time. The DAG does **not** make Locke
/// a causal-inference tool — it merely lets the user pre-register their
/// hypothesised causal pathway so that strong-correlation warnings between
/// declared (or reachable) pairs are *downgraded* (not removed).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CausalDag {
    /// Outgoing adjacency: `edges[a]` contains every `b` such that a → b
    /// was declared. `BTreeMap`/`BTreeSet` for deterministic emit.
    pub edges: std::collections::BTreeMap<String, std::collections::BTreeSet<String>>,
}

#[derive(Debug, PartialEq)]
pub enum CausalDagError {
    CycleIntroduced { from: String, to: String },
    SelfLoop(String),
}

impl CausalDag {
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare `from → to`. Returns Err if the edge would close a cycle
    /// (including a self-loop).
    pub fn add_edge(&mut self, from: &str, to: &str) -> Result<(), CausalDagError> {
        if from == to {
            return Err(CausalDagError::SelfLoop(from.to_string()));
        }
        // Cycle check: if `from` is already reachable from `to`, adding
        // `from → to` would close a cycle.
        if self.is_reachable(to, from) {
            return Err(CausalDagError::CycleIntroduced {
                from: from.to_string(),
                to: to.to_string(),
            });
        }
        self.edges
            .entry(from.to_string())
            .or_default()
            .insert(to.to_string());
        Ok(())
    }

    /// True if there's a *directed* path from `from` to `to` (transitive).
    pub fn is_reachable(&self, from: &str, to: &str) -> bool {
        if from == to {
            return true;
        }
        let mut stack: Vec<&str> = vec![from];
        let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            if let Some(succ) = self.edges.get(n) {
                for s in succ {
                    if s == to {
                        return true;
                    }
                    stack.push(s.as_str());
                }
            }
        }
        false
    }

    /// True if either `a → b` or `b → a` is reachable in the declared DAG.
    /// Used to decide whether a correlation warning has already been
    /// "explained" by the user's assumption.
    pub fn relates(&self, a: &str, b: &str) -> bool {
        self.is_reachable(a, b) || self.is_reachable(b, a)
    }

    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
}

/// Configuration for the causality module.
#[derive(Clone, Debug)]
pub struct CausalConfig {
    pub strong_correlation_threshold: f64,
    pub confounder_threshold: f64,
    pub mode: CausalMode,
    /// Substrings that, when found in column or report metadata, trigger
    /// a `CausalLanguageInLabel` warning.
    pub causal_keywords: Vec<String>,
    /// v0.2 user-declared causal DAG. When a strong-correlation warning
    /// is about to fire between two nodes already related in this DAG,
    /// the warning is **downgraded** (severity dropped one level) and
    /// its message annotated.
    pub assumed_dag: CausalDag,
}

impl Default for CausalConfig {
    fn default() -> Self {
        Self {
            strong_correlation_threshold: 0.7,
            confounder_threshold: 0.4,
            mode: CausalMode::Default,
            causal_keywords: vec![
                "causes".into(),
                "causal".into(),
                "leads_to".into(),
                "due_to".into(),
                "because_of".into(),
                "drives".into(),
            ],
            assumed_dag: CausalDag::new(),
        }
    }
}

/// Top-level causal-guardrail report.
#[derive(Clone, Debug, PartialEq)]
pub struct CausalGuardrailReport {
    pub mode: CausalMode,
    pub claims: Vec<CausalClaim>,
    pub correlations: Vec<CorrelationFinding>,
    pub warnings: Vec<CausalWarning>,
    pub confounder_hints: Vec<ConfounderHint>,
    /// Explicit, prominent disclaimer text emitted with every report.
    pub disclaimer: String,
}

impl CausalGuardrailReport {
    pub const DISCLAIMER: &'static str = "Locke does not infer causal effects. Every warning below is an association-based risk indicator, not a causal claim.";

    pub fn new(
        mode: CausalMode,
        claims: Vec<CausalClaim>,
        correlations: Vec<CorrelationFinding>,
        warnings: Vec<CausalWarning>,
        confounder_hints: Vec<ConfounderHint>,
    ) -> Self {
        Self {
            mode,
            claims,
            correlations,
            warnings,
            confounder_hints,
            disclaimer: Self::DISCLAIMER.into(),
        }
    }
}

/// A sensitivity-style check (deferred — placeholder for v0.2).
#[derive(Clone, Debug, PartialEq)]
pub struct CounterfactualSensitivityCheck {
    pub feature: String,
    pub perturbation_label: String,
    pub note: String,
}

/// Scan a flat list of correlation findings, emit warnings + confounder
/// hints under the given config.
///
/// `correlations` should already be deterministic (sorted by (a, b) with
/// `a < b` lexicographically).
///
/// `target_column` is optional; if present, confounder hints look for
/// candidates correlated with both a feature column and the target.
pub fn audit_correlations(
    correlations: &[CorrelationFinding],
    target_column: Option<&str>,
    config: &CausalConfig,
    label_text: Option<&str>,
    interpreting_model_explanation_as_causal: bool,
) -> CausalGuardrailReport {
    let mut warnings: Vec<CausalWarning> = Vec::new();
    let mut confounders: Vec<ConfounderHint> = Vec::new();

    // 1. Strong-correlation warnings.
    for c in correlations {
        if c.r.abs() >= config.strong_correlation_threshold {
            let mut assumptions = vec!["correlation is computed from observational data only".into()];
            if config.mode == CausalMode::ObservationalOnly {
                assumptions
                    .push("user declared dataset is observational-only — no intervention occurred".into());
            }
            // v0.2: if the user-declared causal DAG already relates these
            // two columns, this correlation is not "surprising news" —
            // surface it at one severity lower so the user sees it but
            // doesn't get re-warned about something they've already
            // acknowledged.
            let acknowledged_in_dag = config.assumed_dag.relates(&c.a, &c.b);
            let message = if acknowledged_in_dag {
                assumptions.push(format!(
                    "user-declared causal DAG already relates `{}` and `{}` — warning downgraded",
                    c.a, c.b
                ));
                format!(
                    "{} and {} show |r|={:.3} ≥ {:.2} (acknowledged by causal DAG; association ≠ verified causation)",
                    c.a, c.b, c.r, config.strong_correlation_threshold
                )
            } else {
                format!(
                    "{} and {} show |r|={:.3} ≥ {:.2}; this is association, not causation",
                    c.a, c.b, c.r, config.strong_correlation_threshold
                )
            };
            // v0.3: severity is now a real field on CausalWarning. The
            // DAG acknowledgement drops it one ordinal level so
            // programmatic consumers see the downgrade without parsing
            // message text.
            let base_sev = CausalWarningKind::StrongCorrelationNoIntervention.default_severity();
            let sev = if acknowledged_in_dag {
                drop_severity_one_level(base_sev)
            } else {
                base_sev
            };
            let warning = CausalWarning::new_with_severity(
                CausalWarningKind::StrongCorrelationNoIntervention,
                sev,
                message,
                &c.a,
                &c.b,
                Some(c.r),
                assumptions,
            );
            warnings.push(warning);
        }
    }

    // 2. Observational-only standing warning, once, if mode is set.
    if config.mode == CausalMode::ObservationalOnly && !correlations.is_empty() {
        warnings.push(CausalWarning::new(
            CausalWarningKind::ObservationalOnly,
            "dataset is declared observational-only; treat every correlation as non-causal until an intervention or quasi-experimental design is added",
            "*",
            "*",
            None,
            vec!["caller passed CausalMode::ObservationalOnly".into()],
        ));
    }

    // 3. Confounder hints.
    if let Some(target) = target_column {
        // v0.7+ B4.3 perf-fix: previously this block (a) stored every
        // correlation twice — both `(a,b)` and `(b,a)` — doubling memory
        // and (b) cloned both keys per inner lookup, allocating two
        // Strings per candidate. Refactor: borrow strings from
        // `correlations` directly (BTreeMap<(&str,&str), f64>), store one
        // canonical orientation only, and look up symmetrically.
        // Additionally: cache the features list once outside the outer
        // loop instead of rebuilding the candidate Vec per feature.

        // Build per-column adjacency: for each column X != target with |r| ≥ conf,
        // record r(X, target). Then for each (X, target) hot pair, find a Z
        // distinct from X and target with |r(Z, X)| and |r(Z, target)| ≥ conf.
        let r_with_target: BTreeMap<&str, f64> = correlations
            .iter()
            .filter_map(|c| {
                if c.a == target {
                    Some((c.b.as_str(), c.r))
                } else if c.b == target {
                    Some((c.a.as_str(), c.r))
                } else {
                    None
                }
            })
            .collect();
        let pairwise: BTreeMap<(&str, &str), f64> = correlations
            .iter()
            .map(|c| {
                let (l, r) = if c.a <= c.b {
                    (c.a.as_str(), c.b.as_str())
                } else {
                    (c.b.as_str(), c.a.as_str())
                };
                ((l, r), c.r)
            })
            .collect();
        let pair_lookup = |a: &str, b: &str| -> Option<f64> {
            let (l, r) = if a <= b { (a, b) } else { (b, a) };
            pairwise.get(&(l, r)).copied()
        };

        // BTreeMap.keys() is already sorted — no extra sort needed.
        let features: Vec<&str> = r_with_target.keys().copied().collect();
        for &feature in &features {
            let r_feat_target = match r_with_target.get(feature) {
                Some(r) if r.abs() >= config.confounder_threshold => *r,
                _ => continue,
            };
            for &z in &features {
                if z == feature {
                    continue;
                }
                let r_z_target = r_with_target.get(z).copied().unwrap_or(0.0);
                let r_z_feature = pair_lookup(z, feature).unwrap_or(0.0);
                if r_z_target.abs() >= config.confounder_threshold
                    && r_z_feature.abs() >= config.confounder_threshold
                {
                    confounders.push(ConfounderHint {
                        candidate: z.to_string(),
                        feature: feature.to_string(),
                        target: target.to_string(),
                        r_with_feature: r_z_feature,
                        r_with_target: r_z_target,
                    });
                    warnings.push(CausalWarning::new(
                        CausalWarningKind::LikelyConfounder,
                        format!(
                            "{} is associated with both {} and {} — could be a confounder",
                            z, feature, target
                        ),
                        z,
                        feature,
                        Some(r_z_feature),
                        vec![
                            "association does not imply confounding without a causal model".into(),
                            format!("|r({}, target)|={:.3}", z, r_z_target),
                        ],
                    ));
                }
            }
            // Use r_feat_target to silence the unused-binding warning and make
            // the feature → target signal visible in the disclaimer assumptions
            // chain when this hot pair triggers nothing else.
            let _ = r_feat_target;
        }
    }

    // 4. Causal-language scan.
    if let Some(text) = label_text {
        let lower = text.to_lowercase();
        for kw in &config.causal_keywords {
            if lower.contains(&kw.to_lowercase()) {
                warnings.push(CausalWarning::new(
                    CausalWarningKind::CausalLanguageInLabel,
                    format!(
                        "metadata contains causal language ({:?}); review whether the analysis actually established causation",
                        kw
                    ),
                    "metadata",
                    kw,
                    None,
                    vec!["Locke does not certify causal claims".into()],
                ));
            }
        }
    }

    // 5. Model-explanation-as-causal warning.
    if interpreting_model_explanation_as_causal {
        warnings.push(CausalWarning::new(
            CausalWarningKind::ModelExplanationAsCausal,
            "model attributions / feature importances are correlational, not causal — do not interpret as effect estimates",
            "model",
            "explanation",
            None,
            vec!["caller flagged that explanations are being read causally".into()],
        ));
    }

    // Deterministic ordering: warnings sort by (kind tag, message, a, b).
    warnings.sort_by(|x, y| {
        x.kind
            .tag()
            .cmp(y.kind.tag())
            .then_with(|| x.a.cmp(&y.a))
            .then_with(|| x.b.cmp(&y.b))
            .then_with(|| x.message.cmp(&y.message))
    });
    confounders.sort_by(|x, y| {
        x.candidate
            .cmp(&y.candidate)
            .then_with(|| x.feature.cmp(&y.feature))
            .then_with(|| x.target.cmp(&y.target))
    });

    CausalGuardrailReport::new(config.mode, vec![], correlations.to_vec(), warnings, confounders)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corr(a: &str, b: &str, r: f64) -> CorrelationFinding {
        CorrelationFinding { a: a.into(), b: b.into(), r, n_pairs: 100 }
    }

    #[test]
    fn strong_correlation_emits_warning() {
        let cfg = CausalConfig::default();
        let r = audit_correlations(&[corr("a", "b", 0.9)], None, &cfg, None, false);
        assert!(r.warnings.iter().any(|w| w.kind == CausalWarningKind::StrongCorrelationNoIntervention));
    }

    #[test]
    fn weak_correlation_does_not_warn() {
        let cfg = CausalConfig::default();
        let r = audit_correlations(&[corr("a", "b", 0.2)], None, &cfg, None, false);
        assert!(!r.warnings.iter().any(|w| w.kind == CausalWarningKind::StrongCorrelationNoIntervention));
    }

    #[test]
    fn observational_mode_adds_disclaimer_warning() {
        let mut cfg = CausalConfig::default();
        cfg.mode = CausalMode::ObservationalOnly;
        let r = audit_correlations(&[corr("a", "b", 0.5)], None, &cfg, None, false);
        assert!(r.warnings.iter().any(|w| w.kind == CausalWarningKind::ObservationalOnly));
    }

    #[test]
    fn confounder_hint_fires_with_target() {
        let cfg = CausalConfig::default();
        let cors = vec![
            corr("age", "y", 0.6),
            corr("income", "y", 0.6),
            corr("age", "income", 0.6),
        ];
        let r = audit_correlations(&cors, Some("y"), &cfg, None, false);
        assert!(!r.confounder_hints.is_empty(), "expected at least one confounder hint, got {:?}", r.confounder_hints);
    }

    #[test]
    fn causal_language_in_label_warns() {
        let cfg = CausalConfig::default();
        let r = audit_correlations(&[], None, &cfg, Some("smoking causes lung cancer"), false);
        assert!(r.warnings.iter().any(|w| w.kind == CausalWarningKind::CausalLanguageInLabel));
    }

    #[test]
    fn model_explanation_flag_warns() {
        let cfg = CausalConfig::default();
        let r = audit_correlations(&[], None, &cfg, None, true);
        assert!(r.warnings.iter().any(|w| w.kind == CausalWarningKind::ModelExplanationAsCausal));
    }

    #[test]
    fn disclaimer_is_always_present() {
        let cfg = CausalConfig::default();
        let r = audit_correlations(&[], None, &cfg, None, false);
        assert_eq!(r.disclaimer, CausalGuardrailReport::DISCLAIMER);
    }

    #[test]
    fn claim_id_is_deterministic() {
        let a = CausalClaim::new("x", "y", CausalDirection::AtoB, "r1", 0.5);
        let b = CausalClaim::new("x", "y", CausalDirection::AtoB, "r1", 0.5);
        assert_eq!(a.id, b.id);
    }

    // ─── v0.2: CausalDag tests ─────────────────────────────────────────

    #[test]
    fn dag_add_edge_then_reachable() {
        let mut d = CausalDag::new();
        d.add_edge("a", "b").unwrap();
        d.add_edge("b", "c").unwrap();
        assert!(d.is_reachable("a", "c"));
        assert!(d.relates("a", "c"));
        assert!(d.relates("c", "a")); // relates() is symmetric
    }

    #[test]
    fn dag_self_loop_is_rejected() {
        let mut d = CausalDag::new();
        assert!(matches!(
            d.add_edge("a", "a"),
            Err(CausalDagError::SelfLoop(_))
        ));
    }

    #[test]
    fn dag_cycle_introduction_is_rejected() {
        let mut d = CausalDag::new();
        d.add_edge("a", "b").unwrap();
        d.add_edge("b", "c").unwrap();
        // Closing a → b → c → a would form a cycle.
        assert!(matches!(
            d.add_edge("c", "a"),
            Err(CausalDagError::CycleIntroduced { .. })
        ));
    }

    #[test]
    fn dag_relates_is_symmetric() {
        let mut d = CausalDag::new();
        d.add_edge("x", "y").unwrap();
        assert!(d.relates("x", "y"));
        assert!(d.relates("y", "x"));
        assert!(!d.relates("x", "z"));
    }

    #[test]
    fn dag_downgrades_strong_correlation_warning() {
        let mut cfg = CausalConfig::default();
        cfg.assumed_dag
            .add_edge("treatment", "outcome")
            .unwrap();
        let cors = vec![corr("treatment", "outcome", 0.9)];
        let r = audit_correlations(&cors, None, &cfg, None, false);
        let w = r
            .warnings
            .iter()
            .find(|w| w.kind == CausalWarningKind::StrongCorrelationNoIntervention)
            .expect("strong-corr warning still emitted");
        // The downgrade is signalled in the message (annotated with
        // "acknowledged by causal DAG") and in the assumptions list.
        assert!(
            w.message.contains("acknowledged by causal DAG"),
            "message should mention DAG acknowledgement; got `{}`",
            w.message
        );
        assert!(w
            .assumptions
            .iter()
            .any(|a| a.contains("causal DAG already relates")));
    }

    #[test]
    fn dag_downgrade_actually_lowers_severity() {
        use crate::report::FindingSeverity::*;
        let mut cfg = CausalConfig::default();
        cfg.assumed_dag.add_edge("treatment", "outcome").unwrap();
        let cors = vec![corr("treatment", "outcome", 0.9)];
        let r = audit_correlations(&cors, None, &cfg, None, false);
        let w = r
            .warnings
            .iter()
            .find(|w| w.kind == CausalWarningKind::StrongCorrelationNoIntervention)
            .unwrap();
        // Default severity is Warning; DAG acknowledgement drops to Notice.
        assert_eq!(w.severity, Notice);
    }

    #[test]
    fn dag_unrelated_pair_keeps_warning_severity() {
        use crate::report::FindingSeverity::*;
        let mut cfg = CausalConfig::default();
        cfg.assumed_dag.add_edge("x", "y").unwrap();
        let cors = vec![corr("a", "b", 0.95)];
        let r = audit_correlations(&cors, None, &cfg, None, false);
        let w = r
            .warnings
            .iter()
            .find(|w| w.kind == CausalWarningKind::StrongCorrelationNoIntervention)
            .unwrap();
        assert_eq!(w.severity, Warning);
    }

    #[test]
    fn default_severities_per_kind_match_brief() {
        use crate::report::FindingSeverity::*;
        assert_eq!(
            CausalWarningKind::StrongCorrelationNoIntervention.default_severity(),
            Warning
        );
        assert_eq!(CausalWarningKind::LikelyConfounder.default_severity(), Warning);
        assert_eq!(CausalWarningKind::CausalLanguageInLabel.default_severity(), Notice);
        assert_eq!(CausalWarningKind::ObservationalOnly.default_severity(), Notice);
        assert_eq!(
            CausalWarningKind::ModelExplanationAsCausal.default_severity(),
            Warning
        );
    }

    #[test]
    fn dag_does_not_downgrade_unrelated_correlations() {
        let mut cfg = CausalConfig::default();
        cfg.assumed_dag.add_edge("x", "y").unwrap();
        let cors = vec![corr("a", "b", 0.95)];
        let r = audit_correlations(&cors, None, &cfg, None, false);
        let w = r
            .warnings
            .iter()
            .find(|w| w.kind == CausalWarningKind::StrongCorrelationNoIntervention)
            .unwrap();
        assert!(!w.message.contains("acknowledged by causal DAG"));
    }
}
