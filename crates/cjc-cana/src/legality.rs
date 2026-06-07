//! Legality gate trait + Phase-1 default implementation.
//!
//! Phase 1 has *no* recommendations to gate — the optimizer still runs its
//! fixed 6-pass sequence (CF → SR → DCE → CSE → LICM → CF). The gate exists
//! so Phase 2 (advisory recommendations) has somewhere to plug in *before*
//! it can change anything.
//!
//! The default gate is intentionally conservative:
//!
//! 1. The empty pass sequence (no proposed change) is always [`Approved`].
//! 2. Any reorder/skip recommendation that would touch a function containing
//!    a strict reduction (`StrictFold` / `KahanFold` / `Unknown`) is
//!    [`Rejected`] with a [`LegalityViolation::StrictReductionPresent`].
//! 3. Any recommendation referencing an unknown function name is
//!    [`Rejected`] with [`LegalityViolation::UnknownFunction`].
//!
//! That's it. We deliberately avoid encoding rules CANA can't yet check —
//! the gate's job is to be *safe by default*, not to be clever.
//!
//! # Why the gate sees both the program AND the features
//!
//! The features cache the reduction histogram per function, which is *exactly*
//! the signal the gate needs. Passing both lets us avoid recomputing the
//! reduction analysis from `&MirProgram` on every verify() call.

use std::collections::BTreeMap;

use cjc_mir::MirProgram;

use crate::features::CanaFeatures;

// ---------------------------------------------------------------------------
// Proposed pass sequence — Phase-2 recommendation shape
// ---------------------------------------------------------------------------

/// A proposed compiler-pass sequence to verify.
///
/// Phase 1 only constructs the empty sequence (no proposed change); Phase 2
/// will populate it with reorder/skip recommendations. We define the shape
/// now so the gate's signature is stable across phases.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PassSequence {
    /// Per-function proposed pass list. Functions not in the map keep the
    /// compiler's default sequence.
    pub per_function: BTreeMap<String, Vec<ProposedPass>>,
}

/// A single proposed compiler-pass action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProposedPass {
    /// Run a pass by name (e.g. `"constant_fold"`, `"dce"`, `"licm"`).
    /// The semantics are identical to the default pipeline — only the
    /// *position* in the sequence may differ.
    Run(String),
    /// Skip a pass by name (the default pipeline runs it; CANA recommends
    /// not running it on this function).
    Skip(String),
}

impl PassSequence {
    /// The "no change" sequence — always passes the default gate.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Iterate over `(function_name, proposed_pass)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ProposedPass)> {
        self.per_function
            .iter()
            .flat_map(|(fname, passes)| passes.iter().map(move |p| (fname.as_str(), p)))
    }
}

// ---------------------------------------------------------------------------
// Verdict
// ---------------------------------------------------------------------------

/// Outcome of running a [`LegalityGate`] over a [`PassSequence`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum LegalityVerdict {
    /// All proposed actions are safe to apply.
    #[default]
    Approved,
    /// One or more violations were detected. Phase-2 advisors should drop
    /// the affected actions or escalate to the user.
    Rejected(Vec<LegalityViolation>),
}

impl LegalityVerdict {
    /// `true` iff verdict is [`Approved`].
    pub fn is_approved(&self) -> bool {
        matches!(self, LegalityVerdict::Approved)
    }

    /// `Some(_)` iff verdict is [`Rejected`].
    pub fn violations(&self) -> Option<&[LegalityViolation]> {
        match self {
            LegalityVerdict::Rejected(v) => Some(v),
            LegalityVerdict::Approved => None,
        }
    }
}

/// A single legality violation against a proposed pass action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LegalityViolation {
    /// A reorder/skip was proposed for a function containing a strict
    /// reduction; the gate refuses to let CANA touch it without explicit
    /// override.
    StrictReductionPresent {
        function: String,
        proposed: ProposedPass,
        strict_count: u32,
    },
    /// The proposed action references a function name not present in the
    /// program. Almost always a stale recommendation from a prior compilation.
    UnknownFunction {
        function: String,
        proposed: ProposedPass,
    },
}

// ---------------------------------------------------------------------------
// Trait surface
// ---------------------------------------------------------------------------

/// A pure-function check that vets a proposed [`PassSequence`] against the
/// determinism / safety contract.
///
/// Implementations MUST be:
/// - Deterministic: same `(program, sequence, features)` → same verdict
/// - Pure: no RNG, no time, no IO, no global state
/// - Total: never panic, regardless of how malformed the inputs are
pub trait LegalityGate {
    /// Vet `sequence` against `program` using cached `features`.
    fn verify(
        &self,
        program: &MirProgram,
        sequence: &PassSequence,
        features: &CanaFeatures,
    ) -> LegalityVerdict;
}

// ---------------------------------------------------------------------------
// DefaultLegalityGate — conservative Phase-1 default
// ---------------------------------------------------------------------------

/// Phase-1 default legality gate.
///
/// Refuses *any* reorder/skip touching a function with a strict reduction,
/// and refuses *any* action referencing an unknown function name. Everything
/// else is approved.
#[derive(Debug, Clone, Default)]
pub struct DefaultLegalityGate;

impl DefaultLegalityGate {
    pub fn new() -> Self {
        Self
    }
}

impl LegalityGate for DefaultLegalityGate {
    fn verify(
        &self,
        program: &MirProgram,
        sequence: &PassSequence,
        features: &CanaFeatures,
    ) -> LegalityVerdict {
        if sequence.per_function.is_empty() {
            // Nothing proposed → nothing to gate → trivially safe.
            return LegalityVerdict::Approved;
        }

        let mut violations = Vec::new();
        let known_fns: std::collections::BTreeSet<&str> =
            program.functions.iter().map(|f| f.name.as_str()).collect();

        for (fname, proposed) in sequence.iter() {
            if !known_fns.contains(fname) {
                violations.push(LegalityViolation::UnknownFunction {
                    function: fname.to_string(),
                    proposed: proposed.clone(),
                });
                continue;
            }
            if let Some(fn_feats) = features.per_fn.get(fname) {
                let strict = fn_feats.reductions.strict_count();
                if strict > 0 {
                    violations.push(LegalityViolation::StrictReductionPresent {
                        function: fname.to_string(),
                        proposed: proposed.clone(),
                        strict_count: strict,
                    });
                }
            }
            // If the function isn't in the features map, it's a brand-new
            // synthetic CANA never indexed — the gate is conservative but
            // not paranoid; we don't fabricate a violation for that case.
        }

        if violations.is_empty() {
            LegalityVerdict::Approved
        } else {
            LegalityVerdict::Rejected(violations)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{extract, CanaFeatures};
    use cjc_mir::{MirBody, MirFnId, MirFunction, MirProgram};

    fn empty_program(fn_names: &[&str]) -> MirProgram {
        let functions: Vec<MirFunction> = fn_names
            .iter()
            .enumerate()
            .map(|(i, n)| MirFunction {
                id: MirFnId(i as u32),
                name: n.to_string(),
                type_params: vec![],
                params: vec![],
                return_type: None,
                body: MirBody {
                    stmts: vec![],
                    result: None,
                },
                is_nogc: false,
                cfg_body: None,
                decorators: vec![],
                vis: cjc_ast::Visibility::Public,
                local_count: 0,
            })
            .collect();
        MirProgram {
            functions,
            struct_defs: vec![],
            enum_defs: vec![],
            entry: MirFnId(0),
        }
    }

    fn extract_or_empty(program: &MirProgram) -> CanaFeatures {
        extract(program)
    }

    #[test]
    fn empty_sequence_always_approved() {
        let program = empty_program(&["__main"]);
        let features = extract_or_empty(&program);
        let gate = DefaultLegalityGate::new();
        let verdict = gate.verify(&program, &PassSequence::empty(), &features);
        assert!(verdict.is_approved());
        assert!(verdict.violations().is_none());
    }

    #[test]
    fn unknown_function_rejected() {
        let program = empty_program(&["__main"]);
        let features = extract_or_empty(&program);
        let gate = DefaultLegalityGate::new();
        let mut seq = PassSequence::empty();
        seq.per_function.insert(
            "ghost".to_string(),
            vec![ProposedPass::Skip("dce".to_string())],
        );
        let verdict = gate.verify(&program, &seq, &features);
        match verdict {
            LegalityVerdict::Rejected(v) => {
                assert_eq!(v.len(), 1);
                assert!(matches!(
                    v[0],
                    LegalityViolation::UnknownFunction { ref function, .. } if function == "ghost"
                ));
            }
            _ => panic!("expected Rejected"),
        }
    }

    #[test]
    fn known_fn_no_strict_reduction_approved() {
        // A function with no body has no reductions → strict_count == 0 → ok.
        let program = empty_program(&["worker"]);
        let features = extract_or_empty(&program);
        let gate = DefaultLegalityGate::new();
        let mut seq = PassSequence::empty();
        seq.per_function.insert(
            "worker".to_string(),
            vec![ProposedPass::Skip("licm".to_string())],
        );
        let verdict = gate.verify(&program, &seq, &features);
        assert!(verdict.is_approved(), "got {:?}", verdict);
    }
}
