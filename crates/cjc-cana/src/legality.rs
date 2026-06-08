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
// PerPassLegalityGate — refined per-pass safety rules
// ---------------------------------------------------------------------------

/// Per-pass safety classification for [`PerPassLegalityGate`].
///
/// The two-tier classification reflects what each MIR optimization pass can
/// *structurally* do to a function's reduction operations:
///
/// - **[`Universal`]**: the pass cannot, by construction, reorder or alter
///   any reduction's bit-exact computation. Safe on every function regardless
///   of `strict_count`. Used for `constant_fold`, `cf_round_2`, `dce`, and
///   `licm`.
/// - **[`NoStrictReductions`]**: the pass *might* touch a reduction's
///   subexpression (e.g. CSE collapsing two occurrences of `a + b` where
///   either occurrence is part of a strict reduction, or SR rewriting
///   `x * 2.0` inside a strict accumulator). Safe only when the function has
///   zero strict reductions. Used for `cse` and `strength_reduce`.
///
/// See [`pass_safety_tier`] for the per-name mapping.
///
/// [`Universal`]: PassSafetyTier::Universal
/// [`NoStrictReductions`]: PassSafetyTier::NoStrictReductions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PassSafetyTier {
    /// Pass is structurally unable to alter any reduction's semantics.
    /// `constant_fold` (only touches literal subexpressions, never runtime
    /// reductions). `dce` (removes unreachable code, which by definition has
    /// no observers including reductions). `licm` (moves loop-invariants
    /// out; a reduction accumulator `acc = acc + ...` depends on the prior
    /// iteration's `acc`, so it's never loop-invariant by definition —
    /// LICM cannot hoist it).
    Universal,
    /// Pass might collapse or rewrite a subexpression that's part of a
    /// strict reduction, which would change the bit-exact computation order.
    /// `cse` (could collapse `a + b` occurrences across reduction
    /// boundaries). `strength_reduce` (rewrites multiplications/divisions —
    /// IEEE 754 guarantees `x + x == x * 2.0` for finite normal floats, but
    /// the safer policy is to skip when `strict_count > 0` until per-
    /// expression analysis lands).
    NoStrictReductions,
}

/// Classify a pass by its safety tier under the per-pass legality model.
///
/// Unknown pass names are conservatively assigned [`PassSafetyTier::NoStrictReductions`]
/// — if a future MIR pass is added that the gate doesn't yet know about, the
/// gate stays safe by default until classified.
pub fn pass_safety_tier(pass_name: &str) -> PassSafetyTier {
    match pass_name {
        // Tier 1: structurally cannot affect reduction order.
        "constant_fold" | "cf" | "cf_round_2" => PassSafetyTier::Universal,
        "dce" | "dead_code_elimination" => PassSafetyTier::Universal,
        "licm" | "loop_invariant_code_motion" => PassSafetyTier::Universal,

        // Tier 2: could touch a reduction's subexpression.
        "cse" | "common_subexpression_elimination" => PassSafetyTier::NoStrictReductions,
        "strength_reduce" | "sr" => PassSafetyTier::NoStrictReductions,

        // Unknown pass — conservative.
        _ => PassSafetyTier::NoStrictReductions,
    }
}

/// A refined legality gate that classifies passes by their structural
/// ability to alter reduction semantics, then applies per-pass safety
/// rules.
///
/// Versus [`DefaultLegalityGate`]: where Default rejects *any* pass touching
/// a function with `strict_count > 0`, `PerPassLegalityGate` allows
/// [`PassSafetyTier::Universal`] passes (CF, DCE, LICM) regardless of
/// strict_count. [`PassSafetyTier::NoStrictReductions`] passes (CSE, SR)
/// remain blocked when `strict_count > 0`.
///
/// **Why this matters:** the §17 / §18 investigation revealed that
/// `DefaultLegalityGate` was vetoing every float-heavy ML workload from
/// every optimization, regardless of cost-model verdict. With
/// `PerPassLegalityGate`, PINN-like programs get at least CF + DCE + LICM
/// recommendations on functions where the trained model identifies
/// positive predicted benefit.
///
/// All other behavior matches `DefaultLegalityGate`: empty sequence
/// approved, unknown functions rejected with
/// [`LegalityViolation::UnknownFunction`].
///
/// Deterministic by construction — same `(program, sequence, features)`
/// always produces the same verdict.
#[derive(Debug, Clone, Default)]
pub struct PerPassLegalityGate;

impl PerPassLegalityGate {
    pub fn new() -> Self {
        Self
    }
}

impl LegalityGate for PerPassLegalityGate {
    fn verify(
        &self,
        program: &MirProgram,
        sequence: &PassSequence,
        features: &CanaFeatures,
    ) -> LegalityVerdict {
        if sequence.per_function.is_empty() {
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

            // Extract the pass name from ProposedPass::Run(_) / Skip(_).
            let pass_name = match proposed {
                ProposedPass::Run(name) => name.as_str(),
                ProposedPass::Skip(name) => name.as_str(),
            };
            let tier = pass_safety_tier(pass_name);

            // Universal passes never need a reduction check. We only
            // consult `strict_count` for NoStrictReductions-tier passes.
            if tier == PassSafetyTier::NoStrictReductions {
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
                // If features don't have the function, defer to
                // DefaultLegalityGate's pattern: not paranoid.
            }
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

    // -----------------------------------------------------------------------
    // PerPassLegalityGate — per-pass safety rules
    // -----------------------------------------------------------------------

    /// Build a `CanaFeatures` for a single named function with a custom
    /// `strict_count`. We can't easily synthesize a `MirProgram` whose
    /// reduction analysis produces a specific strict_count without a
    /// real function body, so we extract from an empty function and then
    /// mutate the feature struct directly to inject strict reductions.
    /// This is a test-only shortcut; production features always come from
    /// `extract()`.
    fn features_with_strict_reductions(fn_name: &str, strict_fold: u32) -> CanaFeatures {
        let program = empty_program(&[fn_name]);
        let mut features = extract(&program);
        if let Some(ff) = features.per_fn.get_mut(fn_name) {
            ff.reductions.strict_fold = strict_fold;
        }
        features
    }

    #[test]
    fn pass_safety_tier_universal_passes() {
        // CF, DCE, LICM are all Tier 1.
        assert_eq!(
            pass_safety_tier("constant_fold"),
            PassSafetyTier::Universal,
        );
        assert_eq!(pass_safety_tier("cf"), PassSafetyTier::Universal);
        assert_eq!(pass_safety_tier("cf_round_2"), PassSafetyTier::Universal);
        assert_eq!(pass_safety_tier("dce"), PassSafetyTier::Universal);
        assert_eq!(
            pass_safety_tier("dead_code_elimination"),
            PassSafetyTier::Universal,
        );
        assert_eq!(pass_safety_tier("licm"), PassSafetyTier::Universal);
        assert_eq!(
            pass_safety_tier("loop_invariant_code_motion"),
            PassSafetyTier::Universal,
        );
    }

    #[test]
    fn pass_safety_tier_no_strict_reductions_passes() {
        assert_eq!(
            pass_safety_tier("cse"),
            PassSafetyTier::NoStrictReductions,
        );
        assert_eq!(
            pass_safety_tier("common_subexpression_elimination"),
            PassSafetyTier::NoStrictReductions,
        );
        assert_eq!(
            pass_safety_tier("strength_reduce"),
            PassSafetyTier::NoStrictReductions,
        );
        assert_eq!(pass_safety_tier("sr"), PassSafetyTier::NoStrictReductions);
    }

    #[test]
    fn pass_safety_tier_unknown_is_conservative() {
        // Unknown pass names default to NoStrictReductions (conservative).
        assert_eq!(
            pass_safety_tier("hypothetical_future_pass"),
            PassSafetyTier::NoStrictReductions,
        );
    }

    #[test]
    fn per_pass_gate_empty_sequence_approved() {
        let program = empty_program(&["__main"]);
        let features = extract_or_empty(&program);
        let gate = PerPassLegalityGate::new();
        let verdict = gate.verify(&program, &PassSequence::empty(), &features);
        assert!(verdict.is_approved());
    }

    #[test]
    fn per_pass_gate_unknown_function_rejected() {
        let program = empty_program(&["__main"]);
        let features = extract_or_empty(&program);
        let gate = PerPassLegalityGate::new();
        let mut seq = PassSequence::empty();
        seq.per_function.insert(
            "ghost".to_string(),
            vec![ProposedPass::Run("constant_fold".to_string())],
        );
        let verdict = gate.verify(&program, &seq, &features);
        assert!(matches!(verdict, LegalityVerdict::Rejected(_)));
    }

    #[test]
    fn per_pass_gate_universal_pass_approved_despite_strict_reductions() {
        // This is the headline change vs DefaultLegalityGate:
        // CF/DCE/LICM are approved even when strict_count > 0.
        let features = features_with_strict_reductions("hot_fn", 5);
        let program = empty_program(&["hot_fn"]);
        let gate = PerPassLegalityGate::new();

        for universal_pass in &[
            "constant_fold",
            "cf",
            "cf_round_2",
            "dce",
            "dead_code_elimination",
            "licm",
            "loop_invariant_code_motion",
        ] {
            let mut seq = PassSequence::empty();
            seq.per_function.insert(
                "hot_fn".to_string(),
                vec![ProposedPass::Run(universal_pass.to_string())],
            );
            let verdict = gate.verify(&program, &seq, &features);
            assert!(
                verdict.is_approved(),
                "Universal pass `{}` should be approved on a function with \
                 strict_count=5; got {:?}",
                universal_pass,
                verdict,
            );
        }
    }

    #[test]
    fn per_pass_gate_no_strict_reductions_pass_rejected_when_strict() {
        // CSE and SR are still blocked when strict_count > 0, matching the
        // user's stated tiering ("CSE: safe when subexpression isn't a
        // reduction"; "SR: safe when multiplication isn't part of a
        // strict reduction"). Conservative: block when ANY strict
        // reduction is present until per-expression analysis lands.
        let features = features_with_strict_reductions("hot_fn", 3);
        let program = empty_program(&["hot_fn"]);
        let gate = PerPassLegalityGate::new();

        for restricted_pass in &["cse", "common_subexpression_elimination", "strength_reduce", "sr"] {
            let mut seq = PassSequence::empty();
            seq.per_function.insert(
                "hot_fn".to_string(),
                vec![ProposedPass::Run(restricted_pass.to_string())],
            );
            let verdict = gate.verify(&program, &seq, &features);
            match verdict {
                LegalityVerdict::Rejected(violations) => {
                    assert!(
                        violations.iter().any(|v| matches!(
                            v,
                            LegalityViolation::StrictReductionPresent { .. }
                        )),
                        "Restricted pass `{}` on strict-reduction function \
                         should be rejected with StrictReductionPresent; got {:?}",
                        restricted_pass,
                        violations,
                    );
                }
                LegalityVerdict::Approved => panic!(
                    "Restricted pass `{}` should be rejected on a function \
                     with strict_count=3",
                    restricted_pass,
                ),
            }
        }
    }

    #[test]
    fn per_pass_gate_no_strict_reductions_pass_approved_when_clean() {
        // CSE and SR are approved when the function has no strict
        // reductions — the conservative gate doesn't add restrictions
        // that DefaultLegalityGate wouldn't impose.
        let program = empty_program(&["clean_fn"]);
        let features = extract(&program); // strict_count = 0 by construction
        let gate = PerPassLegalityGate::new();

        for pass in &["cse", "strength_reduce"] {
            let mut seq = PassSequence::empty();
            seq.per_function.insert(
                "clean_fn".to_string(),
                vec![ProposedPass::Run(pass.to_string())],
            );
            let verdict = gate.verify(&program, &seq, &features);
            assert!(
                verdict.is_approved(),
                "{} should be approved on a strict-reduction-free function; got {:?}",
                pass,
                verdict,
            );
        }
    }

    #[test]
    fn per_pass_gate_strictly_more_permissive_than_default() {
        // A direct head-to-head: a function with strict reductions, CF
        // requested.
        // - DefaultLegalityGate rejects it (any pass on strict_count > 0).
        // - PerPassLegalityGate approves it (CF is Tier 1).
        let features = features_with_strict_reductions("pinn_forward", 4);
        let program = empty_program(&["pinn_forward"]);
        let mut seq = PassSequence::empty();
        seq.per_function.insert(
            "pinn_forward".to_string(),
            vec![ProposedPass::Run("constant_fold".to_string())],
        );

        let default_verdict = DefaultLegalityGate::new().verify(&program, &seq, &features);
        let per_pass_verdict = PerPassLegalityGate::new().verify(&program, &seq, &features);

        assert!(
            !default_verdict.is_approved(),
            "DefaultLegalityGate should still reject (current behavior)",
        );
        assert!(
            per_pass_verdict.is_approved(),
            "PerPassLegalityGate should approve CF on strict-reduction function",
        );
    }

    #[test]
    fn per_pass_gate_deterministic() {
        // Same input → same verdict, every call.
        let features = features_with_strict_reductions("f", 2);
        let program = empty_program(&["f"]);
        let mut seq = PassSequence::empty();
        seq.per_function.insert(
            "f".to_string(),
            vec![
                ProposedPass::Run("cse".to_string()),
                ProposedPass::Run("constant_fold".to_string()),
            ],
        );
        let gate = PerPassLegalityGate::new();
        let v1 = gate.verify(&program, &seq, &features);
        for _ in 0..50 {
            let vn = gate.verify(&program, &seq, &features);
            assert_eq!(v1, vn, "per-pass gate must be deterministic");
        }
    }
}
