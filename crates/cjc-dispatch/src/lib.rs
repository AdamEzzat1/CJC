//! Operator and function dispatch layer for the CJC runtime.
//!
//! This crate provides multiple dispatch resolution for overloaded functions
//! and operators. It matches call-site argument types against registered
//! [`FnSigEntry`] signatures in the [`TypeEnv`], ranking candidates by
//! per-parameter [`Specificity`] to select the most precise overload.
//!
//! # Architecture
//!
//! The dispatch pipeline works in two phases:
//!
//! 1. **Applicability filtering** -- each registered signature is checked
//!    against the call-site argument types. Signatures whose arity or types
//!    do not match are discarded.
//! 2. **Specificity ranking** -- remaining candidates are compared
//!    lexicographically by their per-parameter [`Specificity`] vectors.
//!    The candidate with the highest specificity wins. Ties produce an
//!    [`DispatchResult::Ambiguous`] error.
//!
//! # Key types
//!
//! | Type | Purpose |
//! |------|---------|
//! | [`Specificity`] | Ranks how precisely a parameter matches |
//! | [`DispatchResult`] | Outcome of a dispatch attempt |
//! | [`Dispatcher`] | Performs resolution against a [`TypeEnv`] |
//! | [`CoherenceChecker`] | Detects overlapping function definitions |
//!
//! [`FnSigEntry`]: cjc_types::FnSigEntry
//! [`TypeEnv`]: cjc_types::TypeEnv

use cjc_types::{FnSigEntry, Type, TypeEnv};
use cjc_diag::{Diagnostic, DiagnosticBag, Span};

/// Specificity level of a single parameter match during dispatch resolution.
///
/// Variants are ordered from least specific ([`None`](Specificity::None)) to
/// most specific ([`Concrete`](Specificity::Concrete)). The derived [`Ord`]
/// implementation reflects this ordering, so higher specificity compares
/// greater.
///
/// During resolution, [`Dispatcher::resolve`] computes a [`Specificity`] for
/// every parameter of every candidate signature. The candidate whose
/// specificity vector is lexicographically greatest wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Specificity {
    /// The parameter type does not match the argument type at all.
    ///
    /// A signature containing any `None` specificity is not applicable and
    /// is discarded during the filtering phase.
    None,
    /// The parameter is an unconstrained generic type variable (no trait
    /// bounds).
    ///
    /// Matches any argument type but ranks below [`Constrained`](Specificity::Constrained)
    /// and [`Concrete`](Specificity::Concrete).
    Generic,
    /// The parameter is a generic type variable with one or more trait bounds,
    /// all of which the argument type satisfies.
    ///
    /// Ranks above [`Generic`](Specificity::Generic) because the bounds
    /// narrow the set of acceptable types.
    Constrained,
    /// The parameter type matches the argument type exactly (concrete type
    /// equality via [`TypeEnv::types_match`]).
    ///
    /// This is the highest specificity level and always wins over generic
    /// matches.
    Concrete,
}

/// Outcome of a dispatch resolution attempt by [`Dispatcher::resolve`].
///
/// Callers should match on this enum to determine whether the call can
/// proceed, needs disambiguation, or should be reported as an error via
/// [`Dispatcher::dispatch_error_diagnostic`].
#[derive(Debug)]
pub enum DispatchResult {
    /// Exactly one best-matching signature was found.
    ///
    /// The contained [`FnSigEntry`] is the resolved overload that should be
    /// used for code generation or interpretation.
    Resolved(FnSigEntry),
    /// Two or more signatures tied for the highest specificity.
    ///
    /// The contained vector holds all equally-specific candidates. The caller
    /// should emit an ambiguity diagnostic (error code `E0301`).
    Ambiguous(Vec<FnSigEntry>),
    /// No registered signature matched the call-site argument types.
    ///
    /// `candidates` contains every signature registered under the queried
    /// name (may be empty if the function is entirely undefined). The caller
    /// should emit a "no match" diagnostic (error code `E0302`).
    NoMatch { candidates: Vec<FnSigEntry> },
}

/// Multiple dispatch resolver for overloaded functions and operators.
///
/// A `Dispatcher` borrows a [`TypeEnv`] and uses its registered function
/// signatures ([`TypeEnv::fn_sigs`]) to resolve calls by argument type.
///
/// # Lifetime
///
/// The `'a` lifetime ties the dispatcher to the borrowed [`TypeEnv`]. The
/// dispatcher performs no mutation and can be used for multiple resolutions
/// against the same environment.
///
/// # Examples
///
/// ```ignore
/// let env: TypeEnv = /* ... */;
/// let dispatcher = Dispatcher::new(&env);
///
/// match dispatcher.resolve("add", &[Type::F64, Type::F64]) {
///     DispatchResult::Resolved(sig) => { /* use sig */ }
///     DispatchResult::Ambiguous(sigs) => { /* report ambiguity */ }
///     DispatchResult::NoMatch { .. } => { /* report missing function */ }
/// }
/// ```
pub struct Dispatcher<'a> {
    /// Reference to the type environment containing registered function
    /// signatures and trait implementations.
    env: &'a TypeEnv,
}

impl<'a> Dispatcher<'a> {
    /// Create a new dispatcher backed by the given [`TypeEnv`].
    ///
    /// # Arguments
    ///
    /// * `env` -- Type environment whose [`fn_sigs`](TypeEnv::fn_sigs) map
    ///   will be queried during resolution.
    pub fn new(env: &'a TypeEnv) -> Self {
        Self { env }
    }

    /// Resolve a function call to the most specific matching overload.
    ///
    /// Walk all signatures registered under `name` in the [`TypeEnv`],
    /// discard those whose arity or parameter types are incompatible with
    /// `arg_types`, then select the candidate with the lexicographically
    /// highest per-parameter [`Specificity`] vector.
    ///
    /// # Arguments
    ///
    /// * `name` -- Function or operator name to look up.
    /// * `arg_types` -- Concrete types of the call-site arguments, in order.
    ///
    /// # Returns
    ///
    /// * [`DispatchResult::Resolved`] -- exactly one best match.
    /// * [`DispatchResult::Ambiguous`] -- multiple equally-specific matches.
    /// * [`DispatchResult::NoMatch`] -- no applicable signature (or the name
    ///   is not registered at all).
    pub fn resolve(
        &self,
        name: &str,
        arg_types: &[Type],
    ) -> DispatchResult {
        let candidates = match self.env.fn_sigs.get(name) {
            Some(sigs) => sigs.clone(),
            None => return DispatchResult::NoMatch { candidates: vec![] },
        };

        // Step 1: Filter applicable candidates
        let applicable: Vec<(FnSigEntry, Vec<Specificity>)> = candidates
            .iter()
            .filter_map(|sig| {
                let specs = self.check_applicability(sig, arg_types)?;
                Some((sig.clone(), specs))
            })
            .collect();

        if applicable.is_empty() {
            return DispatchResult::NoMatch { candidates };
        }

        if applicable.len() == 1 {
            return DispatchResult::Resolved(applicable.into_iter().next().unwrap().0);
        }

        // Step 2: Find most specific
        let mut best_indices: Vec<usize> = vec![0];

        for i in 1..applicable.len() {
            let cmp = self.compare_specificity(&applicable[best_indices[0]].1, &applicable[i].1);
            match cmp {
                std::cmp::Ordering::Less => {
                    // New candidate is more specific
                    best_indices = vec![i];
                }
                std::cmp::Ordering::Equal => {
                    // Equally specific — ambiguity
                    best_indices.push(i);
                }
                std::cmp::Ordering::Greater => {
                    // Current best is still more specific
                }
            }
        }

        if best_indices.len() == 1 {
            DispatchResult::Resolved(applicable[best_indices[0]].0.clone())
        } else {
            let ambiguous: Vec<FnSigEntry> = best_indices
                .iter()
                .map(|&i| applicable[i].0.clone())
                .collect();
            DispatchResult::Ambiguous(ambiguous)
        }
    }

    /// Check whether a signature is applicable for the given argument types.
    ///
    /// Return a per-parameter [`Specificity`] vector if every parameter
    /// matches, or [`None`] if the signature is not applicable (wrong arity
    /// or any parameter fails to match).
    fn check_applicability(
        &self,
        sig: &FnSigEntry,
        arg_types: &[Type],
    ) -> Option<Vec<Specificity>> {
        if sig.params.len() != arg_types.len() {
            return None;
        }

        let mut specificities = Vec::with_capacity(arg_types.len());

        for ((_, param_type), arg_type) in sig.params.iter().zip(arg_types) {
            let spec = self.match_param(param_type, arg_type, &sig.type_params);
            if spec == Specificity::None {
                return None;
            }
            specificities.push(spec);
        }

        Some(specificities)
    }

    /// Determine the [`Specificity`] of a single parameter-to-argument match.
    ///
    /// The matching rules, in priority order:
    ///
    /// 1. If either type is an error sentinel, return [`Specificity::Concrete`]
    ///    (error recovery -- let later phases report the real issue).
    /// 2. If the types are equal via [`TypeEnv::types_match`], return
    ///    [`Specificity::Concrete`].
    /// 3. If the parameter is a [`Type::Var`] (inference variable), return
    ///    [`Specificity::Generic`].
    /// 4. If the parameter is a [`Type::Unresolved`] name that corresponds to
    ///    a type parameter in `type_params`, check trait bounds:
    ///    - All bounds satisfied and non-empty: [`Specificity::Constrained`].
    ///    - All bounds satisfied and empty: [`Specificity::Generic`].
    /// 5. Otherwise return [`Specificity::None`] (no match).
    fn match_param(
        &self,
        param_type: &Type,
        arg_type: &Type,
        type_params: &[(String, Vec<String>)],
    ) -> Specificity {
        // Error types always match
        if param_type.is_error() || arg_type.is_error() {
            return Specificity::Concrete;
        }

        // Exact match
        if self.env.types_match(param_type, arg_type) {
            return Specificity::Concrete;
        }

        // Check if param is a type variable
        if let Type::Var(_) = param_type {
            return Specificity::Generic;
        }

        // Check if param is an unresolved name that's a type parameter
        if let Type::Unresolved(name) = param_type {
            for (tp_name, bounds) in type_params {
                if tp_name == name {
                    // Check trait bounds
                    let all_bounds_satisfied = bounds
                        .iter()
                        .all(|bound| self.env.satisfies_trait(arg_type, bound));

                    if all_bounds_satisfied {
                        if bounds.is_empty() {
                            return Specificity::Generic;
                        } else {
                            return Specificity::Constrained;
                        }
                    }
                }
            }
        }

        Specificity::None
    }

    /// Compare two per-parameter specificity vectors lexicographically.
    ///
    /// Return [`Ordering::Less`](std::cmp::Ordering::Less) when `b` is more
    /// specific than `a`, [`Ordering::Greater`](std::cmp::Ordering::Greater)
    /// when `a` is more specific, and [`Ordering::Equal`](std::cmp::Ordering::Equal)
    /// when they are tied.
    fn compare_specificity(
        &self,
        a: &[Specificity],
        b: &[Specificity],
    ) -> std::cmp::Ordering {
        // Lexicographic comparison: higher specificity wins
        for (sa, sb) in a.iter().zip(b) {
            match sb.cmp(sa) {
                std::cmp::Ordering::Greater => return std::cmp::Ordering::Less,
                std::cmp::Ordering::Less => return std::cmp::Ordering::Greater,
                std::cmp::Ordering::Equal => continue,
            }
        }
        std::cmp::Ordering::Equal
    }

    /// Generate a [`Diagnostic`] describing a dispatch failure.
    ///
    /// Call this after [`resolve`](Dispatcher::resolve) returns
    /// [`Ambiguous`](DispatchResult::Ambiguous) or
    /// [`NoMatch`](DispatchResult::NoMatch) to produce a user-facing error
    /// with candidate hints.
    ///
    /// # Arguments
    ///
    /// * `name` -- Name of the function or operator that was being resolved.
    /// * `arg_types` -- Call-site argument types (used in the error message).
    /// * `result` -- The [`DispatchResult`] returned by [`resolve`](Dispatcher::resolve).
    /// * `span` -- Source location of the call expression, attached to the
    ///   diagnostic for IDE integration.
    ///
    /// # Returns
    ///
    /// A [`Diagnostic`] with one of the following error codes:
    ///
    /// | Code | Condition |
    /// |------|-----------|
    /// | `E0301` | Ambiguous resolution -- multiple equally-specific candidates |
    /// | `E0302` | No matching function for the given argument types |
    /// | `E0300` | Internal error (called on a [`Resolved`](DispatchResult::Resolved) result) |
    pub fn dispatch_error_diagnostic(
        &self,
        name: &str,
        arg_types: &[Type],
        result: &DispatchResult,
        span: Span,
    ) -> Diagnostic {
        match result {
            DispatchResult::Ambiguous(candidates) => {
                let mut diag = Diagnostic::error(
                    "E0301",
                    format!("ambiguous method resolution for `{}`", name),
                    span,
                );

                for sig in candidates {
                    let params_str = sig
                        .params
                        .iter()
                        .map(|(n, t)| format!("{}: {}", n, t))
                        .collect::<Vec<_>>()
                        .join(", ");
                    diag.hints.push(format!(
                        "candidate: fn {}({}) -> {}",
                        sig.name, params_str, sig.ret
                    ));
                }

                diag.hints.push(
                    "add a more specific overload or use explicit type annotations to disambiguate"
                        .into(),
                );

                diag
            }
            DispatchResult::NoMatch { candidates } => {
                let arg_types_str = arg_types
                    .iter()
                    .map(|t| format!("{}", t))
                    .collect::<Vec<_>>()
                    .join(", ");

                let mut diag = Diagnostic::error(
                    "E0302",
                    format!(
                        "no matching function `{}` for argument types ({})",
                        name, arg_types_str
                    ),
                    span,
                );

                if !candidates.is_empty() {
                    for sig in candidates {
                        let params_str = sig
                            .params
                            .iter()
                            .map(|(n, t)| format!("{}: {}", n, t))
                            .collect::<Vec<_>>()
                            .join(", ");
                        diag.hints.push(format!(
                            "available: fn {}({})",
                            sig.name, params_str
                        ));
                    }
                } else {
                    diag.hints.push(format!(
                        "function `{}` is not defined; check for typos or define it",
                        name
                    ));
                }

                diag
            }
            DispatchResult::Resolved(_) => {
                // This shouldn't be called for resolved results
                Diagnostic::error("E0300", "internal dispatch error", span)
            }
        }
    }
}

/// Coherence checker that detects overlapping (ambiguous) function definitions.
///
/// Two signatures for the same function name "overlap" when they have
/// identical arity and pairwise-matching parameter types (as determined by
/// [`TypeEnv::types_match`]). Overlapping definitions are reported as error
/// `E0303` because they would always be ambiguous at call sites.
///
/// # Lifetime
///
/// The `'a` lifetime ties the checker to the borrowed [`TypeEnv`].
///
/// # Examples
///
/// ```ignore
/// let env: TypeEnv = /* ... */;
/// let mut diags = DiagnosticBag::new();
/// CoherenceChecker::new(&env).check_overlaps(&mut diags);
///
/// if diags.has_errors() {
///     // report overlapping definitions to the user
/// }
/// ```
pub struct CoherenceChecker<'a> {
    /// Reference to the type environment containing registered function
    /// signatures to check for overlap.
    env: &'a TypeEnv,
}

impl<'a> CoherenceChecker<'a> {
    /// Create a new coherence checker backed by the given [`TypeEnv`].
    ///
    /// # Arguments
    ///
    /// * `env` -- Type environment whose [`fn_sigs`](TypeEnv::fn_sigs) map
    ///   will be inspected for overlapping definitions.
    pub fn new(env: &'a TypeEnv) -> Self {
        Self { env }
    }

    /// Check all registered function signatures for overlapping definitions.
    ///
    /// For every function name, compare each pair of signatures. If two
    /// signatures have the same arity and pairwise-matching parameter types,
    /// emit an `E0303` diagnostic into `diagnostics`.
    ///
    /// # Arguments
    ///
    /// * `diagnostics` -- Mutable reference to a [`DiagnosticBag`] where
    ///   overlap errors will be emitted.
    pub fn check_overlaps(&self, diagnostics: &mut DiagnosticBag) {
        for (name, sigs) in &self.env.fn_sigs {
            for i in 0..sigs.len() {
                for j in (i + 1)..sigs.len() {
                    if self.signatures_overlap(&sigs[i], &sigs[j]) {
                        diagnostics.emit(
                            Diagnostic::error(
                                "E0303",
                                format!("overlapping function definitions for `{}`", name),
                                Span::dummy(),
                            )
                            .with_hint(format!(
                                "both fn {}({}) and fn {}({}) match the same argument types",
                                sigs[i].name,
                                sigs[i].params.len(),
                                sigs[j].name,
                                sigs[j].params.len()
                            )),
                        );
                    }
                }
            }
        }
    }

    /// Return `true` if two signatures overlap (same arity and pairwise-matching
    /// parameter types).
    fn signatures_overlap(&self, a: &FnSigEntry, b: &FnSigEntry) -> bool {
        if a.params.len() != b.params.len() {
            return false;
        }

        a.params
            .iter()
            .zip(b.params.iter())
            .all(|((_, ta), (_, tb))| self.env.types_match(ta, tb))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_types::*;

    fn setup_env() -> TypeEnv {
        let mut env = TypeEnv::new();

        // fn add(a: f32, b: f32) -> f32
        env.register_fn(FnSigEntry {
            name: "add".into(),
            type_params: vec![],
            params: vec![("a".into(), Type::F32), ("b".into(), Type::F32)],
            ret: Type::F32,
            is_nogc: false,
            effects: EffectSet::default(),
        });

        // fn add(a: f64, b: f64) -> f64
        env.register_fn(FnSigEntry {
            name: "add".into(),
            type_params: vec![],
            params: vec![("a".into(), Type::F64), ("b".into(), Type::F64)],
            ret: Type::F64,
            is_nogc: false,
            effects: EffectSet::default(),
        });

        // fn add<T: Numeric>(a: T, b: T) -> T  (generic fallback)
        env.register_fn(FnSigEntry {
            name: "add".into(),
            type_params: vec![("T".into(), vec!["Numeric".into()])],
            params: vec![
                ("a".into(), Type::Unresolved("T".into())),
                ("b".into(), Type::Unresolved("T".into())),
            ],
            ret: Type::Unresolved("T".into()),
            is_nogc: false,
            effects: EffectSet::default(),
        });

        // fn process(x: f64)
        env.register_fn(FnSigEntry {
            name: "process".into(),
            type_params: vec![],
            params: vec![("x".into(), Type::F64)],
            ret: Type::Void,
            is_nogc: false,
            effects: EffectSet::default(),
        });

        // fn process<T: Float>(x: T)
        env.register_fn(FnSigEntry {
            name: "process".into(),
            type_params: vec![("T".into(), vec!["Float".into()])],
            params: vec![("x".into(), Type::Unresolved("T".into()))],
            ret: Type::Void,
            is_nogc: false,
            effects: EffectSet::default(),
        });

        // fn process<T: Numeric>(x: T)
        env.register_fn(FnSigEntry {
            name: "process".into(),
            type_params: vec![("T".into(), vec!["Numeric".into()])],
            params: vec![("x".into(), Type::Unresolved("T".into()))],
            ret: Type::Void,
            is_nogc: false,
            effects: EffectSet::default(),
        });

        env
    }

    #[test]
    fn test_concrete_dispatch() {
        let env = setup_env();
        let dispatcher = Dispatcher::new(&env);

        // add(f64, f64) -> should resolve to concrete f64 overload
        let result = dispatcher.resolve("add", &[Type::F64, Type::F64]);
        match result {
            DispatchResult::Resolved(sig) => {
                assert_eq!(sig.ret, Type::F64);
                assert!(sig.type_params.is_empty());
            }
            _ => panic!("expected resolved dispatch"),
        }
    }

    #[test]
    fn test_concrete_over_generic() {
        let env = setup_env();
        let dispatcher = Dispatcher::new(&env);

        // process(f64) -> concrete f64 should win over generic Float and Numeric
        let result = dispatcher.resolve("process", &[Type::F64]);
        match result {
            DispatchResult::Resolved(sig) => {
                assert!(sig.type_params.is_empty()); // concrete, no type params
            }
            _ => panic!("expected concrete dispatch"),
        }
    }

    #[test]
    fn test_constrained_over_unconstrained() {
        let env = setup_env();
        let dispatcher = Dispatcher::new(&env);

        // add(i32, i32) -> should resolve to Numeric-constrained generic
        let result = dispatcher.resolve("add", &[Type::I32, Type::I32]);
        match result {
            DispatchResult::Resolved(sig) => {
                assert!(!sig.type_params.is_empty());
            }
            _ => panic!("expected generic dispatch"),
        }
    }

    #[test]
    fn test_no_match() {
        let env = setup_env();
        let dispatcher = Dispatcher::new(&env);

        let result = dispatcher.resolve("add", &[Type::Bool, Type::Bool]);
        match result {
            DispatchResult::NoMatch { .. } => {}
            _ => panic!("expected no match"),
        }
    }

    #[test]
    fn test_wrong_arity() {
        let env = setup_env();
        let dispatcher = Dispatcher::new(&env);

        let result = dispatcher.resolve("add", &[Type::F64]);
        match result {
            DispatchResult::NoMatch { .. } => {}
            _ => panic!("expected no match for wrong arity"),
        }
    }

    #[test]
    fn test_undefined_function() {
        let env = setup_env();
        let dispatcher = Dispatcher::new(&env);

        let result = dispatcher.resolve("nonexistent", &[Type::I32]);
        match result {
            DispatchResult::NoMatch { candidates } => {
                assert!(candidates.is_empty());
            }
            _ => panic!("expected no match"),
        }
    }

    #[test]
    fn test_dispatch_error_diagnostic() {
        let env = setup_env();
        let dispatcher = Dispatcher::new(&env);

        let result = dispatcher.resolve("add", &[Type::Bool, Type::Bool]);
        let diag = dispatcher.dispatch_error_diagnostic(
            "add",
            &[Type::Bool, Type::Bool],
            &result,
            Span::new(0, 10),
        );
        assert_eq!(diag.code, "E0302");
        assert!(diag.message.contains("no matching function"));
    }

    #[test]
    fn test_specificity_ordering() {
        assert!(Specificity::Concrete > Specificity::Constrained);
        assert!(Specificity::Constrained > Specificity::Generic);
        assert!(Specificity::Generic > Specificity::None);
    }
}
