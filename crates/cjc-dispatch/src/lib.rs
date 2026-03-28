//! Operator dispatch layer for the CJC runtime.
//!
//! Resolves overloaded operators and function signatures by matching argument
//! types against registered signatures with specificity-based ranking.

use cjc_types::{FnSigEntry, Type, TypeEnv};
use cjc_diag::{Diagnostic, DiagnosticBag, Span};

/// Specificity level of a parameter match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Specificity {
    /// No match.
    None,
    /// Matches via unconstrained generic.
    Generic,
    /// Matches via constrained generic (trait bound).
    Constrained,
    /// Exact concrete type match.
    Concrete,
}

/// Result of dispatch resolution.
#[derive(Debug)]
pub enum DispatchResult {
    /// Single best match found.
    Resolved(FnSigEntry),
    /// Multiple equally-specific matches (ambiguity).
    Ambiguous(Vec<FnSigEntry>),
    /// No matching method found.
    NoMatch { candidates: Vec<FnSigEntry> },
}

/// Multiple dispatch resolver.
pub struct Dispatcher<'a> {
    env: &'a TypeEnv,
}

impl<'a> Dispatcher<'a> {
    pub fn new(env: &'a TypeEnv) -> Self {
        Self { env }
    }

    /// Resolve a function call to the most specific matching overload.
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

    /// Check if a signature is applicable for the given argument types.
    /// Returns per-parameter specificity if applicable, None if not.
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

    /// Determine specificity of a single parameter match.
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

    /// Compare two specificity vectors. Returns ordering of second vs first.
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

    /// Generate a diagnostic for dispatch errors.
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

/// Coherence checker: validates that no overlapping implementations exist.
pub struct CoherenceChecker<'a> {
    env: &'a TypeEnv,
}

impl<'a> CoherenceChecker<'a> {
    pub fn new(env: &'a TypeEnv) -> Self {
        Self { env }
    }

    /// Check for overlapping function definitions (same name, same parameter types).
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
