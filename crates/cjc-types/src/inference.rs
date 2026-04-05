//! Bidirectional type inference engine for CJC.
//!
//! # Design
//!
//! Inference is **local** -- it operates within a single function body.
//! Function parameter types are always required by the grammar; `let`
//! bindings and return types can be inferred.
//!
//! Two modes cooperate during type checking:
//!
//! - **Synthesis** (bottom-up): compute the type of an expression from its
//!   sub-expressions.
//! - **Checking** (top-down): verify an expression has an expected type
//!   propagated from its context.
//!
//! # Constraint workflow
//!
//! 1. Create an [`InferCtx`] at the start of each function body.
//! 2. Generate fresh type variables with [`InferCtx::fresh_var`] wherever a
//!    type is unknown.
//! 3. Record equality constraints with [`InferCtx::constrain`] as the type
//!    checker walks the AST.
//! 4. Call [`InferCtx::solve`] to unify all constraints at once.
//! 5. Use [`InferCtx::resolve_final`] to replace remaining unresolved
//!    variables with sensible defaults (`i64` for integers, `f64` for floats).

use cjc_diag::{Diagnostic, Span};
use super::{Type, TypeVarId, TypeSubst, unify};

/// Inference context for a single function body.
///
/// Collects type constraints during type checking and solves them
/// via unification at the end. Each function body gets its own
/// `InferCtx`, ensuring inference stays local and deterministic.
///
/// # Usage
///
/// ```
/// use cjc_types::inference::InferCtx;
/// use cjc_types::Type;
/// use cjc_diag::Span;
///
/// let mut ctx = InferCtx::new(0);
/// let tv = ctx.fresh_var();           // ?0
/// ctx.constrain(tv.clone(), Type::I64, Span::new(0, 1));
/// let errors = ctx.solve();
/// assert!(errors.is_empty());
/// assert_eq!(ctx.apply(&tv), Type::I64);
/// ```
pub struct InferCtx {
    /// Counter for generating fresh type variables.
    next_var: usize,
    /// Collected constraints: (lhs_type, rhs_type, span_for_error).
    constraints: Vec<(Type, Type, Span)>,
    /// Current substitution (partial solution).
    subst: TypeSubst,
}

impl InferCtx {
    /// Create a new inference context with a starting variable counter.
    ///
    /// # Arguments
    ///
    /// * `start_var` -- The initial type-variable counter. Pass the current
    ///   counter from [`TypeEnv`](super::TypeEnv) so that fresh variables do
    ///   not collide with those already in scope.
    pub fn new(start_var: usize) -> Self {
        Self {
            next_var: start_var,
            constraints: Vec::new(),
            subst: TypeSubst::new(),
        }
    }

    /// Generate a fresh type variable.
    ///
    /// Each call increments the internal counter and returns a unique
    /// `Type::Var(TypeVarId(n))`. The variable is initially unconstrained.
    ///
    /// # Returns
    ///
    /// A `Type::Var` with a globally unique (within this context) identifier.
    pub fn fresh_var(&mut self) -> Type {
        let id = TypeVarId(self.next_var);
        self.next_var += 1;
        Type::Var(id)
    }

    /// Get the current variable counter so it can be passed back to
    /// [`TypeEnv`](super::TypeEnv) after inference completes.
    pub fn next_var_counter(&self) -> usize {
        self.next_var
    }

    /// Add a constraint requiring `lhs` to unify with `rhs`.
    ///
    /// Constraints are accumulated and solved together by [`solve`](Self::solve).
    /// The `span` is attached to any diagnostic emitted if the constraint
    /// turns out to be unsatisfiable.
    ///
    /// # Arguments
    ///
    /// * `lhs` -- Left-hand type (often an inferred variable).
    /// * `rhs` -- Right-hand type (often a concrete or expected type).
    /// * `span` -- Source span for error reporting.
    pub fn constrain(&mut self, lhs: Type, rhs: Type, span: Span) {
        self.constraints.push((lhs, rhs, span));
    }

    /// Solve all constraints. Returns diagnostics for unsolvable ones.
    pub fn solve(&mut self) -> Vec<Diagnostic> {
        let mut errors = Vec::new();
        // Process constraints in order (deterministic)
        let constraints = std::mem::take(&mut self.constraints);
        for (lhs, rhs, span) in constraints {
            let lhs_resolved = self.apply(&lhs);
            let rhs_resolved = self.apply(&rhs);
            match unify(&lhs_resolved, &rhs_resolved, &mut self.subst) {
                Ok(_) => {}
                Err(msg) => {
                    errors.push(
                        Diagnostic::error(
                            "E2004",
                            format!("cannot unify types: {}", msg),
                            span,
                        )
                        .with_hint(format!(
                            "expected `{}`, found `{}`",
                            lhs_resolved, rhs_resolved
                        )),
                    );
                }
            }
        }
        errors
    }

    /// Apply current substitution to a type, resolving all known variables.
    pub fn apply(&self, ty: &Type) -> Type {
        match ty {
            Type::Var(id) => {
                if let Some(bound) = self.subst.get(id) {
                    self.apply(bound)
                } else {
                    ty.clone()
                }
            }
            Type::Tuple(elems) => {
                Type::Tuple(elems.iter().map(|e| self.apply(e)).collect())
            }
            Type::Array { elem, len } => Type::Array {
                elem: Box::new(self.apply(elem)),
                len: *len,
            },
            Type::Tensor { elem, shape } => Type::Tensor {
                elem: Box::new(self.apply(elem)),
                shape: shape.clone(),
            },
            Type::Buffer { elem } => Type::Buffer {
                elem: Box::new(self.apply(elem)),
            },
            Type::Fn { params, ret } => Type::Fn {
                params: params.iter().map(|p| self.apply(p)).collect(),
                ret: Box::new(self.apply(ret)),
            },
            Type::Map { key, value } => Type::Map {
                key: Box::new(self.apply(key)),
                value: Box::new(self.apply(value)),
            },
            Type::Range { elem } => Type::Range {
                elem: Box::new(self.apply(elem)),
            },
            Type::Slice { elem } => Type::Slice {
                elem: Box::new(self.apply(elem)),
            },
            Type::SparseTensor { elem } => Type::SparseTensor {
                elem: Box::new(self.apply(elem)),
            },
            _ => ty.clone(),
        }
    }

    /// Resolve a type fully: apply substitution, and default unresolved
    /// type variables to their default types (i64 for int, f64 for float).
    pub fn resolve_final(&self, ty: &Type) -> Type {
        let applied = self.apply(ty);
        self.default_unresolved_vars(&applied)
    }

    /// Default unresolved type variables to sensible defaults.
    fn default_unresolved_vars(&self, ty: &Type) -> Type {
        match ty {
            // Unresolved type variable: default to i64 (most common)
            Type::Var(_) => Type::I64,
            Type::Tuple(elems) => {
                Type::Tuple(elems.iter().map(|e| self.default_unresolved_vars(e)).collect())
            }
            Type::Array { elem, len } => Type::Array {
                elem: Box::new(self.default_unresolved_vars(elem)),
                len: *len,
            },
            Type::Fn { params, ret } => Type::Fn {
                params: params.iter().map(|p| self.default_unresolved_vars(p)).collect(),
                ret: Box::new(self.default_unresolved_vars(ret)),
            },
            _ => ty.clone(),
        }
    }

    /// Get the current substitution (for inspection/debugging).
    pub fn substitution(&self) -> &TypeSubst {
        &self.subst
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fresh_var() {
        let mut ctx = InferCtx::new(0);
        let v1 = ctx.fresh_var();
        let v2 = ctx.fresh_var();
        assert_eq!(v1, Type::Var(TypeVarId(0)));
        assert_eq!(v2, Type::Var(TypeVarId(1)));
        assert_eq!(ctx.next_var_counter(), 2);
    }

    #[test]
    fn test_constrain_and_solve_success() {
        let mut ctx = InferCtx::new(0);
        let tv = ctx.fresh_var();
        ctx.constrain(tv.clone(), Type::I64, Span::new(0, 1));
        let errors = ctx.solve();
        assert!(errors.is_empty());
        assert_eq!(ctx.apply(&tv), Type::I64);
    }

    #[test]
    fn test_constrain_and_solve_conflict() {
        let mut ctx = InferCtx::new(0);
        let tv = ctx.fresh_var();
        ctx.constrain(tv.clone(), Type::I64, Span::new(0, 1));
        ctx.constrain(tv.clone(), Type::Str, Span::new(2, 3));
        let errors = ctx.solve();
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_apply_nested() {
        let mut ctx = InferCtx::new(0);
        let tv = ctx.fresh_var();
        ctx.constrain(
            Type::Tuple(vec![tv.clone()]),
            Type::Tuple(vec![Type::F64]),
            Span::new(0, 1),
        );
        let errors = ctx.solve();
        assert!(errors.is_empty());
        assert_eq!(ctx.apply(&tv), Type::F64);
    }

    #[test]
    fn test_resolve_final_defaults() {
        let ctx = InferCtx::new(0);
        // Unresolved variable defaults to i64
        assert_eq!(ctx.resolve_final(&Type::Var(TypeVarId(0))), Type::I64);
        // Known types pass through
        assert_eq!(ctx.resolve_final(&Type::F64), Type::F64);
    }

    #[test]
    fn test_transitive_constraints() {
        let mut ctx = InferCtx::new(0);
        let v1 = ctx.fresh_var();
        let v2 = ctx.fresh_var();
        ctx.constrain(v1.clone(), v2.clone(), Span::new(0, 1));
        ctx.constrain(v2.clone(), Type::Bool, Span::new(2, 3));
        let errors = ctx.solve();
        assert!(errors.is_empty());
        assert_eq!(ctx.apply(&v1), Type::Bool);
        assert_eq!(ctx.apply(&v2), Type::Bool);
    }
}
