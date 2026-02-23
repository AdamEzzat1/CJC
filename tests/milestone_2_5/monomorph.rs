// Milestone 2.5 — Monomorphization Tests
//
// Tests for type-level monomorphization: verifying that generic function
// signatures can be resolved to concrete types via unification and
// substitution. This is the compile-time counterpart to runtime dispatch.
//
// - Generic identity function: fn id<T>(x: T) -> T
// - Generic pair: fn pair<A, B>(a: A, b: B) -> (A, B)
// - Bounded generics: fn add<T: Numeric>(a: T, b: T) -> T
// - Nested generics: fn wrap<T>(x: T) -> Tensor<T>
// - Multiple instantiations of the same generic

use cjc_types::*;

#[test]
fn mono_identity_function_resolves() {
    // fn id<T>(x: T) -> T, called with I32 => returns I32
    let mut subst = TypeSubst::new();
    let t0 = TypeVarId(0);

    let param_type = Type::Var(t0);
    let ret_type = Type::Var(t0);
    let arg_type = Type::I32;

    // Unify parameter with argument
    unify(&param_type, &arg_type, &mut subst).unwrap();

    // Apply substitution to return type
    let resolved_ret = apply_subst(&ret_type, &subst);
    assert_eq!(resolved_ret, Type::I32);
}

#[test]
fn mono_pair_function_two_type_params() {
    // fn pair<A, B>(a: A, b: B) -> Tuple(A, B)
    // called with (F64, Bool)
    let mut subst = TypeSubst::new();
    let a = TypeVarId(0);
    let b = TypeVarId(1);

    unify(&Type::Var(a), &Type::F64, &mut subst).unwrap();
    unify(&Type::Var(b), &Type::Bool, &mut subst).unwrap();

    let ret_type = Type::Tuple(vec![Type::Var(a), Type::Var(b)]);
    let resolved = apply_subst(&ret_type, &subst);

    assert_eq!(resolved, Type::Tuple(vec![Type::F64, Type::Bool]));
}

#[test]
fn mono_bounded_generic_satisfies_numeric() {
    // fn add<T: Numeric>(a: T, b: T) -> T
    // Called with I32 -- should satisfy Numeric bound
    let env = TypeEnv::new();
    let mut subst = TypeSubst::new();
    let t0 = TypeVarId(0);

    unify(&Type::Var(t0), &Type::I32, &mut subst).unwrap();

    let inferred = apply_subst(&Type::Var(t0), &subst);
    assert!(
        env.check_bounds(&inferred, &["Numeric".into()]),
        "I32 must satisfy Numeric"
    );
}

#[test]
fn mono_bounded_generic_rejects_bool() {
    // fn add<T: Numeric>(a: T, b: T) -> T
    // Called with Bool -- should NOT satisfy Numeric bound
    let env = TypeEnv::new();
    let mut subst = TypeSubst::new();
    let t0 = TypeVarId(0);

    unify(&Type::Var(t0), &Type::Bool, &mut subst).unwrap();

    let inferred = apply_subst(&Type::Var(t0), &subst);
    assert!(
        !env.check_bounds(&inferred, &["Numeric".into()]),
        "Bool must NOT satisfy Numeric"
    );
}

#[test]
fn mono_nested_generic_tensor_elem() {
    // fn wrap<T>(x: T) -> Tensor<T>
    // Called with F32 => Tensor<F32>
    let mut subst = TypeSubst::new();
    let t0 = TypeVarId(0);

    unify(&Type::Var(t0), &Type::F32, &mut subst).unwrap();

    let ret_type = Type::Tensor {
        elem: Box::new(Type::Var(t0)),
        shape: None,
    };
    let resolved = apply_subst(&ret_type, &subst);

    assert_eq!(
        resolved,
        Type::Tensor {
            elem: Box::new(Type::F32),
            shape: None,
        }
    );
}

#[test]
fn mono_same_generic_two_instantiations() {
    // First call: id<I32>(x) => T0 = I32
    let mut subst1 = TypeSubst::new();
    let t0 = TypeVarId(0);
    unify(&Type::Var(t0), &Type::I32, &mut subst1).unwrap();
    assert_eq!(apply_subst(&Type::Var(t0), &subst1), Type::I32);

    // Second call with fresh var: id<F64>(y) => T1 = F64
    let mut subst2 = TypeSubst::new();
    let t1 = TypeVarId(1);
    unify(&Type::Var(t1), &Type::F64, &mut subst2).unwrap();
    assert_eq!(apply_subst(&Type::Var(t1), &subst2), Type::F64);

    // The two substitutions are independent
    assert!(subst1.get(&t1).is_none());
    assert!(subst2.get(&t0).is_none());
}
