// Milestone 2.5 — Type Unification Tests
//
// Tests for the Hindley-Milner style type unification engine in cjc_types:
// - Concrete type unification
// - Type variable binding and propagation
// - Nested generic types (Tensor<T>, Fn<T>)
// - Occurs check (infinite type prevention)
// - apply_subst with chained substitutions
// - TypeEnv::check_bounds for trait-bounded generics

use cjc_types::*;

// ── Concrete + Variable Unification ─────────────────────────────

#[test]
fn unify_two_vars_bind_transitively() {
    // T0 ~ T1, then T1 ~ F64 => T0 should resolve to F64
    let mut subst = TypeSubst::new();
    let v0 = TypeVarId(0);
    let v1 = TypeVarId(1);

    unify(&Type::Var(v0), &Type::Var(v1), &mut subst).unwrap();
    unify(&Type::Var(v1), &Type::F64, &mut subst).unwrap();

    let resolved = apply_subst(&Type::Var(v0), &subst);
    assert_eq!(resolved, Type::F64, "transitive variable binding failed");
}

#[test]
fn unify_tensor_generic_elem_binds_var() {
    // Tensor<T0> ~ Tensor<I32> => T0 = I32
    let mut subst = TypeSubst::new();
    let v0 = TypeVarId(0);

    let generic_tensor = Type::Tensor {
        elem: Box::new(Type::Var(v0)),
        shape: None,
    };
    let concrete_tensor = Type::Tensor {
        elem: Box::new(Type::I32),
        shape: None,
    };

    let result = unify(&generic_tensor, &concrete_tensor, &mut subst).unwrap();
    assert_eq!(
        result,
        Type::Tensor {
            elem: Box::new(Type::I32),
            shape: None,
        }
    );
    assert_eq!(subst.get(&v0), Some(&Type::I32));
}

#[test]
fn unify_fn_type_binds_param_and_ret() {
    // fn(T0) -> T0 ~ fn(Bool) -> Bool => T0 = Bool
    let mut subst = TypeSubst::new();
    let v0 = TypeVarId(0);

    let generic_fn = Type::Fn {
        params: vec![Type::Var(v0)],
        ret: Box::new(Type::Var(v0)),
    };
    let concrete_fn = Type::Fn {
        params: vec![Type::Bool],
        ret: Box::new(Type::Bool),
    };

    let result = unify(&generic_fn, &concrete_fn, &mut subst).unwrap();
    assert_eq!(
        result,
        Type::Fn {
            params: vec![Type::Bool],
            ret: Box::new(Type::Bool),
        }
    );
    assert_eq!(subst.get(&v0), Some(&Type::Bool));
}

#[test]
fn unify_occurs_check_prevents_infinite_type() {
    // T0 ~ Tensor<T0> should fail (infinite type)
    let mut subst = TypeSubst::new();
    let v0 = TypeVarId(0);

    let infinite = Type::Tensor {
        elem: Box::new(Type::Var(v0)),
        shape: None,
    };
    let result = unify(&Type::Var(v0), &infinite, &mut subst);
    assert!(result.is_err(), "occurs check should reject infinite type");
    assert!(
        result.unwrap_err().contains("infinite type"),
        "error message should mention infinite type"
    );
}

#[test]
fn unify_array_length_mismatch_fails() {
    // Array<I32, 3> ~ Array<I32, 5> => error
    let mut subst = TypeSubst::new();

    let a = Type::Array {
        elem: Box::new(Type::I32),
        len: 3,
    };
    let b = Type::Array {
        elem: Box::new(Type::I32),
        len: 5,
    };
    let result = unify(&a, &b, &mut subst);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("array length mismatch"));
}

#[test]
fn unify_check_bounds_multi_trait() {
    // F64 satisfies [Numeric, Float, Differentiable]
    // I32 does NOT satisfy [Float]
    let env = TypeEnv::new();

    assert!(env.check_bounds(
        &Type::F64,
        &["Numeric".into(), "Float".into(), "Differentiable".into()]
    ));
    assert!(!env.check_bounds(&Type::I32, &["Float".into()]));
    assert!(env.check_bounds(&Type::I32, &["Numeric".into(), "Int".into()]));
    assert!(!env.check_bounds(&Type::Bool, &["Numeric".into()]));
}
