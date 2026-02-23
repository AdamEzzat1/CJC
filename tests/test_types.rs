// CJC Test Suite — cjc-types (7 tests)
// Source: crates/cjc-types/src/lib.rs
// These tests are extracted from the inline #[cfg(test)] modules for regression tracking.

use cjc_types::*;

#[test]
fn test_builtin_types() {
    let env = TypeEnv::new();
    assert_eq!(env.resolve_type_name("i32"), Some(Type::I32));
    assert_eq!(env.resolve_type_name("f64"), Some(Type::F64));
    assert_eq!(env.resolve_type_name("bool"), Some(Type::Bool));
}

#[test]
fn test_trait_satisfaction() {
    let env = TypeEnv::new();
    assert!(env.satisfies_trait(&Type::I32, "Numeric"));
    assert!(env.satisfies_trait(&Type::I32, "Int"));
    assert!(!env.satisfies_trait(&Type::I32, "Float"));

    assert!(env.satisfies_trait(&Type::F64, "Numeric"));
    assert!(env.satisfies_trait(&Type::F64, "Float"));
    assert!(env.satisfies_trait(&Type::F64, "Differentiable"));
    assert!(!env.satisfies_trait(&Type::F64, "Int"));

    assert!(!env.satisfies_trait(&Type::Bool, "Numeric"));
}

#[test]
fn test_type_display() {
    assert_eq!(format!("{}", Type::I32), "i32");
    assert_eq!(format!("{}", Type::F64), "f64");
    assert_eq!(
        format!(
            "{}",
            Type::Tensor {
                elem: Box::new(Type::F32),
                shape: Some(vec![ShapeDim::Known(3), ShapeDim::Known(4)])
            }
        ),
        "Tensor<f32, [3, 4]>"
    );
}

#[test]
fn test_matmul_shape_check() {
    let env = TypeEnv::new();

    // Valid: [3, 4] x [4, 5] -> [3, 5]
    let result = env.check_matmul_shapes(
        &Some(vec![ShapeDim::Known(3), ShapeDim::Known(4)]),
        &Some(vec![ShapeDim::Known(4), ShapeDim::Known(5)]),
    );
    assert!(result.is_ok());
    let shape = result.unwrap().unwrap();
    assert_eq!(shape, vec![ShapeDim::Known(3), ShapeDim::Known(5)]);

    // Invalid: [3, 4] x [5, 6] -> error
    let result = env.check_matmul_shapes(
        &Some(vec![ShapeDim::Known(3), ShapeDim::Known(4)]),
        &Some(vec![ShapeDim::Known(5), ShapeDim::Known(6)]),
    );
    assert!(result.is_err());
}

#[test]
fn test_scope() {
    let mut env = TypeEnv::new();
    env.define_var("x", Type::I32);
    assert_eq!(env.lookup_var("x"), Some(&Type::I32));

    env.push_scope();
    env.define_var("y", Type::F64);
    assert_eq!(env.lookup_var("x"), Some(&Type::I32));
    assert_eq!(env.lookup_var("y"), Some(&Type::F64));

    env.pop_scope();
    assert_eq!(env.lookup_var("x"), Some(&Type::I32));
    assert_eq!(env.lookup_var("y"), None);
}

#[test]
fn test_types_match() {
    let env = TypeEnv::new();
    assert!(env.types_match(&Type::I32, &Type::I32));
    assert!(!env.types_match(&Type::I32, &Type::I64));
    assert!(env.types_match(&Type::Error, &Type::I32)); // Error matches everything
}

#[test]
fn test_value_vs_gc_type() {
    assert!(Type::I32.is_value_type());
    assert!(Type::F64.is_value_type());
    assert!(Type::Struct(StructType {
        name: "Foo".into(),
        type_params: vec![],
        fields: vec![]
    })
    .is_value_type());

    assert!(Type::Class(ClassType {
        name: "Bar".into(),
        type_params: vec![],
        fields: vec![]
    })
    .is_gc_type());

    assert!(!Type::I32.is_gc_type());
    assert!(!Type::Class(ClassType {
        name: "Bar".into(),
        type_params: vec![],
        fields: vec![]
    })
    .is_value_type());
}
