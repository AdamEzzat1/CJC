// CJC Test Suite — cjc-dispatch (8 tests)
// Source: crates/cjc-dispatch/src/lib.rs
// These tests are extracted from the inline #[cfg(test)] modules for regression tracking.

use cjc_dispatch::*;
use cjc_types::*;
use cjc_diag::Span;

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
