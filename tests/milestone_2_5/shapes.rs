// Milestone 2.5 — Shape Unification + Broadcasting Tests
//
// Tests for compile-time shape analysis in cjc_types:
// - unify_shape_dim: Known/Known, Symbolic/Known, Symbolic/Symbolic
// - unify_shapes: rank checking, multi-dim unification
// - broadcast_shapes: NumPy-style broadcasting rules
// - TypeEnv::check_matmul_shapes: inner-dimension compatibility

use cjc_types::*;

// ── Shape Dim Unification ───────────────────────────────────────

#[test]
fn shape_unify_symbolic_to_known_binds() {
    let mut subst = ShapeSubst::new();
    let result = unify_shape_dim(
        &ShapeDim::Symbolic("N".into()),
        &ShapeDim::Known(10),
        &mut subst,
    )
    .unwrap();

    assert_eq!(result, ShapeDim::Known(10));
    assert_eq!(subst.get("N"), Some(&10));
}

#[test]
fn shape_unify_symbolic_already_bound_consistent() {
    let mut subst = ShapeSubst::new();
    subst.insert("N".into(), 7);

    // N is already bound to 7, unifying with 7 should succeed
    let result = unify_shape_dim(
        &ShapeDim::Symbolic("N".into()),
        &ShapeDim::Known(7),
        &mut subst,
    )
    .unwrap();
    assert_eq!(result, ShapeDim::Known(7));
}

#[test]
fn shape_unify_symbolic_already_bound_conflict() {
    let mut subst = ShapeSubst::new();
    subst.insert("M".into(), 5);

    // M is bound to 5, unifying with 8 should fail
    let result = unify_shape_dim(
        &ShapeDim::Symbolic("M".into()),
        &ShapeDim::Known(8),
        &mut subst,
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("already bound"));
}

#[test]
fn shape_unify_two_symbolics_same_name() {
    let mut subst = ShapeSubst::new();
    let result = unify_shape_dim(
        &ShapeDim::Symbolic("N".into()),
        &ShapeDim::Symbolic("N".into()),
        &mut subst,
    )
    .unwrap();
    assert_eq!(result, ShapeDim::Symbolic("N".into()));
}

#[test]
fn shape_unify_shapes_multi_dim() {
    // [N, 3] ~ [4, 3] => N=4, result = [4, 3]
    let mut subst = ShapeSubst::new();
    let a = vec![ShapeDim::Symbolic("N".into()), ShapeDim::Known(3)];
    let b = vec![ShapeDim::Known(4), ShapeDim::Known(3)];

    let result = unify_shapes(&a, &b, &mut subst).unwrap();
    assert_eq!(result, vec![ShapeDim::Known(4), ShapeDim::Known(3)]);
    assert_eq!(subst.get("N"), Some(&4));
}

#[test]
fn shape_broadcast_different_ranks() {
    // [4] broadcast with [3, 4] => [3, 4]
    let a = Some(vec![ShapeDim::Known(4)]);
    let b = Some(vec![ShapeDim::Known(3), ShapeDim::Known(4)]);
    let result = broadcast_shapes(&a, &b).unwrap().unwrap();
    assert_eq!(result, vec![ShapeDim::Known(3), ShapeDim::Known(4)]);
}
