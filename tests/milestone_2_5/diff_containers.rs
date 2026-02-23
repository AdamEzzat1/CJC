// Milestone 2.5 — Differentiable Container Tests
//
// Tests for the AD engine's ability to propagate gradients through
// structured containers (struct fields, map lookups):
// - GradOp::StructField gradient flow
// - GradOp::MapLookup gradient flow
// - Combined forward + backward through containers
// - Gradient accumulation from multiple container accesses

use cjc_ad::{GradGraph, GradOp, GradNode};
use cjc_runtime::Tensor;
use std::cell::RefCell;
use std::rc::Rc;

const TOL: f64 = 1e-10;

#[test]
fn diff_struct_field_gradient_flows_to_parent() {
    // Simulate: loss = struct.field[0] where struct is a parameter tensor
    let mut g = GradGraph::new();

    // Parent parameter (represents a struct with tensor data)
    let parent = g.parameter(Tensor::from_vec(vec![3.0, 7.0], &[2]).unwrap());

    // Create a "field access" node — records that this came from parent
    let field_tensor = Tensor::from_vec(vec![3.0], &[1]).unwrap();
    let field_idx = g.nodes.len();
    g.nodes.push(Rc::new(RefCell::new(GradNode {
        op: GradOp::StructField {
            parent,
            field_index: 0,
            total_fields: 2,
        },
        tensor: field_tensor,
        grad: None,
    })));

    // Sum the field value to get a scalar loss
    let loss = g.sum(field_idx);
    g.backward(loss);

    // The gradient should flow back to the parent parameter
    let grad = g.grad(parent).unwrap();
    assert_eq!(grad.to_vec().len(), 2);
    // Gradient accumulates from the field access
    for &v in &grad.to_vec() {
        assert!(v.is_finite());
    }
}

#[test]
fn diff_map_lookup_gradient_flows_to_map_node() {
    // Simulate: loss = map["key"] where map is a parameter tensor
    let mut g = GradGraph::new();

    let map_node = g.parameter(Tensor::from_vec(vec![5.0, 10.0, 15.0], &[3]).unwrap());

    // Create a "map lookup" node for key_index=1
    let lookup_tensor = Tensor::from_vec(vec![10.0], &[1]).unwrap();
    let lookup_idx = g.nodes.len();
    g.nodes.push(Rc::new(RefCell::new(GradNode {
        op: GradOp::MapLookup {
            map_node,
            key_index: 1,
            total_keys: 3,
        },
        tensor: lookup_tensor,
        grad: None,
    })));

    let loss = g.sum(lookup_idx);
    g.backward(loss);

    // Gradient should flow to the map node
    let grad = g.grad(map_node).unwrap();
    assert_eq!(grad.to_vec().len(), 3);
    for &v in &grad.to_vec() {
        assert!(v.is_finite());
    }
}

#[test]
fn diff_reverse_mode_add_gradient_accumulation() {
    // loss = a + b + a (a is used twice, gradient should accumulate)
    let mut g = GradGraph::new();

    let a = g.parameter(Tensor::from_vec(vec![2.0], &[1]).unwrap());
    let b = g.parameter(Tensor::from_vec(vec![3.0], &[1]).unwrap());

    let ab = g.add(a, b);
    let aba = g.add(ab, a);
    let loss = g.sum(aba);

    g.backward(loss);

    let ga = g.grad(a).unwrap();
    let gb = g.grad(b).unwrap();

    // d(loss)/da = 2 (a appears twice in a + b + a)
    assert!((ga.to_vec()[0] - 2.0).abs() < TOL);
    // d(loss)/db = 1
    assert!((gb.to_vec()[0] - 1.0).abs() < TOL);
}

#[test]
fn diff_zero_grad_resets() {
    let mut g = GradGraph::new();
    let a = g.parameter(Tensor::from_vec(vec![1.0], &[1]).unwrap());
    let loss = g.sum(a);

    g.backward(loss);
    let ga = g.grad(a).unwrap();
    assert!((ga.to_vec()[0] - 1.0).abs() < TOL);

    // Zero out gradients
    g.zero_grad();
    let ga2 = g.grad(a).unwrap();
    assert!((ga2.to_vec()[0] - 0.0).abs() < TOL);
}
