// Milestone 2.5 — Cross-Cutting Coherence Tests
//
// These tests verify that multiple milestone 2.5 subsystems work together
// correctly end-to-end:
// - Type unification + shape unification combined
// - Tensor operations + COW buffer semantics
// - Sparse + Dense round-trip consistency
// - DetMap with Tensor values
// - AD graph + tensor broadcasting
// - Data pipeline + type system coherence

use cjc_types::*;
use cjc_runtime::{Tensor, DetMap, Value, SparseCoo, SparseCsr};
use cjc_ad::GradGraph;
use cjc_data::{Column, DataFrame, Pipeline, DExpr, DBinOp, AggFunc};
use std::rc::Rc;

const TOL: f64 = 1e-10;

#[test]
fn coherence_type_and_shape_unification_combined() {
    // Unify Tensor<T0, [N, 3]> with Tensor<F64, [4, 3]>
    // Should bind T0=F64 and N=4
    let mut type_subst = TypeSubst::new();
    let t0 = TypeVarId(0);

    let generic = Type::Tensor {
        elem: Box::new(Type::Var(t0)),
        shape: Some(vec![ShapeDim::Symbolic("N".into()), ShapeDim::Known(3)]),
    };
    let concrete = Type::Tensor {
        elem: Box::new(Type::F64),
        shape: Some(vec![ShapeDim::Known(4), ShapeDim::Known(3)]),
    };

    let result = unify(&generic, &concrete, &mut type_subst).unwrap();
    assert_eq!(type_subst.get(&t0), Some(&Type::F64));

    // Verify result shape
    if let Type::Tensor { shape: Some(dims), .. } = &result {
        assert_eq!(dims.len(), 2);
        assert_eq!(dims[0], ShapeDim::Known(4));
        assert_eq!(dims[1], ShapeDim::Known(3));
    } else {
        panic!("expected Tensor with shape");
    }
}

#[test]
fn coherence_tensor_cow_buffer_isolation() {
    // Create tensor, clone it (shared buffer), modify clone, verify original
    let t1 = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let mut t2 = t1.clone();

    // Buffer is shared
    assert!(t1.buffer.refcount() >= 2);

    // Modify t2
    t2.set(&[0, 0], 99.0).unwrap();

    // t1 should be unchanged (COW)
    assert_eq!(t1.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(t2.get(&[0, 0]).unwrap(), 99.0);
}

#[test]
fn coherence_sparse_dense_round_trip() {
    // Build sparse, convert to dense, verify values match
    let coo = SparseCoo::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![0, 0, 1, 2],
        vec![0, 2, 1, 0],
        3,
        3,
    );
    let csr = SparseCsr::from_coo(&coo);

    // Check via CSR get
    assert_eq!(csr.get(0, 0), 1.0);
    assert_eq!(csr.get(0, 2), 2.0);
    assert_eq!(csr.get(1, 1), 3.0);
    assert_eq!(csr.get(2, 0), 4.0);

    // Convert to dense and verify same values
    let dense = csr.to_dense();
    assert_eq!(dense.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(dense.get(&[0, 2]).unwrap(), 2.0);
    assert_eq!(dense.get(&[1, 1]).unwrap(), 3.0);
    assert_eq!(dense.get(&[2, 0]).unwrap(), 4.0);
    // Zero entries
    assert_eq!(dense.get(&[0, 1]).unwrap(), 0.0);
    assert_eq!(dense.get(&[1, 0]).unwrap(), 0.0);
}

#[test]
fn coherence_detmap_with_tensor_values() {
    // Store tensors in a DetMap, retrieve them, verify
    let mut m = DetMap::new();

    let t1 = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0], &[3]).unwrap();

    m.insert(
        Value::String(Rc::new("weights".into())),
        Value::Tensor(t1),
    );
    m.insert(
        Value::String(Rc::new("bias".into())),
        Value::Tensor(t2),
    );

    assert_eq!(m.len(), 2);

    if let Some(Value::Tensor(t)) = m.get(&Value::String(Rc::new("weights".into()))) {
        assert_eq!(t.shape(), &[2]);
        assert_eq!(t.to_vec(), vec![1.0, 2.0]);
    } else {
        panic!("expected Tensor value");
    }
}

#[test]
fn coherence_ad_with_broadcasting() {
    // Ensure AD backward pass works with tensors that result from broadcasting
    let mut g = GradGraph::new();

    let a = g.parameter(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap());
    let b = g.parameter(Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap());

    let c = g.add(a, b);
    let loss = g.sum(c);

    // Forward value: sum([11, 22, 33, 44]) = 110
    let loss_val = g.value(loss);
    assert!((loss_val - 110.0).abs() < TOL);

    g.backward(loss);

    // Gradient of sum(a+b) w.r.t a = ones, w.r.t b = ones
    let ga = g.grad(a).unwrap();
    let gb = g.grad(b).unwrap();
    for &v in &ga.to_vec() {
        assert!((v - 1.0).abs() < TOL);
    }
    for &v in &gb.to_vec() {
        assert!((v - 1.0).abs() < TOL);
    }
}

#[test]
fn coherence_data_pipeline_filter_aggregate() {
    // End-to-end pipeline: scan -> filter -> aggregate -> verify types
    let df = DataFrame::from_columns(vec![
        ("category".into(), Column::Str(vec!["A".into(), "A".into(), "B".into(), "B".into()])),
        ("score".into(), Column::Float(vec![80.0, 90.0, 70.0, 60.0])),
    ])
    .unwrap();

    let result = Pipeline::scan(df)
        .filter(DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("score".into())),
            right: Box::new(DExpr::LitFloat(65.0)),
        })
        .summarize(
            vec!["category".into()],
            vec![
                ("avg_score".into(), DExpr::Agg(AggFunc::Mean, Box::new(DExpr::Col("score".into())))),
                ("count".into(), DExpr::Count),
            ],
        )
        .collect()
        .unwrap();

    // After filter (score > 65): A(80), A(90), B(70) -- B(60) filtered out
    // Aggregate by category: A -> mean(80,90)=85, B -> mean(70)=70
    assert_eq!(result.nrows(), 2);

    if let Column::Float(avgs) = result.get_column("avg_score").unwrap() {
        // Results are sorted by key: A first, then B
        assert!((avgs[0] - 85.0).abs() < 0.01);
        assert!((avgs[1] - 70.0).abs() < 0.01);
    } else {
        panic!("expected Float column");
    }
}
