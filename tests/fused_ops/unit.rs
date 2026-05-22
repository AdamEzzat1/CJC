//! GC-06 Phase 3a — unit tests for fused elementwise ops: explicit values,
//! the non-contiguous fallback path, and shape-mismatch errors.

use cjc_runtime::tensor::Tensor;

fn t(data: &[f64]) -> Tensor {
    Tensor::from_vec(data.to_vec(), &[data.len()]).unwrap()
}

#[test]
fn fused_axpy_known_values() {
    // 2 * [1,2,3] + [10,20,30] = [12,24,36]
    let r = t(&[1.0, 2.0, 3.0]).fused_axpy(2.0, &t(&[10.0, 20.0, 30.0])).unwrap();
    assert_eq!(r.to_vec(), vec![12.0, 24.0, 36.0]);
}

#[test]
fn fused_mul_sub_known_values() {
    // [2,3]*[4,5] - [1,1] = [8,15] - [1,1] = [7,14]
    let r = t(&[2.0, 3.0])
        .fused_mul_sub(&t(&[4.0, 5.0]), &t(&[1.0, 1.0]))
        .unwrap();
    assert_eq!(r.to_vec(), vec![7.0, 14.0]);
}

#[test]
fn fused_sub_sq_known_values() {
    // ([5,3] - [1,1])^2 = [4,2]^2 = [16,4]
    let r = t(&[5.0, 3.0]).fused_sub_sq(&t(&[1.0, 1.0])).unwrap();
    assert_eq!(r.to_vec(), vec![16.0, 4.0]);
}

#[test]
fn fused_ops_shape_mismatch_errors() {
    let a = t(&[1.0, 2.0, 3.0]);
    let b = t(&[1.0, 2.0]); // wrong length
    let c = t(&[1.0, 2.0, 3.0]);
    assert!(a.fused_axpy(2.0, &b).is_err());
    assert!(a.fused_mul_sub(&b, &c).is_err());
    assert!(a.fused_sub_sq(&b).is_err());
}

#[test]
fn fused_noncontiguous_matches_unfused() {
    // A transposed tensor exercises whatever path (incl. the non-contiguous
    // fallback); fused must still equal unfused.
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .unwrap()
        .transpose(); // [3,2]
    let b = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3])
        .unwrap()
        .transpose();
    let c = Tensor::from_vec(vec![1.0; 6], &[3, 2]).unwrap();

    assert_eq!(
        a.fused_mul_sub(&b, &c).unwrap().to_vec(),
        a.mul_elem(&b).unwrap().sub(&c).unwrap().to_vec(),
    );
    let d = a.sub(&b).unwrap();
    assert_eq!(
        a.fused_sub_sq(&b).unwrap().to_vec(),
        d.mul_elem(&d).unwrap().to_vec(),
    );
    assert_eq!(
        a.fused_axpy(3.0, &b).unwrap().to_vec(),
        a.scalar_mul(3.0).add(&b).unwrap().to_vec(),
    );
}
