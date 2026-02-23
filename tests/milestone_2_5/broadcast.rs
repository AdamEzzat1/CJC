// Milestone 2.5 — Runtime Broadcasting Tests
//
// Tests for NumPy-style element-wise broadcasting in the Tensor runtime:
// - Same-shape operations (fast path)
// - Scalar broadcast ([1] + [3, 4])
// - Row vector broadcast ([1, N] + [M, N])
// - Column vector broadcast ([M, 1] + [M, N])
// - Broadcasting in arithmetic ops (add, sub, mul_elem, div_elem)
// - Incompatible broadcast error

use cjc_runtime::Tensor;

#[test]
fn bcast_same_shape_fast_path() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();

    let c = a.add(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.to_vec(), vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn bcast_scalar_to_matrix() {
    // [1] + [2, 3] => broadcast scalar to every element
    let scalar = Tensor::from_vec(vec![100.0], &[1]).unwrap();
    let matrix = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

    let result = matrix.add(&scalar).unwrap();
    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(
        result.to_vec(),
        vec![101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
    );
}

#[test]
fn bcast_row_vector_to_matrix() {
    // [1, 3] + [2, 3] => row broadcast
    let row = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
    let mat = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3]).unwrap();

    let result = mat.add(&row).unwrap();
    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(
        result.to_vec(),
        vec![11.0, 22.0, 33.0, 41.0, 52.0, 63.0]
    );
}

#[test]
fn bcast_column_vector_to_matrix() {
    // [2, 1] + [2, 3] => column broadcast
    let col = Tensor::from_vec(vec![100.0, 200.0], &[2, 1]).unwrap();
    let mat = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

    let result = mat.add(&col).unwrap();
    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(
        result.to_vec(),
        vec![101.0, 102.0, 103.0, 204.0, 205.0, 206.0]
    );
}

#[test]
fn bcast_mul_elem_broadcast() {
    // Element-wise multiply with broadcasting: [3] * [2, 3]
    let scale = Tensor::from_vec(vec![2.0, 3.0, 4.0], &[3]).unwrap();
    let mat = Tensor::from_vec(vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0], &[2, 3]).unwrap();

    let result = mat.mul_elem(&scale).unwrap();
    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(result.to_vec(), vec![2.0, 3.0, 4.0, 4.0, 6.0, 8.0]);
}

#[test]
fn bcast_incompatible_shapes_fail() {
    // [2, 3] + [2, 4] => incompatible
    let a = Tensor::zeros(&[2, 3]);
    let b = Tensor::zeros(&[2, 4]);

    assert!(a.add(&b).is_err(), "incompatible shapes should fail");
}
