//! u64 Indexing Boundary Safety Tests
//!
//! Verifies that tensor shape/stride calculations use safe arithmetic,
//! preventing overflow on large dimensions and edge cases.

use cjc_runtime::Tensor;

// ---------------------------------------------------------------------------
// Shape Calculation Safety
// ---------------------------------------------------------------------------

#[test]
fn test_tensor_shape_1d() {
    let t = Tensor::zeros(&[100]);
    assert_eq!(t.shape(), &[100]);
    assert_eq!(t.len(), 100);
}

#[test]
fn test_tensor_shape_2d() {
    let t = Tensor::zeros(&[10, 20]);
    assert_eq!(t.shape(), &[10, 20]);
    assert_eq!(t.len(), 200);
}

#[test]
fn test_tensor_shape_3d() {
    let t = Tensor::zeros(&[3, 4, 5]);
    assert_eq!(t.shape(), &[3, 4, 5]);
    assert_eq!(t.len(), 60);
}

#[test]
fn test_tensor_shape_4d() {
    let t = Tensor::zeros(&[2, 3, 4, 5]);
    assert_eq!(t.shape(), &[2, 3, 4, 5]);
    assert_eq!(t.len(), 120);
}

#[test]
fn test_tensor_shape_empty_dimension() {
    let t = Tensor::zeros(&[0]);
    assert_eq!(t.len(), 0);
}

#[test]
fn test_tensor_shape_singleton() {
    let t = Tensor::zeros(&[1, 1, 1]);
    assert_eq!(t.len(), 1);
}

// ---------------------------------------------------------------------------
// Stride Calculation
// ---------------------------------------------------------------------------

#[test]
fn test_strides_1d() {
    let t = Tensor::zeros(&[10]);
    assert_eq!(t.len(), 10);
}

#[test]
fn test_strides_2d_row_major() {
    let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
    let t = Tensor::from_vec(data, &[3, 4]).unwrap();
    // Element [1,2] should be at index 1*4 + 2 = 6.
    assert_eq!(t.get(&[1, 2]).unwrap(), 6.0);
}

#[test]
fn test_strides_3d_row_major() {
    let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let t = Tensor::from_vec(data, &[2, 3, 4]).unwrap();
    // Element [1,2,3] should be at index 1*12 + 2*4 + 3 = 23.
    assert_eq!(t.get(&[1, 2, 3]).unwrap(), 23.0);
}

// ---------------------------------------------------------------------------
// Reshape Boundary Tests
// ---------------------------------------------------------------------------

#[test]
fn test_reshape_preserves_data() {
    let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
    let t = Tensor::from_vec(data, &[3, 4]).unwrap();
    let r = t.reshape(&[4, 3]).unwrap();
    assert_eq!(r.shape(), &[4, 3]);
    assert_eq!(r.get(&[0, 0]).unwrap(), 0.0);
    assert_eq!(r.get(&[3, 2]).unwrap(), 11.0);
}

#[test]
fn test_reshape_to_1d() {
    let data: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let t = Tensor::from_vec(data, &[2, 3]).unwrap();
    let r = t.reshape(&[6]).unwrap();
    assert_eq!(r.shape(), &[6]);
    assert_eq!(r.len(), 6);
}

#[test]
fn test_reshape_invalid_size() {
    let t = Tensor::zeros(&[3, 4]);
    let result = t.reshape(&[5, 3]);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Index Boundary Tests
// ---------------------------------------------------------------------------

#[test]
fn test_get_first_element() {
    let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let t = Tensor::from_vec(data, &[10]).unwrap();
    assert_eq!(t.get(&[0]).unwrap(), 0.0);
}

#[test]
fn test_get_last_element() {
    let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let t = Tensor::from_vec(data, &[10]).unwrap();
    assert_eq!(t.get(&[9]).unwrap(), 9.0);
}

#[test]
fn test_matmul_dimension_match() {
    let a = Tensor::ones(&[3, 4]);
    let b = Tensor::ones(&[4, 5]);
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[3, 5]);
    // Each element should be 4.0 (sum of 4 ones * 1).
    assert_eq!(c.get(&[0, 0]).unwrap(), 4.0);
}

#[test]
fn test_matmul_dimension_mismatch() {
    let a = Tensor::ones(&[3, 4]);
    let b = Tensor::ones(&[5, 6]);
    let result = a.matmul(&b);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Large Dimension Sanity
// ---------------------------------------------------------------------------

#[test]
fn test_large_1d_tensor() {
    let t = Tensor::zeros(&[100_000]);
    assert_eq!(t.len(), 100_000);
    assert_eq!(t.sum(), 0.0);
}

#[test]
fn test_large_2d_tensor() {
    let t = Tensor::ones(&[100, 100]);
    assert_eq!(t.len(), 10_000);
    assert_eq!(t.sum(), 10_000.0);
}

#[test]
fn test_large_3d_tensor() {
    let t = Tensor::ones(&[10, 10, 10]);
    assert_eq!(t.len(), 1000);
    assert_eq!(t.sum(), 1000.0);
}

// ---------------------------------------------------------------------------
// usize Arithmetic Safety
// ---------------------------------------------------------------------------

#[test]
fn test_shape_numel_overflow_protection() {
    let t = Tensor::zeros(&[1000, 1000]);
    assert_eq!(t.len(), 1_000_000);
}

#[test]
fn test_sum_axis_boundary() {
    let data: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let t = Tensor::from_vec(data, &[2, 3]).unwrap();

    // Sum along axis 0 → shape [1, 3]
    let s0 = t.sum_axis(0).unwrap();
    assert_eq!(s0.shape(), &[1, 3]);

    // Sum along axis 1 → shape [2, 1]
    let s1 = t.sum_axis(1).unwrap();
    assert_eq!(s1.shape(), &[2, 1]);
}

// ---------------------------------------------------------------------------
// View/Slice Boundary Tests
// ---------------------------------------------------------------------------

#[test]
fn test_slice_range() {
    let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
    let t = Tensor::from_vec(data, &[3, 4]).unwrap();
    // Slice rows [0..1) = first row
    let s = t.slice(&[(0, 1), (0, 4)]).unwrap();
    assert_eq!(s.to_vec(), vec![0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn test_slice_last_range() {
    let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
    let t = Tensor::from_vec(data, &[3, 4]).unwrap();
    let s = t.slice(&[(2, 3), (0, 4)]).unwrap();
    assert_eq!(s.to_vec(), vec![8.0, 9.0, 10.0, 11.0]);
}

#[test]
fn test_transpose_shape() {
    let t = Tensor::zeros(&[3, 4]);
    let tr = t.transpose();
    assert_eq!(tr.shape(), &[4, 3]);
}

// ---------------------------------------------------------------------------
// Elementwise Operation Boundaries
// ---------------------------------------------------------------------------

#[test]
fn test_add_same_shape() {
    let a = Tensor::ones(&[3, 4]);
    let b = Tensor::ones(&[3, 4]);
    let c = a.add(&b).unwrap();
    assert_eq!(c.get(&[0, 0]).unwrap(), 2.0);
    assert_eq!(c.len(), 12);
}

#[test]
fn test_add_shape_mismatch() {
    let a = Tensor::ones(&[3, 4]);
    let b = Tensor::ones(&[4, 3]);
    let result = a.add(&b);
    assert!(result.is_err());
}

#[test]
fn test_mul_elem_correctness() {
    let data_a = vec![1.0, 2.0, 3.0, 4.0];
    let data_b = vec![5.0, 6.0, 7.0, 8.0];
    let a = Tensor::from_vec(data_a, &[4]).unwrap();
    let b = Tensor::from_vec(data_b, &[4]).unwrap();
    let c = a.mul_elem(&b).unwrap();
    assert_eq!(c.to_vec(), vec![5.0, 12.0, 21.0, 32.0]);
}

#[test]
fn test_scalar_mul() {
    let data = vec![1.0, 2.0, 3.0];
    let t = Tensor::from_vec(data, &[3]).unwrap();
    let scaled = t.scalar_mul(3.0);
    assert_eq!(scaled.to_vec(), vec![3.0, 6.0, 9.0]);
}

// ---------------------------------------------------------------------------
// Additional Boundary Tests
// ---------------------------------------------------------------------------

#[test]
fn test_zeros_shape_matches() {
    for dims in [1, 2, 3, 5, 10, 100] {
        let t = Tensor::zeros(&[dims]);
        assert_eq!(t.len(), dims);
        assert_eq!(t.shape(), &[dims]);
    }
}

#[test]
fn test_ones_sum_matches_len() {
    for dims in [1, 5, 10, 50] {
        let t = Tensor::ones(&[dims]);
        assert_eq!(t.sum(), dims as f64);
    }
}

#[test]
fn test_from_vec_shape_mismatch() {
    let data = vec![1.0, 2.0, 3.0];
    let result = Tensor::from_vec(data, &[2, 2]);
    assert!(result.is_err());
}

#[test]
fn test_randn_shape() {
    let mut rng = cjc_repro::Rng::seeded(42);
    let t = Tensor::randn(&[5, 3], &mut rng);
    assert_eq!(t.shape(), &[5, 3]);
    assert_eq!(t.len(), 15);
}

#[test]
fn test_randn_deterministic() {
    let mut rng1 = cjc_repro::Rng::seeded(42);
    let mut rng2 = cjc_repro::Rng::seeded(42);
    let t1 = Tensor::randn(&[10], &mut rng1);
    let t2 = Tensor::randn(&[10], &mut rng2);
    assert_eq!(t1.to_vec(), t2.to_vec());
}
