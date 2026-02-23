// Milestone 2.5 — Zero-Copy View Tests
//
// Tests for Tensor operations that create views (shared Buffer, different
// strides/offset) without copying data:
// - slice: creates a sub-tensor view
// - transpose: swaps strides for 2D tensors
// - reshape: shares buffer for contiguous tensors
// - broadcast_to: creates virtual expanded view with stride=0
// - is_contiguous / to_contiguous: materializes non-contiguous views

use cjc_runtime::Tensor;

#[test]
fn view_slice_shares_buffer() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

    // Slice row 0: [(0,1), (0,3)] => shape [1, 3]
    let view = t.slice(&[(0, 1), (0, 3)]).unwrap();
    assert_eq!(view.shape(), &[1, 3]);
    assert_eq!(view.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(view.get(&[0, 1]).unwrap(), 2.0);
    assert_eq!(view.get(&[0, 2]).unwrap(), 3.0);

    // Buffer is shared (refcount > 1)
    assert!(t.buffer.refcount() >= 2, "slice should share buffer");
}

#[test]
fn view_slice_second_row() {
    let t = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3]).unwrap();

    // Slice row 1 only: [(1,2), (0,3)]
    let view = t.slice(&[(1, 2), (0, 3)]).unwrap();
    assert_eq!(view.shape(), &[1, 3]);
    assert_eq!(view.to_vec(), vec![40.0, 50.0, 60.0]);
}

#[test]
fn view_transpose_zero_copy() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let tr = t.transpose();

    assert_eq!(tr.shape(), &[3, 2]);
    // Transposed indexing: tr[i][j] = t[j][i]
    assert_eq!(tr.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(tr.get(&[0, 1]).unwrap(), 4.0);
    assert_eq!(tr.get(&[1, 0]).unwrap(), 2.0);
    assert_eq!(tr.get(&[2, 1]).unwrap(), 6.0);

    // Buffer is shared
    assert!(t.buffer.refcount() >= 2, "transpose should share buffer");
}

#[test]
fn view_transpose_is_not_contiguous() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert!(t.is_contiguous());

    let tr = t.transpose();
    assert!(!tr.is_contiguous(), "transposed tensor should not be contiguous");

    // to_contiguous materializes a fresh copy
    let c = tr.to_contiguous();
    assert!(c.is_contiguous());
    assert_eq!(c.to_vec(), vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn view_reshape_shares_buffer() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let r = t.reshape(&[3, 2]).unwrap();

    assert_eq!(r.shape(), &[3, 2]);
    // Data is the same linear order
    assert_eq!(r.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    // Buffer is shared
    assert!(t.buffer.refcount() >= 2, "reshape should share buffer");
}

#[test]
fn view_broadcast_to_expands_with_stride_zero() {
    // A [1, 3] tensor broadcast to [4, 3]
    let t = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[1, 3]).unwrap();
    let b = t.broadcast_to(&[4, 3]).unwrap();

    assert_eq!(b.shape(), &[4, 3]);
    // Every row should be [10, 20, 30]
    for i in 0..4 {
        assert_eq!(b.get(&[i, 0]).unwrap(), 10.0);
        assert_eq!(b.get(&[i, 1]).unwrap(), 20.0);
        assert_eq!(b.get(&[i, 2]).unwrap(), 30.0);
    }

    // Buffer is shared (no data copy)
    assert!(t.buffer.refcount() >= 2, "broadcast should share buffer");
}
