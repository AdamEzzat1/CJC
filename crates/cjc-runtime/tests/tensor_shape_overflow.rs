//! Shapes from untrusted input (decoders, FFI) whose element count or strides
//! would overflow `usize` are errors, not overflow panics (debug builds) or
//! wrapped element counts (release builds).
//!
//! Found through `cjc-abng`: a tampered blob's tensor shape passed the
//! decoder's saturating size check and then overflowed the unchecked product
//! in `Tensor::from_vec`.

use cjc_runtime::tensor::Tensor;

#[test]
fn unrepresentable_shapes_are_errors_not_overflow() {
    let big = u32::MAX as usize;
    // The first shape is the saturated-product-times-zero case that slipped
    // past the decoder.
    for shape in [vec![big, big, big, 0], vec![0, big, big, big], vec![big, big, big], vec![usize::MAX, 2]] {
        assert!(Tensor::from_vec(vec![], &shape).is_err(), "from_vec {shape:?}");
        assert!(Tensor::from_bytes(&[], &shape, "f64").is_err(), "from_bytes f64 {shape:?}");
        assert!(Tensor::from_bytes(&[], &shape, "f32").is_err(), "from_bytes f32 {shape:?}");
        let empty = Tensor::from_vec(vec![], &[0]).unwrap();
        assert!(empty.reshape(&shape).is_err(), "reshape {shape:?}");
    }
    // The element count fits but the byte count does not.
    assert!(Tensor::from_bytes(&[], &[usize::MAX / 4], "f64").is_err());
    assert!(Tensor::from_bytes(&[], &[usize::MAX / 2], "f32").is_err());
}

#[test]
fn ordinary_shapes_are_unchanged() {
    assert_eq!(Tensor::from_vec(vec![1.0; 6], &[2, 3]).unwrap().shape(), &[2, 3]);
    assert_eq!(Tensor::from_vec(vec![], &[3, 0, 2]).unwrap().len(), 0);
    assert_eq!(Tensor::from_vec(vec![7.0], &[]).unwrap().len(), 1);
    assert!(Tensor::from_vec(vec![1.0; 5], &[2, 3]).is_err());
    let t = Tensor::from_vec(vec![1.0; 6], &[2, 3]).unwrap();
    assert_eq!(t.reshape(&[3, 2]).unwrap().shape(), &[3, 2]);
    assert!(t.reshape(&[4, 2]).is_err());
    let bytes: Vec<u8> = [1.0f64, 2.0].iter().flat_map(|x| x.to_le_bytes()).collect();
    assert_eq!(Tensor::from_bytes(&bytes, &[2], "f64").unwrap().len(), 2);
}
