//! `Tensor::add_assign_unchecked` must add element to element, whatever the layout.
//!
//! A transposed or sliced tensor is a view: its elements are not in buffer order. The
//! in-place path used to pair the two raw buffers, which is only right when both sides
//! are contiguous. Found by the Scholium sweep as a wrong `transpose` gradient.

use cjc_runtime::Tensor;

fn t(data: &[f64], shape: &[usize]) -> Tensor {
    Tensor::from_vec_unchecked(data.to_vec(), shape)
}

fn bits(x: &Tensor) -> Vec<u64> {
    x.to_vec().iter().map(|v| v.to_bits()).collect()
}

#[test]
fn adding_a_transposed_view_matches_add() {
    let a = t(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[3, 2]);
    // [2, 3] -> transpose -> [3, 2], elements in the order 1 4 2 5 3 6.
    let view = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).transpose();
    let want = a.add_unchecked(&view);
    let mut got = a.clone();
    got.add_assign_unchecked(&view);
    assert_eq!(got.to_vec(), vec![11.0, 24.0, 32.0, 45.0, 53.0, 66.0]);
    assert_eq!(bits(&got), bits(&want));
}

#[test]
fn adding_into_a_transposed_view_matches_add() {
    let mut view = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).transpose();
    let b = t(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[3, 2]);
    let want = view.add_unchecked(&b);
    view.add_assign_unchecked(&b);
    assert_eq!(view.to_vec(), vec![11.0, 24.0, 32.0, 45.0, 53.0, 66.0]);
    assert_eq!(bits(&view), bits(&want));
}

#[test]
fn adding_a_slice_view_matches_add() {
    let base = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]);
    let view = base.slice(&[(2, 4), (0, 2)]).unwrap(); // rows 2 and 3: 5 6 7 8
    let mut got = t(&[100.0, 200.0, 300.0, 400.0], &[2, 2]);
    got.add_assign_unchecked(&view);
    assert_eq!(got.to_vec(), vec![105.0, 206.0, 307.0, 408.0]);
    // The source of the view is untouched.
    assert_eq!(base.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn the_contiguous_path_is_unchanged_and_copies_a_shared_buffer() {
    let a = t(&[0.1, 0.2, 0.3], &[3]);
    let held = a.clone(); // shares the buffer
    let mut got = a.clone();
    got.add_assign_unchecked(&t(&[0.2, 0.1, 0.6], &[3]));
    assert_eq!(bits(&got), bits(&a.add_unchecked(&t(&[0.2, 0.1, 0.6], &[3]))));
    assert_eq!(held.to_vec(), vec![0.1, 0.2, 0.3]);
}
