//! Phase I — `Value` size regression guard.
//!
//! `Value` is stored by value in arrays (`Rc<Vec<Value>>`), tuples,
//! frame slots, and scopes — so its `size_of` multiplies across nearly
//! all interpreter memory. Phase I boxed the `SparseTensor(SparseCsr)`
//! variant (88 B, the single largest, but rare — 19 sites) to bring
//! `Value` from 88 → 72 B. This test pins the win so a future variant
//! addition can't silently re-bloat every value: if `Value` grows past
//! 72 B, find the new outlier variant and box it (the
//! `tests/size_probe.rs` helper prints per-type sizes).

use cjc_runtime::Value;

/// The post-Phase-I ceiling. Raising this number is a memory
/// regression — it means some variant's inline payload now exceeds the
/// boxed sparse-matrix bound and should likely be boxed instead.
const VALUE_SIZE_CEILING: usize = 72;

#[test]
fn value_stays_within_size_ceiling() {
    let actual = std::mem::size_of::<Value>();
    assert!(
        actual <= VALUE_SIZE_CEILING,
        "Value grew to {actual} B (ceiling {VALUE_SIZE_CEILING} B) — a new large \
         variant is re-bloating every value; box its payload. Run \
         `cargo test -p cjc-runtime --test size_probe -- --nocapture` to find it."
    );
}

#[test]
fn boxed_sparse_roundtrips_through_value() {
    // The boxed variant must behave identically — construction, the
    // type name, and Display all go through the Box transparently.
    use cjc_runtime::SparseCsr;
    let csr = SparseCsr {
        nrows: 2,
        ncols: 2,
        row_offsets: vec![0, 1, 2],
        col_indices: vec![0, 1],
        values: vec![3.0, 4.0],
    };
    let v = Value::SparseTensor(Box::new(csr));
    assert_eq!(v.type_name(), "SparseTensor");
    // Display reaches through the Box.
    assert!(format!("{v}").contains("SparseTensor(2x2"));
}
