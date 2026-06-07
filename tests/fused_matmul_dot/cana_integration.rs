//! CANA integration — the fusion identifier recognises `fused_matmul_dot`
//! as a native primitive, so it does not re-propose fusion on top of itself.

use cjc_cana::fusion::{is_native_primitive, NATIVE_PRIMITIVES};

#[test]
fn fused_matmul_dot_is_in_native_primitives() {
    assert!(
        NATIVE_PRIMITIVES.contains(&"fused_matmul_dot"),
        "fused_matmul_dot must be registered in NATIVE_PRIMITIVES so the \
         identifier does not propose fusion on top of an already-fused chain"
    );
}

#[test]
fn fused_matmul_dot_is_recognised_by_lookup() {
    assert!(is_native_primitive("fused_matmul_dot"));
}

#[test]
fn unfused_components_still_recognised() {
    // Regression — adding fused_matmul_dot must not remove its components.
    assert!(is_native_primitive("matmul"));
    assert!(is_native_primitive("dot"));
}
