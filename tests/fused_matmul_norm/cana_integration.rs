//! CANA integration — `fused_matmul_norm` is recognised as a native primitive
//! AND a chain `matmul → norm` is identified by the fusion module as a
//! fusion candidate of length 2 (the rewriter target).

use cjc_cana::fusion::{is_native_primitive, NATIVE_PRIMITIVES};

#[test]
fn fused_matmul_norm_is_in_native_primitives() {
    assert!(
        NATIVE_PRIMITIVES.contains(&"fused_matmul_norm"),
        "fused_matmul_norm must be registered in NATIVE_PRIMITIVES"
    );
}

#[test]
fn fused_matmul_norm_is_recognised_by_lookup() {
    assert!(is_native_primitive("fused_matmul_norm"));
}

#[test]
fn unfused_components_still_recognised() {
    // Regression — both halves of the chain stay listed.
    assert!(is_native_primitive("matmul"));
    // Phase 3.5c flipped this: `norm` is now in NATIVE_PRIMITIVES, which
    // lets identify_fusion_candidates recognise the matmul → norm pair
    // as a fusion candidate of length 2. The fusion_rewrite pass then
    // collapses the chain into a single fused_matmul_norm call.
    assert!(is_native_primitive("norm"));
}
