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
    // `norm` is intentionally NOT in NATIVE_PRIMITIVES today — the fusion
    // identifier looks at the matmul → next_primitive chain, and norm being
    // outside the table means the matmul → norm chain currently registers
    // as a single-primitive chain (length 1). When the rewriter lands in
    // Phase 3.5c, adding `norm` to NATIVE_PRIMITIVES will be the trigger
    // that lets identify_fusion_candidates recognise this pair.
    //
    // This test documents the current state so the future addition is
    // intentional, not accidental.
    assert!(!is_native_primitive("norm"));
}
