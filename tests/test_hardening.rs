/// CJC Production Hardening Phase — Integration Test Suite
///
/// Verifies all P0 changes introduced in the hardening phase:
///   H-1: Span-aware unification (unify_spanned)
///   H-2: Match exhaustiveness is a compile-time error
///   H-3: Trait resolution enforcement (missing/undefined/duplicate)
///   H-4: MIR CFG structure (BasicBlock, Terminator, predecessor/successor)
///   H-5: Matmul allocation-free (bit-identical numerical results)
///   H-6: Determinism double-run (same seed → identical output)
mod hardening_tests;
