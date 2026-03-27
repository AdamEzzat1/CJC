//! Integration tests for MIR analysis infrastructure:
//! - Loop analysis (loop tree from CFG)
//! - Reduction analysis (detect/classify accumulation patterns)
//! - Legality verifier (CFG structure, loop integrity, reduction contracts)

mod mir;
