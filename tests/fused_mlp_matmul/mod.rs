//! CANA Phase 3.5b — `GradOp::MlpLayerMatMul` fused autodiff op test suite.
//!
//! Parity properties:
//!   - Forward: `graph.mlp_layer_matmul(input, w1, b1, act, w2)` output tensor
//!     is bit-identical to `graph.matmul(graph.mlp_layer(input, w1, b1, act), w2)`.
//!   - Backward: gradients accumulated for input/w1/b1/w2 are bit-identical
//!     across the two graphs.
//!
//! This is the GradGraph-level analogue of the tensor-level
//! `fused_matmul_norm` and `fused_matmul_dot`. The "dot" in the literal
//! `fused_mlp_matmul_dot` name from the brief reduces to `Mul + Sum`
//! GradOps in autodiff context — building that variant on top of
//! `MlpLayerMatMul` is a clean one-extra-step extension.

pub mod forward_parity;
pub mod backward_parity;
pub mod determinism;
