//! Per-node MLP head configuration + deterministic Xavier init.
//!
//! Naming note (Phase 0.4 Track C-2.3.7): the head is per-*node*, not
//! per-*leaf*. `init_params` runs for the root and every `add_node` /
//! `force_grow` / `force_split`. Pre-0.4 docs called this "per-leaf";
//! the code has always been per-node.
//!
//! Phase 0.3a treats every node as carrying a small fused MLP whose
//! architecture is described by a single graph-wide [`LeafHead`]. Each
//! node owns `params: Vec<Tensor>` — interleaved weight/bias tensors
//! initialized deterministically from a SplitMix64 stream seeded by
//! `(graph.seed, node_id, layer_idx, kind_bit)`.
//!
//! The head is *frozen on first install* (mirroring the codebook's
//! one-shot pattern) and **must be installed before any `add_node`**.
//!
//! Phase 0.3a deliberately stops at "storage + init + canonical bytes" —
//! actual `forward` wiring through the ambient `GradGraph` lives on
//! [`AdaptiveBeliefGraph::leaf_forward`](crate::AdaptiveBeliefGraph::leaf_forward).
//! BLR head, OOD scoring, and structural decisions are Phase 0.3b/c/d.

use cjc_ad::pinn::Activation;
use cjc_repro::Rng;
use cjc_runtime::tensor::Tensor;

use crate::node::NodeId;

/// Parsed leaf-MLP architecture spec, frozen at first install.
#[derive(Debug, Clone)]
pub struct LeafHead {
    /// Number of input features per forward call (one call per node).
    pub input_dim: u32,
    /// Hidden layer widths, in order. Empty = direct in→out (linear regression head).
    pub hidden_dims: Vec<u32>,
    /// Output dimensionality.
    pub output_dim: u32,
    /// Activation applied between layers; the final layer stays linear so
    /// the user can wrap softmax / sigmoid / etc. as needed.
    pub activation: Activation,
    /// SHA-256 of canonical bytes, embedded into the `LeafHeadConfigured`
    /// audit event so a snapshot is only replayable against the matching
    /// architecture.
    pub config_hash: [u8; 32],
}

impl LeafHead {
    /// Construct a fresh `LeafHead`. Computes `config_hash` deterministically.
    pub fn new(
        input_dim: u32,
        hidden_dims: Vec<u32>,
        output_dim: u32,
        activation: Activation,
    ) -> Self {
        let mut head = Self {
            input_dim,
            hidden_dims,
            output_dim,
            activation,
            config_hash: [0u8; 32],
        };
        head.config_hash = cjc_snap::hash::sha256(&head.canonical_bytes());
        head
    }

    /// Number of layers (`hidden_dims.len() + 1`). A direct-linear head
    /// (`hidden_dims.is_empty()`) has 1 layer.
    pub fn num_layers(&self) -> usize {
        self.hidden_dims.len() + 1
    }

    /// `(fan_in, fan_out)` for the `layer_idx`-th layer (0-indexed).
    /// Used by Xavier init.
    pub fn layer_shape(&self, layer_idx: usize) -> (u32, u32) {
        let num_layers = self.num_layers();
        debug_assert!(layer_idx < num_layers, "layer_idx out of range");
        let fan_in = if layer_idx == 0 {
            self.input_dim
        } else {
            self.hidden_dims[layer_idx - 1]
        };
        let fan_out = if layer_idx == num_layers - 1 {
            self.output_dim
        } else {
            self.hidden_dims[layer_idx]
        };
        (fan_in, fan_out)
    }

    /// Total number of param tensors per node (`2 * num_layers`).
    pub fn param_count(&self) -> usize {
        2 * self.num_layers()
    }

    /// Canonical big-endian byte encoding for hashing.
    ///
    /// Layout:
    /// ```text
    ///   input_dim    u32 BE
    ///   output_dim   u32 BE
    ///   activation   u8 (encode_activation_tag)
    ///   n_hidden     u16 BE
    ///   hidden_dims  u32 BE × n_hidden
    /// ```
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(11 + self.hidden_dims.len() * 4);
        out.extend_from_slice(&self.input_dim.to_be_bytes());
        out.extend_from_slice(&self.output_dim.to_be_bytes());
        out.push(encode_activation_tag(self.activation));
        out.extend_from_slice(&(self.hidden_dims.len() as u16).to_be_bytes());
        for &h in &self.hidden_dims {
            out.extend_from_slice(&h.to_be_bytes());
        }
        out
    }
}

/// Errors specific to the leaf-head subsystem.
#[derive(Debug, PartialEq)]
pub enum LeafHeadError {
    /// `set_leaf_head` was called twice on the same graph.
    AlreadyFrozen,
    /// `set_leaf_head` was called on a graph that already has child nodes.
    NotEmptyGraph { n_nodes: u32 },
    /// `input_dim`, `output_dim`, or any `hidden_dims[k]` is zero.
    ZeroDim,
    /// `activation_str` did not match any known activation.
    UnknownActivation(String),
    /// `leaf_set_param` got a tensor whose shape didn't match the
    /// architecture's expected shape for that param index.
    ShapeMismatch {
        node_id: NodeId,
        param_index: u32,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// `leaf_param`/`leaf_set_param` got a `param_index` ≥ `param_count`.
    ParamIndexOutOfRange { param_index: u32, n_params: u32 },
    /// A leaf-head op was called on a graph without an installed head.
    NoLeafHead,
}

impl std::fmt::Display for LeafHeadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LeafHeadError::AlreadyFrozen => write!(f, "abng leaf head: already frozen"),
            LeafHeadError::NotEmptyGraph { n_nodes } => write!(
                f,
                "abng leaf head: must be installed before any add_node \
                 (graph already has {n_nodes} nodes)"
            ),
            LeafHeadError::ZeroDim => write!(f, "abng leaf head: dimensions must be positive"),
            LeafHeadError::UnknownActivation(s) => write!(
                f,
                "abng leaf head: unknown activation {s:?} (expected one of: \
                 tanh sigmoid relu none gelu silu elu selu sin)"
            ),
            LeafHeadError::ShapeMismatch {
                node_id,
                param_index,
                expected,
                got,
            } => write!(
                f,
                "abng leaf head: param[{param_index}] on node {node_id} expected shape \
                 {expected:?}, got {got:?}"
            ),
            LeafHeadError::ParamIndexOutOfRange {
                param_index,
                n_params,
            } => write!(
                f,
                "abng leaf head: param index {param_index} out of range \
                 (head has {n_params} param tensors)"
            ),
            LeafHeadError::NoLeafHead => write!(f, "abng leaf head: no leaf head configured"),
        }
    }
}

/// Parse an activation name (mirroring `cjc_ad::dispatch`'s parse_activation).
pub fn parse_activation(s: &str) -> Result<Activation, LeafHeadError> {
    match s {
        "tanh" => Ok(Activation::Tanh),
        "sigmoid" => Ok(Activation::Sigmoid),
        "relu" => Ok(Activation::Relu),
        "none" => Ok(Activation::None),
        "gelu" => Ok(Activation::Gelu),
        "silu" => Ok(Activation::Silu),
        "elu" => Ok(Activation::Elu),
        "selu" => Ok(Activation::Selu),
        "sin" => Ok(Activation::SinAct),
        other => Err(LeafHeadError::UnknownActivation(other.to_string())),
    }
}

/// Numeric tag for an activation, embedded in canonical bytes + audit.
/// Frozen — new variants must allocate new tags rather than reusing.
pub fn encode_activation_tag(a: Activation) -> u8 {
    match a {
        Activation::Tanh => 0x00,
        Activation::Sigmoid => 0x01,
        Activation::Relu => 0x02,
        Activation::None => 0x03,
        Activation::Gelu => 0x04,
        Activation::Silu => 0x05,
        Activation::Elu => 0x06,
        Activation::Selu => 0x07,
        Activation::SinAct => 0x08,
    }
}

/// Decode an activation tag byte. Returns `None` for unknown tags so the
/// snapshot decoder can surface a clean error instead of panicking.
pub fn decode_activation_tag(tag: u8) -> Option<Activation> {
    Some(match tag {
        0x00 => Activation::Tanh,
        0x01 => Activation::Sigmoid,
        0x02 => Activation::Relu,
        0x03 => Activation::None,
        0x04 => Activation::Gelu,
        0x05 => Activation::Silu,
        0x06 => Activation::Elu,
        0x07 => Activation::Selu,
        0x08 => Activation::SinAct,
        _ => return None,
    })
}

/// Mix a SplitMix64 seed for a `(graph_seed, node_id, layer_idx, kind)`
/// quadruple. `kind` is 0 for weights, 1 for biases.
///
/// Why XOR three independent SplitMix64 outputs? Two `Rng::seeded` streams
/// initialized from related seeds produce correlated low-order bits in
/// their first few draws — that would bias Xavier init across nodes whose
/// ids differ by 1. XORing three independent streams kills the correlation.
fn derive_seed(graph_seed: u64, node_id: NodeId, layer_idx: usize, kind: u8) -> u64 {
    let s1 = Rng::seeded(graph_seed).next_u64();
    let s2 = Rng::seeded(node_id as u64 + 0x9E37_79B9_7F4A_7C15).next_u64();
    let mixed_layer = ((layer_idx as u64) << 1) | (kind as u64 & 1);
    let s3 = Rng::seeded(mixed_layer.wrapping_add(0xBF58_476D_1CE4_E5B9)).next_u64();
    s1 ^ s2 ^ s3
}

/// Initialize a single weight tensor for layer `layer_idx` with Xavier-uniform.
fn init_weight(
    head: &LeafHead,
    graph_seed: u64,
    node_id: NodeId,
    layer_idx: usize,
) -> Tensor {
    let (fan_in, fan_out) = head.layer_shape(layer_idx);
    let limit = (6.0 / (fan_in as f64 + fan_out as f64)).sqrt();
    let n = (fan_out as usize) * (fan_in as usize);
    let mut rng = Rng::seeded(derive_seed(graph_seed, node_id, layer_idx, 0));
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        // Uniform in [-limit, +limit].
        data.push((rng.next_f64() * 2.0 - 1.0) * limit);
    }
    Tensor::from_vec(data, &[fan_out as usize, fan_in as usize])
        .expect("xavier weight tensor build")
}

/// Initialize a single bias tensor (zeros).
fn init_bias(head: &LeafHead, layer_idx: usize) -> Tensor {
    let (_, fan_out) = head.layer_shape(layer_idx);
    Tensor::from_vec(vec![0.0; fan_out as usize], &[fan_out as usize])
        .expect("zero bias tensor build")
}

/// Build the full `Vec<Tensor>` of params for one node.
pub fn init_params(head: &LeafHead, graph_seed: u64, node_id: NodeId) -> Vec<Tensor> {
    let mut out = Vec::with_capacity(head.param_count());
    for layer_idx in 0..head.num_layers() {
        out.push(init_weight(head, graph_seed, node_id, layer_idx));
        out.push(init_bias(head, layer_idx));
    }
    out
}

/// Expected shape of param tensor `k` for the given head.
pub fn expected_param_shape(head: &LeafHead, k: u32) -> Vec<usize> {
    let layer_idx = (k as usize) / 2;
    let is_bias = (k as usize) % 2 == 1;
    let (fan_in, fan_out) = head.layer_shape(layer_idx);
    if is_bias {
        vec![fan_out as usize]
    } else {
        vec![fan_out as usize, fan_in as usize]
    }
}

/// SHA-256 of the canonical concatenation of all params on a leaf.
///
/// Each param tensor contributes its shape (ndim u32 BE + dims u32 BE)
/// followed by raw `f64::to_bits().to_be_bytes()` data. Order is the
/// param index — same as the `Vec<Tensor>` storage order.
pub fn params_hash(params: &[Tensor]) -> [u8; 32] {
    let mut buf = Vec::new();
    for t in params {
        let shape = t.shape();
        buf.extend_from_slice(&(shape.len() as u32).to_be_bytes());
        for &d in shape {
            buf.extend_from_slice(&(d as u32).to_be_bytes());
        }
        for x in t.to_vec() {
            buf.extend_from_slice(&x.to_bits().to_be_bytes());
        }
    }
    cjc_snap::hash::sha256(&buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_head() -> LeafHead {
        // 2 → [4] → 1, tanh
        LeafHead::new(2, vec![4], 1, Activation::Tanh)
    }

    #[test]
    fn num_layers_and_param_count() {
        let h = small_head();
        assert_eq!(h.num_layers(), 2);
        assert_eq!(h.param_count(), 4);
    }

    #[test]
    fn linear_head_param_count_two() {
        // No hidden layers → 1 layer → 2 param tensors.
        let h = LeafHead::new(3, vec![], 2, Activation::None);
        assert_eq!(h.num_layers(), 1);
        assert_eq!(h.param_count(), 2);
    }

    #[test]
    fn layer_shape_progression() {
        let h = small_head();
        assert_eq!(h.layer_shape(0), (2, 4)); // input → hidden
        assert_eq!(h.layer_shape(1), (4, 1)); // hidden → output
    }

    #[test]
    fn init_params_shapes_match_arch() {
        let h = small_head();
        let params = init_params(&h, 42, 0);
        assert_eq!(params.len(), 4);
        assert_eq!(params[0].shape(), &[4, 2]); // W_1
        assert_eq!(params[1].shape(), &[4]);    // b_1
        assert_eq!(params[2].shape(), &[1, 4]); // W_2
        assert_eq!(params[3].shape(), &[1]);    // b_2
    }

    #[test]
    fn xavier_init_within_limits() {
        let h = small_head();
        let params = init_params(&h, 0, 0);
        // Layer 0: fan_in=2, fan_out=4 → limit = sqrt(6/6) = 1.0
        let limit = 1.0f64;
        for &v in &params[0].to_vec() {
            assert!(v.abs() <= limit, "weight {v} exceeded limit {limit}");
        }
        // Biases are zero.
        for &v in &params[1].to_vec() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn init_is_deterministic() {
        let h = small_head();
        let a = init_params(&h, 7, 3);
        let b = init_params(&h, 7, 3);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_vec(), y.to_vec(), "Xavier init not deterministic");
        }
    }

    #[test]
    fn different_node_ids_produce_different_inits() {
        let h = small_head();
        let a = init_params(&h, 0, 0);
        let b = init_params(&h, 0, 1);
        // Distinct nodes → different weights (with overwhelming probability).
        assert_ne!(a[0].to_vec(), b[0].to_vec());
    }

    #[test]
    fn different_seeds_produce_different_inits() {
        let h = small_head();
        let a = init_params(&h, 0, 0);
        let b = init_params(&h, 1, 0);
        assert_ne!(a[0].to_vec(), b[0].to_vec());
    }

    #[test]
    fn config_hash_changes_when_arch_differs() {
        let a = LeafHead::new(2, vec![4], 1, Activation::Tanh);
        let b = LeafHead::new(2, vec![5], 1, Activation::Tanh);
        assert_ne!(a.config_hash, b.config_hash);
    }

    #[test]
    fn config_hash_is_deterministic() {
        let a = LeafHead::new(2, vec![4], 1, Activation::Tanh);
        let b = LeafHead::new(2, vec![4], 1, Activation::Tanh);
        assert_eq!(a.config_hash, b.config_hash);
    }

    #[test]
    fn params_hash_changes_after_in_place_modification() {
        let h = small_head();
        let a = init_params(&h, 42, 0);
        let mut b = a.clone();
        // Tweak one weight — hash must change.
        let mut data = b[0].to_vec();
        data[0] += 1.0;
        b[0] = Tensor::from_vec(data, &[4, 2]).unwrap();
        assert_ne!(params_hash(&a), params_hash(&b));
    }

    #[test]
    fn parse_activation_round_trips_known_names() {
        assert!(matches!(parse_activation("tanh").unwrap(), Activation::Tanh));
        assert!(matches!(
            parse_activation("relu").unwrap(),
            Activation::Relu
        ));
    }

    #[test]
    fn parse_activation_unknown_errs() {
        let err = parse_activation("not_a_real_activation").unwrap_err();
        assert!(matches!(err, LeafHeadError::UnknownActivation(_)));
    }

    #[test]
    fn activation_tag_round_trips() {
        for a in [
            Activation::Tanh,
            Activation::Sigmoid,
            Activation::Relu,
            Activation::None,
            Activation::Gelu,
            Activation::Silu,
            Activation::Elu,
            Activation::Selu,
            Activation::SinAct,
        ] {
            let tag = encode_activation_tag(a);
            let back = decode_activation_tag(tag).unwrap();
            assert_eq!(encode_activation_tag(back), tag);
        }
        assert!(decode_activation_tag(0xFF).is_none());
    }

    #[test]
    fn expected_param_shape_alternates_w_b() {
        let h = small_head();
        // params[0] = W_1 shape [4, 2], params[1] = b_1 shape [4]
        // params[2] = W_2 shape [1, 4], params[3] = b_2 shape [1]
        assert_eq!(expected_param_shape(&h, 0), vec![4, 2]);
        assert_eq!(expected_param_shape(&h, 1), vec![4]);
        assert_eq!(expected_param_shape(&h, 2), vec![1, 4]);
        assert_eq!(expected_param_shape(&h, 3), vec![1]);
    }
}
