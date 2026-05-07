# ABNG Phase 0.3a — Per-leaf MLP head (Design Note)

**Date:** 2026-05-06
**Builds on:** [Phase 0.1](PHASE_0_1_DESIGN.md), [Phase 0.2](PHASE_0_2_DESIGN.md)
**Scope:** Attach a small fused MLP to every ABNG node. Deterministic Xavier init. Forward through the ambient `GradGraph`. **No structural decisions, no BLR head, no OOD, no calibration bins.**

## Why this slice

Phase 0.2 shipped the topology and routing. Phase 0.3a is the smallest step that turns ABNG into a *neural* architecture: every node owns a parameter set; user code can compose them through the existing `grad_graph_*` surface to build a real forward/backward path.

What 0.3a *does* deliver:

1. `LeafHead` configuration on the graph (input dims, hidden layer sizes, output dims, activation). Frozen on first install — same one-shot pattern as the codebook.
2. Each node owns `params: Vec<Tensor>` — interleaved weight/bias tensors for the configured architecture.
3. **Deterministic Xavier init** per node, seeded from `mix(graph.seed, node_id, layer_idx)` via SplitMix64. Same seed → byte-identical weights across runs and platforms.
4. `leaf_forward(node_id, x_idx)` — wires the leaf's MLP into the *ambient* `GradGraph` (the one driven by `grad_graph_new`/`grad_graph_input`/etc.). Returns the output node index plus the param node indices the caller can hand to `grad_graph_backward_collect`.
5. `leaf_param` / `leaf_set_param` / `leaf_params_hash` for direct read/write of the underlying tensors.
6. Snapshot format **v3** (magic `ABNG\x03`). Phase 0.2 v2 snapshots no longer load.
7. ~7 new `abng_*` builtins (499 → 506).

What 0.3a explicitly does **not** deliver:

* **Structural decisions** (Grow/Split/Merge/Prune/Compress/Freeze) — Phase 0.3d.
* **BLR head** for epistemic uncertainty — Phase 0.3b.
* **OOD scoring, calibration bins, drift detector** — Phase 0.3c.
* **Per-leaf forward optimizers** (e.g. wrapping `adam_step` per leaf) — user code orchestrates this with the existing `adam_step` builtin. ABNG provides storage; user code provides the training loop.

## Crate dependency change

Phase 0.3a adds `cjc-ad` as a direct dependency of `cjc-abng`. There is no cycle:

```
cjc-runtime ← cjc-ad ← cjc-abng
            ←       ←  cjc-snap
            ←  cjc-repro ←
```

The `leaf_forward` function uses `cjc_ad::dispatch::with_ambient` to add MLP nodes to the ambient `GradGraph` directly. This is the same pattern Phase 3c established — a satellite crate touching the shared per-thread graph.

## Design

### `LeafHead` config

```rust
pub struct LeafHead {
    pub input_dim:  u32,
    pub hidden_dims: Vec<u32>,    // empty = direct in→out
    pub output_dim: u32,
    pub activation: Activation,   // applied between layers; output stays linear
    pub config_hash: [u8; 32],    // SHA-256 of canonical bytes; embeds in audit
}

pub enum Activation {
    Tanh, Sigmoid, Relu, Gelu, Silu, None,
}
```

Frozen on first install (`set_leaf_head`). Subsequent calls error with `LeafHeadAlreadyFrozen`. **Must be installed before any `add_node`** — otherwise the existing nodes would carry empty `params` mismatching the architecture. Root's params are initialized at `set_leaf_head` time.

### Per-node `params: Vec<Tensor>`

Layout for an MLP `[in] → [h_1] → ... → [h_L] → [out]` is `2 × (L + 1)` tensors:

```
params[0]   = W_1     shape [h_1, input_dim]
params[1]   = b_1     shape [h_1]
params[2]   = W_2     shape [h_2, h_1]
params[3]   = b_2     shape [h_2]
...
params[2L+0] = W_out  shape [output_dim, h_L]
params[2L+1] = b_out  shape [output_dim]
```

Empty `hidden_dims` produces a direct `[output_dim, input_dim]` weight + `[output_dim]` bias — a linear regression head.

### Deterministic Xavier init

For each `(layer_idx, kind)` where `kind ∈ {weight, bias}`:

```
seed_local = splitmix64(graph.seed)
           ⊕ splitmix64(node_id as u64)
           ⊕ splitmix64(layer_idx as u64 << 1 | kind_bit)
```

Then a fresh `Rng::seeded(seed_local)` produces the tensor's fill values. Weights use Xavier-uniform (`limit = sqrt(6 / (fan_in + fan_out))`); biases initialize to zero.

The `SplitMix64` mixer alone makes the per-node seed bit-deterministic; the XOR combination is needed because two SplitMix64 streams with related seeds can have correlated low-order bits, and we want each `(node_id, layer_idx, kind)` triple to behave like an independent draw.

### Audit kinds (extending Phase 0.2)

| Tag  | Kind | Payload |
|------|------|---------|
| 0x05 | `LeafHeadConfigured` | `config_hash: [u8; 32]` |
| 0x06 | `LeafParamsInitialized` | `node_id` (in event header), `params_hash: [u8; 32]` |
| 0x07 | `LeafParamsUpdated` | `node_id` (in event header), `params_hash: [u8; 32]` |

`Initialized` fires when a node's params are filled by Xavier init (root at `set_leaf_head` time, child nodes at `add_node` time after head is configured). `Updated` fires whenever `leaf_set_param` writes back a new tensor.

### `leaf_forward(node_id, x_idx) → (y_idx, param_indices)`

```rust
pub fn leaf_forward(
    graph: &AdaptiveBeliefGraph,
    node_id: NodeId,
    x_idx: usize,
) -> Result<(usize, Vec<usize>), GraphError>
```

Implementation:
1. Borrow the leaf's `params: Vec<Tensor>`.
2. Inside `cjc_ad::dispatch::with_ambient(|g| ...)`:
   - For each tensor in `params`, register it via `g.parameter(t.clone())`. Collect the resulting indices.
   - Walk through layers: `current = g.mlp_layer(current, w_idx, b_idx, activation)` for each hidden layer; final layer uses `Activation::None`.
3. Return `(y_idx, param_indices)` so the caller can pass `param_indices` straight into `grad_graph_backward_collect`.

`leaf_forward` reads `params` by clone — the leaf's stored tensors are independent of the GradGraph. After backward, the user reads gradients, runs Adam, then writes back via `leaf_set_param`. This is the standard "fresh graph each step" pattern.

### Snapshot format v3

```
magic           "ABNG\x03"     (5)
seed            u64 BE         (8)
epoch           u64 BE         (8)
final_hash      [u8; 32]       (32)
codebook_present u8            (1)
[ codebook section ]
head_present    u8             (1)        0x00 = no head, 0x01 = present
if head_present:
  input_dim     u32 BE         (4)
  output_dim    u32 BE         (4)
  activation    u8             (1)
  n_hidden      u16 BE         (2)
  hidden_dims   u32 BE × n_hidden
  config_hash   [u8; 32]       (32)
n_nodes         u32 BE         (4)
per node:
  parent        i32 BE         (4)        -1 = root
  children_kind u8             (1)
  children_payload (variable)
  stats canonical_bytes (24)
  stats_version u64 BE         (8)
  stats_chain_head [u8; 32]    (32)
  n_params      u32 BE         (4)        0 if no head
  for each param p:
    ndim        u8             (1)
    shape       u32 BE × ndim
    data        f64 BE × numel (8 × numel bytes)
n_events        u64 BE         (8)
[ events ]
```

Replay rebuilds:
1. Decode header → install codebook + leaf head from blob.
2. Replay events in order:
   - `LeafHeadConfigured` — verify `config_hash` matches the head from the blob. *Initialize the root's params* deterministically.
   - `NodeAdded` — create the node *and initialize its params* deterministically.
   - `LeafParamsInitialized` — verify `params_hash` matches the live (just-initialized) node's params hash.
   - `LeafParamsUpdated { params_hash }` — the new params bytes live with the node, not in the event payload. So replay can't reconstruct them from the event alone. **Solution:** every `LeafParamsUpdated` event carries the params_hash *witness*, but the actual final params come from the per-node section of the snapshot header. After all events replay, we install the *stored* params into each node and verify their hash equals the most-recent `LeafParamsUpdated` (or `LeafParamsInitialized`) event for that node.
3. Verify per-node `canonical_bytes`, `stats_chain_head`, `children` layouts, and the global `chain_head` all match.

This keeps event payloads small (32-byte witness instead of full tensor blobs) at the cost of requiring the per-node param section in the header. The trade-off favors users with many training steps — the audit log is small even after a million updates.

### New builtins (7)

| Name | Args | Returns | Purpose |
|---|---|---|---|
| `abng_set_leaf_head` | `g, input_dim: i64, hidden_dims: Tensor[Hh], output_dim: i64, activation: String` | `Void` | install + freeze head |
| `abng_leaf_head_dims` | `g` | `Tensor` | `[input_dim, n_hidden, hidden_dims..., output_dim, activation_code]` (or empty Tensor when none) |
| `abng_leaf_param_count` | `g, node_id` | `Int` | number of param tensors on a node |
| `abng_leaf_param` | `g, node_id, k: i64` | `Tensor` | read the k-th param tensor |
| `abng_leaf_set_param` | `g, node_id, k: i64, t: Tensor` | `Void` | write back; emits `LeafParamsUpdated` |
| `abng_leaf_params_hash` | `g, node_id` | `String` (hex) | SHA-256 of the canonical bytes of all params concatenated |
| `abng_leaf_forward` | `g, node_id, x_grad_idx: i64` | `Array<Int>` | `[y_idx, p_0, p_1, ..., p_N]` — output index + param indices |

Total surface after 0.3a: **506 dispatch arms** (499 + 7).

### Determinism contract (preserved)

* Xavier init uses `cjc_repro::Rng` (SplitMix64) with seeds derived from `(graph.seed, node_id, layer_idx, kind_bit)` using `wrapping_add(0x9e3779b97f4a7c15)`-style mixing per the existing `Rng` API. Same seed everywhere → bit-identical params.
* All MLP wiring goes through existing `cjc_ad::GradGraph` methods, which are documented as deterministic.
* Param canonical bytes use `f64::to_bits().to_be_bytes()` — preserves NaN bit patterns.

## Tests (estimate)

* In-crate: 62 → ~85 (+23) — Xavier determinism, head freeze + before-add_node guard, param read/write round-trip, leaf_forward shape correctness, params_hash on init vs after update, snapshot v3 round-trip with head, replay with LeafParamsUpdated witnessing.
* Integration (`tests/abng/`): 89 → ~120 (+31) — every new builtin + AST↔MIR parity for forward + a tiny end-to-end "1-step training round-trip" test that runs Adam through the leaf and verifies determinism.

## Risks

1. **Snapshot break v2 → v3.** Same justification as v1 → v2 — Phase 0.2 just shipped, no real users.
2. **`set_leaf_head` ordering constraint.** Must be called before any `add_node`. Documented + enforced. Mitigation: codebook follows the same one-shot pattern, so users already know the freeze idiom.
3. **Param tensor cloning on forward.** Each `leaf_forward` clones every param into the ambient `GradGraph`. For small heads (4 params × ~100 floats = 800 bytes/leaf) this is fine; for large heads it's the next perf optimization (graph reuse via `set_tensor`+`reforward`). Phase 0.3a deliberately defers this.
4. **Event-log growth from `LeafParamsUpdated`.** Every parameter write back fires one event. A 100-epoch training loop on a 10-node graph emits 1,000 events. Mitigation: events carry only the 32-byte hash, not the params; the `params` blob lives in the per-node section. A 1M-event graph is ~133MB in audit alone — log compaction is Phase 0.4.
