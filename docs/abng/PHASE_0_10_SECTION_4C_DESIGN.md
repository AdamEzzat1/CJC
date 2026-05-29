# Phase 0.10 §4.C — MLP leaf heads, design notes

**Status:** *Not implemented this session.* Design pinned for the next pickup.
**Estimated scope:** 1-2 focused sessions (300-500 LOC including tests).
**Headline-impact estimate:** lift §4.F's AUC 0.6645 ± 0.0018 to ~0.67-0.68 (one std bracket toward the top of the strong-published-model band).
**Cost/benefit reframe vs handoff:** the handoff said §4.C was "the ranking-ceiling lever" — needed to break the 0.61 ceiling. §4.E+§4.F already lifted us to 0.66, so §4.C now reads as **"push from 0.66 to ~0.68"**, not "break 0.61". Smaller ROI; reasonable to defer to Phase 0.11.

---

## What already exists in the codebase

Surprisingly, the *forward* infrastructure for MLP leaf heads is already in place.

### `LeafHead` struct supports `hidden_dims: Vec<u32>`

[`crates/cjc-abng/src/leaf_head.rs:30-44`](../../crates/cjc-abng/src/leaf_head.rs:30) — the `LeafHead`
struct already takes `hidden_dims: Vec<u32>` and an `activation: cjc_ad::pinn::Activation`. The activation
enum supports 9 options: Tanh, Sigmoid, Relu, None, Gelu, Silu, Elu, Selu, SinAct.
[`leaf_head.rs:48-63`](../../crates/cjc-abng/src/leaf_head.rs:48) — `LeafHead::new` hashes the architecture
canonically; `config_hash` is embedded in the `LeafHeadConfigured` audit event.

The harness's existing call:

```rust
g.set_leaf_head(transform.phi_width() as u32, vec![], 1, Activation::None)
```

— `hidden_dims = vec![]` means *no* hidden layers. The BLR sits directly on top of raw `phi`. Setting `hidden_dims = vec![32]` would create a 2-layer MLP with a 32-unit hidden layer — but the harness's training path breaks because it would feed raw `phi` (262-dim) into a BLR expecting 32-dim input.

### `leaf_forward` runs the MLP forward inside `cjc-ad::GradGraph`

[`crates/cjc-abng/src/graph.rs:1386-1428`](../../crates/cjc-abng/src/graph.rs:1386) — `leaf_forward(node_id, x_idx) -> (usize, Vec<usize>)` walks all MLP layers, applies `g.mlp_layer(input, w, b, activation)` per layer (final layer always `Activation::None`), and returns the penultimate-features tensor index. The params live inside the node (`self.nodes[id].params`) and are *cloned* into the GradGraph as registered parameters; updates flow back via `leaf_set_param`.

### `leaf_set_param` / `leaf_set_params_batch` write trained params back

[`graph.rs:1265, 1320`](../../crates/cjc-abng/src/graph.rs:1265) — the user's optimizer (e.g. Adam via `adam_step` builtin) writes new tensors back into the node. This emits `LeafSetParam` / `LeafSetParamBatch` audit events with canonical hashes.

### `feature_version_hash` guards stale BLR state

After updating MLP params, the BLR's `feature_version_hash` must be refreshed (otherwise `BlrError::FeatureVersionStale` fires on the next `blr_update`). [`graph.rs:1782-1786`](../../crates/cjc-abng/src/graph.rs:1782) describes a reset path.

---

## What does *not* exist (the §4.C gap)

The integration glue between (a) the MLP-forward pipeline and (b) the harness's training/eval loop.

### 1. The harness's `train_step` bypasses MLP forward

[`graph.rs:1678-1734`](../../crates/cjc-abng/src/graph.rs:1678) — `train_step(x, phi, y)` validates `phi.len() == blr_feature_dim`, then calls `blr.update(phi, &y_arr)` directly. With `hidden_dims = vec![]`, `phi.len() == 262` and `blr_feature_dim == 262`, so the dimensions match. With `hidden_dims = vec![32]`, `blr_feature_dim == 32` but `phi.len()` is still 262 (it's the categorical-transform output). `validate_blr_inputs` rejects.

### 2. No MLP-loss training driver

For MLP heads to be useful, the MLP weights must be *trained* — not random-Xavier-init-and-forget. The training driver isn't in cjc-abng; it would need to:

```
for batch in train_rows {
    build_gradgraph()
    forward through leaf MLP via leaf_forward → penultimate feats
    sigmoid(linear(penultimate)) → predicted prob
    binary cross-entropy loss
    backward
    adam_step(params, grads, ...)
    leaf_set_param() to write weights back
    refresh BLR feature_version_hash
}
```

Then the BLR head sits on top of the *trained* MLP penultimate features and runs the existing `train_step` for the regression layer.

This is a hybrid: MLP layers trained via gradient descent (Adam), BLR layer trained via the existing Bayesian closed-form update.

### 3. No new `AuditKind` variant

The handoff said §4.C should add `AuditKind::MlpLeafUpdate`. Looking at the existing audit kinds, `LeafSetParam` already covers single-param updates. `LeafSetParamBatch` covers multi-param updates from one optimizer step. So:

- **No new AuditKind needed if we route MLP training through `leaf_set_param`/`leaf_set_params_batch`** — those events already canonical-hash the new params and bind them into the chain.
- **A new `MlpTrainStep` AuditKind would be cleaner** — bundles one Adam step + the resulting param batch into a single event, easier to audit. But not strictly required for §4.C; could be a Phase 0.11 refactor.

This is good news for the determinism contract: **no new canary lock is strictly required** for §4.C, because `LeafSetParam` is already canary-locked.

### 4. No eval-time MLP forward

`blr_predict_with_fallback` ([`graph.rs:???`](../../crates/cjc-abng/src/graph.rs)) expects already-penultimate features. The eval loop in `dataset_a_diabetes130.rs:run_trial` would need to run `leaf_forward` (or a forward-only path) to compute penultimate feats before calling `blr_predict_with_fallback`.

---

## Proposed implementation plan (next session)

### Step 1 — Add `run_trial_mlp` harness function (no ABNG changes)

In [`tests/abng/dataset_a_diabetes130.rs`](../../tests/abng/dataset_a_diabetes130.rs):

```rust
const MLP_HIDDEN: u32 = 32;
const MLP_ACTIVATION: Activation = Activation::Relu;
const MLP_EPOCHS: usize = 5;
const MLP_BATCH_SIZE: usize = 64;
const MLP_LR: f64 = 0.001;

fn build_graph_mlp(seed: u64, transform: &CategoricalTransform) -> AdaptiveBeliefGraph {
    // Like build_graph, but with hidden_dims=vec![MLP_HIDDEN] + MLP_ACTIVATION.
}

fn train_mlp_then_blr(rows: &[Vec<String>], ...) -> TrialResult {
    // 1. Build graph with MLP head (random Xavier-init).
    // 2. For MLP_EPOCHS epochs:
    //    - For each batch of phi:
    //      - Build GradGraph
    //      - leaf_forward → penultimate
    //      - Sigmoid(linear penultimate → 1) → predicted_prob
    //      - BCE loss
    //      - Backward via grad_graph_backward
    //      - adam_step → new params
    //      - leaf_set_params_batch → write back
    // 3. After MLP training: clear BLR posterior (refresh feature_version_hash).
    // 4. For each train row:
    //    - x, phi, y from transform
    //    - run leaf_forward → penultimate_features
    //    - call train_step(x, penultimate_features, y)
    // 5. Eval: leaf_forward + blr_predict_with_fallback.
}

#[test]
#[ignore = "Phase 0.10 §4.C MLP-head trial — multi-minute wall clock"]
fn diabetes130_subsample_trial_mlp_head() { ... }
```

Estimated: ~250 LOC.

### Step 2 — Verify the existing ABNG paths handle the MLP setup correctly

Run the always-run ABNG suite — 629 tests should stay green. Then run the new MLP trial and confirm:

- `validate_blr_inputs` accepts the 32-dim penultimate features.
- `feature_version_hash` refresh prevents stale-BLR errors.
- Determinism holds: chain_head reproducible across runs.

### Step 3 — Measure

Compare against the §4.E/§4.F headline:

- §4.F linear-BLR at 20K: AUC ≈ 0.6312
- §4.C MLP head at 20K: target AUC > 0.65 (≈ linear at 101K)
- Both at 101K: target AUC ≈ 0.67–0.68

Honest reporting: if MLP-head at 101K beats §4.F's 0.6645 by less than 0.005, the increased complexity isn't worth it.

### Step 4 — Multi-seed variance

Add 3+ seeds to confirm the result isn't lucky.

### Step 5 — Decision

Either:
- AUC clearly lifts → ship and re-publish blog with both the linear and MLP baselines.
- AUC moves negligibly → ship the negative result; document that linear is enough at this scale.

---

## Determinism risk assessment

| Path | Touches new state? | Needs new canary lock? |
|---|---|---|
| Setting `hidden_dims = vec![32]` in `set_leaf_head` | Existing `LeafHeadConfigured` event covers it | No |
| `leaf_forward` building a GradGraph | Pure-read on the cjc-abng side; cjc-ad GradGraph is separate | No |
| Adam updates via `adam_step` builtin | No ABNG state | No |
| `leaf_set_param` / `leaf_set_params_batch` | Already canary-locked under existing tests | No |
| `train_step` with new feature_version_hash | Existing `TrainStep` audit event already canonical | No |

**Bottom line: §4.C may not need a single new AuditKind variant or canary lock**, contrary to the handoff's expectation. The existing infrastructure was *designed* for this case.

---

## Why this wasn't implemented this session

Three reasons in priority order:

1. **§4.E + §4.F already hit the stretch goal** (AUC 0.6645 in strong-published-model band). §4.C's marginal ROI is now bounded — at best ~+0.02. The handoff's framing as "the ranking-ceiling lever" doesn't match the post-§4.E reality.
2. **The implementation is ~250 LOC of test-side surgery** plus training-loop debugging, plus multi-seed validation. Not a one-session task even after the design exploration above.
3. **The existing `train_step` doesn't compose with MLP heads** — the harness's training loop needs a major refactor to run forward-through-MLP-then-BLR. Either we add a new method on `AdaptiveBeliefGraph` (e.g. `train_step_with_mlp_forward`) or we do the MLP forward in the test harness directly. Either is cleaner if done deliberately than mid-session.

Recommend Phase 0.11 (or later in a longer focused session) for §4.C.

---

## Files

- This design doc: `docs/abng/PHASE_0_10_SECTION_4C_DESIGN.md`
- LeafHead struct: [`crates/cjc-abng/src/leaf_head.rs:30`](../../crates/cjc-abng/src/leaf_head.rs:30)
- `leaf_forward`: [`crates/cjc-abng/src/graph.rs:1386`](../../crates/cjc-abng/src/graph.rs:1386)
- `train_step`: [`crates/cjc-abng/src/graph.rs:1678`](../../crates/cjc-abng/src/graph.rs:1678)
- `set_blr_prior`: [`crates/cjc-abng/src/graph.rs:1432`](../../crates/cjc-abng/src/graph.rs:1432)
- Harness graph construction: [`tests/abng/dataset_a_diabetes130.rs:347`](../../tests/abng/dataset_a_diabetes130.rs:347)
- Existing `pinn` infrastructure: `crates/cjc-ad/src/pinn.rs`
- Adam builtin: `crates/cjc-runtime/src/builtins.rs` (search `adam_step`)
