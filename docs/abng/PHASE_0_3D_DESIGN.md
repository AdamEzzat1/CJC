# ABNG Phase 0.3d — Structural Decisions (Post-Hoc Design Note)

**Date:** 2026-05-07
**Status:** SHIPPED
**Builds on:** [Phase 0.3c](PHASE_0_3c_DESIGN.md)
**Scope:** structural-decision engine over Phase 0.3a/b/c evidence
— Grow / Split / Merge / Prune / Compress / Freeze / Unfreeze.
Sub-step granularity established mid-flight; this note records the
final decisions for posterity.

## Why post-hoc

The original implementation prompt (`PHASE_0_3D_IMPLEMENTATION_PROMPT.md`)
described the phase as one ~1-week chunk. In practice the work split
naturally into 5 sub-steps as the design surface clarified:

| Sub-step | Scope | Snapshot | Audit kinds added |
|---|---|---|---|
| 0.3d-1 | `Maturity` + `NodeSignature` (lazy, read-only) | (no bump) | (none) |
| 0.3d-2 | `expected_epistemic` capture + calibrated OOD | `\x05 → \x06` | `0x17` |
| 0.3d-3 | `DecisionPolicy` + 6 force-* + `Dense` children | `\x06 → \x07` | `0x10..0x15` |
| 0.3d-4 | `decide_step` engine + Unfreeze + persistent stability | `\x07 → \x08` | `0x16` |
| 0.3d-5 | proptest + bolero + decoder hardening + docs | (no bump) | (none) |

Three magic-byte bumps within one phase is unusual; each was a clean
break per architecture-doc §6 R5 and was enabled by ABNG having no
external users yet. **Phase 0.4 will freeze the format** for the CLI's
lifetime (likely as v9 with all known 0.4 additions consolidated).

## Trigger semantics — designed-vs-shipped

The implementation prompt §2.3 specified rich criteria for each
structural action. Phase 0.3d-4 ships defensible *simplified* variants
that exercise every code path; Phase 0.4 will refine.

| Trigger | Prompt design | 0.3d-4 implementation | 0.4 work |
|---|---|---|---|
| Compress | sub-tree per-leaf signatures within `τ_compress` | sibling Hamming ≤ `τ_compress` of parent's signature | full sub-tree signature equivalence |
| Merge | sibling Hamming ≤ `τ_merge` AND posterior KL < `kl_merge` | Hamming **only**; merge into smaller `NodeId` | + KL-divergence gate, + NIG-aware posterior combination |
| Split | `samples_seen ≥ split_min` AND ΔNLL gain ≥ `nll_split_gain` AND impurity decrease ≥ `impurity_min` | leaf + `samples_seen ≥ split_min` | + held-out bootstrap ΔNLL + impurity |
| Prune | `samples_seen < prune_floor` AND signature stable for `prune_grace_epochs` | exactly that | unchanged |
| Grow | route entropy > `H_grow` AND `samples_seen ≥ grow_min` | leaf + `samples_seen ≥ grow_min` + deterministic-from-(seed, node_id) key unbound | + route entropy gate |
| Freeze | `freeze_after` epochs with bit-identical signatures | exactly that, using `decide_step` calls as "epochs" | unchanged |

The simplifications preserve the *event channel* and *replay
determinism* — the deferred refinements affect *signal quality*, not
*correctness*.

## Decisions worth recording

### D1. Lazy types in 0.3d-1, persistent in 0.3d-4

`Maturity` and `NodeSignature` shipped first as **lazy / read-only**
(no persistent state, no audit kind, no snapshot impact). 0.3d-4
promoted `NodeSignature` to a stability-tracking persistent state via
`last_signature` + `signature_stable_calls` per node. This split kept
0.3d-1 a near-zero-risk PR while still establishing the wire shape.

### D2. Audit-tag allocation skips `0x10..0x16` for 0.3d-2

`ExpectedEpistemicCaptured` lives at tag `0x17`, not `0x10`. The
deviation from FIFO allocation keeps `0x10..0x16` contiguous for the
six structural actions + Unfreeze that arrive in 0.3d-3/4. Architecture
doc §7 #3 only requires "`0x10+` and never re-numbered" — both
schemes satisfy this.

### D3. `set_decision_policy` is install-anytime

Unlike the other one-shot `set_*` builtins (which require
`n_nodes ≤ 1`), `set_decision_policy` can be installed at any point
in the graph's lifecycle. The thresholds are graph-wide configuration
that doesn't depend on per-node initialization order. Documented in
architecture doc §3.2 as the only install-anytime exception.

### D4. Force-* mutations have minimal semantics

`force_merge` only sets `absorbed.is_active = false`; `force_split`
adds two children at deterministic key bytes; `force_compress`
replaces the children container with `Dense` (descendants persist in
the arena, become unreachable through `descend`). These are
deliberately *primitive* — the policy in 0.3d-4 decides *when* to
fire, but doesn't change *how* the mutation executes. The
"semantically correct" merge / split / compress are 0.4's concern.

### D5. Compress orphans descendants

When `force_compress` replaces a node's children container with
`Dense { signature }`, the descendants persist in the arena per
architecture-doc §7 #9 (never reorder pushes). They remain `is_active`
and consume snapshot bytes but are unreachable through `descend`.
This is the simplest interpretation of the "Dense replaces sub-tree"
intent that doesn't violate the never-reorder invariant.

### D6. `Unfreeze` skips `action_counts`

The 6-element `action_counts: [u64; 6]` array tracks structural
*mutations* (Grow/Split/Merge/Prune/Compress/Freeze). Unfreeze is a
*flag flip*, not a mutation, so it doesn't bump any counter. This
keeps `[u64; 6]` exactly aligned with `ActionKind`.

### D7. `decide_step` advances stability even on frozen nodes

The `signature_stable_calls` counter increments for frozen nodes too
— even though no structural action can fire on them. This is so
Phase 0.4's drift-trip auto-Unfreeze path can use the same counter to
decide when a frozen node has been "stable enough to consider
re-activating."

### D8. Decoder allocation hardening (0.3d-5)

The bolero tamper-fuzz revealed that the decoder trusted attacker-
controlled length fields when allocating with `Vec::with_capacity`.
`n_nodes`, `n_events`, `n_params`, and `decode_tensor`'s `numel`
all replaced with `Vec::new()` (or bounded `with_capacity` against
remaining cursor bytes). **This is a security fix**, not just a
test-passing fix — replay is now panic-free under arbitrary byte
flips, which matters for any future deployment that accepts
untrusted snapshots.

## Frozen contracts established in Phase 0.3d

These now sit alongside the 0.1/0.2/0.3a/b/c invariants in
architecture-doc §7:

* Snapshot magic `\x08`
* Audit tags `0x00..0x17` (24 kinds)
* `ChildrenKind` codes `0..5` (Dense added)
* `decide_step` iteration order (NodeId ascending) and trigger
  fall-through (Compress → Merge → Split → Prune → Grow → Freeze)
* `force_merge` / `force_prune` mark inactive but never shrink arena
* `force_compress` orphans descendants
* `Unfreeze` doesn't bump action counts
* `last_signature` advance + `signature_stable_calls` bump happens
  even on frozen nodes

## Test-surface expansion

| Gate | Pre-0.3d | Post-0.3d-5 | Δ |
|---|---:|---:|---:|
| `cargo test -p cjc-abng --lib` | 122 | 227 | +105 |
| `cargo test --test abng` | 175 | 303 | +128 |
| Property tests (4 × 256 cases) | n/a | new | new |
| Bolero fuzz (4 targets) | n/a | new | new |
| `cargo test --workspace --release --lib` | 2,258 | 2,363 | +105 |
| `cargo test --test physics_ml --release` | 107 | 107 | 0 (canary) |

Total ABNG-direct `#[test]` markers: ~530 (227 in-crate + 303
integration).

## What's *not* in Phase 0.3d

Per the simplification table above:

* Welford-smoothed `NodeSignature` profiles
* 3-window ECE / σ stability buffers for `Maturity`
* KL-divergence gate for Merge
* Bootstrap held-out ΔNLL gain for Split
* Route-entropy gate for Grow
* Real NIG-aware merge math (combine BLR posteriors)
* Drift-trip auto-Unfreeze inside `decide_step`
* `cjcl abng …` CLI (the user-facing surface)
* JSON snapshot view, log compaction
* Categorical features in codebook
* `Routed` audit events for explainability

All deferred to Phase 0.4 — see
[`PHASE_0_4_IMPLEMENTATION_PROMPT.md`](PHASE_0_4_IMPLEMENTATION_PROMPT.md)
for the detailed handoff.
