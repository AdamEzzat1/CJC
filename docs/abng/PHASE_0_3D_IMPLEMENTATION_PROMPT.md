# Phase 0.3d Implementation Prompt — Structural Decisions

**Paste this whole file as the next assistant's task.**

---

You are continuing development of **ABNG** (Adaptive Belief Radix Graph)
in the CJC-Lang repository. Phase 0.3c shipped on 2026-05-06 with 175
integration tests + 122 in-crate tests passing, snapshot magic
`ABNG\x05`, 49 `abng_*` builtins, and 16 audit-kind tags `0x00..0x0F`.

## Step 0 — Read before doing anything

Read these files in order, **before** writing any code or even
proposing a design:

1. `docs/abng/ABNG_CURRENT_ARCHITECTURE.md` ← source of truth
2. `docs/abng/PHASE_0_1_DESIGN.md`
3. `docs/abng/PHASE_0_2_DESIGN.md`
4. `docs/abng/PHASE_0_3a_DESIGN.md`
5. `docs/abng/PHASE_0_3b_DESIGN.md`
6. `docs/abng/PHASE_0_3c_DESIGN.md`

Then survey the actual implementation. **Do not assume the design notes
match the code.** When they disagree, the code wins. Specifically:

- `crates/cjc-abng/src/graph.rs` — every install/observe/score method
- `crates/cjc-abng/src/audit.rs` — exact audit-kind tag bytes (frozen)
- `crates/cjc-abng/src/serialize.rs` — snapshot magic + canonical layout
- `crates/cjc-abng/src/dispatch.rs` — 49 `abng_*` arms, the boundary
- `crates/cjc-abng/src/{leaf_head,blr,density,calibration,drift}.rs`
- `tests/abng/*.rs` — the test surface that must continue to pass

The architecture doc lists every frozen tag, magic byte, and field
order. Read **§7 Do Not Change Assumptions** carefully — every item
there is load-bearing.

## Step 1 — Identify any incomplete Phase 0.3c work

Before layering Phase 0.3d, scan for unfinished items:

- §8 of `ABNG_CURRENT_ARCHITECTURE.md` lists 8 known gaps. At least
  three of them (8.1 OOD `epistemic_z`, 8.2 `Maturity`, 8.3 `NodeSignature`)
  are **prerequisites** for Phase 0.3d — building structural decisions
  on top of an unstable evidence layer makes the policy impossible to
  test.
- Run `cargo test -p cjc-abng --lib` and `cargo test --test abng` —
  confirm clean before starting.
- Check `cargo test --workspace --release --lib` — should match the
  Phase 0.3c regression-gate count of 2,258 passing.
- Run `cargo test --test physics_ml --release` — must still pass
  (107/107). This catches any cjc-ad regression.

If anything is broken at start, **stop and report**. Do not paper over
it.

## Step 2 — Phase 0.3d scope

The structural-decision engine. Six actions, each evidence-gated, each
audited.

### 2.1 Prerequisite: `Maturity` (do this first)

Add per-node `Maturity { samples_seen: u64, calibration_stable: bool,
uncertainty_stable: bool, trust_level: u8 }`. Plumb into
`AdaptiveBeliefNode`. Recomputed lazily when read. Choose default
thresholds:

- `min_required_samples`: e.g. 64 for split, 32 for prune.
- `calibration_stable`: 3 consecutive snapshot windows of |ΔECE| < 0.005.
- `uncertainty_stable`: epistemic σ stable to within 5% over 3 windows.
- `trust_level`: 0..4 monotonic from samples_seen + stability flags.

`Maturity.uncertainty_stable` first holding true is the canonical
moment to capture `expected_epistemic` per leaf, fixing gap 8.1.

### 2.2 `NodeSignature` (gap 8.3)

Add four `[u8; 8]` profile hashes per node, computed from Welford-folded
summaries:

- `prediction_profile_hash` — running mean of leaf MLP outputs on
  observed inputs
- `uncertainty_profile_hash` — running (epistemic, aleatoric, total)
- `calibration_profile_hash` — bin counts + ECE
- `routing_profile_hash` — incoming key-byte histogram

Each is a sha256 of canonical bytes truncated to the first 8 bytes.

### 2.3 The six actions

For each, define:
- The evidence-threshold policy that triggers it.
- The audit kind (new tag) and canonical payload bytes.
- The graph-state mutation.
- The replay-determinism contract.

| Action | Evidence | Audit kind | Notes |
|---|---|---|---|
| **Grow** | route entropy at byte position > `H_grow` AND samples_seen ≥ grow_min | `Grow { node_id, key_byte }` (0x10) | Adds a new child — distinct from `NodeAdded` because *the policy* fires it, not user code |
| **Split** | `samples_seen ≥ split_min` AND held-out ΔNLL gain ≥ `nll_split_gain` AND impurity decrease ≥ `impurity_min` | `Split { parent, children: Vec<NodeId> }` (0x11) | Held-out via deterministic stratified bootstrap (SplitMix64-seeded) |
| **Merge** | sibling `NodeSignature` Hamming ≤ `τ_merge` AND posterior KL < `kl_merge` | `Merge { absorbed: Vec<NodeId>, into: NodeId }` (0x12) | Hysteresis-locked post-merge for `freeze_after` epochs |
| **Prune** | `samples_seen < prune_floor` AND signature unchanged for `prune_grace_epochs` AND `protected_evidence == 0` | `Prune { node_id }` (0x13) | Never prune `protected_evidence > 0` nodes |
| **Compress** | sub-tree's per-leaf signatures all match within `τ_compress` | `Compress { signature: NodeSignature }` (0x14) | Replaces sub-tree with `AdaptiveChildren::Dense { sig }` (the variant the design note reserves but 0.2 didn't ship) |
| **Freeze** | `freeze_after` epochs with bit-identical signatures | `Freeze { node_id }` (0x15) | Stats still update; structural ops blocked until drift detector trips |

Use audit tags `0x10..0x15`. Reserve `0x16` for `Unfreeze` once the
drift-trip un-freeze path is wired.

### 2.4 New builtin surface (estimate ~12)

| Name | Purpose |
|---|---|
| `abng_set_decision_policy(g, json_or_struct)` | install thresholds (one-shot) |
| `abng_node_maturity(g, node_id) -> Tensor[4]` | `[samples_seen, cal_stable, unc_stable, trust]` |
| `abng_node_signature(g, node_id) -> Bytes[32]` | 4×8B profile hashes concatenated |
| `abng_decide_step(g) -> Tensor` | one structural-decision sweep; returns count of each action taken |
| `abng_force_grow(g, node_id, key_byte) -> Int` | for testing: bypass policy |
| `abng_force_split(g, node_id) -> Int` | for testing |
| `abng_force_merge(g, ids[], into) -> Void` | for testing |
| `abng_force_prune(g, node_id) -> Void` | for testing |
| `abng_force_compress(g, root_of_subtree) -> Void` | for testing |
| `abng_force_freeze(g, node_id) -> Void` | for testing |
| `abng_is_frozen(g, node_id) -> Bool` | inspection |
| `abng_action_count(g, kind: i64) -> Int` | how many of each action have fired |

Total surface after 0.3d: **61** dispatch arms (49 + 12).

### 2.5 Snapshot v6

Bumping magic to `ABNG\x06`. Header gains a `decision_policy` section
(thresholds + `policy_hash`). Per-node section gains:

- `maturity`: u64 samples_seen + 3 bool flags + u8 trust = 11 bytes
- `signature`: 4×u64 = 32 bytes
- `is_frozen`: u8

`Dense` children variant must be encoded too: kind code = `0x05`,
payload = `signature: [u8; 32]`.

### 2.6 Decision pipeline (one-pass per `abng_decide_step`)

Iterate nodes in `NodeId` order (HARD: never insertion-time order, never
parallel — would break determinism). For each node:

1. Recompute `Maturity` from current evidence.
2. Recompute `NodeSignature` from latest profiles.
3. If `is_frozen`, skip.
4. Try Compress (requires sub-tree).
5. Try Merge (with sibling).
6. Try Split.
7. Try Prune.
8. Try Grow.
9. Try Freeze.

Each fires at most one event per node per `decide_step` call. The
**order is part of the contract** — replay assumes the same fall-through.

## Step 3 — Testing requirements

For Phase 0.3d to be considered shipped, all of these must be green:

### 3.1 Unit tests (in `crates/cjc-abng/src/*` modules)

Each new module (`maturity.rs`, `signature.rs`, `decision.rs`) gets
`#[cfg(test)] mod tests` with at least:

- One test per public method
- A `determinism_double_run` test
- A `canonical_bytes_size` test if it has a snapshot footprint
- A `state_hash_changes_after_*` test for each mutation

### 3.2 Integration tests under `tests/abng/`

Add new files mirroring the existing pattern:

- `tests/abng/decision_tests.rs` — pure-Rust graph-method tests
- `tests/abng/dispatch_p3d.rs` — every new builtin, happy + error paths
- `tests/abng/parity_p3d.rs` — AST↔MIR parity for every new builtin

Wire them into `tests/abng/mod.rs` alongside the existing
`mod blr_tests; mod uncertainty_tests; ...` declarations.

### 3.3 AST↔MIR parity

**Every** new `abng_*` builtin must have at least one parity test that
runs a `.cjcl` snippet through both backends and asserts byte-identical
printed output. This is non-negotiable. Use the existing
`parity_p3a.rs` / `parity_p3c.rs` `assert_parity` helper as the
template.

### 3.4 Property tests

The repo's property-test root is `tests/prop_tests/mod.rs` (registered
in the workspace `Cargo.toml` as `[[test]] name = "prop_tests"`).
Add `tests/prop_tests/abng_decision_props.rs` and register it from
`mod.rs`. Cover at minimum:

- For random `(seed, observation_sequence)`: replay produces
  byte-identical chain head.
- For random `MaturityThresholds`: structural decisions are
  monotonic — `decide_step` called twice with no new observations
  changes nothing the second time.
- For random feature batches: density score is monotonic in
  Mahalanobis distance.

Use `proptest!` macros, ≥ 256 cases per property.

### 3.5 Bolero fuzz tests

The repo's fuzz root is `tests/bolero_fuzz/mod.rs`. Add
`tests/bolero_fuzz/abng_decision_fuzz.rs` and register. Cover:

- **Structural fuzz:** random sequence of (observe, train, decide)
  calls — replay must verify after every step.
- **Numerical fuzz:** random feature vectors fed to `ood_score`,
  `density_score`, `drift_score` — must never panic, must always
  return finite values in `[0, 1]` (where bounded).
- **Tamper fuzz:** random byte flip in serialized blob — `replay`
  must surface a `DecodeError`, never panic.

Use `bolero::check!` macros.

### 3.6 Replay / determinism / round-trip

Every new state-bearing struct (`Maturity`, `NodeSignature`,
`DecisionPolicy`, `DenseChildren`) needs:

- `canonical_bytes()` and `state_hash()` methods
- A snapshot round-trip test (encode → decode → byte-identical)
- A double-run determinism test
- A tamper test (one byte flip → `DecodeError`)

### 3.7 Regression — earlier ABNG phases

After all your changes, the following must still pass *unchanged*:

- `cargo test -p cjc-abng --lib` — count must grow, not shrink. Was
  122 at end of 0.3c.
- `cargo test --test abng` — Was 175 at end of 0.3c. Add 30+ new
  tests; grow to ~210. Existing 175 must still pass.
- `cargo test --workspace --release --lib` — was 2,258. Must stay
  ≥ 2,258 + (new in-crate count).
- `cargo test --test physics_ml --release` — must still be 107/107.
  This is the cross-crate determinism canary.

If any earlier ABNG phase's tests start failing, **stop**, surface the
regression, and fix it before proceeding.

### 3.8 Run order for the gate

```bash
# Fast feedback loop while developing
cargo check -p cjc-abng
cargo test -p cjc-abng --lib
cargo test --test abng

# Pre-commit regression gate
cargo test --test physics_ml --release
cargo test --workspace --release --lib

# Property + fuzz (slow; run before merge)
cargo test --test prop_tests --release
cargo test --test bolero_fuzz --release
```

## Step 4 — Hard constraints (do not violate)

These come from `ABNG_CURRENT_ARCHITECTURE.md` §7 and the project
`CLAUDE.md`. Re-read them before each commit.

1. **Do not weaken replay verification to make tests pass.** If a
   `DecodeError` fires, the test is correct and the implementation is
   wrong. Replay is bit-equality or it errors.
2. **Do not introduce `HashMap`/`HashSet` in canonical paths.**
   `BTreeMap`/`BTreeSet` only.
3. **Do not silently change audit tag bytes, snapshot encoding, or
   frozen API behavior.** Tags `0x00..0x0F` keep their meanings.
   New tags use `0x10..` with documented canonical payload bytes.
4. **Do not change `MAGIC` without bumping the version byte.** v5 → v6
   is a clean break, exactly like every prior bump.
5. **Do not extend the `Value` enum.** Cross-language handles use
   `Int` / `Tensor` / `String` / `Bytes` / `Array` only.
6. **Do not call `cjc_ad::GradGraph::new`** inside cjc-abng. Use
   `cjc_ad::dispatch::with_ambient(...)`. AST↔MIR parity depends on
   this.
7. **Do not add FMA** to any kernel that touches belief state.
8. **Do not parallelize Welford / Kahan reductions** without using
   `cjc_repro::BinnedAccumulator`. Single-threaded determinism is the
   contract today.
9. **Do not paper over a regression.** If an earlier ABNG phase's
   tests fail because of your changes, that's the bug, not the test.
10. **Preserve ABNG's core philosophy:** deterministic, auditable,
    replayable, locally inspectable, structurally adaptive.
    Every Phase 0.3d feature must defend each of those properties.

If a feature would require breaking any constraint above, **stop**,
write a one-page note explaining the conflict, propose an alternative,
and ask before proceeding. This mirrors `CLAUDE.md` HARD RULE.

## Step 5 — When ambiguity arises

Always prefer:

- **Existing ABNG contract** over convenience.
- **Explicit error** over silent fallback.
- **One-shot freeze** over re-installable state (matches every existing
  `set_*` builtin).
- **Witness-only audit events** over full-payload events (matches
  `LeafParamsUpdated` etc., keeps log compact).
- **Per-node arena order** over insertion-time iteration (the only
  determinism-safe order).

When a phase design note says X and the code says Y, the code wins.
Document the gap in the next phase's design note.

## Step 6 — Handoff requirement at end of Phase 0.3d

When done, update:

1. `docs/abng/PHASE_0_3D_DESIGN.md` — design note, post-hoc OK.
2. `docs/abng/ABNG_CURRENT_ARCHITECTURE.md` — bump phase status,
   update §1, §3, §6, §7, §8 as appropriate. Do not let this doc fall
   behind the code again.
3. `CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0023 *.md` — append a
   "Phase 0.3d amendment" section.
4. `~/.claude/projects/C--Users-adame-CJC/memory/project_abng.md` —
   prepend the new "Phase 0.3d deliverable" block.
5. `~/.claude/projects/C--Users-adame-CJC/memory/MEMORY.md` — replace
   the ABNG entry with a one-line pointer to the updated project memory.

Then write a `docs/abng/PHASE_0_4_IMPLEMENTATION_PROMPT.md` for the
*next* phase before closing the session.

---

## Phase 0.4+ scope (not for this session, listed for context)

After Phase 0.3d, the remaining work is:

### 0.4 — CLI + tooling

- `cjcl abng train --config x.toml --seed 42`
- `cjcl abng inspect model.snap [--node ID] [--audit] [--stats] [--tree]`
- `cjcl abng explain prediction.snap` — full RouteEvidence + per-leaf
  BLR coefficients + abstain reason
- `cjcl abng replay model.snap --log audit.log --verify`
- `cjcl abng diff a.snap b.snap`
- JSON snapshot view via `cjc_snap::snap_to_json` adapter
- Log compaction: squash N consecutive `*Updated` events into one
  `StatsSnapshot` (rebased delta chain)
- Snapshot-version negotiation diagnostics ("this is v3, please upgrade")
- `Dense` compressed children variant — design says reserved, Phase
  0.3d's Compress action *needs* it, so it actually lands earlier

### 0.5 — Chess-RL retrofit

- Replace value head only first (Phase 0.5a)
- Then policy head (Phase 0.5b)
- Uncertainty-gated A2C/PPO bootstrap: when `epistemic > τ_abstain`,
  fall back to plain MC return instead of bootstrapping a confidently-
  wrong critic
- End-to-end determinism gate: chess-rl-v2's `weight_hash` must remain
  bit-identical for the non-ABNG comparison runs

### Lineage semantics

ABNG_CURRENT_ARCHITECTURE.md §6.7 notes that dataset and feature-
transform hashes aren't yet wired through the audit log. Phase 0.4's
`cjcl abng explain` is the natural place to formalize this. Add:

- `ProvenanceId { dataset_hash, feature_transform_hash, model_hash }`
- Stamped on `Created` event at graph construction time
- `prediction.snap` carries the full `(input_hash, route_evidence,
  belief_state, ood_score, decision, audit_pointer, provenance_id)`

### Better tooling

- Chain-walk visualizer (per-event diff)
- Per-node trace UI (recent events filtered by node_id)
- Profile zone integration (the `profile_zone_*` builtins exist in
  cjc-runtime — wire ABNG hot paths through them)

---

*Reminder: ABNG's value isn't the math, it's the audit. Don't let
Phase 0.3d's structural decisions silently break the chain.*
