# ABNG Phase 0.4 — Rough Draft Scope

**Status:** ROUGH DRAFT. Not yet a design note. Not yet committed scope.
**Purpose:** Exhaustive enumeration of every item this chat / repo
considers Phase-0.4 work, so a parallel chat's plan can be cross-checked
against it.
**Sources consolidated:**
- `docs/abng/ABNG_CURRENT_ARCHITECTURE.md` (§5 roadmap, §8 undecided, §6 contracts)
- `docs/abng/PHASE_0_1_DESIGN.md`, `..._0_2_..`, `..._0_3a_..`, `..._0_3c_..`, `..._0_3D_..`
- `docs/abng/PHASE_0_3D_IMPLEMENTATION_PROMPT.md` (§ "Phase 0.4+ scope")
- `CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0023 ABNG Adaptive Belief Radix Graph (Phase 0.1).md`
- `~/.claude/projects/.../memory/project_abng.md`

If you're reading this in the *other* chat: scan each item below and
flag anything in your plan I don't have, or anything I have that you
intentionally cut. Items marked **[CRITICAL]** can't slip without
breaking either the CLI story or the structural-decision quality
story.

---

## A. User-facing CLI surface (`cjcl abng …`)

The headline deliverable. Every other phase deferred CLI work to 0.4.

- [A1] **`cjcl abng train --config <toml> --seed <u64>`** — drives a
  training loop that observes, calls `decide_step`, snapshots
  periodically. Config schema TBD; minimum: dataset path, leaf head
  arch, BLR prior, decision-policy thresholds, n_epochs, batch_size.
- [A2] **`cjcl abng inspect <model.snap> [--node ID] [--audit]
  [--stats] [--tree]`** — one tool, several views:
  - default: graph topology summary (n_nodes, depth, ChildrenKind
    histogram, frozen / dense counts)
  - `--node ID`: full per-node dump (stats, BLR, density, calibration,
    drift, maturity, signature, frozen status)
  - `--audit`: chronological event log
  - `--stats`: aggregate evidence counters (per-action_count,
    n_observations, ECE distribution)
  - `--tree`: ASCII-art trie with key bytes and children variants
- [A3] **`cjcl abng explain <prediction.snap>`** — RouteEvidence,
  full path, per-leaf BLR coefficients, density score breakdown, OOD
  composite breakdown, abstain reason if any. Requires lineage
  plumbing (§F).
- [A4] **`cjcl abng replay <model.snap> --log <audit.log>
  --verify`** — replay-equality gate as a first-class user tool.
  Already exists as `abng_replay` builtin; CLI is the wrapper.
- [A5] **`cjcl abng diff <a.snap> <b.snap>`** — structural + stats
  diff: which nodes changed, which audit events differ, which
  per-leaf params differ by `params_hash`. Two-graph arena lookup
  is already supported.
- [A6] **(maybe) `cjcl abng fmt <model.snap>` and
  `cjcl abng cat <model.snap>`** — pretty-printers used internally
  by `inspect` but exposed as flags.

Open: should `cjcl abng explain` accept either a `prediction.snap`
*or* a `(model.snap, input_tensor.bin)` pair so users can re-run a
prediction without serializing it first?

---

## B. Snapshot / serialization improvements

- [B1] **JSON snapshot view via `cjc_snap::snap_to_json` adapter** —
  read-only. Lets `inspect` emit machine-readable output. Schema is
  *not* the on-disk binary format; it's a derived view.
- [B2] **[CRITICAL] Log compaction** — squash N consecutive
  `*Updated` events (LeafParamsUpdated, BlrUpdated, DensityUpdated,
  CalibrationUpdated) into one `StatsSnapshot` event with a single
  hash witness for the post-batch state. Without this, training-time
  audit logs grow unboundedly (~32 MB / 1M training steps with
  witness-only events; bad in 0.5 chess-RL).
  - Need a new `StatsSnapshot` audit kind (tag `0x18` if `0x17` is
    already taken by 0.3d-2's expected-epistemic event)
  - Need an `abng_compact_log(g, until_seq)` builtin
  - Replay must understand both compacted and uncompacted logs
- [B3] **Snapshot-version negotiation diagnostics** — replay of
  `\x05`/`\x06`/`\x07` should error with `UnsupportedVersion { v: u8 }`
  not `BadMagic`. CLI presents "this is v5; current is v8" message
  and points at a future migration tool.
- [B4] **[CRITICAL] Freeze snapshot format for CLI lifetime** —
  Phase 0.3d alone bumped through 4 versions (`\x05 → \x06 → \x07 →
  \x08`). 0.4 should consolidate every known pending field into one
  bump (likely `\x09`) and *not bump again* during the CLI's life.
  - Implies: bake in `Routed` events even if opt-in (§E2).
  - Implies: bake in lineage fields even if optional (§F).
  - Implies: bake in compaction-event tag.
- [B5] **Migration tool** — `cjcl abng migrate <v5.snap> -o
  <v9.snap>` for users who have older snapshots. Could be deferred
  to 0.5 if no real users have v5/v6/v7 graphs yet. **Decision
  needed.**

---

## C. Decision-engine quality refinements (0.3d-4 deferrals)

Phase 0.3d-4 shipped *defensible-but-simplified* triggers. The full
prompt-spec triggers land here. **Without these the structural
decisions are still deterministic, just lower-fidelity.**
ABNG_CURRENT_ARCHITECTURE.md §8.9–§8.11 has the canonical deferral
table.

| Trigger  | 0.3d-4 (now) | 0.4 (planned) |
|---|---|---|
| Compress | sibling Hamming ≤ τ_compress | full sub-tree signature equivalence |
| Merge    | sibling Hamming ≤ τ_merge **only** | + posterior KL < kl_merge gate |
| Split    | leaf + samples_seen ≥ split_min | + held-out ΔNLL ≥ nll_split_gain + impurity ≥ impurity_min |
| Prune    | unchanged | unchanged |
| Grow     | leaf + samples_seen ≥ grow_min + key unbound | + route-entropy > H_grow gate |
| Freeze   | unchanged | unchanged |

Concrete tasks:

- [C1] **Welford-smoothed `NodeSignature` profiles** (gap 8.3 refinement) — currently the four 8-byte profile hashes are lazy hashes of *current* state. Replace with a Welford-running summary so signatures reflect recent history rather than instantaneous values.
- [C2] **3-window ECE / σ stability buffers** for `Maturity` — currently single-threshold (`|ΔECE| < 0.005` evaluated once). Replace with a 3-window ring buffer; flag `calibration_stable` only when 3 *consecutive* windows all hold. Same for `uncertainty_stable`.
- [C3] **[CRITICAL for Merge quality] KL-divergence gate** — Merge currently fires on signature Hamming alone. Add `kl(P_into || P_merged) < kl_merge` precondition using the BLR posteriors. NIG closed-form KL is well-documented.
- [C4] **[CRITICAL for Split quality] Bootstrap held-out ΔNLL gain** — Split currently fires on `samples_seen ≥ split_min`. Add deterministic stratified bootstrap (SplitMix64-seeded) that holds out X% of the leaf's recent observations, computes NLL on both pre-split and post-split partitionings, and fires Split only if ΔNLL ≥ `nll_split_gain` AND impurity decrease ≥ `impurity_min`.
- [C5] **Route-entropy gate for Grow** — Grow currently fires on `samples_seen ≥ grow_min` + key unbound. Add `H(route_byte_distribution) > H_grow` gate so a leaf that always sees the same key byte doesn't grow needlessly.
- [C6] **[CRITICAL] Real NIG-aware merge math** (gap 8.10) — `force_merge` and policy-driven Merge in 0.3d-4 only set `absorbed.is_active = false`. Phase 0.4 must combine BLR posteriors of `into` and `absorbed` — proper conjugate combine of (precision, mean, a, b) and folded sufficient stats so `into` actually inherits `absorbed`'s evidence. Currently a node receiving merges *loses information*, which is wrong.
- [C7] **Drift-trip auto-Unfreeze** (gap 8.11) — `abng_unfreeze` exists as a manual builtin; the design intent is `decide_step` auto-unfreezes a node when its drift score exceeds a threshold. Wire through. New trigger logic in the engine; no new audit kind (reuses `0x16 Unfreeze`).

---

## D. Maturity / Signature follow-on items

These are the parts of `Maturity` and `NodeSignature` that 0.3d-1
landed as stubs and 0.3d-4 partially fleshed out.

- [D1] Persistent **rolling stability buffers** for ECE, σ_epi (3-window default — see C2). Stored in node, hashed in canonical bytes.
- [D2] **`expected_epistemic` policy refinement** (gap 8.1 follow-on): currently auto-captured at first `uncertainty_stable=true` (0.3d-4). Phase 0.4 may add a re-capture mechanism on drift-trip auto-Unfreeze (so the reference σ resets after a real distribution shift).
- [D3] **`protected_evidence` field on node** — used by Prune to guard against pruning nodes the user has explicitly marked important. Designed in the original spec but not yet on `AdaptiveBeliefNode`. Need: new field + `abng_protect_node(g, id, n)` builtin + audit kind.

---

## E. Categorical / route-aware extensions

- [E1] **[CRITICAL for tabular ML] Categorical features in codebook** (gap 8.4) — the `QuantileCodebook` is real-valued only. 0.4 either:
  - extends `QuantileCodebook` with a `categorical: bool` flag per dim and a hashed-id map, or
  - adds a sibling `CategoricalCodebook` variant.
  - Either way: introduce reserved tag bytes `0xFE = missing`,
    `0xFD..0xC0 = reserved`. Phase 0.2's design note already
    pre-allocated these.
- [E2] **`Routed` audit events** (gap 8.6) — opt-in trace mode for explainability. Each `descend` call optionally emits a `Routed { leaf, matched_prefix }` event. New audit tag (`0x19` or wherever 0.4 lands). Default off — log explosion otherwise. Required by `cjcl abng explain` to walk a real prediction's path.
- [E3] **Drift signals beyond feature mean** (gap 8.5) — Phase 0.3c's drift score is L2 z-shift of feature mean only. Add (some subset of):
  - `label_shift_score` — distribution shift in y over time
  - `missingness_shift` — proportion of missing-flag bytes
  - `category_shift` — categorical distribution drift (depends on E1)
  - `route_shift` — per-byte route-distribution drift
  - **Open:** how many of these actually matter? May ship 1–2 in 0.4 and defer the rest.

---

## F. Lineage + provenance

ABNG_CURRENT_ARCHITECTURE.md §6.7 notes lineage is "partially wired"
— `chain_head` covers training events but dataset and feature-
transform hashes don't appear anywhere.

- [F1] Add `ProvenanceId { dataset_hash, feature_transform_hash, model_hash }` struct.
- [F2] Stamp on `Created` event at graph construction time. New event-kind variant or extend the existing `Created` payload (latter is a snapshot bump, but already needed for [B4]).
- [F3] **`prediction.snap` schema** — formalize what a "prediction snapshot" is: `(input_hash, route_evidence, belief_state, ood_score, decision, audit_pointer, provenance_id)`. New `cjc-snap`-encoded type. Required by `cjcl abng explain` (A3).
- [F4] **`abng_predict_snap(g, node_id, x_idx) → Bytes`** builtin that produces a `prediction.snap` blob. The matching `cjcl abng explain` consumes it.
- [F5] **Audit-pointer references** — `prediction.snap.audit_pointer` is a `(seq, new_hash)` pair so a prediction's lineage points back to a specific moment in the model's history. Phase 0.4 model.snap → prediction.snap → model.snap audit-event must be a verified chain.

---

## G. Performance / hot-path optimisation

These are 0.3a/0.3b deferrals that haven't bitten yet but will in 0.5
chess-RL.

- [G1] **Leaf MLP graph reuse** — `leaf_forward` currently clones every param tensor into the ambient `cjc_ad::GradGraph` on every call. Phase 0.3a explicitly deferred this. Phase 0.4 (or possibly 0.5 if 0.5's profiling shows it dominating) should switch to `set_tensor` + `reforward` so the same `GradGraph` is reused across training steps.
- [G2] **Parallelism in Welford updates** — currently single-threaded. Density observe / calibration observe are independent across nodes; could parallelise with `BinnedAccumulator` (already in `cjc-repro`) to keep bit-determinism. Defer until profiling justifies.
- [G3] **Profile-zone integration** — `profile_zone_start` / `profile_zone_stop` / `profile_dump` builtins exist in `cjc-runtime`. Wire ABNG hot paths (`leaf_forward`, `descend`, `decide_step`) through them so users can profile ABNG training loops with the same tooling as chess-RL v2.3.

---

## H. Tooling beyond CLI

- [H1] **Chain-walk visualizer** — render the audit log as a directed chain with per-event diff (especially structural mutations). Could be HTML/SVG output from `cjcl abng inspect --visualize`.
- [H2] **Per-node trace UI** — given a node id, show all events involving it (filtered audit log). Cheap, useful, lives in `inspect`.
- [H3] **Decision-policy template library** — TOML preset configs (`tabular_default.toml`, `chess_rl.toml`, `pinn.toml`) so users don't have to discover all 11 thresholds from scratch.

---

## I. Quality / robustness

- [I1] More **proptest properties** — current 0.3d-5 has 4. Likely add: KL-merge respects monotonicity; ΔNLL split is reversible; auto-unfreeze never fires without a drift event preceding.
- [I2] More **bolero fuzz targets** — cover the new audit kinds and snapshot v9 layout. Re-run the existing 4 to confirm no regression.
- [I3] **Determinism tests for the CLI** — running `cjcl abng train` twice with same seed and config produces byte-identical model.snap. New `tests/abng/cli_tests.rs`.
- [I4] **Bench harness** — `bench/abng_bench/` mirroring `bench/ad_bench/`. Measures `decide_step` cost as graph grows, `leaf_forward` throughput, snapshot encode/decode time.

---

## J. Cross-cutting items I'm uncertain about — flag for cross-check

These are items where the other chat may differ. **Read these and tell
me if the other chat has any of them in 0.4 or routes them to 0.5.**

- [J1] **`Maturity.protected_evidence` plumbing** — is this 0.4 or 0.5? My draft has it as 0.4 (D3). The original spec mentioned it without committing a phase.
- [J2] **`Dense` tombstone semantics** — currently in 0.3d-3 the `Dense` variant orphans descendants in the arena (still indexed by ID, not routable). Phase 0.4 may want to *actually* free them — but that breaks the "never reorder pushes" invariant. Open question.
- [J3] **Re-init under structural change** — when Split spawns child leaves, do they inherit the parent's params or get fresh Xavier init? 0.3d-3 uses fresh Xavier. 0.4 may want a "warm start from parent" option.
- [J4] **Per-node stats chain shape after Merge** — currently the absorbed node's `stats_chain_head` becomes meaningless (because is_active=false). 0.4's NIG-aware merge math (C6) implies `into` should incorporate `absorbed`'s stats — does that advance `into.stats_chain_head` in a defined way? **Need a concrete spec.**
- [J5] **Compaction event hashing** — when `StatsSnapshot` consolidates N updated events, what's the canonical chain-step? Naively: the consolidated event's `previous_hash` = the pre-batch chain head, `new_hash` = `sha256(previous_hash ‖ canonical_payload(StatsSnapshot))`. But replay needs to verify against the *uncompacted* sequence too if older logs exist. **Possibly punt to 0.5.**
- [J6] **Concurrency story** — is 0.4 still single-thread-only? Multi-graph arena is per-thread, but a single graph's `decide_step` could in principle parallelise per-leaf operations. My draft says 0.4 stays single-thread; flag if other chat differs.
- [J7] **A `Routed` event design** — emit per-descend event vs. per-prediction event (only on `predict_snap`)? Per-descend is too noisy for normal ops; per-prediction is bounded by user calls. My draft prefers per-prediction with an opt-in flag.
- [J8] **Snapshot-version negotiation depth** — does 0.4 only diagnose old versions, or actually migrate v5/v6/v7 → v9? My draft says diagnose-only; B5 leaves migration as a possible 0.5 item. The other chat may have migration in 0.4.

---

## K. Items I am confident the other chat *probably* has the same way

These are items so prominent in the existing docs that any planner
hitting "Phase 0.4" would land on them. If the other chat *doesn't*
have these, that's a much bigger flag than the J-items.

- [K1] `cjcl abng inspect` (A2) — lives in every doc since 0.1
- [K2] `cjcl abng explain` (A3) — required by lineage story
- [K3] `cjcl abng replay` (A4) — already a builtin, CLI is the wrapper
- [K4] `cjcl abng diff` (A5) — explicitly the reason for the multi-graph arena design in 0.1
- [K5] Log compaction (B2) — every phase since 0.1 has flagged event-log growth
- [K6] JSON snapshot view (B1) — explicit in 0.1's "Out-of-scope" list as 0.4 work
- [K7] Real NIG-aware merge math (C6) — without it Merge loses info
- [K8] Welford-smoothed signatures (C1) — explicit refinement table
- [K9] Bootstrap held-out ΔNLL split (C4) — explicit refinement table
- [K10] Drift-trip auto-Unfreeze (C7) — explicit deferral §8.11

---

## L. Possible 0.5 items I want to confirm are *not* in 0.4

If the other chat has any of these in 0.4, flag for discussion.

- [L1] Chess-RL value-head retrofit
- [L2] Chess-RL policy-head retrofit
- [L3] Uncertainty-gated A2C/PPO bootstrap
- [L4] End-to-end determinism gate against existing chess-rl-v2 weight hashes
- [L5] Snapshot migration tool (v5/v6/v7 → v9) — possibly 0.4 (B5) but I lean 0.5
- [L6] Multi-thread parallelism (G2) — likely 0.5 if profiling demands it

---

## M. Test bar for Phase 0.4

Inheriting from 0.3d-5, the bar going into 0.4 is:

- 227 in-crate + 303 integration ABNG tests passing (post-0.3d)
- 4 proptest properties × 256 cases each
- 4 bolero fuzz targets
- workspace lib 2,363 passing, physics_ml 107/107

Phase 0.4 must:
- grow each count, not shrink
- not weaken replay verification to land any item
- not introduce `HashMap`/`HashSet` in canonical paths
- not silently change audit tag bytes (existing 0x00..0x17 frozen)
- bump magic exactly once (likely `\x08 → \x09`) and not again during CLI lifetime
- add proptest properties + bolero fuzz for new audit kinds and any new structural mutations
- add CLI determinism tests (same seed → byte-identical snapshot)

---

## N. Estimated size

Rough LOC estimate based on prior-phase ratios:

| Subarea | Est. LOC | New builtins | New audit kinds |
|---|---:|---:|---:|
| CLI (A1–A6) | ~1500 | 2-3 (predict_snap, compact_log, …) | 0 |
| Snapshot/serde (B1–B5) | ~600 | 0 (snap_v9 is encode/decode only) | 1 (StatsSnapshot) |
| Decision quality (C1–C7) | ~800 | 0-2 (refinement) | 0 |
| Maturity/Signature (D1–D3) | ~300 | 1 (protect_node) | 1 (NodeProtected) |
| Categorical / route (E1–E3) | ~700 | 2-4 | 1 (Routed, opt-in) |
| Lineage (F1–F5) | ~500 | 1-2 (predict_snap, set_provenance) | 1 (ProvenanceStamped) |
| Perf (G1–G3) | ~200 | 0 | 0 |
| Tooling (H1–H3) | ~400 | 0 (CLI subcommands) | 0 |
| Testing (I1–I4) | ~500 | 0 | 0 |
| **Total** | **~5500** | **~10** | **~4** |

Compare 0.3d total: ~3500 LOC, 16 new builtins. So 0.4 is **bigger
by LOC** but **smaller by builtin count** — most of the work is CLI
plumbing and engine refinements rather than new public surface.

That's the right shape: 0.3d added the architectural primitives;
0.4 polishes them and exposes them to humans.

---

## Cross-check checklist (for the other chat)

Print or paste this to the other chat. For each item, mark:

```
[A1]  cjcl abng train ............................ ☐ in their plan / ☐ same scope / ☐ different
[A2]  cjcl abng inspect (default+stats+audit+tree) ☐ in their plan / ☐ same scope / ☐ different
[A3]  cjcl abng explain ........................... ☐ in their plan / ☐ same scope / ☐ different
[A4]  cjcl abng replay (CLI) .................... ☐ in their plan / ☐ same scope / ☐ different
[A5]  cjcl abng diff ............................. ☐ in their plan / ☐ same scope / ☐ different
[A6]  cjcl abng fmt/cat .......................... ☐ in their plan / ☐ same scope / ☐ different
[B1]  JSON snapshot view ......................... ☐ in their plan / ☐ same scope / ☐ different
[B2]  Log compaction ............................. ☐ in their plan / ☐ same scope / ☐ different
[B3]  Snapshot-version negotiation ............... ☐ in their plan / ☐ same scope / ☐ different
[B4]  Single-bump format freeze for CLI .......... ☐ in their plan / ☐ same scope / ☐ different
[B5]  Migration tool ............................. ☐ in their plan / ☐ same scope / ☐ different
[C1]  Welford-smoothed signatures ................ ☐ in their plan / ☐ same scope / ☐ different
[C2]  3-window stability buffers ................. ☐ in their plan / ☐ same scope / ☐ different
[C3]  KL-divergence merge gate ................... ☐ in their plan / ☐ same scope / ☐ different
[C4]  Bootstrap held-out ΔNLL split .............. ☐ in their plan / ☐ same scope / ☐ different
[C5]  Route-entropy grow gate .................... ☐ in their plan / ☐ same scope / ☐ different
[C6]  NIG-aware merge math ....................... ☐ in their plan / ☐ same scope / ☐ different
[C7]  Drift-trip auto-Unfreeze ................... ☐ in their plan / ☐ same scope / ☐ different
[D1]  Rolling stability buffers .................. ☐ in their plan / ☐ same scope / ☐ different
[D2]  expected_epistemic re-capture .............. ☐ in their plan / ☐ same scope / ☐ different
[D3]  protected_evidence field ................... ☐ in their plan / ☐ same scope / ☐ different
[E1]  Categorical codebook ....................... ☐ in their plan / ☐ same scope / ☐ different
[E2]  Routed audit events ........................ ☐ in their plan / ☐ same scope / ☐ different
[E3]  Drift signals beyond feature mean .......... ☐ in their plan / ☐ same scope / ☐ different
[F1]  ProvenanceId struct ........................ ☐ in their plan / ☐ same scope / ☐ different
[F2]  Stamp on Created event ..................... ☐ in their plan / ☐ same scope / ☐ different
[F3]  prediction.snap schema ..................... ☐ in their plan / ☐ same scope / ☐ different
[F4]  abng_predict_snap builtin .................. ☐ in their plan / ☐ same scope / ☐ different
[F5]  Audit-pointer references ................... ☐ in their plan / ☐ same scope / ☐ different
[G1]  Leaf MLP graph reuse ....................... ☐ in their plan / ☐ same scope / ☐ different
[G2]  Parallelism in Welford .................... ☐ in their plan / ☐ same scope / ☐ different
[G3]  Profile-zone integration ................... ☐ in their plan / ☐ same scope / ☐ different
[H1]  Chain-walk visualizer ...................... ☐ in their plan / ☐ same scope / ☐ different
[H2]  Per-node trace UI .......................... ☐ in their plan / ☐ same scope / ☐ different
[H3]  Decision-policy template library ........... ☐ in their plan / ☐ same scope / ☐ different
[I1]  More proptest properties ................... ☐ in their plan / ☐ same scope / ☐ different
[I2]  More bolero fuzz targets ................... ☐ in their plan / ☐ same scope / ☐ different
[I3]  CLI determinism tests ...................... ☐ in their plan / ☐ same scope / ☐ different
[I4]  Bench harness .............................. ☐ in their plan / ☐ same scope / ☐ different
```

Anything **only in their plan** → add here. Anything **only here** →
ask why they cut it. Anything with "different scope" → reconcile.

---

*Rough draft. Not committed. Iterate before promoting to a real
phase design note.*
