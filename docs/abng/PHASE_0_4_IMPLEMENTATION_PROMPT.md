# Phase 0.4 Implementation Prompt — `cjcl abng …` CLI + Quality Refinements

**Paste this whole file as the next assistant's task.**

---

You are continuing development of **ABNG** (Adaptive Belief Radix Graph)
in the CJC-Lang repository. Phase 0.3d shipped on 2026-05-07 with **303
integration + 227 in-crate tests passing**, snapshot magic `ABNG\x08`,
**65 `abng_*` builtins**, and **24 audit-kind tags `0x00..0x17`**. The
structural-decision engine (`abng_decide_step`) is functional with
defensible-but-simplified triggers.

## Step 0 — Read before doing anything

Read these files in order, **before** writing any code or even
proposing a design:

1. `docs/abng/ABNG_CURRENT_ARCHITECTURE.md` ← source of truth (post-0.3d)
2. `docs/abng/PHASE_0_3D_DESIGN.md` (post-hoc, captures 0.3d sub-step decisions)
3. `docs/abng/PHASE_0_3D_IMPLEMENTATION_PROMPT.md` (previous phase's prompt)
4. `docs/abng/PHASE_0_3a_DESIGN.md`
5. `docs/abng/PHASE_0_3b_DESIGN.md`
6. `docs/abng/PHASE_0_3c_DESIGN.md`

Then survey the actual implementation. **Do not assume the design notes
match the code.** When they disagree, the code wins.

The architecture doc lists every frozen tag, magic byte, and field
order. **Read §7 (Do Not Change Assumptions) and §8 (Still
Undecided / Gaps) carefully** — every gap with a "0.4" deferral note
is in scope for this phase.

## Step 1 — Identify any incomplete Phase 0.3d work

Before layering Phase 0.4, scan for unfinished items:

- §8 of `ABNG_CURRENT_ARCHITECTURE.md` lists ~7 known gaps deferred to
  0.4. The decision-engine simplifications in §8.9, the merge-math
  deferral in §8.10, and the drift-trip auto-Unfreeze in §8.11 are
  the heart of Phase 0.4's quality refinements.
- Run the standard gates clean:
  - `cargo test -p cjc-abng --lib` (227 expected)
  - `cargo test --test abng` (303 expected)
  - `cargo test --test physics_ml --release` (107/0/2 ignored)
  - `cargo test --workspace --release --lib` (2,363 expected)
  - `cargo test --test prop_tests abng_decision` (4 properties pass)
  - `cargo test --test bolero_fuzz abng_decision` (4 targets pass)

If anything is broken at start, **stop and report**. Do not paper over
it.

## Step 2 — Phase 0.4 scope

Phase 0.4 has **two tracks** that can ship as separate sub-steps:

* **Track A: User-facing CLI** — `cjcl abng …` subcommands, JSON
  snapshot, log compaction, snapshot-version negotiation.
* **Track B: Quality refinements** — Welford-smoothed signatures,
  KL-merge, ΔNLL split, route-entropy grow, real NIG-aware merge
  math, drift-trip auto-Unfreeze.

Tracks A and B can ship in either order; pick the one that unblocks
more user value first. Suggested ordering: **Track B first** (so the
CLI ships reading high-quality evidence), then **Track A** (which
freezes the format for external users).

### 2.1 Track A — `cjcl abng …` CLI

| Subcommand | Purpose | Sketch |
|---|---|---|
| `cjcl abng train --config x.toml --seed 42` | Driver for end-to-end training that wires observe → decide_step → checkpoint | Reads TOML config (seed, policy thresholds, leaf-head shape, observation source). Produces a `.abng` snapshot at the end. |
| `cjcl abng inspect model.snap [--node ID] [--audit] [--stats] [--tree]` | Read-only snapshot viewer | Prints chain head, action counts, per-node maturity / signature / BLR posterior summaries. |
| `cjcl abng explain prediction.snap` | Full RouteEvidence + per-leaf BLR coefficients + abstain reason | Requires `Routed` audit events (§8.6). |
| `cjcl abng replay model.snap --log audit.log --verify` | Run the snapshot's audit log through `replay()` and assert chain integrity | Pure verification — no graph mutation. |
| `cjcl abng diff a.snap b.snap` | Structural diff between two snapshots | Compare topology, action_counts, per-node fingerprints. |

**Additional Track A items:**
- JSON-safe snapshot view via `cjc_snap::snap_to_json` adapter
- Log compaction: squash N consecutive `*Updated` events into one
  `StatsSnapshot` (rebased delta chain)
- Snapshot-version negotiation diagnostics ("this is v3, please
  upgrade") — see architecture doc §8.8
- **Freeze the snapshot format for the CLI's lifetime** (likely as
  v9 with all known 0.4 additions consolidated into a single bump).

### 2.2 Track B — Decision-engine quality refinements

Each item improves the **signal quality** of an existing trigger
without changing the contract surface (no new audit kinds, no
snapshot bump per item). Bundle them into one snapshot bump at end of
Track B.

#### 2.2.1 Welford-smoothed `NodeSignature` profiles
Replace the lazy sha256-truncate-of-state with persistent Welford-
folded summaries per profile (prediction / uncertainty / calibration
/ routing). Each profile owns a Welford accumulator (mean+variance);
the 8-byte signature is `sha256(canonical_bytes_of_welford_state)[..8]`.

**Why:** the current strict signature changes on every observation,
which makes `signature_stable_calls` strict. Welford-smoothing makes
"stability" mean "the running summary is no longer changing
significantly" — which is what the prompt §2.2 originally specified.

**Implementation note:** new persistent state per node (~128 bytes:
4 × Welford-state). Snapshot bump.

#### 2.2.2 3-window ECE / σ stability buffers for `Maturity`

Replace the single-threshold `calibration_stable` / `uncertainty_stable`
with 3-window buffers per the prompt §2.1:
- `calibration_stable`: 3 consecutive snapshot windows of `|ΔECE| < 0.005`
- `uncertainty_stable`: epistemic σ stable to within 5% over 3 windows

Each window is one `decide_step` call. Add ring buffers
`ece_history: [f64; 3]` + `sigma_history: [f64; 3]` per node + fill
counters.

#### 2.2.3 KL-divergence gate for Merge

Add the missing `kl_merge` gate to the Merge trigger. KL between two
NIG posteriors has a closed form:
`KL[N(m1, Σ1) ‖ N(m2, Σ2)] = ½ ( log|Σ2|/|Σ1| − d + tr(Σ2⁻¹Σ1) + (m2−m1)ᵀΣ2⁻¹(m2−m1) )`.
Use the BlrState's `mean` + `precision` as Σ⁻¹. No new persistent
state needed.

#### 2.2.4 Bootstrap held-out ΔNLL gain for Split

Add the missing ΔNLL + impurity gates to the Split trigger:
- Held-out via deterministic stratified bootstrap (SplitMix64-seeded
  from `(graph.seed, node_id, decide_step_call_count)`)
- Compare NLL before vs after a hypothetical split into 2 children
- Threshold against `policy.nll_split_gain()` and `policy.impurity_min()`

This is the most computationally expensive refinement in Track B
(per-decide_step bootstrap for every Split candidate).

#### 2.2.5 Route-entropy gate for Grow

Compute Shannon entropy of the route key-byte distribution at the
candidate node's depth. Compare to `policy.h_grow()`. Requires the
codebook to be installed (skip Grow on codebook-less graphs).

#### 2.2.6 Real NIG-aware merge math

`force_merge` and policy-driven Merge currently only set
`absorbed.is_active = false`. Phase 0.4's version should:
1. Combine BLR posteriors: `Λ_into = Λ_into + Λ_absorbed`,
   `m_into = Λ_into⁻¹ (Λ_into · m_into + Λ_absorbed · m_absorbed)`,
   `a_into = a_into + a_absorbed - a_prior`,
   `b_into = b_into + b_absorbed - b_prior` (subtract prior to avoid
   double-counting).
2. Combine stats: Welford merge of (n, mean, M2) — see Chan/Golub/
   LeVeque parallel Welford combine.
3. Then mark absorbed inactive.

**Snapshot impact:** none (combines existing fields). **Test impact:**
new in-crate tests for the combine math.

#### 2.2.7 Drift-trip auto-Unfreeze inside `decide_step`

Add a 7th step to `decide_step`'s per-node ladder (after Freeze):
if `is_frozen && drift_score > policy.drift_unfreeze_threshold`,
fire Unfreeze. Requires extending `DecisionPolicy` with a 12th
threshold — but the architecture doc §3.2 says DecisionPolicy is
one-shot, so this is a snapshot-format change.

**Implementation note:** either extend `DecisionPolicy` to 12
thresholds (snapshot bump) OR hard-code a sensible default (no bump,
deferred-real). Choose with the user.

### 2.3 New audit kinds (Track A only)

Track A may need:
- `0x18 StatsSnapshot` (log compaction) — rebases N consecutive
  `*Updated` events into one
- `0x19 Routed { leaf, matched_prefix }` (explain mode, opt-in)
- `0x1A ProvenanceStamped { dataset_hash, feature_transform_hash, model_hash }` (lineage — see architecture doc §6.7)

Allocate `0x18..0x1A` in tag order if shipped.

### 2.4 New builtin surface (estimated)

Track A: ~12 new builtins for the CLI subcommands (each subcommand
binds to one or more `abng_cli_*` builtins).
Track B: ~3 new builtins:
- `abng_signature_welford_dump(g, node_id) -> Tensor` (inspect)
- `abng_kl_divergence(g, node_a, node_b) -> Float` (BLR KL helper)
- `abng_route_entropy(g, node_id) -> Float`

Total surface after 0.4: ~80 dispatch arms.

### 2.5 Snapshot v9

**Goal: this should be the LAST snapshot bump for ABNG's pre-1.0
lifecycle.** Bundle ALL of Phase 0.4's persistent-state additions
into one bump:

```
magic                    "ABNG\x09"
... v8 layout ...
+ per-node: Welford state for 4 NodeSignature profiles (~128 bytes)
+ per-node: ECE history [f64; 3] + σ history [f64; 3] + fill counters
+ DecisionPolicy: gain a 12th threshold (drift_unfreeze) → 96 bytes canonical
... events gain 0x18..0x1A as needed ...
```

Once shipped, **freeze the format**. Phase 0.5+ should treat v9 as
permanent.

## Step 3 — Testing requirements

For Phase 0.4 to be considered shipped, all of these must be green:

### 3.1 Unit tests

Each new module / extension gets `#[cfg(test)] mod tests` with at
least:
- `determinism_double_run` — same input → same bytes
- `canonical_bytes_size` — pin the wire shape
- `state_hash_changes_after_*` — for any new mutation
- For each refined trigger: `triggers_when_X`, `does_not_trigger_when_Y`

### 3.2 Integration tests under `tests/abng/`

- `tests/abng/cli_tests.rs` — end-to-end driver tests for each `cjcl abng` subcommand
- `tests/abng/welford_signature_tests.rs` — Welford-smoothed signatures stable under small perturbations
- `tests/abng/merge_math_tests.rs` — NIG-aware merge correctness
- Extend `dispatch_p3d.rs` and `parity_p3d.rs` for the new builtins (Track B's 3 + any Track A inspection builtins)

### 3.3 AST↔MIR parity

Every new `abng_*` builtin must have at least one parity test that
runs a `.cjcl` snippet through both backends and asserts byte-identical
printed output.

### 3.4 Property tests — extend `tests/prop_tests/abng_decision_props.rs`

Add at least 2 new properties:
- **KL-merge symmetry:** `kl_divergence(a, b) ≥ 0` and equals 0 iff posteriors are bit-identical
- **Welford-smoothed signature stability:** observing a stationary stream, the signature's Hamming distance to itself shrinks over time

### 3.5 Bolero fuzz — extend `tests/bolero_fuzz/abng_decision_fuzz.rs`

Add at least 1 new target:
- **CLI fuzz** (Track A): random TOML configs feeding `cjcl abng train` must produce well-defined errors, never panic

### 3.6 Replay / determinism

The v8 → v9 break must:
- reject all v1..v8 magics with `BadMagic`
- byte-round-trip every test case shipping in 0.3d
- have a single `v9_magic_in_blob` + `v8_magic_rejected` pair in
  `serialize.rs::tests`

### 3.7 Regression — earlier phases

After all 0.4 changes, every gate from 0.3d-5 must still pass with
counts only **growing**, never shrinking.

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
`CLAUDE.md`. Re-read before each commit.

1. Do not weaken replay verification to make tests pass.
2. Do not introduce `HashMap`/`HashSet` in canonical paths.
3. Do not silently change audit tag bytes, snapshot encoding, or
   frozen API behavior. Tags `0x00..0x17` keep their meanings. New
   tags use `0x18..` with documented canonical payload bytes.
4. Do not change `MAGIC` without bumping the version byte.
5. Do not extend the `Value` enum.
6. Do not call `cjc_ad::GradGraph::new` inside cjc-abng.
7. Do not add FMA to any kernel that touches belief state.
8. Do not parallelize Welford / Kahan reductions without using
   `cjc_repro::BinnedAccumulator`.
9. Do not paper over a regression.
10. Do not break the `decide_step` iteration order or the trigger
    fall-through (architecture-doc §3.7 / §7 #11).
11. Do not unfreeze without a corresponding audit event (`0x16`).
12. Do not write to a frozen node's structural state.
13. Preserve ABNG's core philosophy: deterministic, auditable,
    replayable, locally inspectable, structurally adaptive.

If a feature would require breaking any constraint above, **stop**,
write a one-page note explaining the conflict, propose an alternative,
and ask before proceeding.

## Step 5 — When ambiguity arises

Always prefer:

- **Existing ABNG contract** over convenience.
- **Explicit error** over silent fallback.
- **One-shot freeze** over re-installable state.
- **Witness-only audit events** over full-payload events for
  per-batch updates (matches `*Updated` pattern).
- **Per-node arena order** over insertion-time iteration.

When a phase design note says X and the code says Y, the code wins.
Document the gap in this design note.

## Step 6 — Handoff requirement at end of Phase 0.4

When done, update:

1. `docs/abng/PHASE_0_4_DESIGN.md` — design note, post-hoc OK.
2. `docs/abng/ABNG_CURRENT_ARCHITECTURE.md` — bump phase status,
   update §1, §3, §6, §7, §8 as appropriate.
3. `CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0023 *.md` — append a
   "Phase 0.4 amendment" section.
4. `~/.claude/projects/C--Users-adame-CJC/memory/project_abng.md` —
   prepend the new "Phase 0.4 deliverable" block.
5. `~/.claude/projects/C--Users-adame-CJC/memory/MEMORY.md` — replace
   the ABNG entry with a one-line pointer to the updated project memory.

Then, if Phase 0.5 is starting, write a
`docs/abng/PHASE_0_5_IMPLEMENTATION_PROMPT.md` for the next phase.

---

## Phase 0.5+ scope (not for this session, listed for context)

After Phase 0.4, the remaining work is:

### 0.5 — Chess-RL retrofit

- Replace value head only first (Phase 0.5a)
- Then policy head (Phase 0.5b)
- Uncertainty-gated A2C/PPO bootstrap: when `epistemic > τ_abstain`,
  fall back to plain MC return instead of bootstrapping a
  confidently-wrong critic
- End-to-end determinism gate: chess-rl-v2's `weight_hash` must
  remain bit-identical for the non-ABNG comparison runs

---

*Reminder: ABNG's value isn't the math, it's the audit. Don't let
Phase 0.4's quality refinements silently break the chain.*
