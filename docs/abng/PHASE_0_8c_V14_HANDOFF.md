# CJC-Lang ABNG Phase 0.8c — v14 Wire-Format Follow-Up

**Date stamped:** 2026-05-11
**Branch:** `claude/abng-v14-wire-format`
**HEAD:** `d61a366` (A1 packed BLR precision)
**Parent:** `5e62967` (B4 columnar AuditLog, on master)
**To continue:** start a new Claude Code session checked out to this branch and paste the prompt at the end of this document.

This doc is the focused continuation of [`PHASE_0_8_HANDOFF.md`](PHASE_0_8_HANDOFF.md) for the v14 wire-format work specifically. The parent handoff covers the broader Phase 0.8 strategy; this one covers what's still ahead on the v14 branch.

---

## What's done — three v14 commits

| Commit | Item | Canary impact | Notes |
|---|---|---|---|
| `0db8522` | v14 magic + dual-acceptance scaffolding | None | Bumps `MAGIC` from `b"ABNG\x0D"` to `b"ABNG\x0E"`. Adds `MAGIC_V13` + `WireVersion` enum threaded through replay. v13 archives stay readable. |
| `d2ce894` | A4: sparse Node48 / Node256 encoding | None | Replaces dense `index[256] + slots[]` with `count u32 + (byte, child_id) × count`. Saves ~228 B per Node48 at 25 children. |
| `d61a366` | A1: packed lower-triangular BLR precision | None (Path B) | Stores `d(d+1)/2` precision entries on disk (vs `d×d`). Saves ~960 B per BlrState at d=16. `canonical_bytes` unchanged → `state_hash` unchanged → canaries unchanged. |

**All 28 SHA-256 canaries embedded in demo files still verify byte-identically.** No re-lock needed yet.

**Gate counts (latest at `d61a366`):**
- `cargo test -p cjc-abng --lib --release` → 294 passed
- `cargo test --test abng --release` → 509 passed (28 canary demos verify their pre-v14 chain heads)

---

## Path A vs Path B — the load-bearing pattern

The architectural lesson from A1 that any future v14 item must respect:

**Path A — modify `canonical_bytes`:** the new packed/optimized representation is the canonical hash input. Forces re-lock of every canary that depends on the affected state. Used when there's no separation between "wire encoding" and "hash domain."

**Path B — modify snapshot encoding only:** the canonical hash input stays the v13 representation; only the on-disk bytes are optimized. In-memory state is unchanged. canaries unaffected. Used when the optimization is "smaller bytes" not "different math."

A1 used Path B. A4 has no canonical_bytes interaction (children layout was never in any hash). The doc's original "Cost. ~150 LOC. Update `BlrState::canonical_bytes`" framing implied Path A; we deliberately chose Path B because it captures the doc's stated win ("~10 MB off the snapshot") without the cost.

**The rule for the next v14 item:** if the doc says "update X::canonical_bytes," first check whether X feeds into chain_head. If it does, ask: can the optimization live in encoding only? If yes, go Path B. If no (because the math itself has changed, e.g., A2), accept the canary re-lock.

---

## What's left — A2, A3, C2, migration doc, push

### Item A2 — fused `AuditKind::TrainStep` event (THE CANARY-LOCKER)

This is the one v14 item that **cannot** use Path B because it changes audit event *content*.

**The change.** Today the per-row training sequence emits two audit events: `BlrUpdated` (BLR state changed) followed by `BeliefUpdate` (Welford observation). A2 introduces a single fused `AuditKind::TrainStep { leaf, value, batch_hash }` event (tag `0x1E`) that captures the entire row's effect in one chain step.

**Wins (from the doc):**
- Halves the audit-log size for training-heavy workloads (~10 MB saved per 10⁵-row run).
- Halves the chain-hash compute per row.
- Stacks with Phase 0.7's streaming SHA-256 for ~30% on per-row training cost.

**Why it locks canaries.** Any workload that calls `Graph::train_step(x, phi, y)` will, post-A2, emit `TrainStep` events instead of `BlrUpdated + BeliefUpdate`. The audit-event payload bytes change → `new_hash` changes → `chain_head` changes → CANARY_HEX changes.

**Estimated scope:**
- ~150 LOC in `cjc-abng`: new `AuditKind::TrainStep` variant, `tag()` arm `0x1E`, `payload_bytes` + `write_payload` arms, replay handling, parity test against the 3-event sequence.
- ~280 LOC in tests: 28 demo files × ~10 lines of CANARY_HEX update each.
- ~100 LOC: `docs/abng/V14_MIGRATION.md` (see skeleton below).

**Shipping plan.** Ship A2 + V14_MIGRATION.md + canary re-lock as **one logical commit batch** (probably 2 commits: one for the code, one for the re-lock+doc). Don't separate the canary re-lock from the code change — the gates won't pass in between.

### Item A3 — Merkle-indexed audit chain

**The change.** Layer a Merkle tree over the existing linear chain. Every 2^k-th event stores a Merkle node hash; verifying any prefix becomes O(log N) instead of O(N).

**Likely canary impact:** none, if structured as a *witness* column. The Merkle tree is computed from `audit.new_hashes()` (which B4 exposed as a zero-copy slice) but the *individual* event hashes are unchanged. The Merkle root is a new field stored alongside (not inside) the event payloads.

**Estimated scope:** ~300 LOC (largest of the v14 items).

**Reading prerequisite:** the B4 columnar AuditLog (commit `5e62967` on master) is the data layout A3 needs. `audit.new_hashes()` returns `&[[u8; 32]]` directly — no AoS-to-SoA materialization required.

### Item C2 — parallel `verify_chain` via Merkle segments

**Depends on:** A3.

**The change.** With a Merkle index in place, chain verification splits into independent subtrees. `std::thread::scope` over the segments (matching C1's no-deps pattern), verify each in parallel.

**Estimated scope:** ~80 LOC after A3 lands.

**Canary impact:** none (verify is read-only over the chain).

### `docs/abng/V14_MIGRATION.md` (NEW, ships with A2)

Skeleton:
```markdown
# v13 → v14 ABNG Snapshot Migration

## Wire-format changes
- Magic: ABNG\x0D → ABNG\x0E
- BLR precision: full d×d → packed lower-triangular d(d+1)/2 (A1)
- Node48/Node256: dense index → sparse (byte, child_id) pairs (A4)
- TrainStep audit kind 0x1E (A2)
- Merkle index trailer (A3)

## Forward-compatibility
- Readers accept BOTH v13 and v14 magic. v13 archives load through
  the legacy dense path.
- Writers always emit v14 from v0.1.x onwards.
- chain_head: BLR-touching workloads see new canary hex; all 28
  demo canaries re-locked in commit <hash>.

## Migration steps for existing v13 snapshots
1. Load via `replay(&fs::read(path)?)` -- v13 magic auto-detected.
2. Re-serialize via `serialize(&g)` -- writes v14.
3. Save back to disk. The snapshot is now v14.

## Verify the migration
...
```

### Push to origin/master

Still pending explicit user approval. Local master at `64ca68d` (Phase 0.8 v13-safe items + doc update); origin at `f4e80c1`. The v14 branch's three commits are not on master yet — that's a separate merge decision once the canary-re-lock cycle completes.

---

## Canary re-lock procedure (for A2)

The 28 canaries live in 25 test files in `tests/test_abng_*.rs`. Each file has one or two `CANARY_HEX` constants. The shape is:

```rust
const CANARY_HEX: &str =
    "d064fb08c546be1b9850bfa91f87f4aed95682aa4fb7f4533cf1ac4da0d87807";
assert_eq!(chain, CANARY_HEX, "...");
```

### Files to update

```
tests/test_abng_adaptive_triggers_cjcl.rs
tests/test_abng_calibration_cjcl.rs
tests/test_abng_calibration_scaled_cjcl.rs
tests/test_abng_compact_scaled_cjcl.rs
tests/test_abng_compress_trigger_cjcl.rs
tests/test_abng_drift_detection_cjcl.rs
tests/test_abng_drift_scaled_cjcl.rs
tests/test_abng_freeze_trigger_cjcl.rs
tests/test_abng_grow_trigger_cjcl.rs
tests/test_abng_lineage_attestation.rs
tests/test_abng_lineage_attestation_cjcl.rs
tests/test_abng_lineage_scaled_cjcl.rs
tests/test_abng_maturity_inspection_cjcl.rs
tests/test_abng_maturity_scaled_cjcl.rs
tests/test_abng_ood_detection_cjcl.rs
tests/test_abng_ood_scaled_cjcl.rs
tests/test_abng_pinn_scaled_cjcl.rs
tests/test_abng_pinn_uncertainty.rs
tests/test_abng_pinn_uncertainty_cjcl.rs
tests/test_abng_prune_trigger_cjcl.rs
tests/test_abng_split_trigger_cjcl.rs
tests/test_abng_tabular_gp.rs
tests/test_abng_tabular_gp_cjcl.rs
tests/test_abng_tabular_scaled_cjcl.rs
tests/test_chess_rl_v2_6_abng.rs
```

Total: 27 `CANARY_HEX` constants across 25 files.

### Re-lock recipe

1. **Ship the A2 code change first.** All 27 canaries will fail (they verify old v13-era hex against new v14-with-TrainStep hex). That's expected.
2. **Run each canary demo with `--nocapture`** to extract the new chain_head hex. Most demos print a line like `canary chain_head = <hex>` near the assertion. If they don't, add a `println!` temporarily.
3. **Update each CANARY_HEX const** to the new value.
4. **Re-run gates.** All 27 should now pass.
5. **Document the migration** in `V14_MIGRATION.md` — link to the commit that re-locks each canary so future readers can map v13 → v14.

The re-lock is mechanical but tedious; budget ~2 hours for the full cycle if all 27 demos exit cleanly. If any demo errors (e.g., a workload that doesn't yet support A2's new audit kind), fix the demo first, then re-derive its hex.

---

## Verification loop (every commit on this branch)

```bash
cd C:/Users/adame/CJC/.claude/worktrees/flamboyant-fermi-208353  # or wherever you check out the v14 branch

# Standard gates (must stay green):
cargo test -p cjc-abng --lib --release           # ≥ 294
cargo test --test abng --release                  # ≥ 509 (PRE-A2)
                                                  # ≥ 509 with re-locked canaries (POST-A2)

# Per-feature gates:
cargo test --test abng --release --features compression  # B3 path (currently 513)

# Workspace cross-crate safety:
cargo test --workspace --release --lib            # exit 0; floor ≥ 2,465
```

After A2 lands with the canary re-lock, the abng integration count should stay at 509 — same number of demo tests, just with new locked hex.

---

## Recommended next-session prompt (paste into a fresh Claude Code session)

```
# CJC-Lang ABNG Phase 0.8c v14 — A2 Fused TrainStep + Canary Re-Lock

## ROLE

You are continuing the ABNG Phase 0.8c v14 wire-format work. The
branch `claude/abng-v14-wire-format` has three commits in
(scaffolding + A4 + A1, none of which broke canaries). The next
item is A2 — and unlike A1/A4, A2 will re-lock all 27 SHA-256
canaries in the demo suite because it changes audit-event content.

Read `docs/abng/PHASE_0_8c_V14_HANDOFF.md` for the full context
before starting.

## PRIME DIRECTIVES (unchanged from the parent handoff)

1. Wire-format v14 is authorized; A2's content change is the
   intentional trigger for canary re-lock.
2. v13 archives must stay readable through the dual-magic
   dispatch.
3. No new `Value` enum variants.
4. Both executors (cjc-eval + cjc-mir-exec) must produce
   byte-identical output.
5. Performance claims must be measured.

## SCOPE — A2 ONLY

Ship A2 (fused `AuditKind::TrainStep`) as one logical batch:

  1. Add `AuditKind::TrainStep { leaf, value, batch_hash }` with
     tag 0x1E. Implement `payload_bytes`, `write_payload`, replay
     handling.
  2. Update `Graph::train_step(x, phi, y)` to emit ONE
     `TrainStep` event instead of `BlrUpdated + BeliefUpdate` when
     running under v14.
  3. Replay must reconstruct the same graph state from a
     `TrainStep` event as it would from the 3-event sequence.
     This is the parity test.
  4. Re-lock all 27 CANARY_HEX consts. Do this in a single commit
     AFTER the code change so the diff cleanly shows "code changed
     here; canaries shifted there."
  5. Ship `docs/abng/V14_MIGRATION.md` covering all v14 changes
     (A1 + A2 + A4 — A3/C2 will append later).

Do NOT attempt A3 or C2 in this session. They're separate items
and the canary re-lock for A2 is already a multi-file commit.

## START

1. Read `docs/abng/PHASE_0_8c_V14_HANDOFF.md` end to end.
2. Run the gate suite to confirm 294 / 509 baseline.
3. Propose the A2 code design (audit event shape, replay path,
   parity test). Ask for confirmation before starting the canary
   re-lock — the user wants to verify the design first since the
   re-lock is mechanical but irreversible-feeling.
```

---

*This doc is a focused companion to [`PHASE_0_8_HANDOFF.md`](PHASE_0_8_HANDOFF.md). The parent has the architectural context (why ABNG, what's v14, what's the wire-format contract); this doc has the operational specifics for the v14 follow-up branch. Future v14 items (post-A2) should append their notes here; future post-v14 work (v15, etc.) should start a fresh handoff doc.*
