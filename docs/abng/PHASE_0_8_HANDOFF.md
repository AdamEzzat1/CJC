# CJC-Lang ABNG Phase 0.8 — Big-Data Scaling + Wire-Format Evolution

**Date stamped:** 2026-05-09
**Branch (recommended):** `claude/abng-phase-0-8` (forked from `master @ 96ea80e`)
**Phase 0.7 HEAD at handoff:** `96ea80e` (Item 4 fused `abng_train_step` builtin)
**Master state at handoff:** `96ea80e` — Phase 0.6 + Phase 0.7 are fully merged.
**To continue:** start a new Claude Code session inside a fresh worktree forked from `master @ 96ea80e` and paste the prompt at the end of this document (`## Recommended next-session prompt`).

---

## Status update — 2026-05-11

**Master HEAD at this update:** `5e62967` (B4 columnar AuditLog).
**Wire format:** v13 (`b"ABNG\x0D"`) unchanged. All 28 SHA-256 canaries still valid.

### Phase 0.8 items shipped

| Item | Commit | Branch | What it added |
|---|---|---|---|
| **B1** Memory-mapped audit log replay | `a524992` | `claude/flamboyant-fermi-208353` | `replay_mmap` / `replay_mmap_with_outcome`; new `DecodeError::Io` variant. |
| **C1** Parallel `route_to_leaf_batch` | `29604f4` | `claude/flamboyant-fermi-208353` (canonical from `claude/elastic-kirch-db47b2:faf123c`) | New `route_to_leaf_batch_par(xs, n, n_threads)` via `std::thread::scope`. No new deps; `descend_with_children` Sync view of routing data. |
| **D2a** SIMD-friendly Kahan accumulators | `ac9b576` | `claude/flamboyant-fermi-208353` | `KahanAccumulatorF64x4` / `F64x8` in `cjc-repro`. Plain-array `[f64; N]` for cross-platform determinism. **BlrState refactor (D2b) deferred to v14** — reordering BLR posterior adds would change `canonical_bytes` and break canaries. |
| **B2** Streaming snapshot encode | `59b608b` | `claude/flamboyant-fermi-208353` | `serialize_into(g, w: &mut dyn Write)`. 6 internal `encode_*` helpers refactored to `Write` sinks. `serialize()` becomes a thin wrapper. |
| **B3** zstd-compressed snapshot | `5a32d8d` | `claude/flamboyant-fermi-208353` | Optional `compression` Cargo feature. New magic `b"ABNGZ\x01"` (orthogonal to v13/v14). `serialize_compressed`, `serialize_into_compressed`. `replay` auto-detects + dispatches. |
| **B4** Columnar AuditLog | `5e62967` | `claude/flamboyant-fermi-208353` | `Vec<AuditEvent>` → `AuditLog` (SoA per-column Vecs). Zero-copy `previous_hashes()` / `new_hashes()` slices — prerequisite for A3 (Merkle) + C2 (parallel verify). |
| **C3** Per-thread arena observability | `34047f5` | `claude/elastic-kirch-db47b2` (not merged here) | `arena_graph_count` / `arena_next_id` observability. 7 isolation tests + concurrent training scaling bench. |
| **D1** Cholesky factor caching | `8c86f82` | `claude/elastic-kirch-db47b2` (not merged here) | Caches Cholesky factor in `BlrState`; 2-10× faster `predict()` at d≥8. |

### Items remaining

**Phase 0.8 is now COMPLETE as of branch `claude/abng-v14-wire-format`.** All 11 numbered items have shipped:

| Item | Commit | Branch | What it added |
|---|---|---|---|
| **A4** Compact `AdaptiveChildren` snapshot | `d2ce894` | `claude/abng-v14-wire-format` | Sparse `Node48`/`Node256` encoding (`count u32 + (byte, child_id)×count` replaces dense `index[256] + slots[]`). ~228 B saved per Node48 at 25 children. Path B (encoding-only) — zero canary impact. |
| **A1** Packed lower-triangular BLR precision | `d61a366` | `claude/abng-v14-wire-format` | `d(d+1)/2` entries on disk vs `d×d` (full matrix). ~960 B saved per BlrState at d=16. Path B (encoding-only) — zero canary impact. |
| **A2** Fused `AuditKind::TrainStep` audit | `457c3c1` + `eb18f1a` | `claude/abng-v14-wire-format` | New audit kind (tag `0x1E`) collapses `BlrUpdated + BeliefUpdate` into one chain event for `train_step` workloads. 38% audit-log size reduction. Path A — 6 canaries re-locked. |
| **A3** Merkle-indexed audit chain | `2a5801d` | `claude/abng-v14-wire-format` | `MerkleTree::build`/`proof`/`verify_proof`; `Graph::merkle_root` + `merkle_tree`; snapshot trailer (`0x01` + 32-byte root). External O(log N) inclusion proofs. Zero canary impact (witness column). |
| **C2** Parallel `verify_chain` via Merkle segments | `43bf29e` | `claude/abng-v14-wire-format` | `verify_chain_par(n_threads)` via `std::thread::scope`, threshold-gated at 10K events. Zero canary impact (read-only). |
| **D2b** SIMD Kahan refactor of `BlrState::update` | `7227e30` | `claude/abng-v14-wire-format` | Replaces scalar Kahan with `KahanAccumulatorF64x4` for the n-axis reductions (xtx, xty, yty). Bit-identical at n ≤ 4, bit-different at n ≥ 5. **3 canaries re-locked** — only the 3 "*_scaled_cjcl" demos with batched n ≥ 5 per leaf shifted; everything else (post-A2 train_step demos, trigger demos) stayed byte-stable. |
| **D3** Fused matvec kernel for `Λ_old · μ_old` | `2803242` | `claude/abng-v14-wire-format` | `matvec_plus_xty_kahan` free helper with scalar path at d ≤ 4 (bit-identical to pre-D3) and F64x4 fused path at d ≥ 8 with d % 4 == 0. Zero canary impact (every locked canary uses d=4). |

**Bonus items shipped on a different branch (not merged here):**

| Item | Commit | Branch |
|---|---|---|
| **C3** Per-thread arena observability | `34047f5` | `claude/elastic-kirch-db47b2` |
| **D1** Cholesky factor caching | `8c86f82` | `claude/elastic-kirch-db47b2` |

**Final Phase 0.8 canary state: 9 canaries re-locked (6 from A2, 3 from D2b); 19 canaries unchanged.**

The original handoff said "all seven remaining items depend on the v14 wire bump" and that proved true: A1/A2/A3/A4/D2b/D3 all shipped under v14 magic `ABNG\x0E`. C2 didn't actually require the wire bump (it's a read-only API) but rode along on the same branch for narrative coherence.

### Honest macro impact — PINN scale bench

5 trials each at Phase 0.7 (`96ea80e`) and Phase 0.8 (`5e62967`), `bench/abng_pinn_scale` on Windows MSVC laptop (no thermal pinning, no CPU isolation):

| n_collocation | Phase 0.7 min / median | Phase 0.8 min / median | min×× / median×× |
|---|---:|---:|---:|
| 1,000 | 23.8 ms / ~34.5 ms | 11.2 ms / ~27 ms | **2.13× / 1.28×** |
| 10,000 | 174.5 ms / ~411 ms | 130.1 ms / ~350 ms | **1.34× / 1.18×** |
| 100,000 | 2,636 ms / ~3,050 ms | 1,002 ms / ~2,239 ms | **2.63× / 1.36×** |

**Honest reading:** Phase 0.8 produces a 1.2-1.5× macro speedup on the PINN bench (median) — modest, real, and outside the trial-to-trial variance band. The min-of-trials number is variance-inflated and should not be cited.

**L2 error byte-identical across all trials and both phases** at every scale (4.192e-2 / 9.351e-3 / 1.800e-3). The wire-format determinism contract held cleanly through B1+C1+B2+D2a+B3+B4.

**Why isn't the speedup bigger?** The Phase 0.8 micro-bench wins (3-4× SIMD Kahan, 2.5× parallel route, 4.3× zstd-friendly audit) don't directly target the PINN training inner loop. `route_to_leaf_batch` isn't called per-row in PINN; SIMD Kahan isn't yet wired into `cjc-abng`; the audit log work is overhead-neutral. The modest speedup likely traces to **compiler-level** effects from B4's columnar `AuditLog` reshaping a hot struct (8 small `Vec` pushes vs. 1 AoS `Vec` push), which the optimizer inlines differently.

### Phase 0.8 lessons learned

1. **"Wire format unchanged" is the most important phrase.** Every commit in Phase 0.8 held this line, which is why merging six in-place refactors didn't break a single canary. The decision to defer D2b/D3 to v14 (because they'd reorder BLR `canonical_bytes`) was the load-bearing one — see the D2a commit (`ac9b576`) for the analysis.

2. **`std::simd` is still nightly.** The doc claimed it was stable on Rust 1.84+; it isn't (`portable_simd` is nightly-only). Plain `[f64; N]` arrays with auto-vectorization gave 3-4× SIMD wins **and** byte-identical cross-platform results without any unsafe or per-arch code. The compiler is doing more than people credit it with.

3. **Threshold-gating parallel work is mandatory at sub-microsecond per-row cost.** Rayon's per-task setup at n=1024 made the parallel `route_to_leaf_batch` **33× slower** than sequential. Adding a 10,000-row threshold (and then settling on `std::thread::scope` for explicit thread-count control) was the difference between "items must be reverted" (Phase 0.7 discipline) and "ships as a clear win at scale."

4. **B4 (columnar AuditLog) is the v14 enabler.** A3 (Merkle index) and C2 (parallel verify) both need `&[[u8; 32]]` slices over chain heads. Without columnar storage they'd have to materialize the slice from an AoS `Vec<AuditEvent>` on every call. Doing B4 first means A3/C2 can land as simple consumers of `audit.new_hashes()`.

5. **Macro-bench variance dominates micro-bench wins.** A single-run PINN bench on a thermally-unpinned laptop has 2-3× trial variance. The Phase 0.8 micro wins (3-4× on isolated primitives) translate to 1.2-1.5× macro improvements — but only after min-of-trials methodology beats the noise floor. **Future handoff docs should specify min-of-N + warmup in their bench prescriptions.**

### Workflow notes for the v14 follow-up branch

- **Fork from `master @ 5e62967`** (or whatever HEAD is after this update commits). Do NOT fork from the original handoff's `96ea80e` — you'd lose six shipped items.
- **Ship A1+A2+A3+A4 as one v14 commit batch.** The doc's original guidance still applies: one wire-format bump captures all four. Doing them in four separate commits is fine, but they should not land on master until the canary-relock is complete.
- **Canary re-lock is mandatory** for any v14 commit. Re-run the 28 canary workloads, record new hex, update demo files, document the migration in `docs/abng/V14_MIGRATION.md`.
- **C2 follows A3** — it needs the Merkle index to slice. D2b/D3 follow the v14 bump independently.

### Workflow for re-running the macro benches

The PINN bench (`bench/abng_pinn_scale`) has no warmup or min-of-trials — single-run-per-scale. For honest comparison:

```bash
# At each commit you want to measure, run 5+ trials and take the min:
for i in 1 2 3 4 5; do cargo run -p abng-pinn-scale --release 2>&1 | grep n_coll; done
```

Future improvement: extend the bench to do min-of-7 internally (matches the `bench_kahan_simd_vs_scalar` and `bench_serialize_compressed` convention).

---

## What's done — Phase 0.7 baseline

### Surface

| Property | Value |
|---|---|
| Snapshot magic | `b"ABNG\x0D"` (v13 — UNCHANGED through all of Phase 0.7) |
| Builtin count | **79** `abng_*` arms in `dispatch.rs` (+1 from 0.6: `abng_train_step`) |
| Audit kinds | **30** (tags `0x00..0x1D`, frozen) |
| `DecisionPolicy` | 14 thresholds (112 bytes, unchanged) |
| Per-node state | unchanged from v13 |
| CLI surface | unchanged |

### Phase 0.7 measured perf wins (6 commits, all canary-preserving)

| Commit | Item | Target operation | Measured speedup | Predicted speedup |
|---|---|---|---:|---|
| `7bd8b2f` | **B** — `QuantileCodebook::encode_into` | route encode @ d=8 | **2.37×** | 50–150 ns/route |
| `b8990a0` | **A** — `AuditEvent::write_payload` | `verify_chain` (10k events) | **1.25×** | 10–20% on replay |
| `f19da46` | **F** — stack-array `advance_stats_chain` | per-observation chain advance | **1.24×** | 30–80 ns/observe |
| `f56628d` | **C** — streaming SHA-256 hasher | every audit chain step | **1.18×** | 100–200 ns/event |
| `95d73ee` | **E** — `AdaptiveChildren::iter_sorted` | child walks (routing/decisions) | zero-alloc (no isolated bench) | feeds into above |
| `96ea80e` | **Item 4** — fused `abng_train_step` | per-row training | **1.17×** at Rust API | ~5 µs/row at language level |
| `711762d` | bench infra | `blr_state_update_direct` isolation bench | (foundational) | — |

**Cumulative effect on a representative training row:** route + train_step + audit emit + chain hash + observation Welford → roughly 4–5 µs saved per row. At 10⁵ rows = ~400–500 ms saved per training run, plus 25%+ on chain verification.

### Key architectural lesson from Phase 0.7

**Caller-provided buffers reliably win; struct-field scratch reliably loses on Windows MSVC.** Three patterns succeeded:
- `&mut Vec<u8>` parameters (B, A, E in routing path)
- `&mut Sha256` streaming state (C)
- Fixed-size `[u8; N]` stack arrays (F)

One pattern was reverted (Item 1 — `BlrState.xtx_buf` field): every access through `&mut self.field` defeated the optimizer's aliasing analysis, producing a measurable *slowdown*. Phase 0.8 work should preserve this idiom: any new buffer reuse should travel as a parameter to a function, never as a field on a struct that's the receiver of an `&mut self` method.

### Test counts at end-of-0.7 (the floor — Phase 0.8 must keep these passing)

| Gate | Count |
|---|---:|
| `cargo test -p cjc-abng --lib` | **293** (was 277 pre-0.7) |
| `cargo test -p cjc-snap --lib` | **97** (was 90 pre-0.7) |
| `cargo test --test abng` | **487** (was 483 pre-0.7) |
| `cargo test --test prop_tests abng_decision` | **8 × 256 cases** |
| `cargo test --test bolero_fuzz abng_decision` | **8** |
| `cargo test -p cjc-cli --test abng_cli_integration` | **43** |
| `cargo test -p cjc-cli --lib toml_min` | **29** |
| 4 Rust application demos | 13+10+16+19 = **58** |
| 9 Phase 0.5 CJC-Lang demos | **81** |
| 5 Phase 0.6 trigger demos | **39** |
| 8 Phase 0.6 scaled demos | **55** |
| `cargo test --workspace --release --lib` | **2,465** (floor was ≥2,440) |

### Locked SHA-256 canaries (28 — UNCHANGED through Phase 0.7)

Every one verified byte-identically on every Phase 0.7 commit. **These remain the integrity contract for Phase 0.8.**

### Frozen contracts as of v13 (still frozen — but Phase 0.8 may bump to v14)

- `MAGIC = b"ABNG\x0D"` (v13)
- `b"ABNG-PRED\x02"` (prediction snapshot)
- Audit-kind tag bytes `0x00..0x1D`
- `ChildrenKind` codes `0..5`, `Activation` codes `0x00..0x08`
- `action_counts` is `[u64; 6]`
- `N_THRESHOLDS = 14`
- `NodeStats::canonical_bytes` is **32 bytes**
- All per-node state mounting order

**Phase 0.8 EXPLICITLY AUTHORIZES a v14 wire-format bump** if the perf or scaling benefit justifies it. New audit kinds, packed-precision matrix, Merkle-indexed chains — all of these were deferred from Phase 0.7 because they required a v14 bump, and they are now in scope.

---

## Phase 0.8 themes

Phase 0.7's theme was "make existing workloads faster without breaking anything." Phase 0.8's theme is **two-pronged**:

1. **Big-data scaling** — make ABNG usable on workloads with 10⁶+ samples, 10⁴+ nodes, GB-scale audit logs. The current state is fine for ≤10⁵ samples but starts to choke beyond that.
2. **Wire-format evolution** — finally bump to v14 to unlock perf wins that Phase 0.7 had to defer. v14 will be designed as a *forward-compatible* superset of v13: replay tools can decode both, snapshot writers always emit v14.

The 11 items below are organized into 4 tracks. Pick by leverage × risk × dependency order.

---

## Track A — Wire-format v14 (one bump, several wins)

Doing one v14 bump that captures multiple deferred wins is much cheaper than doing serial bumps. These four items should ship as one logical v14 release.

### Item A1 — Packed lower-triangular `BlrState.precision`

**The change.** Today `BlrState.precision` serializes as `d × d × 8 bytes` (full f64 matrix). Since the matrix is symmetric positive-definite by construction, only `d(d+1)/2` entries are needed. At d=4 this saves 6 floats = 48 bytes per BLR head per node. At d=16, saves 120 floats = 960 bytes/node. Across 10⁴ nodes that's ~10 MB on the snapshot.

**Determinism.** Identical posterior math; the diagonal-and-below ordering of canonical bytes must be specified explicitly so cross-platform CI catches drift.

**Cost.** ~150 LOC. New encode/decode in `serialize.rs::decode_blr_state`. Update `BlrState::canonical_bytes`. Update the size assertion in `canonical_bytes_size` test.

**Risk.** Wire-format change → bumps magic to `ABNG\x0E`. All 28 canaries will need to be re-locked at v14. The replay path must accept BOTH magics for one phase (deprecate v13 in Phase 0.9).

### Item A2 — Fused `AuditKind::TrainStep` (one chain step per row)

**The change.** Phase 0.7 Item 4 shipped a fused *Rust method* but had to emit two audit events (BlrUpdated + BeliefUpdate) per call to preserve v13. v14 introduces `AuditKind::TrainStep { leaf, value, batch_hash }` — one event per row, one chain hash per row.

**Wins.**
- Halves the audit log size for training-heavy workloads (~10 MB saved per 10⁵-row run).
- Halves the chain-hash compute (one SHA-256 per row instead of two).
- Stacks with Phase 0.7's streaming SHA-256 to give ~30% on per-row training cost.

**Cost.** ~120 LOC. New variant in `AuditKind` (tag `0x1E`), new payload encoding, new replay handling, parity test against the 3-event sequence (assert that `train_step` emits the same *graph state* even though the audit log shape differs).

**Risk.** Replay tools must handle both shapes. The 28 canaries need re-locking under v14 because the audit log structure changes for any workload that uses `train_step` (i.e., chess RL, tabular GP demos).

### Item A3 — Merkle-indexed audit chain

**The change.** Today's audit chain is a linear hash list — verifying event N requires hashing all events 0..N. v14 introduces a Merkle tree layered over the linear chain: every 2^k-th event also stores a Merkle node hash, so verifying any prefix is `O(log N)` instead of `O(N)`.

**Wins.**
- `verify_chain` on a 10⁶-event log goes from O(N) ≈ 1 second to O(log N) ≈ a few microseconds.
- Enables "verify-from-checkpoint" workflows: load a snapshot, verify chain tail, trust the prefix because the Merkle root is signed.
- Enables parallel verify: each Merkle subtree can be verified independently across threads.

**Cost.** ~300 LOC. New `merkle_index: Vec<[u8; 32]>` field on `AdaptiveBeliefGraph`, new audit kind for Merkle-checkpoint events, replay path updates, snapshot encode/decode.

**Risk.** Largest of the v14 items. The Merkle layer must be deterministic across platforms, which means the order in which Merkle nodes are computed has to be specified. Cross-platform CI gate is mandatory.

### Item A4 — Compact `AdaptiveChildren` snapshot for Node48/Node256

**The change.** Today Node48 serializes as `index[256] u8` + `slots: Vec<Option<NodeId>>`. For sparse Node48 nodes (say 25 children), the 256-byte index is mostly `0xFF` filler. v14 introduces a sparse encoding: `n_children u32` + `(byte, child_id) × n_children`. At Node48 with 25 children, saves ~150 bytes/node.

**Wins.** ~10–30% reduction in snapshot size for trees with many medium-density nodes.

**Cost.** ~80 LOC. New encoding branch in `serialize.rs`, decode-side detection (the encoder picks the smaller of dense vs sparse).

**Risk.** Low — the in-memory representation doesn't change, only the serialized form.

---

## Track B — Big-data ergonomics (no wire-format change)

These items make ABNG handle large workloads without changing the v13 wire format. They can ship before, after, or alongside Track A.

### Item B1 — Memory-mapped audit log replay

**The change.** Today `replay()` reads the entire snapshot blob into RAM. For a 10⁶-event log (~100 MB) that's wasteful — most events are read once, sequentially, and could come from a memory-mapped file. v14-or-not: snapshot format is unchanged; this is a *reader-side* optimization.

**Wins.**
- Replay starts immediately (no full-blob read). Streaming-replay use cases become tractable.
- RAM peak during replay drops from O(snapshot_size) to O(working_set), typically 100× smaller.

**Cost.** ~150 LOC. Add `replay_mmap(path: &Path)` alongside `replay(bytes: &[u8])`, both feed into the same internal loop.

**Risk.** Low. mmap on Windows requires careful handling around concurrent writes (which we don't do — snapshots are write-once read-many).

### Item B2 — Streaming snapshot encode (no full-buffer materialization)

**The change.** Today `serialize()` builds a `Vec<u8>` containing the entire snapshot. For a 50 MB snapshot, that's a single 50 MB allocation. Streaming encode writes into a `&mut dyn Write` (or a pre-sized `Vec<u8>` in chunks).

**Wins.**
- Lets the caller pipe the snapshot directly to a file or socket without materializing it.
- Removes the peak-memory spike during snapshot save.

**Cost.** ~200 LOC. Refactor `serialize` into `serialize_into(g: &Graph, w: &mut dyn Write)`.

**Risk.** Medium — the existing `serialize() -> Vec<u8>` is the public API, so we'd add `serialize_into` alongside (`serialize` becomes a thin wrapper). All existing callers continue to work.

### Item B3 — Compressed snapshot format (zstd, optional)

**The change.** Audit-log payloads are highly repetitive (every event has the same 8+8+4+1=21 byte header). zstd compression on the audit-log section typically gives 5–10× shrink. The codebook and per-node states are less compressible.

**Wins.**
- Snapshot-on-disk size: 50 MB → ~10 MB for typical workloads.
- Faster I/O for replay-from-disk.

**Cost.** ~100 LOC. Add an optional `zstd` dependency (gate behind a Cargo feature). Snapshot magic stays `\x0D` for uncompressed, gain a new `\x0E`-prefixed magic only when compression is enabled.

**Risk.** Adds a dependency. zstd has well-tested deterministic mode; verify cross-platform output.

### Item B4 — `AuditEvent` columnar storage (SoA in memory)

**The change.** Today the in-memory audit log is `Vec<AuditEvent>` (array-of-structs). For batch operations (Merkle indexing, verify-segment-parallel, snapshot encode), columnar is much friendlier: `Vec<u64>` for seq, `Vec<u32>` for node_id, `Vec<[u8; 32]>` for chain_heads, `Vec<u8>` for variant tags + payload offsets.

**Wins.**
- Enables fast `Vec<chain_head>` slicing for Merkle layer construction (Item A3).
- Better cache utilization on full-log scans.

**Cost.** ~400 LOC. Significant refactor — every `g.audit.push(event)` and `for event in &g.audit` callsite touches.

**Risk.** Medium-high. The behavioral surface is unchanged but the internal API churns. Must NOT change the wire format.

---

## Track C — Concurrency (with determinism preserved)

Big workloads benefit from parallelism, but ABNG's determinism contract is per-thread + per-row + Kahan-ordered. Phase 0.8 introduces parallelism only where the math associates correctly under reordering.

### Item C1 — Parallel `route_to_leaf_batch`

**The change.** Each row's routing is independent (read-only walk over the radix tree). Use `rayon::par_iter` over batch rows; collect leaf ids in row-order. No determinism risk because each row's leaf id is a pure function of `(graph, row_input)`.

**Wins.** Near-linear speedup over thread count for batch routing on large `n`. Inference workloads (where you route 10⁶ test points without observing) become CPU-bound on cores instead of single-threaded interpreter overhead.

**Cost.** ~50 LOC. Add `rayon` dependency, gate behind a `parallel` Cargo feature.

**Risk.** Very low. The routing path is read-only on the graph.

### Item C2 — Parallel `verify_chain` via Merkle segments

**Depends on:** Item A3 (Merkle-indexed audit chain).

**The change.** With a Merkle index, the chain verification splits into independent subtrees. `par_iter` over the segments, verify each in parallel.

**Wins.** Linear speedup with thread count on chain verification, which is currently the bottleneck for snapshot round-trip on large logs.

**Cost.** ~80 LOC after A3 lands.

**Risk.** Low after A3.

### Item C3 — Per-thread arena for transient training scratch

**The change.** Move the existing per-thread `RefCell<BTreeMap<i64, Graph>>` arena into a `thread_local!` arena that supports per-thread training contexts. Different threads can train on different graphs concurrently with no contention.

**Wins.** Multi-graph training workloads (e.g., ensembles, hyperparameter sweeps) scale linearly.

**Cost.** ~60 LOC. The arena infrastructure already exists; this just makes it explicitly per-thread.

**Risk.** Low. Doesn't change the math at all.

---

## Track D — Numerical kernels (deferred from Phase 0.7)

These items were considered for Phase 0.7 but deferred for risk or scope reasons. Phase 0.8 picks them up.

### Item D1 — Cholesky factor caching with rank-1 update

**The change.** `BlrState::update` recomputes a fresh Cholesky on every call (`O(d³)/3` flops). Cache the factor `L` and update via Givens rotations on each new row (`O(d²)`).

**Wins.** ~30% on `update` math at d=4, ~70% at d=16+. For PINN-style workloads with d=32 features, this is the dominant cost.

**Cost.** ~250 LOC. New `BlrState.cached_l: Option<Tensor>` field (NOT in `canonical_bytes`). Rank-1 Givens update math. Invariance test: `state_after_update.precision == cholesky_factor(L_after_rank1).precision_reconstructed`.

**Risk.** Medium-high. The rank-1 path's intermediate floats can diverge in ULPs from the direct decomposition's floats — but only when `predict()` is called. For determinism, EITHER (a) keep the rank-1 path bit-equal by carefully replicating the operation order, OR (b) recompute the Cholesky from precision before any `predict()` call (caching only between updates). Option (b) is safer and still wins ~70% on training-only loops.

### Item D2 — SIMD `KahanAccumulatorF64x4` / x8

**The change.** Phase 0.7 handoff Item 3 — for d ≥ 8, the Cholesky inner loops become the bottleneck. SIMD-vectorize via `std::simd` (stable on Rust 1.84+).

**Wins.** ~3× on Cholesky math at d=32. Pure ABNG demos use d=4 so don't benefit; PINN workloads with high-dim features do.

**Cost.** ~150 LOC. New `KahanAccumulatorF64x4` / x8 in `cjc-repro/src/kahan.rs`. Refactor `BlrState::update` inner loops to use SIMD when `d % 4 == 0`.

**Risk.** SIMD lane ordering is the determinism gate. Cross-platform CI must verify lane-order is x86_64 == aarch64 == arm64 byte-identically.

### Item D3 — Fused matmul kernel for `lambda_old · mu_old`

**The change.** Inside `BlrState::update`, the term `Λ_old · μ_old` is a d×d * d matrix-vector multiply (16 muls + 12 adds at d=4). Hand-rolled with no FMA. Auto-vectorization is fine at d=4 but should be explicitly tested at d=16+.

**Wins.** Modest at d=4 (~5%); meaningful at d=16+.

**Cost.** ~50 LOC. Restructure the inner loop, add a unit bench at multiple d values.

**Risk.** Low. Same scalar arithmetic, just reordered for cache + SIMD.

---

## Recommended order

```
Phase 0.8a (big-data ergonomics, no wire-format change):
   Item B1 (mmap replay)  →  Item C1 (parallel route_to_leaf_batch)  →  Item C3 (per-thread arena)
                                                                              ↓
                                                    measured on chess RL + PINN scaled demos
                                                                              ↓
Phase 0.8b (numerical kernels, no wire-format change):
   Item D1 (Cholesky caching)  →  Item D2 (SIMD Kahan) — gated by cross-platform CI
                                                              ↓
                                            measured on tabular_scaled + lineage_scaled
                                                              ↓
Phase 0.8c (wire-format v14):
   Item A1 (packed precision)  →  Item A2 (fused TrainStep audit)  →  Item A3 (Merkle index)
                                                                              ↓
                                       Item A4 (sparse Node48/256)  + Item C2 (parallel verify)
                                                                              ↓
                                                     re-lock all 28 canaries at v14
                                                                              ↓
Phase 0.8d (compression, optional):
   Item B3 (zstd snapshot)  +  Item B2 (streaming snapshot encode)
```

Ship 0.8a first. The big-data items unlock realistic workload sizes that 0.8b and 0.8c can be measured against. v14 (0.8c) is the highest-leverage single bump; do it once, do it carefully, do it last.

---

## Verification loop (run before merge of EVERY commit)

Same as Phase 0.7 plus a v14-canary update sequence (when v14 lands):

```bash
cd C:/Users/adame/CJC/.claude/worktrees/abng-phase-0-8

# Phase 0.7 baseline gates (FLOOR — must remain at or above)
cargo test -p cjc-abng --lib                       # ≥ 293
cargo test --test abng                              # ≥ 487
cargo test --test prop_tests abng_decision          # ≥ 8 × 256
cargo test --test bolero_fuzz abng_decision         # ≥ 8
cargo test -p cjc-cli --test abng_cli_integration   # ≥ 43
cargo test -p cjc-cli --lib toml_min                # ≥ 29
cargo test -p cjc-snap --lib                        # ≥ 97

# 26 demo files (canary verification)
# (see Phase 0.7 handoff for the complete list)

# Broader workspace gate
cargo test --workspace --release --lib              # ≥ 2,465
```

For Track C (concurrency) items, add a determinism gate:

```bash
# Run the same workload 5× under parallel features and assert chain_heads are identical
cargo test --test abng_parallel_determinism --release --features parallel
```

For Track A (v14) items, the canary-relock workflow:

```bash
# 1. Run canary workloads, record the new v14 chain_heads
cargo test --test test_abng_pinn_uncertainty -- --nocapture | grep CANARY
cargo test --test test_abng_tabular_gp -- --nocapture | grep CANARY
# ... (28 canaries total)

# 2. Update CANARY_HEX in each demo file
# 3. Re-run gates — every demo must pass with the new locked hex
# 4. Document the v13→v14 migration in docs/abng/V14_MIGRATION.md
```

---

## Documentation updates expected

Mirror the Phase 0.7 documentation discipline:

| Doc | Update |
|---|---|
| `docs/abng/ABNG_CURRENT_ARCHITECTURE.md` | Update header (builtin count, magic byte if v14 ships); §1 phase status table; §3.4 builtin surface; Appendix A file map; Appendix B test counts |
| `docs/abng/PHASE_0_8_DESIGN.md` (NEW) | Post-hoc design note covering all shipped items, with per-item bullets, test counts, "What's *not* in 0.8" section |
| `docs/abng/V14_MIGRATION.md` (NEW, if v14 ships) | v13→v14 migration guide for snapshot users (replay tools, snapshot importers, etc.) |
| `~/.claude/projects/C--Users-adame-CJC/memory/project_abng.md` | Prepend "Phase 0.8 deliverable" block above the Phase 0.7 block |
| `~/.claude/projects/C--Users-adame-CJC/memory/MEMORY.md` | Update the one-line ABNG entry to reflect Phase 0.8 status |

---

## Recommended next-session prompt (stacked role group prompt)

Paste the following into a fresh Claude Code session inside the `abng-phase-0-8` worktree:

```
# CJC-Lang ABNG Phase 0.8 — Stacked Role Group Prompt

## ROLE

You are a stacked systems team working inside the CJC-Lang
(Computational Jacobian Core) compiler repository, specifically
inside the ABNG (Adaptive Belief Network Graph) sub-system at
`crates/cjc-abng/`. You are continuing the ABNG roadmap from Phase
0.7 (already merged to master at HEAD `96ea80e`).

You consist of:

1. **Lead ABNG Architect** — owns ABNG's external contract: which
   builtins exist, their semantics, the `Value` enum's invariance,
   and the snapshot wire format. Phase 0.8 EXPLICITLY AUTHORIZES a
   v14 wire-format bump (Track A items). v14 must be a forward-
   compatible superset of v13: replay tools accept BOTH magics
   during the transition; snapshot writers always emit v14.

2. **Compiler / Interpreter Performance Engineer** — owns the
   Lexer → Parser → AST → HIR → MIR → Exec data flow.

3. **Big-Data Engineer** — owns memory footprint, I/O scaling,
   and out-of-RAM workflows. Phase 0.8's Track B items (mmap
   replay, streaming encode, compression) are this engineer's
   beat. Targets: 10⁶+ samples, 10⁴+ nodes, GB-scale audit logs.

4. **Concurrency Engineer** — owns parallelism with deterministic
   output. Phase 0.8's Track C items (parallel route, parallel
   verify, per-thread arena) need rigorous determinism gates:
   same input + same seed must produce byte-identical chain_heads
   regardless of thread count.

5. **Numerical Computing Engineer** — owns deterministic
   floating-point. Phase 0.8's Track D items (Cholesky caching,
   SIMD Kahan, fused matmul) sit here. The 28 locked SHA-256
   canaries are the gate (or in Track A, the v14-relocked canaries).

6. **Determinism & Reproducibility Auditor** — enforces bit-
   identical output across runs AND platforms AND thread counts.
   The cross-platform CI matrix (Linux + Windows + macOS) is the
   gate; for Track C, a determinism-vs-thread-count CI test is
   added. Any regression on the canaries is a release-blocker.

7. **QA Automation Engineer** — owns the gate suite:
   - `cargo test -p cjc-abng --lib` (≥ 293)
   - `cargo test -p cjc-snap --lib` (≥ 97)
   - `cargo test --test abng` (≥ 487)
   - `cargo test --test prop_tests abng_decision` (≥ 8 × 256)
   - `cargo test --test bolero_fuzz abng_decision` (≥ 8)
   - `cargo test -p cjc-cli --test abng_cli_integration` (≥ 43)
   - All 26 demo files (≥ 233 tests)
   - `cargo test --workspace --release --lib` (≥ 2,465)

## PRIME DIRECTIVES

1. **The 28 SHA-256 canaries are the integrity gate.** Every Track
   B/C/D commit must verify them byte-identically against v13. Track
   A commits re-lock them at v14 — the relock procedure is a one-
   shot per item and must include a parity test against v13's
   chain_head computed from a v13 snapshot.

2. **No hidden non-determinism.** Kahan/SplitMix64/BTreeMap discipline
   holds. SIMD Kahan (D2) MUST verify lane-order determinism across
   x86_64 / aarch64 / arm64. Parallel ops (C1, C2) MUST verify chain
   determinism across thread counts.

3. **Preserve `Value` enum layout.** No new `Value` variant.

4. **Both executors must agree.** Every new builtin works in both
   `cjc-eval` (AST-walk) AND `cjc-mir-exec` (MIR register-machine)
   with byte-identical output. Parity tests in `tests/abng/parity.rs`
   are the regression gate.

5. **Performance claims must be measured.** Every perf item ships
   with a bench number. Items that don't show measured speedup are
   reverted (Phase 0.7's Item 1 was reverted under this rule).

6. **Caller-provided buffers, not struct-field scratch.** Phase 0.7's
   architectural lesson: any new buffer-reuse pattern travels as
   `&mut Vec<u8>` parameters, NOT as fields on a struct that's the
   receiver of `&mut self` methods. The Windows MSVC optimizer
   pessimizes the latter.

7. **All 26 Phase 0.6/0.7 demos MUST continue to pass.** Track A's
   v14 work re-locks the 28 canaries embedded in those demos; Track
   B/C/D's no-wire-format work preserves them byte-identically.

8. **v14 is the only wire-format bump in Phase 0.8.** Don't bump
   again. If a perf optimization requires v15, defer to Phase 0.9.

## SCOPE — Phase 0.8 11 items across 4 tracks

Read `docs/abng/PHASE_0_8_HANDOFF.md` for the full breakdown.

Track A (wire-format v14, one bump captures multiple wins):
  A1. Packed lower-triangular `BlrState.precision`
  A2. Fused `AuditKind::TrainStep` (one chain step per row)
  A3. Merkle-indexed audit chain (O(log N) verify)
  A4. Compact `AdaptiveChildren` snapshot for sparse Node48/Node256

Track B (big-data ergonomics, no wire-format change):
  B1. Memory-mapped audit log replay
  B2. Streaming snapshot encode (no full-buffer materialization)
  B3. Compressed snapshot format (zstd, optional Cargo feature)
  B4. `AuditEvent` columnar storage (SoA in memory)

Track C (concurrency, with determinism preserved):
  C1. Parallel `route_to_leaf_batch`
  C2. Parallel `verify_chain` via Merkle segments (depends on A3)
  C3. Per-thread arena for transient training scratch

Track D (numerical kernels, deferred from Phase 0.7):
  D1. Cholesky factor caching with rank-1 update
  D2. SIMD `KahanAccumulatorF64x4` / x8
  D3. Fused matmul kernel for `lambda_old · mu_old`

## RECOMMENDED ORDER

1. **Phase 0.8a:** B1 → C1 → C3 (big-data ergonomics, fast wins)
2. **Phase 0.8b:** D1 → D2 (numerical kernels, gated by cross-platform CI)
3. **Phase 0.8c:** A1 → A2 → A3 → A4 + C2 (one v14 bump captures all)
4. **Phase 0.8d:** B3 + B2 (compression, optional)

## TEST PLACEMENT

Same locations as Phase 0.7:
- In-crate unit tests → `crates/cjc-abng/src/<module>.rs::tests`
- Integration tests → `tests/abng/<feature>_tests.rs`
- Property tests → `tests/prop_tests/abng_decision_props.rs`
- Fuzz targets → `tests/bolero_fuzz/abng_decision_fuzz.rs`
- Benchmarks → `bench/abng_micro/main.rs` (extend) + new `bench/abng_at_scale/` if needed

NEW for Phase 0.8:
- Cross-platform CI: extend `.github/workflows/cross-platform-determinism.yml` with v14 magic check + thread-count determinism gate
- Determinism-vs-thread-count test: `tests/abng_parallel_determinism.rs` (new)
- v14 migration test: `tests/abng/v13_to_v14_migration_tests.rs` (new, only if v14 lands)

## VERIFICATION LOOP (run before EVERY commit)

```bash
cd C:/Users/adame/CJC/.claude/worktrees/abng-phase-0-8
cargo test -p cjc-abng --lib                          # ≥ 293
cargo test -p cjc-snap --lib                          # ≥ 97
cargo test --test abng                                 # ≥ 487
cargo test --test prop_tests abng_decision             # ≥ 8 × 256
cargo test --test bolero_fuzz abng_decision            # ≥ 8
cargo test -p cjc-cli --test abng_cli_integration      # ≥ 43
cargo test -p cjc-cli --lib toml_min                   # ≥ 29
# All 26 demo files (see PHASE_0_8_HANDOFF.md verification loop)
cargo test --workspace --release --lib                 # ≥ 2,465
```

## USER PREFERENCES

- **Small, stop-and-confirm units** rather than batched changes.
- **Commit per item once gates pass.** Each numbered item is a
  logical commit. Items A1–A4 may share a common v14-magic-bump
  scaffold commit, but each item's payload changes ship as
  separate commits on top.
- **Performance claims must be measured.** No asserting speedups
  without a bench number.
- **Caller-provided buffers, not struct-field scratch** —
  Phase 0.7's hard-won lesson.
- **Cross-platform CI must remain green.** Every platform on
  every commit.
- **The user's name is Adam Ezzat.** (From auto-memory.)

## START

1. Verify the baseline by running the 7 gate suites + 26 demo
   files. All should pass with the v13 floor counts from
   `PHASE_0_8_HANDOFF.md`.
2. Read `docs/abng/PHASE_0_8_HANDOFF.md` for the full per-item
   breakdown.
3. Propose Item B1 (memory-mapped audit log replay) as the first
   concrete unit of work. It's no-wire-format-change, contained
   to `serialize.rs`, and the bench can be measured against
   replay_smart_vs_naive.
4. Ask for user confirmation before starting.
```

---

## Files added or substantially modified during Phase 0.7 (for context)

To know what code surface Phase 0.8 will be touching, read these files
first — they have the most recent design context:

### Heavily modified in Phase 0.7

- `crates/cjc-abng/src/audit.rs` — added `write_payload`, refactored `compute_new_hash` to use streaming SHA-256
- `crates/cjc-abng/src/codebook.rs` — added `encode_into`
- `crates/cjc-abng/src/children.rs` — added `iter_sorted` + `ChildrenSortedIter`
- `crates/cjc-abng/src/signature.rs` — refactored `routing_observation_value` to streaming + `iter_sorted`
- `crates/cjc-abng/src/node.rs` — stack-array `advance_stats_chain`
- `crates/cjc-abng/src/graph.rs` — added `train_step` method, refactored `verify_chain` + `route_to_leaf_batch` for buffer reuse
- `crates/cjc-abng/src/dispatch.rs` — added `abng_train_step` builtin (79 total)
- `crates/cjc-snap/src/hash.rs` — added streaming `Sha256` hasher
- `bench/abng_micro/main.rs` — added 3 new benches (direct BlrState update, codebook encode paired, train_step paired, verify_chain_10k)

### New test surface in Phase 0.7

- `crates/cjc-abng/src/codebook.rs::tests` — 4 new in-crate tests
- `crates/cjc-abng/src/audit.rs::tests` — 3 new in-crate tests
- `crates/cjc-abng/src/children.rs::tests` — 7 new in-crate tests
- `crates/cjc-abng/src/signature.rs::tests` — 2 new in-crate tests
- `crates/cjc-snap/src/hash.rs::tests` — 7 new in-crate tests (streaming Sha256)
- `tests/abng/dispatch.rs` — 3 new train_step tests
- `tests/abng/parity.rs` — 1 new train_step AST↔MIR parity test

### Reference design notes

- `docs/abng/PHASE_0_7_HANDOFF.md` — predecessor of this document
- `docs/abng/PHASE_0_6_TRAINING_PIPELINE.md` — Phase 0.6 Item 8 deliverable
- `docs/abng/ABNG_CURRENT_ARCHITECTURE.md` — source-of-truth contract

---

*This handoff was generated at the end of Phase 0.7, which shipped 6
performance items (B, A, F, C, E, Item 4) with measured speedups
ranging 1.17×–2.37× and a reverted 7th item (Item 1, BlrState
struct-field scratch — measured slower on Windows MSVC, replaced
with the lessons it taught about caller-provided buffers). The merge
to master landed at HEAD `96ea80e`. The branch
`claude/abng-phase-0-8` should be forked from that commit; the next
session should commit Phase 0.8 work as one PR per item once each
item's gates pass. Phase 0.8 EXPLICITLY AUTHORIZES the v14 wire-format
bump for Track A items.*
