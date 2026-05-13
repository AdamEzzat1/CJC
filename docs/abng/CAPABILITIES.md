# ABNG Phase 0.8 — Capabilities Reference

**Last updated:** 2026-05-12 (post-D2b + D3 close-out).
**Wire format:** v14 (`ABNG\x0E`).
**Phase 0.8 status:** all 11 items shipped on `claude/abng-v14-wire-format`.

This doc is the single-page reference for the new ABNG
capabilities introduced in Phase 0.8 — what each one enables,
which API surfaces it lives behind, what the demo proves about
it, and the concrete numbers it produced on the demo data.

For wire-format details (magic bytes, audit-event encoding,
backward compatibility), see [`V14_MIGRATION.md`](V14_MIGRATION.md).
For the design history and item-by-item rationale, see
[`PHASE_0_8_HANDOFF.md`](PHASE_0_8_HANDOFF.md) and
[`PHASE_0_8c_V14_HANDOFF.md`](PHASE_0_8c_V14_HANDOFF.md).

---

## Headline capabilities (each with a demo + visualization)

### A3 — External Merkle inclusion proofs

**What it enables.** A regulator/auditor with `(merkle_root, leaf_hash, index, n_leaves, proof)` can verify that a specific training event occurred at a specific position in the audit chain, **in `O(log N)` SHA-256 ops, without downloading the full audit log.** Before A3, the only verification path was a full sequential walk over every audit event.

**API surface (Rust only on this branch).**

```rust
let root: [u8; 32] = g.merkle_root();
let tree: MerkleTree = g.merkle_tree();
let proof: Vec<[u8; 32]> = tree.proof(i);
let ok: bool = MerkleTree::verify_proof(leaf, i, n_leaves, &proof, root);
```

Plus the snapshot trailer block (33 bytes per snapshot: `0x01` tag + 32-byte root) that any v14-aware reader cross-checks against the recomputed root.

**Demo.** [`tests/abng/demo_a3_merkle_proof.rs`](../../tests/abng/demo_a3_merkle_proof.rs)
- Workflow: issuer trains 16-event graph → publishes root + proof → auditor verifies → tamper-detection contrast.
- SVG: [`bench_results/phase_0_8_demos/a3_merkle_proof_tree.svg`](../../bench_results/phase_0_8_demos/a3_merkle_proof_tree.svg) — Merkle tree with the highlighted leaf, the proof path (red), and the witness siblings (orange).
- Numbers (N=16): proof = 4 hashes (128 bytes); verification cost = 4 SHA-256 ops vs 16 for a full-log walk → **4× cheaper at N=16, scaling as N/log₂(N).**

---

### A2 — Fused per-row training events

**What it enables.** Replacing the pre-A2 3-call sequence (`blr_update + observe`) with a single `train_step` call collapses the per-row audit footprint from 2 chain events (`BlrUpdated` + `BeliefUpdate`) into 1 (`TrainStep`, tag `0x1E`). The same Welford state + BLR posterior result; half the chain events.

**API surface (cjcl + Rust).**

```rust
g.train_step(&x, &phi, y)?;
```

```cjcl
abng_train_step(g, x, phi, y);
```

**Demo.** [`tests/abng/demo_a2_fused_training.rs`](../../tests/abng/demo_a2_fused_training.rs)
- Workflow: side-by-side 100-row training run, pre-A2 vs v14 A2.
- SVG: [`bench_results/phase_0_8_demos/a2_fused_training_compactness.svg`](../../bench_results/phase_0_8_demos/a2_fused_training_compactness.svg) — grouped bar chart of audit events + payload bytes.
- Numbers (N=100 rows):
  - Audit events: 219 → 119 (**−100, exactly one per row**)
  - Payload bytes: 17,797 → 11,697 (**−6,100 B, 34.3% reduction**)
  - Welford + BLR state: byte-identical on all 4 leaves ✓
  - Chain heads diverge (by design — A2's whole point)

---

### C2 — Parallel chain verification

**What it enables.** `verify_chain_par(n_threads)` splits the audit log into `n_threads` chunks and verifies them concurrently via `std::thread::scope`. At chain sizes ≥ 10,000 events (the threshold gate), the per-thread chunk verification + cross-chunk linkage check delivers measurable speedup.

**API surface (Rust only).**

```rust
g.verify_chain_par(n_threads)?;
```

**Demo.** [`tests/abng/demo_c2_parallel_verify.rs`](../../tests/abng/demo_c2_parallel_verify.rs)
- Workflow: build a 15,000-event chain → wall-clock `verify_chain_par(k)` at k ∈ {1, 2, 4, 8} → tamper test at every k.
- SVG: [`bench_results/phase_0_8_demos/c2_parallel_verify_scalability.svg`](../../bench_results/phase_0_8_demos/c2_parallel_verify_scalability.svg) — bar chart of speedup vs sequential reference.
- Numbers (N=15,000, on the development laptop):
  - Sequential reference: 22.6 ms (median of 5)
  - k=2: 2.18× speedup
  - k=4: **3.82× speedup** (likely the peak — matches 4 physical cores)
  - k=8: 3.76× (hyperthreading not helping)
- Tamper detection: caught at every k ✓
- Below the threshold (e.g. 500 events), `verify_chain_par` transparently falls through to sequential — same outcome as `verify_chain`.

---

### B3 — ZSTD-wrapped snapshots

**What it enables.** `serialize_compressed` produces a self-identifying snapshot blob (magic `ABNGZ\x01`) wrapped in a single zstd frame. Any tool that consumes ZSTD frames (S3 lifecycle, network proxies, archival systems) can store/retrieve ABNG snapshots transparently. `replay` auto-detects the wrapping and dispatches to the zstd decoder.

**API surface (Rust only, gated on `feature = "compression"`).**

```rust
let blob: Vec<u8> = serialize_compressed(&g, /* level */ 3);
let g2 = replay(&blob)?; // auto-detects the wrapping
```

**Demo.** [`tests/abng/demo_b3_zstd_snapshot.rs`](../../tests/abng/demo_b3_zstd_snapshot.rs)
- Workflow: serialize at multiple chain sizes, compressed + uncompressed → assert round-trip yields the same chain head and Merkle root.
- SVG: [`bench_results/phase_0_8_demos/b3_zstd_snapshot.svg`](../../bench_results/phase_0_8_demos/b3_zstd_snapshot.svg) — grouped bar chart, uncompressed vs compressed at multiple N.
- Numbers (level 3):
  - N=100: 14,127 B → 7,614 B (**1.86×**)
  - N=1,000: 137,427 B → 74,920 B (**1.83×**)
  - N=5,000: 685,427 B → 376,345 B (**1.82×**)
- Compression ratio stable at ~1.82× as the audit log scales — the audit log dominates snapshot bytes, so ratio holds.

---

## Infrastructure capabilities (no standalone demo)

### B1 — mmap snapshot replay

**What it enables.** `replay_mmap` / `replay_mmap_with_outcome` open a snapshot file via `mmap` instead of `fs::read`. For multi-GB historical archives, the OS handles paging on demand — no application-side RAM allocation for the whole file.

**API surface.**

```rust
let g = replay_mmap(&path)?;
let (g, outcome) = replay_mmap_with_outcome(&path)?;
```

**Why no standalone demo.** No realistic workload in the current CJC-Lang ecosystem produces multi-GB snapshots. The capability is operational ("when you eventually have huge archives, this will Just Work"), not workflow. Coverage lives in [`tests/abng/serialize_mmap.rs`](../../tests/abng/serialize_mmap.rs).

### B4 — Columnar `AuditLog`

**What it enables.** The audit log is stored as Struct-of-Arrays (`Vec<u64>` for seqs, `Vec<NodeId>` for node_ids, `Vec<[u8; 32]>` for new_hashes / previous_hashes, etc.) instead of `Vec<AuditEvent>` (Array-of-Structs). Critical primitive: `audit.new_hashes() -> &[[u8; 32]]` is a zero-copy slice over the chain witness column.

**Why no standalone demo.** B4 is the structural prerequisite for A3 (Merkle build needs a `&[[u8; 32]]` leaf list) and C2 (parallel chunks need direct column access). Calling B4 a "capability" the headline list does is generous; it's invisible infrastructure that enables the others. Its demo IS A3 + C2.

---

## Performance refactors (no canary-visible API change, no standalone demo)

### D2b — SIMD-friendly Kahan in `BlrState::update`

Replaces three scalar `KahanAccumulatorF64` reduction sites inside `BlrState::update` (`xtx`, `xty`, `yty`) with `KahanAccumulatorF64x4`, processing 4 rows per chunk via `add_lanes`. Bit-identical at `n ≤ 4`, bit-different at `n ≥ 5` (3 canaries re-locked). Win: ~3× on Cholesky-friendly inner loops at d ≥ 8. At d=4 (the default for demos) the absolute speedup is marginal; D2b's value is structural readiness for PINN/tabular workloads at higher feature widths.

### D3 — Fused matvec kernel for `Λ_old · μ_old`

Extracts the rhs computation into a named free helper, `matvec_plus_xty_kahan`, with a scalar path at `d ≤ 4` (bit-identical to pre-D3) and an F64x4 fused path at `d ≥ 8 && d % 4 == 0`. Zero canary impact at d=4 (every currently locked canary takes the scalar path).

---

## Gates this branch holds

| Gate | Count |
|---|---:|
| `cargo test -p cjc-abng --lib --release` | 320 |
| `cargo test --test abng --release` | 542 |
| `cargo test --test abng --release --features compression` | 549 |
| SHA-256 canaries (28 total) | 9 re-locked (6 A2, 3 D2b), 19 unchanged |
| Canary binaries (25 files) | all green |

---

## Honest framing

The headline capabilities split into "ship-ready for known consumers" and "infrastructure for hypothetical consumers":

| Capability | Has a current consumer? |
|---|---|
| **A2** fused training | Yes — the 6 migrated training demos use it today. |
| **A3** Merkle proofs | Not yet — no CJC-Lang user is producing inclusion proofs in production. The capability is structural readiness for that workflow. |
| **C2** parallel verify | Marginal — most current chains are below the 10K-event threshold. Audit-heavy workloads will benefit. |
| **B3** zstd snapshots | Yes — anyone archiving snapshots to cloud storage gains the 1.82× shrink immediately. |
| **B1** mmap replay | Not yet — no current workload produces multi-GB snapshots. |
| **B4** columnar audit | Indirectly — every A3 + C2 user benefits without knowing it. |

The Phase 0.8 wins concentrate in a "future-ready" zone: the infrastructure for tamper-evident external attestation, scalable verification, and snapshot interop with the broader ecosystem is now in place. The next phase's job is bringing consumers to that infrastructure.
