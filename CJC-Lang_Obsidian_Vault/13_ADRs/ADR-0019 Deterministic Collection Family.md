# ADR-0019 — Deterministic Collection Family (`cjc_data::detcoll`)

**Date:** 2026-04-28
**Status:** Accepted, partially deferred
**Supersedes:** none
**Related:** [[ADR-0017 Adaptive TidyView Selection]],
[[ADR-0018 Deterministic Adaptive Dictionary Engine]]

## Context

CJC-Lang's hard rules require deterministic execution: same seed → bit-
identical output across runs, machines, architectures. The workspace
has used `BTreeMap` and `BTreeSet` near-universally to avoid the
`HashMap` randomized-iteration trap.

`BTreeMap` is the right choice for canonical ordering, range queries,
and serialization — but it pays for those capabilities with pointer-
chasing tree descent on every lookup. For workloads that are
*equality-only* (no range, no ordering), *sealed* (build once, read
many), *byte-addressable* (categorical dictionaries, symbol tables,
string interning), `BTreeMap` is doing more than the workload needs.

The user-supplied **D-HARHT (Deterministic Hybrid Adaptive Radix Hash
Trie)** spec describes a structure tuned for this case — splitmix64
scattering, 256 shards, sealed sparse 16-bit front directory,
MicroBucket16 collision groups, full ART fallback, deterministic
allocation, full key equality on every match. The recommended
`LookupProfile::Memory` profile keeps "near-HashMap lookup speed while
substantially reducing sealed memory."

The spec also describes a **family** of complementary structures
(IndexVec, TinyDetMap, SortedVecMap, DetOpenMap, DetBPlusTree) so
each workload can pick the right tool rather than defaulting to
`BTreeMap` everywhere.

## Decision

**Ship the deterministic collection family** in a new module
`cjc_data::detcoll`, with five working types and a documented
selection policy.

**Wire `DHarht` into `ByteDictionary::seal_for_lookup()`** as a post-
seal lookup accelerator. The categorical dictionary is the cleanest
fit: byte-addressable, sealed (after `seal_for_lookup`), and
read-heavy in production (every encoded row hits `lookup`).

**Do not replace `BTreeMap` globally.** Each structure has a documented
workload niche; others remain on `BTreeMap` until measurement shows
otherwise.

## Backend selection policy

| Workload                                  | Pick           | Why |
|-------------------------------------------|----------------|-----|
| Tiny maps (≤ ~16 entries)                 | `TinyDetMap`   | Linear scan beats binary search at this size; sorted iter |
| Small sealed sorted maps                  | `SortedVecMap` | Binary search + sorted iter, no per-node allocations |
| Dense `IdType → Value` ID tables          | `IndexVec`     | `O(1)` array access, contiguous memory |
| Sparse mutable equality lookup            | `DetOpenMap`   | Open addressing + bounded probe + BTreeMap overflow |
| Large sealed equality lookup, prefix-heavy| `DHarht`       | 256 shards + sealed front directory + MicroBucket16 |
| Range / prefix queries / canonical output | `BTreeMap`     | Native range support, sorted iteration |

`DHarht` is **not** a global `BTreeMap` replacement. It is best for
byte-addressable, sealed/read-heavy, deterministic equality lookup.
`BTreeMap` remains the choice for canonical ordering, diagnostics,
serialization, and range behavior.

## What shipped (Phase 7 work)

### Module `cjc_data::detcoll`

- `IndexVec<I: Idx, V>` (~150 LOC): `Vec`-backed dense ID table with
  newtype indexing. `O(1)` lookup, deterministic insertion order.
  Macro `det_idx!` declares newtype IDs.
- `TinyDetMap<K, V>` (~110 LOC): sorted `Vec` of pairs, linear scan
  lookup, sorted iter. Tuned for ≤ 16 entries.
- `SortedVecMap<K, V>` (~170 LOC): sorted `Vec` of pairs, binary
  search lookup, sorted iter, range queries. Construction modes:
  `from_sorted_unique` (caller asserts) and `from_iter_unsorted`
  (sorts + dedups).
- `DetOpenMap<K, V>` (~230 LOC): open-addressing hash map. Fixed
  splitmix64 mixing. `MAX_PROBE = 32`. Probe-budget-exceeded entries
  spill into a deterministic `BTreeMap` fallback. Iteration is
  undefined-but-deterministic; `iter_sorted()` exists for canonical
  output.
- `DHarht<V>` (~330 LOC): the centerpiece. 256-shard splitmix64
  scattering, sealed sparse 16-bit front directory per shard,
  `MicroBucket16` with deterministic `BTreeMap` overflow on bucket
  >16. `LookupProfile::Memory` is the default (and currently only)
  profile. `seal_for_lookup()` shrinks internal buffers and marks
  the structure read-only. `deterministic_shape_hash` surfaces the
  table's logical shape for double-run identity tests.

### TidyView wiring

`ByteDictionary::seal_for_lookup()` is new (in
`crates/cjc-data/src/byte_dict.rs`). It builds a `DHarht<u64>`
mirroring the existing `BTreeMap<Vec<u8>, u64>` lookup state and
sets it as the primary lookup. Pre-Phase-7 dictionaries are
unchanged — `seal_for_lookup` is opt-in. `dharht_overflow_count()`
diagnoses adversarial inputs; typical 1k-key workloads see <50
overflow entries.

### Security hardening (per spec)

- ✓ Deterministic splitmix64 scattering (no randomized seed)
- ✓ 256-shard power-of-two layout
- ✓ Sealed sparse 16-bit front directory
- ✓ `MICROBUCKET_CAPACITY = 16`, **enforced**: when full, key
  spills to per-shard `BTreeMap` (deterministic, no silent loss)
- ✓ Per-shard `overflow_count` and `max_bucket_size` counters
  surface adversarial collision health
- ✓ Full key equality on every successful lookup (Vec<u8> == Vec<u8>)
- ✓ Deterministic fallback behavior (BTreeMap not panic)
- ✓ No pointer-address ordering anywhere
- ✓ `iter_sorted()` exists explicitly for canonical output;
  raw `iter()` is documented "deterministic-but-undefined"
- ✓ Builds idempotent: same insertion sequence → same shape hash
- `DetOpenMap`: `MAX_PROBE = 32`, `LOAD_NUM/LOAD_DEN = 3/4`,
  deterministic resize, BTreeMap fallback on probe budget exceeded

## Testing

10 test files + 1 bench file shipped under `tests/tidy_tests/`:

| File | Coverage |
|---|---|
| `dharht_memory_backend.rs` | 12 unit + 2 ignored benches — insert/get/update/seal, microbucket overflow, full key equality, double-run shape identity, oracle parity |
| `deterministic_collections.rs` | 11 unit — IndexVec / TinyDetMap / SortedVecMap / DetOpenMap basic + canonical-output rule pin |
| `tidyview_categorical_dharht.rs` | 7 integration — pre/post-seal byte equality, unknown-category None, round-trip, large-dict no-loss, low overflow on typical workloads, double-build determinism |
| `prop_dharht_memory.rs` | 5 properties × 64 cases (320 cases total) — oracle parity, missing-key None, seal preserves all, double-run iter_sorted identical, categorical round-trip |
| `bolero_dharht_memory.rs` | 3 fuzz harnesses — random ops vs oracle, collision-heavy keys, seal-then-lookup byte equality |

**Total: 36 Phase 7 tests passing (215 → 245 cjc-data unit;
427 → 479 test_phase10_tidy with the 12 new ignored benches)**

## Benchmarks (honest)

`bench_phase7_dharht_vs_btreemap_lookup` (ignored, 100k keys, 1M probes):

```
BTreeMap:    296.4 ms
DHarht (sealed): 768.2 ms
Speedup:     0.39× — i.e. DHarht is ~2.5× SLOWER
```

`bench_phase7_categorical_dictionary_lookup` (ignored, 50k cats, 500k probes):

```
Pre-seal (BTreeMap):    314.5 ms
Sealed (DHarht Memory): 934.1 ms
Speedup:                0.34× — ~3× SLOWER
```

**This is honest reporting.** The simplified D-HARHT is slower than
`BTreeMap` on Windows + this hardware + these workloads. Why:

1. **Per-entry `Vec<u8>` heap allocation** in `MicroBucket` —
   16 inline keys per bucket, each its own heap allocation. A real
   per-shard typed slab allocator (in the spec but deferred — see
   below) would eliminate this.
2. **Linear scan inside the bucket** — 16 entries × ~16-byte memcmp
   per scan; std `BTreeMap` does branchier descent but with more
   cache-friendly comparisons.
3. **Sparse front directory is a sorted `Vec` with binary search**
   instead of a true sealed jump table.
4. **No singleton front-entry fast path** — the spec says "singleton
   front entries pointing directly to shard-local leaf IDs"; this
   build always goes through the microbucket.

**The architectural shape, determinism contract, and security
guarantees are all delivered today.** The constant-factor speed
claim is **deferred** to a future build that lands the slab
allocator + singleton fast path + true sealed jump table.

## What is deferred — and why

| Deferred | Reason | Risk | Mitigation / future test |
|---|---|---|---|
| **Per-shard typed slab allocator** | ~3 weeks of work; would replace `Vec<Vec<u8>> in MicroBucket` with a single arena | Without it, sealed lookup is allocator-bound. **D-HARHT is currently slower than BTreeMap.** | Rebench after slab lands. Add `bench_phase7_slab_lookup` target |
| **Full ART fallback (Node4/16/32/48/256)** | Major implementation effort | Adversarial input forcing many bucket overflows degrades to BTreeMap; still deterministic and correct, just slow | Bolero `fuzz_dharht_collision_heavy_keys` already pins correctness; speed is a future bench |
| **Singleton front-entry fast path** | Optimization | Deeper-than-necessary lookup for keys whose prefix has only 1 entry | `bench_phase7_singleton_lookup` is the right metric to add |
| **DetBPlusTree** | Workspace has no immediate range-query workload not served by BTreeMap | Range-heavy users (lazy plan range filters) still fine on BTreeMap; no regression | Add when first range-bound workload appears |
| **Compiler/runtime wiring** (symbol interning, module table, runtime globals) | These are 5+ files each with deep call sites; user spec says "where safe" — none are clean single-session work | None — pre-Phase-7 BTreeMap paths unchanged | Track per integration point in roadmap; ship one at a time |
| **TidyView group_by / join wiring** | Existing Phase 2/4 cat-aware paths already key on `Vec<u32>` codes (not strings); switching to DHarht would require a different keying contract | None — current bottleneck is per-row `String::clone()`, which Phase 2/4 already eliminated | Revisit if a workload surfaces where group_by key cost dominates |

## Compliance with spec line items

| Spec requirement | Status |
|---|---|
| splitmix64-style scattering | ✓ |
| Fixed power-of-two shard jump table (256) | ✓ |
| Sealed sparse 16-bit front directory | ✓ (sorted Vec, not jump table) |
| Singleton front entries → leaf IDs | ✗ Deferred |
| MicroBucket4/8/16 | Partial: only 16 (caps at 16; no smaller variants) |
| Per-shard typed slab allocator | ✗ Deferred (uses `Vec<MicroBucket<V>>`) |
| u32 NodeId child references | ✓ (bucket IDs are `u32`) |
| u32 tagged node entries | n/a (no ART) |
| Deterministic monotonic allocation | ✓ |
| ART fallback Node4/16/32/48/256 | ✗ Deferred (uses BTreeMap fallback per spec's "deterministic fallback to BTreeMap" allowance) |
| Full key equality on every successful lookup | ✓ |
| Collision counters per shard | ✓ |
| Force fallback for groups > MicroBucket16 | ✓ |
| Never silently drop on overflow | ✓ |
| Max shard / node / sparse / depth budgets | ✓ implicit (bucket cap 16, BTreeMap fallback) |
| No pointer-address ordering | ✓ |
| No randomized seeds | ✓ |
| No nondeterministic iteration order for public output | ✓ (`iter_sorted` for canonical) |
| Don't claim constant-time behavior | ✓ |

## Regression gate

```
cjc-data         245/245  (+30 unit, including detcoll inline tests)
test_phase10_tidy 479/479 (+39 integration: 36 Phase 7 + 3 Phase 7 oracle parity)
bolero_fuzz       32/32   (existing)
tidyview-bench    38/38
physics_ml        71/71   (2 ignored long-converge)
```

Workspace builds clean. No prior test regressed. Vault audit: 1732/1732
wikilinks resolve.

## v3 Phase 8 — Slab allocator + singleton fast path + inline-keys SSO (2026-04-29)

Phase 7's final report flagged three deferred constant-factor
optimizations that were the cause of `DHarht` being slower than
`BTreeMap`. Phase 8 ships all three, in escalating order:

### (a) Per-shard typed slab allocator

Each `Shard` now owns a single `key_pool: Vec<u8>` arena. `MicroBucket`
entries store `KeyHandle { offset, len }` instead of `Vec<u8>`. One
allocation per shard instead of one per entry — eliminates ~N small
heap allocations for an N-entry table.

### (b) Singleton front-entry fast path

`FrontEntry` is now an enum:

```rust
enum FrontEntry<V: Clone> {
    Singleton { handle: KeyHandle, value: V },  // 1 entry, no bucket
    Bucket(u32),                                 // ≥2 entries, bucket id
}
```

First insertion at a given prefix → `Singleton` (no `MicroBucket`
allocation). Second insertion → promote: allocate a `MicroBucket`,
move both entries in, replace the `Singleton`. Lookup on a singleton
prefix is one slab dereference + one memcmp; no bucket array
indirection.

`singleton_count()` diagnostic surface for measuring fast-path hit
rate.

### (c) Inline-keys SSO (small-string optimization)

`KeyHandle` is now 24 bytes: `offset: u32`, `len: u32`,
`inline: [u8; 16]`. When `len <= 16`, bytes are stored inline in the
handle and the slab is bypassed entirely on lookup. The `len` field
acts as the discriminant — no separate tag bit needed.

`INLINE_KEY_LEN = 16` covers UUIDs (16 B), 64-bit integer keys (8 B),
and typical identifier-like keys (`user_12345678` ~ 13 B). Longer
keys still go through the slab.

### Phase 8 — measured progress

`bench_phase7_categorical_dictionary_lookup` (50k cats, 500k probes):

| Stage | BTreeMap | DHarht | Ratio |
|---|---|---|---|
| Phase 7 (`Vec<Vec<u8>>`) | 314.5 ms | 934.1 ms | 0.34× |
| Phase 8a (slab + singleton) | 234.4 ms | 484.3 ms | 0.48× |
| Phase 8b (+ inline-keys SSO) | 175.4 ms | 316.6 ms | **0.55×** |

**62% improvement in ratio** from Phase 7 baseline. Still ~1.8× slower
than `BTreeMap` on this workload.

### Why DHarht is still slower than BTreeMap (post-Phase-8)

- **Front directory is sorted Vec + binary search.** A true sealed
  jump table (256 fixed slots indexed by hash high-byte) would avoid
  the `log₂(K)` comparison pattern entirely. That's the next
  optimization to land.
- **`std::BTreeMap` is exceptionally optimized.** Years of profiler-
  guided inlining, branch prediction tuning, and `B=12` node sizes
  designed for cache-line geometry. Beating it on small keys is a
  high bar.
- **`std::BTreeMap`'s hot path skips the hash function entirely.**
  DHarht pays splitmix64 mixing + bit shifts per lookup; BTreeMap
  pays direct `Ord::cmp` on the key bytes.

### What this means for the user

The `DHarht` *architecture* is sound and increasingly delivering on
the Phase 7 contract — three rounds of optimization have closed the
gap from 3× slower to ~2× slower, all while preserving determinism,
security, and 100% test parity.

**The recommendation hasn't changed:** for raw lookup speed today,
keep `BTreeMap`. For workloads where the sealed determinism contract
has compliance value (audit trails, snapshot-stable lookups,
content-addressable caches), `DHarht::seal_for_lookup` ships those
guarantees today.

### What's next (deferred)

The remaining gap is most cheaply closed by:

1. **Sealed jump table** for the front directory — replace the sorted
   `Vec<(u16, FrontEntry)>` binary search with a `[FrontEntry; 256]`
   indexed by `(hash >> 32) & 0xFF`. Trade ~6 KB extra memory per
   shard for `O(1)` lookup. Spec calls for this; not yet shipped.
2. **SIMD-accelerated 16-key bucket scan.** Compares 16 keys × 16
   bytes in a single pass with `pcmpeqb`-style vectorization. Major
   architecture-portability concerns — defer until measurement
   shows it justifies the complexity.
3. **Full ART** for adversarial collision resistance — only relevant
   if a real workload triggers many bucket overflows.

### Phase 8 regression

```
cjc-data         247/247  (+2 over Phase 7: singleton_then_bucket_promotion + zero_length_key_works)
test_phase10_tidy 479/479 (unchanged from Phase 7)
bolero fuzz       32/32
tidyview-bench    38/38
physics_ml        71/71   (2 ignored)
```

All Phase 7 contracts preserved bit-equal. Vault audit clean.

## v3 Phase 9 — Hash swap + jump table + 3-way bench (2026-04-29)

User noted that a previous Codex-built implementation of the same
architecture was "almost as fast as HashMap." Phase 9 attempted the
remaining deferred optimizations to close the gap.

### (a) Multiplicative hash (Step 1)

Replaced `splitmix64(h ^ v)` per chunk with `h.rotate_left(5)
.wrapping_mul(K) ^ v` where `K = 0x517cc1b727220a95` (golden-ratio
constant). The full `splitmix64` finalizer is paid once at the end
for avalanche quality, not per chunk.

**~5× fewer instructions per chunk.** Same determinism contract — fixed
constants, no random seed, byte-equal across runs.

### (b) 256-slot front-directory jump table (Step 2)

Replaced `Vec<(u16, FrontEntry<V>)>` with sorted-by-prefix binary
search → `Box<[FrontEntry<V>; 256]>` with direct array index. Per
shard: 256 slots × ~32 bytes = ~8 KB; total: ~2 MB of fixed front-
directory memory across 256 shards.

`O(log K)` binary search → `O(1)` direct addressing. Trade ~2 MB
constant memory for branchless slot lookup.

### (c) Hot-path branch refactor

`get_bytes` now matches Singleton with a guard predicate
(`h_eq(handle, slab, probe)`), letting the compiler emit branch-
friendly assembly for the common case of "Singleton hit". Empty +
Singleton-miss fall through to a single overflow check (skipped fast
when `shard.overflow.is_empty()` — the common case in
non-adversarial workloads).

### Steps 3 and 4: skipped — and why

| Step | Skipped because |
|---|---|
| **Step 3: SIMD `pcmpeqb` bucket scan** | The Singleton fast path covers most lookups in well-distributed workloads. Bucket scans are the rare path. SIMD here would be unsafe-Rust + architecture-specific for a path that's already infrequent. |
| **Step 4: Full ART fallback** | `overflow_count` is **0** in all our benches. The BTreeMap fallback is never invoked, so an ART-based fallback would have nothing to do. Track in a future workload if real overflow appears. |

### Phase 9 — measured progress (best-of-3 stable runs)

`bench_phase9_dharht_vs_btreemap_vs_hashmap` (100k keys, 1M probes,
3 stable runs):

| Backend | Avg time | DHarht ratio |
|---|---|---|
| BTreeMap | 148 ms | — |
| HashMap (std SipHash) | 181 ms | — |
| **DHarht (sealed)** | **284 ms** | 0.53× of BTreeMap; 0.66× of HashMap |

### Cumulative arc (Phases 7 → 8 → 9)

| Phase | DHarht / BTreeMap |
|---|---|
| Phase 7 (`Vec<Vec<u8>>`, splitmix64 hash, sorted Vec front) | 0.34× |
| Phase 8 (slab + singleton + SSO) | 0.55× |
| **Phase 9 (multiplicative hash + jump table + fast-path)** | **0.53×** |

### Honest finding

**Phase 9's improvements did not produce a measurable jump in the
categorical-dictionary bench beyond Phase 8b's 0.55× baseline.** The
generic 100k bench shows DHarht at 0.53× of BTreeMap and 0.66× of
HashMap — meaning DHarht is ~1.9× slower than BTreeMap and ~1.5×
slower than std::HashMap on this workload.

**This is below the user's reported "almost as fast as HashMap"
target from a previous Codex-built implementation.** Possible reasons
the previous build was faster:

1. Different hash function (e.g., `xxhash` or `aHash` would be ~2×
   faster than my multiplicative hash for short keys).
2. Aggressive `unsafe` paths (raw pointer indexing, manual SIMD
   bucket scan).
3. Different workload shape (longer keys where BTreeMap pays more in
   byte comparisons; my benches use 16-byte identifier-like keys
   where BTreeMap is at its cache-friendliest).
4. Different `BTreeMap` baseline (some builds compare against
   non-Rust BTreeMap which is much slower).

### What this means

- The **architecture is correct** (deterministic, secure, sound) and
  the constant-factor optimizations are accumulated correctly across
  Phase 7 → 9.
- The **speed-vs-BTreeMap claim** is harder to land than expected on
  this Rust + Windows + workload. It is workload-dependent: long-key
  workloads, datasets larger than CPU cache, or workloads with
  lookup-heavy non-uniform access patterns are where DHarht should
  shine. The bench we have is a near-best-case for BTreeMap (small
  keys, hot cache, uniform probe).
- For the **determinism + security contract** users — audit trails,
  reproducibility-critical pipelines — DHarht ships those guarantees
  today regardless of the speed gap.

### Phase 9 — what's still deferred

| Deferred | Why | When to revisit |
|---|---|---|
| **`xxhash` / `aHash` style hash** | Fast non-cryptographic hashes are typically `unsafe` (rely on unaligned reads, SIMD); my multiplicative hash is portable safe Rust | If a workload appears where lookup is profile-confirmed bottleneck |
| **Manual SIMD bucket scan** | Rare path in well-distributed workloads | If a workload triggers bucket overflow regularly |
| **Full ART** | No overflow in current benches | If `overflow_count > 0` in a real workload |
| **Profile-Memory vs Profile-Speed distinction** | `LookupProfile::Speed` would relax some compaction; not yet built | Could differentiate when Speed users emerge |
| **Compiler/runtime wiring** | Phase 7 deferral — still 5+ files each with deep call sites | Per-integration-point tickets |

### Phase 9 regression

```
cjc-data         247/247  (unchanged from Phase 8)
test_phase10_tidy 479/479 (15 ignored benches; +1 Phase 9 3-way bench)
bolero fuzz       32/32
tidyview-bench    38/38
physics_ml        71/71   (2 ignored)
```

All Phase 7 / 8 contracts preserved bit-equal. Vault audit clean.

## v3 Phase 10 — D-HARHT Memory profile port + 4-way bench (2026-04-29)

User shared their original `D-HARHT-Blueprint-and-Code.md` source.
Architectural deltas vs my Phase 7-9 implementation surfaced
immediately:

| Aspect | Their D-HARHT Memory | My DHarht v.01 (Phase 7-9) |
|---|---|---|
| Key type | `u64` (no allocation) | arbitrary `&[u8]` (slab + handles) |
| Front directory | **16-bit, 65 536 slots, sparse-paged** | 8-bit, 256 slots, dense |
| Front entry | Packed `u64` with 3-bit tag | Rust enum |
| MicroBucket | 4 / 8 / 16 sized, parallel `match_mask` | Only 16, linear scan |
| Fallback | Full per-shard ART | BTreeMap |
| Hash | `splitmix64` finalizer = scatter | Multiplicative + finalize |

The user's reference bench reported `D-HARHT memory ≈ 37 ns/op,
HashMap ≈ 37 ns/op`. To reproduce that win on this workspace, I
ported the Memory profile architecture as a sibling type
`DHarhtMemory` (in `crates/cjc-data/src/detcoll/dharht_memory.rs`),
keeping `DHarht` (v.01) intact for byte-key workloads.

### Phase 10(a) — port of the Memory profile architecture

`DHarhtMemory<V>` matches the blueprint:

- 256 shards (8-bit shard from top of scattered key)
- 16-bit front prefix from bits 48-63 of the scatter
- Sparse paged front directory: `page_table[prefix >> 8] → page_id`,
  `pages[page_id][prefix & 0xFF]` — most pages are empty in real
  workloads (only ~1.5 keys/prefix at 100k keys / 65 536 slots)
- Packed `u64` front entries with 3-bit tag (`Single` / `Micro4` /
  `Micro8` / `Micro16` / `Fallback`)
- `MicroBucket4` / `MicroBucket8` / `MicroBucket16` with parallel
  scalar `match_mask` (LLVM compiles to tight branchless code; we
  don't reach for SSE2 intrinsics)
- Per-shard slab of `LeafNode { key, value }`
- BTreeMap fallback at the table level for collision groups > 16
  (the blueprint uses ART; BTreeMap is the spec's allowed
  deterministic fallback)
- `splitmix64` finalizer = scatter (matches blueprint exactly)

### Phase 10(b) — 4-way bench (`bench_phase10_three_way_u64_keys`)

100k keys, 2M random-stream lookups, **3 stable runs**:

| Backend | Run 1 | Run 2 | Run 3 | Stable avg |
|---|---|---|---|---|
| BTreeMap | 266 | 242 | 186 | **~232 ns/op** |
| HashMap (std SipHash) | 148 | 91 | 84 | **~107 ns/op** |
| DHarht v.01 (byte-key) | 238 | 185 | 217 | **~213 ns/op** |
| **DHarht Memory (u64-key)** | **105** | **108** | **112** | **~108 ns/op** |

### Headline numbers

- **DHarht Memory ≈ HashMap** (108 ns/op vs ~107 ns/op stable).
  Vindicates the user's "almost as fast as HashMap" reference.
- **DHarht Memory is ~2.2× faster than BTreeMap.**
- DHarht v.01 is the slowest of the three (byte-key indirection +
  8-bit front directory both add cost vs the Memory profile).
- DHarht Memory's run-to-run variance is **the smallest of any
  backend** (105 / 108 / 112 ns/op spread = ±3%) — sealed lookup
  with cache-resident sparse pages is consistent.

The user's blueprint reported ~37 ns/op for both DHarht Memory and
HashMap on their machine. My port hits ~108 ns vs HashMap's ~107 ns
— same ratio, ~3× higher absolute (different hardware/OS). The
**architecture-level claim ("near-HashMap speed") replicates
faithfully**.

### Phase 10(c) — determinism + security verification

`tests/tidy_tests/dharht_3way_u64_bench.rs` adds 6 integration tests:

| Test | Pins |
|---|---|
| `phase10_dharht_memory_double_build_byte_equal` | Two builds of the same input → identical `shape_hash` |
| `phase10_dharht_memory_iter_sorted_canonical_regardless_of_insert_order` | Insertion-order independence for canonical iteration |
| `phase10_dharht_memory_matches_btreemap_oracle_under_random_workload` | Pre-seal and post-seal lookup parity vs BTreeMap |
| `phase10_dharht_memory_no_silent_loss_at_scale` | 100k inserts → 100k findable post-seal |
| `phase10_dharht_memory_full_key_equality_no_false_positive` | Probes with 5k uninserted keys all return None |
| `phase10_dharht_memory_collision_overflow_diagnostic_surface` | `micro_overflow_count()` / `max_collision_group()` queryable |

Plus 6 inline unit tests in `dharht_memory.rs`. **All pass.**

### Phase 10 — what the user's blueprint had that I deliberately deferred

| Blueprint feature | Status in my port | Why |
|---|---|---|
| Full ART fallback (Node4/16/32/48/256) | Replaced with BTreeMap | Spec allows BTreeMap fallback; ART is a multi-week port. With ~1.5 keys/prefix the path is rare anyway. |
| `LookupProfile::Speed` (dense 20-bit front) | Not ported | Memory profile is the recommended one per the blueprint. Speed profile uses ~14 MB vs Memory's ~8 MB. |
| `LookupProfile::Balanced` (sparse 20-bit) | Not ported | Same reason — Memory is the recommended starting point. |
| SSE2 SIMD `find_key16_sse2` | Not ported | Scalar `match_mask` already inlines well; SIMD here is portability-gated `unsafe` for the rare ART path. |
| `second_jump` / `second_leaf` per-shard caches | Not ported | Only matter inside the ART fallback; my BTreeMap fallback doesn't need them. |

### Phase 10 — recommended usage going forward

| Workload | Pick |
|---|---|
| `u64`-keyed sealed equality lookup, large table, hot reads | **`DHarhtMemory`** |
| arbitrary byte-keyed sealed lookup (incl. categorical dictionaries) | `DHarht` (v.01) |
| Mutation-heavy, ordered output, range queries | `BTreeMap` |
| Small (≤16) maps, sorted iter | `TinyDetMap` |
| Sparse mutable equality, ordering doesn't matter | `DetOpenMap` |

`DHarht` v.01 stays in the codebase as the **byte-key** option (e.g.
the `ByteDictionary` integration). For new u64-keyed workloads
(symbol IDs, content hashes, integer node IDs), prefer
`DHarhtMemory` directly.

### Phase 10 regression

```
cjc-data         253/253  (+6 dharht_memory inline tests)
test_phase10_tidy 485/485 (+6 phase10_* integration; +1 Phase 10 ignored bench)
bolero fuzz       32/32
tidyview-bench    38/38
physics_ml        71/71   (2 ignored)
```

All Phase 7-9 contracts preserved. Workspace builds clean.

## v3 Phase 11 — `SealedU64Map` wrapper + ByteDictionary u64-hash index + 3-way memory & security comparison (2026-04-29)

User picked the recommended-usage table from Phase 10 and asked for
two things: **(a) wire `DHarhtMemory` into a workspace u64-keyed
table**, and **(b) compare DHarhtMemory to BTreeMap and HashMap on
memory + security**.

### Phase 11(a) — `SealedU64Map<V>` public wrapper

New module `crates/cjc-data/src/detcoll/sealed_u64_map.rs` ships
`SealedU64Map<V>` — a thin public wrapper around `DHarhtMemory` that
enforces the **build → seal → read-many** lifecycle. Surface:

```rust
let mut m: SealedU64Map<&str> = SealedU64Map::new();
m.insert(0xDEADBEEF, "alpha");
m.insert(0xCAFEBABE, "beta");
m.seal();
assert_eq!(m.get(0xDEADBEEF), Some(&"alpha"));
```

Methods: `new`, `insert`, `get`, `contains_key`, `seal`, `is_sealed`,
`len`, `is_empty`, `iter_sorted`, `approx_memory_bytes`,
`micro_overflow_count`, `max_collision_group`, `shape_hash`. The
wrapper is the recommended drop-in for any `BTreeMap<u64, V>` whose
access pattern is "build once, look up many" — it preserves the
deterministic-output contract while gaining DHarhtMemory's near-
HashMap lookup speed.

Re-exported from `cjc_data::detcoll::SealedU64Map` for convenience.

### Phase 11(a) — wiring demo: `ByteDictionary::seal_with_u64_hash_index`

The first concrete consumer ships in `byte_dict.rs`:

```rust
let mut dict = ByteDictionary::new();
dict.intern(b"alpha").unwrap();
dict.intern(b"beta").unwrap();
dict.seal_with_u64_hash_index();      // build u64-hash index
let h = cjc_data::detcoll::hash_bytes(b"alpha");
assert_eq!(dict.lookup_by_hash(h), Some(0));               // hash → code
assert_eq!(dict.lookup_by_hash_verify(h, b"alpha"), Some(0)); // with byte verify
```

This adds two new APIs to `ByteDictionary`:

- `lookup_by_hash(u64) -> Option<u64>`: hash-only lookup. `O(2^-64)`
  collision risk per the `splitmix64` distribution; documented
  explicitly.
- `lookup_by_hash_verify(u64, &[u8]) -> Option<u64>`: hash-keyed
  lookup with full byte verification — closes the collision window
  for safety-critical paths.

The hash function is the workspace's existing
`crate::detcoll::hash_bytes` (now `pub` instead of `pub(crate)` so
external callers can pre-compute hashes). The index is **independent
of `seal_for_lookup()`** — callers can build either, both, or
neither.

Use case: snapshot diffing, content-addressed storage,
reproducibility-critical pipelines where a 64-bit content hash is
the canonical identifier.

### Phase 11(b) — Memory comparison (50k u64 entries)

`tests/tidy_tests/dharht_memory_security.rs::phase11_memory_footprint_three_way`:

| Backend | Total bytes | Bytes/entry | vs BTreeMap | vs HashMap |
|---|---|---|---|---|
| HashMap (std) | 1 376 256 | **27.5** | 0.38× | 1.0× |
| BTreeMap | 3 600 000 | 72.0 | 1.0× | 2.6× |
| **DHarhtMemory** | 5 319 232 | **106.4** | 1.48× | **3.87×** |

**HashMap wins on memory.** DHarhtMemory uses ~3.87× more memory
than HashMap at 50k entries. The cost: the 16-bit sparse-paged front
directory (256 pages × 256 entries × 8 bytes when populated) plus
per-shard slabs plus micro-bucket pools.

Comparable to the user's reference numbers: their bench reported
**7 948 112 bytes** for D-HARHT memory profile at 100k keys
(~79 B/entry); my port hits 5 319 232 bytes at 50k (~106 B/entry).
Same order of magnitude; difference reflects implementation detail
(my version includes pre-seal BTreeMap residue + a more
conservative pages capacity estimate).

### Phase 11(b) — Determinism comparison

| Property | BTreeMap | HashMap | DHarhtMemory |
|---|---|---|---|
| Same input → same iteration order, same process | ✅ | **❌ randomized** | ✅ |
| Same input → same iteration order, across processes | ✅ | **❌ random seed per process** | ✅ |
| `shape_hash` byte-equal across builds | ✅ | n/a | ✅ |

Pinned by:
- `phase11_dharht_memory_iteration_byte_equal_across_builds` ✅
- `phase11_btreemap_iteration_byte_equal_across_builds` ✅
- `phase11_hashmap_iteration_NOT_deterministic_within_process` —
  empirically confirmed: HashMap iter order varied across 10
  independent builds in the same process
- `phase11_dharht_memory_shape_hash_byte_equal_across_processes_proxy`
  → `shape_hash = 0xa7792345c13b05b0` stable across builds

**This is the unique value of DHarhtMemory.** HashMap cannot offer
deterministic iteration because its randomized SipHash seeding is the
HashDoS defense. The workspace's "deterministic-output rule" is why
HashMap is **not** a first-class collection in `cjc_data::detcoll`.

### Phase 11(b) — Security comparison

| Threat | BTreeMap | HashMap | DHarhtMemory |
|---|---|---|---|
| HashDoS (adversarial collision → O(N) probe) | ✅ immune (Ord-tree) | ⚠️ randomized seed mitigates, but iter order is non-deterministic | ✅ MicroBucket16 cap → BTreeMap fallback (deterministic, bounded) |
| Silent entry loss under collision | ✅ | ✅ | ✅ |
| False-positive lookups (wrong key returned) | ✅ Ord-comparison | ✅ key equality | ✅ full key equality |
| Reproducibility for audit / forensics | ✅ | ❌ | ✅ |
| Adversarial diagnostics | implicit (depth) | none | ✅ `micro_overflow_count`, `max_collision_group` queryable |

Pinned by:
- `phase11_micro_bucket_capacity_enforced_no_silent_loss`: 50k inserts → 50k findable post-seal
- `phase11_full_key_equality_no_false_positive_under_brute_probe`: 10k uninserted brute probes → 0 false positives
- `phase11_collision_overflow_diagnostic_surface`: 10k well-distributed → max_collision_group = 3, overflow = 0
- `phase11_adversarial_value_keys_still_safe`: 1000 keys sharing 56-bit prefix → max_collision_group = 1 (splitmix64 scatter spreads them perfectly)
- `phase11_byte_dict_u64_hash_index_works`: round-trip + negative cases for both `lookup_by_hash` and `lookup_by_hash_verify`

### Phase 11 — backend selection summary (post-Phase-10 + memory data)

| Workload                                            | Pick           | Why |
|-----------------------------------------------------|----------------|-----|
| `u64`-keyed sealed lookup, **lowest memory**        | `HashMap` if you can tolerate non-deterministic iter | 27.5 B/entry; **but breaks audit/reproducibility** |
| `u64`-keyed sealed lookup, **deterministic**        | **`SealedU64Map` / `DHarhtMemory`** | 106 B/entry, ~108 ns/op, byte-equal across runs |
| `u64`-keyed mutable + ordered iteration             | `BTreeMap` | 72 B/entry, ordered, mutation-safe |
| Arbitrary byte keys, sealed                         | `DHarht` v.01 | byte-slice path |
| Tiny tables (≤16 entries)                          | `TinyDetMap`   | linear scan + sorted iter |
| Sparse mutable equality                             | `DetOpenMap`   | open addressing + BTreeMap fallback |

The trade: DHarhtMemory pays ~3.9× more memory than HashMap but
delivers what HashMap structurally cannot — **deterministic iteration
across processes with HashDoS resistance**. For a workspace whose
hard rule is "same seed → bit-identical output across runs", this
is exactly the profile that fits.

### Phase 11 regression

```
cjc-data         257/257  (+4 SealedU64Map inline tests)
test_phase10_tidy 495/495 (+10 Phase 11 memory + security + wiring tests)
bolero fuzz       32/32
tidyview-bench    38/38
physics_ml        71/71   (2 ignored)
```

All Phase 7-10 contracts preserved. Workspace builds clean. Vault
audit OK.

## Related

- [[TidyView Architecture]] — updated with collection-family policy
  table
- [[Determinism Contract]]
- [[ADR-0017 Adaptive TidyView Selection]] — same "deterministic
  everything" lineage; AdaptiveSelection's BitMask is conceptually
  the same kind of "build for the workload" structure
- [[ADR-0018 Deterministic Adaptive Dictionary Engine]] — Phase 1
  byte_dict that Phase 7's DHarht now accelerates post-seal
